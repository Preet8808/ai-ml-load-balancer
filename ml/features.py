"""
Shared feature engineering for training and serving (Phase 4).

This module is imported by both the offline training pipeline and the live
balancer decision engine. One shared definition is what prevents the
training/serving mismatch the project plan flags as a key challenge.

Design rule: every feature must be computable from a single collector
snapshot plus the incoming request. No rolling windows or history-dependent
transforms, because at serving time the balancer holds one `ServerMetrics`
snapshot per server. The collector already publishes `rt_ewma_ms`,
`rt_p95_5s_ms` and `cpu_delta_5s`, which carry the history that a rolling
window would otherwise supply.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Log schema (Phase 2) — the 16 columns written to Parquet
# ---------------------------------------------------------------------------

RAW_COLUMNS: List[str] = [
    "timestamp", "run_id", "req_type", "payload_kb", "server_id",
    "srv_cpu_pct", "srv_mem_pct", "inflight", "req_rate_5s",
    "rt_ewma_ms", "rt_p95_5s_ms", "err_rate_5s", "cpu_limit",
    "cpu_delta_5s", "response_ms", "status",
]

TARGET_COLUMN = "response_ms"
TARGET_LOG_COLUMN = "log_response_ms"  # log1p(response_ms)

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

# Server state, exactly as published by MetricsCollector
STATE_FEATURES: List[str] = [
    "srv_cpu_pct",
    "srv_mem_pct",
    "inflight",
    "req_rate_5s",
    "rt_ewma_ms",
    "rt_p95_5s_ms",
    "err_rate_5s",
    "cpu_limit",
    "cpu_delta_5s",
]

# Request features, known before a server is chosen
REQUEST_FEATURES: List[str] = [
    "payload_kb",
]

REQ_TYPES: List[str] = ["light", "cpu", "io", "payload"]

FEATURE_COLUMNS: List[str] = (
    STATE_FEATURES + REQUEST_FEATURES + [f"req_type_{t}" for t in REQ_TYPES]
)

# Latencies are heavy-tailed. log1p keeps a single outlier from dominating
# the squared-error objective, and predictions map back with expm1.
TARGET_CAP_MS = 5000.0


# Maps each logged feature column to the corresponding ServerMetrics field.
# The names differ because the log schema is prefixed to make the columns
# self-describing, and reading the wrong attribute silently yields 0.0.
METRICS_FIELD: Dict[str, str] = {
    "srv_cpu_pct": "cpu_pct",
    "srv_mem_pct": "mem_pct",
    "inflight": "inflight_requests",
    "req_rate_5s": "req_rate_5s",
    "rt_ewma_ms": "rt_ewma_ms",
    "rt_p95_5s_ms": "rt_p95_5s_ms",
    "err_rate_5s": "error_rate_5s",
    "cpu_limit": "cpu_limit",
    "cpu_delta_5s": "cpu_delta_5s",
}


def feature_columns() -> List[str]:
    """Ordered feature names. Used to validate loaded models."""
    return list(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Training-time transformation
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) from raw log rows.

    Returns X as a float frame with FEATURE_COLUMNS in order, and y as
    log1p(response_ms) with the target clipped at TARGET_CAP_MS.
    """
    df = df.copy()

    # Rows the proxy could not attribute to a server carry no state
    if "server_id" in df.columns:
        df = df[df["server_id"].astype(str).str.len() > 0]

    # Failed requests are not observations of response time. A 503 returns
    # quickly, so keeping them teaches the model that a failing server is a
    # fast one - the opposite of what a router should learn. Errors stay in
    # the logs and reach the model only through `err_rate_5s`.
    if "status" in df.columns:
        status = pd.to_numeric(df["status"], errors="coerce")
        df = df[status < 400]

    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df[target.notna()]
    target = target.clip(upper=TARGET_CAP_MS)

    work = pd.DataFrame(index=df.index)

    for col in STATE_FEATURES + REQUEST_FEATURES:
        if col in df.columns:
            work[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            work[col] = np.nan
    work = work.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    req_type = df["req_type"].astype(str)
    for t in REQ_TYPES:
        work[f"req_type_{t}"] = (req_type == t).astype(float)

    y = np.log1p(target.to_numpy())
    return work[FEATURE_COLUMNS].astype(float), pd.Series(y, index=work.index, name="log_response_ms")


def to_ms(pred_log: np.ndarray) -> np.ndarray:
    """Map a log-scale prediction back to milliseconds."""
    return np.clip(np.expm1(np.asarray(pred_log, dtype=float)), 0.0, TARGET_CAP_MS)


# ---------------------------------------------------------------------------
# Serving-time transformation
# ---------------------------------------------------------------------------

def build_serving_features(
    req_type: str,
    payload_kb: float,
    metrics: Any,
) -> pd.DataFrame:
    """
    Build a one-row feature frame for a live routing decision.

    `metrics` may be a `ServerMetrics` dataclass or any object/dict exposing
    the same fields. Returns a DataFrame with FEATURE_COLUMNS in the same
    order used at training time.
    """
    def get(column: str, default: float = 0.0) -> float:
        # The logged column name and the metrics field name differ, so read
        # through METRICS_FIELD rather than assuming they match.
        field = METRICS_FIELD.get(column, column)
        if isinstance(metrics, dict):
            value = metrics.get(field, default)
        else:
            value = getattr(metrics, field, default)
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
        return default if not np.isfinite(out) else out

    row = {col: get(col) for col in STATE_FEATURES}
    # payload_kb comes from the request, not the server
    try:
        row["payload_kb"] = float(payload_kb)
    except (TypeError, ValueError):
        row["payload_kb"] = 0.0

    for t in REQ_TYPES:
        row[f"req_type_{t}"] = 1.0 if req_type == t else 0.0

    return pd.DataFrame([row], columns=FEATURE_COLUMNS, dtype=float)


def build_serving_matrix(
    req_type: str,
    payload_kb: float,
    servers: List[Any],
) -> pd.DataFrame:
    """
    Build one feature row per candidate server for a single request.

    This is the router's hot path: given N servers it produces an N-row
    matrix, and the model returns N predicted response times.
    """
    rows = [
        build_serving_features(req_type, payload_kb, getattr(s, "metrics", None))
        for s in servers
    ]
    return pd.concat(rows, ignore_index=True)[FEATURE_COLUMNS].astype(float)
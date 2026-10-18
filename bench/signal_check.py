"""
Signal gate for the dataset.

A load-aware router is only useful if the *server state features* carry
information about response time. A dataset where metrics-only R^2 is near
zero cannot produce a meaningful model no matter which regressor is used,
so this check runs before Phase 4 training.

It also catches the specific failure modes that produced an unusable dataset
earlier: constant features, a single shared run_id, and unreadable files.
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

# Features that must vary and must explain response time on their own.
STATE_FEATURES = [
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

MIN_STATE_R2 = 0.30
MIN_ROWS = 5000
MIN_RUNS = 4


def load_parquet_dir(data_dir: str) -> tuple[pd.DataFrame, list[str]]:
    """Load all Parquet files, reporting any that cannot be read."""
    files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    frames = []
    unreadable = []
    for f in sorted(files):
        try:
            frames.append(pd.read_parquet(f))
        except Exception as exc:
            unreadable.append(f"{os.path.basename(f)}: {str(exc)[:70]}")
    if not frames:
        return pd.DataFrame(), unreadable
    return pd.concat(frames, ignore_index=True), unreadable


def feature_diagnostics(df: pd.DataFrame) -> list[str]:
    """Report features that are constant or entirely zero (useless signals)."""
    problems = []
    for col in STATE_FEATURES:
        if col not in df.columns:
            problems.append(f"{col}: missing")
            continue
        s = df[col]
        if s.isna().all():
            problems.append(f"{col}: all NaN")
        elif s.nunique() <= 1:
            problems.append(f"{col}: constant ({s.iloc[0]!r})")
        elif (s == 0).mean() > 0.98:
            problems.append(f"{col}: {(s == 0).mean() * 100:.1f}% zeros")
    return problems


def state_r2(df: pd.DataFrame) -> float:
    """Cross-validated R^2 using only live server-state features."""
    X = df[STATE_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = np.log1p(df["response_ms"].astype(float).clip(upper=5000))
    model = Ridge(alpha=1.0)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(model, X, y, cv=cv)
    return float(r2_score(y, pred))


def endpoint_r2(df: pd.DataFrame) -> float:
    """Reference R^2 from endpoint identity alone, for comparison."""
    X = pd.get_dummies(df["req_type"].astype(str), dtype=float)
    y = np.log1p(df["response_ms"].astype(float).clip(upper=5000))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(Ridge(alpha=1e-3), X, y, cv=cv)
    return float(r2_score(y, pred))


def signal_gate(data_dir: str) -> bool:
    """Full gate. Returns True only if the dataset is fit for training."""
    print("-" * 68)
    print("  SIGNAL GATE")
    print("-" * 68)

    df, unreadable = load_parquet_dir(data_dir)

    if unreadable:
        print(f"  [FAIL] {len(unreadable)} unreadable Parquet file(s):")
        for u in unreadable[:5]:
            print(f"           {u}")

    if df.empty:
        print("  [FAIL] no readable rows")
        print("-" * 68)
        return False

    print(f"  rows      : {len(df):,}")
    print(f"  run_ids   : {df['run_id'].nunique()}")

    failures = list(unreadable)

    # 1. Volume
    if len(df) < MIN_ROWS:
        print(f"  [FAIL] only {len(df):,} rows, need >= {MIN_ROWS:,}")
        failures.append("row count")

    # 2. Run isolation — one run_id per file is what makes grouped CV valid
    n_runs = df["run_id"].nunique()
    if n_runs < MIN_RUNS:
        print(f"  [FAIL] only {n_runs} distinct run_id(s), need >= {MIN_RUNS}")
        print("         (a single run_id means run-grouped CV will leak)")
        failures.append("run isolation")

    # 3. Feature health
    problems = feature_diagnostics(df)
    if problems:
        print(f"  [WARN] {len(problems)} degenerate feature(s):")
        for p in problems:
            print(f"           {p}")

    hard_feature_failures = [p for p in problems if "missing" in p or "all NaN" in p]
    if hard_feature_failures:
        failures.append("missing features")

    # 4. Does server state actually predict response time?
    r2_state = state_r2(df)
    r2_endpoint = endpoint_r2(df)
    print(f"  R2 state-only  : {r2_state:+.4f}   (need >= {MIN_STATE_R2:+.2f})")
    print(f"  R2 endpoint-only: {r2_endpoint:+.4f}   (reference)")

    if r2_state < MIN_STATE_R2:
        print(f"  [FAIL] server-state features explain almost nothing (R2={r2_state:+.4f}).")
        print("         A model trained here would only memorise endpoint cost.")
        failures.append("state signal")

    if r2_state < r2_endpoint:
        print("  [WARN] server state explains less than endpoint identity alone.")

    # 5. Label sanity
    n_unique_rt = df["response_ms"].nunique()
    print(f"  response_ms: mean={df['response_ms'].mean():.1f} "
          f"p50={df['response_ms'].median():.1f} "
          f"p95={df['response_ms'].quantile(0.95):.1f} "
          f"max={df['response_ms'].max():.1f} ({n_unique_rt} distinct)")

    if n_unique_rt < 50:
        print("  [FAIL] response_ms has almost no variation")
        failures.append("label variation")

    print("-" * 68)
    if failures:
        print(f"  GATE FAILED: {failures}")
        print("-" * 68)
        return False

    print(f"  GATE PASSED — {len(df):,} rows, {n_runs} runs, state R2={r2_state:+.4f}")
    print("-" * 68)
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data"
    sys.exit(0 if signal_gate(target) else 2)
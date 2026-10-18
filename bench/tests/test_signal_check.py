"""
Tests for the dataset signal gate.

The gate is what stands between a feature-free dataset and a Phase 4 training
run that reports a large improvement while learning nothing. These tests build
synthetic datasets that reproduce both failure modes seen in practice:

  - the original dataset, where latency depended only on endpoint identity and
    every server-state feature was constant
  - a healthy dataset, where server state genuinely drives latency
"""

import numpy as np
import pandas as pd
import pytest

from bench.signal_check import (
    MIN_RUNS,
    MIN_STATE_R2,
    feature_diagnostics,
    signal_gate,
    state_r2,
)

COLUMNS = [
    "timestamp", "run_id", "req_type", "payload_kb", "server_id",
    "srv_cpu_pct", "srv_mem_pct", "inflight", "req_rate_5s",
    "rt_ewma_ms", "rt_p95_5s_ms", "err_rate_5s", "cpu_limit",
    "cpu_delta_5s", "response_ms", "status",
]


def write_frame(df: pd.DataFrame, path) -> None:
    df[COLUMNS].to_parquet(path)


def healthy_dataset(n_runs: int = 6, rows_per_run: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Latency rises with CPU load and in-flight count, as it should."""
    rng = np.random.default_rng(seed)
    frames = []
    for r in range(n_runs):
        cpu = rng.uniform(0, 100, rows_per_run)
        inflight = rng.integers(0, 25, rows_per_run)
        rate = rng.uniform(0, 120, rows_per_run)
        ewma = 3.0 + 0.35 * cpu + rng.normal(0, 3, rows_per_run)
        rt = (
            4.0
            + 0.55 * cpu
            + 6.0 * inflight
            + 0.02 * rate
            + rng.normal(0, 6, rows_per_run)
        ).clip(0.5, 4000)
        frames.append(pd.DataFrame({
            "timestamp": np.arange(rows_per_run) / 20.0,
            "run_id": f"run_{r}",
            "req_type": rng.choice(["light", "cpu", "io", "payload"], rows_per_run),
            "payload_kb": rng.choice([0.0, 2.0, 8.0, 32.0], rows_per_run),
            "server_id": rng.choice(["backend1", "backend2", "backend3", "backend4"], rows_per_run),
            "srv_cpu_pct": cpu,
            "srv_mem_pct": rng.uniform(5, 90, rows_per_run),
            "inflight": inflight,
            "req_rate_5s": rate,
            "rt_ewma_ms": ewma,
            "rt_p95_5s_ms": ewma * 2.2,
            "err_rate_5s": rng.uniform(0, 0.05, rows_per_run),
            "cpu_limit": rng.choice([2.0, 1.0, 1.0, 0.5], rows_per_run),
            "cpu_delta_5s": rng.normal(0, 5, rows_per_run),
            "response_ms": rt,
            "status": 200,
        }))
    return pd.concat(frames, ignore_index=True)


def memorisable_dataset(n_runs: int = 6, rows_per_run: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Latency depends only on endpoint; server state is constant or noise."""
    rng = np.random.default_rng(seed)
    frames = []
    for r in range(n_runs):
        req_type = rng.choice(["light", "cpu", "io", "payload"], rows_per_run)
        base = np.select(
            [req_type == "io", req_type == "payload", req_type == "cpu"],
            [55.0, 12.0, 5.0],
            default=4.0,
        )
        frames.append(pd.DataFrame({
            "timestamp": np.arange(rows_per_run) / 20.0,
            "run_id": f"run_{r}",
            "req_type": req_type,
            "payload_kb": 0.0,
            "server_id": "backend1",
            "srv_cpu_pct": 100.0,          # the original saturation bug
            "srv_mem_pct": 100.0,
            "inflight": 0,
            "req_rate_5s": rng.normal(6, 0.5, rows_per_run),
            "rt_ewma_ms": rng.normal(6, 0.5, rows_per_run),
            "rt_p95_5s_ms": rng.normal(8, 0.5, rows_per_run),
            "err_rate_5s": 0.0,
            "cpu_limit": 1.0,
            "cpu_delta_5s": 0.0,
            "response_ms": base + rng.normal(0, 1.5, rows_per_run),
            "status": 200,
        }))
    return pd.concat(frames, ignore_index=True)


def test_healthy_dataset_passes_the_gate(tmp_path):
    write_frame(healthy_dataset(), tmp_path / "a.parquet")
    assert signal_gate(str(tmp_path)) is True


def test_memorisable_dataset_fails_the_gate(tmp_path):
    write_frame(memorisable_dataset(), tmp_path / "b.parquet")
    assert signal_gate(str(tmp_path)) is False


def test_gate_rejects_single_run_dataset(tmp_path):
    """One run_id means run-grouped CV cannot be done without leaking."""
    write_frame(healthy_dataset(n_runs=1), tmp_path / "c.parquet")
    assert signal_gate(str(tmp_path)) is False


def test_gate_rejects_empty_directory(tmp_path):
    assert signal_gate(str(tmp_path)) is False


def test_gate_rejects_corrupt_parquet(tmp_path):
    (tmp_path / "broken.parquet").write_bytes(b"PAR1 not actually parquet")
    assert signal_gate(str(tmp_path)) is False


def test_feature_diagnostics_flags_constant_features():
    df = memorisable_dataset(n_runs=2, rows_per_run=50)
    problems = feature_diagnostics(df)
    joined = " ".join(problems)
    assert "srv_cpu_pct" in joined
    assert "inflight" in joined
    assert "cpu_delta_5s" in joined


def test_state_r2_is_high_for_load_driven_latency():
    df = healthy_dataset()
    assert state_r2(df) > MIN_STATE_R2


def test_state_r2_is_near_zero_for_endpoint_only_latency():
    df = memorisable_dataset()
    assert state_r2(df) < MIN_STATE_R2


def test_gate_thresholds_are_reachable():
    """A dataset that meets the documented minimums must not trip on volume."""
    df = healthy_dataset(n_runs=MIN_RUNS, rows_per_run=1500)
    assert len(df) >= 5000
    assert df["run_id"].nunique() >= MIN_RUNS
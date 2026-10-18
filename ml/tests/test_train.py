"""
Training-pipeline tests for Phase 4.

Two properties matter more than the scores themselves:

  * the split is by run, never by row. Rows inside a run share a server state
    that evolves over seconds, so a random row split puts near-duplicates on
    both sides and every reported score becomes optimistic;
  * the artifact that gets saved is the artifact that can be served. A model
    dumped with a different feature order than the router builds at decision
    time is the classic training/serving mismatch, and it fails silently.

The pipeline is therefore exercised end to end on a synthetic dataset whose
latency is a known function of server state, so "did it learn anything" has a
definite answer instead of depending on the collected data being present.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from balancer.interfaces import ServerInfo, ServerMetrics
from ml.features import FEATURE_COLUMNS, build_serving_matrix, feature_columns
from ml.train import (
    build_models,
    evaluate,
    best_model_name,
    load_dataset,
    split_by_run,
    train,
)

# Per-request cost by endpoint, in milliseconds. The spread is what makes a
# server's own state matter: routing a `cpu` request to a busy server costs far
# more than routing an `light` one there.
ENDPOINT_COST_MS = {"light": 1.0, "cpu": 25.0, "io": 6.0, "payload": 14.0}


def synthetic_run(run_id: str, n_rows: int = 150, seed: int = 0) -> pd.DataFrame:
    """
    One run whose latency is a deterministic function of server state.

    `rt_ewma_ms` is deliberately a weak, uninformative proxy so the naive
    baseline is genuinely naive; otherwise the comparison measures nothing.
    """
    rng = np.random.default_rng(seed)
    start = 1_700_000_000.0 + seed * 10_000
    timestamps = start + np.arange(n_rows, dtype=float)

    req_type = rng.choice(list(ENDPOINT_COST_MS), size=n_rows)
    payload_kb = np.array([{"light": 0.0, "cpu": 0.0, "io": 0.0, "payload": 32.0}[r] for r in req_type])
    cpu_pct = rng.uniform(5.0, 95.0, n_rows)
    inflight = rng.integers(0, 12, n_rows).astype(float)
    endpoint_cost = np.array([ENDPOINT_COST_MS[r] for r in req_type])

    # Latency depends on how busy the server is *and* on what was asked of it.
    # inflight has a superlinear effect, which is where the queueing lives.
    response_ms = (
        5.0
        + 0.6 * cpu_pct
        + 4.0 * inflight
        + 0.35 * inflight ** 2
        + endpoint_cost
        + rng.normal(0.0, 2.0, n_rows)
    )
    response_ms = np.clip(response_ms, 0.5, None)

    return pd.DataFrame({
        "timestamp": timestamps,
        "run_id": run_id,
        "req_type": req_type,
        "payload_kb": payload_kb,
        "server_id": "backend1",
        "srv_cpu_pct": cpu_pct,
        "srv_mem_pct": rng.uniform(10.0, 60.0, n_rows),
        "inflight": inflight,
        "req_rate_5s": rng.uniform(20.0, 90.0, n_rows),
        "rt_ewma_ms": rng.uniform(5.0, 25.0, n_rows),
        "rt_p95_5s_ms": rng.uniform(10.0, 40.0, n_rows),
        "err_rate_5s": np.zeros(n_rows),
        "cpu_limit": np.full(n_rows, 1.0),
        "cpu_delta_5s": rng.normal(0.0, 1.0, n_rows),
        # A failing request has no meaningful response time; keep a few so the
        # error-row filter is exercised.
        "status": np.where(rng.random(n_rows) < 0.03, 503, 200),
        "response_ms": response_ms,
    })


def synthetic_dataset(n_runs: int = 8, n_rows: int = 150) -> pd.DataFrame:
    return pd.concat(
        [synthetic_run(f"run{i:02d}", n_rows=n_rows, seed=i) for i in range(n_runs)],
        ignore_index=True,
    )


def write_dataset(df: pd.DataFrame, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for run_id, group in df.groupby("run_id"):
        group.to_parquet(directory / f"{run_id}.parquet", index=False)
    return directory


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def test_split_by_run_never_shares_a_run():
    """The leakage guard: a run must land entirely in one split."""
    df = synthetic_dataset(n_runs=10)

    train_mask, val_mask, test_mask = split_by_run(df, df.index)[:3]

    assert not (train_mask & test_mask).any()
    assert not (train_mask & val_mask).any()
    assert not (val_mask & test_mask).any()

    runs = df["run_id"].to_numpy()
    for mask in (train_mask, val_mask, test_mask):
        selected = set(runs[mask])
        others = set(runs[~mask])
        assert not (selected & others), "a run_id appears on both sides of the split"


def test_split_by_run_always_produces_a_test_set():
    """With two runs a plain 70/15/15 split would leave the test set empty."""
    df = synthetic_dataset(n_runs=2)
    y_index = df.index
    _, _, test_mask, test_runs = split_by_run(df, y_index)

    assert test_mask.any()
    assert test_runs


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_dataset_drops_each_runs_warmup_window(tmp_path):
    """
    The warm-up window is dropped per run, not once for the whole dataset.

    A run's first seconds are its coldest, so they do not describe steady
    behaviour. Dropping them globally instead of per run would keep the cold
    part of every later run.
    """
    df = synthetic_dataset(n_runs=3, n_rows=50)
    data_dir = write_dataset(df, tmp_path / "data")

    loaded, info = load_dataset(str(data_dir), warmup_s=20.0)

    assert info["rows_raw"] == 150
    assert info["warmup_dropped"] == 60  # 20 rows dropped from each of 3 runs
    assert len(loaded) == 90
    assert info["runs"] == 3

    for run_id, group in loaded.groupby("run_id"):
        start = df[df["run_id"] == run_id]["timestamp"].min()
        assert group["timestamp"].min() >= start + 20.0


def test_load_dataset_reports_unreadable_files(tmp_path):
    """A corrupt file must be reported and skipped, not crash the pipeline."""
    data_dir = write_dataset(synthetic_dataset(n_runs=2), tmp_path / "data")
    (data_dir / "corrupt.parquet").write_bytes(b"not a parquet file")

    loaded, info = load_dataset(str(data_dir), warmup_s=0.0)

    assert len(info["unreadable"]) == 1
    assert "corrupt.parquet" in info["unreadable"][0]
    assert not loaded.empty


# ---------------------------------------------------------------------------
# Metrics and model selection
# ---------------------------------------------------------------------------

def test_evaluate_rewards_a_perfect_prediction():
    y_log = np.log1p(np.array([10.0, 100.0, 1000.0]))
    metrics = evaluate(y_log, y_log, np.expm1(y_log))

    assert metrics["r2_log"] == pytest.approx(1.0)
    assert metrics["mae_log"] == pytest.approx(0.0)
    assert metrics["mae_ms"] == pytest.approx(0.0)


def test_best_model_name_ignores_the_naive_baseline():
    """
    The baseline must never be selected as the winner.

    It is stored in the same dict as the models, and it is often the most
    accurate entry, so picking the minimum without excluding it would ship the
    heuristic this project exists to beat.
    """
    results = {
        "naive": {"metrics": {"mae_log": 0.01}},
        "ridge": {"metrics": {"mae_log": 0.50}, "model": object()},
        "xgboost": {"metrics": {"mae_log": 0.20}, "model": object()},
    }
    assert best_model_name(results) == "xgboost"
    assert best_model_name({"naive": {"metrics": {"mae_log": 0.01}}}) is None


def test_build_models_registers_the_expected_estimators():
    models = build_models()
    assert set(models) == {"ridge", "random_forest", "xgboost"}
    assert models["ridge"] is not None
    assert models["random_forest"] is not None


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

@pytest.fixture
def fast_models(monkeypatch):
    """A one-model registry so the end-to-end test stays quick and deterministic."""
    monkeypatch.setattr(
        "ml.train.build_models",
        lambda: {"ridge": Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])},
    )


def test_train_beats_the_naive_baseline_on_a_learnable_signal(tmp_path, fast_models):
    """
    The pipeline has to actually learn, not merely run.

    The synthetic latency is a function of cpu_pct, inflight and the endpoint
    cost, while the baseline only sees a noisy rt_ewma_ms. A pipeline that
    builds its features correctly must therefore beat the baseline clearly; if
    this fails, feature construction or the target transform has drifted.
    """
    data_dir = write_dataset(synthetic_dataset(n_runs=10, n_rows=200), tmp_path / "data")

    results = train(str(data_dir), str(tmp_path / "models"))

    naive = results["naive"]["metrics"]["mae_log"]
    model = results["ridge"]["metrics"]["mae_log"]
    assert model < naive * 0.7, f"model MAE {model:.4f} did not clearly beat naive {naive:.4f}"


def test_train_saves_a_servable_artifact(tmp_path, fast_models):
    """
    The saved bundle must be directly usable by the routing path.

    This closes the loop between training and serving: the feature order in the
    bundle is compared against what the router builds, and a real prediction is
    taken from the loaded model over real ServerInfo objects.
    """
    data_dir = write_dataset(synthetic_dataset(n_runs=8, n_rows=200), tmp_path / "data")
    model_dir = tmp_path / "models"

    results = train(str(data_dir), str(model_dir))

    bundles = sorted(model_dir.glob("*.joblib"))
    assert bundles, "training saved no model bundle"
    reports = sorted(model_dir.glob("training_report_*.json"))
    assert reports, "training saved no report"

    import joblib

    bundle = joblib.load(bundles[0])
    assert bundle["feature_columns"] == feature_columns()
    assert bundle["target"] == "log1p(response_ms)"
    assert bundle["train_rows"] > 0

    report = json.loads(reports[0].read_text())
    assert report["meta"]["best_model"] == "ridge"
    assert report["meta"]["rows"] > 0
    assert report["meta"]["test_runs"]
    assert "feature_importance" in report or "cv_r2_log" in report["ridge"]["metrics"]

    servers = [
        ServerInfo(
            server_id=f"backend{i}",
            host=f"backend{i}",
            port=8000,
            weight=1.0,
            metrics=ServerMetrics(
                server_id=f"backend{i}",
                timestamp=1_700_000_000.0,
                cpu_pct=20.0 * i,
                mem_pct=30.0,
                load_avg=1.0,
                net_io_bytes_per_s=1000.0,
                inflight_requests=float(i),
                req_rate_5s=40.0,
                rt_ewma_ms=12.0,
                rt_p95_5s_ms=25.0,
                error_rate_5s=0.0,
                cpu_limit=1.0,
                cpu_delta_5s=0.0,
            ),
        )
        for i in range(1, 5)
    ]

    X_serve = build_serving_matrix("cpu", 0.0, servers)
    assert list(X_serve.columns) == bundle["feature_columns"]

    preds = bundle["model"].predict(X_serve)
    assert len(preds) == len(servers)
    assert np.isfinite(preds).all()

    # The busiest server is the one the synthetic signal makes slowest.
    assert preds[0] < preds[-1]

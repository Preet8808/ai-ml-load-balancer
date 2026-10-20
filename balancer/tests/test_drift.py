"""
Unit tests for the drift guard, its wiring into the ML router, and the
decision log.

The behaviours under test are what "demote a model whose live error grows" has
to mean to be useful: it measures the model's error in the units the model was
selected on, it does not demote on noise, it re-promotes by itself, and it does
not flap between the two around the threshold.
"""

import math
import time
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from balancer.decision_log import DECISION_SCHEMA, make_decision_logger
from balancer.drift import DriftGuard
from balancer.interfaces import RequestContext, RequestType, ServerInfo, ServerMetrics
from balancer.ml_router import MLRouter

FAR_OFF = 500.0   # ~3.8 absolute error on the log scale against a 10 ms prediction


def make_metrics(age_s: float = 0.5, healthy: bool = True) -> ServerMetrics:
    return ServerMetrics(
        server_id="s",
        timestamp=time.time() - age_s,
        cpu_pct=10.0,
        mem_pct=20.0,
        load_avg=0.1,
        net_io_bytes_per_s=0.0,
        inflight_requests=0,
        req_rate_5s=1.0,
        rt_ewma_ms=10.0,
        rt_p95_5s_ms=12.0,
        error_rate_5s=0.0,
        cpu_limit=1.0,
        cpu_delta_5s=0.0,
        healthy=healthy,
    )


def make_server(server_id: str, inflight: int = 0) -> ServerInfo:
    server = ServerInfo(server_id=server_id, host=server_id, port=8000)
    server.metrics = make_metrics()
    server.inflight_local = inflight
    return server


def make_request() -> RequestContext:
    return RequestContext(
        request_id="r1", method="GET", path="/cpu", req_type=RequestType.CPU
    )


def make_router() -> MLRouter:
    """A router with a constant predictor, so the guard is the thing under test."""
    router = MLRouter(model_path=str(Path("does_not_exist.joblib")))
    assert router.model is None and router.load_error
    # Inject a model that always predicts log1p(53.6 ms) = 4.0.
    router.model = object()
    router._predict = lambda matrix: [4.0] * len(matrix)
    return router


# ---------------------------------------------------------------------------
# DriftGuard: measurement
# ---------------------------------------------------------------------------

def test_error_is_measured_on_the_models_own_scale():
    guard = DriftGuard()
    guard.observe(100.0, 100.0)
    assert guard.live_mae == 0.0

    guard.reset()
    guard.observe(100.0, 200.0)
    assert guard.live_mae == pytest.approx(math.log1p(200.0) - math.log1p(100.0))


def test_observations_that_cannot_be_scored_are_ignored_not_counted():
    guard = DriftGuard()

    assert guard.observe(None, 10.0) is None
    assert guard.observe(10.0, None) is None
    assert guard.observe(float("nan"), 10.0) is None
    assert guard.observe(10.0, float("inf")) is None
    assert guard.observe(0.0, 10.0) is None
    assert guard.observe(-5.0, 10.0) is None
    assert guard.observe(10.0, -1.0) is None

    status = guard.status()
    assert status["samples_total"] == 0
    assert status["ignored"] == 7
    assert status["live_mae_log"] is None


def test_accurate_predictions_never_demote():
    guard = DriftGuard(window=20, min_samples=5, threshold=0.5)
    for _ in range(50):
        assert guard.observe(100.0, 100.0) is None
    assert not guard.demoted
    assert guard.status()["demotions"] == 0


# ---------------------------------------------------------------------------
# DriftGuard: demotion and re-promotion
# ---------------------------------------------------------------------------

def test_demotes_once_the_window_is_bad_and_the_minimum_is_met():
    guard = DriftGuard(window=20, min_samples=5, threshold=0.5)

    for _ in range(4):
        assert guard.observe(10.0, FAR_OFF) is None
    assert not guard.demoted, "min_samples gates the decision"

    assert guard.observe(10.0, FAR_OFF) == "demoted"
    assert guard.demoted
    assert guard.status()["demotions"] == 1
    assert guard.status()["samples_in_window"] == 0, "the window is cleared, not carried over"


def test_promotes_only_after_a_full_clean_window():
    guard = DriftGuard(window=10, min_samples=3, threshold=0.5, hysteresis=0.8)
    for _ in range(3):
        guard.observe(10.0, FAR_OFF)
    assert guard.demoted

    for _ in range(9):
        assert guard.observe(50.0, 50.0) is None
    assert guard.demoted, "a partial window is not enough evidence"

    assert guard.observe(50.0, 50.0) == "promoted"
    assert not guard.demoted
    assert guard.status()["promotions"] == 1


def test_the_error_that_demoted_does_not_immediately_promote():
    """
    Hysteresis: an error hovering just above the threshold must not flap.

    100 ms predicted against 280 ms actual is an error of 1.02 on the log
    scale, so with a threshold of 1.0 and a 0.8 hysteresis it demotes and then
    never re-promotes while the same error keeps arriving.
    """
    guard = DriftGuard(window=6, min_samples=2, threshold=1.0, hysteresis=0.8)

    assert guard.observe(100.0, 280.0) is None
    assert guard.status()["live_mae_log"] == pytest.approx(1.023, abs=0.01)

    assert guard.observe(100.0, 280.0) == "demoted"

    for _ in range(12):
        assert guard.observe(100.0, 280.0) is None
    assert guard.demoted
    assert guard.status()["promotions"] == 0


def test_configure_changes_threshold_window_and_minimum():
    guard = DriftGuard(window=50, min_samples=10, threshold=2.0)
    guard.observe(100.0, 100.0)

    status = guard.configure(threshold=0.01, window=5, min_samples=1)

    assert status["threshold"] == 0.01
    assert status["window"] == 5
    assert status["min_samples"] == 1
    assert status["samples_in_window"] == 1, "resizing keeps the window contents"


# ---------------------------------------------------------------------------
# Router wiring
# ---------------------------------------------------------------------------

def test_router_demotes_itself_when_its_predictions_stop_matching_reality():
    router = make_router()
    router.drift.configure(window=10, min_samples=5, threshold=0.5)
    servers = [make_server("backend1"), make_server("backend2")]

    for _ in range(5):
        choice = router.choose_server(make_request(), servers)
        assert choice.model_driven
        assert choice.predicted_rt_ms == pytest.approx(math.expm1(4.0))
        router.observe_outcome(choice.predicted_rt_ms, 400.0, 200)

    assert router.drift.demoted

    choice = router.choose_server(make_request(), servers)
    assert choice.reason.startswith("drift-fallback")
    assert not choice.model_driven
    assert router.drift_routed == 1
    # The shadow prediction is what lets the guard see a recovery.
    assert choice.predicted_rt_ms == pytest.approx(math.expm1(4.0))


def test_router_promotes_the_model_back_after_clean_observations():
    router = make_router()
    router.drift.configure(window=6, min_samples=2, threshold=0.5)
    servers = [make_server("backend1"), make_server("backend2")]

    router.observe_outcome(50.0, 400.0, 200)
    router.observe_outcome(50.0, 400.0, 200)
    assert router.drift.demoted
    assert router.choose_server(make_request(), servers).reason.startswith("drift-fallback")

    for _ in range(6):
        router.observe_outcome(50.0, 50.0, 200)

    assert not router.drift.demoted
    choice = router.choose_server(make_request(), servers)
    assert choice.model_driven


def test_failed_requests_are_not_observations_about_the_model():
    router = make_router()
    router.observe_outcome(10.0, 900.0, 503)
    assert router.drift.status()["samples_total"] == 0


def test_the_drift_guard_has_a_high_default_threshold():
    """
    The default is a "worse than the naive baseline" line, not a tuned value:
    the training report measures XGBoost at 0.59 MAE(log) and the naive
    baseline at 1.27.
    """
    assert DriftGuard().threshold == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_decision_logger_writes_its_own_schema_and_prefix(tmp_path):
    log_dir = str(tmp_path / "decisions")
    logger = make_decision_logger(run_id="run_d", log_dir=log_dir)
    await logger.start()
    await logger.start_run("run_d")

    await logger.log({
        "timestamp": 1.0,
        "request_id": "req-1",
        "algorithm": "ml",
        "server_id": "backend1",
        "reason": "ml(pred 12.3ms, inflight 1)",
        "predicted_ms": 12.3,
        "actual_ms": 14.0,
        "status": 200,
        "retries": 0,
        "model_driven": True,
        "inflight": 1,
        "feed_age_ms": 400.0,
        "req_type": "cpu",
    })
    await logger.stop()

    files = list(Path(log_dir).glob("decisions_run_d_*.parquet"))
    assert len(files) == 1, "the decision log has its own filename prefix"

    table = pq.read_table(files[0])
    assert table.column_names == [field.name for field in DECISION_SCHEMA]
    assert table.num_rows == 1

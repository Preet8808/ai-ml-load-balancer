"""
Unit tests for the ML routing algorithm (Phase 5).

The routing logic is tested against a scripted estimator, so the tests do not
depend on a trained artifact or on which model happened to win. Loading is
tested separately against artifacts written to a temporary directory.

The point of these tests is the serving contract, not the model's accuracy:
the frame handed to the estimator must carry the training column order, the
prediction must be reported in milliseconds, and a router without a usable
model must fall back to a heuristic instead of failing to route.
"""

import json

import joblib
import numpy as np
import pytest

from balancer.algorithms import algorithm_names, get_algorithm
from balancer.interfaces import RequestContext, RequestType, ServerInfo, ServerMetrics
from balancer.ml_router import MLRouter, find_artifact
from ml.features import FEATURE_COLUMNS, to_ms


class FakeModel:
    """Stand-in for a fitted estimator: returns one scripted value per row."""

    def __init__(self, predictions):
        self.predictions = list(predictions)
        self.frame = None
        self.calls = 0

    def predict(self, frame):
        self.frame = frame
        self.calls += 1
        return np.asarray(self.predictions[: len(frame)], dtype=float)


def make_metrics(server_id: str, inflight: int = 0, rt_ewma_ms: float = 50.0) -> ServerMetrics:
    return ServerMetrics(
        server_id=server_id,
        timestamp=1000.0,
        cpu_pct=20.0,
        mem_pct=30.0,
        load_avg=0.5,
        net_io_bytes_per_s=1000.0,
        inflight_requests=inflight,
        req_rate_5s=10.0,
        rt_ewma_ms=rt_ewma_ms,
        rt_p95_5s_ms=rt_ewma_ms * 2,
        error_rate_5s=0.0,
        cpu_limit=1.0,
        cpu_delta_5s=0.0,
        healthy=True,
        stale=False,
    )


def make_server(server_id: str, inflight: int = 0, rt_ewma_ms: float = 50.0) -> ServerInfo:
    return ServerInfo(
        server_id=server_id,
        host=f"host-{server_id}",
        port=8000,
        metrics=make_metrics(server_id, inflight=inflight, rt_ewma_ms=rt_ewma_ms),
    )


def make_request(req_type: RequestType = RequestType.LIGHT, payload_kb: float = 0.0) -> RequestContext:
    return RequestContext(
        request_id="test-req-1",
        method="GET",
        path="/ping",
        req_type=req_type,
        payload_kb=payload_kb,
    )


def make_router(tmp_path, predictions) -> MLRouter:
    """A router with a scripted model, bypassing artifact loading."""
    router = MLRouter(model_dir=tmp_path)
    router.model = FakeModel(predictions)
    router.path = tmp_path / "scripted.joblib"
    return router


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_ml_algorithm_is_selectable():
    assert "ml" in algorithm_names()
    assert get_algorithm("ml").name == "ml"


def test_unknown_algorithm_lists_ml_as_available():
    with pytest.raises(ValueError, match="ml"):
        get_algorithm("nope")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_picks_the_server_with_the_lowest_prediction(tmp_path):
    router = make_router(tmp_path, predictions=[3.0, 2.0, 5.0])
    servers = [make_server("s1"), make_server("s2"), make_server("s3")]

    choice = router.choose_server(make_request(), servers)

    assert choice.server.server_id == "s2"
    assert router.decisions == 1
    assert router.fallbacks == 0


def test_frame_handed_to_the_model_keeps_the_training_column_order(tmp_path):
    router = make_router(tmp_path, predictions=[1.0, 2.0])
    servers = [make_server("s1"), make_server("s2")]

    router.choose_server(make_request(RequestType.PAYLOAD, payload_kb=32.0), servers)

    frame = router.model.frame
    assert list(frame.columns) == FEATURE_COLUMNS
    assert len(frame) == 2, "one row per candidate server"
    # The request's own attributes must reach the model, not defaults: a
    # payload request that arrives as payload_kb=0 with all req_type dummies
    # at zero is the training/serving mismatch this asserts against.
    assert frame["payload_kb"].iloc[0] == 32.0
    assert frame["req_type_payload"].iloc[0] == 1.0


def test_prediction_is_reported_in_milliseconds(tmp_path):
    router = make_router(tmp_path, predictions=[4.0, 2.0])
    servers = [make_server("s1"), make_server("s2")]

    choice = router.choose_server(make_request(), servers)

    # The model predicts log1p(ms); the choice reports ms for the dashboard.
    assert choice.predicted_rt_ms == pytest.approx(float(to_ms(np.array([2.0]))[0]))
    assert "ml(pred" in choice.reason


def test_unseparated_predictions_break_ties_by_inflight(tmp_path):
    # A flat model must not concentrate every request on one backend.
    router = make_router(tmp_path, predictions=[3.0, 3.0, 3.0])
    servers = [
        make_server("s1", inflight=7),
        make_server("s2", inflight=1),
        make_server("s3", inflight=4),
    ]

    choice = router.choose_server(make_request(), servers)

    assert choice.server.server_id == "s2"


def test_no_healthy_servers_returns_none(tmp_path):
    router = make_router(tmp_path, predictions=[1.0])
    unhealthy = [ServerInfo(server_id="s1", host="h1", port=8000, metrics=None)]

    assert router.choose_server(make_request(), unhealthy) is None


def test_a_failing_predict_falls_back_rather_than_failing_the_request(tmp_path):
    class Broken(FakeModel):
        def predict(self, frame):
            raise RuntimeError("estimator exploded")

    router = make_router(tmp_path, predictions=[])
    router.model = Broken([])
    servers = [make_server("s1", inflight=5), make_server("s2", inflight=1)]

    choice = router.choose_server(make_request(), servers)

    assert choice.server.server_id == "s2", "falls back to least-connections"
    assert "ml-fallback" in choice.reason
    assert router.fallbacks == 1


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_missing_artifact_falls_back_and_says_so(tmp_path):
    router = MLRouter(model_dir=tmp_path)
    servers = [make_server("s1", inflight=4), make_server("s2", inflight=0)]

    assert router.model is None
    assert "no model artifact" in router.load_error

    status = router.status()
    assert status["loaded"] is False
    assert status["load_error"]

    choice = router.choose_server(make_request(), servers)
    assert choice.server.server_id == "s2"
    assert router.status()["fallbacks"] == 1


def test_artifact_round_trips_through_the_loader(tmp_path):
    bundle = {
        "model": FakeModel([1.0]),
        "feature_columns": list(FEATURE_COLUMNS),
        "target": "log1p(response_ms)",
        "data_hash": "abc123",
        "train_rows": 100,
        "runs": 3,
    }
    joblib.dump(bundle, tmp_path / "random_forest_abc123.joblib")

    router = MLRouter(model_dir=tmp_path)

    assert router.model is not None, router.load_error
    status = router.status()
    assert status["loaded"] is True
    assert status["estimator"] == "FakeModel"
    assert status["trained_on"] == "abc123"
    assert status["train_rows"] == 100


def test_artifact_with_a_different_feature_layout_is_refused(tmp_path):
    # Silently serving a model whose columns are ordered differently produces
    # plausible-looking wrong numbers, so it must not load at all.
    joblib.dump(
        {"model": FakeModel([1.0]), "feature_columns": list(reversed(FEATURE_COLUMNS))},
        tmp_path / "random_forest_abc123.joblib",
    )

    router = MLRouter(model_dir=tmp_path)

    assert router.model is None
    assert "feature columns" in router.load_error


def test_find_artifact_prefers_the_best_model_of_the_newest_report(tmp_path):
    for name in ("ridge_aaa111", "xgboost_aaa111", "random_forest_zzz999"):
        joblib.dump({"model": FakeModel([1.0])}, tmp_path / f"{name}.joblib")
    (tmp_path / "training_report_aaa111.json").write_text(
        json.dumps({"meta": {"best_model": "xgboost", "data_hash": "aaa111"}})
    )

    found = find_artifact(tmp_path)

    assert found is not None
    # Alphabetical order alone would have picked random_forest_zzz999.
    assert found.name == "xgboost_aaa111.joblib"


def test_model_path_env_var_wins(tmp_path, monkeypatch):
    chosen = tmp_path / "chosen.joblib"
    joblib.dump({"model": FakeModel([1.0])}, chosen)
    joblib.dump({"model": FakeModel([1.0])}, tmp_path / "newer.joblib")
    monkeypatch.setenv("MODEL_PATH", str(chosen))

    # find_artifact must honour MODEL_PATH over anything the directory says.
    assert find_artifact(tmp_path) == chosen

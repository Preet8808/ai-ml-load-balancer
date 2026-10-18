"""
Feature-engineering tests for Phase 4.

The most important test here is `test_serving_matches_training`: a model
trained offline must see exactly the same feature values at routing time. A
mismatch between the two paths is the classic way a load-aware router ships
looking accurate in a notebook and behaves randomly in production.
"""

import numpy as np
import pandas as pd
import pytest

from balancer.interfaces import ServerInfo, ServerMetrics
from ml.features import (
    FEATURE_COLUMNS,
    STATE_FEATURES,
    TARGET_CAP_MS,
    build_features,
    build_serving_features,
    build_serving_matrix,
    feature_columns,
    to_ms,
)


def make_row(**overrides):
    row = {
        "timestamp": 1_700_000_000.0,
        "run_id": "r01",
        "req_type": "cpu",
        "payload_kb": 8.0,
        "server_id": "backend2",
        "srv_cpu_pct": 55.0,
        "srv_mem_pct": 22.0,
        "inflight": 3,
        "req_rate_5s": 42.0,
        "rt_ewma_ms": 91.0,
        "rt_p95_5s_ms": 180.0,
        "err_rate_5s": 0.02,
        "cpu_limit": 1.0,
        "cpu_delta_5s": -4.5,
        "response_ms": 95.5,
        "status": 200,
    }
    row.update(overrides)
    return row


def test_feature_schema_is_stable():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
    for name in STATE_FEATURES:
        assert name in FEATURE_COLUMNS
    for req_type in ["light", "cpu", "io", "payload"]:
        assert f"req_type_{req_type}" in FEATURE_COLUMNS
    assert feature_columns() == FEATURE_COLUMNS


def test_build_features_shape_and_order():
    df = pd.DataFrame([make_row(), make_row(run_id="r02")])
    X, y = build_features(df)

    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == 2
    assert len(y) == 2
    assert X.dtypes.eq(float).all()

    expected = np.log1p(95.5)
    assert y.iloc[0] == pytest.approx(expected)


def test_build_features_one_hot_encodes_req_type():
    df = pd.DataFrame([make_row(req_type="io"), make_row(req_type="ping_like")])
    X, _ = build_features(df)

    assert X["req_type_io"].tolist() == [1.0, 0.0]
    assert X["req_type_cpu"].tolist() == [0.0, 0.0]


def test_unknown_req_type_does_not_crash():
    df = pd.DataFrame([make_row(req_type="something-new")])
    X, _ = build_features(df)
    assert X["req_type_light"].iloc[0] == 0.0
    assert X["req_type_cpu"].iloc[0] == 0.0


def test_nan_and_inf_are_cleaned():
    df = pd.DataFrame([
        make_row(srv_cpu_pct=float("nan"), rt_ewma_ms=float("inf"), inflight=None),
    ])
    X, _ = build_features(df)
    assert np.isfinite(X.to_numpy()).all()


def test_target_is_capped_at_sla():
    df = pd.DataFrame([make_row(response_ms=99_000.0)])
    _, y = build_features(df)
    assert y.iloc[0] == pytest.approx(np.log1p(TARGET_CAP_MS))


def test_to_ms_round_trips_log_scale():
    values = np.array([0.0, 5.0, 250.0, 3000.0])
    back = to_ms(np.log1p(values))
    assert back == pytest.approx(values, rel=1e-6)


def test_to_ms_never_negative_or_unbounded():
    back = to_ms(np.array([-50.0, 50.0]))
    assert (back >= 0).all()
    assert (back <= TARGET_CAP_MS).all()


def test_serving_matches_training():
    """The serving path must reproduce the training feature values exactly."""
    row = make_row()
    df = pd.DataFrame([row])
    X, _ = build_features(df)

    metrics = ServerMetrics(
        server_id="backend2",
        timestamp=row["timestamp"],
        cpu_pct=row["srv_cpu_pct"],
        mem_pct=row["srv_mem_pct"],
        load_avg=0.0,
        net_io_bytes_per_s=0.0,
        inflight_requests=row["inflight"],
        req_rate_5s=row["req_rate_5s"],
        rt_ewma_ms=row["rt_ewma_ms"],
        rt_p95_5s_ms=row["rt_p95_5s_ms"],
        error_rate_5s=row["err_rate_5s"],
        cpu_limit=row["cpu_limit"],
        cpu_delta_5s=row["cpu_delta_5s"],
    )

    served = build_serving_features("cpu", 8.0, metrics)

    assert list(served.columns) == FEATURE_COLUMNS
    for col in FEATURE_COLUMNS:
        assert served[col].iloc[0] == pytest.approx(X[col].iloc[0]), f"mismatch on {col}"


def test_serving_handles_missing_metrics():
    served = build_serving_features("ping_like", 0.0, None)
    assert list(served.columns) == FEATURE_COLUMNS
    assert np.isfinite(served.to_numpy()).all()
    # Every request-type flag off, state all zero
    assert served[STATE_FEATURES].to_numpy().sum() == 0.0


def test_build_serving_matrix_one_row_per_server():
    def server(sid, cpu):
        return ServerInfo(
            server_id=sid,
            host=sid,
            port=8000,
            metrics=ServerMetrics(
                server_id=sid,
                timestamp=1.0,
                cpu_pct=cpu,
                mem_pct=10.0,
                load_avg=0.0,
                net_io_bytes_per_s=0.0,
                inflight_requests=1,
                req_rate_5s=10.0,
                rt_ewma_ms=20.0,
                rt_p95_5s_ms=30.0,
                error_rate_5s=0.0,
                cpu_limit=1.0,
                cpu_delta_5s=0.0,
            ),
        )

    servers = [server("b1", 10.0), server("b2", 90.0), server("b3", 50.0)]
    matrix = build_serving_matrix("io", 2.0, servers)

    assert len(matrix) == 3
    assert list(matrix.columns) == FEATURE_COLUMNS
    assert matrix["srv_cpu_pct"].tolist() == [10.0, 90.0, 50.0]
    assert matrix["req_type_io"].tolist() == [1.0, 1.0, 1.0]


def test_features_do_not_leak_the_label():
    """No feature may be derived from response_ms."""
    assert "response_ms" not in FEATURE_COLUMNS
    assert "status" not in FEATURE_COLUMNS
    assert not any("response" in name for name in FEATURE_COLUMNS)
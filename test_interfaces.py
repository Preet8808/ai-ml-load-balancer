"""Tests for the AI/ML load balancer interfaces."""

import pytest

from balancer.interfaces import (
    RequestContext,
    RequestType,
    ServerChoice,
    ServerInfo,
    ServerMetrics,
)


def test_light_request_creation():
    """A light request can be constructed with all defaults."""
    req = RequestContext(
        request_id="req-1",
        method="GET",
        path="/ping",
        req_type=RequestType.LIGHT,
    )
    assert req.request_id == "req-1"
    assert req.method == "GET"
    assert req.path == "/ping"
    assert req.req_type == RequestType.LIGHT
    assert req.payload_kb == 0.0


def test_server_info_defaults():
    """ServerInfo default metrics produce healthy=True."""
    metrics = ServerMetrics(
        server_id="s1",
        timestamp=100.0,
        cpu_pct=10.0,
        mem_pct=20.0,
        load_avg=0.5,
        net_io_bytes_per_s=1000.0,
        inflight_requests=0,
        req_rate_5s=1.0,
        rt_ewma_ms=10.0,
        rt_p95_5s_ms=20.0,
        error_rate_5s=0.0,
        cpu_limit=2.0,
        cpu_delta_5s=0.0,
    )
    server = ServerInfo(
        server_id="s1",
        host="localhost",
        port=8001,
        metrics=metrics,
    )
    assert server.healthy is True


def test_server_missing_metrics_is_unhealthy():
    """Server without metrics is not healthy."""
    server = ServerInfo(
        server_id="s2",
        host="localhost",
        port=8002,
        metrics=None,
    )
    assert server.healthy is False


def test_stale_metrics_make_server_unhealthy():
    """A server with stale metrics is not healthy."""
    metrics = ServerMetrics(
        server_id="s3",
        timestamp=100.0,
        cpu_pct=10.0,
        mem_pct=20.0,
        load_avg=0.5,
        net_io_bytes_per_s=1000.0,
        inflight_requests=0,
        req_rate_5s=1.0,
        rt_ewma_ms=10.0,
        rt_p95_5s_ms=20.0,
        error_rate_5s=0.0,
        cpu_limit=2.0,
        cpu_delta_5s=0.0,
        stale=True,
    )
    server = ServerInfo(
        server_id="s3",
        host="localhost",
        port=8003,
        metrics=metrics,
    )
    assert server.healthy is False


def test_unhealthy_server_is_unhealthy():
    """A server reporting unhealthy metrics is not healthy."""
    metrics = ServerMetrics(
        server_id="s4",
        timestamp=100.0,
        cpu_pct=99.0,
        mem_pct=95.0,
        load_avg=50.0,
        net_io_bytes_per_s=1000.0,
        inflight_requests=100,
        req_rate_5s=500.0,
        rt_ewma_ms=5000.0,
        rt_p95_5s_ms=10000.0,
        error_rate_5s=0.5,
        cpu_limit=1.0,
        cpu_delta_5s=10.0,
        healthy=False,
    )
    server = ServerInfo(
        server_id="s4",
        host="localhost",
        port=8004,
        metrics=metrics,
    )
    assert server.healthy is False


def test_server_choice_defaults():
    """ServerChoice has default reason and no prediction."""
    server = ServerInfo(
        server_id="s5",
        host="localhost",
        port=8005,
    )
    choice = ServerChoice(server=server)
    assert choice.server == server
    assert choice.reason == ""
    assert choice.predicted_rt_ms is None
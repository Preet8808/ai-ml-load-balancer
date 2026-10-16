"""
Unit tests for Phase 1: All routing algorithms.

Tests:
- Every algorithm returns a valid ServerChoice for healthy servers
- Every algorithm returns None when no healthy servers
- RoundRobin cycles through all servers
- WeightedRoundRobin distributes within 2% of weights over 10,000 picks
- LeastConnections picks the server with fewest in-flight requests
- WeightedLeastConnections accounts for weight
- LeastResponseTime picks the server with lowest EWMA response time
"""

import sys
sys.path.insert(0, 'C:/Users/DELL/projects/ai_ml_load_balancer')

import pytest
from balancer.algorithms import (
    Random,
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    WeightedLeastConnections,
    LeastResponseTime,
    get_algorithm,
    ALGORITHMS,
)
from balancer.interfaces import (
    RequestContext,
    RequestType,
    ServerChoice,
    ServerInfo,
    ServerMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def make_server(server_id: str, weight: float = 1.0, inflight: int = 0, rt_ewma_ms: float = 50.0) -> ServerInfo:
    return ServerInfo(
        server_id=server_id,
        host=f"host-{server_id}",
        port=8000,
        weight=weight,
        metrics=make_metrics(server_id, inflight=inflight, rt_ewma_ms=rt_ewma_ms),
    )


def make_request() -> RequestContext:
    return RequestContext(
        request_id="test-req-1",
        method="GET",
        path="/ping",
        req_type=RequestType.LIGHT,
    )


def make_servers_equal(n: int = 3) -> list[ServerInfo]:
    return [make_server(f"s{i+1}") for i in range(n)]


# ---------------------------------------------------------------------------
# get_algorithm registry
# ---------------------------------------------------------------------------

def test_all_algorithms_in_registry():
    expected = {"random", "round_robin", "weighted_rr", "least_conn", "weighted_least_conn", "least_response_time"}
    assert set(ALGORITHMS.keys()) == expected


def test_get_algorithm_valid():
    for name in ALGORITHMS:
        alg = get_algorithm(name)
        assert alg.name == name


def test_get_algorithm_invalid():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        get_algorithm("nonexistent")


# ---------------------------------------------------------------------------
# No healthy servers → None
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alg_name", list(ALGORITHMS.keys()))
def test_no_healthy_servers_returns_none(alg_name):
    alg = get_algorithm(alg_name)
    servers = [
        ServerInfo(server_id="s1", host="h1", port=8000, metrics=None),  # no metrics = unhealthy
    ]
    result = alg.choose_server(make_request(), servers)
    assert result is None


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------

def test_random_returns_valid_choice():
    alg = Random()
    servers = make_servers_equal(3)
    result = alg.choose_server(make_request(), servers)
    assert isinstance(result, ServerChoice)
    assert result.server in servers
    assert result.reason == "random"


def test_random_picks_all_servers_over_many_calls():
    alg = Random()
    servers = make_servers_equal(3)
    ids_seen = set()
    for _ in range(200):
        choice = alg.choose_server(make_request(), servers)
        ids_seen.add(choice.server.server_id)
    assert ids_seen == {"s1", "s2", "s3"}


# ---------------------------------------------------------------------------
# RoundRobin
# ---------------------------------------------------------------------------

def test_round_robin_cycles_all_servers():
    alg = RoundRobin()
    servers = make_servers_equal(3)
    seen = [alg.choose_server(make_request(), servers).server.server_id for _ in range(6)]
    # Each server should appear twice in 6 picks (3 equal-weight servers)
    assert seen.count("s1") == 2
    assert seen.count("s2") == 2
    assert seen.count("s3") == 2


def test_round_robin_reason():
    alg = RoundRobin()
    result = alg.choose_server(make_request(), make_servers_equal(2))
    assert result.reason == "round_robin"


# ---------------------------------------------------------------------------
# WeightedRoundRobin — within 2% of weights over 10,000 picks
# ---------------------------------------------------------------------------

def test_weighted_round_robin_proportions():
    alg = WeightedRoundRobin()
    servers = [
        make_server("s1", weight=2.0),
        make_server("s2", weight=1.0),
        make_server("s3", weight=1.0),
    ]
    counts = {"s1": 0, "s2": 0, "s3": 0}
    N = 10_000
    for _ in range(N):
        choice = alg.choose_server(make_request(), servers)
        counts[choice.server.server_id] += 1

    # Expected proportions: s1=50%, s2=25%, s3=25%
    assert abs(counts["s1"] / N - 0.50) < 0.02, f"s1={counts['s1']/N:.3f}"
    assert abs(counts["s2"] / N - 0.25) < 0.02, f"s2={counts['s2']/N:.3f}"
    assert abs(counts["s3"] / N - 0.25) < 0.02, f"s3={counts['s3']/N:.3f}"


# ---------------------------------------------------------------------------
# LeastConnections
# ---------------------------------------------------------------------------

def test_least_connections_picks_minimum_inflight():
    alg = LeastConnections()
    servers = [
        make_server("s1", inflight=10),
        make_server("s2", inflight=2),   # least in-flight
        make_server("s3", inflight=7),
    ]
    result = alg.choose_server(make_request(), servers)
    assert result.server.server_id == "s2"
    assert result.reason == "least_conn"


def test_least_connections_single_server():
    alg = LeastConnections()
    servers = [make_server("s1", inflight=0)]
    result = alg.choose_server(make_request(), servers)
    assert result.server.server_id == "s1"


# ---------------------------------------------------------------------------
# WeightedLeastConnections
# ---------------------------------------------------------------------------

def test_weighted_least_connections_accounts_for_weight():
    alg = WeightedLeastConnections()
    # s1 has 10 in-flight but weight=2 → score=5
    # s2 has 6  in-flight but weight=1 → score=6   (s1 wins)
    servers = [
        make_server("s1", weight=2.0, inflight=10),
        make_server("s2", weight=1.0, inflight=6),
    ]
    result = alg.choose_server(make_request(), servers)
    assert result.server.server_id == "s1"
    assert result.reason == "weighted_least_conn"


# ---------------------------------------------------------------------------
# LeastResponseTime
# ---------------------------------------------------------------------------

def test_least_response_time_picks_fastest():
    alg = LeastResponseTime()
    servers = [
        make_server("s1", rt_ewma_ms=200.0),
        make_server("s2", rt_ewma_ms=50.0),   # fastest
        make_server("s3", rt_ewma_ms=120.0),
    ]
    result = alg.choose_server(make_request(), servers)
    assert result.server.server_id == "s2"
    assert result.reason == "least_response_time"

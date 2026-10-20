"""
Unit tests for the serving safeguards (Phases 5-6).

These encode the plan's exit criteria: a failing backend is bypassed in
milliseconds rather than after a health-check interval, a recovered backend is
ramped back in, and a stopped metrics feed keeps serving instead of failing.
"""

import time

import pytest

from balancer import resilience
from balancer.interfaces import RequestContext, RequestType, ServerChoice, ServerInfo, ServerMetrics


def make_metrics(age_s: float = 0.5, healthy: bool = True, inflight: int = 0) -> ServerMetrics:
    return ServerMetrics(
        server_id="s",
        timestamp=time.time() - age_s,
        cpu_pct=10.0,
        mem_pct=20.0,
        load_avg=0.1,
        net_io_bytes_per_s=0.0,
        inflight_requests=inflight,
        req_rate_5s=1.0,
        rt_ewma_ms=10.0,
        rt_p95_5s_ms=12.0,
        error_rate_5s=0.0,
        cpu_limit=1.0,
        cpu_delta_5s=0.0,
        healthy=healthy,
    )


def make_server(
    server_id: str,
    age_s: float = 0.5,
    healthy: bool = True,
    inflight: int = 0,
    weight: float = 1.0,
) -> ServerInfo:
    server = ServerInfo(server_id=server_id, host=server_id, port=8000, weight=weight)
    server.metrics = make_metrics(age_s=age_s, healthy=healthy, inflight=inflight)
    server.inflight_local = inflight
    return server


class FirstServer:
    """Algorithm stub: chooses the first healthy candidate and records what it saw."""

    def __init__(self):
        self.seen = []

    def choose_server(self, request, servers):
        # The real algorithms all filter on health; the stub must too, or it
        # would route to a server the collector has marked dead.
        healthy = [s for s in servers if s.healthy]
        self.seen.append([s.server_id for s in healthy])
        return ServerChoice(server=healthy[0], reason="first") if healthy else None


def make_request() -> RequestContext:
    return RequestContext(
        request_id="r1", method="GET", path="/ping", req_type=RequestType.LIGHT
    )


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

def test_circuit_opens_on_the_threshold_failure():
    server = make_server("s1")

    for _ in range(resilience.BREAKER_THRESHOLD - 1):
        resilience.record_outcome(server, ok=False)

    assert not resilience.circuit_open(server)
    resilience.record_outcome(server, ok=False)
    assert resilience.circuit_open(server)
    assert server.circuit_opens == 1


def test_a_success_resets_the_failure_count_below_the_threshold():
    server = make_server("s1")
    resilience.record_outcome(server, ok=False)
    resilience.record_outcome(server, ok=False)
    resilience.record_outcome(server, ok=True)

    resilience.record_outcome(server, ok=False)
    assert not resilience.circuit_open(server), "two failures, not three"


def test_open_circuit_is_excluded_from_the_candidates():
    good = make_server("good")
    bad = make_server("bad")
    for _ in range(resilience.BREAKER_THRESHOLD):
        resilience.record_outcome(bad, ok=False, now=time.time())

    algorithm = FirstServer()
    choice = resilience.choose(make_request(), [bad, good], algorithm)

    assert choice.server is good
    assert algorithm.seen == [["good"]]


def test_a_failed_probe_after_the_cooldown_reopens_the_circuit():
    server = make_server("s1")
    for _ in range(resilience.BREAKER_THRESHOLD):
        resilience.record_outcome(server, ok=False, now=100.0)

    assert not resilience.circuit_open(server, now=100.0 + resilience.BREAKER_COOLDOWN_S + 1)
    resilience.record_outcome(server, ok=False, now=111.0)

    assert resilience.circuit_open(server, now=111.0)
    assert server.circuit_opens == 1, "re-opening the same incident is not a new one"


def test_a_successful_probe_closes_the_circuit_and_starts_slow_start():
    server = make_server("s1")
    for _ in range(resilience.BREAKER_THRESHOLD):
        resilience.record_outcome(server, ok=False, now=100.0)

    resilience.record_outcome(server, ok=True, now=200.0)

    assert not server.circuit_tripped
    assert server.consecutive_failures == 0
    assert server.slow_start_until == pytest.approx(200.0 + resilience.SLOW_START_S)


# ---------------------------------------------------------------------------
# Slow start
# ---------------------------------------------------------------------------

def test_slow_start_fraction_ramps_from_a_tenth_to_full():
    server = make_server("s1")
    now = 1000.0
    server.slow_start_until = now + resilience.SLOW_START_S

    assert resilience.slow_start_fraction(server, now) == pytest.approx(0.1)
    assert resilience.slow_start_fraction(
        server, now + resilience.SLOW_START_S / 2
    ) == pytest.approx(0.5)
    assert resilience.slow_start_fraction(server, now + resilience.SLOW_START_S + 1) == 1.0


def test_a_ramping_server_is_dropped_while_it_ramps_and_kept_afterwards():
    ramping = make_server("ramping")
    steady = make_server("steady")
    ramping.slow_start_until = time.time() + resilience.SLOW_START_S

    # At the start of the ramp only the lowest decile of draws is admitted, so
    # a high draw drops it but a low one keeps it.
    dropped = resilience.choose(
        make_request(), [ramping, steady], FirstServer(), rng=lambda: 0.99
    )
    kept = resilience.choose(
        make_request(), [ramping, steady], FirstServer(), rng=lambda: 0.0
    )

    assert dropped.server is steady
    assert kept.server is ramping


# ---------------------------------------------------------------------------
# Stale feed
# ---------------------------------------------------------------------------

def test_feed_age_is_the_newest_snapshot():
    old = make_server("old", age_s=10.0)
    new = make_server("new", age_s=0.5)

    assert resilience.feed_age([old, new]) == pytest.approx(0.5, abs=0.2)
    assert not resilience.feed_is_stale([old, new])


def test_a_stale_feed_routes_on_the_last_snapshot_without_consulting_the_algorithm():
    stale = make_server("stale", age_s=10.0)
    algorithm = FirstServer()

    choice = resilience.choose(make_request(), [stale], algorithm)

    assert choice.server is stale
    assert choice.reason.startswith("stale-feed")
    assert algorithm.seen == []


def test_a_stale_feed_with_a_dead_cluster_still_refuses_to_route():
    # The collector labels servers unhealthy after repeated poll failures; a
    # dead cluster must not be routed to just because the snapshots are old.
    dead = make_server("dead", age_s=10.0, healthy=False)

    assert resilience.choose(make_request(), [dead], FirstServer()) is None


def test_a_fresh_feed_uses_the_algorithm_as_before():
    server = make_server("s1", age_s=0.2)
    algorithm = FirstServer()

    choice = resilience.choose(make_request(), [server], algorithm)

    assert choice.reason == "first"
    assert algorithm.seen == [["s1"]]


# ---------------------------------------------------------------------------
# Retry and least-loaded fallback
# ---------------------------------------------------------------------------

def test_a_retry_cannot_choose_the_server_that_just_failed():
    first = make_server("first")
    second = make_server("second")

    choice = resilience.choose(make_request(), [first, second], FirstServer(), tried=["first"])

    assert choice.server is second


def test_least_loaded_accounts_for_server_weight():
    big = make_server("big", inflight=2, weight=2.0)
    small = make_server("small", inflight=2, weight=1.0)

    # 2 in flight at weight 2 is the same queue as 1 in flight at weight 1.
    assert resilience.least_loaded([small, big]) is big


def test_retrying_is_limited_to_server_side_failures():
    assert 500 in resilience.RETRYABLE_STATUS
    assert 503 in resilience.RETRYABLE_STATUS
    assert 404 not in resilience.RETRYABLE_STATUS
    assert 200 not in resilience.RETRYABLE_STATUS

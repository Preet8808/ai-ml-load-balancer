"""
Serving safeguards: circuit breaking, slow start, and the stale-feed fallback.

The routing algorithms decide *where* a request should go. This module decides
whether a server is currently worth choosing, and what to do when the metrics
feed stops. Keeping it out of the algorithms means every policy gets the same
protection, and the policies stay functions of the state they are handed.

A request meets three mechanisms, in this order:

- **Slow start.** A server that has just come back receives a growing share of
  traffic over SLOW_START_S instead of its full weight immediately. Without it
  a recovered backend takes a full share of connections while its caches and
  pools are still cold.
- **Circuit breaker.** BREAKER_THRESHOLD consecutive proxy failures (connection
  errors, timeouts, 5xx) open that server's circuit for BREAKER_COOLDOWN_S.
  The collector's health checks also fail a dead backend out, but at a 2 s
  interval and a 3-failure threshold that takes about 6 s; the breaker reacts
  in milliseconds.
- **Stale-feed fallback.** When every metrics snapshot is older than
  STALE_AFTER_S the collector has stopped producing fresh state. The proxy then
  routes on the last known snapshot with least-connections instead of refusing
  to route: a monitoring outage should not become a serving outage. This is the
  plan's M6 criterion - "stopping the metrics feed triggers a switch to Least
  Connections without errors" - made explicit and countable.

Retry policy lives here as a constant too: one retry on a 5xx, on a different
server, is what turns a single failing backend into a slow request rather than
a failed one. The trade-off is documented in docs/resilience.md: a request whose
response was lost may be executed twice.
"""

from __future__ import annotations

import os
import random
import time
from typing import Iterable, List, Optional, Sequence

from balancer.interfaces import RequestContext, ServerChoice, ServerInfo

# How old the newest snapshot may be before routing is considered to be running
# on a dead feed. The collector polls every second, so 3 s means three missed
# polls.
STALE_AFTER_S = float(os.environ.get("STALE_AFTER_S", "3.0"))

# Consecutive proxy failures that open a circuit, and how long it stays open
# before one probe is allowed through.
BREAKER_THRESHOLD = int(os.environ.get("BREAKER_THRESHOLD", "3"))
BREAKER_COOLDOWN_S = float(os.environ.get("BREAKER_COOLDOWN_S", "10"))

# How long a recovered server is ramped up from 10% to its full share.
SLOW_START_S = float(os.environ.get("SLOW_START_S", "10"))

# Responses that count as a server failure for the breaker and are retried once.
RETRYABLE_STATUS = frozenset({500, 502, 503, 504})


def circuit_open(server: ServerInfo, now: Optional[float] = None) -> bool:
    """True while this server's circuit is open and it must not be chosen."""
    return (now if now is not None else time.time()) < server.circuit_open_until


def feed_age(servers: Sequence[ServerInfo], now: Optional[float] = None) -> Optional[float]:
    """Age in seconds of the *newest* snapshot, or None if nothing was ever polled."""
    now = now if now is not None else time.time()
    ages = [now - s.metrics.timestamp for s in servers if s.metrics is not None]
    return min(ages) if ages else None


def feed_is_stale(servers: Sequence[ServerInfo], now: Optional[float] = None) -> bool:
    """True when every server's snapshot is older than STALE_AFTER_S."""
    age = feed_age(servers, now)
    return age is not None and age > STALE_AFTER_S


def slow_start_fraction(server: ServerInfo, now: Optional[float] = None) -> float:
    """Share of its normal traffic the server should receive right now, 0.1-1.0."""
    remaining = server.slow_start_until - (now if now is not None else time.time())
    if remaining <= 0:
        return 1.0
    return max(0.1, 1.0 - remaining / SLOW_START_S)


def record_outcome(server: ServerInfo, ok: bool, now: Optional[float] = None) -> None:
    """
    Feed one proxied request's outcome into the server's breaker state.

    A success while the circuit is tripped closes it and starts the slow-start
    ramp. A failure while tripped re-opens it immediately: the probe that the
    cooldown allowed failed, so the server is not back yet.
    """
    now = now if now is not None else time.time()

    if ok:
        if server.circuit_tripped:
            server.circuit_tripped = False
            server.circuit_open_until = 0.0
            server.slow_start_until = now + SLOW_START_S
        server.consecutive_failures = 0
        return

    server.consecutive_failures += 1
    if server.circuit_tripped or server.consecutive_failures >= BREAKER_THRESHOLD:
        if not server.circuit_tripped:
            server.circuit_opens += 1
        server.circuit_tripped = True
        server.circuit_open_until = now + BREAKER_COOLDOWN_S


def least_loaded(servers: Iterable[ServerInfo]) -> Optional[ServerInfo]:
    """
    Pick the server with the least in-flight work per unit of weight.

    Used only when normal routing cannot answer - a stale feed, or a stale
    snapshot for a server the collector has not re-polled. It deliberately
    ignores freshness, because the caller has already decided that no fresher
    information exists.
    """
    candidates = [s for s in servers if s.metrics is not None]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda s: (s.inflight / max(s.weight, 1e-6), s.metrics.cpu_pct),
    )


def _apply_slow_start(
    servers: Sequence[ServerInfo],
    now: float,
    rng,
) -> List[ServerInfo]:
    """Drop a deterministic-random share of each ramping server's traffic."""
    kept: List[ServerInfo] = []
    for server in servers:
        if server.slow_start_until > now and rng() >= slow_start_fraction(server, now):
            continue
        kept.append(server)
    return kept


def choose(
    context: RequestContext,
    servers: Sequence[ServerInfo],
    algorithm,
    tried: Sequence[str] = (),
    now: Optional[float] = None,
    rng=random.random,
) -> Optional[ServerChoice]:
    """
    Choose a server with the safeguards applied.

    `tried` holds server ids already attempted for this request, so a retry
    cannot go back to the backend that just failed. Returns None when nothing
    can serve, which the proxy turns into a 503.
    """
    now = now if now is not None else time.time()
    skipped = set(tried)

    pool = [s for s in servers if s.server_id not in skipped]
    pool = [s for s in pool if not circuit_open(s, now)]
    pool = _apply_slow_start(pool, now, rng)

    if not pool:
        return None

    if feed_is_stale(servers, now):
        # Serve from the last snapshot rather than failing every request. Only
        # servers the collector last saw healthy are considered, so a dead
        # cluster still answers 503 instead of being routed to blindly. The
        # algorithm is not consulted at all: its inputs are the stale snapshot.
        alive = [s for s in pool if s.metrics is not None and s.metrics.healthy]
        server = least_loaded(alive)
        if server is None:
            return None
        return ServerChoice(server=server, reason="stale-feed(least_conn)")

    return algorithm.choose_server(context, pool)

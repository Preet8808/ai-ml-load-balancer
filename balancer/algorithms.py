"""
Routing algorithms for the AI/ML Load Balancer.

Every algorithm implements choose_server(request, servers) so that
benchmarks can swap algorithms by changing a single config value.
"""

from __future__ import annotations

import itertools
import random
import math
from typing import List

from balancer.interfaces import (
    Algorithm,
    RequestContext,
    ServerChoice,
    ServerInfo,
)


class Random(Algorithm):
    """
    Random algorithm - uniformly random choice among healthy servers.
    """

    name = "random"

    def choose_server(
        self,
        request: RequestContext,
        servers: List[ServerInfo],
    ) -> ServerChoice | None:
        healthy = [s for s in servers if s.healthy]
        if not healthy:
            return None

        server = random.choice(healthy)
        server.sent_since_snapshot += 1

        return ServerChoice(server=server, reason="random")


class RoundRobin(Algorithm):
    """
    Round Robin algorithm that cycles through servers in order.
    Weights are used to determine the relative number of times each server
    is selected in the cycle.
    """

    name = "round_robin"

    def __init__(self):
        self._server_ids = None
        self._cycle = None

    def _rebuild_cycle(self, servers: List[ServerInfo]):
        """Rebuild the round-robin cycle whenever the healthy server set changes."""
        self._cycle = itertools.cycle(list(servers))

    def choose_server(
        self,
        request: RequestContext,
        servers: List[ServerInfo],
    ) -> ServerChoice | None:
        healthy = [s for s in servers if s.healthy]
        if not healthy:
            return None

        current_ids = [s.server_id for s in healthy]
        if current_ids != self._server_ids:
            self._server_ids = current_ids
            self._rebuild_cycle(healthy)

        server = next(self._cycle)
        server.sent_since_snapshot += 1

        return ServerChoice(server=server, reason="round_robin")


class WeightedRoundRobin(Algorithm):
    """
    Weighted Round Robin - cycles through servers with weights
    proportional to their CPU limits (or configured weights).
    """

    name = "weighted_rr"

    def __init__(self):
        self._servers = None
        self._cycle = None

    def _rebuild_cycle(self, servers: List[ServerInfo]):
        weighted = []
        for server in servers:
            weight = max(1, int(server.weight))
            for _ in range(weight):
                weighted.append(server)
        self._cycle = itertools.cycle(weighted)

    def choose_server(
        self,
        request: RequestContext,
        servers: List[ServerInfo],
    ) -> ServerChoice | None:
        healthy = [s for s in servers if s.healthy]
        if not healthy:
            return None

        server_ids = [s.server_id for s in healthy]
        if self._servers != server_ids:
            self._servers = server_ids
            self._rebuild_cycle(healthy)

        server = next(self._cycle)
        server.sent_since_snapshot += 1

        return ServerChoice(server=server, reason="weighted_rr")


class LeastConnections(Algorithm):
    """
    Least Connections algorithm that routes to the server with the fewest
    in-flight requests.
    """

    name = "least_conn"

    def choose_server(
        self,
        request: RequestContext,
        servers: List[ServerInfo],
    ) -> ServerChoice | None:
        healthy = [s for s in servers if s.healthy]
        if not healthy:
            return None

        server = min(healthy, key=lambda s: s.metrics.inflight_requests if s.metrics else 0)
        server.sent_since_snapshot += 1

        return ServerChoice(server=server, reason="least_conn")


class WeightedLeastConnections(Algorithm):
    """
    Weighted Least Connections - routes to the server with the fewest
    in-flight requests divided by its weight.
    """

    name = "weighted_least_conn"

    def choose_server(
        self,
        request: RequestContext,
        servers: List[ServerInfo],
    ) -> ServerChoice | None:
        healthy = [s for s in servers if s.healthy]
        if not healthy:
            return None

        server = min(healthy, key=lambda s: (
            s.metrics.inflight_requests / s.weight if s.metrics else 0
        ))
        server.sent_since_snapshot += 1

        return ServerChoice(server=server, reason="weighted_least_conn")


class LeastResponseTime(Algorithm):
    """
    Least Response Time algorithm that routes to the server with the
    lowest recent average response time.
    """

    name = "least_response_time"

    def choose_server(
        self,
        request: RequestContext,
        servers: List[ServerInfo],
    ) -> ServerChoice | None:
        healthy = [s for s in servers if s.healthy]
        if not healthy:
            return None

        server = min(healthy, key=lambda s: (
            s.metrics.rt_ewma_ms if s.metrics else float("inf")
        ))
        server.sent_since_snapshot += 1

        return ServerChoice(server=server, reason="least_response_time")


# Registry of all available algorithms
ALGORITHMS = {
    "random": Random,
    "round_robin": RoundRobin,
    "weighted_rr": WeightedRoundRobin,
    "least_conn": LeastConnections,
    "weighted_least_conn": WeightedLeastConnections,
    "least_response_time": LeastResponseTime,
}


def get_algorithm(name: str) -> Algorithm:
    """Create the routing algorithm specified by name."""
    cls = ALGORITHMS.get(name)
    if cls is None:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(ALGORITHMS.keys())}")
    return cls()
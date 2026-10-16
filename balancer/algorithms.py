"""
Routing algorithms for the AI/ML Load Balancer.

Every algorithm implements choose_server(request, servers) so that
benchmarks can swap algorithms by changing a single config value.
"""

from __future__ import annotations

import itertools
from typing import List

from balancer.interfaces import (
    Algorithm,
    RequestContext,
    ServerChoice,
    ServerInfo,
)


class RoundRobin(Algorithm):
    """
    Round Robin algorithm that cycles through servers in order.
    Weights are used to determine the relative number of times each server
    is selected in the cycle.
    """

    name = "round_robin"

    def __init__(self):
        self._servers = None
        self._cycle = None

    def _rebuild_cycle(self, servers: List[ServerInfo]):
        """Rebuild the weighted cycle whenever the server list changes."""
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
        """
        Select a server for the given request using round robin.
        Returns None if no healthy servers are available.
        """
        healthy = [s for s in servers if s.healthy]
        if not healthy:
            return None

        if self._servers is not healthy:
            self._servers = healthy
            self._rebuild_cycle(healthy)

        server = next(self._cycle)
        server.sent_since_snapshot += 1

        return ServerChoice(server=server, reason="round_robin")


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

        if self._servers is not healthy:
            self._servers = healthy
            self._rebuild_cycle(healthy)

        server = next(self._cycle)
        server.sent_since_snapshot += 1

        return ServerChoice(server=server, reason="weighted_rr")
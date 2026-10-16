"""
Core interfaces for the AI/ML Load Balancer.

This module defines the contract between the proxy, the decision engine,
the metrics collector, and the backends. All routing algorithms implement
the same interface so benchmarks can swap algorithms by changing a single config value.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RequestType(str, Enum):
    """Type of incoming request, used as a feature for ML routing."""
    LIGHT = "light"       # /ping, about 5 ms — cheap health check
    CPU = "cpu"           # small and large computation
    IO = "io"             # simulated database wait with sleep
    PAYLOAD = "payload"   # large response body


@dataclass
class RequestContext:
    """
    Information about an incoming request, used for routing decisions.
    """
    request_id: str
    method: str
    path: str
    req_type: RequestType
    payload_kb: float = 0.0
    client_ip: str = ""


@dataclass
class ServerMetrics:
    """
    A snapshot of one backend server's state, collected every second.
    """
    server_id: str
    timestamp: float
    cpu_pct: float          # CPU utilisation (0-100)
    mem_pct: float          # Memory used (0-100)
    load_avg: float         # 1-minute load average
    net_io_bytes_per_s: float
    inflight_requests: int
    req_rate_5s: float      # requests per second over last 5 s
    rt_ewma_ms: float       # exponentially weighted moving average of response time
    rt_p95_5s_ms: float     # p95 response time over last 5 s
    error_rate_5s: float    # error rate over last 5 s
    cpu_limit: float        # CPU limit (e.g. 2.0 means two cores)
    cpu_delta_5s: float     # change in CPU over last 5 s (trend)
    healthy: bool = True
    stale: bool = False     # True if snapshot is older than 3 s


@dataclass
class ServerInfo:
    """
    Full server descriptor including static configuration and live metrics.
    """
    server_id: str
    host: str
    port: int
    weight: float = 1.0
    cpu_limit: float = 1.0
    mem_limit_mb: float = 512.0
    metrics: ServerMetrics | None = None
    # Count of requests routed to this server since the last metrics snapshot
    sent_since_snapshot: int = 0
    health_check_failures: int = 0
    health_check_successes: int = 0

    @property
    def healthy(self) -> bool:
        return (
            self.metrics is not None
            and self.metrics.healthy
            and not self.metrics.stale
        )


@dataclass
class ServerChoice:
    """
    Result of a routing decision.
    """
    server: ServerInfo
    reason: str = ""
    predicted_rt_ms: float | None = None


class Algorithm(ABC):
    """
    Abstract base class for all routing algorithms.

    Every algorithm implements choose_server(request, servers) so that
    benchmarks can swap algorithms by changing a single config value.
    """

    name: str = "base"

    @abstractmethod
    def choose_server(
        self,
        request: RequestContext,
        servers: list[ServerInfo],
    ) -> ServerChoice | None:
        """
        Select a server for the given request.

        Returns a ServerChoice with the selected server, or None if no
        suitable server is available (the proxy should respond with 503).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Log schema — one row per (request, chosen server) decision
# ---------------------------------------------------------------------------
# Columns:
#   timestamp,
#   run_id,
#   req_type,
#   payload_kb,
#   server_id,
#   srv_cpu_pct,
#   srv_mem_pct,
#   inflight,
#   req_rate_5s,
#   rt_ewma_ms,
#   rt_p95_5s_ms,
#   err_rate_5s,
#   cpu_limit,
#   cpu_delta_5s,
#   response_ms,
#   status
#
# Example row (illustrative):
#   1729084201.3, r07, cpu, 2, s2, 63.0, 41.5, 7, 22, 118.0, 240.0, 0.00, 1.0, +9.5, 152.4, 200
#
# ---------------------------------------------------------------------------
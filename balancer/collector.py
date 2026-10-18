import asyncio
import time
import httpx
import logging
import collections
from typing import List, Dict, Deque

from .interfaces import ServerInfo, ServerMetrics

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects metrics from backend servers and maintains their health status.

    The collector is a *transport*: every value it publishes comes from the
    backend's own /metrics endpoint. It only derives `cpu_delta_5s` (a trend
    over the 60 s history) and keeps the health state machine.

    Earlier versions synthesised `req_rate_5s`, `rt_ewma_ms` and
    `rt_p95_5s_ms` from the latency of the polling request itself, which made
    them uncorrelated with real response times and unusable as ML features.
    """

    def __init__(
        self,
        servers: List[ServerInfo],
        poll_interval: float = 1.0,
        poll_timeout: float = 2.0,
        stale_threshold_s: float = 3.0,
        failures_to_unhealthy: int = 3,
        successes_to_healthy: int = 3,
    ):
        self.servers = servers
        self.poll_interval = poll_interval
        self.stale_threshold_s = stale_threshold_s
        self.failures_to_unhealthy = failures_to_unhealthy
        self.successes_to_healthy = successes_to_healthy
        self._running = False
        self._client = httpx.AsyncClient(timeout=poll_timeout)
        self._history: Dict[str, Deque[Dict]] = {
            s.server_id: collections.deque(maxlen=60) for s in servers
        }
        self._task: asyncio.Task | None = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"Metrics collector started for {len(self.servers)} servers")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        await self._client.aclose()

    async def _poll_loop(self):
        while self._running:
            started = time.time()
            tasks = [self._poll_server(server) for server in self.servers]
            await asyncio.gather(*tasks)
            elapsed = time.time() - started
            await asyncio.sleep(max(0.0, self.poll_interval - elapsed))

    def _calculate_rolling_avg(self, server_id: str, field: str, window_s: int) -> float:
        history = self._history.get(server_id)
        if not history:
            return 0.0

        now = time.time()
        relevant = [h[field] for h in history if now - h["timestamp"] <= window_s and field in h]
        if not relevant:
            return float(history[-1].get(field, 0.0))
        return sum(relevant) / len(relevant)

    @staticmethod
    def _num(data: Dict, key: str, default: float = 0.0) -> float:
        value = data.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    async def _poll_server(self, server: ServerInfo):
        url = f"http://{server.host}:{server.port}/metrics"
        try:
            started = time.perf_counter()
            response = await self._client.get(url)
            poll_latency_ms = (time.perf_counter() - started) * 1000.0

            if response.status_code != 200:
                await self._handle_failure(server)
                return

            data = response.json()
            now = time.time()
            data["timestamp"] = now
            self._history[server.server_id].append(data)

            previous = server.metrics
            was_unhealthy = previous is not None and not previous.healthy

            server.health_check_failures = 0
            if was_unhealthy:
                server.health_check_successes += 1
                healthy_now = server.health_check_successes >= self.successes_to_healthy
                if healthy_now:
                    logger.info(f"Server {server.server_id} is now HEALTHY")
            else:
                server.health_check_successes = 0
                healthy_now = True

            cpu_now = self._num(data, "cpu_pct")
            cpu_avg_5s = self._calculate_rolling_avg(server.server_id, "cpu_pct", 5)

            server.metrics = ServerMetrics(
                server_id=server.server_id,
                timestamp=now,
                cpu_pct=cpu_now,
                mem_pct=self._num(data, "mem_pct"),
                load_avg=self._num(data, "load_avg"),
                net_io_bytes_per_s=(
                    self._num(data, "net_io_bytes_per_s")
                    or self._num(data, "net_io_bytes_recv") + self._num(data, "net_io_bytes_sent")
                ),
                inflight_requests=int(self._num(data, "inflight")),
                req_rate_5s=self._num(data, "req_rate_5s"),
                rt_ewma_ms=self._num(data, "rt_ewma_ms"),
                rt_p95_5s_ms=self._num(data, "rt_p95_5s_ms"),
                error_rate_5s=self._num(data, "error_rate_5s"),
                cpu_limit=server.cpu_limit,
                cpu_delta_5s=cpu_now - cpu_avg_5s,
                healthy=healthy_now,
                stale=False,
                poll_latency_ms=poll_latency_ms,
            )

            server.sent_since_snapshot = 0

        except Exception as e:
            logger.debug(f"Error polling {server.server_id}: {e}")
            await self._handle_failure(server)

    async def _handle_failure(self, server: ServerInfo):
        server.health_check_successes = 0
        server.health_check_failures += 1

        if server.metrics is None:
            return

        if time.time() - server.metrics.timestamp > self.stale_threshold_s:
            server.metrics.stale = True

        if server.metrics.healthy and server.health_check_failures >= self.failures_to_unhealthy:
            server.metrics.healthy = False
            logger.warning(
                f"Server {server.server_id} marked UNHEALTHY after "
                f"{server.health_check_failures} failures"
            )

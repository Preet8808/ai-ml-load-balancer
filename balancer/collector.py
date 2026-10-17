import asyncio
import time
import httpx
import logging
import collections
from typing import Dict, List, Optional, Deque
from .interfaces import ServerInfo, ServerMetrics

logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    Collects metrics from backend servers and maintains their health status.
    Maintains a 60s history for rolling windows and calculates EWMA.
    """
    def __init__(self, servers: List[ServerInfo], poll_interval: float = 1.0, alpha: float = 0.3):
        self.servers = servers
        self.poll_interval = poll_interval
        self.alpha = alpha  # EWMA smoothing factor
        self._running = False
        self._client = httpx.AsyncClient(timeout=0.5)
        # History stores metrics for the last 60 seconds
        self._history: Dict[str, Deque[Dict]] = {s.server_id: collections.deque(maxlen=60) for s in servers}
        
    async def start(self):
        self._running = True
        asyncio.create_task(self._poll_loop())
        logger.info(f"Metrics collector started for {len(self.servers)} servers")

    async def stop(self):
        self._running = False
        await self._client.aclose()

    async def _poll_loop(self):
        while self._running:
            tasks = [self._poll_server(server) for server in self.servers]
            await asyncio.gather(*tasks)
            await asyncio.sleep(self.poll_interval)

    def _calculate_rolling_avg(self, server_id: str, field: str, window_s: int) -> float:
        history = self._history.get(server_id, [])
        if not history:
            return 0.0
        
        now = time.time()
        relevant = [h[field] for h in history if now - h['timestamp'] <= window_s]
        if not relevant:
            return history[-1][field] if history else 0.0
            
        return sum(relevant) / len(relevant)

    async def _poll_server(self, server: ServerInfo):
        url = f"http://{server.host}:{server.port}/metrics"
        try:
            start_time = time.time()
            response = await self._client.get(url)
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                data['timestamp'] = time.time()
                data['latency_ms'] = latency_ms
                
                # Update history
                self._history[server.server_id].append(data)
                
                # Calculate EWMA for response time
                old_rt = server.metrics.rt_ewma_ms if server.metrics else latency_ms
                new_rt = (self.alpha * latency_ms) + (1 - self.alpha) * old_rt
                
                # Calculate rolling stats
                cpu_5s = self._calculate_rolling_avg(server.server_id, 'cpu_pct', 5)
                
                # Update metrics
                metrics = ServerMetrics(
                    server_id=server.server_id,
                    timestamp=data['timestamp'],
                    cpu_pct=data.get("cpu_pct", 0.0),
                    mem_pct=data.get("mem_pct", 0.0),
                    load_avg=data.get("load_avg", 0.0),
                    net_io_bytes_per_s=data.get("net_io_bytes_recv", 0.0) + data.get("net_io_bytes_sent", 0.0),
                    inflight_requests=server.metrics.inflight_requests if server.metrics else 0,
                    req_rate_5s=self._calculate_rolling_avg(server.server_id, 'latency_ms', 5), # Not really rate, but placeholder
                    rt_ewma_ms=new_rt,
                    rt_p95_5s_ms=latency_ms, # Real p95 would need more work, using current for now
                    error_rate_5s=0.0,
                    cpu_limit=server.cpu_limit,
                    cpu_delta_5s=data.get("cpu_pct", 0.0) - cpu_5s,
                    healthy=True,
                    stale=False
                )
                server.metrics = metrics
                
                # Reset "sent since snapshot" counter
                server.sent_since_snapshot = 0
                
                # Health state machine: 3 successes to restore
                server.health_check_failures = 0
                if not server.metrics.healthy:
                    server.health_check_successes += 1
                    if server.health_check_successes >= 3:
                        server.metrics.healthy = True
                        server.metrics.stale = False
                        logger.info(f"Server {server.server_id} is now HEALTHY")
            else:
                await self._handle_failure(server)
                
        except Exception as e:
            logger.debug(f"Error polling {server.server_id}: {e}")
            await self._handle_failure(server)

    async def _handle_failure(self, server: ServerInfo):
        server.health_check_successes = 0
        server.health_check_failures += 1
        
        # Mark stale if missing for 3 seconds
        if server.metrics:
            if time.time() - server.metrics.timestamp > 3.0:
                server.metrics.stale = True
        
        # Health state machine: 3 failures to remove
        if server.metrics and server.metrics.healthy and server.health_check_failures >= 3:
            server.metrics.healthy = False
            logger.warning(f"Server {server.server_id} marked UNHEALTHY after 3 failures")

"""
Probe mode for paired counterfactual labels (Phase 3, Step 3).

Sends identical light requests to ALL servers simultaneously so we can
compare which server was actually fastest. This gives us paired labels
that allow computing top-1 ranking accuracy and Spearman correlation.

Usage:
    from bench.probe_mode import ProbeRunner
    runner = ProbeRunner(servers_config)
    await runner.run_probe(rounds=100)
"""

import asyncio
import time
import json
import httpx
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ProbeResult:
    """Result of a single probe round — one request sent to all servers."""
    timestamp: float
    server_id: str
    response_ms: float
    status: int
    cpu_pct: float
    mem_pct: float
    inflight: int
    cpu_limit: float


class ProbeRunner:
    """
    Sends identical /ping requests to every server at the same instant,
    records response times, and stores results to Parquet for ranking analysis.
    """

    def __init__(self, servers: List[Dict[str, Any]], output_dir: str = "logs"):
        """
        Args:
            servers: List of dicts with 'server_id', 'host', 'port', 'cpu_limit'
            output_dir: Directory to write probe results
        """
        self.servers = servers
        self.output_dir = output_dir
        self.results: List[Dict[str, Any]] = []
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=5.0)
        return self._client

    async def _probe_server(self, server: Dict[str, Any]) -> Dict[str, Any]:
        """Send a single /ping to one server and measure response time."""
        url = f"http://{server['host']}:{server['port']}/ping"
        client = await self._get_client()

        try:
            # Fetch metrics first for the feature snapshot
            metrics_resp = await client.get(
                f"http://{server['host']}:{server['port']}/metrics"
            )
            metrics_data = metrics_resp.json() if metrics_resp.status_code == 200 else {}

            # Send the actual probe request
            start = time.time()
            resp = await client.get(url)
            elapsed_ms = (time.time() - start) * 1000

            return {
                "timestamp": start,
                "server_id": server["server_id"],
                "response_ms": elapsed_ms,
                "status": resp.status_code,
                "cpu_pct": metrics_data.get("cpu_pct", 0.0),
                "mem_pct": metrics_data.get("mem_pct", 0.0),
                "inflight": metrics_data.get("inflight", 0),
                "cpu_limit": server.get("cpu_limit", 1.0),
            }
        except Exception as e:
            return {
                "timestamp": time.time(),
                "server_id": server["server_id"],
                "response_ms": 5000.0,  # timeout penalty
                "status": 0,
                "cpu_pct": 0.0,
                "mem_pct": 0.0,
                "inflight": 0,
                "cpu_limit": server.get("cpu_limit", 1.0),
            }

    async def run_probe(self, rounds: int = 100, interval: float = 1.0) -> List[Dict[str, Any]]:
        """
        Run multiple probe rounds. Each round sends to all servers concurrently.

        Args:
            rounds: Number of probe rounds
            interval: Seconds between rounds

        Returns:
            List of result dicts
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        for r in range(rounds):
            # Fire all probes simultaneously
            tasks = [self._probe_server(s) for s in self.servers]
            results = await asyncio.gather(*tasks)
            self.results.extend(results)
            print(f"  Probe round {r+1}/{rounds} — {len(results)} servers probed")
            await asyncio.sleep(interval)

        # Save to parquet
        self._save_results()
        return self.results

    def _save_results(self):
        """Save probe results to Parquet file."""
        if not self.results:
            return

        df = pd.DataFrame(self.results)
        filename = f"probe_{int(time.time())}.parquet"
        filepath = f"{self.output_dir}/{filename}"
        df.to_parquet(filepath, index=False)
        print(f"  Saved {len(self.results)} probe results to {filepath}")

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def get_ranking_stats(self) -> Dict[str, Any]:
        """
        Compute basic ranking statistics from probe results.
        Returns top-1 accuracy per round and average response times per server.
        """
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        # Group by timestamp (each timestamp = one probe round)
        # Find the fastest server in each round
        df_round = df.groupby("timestamp").apply(
            lambda g: g.loc[g["response_ms"].idxmin(), "server_id"],
            include_groups=False
        )
        top1_counts = df_round.value_counts().to_dict()

        avg_by_server = df.groupby("server_id")["response_ms"].mean().to_dict()

        return {
            "total_rounds": len(df_round),
            "top1_wins": top1_counts,
            "avg_response_ms": avg_by_server,
        }

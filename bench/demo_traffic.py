"""
Small host-driven traffic generator for live demos.

This is deliberately not a benchmark client: it runs on the host and is limited
by the Windows -> Docker forwarding path (about 60-70 requests/s however fast
the stack is). It exists so a demo can show routing behaviour without starting
Locust. Every number quoted in the report comes from `bench/run_matrix.py` or
`bench/ab_compare.py`.

Usage:
  python bench/demo_traffic.py --seconds 60 --rate 24
  python bench/demo_traffic.py --seconds 45 --rate 30 --url http://localhost:9090
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time

import httpx

ENDPOINTS = [("/ping", 0.40), ("/cpu", 0.30), ("/payload", 0.20), ("/io", 0.10)]
PATHS = [path for path, _ in ENDPOINTS]
WEIGHTS = [weight for _, weight in ENDPOINTS]


async def worker(
    client: httpx.AsyncClient,
    rate: float,
    stop_at: float,
    stats: dict,
    lock: asyncio.Lock,
) -> None:
    interval = 1.0 / rate
    while time.time() < stop_at:
        started = time.time()
        path = random.choices(PATHS, weights=WEIGHTS)[0]
        try:
            response = await client.get(path, timeout=10)
            async with lock:
                stats["reqs"] += 1
                stats["codes"][response.status_code] = stats["codes"].get(response.status_code, 0) + 1
        except Exception:
            async with lock:
                stats["errors"] += 1
        delay = started + interval - time.time()
        if delay > 0:
            await asyncio.sleep(delay)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Demo traffic (not a benchmark)")
    parser.add_argument("--url", default="http://localhost:9090")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--rate", type=float, default=24, help="requests/s across all workers")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    stats: dict = {"reqs": 0, "errors": 0, "codes": {}}
    lock = asyncio.Lock()
    stop_at = time.time() + args.seconds

    async with httpx.AsyncClient(base_url=args.url, limits=httpx.Limits(max_connections=32)) as client:
        try:
            algorithm = (await client.get("/admin/algorithm", timeout=5)).json()
            print(f"  algorithm: {algorithm.get('algorithm')}")
        except Exception as exc:
            print(f"  WARNING: cannot read /admin/algorithm: {exc}")

        per_worker = args.rate / args.concurrency
        tasks = [asyncio.create_task(worker(client, per_worker, stop_at, stats, lock))
                 for _ in range(args.concurrency)]

        while time.time() < stop_at:
            await asyncio.sleep(min(5, max(0.2, stop_at - time.time())))
            elapsed = args.seconds - max(0.0, stop_at - time.time())
            print(f"  {elapsed:4.0f}s  sent {stats['reqs']:5d}  errors {stats['errors']:3d}  "
                  f"codes {stats['codes']}", flush=True)

        await asyncio.gather(*tasks)

        try:
            data = (await client.get("/stats", timeout=10)).json()
            recent = (data.get("decisions") or {}).get("recent") or []
            counts: dict[str, int] = {}
            for decision in recent:
                key = decision.get("server_id") or decision.get("server") or "?"
                counts[key] = counts.get(key, 0) + 1
            print(f"  last {len(recent)} decisions by server: {counts}")
        except Exception as exc:
            print(f"  WARNING: cannot read /stats: {exc}")


if __name__ == "__main__":
    asyncio.run(main())

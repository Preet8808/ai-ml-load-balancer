"""Diagnose where proxy latency is actually spent.

Fires concurrent requests at one endpoint while sampling the balancer's view
of each backend, so we can tell whether time is spent inside the backend or
somewhere in between.
"""
import asyncio
import statistics
import sys
import time

import httpx

BALANCER = "http://localhost:9090"
ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "/cpu"
CONCURRENCY = int(sys.argv[2]) if len(sys.argv) > 2 else 40
SAMPLES = int(sys.argv[3]) if len(sys.argv) > 3 else 12


async def sample(observations):
    async with httpx.AsyncClient(timeout=10) as c:
        for _ in range(SAMPLES):
            try:
                r = await c.get(f"{BALANCER}/stats", timeout=5)
                for s in r.json()["servers"]:
                    m = s["metrics"] or {}
                    observations.append(
                        (s["server_id"], m.get("inflight_requests", 0),
                         m.get("cpu_pct", 0), m.get("rt_ewma_ms", 0))
                    )
            except Exception:
                pass
            await asyncio.sleep(0.25)


async def worker(client, latencies, errors):
    try:
        t0 = time.perf_counter()
        r = await client.get(f"{BALANCER}{ENDPOINT}", timeout=60)
        latencies.append((time.perf_counter() - t0) * 1000)
        if r.status_code != 200:
            errors.append(r.status_code)
    except Exception as exc:
        errors.append(type(exc).__name__)


async def main():
    observations = []
    sampler = asyncio.create_task(sample(observations))

    latencies, errors = [], []
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    async with httpx.AsyncClient(timeout=60, limits=limits) as client:
        await asyncio.gather(*[
            worker(client, latencies, errors) for _ in range(CONCURRENCY)
        ])

    await sampler

    latencies.sort()
    print(f"endpoint={ENDPOINT} concurrency={CONCURRENCY} n={len(latencies)} errors={len(errors)}")
    if latencies:
        print(f"  min={latencies[0]:.0f}ms  p50={statistics.median(latencies):.0f}ms  "
              f"p95={latencies[int(len(latencies) * 0.95) - 1]:.0f}ms  max={latencies[-1]:.0f}ms")

    if not observations:
        print("  no samples")
        return

    per_server = {}
    for sid, infl, cpu, ewma in observations:
        per_server.setdefault(sid, []).append((infl, cpu, ewma))

    print("\n  backend view while loaded (inflight / cpu% / ewma_ms):")
    for sid, rows in sorted(per_server.items()):
        infls = [r[0] for r in rows]
        cpus = [r[1] for r in rows]
        print(f"    {sid:9s} inflight avg={statistics.mean(infls):5.2f} "
              f"max={max(infls):3d}   cpu avg={statistics.mean(cpus):5.1f}  "
              f"ewma avg={statistics.mean([r[2] for r in rows]):7.1f}")

    print("\n  If inflight stays near zero while client latency is large,")
    print("  the time is being spent outside backend request handling.")


asyncio.run(main())
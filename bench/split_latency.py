"""Split proxy latency into: backend loop vs balancer loop vs container network.

Runs inside the balancer container and times three paths side by side:
  A  balancer -> backend /metrics   (what the collector's poll latency measures)
  B  balancer -> its own /stats     (balancer's own event loop responsiveness)
  C  balancer -> its own /ping      (full proxy path, loopback)
"""
import asyncio
import statistics
import subprocess
import sys
import time

SCRIPT = r'''
import asyncio, httpx, statistics, time

async def timeit(c, url, n=12):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        try:
            await c.get(url, timeout=30)
        except Exception:
            continue
        ts.append((time.perf_counter() - t) * 1000)
        await asyncio.sleep(0.1)
    return ts

async def main():
    limits = httpx.Limits(max_connections=64, max_keepalive_connections=64)
    async with httpx.AsyncClient(timeout=30, limits=limits) as c:
        await c.get("http://backend1:8000/health")
        await c.get("http://localhost:8080/ping")

        a = await timeit(c, "http://backend1:8000/metrics")
        b = await timeit(c, "http://localhost:8080/stats")
        d = await timeit(c, "http://localhost:8080/ping")

    def show(name, ts):
        if ts:
            print(f"  {name:42s} n={len(ts):3d} mean={statistics.mean(ts):8.1f}ms "
                  f"p50={statistics.median(ts):8.1f}ms max={max(ts):8.1f}ms")
        else:
            print(f"  {name:42s} no samples")

    show("A balancer -> backend /metrics", a)
    show("B balancer -> own /stats (loop test)", b)
    show("C balancer -> own /ping (full proxy)", d)

asyncio.run(main())
'''

result = subprocess.run(
    ["docker", "exec", "ai_ml_load_balancer-balancer-1", "python", "-c", SCRIPT],
    capture_output=True, text=True,
)
print(result.stdout or result.stderr)
sys.exit(result.returncode)
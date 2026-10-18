"""Run host-driven load and, from inside the balancer, time the same paths.

If loopback proxying stays fast while host-originated requests are slow, the
bottleneck is the Windows -> Docker Desktop forwarding path rather than the
balancer or the backends.
"""
import subprocess
import sys
import time

LOAD = r'''
import os, time
os.environ["TRAFFIC_PROFILE"] = "steady"
os.environ["TRAFFIC_USERS"] = "200"
os.environ["BENCH_DURATION"] = "35"
os.environ["SPAWN_RATE"] = "40"
from locust.env import Environment
'''

LOAD_CMD = [
    sys.executable, "-m", "locust",
    "-f", "bench/traffic_profiles.py",
    "--host", "http://localhost:9090",
    "--headless", "-t", "45s", "--only-summary", "--loglevel", "ERROR",
]

PROBE = r'''
import asyncio, httpx, statistics, time

async def timeit(c, url, n=10):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        try:
            await c.get(url, timeout=30)
        except Exception:
            pass
        ts.append((time.perf_counter() - t) * 1000)
        await asyncio.sleep(0.05)
    return ts

async def main():
    limits = httpx.Limits(max_connections=64, max_keepalive_connections=64)
    async with httpx.AsyncClient(timeout=30, limits=limits) as c:
        for name, url in [
            ("loopback proxy /ping   ", "http://localhost:8080/ping"),
            ("balancer -> backend    ", "http://backend1:8000/metrics"),
            ("loopback /stats (loop) ", "http://localhost:8080/stats"),
        ]:
            ts = await timeit(c, url)
            ts.sort()
            print(f"  {name} p50={statistics.median(ts):8.1f}ms  max={max(ts):8.1f}ms")

asyncio.run(main())
'''


def main():
    load = subprocess.Popen(LOAD_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(20)
        for round_no in range(3):
            print(f"probe {round_no + 1} (host load running):")
            result = subprocess.run(
                ["docker", "exec", "ai_ml_load_balancer-balancer-1", "python", "-c", PROBE],
                capture_output=True, text=True,
            )
            print(result.stdout or result.stderr)
            time.sleep(3)
    finally:
        load.terminate()
        try:
            load.wait(timeout=15)
        except subprocess.TimeoutExpired:
            load.kill()
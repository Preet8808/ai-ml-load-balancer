"""Dump each backend's cumulative counters from inside the balancer container."""
import asyncio
import subprocess
import sys

SCRIPT = r'''
import asyncio, httpx

async def main():
    async with httpx.AsyncClient(timeout=10) as c:
        for h in ["backend1", "backend2", "backend3", "backend4"]:
            m = (await c.get(f"http://{h}:8000/metrics")).json()
            print(
                f"{h}: total_requests={m['total_requests']:7d} uptime={m['uptime_s']:7.0f}s "
                f"rate5s={m['req_rate_5s']:6.2f} inflight={m['inflight']:3d} "
                f"cpu={m['cpu_pct']:5.1f} mem={m['mem_pct']:5.1f} ewma={m['rt_ewma_ms']:7.1f}"
            )

asyncio.run(main())
'''

result = subprocess.run(
    ["docker", "exec", "ai_ml_load_balancer-balancer-1", "python", "-c", SCRIPT],
    capture_output=True,
    text=True,
)
print(result.stdout or result.stderr)
sys.exit(result.returncode)
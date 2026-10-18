"""
Smoke-test the fault-injection endpoints through the compose network.

Verifies that /slow and /fault actually degrade a backend for a sustained
window (rather than for a single request), because a fault that does not
persist produces no error-rate signal and no routing lesson.
"""

import subprocess
import sys
import time

SCRIPT = r'''
import asyncio, httpx, time

async def main():
    async with httpx.AsyncClient(timeout=30) as c:
        # /slow should return immediately and then slow every later request
        t = time.perf_counter()
        r = await c.get("http://backend3:8000/slow", params={"ms": 300, "duration_s": 8})
        control_ms = (time.perf_counter() - t) * 1000
        print(f"  /slow control call: {r.status_code} in {control_ms:.0f}ms  {r.text}")

        for i in range(3):
            t = time.perf_counter()
            await c.get("http://backend3:8000/ping")
            print(f"  backend3 /ping while slow: {(time.perf_counter()-t)*1000:7.0f}ms")

        t = time.perf_counter()
        await c.get("http://backend1:8000/ping")
        print(f"  backend1 /ping (control) : {(time.perf_counter()-t)*1000:7.0f}ms")

        # /fault should produce a mix of 200 and 503 and be visible in /metrics
        r = await c.get("http://backend3:8000/fault", params={"error_rate": 0.5, "duration_s": 8})
        print(f"  /fault control call: {r.status_code}  {r.text}")

        codes = []
        for i in range(30):
            code = (await c.get("http://backend3:8000/ping")).status_code
            codes.append(code)
        print(f"  30 requests: {codes.count(200)} ok, {codes.count(503)} failed")

        for _ in range(12):
            m = (await c.get("http://backend3:8000/metrics")).json()
            if m["error_rate_5s"] > 0:
                break
            await asyncio.sleep(1)

        print(f"  backend3 metrics: err_rate_5s={m['error_rate_5s']:.3f} "
              f"inflight={m['inflight']} queue_depth={m['queue_depth']} "
              f"rt_ewma_ms={m['rt_ewma_ms']:.1f} rt_p95_5s_ms={m['rt_p95_5s_ms']:.1f}")

        # Clear the faults so the next experiment starts clean
        await c.get("http://backend3:8000/slow", params={"ms": 0, "duration_s": 0})
        await c.get("http://backend3:8000/fault", params={"error_rate": 0.0, "duration_s": 0})
        print("  faults cleared")

asyncio.run(main())
'''


def main():
    result = subprocess.run(
        ["docker", "exec", "ai_ml_load_balancer-balancer-1", "python", "-c", SCRIPT],
        capture_output=True, text=True,
    )
    print(result.stdout or result.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

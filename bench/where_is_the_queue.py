"""Compare where proxy requests are: balancer in-flight vs backend in-flight.

If the balancer reports hundreds of in-flight proxy requests while the backends
report a handful, the queue is between them (pool, event loop). If the balancer
also reports a handful, the load generator is not sending what we think.
"""
import subprocess
import sys

PROBE = r'''
import asyncio, httpx, json, statistics, time

async def main():
    async with httpx.AsyncClient(timeout=10) as c:
        bal_total, bal_per = [], []
        back_total = []
        for _ in range(14):
            try:
                st = (await c.get("http://localhost:8080/stats")).json()
                bal_total.append(st.get("inflight_total", -1))
                bal_per.append([s.get("inflight_local", 0) for s in st["servers"]])
            except Exception:
                pass

            bt = 0
            for h in ["backend1", "backend2", "backend3", "backend4"]:
                try:
                    m = (await c.get(f"http://{h}:8000/metrics")).json()
                    bt += m["inflight"]
                except Exception:
                    pass
            back_total.append(bt)
            await asyncio.sleep(0.4)

    print(f"  balancer inflight_total : mean={statistics.mean(bal_total):7.1f} "
          f"min={min(bal_total)} max={max(bal_total)}")
    print(f"  backends inflight (sum) : mean={statistics.mean(back_total):7.1f} "
          f"min={min(back_total)} max={max(back_total)}")
    if bal_per:
        print(f"  balancer per-server last sample: {bal_per[-1]}")
    print(f"  gap (waiting in the proxy) : {statistics.mean(bal_total) - statistics.mean(back_total):7.1f}")

asyncio.run(main())
'''


def main():
    load_cmd = [
        sys.executable, "-m", "locust",
        "-f", "bench/traffic_profiles.py",
        "--host", "http://localhost:9090",
        "--headless", "-t", "45s", "--only-summary", "--loglevel", "ERROR",
    ]
    load = subprocess.Popen(load_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import time
        time.sleep(22)
        result = subprocess.run(
            ["docker", "exec", "ai_ml_load_balancer-balancer-1", "python", "-c", PROBE],
            capture_output=True, text=True,
        )
        print(result.stdout or result.stderr)
    finally:
        load.terminate()
        try:
            load.wait(timeout=15)
        except subprocess.TimeoutExpired:
            load.kill()


if __name__ == "__main__":
    main()
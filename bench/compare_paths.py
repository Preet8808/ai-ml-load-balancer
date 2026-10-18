"""
Compare the balancer path against a direct backend path, from a separate container.

Everything runs inside the compose network, but the generator gets its own
cgroup. That distinction matters on this host: the machine is 2 physical cores
exposed as 4 vCPU, and the generator alone burns about one core, so running it
inside the balancer container starves the component under test and the resulting
"collapse" says nothing about the balancer.

The probe reports the generator's own CPU consumption so it is possible to tell
a loaded stack from a throttled generator, and includes a sequential mode for
the uncontended per-request floor.

Usage:
  python bench/compare_paths.py --sequential 200
  python bench/compare_paths.py --targets balancer:8080 backend1:8000 --users 1 10 50
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bench.quick_dataset import run_generator  # noqa: E402

IN_NETWORK = r'''
import argparse, asyncio, statistics, time

import httpx


def split_target(spec):
    host, _, port = spec.partition(":")
    return host, int(port or 8000)


async def worker(client, url, deadline, lat, codes):
    while time.perf_counter() < deadline:
        t = time.perf_counter()
        try:
            r = await client.get(url)
            codes[r.status_code] = codes.get(r.status_code, 0) + 1
        except Exception as exc:
            codes[type(exc).__name__] = codes.get(type(exc).__name__, 0) + 1
        lat.append((time.perf_counter() - t) * 1000.0)


def pct(lat, p):
    return lat[min(len(lat) - 1, int(len(lat) * p))] if lat else float("nan")


async def run(spec, path, users, duration):
    host, port = split_target(spec)
    url = f"http://{host}:{port}{path}"
    limits = httpx.Limits(max_connections=users + 50, max_keepalive_connections=users + 50)
    lat, codes = [], {}
    c0, w0 = time.process_time(), time.perf_counter()
    async with httpx.AsyncClient(timeout=30.0, limits=limits) as client:
        deadline = time.perf_counter() + duration
        await asyncio.gather(*[worker(client, url, deadline, lat, codes) for _ in range(users)])
    wall = time.perf_counter() - w0
    cpu = time.process_time() - c0
    lat.sort()

    print(f"  {spec:<18} users={users:<4} n={len(lat):<6} {len(lat)/wall:>7.1f} req/s  "
          f"p50={statistics.median(lat) if lat else float('nan'):>7.1f}  "
          f"p95={pct(lat, 0.95):>7.1f}  max={max(lat) if lat else float('nan'):>7.1f}  "
          f"gen_cpu={cpu/wall:>4.2f}  {codes}")


async def sequential(spec, path, n):
    host, port = split_target(spec)
    url = f"http://{host}:{port}{path}"
    lat, codes = [], {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        for _ in range(3):
            await client.get(url)
        for _ in range(n):
            t = time.perf_counter()
            r = await client.get(url)
            lat.append((time.perf_counter() - t) * 1000.0)
            codes[r.status_code] = codes.get(r.status_code, 0) + 1
    lat.sort()
    print(f"  {spec:<18} sequential n={len(lat):<5} p50={statistics.median(lat):>7.1f}  "
          f"p95={pct(lat, 0.95):>7.1f}  min={lat[0]:>7.1f}  max={lat[-1]:>7.1f}  {codes}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["balancer:8080", "backend1:8000"])
    ap.add_argument("--path", default="/ping")
    ap.add_argument("--users", nargs="+", type=int, default=[1, 10, 50])
    ap.add_argument("--duration", type=int, default=15)
    ap.add_argument("--sequential", type=int, default=0)
    args = ap.parse_args()

    if args.sequential:
        print(f"  sequential {args.sequential} x {args.path} (uncontended floor)")
        for t in args.targets:
            await sequential(t, args.path, args.sequential)
        return

    print(f"  concurrent {args.duration}s x {args.path}   "
          f"(gen_cpu = cores the generator burned)")
    for users in args.users:
        for t in args.targets:
            await run(t, args.path, users, args.duration)


asyncio.run(main())
'''


def main():
    parser = argparse.ArgumentParser(description="Compare balancer vs direct backend path")
    parser.add_argument("--targets", nargs="+", default=["balancer:8080", "backend1:8000"])
    parser.add_argument("--path", default="/ping")
    parser.add_argument("--users", nargs="+", type=int, default=[1, 10, 50])
    parser.add_argument("--duration", type=int, default=15)
    parser.add_argument("--sequential", type=int, default=0)
    parser.add_argument("--cpus", default=None, help="generator CPU quota (default 1.0)")
    args = parser.parse_args()

    result = run_generator(
        [
            "python", "-c", IN_NETWORK,
            "--targets", *args.targets,
            "--path", args.path,
            "--users", *[str(u) for u in args.users],
            "--duration", str(args.duration),
            "--sequential", str(args.sequential),
        ],
        # Generous: this probe reports metrics, so a starved generator would
        # corrupt the very comparison it is meant to produce.
        timeout=args.duration * len(args.users) * len(args.targets) + 240,
        cpus=args.cpus or "1.0",
    )
    print(result.stdout or result.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

"""
Find the operating point where server state actually explains latency.

Too light and every request costs the same fixed overhead, so the target is
pure noise; too heavy and the proxy queues while the backends sit idle, so the
target is again unrelated to server state. Between those extremes there is a
partially loaded regime where routing decisions have measurable consequences.

Two sweeps are supported:

  --levels    vary concurrency at one profile (where is the knee?)
  --profiles  vary the traffic shape at one concurrency (does any shape fall
              off the cliff?)

Each cell is one short run: seal, collect, then report throughput, the latency
distribution and the cross-validated R^2 from server-state features alone.

Usage:
  python bench/sweep_users.py --levels 40 50 60 70 --duration 45 --degrade
  python bench/sweep_users.py --profiles steady burst wave --levels 50 --duration 45
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bench.quick_dataset import (  # noqa: E402
    BALANCER_URL,
    DATA_DIR,
    LOGS_DIR,
    _post,
    collect_sealed_file,
    degrade_backend,
    run_locust,
    set_algorithm,
)
from bench.signal_check import endpoint_r2, state_r2  # noqa: E402


def measure(
    level: int,
    duration: int,
    profile: str,
    policy: str,
    degrade: bool = False,
    fault_server: str = "backend3",
    slow_ms: int = 300,
    fault_error_rate: float = 0.35,
) -> dict:
    run_id = f"sweep_{profile}_{level}u"
    set_algorithm(policy)
    _post("/admin/run", run_id=run_id)
    time.sleep(2)

    # Degrade after the run is open, so the fault window covers the whole load
    # window. This is the condition under which state features must carry the
    # signal, so the sweep has to include it rather than measure a clean stack.
    if degrade:
        degrade_backend(fault_server, slow_ms, fault_error_rate, duration + 15)

    run_locust(profile, DATA_DIR / f"locust_{run_id}", duration, level)

    time.sleep(2)
    sealed = _post("/admin/seal").get("sealed")
    if not sealed:
        return {"level": level, "profile": profile, "error": "nothing sealed"}

    dest = collect_sealed_file(run_id)
    if dest is None:
        return {"level": level, "profile": profile, "error": "sealed file not collected"}

    try:
        df = pd.read_parquet(dest)
    except Exception as exc:
        return {"level": level, "profile": profile, "error": f"unreadable parquet: {str(exc)[:60]}"}
    if df.empty:
        return {"level": level, "profile": profile, "error": "no rows"}

    span = float(df["timestamp"].max() - df["timestamp"].min())
    ok = df[df["status"] < 400]
    return {
        "level": level,
        "profile": profile,
        "rows": len(df),
        "req_s": len(df) / span if span > 0 else 0.0,
        "err_pct": float((df["status"] >= 400).mean() * 100.0),
        # Latency percentiles describe successful requests only. A 503 returns
        # in a millisecond and would make a failing server look like the fastest
        # one in the pool.
        "p50": float(ok["response_ms"].median()) if len(ok) else float("nan"),
        "p95": float(ok["response_ms"].quantile(0.95)) if len(ok) else float("nan"),
        "max": float(ok["response_ms"].max()) if len(ok) else float("nan"),
        "inflight_mean": float(df["inflight"].mean()),
        "cpu_mean": float(df["srv_cpu_pct"].mean()),
        "r2_state": state_r2(df),
        "r2_endpoint": endpoint_r2(df),
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep load levels or traffic profiles")
    parser.add_argument("--levels", nargs="*", type=int, default=[40, 50, 60, 70])
    parser.add_argument("--profiles", nargs="*", default=None,
                        help="sweep these traffic profiles at --levels[0] users")
    parser.add_argument("--duration", type=int, default=45)
    parser.add_argument("--profile", default="steady")
    parser.add_argument("--policy", default="round_robin")
    parser.add_argument("--degrade", action="store_true",
                        help="inject latency+errors into one backend during each cell")
    parser.add_argument("--fault-server", default="backend3")
    parser.add_argument("--slow-ms", type=int, default=300)
    parser.add_argument("--fault-error-rate", type=float, default=0.35)
    args = parser.parse_args()

    if args.profiles:
        cells = [(args.levels[0], p) for p in args.profiles]
        mode = f"profiles={','.join(args.profiles)} users={args.levels[0]}"
    else:
        cells = [(lv, args.profile) for lv in args.levels]
        mode = f"profile={args.profile} levels={args.levels}"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    header = (
        f"{'profile':>11} {'users':>6} {'rows':>8} {'req/s':>8} {'err%':>6} "
        f"{'p50 ms':>9} {'p95 ms':>9} {'max ms':>9} {'inflt':>6} {'cpu%':>6} "
        f"{'R2 state':>10} {'R2 endpt':>10}"
    )
    print("=" * len(header))
    print(f"  Sweep: {mode} policy={args.policy} duration={args.duration}s")
    if args.degrade:
        print(f"  Degraded: {args.fault_server} +{args.slow_ms}ms, "
              f"{args.fault_error_rate:.0%} errors for the whole cell")
    print("=" * len(header))

    results = []
    for level, profile in cells:
        print(f"\n  --- {profile} @ {level} users ---")
        try:
            results.append(measure(
                level, args.duration, profile, args.policy,
                degrade=args.degrade,
                fault_server=args.fault_server,
                slow_ms=args.slow_ms,
                fault_error_rate=args.fault_error_rate,
            ))
        except Exception as exc:
            print(f"    ERROR: {exc}")
            results.append({"level": level, "profile": profile, "error": str(exc)})

    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['profile']:>11} {r['level']:>6}   {r['error']}")
            continue
        print(
            f"{r['profile']:>11} {r['level']:>6} {r['rows']:>8,} {r['req_s']:>8.1f} "
            f"{r['err_pct']:>6.1f} {r['p50']:>9.1f} {r['p95']:>9.1f} {r['max']:>9.1f} "
            f"{r['inflight_mean']:>6.2f} {r['cpu_mean']:>6.1f} "
            f"{r['r2_state']:>+10.4f} {r['r2_endpoint']:>+10.4f}"
        )
    print("=" * len(header))
    print("  Target: R2 state >= 0.30 with proxy queueing kept modest (p95 well under 1s).")


if __name__ == "__main__":
    main()

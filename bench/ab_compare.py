"""
A/B harness for Phase 5 / M6: does the ML router actually beat round-robin?

Same profile, same concurrency, same duration, same hardware; only the routing
algorithm changes. Policies are interleaved within each repeat rather than run
in blocks, because the host drifts over a session (thermal state, background
work) and a block design would attribute that drift to whichever policy ran
second.

Every metric comes from the balancer's own Parquet log - the same 16-column
rows the dataset and the signal gate use - so this does not re-implement the
measurement, and the routing distribution it reports is read back from
`server_id` rather than assumed.

Usage:
  python bench/ab_compare.py --profiles mixed_cost --policies round_robin ml --repeats 3
  python bench/ab_compare.py --profiles degraded --repeats 2 --duration 90
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bench.quick_dataset import (  # noqa: E402
    BALANCER_URL,
    LOGS_DIR,
    PROFILE_USERS,
    _post,
    collect_sealed_file,
    degrade_backend,
    run_locust,
    set_algorithm,
    warm_up,
)

RESULTS_DIR = PROJECT_ROOT / "scratch" / "ab"


@dataclass
class RunResult:
    profile: str
    policy: str
    repeat: int
    run_id: str
    file: str
    rows: int = 0
    span_s: float = 0.0
    req_s: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    error_pct: float = 0.0
    max_ms: float = 0.0
    share: dict = field(default_factory=dict)
    ml: dict | None = None


def measure(path: Path) -> dict:
    """
    Latency and error statistics for one run, from its Parquet file.

    Percentiles cover successful requests only (a 503 returns fast and would
    flatter the router that caused it), while the error rate is counted over
    every row so the two policies are compared on the same offered load.
    """
    df = pd.read_parquet(path)
    span = float(df["timestamp"].max() - df["timestamp"].min()) if len(df) else 0.0
    ok = df[df["status"] < 400]
    share = (
        (df["server_id"].value_counts(normalize=True) * 100).round(1).to_dict()
        if "server_id" in df.columns and len(df) else {}
    )
    stats = {
        "rows": int(len(df)),
        "span_s": round(span, 1),
        "req_s": round(len(df) / span, 1) if span > 0 else 0.0,
        "p50": float(ok["response_ms"].median()) if len(ok) else 0.0,
        "p95": float(ok["response_ms"].quantile(0.95)) if len(ok) else 0.0,
        "p99": float(ok["response_ms"].quantile(0.99)) if len(ok) else 0.0,
        "max_ms": float(ok["response_ms"].max()) if len(ok) else 0.0,
        "error_pct": round(float((df["status"] >= 400).mean() * 100.0), 2) if len(df) else 0.0,
        "share": {k: float(v) for k, v in share.items()},
    }
    return stats


def ml_status() -> dict | None:
    """Serving state of the ML router, if it is the live algorithm."""
    try:
        return httpx.get(f"{BALANCER_URL}/stats", timeout=10).json().get("ml")
    except Exception:
        return None


def run_one(profile: str, policy: str, repeat: int, duration: int, users: int,
            fault_server: str, fault_ms: int, fault_error_rate: float) -> RunResult:
    run_id = f"ab_{profile}_{policy}_{repeat}"
    result = RunResult(profile=profile, policy=policy, repeat=repeat, run_id=run_id, file="")

    print(f"\n  -> {profile} / {policy}  (repeat {repeat}, {users} users, {duration}s)")
    if not set_algorithm(policy):
        raise RuntimeError(f"could not select algorithm {policy}")

    if profile == "degraded":
        degrade_backend(fault_server, fault_ms, fault_error_rate, duration + 20)

    _post("/admin/run", run_id=run_id)
    started = time.time()
    run_locust(profile, RESULTS_DIR / run_id, duration, users)
    _post("/admin/seal")
    print(f"     wall {time.time() - started:.0f}s")

    sealed = collect_sealed_file_to(run_id)
    if sealed is None:
        print("     WARNING: no sealed file; run discarded")
        return result

    result.file = sealed.name
    result.ml = ml_status()
    for key, value in measure(sealed).items():
        setattr(result, key, value)
    return result


def collect_sealed_file_to(run_id: str) -> Path | None:
    """Move the newest sealed file for run_id out of logs/ into scratch/ab."""
    candidates = sorted(
        LOGS_DIR.glob(f"{run_id}_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = RESULTS_DIR / candidates[0].name
    shutil.move(str(candidates[0]), str(dest))
    return dest


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "  ".join("-" * w for w in widths)
    body = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(row)) for row in rows]
    return "\n".join([line, rule, *body])


def summarise(results: list[RunResult]) -> None:
    used = [r for r in results if r.rows > 0]
    policies = sorted({r.policy for r in used})
    profiles = sorted({r.profile for r in used})

    print("\n" + "=" * 78)
    print("  A/B summary (latency over successful requests, errors over all rows)")
    print("=" * 78)

    rows = []
    for profile in profiles:
        for policy in policies:
            group = [r for r in used if r.profile == profile and r.policy == policy]
            if not group:
                continue
            rows.append([
                profile, policy, str(len(group)),
                f"{statistics.mean(r.req_s for r in group):.0f}",
                f"{statistics.mean(r.p50 for r in group):.0f}",
                f"{statistics.mean(r.p95 for r in group):.0f}",
                f"{statistics.mean(r.p99 for r in group):.0f}",
                f"{statistics.mean(r.error_pct for r in group):.2f}%",
            ])
    print(table(
        ["profile", "policy", "runs", "req/s", "p50 ms", "p95 ms", "p99 ms", "errors"],
        rows,
    ))

    # Paired comparison: each repeat contributes one (policy A, policy B) pair
    # measured back to back, so the host's condition cancels out.
    print("\n  Paired per-repeat comparison "
          "(a ratio below 1.00 means the second policy was faster)")
    pairs = []
    if len(policies) == 2:
        base, challenger = policies
        for profile in profiles:
            for repeat in sorted({r.repeat for r in used}):
                a = next((r for r in used if r.profile == profile and r.policy == base and r.repeat == repeat), None)
                b = next((r for r in used if r.profile == profile and r.policy == challenger and r.repeat == repeat), None)
                if not (a and b):
                    continue
                pairs.append([
                    profile, str(repeat),
                    f"{a.p95:.0f} -> {b.p95:.0f}",
                    f"{b.p95 / a.p95:.2f}" if a.p95 else "-",
                    f"{a.p99:.0f} -> {b.p99:.0f}",
                    f"{b.p99 / a.p99:.2f}" if a.p99 else "-",
                    f"{a.error_pct:.2f}% -> {b.error_pct:.2f}%",
                ])
        if pairs:
            print(table(
                ["profile", "repeat", "p95 base -> new", "ratio", "p99 base -> new", "ratio", "errors"],
                pairs,
            ))
            ratios95 = [float(p[3]) for p in pairs if p[3] != "-"]
            ratios99 = [float(p[5]) for p in pairs if p[5] != "-"]
            if ratios95:
                wins = sum(1 for x in ratios95 if x < 1.0)
                print(f"\n  {challenger} vs {base}: p95 median ratio "
                      f"{statistics.median(ratios95):.2f}, "
                      f"faster in {wins}/{len(ratios95)} pairs")
            if ratios99:
                wins = sum(1 for x in ratios99 if x < 1.0)
                print(f"  {challenger} vs {base}: p99 median ratio "
                      f"{statistics.median(ratios99):.2f}, "
                      f"faster in {wins}/{len(ratios99)} pairs")

    # Routing distribution: the clearest evidence that the two policies really
    # routed differently, and where one of them sent its traffic.
    print("\n  Routing distribution (% of requests per backend)")
    rows = []
    for r in sorted(used, key=lambda r: (r.profile, r.policy, r.repeat)):
        server_ids = sorted(r.share)
        rows.append(
            [r.profile, r.policy, str(r.repeat)]
            + [f"{r.share.get(s, 0.0):.0f}%" for s in server_ids]
        )
    if rows:
        print(table(["profile", "policy", "rep"] + sorted({s for r in used for s in r.share}), rows))

    print("\n  Model serving state after each run")
    rows = []
    for r in sorted(used, key=lambda r: (r.profile, r.policy, r.repeat)):
        if not r.ml:
            continue
        rows.append([
            r.policy, r.profile, str(r.repeat),
            str(r.ml.get("artifact")), str(r.ml.get("estimator")),
            str(r.ml.get("decisions")), str(r.ml.get("fallbacks")),
            "yes" if r.ml.get("loaded") else "NO",
        ])
    if rows:
        print(table(
            ["policy", "profile", "rep", "artifact", "estimator", "decisions", "fallbacks", "loaded"],
            rows,
        ))
        print("\n  A run with fallbacks > 0 was not fully model-driven; treat it as "
              "evidence about the fallback, not the model.")

    out = RESULTS_DIR / f"ab_report_{int(time.time())}.json"
    out.write_text(json.dumps([r.__dict__ for r in results], indent=2, default=str))
    print(f"\n  raw results: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B compare routing algorithms")
    parser.add_argument("--profiles", nargs="*", default=["mixed_cost"])
    parser.add_argument("--policies", nargs="*", default=["round_robin", "ml"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--users", type=int, default=0, help="0 = the profile's tuned value")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--fault-server", default="backend3")
    parser.add_argument("--fault-ms", type=int, default=400)
    parser.add_argument("--fault-error-rate", type=float, default=0.35)
    args = parser.parse_args()

    if len(args.policies) != 2 and args.repeats:
        print("note: the paired table needs exactly two policies to be meaningful")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        httpx.get(f"{BALANCER_URL}/stats", timeout=5).raise_for_status()
    except Exception as exc:
        print(f"Balancer not reachable at {BALANCER_URL} ({exc}).")
        sys.exit(1)

    # Randomised paired order: within a repeat both policies run back to back,
    # and the order inside each pair is shuffled, so neither policy is
    # systematically the one that follows a cold cache or a thermal ramp.
    rng = random.Random(args.seed)
    plan = []
    for repeat in range(args.repeats):
        for profile in args.profiles:
            pair = list(args.policies)
            rng.shuffle(pair)
            plan += [(profile, policy, repeat) for policy in pair]

    print("=" * 78)
    print("  A/B routing comparison")
    print(f"  profiles : {args.profiles}")
    print(f"  policies : {args.policies}   repeats: {args.repeats}")
    print(f"  duration : {args.duration}s per run   {len(plan)} runs "
          f"(~{len(plan) * (args.duration + 12) / 60:.0f} min)")
    print("=" * 78)

    warm_up(args.warmup, PROFILE_USERS.get(args.profiles[0], 30), args.policies[0])

    results = []
    for profile, policy, repeat in plan:
        users = args.users or PROFILE_USERS.get(profile, 30)
        try:
            results.append(run_one(
                profile, policy, repeat, args.duration, users,
                args.fault_server, args.fault_ms, args.fault_error_rate,
            ))
        except Exception as exc:
            print(f"     run failed: {exc}")

    summarise(results)


if __name__ == "__main__":
    main()

"""
Dataset generation driver (Phase 3).

For each (profile, policy) combination this:
  1. sets the routing algorithm on the balancer
  2. calls POST /admin/run?run_id=... which seals the previous Parquet file
     and opens a new one tagged with the run id
  3. drives load with Locust
  4. calls POST /admin/seal so the file is closed and complete on disk
  5. moves the sealed file into data/ and writes a sidecar metadata JSON

Because every file is sealed before collection, each one is a valid,
independently readable Parquet file with exactly one run_id.

Usage:
  python bench/quick_dataset.py --runs 1 --duration 120
  python bench/quick_dataset.py --profiles steady mixed_cost --policies round_robin
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BALANCER_URL = "http://localhost:9090"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
BALANCER_CONTAINER = "ai_ml_load_balancer-balancer-1"
DOCKER_BIN = r"C:\Program Files\Docker\Docker\resources\bin"

PROFILES = ["steady", "ramp", "burst", "wave", "mixed_cost", "degraded"]
POLICIES = ["round_robin", "least_conn", "random"]

# Concurrency per profile. Chosen to saturate the 4.5 cores of cgroup quota
# shared across the four backends, so CPU throttling and queueing appear.
PROFILE_USERS = {
    "steady": 100,
    "ramp": 150,
    "burst": 200,
    "wave": 120,
    "mixed_cost": 140,
    "degraded": 120,
}
PROFILE_SPAWN_RATE = 20
PROFILE_WAIT = 5.0  # seconds of idle collection after sealing


def _post(path: str, **params) -> dict:
    r = httpx.post(f"{BALANCER_URL}{path}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def set_algorithm(policy: str) -> bool:
    try:
        _post("/admin/algorithm", name=policy)
        return True
    except Exception as exc:
        print(f"    WARNING: could not set algorithm {policy}: {exc}")
        return False


def run_locust(profile: str, out_prefix: Path, duration: int, users: int) -> bool:
    env = os.environ.copy()
    env["TRAFFIC_PROFILE"] = profile
    env["TRAFFIC_USERS"] = str(users)
    env["BENCH_DURATION"] = str(duration)
    env["SPAWN_RATE"] = str(PROFILE_SPAWN_RATE)
    # Bound randomness so a rerun of the same cell is comparable
    env["PYTHONHASHSEED"] = "0"

    cmd = [
        sys.executable, "-m", "locust",
        "-f", str(PROJECT_ROOT / "bench" / "traffic_profiles.py"),
        "--host", BALANCER_URL,
        "--headless",
        "-t", f"{duration + 30}s",
        "--csv", str(out_prefix),
        "--only-summary",
        "--loglevel", "WARNING",
    ]
    print(f"    locust: profile={profile} users={users} {duration}s")
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, timeout=duration + 120)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("    locust exceeded timeout; stopping it")
        return False


def slow_backend(server: str, ms: int, background: bool = True) -> bool:
    """
    Fault injection: make a backend answer slowly for a while.

    Runs inside the balancer container so it can reach the backends over the
    compose network, which the host cannot do.
    """
    script = (
        "import httpx,time;"
        f"r=httpx.get('http://{server}:8000/slow',params={{'ms':{ms}}},timeout={ms / 1000 + 10});"
        "print(r.status_code)"
    )
    cmd = ["docker", "exec", BALANCER_CONTAINER, "python", "-c", script]
    env = os.environ.copy()
    env["PATH"] = DOCKER_BIN + os.pathsep + env.get("PATH", "")

    try:
        if background:
            subprocess.Popen(
                cmd, cwd=PROJECT_ROOT, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            return subprocess.run(
                cmd, cwd=PROJECT_ROOT, env=env, timeout=ms / 1000 + 30
            ).returncode == 0
        return True
    except Exception as exc:
        print(f"    WARNING: fault injection failed: {exc}")
        return False


def collect_sealed_file(run_id: str) -> Path | None:
    """Move the newest sealed Parquet file for run_id into data/."""
    candidates = sorted(
        LOGS_DIR.glob(f"{run_id}_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print(f"    WARNING: no sealed parquet found for {run_id}")
        return None

    src = candidates[0]
    dest = DATA_DIR / src.name
    shutil.move(str(src), str(dest))
    return dest


def verify_parquet(path: Path) -> dict:
    """Confirm the file is readable and report basic shape."""
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    return {
        "ok": True,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "run_ids": sorted(df["run_id"].unique().tolist()),
        "response_ms_p50": float(df["response_ms"].median()) if len(df) else 0.0,
        "response_ms_max": float(df["response_ms"].max()) if len(df) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate the load-balancer dataset")
    parser.add_argument("--runs", type=int, default=1, help="runs per (profile, policy)")
    parser.add_argument("--duration", type=int, default=120, help="seconds per run")
    parser.add_argument("--profiles", nargs="*", default=PROFILES)
    parser.add_argument("--policies", nargs="*", default=POLICIES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(DATA_DIR))
    parser.add_argument("--fault-server", default="backend3",
                        help="backend to slow during the degraded profile")
    parser.add_argument("--fault-ms", type=int, default=400,
                        help="how long the faulted backend sleeps per request")
    args = parser.parse_args()

    data_dir = Path(args.output)
    data_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    combos = [
        (p, pol)
        for p in args.profiles
        for pol in args.policies
        for _ in range(args.runs)
    ]

    print("=" * 68)
    print("  Dataset generation")
    print(f"  profiles : {args.profiles}")
    print(f"  policies : {args.policies}")
    print(f"  runs/cell: {args.runs}   duration: {args.duration}s")
    print(f"  total runs: {len(combos)}   est. time: {len(combos) * (args.duration + 15) / 60:.0f} min")
    print(f"  output   : {data_dir}")
    print("=" * 68)

    try:
        r = httpx.get(f"{BALANCER_URL}/stats", timeout=5)
        if r.status_code != 200:
            raise RuntimeError(f"status {r.status_code}")
    except Exception as exc:
        print(f"\nBalancer not reachable at {BALANCER_URL} ({exc}).")
        print("Start it with:  docker compose up -d --build")
        sys.exit(1)

    started = time.time()
    index = 0
    manifest = []

    for profile in args.profiles:
        for policy in args.policies:
            for repeat in range(args.runs):
                index += 1
                run_id = f"{profile}_{policy}_{repeat}"
                users = PROFILE_USERS.get(profile, 80)

                print(f"\n[{index}/{len(combos)}] {run_id}  (users={users}, {args.duration}s)")

                set_algorithm(policy)

                # Seal the previous run and open a fresh file for this one
                info = _post("/admin/run", run_id=run_id)
                if info.get("previous"):
                    print(f"    sealed previous: {Path(info['previous']['path']).name}"
                          f" ({info['previous']['rows']} rows)")

                time.sleep(PROFILE_WAIT)

                csv_prefix = data_dir / f"locust_{run_id}"

                # Degraded profile: slow one backend mid-run so the model sees
                # a server whose latency climbs while its reported state lags
                if profile == "degraded":
                    time.sleep(min(20, args.duration // 3))
                    if slow_backend(args.fault_server, args.fault_ms):
                        print(f"    injected fault: {args.fault_server} +{args.fault_ms}ms")

                run_locust(profile, csv_prefix, args.duration, users)

                time.sleep(PROFILE_WAIT)

                # Seal so the file is complete, then collect it
                sealed = _post("/admin/seal").get("sealed")
                if not sealed:
                    print("    WARNING: nothing to seal")
                    continue

                print(f"    sealed: {Path(sealed['path']).name} ({sealed['rows']} rows)")

                dest = collect_sealed_file(run_id)
                if dest is None:
                    continue

                check = verify_parquet(dest)
                print(f"    collected: {dest.name} -> {check}")

                meta = {
                    "run_id": run_id,
                    "profile": profile,
                    "policy": policy,
                    "users": users,
                    "duration_s": args.duration,
                    "seed": args.seed,
                    "file": dest.name,
                    "started_at": sealed["path"],
                    "verified": check,
                }
                manifest.append(meta)
                with open(data_dir / f"meta_{run_id}.json", "w") as f:
                    json.dump(meta, f, indent=2)

    elapsed = time.time() - started
    print("\n" + "=" * 68)
    print(f"  Done in {elapsed / 60:.1f} min")

    files = sorted(data_dir.glob("*.parquet"))
    total_rows = 0
    bad = []
    for f in files:
        try:
            total_rows += len(pd.read_parquet(f))
        except Exception as exc:
            bad.append((f.name, str(exc)[:60]))

    print(f"  parquet files : {len(files)}")
    print(f"  total rows    : {total_rows:,}")
    if bad:
        print(f"  unreadable    : {bad}")
    print(f"  manifest      : {data_dir / 'manifest.json'}")
    with open(data_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("=" * 68)

    if total_rows == 0:
        print("\nNo rows collected. Check balancer logs before training.")
        sys.exit(1)

    # Signal gate: refuse to hand a feature-free dataset to Phase 4
    print("\n  Running signal gate...")
    from bench.signal_check import signal_gate
    ok = signal_gate(str(data_dir))
    if not ok:
        print("\n  SIGNAL GATE FAILED — dataset rejected. Fix instrumentation first.")
        sys.exit(2)
    print("\n  Signal gate passed. Dataset is ready for Phase 4.")


if __name__ == "__main__":
    main()
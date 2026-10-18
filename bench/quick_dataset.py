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
import random
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

# CPU the one-shot load generator may use. The generator is a full HTTP client
# stack and burns roughly a core at a few hundred requests per second, so on a
# 4-vCPU (2 physical core) host it has to be capped and kept out of the
# balancer's cgroup. Letting it run inside the balancer container meant the
# generator starved the very component under test, which looked exactly like a
# proxy bottleneck: throughput fell as concurrency rose while the backends sat
# idle. The generator is now measured rather than assumed to be free.
GENERATOR_CPUS = os.environ.get("GENERATOR_CPUS", "1.0")
GENERATOR_MEM = os.environ.get("GENERATOR_MEM", "512m")

PROFILES = ["steady", "ramp", "burst", "wave", "mixed_cost", "degraded"]
POLICIES = ["round_robin", "least_conn", "random"]

# Concurrency per profile.
#
# The previous values (40-80) were chosen from a saturation curve that turned
# out to be an artefact: the load generator ran inside the balancer container
# and, at ~5 ms of client CPU per request, burned about a full core there,
# starving the router. That looked like a proxy bottleneck ("the proxy queues
# while the backends sit idle") and led to tuning against a measurement of the
# test harness rather than of the balancer.
#
# Measured on this host, with the generator in its own container:
#   * an uncontended /ping costs 9.7 ms through the balancer, 3.8 ms direct;
#   * the generator needs ~1 core to drive ~200 req/s;
#   * the host is 2 physical cores (4 vCPU), so ~120-150 req/s is the rate that
#     can be driven without the generator or the host becoming the bottleneck
#     instead of the backends.
#
# Load is therefore moderated rather than maximised. The dataset does not need
# saturation: with heterogeneous backends (cpu_limit 0.8 down to 0.15) and a
# cost-heterogeneous endpoint mix, a cheap server and an expensive one diverge
# at quite low offered load, and that divergence is the learnable signal.
#
# mixed_cost carries the same concurrency as the rest even though its endpoint
# mix costs several times more per request, so it gets fewer users to keep the
# offered *work* comparable across profiles.
PROFILE_USERS = {
    "steady": 30,
    "ramp": 30,
    "burst": 30,
    "wave": 30,
    "mixed_cost": 24,
    "degraded": 30,
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


def _docker(args: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PATH"] = DOCKER_BIN + os.pathsep + env.get("PATH", "")
    return subprocess.run(
        ["docker", *args], cwd=PROJECT_ROOT, env=env,
        capture_output=True, text=True, timeout=timeout,
    )


def balancer_network() -> str:
    """Network the running stack is attached to, so one-shot containers can join it."""
    out = _docker([
        "inspect", "-f",
        "{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}",
        BALANCER_CONTAINER,
    ])
    net = (out.stdout or "").strip()
    if not net:
        raise RuntimeError(
            f"could not determine the network of {BALANCER_CONTAINER}; "
            "is the stack up? (docker compose up -d)"
        )
    return net


def balancer_image() -> str:
    """Image the balancer runs, reused as the load generator's image."""
    out = _docker(["inspect", "-f", "{{.Config.Image}}", BALANCER_CONTAINER])
    image = (out.stdout or "").strip()
    if not image:
        raise RuntimeError(f"could not determine the image of {BALANCER_CONTAINER}")
    return image


def run_generator(
    args: list[str],
    timeout: float,
    cpus: str | None = None,
    env: dict[str, object] | None = None,
) -> subprocess.CompletedProcess:
    """
    Run a command in a throwaway container on the stack's network.

    The load generator gets its own cgroup so its CPU use is visible and, more
    importantly, bounded: it can no longer take CPU away from the balancer it is
    supposed to be measuring. `env` becomes container environment variables and
    must therefore be built here rather than folded into `args`, because `docker
    run` only accepts options before the image name.
    """
    env_args: list[str] = []
    for key, value in (env or {}).items():
        env_args += ["-e", f"{key}={value}"]
    # bench/ is mounted rather than taken from the image so that editing a
    # traffic profile does not require rebuilding it. Read-only: the generator
    # only ever reads its script. as_posix() because Docker wants forward
    # slashes even for a Windows host path.
    bench_mount = f"{PROJECT_ROOT.as_posix()}/bench:/app/bench:ro"
    return _docker(
        [
            "run", "--rm",
            "--network", balancer_network(),
            "--cpus", cpus or GENERATOR_CPUS,
            "--memory", GENERATOR_MEM,
            "-v", bench_mount,
            *env_args,
            balancer_image(),
            *args,
        ],
        timeout=timeout,
    )


def run_locust(profile: str, out_prefix: Path, duration: int, users: int) -> bool:
    """
    Drive load from a separate container on the compose network.

    The load generator deliberately does NOT run on the Windows host. Driving
    60+ concurrent connections through Docker Desktop's port forwarder roughly
    halved throughput and doubled latency versus generating the load inside the
    compose network, and that extra delay is an artefact of the host's network
    stack rather than anything the load balancer controls. It would show up in
    the labels and drown out the signal we are trying to measure.

    It also no longer runs inside the balancer container. Measured on this host
    (4 vCPU over 2 physical cores) the generator alone burns about one core, so
    sharing the balancer's cgroup starved the router and produced a throughput
    collapse that looked like a proxy bottleneck while the backends idled.

    `out_prefix` is kept for callers that pass a scratch path; request statistics
    come from the balancer's own log, so Locust writes no CSV here.
    """
    cmd = [
        "python", "-m", "locust",
        "-f", "bench/traffic_profiles.py",
        "--host", "http://balancer:8080",
        "--headless",
        "-t", f"{duration + 30}s",
        "--only-summary",
        "--loglevel", "ERROR",
    ]

    print(f"    locust (own container, {GENERATOR_CPUS} cpu): "
          f"profile={profile} users={users} {duration}s")
    try:
        result = run_generator(
            cmd,
            timeout=duration + 180,
            env={
                "TRAFFIC_PROFILE": profile,
                "TRAFFIC_USERS": users,
                "BENCH_DURATION": duration,
                "SPAWN_RATE": PROFILE_SPAWN_RATE,
            },
        )
        if result.returncode != 0:
            print(f"    WARNING: locust exited {result.returncode}")
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
            for line in tail:
                print(f"      {line}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("    locust exceeded timeout; stopping it")
        return False


def backend_control(server: str, path: str, params: dict, timeout_s: float = 30.0) -> bool:
    """
    Call a control endpoint on a backend through the compose network.

    Commands run inside the balancer container, which shares a network with the
    backends; the host cannot reach them because no backend ports are published.
    """
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"http://{server}:8000{path}" + (f"?{query}" if query else "")
    script = (
        "import httpx;"
        f"r=httpx.get('{url}', timeout={timeout_s});"
        "print(r.status_code, r.text)"
    )
    cmd = ["docker", "exec", BALANCER_CONTAINER, "python", "-c", script]
    env = os.environ.copy()
    env["PATH"] = DOCKER_BIN + os.pathsep + env.get("PATH", "")

    try:
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT, env=env,
            capture_output=True, text=True, timeout=timeout_s + 30,
        )
        if result.returncode != 0:
            print(f"    WARNING: control call failed: "
                  f"{(result.stderr or '').strip().splitlines()[-1:]}")
            return False
        return True
    except Exception as exc:
        print(f"    WARNING: control call failed: {exc}")
        return False


def degrade_backend(server: str, latency_ms: int, error_rate: float, duration_s: float) -> bool:
    """
    Make one backend both slower and unreliable for a while.

    The latency fault alone produces no errors, which leaves error_rate_5s a
    constant zero and gives a router nothing to learn from on that axis.
    """
    slow = backend_control(
        server, "/slow",
        {"ms": latency_ms, "duration_s": duration_s},
        timeout_s=max(10.0, latency_ms / 1000 + 10),
    )
    faulty = backend_control(
        server, "/fault",
        {"error_rate": error_rate, "duration_s": duration_s},
    )
    return slow and faulty


def warm_up(seconds: int, users: int, policy: str = "round_robin") -> None:
    """
    Drive throwaway load before the measured matrix.

    The first cell of a matrix is systematically slower than the rest: page
    caches are cold, connection pools are empty and every backend is still
    warming up. Measured across a 6-cell pilot, p50 fell from 42.9 ms in the
    first cell to 27.0 ms by the fourth, and each later cell produced more rows
    than the one before it. Left in place, that drift correlates the labels with
    the order the cells happened to run in, which is exactly the kind of
    confound the state features are supposed to explain away.

    The warm-up run is sealed like any other and then deleted, so it cannot
    reach the dataset or the signal gate.
    """
    print(f"  warming up for {seconds}s at {users} users...")
    set_algorithm(policy)
    _post("/admin/run", run_id="warmup")
    run_locust("steady", DATA_DIR / "locust_warmup", seconds, users)
    _post("/admin/seal")
    removed = 0
    for path in LOGS_DIR.glob("warmup_*.parquet"):
        path.unlink(missing_ok=True)
        removed += 1
    print(f"  warm-up discarded ({removed} file(s))")


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

    span = float(df["timestamp"].max() - df["timestamp"].min()) if len(df) else 0.0
    ok = df[df["status"] < 400] if "status" in df.columns else df
    return {
        "ok": True,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "run_ids": sorted(df["run_id"].unique().tolist()),
        # Throughput and percentiles describe successful requests; count them
        # over the whole file so the offered load of each cell is comparable.
        "req_s": round(len(df) / span, 1) if span > 0 else 0.0,
        "response_ms_p50": float(ok["response_ms"].median()) if len(ok) else 0.0,
        "response_ms_p95": float(ok["response_ms"].quantile(0.95)) if len(ok) else 0.0,
        "response_ms_max": float(ok["response_ms"].max()) if len(ok) else 0.0,
        "error_pct": round(float((df["status"] >= 400).mean() * 100.0), 1) if len(df) else 0.0,
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
    parser.add_argument("--fault-error-rate", type=float, default=0.35,
                        help="fraction of requests the faulted backend fails")
    parser.add_argument("--warmup", type=int, default=25,
                        help="seconds of throwaway load before the matrix, to "
                             "remove the cold-start drift from the first cell")
    parser.add_argument("--ordered", action="store_true",
                        help="run cells in profile/policy order instead of "
                             "shuffling them (shuffling is the default so that "
                             "drift over a long matrix does not line up with any "
                             "one profile)")
    args = parser.parse_args()

    data_dir = Path(args.output)
    data_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    combos = [
        (p, pol, r)
        for p in args.profiles
        for pol in args.policies
        for r in range(args.runs)
    ]

    # Cell order is shuffled by default. Over a 40 minute matrix the host itself
    # drifts (background processes, thermal state, image housekeeping) and a
    # measured PILOT showed the first cell running ~40% slower than the last. If
    # cells run in profile order, that drift is confounded with the profiles
    # themselves, so the model can partly "predict" a profile by knowing when it
    # ran. Shuffling with a fixed seed spreads the drift across profiles and
    # keeps the matrix reproducible.
    if not args.ordered:
        random.Random(args.seed).shuffle(combos)

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

    # A previous run may have died mid-load (Ctrl-C, killed process), leaving a
    # writer installed on a file that has since been removed. Seal it up front
    # so the first /admin/run does not inherit that state.
    try:
        pending = _post("/admin/seal").get("sealed")
        if pending:
            print(f"  recovered dangling run {pending.get('run_id')} "
                  f"({pending.get('rows')} rows)")
    except Exception as exc:
        print(f"  WARNING: could not clear a pending run: {exc}")

    if args.warmup > 0:
        warm_up(args.warmup, PROFILE_USERS.get("steady", 30), args.policies[0])

    for profile, policy, repeat in combos:
        index += 1
        run_id = f"{profile}_{policy}_{repeat}"
        users = PROFILE_USERS.get(profile, 30)

        print(f"\n[{index}/{len(combos)}] {run_id}  (users={users}, {args.duration}s)")

        # A cell that takes far longer than its scripted load means the host
        # stalled (a sleep/suspend, or the Docker VM hanging). The cell's rows
        # then cover a shorter load window than every other cell, and nothing
        # in the data shows that except the wall clock, so it is recorded here.
        cell_started = time.time()

        set_algorithm(policy)

        # Seal the previous run and open a fresh file for this one
        info = _post("/admin/run", run_id=run_id)
        if info.get("previous"):
            print(f"    sealed previous: {Path(info['previous']['path']).name}"
                  f" ({info['previous']['rows']} rows)")

        time.sleep(PROFILE_WAIT)

        csv_prefix = data_dir / f"locust_{run_id}"

        # Degraded profile: make one backend slow and unreliable, so
        # the model sees a server whose state and error rate diverge
        # from its peers
        if profile == "degraded":
            time.sleep(min(15, args.duration // 4))
            fault_window = args.duration * 0.75
            if degrade_backend(args.fault_server, args.fault_ms,
                               args.fault_error_rate, fault_window):
                print(f"    injected fault: {args.fault_server} "
                      f"+{args.fault_ms}ms, {args.fault_error_rate:.0%} errors "
                      f"for {fault_window:.0f}s")

        ok = run_locust(profile, csv_prefix, args.duration, users)

        time.sleep(PROFILE_WAIT)

        # Seal so the file is complete, then collect it
        sealed = _post("/admin/seal").get("sealed")
        if not sealed:
            print("    WARNING: nothing to seal")
            continue

        if sealed.get("error"):
            print(f"    WARNING: seal reported {sealed['error']}")

        print(f"    sealed: {Path(sealed['path']).name} ({sealed['rows']} rows)")

        dest = collect_sealed_file(run_id)
        if dest is None:
            continue

        check = verify_parquet(dest)
        print(f"    collected: {dest.name} -> {check}")

        cell_wall = time.time() - cell_started
        meta = {
            "run_id": run_id,
            "profile": profile,
            "policy": policy,
            "users": users,
            "duration_s": args.duration,
            "seed": args.seed,
            "file": dest.name,
            "started_at": sealed["path"],
            "wall_s": round(cell_wall, 1),
            "verified": check,
        }

        # Generous: the scripted overhead is ~35 s (waits, fault window, locust
        # shutdown, collection), so anything past duration+120 s is a stall
        # rather than slow load.
        if cell_wall > args.duration + 120:
            meta["suspect"] = (
                f"cell took {cell_wall:.0f}s against a {args.duration}s load run; "
                "the host stalled and this cell's window is truncated"
            )
            print(f"    WARNING: {meta['suspect']}")

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
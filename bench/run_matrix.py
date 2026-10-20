"""
Phase 7 benchmark campaign (M8): 5 algorithms x 5 scenarios x 5 repetitions.

Design
------
Algorithms: round_robin, weighted_rr (weighted round robin), least_conn,
least_response_time, ml. `ml` is the herd-protected router, i.e. the plan's
"ML-Predictive (final version)".

Scenarios:
  steady    constant concurrency, identical servers
  ramp      concurrency climbs from 15% to the target
  burst     periodic spikes above the baseline
  degraded  mixed endpoint cost while backend3 is slowed and 35% faulty
  failure   steady traffic while backend3 is stopped at 40% and restarted at 70%

Deviations from the plan, stated rather than hidden:
  * Run length is 60 s, not 5 min. 125 x 6 min is about 12.5 h, which does not
    fit one session on this laptop; the design (5x5x5, randomised order, 5
    repeats) is unchanged and the shorter window is recorded in the manifest.
  * Warm-up is 12 s per run, not 1 min, and it is discarded. The stack is also
    warmed once before the campaign.
  * The generator runs in its own cgroup and its CPU is sampled during each
    run, because M8 requires it to stay under about 70%.

Every result is appended to `scratch/matrix/results.jsonl` as it completes, so
an interrupted campaign resumes instead of restarting. A failed run is recorded
with its error; `--retry-failed` reruns only those.

Usage:
  python bench/run_matrix.py --repeats 5 --duration 60 --warmup 12
  python bench/run_matrix.py --scenarios steady --algorithms ml --repeats 1
  python bench/run_matrix.py --retry-failed
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bench.ab_compare import measure  # noqa: E402
from bench.quick_dataset import (  # noqa: E402
    BALANCER_CONTAINER,
    BALANCER_URL,
    LOGS_DIR,
    PROFILE_USERS,
    _docker,
    _post,
    degrade_backend,
    run_locust,
    set_algorithm,
)

MATRIX_DIR = PROJECT_ROOT / "scratch" / "matrix"
RAW_DIR = MATRIX_DIR / "raw"
RESULTS = MATRIX_DIR / "results.jsonl"
MANIFEST = MATRIX_DIR / "manifest.json"

FAULT_SERVER = "backend3"
MIXED_MIX = "ping:0.2,cpu:0.4,io:0.25,payload:0.15"

ALGORITHMS = ["round_robin", "weighted_rr", "least_conn", "least_response_time", "ml"]

SCENARIOS = {
    "steady": {"profile": "steady", "users": 30},
    "ramp": {"profile": "ramp", "users": 30},
    "burst": {"profile": "burst", "users": 30},
    "degraded": {
        "profile": "degraded",
        "users": 24,
        "env": {"ENDPOINT_MIX": MIXED_MIX},
        "degrade": True,
    },
    "failure": {"profile": "steady", "users": 30, "kill": True},
}


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------

class CpuSampler(threading.Thread):
    """Sample docker CPU% for a few containers at fixed points in the run."""

    def __init__(self, containers: list[str], duration: float, points=(0.25, 0.5, 0.75)):
        super().__init__(daemon=True)
        self.containers = [c for c in containers if c]
        self.duration = duration
        self.points = points
        self.samples: list[dict] = []
        self._stop = threading.Event()

    def run(self) -> None:
        started = time.time()
        for fraction in self.points:
            delay = started + fraction * self.duration - time.time()
            if delay > 0:
                self._stop.wait(delay)
            if self._stop.is_set():
                return
            for name in self.containers:
                out = _docker(
                    ["stats", "--no-stream", "--format", "{{.CPUPerc}}", name],
                    timeout=30,
                )
                text = (out.stdout or "").strip().rstrip("%")
                try:
                    value = float(text)
                except ValueError:
                    continue
                self.samples.append({"container": name, "pct": value})

    def stop(self) -> None:
        self._stop.set()

    def summary(self) -> dict:
        out: dict[str, dict] = {}
        for sample in self.samples:
            out.setdefault(sample["container"], []).append(sample["pct"])
        return {
            name: {"median_pct": round(statistics.median(values), 1), "max_pct": round(max(values), 1)}
            for name, values in out.items()
        }


def jain(values: list[float]) -> float:
    """Jain's fairness index over per-server request counts."""
    values = [v for v in values if v > 0]
    if not values:
        return 0.0
    return (sum(values) ** 2) / (len(values) * sum(v * v for v in values))


def fault_timeline(df: pd.DataFrame, fault: dict) -> dict:
    """When the fault hit, when traffic left the server, when it came back."""
    stop = fault.get("stop_initiated")
    start = fault.get("start_initiated")
    if not stop:
        return {}

    after_stop = df[df["timestamp"] >= stop]
    during = after_stop[after_stop["timestamp"] <= (start or stop + 3600)]
    hit = during[during["server_id"] == FAULT_SERVER]

    out = {
        "stop_at": round(stop, 3),
        "start_at": round(start, 3) if start else None,
        # The clock starts when the stop is issued; the container dies a moment
        # later, so a few in-flight requests can still land on it.
        "last_hit_after_stop_s": round(float(hit["timestamp"].max() - stop), 2) if len(hit) else 0.0,
        "requests_to_faulted_while_down": int(len(hit)),
        "error_pct_while_down": round(float((during["status"] >= 500).mean() * 100), 2) if len(during) else 0.0,
    }
    if start:
        resumed = df[(df["timestamp"] >= start) & (df["server_id"] == FAULT_SERVER)]
        out["traffic_back_after_start_s"] = (
            round(float(resumed["timestamp"].min() - start), 2) if len(resumed) else None
        )
    return out


def extra_metrics(path: Path, fault: dict) -> dict:
    """SLA share and fairness, plus the fault timeline when there is one."""
    df = pd.read_parquet(path)
    ok = df[df["status"] < 400]
    counts = df["server_id"].value_counts()
    out = {
        "sla_violation_pct": round(float((ok["response_ms"] > 500).mean() * 100), 2) if len(ok) else 0.0,
        "jain": round(jain(counts.tolist()), 4),
    }
    if fault:
        out["fault"] = fault_timeline(df, fault)
    return out


# ---------------------------------------------------------------------------
# Fault helpers
# ---------------------------------------------------------------------------

def stop_backend(server: str) -> None:
    _docker(["compose", "stop", server], timeout=180)


def start_backend(server: str) -> None:
    _docker(["compose", "start", server], timeout=180)


def backend_ready(server: str, timeout_s: float = 60.0) -> bool:
    """Poll /health from inside the compose network until the backend answers."""
    deadline = time.time() + timeout_s
    script = (
        "import httpx,sys;"
        f"r=httpx.get('http://{server}:8000/health', timeout=2);"
        "sys.exit(0 if r.status_code == 200 else 1)"
    )
    while time.time() < deadline:
        result = _docker(["exec", BALANCER_CONTAINER, "python", "-c", script], timeout=20)
        if result.returncode == 0:
            return True
        time.sleep(2)
    return False


def schedule_failure(duration: float, out: dict) -> list[threading.Timer]:
    """Stop the fault server at 40% of the run and restart it at 70%."""

    def stop() -> None:
        out["stop_initiated"] = time.time()
        stop_backend(FAULT_SERVER)

    def restart() -> None:
        out["start_initiated"] = time.time()
        start_backend(FAULT_SERVER)

    timers = [
        threading.Timer(duration * 0.4, stop),
        threading.Timer(duration * 0.7, restart),
    ]
    for timer in timers:
        timer.daemon = True
        timer.start()
    return timers


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run_one(scenario: str, algorithm: str, repeat: int, args) -> dict:
    spec = SCENARIOS[scenario]
    run_id = f"mx_{scenario}_{algorithm}_{repeat}"
    users = spec.get("users") or PROFILE_USERS.get(spec["profile"], 30)

    result = {
        "run_id": run_id,
        "scenario": scenario,
        "algorithm": algorithm,
        "repeat": repeat,
        "status": "ok",
        "started_at": time.time(),
        "duration_s": args.duration,
        "warmup_s": args.warmup,
        "users": users,
        "seed": args.seed,
        "commit": args.commit,
    }
    print(f"\n  -> {scenario} / {algorithm} (repeat {repeat}, {users} users, {args.duration}s)")

    if not set_algorithm(algorithm):
        result.update(status="failed", error=f"could not select {algorithm}")
        return result

    # -- warm-up, discarded -------------------------------------------------
    if args.warmup > 0:
        _post("/admin/run", run_id=f"warmup_{run_id}")
        run_locust(spec["profile"], MATRIX_DIR, args.warmup, users, name=f"matrix_warm_{run_id}")
        _post("/admin/seal")
        for path in LOGS_DIR.glob(f"warmup_{run_id}_*.parquet"):
            path.unlink(missing_ok=True)

    fault: dict = {}
    timers: list[threading.Timer] = []
    sampler: CpuSampler | None = None
    try:
        if spec.get("degrade"):
            degrade_backend(
                FAULT_SERVER, args.fault_ms, args.fault_error_rate, args.duration + 10
            )
        if spec.get("kill"):
            timers = schedule_failure(args.duration, fault)

        _post("/admin/run", run_id=run_id)
        sampler = CpuSampler([BALANCER_CONTAINER, f"matrix_gen_{run_id}"], args.duration)
        sampler.start()
        started = time.time()
        ok = run_locust(
            spec["profile"],
            MATRIX_DIR,
            args.duration,
            users,
            name=f"matrix_gen_{run_id}",
            extra_env=spec.get("env"),
        )
        _post("/admin/seal")
        result["measured_wall_s"] = round(time.time() - started, 1)
        if not ok:
            result.update(status="failed", error="locust exited non-zero")
    except Exception as exc:  # noqa: BLE001 - a failed cell must not end the campaign
        result.update(status="failed", error=repr(exc))
    finally:
        if sampler is not None:
            sampler.stop()
        if spec.get("kill"):
            for timer in timers:
                timer.cancel()
            # Whatever happened, the next cell must not inherit a dead backend.
            start_backend(FAULT_SERVER)
            result["backend_ready"] = backend_ready(FAULT_SERVER)
        if spec.get("degrade"):
            # Let the /slow and /fault windows expire before the next run.
            time.sleep(12)

    if result["status"] != "ok":
        return result

    sealed = collect(run_id)
    if sealed is None:
        result.update(status="failed", error="no sealed Parquet file")
        return result

    result["file"] = sealed.name
    result.update(measure(sealed))
    result.update(extra_metrics(sealed, fault))
    result["cpu"] = sampler.summary() if sampler else {}
    try:
        result["ml"] = httpx.get(f"{BALANCER_URL}/stats", timeout=10).json().get("ml")
    except Exception:
        result["ml"] = None
    return result


def collect(run_id: str) -> Path | None:
    """Move the newest sealed file for run_id into scratch/matrix/raw."""
    candidates = sorted(
        LOGS_DIR.glob(f"{run_id}_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / candidates[0].name
    candidates[0].replace(dest)
    return dest


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------

def load_results() -> list[dict]:
    if not RESULTS.exists():
        return []
    rows = []
    for line in RESULTS.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def git_commit() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
        capture_output=True, text=True,
    ).stdout.strip())
    return commit or "unknown", dirty


def main() -> None:
    global FAULT_SERVER

    parser = argparse.ArgumentParser(description="Phase 7 benchmark matrix")
    parser.add_argument("--algorithms", nargs="*", default=ALGORITHMS)
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fault-server", default=FAULT_SERVER)
    parser.add_argument("--fault-ms", type=int, default=400)
    parser.add_argument("--fault-error-rate", type=float, default=0.35)
    parser.add_argument("--retry-failed", action="store_true",
                        help="rerun cells whose only recorded attempt failed")
    args = parser.parse_args()
    FAULT_SERVER = args.fault_server
    args.commit, dirty = git_commit()

    # A stable plan: same seed, same order, so resuming skips the same cells.
    import random
    rng = random.Random(args.seed)
    plan = [
        (scenario, algorithm, repeat)
        for repeat in range(args.repeats)
        for scenario in args.scenarios
        for algorithm in args.algorithms
    ]
    rng.shuffle(plan)

    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = load_results()
    ok_cells = {(r["scenario"], r["algorithm"], r["repeat"]) for r in existing if r.get("status") == "ok"}
    failed_cells = {(r["scenario"], r["algorithm"], r["repeat"]) for r in existing if r.get("status") != "ok"}
    if args.retry_failed:
        todo = [cell for cell in plan if cell in failed_cells or cell not in ok_cells]
    else:
        todo = [cell for cell in plan if cell not in ok_cells]

    manifest = {
        "started_at": time.time(),
        "commit": args.commit,
        "dirty_tree": dirty,
        "seed": args.seed,
        "duration_s": args.duration,
        "warmup_s": args.warmup,
        "repeats": args.repeats,
        "algorithms": args.algorithms,
        "scenarios": args.scenarios,
        "scenario_specs": SCENARIOS,
        "total_cells": len(plan),
        "already_ok": len(ok_cells),
        "planned_this_session": len(todo),
        "estimated_minutes": round(len(todo) * (args.duration + args.warmup + 12) / 60, 0),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))

    print("=" * 78)
    print("  Phase 7 matrix")
    print(f"  commit   : {args.commit}{' (dirty tree)' if dirty else ''}")
    print(f"  design   : {len(args.algorithms)} algorithms x {len(args.scenarios)} scenarios "
          f"x {args.repeats} repeats = {len(plan)} cells")
    print(f"  this run : {len(todo)} cells, {args.duration}s measured + {args.warmup}s warm-up "
          f"(~{manifest['estimated_minutes']:.0f} min)")
    print("=" * 78)

    if not todo:
        print("  nothing to do")
        return

    for index, (scenario, algorithm, repeat) in enumerate(todo, start=1):
        print(f"\n[{index}/{len(todo)}] {scenario} / {algorithm} / {repeat}")
        try:
            result = run_one(scenario, algorithm, repeat, args)
        except Exception as exc:  # noqa: BLE001
            result = {
                "run_id": f"mx_{scenario}_{algorithm}_{repeat}",
                "scenario": scenario,
                "algorithm": algorithm,
                "repeat": repeat,
                "status": "failed",
                "error": repr(exc),
                "started_at": time.time(),
            }
        with RESULTS.open("a") as handle:
            handle.write(json.dumps(result, default=str) + "\n")
        if result.get("status") == "ok":
            print(f"     {result['rows']} rows  {result['req_s']} req/s  "
                  f"p95 {result['p95']:.0f} ms  errors {result['error_pct']:.2f}%")
        else:
            print(f"     FAILED: {result.get('error')}")

    done = load_results()
    ok = sum(1 for r in done if r.get("status") == "ok")
    failed = len(done) - ok
    print("\n" + "=" * 78)
    print(f"  campaign: {ok} ok, {failed} failed, {len(plan)} cells in the design")
    print(f"  manifest: {MANIFEST}")
    print(f"  results : {RESULTS}")
    print("=" * 78)


if __name__ == "__main__":
    main()

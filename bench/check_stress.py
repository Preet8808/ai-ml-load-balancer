"""
Above-capacity stress check.

Drives far more concurrent users than the stack can serve, then reports:
- client-visible outcomes (2xx/4xx/5xx/timeouts) and latency percentiles,
- whether the balancer stayed responsive on /stats while overloaded,
- its memory footprint before and after (no leak),
- whether the container restarted (no crash).
"""

import concurrent.futures
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCKER = os.environ.get(
    "DOCKER_BIN", r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
)
if not os.path.exists(DOCKER):
    DOCKER = "docker"
CONTAINER = "ai_ml_load_balancer-balancer-1"
BASE = "http://localhost:9090"
PATHS = ["/ping", "/cpu", "/io", "/payload"]
USERS = 80
DURATION_S = 60


def get(path, timeout=5):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as response:
        return json.load(response)


def hit(index):
    with urllib.request.urlopen(BASE + PATHS[index % len(PATHS)], timeout=30) as response:
        return response.status


def mem_usage():
    result = subprocess.run(
        [DOCKER, "stats", "--no-stream", "--format", "{{.MemUsage}}", CONTAINER],
        capture_output=True,
        text=True,
    )
    return (result.stdout or result.stderr).strip()


def restarts():
    result = subprocess.run(
        [DOCKER, "inspect", "--format", "{{.RestartCount}}", CONTAINER],
        capture_output=True,
        text=True,
    )
    return (result.stdout or "").strip()


before_stats = get("/stats")
before_mem = mem_usage()
print(f"before: mem={before_mem} restarts={restarts()}")

stop = threading.Event()
outcomes = []   # (status, latency_ms)
lock = threading.Lock()
samples = []


def user(worker_id):
    index = worker_id
    while not stop.is_set():
        started = time.time()
        try:
            status = hit(index)
        except urllib.error.HTTPError as exc:
            status = exc.code
        except Exception:
            status = 0
        latency = (time.time() - started) * 1000
        with lock:
            outcomes.append((status, latency))
        index += USERS


def sampler():
    while not stop.is_set():
        started = time.time()
        try:
            stats = get("/stats", timeout=5)
            samples.append({
                "at": started,
                "ok": True,
                "latency_ms": (time.time() - started) * 1000,
                "inflight": stats.get("inflight_total"),
                "decision_ms": (stats.get("ml") or {}).get("decision_ms_mean"),
                "feed_age_ms": (stats.get("feed") or {}).get("age_ms"),
            })
        except Exception as exc:
            samples.append({"at": started, "ok": False, "error": str(exc)})
        time.sleep(5)


threads = [threading.Thread(target=user, args=(i,), daemon=True) for i in range(USERS)]
for thread in threads:
    thread.start()
sampler_thread = threading.Thread(target=sampler, daemon=True)
sampler_thread.start()

time.sleep(DURATION_S)
stop.set()
time.sleep(2)

after_mem = mem_usage()
codes = {}
for status, _ in outcomes:
    codes[status] = codes.get(status, 0) + 1

ok_latencies = [lat for status, lat in outcomes if status == 200]
p50 = statistics.median(ok_latencies) if ok_latencies else None
p95 = (
    sorted(ok_latencies)[int(len(ok_latencies) * 0.95)]
    if ok_latencies
    else None
)

print(f"after : mem={after_mem} restarts={restarts()}")
print(f"requests: {len(outcomes)} in {DURATION_S}s "
      f"({len(outcomes) / DURATION_S:.0f} req/s)")
print(f"codes: {dict(sorted(codes.items()))}")
print(f"2xx latency: p50={p50:.0f} ms p95={p95:.0f} ms" if p50 else "no 2xx")

failed_samples = [s for s in samples if not s["ok"]]
print(f"/stats samples: {len(samples)}, failed: {len(failed_samples)}")
decision = [s["decision_ms"] for s in samples if s.get("decision_ms")]
if decision:
    print(f"decision cost during load: mean={statistics.mean(decision):.2f} ms "
          f"max={max(decision):.2f} ms")
print(f"max in-flight: {max((s.get('inflight') or 0) for s in samples)}")
print(f"feed age max: {max((s.get('feed_age_ms') or 0) for s in samples):.0f} ms")

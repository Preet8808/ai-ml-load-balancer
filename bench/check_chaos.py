"""
Live chaos check: kill a backend under load, measure how fast it is bypassed,
restart it, and confirm it is ramped back in with slow start.

Evidence collected:
- how long after `docker compose stop` the breaker opened (bypass latency),
- client-visible errors during the whole run,
- that traffic stops reaching the dead backend (sent_since_snapshot stays 0),
- recovery time and the slow-start window on the way back,
- that the balancer process never restarted.
"""

import json
import os
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
BASE = "http://localhost:9090"
PATHS = ["/ping", "/cpu", "/io", "/payload"]


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=15) as response:
        return json.load(response)


def hit(index):
    with urllib.request.urlopen(BASE + PATHS[index % len(PATHS)], timeout=30) as response:
        return response.status


def server(stats, server_id):
    return next(s for s in stats["servers"] if s["server_id"] == server_id)


def compose(*args):
    return subprocess.run(
        [DOCKER, "compose", *args],
        cwd=PROJECT,
        capture_output=True,
        text=True,
    )


def describe(result):
    """Last non-empty line of a compose call, which may write to stdout or stderr."""
    lines = [
        line
        for line in f"{result.stdout or ''}\n{result.stderr or ''}".splitlines()
        if line.strip()
    ]
    return lines[-1] if lines else f"exit {result.returncode}"


stop = threading.Event()
outcomes = []          # (timestamp, status)
lock = threading.Lock()


def drive(worker_id):
    index = worker_id
    while not stop.is_set():
        started = time.time()
        try:
            status = hit(index)
        except Exception:
            status = 0
        with lock:
            outcomes.append((started, status))
        index += 4
        time.sleep(0.01)


threads = [threading.Thread(target=drive, args=(i,), daemon=True) for i in range(4)]
for thread in threads:
    thread.start()

time.sleep(3)
kill_at = time.time()
print(describe(compose("stop", "backend3")))

open_after = None
deadline = time.time() + 15
while time.time() < deadline:
    stats = get("/stats")
    backend3 = server(stats, "backend3")
    if backend3["circuit_open"]:
        open_after = time.time() - kill_at
        print(f"circuit opened {open_after * 1000:.0f} ms after the kill "
              f"({backend3['circuit_opens']} open(s), healthy={backend3['healthy']})")
        break
    time.sleep(0.1)
else:
    print("circuit never opened within 15 s")

# Let the collector also notice the death, then confirm nothing is being sent.
time.sleep(6)
stats = get("/stats")
backend3 = server(stats, "backend3")
print(f"6 s after kill: healthy={backend3['healthy']}, circuit_open={backend3['circuit_open']}, "
      f"sent_since_snapshot={backend3['sent_since_snapshot']}")
print("  traffic elsewhere:",
      {s["server_id"]: s["sent_since_snapshot"] for s in stats["servers"]})

restart_at = time.time()
print(describe(compose("start", "backend3")))

recovered_after = None
slow_start_seen = False
deadline = time.time() + 40
while time.time() < deadline:
    stats = get("/stats")
    backend3 = server(stats, "backend3")
    if backend3["healthy"] and not backend3["circuit_open"]:
        if recovered_after is None:
            recovered_after = time.time() - restart_at
        if backend3["slow_start"]:
            slow_start_seen = True
        if backend3["slow_start"] is False and recovered_after is not None:
            break
    time.sleep(0.25)

time.sleep(3)
stop.set()
time.sleep(0.3)

stats = get("/stats")
backend3 = server(stats, "backend3")
codes = [status for _, status in outcomes]
errors = [status for status in codes if status >= 400 or status == 0]

print()
print(f"client requests: {len(codes)}  errors: {len(errors)} "
      f"({len(errors) / max(len(codes), 1) * 100:.2f}%)  codes={sorted(set(codes))}")
print(f"recovered {recovered_after:.1f} s after restart" if recovered_after else "did not recover")
print(f"slow start observed: {slow_start_seen}")
print(f"backend3 now: healthy={backend3['healthy']}, circuit_open={backend3['circuit_open']}, "
      f"sent_since_snapshot={backend3['sent_since_snapshot']}")
print(f"resilience: {json.dumps(stats['resilience'])}")

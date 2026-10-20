"""
Live check of the stale-feed fallback.

Pauses the collector, waits for every snapshot to age past STALE_AFTER_S,
and confirms that requests keep being served (200s) by the least-connections
fallback instead of failing, then resumes polling and confirms normal routing
returns. This is the plan's M6 criterion, made countable through /stats.
"""

import concurrent.futures
import json
import time
import urllib.request

BASE = "http://localhost:9090"
PATHS = ["/ping", "/cpu", "/io"]


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=15) as response:
        return json.load(response)


def post(path):
    request = urllib.request.Request(BASE + path, method="POST")
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def hit(index):
    with urllib.request.urlopen(BASE + PATHS[index % len(PATHS)], timeout=30) as response:
        return response.status


def drive(count, workers=10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(hit, range(count)))


def wait_for_stale(limit_s=20.0):
    deadline = time.time() + limit_s
    while time.time() < deadline:
        stats = get("/stats")
        if stats["feed"]["stale"]:
            return stats
        time.sleep(0.25)
    return get("/stats")


print("algorithm:", post("/admin/algorithm?name=ml"))
print("baseline requests:", sorted(set(drive(20))))

stats = get("/stats")
print(f"feed age {stats['feed']['age_ms']:.0f} ms, stale={stats['feed']['stale']}, "
      f"stale_routing={stats['resilience']['stale_routing']}")

post("/admin/collector?enabled=false")
stats = wait_for_stale()
print(f"collector paused: feed age {stats['feed']['age_ms']:.0f} ms, "
      f"stale={stats['feed']['stale']}")

before = get("/stats")
codes = drive(30)
after = get("/stats")

print("during stale feed:", len(codes), "requests, codes", sorted(set(codes)))
print("  stale_routing delta:", after["resilience"]["stale_routing"] - before["resilience"]["stale_routing"])
print("  ml decisions delta :", after["ml"]["decisions"] - before["ml"]["decisions"])
print("  retries delta      :", after["resilience"]["retries"] - before["resilience"]["retries"])

post("/admin/collector?enabled=true")
time.sleep(3)
stats = get("/stats")
print(f"collector resumed: feed age {stats['feed']['age_ms']:.0f} ms, stale={stats['feed']['stale']}")
print("post-resume requests:", sorted(set(drive(20))))

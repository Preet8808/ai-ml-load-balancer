"""
Live check of the decision log and the drift guard.

Runs against a live stack and prints, in order:

1. the live MAE the guard computes from ordinary traffic,
2. a demotion, forced by lowering the guard's threshold - the model is healthy,
   so this demonstrates the *mechanism* honestly rather than pretending a
   broken model was shipped,
3. that routing really switched (reason, model_driven, drift_routed),
4. a promotion back once the threshold is restored and a clean window passes,
5. the sealed decision file, read back from disk with its own schema.
"""

import atexit
import concurrent.futures
import glob
import json
import os
import time
import urllib.request

import pyarrow.parquet as pq

BASE = "http://localhost:9090"
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = ["/ping", "/cpu", "/io"]
RUN_ID = f"drift_check_{int(time.time())}"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=20) as response:
        return json.load(response)


def post(path):
    request = urllib.request.Request(BASE + path, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def hit(index):
    with urllib.request.urlopen(BASE + PATHS[index % len(PATHS)], timeout=30) as response:
        return response.status


def drive(count, workers=8):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(hit, range(count)))


def stats():
    return get("/stats")


def _restore_threshold():
    """Never leave the live stack with the deliberately low demo threshold."""
    try:
        post("/admin/drift?threshold=1.0")
    except Exception:
        pass


atexit.register(_restore_threshold)

print("algorithm:", post("/admin/algorithm?name=ml")["algorithm"])
print("drift:", post("/admin/drift?threshold=1.0")["drift"])

run = post(f"/admin/run?run_id={RUN_ID}")
print("run started, decisions file:", run["decisions"]["current"])

# -- 1. ordinary traffic ----------------------------------------------------
codes = drive(120)
s = stats()
drift = s["ml"]["drift"]
print(f"\n120 requests, codes={sorted(set(codes))}")
print(f"live MAE(log): {drift['live_mae_log']} over {drift['samples_in_window']} samples, "
      f"threshold {drift['threshold']}, demoted={drift['demoted']}")
recent = s["decisions"]["recent"]
if recent:
    row = recent[-1]
    print(f"last decision: {row['server_id']} predicted={row['predicted_ms']} ms "
          f"actual={row['actual_ms']} ms status={row['status']} model={row['model_driven']}")
    print(f"  reason: {row['reason']}")

# -- 2/3. force a demotion --------------------------------------------------
post("/admin/drift?threshold=0.15")
drive(120)
s = stats()
drift = s["ml"]["drift"]
print(f"\nthreshold lowered to 0.15: demoted={drift['demoted']} "
      f"(demotions {drift['demotions']}), drift_routed={s['ml']['drift_routed']}")
row = s["decisions"]["recent"][-1]
print(f"last decision: model_driven={row['model_driven']} reason={row['reason']}")

# -- 4. restore and let it promote -----------------------------------------
post("/admin/drift?threshold=1.0")
promoted = None
for _ in range(30):
    drive(40)
    drift = stats()["ml"]["drift"]
    if not drift["demoted"]:
        promoted = drift
        break
print(f"\nafter restoring the threshold: demoted={drift['demoted']} "
      f"live MAE={drift['live_mae_log']} promotions={drift['promotions']}")
row = stats()["decisions"]["recent"][-1]
print(f"last decision: model_driven={row['model_driven']} reason={row['reason']}")

# -- 5. the file ------------------------------------------------------------
sealed = post("/admin/seal")
print("\nsealed:", sealed["sealed"]["path"], sealed["sealed"]["rows"], "rows")
decision_path = sealed["decisions"]["path"]
table = pq.read_table(decision_path)
print("decision file:", os.path.basename(decision_path), table.num_rows, "rows")
print("columns:", ", ".join(table.column_names))

frame = table.to_pandas()
scored = frame[(frame["predicted_ms"].notna()) & (frame["status"] < 400) & (frame["retries"] == 0)]
if len(scored):
    import numpy as np

    error = np.abs(np.log1p(scored["actual_ms"]) - np.log1p(scored["predicted_ms"]))
    print(f"MAE(log) recomputed from the file: {error.mean():.4f} over {len(scored)} scored rows")
print("model-driven rows:", int(frame["model_driven"].sum()), "of", len(frame))
print("fallback rows:", int((~frame["model_driven"]).sum()))
leftover = len(glob.glob(os.path.join(PROJECT, "decisions", "*.parquet")))
print("files in decisions/:", leftover)

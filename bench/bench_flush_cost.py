"""Time Parquet flushes into a growing file on the bind mount vs local disk."""
import subprocess
import sys

SCRIPT = r'''
import os, time, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

schema = pa.schema([("a", pa.float64()), ("b", pa.string()), ("c", pa.int64())])
df = pd.DataFrame({"a": [1.0] * 50, "b": ["backend1"] * 50, "c": list(range(50))})

for target in ["/app/logs/grow.parquet", "/tmp/grow.parquet"]:
    if os.path.exists(target):
        os.remove(target)
    w = pq.ParquetWriter(target, schema)
    times = []
    for i in range(40):
        t = time.perf_counter()
        w.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False))
        times.append((time.perf_counter() - t) * 1000)
    w.close()
    times.sort()
    print(f"{target}")
    print(f"   flushes={len(times)} mean={sum(times)/len(times):7.1f}ms "
          f"p50={times[len(times)//2]:7.1f}ms p95={times[int(len(times)*0.95)]:7.1f}ms "
          f"max={max(times):7.1f}ms size={os.path.getsize(target)} bytes")
'''

result = subprocess.run(
    ["docker", "exec", "ai_ml_load_balancer-balancer-1", "python", "-c", SCRIPT],
    capture_output=True, text=True,
)
print(result.stdout or result.stderr)
sys.exit(result.returncode)
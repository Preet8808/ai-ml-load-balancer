"""Quick peek at collected data: do the state features carry signal?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bench.signal_check import endpoint_r2, state_r2, load_parquet_dir

target = sys.argv[1] if len(sys.argv) > 1 else "data"
df, unreadable = load_parquet_dir(target)

if unreadable:
    print("unreadable:", unreadable)
if df.empty:
    print("no rows")
    raise SystemExit(0)

pd.set_option("display.width", 200)
print(f"rows: {len(df):,}   runs: {df['run_id'].nunique()}   "
      f"files: {len(list(Path(target).glob('*.parquet')))}")
print(f"time span: {df['timestamp'].max() - df['timestamp'].min():.0f}s")
print(f"status: {df['status'].value_counts().to_dict()}")
print()

print("state feature spread (distinct values / std):")
for c in ["srv_cpu_pct", "srv_mem_pct", "inflight", "req_rate_5s",
          "rt_ewma_ms", "rt_p95_5s_ms", "err_rate_5s", "cpu_delta_5s", "payload_kb"]:
    s = pd.to_numeric(df[c], errors="coerce")
    print(f"  {c:15s} n_unique={s.nunique():6d}  mean={s.mean():9.2f}  "
          f"std={s.std():9.2f}  zeros={(s == 0).mean() * 100:5.1f}%")

print()
print("corr(rt_ewma_ms, response_ms) =",
      round(float(pd.to_numeric(df['rt_ewma_ms']).corr(pd.to_numeric(df['response_ms']), method='spearman')), 3))
print()
print("response_ms:", df['response_ms'].describe().round(1).to_dict())
print()
print("R2 state-only   :", round(state_r2(df), 4))
print("R2 endpoint-only:", round(endpoint_r2(df), 4))
print()
print("per-server mean response_ms:")
print(df.groupby("server_id")["response_ms"].agg(["count", "mean", "median", "max"]).round(1).to_string())
"""
Summarise sweep cells already sitting in data/, without re-running the load.

The first sweep run collected its Parquet files but the report step looked in
the wrong place, so the numbers were never printed. Re-reading the sealed files
is cheaper than repeating four minutes of load, and it keeps the report step
testable independently of the driver.

Usage:
  python bench/sweep_report.py
  python bench/sweep_report.py --pattern "sweep_*.parquet"
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bench.quick_dataset import DATA_DIR  # noqa: E402
from bench.signal_check import endpoint_r2, state_r2  # noqa: E402

LEVEL_RE = re.compile(r"_(\d+)u(?:_\d+)?\.parquet$")


def level_of(path: Path) -> int:
    m = LEVEL_RE.search(path.name)
    return int(m.group(1)) if m else -1


def summarise(path: Path) -> dict:
    df = pd.read_parquet(path)
    if df.empty:
        return {"path": path, "error": "no rows"}

    span = float(df["timestamp"].max() - df["timestamp"].min())
    return {
        "path": path,
        "level": level_of(path),
        "rows": len(df),
        "req_s": len(df) / span if span > 0 else 0.0,
        "p50": float(df["response_ms"].median()),
        "p95": float(df["response_ms"].quantile(0.95)),
        "max": float(df["response_ms"].max()),
        "err_pct": float((df["status"] >= 400).mean() * 100.0),
        "inflight_mean": float(df["inflight"].mean()),
        "inflight_max": float(df["inflight"].max()),
        "cpu_mean": float(df["srv_cpu_pct"].mean()),
        "r2_state": state_r2(df),
        "r2_endpoint": endpoint_r2(df),
    }


def main():
    parser = argparse.ArgumentParser(description="Report sweep cells from data/")
    parser.add_argument("--pattern", default="sweep_*.parquet")
    parser.add_argument("--dir", default=str(DATA_DIR))
    args = parser.parse_args()

    files = sorted(Path(args.dir).glob(args.pattern), key=level_of)
    if not files:
        print(f"  no files matching {args.pattern} in {args.dir}")
        return

    results = []
    for f in files:
        try:
            results.append(summarise(f))
        except Exception as exc:
            results.append({"path": f, "error": str(exc)[:70]})

    header = (
        f"{'users':>6} {'rows':>8} {'req/s':>8} {'p50 ms':>9} {'p95 ms':>9} "
        f"{'max ms':>9} {'err%':>6} {'inflt':>6} {'cpu%':>6} {'R2 state':>10} {'R2 endpt':>10}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['path'].name:>40}  {r['error']}")
            continue
        print(
            f"{r['level']:>6} {r['rows']:>8,} {r['req_s']:>8.1f} {r['p50']:>9.1f} "
            f"{r['p95']:>9.1f} {r['max']:>9.1f} {r['err_pct']:>6.1f} "
            f"{r['inflight_mean']:>6.2f} {r['cpu_mean']:>6.1f} "
            f"{r['r2_state']:>+10.4f} {r['r2_endpoint']:>+10.4f}"
        )
    print("=" * len(header))
    print("  Target: R2 state >= 0.30 with proxy queueing kept modest (p95 well under 1s).")


if __name__ == "__main__":
    main()

"""
Print one line per generated cell from data/manifest.json.

The signal gate only reports aggregates, and one bad cell can hide inside a
passing total. This shows throughput, percentiles and error rate per cell so a
profile that silently flattened, or a run that collapsed, is visible before the
dataset is used for training.

Usage:
  python bench/manifest_report.py
  python bench/manifest_report.py --manifest data/manifest.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bench.quick_dataset import DATA_DIR  # noqa: E402


def discover_cells(path: Path) -> list[dict]:
    """
    Load the manifest, or fall back to the per-cell sidecars.

    manifest.json is written once, when the whole matrix finishes, so during a
    long collection there is nothing to read. Each cell writes its own
    meta_<run_id>.json as it completes, which allows a run in progress to be
    inspected and an early bad cell to be spotted.
    """
    if path.exists():
        return json.loads(path.read_text()), path

    sidecars = sorted(path.parent.glob("meta_*.json"))
    if not sidecars:
        return [], path
    return [json.loads(p.read_text()) for p in sidecars], path.parent


def main():
    parser = argparse.ArgumentParser(description="Report per-cell dataset health")
    parser.add_argument("--manifest", default=str(DATA_DIR / "manifest.json"))
    args = parser.parse_args()

    path = Path(args.manifest)
    cells, source = discover_cells(path)
    if not cells:
        print(f"  no manifest or per-cell sidecars at {path.parent}")
        return

    print(f"  source: {source}")
    header = (
        f"{'cell':<28} {'usr':>4} {'rows':>7} {'req/s':>7} "
        f"{'p50':>7} {'p95':>7} {'max':>7} {'err%':>6}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    total_rows = 0
    ok_rows = [c.get("verified", {}).get("rows", 0) for c in cells if c.get("verified", {}).get("ok")]
    median_rows = float(np.median(ok_rows)) if ok_rows else 0.0

    flagged = []
    for c in cells:
        v = c.get("verified", {})
        if not v.get("ok", False):
            print(f"{c['run_id']:<28}   FAILED: {v.get('error', 'unknown')}")
            flagged.append(f"{c['run_id']}: unreadable")
            continue
        total_rows += v.get("rows", 0)

        notes = []
        if c.get("suspect"):
            notes.append("host stalled")
        # A truncated cell is short on rows but internally consistent, so the
        # row count is the only place the damage shows up.
        if median_rows and v.get("rows", 0) < median_rows * 0.6:
            notes.append(f"only {v.get('rows', 0) / median_rows * 100:.0f}% of median rows")

        # A cell whose rows span far more wall time than the scripted load ran
        # for has a hole in it. This is what a host sleep looks like: the rows
        # either side of the suspend are fine and the span between them is not,
        # and any request in flight across it records the sleep as latency.
        span_s = v.get("rows", 0) / v["req_s"] if v.get("req_s") else 0.0
        duration_s = c.get("duration_s", 0) or 0
        if duration_s and span_s > duration_s * 1.4:
            notes.append(f"span {span_s:,.0f}s vs {duration_s}s window")

        note = ("  <- " + "; ".join(notes)) if notes else ""

        print(
            f"{c['run_id']:<28} {c.get('users', 0):>4} {v.get('rows', 0):>7,} "
            f"{v.get('req_s', 0):>7.1f} {v.get('response_ms_p50', 0):>7.1f} "
            f"{v.get('response_ms_p95', 0):>7.0f} {v.get('response_ms_max', 0):>7.0f} "
            f"{v.get('error_pct', 0):>6.1f}{note}"
        )
        if notes:
            flagged.append(c["run_id"])

    print("-" * len(header))
    print(f"{'TOTAL':<28} {'':>4} {total_rows:>7,} across {len(cells)} cell(s)")
    print(f"  median cell: {median_rows:,.0f} rows")

    if flagged:
        print("-" * len(header))
        print(f"  SUSPECT CELLS ({len(flagged)}) - re-run these before training:")
        for name in flagged:
            print(f"    {name}")
    print("=" * len(header))
    print("  A row count far below its neighbours means that cell was throttled")
    print("  or truncated rather than shaped by its profile.")


if __name__ == "__main__":
    main()

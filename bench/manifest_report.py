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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bench.quick_dataset import DATA_DIR  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Report per-cell dataset health")
    parser.add_argument("--manifest", default=str(DATA_DIR / "manifest.json"))
    args = parser.parse_args()

    path = Path(args.manifest)
    if not path.exists():
        print(f"  no manifest at {path}")
        return

    cells = json.loads(path.read_text())
    header = (
        f"{'cell':<28} {'usr':>4} {'rows':>7} {'req/s':>7} "
        f"{'p50':>7} {'p95':>7} {'max':>7} {'err%':>6}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    total_rows = 0
    for c in cells:
        v = c.get("verified", {})
        if not v.get("ok", False):
            print(f"{c['run_id']:<28}   FAILED: {v.get('error', 'unknown')}")
            continue
        total_rows += v.get("rows", 0)
        print(
            f"{c['run_id']:<28} {c.get('users', 0):>4} {v.get('rows', 0):>7,} "
            f"{v.get('req_s', 0):>7.1f} {v.get('response_ms_p50', 0):>7.1f} "
            f"{v.get('response_ms_p95', 0):>7.0f} {v.get('response_ms_max', 0):>7.0f} "
            f"{v.get('error_pct', 0):>6.1f}"
        )

    print("-" * len(header))
    print(f"{'TOTAL':<28} {'':>4} {total_rows:>7,} across {len(cells)} cell(s)")
    print("=" * len(header))
    print("  A row count far below its neighbours means that cell was throttled")
    print("  rather than shaped by its profile; check the generator and host load.")


if __name__ == "__main__":
    main()

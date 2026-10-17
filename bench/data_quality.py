"""
Data quality checks and dataset statistics for Phase 3.

Reads all Parquet files in a directory and produces:
  - Row count and schema verification
  - Distribution of response times
  - Per-server and per-profile counts
  - Missing/NaN check
  - Basic summary stats
"""

import os
import glob
import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, Any, List


EXPECTED_COLUMNS = [
    "timestamp", "run_id", "req_type", "payload_kb", "server_id",
    "srv_cpu_pct", "srv_mem_pct", "inflight", "req_rate_5s",
    "rt_ewma_ms", "rt_p95_5s_ms", "err_rate_5s", "cpu_limit",
    "cpu_delta_5s", "response_ms", "status",
]


def load_all_parquet(data_dir: str) -> pd.DataFrame:
    """Load all Parquet files from a directory tree into a single DataFrame."""
    files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    if not files:
        print(f"No Parquet files found in {data_dir}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df["source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def check_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Verify all expected columns are present and report missing ones."""
    present = set(df.columns)
    expected = set(EXPECTED_COLUMNS)
    missing = expected - present
    extra = present - expected
    return {
        "missing_columns": list(missing),
        "extra_columns": list(extra),
        "schema_ok": len(missing) == 0,
    }


def check_nans(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for NaN values in any column."""
    nan_counts = df.isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0].to_dict()
    return {
        "columns_with_nans": cols_with_nans,
        "has_nans": len(cols_with_nans) > 0,
    }


def compute_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics for the dataset."""
    stats = {
        "total_rows": len(df),
        "unique_servers": df["server_id"].nunique() if "server_id" in df.columns else 0,
        "req_type_counts": df["req_type"].value_counts().to_dict() if "req_type" in df.columns else {},
        "server_counts": df["server_id"].value_counts().to_dict() if "server_id" in df.columns else {},
        "response_ms": {
            "mean": df["response_ms"].mean() if "response_ms" in df.columns else None,
            "median": df["response_ms"].median() if "response_ms" in df.columns else None,
            "p95": df["response_ms"].quantile(0.95) if "response_ms" in df.columns else None,
            "p99": df["response_ms"].quantile(0.99) if "response_ms" in df.columns else None,
            "min": df["response_ms"].min() if "response_ms" in df.columns else None,
            "max": df["response_ms"].max() if "response_ms" in df.columns else None,
        },
        "status_codes": df["status"].value_counts().to_dict() if "status" in df.columns else {},
    }

    if "server_id" in df.columns and "response_ms" in df.columns:
        stats["per_server_response"] = df.groupby("server_id")["response_ms"].agg(
            ["mean", "median", "count"]
        ).to_dict()

    return stats


def run_quality_check(data_dir: str) -> None:
    """Run the full data quality check and print a report."""
    print(f"\n{'='*60}")
    print(f"  Dataset Quality Report")
    print(f"  Directory: {data_dir}")
    print(f"{'='*60}\n")

    df = load_all_parquet(data_dir)
    if df.empty:
        print("  No data to analyze.")
        return

    # Schema
    schema = check_schema(df)
    print(f"  Schema check: {'PASS' if schema['schema_ok'] else 'FAIL'}")
    if schema["missing_columns"]:
        print(f"    Missing columns: {schema['missing_columns']}")
    if schema["extra_columns"]:
        print(f"    Extra columns: {schema['extra_columns']}")

    # NaNs
    nans = check_nans(df)
    print(f"  NaN check: {'PASS' if not nans['has_nans'] else 'FAIL'}")
    if nans["has_nans"]:
        for col, count in nans["columns_with_nans"].items():
            print(f"    {col}: {count} NaNs")

    # Stats
    stats = compute_stats(df)
    print(f"\n  Total rows: {stats['total_rows']}")
    print(f"  Unique servers: {stats['unique_servers']}")

    print(f"\n  Request type distribution:")
    for rtype, count in stats["req_type_counts"].items():
        pct = count / stats["total_rows"] * 100
        print(f"    {rtype}: {count} ({pct:.1f}%)")

    print(f"\n  Server distribution:")
    for sid, count in stats["server_counts"].items():
        pct = count / stats["total_rows"] * 100
        print(f"    {sid}: {count} ({pct:.1f}%)")

    print(f"\n  Response time (ms):")
    rt = stats["response_ms"]
    print(f"    mean:   {rt['mean']:.2f}")
    print(f"    median: {rt['median']:.2f}")
    print(f"    p95:    {rt['p95']:.2f}")
    print(f"    p99:    {rt['p99']:.2f}")
    print(f"    min:    {rt['min']:.2f}")
    print(f"    max:    {rt['max']:.2f}")

    print(f"\n  Status codes:")
    for code, count in stats["status_codes"].items():
        print(f"    {code}: {count}")

    print(f"\n  Target: 200,000+ rows from 20+ runs")
    print(f"  Current: {stats['total_rows']} rows")
    if stats["total_rows"] >= 200000:
        print(f"  ✓ Target met")
    else:
        print(f"  ✗ Below target (need {200000 - stats['total_rows']} more)")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    run_quality_check(data_dir)

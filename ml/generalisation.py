"""
Unseen-profile generalisation test (Phase 7).

Every training and validation split in `ml/train.py` is by run, but every split
still sees all six traffic profiles. This test is stricter: it holds out *every*
run of one profile, trains the shipped estimator on the rest, and reports what
the model does on a traffic shape it has never seen.

The comparison is the same one M5 uses - MAE(log) against the server's own EWMA
- so the number is directly comparable to `docs/dataset.md`.

Usage:
  python ml/generalisation.py --profile burst
  python ml/generalisation.py --profile wave --data data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.train import (  # noqa: E402
    TARGET_COLUMN,
    build_features,
    build_models,
    evaluate,
    load_dataset,
    spearman,
)


def run(data_dir: str, profile: str, output_dir: str) -> dict:
    df, info = load_dataset(data_dir)
    if df.empty:
        raise SystemExit(f"no usable data in {data_dir}")

    runs = df["run_id"].astype(str)
    held = df[runs.str.startswith(f"{profile}_")]
    train = df[~runs.str.startswith(f"{profile}_")]
    if held.empty:
        raise SystemExit(f"no runs start with '{profile}_' in {data_dir}")

    X_train, y_train = build_features(train)
    X_test, y_test_series = build_features(held)
    y_test = y_test_series.to_numpy()
    y_ms = held.loc[y_test_series.index, TARGET_COLUMN].to_numpy(dtype=float)

    model = build_models()["xgboost"]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = evaluate(y_test, pred, y_ms)
    metrics["spearman"] = spearman(pred, y_test)

    naive_log = np.log1p(X_test["rt_ewma_ms"].to_numpy(dtype=float))
    naive = evaluate(y_test, naive_log, y_ms)
    naive["spearman"] = spearman(naive_log, y_test)

    improvement = (
        (naive["mae_log"] - metrics["mae_log"]) / naive["mae_log"] * 100
        if naive["mae_log"]
        else float("nan")
    )

    result = {
        "profile": profile,
        "held_out_runs": sorted(held["run_id"].unique().tolist()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "model": metrics,
        "naive": naive,
        "improvement_pct": round(float(improvement), 1),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"generalisation_{profile}.json"
    path.write_text(json.dumps(result, indent=2, default=str))

    print(f"\n  held out      : {', '.join(result['held_out_runs'])}")
    print(f"  train rows    : {len(X_train):,} (from {info['runs'] - len(result['held_out_runs'])} runs)")
    print(f"  test rows     : {len(X_test):,}")
    print(f"  naive MAE(log): {naive['mae_log']:.4f}")
    print(f"  model MAE(log): {metrics['mae_log']:.4f}  ({improvement:+.1f}% vs naive)")
    print(f"  model MAE(ms) : {metrics['mae_ms']:.1f}")
    print(f"  R2(log)       : {metrics['r2_log']:+.4f}")
    print(f"  Spearman      : {metrics['spearman']:.3f}")
    print(f"  report        : {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Unseen-profile generalisation test")
    parser.add_argument("--data", default="data")
    parser.add_argument("--profile", default="burst")
    parser.add_argument("--output", default="scratch/ablations")
    args = parser.parse_args()
    run(args.data, args.profile, args.output)


if __name__ == "__main__":
    main()

"""
Model training pipeline (Phase 4).

Trains and evaluates, on log1p(response_ms):

  0. naive baseline   predict the server's own reported rt_ewma_ms
  1. Ridge regression linear reference
  2. Random Forest     nonlinear reference
  3. XGBoost           expected best model

Splitting is by `run_id`, never by row. Adjacent rows inside one run share a
server state that evolves over seconds, so a random row split leaks the
validation set into training and inflates every score. The plan's metric is
grouped by run for exactly this reason.

Run:
  python ml/train.py --data data --output models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import (
    FEATURE_COLUMNS,
    STATE_FEATURES,
    TARGET_COLUMN,
    TARGET_CAP_MS,
    build_features,
    feature_columns,
    to_ms,
)

SEED = 42
WARMUP_S = 20.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_dir: str, warmup_s: float = WARMUP_S) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load all Parquet files and drop each run's warm-up window."""
    from bench.signal_check import load_parquet_dir

    df, unreadable = load_parquet_dir(data_dir)
    info: Dict[str, Any] = {"unreadable": unreadable, "rows_raw": int(len(df))}

    if df.empty:
        return df, info

    before = len(df)
    if "timestamp" in df.columns and "run_id" in df.columns:
        keep = pd.Series(True, index=df.index)
        for run_id, group in df.groupby("run_id"):
            start = group["timestamp"].min()
            keep.loc[group.index] = group["timestamp"] >= start + warmup_s
        df = df[keep]

    df = df.reset_index(drop=True)
    info["rows_after_warmup"] = int(len(df))
    info["warmup_dropped"] = int(before - len(df))
    info["runs"] = int(df["run_id"].nunique()) if "run_id" in df.columns else 0
    return df, info


def split_by_run(df: pd.DataFrame, y_index: pd.Index) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Split rows into train/val/test by run_id.

    Runs are shuffled with a fixed seed, then partitioned 70/15/15. Returns
    boolean masks aligned to y_index.
    """
    runs = df.loc[y_index, "run_id"].to_numpy()
    unique = np.array(sorted(pd.unique(runs)))
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique)

    n = len(unique)
    n_train = max(1, int(n * 0.70))
    n_val = max(1, int(n * 0.15)) if n >= 3 else 0

    train_runs = set(unique[:n_train])
    val_runs = set(unique[n_train:n_train + n_val])
    test_runs = set(unique[n_train + n_val:])

    # Guarantee a non-empty test set
    if not test_runs:
        test_runs = {unique[-1]}

    train_mask = np.isin(runs, list(train_runs))
    val_mask = np.isin(runs, list(val_runs))
    test_mask = np.isin(runs, list(test_runs))

    return train_mask, val_mask, test_mask, sorted(str(r) for r in test_runs)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(y_true_log: np.ndarray, y_pred_log: np.ndarray, y_true_ms: np.ndarray) -> Dict[str, float]:
    """Metrics on the log scale plus error in the original ms scale."""
    pred_ms = to_ms(y_pred_log)
    clipped_ms = np.clip(y_true_ms, 0.0, TARGET_CAP_MS)
    return {
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "mae_ms": float(mean_absolute_error(clipped_ms, pred_ms)),
        "p90_abs_err_ms": float(np.percentile(np.abs(clipped_ms - pred_ms), 90)),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, which is what matters for picking the fastest server."""
    if len(a) < 3:
        return float("nan")
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_models() -> Dict[str, Any]:
    ridge = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=SEED)),
    ])
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=SEED,
    )
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=SEED,
            tree_method="hist",
        )
    except ImportError:
        xgb_model = None
    return {"ridge": ridge, "random_forest": rf, "xgboost": xgb_model}


def cross_validate(model, X, y, groups, n_splits: int = 5) -> Dict[str, float]:
    """
    GroupKFold by run_id, so folds never share a run.

    Two details matter here. Indexing is positional through numpy, because the
    training path keeps X as a DataFrame while y is a log-transformed ndarray,
    and mixing the two access styles is what broke this. And every fold fits a
    clone rather than the model handed in: fitting that one would leave it
    trained on a single fold's subset, so the artifact saved afterwards would
    silently be a model that never saw part of its own training split.
    """
    unique_groups = len(np.unique(groups))
    splits = min(n_splits, unique_groups)
    if splits < 2:
        return {}

    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y_values = np.asarray(y)

    cv = GroupKFold(n_splits=splits)
    preds = np.full(len(y_values), np.nan)
    for train_idx, test_idx in cv.split(X_values, y_values, groups=groups):
        fold_model = clone(model)
        fold_model.fit(X_values[train_idx], y_values[train_idx])
        preds[test_idx] = fold_model.predict(X_values[test_idx])

    ok = ~np.isnan(preds)
    return evaluate(y_values[ok], preds[ok], np.expm1(y_values[ok]))


def measure_inference(model, X: pd.DataFrame, n_servers: int = 5, trials: int = 200) -> Dict[str, float]:
    """Per-decision latency: one batched prediction across N candidate servers."""
    sample = X.head(max(n_servers, 1))
    if len(sample) < n_servers:
        sample = pd.concat([sample] * n_servers, ignore_index=True).head(n_servers)

    # Warm up so import/first-call cost is not counted
    for _ in range(5):
        model.predict(sample)

    times = []
    for _ in range(trials):
        start = time.perf_counter()
        model.predict(sample)
        times.append((time.perf_counter() - start) * 1000.0)

    return {
        "batch_mean_ms": float(np.mean(times)),
        "batch_p99_ms": float(np.percentile(times, 99)),
    }


# ---------------------------------------------------------------------------
# Main training run
# ---------------------------------------------------------------------------

def train(data_dir: str = "data", output_dir: str = "models") -> Dict[str, Any]:
    print("=" * 70)
    print("  Phase 4: model training")
    print("=" * 70)

    df, info = load_dataset(data_dir)
    if df.empty:
        print(f"No usable data in {data_dir}")
        return {}

    if info["unreadable"]:
        print(f"  WARNING: {len(info['unreadable'])} unreadable file(s) skipped")

    print(f"  rows: {info['rows_raw']:,} raw -> {info['rows_after_warmup']:,} after warm-up")
    print(f"  runs: {info['runs']}")

    X, y = build_features(df)
    y_log = y.to_numpy()
    y_ms = df.loc[y.index, TARGET_COLUMN].to_numpy(dtype=float)

    train_mask, val_mask, test_mask, test_runs = split_by_run(df, y.index)
    groups_all = df.loc[y.index, "run_id"].to_numpy()

    X_train, y_train = X[train_mask], y_log[train_mask]
    X_val, y_val = X[val_mask], y_log[val_mask]
    X_test, y_test = X[test_mask], y_log[test_mask]
    y_test_ms = y_ms[test_mask]

    print(f"  train {len(X_train):,} rows | val {len(X_val):,} | test {len(X_test):,}")
    print(f"  held-out runs: {test_runs}")

    results: Dict[str, Any] = {}

    # -- naive baseline ----------------------------------------------------
    # The server's own reported EWMA. This is what any heuristic router can
    # already do without a model, so a model must beat it to be worth shipping.
    print("\n  [1/4] naive baseline (server's reported rt_ewma_ms)")
    ewma_log = np.log1p(X["rt_ewma_ms"].to_numpy(dtype=float))
    baseline = evaluate(y_test, ewma_log[test_mask], y_test_ms)
    baseline["spearman"] = spearman(ewma_log[test_mask], y_test)
    baseline["rank_of_4"] = float("nan")
    results["naive"] = {"metrics": baseline}
    print(f"        MAE(log)={baseline['mae_log']:.4f}  MAE(ms)={baseline['mae_ms']:.1f}  "
          f"R2={baseline['r2_log']:+.4f}  rho={baseline['spearman']:.3f}")

    # -- models ------------------------------------------------------------
    models = build_models()
    entries = [(n, m) for n, m in models.items() if m is not None]
    total = len(entries) + 1
    for i, (name, model) in enumerate(entries, start=2):
        print(f"\n  [{i}/{total}] {name}")
        if len(X_val):
            fit_kwargs = {}
            if name == "xgboost":
                fit_kwargs = dict(eval_set=[(X_val, y_val)], verbose=False)
            model.fit(X_train, y_train, **fit_kwargs)
        else:
            model.fit(X_train, y_train)

        pred = model.predict(X_test)
        metrics = evaluate(y_test, pred, y_test_ms)
        metrics["spearman"] = spearman(pred, y_test)

        cv = cross_validate(model, X_train, y_train, groups_all[train_mask])
        if cv:
            metrics["cv_r2_log"] = cv["r2_log"]
            metrics["cv_mae_ms"] = cv["mae_ms"]

        timing = measure_inference(model, X)
        metrics.update({f"infer_{k}": v for k, v in timing.items()})

        results[name] = {"metrics": metrics, "model": model}

        print(f"        MAE(log)={metrics['mae_log']:.4f}  MAE(ms)={metrics['mae_ms']:.1f}  "
              f"R2={metrics['r2_log']:+.4f}  rho={metrics['spearman']:.3f}")
        if "cv_r2_log" in metrics:
            print(f"        CV R2={metrics['cv_r2_log']:+.4f}  CV MAE(ms)={metrics['cv_mae_ms']:.1f}")
        print(f"        inference over 5 servers: {timing['batch_mean_ms']:.3f} ms "
              f"(p99 {timing['batch_p99_ms']:.3f} ms)")

    # -- feature importance ------------------------------------------------
    best = best_model_name(results)
    if best:
        model = results[best]["model"]
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            order = np.argsort(importances)[::-1]
            print(f"\n  feature importance ({best}):")
            for idx in order:
                print(f"    {FEATURE_COLUMNS[idx]:20s} {importances[idx]:.4f}")
            results[best]["feature_importance"] = [
                [FEATURE_COLUMNS[i], float(importances[i])] for i in order
            ]

    # -- persist -----------------------------------------------------------
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_hash = hashlib.sha1(
        "".join(sorted(df["run_id"].astype(str).unique())).encode()
    ).hexdigest()[:10]

    saved = []
    for name, entry in results.items():
        if "model" not in entry:
            continue
        path = out / f"{name}_{data_hash}.joblib"
        joblib.dump(
            {
                "model": entry["model"],
                "feature_columns": feature_columns(),
                "target": "log1p(response_ms)",
                "data_hash": data_hash,
                "train_rows": int(len(X_train)),
                "runs": info["runs"],
            },
            path,
        )
        saved.append(path.name)
        print(f"  saved {path.name}")

    summary = {
        name: {k: v for k, v in entry.items() if k != "model"}
        for name, entry in results.items()
    }
    summary["meta"] = {
        "data_hash": data_hash,
        "rows": int(len(df)),
        "runs": info["runs"],
        "train_runs": int(len(np.unique(groups_all[train_mask]))),
        "test_runs": test_runs,
        "feature_columns": feature_columns(),
        "best_model": best,
    }
    summary_path = out / f"training_report_{data_hash}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  saved {summary_path.name}")

    # -- verdict -----------------------------------------------------------
    print("\n" + "-" * 70)
    if best:
        best_mae = results[best]["metrics"]["mae_log"]
        naive_mae = results["naive"]["metrics"]["mae_log"]
        improvement = (naive_mae - best_mae) / naive_mae * 100 if naive_mae else float("nan")
        print(f"  best model        : {best}")
        print(f"  naive MAE(log)    : {naive_mae:.4f}")
        print(f"  best MAE(log)     : {best_mae:.4f}")
        print(f"  improvement       : {improvement:+.1f}%")
        print(f"  M5 target (>=30%) : {'MET' if improvement >= 30 else 'NOT MET'}")
    print("-" * 70)

    return results


def best_model_name(results: Dict[str, Any]) -> str | None:
    candidates = [(n, e) for n, e in results.items() if n != "naive" and "model" in e]
    if not candidates:
        return None
    return min(candidates, key=lambda kv: kv[1]["metrics"]["mae_log"])[0]


def main():
    parser = argparse.ArgumentParser(description="Train load-balancer response-time models")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="models")
    args = parser.parse_args()
    train(args.data, args.output)


if __name__ == "__main__":
    main()
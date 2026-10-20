# Matrix artifacts

Committed evidence for the Phase 7 campaign (125 cells). Regenerate with
`python bench/analyse_matrix.py`; the analysis prose is in `../results.md`.

| File | Content |
|---|---|
| `results.jsonl` | one JSON object per attempted cell: status, metrics, CPU samples, fault timeline, ML state, commit, seed |
| `summary.csv` | one row per successful run with the columns the tables use |
| `manifest.json` | design as launched: commit, seed, durations, algorithms, scenarios, expected cells |
| `ablation_*.json` | full / no-trend / no-request-type training reports |
| `generalisation_*.json` | burst and wave held-out profile reports |
| `herd_ablation.txt` | raw output of the `ml` vs `ml_predictive` burst A/B |

## What is not committed

The request-level Parquet files (one per cell, about 2-4 MB each) are gitignored:
125 files would add roughly 400 MB. They are reproducible with
`python bench/run_matrix.py` (resumable) and every table and chart is generated
from the committed summary plus those files.

## How to read `results.jsonl`

- `status`: `ok` or `failed` with `error`; the analysis only uses `ok`.
- `share`: percent of requests per server, from `server_id`.
- `cpu`: `docker stats` samples taken at 25%, 50% and 75% of the run.
- `fault`: only for the failure scenario: stop/restart times, last request to
  the stopped server, 5xx share while it was down, when traffic returned.
- `ml`: `/stats.ml` after the run: decisions, fallbacks, drift state and the
  live MAE in log space.
- `commit`: the commit the runner started from; `drift_threshold`: the pinned
  guard threshold for ML cells (see `../results.md`, deviations).

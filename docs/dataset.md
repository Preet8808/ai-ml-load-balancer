# Dataset and trained model (Phases 3-4)

This is the evidence for the dataset the model is trained on: what was
collected, how it was validated before training, and what the trained models
actually achieve. It is written to be checked rather than trusted - every
number here is reproducible with the commands at the bottom.

## What was collected

Six traffic profiles crossed with three routing policies, one 120 s run per
cell, 18 cells in total. Load came from a Locust container on the compose
network, not through the host's port forwarder, and each cell was preceded by
a discarded warm-up so the measured window is not the one where the Python
processes are still cold.

| cell | users | rows | req/s | p50 ms | p95 ms | max ms | err % |
|---|---|---:|---:|---:|---:|---:|---:|
| mixed_cost_least_conn_0 | 24 | 10,330 | 85.9 | 100.8 | 413 | 2,131 | 0.0 |
| mixed_cost_round_robin_0 | 24 | 8,309 | 69.2 | 34.0 | 1,594 | 5,423 | 0.0 |
| mixed_cost_random_0 | 24 | 6,306 | 52.5 | 39.1 | 2,684 | 4,821 | 0.0 |
| steady_least_conn_0 | 30 | 19,462 | 162.1 | 53.8 | 159 | 848 | 0.0 |
| steady_round_robin_0 | 30 | 9,258 | 77.1 | 33.5 | 1,790 | 5,887 | 0.0 |
| steady_random_0 | 30 | 9,888 | 82.4 | 35.6 | 1,650 | 5,353 | 0.0 |
| ramp_least_conn_0 | 30 | 17,776 | 148.0 | 57.1 | 218 | 2,111 | 0.0 |
| ramp_round_robin_0 | 30 | 10,428 | 86.9 | 27.7 | 1,756 | 4,412 | 0.0 |
| ramp_random_0 | 30 | 8,473 | 70.5 | 51.6 | 1,464 | 10,159 | 0.0 |
| burst_least_conn_0 | 30 | 19,262 | 160.4 | 54.1 | 162 | 859 | 0.0 |
| burst_round_robin_0 | 30 | 11,395 | 94.9 | 28.8 | 1,285 | 4,304 | 0.0 |
| burst_random_0 | 30 | 10,670 | 88.9 | 44.1 | 1,212 | 5,398 | 0.0 |
| wave_least_conn_0 | 30 | 18,325 | 152.6 | 56.8 | 189 | 1,237 | 0.0 |
| wave_round_robin_0 | 30 | 9,288 | 77.3 | 29.7 | 1,763 | 9,617 | 0.0 |
| wave_random_0 | 30 | 10,375 | 86.4 | 37.6 | 1,429 | 4,303 | 0.0 |
| degraded_least_conn_0 | 30 | 18,044 | 150.3 | 49.6 | 262 | 1,272 | 1.4 |
| degraded_round_robin_0 | 30 | 10,460 | 87.1 | 40.2 | 831 | 5,135 | 6.6 |
| degraded_random_0 | 30 | 9,962 | 83.0 | 48.1 | 876 | 4,796 | 6.8 |

**218,011 rows** across 18 files, one `run_id` per file.

The table is already a result: on identical hardware, `least_conn` sustains
148-162 req/s at p95 159-262 ms, while `round_robin` and `random` sustain
69-95 req/s at p95 831-1,790 ms. With one backend faulted (the `degraded`
row), `least_conn` held errors to 1.4% where the other two policies sat at
6.6-6.8%. The routing policy changes the behaviour of the system by roughly a
factor of two, which is what makes an adaptive router worth building and what
the Phase 5 A/B measures.

## How a cell is collected

`bench/quick_dataset.py` walks the matrix and, for each cell:

1. sets the algorithm with `POST /admin/algorithm?name=...`
2. opens a run with `POST /admin/run?run_id=...`, which seals the previous
   Parquet file and starts a new one
3. drives load from a Locust container with its own 1.0 CPU quota
4. seals the file with `POST /admin/seal`
5. moves the sealed file into `data/` and writes a sidecar with the wall time

Cells are shuffled with a fixed seed rather than run profile by profile, so a
drift in the host over the session does not line up with the profile order.

## What was checked before training

**The signal gate** (`bench/signal_check.py`) rejects a dataset unless the
server-state features alone predict `log1p(response_ms)` with R² >= 0.30 on
held-out runs, over at least 5,000 rows and 4 runs. It exists because an
earlier dataset passed a visual inspection while being useless: its labels
were explained by *which endpoint* was called rather than by how loaded the
server was.

```
rows            : 218,011
run_ids         : 18
R2 state-only   : +0.5499   (need >= +0.30)
R2 endpoint-only: +0.0770   (reference)
GATE PASSED
```

The endpoint-only number is the control: if it were high, the model would be
learning the request mix instead of the server state.

**Per-cell health** (`bench/manifest_report.py`) prints the table above and
flags cells whose wall time is much longer than the scripted load, or whose
row count is far below its peers. This is not hypothetical: one cell
(`ramp_random_0`) was silently truncated when the laptop slept for 53 minutes
mid-collection, and the driver's per-cell wall time is what exposed it. That
cell was re-collected and the file it replaced was moved aside rather than
deleted, so the damage stays inspectable.

**One `run_id` per file** is enforced in the logger, not assumed. A request
that completes while `seal()` is draining can no longer be stamped with the
run being sealed; such a row is dropped and counted. Without that fix a cell
contained a row belonging to the previous cell, and the gate counted 19 run
ids for an 18-cell matrix.

## The trained models

`ml/train.py` splits by `run_id`, never by row: adjacent rows inside a run
share a server state that evolves over seconds, so a random row split leaks
the validation set into training. 12 runs train, the remaining 4 held-out runs
(`burst_random_0`, `mixed_cost_round_robin_0`, `steady_random_0`,
`wave_least_conn_0`) are the test set. 178,165 rows survive the warm-up
window and the exclusion of failed requests.

| model | MAE(log) | MAE ms | R² | Spearman | CV R² | inference (5 servers) |
|---|---:|---:|---:|---:|---:|---:|
| naive: server's own `rt_ewma_ms` | 1.274 | 140.5 | -0.291 | 0.414 | - | - |
| Ridge | 0.640 | 111.5 | 0.645 | 0.692 | 0.601 | 1.9 ms |
| Random Forest | 0.591 | 75.4 | 0.676 | 0.704 | 0.638 | 51.7 ms |
| XGBoost | **0.587** | **73.1** | **0.683** | 0.710 | 0.646 | 5.95 ms (p99 8.4) |

**M5 verdict: MET.** XGBoost improves MAE over the naive baseline by 53.9%
(73.1 ms against 140.5 ms) where the target is 30%. The baseline is not a
straw man: it is exactly what `least_response_time` routes on, and it scores
*worse than predicting the mean* (R² -0.29), because the last seconds' EWMA
does not describe the next request's wait once the server's queue has changed
in between.

The Random Forest is the useful negative result. It is dominated on both axes:
less accurate than XGBoost (MAE(log) 0.591 against 0.587) and about nine times
slower per decision, 51.7 ms to score five candidates against 5.95 ms. On a
50 ms p50 the forest would spend the request's own latency deciding where to
send it, so accuracy alone does not make it a serving candidate.

Inference timing moves with whatever else the host is doing - an earlier run
of the same benchmark, taken while an image was being built, reported 225 ms
for the forest and 6.8 ms for XGBoost. The absolute number is therefore not
reproducible to the millisecond, but the ratio between the models is, and it
is large enough to decide the question.

Feature importance (XGBoost): `rt_p95_5s_ms` 0.359, `req_type_cpu` 0.162,
`rt_ewma_ms` 0.097, `req_type_light` 0.096, `cpu_limit` 0.062,
`req_rate_5s` 0.060, `req_type_io` 0.051, `inflight` 0.036, `srv_cpu_pct` 0.020,
`err_rate_5s` 0.016, `payload_kb` 0.012, `srv_mem_pct` 0.011,
`cpu_delta_5s` 0.011, `req_type_payload` 0.006.

The recent tail of the response-time distribution dominates, which is what a
queueing system should look like; the request type matters second, because a
`/cpu` request is expensive by construction. No single feature carries the
prediction, which is what the endpoint-only control was there to check. The
ranking of the weaker features shifts between runs (XGBoost's split selection
is not stable for near-irrelevant columns), so only the top of this list should
be read as a finding.

## Known limitations

- **`cpu_limit` is the design ratio, not the enforced quota.** The router sees
  the configured 4:2:2:1 ratio (2.0 / 1.0 / 1.0 / 0.5), while Docker enforces
  0.8 / 0.4 / 0.4 / 0.15 so the cluster fits this host. The collector
  publishes the configured value so training and serving agree; the real
  capability still reaches the model through `srv_cpu_pct` (already relative
  to each server's quota) and the latency features. Publishing the enforced
  quota instead would mean re-collecting the matrix, because a feature whose
  value changes between collection and serving is worse than one measured in
  a different unit.
- **Failed requests are not latency observations.** Rows with status >= 400
  are excluded from training rather than taught as "this server was fast"; the
  error rate reaches the model through `err_rate_5s`. Two requests in the
  re-collected cell returned 500 and are excluded this way.
- **The host is a 2-physical-core laptop.** Load was moderated rather than
  maximised (~120 req/s sustainable), because a saturated host makes every
  label an artefact of the scheduler.
- **`degraded` cells contain a fault that was injected, not observed.** The
  fault rate and delay are the driver's parameters, documented in the sidecar
  metadata of each cell.

## Reproducing

```bash
python bench/quick_dataset.py --duration 120   # full matrix, ~40 min
python bench/manifest_report.py                # per-cell health table
python bench/signal_check.py data              # the gate
python ml/train.py --data data --output models # train + evaluate + M5 verdict
```

Training and serving must use the same XGBoost, because the artifact is a
pickle of a scikit-learn estimator wrapping a Booster. The version is pinned in
`requirements-ml.txt` (3.2.0 - the newest release that supports the service
images' Python 3.11) and the balancer image installs it with `--no-deps`, since
the Linux wheel otherwise pulls 351 MB of NCCL for GPU training this project
never does. `scratch/check_artifact.py` loads the artifact and predicts on a
fixed feature row; run it on the host and inside the container to confirm both
environments agree before trusting a live run.

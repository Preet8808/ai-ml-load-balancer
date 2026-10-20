# AI/ML-Based Intelligent Load Balancer

A Layer-7 HTTP load balancer that uses machine learning to predict per-server response times and route requests to the fastest backend. Built as a semester project with a 14-week timeline.

## Repository Layout

| Directory | Purpose |
|-----------|---------|
| `balancer/` | Reverse proxy, decision engine, routing algorithms |
| `backend/` | FastAPI/Flask backend apps with `/metrics` and `/health` endpoints |
| `agent/` | psutil metrics agent running inside each backend container |
| `ml/` | Feature engineering, model training, inference |
| `bench/` | Benchmark suites, Locust traffic scripts, resilience/drift checks |
| `docs/` | Hypotheses, design docs, architecture diagrams |
| `logs/`, `decisions/` | Per-run request and decision Parquet files (generated, gitignored) |

## Quick Start

```bash
# Install dependencies: the runtime requirements plus the ML stack, which
# tests, the signal gate and training need (the service images install
# requirements.txt only)
pip install -r requirements-dev.txt

# Run tests
python -m pytest

# Start the full stack (requires Docker Compose)
docker compose up
```

On a machine with `make` (Linux, macOS, CI), the same entry points are wrapped:

```bash
make up          # build and start, dashboard on :9090
make test        # 121 tests
make benchmark   # 125-cell Phase 7 matrix (resumable, ~3 h)
make analyse     # tables, confidence intervals, charts
```

## Deliverables

| Document | What it is |
|---|---|
| `docs/report.md` | Final report: design, experiment, results, threats to validity |
| `docs/results.md` | Phase 7 matrix analysis: tables, H1-H4, charts |
| `docs/ablations.md` | Feature ablations, unseen-profile generalisation, herd ablation |
| `docs/dataset.md` | Dataset, signal gate, model selection, M5 verdict |
| `docs/serving.md` | Serving path and the exploratory A/B evidence, including the burst session loss |
| `docs/resilience.md` | Retry, breaker, slow start, stale feed, stress evidence |
| `docs/monitoring.md` | Decision log, serving cost, drift guard |
| `docs/slides.md` | 15-slide final deck |
| `docs/demo.md` | Five-minute live demo script |
| `docs/viva.md` | Anticipated questions with honest answers |
| `docs/traceability.md` | Every plan deliverable mapped to its evidence |

## Dataset generation

The dataset is produced by `bench/quick_dataset.py`, which drives load at each
(profile, policy) cell, seals the balancer's Parquet log per run, and runs a
signal gate before the data is considered usable for training.

```bash
# Start the stack (balancer on http://localhost:9090)
docker compose up -d --build

# Generate the full matrix: 6 traffic profiles x 3 policies, 120 s per cell
python bench/quick_dataset.py --duration 120

# Inspect it per cell, then re-run the gate on its own
python bench/manifest_report.py
python bench/signal_check.py data
```

The gate rejects a dataset unless the server-state features alone predict
`log1p(response_ms)` with R² ≥ 0.30 on held-out folds, with at least 5,000 rows
across at least 4 distinct runs. It exists because an earlier dataset passed
inspection while being useless: `response_ms` was explained by *which endpoint*
was called rather than by how loaded the server was, so no routing decision
could have improved it.

### Operating point

Load is moderate rather than maximal, because this project is developed on a
2-physical-core laptop and the measurement is easy to confuse with the harness:

| Quantity | Measured |
|---|---|
| Uncontended `/ping` through the balancer | 9.7 ms p50 |
| Uncontended `/ping` direct to a backend | 3.8 ms p50 |
| Load generator's own cost | ~1 core per 200 req/s |
| Sustainable offered load | ~120 req/s |

Two consequences are designed in. The load generator runs as its **own
container** on the compose network rather than inside the balancer, because
sharing the balancer's CPU quota made the router look like the bottleneck while
the backends sat idle. And backend CPU quotas keep the 4:2:2:1 ratio that makes
the cluster heterogeneous while fitting inside the machine, since quotas
summing to more than the host can deliver make every latency sample a scheduler
artefact. See `docs/instrumentation_audit.md` for the full analysis.

### Traffic profiles

| Profile | Shape |
|---|---|
| `steady` | constant concurrency |
| `ramp` | concurrency climbs from 15% to the target |
| `burst` | periodic 1.6x spikes over a 0.5x baseline |
| `wave` | sinusoidal, 120 s period |
| `mixed_cost` | constant concurrency, heavier endpoint mix |
| `degraded` | constant concurrency while one backend is slowed and faulted |

`bench/traffic_profiles.py` can also be run standalone against a host:

```bash
TRAFFIC_PROFILE=burst TRAFFIC_USERS=30 python -m locust -f bench/traffic_profiles.py \
  --host=http://localhost:9090 --headless
```

## Live demo

The balancer serves its own dashboard at <http://localhost:9090/>. It shows each
backend's CPU relative to its quota, in-flight requests, requests routed in the
last second, EWMA and p95 latency, and error rate, and it switches the routing
algorithm live - so the same traffic can be shown being routed differently
without restarting anything. The **Demo controls** card can also start browser
traffic (opening and sealing a run), inject a backend3 slow/error fault, and
stress or restore the drift guard, so a live demo does not need a terminal for
those steps. The browser driver is intentionally not a benchmark client.

```bash
docker compose up -d                                   # dashboard at localhost:9090
curl -X POST "http://localhost:9090/admin/algorithm?name=ml"
```

The trained artifacts are committed, so `ml` works on a fresh clone. `data/` is
not committed (20 MB of Parquet, rebuilt by the collector); after collecting
your own, retrain with `python ml/train.py --data data --output models`.

Three model algorithms are selectable. `ml` is the shipped router and `ml_herd`
is the same class under the Phase 7 name: predicted latency times outstanding
work. `ml_predictive` is the plan's no-herd-protection ablation - predicted
latency only - and exists to be measured, not served. `python bench/run_matrix.py`
runs the Phase 7 campaign; `ml/train.py --drop-features …` runs the feature
ablations.

`/stats` reports the model's serving state too, so a real prediction-driven run
can be told apart from one that silently fell back to a heuristic:

```json
"ml": {"loaded": true, "artifact": "xgboost_032d9bab9a.joblib",
       "estimator": "XGBRegressor", "decisions": 4213, "fallbacks": 0,
       "decision_ms_mean": 1.38, "decision_ms_max": 9.54}
```

With no artifact present, or one that cannot be loaded, the router keeps
routing on least-connections rather than failing, and `fallbacks` counts it.

### A/B: does the ML router actually help?

These are exploratory single-profile A/Bs. The pre-registered 125-cell matrix
below is the primary comparison; the exploratory sessions are kept because they
show the herding/queue correction and the session-to-session variation.

```bash
python bench/ab_compare.py --profiles mixed_cost --policies round_robin ml --repeats 3
python bench/ab_compare.py --profiles degraded --repeats 2 --duration 90
```

Same profile, concurrency and duration under each policy, with the policies
interleaved inside each repeat and the order inside a pair shuffled, so host
drift does not attach itself to one policy. Latency is read back from the
balancer's own Parquet log - the same rows the dataset is built from - and the
routing distribution is read from `server_id`, so "which server got the
traffic" is measured rather than assumed.

`mixed_cost`, 24 users, three repeats (least_conn and ml re-measured on the
shipped code; round_robin from the earlier session that ran all three):

| policy | req/s | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| least_conn | 175 | 24 ms | 76 ms | 137 ms |
| ml | 134 | 59 ms | 145 ms | 229 ms |
| round_robin | 77 | 35 ms | 1205 ms | 2823 ms |

The learned router beats the plan's round-robin baseline by about 9x on p95 and
1.7x on throughput, and loses to least_conn by about 45% on p95 (median paired
ratio 1.46, 0 of 3 repeats). Both held in every repeat across two sessions.
`docs/serving.md` explains why, and documents the herding bug the first serving
version had and the queue correction that fixed it.

On `burst`, that exploratory comparison reverses: least_conn 211 req/s / p95
77 ms against ml 165 / 138 ms, ml losing 0 of 3 pairs (1.66-1.95x worse). The
plan's M6 criterion remains **not met**; the Phase 7 matrix later found a small,
non-significant mean win (243 vs 275 ms), not a consistent burst win.
`docs/serving.md` shows the routing distribution that explains the exploratory
loss.

Under the `degraded` profile - one backend 400 ms slower and failing 35% of
requests - ml routes 0-1% of traffic to the faulted backend against
round_robin's fixed 25%. On the shipped stack the shared breaker and retry keep
both policies near zero errors, so the difference shows up in latency:

| policy | req/s | p50 | p95 | p99 | errors | share to faulted backend |
|---|---:|---:|---:|---:|---:|---:|
| ml | 154 | 62 ms | 174 ms | 299 ms | **0.00%** | 0% |
| round_robin | 147 | 41 ms | 423 ms | 686 ms | **0.01%** | 1-4% |

Before the safeguards existed, the same pairing measured 0.26% against 8.73%
errors and 162 against 414 ms p95. `docs/serving.md` keeps both tables: the
first shows the router's avoidance, the second the whole serving stack.

### Phase 7 matrix (primary comparison)

125 cells (5 algorithms x 5 scenarios x 5 repeats) completed in one randomised
session from commit `3d9246e`: **125 ok, 0 failed**, seed 42, 60 s measured +
12 s warm-up. Full evidence: `docs/results/`, prose and charts: `docs/results.md`.

| Scenario | ML p95 | Best baseline p95 | Change |
|---|---:|---:|---:|
| Steady | 433 | 214 (least conn) | +102% |
| Ramp | 273 | 169 (least conn) | +62% |
| Burst | 243 | 275 (least conn) | -12% (n.s.) |
| Degraded | 182 | 269 (least RT) | -32% (n.s.) |
| Failure | 734 | 209 (least conn) | +252% |

**H1 is not met**: 0/5 significant wins (needs 3). The favourable burst and
degraded means do not survive Holm correction at five runs; ML loses steady,
ramp and failure, and least connections has the best throughput in all five.
**H2 is met** only against round robin (0.55x; 2.02x worse than least
connections). **H3 is met** on the mean (1.96 ms; one 126.72 ms scheduler tail).
**H4 is met** (worst redirect 1.13 s, 0.00% final 5xx; traffic returns 7.7 s
after restart). Live MAE is 1.394 steady and 2.421 failure versus 0.5868
offline; the collapsed runs send 4-7% to backend4 and lose capacity. The drift
guard was pinned at 10 for the matrix; at the shipped 1.0 it would demote. The
herd ablation is the clearest live win: burst p95 342 vs 700 ms and 102 vs 73
req/s, herd faster in 3/3 paired runs.

### Resilience under failure

Every algorithm shares three safeguards, so the router is not the only thing
standing between a failing backend and the client:

- **Stale-feed fallback.** If every metrics snapshot is older than 3 s, routing
  switches to least-connections over the last known state, only to servers the
  collector last saw healthy. Pausing the collector
  (`POST /admin/collector?enabled=false`) demonstrates it: 30 requests, all
  200, with `ml.decisions` flat while `resilience.stale_routing` rises by 30.
- **Circuit breaker.** Three consecutive proxy failures open a server for 10 s,
  then allow one probe. In a live kill test the circuit opened **1.09 s** after
  `docker compose stop backend3`, and 2,934 requests spanning the kill and the
  restart completed with **zero errors**.
- **Slow start and one retry.** A recovered server is ramped from 10% to its
  full share over 10 s, and a failed request is retried once on a different
  server. The restarted backend was serving again 5.0 s later, with the
  slow-start window observed in `/stats`.

An 80-user stress run - 7,353 requests over 60 s, above what 1.75 CPU of
backend quota can serve - produced no errors, no restart and +3% RSS
(141.9 -> 146.6 MiB). Overload appeared as queueing (p50 625 ms, p95 902 ms),
not failure. `docs/resilience.md` has the raw output and the trade-offs.

### Monitoring: decision log and drift guard

Every decision is written to its own Parquet file (`decisions/`, one run per
file) with the reason, the prediction, the outcome, the retries, the metrics
age and whether a model made the call. It joins to the 16-column request log on
`(run_id, timestamp)` - the 16-column schema has no id - and was verified exact
on a 2,601-request run. `/stats.decisions.recent` feeds a predicted-vs-actual
table on the dashboard.

The drift guard scores the served server's prediction against the outcome in
log space - the metric the model was selected on - and demotes the model to
least-connections when the rolling MAE exceeds 1.0 (the naive baseline is
1.27, XGBoost trained at 0.59). It keeps shadow-scoring while demoted, clears
the window on demotion, and promotes itself back after a clean full window.
Measured live: `live MAE(log) 0.4603` on ordinary traffic; lowering the
threshold demoted it (`drift_routed` 112, every reason `drift-fallback`), and
restoring it promoted it back (`promotions 1`). In the Phase 7 matrix the mean
live MAE is 1.394 in steady, 1.151 in ramp, 0.898 in burst, 0.519 in degraded
and 2.421 in failure; the matrix pins the guard at 10 to measure routing, so no
cell demoted. At 1.0 the steady, ramp and failure windows would have crossed.
See `docs/monitoring.md` and `docs/results.md`.

## Milestones

| Milestone | Week | Focus | Status |
|-----------|------|-------|--------|
| M1 | 1 | Project skeleton and CI green | done |
| M2 | 3 | Baseline balancer works end-to-end | done |
| M3 | 5 | Live metrics and logs flowing | done |
| M4 | 6 | Dataset v1 (200k+ rows) | partial - 218,011 rows, 18 runs (plan asked 20+) |
| M5 | 8 | Model beats naive baseline by 30%+ MAE | done - +53.9% (XGBoost) |
| M6 | 9 | MVP: ML balancer demo | partial - stale-feed fallback, routing overhead and failure redirect met; burst-vs-least_conn is not a significant win in the matrix, and the earlier A/B lost |
| M7 | 11 | Feature freeze, all tests green | done - 121 tests; stress, chaos, slow start, decision log, drift guard, `ml_predictive`/`ml_herd` split, p95-over-time panel and Nginx/HAProxy reference configs |
| M8 | 12 | Raw results complete | done - 125/125 cells, 0 failed; tables, CIs, CDFs, burst/failure timelines and analysis committed |
| M9 | 14 | Final submission | partial - report, slides, demo and viva written; no screen recording and no timed fresh-machine setup |

See `docs/dataset.md` for the per-cell table, the signal gate, the model
comparison and the M5 verdict, `docs/serving.md` for the serving path and the
A/B evidence, `docs/resilience.md` for the safeguards, and
`docs/monitoring.md` for the decision log and drift guard.

### Open items against the plan

Phase 5 and 6 name a few things this repo does not have yet. They are listed
rather than implied to be done:

- **Reference configs are configs, not measurements.** Nginx and HAProxy start,
  serve traffic through the same backends and are documented in `deploy/`, but
  they write no request log of the shape this project measures from, so they are
  not in the Phase 7 campaign yet. A client-side harness is the remaining work.
- **Phase 7 matrix.** The 125-cell campaign is complete (125/125 ok). The
  pre-registered result is negative on H1: ML is not significantly better in
  three scenarios. The 18-cell dataset campaign documented in `docs/dataset.md`
  remains the reduced campaign the model was trained on.
- **M6's burst criterion.** The matrix mean is 243 vs 275 ms in ML's favour but
  not significant; the earlier dedicated A/B lost (138 vs 77 ms). M6 stays
  partial because there is no consistent, significant burst win.
- **M9 evidence.** The demo and runbook are written, but there is no screen
  recording and no timed fresh-machine setup. Those are the two missing pieces
  of the plan's M9 checklist.

## License
MIT
# AI/ML-Based Intelligent Load Balancer

A Layer-7 HTTP load balancer that uses machine learning to predict per-server response times and route requests to the fastest backend. Built as a semester project with a 14-week timeline.

## Repository Layout

| Directory | Purpose |
|-----------|---------|
| `balancer/` | Reverse proxy, decision engine, routing algorithms |
| `backend/` | FastAPI/Flask backend apps with `/metrics` and `/health` endpoints |
| `agent/` | psutil metrics agent running inside each backend container |
| `ml/` | Feature engineering, model training, inference |
| `bench/` | Benchmark suites, Locust traffic scripts, run matrix |
| `docs/` | Hypotheses, design docs, architecture diagrams |

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
without restarting anything.

```bash
docker compose up -d                                   # dashboard at localhost:9090
curl -X POST "http://localhost:9090/admin/algorithm?name=ml"
```

The trained artifacts are committed, so `ml` works on a fresh clone. `data/` is
not committed (20 MB of Parquet, rebuilt by the collector); after collecting
your own, retrain with `python ml/train.py --data data --output models`.

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

`mixed_cost`, 24 users, three repeats:

| policy | req/s | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| least_conn | 161 | 33 ms | 100 ms | 178 ms |
| ml | 135 | 56 ms | 133 ms | 212 ms |
| round_robin | 77 | 35 ms | 1205 ms | 2823 ms |

The learned router beats the plan's round-robin baseline by about 9x on p95 and
1.7x on throughput, and loses to least_conn by about 40% on p95. Both held in
every repeat. `docs/serving.md` explains why, and documents the herding bug the
first serving version had and the queue correction that fixed it.

Under the `degraded` profile - one backend 400 ms slower and failing 35% of
requests - ml routes 0-1% of traffic to the faulted backend against
round_robin's fixed 25%, which shows up where it matters:

| policy | req/s | p95 | p99 | errors |
|---|---:|---:|---:|---:|
| ml | 160 | 162 ms | 279 ms | **0.26%** |
| round_robin | 136 | 414 ms | 526 ms | **8.73%** |

## Milestones

| Milestone | Week | Focus | Status |
|-----------|------|-------|--------|
| M1 | 1 | Project skeleton and CI green | done |
| M2 | 3 | Baseline balancer works end-to-end | done |
| M3 | 5 | Live metrics and logs flowing | done |
| M4 | 6 | Dataset v1 (200k+ rows) | done - 218,011 rows, 18 cells |
| M5 | 8 | Model beats naive baseline by 30%+ MAE | done - +53.9% (XGBoost) |
| M6 | 9 | MVP: ML balancer demo | partial - demo works and beats round_robin; staleness fallback and the burst-vs-least_conn trial are open |
| M7 | 11 | Feature freeze, all tests green | partial - 90 unit/model tests green in CI; stress, chaos and slow start are open |
| M8 | 12 | Raw results complete | planned |
| M9 | 14 | Final submission | planned |

See `docs/dataset.md` for the per-cell table, the signal gate, the model
comparison and the M5 verdict, and `docs/serving.md` for the serving path and
the A/B evidence.

### Open items against the plan

Phase 5 and 6 name a few things this repo does not have yet. They are listed
rather than implied to be done:

- **Staleness fallback.** When the collector's snapshots go stale, every
  server is unhealthy and the balancer answers 503. The plan asks for a switch
  to least-connections instead. The missing-model fallback does exist: a bad or
  absent artifact routes on least-connections and says so in `/stats`.
- **Drift guard.** No rolling-MAE monitor that demotes a model whose live
  error grows; served predictions are not persisted, so it needs a decision
  log first.
- **`ml_predictive` vs `ml_herd`.** Herd protection is built into `ml` (the
  in-flight queue term) rather than exposed as two selectable algorithms.
- **Resilience.** No circuit breaker, single retry, or slow start on rejoin. A
  dead backend is failed out by health checks after three failed polls, about
  6 s at the configured 2 s interval.
- **Phase 6 hardening.** No stress run above capacity, no chaos script, no
  Nginx/HAProxy reference configs. The dashboard shows per-server CPU,
  in-flight, EWMA, p95 and errors, and the ML artifact/decision counters, but
  not predicted-vs-actual, a decision log, or p95 over time.

## License
MIT
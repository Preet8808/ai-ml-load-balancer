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
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

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

## Milestones

| Milestone | Week | Focus |
|-----------|------|-------|
| M1 | 1 | Project skeleton and CI green |
| M2 | 3 | Baseline balancer works end-to-end |
| M3 | 5 | Live metrics and logs flowing |
| M4 | 6 | Dataset v1 (200k+ rows) |
| M5 | 8 | Model beats naive baseline by 30%+ MAE |
| M6 | 9 | MVP: ML balancer demo |
| M7 | 11 | Feature freeze, all tests green |
| M8 | 12 | Raw results complete |
| M9 | 14 | Final submission |

## License
MIT
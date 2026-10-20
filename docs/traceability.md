# Requirements traceability

Every deliverable the plan names, where it lives, and its honest status. "Done"
means there is a command or a committed artifact that demonstrates it; "partial"
means something the plan asked for is missing and is named.

## Milestones

| Milestone | Required | Evidence | Status |
|---|---|---|---|
| M1 | Repo, CI, requirements, hypotheses draft | `README.md`, `.github/workflows/`, `requirements.txt`, `docs/hypotheses.md` | done |
| M2 | Baseline balancer end to end | round robin / least connections behind `choose_server`, `docker-compose.yml`, `bench/check_stack.py` | done |
| M3 | Live metrics and logs flowing | 1 Hz collector, `/metrics`, 16-column request log, `/stats` | done |
| M4 | Dataset v1, 200k+ rows, 20+ runs | 218,011 rows in 18 runs (six profiles x three policies) | partial - 18 runs, not 20+; documented |
| M5 | Model beats naive by 30%+ MAE | XGBoost 0.5868 vs naive 1.2738 = **+53.9%**, R² 0.683, CV 0.6455 | done - `docs/dataset.md` |
| M6 | MVP: ML balancer demo | live A/B evidence, stale-feed fallback, routing overhead 1.96 ms; 125-cell matrix | partial - H1 not met; burst is not a significant win and the earlier A/B lost; kept as a negative result |
| M7 | Tests green, stress/chaos/slow start, dashboard, reference configs | 121 tests; `docs/resilience.md`; dashboard with p95 over time; `deploy/` | done |
| M8 | 125 runs completed or documented failures | `docs/results.md`, `docs/results/results.jsonl`, `bench/run_matrix.py` | done - 125/125 ok, 0 failed; 60 s runs, deviation documented |
| M9 | Report, README, slides, demo, viva | `docs/report.md`, `docs/slides.md`, `docs/demo.md`, this repository | partial - no screen recording; setup not timed on a fresh machine |

## Phase deliverables

| Plan item | Where | Status |
|---|---|---|
| Backends with `/metrics`, `/health`, fault endpoints | `backend/`, `docker-compose.yml` | done |
| YAML config selecting algorithm, servers, thresholds | `config.yaml`, `balancer/config.py` | done |
| Six algorithms behind one interface | `balancer/algorithms.py`, `balancer/ml_router.py` | done |
| ML decision engine, herd protection, fallback | `balancer/ml_router.py` | done |
| SLA guard | `sla_ms` in `balancer/main.py`; argument in `docs/results.md` | partial - with an argmin policy the planned filter is provably redundant; SLA is configurable, reported and measured instead |
| Stale-feed fallback | `balancer/resilience.py`, `bench/check_stale_feed.py` | done |
| Retry, circuit breaker, slow start | `balancer/resilience.py`, `bench/check_chaos.py` | done |
| Metrics agent and collector | `backend/main.py`, `balancer/collector.py` | done |
| Six traffic profiles | `bench/traffic_profiles.py` | done |
| Fault injection | `/slow`, `/fault`, `bench/inject_fault.sh` | done |
| Signal gate (metrics-only R² >= 0.3) | `bench/signal_check.py` | done - 0.5499 |
| Run-level split, naive baseline, model comparison | `ml/train.py`, `docs/dataset.md` | done |
| Ranking test | Spearman 0.710, documented in `docs/dataset.md` | done - top-1 server ranking is not identifiable from observational logs; rationale in `docs/dataset.md` |
| Decision log (separate schema, join key) | `balancer/decision_log.py`, `docs/monitoring.md` | done |
| Drift guard | `balancer/drift.py`, `bench/check_drift.py` | done |
| Unit, integration, model tests in CI | `balancer/tests/`, `ml/tests/`, GitHub Actions | done - 121 tests |
| Load, stress, chaos checks | `bench/check_stress.py`, `bench/check_chaos.py` | done - evidence recorded |
| Dashboard: CPU, in-flight, predicted vs actual, p95 over time, decisions | `balancer/dashboard.html` | done |
| Nginx and HAProxy reference configs | `deploy/nginx.conf`, `deploy/haproxy.cfg` | done - configs validated; no measured comparison |
| 125-run matrix, randomised order, seeded | `bench/run_matrix.py` | done - 125/125 ok, 0 failed, 5x5x5, seed 42 |
| Per-run manifest: seed, config, commit | `docs/results/manifest.json`, `results.jsonl` | done |
| Raw results (CSV) | `docs/results/summary.csv` | done - per-run summary; request-level raw Parquet gitignored |
| Ablations: no trend, no request type, no herd | `docs/ablations.md`, `docs/results/` | done - offline reports and live herd A/B (342 vs 700 ms p95) |
| Unseen-profile generalisation | `ml/generalisation.py`, `docs/ablations.md` | done - burst +59.6%, wave +58.8% |
| Statistics: means, 95% CI, bootstrap, Welch/Mann-Whitney | `bench/analyse_matrix.py`, `docs/results.md` | done |
| Latency CDFs, burst time series (latency + per-server CPU), p95 bars, failure timeline, overhead table | `docs/img/`, `docs/results.md` | done |
| H1-H4 verdicts | `docs/results.md` | done - H1 **not met**; H2, H3, H4 met |
| Report | `docs/report.md` | done |
| README with setup | `README.md` | done - not timed on a fresh machine |
| 15-slide deck | `docs/slides.md` | done - 15 slides |
| 5-minute demo | `docs/demo.md` | done - script; no recording |

## Deliberate non-goals

- RL, LSTM and Prometheus/Grafana are stretch goals in the plan and were not
  attempted. The plan allows them only after M6 and only if the benchmark is on
  schedule; M6's burst criterion was not met, so they stayed out.
- No cloud deployment: the plan's test bed is a local compose stack.

## Known gaps, stated rather than hidden

1. 18 dataset runs, not 20+ (M4 wording).
2. H1 is not met: ML is not significantly better in three scenarios. It loses
   steady, ramp and failure; burst is a non-significant mean win in the matrix
   and a loss in the earlier A/B, so there is no consistent burst claim.
3. Live prediction error is high (MAE 1.394 steady, 2.421 failure vs 0.5868
   offline), and the collapsed runs avoid backend4. The matrix pins the guard at
   10; at the shipped 1.0 it would demote. That is a finding, not a tested fix.
4. Nginx/HAProxy are configs, not measured baselines.
5. No screen recording and no fresh-machine timing for the README.
6. The 60 s window is shorter than the plan's 5 minutes; burst has one spike and
   the ramp is steeper. The campaign is one host/session, so absolute throughput
   is not portable.
7. Top-1 server-ranking accuracy is not identifiable from observational logs;
   Spearman is reported instead. The SLA filter is provably redundant under
   argmin and is configurable/reported rather than a routing filter. CPU
   imbalance is a traffic-weighted proxy, not a fixed-clock instrument.

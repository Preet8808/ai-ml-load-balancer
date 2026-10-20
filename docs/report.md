# AI/ML Load Balancer - Final Report

**Author:** Preet
**Project:** ML-driven request routing for heterogeneous HTTP backends
**Repository:** <https://github.com/Preet8808/ai-ml-load-balancer>
**Plan:** *AI/ML Load Balancer Project Plan*, 14 weeks, milestones M1-M9

---

## 1. Abstract

This project builds a load balancer that predicts each backend's response time
for the request in front of it and routes to the server with the lowest
predicted wait, while keeping a set of traditional algorithms behind the same
interface. It compares the learned router against round robin, weighted round
robin, least connections and least response time on five traffic scenarios,
five repetitions each, at the run level.

The model beats the naive baseline from the plan by 53.9% MAE (log space) on a
run-held-out test set, and keeps that advantage on two traffic profiles it never
saw during training. On live traffic the picture is mixed and is reported as
such: the learned router has the lowest mean p95 in the burst and degraded
scenarios, but it is worse than least-connections in steady, ramp and failure.
H1 (three significant wins) is **not met**; H2-H4 pass, H2 only against the
weakest baseline. The negative result is traced to live prediction error and a
capacity-avoidance failure mode, and is documented rather than smoothed over.

---

## 2. Problem and hypotheses

Four backends share a host but not a CPU quota: 0.8, 0.4, 0.4 and 0.15 cores.
Request costs vary by endpoint (a `ping` is cheap, a `payload` or `cpu` request
is several times more expensive). A router that balances *connections* treats
those servers as equal and over-sends the slow one; a router that predicts the
*response time of this request on this server* can price both the hardware and
the request.

The plan fixes the hypotheses before the campaign:

| # | Hypothesis | Test |
|---|---|---|
| H1 | ML p95 is lower than the best traditional baseline in at least 3 of 5 scenarios, with statistical significance | run-level p95, Welch t-test, Holm correction |
| H2 | On steady traffic with identical servers, ML is not more than 5% worse than Round Robin | p95 ratio, steady scenario |
| H3 | Routing adds under 5 ms per request | decision timer in the router |
| H4 | After a server failure, traffic is redirected within 5 s | request log timestamps around the stop |

---

## 3. System

```text
Locust (own container, 1 CPU)
      |
      v
Balancer (FastAPI, 2 CPU)  ----->  /metrics collector (1 Hz, threaded)
  |   algorithm: round_robin | weighted_rr | least_conn |
  |              least_response_time | random | ml
  |   resilience: retry once on another server, circuit breaker,
  |               slow start, stale-feed fallback
  |   logging: requests.parquet (16 columns) + decisions.parquet (14 columns)
  v
backend1 (0.8 CPU)  backend2 (0.4)  backend3 (0.4)  backend4 (0.15)
      \____________________ | ____________________/
                    /health, /metrics, /ping, /cpu, /io, /payload
```

Every algorithm implements the same interface, so the proxy, the resilience
layer, the logs and the dashboard are identical across policies; only
`choose_server(request, servers)` changes. That is what makes the A/B
comparisons attributable to the algorithm.

Key decisions:

- **Metrics-only signal gate.** A model must reach R² >= 0.3 using only
  server-state features (no request type, no payload) before request features
  are allowed to help. It passed at R² 0.5499 on the held-out runs.
- **Log-space target.** `log1p(response_ms)` keeps the heavy tail from
  dominating squared error; the plan's M5 MAE is measured in that space.
- **Herd protection.** The score is `predicted_ms * (1 + inflight)`: the
  prediction prices the request, the live in-flight count prices the queue the
  2-second-old snapshot cannot see.
- **One safety layer.** Retry, breaker, slow start and stale-feed fallback live
  in `resilience.py` and apply to every algorithm, so resilience cannot be
  credited to one policy.
- **Two logs.** The request log stays exactly the plan's 16 columns; routing
  decisions write their own 14-column Parquet with the prediction, so the two
  can be joined on `(run_id, timestamp)` without contaminating the training
  schema.

---

## 4. Dataset and model

The dataset is 218,011 requests over 18 runs: six traffic profiles (steady,
ramp, burst, wave, mixed cost, degraded) crossed with three collection policies
(round robin, least connections, random). Failed requests are excluded from
training; their effect reaches the model through `err_rate_5s` instead. Each
run's warm-up window is dropped.

The model comparison is by run, never by row:

| Model | MAE(log) | MAE(ms) | R²(log) | Spearman |
|---|---|---|---|---|
| Naive (server's own EWMA) | 1.2738 | 140.5 | -0.2907 | 0.414 |
| Ridge | 0.6402 | 111.5 | 0.6449 | 0.692 |
| Random Forest | 0.5912 | 75.4 | 0.6757 | 0.704 |
| **XGBoost (shipped)** | **0.5868** | **73.1** | **0.6827** | **0.710** |

Grouped 5-fold cross-validation on the training runs gives R² 0.6455, so the
result is not one lucky split. Against the plan's 30% target the shipped model
is **53.9% better in MAE**. Full detail: `docs/dataset.md`.

The plan also asks for top-1 ranking accuracy. It is not identifiable from this
data: the log records the latency of the server that served each request and
never the same request on the other three, so there is no observed "fastest
server" to rank against. Spearman is reported instead, and the limitation is
stated in `docs/dataset.md` rather than replaced with a proxy that would look
better than it is.

---

## 5. Serving path

`MLRouter.choose_server` builds one feature row per healthy server, predicts all
of them in a single `inplace_predict` call, converts to milliseconds, and
charges each candidate its outstanding work. Ties within 1 ms are broken by
in-flight count, so an uninformative model still cannot pile every request on
one backend.

Failure modes, all measured on live traffic:

- **No artifact / malformed artifact** -> least connections, recorded as
  `ml-fallback`.
- **Stale metrics feed** (> 3 s) -> route from the last snapshot's healthy
  servers; a 30-request check during a 3,109 ms feed stop returned 30/30 200s.
- **Drift** -> the guard scores predictions against outcomes in log space and
  demotes to least connections above the threshold. Shadow scoring continues
  while demoted, and a clean window promotes the model back. Demonstrated live
  by lowering the threshold to 0.15, observing 112 `drift-fallback` decisions,
  and restoring it.

**SLA guard.** The plan excludes servers predicted above the SLA. With an argmin
policy that filter is redundant: the minimum score is below the SLA whenever any
candidate is, so it cannot change the choice; it only makes sense with a
softmax/two-choices picker, which was rejected because it destabilises routing
against a stale snapshot. The SLA is instead configured (`sla_ms`, 500 ms),
reported in `/stats`, and measured per run as the share of successful requests
over 500 ms. `docs/results.md` records that argument rather than claiming a rule
that never fires.

The decision log turned out to be essential: it is how "was the model actually
routing?" is answered. See `docs/monitoring.md`.

---

## 6. Resilience

Shared by all algorithms:

| Safeguard | Behaviour | Evidence |
|---|---|---|
| Retry | one retry on a different server for 500/502/503/504, logged as one row | chaos runs, 0 errors |
| Circuit breaker | 3 failures open it for 10 s; half-open probe | killed backend bypassed in 1.09 s |
| Slow start | recovered server ramps its share over 10 s | rejoin check |
| Stale feed | snapshot fallback above 3 s | 3,109 ms stop, 30/30 200s |
| Stress | 80 users / 60 s: 7,353 requests, all 200, no restart, RSS +3% | `docs/resilience.md` |

The safeguards are deliberately not part of the routing algorithm: an ML run and
a least-connections run get the same retry and breaker behaviour, so a
comparison isolates routing.

---

## 7. Test strategy

121 automated tests, all green in CI (Python 3.11, Linux):

- **Unit:** algorithms, fallback, router layout, tie-breaking, drift maths and
  hysteresis, logger schemas, resilience state machine, the 60 s latency window.
- **Integration:** proxy end to end, retry semantics, stale feed, admin
  endpoints, decision/request log join.
- **Model:** artifact layout, serving array order, prediction units.
- **Load / stress / chaos:** Locust drivers and the scripts in `bench/`:
  `check_stack.py`, `check_stress.py`, `check_chaos.py`, `check_faults.py`,
  `check_stale_feed.py`, `check_drift.py`.

Load, stress and chaos are operational checks with recorded evidence, not
pytest cases: they need the compose stack and a real container kill.

---

## 8. Experiment design

Five algorithms x five scenarios x five repetitions = 125 cells.

| Scenario | Definition |
|---|---|
| steady | constant 30 users, identical servers |
| ramp | concurrency climbs 15% -> 100% over the run |
| burst | periodic spikes above the 30-user baseline |
| degraded | mixed endpoint cost while backend3 is slowed by 400 ms and returns 35% errors |
| failure | steady traffic while backend3 is stopped at 40% and restarted at 70% of the run |

Order is shuffled with a fixed seed, so host drift is spread across policies
rather than attributed to whichever policy ran second. Every cell records a
manifest: commit, seed, duration, users, CPU samples, and the fault timeline.

Deviations from the plan, all recorded in the manifest:

1. **Run length 60 s, not 5 min.** 125 x 6 min is about 12.5 h of continuous
   load on one laptop; the design is unchanged but the window is shorter. p95
   from ~6,000-9,000 requests per cell is still stable, and repetitions carry
   the uncertainty.
2. **Warm-up 12 s per cell, not 1 min.** Discarded, as the plan requires. The
   stack is also warmed before the campaign. A 1-minute warm-up per cell would
   add two hours of heat for no extra independent variable.
3. **Drift guard threshold pinned at 10 for ML cells.** The guard is a safety
   layer, not the policy under test. At its shipped threshold of 1.0 it demoted
   the model during the ramp's low-load phase, which would have made the cell a
   least_conn measurement with an `ml` label. The live MAE is recorded on every
   cell, so the model's real error remains visible.

---

## 9. Results

The campaign completed **125/125 cells, 0 failed**, in one randomised session
from commit `3d9246e` (seed 42, 60 s measured + 12 s discarded warm-up, drift
guard pinned at 10.0). All tables and charts are generated by
`bench/analyse_matrix.py`; the per-cell records, summary CSV and manifest are in
`docs/results/`. The full tables, CDFs, burst CPU timeline and failure timeline
are in `docs/results.md`.

### p95 by scenario (ms, mean of 5 runs; bootstrap 95% CI in brackets)

| Scenario | Round Robin | Weighted RR | Least Conn | Least RT | ML (herd) |
|---|---|---|---|---|---|
| Steady | 789 [484-1,094] | 396 [318-475] | **214 [141-287]** | 302 [232-379] | 433 [211-821] |
| Ramp | 575 [489-664] | 287 [242-329] | **169 [118-219]** | 598 [282-1,136] | 273 [190-379] |
| Burst | 733 [556-935] | 372 [285-463] | 275 [148-450] | 450 [311-598] | **243 [179-308]** |
| Mixed cost + degraded server | 474 [444-513] | 433 [334-524] | 332 [247-427] | 269 [180-386] | **182 [154-218]** |
| Server failure mid-run | 612 [438-855] | 555 [387-813] | **209 [145-302]** | 636 [428-845] | 734 [367-1,119] |

### Throughput (requests/s, mean of 5 runs)

| Scenario | Round Robin | Weighted RR | Least Conn | Least RT | ML (herd) |
|---|---|---|---|---|---|
| Steady | 112 | 118 | **150** | 129 | 109 |
| Ramp | 133 | 143 | **160** | 109 | 121 |
| Burst | 120 | 124 | **141** | 107 | 127 |
| Mixed cost + degraded server | 113 | 101 | **122** | 118 | 120 |
| Server failure mid-run | 130 | 111 | **155** | 92 | 80 |

All 25 cells have 0.00% final 5xx: the breaker, retry and health checks are
shared, so the failure cost is latency, not errors. ML's SLA >500 ms shares are
7.0, 1.0, 0.5, 0.5 and 16.2% in steady/ramp/burst/degraded/failure;
least-connections is 0.6, 0.3, 2.0, 1.6 and 0.5%. The failure cell is the clear
regression.

### Hypothesis verdicts

- **H1: not met.** ML is significantly better on 0 of 5 scenarios (needs 3).
  Mean p95 is lower in burst (243 vs 275 ms, -11.7%) and degraded (182 vs 269 ms,
  -32.5%), but neither survives Holm correction at n=5 (both adjusted p=1.0).
  It is worse in steady (+101.9% vs least-connections), ramp (+62.1%) and failure
  (+251.8%).
- **H2: met, with a caveat.** Steady ML p95 is 0.55x round robin (433 vs 789 ms),
  but 2.02x worse than least-connections (214 ms). It passes against the weakest
  baseline, not against the strongest.
- **H3: met on the mean.** ML decision cost is 1.96 ms; the worst single decision
  is 126.72 ms, a scheduler/Python tail outlier. The H3 bar is mean < 5 ms.
- **H4: met.** Traffic left the stopped server within 1.13 s in the worst of 25
  failure runs; final errors are 0.00%; traffic returns 7.7 s after restart on
  average, including slow start. This is a platform property shared by all five
  policies, not an ML-specific win.

### What the numbers say

The negative is the headline: the offline model win does not become a consistent
live routing win. The mean p95 advantage in degraded is genuine but not
statistically separated; burst is a small, non-significant mean gain. ML loses
steady, ramp and failure, and has lower throughput in all five scenarios. The
one-scenario means are also pulled by real failure-mode runs: steady ML p95s are
195, 304, 199, 267 and 1200 ms (median 267); failure p95s are 1335, 368, 1222,
356 and 391 ms (median 391). The 1200 ms steady run and two collapsed failure
runs completed only 42-45 req/s, sent 4-7% of traffic to the 0.15-core backend4,
and recorded live MAE near 3.0. When prediction error rises, the router avoids
the small server, loses that capacity to the other three, and queues build. The
runs are kept, not deleted; they are why the confidence intervals are wide and
why the tests do not reject equality, but the practical regression remains.

The model's live MAE is 1.394 (steady), 1.151 (ramp), 0.898 (burst), 0.519
(degraded) and 2.421 (failure), versus 0.5868 offline. The drift guard was pinned
at 10.0 to measure routing rather than the guard, so no cell demoted; at the
shipped 1.0 the steady/ramp/failure windows would have crossed. The herd
ablation confirms the queue term is essential: on burst, `ml` beats
`ml_predictive` 3/3 pairs, p95 342 vs 700 ms (median ratio 2.21x), 102 vs 73
req/s. The routing machinery works; live prediction quality is the bottleneck.

The full test grid and p-values are in `docs/results.md`; the key conclusion is
that H1 must be reported as not met, not reframed.

---

## 10. Ablations and generalisation

Offline, on the same run split and XGBoost hyperparameters as the shipped
model:

| Arm | MAE(log) | MAE(ms) | R² | Change vs full |
|---|---|---|---|---|
| Full feature set | 0.5868 | 73.1 | 0.6827 | - |
| No trend features (`cpu_delta_5s`, `rt_p95_5s_ms`) | 0.6124 | 78.4 | 0.6565 | +4.4% MAE |
| No request-type features | 0.6955 | 122.1 | 0.5793 | +18.5% MAE |

Request type is the more valuable feature group: without it the model cannot
know that a `payload` request costs several times a `ping`, which is the core
signal of the mixed-cost scenario.

Holding out a whole profile, so the model never sees that traffic shape:

| Held-out profile | Naive MAE(log) | Model MAE(log) | Improvement | R² | Spearman |
|---|---|---|---|---|---|
| burst | 1.3677 | 0.5530 | +59.6% | 0.6797 | 0.708 |
| wave | 1.3385 | 0.5521 | +58.8% | 0.7013 | 0.737 |

The no-herd-protection arm is measured live rather than offline (it is a routing
rule, not a feature): `ml_predictive` sends by prediction alone and is compared
with `ml` on burst traffic. The herd term is faster in 3/3 paired repeats
(p95 342 vs 700 ms, median ratio 2.21x; 102 vs 73 req/s). Without it the router
sends 46-69% to backend1 and only 1-3% to backend4; with it the shares are
29/28/29/14. Full output: `docs/results/herd_ablation.txt`.

---

## 11. Threats to validity

- **One host, one laptop.** Absolute throughput is not portable; only
  within-session paired comparisons are. Session-to-session throughput varies
  20-40%, which is why the campaign uses one session and randomised order.
- **Run length.** 60 s windows stress the p95 estimator less than the plan's
  5-minute windows and give the burst scenario one spike rather than several.
- **Client-bound extremes.** Ad-hoc host-side load scripts top out at 64-70
  req/s; they are never quoted as balancer throughput. The campaign uses Locust
  in its own cgroup at 12-13% CPU.
- **Drift guard pinned.** The campaign measures routing, not the guard. The
  guard's own behaviour, including the cold-ramp demotion, is reported
  separately.
- **One fault server.** `backend3` is the degraded/killed server everywhere.
  The other backends differ only by CPU quota, which is the intended treatment.
- **Decision cost tail.** ML's mean decision cost is 1.96 ms, but one decision
  took 126.72 ms, traced to a scheduler/Python pause rather than model scoring.
  H3 is a mean-cost hypothesis; the maximum is reported rather than trimmed.
- **Heavy tails at n=5.** One bad ML steady run and two bad failure runs move the
  mean substantially. Medians and confidence intervals are reported alongside;
  deleting the runs would be cherry-picking. The failure regression is large
  enough to remain practically important even though the run-level test does not
  reject equality.

---

## 12. Conclusion

The prediction task is solved well beyond the plan's bar: 53.9% MAE improvement,
held out by run, and it survives a whole unseen traffic profile. The routing
task is harder, and the pre-registered 125-cell matrix gives a conditional and
mostly negative answer. ML has the lowest mean p95 in burst and degraded, but
neither difference is statistically significant; it is worse than
least-connections in steady, ramp and the failure scenario. H1 is not met. The
live model error (MAE 1.394 steady, 2.421 failure, versus 0.5868 offline) and a
capacity-avoidance failure mode - the router sends only 4-7% to the small
backend in the collapsed runs - explain the gap. The report keeps those runs and
reports the medians, because removing them would turn a real failure into a
better-looking average.

The most useful engineering results are the safeguards around the model: the
queue term (which cuts burst p95 from 700 to 342 ms in the paired ablation), the
stale-feed fallback, the circuit breaker with slow start, the decision log that
makes "was it really the model?" answerable, and a drift guard whose cold-ramp
demotion is itself a finding. The project's contribution is a reproducible
negative result and the instrumentation that makes it trustworthy, not a claim
that ML beats the heuristics everywhere.

Milestone status: M1-M3, M5, M7 and M8 are done; M4 is partial (18 dataset runs,
not the plan's 20+); M6 is partial (H1 not met; burst not a significant,
consistent win); M9 is partial (no screen recording and no timed fresh-machine
setup). The full map is in `docs/traceability.md`.

---

## 13. Reproducibility

```bash
docker compose up -d --build                     # stack + dashboard on :9090
python -m pytest -q                              # 121 tests
python ml/train.py --data data --output models   # retrain; needs data/
python bench/run_matrix.py                       # 125-cell campaign (resumable)
python bench/analyse_matrix.py                   # tables + charts
```

`data/`, `logs/`, `decisions/` and the matrix raw files are gitignored: they are
generated, reproducible from the commands above, and too large for the repo. The
per-run manifest and summary CSV are committed under `docs/results/`.

# Final presentation - 15 slides

Each slide is a heading, the bullets to put on it, and a note of what to say.
Numbers trace to `docs/results.md`, `docs/dataset.md` and `docs/monitoring.md`.

---

## Slide 1 - Title

**An ML load balancer for heterogeneous backends**

- Predict response time per request and per server, then route to the lowest
  predicted wait
- Compared against four traditional algorithms on five traffic scenarios
- Preet - 14-week project

*Say:* one sentence on the problem: the four backends share a host but not a
CPU quota, so equal connection counts are not equal load.

---

## Slide 2 - Why least-connections is not enough

- Backends: 0.8, 0.4, 0.4 and 0.15 CPU
- Requests: `ping` is ~5x cheaper than `payload`
- Least connections counts *requests*, not *work*
- The plan's question: can a 2-second metrics feed predict the wait well enough
  to beat that?

*Say:* if I send the same number of payload requests to the 0.15-core server and
the 0.8-core server, the queue on the small one is 5x longer.

---

## Slide 3 - Hypotheses, fixed before the campaign

- **H1** ML p95 beats the best traditional baseline in >= 3 of 5 scenarios,
  with significance
- **H2** Steady traffic: not more than 5% worse than round robin
- **H3** Routing overhead under 5 ms
- **H4** Traffic leaves a failed server within 5 s

*Say:* the plan asks for these before any final run; they are listed in
`docs/report.md` and not adjusted afterwards.

---

## Slide 4 - Architecture

```text
Locust (1 CPU) -> Balancer (2 CPU) -> backend1 0.8 | backend2 0.4
                        |              backend3 0.4 | backend4 0.15
                        +- 1 Hz metrics collector
                        +- resilience: retry, breaker, slow start
                        +- logs: requests (16 cols) + decisions (14 cols)
```

*Say:* every algorithm sits behind one `choose_server` interface, so the proxy,
retry, breaker and logging are identical across policies.

---

## Slide 5 - Six algorithms, one interface

- round robin, weighted round robin, least connections, weighted least
  connections, least response time, random
- `ml` = predicted latency x (1 + in-flight)
- `ml_predictive` = prediction only (the plan's ablation)
- Switching is an admin call, no restart

*Say:* the A/B comparison changes only the decision function.

---

## Slide 6 - Dataset and signal gate

- 218,011 requests, 18 runs: 6 profiles x 3 collection policies
- Faults injected during collection; failed requests excluded from training
- Gate: metrics-only R² >= 0.3 -> **passed at 0.5499**
- Split by run, never by row

*Say:* the gate is the plan's defence against leakage: if the server state alone
cannot predict, request features would be doing the work.

---

## Slide 7 - Model selection (M5)

| Model | MAE(log) | R² | Spearman |
|---|---|---|---|
| Naive EWMA | 1.2738 | -0.29 | 0.414 |
| Ridge | 0.6402 | 0.645 | 0.692 |
| Random Forest | 0.5912 | 0.676 | 0.704 |
| **XGBoost** | **0.5868** | **0.683** | **0.710** |

- **+53.9% MAE over the naive baseline**, target was 30%
- Grouped CV R² 0.6455

*Say:* the model is not just fitting one split; the CV number is on the training
runs.

---

## Slide 8 - Serving

- One feature row per healthy server, one batch prediction
- Queue term: `pred * (1 + inflight)`; ties broken by in-flight count
- Stale feed (> 3 s): serve from the last snapshot
- No artifact: least connections, labelled `ml-fallback`

*Say:* the queue term is what stops the model from sending everything to the
server that looked idle two seconds ago.

---

## Slide 9 - Decision log and drift guard

- 14-column Parquet beside the 16-column request log
- Join on `(run_id, timestamp)`: 2,601/2,601 exact in the check
- Drift: live MAE in log space, demote > 1.0, promote < 0.8
- Demonstrated live: 112 `drift-fallback` decisions, then promotion

*Say:* without the decision log I could not tell a model-driven run from one
that silently fell back.

---

## Slide 10 - Resilience

| Check | Result |
|---|---|
| Metrics feed stopped 3.1 s | 30/30 requests 200 |
| Backend killed | bypassed in 1.09-1.13 s, 0 errors, traffic back in 5.0-7.7 s |
| 80 users / 60 s | 7,353 requests, all 200, RSS +3% |
| Slow start | 10 s ramp on rejoin |

*Say:* these safeguards are shared, so an ML win cannot come from better retry
behaviour.

---

## Slide 11 - Experiment design

- 5 algorithms x 5 scenarios x 5 repetitions = **125 cells**
- Scenarios: steady, ramp, burst, mixed cost + degraded server, server failure
- Randomised order, fixed seed, per-run manifest
- Deviations from the plan: 60 s runs (not 5 min), 12 s warm-up, drift threshold
  pinned for ML cells

*Say:* the deviations are in the manifest and the report; 125 x 6 minutes is
12.5 hours on one laptop.

---

## Slide 12 - Results: p95 by scenario

- Chart: `docs/img/p95_by_scenario.png`
- Mean of 5 runs, ms; `n.s.` = not significant after Holm

| Scenario | ML | Best baseline | Change |
|---|---|---|---|
| Steady | 433 | 214 (least conn) | +102% |
| Ramp | 273 | 169 (least conn) | +62% |
| Burst | 243 | 275 (least conn) | -12% n.s. |
| Degraded | 182 | 269 (least RT) | -32% n.s. |
| Failure | 734 | 209 (least conn) | +252% |

*Say:* ML is lowest on the mean in burst and degraded, but neither difference is
significant. It loses steady, ramp and failure; least-connections wins three of
five. Do not skip the losses.

---

## Slide 13 - Hypotheses

- H1: **not met** - 0/5 significant wins (needs 3); burst and degraded are
  favourable means but Holm-adjusted p=1.0
- H2: **met** - 0.55x round robin (433 vs 789 ms), but 2.02x worse than least
  connections (214 ms)
- H3: **met** - mean decision 1.96 ms; max 126.72 ms is a scheduler tail
- H4: **met** - worst failover 1.13 s, 0.00% final 5xx; traffic returns 7.7 s
  after restart (slow start)

*Say:* H3 and H4 are the engineering claims; H1 is the research claim and it
failed. H2 passes only against the weakest baseline.

---

## Slide 14 - Ablations and generalisation

- No trend features: +4.4% MAE; no request type: +18.5% MAE
- Unseen profile: burst +59.6%, wave +58.8% vs naive
- Live herd A/B, burst: p95 **342 vs 700 ms** (2.21x), 102 vs 73 req/s; herd
  wins 3/3 paired runs
- No-herd routing sends 46-69% to backend1 and 1-3% to backend4

*Say:* request type is the feature group the model cannot lose; the queue term
is the clearest live win, even when the model's prediction is imperfect.

---

## Slide 15 - Honest negatives and lessons

- H1 is not met; mean ML p95 is worse in steady, ramp and failure (failure 734
  vs 209 ms, SLA 16.2% vs 0.5%)
- Burst is not consistent: matrix 243 vs 275 ms (n.s.), earlier A/B 138 vs 77 ms
- Live MAE is 1.394 steady / 2.421 failure vs 0.5868 offline; collapsed runs
  send 4-7% to backend4 and lose capacity
- Drift guard was pinned at 10; at shipped 1.0 it would demote; that is a next
  experiment, not a claimed fix
- M9 remains partial: no screen recording, no timed fresh-machine setup

*Say:* the negative results are the most credible part: they come from the same
logs as the wins. The next step is retraining on fault transients, adding an
explicit capacity term, and testing the tighter guard.

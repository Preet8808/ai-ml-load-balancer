# Serving the model, and what the A/B measured

Phase 4 answered a modelling question: a model trained on the collected state
predicts a request's response time 53.9% better (MAE) than the server's own
reported EWMA. That says the signal is there. It does not say that a router
built on it beats a heuristic, because a router has to decide *before* the
request runs, with a state snapshot that is already a little old, and it has to
decide fast enough not to become the bottleneck itself.

This document records the serving path, the measurement that tests it, the
first result - which was a loss - and what fixing it changed.

## How a decision is made

For every request, in `balancer/ml_router.py`:

1. The request's type and declared payload size are read from the request
   (before the body is proxied), and one feature row is built per healthy
   server with the shared transform in `ml/features.py`.
2. One batched `predict` returns each candidate's predicted response time on
   the log1p scale; `to_ms` maps it back to milliseconds.
3. Each candidate is charged its outstanding work:

   ```
   expected wait = predicted service time x (1 + in-flight requests)
   ```

   The prediction comes from a metrics snapshot the collector refreshes every
   two seconds; the in-flight count is live, read from this router's own
   counter. Step 3 is what keeps the router from sending every request in a
   snapshot window to the same backend. (Charging it in units of the
   configured weight instead was tried and measured worse - see "A variant
   that lost".)
4. The lowest expected wait wins; servers within 5 ms of it are treated as
   tied and the least loaded of those is chosen. The choice, the raw
   prediction, and the in-flight count are written to the request log.

Serving does not use pandas: `build_serving_array` produces one numpy matrix
and the estimator is called through XGBoost's `inplace_predict`, which skips
the DMatrix construction that the scikit-learn wrapper does on every call.

## The measurement

`bench/ab_compare.py` runs the policies back to back on the same stack, same
traffic profile, same user count, with the offered load and the policies
interleaved per repeat rather than run in two blocks. It reads latency from the
balancer's own Parquet log - not from the load generator - over successful
requests only, and counts errors over every row, so a policy cannot look fast
by failing quickly.

```
python bench/ab_compare.py --profiles mixed_cost --policies round_robin ml --repeats 3
python bench/ab_compare.py --profiles mixed_cost --policies least_conn ml --repeats 3
```

`round_robin` is the plan's baseline; `least_conn` is there because it is the
strongest heuristic this project has, and beating the weakest one would not
prove much.

## First result: the router lost

The first paired run, with the model served exactly as trained and the router
picking the minimum prediction:

| policy | req/s | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| round_robin | 127 | 22 ms | 418 ms | 627 ms |
| ml | 68 | 172 ms | 527 ms | 800 ms |

The ML router was slower at the median by a factor of eight and moved half the
throughput. Both causes were in the serving path, and the request log said
where to look:

- **Every backend's median rose to about 180 ms**, including backend1, which
  was carrying less load than it carried under round_robin. A per-server
  problem would have left the lightly loaded servers fast. So the delay was in
  the balancer, not in the backends.
- **Within a one-second window, the busiest server's share was 0.81** under
  `ml` against 0.25 under round_robin. The router was herding: the model scores
  a snapshot, all requests in the window see the same scores, and they all go
  to the same backend until the next snapshot arrives.
- The decision itself cost **5.0 ms** inside the image, spent on building one
  DataFrame per candidate plus a concat plus a DMatrix, on the event loop.

## The fixes

1. **One numpy matrix instead of a frame per candidate.** 5.0 ms down to
   0.8 ms; the same predictions, asserted element for element by a test.
2. **Charge the queue.** In-flight work is the load the snapshot cannot see,
   and it is the quantity a queueing server actually responds to. This alone
   took the router from 68 req/s and a 172 ms median to 135 req/s and 56 ms,
   and put the busiest server's share back to a sane 23-38%.

`/stats` now reports `decision_ms_mean` and `decision_ms_max`, and the
dashboard shows the mean, so the router's own cost is observable during a demo
rather than inferred afterwards.

## Result after the fixes

Paired runs on `mixed_cost` at 24 users, three repeats each, with the policies
interleaved inside every repeat:

| policy | req/s | p50 | p95 | p99 | errors |
|---|---:|---:|---:|---:|---:|
| least_conn | 161 | 33 ms | 100 ms | 178 ms | 0.00% |
| ml | 135 | 56 ms | 133 ms | 212 ms | 0.00% |
| round_robin | 77 | 35 ms | 1205 ms | 2823 ms | 0.00% |

`ml` appears in two sessions because an A/B pairs two policies; the row above
is the one paired against least_conn. In the session paired against
round_robin, ml measured 133 req/s / 56 / 141 / 222 ms. Session-to-session
numbers move by tens of percent on this laptop - round_robin alone measured
77 and 127 req/s in two sessions - so the table is context, and the paired
ratios are the claim:

| pair | p95 ratio, median | ml faster in | p99 ratio, median |
|---|---:|---:|---:|
| ml vs round_robin | 0.11 (about 9x better) | 3 / 3 repeats | 0.12 |
| ml vs least_conn | 1.40 (40% worse) | 0 / 3 repeats | 1.25 |

Both directions held in every repeat, so neither is a drift artefact.

**Against the plan's baseline, the learned router wins clearly**: about 9x
better p95, 8x better p99, and 1.7x the throughput. round_robin's tail is set
by the 0.15-CPU backend - it gives that server a quarter of the traffic no
matter what, and a queue there takes seconds to drain. ml sends it 9-13%,
because the model's latency features price it correctly.

**Against least_conn, it loses**: 40% worse p95, 16% less throughput, and a
higher median. That is the honest result. The reasons are visible in the data:

- least_conn reacts to the in-flight count at the instant of the decision, and
  it divides by server weight. ml's strongest inputs (`rt_p95_5s_ms`,
  `rt_ewma_ms`, `req_rate_5s`) are sampled every two seconds, and its queue
  correction is an approximation of the quantity least_conn measures exactly.
- least_conn leans on the 2x backend (36% of traffic); ml spreads more evenly
  (33 / 27 / 28 / 12), so its median request lands on a 1x backend. That is
  why its p50 is worse than both heuristics while its tail is far better than
  round_robin's.
- The model costs 1.4 ms per decision; least_conn costs nothing.

None of that contradicts M5: the model predicts response time 53.9% better
than the naive baseline. It says that on a four-server cluster with two-second
snapshots, most of what a learned router can add is already captured by
"send the next request to the least loaded server", and the remaining edge -
knowing that a `/cpu` request is expensive and that the 0.15-CPU backend should
be avoided - is real but not enough to overcome fresher state.

### A variant that lost

Charging the queue in units of server weight,

```
expected wait = predicted service time x (1 + in-flight / weight)
```

looks better on paper and matched least_conn's routing distribution almost
exactly (36 / 28 / 25 / 11). It measured worse in every repeat: ml 100 req/s,
p95 280 ms against least_conn's 145 req/s and 135 ms in the same session. The
weight discounts the queue on the 2x backend, so the router sends it
least_conn's share of traffic but in bursts that the stale prediction cannot
price, and its queue spikes. It was reverted, and the numbers are kept here
because a rejected variant with a measurement is more useful than a missing
one.

## Under a fault

The `degraded` profile adds 400 ms to every request on backend3 and fails 35%
of them, for the whole run. Throughput necessarily falls; the question is
where the delay and the errors land.

| policy | req/s | p50 | p95 | p99 | errors | share to the faulted backend |
|---|---:|---:|---:|---:|---:|---:|
| ml | 160 | 54 ms | 162 ms | 279 ms | 0.26% | 0-1% |
| round_robin | 136 | 10 ms | 414 ms | 526 ms | 8.73% | 25% |

Two paired repeats, 30 users. round_robin cannot avoid the faulted backend - it
gives it a quarter of the traffic no matter what - so a quarter of its requests
pay 400 ms or a 503, and 8.73% of all requests fail. The router sends it 0-1%:
the fault is visible in the snapshot the model scores (`err_rate_5s` rises,
`rt_ewma_ms` and `rt_p95_5s_ms` jump), and the 400 ms sleep keeps that
backend's live queue longer, so both terms of the score point away from it.
Errors drop 34x, throughput rises 18%, p95 improves 2.6x.

The one metric round_robin wins is p50 (10 ms against 54 ms). Its successful
requests are mostly the three-quarters that never touched the faulted backend,
while the router's successes are served by backends 1 and 2 carrying the full
160 req/s. A median that excludes failures is not the comparison that matters
here; the error column is.

This is the scenario the router is for: a static distribution keeps feeding a
broken server forever, and a per-request prediction that reads the server's own
error and latency signal does not.

## What this does and does not show

- It is a four-server cluster on one laptop, with two physical cores shared by
  the balancer (2 CPU), four backends (1.75 CPU total) and the load generator
  (1 CPU). Absolute numbers will not transfer to other hardware; the paired
  comparisons are what the claim rests on.
- The model is scored against a two-second-old snapshot. Part of what the
  queue term adds is information the model would have had with fresher
  metrics; a production deployment would poll faster and could fold in-flight
  work into the model's inputs instead of correcting after the fact.
- Inference runs on the event loop, so `ml` has a CPU ceiling of roughly one
  decision per `decision_ms_mean`. It is affordable here (~1.4 ms) but it is
  the reason the algorithm is not free.

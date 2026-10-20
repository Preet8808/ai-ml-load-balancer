# Decision log and drift guard

Two Phase 5 deliverables that go together: a model that is served needs a
record of what it decided and evidence that it is still right. Code:
`balancer/decision_log.py`, `balancer/drift.py`,
`balancer/ml_router.py`; tests: `balancer/tests/test_drift.py`.

## The decision log

The request log is the 16-column dataset schema and must keep exactly that
shape, and it has no id column. A routing decision needs different fields, so
it gets its own Parquet file, joined to the request log on
`(run_id, timestamp)` - the run and the request's start time, which both files
record from the same value:

| column | meaning |
|---|---|
| `timestamp` | when the request started; the join key with the request log |
| `request_id` | the decision's own identifier (unique within the run) |
| `run_id` | the run, same id as the request file |
| `algorithm` | the policy that was live |
| `server_id` | the server that served |
| `reason` | the decision string, e.g. `ml(pred 12.3ms, inflight 1)` |
| `predicted_ms` | the model's estimate for *that* server, null when none |
| `actual_ms` | what the client waited, including a retry if one happened |
| `status` | HTTP status the client received |
| `retries` | 0 or 1 |
| `model_driven` | true only when a model made the choice |
| `inflight` | the server's queue at decision time |
| `feed_age_ms` | age of the metrics snapshot the decision saw |
| `req_type` | the workload class |

Files land in `decisions/` (gitignored) and follow the same run lifecycle as
the request log: `POST /admin/run?run_id=...` seals the previous file and
opens a new one, so a file holds one run and can be joined to that run's
request file. Rows outside a run are dropped rather than mislabelled, for the
same reason as the request log.

Reading one back and joining it:

```python
import numpy as np, pandas as pd
decisions = pd.read_parquet("decisions/decisions_<run>_<ts>.parquet")
requests = pd.read_parquet("logs/<run>_<ts>.parquet")
merged = decisions.merge(requests, on=["run_id", "timestamp"], suffixes=("_dec", "_req"))

scored = merged[(merged.status_dec < 400) & (merged.retries == 0)
                & merged.predicted_ms.notna()]
scored["error_log"] = abs(np.log1p(scored.actual_ms) - np.log1p(scored.predicted_ms))
```

Checked on a 2,601-request mixed_cost run: the inner join returned 2,601 rows,
and `actual_ms`, `status` and `server_id` agreed exactly with the request log
on every one. The join is exact because both rows are written from the same
`request_started` float, not because timestamps were rounded to match.

`/stats.decisions.recent` carries the last 15 rows (rounded) so the dashboard
can show predicted-versus-actual without reading a file, and
`/admin/logger` reports both loggers' state.

### Serving cost

The log is buffered (50 rows or 2 s) and flushed in a worker thread, but that
thread holds the GIL while it builds the Arrow table, so the conversion cost is
visible on the request path. The first version built a pandas DataFrame first
and measured 8.4 ms per 50-row batch; `pa.Table.from_pylist` does the same job
in 2.1 ms (200 rows: 10.4 -> 3.4 ms), with identical files. That change is in
`balancer/logger.py` and applies to both logs.

After it, a mixed_cost A/B run with `ml` measured 130 req/s, p50 61 ms,
p95 136 ms, p99 222 ms - the earlier measurement of the same profile in the
previous session was 134 / 59 / 145 / 229, so the decision log does not move
the serving numbers. Mean decision cost was 1.57 ms over 2,601 decisions.

The decision-cost *maximum* is a different story: it was 20.4 ms in that run,
and 19.5 ms before the `from_pylist` change. It is a single stall, not a
distribution - the mean is unchanged, and nothing in the logger explains why
exactly one decision in 2,601 took 20 ms. The Phase 7 matrix later recorded one
126.72 ms decision across the ML cells, with a mean of 1.96 ms; that is the tail
quoted in `docs/results.md`. Both are most plausibly GC or scheduler pauses on
this host; the honest statement is that the mean is the measured routing
overhead and the max is an unexplained one-off.

## The drift guard

The plan asks for a rolling-MAE monitor that demotes a model whose live error
grows. `DriftGuard` keeps the last `window` (200) absolute errors in **log
space** - `|log1p(actual) - log1p(predicted)|`, the same metric the model was
selected on - and:

- **demotes** when at least `min_samples` (50) observations are in the window
  and the mean exceeds `threshold` (default 1.0);
- **promotes** when a *full* window of fresh observations stays below
  `threshold * hysteresis` (0.8).

The threshold default is a line, not a tuning: `docs/dataset.md` measures
XGBoost at 0.5868 MAE(log) and the naive baseline at 1.2738, so 1.0 means "the
model has become no better than predicting the mean".

Three details make it work rather than just exist:

1. **The error is measured on the server that served**, whether or not the
   model chose it. If demotion were the end of the story the guard would have
   no fresh evidence and could only be re-enabled by hand. The router therefore
   keeps predicting while demoted (a *shadow* prediction) and reports it with
   the least-connections choice, so the window keeps filling.
2. **The window is cleared on demotion**, so re-promotion needs a full window
   of new evidence and cannot ride on the tail of the bad patch.
3. **Retried and failed requests are not scored.** The proxy calls
   `observe_outcome` only for a first attempt, and the router drops non-2xx
   outcomes: a retry's latency contains the failed attempt, and an error page
   says nothing about the model's accuracy.

While demoted, every decision's `reason` starts with `drift-fallback`, its
`model_driven` is false, and `ml.drift_routed` counts them. The model's state
is in `/stats.ml.drift`:

```json
"drift": {"demoted": false, "live_mae_log": 0.4603, "threshold": 1.0,
          "window": 200, "min_samples": 50, "samples_in_window": 120,
          "demotions": 0, "promotions": 0, "last_event": null}
```

### Watching it fire

Injecting a broken model would be the real test, but the mechanism can also be
demonstrated on healthy traffic by lowering the threshold with
`POST /admin/drift?threshold=0.15` (and raising it again). `bench/check_drift.py`
does exactly that against the live stack:

```
algorithm: ml
live MAE(log): 0.4603 over 120 samples, threshold 1.0, demoted=False
last decision: backend2 predicted=94.1 ms actual=57.7 ms status=200 model=True
  reason: ml(pred 94.1ms, inflight 1)

threshold lowered to 0.15: demoted=True (demotions 1), drift_routed=112
last decision: model_driven=False reason=drift-fallback(least_conn, pred 279.9ms)

after restoring the threshold: demoted=False live MAE=0.7593 promotions=1
last decision: model_driven=True reason=ml(pred 121.4ms, inflight 2)

sealed: logs/drift_check_...parquet 640 rows
decision file: decisions_drift_check_...parquet 640 rows
columns: timestamp, request_id, run_id, algorithm, server_id, reason,
         predicted_ms, actual_ms, status, retries, model_driven, inflight,
         feed_age_ms, req_type
MAE(log) recomputed from the file: 0.8200 over 640 scored rows
model-driven rows: 135 of 640
```

The file's MAE (0.82) is higher than the pre-demotion live value (0.46)
because a deliberately wrong threshold forced most of the run onto
least-connections, where the shadow predictions are for servers the model did
not choose. That is the honest number for that run, not a regression to
explain away.

## Dashboard

The ML panel shows the guard: live MAE next to the threshold, window fill,
demotions and promotions, and a red `DEMOTED` marker while it is in force. A
"Recent decisions" table below the servers shows time, server, predicted ms,
actual ms, the log-space error, status, whether a model made the call, and the
reason string for the last 15 decisions. Together they close the
predicted-vs-actual and decision-log gaps against the plan.

A latency panel plots p95 over time. The server computes a 60 s window of
successful requests on every `/stats` poll (`latency` in the payload) and reports
nearest-rank p50/p95/p99; the page keeps the last 120 polls, so the chart is two
minutes of five-second windows rather than a client-side re-aggregation. Failed
requests are excluded because a 503 returns fast and would drag the p95 down;
error rate is reported separately, as in the A/B harness. The window is a
`deque` of at most 20,000 `(timestamp, ms)` pairs - enough for 60 s at this
host's peak rate - and the percentile is computed once per poll, not per
request.

## Limits

- **Live MAE is not the training MAE.** The training number is measured on the
  collected campaign; serving traffic here is lighter (`/ping` dominates), so
  live MAE can be *better* than the training report (0.4603 in the demo above).
  It can also be much worse: the Phase 7 matrix measured 1.394 in steady and
  2.421 in failure against 0.5868 offline. The guard is a relative-change
  detector, not a re-validation.
- **The window is a request count, not time.** 200 observations is a few
  seconds at demo rates; a slower deployment would need a larger window and a
  longer minimum.
- **One prediction per request.** The model scores every candidate but only
  the chosen server's prediction gets an outcome, so the guard measures the
  error on the traffic actually sent, not counterfactual quality.
- **Promotion is automatic by design.** If the model is systematically wrong
  for traffic the guard rarely sees, the guard cannot know. A per-workload
  guard would be the next step.
- **The log is per-run.** Decisions outside a run are in `/stats` and the ring
  buffer only; start a run before a demo you want to keep.

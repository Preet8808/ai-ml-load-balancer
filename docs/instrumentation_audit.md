# Instrumentation Audit — Phase 3 Dataset Defects

The first dataset produced by the Phase 3 pipeline was unusable. Every defect
below was found by inspecting the data itself rather than by trusting the
pipeline to report its own success. They are documented because each one
produced *plausible-looking* output: files existed, columns existed, and row
counts were non-zero.

## How it was caught

The collection script copied 9 Parquet snapshots and all 9 were unreadable.
Dumping the first and last four bytes showed a `PAR1` header followed by
garbage: the Parquet footer was never written because the writer was still
open. After a graceful container stop (which triggers the shutdown hook and
closes the writer), 8,481 rows became readable — enough to audit.

## Signal test

Same 5-fold cross-validation, target `log1p(response_ms)`:

| Feature set | MAE | R² |
|---|---|---|
| Predict training mean | 0.486 | 0.000 |
| Endpoint type only | 0.276 | 0.745 |
| Endpoint × server lookup | 0.245 | 0.779 |
| **Live server metrics** | **0.479** | **0.018** |

The live server-state features — the entire premise of an AI/ML load balancer
— explained **1.8%** of variance. All apparent predictive power came from
memorising that `/io` is slow, because `/io` slept for a hardcoded
`50 / cpu_limit` milliseconds. A model trained on this dataset would have
reported a large improvement over the naive baseline (which itself uses a
broken feature) while routing no better than round robin.

## Defects

### File integrity

1. **Parquet files were never sealed.** `RequestLogger` held a single
   `ParquetWriter` open for the whole container lifetime, so the footer was
   written only on shutdown. Any copy taken during a run was corrupt.
   *Fix:* the writer rotates per run and `/admin/seal` closes it explicitly.

2. **One cumulative file for all runs.** Every run appended to the same file,
   so per-run files could not be isolated.
   *Fix:* `/admin/run?run_id=...` seals the current file and opens a new one.

3. **`run_id` was constant.** Derived from process start time, so all rows
   shared one id, making run-grouped cross-validation impossible and leaking
   information between folds.
   *Fix:* `run_id` is supplied per run by the collector.

4. **Collection script copied open files**, and `make_dataset.sh` still
   targeted port 8080 after the balancer moved to 9090.
   *Fix:* one Python driver owns the pipeline; the shell script is a wrapper.

### Backend metrics (`backend/main.py`)

5. **`cpu_pct` and `mem_pct` were monotonic counters.** They incremented by a
   fixed amount per request and were capped at 100, so they saturated within
   seconds and then reported a constant 100 forever — measured p50 = p95 = max
   = 100.
   *Fix:* read real usage from cgroup v2 (`cpu.stat` deltas normalised by the
   `cpu.max` quota; `memory.current` over `memory.max`), with a psutil
   fallback outside containers.

6. **`req_rate_5s`, `error_rate_5s`, `load_avg`, `net_io` were
   `random.uniform()` noise**, regenerated on every `/metrics` call and
   uncorrelated with any real load.
   *Fix:* derived from observed request events in a 5-second sliding window,
   plus `/proc/loadavg` and `/proc/net/dev` byte deltas.

7. **`inflight` was never incremented.** It was only decremented on `/cpu`, so
   it stayed at zero — 97.4% of rows.
   *Fix:* HTTP middleware increments on entry and decrements on exit.

8. **Latency statistics did not exist.** There was no per-server latency
   tracking at all, only the endpoint's own fixed sleep.
   *Fix:* middleware records every request's latency and status; the backend
   reports EWMA, p95 over 5 s, and error rate from that window.

9. **`/cpu` work scaled *with* CPU limit** (`cpu_limit * 1000` iterations), so
   weaker backends did less work and answered faster — the opposite of the
   intended heterogeneity.
   *Fix:* constant work per request, so a 0.5-core backend is measurably
   slower than a 2-core backend for identical work.

10. **CPU work blocked the event loop**, delaying `/metrics` and making
    backends look stale to the collector.
    *Fix:* offloaded with `asyncio.to_thread`, so cgroup throttling produces
    queueing without stalling metrics.

11. **`payload_kb` was always zero** because traffic was GET-only.
    *Fix:* the `/payload` share of the traffic mix now POSTs a body of varied
    size, so the feature is real and known at routing time.

### Collector (`balancer/collector.py`)

12. **`rt_ewma_ms` was the latency of the `/metrics` poll itself**, not a
    response time — measured correlation with actual `response_ms` of
    **0.013**.
    *Fix:* the collector passes through the backend's own EWMA.

13. **`req_rate_5s` was a rolling average of that same poll latency**,
    explicitly annotated in the code as
    `"Not really rate, but placeholder"`.
    *Fix:* passes through the backend's measured request rate.

14. **`rt_p95_5s_ms` was the current poll latency**, not a percentile.
    *Fix:* passes through the backend's real p95.

15. **`error_rate_5s` was hardcoded to `0.0`.**
    *Fix:* computed from 5xx responses in the window.

16. **The health state machine could not recover.** New metrics were
    constructed with `healthy=True`, so the `if not server.metrics.healthy`
    branch was unreachable and a single successful poll restored a server
    regardless of the 3-success threshold.
    *Fix:* health transitions are evaluated against the *previous* snapshot
    before the new one is installed.

17. **Monitoring traffic polluted the application statistics**, so the
    backend measured the collector rather than the load.
    *Fix:* `/metrics` and `/health` are excluded from the tracked window, and
    responses carry `cache-control: no-store`.

## Guards added

`bench/signal_check.py` is a gate that runs before Phase 4 training and
rejects the dataset unless:

- no Parquet file is unreadable
- at least 5,000 rows across at least 4 distinct `run_id`s
- no state feature is constant or nearly all zeros
- **cross-validated R² from server-state features alone is ≥ 0.30**
- the label has real variation

The last condition is the one that matters. Without it the pipeline happily
reported success on a dataset where the features carried no information.

---

# Part 2 — Latency that was not the backend's fault

Once the gate rejected the first corrected dataset (R² = 0.149), the next
defect class turned up: latency the backend did not cause. Four separate
mechanisms were adding seconds to every request while the backends sat idle.

## The logger blocked the request path

`RequestLogger.log()` acquired an `asyncio.Lock` and held it across the Parquet
write. With ~200 concurrent proxy requests all queueing behind that lock and a
20 ms write to the host bind mount, `/ping` — which the backend answers in
microseconds — measured **2.6 s of pure logging latency**.

Measured cost of a 50-row flush:

| Target | mean | p95 | max |
|---|---|---|---|
| `/app/logs` (host bind mount) | 19.6 ms | 46.5 ms | 52.7 ms |
| `/tmp` (container local) | 2.1 ms | 5.9 ms | 10.2 ms |

*Fix:* `log()` now only appends to a list — atomic inside a single-threaded
event loop — and the Parquet write runs in a worker thread. The request path
takes no lock and performs no I/O.

## The proxy connection pool was too small

`httpx.AsyncClient` defaults to `max_connections=100` shared across all four
backends. Everything above that queued inside `client.request()`, and because
the proxy measures that wait, it attributed queueing to the backend.

*Fix:* limits are configurable (`PROXY_MAX_CONNECTIONS`, default 500) with
keepalive aligned to the pool.

## CPU work starved the event loop

`asyncio.to_thread` without a bound let many GIL-holding threads run at once.
The backend's event loop then could not answer `/metrics`, so the only
queue-depth signal a router could observe read a flat zero — which is exactly
the signal a load balancer needs most.

An intermediate attempt used a `ProcessPoolExecutor`, which removed the GIL
contention but pushed poll latency to 1.5 s under load.

*Fix:* CPU work is bounded by an `asyncio.Semaphore` whose width equals the
container's CPU quota, mirroring a fixed worker pool. At most `CPU_WORKERS`
threads ever hold the GIL, so the loop stays responsive while requests queue
where a router can see them.

## /cpu scaled its work with the CPU limit

`/cpu` did `cpu_limit * 1000` iterations, so a 0.5-core backend did *less*
work and answered *faster* — the exact opposite of the heterogeneity the
project needs. *Fix:* constant work; the quota decides how long it takes.

## Where the queue actually was

With 200 concurrent users, instrumenting both sides at once settled it:

```
balancer inflight_total : mean=181.3  (min=175, max=187)
backends inflight (sum) : mean=0.3
balancer per-server     : [44, 49, 43, 44]
gap (waiting in proxy)  : 181.0
```

One hundred and eighty-one requests were inside the balancer while the backends
held three tenths of one request between them. `/ping`, which does no work, had
the same latency as `/cpu`, which does 28 ms of work — the signature of a
uniform queue rather than endpoint-specific cost.

## Saturation curve

Aggregate latency against concurrent users (in-container Locust, round robin):

| Users | req/s | avg | median |
|---|---|---|---|
| 20 | 145 | 36 ms | 27 ms |
| 40 | 118 | 224 ms | 180 ms |
| 80 | 97 | 641 ms | 480 ms |
| 200 | 87 | 1853 ms | 1400 ms |

Throughput peaks near 20 users and then falls while latency climbs: the stack
is CPU-bound and past roughly 80 concurrent users the proxy degrades without
the backends getting busier. Dataset runs are therefore sized to 60–110 users
so that server state, rather than proxy collapse, explains the latency.

## Result

After these fixes, the same signal test that scored 0.018 on the original
dataset and 0.149 on the first corrected one now scores **R² ≈ 0.51–0.55 from
server-state features alone**, with endpoint identity contributing essentially
nothing (R² ≈ 0.000). The features describe load, not endpoint identity.
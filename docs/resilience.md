# Serving safeguards

The routing algorithms answer *where* a request should go. This layer answers a
different question: *is this server worth choosing right now, and what should
happen when something breaks?* It sits between the proxy and every algorithm, so
`round_robin`, `least_conn`, `ml` and the rest all get the same protection.

Code: `balancer/resilience.py` (policy), `balancer/main.py` (proxy loop),
`balancer/collector.py` (feed control). Tests:
`balancer/tests/test_resilience.py`.

## Why these three mechanisms

The project plan makes two promises that a plain load balancer does not keep:

- **Phase 5 (M6):** "stopping the metrics feed triggers a switch to Least
  Connections without errors". Routing on a frozen snapshot while claiming to
  be metrics-driven is not that; refusing to route is worse.
- **Phase 6 (M7):** "a killed backend is bypassed within 5 s, and a recovered
  backend is ramped in". The collector's health checks poll every second and
  need three failures, so they take about 6 s to react - too slow for the
  first criterion - and they have no notion of a ramp.

## What was built

| Mechanism | Trigger | Effect |
|---|---|---|
| Stale-feed fallback | newest snapshot older than `STALE_AFTER_S` (3 s) | route with least-connections over the last snapshot, only to servers the collector last saw healthy; count as `stale_routing` |
| Circuit breaker | `BREAKER_THRESHOLD` (3) consecutive proxy failures | skip the server for `BREAKER_COOLDOWN_S` (10 s), then allow one probe; a failed probe re-opens immediately |
| Slow start | a tripped server succeeds again, or the collector marks it healthy again | admit it with a share ramping from 10% to 100% over `SLOW_START_S` (10 s) |
| Single retry | the chosen server returns 500/502/503/504, or the connection fails | retry once on a different server, excluding the one that failed |

A failure for the breaker is a connection error, timeout, or any 5xx response.
4xx responses are the client's answer and never count against a server.

All four are observable through `/stats`:

```json
"feed":       {"age_ms": 182.0, "stale": false, "collector": {"running": true, "enabled": true}},
"resilience": {"retries": 6, "stale_routing": 0, "circuits_open": [], "circuit_opens": 2}
```

and per server: `circuit_open`, `circuit_tripped`, `consecutive_failures`,
`circuit_opens`, `slow_start`. The dashboard shows feed age, retries, stale
routing, and a per-server state badge (`healthy` / `slow start` / `circuit open`
/ `down`).

## Demonstrating the stale-feed switch

`POST /admin/collector?enabled=false` pauses polling without stopping the
balancer. `bench/check_stale_feed.py` runs the whole sequence and prints:

```
algorithm: {'status': 'ok', 'algorithm': 'ml'}
baseline requests: [200]
feed age 182 ms, stale=False, stale_routing=0
collector paused: feed age 3109 ms, stale=True
during stale feed: 30 requests, codes [200]
  stale_routing delta: 30
  ml decisions delta : 0
  retries delta      : 0
collector resumed: feed age 948 ms, stale=False
post-resume requests: [200]
```

The `ml decisions delta: 0` is the important number: during the outage the
model was not consulted at all, and every request was served by the
least-connections fallback. When polling resumes, the model takes over again.

If the backends are actually dead - not just unmonitored - the snapshots are
stale *and* the collector has marked them unhealthy, so the fallback still
refuses to route and the client gets a 503. A dead cluster is not made to look
alive by a monitoring outage.

## Chaos: killing a backend under load

`bench/check_chaos.py` drives traffic, runs `docker compose stop backend3`,
keeps driving, restarts it, and reports:

```
circuit opened 1089 ms after the kill (2 open(s), healthy=True)
6 s after kill: healthy=False, circuit_open=True, sent_since_snapshot=7
  traffic elsewhere: {'backend1': 0, 'backend2': 0, 'backend3': 7, 'backend4': 0}
client requests: 2934  errors: 0 (0.00%)  codes=[200]
recovered 5.0 s after restart
slow start observed: True
backend3 now: healthy=True, circuit_open=False, sent_since_snapshot=9
resilience: {"retries": 6, "stale_routing": 30, "circuits_open": [], "circuit_opens": 2}
```

Reading it:

- **Bypass: 1.09 s.** The breaker opened after three consecutive failed
  requests, three seconds before the collector's health check would have
  noticed. The plan asks for under 5 s.
- **Zero client errors** across 2,934 requests spanning the kill, the dead
  period, and the restart. The six `retries` are the requests that had already
  been sent to `backend3` when it died; they were re-sent to a live server.
- **Recovery: 5.0 s** - container start plus three successful health polls -
  and `slow_start` was observed on the way back, so the recovered backend did
  not take a full share immediately.
- `circuit_opens: 2` at that point includes one open from an earlier aborted
  attempt; the counter is cumulative since the balancer started.

## Stress: above capacity

`bench/check_stress.py`, 80 concurrent users for 60 s against a stack with
1.75 CPU of backend quota:

```
before: mem=141.9MiB / 512MiB restarts=0
after : mem=146.6MiB / 512MiB restarts=0
requests: 7353 in 60s (123 req/s)
codes: {200: 7353}
2xx latency: p50=625 ms p95=902 ms
/stats samples: 12, failed: 0
decision cost during load: mean=1.30 ms max=1.39 ms
max in-flight: 74
feed age max: 1101 ms
```

- No crash and no restart: `RestartCount` stayed 0.
- No leak: RSS moved 141.9 -> 146.6 MiB (+3.3%) over 7,353 requests and
  settled; the 512 MiB limit was never approached.
- Overload is expressed as **queueing**, not failure: every request succeeded,
  with p50 rising from about 30 ms to 625 ms. The balancer's own control plane
  stayed responsive (`/stats` answered all 12 samples) and kept deciding in
  1.3 ms while saturated.
- The feed stayed fresh (max age 1.1 s) - the collector was not starved by the
  request path.

## Burst A/B: a negative result

The plan's M6 criterion is that "in the burst scenario the ML balancer shows
lower p95 than Least Connections in at least one early trial". It does not:

```
profile  policy      runs  req/s  p50 ms  p95 ms  p99 ms  errors
burst    least_conn  3     211    26      77      139     0.00%
burst    ml          3     165    52      138     217     0.00%

burst    0       77 -> 136   1.76   133 -> 212   1.59   0.00% -> 0.00%
burst    1       74 -> 145   1.95   133 -> 235   1.76   0.00% -> 0.00%
burst    2       81 -> 134   1.66   149 -> 206   1.38   0.00% -> 0.00%
```

ML lost all three pairs (median p95 ratio 1.76) with 35% less throughput.
Routing distribution shows why: under bursts the model sent 40-42% of traffic
to `backend3` (a 1x server) while giving `backend1` (2x CPU) only 22%, whereas
least-connections split 34/28/25/13. The model's queue term is unweighted, so
one in-flight request counts the same on a 1x and a 2x server; the
capacity-weighted variant was tried and measured worse in the mixed profile, so
it was reverted (`docs/serving.md`). The honest summary is that the model is
useful when capacity is heterogeneous and latency is the goal, and it is not a
better burst absorber than least-connections on this stack.

## Trade-offs and limitations

- **Retrying is not free.** A request whose response was lost (timeout after
  the backend processed it) may be executed twice. All benchmark endpoints are
  stateless, but a production deployment should limit retries to idempotent
  methods or add request de-duplication.
- **5xx counts against the server.** A server that returns 500s because of a
  bug in a *specific* request will be penalized for every client that repeats
  it. This is the standard trade-off; the cooldown bounds the damage to 10 s.
- **The breaker and the health checker are independent.** The breaker reacts to
  proxied traffic, the collector to `/metrics` polling. A backend that fails
  `/metrics` but serves traffic stays routable (correct), and one that serves
  `/metrics` but fails requests is still caught by the breaker (also correct).
- **Slow start is stochastic.** It drops a share of requests from the candidate
  pool per decision; it is a ramp in expectation, not a token bucket.
- **`STALE_AFTER_S` is a demo-scale constant.** Three missed one-second polls
  is right for this stack; a slower collector would need a larger value.
- **Retries change the dataset.** The Parquet log records one row per client
  request with the final outcome and total latency. A request that failed once
  and succeeded on retry is logged as a 200, so the dataset measures what the
  client saw; the backend's own `err_rate_5s` still records the failure. The
  218k-row training set predates this layer, so retried requests are rare in it.

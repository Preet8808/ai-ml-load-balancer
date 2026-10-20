# AI/ML Load Balancer - Project Hypotheses

*Drafted during Phase 0 and frozen before the Phase 7 campaign. The verdicts
below are from the 125-cell matrix; the full evidence is in `docs/results.md`.*

## H1: ML-Predictive has lower p95 latency than the best traditional baseline

**Statement:** ML-Predictive has lower p95 latency than the best traditional baseline in at least
3 of 5 scenarios, with statistical significance (p < 0.05, Mann-Whitney U test).

**Rationale:** Traditional algorithms (Round Robin, Least Connections) react to existing load rather than
predicting future saturation. With bursty traffic and heterogeneous servers, this leads to
uneven utilization and tail latency spikes. The ML model anticipates saturation from current load,
trend features, and request type, allowing proactive load shifting.

**Evidence:** Burst scenario time-series showing per-server CPU and latency; CDF comparison
across all 5 scenarios.

---

## H2: ML performs near-optimally in uniform conditions

**Statement:** In the easy case (identical servers, steady traffic), ML is not more than 5% worse
than Round Robin on p95 latency.

**Rationale:** When all servers are equivalent and traffic is steady, Round Robin is near-optimal.
The ML balancer should recognize server equivalence and behave similarly, with minimal overhead.

**Evidence:** Steady traffic scenario comparison; overhead measurement confirming < 5ms added latency.

---

## H3: Routing overhead is under 5 ms per request

**Statement:** The ML decision engine adds under 5 ms per request on average.

**Rationale:** The model must make predictions in real-time without significantly degrading
throughput. Gradient-boosted trees provide sub-millisecond inference for small numbers of servers.

**Evidence:** Per-request latency histogram; CPU usage comparison between ML and baseline algorithms.

---

## H4: Failover is transparent and fast

**Statement:** After a server failure, traffic is redirected away from the failed server within 5 seconds,
with no manual intervention and zero dropped requests.

**Rationale:** Health checks and circuit breaker mechanisms must detect failures quickly and
route around them. The fallback to Least Connections ensures no single point of failure.

**Evidence:** Failure timeline from chaos testing; recovery time measurement across 5 repetitions.

---

## Hypotheses Status

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1 | **Not met** | 0/5 significant wins (needs 3); burst and degraded means favour ML but Holm-adjusted p=1.0; steady, ramp and failure favour least-connections |
| H2 | **Met, with caveat** | steady ML p95 433 vs Round Robin 789 ms (0.55x); least-connections is 214 ms (2.02x better than ML), so this is a weak-baseline check, not a live win |
| H3 | **Met on the mean** | ML decision mean 1.96 ms; one 126.72 ms scheduler/Python tail outlier is reported |
| H4 | **Met** | worst redirect 1.13 s of 25 failure runs; final 5xx 0.00%; traffic returns 7.7 s after restart including slow start |

H2's original wording assumes identical servers; the test bed keeps the 4:2:2:1
CPU quotas, so the steady comparison against round robin is the harder
heterogeneous case. H4's "zero dropped requests" is the final 5xx rate: retries
can fail and recover inside the run, but no request ends in a 5xx.
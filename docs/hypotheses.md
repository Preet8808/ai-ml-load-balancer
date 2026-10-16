# AI/ML Load Balancer - Project Hypotheses

*Drafted during Phase 0 — will be finalized before Phase 5 (M6).*

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

| Hypothesis | Status | Finalized |
|------------|--------|-----------|
| H1 | ✅ Planned | To be frozen at M6 (Week 9) |
| H2 | ✅ Planned | To be frozen at M6 (Week 9) |
| H3 | ✅ Planned | To be frozen at M6 (Week 9) |
| H4 | ✅ Planned | To be frozen at M6 (Week 9) |
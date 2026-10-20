# Viva preparation - likely questions and honest answers

Numbers trace to `docs/results.md`, `docs/dataset.md`, `docs/ablations.md` and
`docs/monitoring.md`.

**Why XGBoost and not a neural network?**
The modelled dataset is tabular: 1.78 x 10^5 rows after warm-up and failure
exclusion, 121k in the training split, 14 features with strong interactions
(request type x server quota x queue). Gradient-boosted trees are the strong
baseline there, train in under a minute on a laptop, and predict in ~6 ms for a
four-row batch. A neural net would add tuning and serving cost for no expected
gain on this shape, and the plan's bar was MAE, not novelty.

**How do you know the test set is not leaked?**
The split is by `run_id`, never by row: adjacent rows inside a run share a server
state that evolves over seconds. The signal gate also forbids request features
until server state alone reaches R² 0.3, which it does (0.5499). The generalisation
test goes further and holds out an entire traffic profile.

**Why MAE in log space?**
Latency is heavy-tailed; one 4-second request would dominate a millisecond MAE.
`log1p` keeps the tail from owning the objective and makes relative error the
thing being optimised. The plan's M5 metric is defined in that space, and the
same metric is used for training, drift and the matrix.

**What does the in-flight term actually do?**
The features come from a snapshot up to 2 s old. Two servers that looked idle
then may differ by 40 requests now. The score `pred * (1 + inflight)` charges
each candidate its outstanding work, so the model cannot herd onto a server whose
queue it cannot see. Without it (`ml_predictive`), the rounding of near-ties
concentrates traffic. Measured on burst: p95 700 ms without the term vs 342 ms
with it (median ratio 2.21x), 73 vs 102 req/s, herd faster in 3/3 paired runs;
without it 46-69% of traffic goes to backend1 and 1-3% to backend4. Details in
`docs/results.md`.

**Why does ML lose on burst?**
In the pre-registered matrix it does not lose: mean p95 is 243 vs 275 ms for
least-connections, but that -11.7% difference is not significant (Welch p=0.751,
Holm 1.0). In the earlier dedicated burst A/B the direction was the opposite
(138 vs 77 ms). So the honest answer is that there is no consistent burst win,
not that the matrix proves a loss. The mechanism is plausible: least connections
reacts to the queue that exists *now*, while the model prices a snapshot up to
two seconds old, and a burst changes the queue faster than that.

**Why not just use least connections, then?**
On this campaign, least-connections is the strongest overall: lowest p95 in
three of five scenarios and highest throughput in all five. ML's case is
conditional: it is lower in degraded (182 vs 269 ms for the best baseline) and
burst (243 vs 275 ms), but neither is significant, and it loses steady, ramp and
failure. The defensible answer is that least-connections is a strong default and
the live evidence does not justify replacing it with ML. The ML contribution is
the offline prediction result, the degraded mean, and the queue-term mechanism,
not a universal win.

**Why does the failure scenario get worse if H4 passes?**
H4 is about redirecting away from the dead server within 5 s and avoiding final
5xx; that passes (worst 1.13 s, 0.00% errors). The failure cost is queueing after
the redirect: ML p95 734 vs 209 ms for least-connections, SLA 16.2% vs 0.5%. Two
runs collapsed at 41-43 req/s and sent under 7% to backend4. The breaker keeps
errors at zero; it cannot stop the remaining three servers from queueing.

**Why is H1 not met, and what does that mean?**
H1 needs three significant wins; ML has zero. The mean is favourable in burst
and degraded, but the bootstrap intervals are wide at five runs and the
Holm-adjusted p-values are 1.0; steady (+102%), ramp (+62%) and failure (+252%)
go the other way. The pre-registered live claim is therefore not supported. M5,
the offline 53.9% MAE improvement, still stands, but it is a different claim.
The project reports the negative rather than relabelling or dropping runs.

**How is the model updated without downtime?**
Artifacts are joblib files on a read-only mount, and the router takes the newest
one when it is instantiated. Re-selecting the algorithm (`POST
/admin/algorithm?name=ml`) builds a fresh router against the current artifact,
so a retrained model is picked up without restarting the container; there is no
file watcher, and a dedicated reload endpoint is future work. A model whose
feature layout does not match the serving array is rejected and the router falls
back to least connections rather than producing garbage.

**What happens if the metrics feed dies?**
Above 3 s of staleness the router serves from the last snapshot's healthy
servers; if there is no snapshot at all it falls back to least connections.
Measured: a 3,109 ms feed stop returned 30/30 200s. `docs/resilience.md`.

**What happens when the model is wrong?**
The drift guard scores predictions against outcomes in log space. Above the
threshold it demotes to least connections, keeps shadow-scoring, and promotes
back after a clean window. It is demonstrated live in `docs/monitoring.md`. In
the matrix the live MAE is 1.394 (steady), 1.151 (ramp), 0.898 (burst), 0.519
(degraded) and 2.421 (failure) against 0.5868 offline. The bad runs are around
3.0 and coincide with the capacity-avoidance collapse; at the shipped 1.0
threshold the guard would have been active, but the campaign pinned it at 10.0.

**Isn't pinning the drift threshold in the campaign cheating?**
No, and the distinction matters. The campaign compares *routing policies*. The
guard is a safety layer that can substitute least connections; at its shipped
threshold it did exactly that on the first ramp cell (live MAE 1.64), which would
have produced a cell labelled `ml` whose p95 was least connections'. Pinning it
isolates the policy; the live MAE is recorded for every cell, including the bad
windows (1.394 steady, 2.421 failure), and the demotion is reported as a finding,
including the recommendation not to trust the model on low-load ramp or failure
transients. The matrix does not claim the pinned policy is production-ready.

**Why 60-second runs when the plan says 5 minutes?**
125 x 6 min is 12.5 hours of continuous load on one laptop; the plan itself
schedules 2-3 overnight sessions. The design (5x5x5, randomised, 5 reps) is
unchanged, and 60 s still yields 6,000-9,000 requests per cell. The deviation is
in the manifest and the report; the consequence - fewer burst cycles per run - is
listed as a threat to validity.

**Five repetitions is low for a t-test; is H1 meaningful?**
The independent unit is the run, and 5 per cell is what the plan specifies. With
clean separation, the nonparametric test's minimum two-sided p at n=5 per group
is 0.0079, so a consistent win is detectable; a noisy one should not be claimed.
Both Welch and Mann-Whitney are reported, with Holm correction across the 20
comparisons, and the CIs are shown so a reader can see the uncertainty rather
than just a p-value.

**How do you know the load generator is not the bottleneck?**
The generator runs in its own cgroup at 12-13% CPU, sampled during every cell;
the M8 criterion was under ~70%. Host-side ad-hoc scripts top out at 64-70 req/s
and are never quoted as balancer throughput.

**Why is the balancer at ~130% CPU?**
That is percent of one core, so about 1.3 cores of its 2-core quota, at 100-150
req/s with retries and two Parquet logs. The ML routing decision is 1.96 ms mean,
with a 126.72 ms tail outlier, so H3 passes on the mean; the rest is proxying,
logging and Python overhead. It is the cost of the observability the project
needs, not just the model.

**Is this novel?**
No new algorithm. The contribution is an honest integration and measurement: a
learned router behind the same interface as five heuristics, a queue-aware score,
a decision log that makes "was it really the model?" answerable, a drift guard
with shadow scoring, and a benchmark that reports where the model loses. The
plan asks for exactly that discipline.

**What would you do with another month?**
Four things, in order: retrain with fault transients and add an explicit capacity
term so the router does not avoid backend4 when its predicted latency is high;
test the guard at the shipped 1.0 threshold and compare the demoted policy; a
measured Nginx/HAProxy baseline through the same client harness; and finish M9
with a screen recording and timed fresh-machine setup. A shorter prediction
horizon or queue-wait target is a plausible model-side experiment, but the
capacity failure is the more specific finding from this campaign.

**Why keep `ml` and `ml_herd` as separate names?**
`ml` is the stable name used in configs, docs and earlier results; `ml_herd` is
the plan's name for the same class, added so the Phase 7 matrix can label the two
variants explicitly. They are the same code path, not two implementations.

**Why exclude failed requests from training?**
A 503's latency measures the failure path, not the server's service time, and
including it teaches the model to predict fast responses from broken servers.
Failures still reach the model as `err_rate_5s`, so health is not hidden.

**What is Jain's index telling you?**
How evenly requests are spread. It is reported with its limitation: equal request
counts are not equal *work* when quotas differ, which is precisely why the
project argues for cost-aware routing. A high Jain on this test bed is not
automatically good.

**The plan asks for an SLA guard - where is it?**
With an argmin policy the planned filter is provably redundant: if any candidate
is predicted under the SLA, the minimum is under the SLA, so excluding the over-
SLA candidates cannot change the choice. It only matters with a softmax or
two-choices picker, which this router does not use because it destabilises
routing against a stale snapshot. The SLA is configurable and reported, and the
observed over-500 ms share is measured per run; the reasoning is in
`docs/results.md` rather than a filter that can never fire.

**Why is there no top-1 ranking accuracy in the model table?**
Because it is not identifiable from observational data. The log records the
latency of the server that served each request; it never observes the same
request on the other three, so there is no observed "fastest server" per
request. Spearman is the honest substitute available; computing per-decision
ranking accuracy would need active probing or replaying every request against
every server, which changes the system. `docs/dataset.md` states the limitation.

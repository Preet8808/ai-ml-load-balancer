# Five-minute live demo

Everything below has been run on this laptop. If live traffic misbehaves, the
same behaviour is recorded in `docs/results.md` and `docs/monitoring.md`, so the
demo can fall back to the evidence without pretending the live run happened.

## Before the demo (2 minutes)

```bash
docker compose up -d
curl.exe -s http://localhost:9090/stats     # healthy: 4, feed age well under 3 s
```

Open <http://localhost:9090/> in a browser. Keep `docs/results.md` open in a
second tab for the last minute.

Start a run so the request and decision Parquet files are actually written
(without an active run the balancer still routes, but the logger drops the rows
by design, because they would not belong to a run):

```powershell
curl.exe -X POST "http://localhost:9090/admin/run?run_id=demo"
```

Three terminals:

- **A** - traffic: `python bench/demo_traffic.py --seconds 120 --rate 24`
- **B** - admin calls (`curl.exe`, not PowerShell's `curl` alias)
- **C** - optional: `docker compose logs -f balancer`

## 0:00-0:30 - What this is

One sentence: four backends on one host with different CPU quotas, and a router
that predicts the response time of *this request* on *this server* instead of
counting connections. Point at the dashboard: per-server CPU relative to quota,
in-flight, feed age, the latency panel, the decision table.

## 0:30-1:15 - Algorithms behind one interface

Start the traffic in A, then switch policies in B while it runs:

```powershell
curl.exe -X POST "http://localhost:9090/admin/algorithm?name=round_robin"
curl.exe -X POST "http://localhost:9090/admin/algorithm?name=least_conn"
curl.exe -X POST "http://localhost:9090/admin/algorithm?name=ml"
```

Watch the **reason** column in the decision table change from `round_robin` to
`least_conn` to `ml(pred 42.1ms, inflight 3)`. Same traffic, same backends, only
the decision function changed.

## 1:15-2:00 - The decision log

Point at the last 15 decisions: server, predicted vs actual, status, reason. The
prediction is what the model expected; the actual is what the client waited.
Note that they are written to a separate 14-column Parquet file, joined to the
request log on `(run_id, timestamp)`, so the 16-column training schema stays
exactly as the plan specifies.

## 2:00-3:00 - A degraded server

The backends' control endpoints are only reachable inside the compose network:

```powershell
docker compose exec balancer python -c "import httpx; print(httpx.get('http://backend3:8000/slow', params={'ms': 400, 'duration_s': 45}).text); print(httpx.get('http://backend3:8000/fault', params={'error_rate': 0.35, 'duration_s': 45}).text)"
```

Backend3 is now 400 ms slower and returns 503 on 35% of requests. Expected, over
the next few seconds:

- the circuit opens after three failures (tile on backend3),
- the share of new work moves to the other three servers,
- **no user-visible errors**: the retry sends a failed request to another
  server and it is logged as one row.

Point at the error tile staying at 0.00% while backend3's own error rate is 35%.

## 3:00-3:50 - The drift guard

Lower the threshold so the guard demotes a *healthy* model on real traffic:

```powershell
curl.exe -X POST "http://localhost:9090/admin/drift?threshold=0.15"
```

Within a few hundred decisions (about 10 s at this rate) the ML row shows
`demoted`, `drift_routed` starts climbing, and every reason becomes
`drift-fallback`. Restore it:

```powershell
curl.exe -X POST "http://localhost:9090/admin/drift?threshold=1.0"
```

After a clean window it promotes itself back (`promotions 1`). The point: the
model can be taken out of the path without a restart, and shadow predictions
keep scoring it while it is out.

## 3:50-4:30 - Results

Stop the traffic (Ctrl-C in A). Show `docs/results.md`:

- **M5**: XGBoost +53.9% MAE over the naive baseline (target 30%), R² 0.683,
  grouped CV 0.6455.
- **Matrix**: 125 cells; ML has the lowest mean p95 in degraded and burst, but
  neither difference is significant; it loses steady, ramp and failure, and H1
  is not met. Least connections wins three of five.
- **Ablations**: request type is the feature group that matters (+18.5% MAE
  without it); unseen burst and wave profiles still beat naive by ~59%; the herd
  term cuts burst p95 from 700 to 342 ms (3/3 paired runs).

## 4:30-5:00 - The honest part

The model is not uniformly better. H1 is not met: ML loses steady, ramp and
failure, and the failure p95 is 734 ms vs 209 ms for least connections. Live MAE
is 1.4 in steady and 2.4 in failure versus 0.5868 offline, and the collapsed runs
send only 4-7% to the small backend, losing capacity. The drift guard was pinned
at 10 for the matrix; at its shipped 1.0 it would demote, which is a next
experiment, not a claimed win. H3 and H4 still pass. Close on the engineering
answer: the decision log, queue term and drift guard are what make a learned
router debuggable, and the negative is measured from the same logs as the wins.

## After the demo

```powershell
curl.exe -X POST "http://localhost:9090/admin/seal"
```

This closes the Parquet footer so `logs/demo_*.parquet` and the matching decision
file are readable. The run id is in every row, so the two logs join on
`(run_id, timestamp)`.

## If something goes wrong

| Symptom | Cause | What to say / do |
|---|---|---|
| dashboard tiles stale | metrics feed stopped | wait 3 s: the stale-feed fallback takes over; 30/30 200s was measured |
| 503s during the fault | retry could not find a healthy server | check `healthy` count; stop the fault by waiting it out |
| ML row says `ml-fallback` | no model loaded | `docker compose logs balancer`; artifacts are committed, restart fixes it |
| logger rows 0, `dropped` climbing | no active run | start one with `/admin/run?run_id=demo` before traffic |
| traffic tool slow | host client-bound, ~60-70 req/s max | it is a demo driver, not the benchmark; numbers come from Locust |

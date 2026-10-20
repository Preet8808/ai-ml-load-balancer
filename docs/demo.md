# Five-minute live demo

The fast path is **dashboard-only** after the stack is up: the **Demo controls**
card starts the traffic, opens and seals the run, injects a backend fault, and
changes the drift threshold. No `curl` calls are needed during the talk.

The commands in the fallback section still work, and the measured evidence in
`docs/results.md` and `docs/monitoring.md` is what backs the claims. If live
traffic misbehaves, fall back to that evidence without pretending the live run
happened. The browser traffic driver is a demo aid, not the benchmark client;
the reported numbers come from Locust and `bench/run_matrix.py`.

## Before the demo (2 minutes)

```bash
docker compose up -d
curl.exe -s http://localhost:9090/stats     # healthy: 4, feed age well under 3 s
```

Open <http://localhost:9090/> in a browser. Keep `docs/results.md` open in a
second tab for the last minute. In the **Demo controls** card, click **Reset to
ML** once. That sets the policy to `ml`, restores the drift threshold to `1.0`,
and clears any leftover backend3 fault. The existing run is sealed when you
start the demo traffic.

The browser tab must stay open and active while it drives traffic (about 24
requests/s). Keep the demo on the same machine as the stack; this is a demo
driver, not a load generator for capacity numbers.

### The controls

| Button | What it does |
|---|---|
| **Start traffic** | Opens a new run, then sends about 24 req/s from this tab until stopped. |
| **Stop + seal** | Stops the browser driver, waits for in-flight requests, and seals the Parquet files. |
| **Degrade 45 s** | backend3 gets 400 ms extra latency and a 35% injected error rate for 45 s. |
| **Fail 12 s** | backend3 returns 503 on every request for 12 s. The circuit breaker opens; retries keep client-visible errors at 0. This is an error fault, not a container kill. |
| **Clear fault** | Clears the slow and error injections immediately. |
| **Stress (0.15)** | Lowers the ML drift threshold so a healthy model demotes on real traffic. Only works while `ml` is selected. |
| **Restore (1.0)** | Restores the shipped drift threshold. |
| **Reset to ML** | `ml` + threshold `1.0` + clear backend3 fault. |

Optional terminal for logs:

```bash
docker compose logs -f balancer
```

## 0:00-0:30 - What this is

One sentence: four backends on one host with different CPU quotas, and a router
that predicts the response time of *this request* on *this server* instead of
counting connections. Point at the dashboard: per-server CPU relative to quota,
in-flight, feed age, the latency panel, the decision table.

## 0:30-1:15 - Algorithms behind one interface

Click **Start traffic** in the Demo controls card. The status changes to
`running · … req`, and the same traffic is now live. While it runs, click the
**round_robin**, **least_conn**, and **ml** pills in the Routing algorithm card
above. No terminal is needed.

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

Click **Degrade 45 s**. backend3 is now 400 ms slower and returns 503 on 35% of
requests. Expected, over the next few seconds:

- its error rate rises and the circuit opens after three failures (the backend3
  card changes to `circuit open`),
- the share of new work moves to the other three servers,
- **no user-visible errors**: the retry sends a failed request to another
  server and it is logged as one row.

Point at backend3's own error rate (35%) and then at the decision table: the
failed attempt is retried on another server, so the logged row is a 200. The
fleet **mean errors** tile is diluted (roughly 35%/4), not zero; do not claim it
is zero. For a stronger "unhealthy server" beat, click **Fail 12 s** while
traffic is running: every request to backend3 returns 503, the circuit opens,
and its request rate drops to near zero. Then click **Clear fault** to let it
recover.

The Fail button is an injected error fault, not a container kill. The health
probe stays green because the process is alive; the circuit is what takes it out
of rotation. Actual container-stop evidence is the H4 matrix, not this button.

## 3:00-3:50 - The drift guard

Click **Stress (0.15)**. The threshold drops, and within a few hundred decisions
(about 10 s at this rate) the ML row shows **Demoted to least-connections**,
`drift_routed` starts climbing, and every reason becomes `drift-fallback`.
Click **Restore (1.0)**.

After a clean window it promotes itself back (`promotions 1`). The point: the
model can be taken out of the path without a restart, and shadow predictions
keep scoring it while it is out. The Stress button only works while `ml` is
selected; if it is not, click the ml pill first.

## 3:50-4:30 - Results

Click **Stop + seal** in the Demo controls card. Show `docs/results.md`:

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

**Stop + seal** has already closed the Parquet footer, so `logs/demo_ui_*.parquet`
and the matching decision file are readable. The run id is in every row, so the
two logs join on `(run_id, timestamp)`.

## Command fallback (if the panel is unavailable)

The panel is the fast path, but the underlying admin calls are unchanged. Use a
separate terminal only if the browser controls are not usable:

```powershell
# Reset the shipped demo state
curl.exe -X POST "http://localhost:9090/admin/algorithm?name=ml"
curl.exe -X POST "http://localhost:9090/admin/drift?threshold=1.0"
docker compose exec balancer python -c "import httpx; print(httpx.get('http://backend3:8000/slow', params={'ms': 0, 'duration_s': 0}).text); print(httpx.get('http://backend3:8000/fault', params={'error_rate': 0, 'duration_s': 0}).text)"

# Start a run and host traffic
curl.exe -X POST "http://localhost:9090/admin/run?run_id=demo"
python bench/demo_traffic.py --seconds 330 --rate 24

# Degrade backend3 (same operation as the panel's Degrade button)
docker compose exec balancer python -c "import httpx; print(httpx.get('http://backend3:8000/slow', params={'ms': 400, 'duration_s': 45}).text); print(httpx.get('http://backend3:8000/fault', params={'error_rate': 0.35, 'duration_s': 45}).text)"

# Drift stress and restore
curl.exe -X POST "http://localhost:9090/admin/drift?threshold=0.15"
curl.exe -X POST "http://localhost:9090/admin/drift?threshold=1.0"

# Seal
curl.exe -X POST "http://localhost:9090/admin/seal"
```

## If something goes wrong

| Symptom | Cause | What to say / do |
|---|---|---|
| dashboard tiles stale | metrics feed stopped | wait 3 s: the stale-feed fallback takes over; 30/30 200s was measured |
| Demo controls card missing | stale page or old balancer process | refresh; if the admin actions 404, `docker compose restart balancer` |
| Start traffic says the run failed | balancer or logger unavailable | check the live dot; fall back to `/admin/run` in the command section |
| 503s during the fault | retry could not find a healthy server | check `healthy` count; click Clear fault or wait out the 12/45 s window |
| Fail says backend3 is still healthy | the health probe is alive; this is an error fault | point at `circuit open` and backend3's request rate, not the healthy tile |
| ML row says `ml-fallback` | no model loaded | `docker compose logs balancer`; artifacts are committed, restart fixes it |
| logger rows 0, `dropped` climbing | no active run | click Start traffic (it opens a run), or `/admin/run?run_id=demo` |
| traffic slow | browser driver on the same host, ~20-24 req/s | it is a demo driver, not the benchmark; numbers come from Locust |

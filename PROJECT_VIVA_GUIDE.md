# AI/ML Layer-7 Load Balancer - Complete Viva Guide

**Project:** AI/ML-Based Intelligent Load Balancer  
**Author:** Preet  
**Repository:** `C:\Users\DELL\projects\ai_ml_load_balancer`  
**Main stack:** Python 3.11, FastAPI, Uvicorn, httpx, Docker Compose, XGBoost, pandas/PyArrow, Locust  
**One-line description:** A software Layer-7 HTTP reverse proxy that sends each request to one of four heterogeneous FastAPI backends, using a traditional load-balancing algorithm or an XGBoost model that predicts each backend's response time.

> **How to use this guide**
> 1. Read Sections 0-3 first. They contain the answers to the questions a professor is most likely to ask.
> 2. Use Sections 4-13 when you need the detailed explanation, formulas, or evidence.
> 3. Section 14 is a ready-to-follow five-minute demo.
> 4. Section 18 lists places where older documentation disagrees with the current code. Quote the code-accurate values there.
>
> **Honest headline:** The offline prediction result is strong (53.9% MAE improvement over the naive baseline), but the live routing result is mixed. H1 is **not met**. Least-connections is the strongest policy overall in the 125-cell comparison; ML has lower mean p95 only in burst and degraded, and those differences are not statistically significant. Never present this project as "ML always beats traditional load balancing."

---

## 0. The 60-second answer (memorize this)

> "This is a Layer-7 HTTP load balancer. Four FastAPI backends run on one Docker bridge network with different CPU quotas: 0.8, 0.4, 0.4 and 0.15 cores. The balancer is itself a FastAPI/Uvicorn reverse proxy. It supports random, round-robin, weighted round-robin, least-connections, weighted least-connections, least-response-time and ML policies behind one `choose_server()` interface.
>
> For ML, an XGBoost model predicts `log1p(response_ms)` for every healthy backend using 14 features: nine server-state features, request payload size, and four request-type one-hot features. The router converts the prediction back to milliseconds, multiplies by `(1 + inflight)` to account for queueing, and sends the request to the lowest expected wait. A metrics collector polls each backend's `/metrics` endpoint about once per second. A circuit breaker, one retry, slow start and a stale-feed fallback protect every algorithm equally.
>
> The offline model beats the naive EWMA baseline by 53.9% MAE. On 125 live benchmark cells, however, least-connections is stronger overall; ML wins the mean only in burst and degraded, not significantly. H1 is not met, and the report says so honestly."

If the professor says **"Explain your project in two minutes"**, add:

- **Why L7:** the router needs to know the request path/type, HTTP status, latency and payload size. An L4 balancer only sees IP addresses and ports.
- **Why four heterogeneous backends:** the research question is whether request-aware prediction can beat connection-counting when servers have different capacities.
- **What is measured:** p50/p95/p99 latency, throughput, error rate, SLA violations, CPU imbalance and routing distribution, at the run level.
- **What was found:** least-connections is a very strong baseline; ML is conditionally useful; the queue term is essential to prevent herding; decision logging and drift detection make the model debuggable.

---

## 1. Ten facts to memorize

| # | Fact |
|---|---|
| 1 | Four backends, CPU quotas **0.8 / 0.4 / 0.4 / 0.15** cores. Config design weights are **2.0 / 1.0 / 1.0 / 0.5** (a 4:2:2:1 ratio). |
| 2 | Host port **9090** maps to balancer container port **8080**; backends listen on internal port **8000** and are not published on the host. |
| 3 | Collector polls each backend about every **1 s**; health needs **3 consecutive failures** to mark down and **3 successes** to recover; feed is stale after **3 s**. |
| 4 | Circuit breaker: **3 consecutive proxy failures** -> open for **10 s**; slow start: **10% to 100% over 10 s**. |
| 5 | Retry: **one** retry, on a different server, when attempt 0 produces **500/502/503/504** (a transport error is mapped to 502, so it also retries). |
| 6 | ML model: XGBoost, **14 features**, target `log1p(response_ms)`, score = `predicted_ms * (1 + inflight)`. |
| 7 | Offline result: MAE(log) **0.5868**, MAE(ms) **73.1**, R^2(log) **0.6827**, grouped CV R^2 **0.6455**, **53.9%** better MAE than naive. |
| 8 | Live comparison: **125 cells** (5 algorithms x 5 scenarios x 5 repeats), **0 failed**. Least-connections best overall. **H1 not met.** |
| 9 | H2 met only against round-robin; H3 met on mean decision cost **1.96 ms**; H4 met with worst redirect **1.13 s** and **0.00%** final 5xx. |
| 10 | Current automated suite: **129 tests** (older docs say 121). FastAPI reports four `on_event` deprecation warnings, but tests pass. |

---

## 2. Top 30 high-probability questions and short answers

| # | Question | Answer to give |
|---|---|---|
| 1 | What is your project? | A Layer-7 HTTP load balancer/reverse proxy over four heterogeneous FastAPI backends; it compares traditional policies with an XGBoost router that predicts per-server response time. |
| 2 | Why is load balancing needed? | To distribute work, avoid overloading one server, improve latency and availability, survive failures, and use heterogeneous capacity efficiently. |
| 3 | Why is this a Layer-7 balancer? | It parses HTTP requests/responses and routes using path, request type, payload, status, latency and server metrics. L4 would see only IP/port/TCP. |
| 4 | L4 vs L7? | L4 forwards TCP/UDP flows without understanding HTTP; fast but blind to URLs/headers. L7 terminates HTTP, can route by path/header, retry on status, log requests and inspect application metrics. |
| 5 | Reverse proxy vs forward proxy? | A forward proxy acts for clients toward arbitrary servers; a reverse proxy sits in front of known servers, hides them, and represents them to clients. This is a reverse proxy plus a selection policy. |
| 6 | Reverse proxy vs NAT? | NAT rewrites addresses/ports at L3/L4 and usually keeps one connection semantically; an L7 proxy terminates the client connection and creates a separate upstream connection, so it can retry, modify headers, log and buffer. Docker port publishing is host port mapping (NAT on Linux; Docker Desktop forwarding on Windows/macOS); the balancer is the proxy. |
| 7 | How does a request travel? | Browser/curl -> host:9090 -> Docker port mapping -> balancer:8080 -> algorithm/ML chooses backend -> shared httpx pool -> backendN:8000 -> response -> back to client. |
| 8 | How does the balancer find backends? | Docker Compose creates a user-defined bridge network `balancer-net` with embedded DNS; `backend1`...`backend4` resolve to container IPs. Config uses hostnames, not hard-coded IPs. |
| 9 | How does the client find the balancer? | Host clients use `localhost:9090` via Docker port publishing. The benchmark Locust container runs on the Compose network and targets `balancer:8080` directly. In production, a DNS name/VIP or external load balancer would front it. |
| 10 | Which algorithms are implemented? | Random, round-robin, weighted round-robin, least-connections, weighted least-connections, least-response-time, and ML (`ml`, `ml_herd`, `ml_predictive` ablation). |
| 11 | How does least-connections work? | Choose the healthy backend with the smallest `inflight` count: `argmin_s inflight(s)`. |
| 12 | How does weighted round-robin work? | Repeat each server in the cycle in proportion to its weight; the project's intended design is 4:2:2:1, but the code uses `max(1, int(weight))`, so the 0.5 weight becomes one slot. See Section 6.4. |
| 13 | What does the ML model predict? | `log1p(response_ms)` for each candidate server for the current request; then the router maps back with `expm1` and clips to 0-5000 ms. |
| 14 | What are the features? | Nine server-state features: CPU %, memory %, inflight, request rate, EWMA latency, p95 latency, error rate, CPU limit, CPU delta. Plus payload KB and four request-type one-hots. |
| 15 | Why XGBoost? | The data is tabular with nonlinear interactions; gradient-boosted trees train quickly, predict fast, handle mixed scales, and beat Ridge and Random Forest on this dataset. A neural net adds cost for no expected gain. |
| 16 | How do you prevent data leakage? | Split by `run_id`, never by row; 12 runs train, 2 validation, 4 test. Also a signal gate requires server-state-only R^2 >= 0.30, and whole-profile holdouts test burst/wave. |
| 17 | Why log1p? | Latency is heavy-tailed; one 4-second request would dominate a millisecond MAE. `log1p` makes the target closer to relative error and keeps the tail from owning the objective. |
| 18 | What is the queue/herd term? | The metrics snapshot is up to ~1 s old, so the model cannot see requests sent since the poll. Score each server as `predicted_ms * (1 + inflight)`; without it the router herds onto one backend. |
| 19 | How does health checking work? | The collector GETs `/metrics` about once per second. A failed poll increments failure count; 3 consecutive failures mark the server unhealthy; 3 successful polls recover it. The feed is stale after 3 s. |
| 20 | What happens if a backend dies? | Requests fail, the breaker counts consecutive failures, opens after 3, and new work avoids the server for 10 s. One retry goes to a different server. The collector eventually marks it unhealthy; on recovery slow start ramps it back. |
| 21 | What happens if the metrics feed dies? | Above 3 s of staleness, the router stops consulting the algorithm/ML and chooses least-loaded among servers last seen healthy. If no backend is known healthy, it returns 503. |
| 22 | Circuit breaker vs health check? | Health check is an active periodic poll of `/metrics`; the circuit breaker is passive, based on actual proxied request failures. The breaker reacts in ~1 s; the health state needs 3 failed polls. |
| 23 | Is retry safe? | It is safe for this demo's stateless GET/POST, but generally one retry can duplicate a non-idempotent request if the response was lost. The guide/report explicitly states this trade-off. |
| 24 | 502 vs 503? | 502 Bad Gateway: the balancer could not get a valid response from the chosen upstream (connection error, timeout, protocol error). 503 Service Unavailable: no healthy/routable server is available. |
| 25 | Why p95, not average? | Averages hide tail latency. p95 says 95% of successful requests were at or below that value, which reflects user experience under bursts and queueing. |
| 26 | What did the experiment find? | Least-connections is strongest overall: lowest p95 in 3/5 scenarios and highest throughput in all 5. ML has lower mean p95 in burst and degraded but not significantly; it loses steady, ramp and failure. H1 is not met. |
| 27 | Why keep ML if it is not uniformly better? | The offline prediction win, the degraded/burst means and the queue-term mechanism are real; the project's contribution is the measured, honest comparison and the observability/reliability machinery, not a claim of universal superiority. |
| 28 | What are the two logs? | A 16-column request Parquet for the dataset (one row per client request, final outcome and total latency) and a 14-column decision Parquet with reason, prediction, retries, feed age and `model_driven`. Join on `(run_id, timestamp)`. |
| 29 | Is the dashboard flow a packet capture? | No. It is a sampled animation whose rate and split come from `/stats`; it is presentation only. New work is rerouted after a failure, never migrated from one backend to another. |
| 30 | Biggest limitations? | One laptop/host, 60-second matrix runs, five repetitions, one fault server, a single balancer (SPOF), no TLS/auth, and live model error higher than offline. See Section 13.5. |

---

## 3. Architecture

### 3.1 Diagram

```text
                         +----------------------------------+
                         | Client / browser / curl / Locust |
                         +----------------+-----------------+
                                          | HTTP
                                          v
                     host:9090  (Docker port mapping)
                                          |
                                          v
+-------------------------------------------------------------------+
| Balancer container: FastAPI + Uvicorn, internal port 8080         |
|                                                                   |
|  /                 dashboard (HTML)                                |
|  /stats            live JSON snapshot                              |
|  /admin/...        algorithm, collector, run, drift, demo controls |
|  /{path}           reverse proxy                                   |
|                                                                   |
|  Request path:                                                     |
|    classify request -> resilience.choose -> algorithm/ML ->        |
|    httpx.AsyncClient -> backend                                    |
|                                                                   |
|  Background: MetricsCollector (GET /metrics ~1 Hz)                 |
|  Logging: RequestLogger -> logs/*.parquet + decisions/*.parquet    |
+----------------+----------------+----------------+----------------+
                 |                |                |
       Docker bridge network `balancer-net` + embedded DNS
                 |                |                |
        +--------+------+  +------+-------+  +-----+--------+  +-----+--------+
        | backend1      |  | backend2     |  | backend3     |  | backend4     |
        | internal:8000 |  | internal:8000|  | internal:8000|  | internal:8000|
        | 0.8 CPU       |  | 0.4 CPU      |  | 0.4 CPU      |  | 0.15 CPU     |
        +---------------+  +--------------+  +--------------+  +--------------+
```

### 3.2 Components

| Component | Technology | Responsibility |
|---|---|---|
| Client | Browser, curl, Locust | Sends HTTP requests; benchmark client is Locust, not the dashboard. |
| Reverse proxy | FastAPI + Uvicorn | Receives HTTP, chooses backend, forwards request, returns response. |
| Upstream client | `httpx.AsyncClient` | Async HTTP/1.1, connection pool, timeouts, retries at proxy layer. |
| Metrics collector | `asyncio` background task + httpx | Polls each backend's `/metrics` about once per second and updates `ServerMetrics`. |
| Algorithms | `balancer/algorithms.py` | Stateless selection policies behind one interface. |
| ML router | `balancer/ml_router.py` + artifact | Batched prediction, queue correction, tie-break, fallback. |
| Resilience | `balancer/resilience.py` | Circuit breaker, slow start, stale-feed fallback, retry constant. |
| Drift guard | `balancer/drift.py` | Rolling log-MAE monitor; demotes/promotes model. |
| Request log | `balancer/logger.py` | 16-column Parquet, one row per client request. |
| Decision log | `balancer/decision_log.py` | 14-column Parquet with prediction/reason/outcome. |
| Dashboard | `balancer/dashboard.html` | Polls `/stats`, shows servers, latency, decisions, demo controls. |
| Backends | FastAPI + Uvicorn, internal :8000 | Expose application endpoints and `/metrics`; inject faults for demos. |
| Isolation | Docker + cgroup v2 | CPU/memory quotas, bridge network, DNS, port publishing. |
| Offline ML | pandas, scikit-learn, XGBoost, joblib | Feature engineering, training, evaluation, artifact. |
| Load generation | Locust `FastHttpUser` | Generates reproducible profiles for dataset and matrix. |
| Optional baselines | Nginx, HAProxy configs | Reference least-connections proxies in `deploy/`; not in default stack. |

### 3.3 What is actually deployed

- `docker compose up -d` starts **five** services: balancer + four backends.
- The directory `agent/` contains a psutil metrics agent, but the default deployed backend already exposes `/metrics`; the agent is not a separate Compose service in the current stack.
- The dashboard is served by the balancer itself at `/`; it is not a separate frontend server.
- The trained model is mounted read-only from `models/`; request/decision logs are mounted from `logs/` and `decisions/`.
- Nginx and HAProxy are optional reference configs, not part of the primary 125-cell measurement.

---

## 4. End-to-end request lifecycle

This is the sequence to draw on a whiteboard if asked **"walk me through one request."**

1. **Client connection.** The browser/curl connects to `localhost:9090`. Docker's port publishing forwards that to the balancer container's port `8080`. This is a TCP connection at L4; the HTTP request is L7.
2. **FastAPI routing.** Uvicorn accepts the HTTP request. FastAPI checks explicit routes first: `/`, `/stats`, `/config`, `/admin/...`, `/docs`, `/openapi.json`. Anything else falls to the catch-all proxy route.
3. **Classify request.** The proxy classifies the path: `/cpu...` -> `cpu`, `/io...` -> `io`, `/payload...` -> `payload`, everything else -> `light`. This becomes a request-type one-hot feature for ML.
4. **Build request context.** A `RequestContext` gets a UUID, method, path, type, and declared `Content-Length` converted to KB. The body is then read and `payload_kb` is replaced with the actual body length. This is important because `payload_kb` is an ML feature known **before** choosing a server.
5. **Take a state snapshot.** For each server, `ServerInfo` holds static config plus the latest `ServerMetrics`: CPU, memory, inflight, request rate, EWMA latency, p95 latency, error rate, CPU limit, CPU delta.
6. **Apply resilience filters.** `resilience.choose()` removes servers already tried for this request, removes servers whose circuit is open, and probabilistically removes a share of a slow-starting server. If the newest snapshot is older than 3 s, it does not call the normal algorithm; it uses last-known healthy servers with least-loaded selection.
7. **Choose a server.** The selected algorithm chooses one. For ML, one feature row is built per healthy candidate, all rows are scored in one `inplace_predict` call, predictions are converted to milliseconds, and each server's expected wait is `predicted_ms * (1 + inflight)`. The minimum wins; scores within 5 ms are ties, broken by lower inflight then lower prediction.
8. **Send upstream.** The proxy increments `inflight_local` for the chosen server, builds `http://backendN:8000/<path>?<query>`, and calls the shared async `httpx` client. Docker DNS resolves `backendN`; the httpx pool may reuse an existing TCP connection or open a new one.
9. **Backend processing.** The backend handles `/ping`, `/cpu`, `/io`, `/payload` or the catch-all, updates its own 5-second request statistics, and returns a JSON response and HTTP status.
10. **Receive response.** The proxy stores status, body and headers, decrements `inflight_local`, and records the outcome in the breaker: status < 500 counts as success; connection errors, timeouts and 5xx count as failures.
11. **Retry if needed.** If this was attempt 0 and the status is 500/502/503/504, the proxy retries **once** on a different server (the failed server is in `tried`). A connection error/timeout is mapped to 502, so it also enters this retry path. If the first attempt returned a valid 2xx/3xx/4xx, it does not retry.
12. **Log final outcome.** The request log gets exactly one row for the client request, using the server that actually served the final attempt, total client-visible duration, and final status. The decision log gets one row with the reason, prediction, retries, feed age, inflight and `model_driven`.
13. **Update observability.** Successful latencies go into the 60-second p50/p95/p99 window; the last 15 decisions feed the dashboard; ML calls `observe_outcome()` to update the drift guard (first attempt only, successful requests only).
14. **Return to client.** The proxy returns the upstream status, body and headers. `Host` and `Content-Length` are removed before forwarding; `httpx` recomputes `Content-Length`. The client sees the balancer, not the backend.

### 4.1 Failure branches

| Situation | What happens |
|---|---|
| Backend returns 503 on first attempt | Breaker sees a failure; one retry is sent to another server; if retry succeeds, client gets 200 and the decision row says `retries=1`. |
| Backend connection refused/timeout | Same as above; proxy maps final upstream error to **502**. |
| All servers circuit-open or unhealthy | `resilience.choose()` returns `None`; proxy returns **503** `"No healthy servers available"`. |
| Metrics feed paused >3 s | Normal algorithm skipped; least-loaded among last-known healthy; reason starts `stale-feed(least_conn)`. |
| ML artifact missing/corrupt | ML router falls back to least-connections; reason `ml-fallback`; `/stats.ml.load_error` shows why. |
| Drift guard demoted model | Routing uses least-connections but still shadow-predicts; reason `drift-fallback(...)`; `model_driven=false`. |
| First attempt 404/400 | No retry (client's answer), breaker counts it as success because status <500. |
| Both attempts fail | Client gets final 5xx (or 502 if no content); one request-log row records the final outcome. |

### 4.2 Where the latency comes from

`response_ms` is client-visible end-to-end time: time waiting in the balancer, route decision, upstream connection wait/pool wait, backend queueing and service time, response transfer, and a possible retry. It is **not** just the model inference time. H3 measures only the router's decision timer (mean 1.96 ms), not total latency.

Useful intuition:

```text
client latency = proxy/queueing + decision + upstream wait + backend service + response transfer
```

For a queueing system, Little's Law is a useful approximation:

```text
L = lambda * W
in-flight requests = throughput * average time in system
```

That is why a high in-flight count and a high p95 usually appear together.

---

## 5. Computer Networks concepts used in the project

### 5.1 OSI / TCP-IP mapping

| OSI layer | Project example | Notes |
|---|---|---|
| 7 - Application | HTTP endpoints `/ping`, `/cpu`, `/io`, `/payload`; routing by URL path; `/metrics`; DNS | The load balancer is an **L7** device because it understands HTTP. |
| 6 - Presentation | JSON serialization; Parquet for logs | HTTP bodies are JSON; Parquet is storage, not network protocol. |
| 5 - Session | HTTP is stateless; no explicit session layer | Keep-alive is a TCP connection-reuse mechanism, not an HTTP session. |
| 4 - Transport | TCP; ports 9090/8080/8000; connection pool; retransmission/ordering | HTTP/1.1 uses TCP; the OS handles reliability. |
| 3 - Network | IP addressing; Docker bridge subnet; host port publishing | Docker assigns private container IPs; host port mapping/forwarding exposes 9090. |
| 2 - Data link | Docker bridge, veth pairs, Ethernet | The bridge behaves like a virtual L2 switch between containers. |
| 1 - Physical | Host NIC/CPU/memory | Not part of the application logic. |

Do **not** confuse the project's "routing algorithm" with network routing protocols such as OSPF or BGP. The project chooses an **application server**; Docker/Linux handles packet delivery.

### 5.2 L4 vs L7 load balancing

| | L4 load balancer | L7 load balancer (this project) |
|---|---|---|
| Sees | Source/destination IP, ports, TCP/UDP | Full HTTP request: method, path, headers, body, status |
| Routing decision | Connection/flow based | URL, request type, payload, cookies, headers, server metrics |
| Can retry on HTTP 5xx? | Not at HTTP level | Yes; this project retries once on 500/502/503/504 |
| Can log request latency/status? | Only connection-level | Yes; full request/decision logs |
| Cost | Lower | Higher: parses HTTP, runs Python/ML, buffers bodies |
| Example | Linux IPVS, cloud NLB | Nginx, HAProxy, Envoy, this FastAPI proxy |

Important nuance: an L7 balancer still uses TCP underneath. "L7" describes what it **inspects**, not a replacement for L4.

### 5.3 Reverse proxy, forward proxy and NAT

- **Forward proxy:** a client explicitly uses it to reach other servers; it acts on behalf of clients.
- **Reverse proxy:** clients connect to it as if it were the server; it forwards to hidden upstreams. A load balancer is a reverse proxy with a server-selection policy.
- **NAT:** rewrites source/destination IP/port at L3/L4. Docker publishes `ports: "9090:8080"` through host port mapping (NAT on Linux; Docker Desktop's forwarding layer on Windows/macOS). It does not parse HTTP.
- **This project:** host:9090 -> Docker port mapping -> balancer:8080 -> new HTTP connection -> backend:8000. The client TCP connection terminates at the balancer; the upstream connection is separate. That is why the proxy can retry, change headers, log status, and choose a different server.

### 5.4 HTTP and TCP

- HTTP/1.1 request/response over TCP. The project does not use HTTP/2 or TLS in the default stack.
- Methods used: `GET` for most endpoints, `POST` for `/payload`, admin/demo/run endpoints.
- Status codes: 200 success, 400 bad admin parameter, 404 unknown admin route, 500 internal proxy error, 502 bad gateway/upstream error, 503 no healthy server or injected fault.
- TCP provides ordered, reliable byte delivery, retransmission, flow control and congestion control. Application-level retry is different: it re-sends an HTTP request after a failure and can duplicate a non-idempotent operation.
- The classic TCP handshake is SYN, SYN-ACK, ACK; closing is FIN/ACK in both directions. Connection pooling avoids a new handshake for every request.
- A socket/connection is identified by the 4-tuple: source IP, source port, destination IP, destination port (plus protocol). The client's source port is usually ephemeral.
- HTTP is stateless. The project does not use sticky sessions; each request can go to any backend because the demo endpoints are stateless. If sessions mattered, you would need cookies/affinity or a shared session store.

### 5.5 DNS and Docker service discovery

- Docker Compose creates a user-defined bridge network named `balancer-net`.
- Docker's embedded DNS server (inside the Docker network, commonly `127.0.0.11`) resolves service/container names such as `backend1`, `backend2`, etc.
- The balancer config uses `host: backend1`, not a fixed container IP. If a container restarts and gets a new IP, the name still resolves.
- This is **service discovery at the Docker DNS level**, not a client-facing DNS load balancer and not Consul/Kubernetes.
- DNS itself is an application-layer protocol (normally UDP/TCP port 53). It maps names to IPs; it does not choose the least-loaded backend in this project.

### 5.6 Ports, NAT and Docker networking

| Port | Where | Purpose |
|---|---|---|
| 9090 | Host | Published port used by browser/curl and the host-side `demo_traffic.py`. The benchmark Locust container does **not** use it. |
| 8080 | Balancer container | Uvicorn listening port (mapped from host 9090). Locust on the Compose network targets `http://balancer:8080`. |
| 8000 | Each backend container | FastAPI/Uvicorn internal port; not published. |

- `EXPOSE 8000` in a Dockerfile is documentation; it does not publish a port. `ports:` in Compose performs host publishing.
- Backend ports are deliberately not published. The balancer and other containers on `balancer-net` can reach them directly.
- **Important nuance:** the backend's `/slow` and `/fault` control endpoints are *not host-published*, but they are **not private** either: the balancer's catch-all proxy forwards `/slow` and `/fault` from `localhost:9090` to a chosen backend. There is no auth or path block, so they are demo-only exposure. Production must block or remove these paths.
- Docker bridge networking is a virtual L2 segment; container-to-container traffic is switched on the bridge and routed at L3 when needed. Host-to-container published ports are forwarded by Docker's port mapping (NAT on Linux, Docker Desktop forwarding on Windows/macOS).
- `localhost` from the host means the host loopback. Inside a container, `localhost` means that container itself; use service names for other containers.

### 5.7 Keep-alive, connection pools and queueing

- The balancer uses one shared `httpx.AsyncClient` with:
  - `max_connections=500`
  - `max_keepalive_connections=500`
  - `keepalive_expiry=60 s`
  - upstream timeout 30 s (`PROXY_TIMEOUT_S`)
- Keep-alive avoids repeating TCP handshakes. Without it, every proxied request would pay an extra round trip and connection setup.
- The pool is shared across all upstream hosts. If more than 500 connections are needed, requests wait in the client pool; that wait is not backend service time but can appear in client latency. This is why the pool is documented in the code.
- Backends also queue. `/cpu` uses a semaphore whose width is the rounded CPU quota; excess CPU requests wait asynchronously. `/io` deliberately increases its sleep with concurrency to simulate a shared downstream dependency. `inflight` captures this queue.
- Overload should appear as **queueing/increased latency**, not necessarily errors. The 80-user stress run had 7,353 requests, all 200, but p50 625 ms and p95 902 ms.

### 5.8 Health checks and failover

- **Active health check:** collector GETs `/metrics` about every 1 s. A 200 response means the server answered; non-200/exception counts as a failure. Three consecutive failures mark it unhealthy; three successes recover it.
- **Passive health / circuit breaker:** every proxied request outcome updates the breaker. Three consecutive proxy failures open the circuit for 10 s. This reacts to actual user traffic, often before the next health poll.
- **Stale feed:** if the newest snapshot is older than 3 s, the collector is considered dead; the router falls back to last-known healthy servers and least-loaded selection.
- **Slow start:** a recovered server begins at a 10% share and ramps to 100% over 10 s. The implementation drops a random share of decisions; it is a ramp in expectation, not a token bucket.
- **Retry:** one retry on a different server for 500/502/503/504, including a transport error mapped to 502. It reduces client-visible errors but can duplicate a non-idempotent request if the original response was lost.

### 5.9 Latency, throughput and percentiles

| Term | Meaning in this project |
|---|---|
| Latency | Client-visible `response_ms` for one request (including proxy, queueing, retry). |
| Throughput | Requests per second over a run, usually rows / run span. |
| p50 | Median; half the successful requests are at or below it. |
| p95 | 95th percentile; a tail-latency measure. Computed by nearest rank: sorted values, index `ceil(0.95*n)-1`. |
| p99 | 99th percentile; more tail-sensitive, noisier with small samples. |
| SLA | `sla_ms: 500`; measured as the share of successful requests over 500 ms. It is reported, not used as an argmin filter. |
| EWMA | Exponentially weighted moving average; backend `rt_ewma_ms` uses alpha 0.15. |
| Jain's index | Fairness measure: `(sum x_i)^2 / (n * sum x_i^2)`; 1 = perfectly even. Equal request counts do not mean equal work on heterogeneous servers. |
| CPU imbalance | Spread of per-server CPU utilization; lower is more balanced. |

Why p95 matters: a mean can look healthy while 5% of users wait several seconds. The project reports p95 and CIs at the **run level**, not pooled per-request, because requests inside a run are correlated.

### 5.10 Proxy behavior and headers

- The proxy removes `Host` (the upstream should see backendN, not localhost:9090) and `Content-Length` (httpx recomputes it after reading the body). Other request headers are forwarded.
- The proxy reads the entire request body before choosing, because payload size is a feature. This is not a streaming proxy; large uploads would consume memory.
- The response body is read fully and returned as a FastAPI `Response`; response headers are forwarded. There is no response caching and no compression logic in the balancer.
- The implementation does **not** add `X-Forwarded-For`, `X-Forwarded-Proto` or `X-Forwarded-Host`. If asked, say this is a known production gap; adding them requires a trusted-proxy policy.
- It does not implement hop-by-hop header filtering, WebSocket upgrade, HTTP/2, or TLS termination. These are production features, not claims of this project.

### 5.11 What is not implemented (say this honestly)

- No TLS/HTTPS; traffic is plain HTTP in the demo.
- No HTTP/2, WebSocket or streaming proxy.
- No sticky sessions/cookie affinity.
- No authentication/authorization on the dashboard or admin endpoints.
- No rate limiting, WAF, DDoS protection or request validation beyond FastAPI parameters.
- No high availability for the balancer itself: one process/host is a single point of failure.
- No distributed coordination, consensus, Kubernetes or service mesh.
- No distributed tracing/OpenTelemetry; metrics and Parquet logs only.
- No active replay of each request against all backends, so top-1 ranking accuracy is not directly observable.
- No Nginx/HAProxy measurement in the primary 125-cell matrix (reference configs exist separately).

---

## 6. Load-balancing algorithms in detail

### 6.1 Common interface

Every policy implements:

```python
class Algorithm:
    def choose_server(self, request, servers) -> ServerChoice | None:
        ...
```

The proxy, resilience layer, logging, dashboard and metrics are identical across policies. Only `choose_server()` changes. This is what makes the comparison attributable to the algorithm rather than to different plumbing.

`ServerChoice` carries:

- `server`: the chosen `ServerInfo`
- `reason`: for logs/dashboard, e.g. `round_robin`, `least_conn`, `ml(pred 42.1ms, inflight 3)`
- `predicted_rt_ms`: optional model prediction
- `model_driven`: true only when the model made the choice

### 6.2 Algorithm comparison

| Algorithm | Selection rule | Complexity | Strength | Weakness |
|---|---|---|---|---|
| Random | Uniform choice among healthy servers | O(n) to filter, O(1) choice | Simple, no state, unbiased | Ignores load and capacity; high variance. |
| Round-robin | Cycle through healthy servers in order | O(1) | Even request count, no state | Assumes homogeneous servers; ignores queue and request cost. |
| Weighted RR | Repeat each server `max(1, int(weight))` times in a cycle | O(1) after build | Approximates capacity ratio | Static; fractional weight truncation (see 6.4); ignores current load. |
| Least-connections | `argmin inflight` | O(n) | Reacts to current queue; strong on heterogeneous/unbalanced load | Does not know request cost or server capacity; ties arbitrary. |
| Weighted least-connections | `argmin inflight/weight` | O(n) | Capacity-aware queue balancing | Requires accurate weights; not in primary matrix. |
| Least-response-time | `argmin rt_ewma_ms` | O(n) | Uses observed server latency | EWMA is stale and can herd; ignores current queue; weak in results. |
| ML (herd) | `argmin predicted_ms * (1 + inflight)` | O(n * inference) | Request-aware, nonlinear, capacity/latency features; good degraded mean | Live prediction error; can avoid small backend; not uniformly better. |
| ML predictive | `argmin predicted_ms` | O(n * inference) | Pure prediction ablation | Herds badly on stale snapshots; measured worse in burst. |

### 6.3 The traditional policies

**Random.** `random.choice(healthy)`. It is a baseline showing what happens when no load information is used. It is implemented but not one of the five policies in the primary 125-cell matrix.

**Round-robin.** Maintains an `itertools.cycle` over the current healthy set. When the set of healthy server IDs changes, the cycle is rebuilt. It is O(1) per request and assumes equal capacity and equal request cost. With 0.8/0.4/0.4/0.15 quotas, that assumption is false.

**Weighted round-robin.** Builds a cycle by repeating each server `max(1, int(weight))` times. Intended design ratio is 4:2:2:1, but `int(0.5)` is 0, so the 0.5-weight backend gets one slot. Effective slots are 2:1:1:1. This is a code-level nuance worth knowing; weighted least-connections uses the exact float weight.

**Least-connections.** Picks the server with the lowest `inflight` property. `inflight` is the proxy's local live count when >0, otherwise the backend-reported snapshot. It is simple, cheap and reacts to the queue that exists now. It is the strongest policy overall in this project.

**Weighted least-connections.** Picks `inflight / max(weight, 1e-6)`, so a 2x server can carry twice the in-flight work before looking as loaded as a 1x server. It is implemented but not part of the primary matrix.

**Least-response-time.** Picks the lowest `rt_ewma_ms` from the last backend snapshot. It does not include current inflight. A server with a low historical EWMA can still have a large queue, and the EWMA is stale; this is a classic herd/staleness problem.

### 6.4 Why weights and Docker quotas differ

`balancer/config.yaml` has:

- `weight`: 2.0, 1.0, 1.0, 0.5 (design ratio)
- `cpu_limit`: 2.0, 1.0, 1.0, 0.5 (feature value)

`docker-compose.yml` has actual CPU ceilings 0.8, 0.4, 0.4, 0.15. The actual quotas are scaled down to fit the 4-vCPU test host. The config values preserve the 4:2:2:1 ratio and are what the collector publishes as `cpu_limit`; this keeps training and serving on the same feature distribution. The real performance still differs because `srv_cpu_pct` is normalized by each backend's own quota and the latency features reflect actual behavior.

If asked "why not use the Docker quota directly?", answer: the feature was trained on the configured design value; changing it at serving time would create train/serve skew. The enforced quota is an implementation detail, while `cpu_limit` is the stable capacity signal.

### 6.5 ML variants

- `ml`: shipped router, queue/herd correction enabled.
- `ml_herd`: the same class under the plan's name; not a second implementation.
- `ml_predictive`: the ablation with **no** queue correction; exists to measure why the queue term matters.

Do not claim `ml_herd` is a different model. It is a naming alias.

### 6.6 Which policy would you choose for a scenario?

| Scenario | Sensible choice | Reason |
|---|---|---|
| Homogeneous servers, equal requests | Round-robin | Simple, even, no metrics needed. |
| Heterogeneous servers, steady load | Least-connections or weighted least-connections | Reacts to queue; capacity-aware. |
| Request costs vary enormously | ML or request-aware routing | A `/ping` and a `/payload` should not consume the same share. |
| Bursty traffic | Least-connections is a strong default | It sees the queue now; a stale prediction may lag. |
| Faulted/degraded backend | ML or breaker + least-connections | Error/latency features can avoid the fault; breaker handles hard failure. |
| Metrics unavailable | Least-connections on last-known state, or fail closed | Do not trust stale predictions. |

---

## 7. ML pipeline in detail

### 7.1 Problem formulation

For each request \(r\) and candidate server \(s\):

```text
input:  x(r, s) = server state + request features
target: y = log1p(response_ms)
model:  f(x) -> predicted log latency
output: predicted_ms = clip(expm1(f(x)), 0, 5000)
score:  predicted_ms * (1 + inflight(s))     # shipped ML router
choice: argmin score, with 5 ms tie tolerance
```

This is a **regression** problem, not classification. The router does not predict "healthy/unhealthy"; it predicts how long the request would take on each server and picks the lowest expected wait.

### 7.2 Data collection

- Dataset: **218,011 requests** across **18 Parquet files/runs**.
- Six traffic profiles (steady, ramp, burst, wave, mixed cost, degraded) x three collection policies (round-robin, least-connections, random), one 120 s run per cell.
- Load generated by Locust in its own container with a CPU quota; it is not run inside the balancer container, because sharing CPU would make the generator look like the balancer/backends.
- Each run is opened with `POST /admin/run?run_id=...`, measured, then sealed with `POST /admin/seal`, so each file has one run ID and a valid Parquet footer.
- The training pipeline drops warm-up rows (default 20 s) and failed requests (`status >= 400`), leaving **178,165 rows**. The training split is **121,061 rows**.

Why exclude errors from training? A 503 returns quickly; if treated as a latency observation, the model would learn that a failing server is "fast." Errors still reach the model as the `err_rate_5s` feature.

### 7.3 Split, signal gate and leakage control

- Split by **`run_id`**, not by row: 12 runs train, 2 validation, 4 test. Adjacent rows inside a run share a slowly evolving server state, so a random row split would leak validation information into training.
- The test runs include `burst_random_0`, `mixed_cost_round_robin_0`, `steady_random_0`, `wave_least_conn_0`.
- **Signal gate:** server-state features alone must predict `log1p(response_ms)` with R^2 >= 0.30 on held-out runs, with at least 5,000 rows and 4 runs. The collected data passed at **R^2 0.5499**; endpoint-only control was **0.0770**. This prevents a dataset where the endpoint identity, not server load, explains the label.
- **Generalisation holdout:** remove an entire traffic profile (burst or wave) from training and score it. Model MAE(log) 0.5530 on burst and 0.5521 on wave, about 59% better than naive. Caveat: same four backends and hardware, so this is traffic-shape generalisation, not hardware generalisation.

### 7.4 Features (14 total)

| Group | Feature | Meaning |
|---|---|---|
| Server state (9) | `srv_cpu_pct` | CPU utilisation normalized by quota. |
| | `srv_mem_pct` | Memory used as % of container limit. |
| | `inflight` | Requests currently being served/queued at snapshot. |
| | `req_rate_5s` | Completed requests per second over last 5 s. |
| | `rt_ewma_ms` | EWMA of observed response times. |
| | `rt_p95_5s_ms` | p95 response time over last 5 s. |
| | `err_rate_5s` | Share of 5xx responses over last 5 s. |
| | `cpu_limit` | Configured capacity (2/1/1/0.5). |
| | `cpu_delta_5s` | Current CPU minus rolling 5 s average (trend). |
| Request (1) | `payload_kb` | Request body size in KB. |
| Request type (4) | `req_type_light` | `/ping`/other light request. |
| | `req_type_cpu` | `/cpu` CPU-bound request. |
| | `req_type_io` | `/io` simulated I/O wait. |
| | `req_type_payload` | `/payload` large response request. |

Not every metric is a feature: `load_avg`, `net_io_bytes_per_s` and `poll_latency_ms` are collected but not in the 14-column model input. `poll_latency_ms` is diagnostic only.

### 7.5 Models and hyperparameters

| Model | Purpose | Configuration |
|---|---|---|
| Naive | Baseline | Predict the server's own `rt_ewma_ms`; no learning. |
| Ridge | Linear reference | StandardScaler + Ridge(alpha=1.0). |
| Random Forest | Nonlinear reference | 200 trees, max depth 14, min samples leaf 5. |
| XGBoost | Shipped model | 400 estimators, max depth 6, learning rate 0.06, subsample 0.8, colsample 0.8, lambda 1.0, hist tree method. |

Why XGBoost: tabular data, nonlinear interactions (request type x quota x queue), fast training, strong accuracy, feature importance, fast CPU inference. Random Forest was close in accuracy but about nine times slower at serving (51.7 ms vs 5.95 ms for a five-candidate batch in the project report); a neural network would add tuning/serving complexity without a clear gain.

### 7.6 Offline results

| Model | MAE(log) | MAE(ms) | R^2(log) | Spearman |
|---|---:|---:|---:|---:|
| Naive (server EWMA) | 1.2738 | 140.5 | -0.2907 | 0.414 |
| Ridge | 0.6402 | 111.5 | 0.6449 | 0.692 |
| Random Forest | 0.5912 | 75.4 | 0.6757 | 0.704 |
| **XGBoost (shipped)** | **0.5868** | **73.1** | **0.6827** | **0.710** |

Grouped 5-fold CV R^2 is **0.6455**, so the result is not one lucky split. Against the plan's 30% MAE-improvement target, XGBoost is **53.9% better** than naive.

Feature importance (top): `rt_p95_5s_ms` 0.359, `req_type_cpu` 0.162, `rt_ewma_ms` 0.097, `req_type_light` 0.096, `cpu_limit` 0.062, `req_rate_5s` 0.060. No single feature carries the prediction.

### 7.7 Serving path

1. Load artifact: `MODEL_PATH` if set; otherwise newest `training_report_*.json`, then `{best_model}_{data_hash}.joblib`. Fallback to newest `.joblib`.
2. Validate feature columns against `ml.features.feature_columns()`; a mismatch refuses the artifact.
3. On each decision, build one row per healthy candidate with `build_serving_array` (numpy, not pandas).
4. Call `inplace_predict` if available; otherwise `predict`.
5. Convert log predictions with `expm1` and clip to 0-5000 ms.
6. Score with queue term, choose minimum, break ties by lower inflight then lower prediction.
7. If no model, load error, or prediction exception -> least-connections fallback and increment `fallbacks`.
8. If drift guard demoted -> least-connections but still compute a shadow prediction for the chosen server; increment `drift_routed`.
9. `status()` exposes loaded state, artifact, estimator, decision count, fallbacks, drift state, decision mean/max and training hash through `/stats`.

Why batched prediction? One feature matrix for all candidates and one model call avoids per-server Python overhead. Why numpy? Building one DataFrame per candidate plus `pd.concat` cost about 5 ms per decision; the numpy path cost about 0.8 ms in the image, with identical predictions asserted by tests.

### 7.8 Queue/herd correction and tie-breaking

- The feature snapshot is not instantaneous. The collector refreshes roughly once per second, and a request can arrive much faster than that.
- Herd problem: a static predictor sees the same idle-looking snapshot for many requests; they all choose the same server; its queue explodes while other servers sit idle. The first serving version measured p50 **22 ms -> 172 ms** (about 8x) and the busiest server's one-second share **0.25 -> 0.81**.
- Fix: `score = predicted_ms * (1 + inflight)`. Each outstanding request is charged one predicted service time. This is a heuristic, not an exact queueing model.
- A capacity-weighted variant (`inflight / weight`) was tried and reverted: it sent too much burst traffic to the 2x backend and its p95 nearly doubled in the mixed profile.
- Without the queue term (`ml_predictive`), burst p95 is **700 ms** vs **342 ms** with it; throughput 73 vs 102 req/s; herd faster in 3/3 paired runs. Without it, 46-69% of traffic goes to backend1 and only 1-3% to backend4; with it, the shares are 29/28/29/14.
- Tie rule: candidates whose score is within **5.0 ms** of the best are tied; choose the one with lower `inflight`, then lower predicted latency. Older docs say 1 ms; the code uses 5 ms.

### 7.9 Drift guard

Goal: detect when the model's live error grows and temporarily stop using it.

```text
error_i = |log1p(actual_ms) - log1p(predicted_ms)|
live_mae = mean(error_i) over the last `window` observations
```

Defaults:

- window = 200 observations
- min samples = 50
- demotion threshold = 1.0 log-MAE
- promotion hysteresis = 0.8 (must be below 0.8 over a full window)
- while demoted, still shadow-predict; routing uses least-connections
- clear the error window on demotion so re-promotion needs fresh evidence
- do not score failed requests or retried requests

Why threshold 1.0: XGBoost offline MAE(log) is 0.5868 and naive is 1.2738. A threshold of 1.0 sits between them: it means the model has degraded substantially, though not necessarily all the way to the naive baseline. It is a line, not a finely tuned value. The matrix pinned it at 10.0 to compare routing policies; at the shipped 1.0, steady/ramp/failure windows would have demoted. That is reported as a finding, not hidden.

### 7.10 Why no top-1 accuracy?

The observational log records the latency of the server that served each request, never what the other three would have done for the same request. Therefore there is no observed "fastest server" label per request. Reporting top-1 accuracy would require replaying every request against every server (active probing), which changes the system. The project reports Spearman rank correlation and states the limitation.

### 7.11 Live results and interpretation

| Scenario | Round Robin | Weighted RR | Least Conn | Least RT | ML (herd) |
|---|---:|---:|---:|---:|---:|
| Steady | 789 | 396 | **214** | 302 | 433 |
| Ramp | 575 | 287 | **169** | 598 | 273 |
| Burst | 733 | 372 | 275 | 450 | **243** |
| Mixed cost + degraded | 474 | 433 | 332 | 269 | **182** |
| Server failure mid-run | 612 | 555 | **209** | 636 | 734 |

Values are mean p95 in ms over five runs. ML has the lowest mean p95 in burst and degraded, but neither difference is significant after Holm correction at n=5. It is worse than least-connections in steady, ramp and failure. H1 requires three significant wins; ML has **zero**, so H1 is **not met**.

Throughput (req/s, mean of 5):

| Scenario | Round Robin | Weighted RR | Least Conn | Least RT | ML (herd) |
|---|---:|---:|---:|---:|---:|
| Steady | 112 | 118 | **150** | 129 | 109 |
| Ramp | 133 | 143 | **160** | 109 | 121 |
| Burst | 120 | 124 | **141** | 107 | 127 |
| Mixed cost + degraded | 113 | 101 | **122** | 118 | 120 |
| Failure | 130 | 111 | **155** | 92 | 80 |

Least-connections has the highest throughput in all five scenarios. Errors are ~0.00% for all policies because the shared retry/breaker layer absorbs failures; the cost shows up in latency and SLA.

Live MAE(log) on served traffic: steady 1.394, ramp 1.151, burst 0.898, degraded 0.519, failure 2.421, versus 0.5868 offline. The failure/steady collapses coincide with the router sending only 4-7% of traffic to the small 0.15-core backend, losing capacity; this is a capacity-avoidance failure mode, not an error-rate failure.

### 7.12 Honest ML conclusions

- Offline prediction: strong and reproducible.
- Live routing: conditional, not uniformly better.
- Least-connections is the best default on this testbed.
- The queue term is essential; without it, ML is much worse under burst.
- ML's clearest live advantage is degraded/fault avoidance (mean), but it is not statistically significant in the matrix.
- At the shipped drift threshold, the guard would demote on some live windows; the matrix pinned it to isolate routing, which is an experiment-design choice and a limitation.

---

## 8. Resilience and failure handling

### 8.1 Mechanisms

| Mechanism | Trigger | Action | Evidence |
|---|---|---|---|
| Active health check | `/metrics` poll failure | 3 consecutive failures -> unhealthy; 3 successes -> healthy + slow start | collector |
| Circuit breaker | 3 consecutive proxied failures | Skip server for 10 s; next success closes; failure reopens | chaos: 1.09 s bypass |
| Retry | 500/502/503/504 on attempt 0; transport error becomes 502 | One retry on a different server | 2,934 chaos requests, 0 errors |
| Slow start | Server recovered/tripped success | 10% -> 100% over 10 s, stochastic admission | observed in `/stats` |
| Stale-feed fallback | newest snapshot >3 s | normal algorithm/ML skipped; least-loaded among last-known healthy | 30/30 200s after 3.1 s pause |

### 8.2 Circuit breaker state

```text
CLOSED --3 consecutive failures--> OPEN (10 s)
OPEN --cooldown expires--> eligible again (time-based half-open window)
  success -> CLOSED + slow start
  failure -> OPEN (10 s again)
```

Implementation nuance: the code uses `circuit_open_until` and `circuit_tripped`; after cooldown, the server is eligible, and a successful request clears `circuit_tripped` and starts slow start. There is no separate per-server semaphore limiting the half-open probe to exactly one request, so under concurrency more than one request can be admitted right after cooldown. The docs call it a half-open probe; describe the intended behavior and mention the time-based implementation if pressed.

### 8.3 What happens in each failure scenario

**Backend returns 503 (injected fault, process alive):**
- First attempt fails; breaker increments consecutive failures.
- Retry goes to another server; client usually sees 200.
- After three consecutive failures, circuit opens for 10 s; new work avoids it.
- Health probe may stay green because `/metrics` still answers; the dashboard shows "probe up - not routable" and "routable 3/4". This is a key distinction.

**Backend container is killed:**
- Existing connection fails -> proxy maps to 502, retries once.
- Breaker opens after 3 failures (about 1 s in the chaos test).
- Collector marks unhealthy after 3 failed polls; when restarted, 3 successful polls and slow start.
- H4: worst traffic redirect 1.13 s; final 5xx 0.00%; traffic returns after 7.7 s on average.

**Metrics collector paused:**
- Snapshots age; after 3 s, `feed_is_stale` is true.
- Normal algorithm/ML is skipped; least-loaded among last-known healthy is chosen.
- If all last-known servers are also unhealthy, returns 503.
- Resume collector; normal routing returns.

**No artifact / corrupt artifact:**
- ML router falls back to least-connections; `ml-fallback` reason; `fallbacks` increments.
- `/stats.ml.load_error` explains why.

**Drift:**
- Guard demotes; routing least-connections; shadow prediction continues.
- `drift_routed` increments; `reason` starts `drift-fallback`.
- A clean full window promotes back.

**All servers down:**
- No candidate; HTTP 503 with `"No healthy servers available"`.
- A dead cluster is not made to look alive by stale fallback: `least_loaded` only considers `metrics.healthy`.

### 8.4 Why the circuit breaker is separate from health checks

- Health check asks "does `/metrics` answer?" It can be green while the application returns 503 for every user request.
- Breaker asks "are actual proxied requests failing?" It catches application failures and connection failures even if the process is alive.
- Conversely, a backend that fails `/metrics` but serves user traffic would be marked unhealthy by the collector; that is an intentional policy (monitoring is part of service health).
- Two mechanisms are independent, which is why the dashboard shows both "health probes" and "routable."

### 8.5 Retry trade-offs

- One retry only; no exponential backoff/jitter or retry budget.
- Retry only for 500/502/503/504 (including a transport error mapped to 502); 4xx is the client's answer.
- A lost response after the backend processed the request may cause duplicate execution. Demo endpoints are stateless; production should restrict retries to idempotent methods or add de-duplication.
- Retrying a server that returns 500 because of a request-specific bug penalizes that server for every client repeating it; the 10 s cooldown bounds the damage.

### 8.6 Stress and chaos evidence

- 80 concurrent users, 60 s, 7,353 requests, all 200; p50 625 ms, p95 902 ms; no restart; RSS 141.9 -> 146.6 MiB (+3.3%); `/stats` stayed responsive; feed age max 1.1 s.
- Chaos kill: breaker opened 1.09 s after `docker compose stop backend3`; 2,934 requests, 0 errors; recovered 5.0 s after restart; slow start observed.
- Stale feed: 3,109 ms pause, 30 requests all 200, stale_routing +30, ML decisions +0.

---

## 9. Backend design and metrics

### 9.1 Endpoints

| Endpoint | Purpose | Notes |
|---|---|---|
| `GET /health` | Simple liveness/health JSON | Exists, but the balancer's collector actually uses `/metrics` for health+state. |
| `GET /metrics` | Full metrics JSON | Polled by collector; `cache-control: no-store`; excluded from app statistics. |
| `GET /ping` | Cheap light request | `{"id": ..., "type": "ping", "result": "pong"}`. |
| `GET /cpu` | Constant CPU work | 80,000 iterations in a thread; semaphore width = CPU quota. |
| `GET /io` | Simulated I/O wait | Wait grows with concurrency: `25 ms * sqrt(inflight) * jitter / quota^0.25`, capped 6x. |
| `GET/POST /payload` | Large JSON response | 2,000 items by default; POST body size is the payload feature. |
| `GET /slow?ms=&duration_s=` | Inject latency for a window | Control endpoint; backend port not host-published, but reachable through the balancer catch-all (`localhost:9090/slow`). |
| `GET /fault?error_rate=&duration_s=` | Inject 503s for a window | Control endpoint; backend port not host-published, but reachable through the balancer catch-all (`localhost:9090/fault`). |
| `GET/POST/PUT/DELETE/PATCH /{path}` | Catch-all | Does 1/8 CPU work; makes unknown paths represent "light" traffic. |

### 9.2 Metric definitions

| Metric | Definition |
|---|---|
| `cpu_pct` | `(cpu usage delta / elapsed) / CPU_LIMIT * 100`, clipped 0-100; cgroup v2 `cpu.stat` or psutil fallback. |
| `mem_pct` | cgroup `memory.current / memory.max * 100`; fallback psutil. |
| `load_avg` | `/proc/loadavg` 1-minute value; fallback psutil. |
| `net_io_bytes_per_s` | `/proc/net/dev` byte delta / elapsed; collected but not a model feature. |
| `inflight` | Middleware count of in-flight application requests; excludes `/metrics`, `/health`, `/fault`, `/slow`. |
| `req_rate_5s` | Number of application events in last 5 s / 5. |
| `rt_ewma_ms` | EWMA alpha 0.15 of observed application latency. |
| `rt_p95_5s_ms` | Nearest-rank p95 over last 5 s of events. |
| `error_rate_5s` | 5xx count / events in last 5 s. |
| `cpu_limit` | Configured capacity from server config (2, 1, 1, 0.5). |
| `cpu_delta_5s` | Current CPU minus rolling 5 s average; computed by collector. |
| `poll_latency_ms` | Latency of the `/metrics` poll itself; diagnostic only. |

### 9.3 Why `/metrics` is excluded from application statistics

If monitoring requests counted as traffic, they would inflate `req_rate_5s` and pollute the training data. The middleware explicitly excludes `/metrics`, `/health`, `/fault`, `/slow`. The `/metrics` response sets `cache-control: no-store` so proxies do not serve stale metrics.

### 9.4 Fault injection for demos

- `degrade`: backend3 gets +400 ms delay and 35% errors for 45 s.
- `fail`: backend3 returns 503 for every request for 12 s; health probe stays green; breaker opens.
- `clear`: removes both injections.
- The backend ports are not host-published, but this does **not** make the controls private. The dashboard uses the balancer's `/admin/demo/*` wrappers; the same backend paths are also forwarded by the public catch-all, so `GET http://localhost:9090/slow?...` and `GET http://localhost:9090/fault?...` reach a backend. There is no authentication. This is a deliberate demo convenience and a production security gap.

---

## 10. Logging, observability and dashboard

### 10.1 Request log: 16 columns

`logs/<run_id>_<timestamp>.parquet`, one row per client request, final outcome and total latency.

```text
timestamp, run_id, req_type, payload_kb, server_id,
srv_cpu_pct, srv_mem_pct, inflight, req_rate_5s,
rt_ewma_ms, rt_p95_5s_ms, err_rate_5s, cpu_limit,
cpu_delta_5s, response_ms, status
```

Why exactly 16? It is the dataset/training schema. Keeping it stable means all training/analysis code and prior runs remain compatible.

### 10.2 Decision log: 14 columns

`decisions/decisions_<run_id>_<timestamp>.parquet`, one row per client request.

```text
timestamp, request_id, run_id, algorithm, server_id, reason,
predicted_ms, actual_ms, status, retries, model_driven,
inflight, feed_age_ms, req_type
```

Why separate? The request log must stay exactly 16 columns and has no request ID. Routing decisions need reason/prediction/retry/feed-age fields. The join key is `(run_id, timestamp)`; both files write the same request-start float. A 2,601-request run joined exactly on every row.

### 10.3 Logging behavior

- Buffered 50 rows or 2 s; flush in a worker thread; `log()` is non-blocking on the event loop.
- `pa.Table.from_pylist` is used instead of pandas for speed.
- `POST /admin/run?run_id=...` seals the previous file and starts a new one.
- `POST /admin/seal` flushes/closes so the Parquet footer is valid.
- Rows outside a run are dropped and counted, not mislabelled.
- One row per client request even if a retry happened; `response_ms` includes both attempts, final server wins.

### 10.4 `/stats` payload

`/stats` is the live control-plane snapshot. Important fields:

- `servers[]`: id, host, port, weight, cpu_limit, healthy, sent_since_snapshot, inflight_local, metrics, health_failures, circuit_open, consecutive_failures, circuit_opens, slow_start
- `algorithm`, `sla_ms`, `inflight_total`
- `ml`: loaded, artifact, estimator, load_error, decisions, fallbacks, drift_routed, drift (live MAE/threshold/window/samples/demotions/promotions), decision_ms_mean/max, trained_on, train_rows
- `feed`: age_ms, stale, collector status
- `resilience`: retries, stale_routing, circuits_open, circuit_opens
- `latency`: 60 s window p50/p95/p99, samples
- `decisions.recent`: last 15 decisions with server, reason, predicted vs actual, status, retries, model_driven

### 10.5 Dashboard panels

- **Routable** count vs **health probes**: a backend can be probe-up but circuit-open, so not routable.
- Per-server CPU relative to quota, inflight, EWMA/p95, errors, feed age.
- Alert/banner naming excluded backends.
- Traffic-share bar (5 s routed rate, browser-derived).
- p50/p95 chart over 60 s successful requests.
- "What changed?" start-vs-now snapshot (directional, not significance).
- Intervention timeline and run summary.
- ML panel: live MAE, threshold, window fill, demotions/promotions.
- Recent decisions table: predicted vs actual, error, status, model-driven.
- **Live request flow**: a sampled animation from `/stats`; dots represent approximate req/s, not packets. A failed backend is removed from future decisions; requests are never migrated from it.
- Demo controls: start/stop/seal, degrade, fail, clear, stress drift threshold, restore.

### 10.6 Metrics vs logs vs traces

- **Metrics:** numeric aggregates (`/metrics`, `/stats`); good for dashboards and alerts.
- **Logs:** per-request/decision Parquet; good for post-hoc analysis and reproducibility.
- **Traces:** not implemented. A production version would add OpenTelemetry spans for client -> balancer -> backend and propagate trace IDs.
- `poll_latency_ms` is diagnostic, never a model feature.

---

## 11. API reference and commands

### 11.1 Balancer endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Dashboard HTML. |
| GET | `/stats` | Live JSON snapshot. |
| GET | `/config` | Current config and server state. |
| GET | `/docs` | FastAPI Swagger UI. |
| GET | `/admin/algorithm` | Current algorithm and available names. |
| POST | `/admin/algorithm?name=ml` | Switch algorithm at runtime. |
| POST | `/admin/collector?enabled=false` | Pause metrics polling; demonstrate stale feed. |
| POST | `/admin/run?run_id=demo` | Seal previous run, start a new Parquet run. |
| POST | `/admin/seal` | Flush/close current Parquet files. |
| GET | `/admin/logger` | Logger state, current files, row counts, sealed runs. |
| POST | `/admin/drift?threshold=0.15` | Reconfigure drift guard (only ML). |
| POST | `/admin/demo/degrade?server_id=backend3&duration_s=45&delay_ms=400&error_rate=0.35` | Demo fault. |
| POST | `/admin/demo/fail?server_id=backend3&duration_s=12` | 100% error fault. |
| POST | `/admin/demo/clear?server_id=backend3` | Clear fault. |
| GET/POST/PUT/DELETE/PATCH | `/{path}` | Reverse proxy catch-all. |

### 11.2 Backend endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Simple health JSON. |
| GET | `/metrics` | Metrics for collector. |
| GET | `/ping` | Light request. |
| GET | `/cpu` | CPU-bound request. |
| GET | `/io` | I/O wait request. |
| GET/POST | `/payload` | Large JSON response; POST body size varies. |
| GET | `/slow?ms=&duration_s=` | Inject delay. |
| GET | `/fault?error_rate=&duration_s=` | Inject 503s. |
| GET/POST/PUT/DELETE/PATCH | `/{path}` | Catch-all. |

**Control-endpoint caveat:** `/slow` and `/fault` are not host-published on the backend ports, but the balancer's catch-all also forwards `localhost:9090/slow` and `localhost:9090/fault` to a backend. They are unauthenticated demo controls, not an isolated admin surface.

### 11.3 Useful commands (Windows PowerShell)

```powershell
# Stack up
docker compose up -d --build

# Is it alive?
curl.exe -s http://localhost:9090/stats
curl.exe -s http://localhost:9090/config
docker compose ps

# Switch algorithm
curl.exe -X POST "http://localhost:9090/admin/algorithm?name=ml"
curl.exe -X POST "http://localhost:9090/admin/algorithm?name=least_conn"

# Pause/resume metrics collector
curl.exe -X POST "http://localhost:9090/admin/collector?enabled=false"
curl.exe -X POST "http://localhost:9090/admin/collector?enabled=true"

# Start/seal a run for logs
curl.exe -X POST "http://localhost:9090/admin/run?run_id=demo"
curl.exe -X POST "http://localhost:9090/admin/seal"

# Drift demo
curl.exe -X POST "http://localhost:9090/admin/drift?threshold=0.15"
curl.exe -X POST "http://localhost:9090/admin/drift?threshold=1.0"

# Demo fault
curl.exe -X POST "http://localhost:9090/admin/demo/degrade?duration_s=45"
curl.exe -X POST "http://localhost:9090/admin/demo/fail?duration_s=12"
curl.exe -X POST "http://localhost:9090/admin/demo/clear"

# Logs
docker compose logs -f balancer

# Tests
python -m pytest -q
```

### 11.4 Example requests

```powershell
curl.exe -s http://localhost:9090/ping
# {"id":"...","type":"ping","result":"pong"}

curl.exe -s http://localhost:9090/cpu
curl.exe -s http://localhost:9090/io
curl.exe -s -X POST http://localhost:9090/payload -H "Content-Type: application/json" -d "{}"
```

The exact backend `id` depends on which server the policy chose; that is the point of the demo. The dashboard's decision reason shows the policy and predicted latency.

---

## 12. Docker, deployment and configuration

### 12.1 Compose services

| Service | Internal port | Host port | CPU | Memory | Notes |
|---|---:|---:|---:|---:|---|
| balancer | 8080 | 9090 | 2.0 | 512m | Uvicorn, proxy, dashboard, logs. |
| backend1 | 8000 | none | 0.8 | 512m | Config weight 2.0, cpu_limit 2.0. |
| backend2 | 8000 | none | 0.4 | 256m | Weight 1.0, cpu_limit 1.0. |
| backend3 | 8000 | none | 0.4 | 512m | Weight 1.0, cpu_limit 1.0; demo fault target. |
| backend4 | 8000 | none | 0.15 | 128m | Weight 0.5, cpu_limit 0.5; small capacity. |

During benchmarking, a Locust container joins the same network with its own 1.0 CPU quota. It is not in the default five-service Compose file.

### 12.2 Network and volumes

- Network: user-defined bridge `balancer-net`; embedded DNS.
- Volumes:
  - `./logs:/app/logs`
  - `./decisions:/app/decisions`
  - `./balancer:/app/balancer:ro`
  - `./ml:/app/ml:ro`
  - `./models:/app/models:ro`
- Read-only code/model mounts mean the container cannot write back into the source tree. The dashboard HTML is read per request, so editing it shows on reload; Python code changes require a container restart because there is no auto-reload.
- Model artifacts are mounted rather than baked in, so retraining does not require an image rebuild.

### 12.3 Why Docker Compose?

- Reproducible Python/runtime environment; no host pip conflicts.
- CPU/memory quotas via cgroups create genuinely heterogeneous backends.
- Bridge DNS gives stable service names and hides backend ports.
- One command starts the whole testbed.
- Not a production orchestrator; no scheduler, autoscaling, rolling upgrades or HA.

### 12.4 Configuration values

| Setting | Value |
|---|---|
| `algorithm` | `round_robin` by default in config; runtime switch via `/admin/algorithm`. |
| `sla_ms` | 500 ms. |
| `metrics_poll_interval_s` | 1 (collector defaults to 1 s; see caveats). |
| `stale_threshold_s` | 3.0 s. |
| `health_check_interval_s` | 2 (config field; actual collector default is 1 s; see caveats). |
| `health_check_failures_threshold` | 3. |
| `health_check_success_threshold` | 3. |
| `fallback_to_least_conn` | true. |
| `PROXY_TIMEOUT_S` | 30 s. |
| `PROXY_MAX_CONNECTIONS` / keepalive | 500 / 500 / 60 s. |
| Circuit | 3 failures, 10 s cooldown. |
| Slow start | 10 s, 10% -> 100%. |
| Retry | 1. |

### 12.5 Scaling and production gaps

- Add more backends by extending `config.yaml` and Compose; algorithms are O(n) per decision, so a few hundred candidates would need a smarter index/top-k.
- The balancer itself is a single point of failure; production would use multiple balancer replicas behind a VIP/DNS, health-checked, with shared or replicated state.
- For TLS, terminate at the balancer (or a front proxy), manage certificates, add HSTS, and define trusted `X-Forwarded-*` behavior.
- Add authn/authz for admin/dashboard; they are demo-only and unauthenticated.
- Add rate limiting, request size limits, streaming, backpressure and circuit breakers for the proxy itself.
- Add OpenTelemetry traces and metrics export (Prometheus) for production observability.
- Consider Nginx/Envoy/HAProxy + an ML sidecar for better baseline performance, or Rust/C++ for lower overhead; this project prioritizes experimentation and measurement.

---

## 13. Tests, experiments and results

### 13.1 Automated tests

- Current suite: **129 tests** collected; the workspace run recorded 129 passed, with four FastAPI `on_event` deprecation warnings.
- CI: `.github/workflows/python-tests.yml`, Ubuntu 24.04, Python 3.11, `pip install -r requirements-dev.txt`, `python -m pytest -v`.
- Coverage areas:
  - algorithms and fallback behavior
  - router feature layout/tie-breaking
  - drift math and hysteresis
  - logger/decision schemas and join
  - resilience state machine
  - 60 s latency window
  - proxy end-to-end, retry semantics, stale feed
  - admin/demo endpoints
  - model artifact layout/serving units
- Some older docs say 121 tests; the actual current count is 129.

### 13.2 Experiment design

- **Dataset campaign:** 6 profiles x 3 collection policies = 18 runs, 120 s each. Used to train/evaluate the model.
- **Primary matrix:** 5 algorithms (round robin, weighted RR, least connections, least response time, ML herd) x 5 scenarios (steady, ramp, burst, mixed-cost degraded, failure) x 5 repetitions = **125 cells**.
- Per cell: 12 s discarded warm-up + 60 s measured; randomized order, seed 42; one session; 125/125 completed, 0 failed.
- The fault scenario stops backend3 at 40% and restarts it at 70% (per docs); degraded adds 400 ms + 35% errors.
- Unit of analysis is the **run**, not the request: one p95/throughput/error rate per run; five repetitions give uncertainty. Pooling thousands of correlated requests would make everything look significant.
- Statistics: percentile bootstrap 95% CI of mean p95; Welch t-test and Mann-Whitney U on run-level p95; Holm correction across 20 comparisons.
- Metrics: p50/p95/p99, throughput, errors, SLA violations, Jain's index, CPU imbalance, routing shares.
- Why 60 s instead of 5 min: 125 x 6 min is ~12.5 h on one laptop; 60 s still yields 6,000-9,000 requests/cell. Consequence: fewer burst cycles; listed as a validity threat.

### 13.3 Hypothesis verdicts

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1: ML p95 lower than best traditional baseline in >=3/5 scenarios, significant | **Not met** | 0/5 significant wins; burst/degraded favorable but not significant; steady/ramp/failure worse. |
| H2: ML no worse than 5% vs round-robin on steady | **Met, with caveat** | 433 vs 789 ms (0.55x), but least-connections is 214 ms (2.02x better than ML); testbed is heterogeneous, not identical. |
| H3: routing overhead <5 ms mean | **Met on mean** | 1.96 ms mean; max 126.72 ms scheduler/Python outlier. |
| H4: traffic redirect within 5 s after failure | **Met** | worst 1.13 s; 0.00% final 5xx; traffic returns 7.7 s after restart incl. slow start. |

### 13.4 Ablations and generalisation

| Experiment | Result |
|---|---|
| Full features | MAE(log) 0.5868, MAE(ms) 73.1, R^2 0.6827. |
| No trend features (`cpu_delta_5s`, `rt_p95_5s_ms`) | MAE 0.6124 (+4.4%); trend features are less valuable than planned. |
| No request-type features | MAE 0.6955 (+18.5%); request type is the critical group. |
| Held-out burst profile | Model MAE 0.5530 vs naive 1.3677 (+59.6%). |
| Held-out wave profile | Model MAE 0.5521 vs naive 1.3385 (+58.8%). |
| Herd ablation | p95 342 vs 700 ms; 102 vs 73 req/s; 3/3 paired. |

### 13.5 Threats to validity / limitations

- One host/laptop; absolute throughput is not portable; only within-session comparisons are defensible. Session-to-session throughput varies 20-40%.
- 60 s runs and 5 repetitions give wide CIs; one bad run moves the mean. Medians and CIs are reported; runs are not deleted.
- One fault server (backend3) in all fault scenarios.
- Matrix pinned drift threshold at 10.0, so it measures routing policy, not the production drift guard; at 1.0 some cells would demote.
- Live ML error (1.394 steady, 2.421 failure) is much higher than offline 0.5868.
- Failure/steady collapsed runs send only 4-7% to the small backend, losing capacity; that is a real failure mode, not noise to hide.
- No top-1 ranking accuracy because counterfactual performance is unobserved.
- No Nginx/HAProxy measurement in the primary matrix.
- H3 is a mean-cost claim; the 126.72 ms maximum is an unexplained tail.
- No TLS/auth/HA; demo-scale, not production. In particular, `/slow` and `/fault` are unauthenticated and reachable through the balancer catch-all even though backend ports are not published.

### 13.6 The one-paragraph result to memorize

> "Offline, XGBoost predicts response time 53.9% better in MAE than the server's own EWMA, with grouped CV R^2 0.6455 and whole-profile generalisation around 59%. Live, the 125-cell matrix shows least-connections is the strongest overall: lowest p95 in three of five scenarios and highest throughput in all five. ML has lower mean p95 in burst and degraded, but neither difference is statistically significant; it loses steady, ramp and failure. H1 is not met. The queue term is essential (342 vs 700 ms burst p95), H3 passes on the mean (1.96 ms), and H4 passes (1.13 s worst redirect, 0% final 5xx). The honest conclusion is that the prediction problem is solved better than the routing problem on this testbed."

---

## 14. Five-minute live demo script

### 14.1 Before the demo (2 minutes)

```powershell
docker compose up -d
curl.exe -s http://localhost:9090/stats
```

Check:

- 4 servers, 4 healthy.
- `feed.age_ms` well under 3000.
- `ml.loaded: true` (if using ML).
- `feed.collector.enabled: true`.
- No leftover fault on backend3.

Open `http://localhost:9090/`. Keep `docs/results.md` open in a second tab. Click **Reset to ML** once: sets algorithm `ml`, drift threshold `1.0`, clears backend3 fault.

The browser tab must stay active while its demo driver runs (about 24 req/s). It is a demo aid, not the benchmark client.

### 14.2 Timeline

**0:00-0:30 - What this is.**
Say the 60-second answer. Point at:
- four backend cards with different CPU quotas
- routable vs health probes
- feed age
- p95 chart
- decision table

**0:30-1:15 - Algorithms.**
Click **Start traffic**. Then click the `round_robin`, `least_conn`, `ml` pills. Watch the decision `reason` change:
- `round_robin`
- `least_conn`
- `ml(pred 42.1ms, inflight 3)`

Say: same traffic, same backends, only `choose_server()` changed.

**1:15-2:00 - Decision log.**
Point at the recent decisions table: server, predicted vs actual, status, retry, model-driven. Say it is a separate 14-column Parquet joined on `(run_id, timestamp)`; the 16-column training schema is untouched.

**2:00-3:00 - Degraded backend.**
Click **Degrade 45 s**. backend3 gets +400 ms and 35% errors. Expected:
- error rate rises
- three consecutive failures can open circuit
- traffic share moves away
- client errors stay near zero because of retry
- dashboard shows "probe up - not routable" if breaker opens

Do not claim fleet errors are zero: mean errors are diluted (~35%/4), and the request log records retried requests as final 200. For a deterministic circuit-open demo, click **Fail 12 s**; every backend3 request returns 503 but `/metrics` stays alive.

**3:00-3:50 - Drift.**
Click **Stress (0.15)**. After enough decisions, ML demotes; reasons become `drift-fallback(least_conn, ...)`, `drift_routed` rises. Click **Restore (1.0)**; after a clean window, it promotes back. Say the model keeps shadow-scoring while demoted, so it can recover automatically.

**3:50-4:30 - Results.**
Click **Stop + seal**. Show run summary: sealed row counts, p95 start/end, routing distribution. Then show `docs/results.md`:
- M5: +53.9% MAE, R^2 0.683, CV 0.6455.
- 125 cells; ML lower mean p95 in burst/degraded, not significant; loses steady/ramp/failure; least-connections strongest overall; H1 not met.
- Herd ablation 342 vs 700 ms.
- H3/H4 pass.

**4:30-5:00 - Honest part.**
Say the model is not uniformly better. Live MAE is 1.4 steady and 2.4 failure vs 0.5868 offline. The collapsed runs avoid the small backend and lose capacity. The matrix pinned the guard at 10 to measure routing; at 1.0 it would demote. Close with the engineering contributions: queue term, decision log, drift guard, resilience layer, reproducible measurement.

### 14.3 If the dashboard fails

```powershell
# Reset
curl.exe -X POST "http://localhost:9090/admin/algorithm?name=ml"
curl.exe -X POST "http://localhost:9090/admin/drift?threshold=1.0"
curl.exe -X POST "http://localhost:9090/admin/demo/clear"

# Start a run and traffic
curl.exe -X POST "http://localhost:9090/admin/run?run_id=demo"
python bench/demo_traffic.py --seconds 330 --rate 24

# Degrade / fail / clear
curl.exe -X POST "http://localhost:9090/admin/demo/degrade?duration_s=45"
curl.exe -X POST "http://localhost:9090/admin/demo/fail?duration_s=12"
curl.exe -X POST "http://localhost:9090/admin/demo/clear"

# Seal
curl.exe -X POST "http://localhost:9090/admin/seal"
```

If live traffic misbehaves, fall back to `docs/results.md` and `docs/resilience.md` and say so; do not pretend a live run happened.

### 14.4 Troubleshooting quick reference

| Symptom | Likely cause | What to do |
|---|---|---|
| Dashboard/`/stats` unreachable | Balancer stopped, wrong port, host 9090 occupied | `docker compose ps`; `docker compose logs -f balancer`; check the port mapping. |
| `/stats` shows 0 healthy | Backends down/starting, collector failing, network/DNS problem | `docker compose ps`; `docker compose logs backend1`; test `docker compose exec balancer python -c "import httpx; print(httpx.get('http://backend1:8000/metrics').status_code)"`. |
| 503 `No healthy servers available` | All circuits open, all collector health states down, or stale feed with no known healthy server | Inspect `/stats`; clear faults; wait for health/slow start; check collector. |
| 502 from the proxy | Backend connection error/timeout or upstream protocol failure | Check backend container/logs; confirm internal port 8000; check `x-proxy-error` header/detail. |
| ML reason says `ml-fallback` | No artifact, load error, feature mismatch, or prediction exception | Check `/stats.ml.load_error`; verify `models/` mount and feature columns; re-select `ml` after fixing. |
| ML not loaded (`ml.loaded=false`) | Artifact missing/corrupt or sklearn/joblib mismatch | Check balancer logs; verify `models/xgboost_*.joblib`; fallback still routes via least-connections. |
| Feed is stale | Collector paused or failing | `POST /admin/collector?enabled=true`; check backend `/metrics`; inspect `feed.collector`. |
| Logger rows remain 0 | No run started, or rows outside a run are dropped | `POST /admin/run?run_id=...`; inspect `/admin/logger`; seal when done. |
| Algorithm switch reverts after restart | Runtime switch is in-memory; config still says `round_robin` | Edit `balancer/config.yaml` and restart if it must persist. |
| New model not used | Balancer loaded the old artifact at startup | Re-select `ml` (reloads artifact) or restart the balancer. |
| High latency / p95 rising | Queueing, CPU saturation, pool wait, or expected stress | Check inflight, CPU/quota, feed age, `/stats.latency`, pool limits; reduce users. |
| Tests fail | Environment/dependency/version mismatch | `python -m pytest -q`; install `requirements-dev.txt`; Python 3.11. |
| Port 9090 conflict | Another process has the port | Stop it or change `9090:8080` in Compose. |

---

## 15. Additional Q&A bank

### 15.1 CN fundamentals

**Q: What is a load balancer?**  
A device/software that distributes client requests across multiple servers to improve availability, scalability, latency and utilization, and to avoid a single server becoming a bottleneck.

**Q: Which OSI layer is your balancer?**  
Layer 7, application. It parses HTTP requests and responses. It still uses TCP at Layer 4 underneath.

**Q: What is the difference between a router and a load balancer?**  
A router forwards packets between IP networks using L3 routing tables/protocols. A load balancer selects an application server for a request; this one is an L7 reverse proxy, not a network router.

**Q: What is a switch?**  
A classic L2 device that forwards Ethernet frames within a LAN using MAC addresses. Docker's bridge behaves like a virtual L2 switch. A multilayer switch can also route at L3.

**Q: What is NAT?**  
Network Address Translation rewrites IP/port addresses. Docker's published port 9090 -> container 8080 uses host port mapping (NAT on Linux; Docker Desktop's forwarding on Windows/macOS). The balancer itself is a separate L7 proxy, not just NAT.

**Q: What is DNS?**  
A distributed naming system that maps names to IP addresses. Docker's embedded DNS resolves service names; client-facing DNS load balancing is not used here.

**Q: What is a port?**  
A 16-bit transport-layer identifier for a service on a host. 9090 is the host-published balancer port; 8080 is the container's Uvicorn port; 8000 is the backend port.

**Q: TCP vs UDP?**  
TCP is connection-oriented, reliable, ordered and congestion-controlled; HTTP/1.1 uses it. UDP is connectionless and best-effort, used by DNS queries and real-time traffic. This project uses TCP.

**Q: What is a TCP handshake?**  
SYN -> SYN-ACK -> ACK. It establishes sequence numbers and connection state. Keep-alive/pooling avoids repeating it per request.

**Q: What is a socket?**  
An endpoint identified by IP address, port and protocol; a connection is commonly identified by a 4-tuple (source IP/port, destination IP/port).

**Q: What is a reverse proxy?**  
A server that fronts backend servers and forwards client requests to them, hiding backend topology and often adding TLS, caching, compression, retries and logging.

**Q: Forward vs reverse proxy?**  
Forward proxy represents clients; reverse proxy represents servers. A load balancer is normally a reverse proxy.

**Q: What is active vs passive health checking?**  
Active: periodically send a probe (`/metrics` here) and mark down after failures. Passive: infer health from real request outcomes (circuit breaker here). This project uses both.

**Q: Why not just ping the backend?**  
ICMP ping checks host reachability, not whether the HTTP application can serve requests. An HTTP probe checks the app-level path.

**Q: What is keep-alive?**  
Persistent TCP connection reuse across multiple HTTP requests. It reduces handshake and slow-start overhead. The balancer's httpx pool reuses upstream connections.

**Q: What is a connection pool?**  
A set of reusable connections with limits. Here max 500 connections/keepalive, 60 s expiry. It prevents unbounded connection creation and controls concurrency.

**Q: What is throughput vs latency?**  
Throughput is requests per second; latency is time per request. A system can have high throughput and high latency if requests queue. The stress test showed 123 req/s but p95 902 ms.

**Q: What is p95?**  
The 95th percentile of latency: 95% of observed successful requests are at or below it. It captures tail latency better than the mean.

**Q: What is an SLA?**  
A service-level agreement/target. Here `sla_ms=500`; the project measures the percentage of successful requests above 500 ms rather than enforcing a hard filter.

**Q: Why is HTTP stateless?**  
Each request is independent; the server does not need to remember previous requests. This project's endpoints are stateless, so no sticky sessions are needed. Stateful apps would need affinity or shared state.

**Q: What is idempotency?**  
An operation that can be repeated without changing the final result (e.g., GET). Retrying a non-idempotent POST can duplicate work; the project warns about this.

### 15.2 Algorithms and architecture

**Q: Why does round-robin fail on heterogeneous servers?**  
It gives equal request counts, not equal work. A 0.15-core backend gets the same share as a 0.8-core backend and saturates.

**Q: Why is least-connections strong?**  
It directly reacts to the current queue. It is cheap, robust and does not depend on a model's prediction accuracy. On this testbed it has the best throughput and p95 in most scenarios.

**Q: What is the weakness of least-connections?**  
It ignores request cost and server capacity; one heavy request and one light request both count as one connection. Weighted least-connections addresses capacity, but still not request cost.

**Q: Why not always use weighted least-connections?**  
It needs accurate weights; if the quotas or workload change, the weights can be wrong. The project implements it but does not include it in the primary matrix.

**Q: How does the ML router choose in one sentence?**  
Predict each healthy backend's response time for this request, multiply by `(1 + inflight)` to price its queue, and choose the minimum expected wait.

**Q: Why batch predictions?**  
One matrix and one estimator call for all candidates is faster and simpler than one call per server, and it keeps the decision within the 5 ms mean target.

**Q: Why remove Host and Content-Length?**  
The upstream should see its own host, and httpx should recompute Content-Length after the body is read. Forwarding the original Host could confuse virtual-host routing; forwarding the original length can mismatch the body.

**Q: Why read the body before routing?**  
`payload_kb` is a model feature known only from the body (or Content-Length). Reading it after selection would make the feature always zero or stale for POST payload traffic.

**Q: Why is there no X-Forwarded-For?**  
It is not implemented. In production, add it only from trusted proxies and define how downstream services use it. Without it, backends see the proxy/container source, not the original client.

**Q: Why no TLS?**  
The demo is local and plain HTTP. Production would terminate TLS at the balancer or a front proxy, use certificates and HSTS, and possibly re-encrypt upstream.

**Q: Why is the balancer a single point of failure?**  
There is one balancer process/container. If it dies, clients lose the entry point. Production would run multiple replicas behind a VIP/DNS and health-check them.

**Q: Where does Docker's service discovery fit?**  
Docker's embedded DNS maps service names to container IPs on `balancer-net`. It is not a client-facing GSLB; it simply lets the balancer use stable names.

### 15.3 ML and statistics

**Q: Why not a neural network?**  
Tabular data with strong interactions; gradient-boosted trees are a strong baseline, train in under a minute, and predict in milliseconds. A neural net adds tuning/serving cost without a measured gain for this problem.

**Q: Why log1p instead of raw milliseconds?**  
Latency is heavy-tailed. Log space reduces the influence of a few multi-second outliers and makes relative error the target; predictions are mapped back with expm1.

**Q: Why group by run?**  
Requests within a run share server state that evolves over seconds. A random row split would put near-duplicate states in train and test, inflating scores. Grouping by run prevents leakage.

**Q: What is the signal gate?**  
A dataset acceptance test: server-state-only features must achieve R^2 >= 0.30 on held-out runs. It prevents a dataset where the endpoint identity, not load, explains latency. The actual value was 0.5499.

**Q: Why exclude failed requests?**  
A 503's latency is not a service-time observation; including it teaches the model that failing servers are fast. Error rate is still a feature.

**Q: Why is top-1 accuracy missing?**  
The log only observes the chosen server's latency, not the counterfactual latency on the other servers. Spearman correlation is reported instead; active replay would be needed for top-1.

**Q: What is MAE?**  
Mean absolute error: average `|prediction - actual|`. In log space it treats relative errors more evenly; in ms it is interpretable but tail-sensitive.

**Q: What is R^2?**  
Coefficient of determination: how much variance the model explains relative to predicting the mean. It can be negative if the model is worse than the mean.

**Q: What is Spearman correlation?**  
Rank correlation between predicted and actual latency; useful when the goal is ordering requests/servers rather than exact milliseconds.

**Q: What does the drift guard do?**  
Tracks rolling log-MAE on successful first attempts. If it exceeds 1.0 after at least 50 samples, demote to least-connections. Keep shadow-predicting, clear the window, and promote after a full 200-sample window below 0.8.

**Q: Why keep predicting while demoted?**  
Otherwise the guard would have no fresh evidence and could only be re-enabled manually. Shadow predictions let it promote itself.

**Q: Why is the live MAE higher than offline?**  
Live traffic differs from the training campaign (lighter `/ping` mix, different concurrency, fault transients, stale snapshots). The matrix recorded 1.394 steady and 2.421 failure vs 0.5868 offline. The guard and the negative result are reported.

**Q: Why five repetitions?**  
The plan specifies five; the run is the independent unit. With n=5, only a clean separation reaches significance, and CIs are wide. The project reports both tests and CIs rather than hiding uncertainty.

**Q: What is Holm correction?**  
A multiple-comparison correction: sort p-values and apply progressively stricter thresholds so the chance of false positives across 20 comparisons is controlled. It is why some nominal p-values are no longer significant.

### 15.4 Resilience and operations

**Q: Circuit breaker vs retry?**  
Retry handles one failed request immediately; the breaker remembers repeated failures and stops sending new work to that server for a cooldown. They complement each other.

**Q: What is a half-open circuit?**  
After the cooldown, allow a probe/traffic to test recovery. Success closes the circuit and starts slow start; failure reopens it. This implementation uses a time-based eligible window rather than a strict single-probe semaphore.

**Q: Why slow start?**  
A recovered backend may have cold caches, empty pools and a cold JIT/process. Sending full traffic immediately can overload it. The ramp is 10% to 100% over 10 s, stochastic per decision.

**Q: What is stale-feed fallback?**  
If all metrics are older than 3 s, stop trusting the model/algorithm inputs; use last-known healthy servers with least-loaded selection. It turns a monitoring outage into a degraded but working system, not a total outage.

**Q: What if the metrics feed dies and all servers are also dead?**  
The fallback only considers last-known healthy servers; if none, it returns 503. It does not blindly route to dead servers.

**Q: Why does the dashboard show "probe up - not routable"?**  
The process can answer `/metrics` while its circuit is open due to user-facing 5xx. Health probe success and routability are different states.

**Q: What is the 503 vs 502 distinction again?**  
502: upstream/gateway failure (bad response/connection). 503: service unavailable, often no healthy server. The code maps upstream httpx errors to 502 and no-candidate to 503.

**Q: How do you demonstrate failover?**  
Use **Fail 12 s** on backend3; the process stays alive, `/metrics` stays green, but the breaker opens and traffic moves; retries keep client-visible errors near zero. For a real container kill, use `docker compose stop backend3`; the collector eventually marks it unhealthy and H4 evidence covers it.

**Q: How do you demonstrate stale feed?**  
`POST /admin/collector?enabled=false`; after 3 s, `/stats.feed.stale=true` and reasons become `stale-feed(least_conn)`. Resume with `enabled=true`.

**Q: How do you demonstrate drift?**  
Lower threshold to 0.15; after enough decisions the guard demotes; reasons become `drift-fallback`; restore to 1.0 and wait for promotion.

### 15.5 Project and results

**Q: What is novel here?**  
No new algorithm. The contribution is an honest integration and measurement: a learned router behind the same interface as traditional policies, queue-aware scoring, decision logging, drift guard with shadow scoring, and a benchmark that reports where ML loses.

**Q: Why not Nginx/HAProxy?**  
They are mature L7 proxies and good baselines; reference configs exist in `deploy/`. The custom Python proxy is needed to integrate per-request ML, detailed decisions and run-level logs. The primary 125-cell matrix does not benchmark Nginx/HAProxy; that is future work.

**Q: Why Python if it is slower?**  
Async I/O and a small model keep the mean decision at ~2 ms; the goal is experimentation and observability, not maximum throughput. A production high-throughput version could use C++/Rust/Envoy or a sidecar.

**Q: What is the biggest weakness?**  
The live model error and capacity-avoidance failure mode; the single host; 60 s runs; no HA/TLS/auth; and the fact that least-connections is a very strong baseline. The project reports these rather than hiding them.

**Q: What would you do next?**  
Retrain with fault transients and an explicit capacity term; test the drift guard at the shipped threshold; measure Nginx/HAProxy through the same harness; add TLS/auth/HA; add tracing; consider a shorter prediction horizon/queue-wait target.

---

## 16. Evidence map: what to show when the professor asks

| If asked about | Show | Say |
|---|---|---|
| Architecture | `docs/report.md` diagram, `docker compose ps`, dashboard | Five services; balancer + four backends on a bridge network. |
| Request flow | Dashboard "Live request flow" + `/stats` | It is a sampled animation, not a packet trace; new work is rerouted, never migrated. |
| Algorithms | Dashboard pills; `balancer/algorithms.py` | Same `choose_server()` interface; only the decision function changes. |
| ML model | `/stats.ml`; `models/training_report_032d9bab9a.json`; `docs/dataset.md` | XGBoost, 14 features, log target, grouped split, 53.9% MAE improvement. |
| Live results | `docs/results.md` p95/throughput tables | 125 cells; least-connections strongest; ML burst/degraded means not significant; H1 not met. |
| Herd term | `docs/results.md` herd ablation; `/stats.decisions` | 342 vs 700 ms p95; 102 vs 73 req/s; 3/3 paired. |
| Resilience | `/stats.resilience`, `docs/resilience.md` | Breaker 3/10 s, retry 1, slow start 10 s, stale feed 3 s. |
| Health vs routable | Dashboard "routable" tile and backend badges | Probe up can be circuit-open; they are different states. |
| Logs | `/admin/logger`, `logs/`, `decisions/` | 16-column request log + 14-column decision log; join on `(run_id, timestamp)`. |
| Docker network | `docker network inspect ai_ml_load_balancer_balancer-net` | Embedded DNS; backend ports not host-published; host 9090 -> 8080; `/slow` and `/fault` still reachable via the catch-all. |
| Tests | `python -m pytest -q` | 129 tests (not 121); CI on Python 3.11. |
| Drift guard | Dashboard ML panel; `/admin/drift`; `docs/monitoring.md` | Rolling log-MAE; demote/promote; shadow scoring. |
| Demo | `docs/demo.md`; dashboard Demo controls | Five-minute path; faults and threshold changes are buttons. |
| Honest limitations | `docs/report.md` sections 11-12 | One host, 60 s, n=5, live error, capacity avoidance, no HA/TLS. |

### 16.1 If the professor opens the code, walk this order

1. `balancer/main.py` -> `proxy()`: classify path, read body, build context, retry loop, call `resilience.choose()`.
2. `balancer/resilience.py` -> `choose()`: circuit breaker, slow start, stale-feed fallback, server selection.
3. `balancer/algorithms.py` -> `get_algorithm()` and one `choose_server()` implementation: show that all policies share the interface.
4. `balancer/ml_router.py` -> `choose_server()`: build matrix, predict, `expm1`, queue score, tie-break, drift fallback.
5. `ml/features.py` -> `FEATURE_COLUMNS`, `build_serving_array()`: the exact 14-feature contract shared by training and serving.
6. `balancer/collector.py` -> `_poll_server()`: how `/metrics` becomes health + state.
7. `balancer/drift.py` -> `observe()`: log-MAE window, demotion, shadow prediction, promotion.
8. `balancer/logger.py` + `decision_log.py`: 16-column request log and 14-column decision log; join on `(run_id, timestamp)`.
9. `backend/main.py`: `/ping`, `/cpu`, `/io`, `/payload`, `/metrics`, fault injection.
10. `docker-compose.yml`: bridge network, quotas, port mapping, read-only mounts.

If asked "where is the main logic?", answer: **`balancer/main.py::proxy` for the data path, `balancer/resilience.py::choose` for safety, and `balancer/ml_router.py::choose_server` for the ML decision.**

---

## 17. Numbers and formulas cheat sheet

### 17.1 Values

| Item | Value |
|---|---|
| Backends | 4: 0.8, 0.4, 0.4, 0.15 CPU |
| Config weights | 2.0, 1.0, 1.0, 0.5 (4:2:2:1 design) |
| Host/container ports | 9090 -> 8080; backend 8000 |
| Collector poll | ~1 s |
| Stale threshold | 3 s |
| Health failures/successes | 3 / 3 |
| Circuit | 3 failures / 10 s |
| Slow start | 10 s, 10% -> 100% |
| Retry | 1, on 500/502/503/504 |
| Proxy timeout | 30 s |
| Pool | 500 connections / 500 keepalive / 60 s |
| SLA | 500 ms |
| ML features | 14 |
| ML target | `log1p(response_ms)`, cap 5000 ms |
| Model | XGBoost 400 trees, depth 6, lr 0.06 |
| Offline MAE(log) | 0.5868 |
| Offline MAE(ms) | 73.1 |
| Offline R^2 | 0.6827 |
| CV R^2 | 0.6455 |
| MAE improvement | 53.9% |
| Dataset | 218,011 rows / 18 runs |
| Filtered rows | 178,165 |
| Train rows | 121,061 |
| Matrix | 125 cells, 0 failed |
| Tests | 129 |
| Decision mean/max | 1.96 ms / 126.72 ms |
| H4 | 1.13 s worst / 0.00% final 5xx / 7.7 s return |

### 17.2 Formulas

```text
Round robin:              i = (i + 1) mod N
Weighted RR:              max(1, int(weight)) copies per server
Least connections:        argmin_s inflight(s)
Weighted least conn:      argmin_s inflight(s) / weight(s)
Least response time:      argmin_s rt_ewma_ms(s)
ML score (shipped):       predicted_ms(s) * (1 + inflight(s))
ML score (ablation):      predicted_ms(s)
ML tie:                   score <= best + 5.0 ms -> least inflight, then prediction
Target:                   y = log1p(response_ms)
Back to ms:               predicted_ms = clip(expm1(y), 0, 5000)
Drift error:               |log1p(actual) - log1p(predicted)|
Drift MAE:                 mean(last N errors)
Percentile (nearest rank): sorted[ceil(q * n) - 1]
Jain:                      (sum x_i)^2 / (n * sum x_i^2)
Little's Law:              L = lambda * W
SLA violation:             successful requests with response_ms > 500 / all successful
```

### 17.3 Glossary

| Term | Meaning |
|---|---|
| Reverse proxy | Server-side proxy that fronts upstream servers. |
| Upstream | A backend server behind the proxy. |
| L4/L7 | Transport layer vs application layer; this is L7 HTTP. |
| Health check | Probe that decides whether a server should receive traffic. |
| Circuit breaker | Stops sending work to a repeatedly failing server for a cooldown. |
| Slow start | Gradual traffic ramp for a recovered server. |
| Stale feed | Metrics snapshot older than the threshold. |
| Herding | Many requests choosing the same server based on a stale snapshot. |
| EWMA | Exponentially weighted moving average. |
| p95 | 95th percentile latency. |
| SLA | Service-level agreement/target. |
| MAE | Mean absolute error. |
| R^2 | Fraction of variance explained relative to the mean. |
| Spearman | Rank correlation. |
| Grouped split | Split by run so related rows do not leak across train/test. |
| Log1p | `log(1+x)`, used to compress heavy-tailed latency. |
| Artifact | Serialized trained model (`.joblib`). |
| Drift | Live model error exceeding a threshold. |
| cgroup | Linux control group used for CPU/memory quotas. |
| Bridge network | Virtual Docker L2 network with embedded DNS. |
| ASGI | Asynchronous server gateway interface used by FastAPI/Uvicorn. |
| httpx | Async Python HTTP client used for proxying and polling. |
| Parquet | Columnar file format used for request/decision logs. |
| Locust | Load-testing tool used to generate traffic. |

---

## 18. Known inconsistencies and tricky points (do not get caught)

This section is important because older docs and comments disagree with the current code in a few places.

| Topic | Older docs/comment | Current code / correct viva answer |
|---|---|---|
| Test count | `121 tests` in `README.md`, `docs/report.md`, `docs/traceability.md` | Current suite has **129 tests**. Some docs are stale. |
| ML tie threshold | Some docs say **1 ms** | Code uses `TIE_MS = 5.0` ms. Quote **5 ms**. |
| Collector interval | Some docs say **2 s** / "2-second-old snapshot" | `MetricsCollector` default and `config.yaml` intent are **1 s**; `/stats.feed.collector.poll_interval_s` is the live truth. The queue term still matters because requests arrive faster than polls. |
| Health check interval | Config has `health_check_interval_s: 2`; Dockerfile comment says `/health` every 2 s | Actual collector polls `/metrics` with its default **1 s**, and health is derived from that poll. The `/health` endpoint exists but is not the collector's health feed. |
| Weighted RR weights | Config says 4:2:2:1 (2/1/1/0.5) | Code uses `max(1, int(weight))`: 0.5 becomes 1, so WRR's effective slots are **2:1:1:1**. Weighted least-connections uses exact floats. |
| Algorithms in primary matrix | Docs list five | Matrix uses round-robin, weighted RR, least-connections, least-response-time, ML. `random` and `weighted_least_conn` are implemented but not in the 125-cell primary comparison. |
| Drift threshold in matrix | Docs explain pinned at 10 | The 125-cell matrix pinned the guard at **10.0** to measure routing policy, not the production guard. At shipped **1.0**, some live windows would demote. This is a stated experiment-design choice and limitation. |
| H1 | "AI/ML wins" assumption | **H1 is not met.** ML has 0/5 significant wins; lowest mean p95 only burst/degraded, not significant; loses steady/ramp/failure. |
| H1 model name | `docs/hypotheses.md` says "ML-Predictive" | The matrix treats the shipped `ml` policy (which includes herd protection) as the final ML router; pure `ml_predictive` is a separate no-queue ablation. The H1 verdict applies to the matrix ML row. |
| `ml` vs `ml_herd` | Sometimes presented as separate | They are **the same class/code path**. `ml` is the stable name; `ml_herd` is the plan's name. `ml_predictive` is the no-queue ablation. |
| Dataset rows | 218,011 vs 178,165 vs 121,061 | 218,011 collected; 178,165 after warm-up/failure filtering; 121,061 in the training split. Use the right number for the context. |
| Agent | `agent/` exists | It is not a separate default Compose service; the deployed backend exposes `/metrics` itself. |
| Docker-compose CPU quotas vs config weights | 0.8/0.4/0.4/0.15 vs 2/1/1/0.5 | Actual Docker quotas are scaled to the host; config values preserve the design ratio and are used as stable model features. See 6.4. |
| "Health checks every 2 s take 6 s" | Some prose | Code's actual collector poll is 1 s; 3 failures is roughly 3 s (plus scheduling). The breaker is faster (~1.1 s). State the actual behavior. |
| Backend controls "private" | Balancer code comment says only the balancer can reach `/slow` and `/fault` because they are not host-published | Not host-published, but the balancer's catch-all forwards those paths from `localhost:9090`; there is no auth. Treat them as demo-only and block them in production. |
| No TLS | Not a bug | Demo is plain HTTP. Say "not implemented; production would add TLS termination, certificates and forwarded headers." |
| No `X-Forwarded-For` | Not implemented | Backends see the proxy/container source. Add only with a trusted-proxy policy in production. |
| "All 7 algorithms compared" | Not true | The primary matrix has 5 policies. Additional algorithms are implemented for experimentation/demo. |
| "121 tests all green" | Stale | 129 tests; 4 FastAPI `on_event` warnings. |
| `on_event` warnings | FastAPI deprecation | Tests pass; migration to lifespan is future cleanup. |

**How to answer if the professor spots a discrepancy:**  
> "That number in the older prose is stale. The current code uses 5 ms for ties / a 1 s collector / 129 tests. The report documents the discrepancy and the code is the source of truth."

---

## 19. Last-minute checklist

### 19.1 Before entering the room

- [ ] Stack up: `docker compose up -d`; `docker compose ps` shows 5 running.
- [ ] `curl.exe -s http://localhost:9090/stats` shows 4 servers, 4 healthy, feed age < 3 s.
- [ ] Dashboard open at `http://localhost:9090/`.
- [ ] `docs/results.md` open in a second tab.
- [ ] Algorithm reset to `ml`; drift threshold 1.0; backend3 fault cleared.
- [ ] Know how to switch to `least_conn` and back.
- [ ] Know the five headline numbers: 53.9%, 0.5868, 125 cells, H1 not met, 129 tests.

### 19.2 Three sentences to never forget

1. **"It is an L7 HTTP reverse proxy, not an L4 router."**
2. **"The ML model predicts response time; the queue term makes the decision; the drift guard and breaker keep it safe."**
3. **"Least-connections is the strongest overall on this testbed; ML is conditionally useful, and H1 is not met. We report that honestly."**

### 19.3 If you forget everything else

- Architecture: client -> host 9090 -> balancer 8080 -> backend 8000.
- Selection: `choose_server()` -> algorithm/ML -> httpx -> backend.
- ML: 14 features -> XGBoost -> `log1p` prediction -> expm1 -> `pred * (1+inflight)` -> minimum.
- Safety: 3 failures -> breaker 10 s; 3 failed polls -> unhealthy; 3 s stale -> least-conn fallback; one retry; 10 s slow start.
- Results: offline +53.9% MAE; live least-connections wins; H1 not met; H2-H4 pass (H2 only against round-robin; H3 on the mean).

---

*End of guide. The code in `balancer/`, `ml/`, `backend/`, and `docker-compose.yml` is the source of truth when a number or behavior is ambiguous.*

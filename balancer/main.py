"""
AI/ML Load Balancer - Main entry point.

Phase 2 Implementation:
- Reverse proxy with pluggable routing
- Live metrics collection with MetricsCollector
- Parquet request logger for data generation
- Enhanced /stats endpoint
"""

import asyncio
import collections
import math
import os
import sys
import time
import uuid
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse

from balancer.algorithms import get_algorithm, algorithm_names, Algorithm
from balancer.interfaces import RequestContext, ServerChoice, ServerInfo, RequestType
from balancer.collector import MetricsCollector
from balancer.decision_log import make_decision_logger
from balancer.logger import RequestLogger
from balancer import resilience

CONFIG_PATH = os.environ.get("BALANCER_CONFIG", "balancer/config.yaml")
DASHBOARD_PATH = Path(__file__).with_name("dashboard.html")
ALGORITHM: Optional[str] = None
SERVERS: List[ServerInfo] = []
SLA_MS = 500
client: Optional[httpx.AsyncClient] = None
algorithm_instance: Optional[Algorithm] = None
collector: Optional[MetricsCollector] = None
req_logger: Optional[RequestLogger] = None
decision_logger: Optional[RequestLogger] = None

# Recent routing decisions, newest last. The decision log file follows the run
# lifecycle and only exists inside a run; this ring is always on, so the
# dashboard can show predicted-versus-actual without a run being started.
RECENT_DECISIONS: "collections.deque[Dict[str, Any]]" = collections.deque(maxlen=100)

# Successful requests' user-visible latency, for the dashboard's p95-over-time
# panel. A time window, not a request count: 60 s at the traffic rates this
# host can produce needs ~9,000 entries, so the cap is deliberately generous.
# Percentiles are nearest-rank over the window and are computed once per
# /stats poll, not per request.
RECENT_LATENCIES: "collections.deque[tuple[float, float]]" = collections.deque(maxlen=20000)
LATENCY_WINDOW_S = 60.0


def latency_window(now: float) -> Dict[str, Any]:
    """p50/p95/p99 of successful requests in the last LATENCY_WINDOW_S."""
    cutoff = now - LATENCY_WINDOW_S
    values = sorted(ms for ts, ms in RECENT_LATENCIES if ts >= cutoff)
    if not values:
        return {"window_s": LATENCY_WINDOW_S, "samples": 0,
                "p50": None, "p95": None, "p99": None}

    def at(q: float) -> float:
        # Nearest-rank: ceil(q * n) is the 1-based rank of the quantile.
        index = max(0, min(len(values) - 1, math.ceil(q * len(values)) - 1))
        return round(values[index], 1)

    return {
        "window_s": LATENCY_WINDOW_S,
        "samples": len(values),
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
    }

# Resilience counters, surfaced through /stats. They count routing decisions and
# retries rather than requests: a stale-feed decision can also happen on the
# single retry of a failed attempt.
STATS = {"retries": 0, "stale_routing": 0}


def load_config():
    """Load configuration from YAML file."""
    global ALGORITHM, SERVERS, SLA_MS, collector, req_logger, decision_logger
    
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {
            "algorithm": "round_robin",
            "servers": [
                {"server_id": "backend1", "host": "backend1", "port": 8000, "weight": 2.0, "cpu_limit": 2.0, "mem_limit_mb": 512},
                {"server_id": "backend2", "host": "backend2", "port": 8000, "weight": 1.0, "cpu_limit": 1.0, "mem_limit_mb": 256},
                {"server_id": "backend3", "host": "backend3", "port": 8000, "weight": 1.0, "cpu_limit": 1.0, "mem_limit_mb": 512},
                {"server_id": "backend4", "host": "backend4", "port": 8000, "weight": 0.5, "cpu_limit": 0.5, "mem_limit_mb": 128},
            ],
            "sla_ms": 500,
            "run_id": "run_0"
        }
    
    ALGORITHM = config.get("algorithm", "round_robin")
    SLA_MS = config.get("sla_ms", 500)
    
    SERVERS = []
    for s in config.get("servers", []):
        server = ServerInfo(
            server_id=s.get("server_id", s.get("id", "unknown")),
            host=s["host"],
            port=s["port"],
            weight=s.get("weight", 1.0),
            cpu_limit=s.get("cpu_limit", 1.0),
            mem_limit_mb=s.get("mem_limit_mb", 512),
            metrics=None,
        )
        SERVERS.append(server)
        
    collector = MetricsCollector(SERVERS)
    run_id = config.get("run_id", f"run_{int(time.time())}")
    req_logger = RequestLogger(run_id=run_id)
    decision_logger = make_decision_logger(run_id=run_id)


async def init():
    """Initialize the balancer components."""
    global client, algorithm_instance
    load_config()
    algorithm_instance = get_algorithm(ALGORITHM)

    # Pool sizing matters here: httpx defaults to 100 connections shared
    # across all backends. Anything above that queues inside client.request(),
    # and the proxy then attributes that queueing to the backend's latency.
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(float(os.environ.get("PROXY_TIMEOUT_S", "30"))),
        limits=httpx.Limits(
            max_connections=int(os.environ.get("PROXY_MAX_CONNECTIONS", "500")),
            max_keepalive_connections=int(os.environ.get("PROXY_MAX_KEEPALIVE", "500")),
            keepalive_expiry=float(os.environ.get("PROXY_KEEPALIVE_EXPIRY_S", "60")),
        ),
    )
    
    if collector:
        await collector.start()
    if req_logger:
        await req_logger.start()
    if decision_logger:
        await decision_logger.start()
        
    print(f"Loaded balancer config: algorithm={ALGORITHM}, servers={len(SERVERS)}")


async def close():
    """Clean up resources."""
    global client, collector, req_logger, decision_logger
    if collector:
        await collector.stop()
    if req_logger:
        await req_logger.stop()
    if decision_logger:
        await decision_logger.stop()
    if client:
        await client.aclose()


app = FastAPI(title="AI/ML Load Balancer")


@app.on_event("startup")
async def startup():
    await init()


@app.on_event("shutdown")
async def shutdown():
    await close()


@app.post("/admin/algorithm")
async def set_algorithm(name: str):
    """Update the routing algorithm at runtime."""
    global algorithm_instance, ALGORITHM
    try:
        algorithm_instance = get_algorithm(name)
        ALGORITHM = name
        return {"status": "ok", "algorithm": name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/admin/collector")
async def set_collector(enabled: bool = True):
    """
    Pause or resume metrics polling.

    Pausing is how the stale-feed fallback is demonstrated: once the newest
    snapshot is older than STALE_AFTER_S, routing switches to least-connections
    over the last known state instead of failing every request.
    """
    if collector is None:
        raise HTTPException(status_code=503, detail="Collector not initialised")
    collector.set_enabled(enabled)
    return {"status": "ok", "collector": collector.status()}


@app.post("/admin/run")
async def start_run(run_id: str):
    """
    Seal the current Parquet file and begin a new run.

    Call this before each traffic run so every file is complete (footer
    written), independently readable, and tagged with its own run_id. The
    decision log rotates with it, so a run's decisions and its requests can be
    joined on (run_id, timestamp).
    """
    if req_logger is None or decision_logger is None:
        raise HTTPException(status_code=503, detail="Logger not initialised")
    if not run_id or "/" in run_id or "\\" in run_id:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    info = await req_logger.start_run(run_id)
    decisions = await decision_logger.start_run(run_id)
    return {"status": "ok", **info, "decisions": decisions}


@app.post("/admin/seal")
async def seal_run():
    """Flush and close the current Parquet files so they can be collected."""
    if req_logger is None or decision_logger is None:
        raise HTTPException(status_code=503, detail="Logger not initialised")
    info = await req_logger.seal()
    decisions = await decision_logger.seal()
    return {"status": "ok", "sealed": info, "decisions": decisions}


@app.get("/admin/logger")
async def logger_status():
    """Inspect the loggers: current files, row counts, and sealed runs."""
    if req_logger is None or decision_logger is None:
        raise HTTPException(status_code=503, detail="Logger not initialised")
    return {**req_logger.status(), "decisions": decision_logger.status()}


@app.post("/admin/drift")
async def configure_drift(
    threshold: Optional[float] = None,
    window: Optional[int] = None,
    min_samples: Optional[int] = None,
):
    """
    Re-tune the drift guard at runtime.

    Being able to lower the threshold is how the demotion path is demonstrated
    honestly: the guard then demotes a *healthy* model on real traffic, the
    reason on every decision changes to a least-connections fallback, and
    raising the threshold again promotes it back after a clean window.
    """
    guard = getattr(algorithm_instance, "drift", None)
    if guard is None:
        raise HTTPException(
            status_code=400,
            detail=f"algorithm '{ALGORITHM}' has no drift guard (only 'ml' does)",
        )
    return {
        "status": "ok",
        "drift": guard.configure(threshold=threshold, window=window, min_samples=min_samples),
    }


@app.get("/admin/algorithm")
async def get_algorithm_name():
    """The routing algorithm that is live right now."""
    return {"algorithm": ALGORITHM, "available": algorithm_names()}


def _demo_server(server_id: str) -> ServerInfo:
    target = next((s for s in SERVERS if s.server_id == server_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Unknown server_id: {server_id}")
    return target


async def _backend_control(
    target: ServerInfo, path: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Drive a backend's demo-only /slow or /fault control endpoint.

    Those endpoints are deliberately not published on the host: only the
    balancer can reach backend3 inside the compose network. This thin proxy
    keeps the dashboard's demo buttons working without exposing another port
    and without adding Docker control to the balancer.
    """
    url = f"http://{target.host}:{target.port}{path}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as control:
            response = await control.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"{target.server_id}{path} control failed: {exc}",
        )
    return {"path": path, "request": params, "response": payload}


@app.post("/admin/demo/degrade")
async def demo_degrade(
    server_id: str = "backend3",
    duration_s: float = 45.0,
    delay_ms: int = 400,
    error_rate: float = 0.35,
):
    """
    Make one backend slow and flaky for a bounded window.

    Demo-only convenience around the existing backend controls. It does not
    change routing: the router discovers the fault from the same metrics it
    would see for a real dependency slowdown.
    """
    target = _demo_server(server_id)
    actions = [
        await _backend_control(
            target, "/slow", {"ms": max(0, delay_ms), "duration_s": max(0.0, duration_s)}
        ),
        await _backend_control(
            target,
            "/fault",
            {"error_rate": min(1.0, max(0.0, error_rate)), "duration_s": max(0.0, duration_s)},
        ),
    ]
    return {
        "status": "ok",
        "server_id": target.server_id,
        "duration_s": duration_s,
        "actions": actions,
    }


@app.post("/admin/demo/fail")
async def demo_fail(
    server_id: str = "backend3",
    duration_s: float = 12.0,
):
    """
    Inject a 100% error fault for a short window.

    The backend stays alive, so its health probe remains green; the circuit
    breaker opens from the request failures and new work moves away. That is a
    useful demo of the failure path without requiring a Docker command. Actual
    container-kill evidence remains the H4 matrix runs in docs/results.md.
    """
    target = _demo_server(server_id)
    actions = [
        await _backend_control(target, "/slow", {"ms": 0, "duration_s": 0}),
        await _backend_control(
            target, "/fault", {"error_rate": 1.0, "duration_s": max(0.0, duration_s)}
        ),
    ]
    return {
        "status": "ok",
        "server_id": target.server_id,
        "duration_s": duration_s,
        "actions": actions,
    }


@app.post("/admin/demo/clear")
async def demo_clear(server_id: str = "backend3"):
    """Clear slow and fault injection on one backend."""
    target = _demo_server(server_id)
    actions = [
        await _backend_control(target, "/slow", {"ms": 0, "duration_s": 0}),
        await _backend_control(target, "/fault", {"error_rate": 0.0, "duration_s": 0}),
    ]
    return {"status": "ok", "server_id": target.server_id, "actions": actions}


@app.api_route("/admin/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def admin_not_found(path: str):
    """
    Keep unknown /admin paths away from the proxy.

    Without this they match the catch-all proxy route below, are forwarded to
    a backend and logged as a request to it, so a mistyped admin path both
    answered with a backend's 404 and added a row to the dataset's log.
    """
    raise HTTPException(status_code=404, detail=f"No such admin endpoint: /admin/{path}")


@app.get("/config")
async def get_config():
    """Get current configuration."""
    return {
        "algorithm": ALGORITHM,
        "servers": [
            {
                "server_id": s.server_id,
                "host": s.host,
                "port": s.port,
                "weight": s.weight,
                "cpu_limit": s.cpu_limit,
                "mem_limit_mb": s.mem_limit_mb,
                "healthy": s.healthy,
                "metrics": s.metrics.__dict__ if s.metrics else None,
            }
            for s in SERVERS
        ],
        "sla_ms": SLA_MS,
    }


@app.get("/stats")
async def stats():
    """Expose per-server load, predictions, and decision counts."""
    now = time.time()
    result = []
    for s in SERVERS:
        result.append({
            "server_id": s.server_id,
            "host": s.host,
            "port": s.port,
            "weight": s.weight,
            "cpu_limit": s.cpu_limit,
            "mem_limit_mb": s.mem_limit_mb,
            "healthy": s.healthy,
            "sent_since_snapshot": getattr(s, 'sent_since_snapshot', 0),
            "inflight_local": getattr(s, 'inflight_local', 0),
            "metrics": s.metrics.__dict__ if s.metrics else None,
            "health_failures": getattr(s, 'health_check_failures', 0),
            "circuit_open": resilience.circuit_open(s, now),
            "consecutive_failures": s.consecutive_failures,
            "circuit_opens": s.circuit_opens,
            "slow_start": s.slow_start_until > now,
        })
    # An ML router reports its own serving state, so a demo can show whether
    # predictions are driving the decisions or a fallback is.
    describe = getattr(algorithm_instance, "status", None)
    ml_status = describe() if callable(describe) else None

    age = resilience.feed_age(SERVERS, now)
    return {
        "servers": result,
        "algorithm": ALGORITHM,
        "sla_ms": SLA_MS,
        "inflight_total": sum(getattr(s, 'inflight_local', 0) for s in SERVERS),
        "ml": ml_status,
        "feed": {
            "age_ms": round(age * 1000.0, 1) if age is not None else None,
            "stale": resilience.feed_is_stale(SERVERS, now),
            "collector": collector.status() if collector else None,
        },
        "resilience": {
            "retries": STATS["retries"],
            "stale_routing": STATS["stale_routing"],
            "circuits_open": [s.server_id for s in SERVERS if resilience.circuit_open(s, now)],
            "circuit_opens": sum(s.circuit_opens for s in SERVERS),
        },
        # The dashboard's latency panel: a 60 s window, recomputed per poll.
        "latency": latency_window(now),
        # Newest last, rounded for the wire. The full-precision rows go to the
        # decision log; this is what the dashboard's predicted-vs-actual table
        # and a demo read.
        "decisions": {
            "logged": len(RECENT_DECISIONS),
            "recent": [
                {
                    "timestamp": round(row["timestamp"], 3),
                    "server_id": row["server_id"],
                    "reason": row["reason"],
                    "predicted_ms": (
                        round(row["predicted_ms"], 1)
                        if row["predicted_ms"] is not None
                        else None
                    ),
                    "actual_ms": round(row["actual_ms"], 1),
                    "status": row["status"],
                    "retries": row["retries"],
                    "model_driven": row["model_driven"],
                }
                for row in list(RECENT_DECISIONS)[-15:]
            ],
        },
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """
    Live dashboard.

    Served from a file, and read on each request, so an edit to the page shows
    up on reload without restarting the balancer. `/` is free for this: the
    load generator only ever requests /ping, /cpu, /io and /payload, so the
    dashboard can never be counted as load traffic.
    """
    try:
        return HTMLResponse(DASHBOARD_PATH.read_text(encoding="utf-8"))
    except OSError as exc:
        return HTMLResponse(
            f"<h1>Dashboard unavailable</h1><p>{DASHBOARD_PATH}: {exc}</p>",
            status_code=500,
        )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    """Forward the request to the chosen backend server."""
    if path.startswith("cpu"):
        req_type_str = "cpu"
    elif path.startswith("io"):
        req_type_str = "io"
    elif path.startswith("payload"):
        req_type_str = "payload"
    else:
        req_type_str = "light"

    # Content-Length is the only payload size known before the routing
    # decision, because the body has not been read yet. payload_kb is a model
    # feature, so a router that reads it after the choice would always see
    # zero for the POST /payload traffic that exists to make it vary.
    try:
        declared_kb = max(float(request.headers.get("content-length", 0) or 0), 0.0) / 1024.0
    except (TypeError, ValueError):
        declared_kb = 0.0

    context = RequestContext(
        request_id=str(uuid.uuid4()),
        method=request.method,
        path=f"/{path}",
        req_type=RequestType(req_type_str),
        payload_kb=declared_kb,
    )

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    body = await request.body()
    context.payload_kb = len(body) / 1024.0

    # Sanitize label/log values so a non-numeric field can never abort a write
    def _f(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _state_of(server: ServerInfo) -> Dict[str, float]:
        """The server's pre-request snapshot, exactly as the Parquet row records it."""
        metrics = server.metrics
        return {
            "srv_cpu_pct": _f(metrics.cpu_pct if metrics else 0.0),
            "srv_mem_pct": _f(metrics.mem_pct if metrics else 0.0),
            "inflight": int(_f(metrics.inflight_requests if metrics else 0.0)),
            "req_rate_5s": _f(metrics.req_rate_5s if metrics else 0.0),
            "rt_ewma_ms": _f(metrics.rt_ewma_ms if metrics else 0.0),
            "rt_p95_5s_ms": _f(metrics.rt_p95_5s_ms if metrics else 0.0),
            "err_rate_5s": _f(metrics.error_rate_5s if metrics else 0.0),
            "cpu_limit": _f(metrics.cpu_limit if metrics else server.cpu_limit),
            "cpu_delta_5s": _f(metrics.cpu_delta_5s if metrics else 0.0),
        }

    request_started = time.time()
    tried: List[str] = []
    final: Optional[Dict[str, Any]] = None

    for attempt in (0, 1):
        if attempt == 1:
            STATS["retries"] += 1

        # Safeguards sit behind this call: circuit-broken servers are skipped,
        # a recovering server is ramped in, and a dead metrics feed routes on
        # the last snapshot instead of failing every request.
        choice: Optional[ServerChoice] = resilience.choose(
            context, SERVERS, algorithm_instance, tried=tried
        )
        if choice is None:
            break
        if choice.reason.startswith("stale-feed"):
            STATS["stale_routing"] += 1

        server = choice.server
        target_url = f"http://{server.host}:{server.port}/{path}"
        if request.url.query:
            target_url += f"?{request.url.query}"

        state_before = _state_of(server)
        server.inflight_local = getattr(server, 'inflight_local', 0) + 1

        status = 0
        content: Optional[bytes] = None
        response_headers: Dict[str, str] = {}

        try:
            resp = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
            )
            status = resp.status_code
            content = resp.content
            response_headers = dict(resp.headers)
        except httpx.HTTPError as exc:
            # Connection error, timeout or protocol error: the backend did not
            # serve this request, so it is both a client-visible failure and a
            # signal for the breaker.
            status = 502
            response_headers = {"x-proxy-error": str(exc)}
        except Exception as exc:
            status = 500
            response_headers = {"x-proxy-error": str(exc)}
        finally:
            server.inflight_local = max(0, getattr(server, 'inflight_local', 0) - 1)

        resilience.record_outcome(server, ok=status < 500)
        tried.append(server.server_id)
        final = {
            "server_id": server.server_id,
            "state": state_before,
            "status": status,
            "content": content,
            "headers": response_headers,
            # Decision detail, for the decision log and the drift guard.
            "attempt": attempt,
            "reason": choice.reason,
            "predicted_ms": choice.predicted_rt_ms,
            "model_driven": bool(choice.model_driven),
            "inflight_at_decision": int(getattr(server, "inflight", 0)),
            "feed_age_ms": (resilience.feed_age(SERVERS) or 0.0) * 1000.0,
        }

        # One retry, on a different server, for server-side failures only. A
        # 404 or a 505 is the client's answer; retrying it would just add load.
        if attempt == 1 or status not in resilience.RETRYABLE_STATUS:
            break

    if final is None:
        raise HTTPException(status_code=503, detail="No healthy servers available")

    # One row per client request, recording the outcome the client received.
    # A retried request therefore appears once, with its total latency; the
    # failed first attempt stays visible in the backend's own err_rate_5s
    # instead of inflating the request count in the dataset.
    duration_ms = (time.time() - request_started) * 1000
    if final["status"] < 400:
        # Successful requests only, matching the A/B harness: a 503 returns
        # fast and would drag the dashboard's p95 down.
        RECENT_LATENCIES.append((time.time(), duration_ms))
    if req_logger is not None:
        try:
            await req_logger.log({
                "timestamp": request_started,
                "req_type": context.req_type.value,
                "payload_kb": _f(context.payload_kb),
                "server_id": final["server_id"],
                "response_ms": _f(duration_ms),
                "status": int(final["status"]),
                **final["state"],
            })
        except Exception as exc:
            print(f"logger failed: {exc}")

    # One decision row per client request: what was chosen, why, and what the
    # client actually waited. Only the always-on copy is written outside a run.
    entry = {
        "timestamp": request_started,
        "request_id": context.request_id,
        "algorithm": ALGORITHM,
        "server_id": final["server_id"],
        "reason": final["reason"],
        "predicted_ms": (
            float(final["predicted_ms"]) if final["predicted_ms"] is not None else None
        ),
        "actual_ms": _f(duration_ms),
        "status": int(final["status"]),
        "retries": int(final["attempt"]),
        "model_driven": final["model_driven"],
        "inflight": final["inflight_at_decision"],
        "feed_age_ms": round(final["feed_age_ms"], 1),
        "req_type": context.req_type.value,
    }
    RECENT_DECISIONS.append(entry)
    if decision_logger is not None:
        try:
            await decision_logger.log(dict(entry))
        except Exception as exc:
            print(f"decision logger failed: {exc}")

    # Feedback for algorithms that monitor themselves. A retried request is
    # excluded: its latency contains a failed first attempt, so it is not an
    # observation about the prediction that was made.
    if final["attempt"] == 0:
        algorithm_instance.observe_outcome(
            entry["predicted_ms"], entry["actual_ms"], entry["status"]
        )

    if final["content"] is None:
        raise HTTPException(
            status_code=502,
            detail=f"Backend {final['server_id']} unreachable",
        )

    return Response(
        content=final["content"],
        status_code=final["status"],
        headers=final["headers"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

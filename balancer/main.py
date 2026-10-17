"""
AI/ML Load Balancer - Main entry point.

Phase 2 Implementation:
- Reverse proxy with pluggable routing
- Live metrics collection with MetricsCollector
- Parquet request logger for data generation
- Enhanced /stats endpoint
"""

import asyncio
import os
import sys
import time
import uuid
import yaml
from typing import Optional, List, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse

from balancer.algorithms import get_algorithm, Algorithm
from balancer.interfaces import RequestContext, ServerChoice, ServerInfo, RequestType
from balancer.collector import MetricsCollector
from balancer.logger import RequestLogger

CONFIG_PATH = os.environ.get("BALANCER_CONFIG", "balancer/config.yaml")
ALGORITHM: Optional[str] = None
SERVERS: List[ServerInfo] = []
SLA_MS = 500
client: Optional[httpx.AsyncClient] = None
algorithm_instance: Optional[Algorithm] = None
collector: Optional[MetricsCollector] = None
req_logger: Optional[RequestLogger] = None


def load_config():
    """Load configuration from YAML file."""
    global ALGORITHM, SERVERS, SLA_MS, collector, req_logger
    
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


async def init():
    """Initialize the balancer components."""
    global client, algorithm_instance
    load_config()
    algorithm_instance = get_algorithm(ALGORITHM)
    client = httpx.AsyncClient(timeout=30.0)
    
    if collector:
        await collector.start()
    if req_logger:
        await req_logger.start()
        
    print(f"Loaded balancer config: algorithm={ALGORITHM}, servers={len(SERVERS)}")


async def close():
    """Clean up resources."""
    global client, collector, req_logger
    if collector:
        await collector.stop()
    if req_logger:
        await req_logger.stop()
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
            "metrics": s.metrics.__dict__ if s.metrics else None,
            "health_failures": getattr(s, 'health_check_failures', 0)
        })
    return {
        "servers": result, 
        "algorithm": ALGORITHM,
        "sla_ms": SLA_MS,
    }


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

    context = RequestContext(
        request_id=str(uuid.uuid4()),
        method=request.method,
        path=f"/{path}",
        req_type=RequestType(req_type_str),
        payload_kb=0.0,
    )

    choice: Optional[ServerChoice] = algorithm_instance.choose_server(context, SERVERS)
    if choice is None:
        raise HTTPException(status_code=503, detail="No healthy servers available")

    server = choice.server
    target_url = f"http://{server.host}:{server.port}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    server.sent_since_snapshot = getattr(server, 'sent_since_snapshot', 0) + 1

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    body = await request.body()
    context.payload_kb = len(body) / 1024.0

    # Save state before request for training data
    state_before = None
    if server.metrics:
        state_before = {
            "srv_cpu_pct": server.metrics.cpu_pct,
            "srv_mem_pct": server.metrics.mem_pct,
            "inflight": server.metrics.inflight_requests,
            "req_rate_5s": server.metrics.req_rate_5s,
            "rt_ewma_ms": server.metrics.rt_ewma_ms,
            "rt_p95_5s_ms": server.metrics.rt_p95_5s_ms,
            "err_rate_5s": server.metrics.err_rate_5s,
            "cpu_limit": server.metrics.cpu_limit,
            "cpu_delta_5s": server.metrics.cpu_delta_5s,
        }

    # Increment in-flight count
    if server.metrics:
        server.metrics.inflight_requests += 1

    start_time = time.time()
    try:
        resp = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
        )
        duration_ms = (time.time() - start_time) * 1000

        # Log request data
        if req_logger and state_before:
            await req_logger.log({
                "timestamp": start_time,
                "req_type": context.req_type.value,
                "payload_kb": context.payload_kb,
                "server_id": server.server_id,
                "response_ms": duration_ms,
                "status": resp.status_code,
                **state_before
            })

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )
    except httpx.ConnectError:
        if hasattr(server, 'health_check_failures'):
            server.health_check_failures = getattr(server, 'health_check_failures', 0) + 1
        raise HTTPException(status_code=502, detail=f"Backend {server.server_id} unreachable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")
    finally:
        # Decrement in-flight count
        if server.metrics:
            server.metrics.inflight_requests = max(0, server.metrics.inflight_requests - 1)

"""
AI/ML Load Balancer - Main entry point.

This is a minimal reverse proxy implementation for Phase 0/M1.
It supports HTTP forwarding with Round Robin routing and health checks.
"""

import asyncio
import os
import signal
import sys
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from balancer.interfaces import Algorithm, ServerChoice, ServerInfo, RequestType
from balancer.algorithms import RoundRobin


CONFIG = {
    "algorithm": os.environ.get("ALGORITHM", "round_robin"),
    "servers": [
        ServerInfo(server_id="backend1", host="backend1", port=8000, weight=2.0, cpu_limit=2.0, mem_limit_mb=512),
        ServerInfo(server_id="backend2", host="backend2", port=8000, weight=1.0, cpu_limit=1.0, mem_limit_mb=256),
        ServerInfo(server_id="backend3", host="backend3", port=8000, weight=1.0, cpu_limit=1.0, mem_limit_mb=512),
        ServerInfo(server_id="backend4", host="backend4", port=8000, weight=0.5, cpu_limit=0.5, mem_limit_mb=128),
    ],
    "sla_ms": 500,
}

ALGORITHMS = {
    "round_robin": RoundRobin,
}

algorithm: Optional[Algorithm] = None
client: Optional[httpx.AsyncClient] = None


def get_algorithm(name: str) -> Algorithm:
    """Create the routing algorithm specified by name."""
    cls = ALGORITHMS.get(name)
    if cls is None:
        raise ValueError(f"Unknown algorithm: {name}")
    return cls()


async def init():
    """Initialize the balancer components."""
    global algorithm, client
    algorithm = get_algorithm(CONFIG["algorithm"])
    client = httpx.AsyncClient(timeout=30.0)


async def close():
    """Clean up resources."""
    global client
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
    global algorithm
    try:
        algorithm = get_algorithm(name)
        CONFIG["algorithm"] = name
        return {"status": "ok", "algorithm": name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/stats")
async def stats():
    """Expose per-server load, predictions, and decision counts."""
    result = []
    for s in CONFIG["servers"]:
        result.append({
            "server_id": s.server_id,
            "host": s.host,
            "port": s.port,
            "weight": s.weight,
            "healthy": s.healthy,
            "sent_since_snapshot": s.sent_since_snapshot,
        })
    return {"servers": result, "algorithm": CONFIG["algorithm"]}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    """Forward the request to the chosen backend server."""
    req_type = RequestType.LIGHT
    if path.startswith("cpu"):
        req_type = RequestType.CPU
    elif path.startswith("io"):
        req_type = RequestType.IO

    from balancer.interfaces import RequestContext
    import uuid

    context = RequestContext(
        request_id=str(uuid.uuid4()),
        method=request.method,
        path=f"/{path}",
        req_type=req_type,
    )

    choice: Optional[ServerChoice] = algorithm.choose_server(context, CONFIG["servers"])
    if choice is None:
        raise HTTPException(status_code=503, detail="No healthy servers available")

    server = choice.server
    target_url = f"http://{server.host}:{server.port}/{path}"

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    body = await request.body()

    try:
        resp = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )
    except httpx.ConnectError:
        server.health_check_failures += 1
        raise HTTPException(status_code=502, detail=f"Backend {server.server_id} unreachable")
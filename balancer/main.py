"""
AI/ML Load Balancer - Main entry point.

This is the reverse proxy implementation for Phase 1.
It supports HTTP forwarding with pluggable routing algorithms
configured via YAML.
"""

import asyncio
import os
import signal
import sys
from typing import Optional

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from balancer.algorithms import get_algorithm
from balancer.interfaces import RequestContext, ServerChoice


CONFIG_PATH = os.environ.get("BALANCER_CONFIG", "balancer/config.yaml")
ALGORITHM: Optional[str] = None
SERVERS = []
SLA_MS = 500
client: Optional[httpx.AsyncClient] = None
algorithm_instance = None


def load_config():
    """Load configuration from YAML file."""
    global ALGORITHM, SERVERS, SLA_MS
    
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback to default config if file not found
        config = {
            "algorithm": "round_robin",
            "servers": [
                {"server_id": "backend1", "host": "backend1", "port": 8000, "weight": 2.0, "cpu_limit": 2.0, "mem_limit_mb": 512},
                {"server_id": "backend2", "host": "backend2", "port": 8000, "weight": 1.0, "cpu_limit": 1.0, "mem_limit_mb": 256},
                {"server_id": "backend3", "host": "backend3", "port": 8000, "weight": 1.0, "cpu_limit": 1.0, "mem_limit_mb": 512},
                {"server_id": "backend4", "host": "backend4", "port": 8000, "weight": 0.5, "cpu_limit": 0.5, "mem_limit_mb": 128},
            ],
            "sla_ms": 500,
        }
    
    ALGORITHM = config.get("algorithm", "round_robin")
    SLA_MS = config.get("sla_ms", 500)
    
    # Convert server dicts to ServerInfo objects
    from balancer.interfaces import ServerInfo, ServerMetrics
    
    SERVERS = []
    for s in config.get("servers", []):
        server = ServerInfo(
            server_id=s["server_id"],
            host=s["host"],
            port=s["port"],
            weight=s.get("weight", 1.0),
            cpu_limit=s.get("cpu_limit", 1.0),
            mem_limit_mb=s.get("mem_limit_mb", 512),
            metrics=None,  # Will be populated by metrics collection
        )
        SERVERS.append(server)


async def init():
    """Initialize the balancer components."""
    global client, algorithm_instance
    load_config()
    algorithm_instance = get_algorithm(ALGORITHM)
    client = httpx.AsyncClient(timeout=30.0)
    print(f"Loaded balancer config: algorithm={ALGORITHM}, servers={len(SERVERS)}")


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
    global algorithm_instance, ALGORITHM
    try:
        algorithm_instance = get_algorithm(name)
        ALGORITHM = name
        # Update config file
        config = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f) or {}
        config["algorithm"] = name
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
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
        })
    return {
        "servers": result, 
        "algorithm": ALGORITHM,
        "sla_ms": SLA_MS,
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    """Forward the request to the chosen backend server."""
    # Determine request type based on path for feature extraction
    if path.startswith("cpu"):
        req_type_str = "cpu"
    elif path.startswith("io"):
        req_type_str = "io"
    elif path.startswith("payload"):
        req_type_str = "payload"
    else:
        req_type_str = "light"  # Default for /ping and unknown paths

    from balancer.interfaces import RequestType
    import uuid

    context = RequestContext(
        request_id=str(uuid.uuid4()),
        method=request.method,
        path=f"/{path}",
        req_type=RequestType(req_type_str),
        payload_kb=0.0,  # Could extract from request body if needed
    )

    choice: Optional[ServerChoice] = algorithm_instance.choose_server(context, SERVERS)
    if choice is None:
        raise HTTPException(status_code=503, detail="No healthy servers available")

    server = choice.server
    target_url = f"http://{server.host}:{server.port}/{path}"

    # Track that we sent a request to this server (for herd protection)
    server.sent_since_snapshot = getattr(server, 'sent_since_snapshot', 0) + 1

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
        # Mark server as potentially unhealthy
        if hasattr(server, 'health_check_failures'):
            server.health_check_failures = getattr(server, 'health_check_failures', 0) + 1
        raise HTTPException(status_code=502, detail=f"Backend {server.server_id} unreachable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")
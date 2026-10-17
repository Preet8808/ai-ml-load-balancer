"""Backend server that exposes /ping, /cpu, /io, /payload, and /health endpoints.

This is the core application that will run inside Docker containers
with different resource limits to create heterogeneity.
"""

import asyncio
import os
import time
import random
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

app = FastAPI(title="Backend Server")

BACKEND_ID = os.environ.get("BACKEND_ID", "1")

# Track metrics for this backend
metrics_state = {
    "cpu_pct": 0.0,
    "mem_pct": 0.0,
    "load_avg": 0.0,
    "inflight": 0,
    "req_rate_5s": 0.0,
    "rt_ewma_ms": 0.0,
    "rt_p95_5s_ms": 0.0,
    "error_rate_5s": 0.0,
    "net_io_bytes_per_s": 0.0,
    "cpu_limit": float(os.environ.get("BACKEND_CPUS", "1.0")),
    "mem_limit_mb": int(os.environ.get("BACKEND_MEM", "512")),
}


@app.get("/health")
async def health():
    """Health check endpoint - always returns healthy unless explicitly marked."""
    return {"status": "healthy", "id": BACKEND_ID, "cpu_pct": metrics_state["cpu_pct"]}


@app.get("/ping")
async def ping():
    """Light request - returns immediately."""
    return {"id": BACKEND_ID, "type": "ping", "result": "pong"}


@app.get("/cpu")
async def cpu():
    """CPU-bound request - does computation proportional to backend capacity."""
    # Simulate work based on CPU limit - limited servers do less work
    cpu_work = int(metrics_state["cpu_limit"] * 1000)
    result = sum(i * i for i in range(cpu_work))
    # Update metrics
    metrics_state["cpu_pct"] = min(100.0, metrics_state["cpu_pct"] + 5.0)
    metrics_state["inflight"] = max(0, metrics_state["inflight"] - 1)
    return {"id": BACKEND_ID, "type": "cpu", "computation": result, "cpu_limit": metrics_state["cpu_limit"]}


@app.get("/io")
async def io():
    """I/O-bound request - simulates database wait with async sleep."""
    # Wait time varies based on backend capacity
    wait_ms = int(50 / metrics_state["cpu_limit"]) if metrics_state["cpu_limit"] > 0 else 50
    await asyncio.sleep(wait_ms / 1000.0)
    # Update metrics
    metrics_state["mem_pct"] = min(100.0, metrics_state["mem_pct"] + 3.0)
    return {"id": BACKEND_ID, "type": "io", "waited_ms": wait_ms, "cpu_limit": metrics_state["cpu_limit"]}


@app.get("/payload")
async def payload():
    """Heavy response - returns large JSON payload."""
    item_count = int(100 * metrics_state["cpu_limit"])
    data = {"id": BACKEND_ID, "items": list(range(item_count))}
    return JSONResponse(content=data)


@app.get("/metrics")
async def metrics():
    """Internal metrics endpoint - exposed to load balancer."""
    # Update some state to simulate real-world changes
    metrics_state["req_rate_5s"] = random.uniform(10, 100)
    metrics_state["error_rate_5s"] = random.uniform(0, 0.02)
    metrics_state["load_avg"] = random.uniform(0.1, 2.0)
    metrics_state["net_io_bytes_per_s"] = random.uniform(1000, 10000)
    return JSONResponse(content={
        "backend_id": BACKEND_ID,
        "timestamp": time.time(),
        **metrics_state,
    })


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request, path: str):
    """Catch-all for unknown routes."""
    result = sum(i * i for i in range(5000))
    return {"id": BACKEND_ID, "path": path, "result": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
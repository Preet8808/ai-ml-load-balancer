"""Backend server that exposes /ping, /cpu, /io, /payload, and /health endpoints."""

import asyncio
import os
import time
import random
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

app = FastAPI()
BACKEND_ID = os.environ.get("BACKEND_ID", "1")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "id": BACKEND_ID}


@app.get("/ping")
async def ping():
    """Light request - returns immediately."""
    return {"id": BACKEND_ID, "type": "ping", "result": "pong"}


@app.get("/cpu")
async def cpu():
    """CPU-bound request - does some computation."""
    # Simulate CPU work with a computation
    result = sum(i * i for i in range(10000))
    return {"id": BACKEND_ID, "type": "cpu", "computation": result}


@app.get("/io")
async def io():
    """I/O-bound request - simulates database wait."""
    await asyncio.sleep(0.1)  # 100ms simulated I/O
    return {"id": BACKEND_ID, "type": "io", "waited_ms": 100}


@app.get("/payload")
async def payload():
    """Heavy response - returns large JSON."""
    data = {"id": BACKEND_ID, "items": list(range(1000))}
    return JSONResponse(content=data)


@app.get("/metrics")
async def metrics():
    """Internal metrics endpoint (populated by metrics agent)."""
    # This will be populated by the metrics agent running alongside
    return JSONResponse(content={
        "backend_id": BACKEND_ID,
        "timestamp": time.time(),
        "cpu_pct": random.uniform(10, 50),
        "mem_pct": random.uniform(20, 60),
    })


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request, path: str):
    """Catch-all for unknown routes - route as CPU-intensive."""
    result = sum(i * i for i in range(5000))
    return {"id": BACKEND_ID, "path": path, "result": result}
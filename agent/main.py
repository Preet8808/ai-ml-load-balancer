"""
Metrics agent for backend servers.

Collects CPU, memory, load average, network I/O and other system state
from each backend and exposes it via HTTP /metrics endpoint.
"""

import os
import time
import psutil
import asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="Metrics Agent")

BACKEND_ID = os.environ.get("BACKEND_ID", "1")


def get_system_metrics():
    """Collect current system metrics using psutil."""
    # CPU metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # Memory metrics
    mem = psutil.virtual_memory()
    mem_percent = mem.percent
    
    # Load average (Linux/Unix)
    try:
        load_avg = os.getloadavg()[0]  # 1-minute load average
    except (AttributeError, ValueError):
        load_avg = 0.0
    
    # Network I/O
    net_io = psutil.net_io_counters()
    net_bytes_sent = net_io.bytes_sent
    net_bytes_recv = net_io.bytes_recv
    
    # Process-specific metrics
    process = psutil.Process()
    num_threads = process.num_threads()
    open_files = len(process.open_files())
    
    return {
        "backend_id": BACKEND_ID,
        "timestamp": time.time(),
        "cpu_pct": cpu_percent,
        "mem_pct": mem_percent,
        "load_avg": load_avg,
        "net_io_bytes_sent": net_bytes_sent,
        "net_io_bytes_recv": net_bytes_recv,
        "num_threads": num_threads,
        "open_files": open_files,
        "cpu_count": psutil.cpu_count(),
        "memory_total_mb": mem.total // (1024 * 1024),
        "memory_available_mb": getattr(mem, 'available', mem.total) // (1024 * 1024),
    }


@app.get("/metrics")
async def metrics():
    """Expose system metrics as JSON."""
    return JSONResponse(content=get_system_metrics())


@app.get("/health")
async def health():
    """Health check endpoint for the agent itself."""
    return {"status": "healthy", "backend_id": BACKEND_ID, "timestamp": time.time()}


async def background_metrics_updater():
    """Background task to update metrics periodically (for future implementation)."""
    while True:
        await asyncio.sleep(1)  # Update every second
        # This would be used for the async collector in Phase 2
        pass


@app.on_event("startup")
async def startup():
    # Start background metrics updater
    asyncio.create_task(background_metrics_updater())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
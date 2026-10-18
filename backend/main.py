"""Backend server that exposes /ping, /cpu, /io, /payload, and /health endpoints.

The backend reports *real* resource usage and request statistics, which the
balancer's metrics collector consumes and the ML models learn from:

  - cpu_pct        : cgroup v2 CPU usage delta / (elapsed * cpu quota)
  - mem_pct        : cgroup v2 memory.current / memory.max
  - load_avg       : /proc/loadavg (1-minute)
  - net_io_*       : /proc/net/dev byte counters, delta per second
  - inflight       : requests currently being served
  - req_rate_5s    : completed requests per second over the last 5 s
  - rt_ewma_ms     : EWMA of observed response times for this server
  - rt_p95_5s_ms   : p95 of observed response times over the last 5 s
  - error_rate_5s  : share of 5xx responses over the last 5 s

/metrics and /health are excluded from the application statistics so that
monitoring traffic does not pollute the training data.

Endpoints are deliberately heterogeneous under load:

  - /cpu uses real CPU work, so cgroup throttling produces genuine queueing.
  - /io sleeps proportionally to the number of concurrent requests on this
    backend, simulating contention on a shared downstream dependency.
"""

import asyncio
import math
import os
import time
import random
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Backend Server")

BACKEND_ID = os.environ.get("BACKEND_ID", "1")
CGROUP = Path("/sys/fs/cgroup")

# ---------------------------------------------------------------------------
# Capacity (set explicitly by docker-compose, else derived from cgroup limits)
# ---------------------------------------------------------------------------

def _cgroup_cpu_limit() -> float:
    """Return the CPU quota in cores (e.g. 2.0) or the host CPU count."""
    try:
        quota, period = (CGROUP / "cpu.max").read_text().split()
        if quota != "max":
            return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    return float(os.cpu_count() or 1)


def _cgroup_mem_limit_mb() -> float:
    """Return the memory limit in MB, or total host memory."""
    try:
        raw = (CGROUP / "memory.max").read_text().strip()
        if raw != "max":
            return int(raw) / (1024 * 1024)
    except (OSError, ValueError):
        pass
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 * 1024)
    except Exception:
        return 512.0


CPU_LIMIT = float(os.environ.get("BACKEND_CPUS") or _cgroup_cpu_limit())
MEM_LIMIT_MB = float(os.environ.get("BACKEND_MEM") or _cgroup_mem_limit_mb())

# CPU work per /cpu request. Constant work means a 0.5-core backend is
# measurably slower than a 2-core backend for the same request.
#
# Sized so the stack lands in a partially-loaded regime across the bench:
# measured aggregate latency was 36 ms at 20 users, 224 ms at 40, 641 ms at 80
# and 1853 ms at 200. Past roughly 80 concurrent users the proxy collapses and
# the balancer queues ~180 requests with the backends idle, which tells us
# nothing about load. 80k iterations (~28 ms) keeps throughput near 200 req/s
# while still saturating a backend's CPU quota.
CPU_WORK_ITERATIONS = int(os.environ.get("CPU_WORK_ITERATIONS", "80000"))
IO_BASE_MS = float(os.environ.get("IO_BASE_MS", "25"))

# ---------------------------------------------------------------------------
# Request statistics (application traffic only)
# ---------------------------------------------------------------------------

WINDOW_S = 5.0
EVENT_BUFFER = 20000

_events: Deque[Tuple[float, float, int]] = deque(maxlen=EVENT_BUFFER)
_inflight = 0
_ewma_ms = 0.0
_ewma_alpha = 0.15
_total_requests = 0
_total_errors = 0
_started_at = time.time()

# Resource sampling state
_last_cpu_usec: Optional[int] = None
_last_cpu_sample_ts: Optional[float] = None
_last_net_bytes: Optional[int] = None
_last_net_sample_ts: Optional[float] = None


def _read_cpu_usec() -> Optional[int]:
    """Total CPU time consumed by this cgroup, in microseconds."""
    try:
        for line in (CGROUP / "cpu.stat").read_text().splitlines():
            if line.startswith("usage_usec"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil
        proc = psutil.Process()
        cpu = proc.cpu_times()
        return int((cpu.user + cpu.system) * 1_000_000)
    except Exception:
        return None


def _read_mem_pct() -> float:
    """Memory utilisation as a percentage of the container limit."""
    try:
        current = int((CGROUP / "memory.current").read_text().strip())
        limit_raw = (CGROUP / "memory.max").read_text().strip()
        if limit_raw != "max":
            limit = int(limit_raw)
            if limit > 0:
                return min(100.0, current / limit * 100.0)
    except (OSError, ValueError):
        pass
    try:
        import psutil
        return psutil.virtual_memory().percent
    except Exception:
        return 0.0


def _read_net_bytes() -> Optional[int]:
    """Total bytes received + sent across all interfaces."""
    try:
        total = 0
        with open("/proc/net/dev", "r") as f:
            for line in f.readlines()[2:]:
                fields = line.split()
                if len(fields) >= 10:
                    total += int(fields[1]) + int(fields[9])
        return total
    except (OSError, ValueError, IndexError):
        return None


def _read_load_avg() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        try:
            import psutil
            return psutil.getloadavg()[0]
        except Exception:
            return 0.0


def _cpu_pct_sample() -> float:
    """CPU utilisation since the previous sample, normalised by the quota."""
    global _last_cpu_usec, _last_cpu_sample_ts

    now = time.time()
    usec = _read_cpu_usec()
    if usec is None:
        return 0.0

    if _last_cpu_usec is None or _last_cpu_sample_ts is None:
        _last_cpu_usec, _last_cpu_sample_ts = usec, now
        return 0.0

    elapsed = now - _last_cpu_sample_ts
    if elapsed <= 0:
        return 0.0

    delta_usec = max(0, usec - _last_cpu_usec)
    _last_cpu_usec, _last_cpu_sample_ts = usec, now

    # CPU-seconds consumed / wall-seconds / cores available
    utilisation = (delta_usec / 1_000_000.0) / elapsed / max(CPU_LIMIT, 1e-6) * 100.0
    return max(0.0, min(100.0, utilisation))


def _net_io_bytes_per_s() -> float:
    global _last_net_bytes, _last_net_sample_ts

    now = time.time()
    total = _read_net_bytes()
    if total is None:
        return 0.0

    if _last_net_bytes is None or _last_net_sample_ts is None:
        _last_net_bytes, _last_net_sample_ts = total, now
        return 0.0

    elapsed = now - _last_net_sample_ts
    if elapsed <= 0:
        return 0.0

    delta = max(0, total - _last_net_bytes)
    _last_net_bytes, _last_net_sample_ts = total, now
    return delta / elapsed


def _window_events() -> list[Tuple[float, float, int]]:
    cutoff = time.time() - WINDOW_S
    # deque is chronological; walk from the right until we leave the window
    out = []
    for ev in reversed(_events):
        if ev[0] < cutoff:
            break
        out.append(ev)
    return out


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(p / 100.0 * len(ordered)) - 1))
    return ordered[idx]


def _apply_metrics_response(resp) -> None:
    """Set the no-store cache header so proxies never reuse stats."""
    resp.headers["cache-control"] = "no-store"


# ---------------------------------------------------------------------------
# Fault injection control (used by the dataset driver's `degraded` profile)
# ---------------------------------------------------------------------------

_fault_error_rate = 0.0
_fault_until = 0.0


def _fault_active() -> bool:
    """True if this request should be failed on purpose."""
    if _fault_error_rate <= 0.0 or time.time() >= _fault_until:
        return False
    return random.random() < _fault_error_rate


@app.get("/fault")
async def fault(error_rate: float = 0.0, duration_s: float = 30.0):
    """
    Fail a fraction of requests for a while, so error-rate metrics carry real
    information. Excluded from the application statistics like /metrics, so
    control traffic cannot pollute the load the router is meant to observe.
    """
    global _fault_error_rate, _fault_until
    _fault_error_rate = max(0.0, min(1.0, error_rate))
    _fault_until = time.time() + max(0.0, duration_s)
    return {
        "id": BACKEND_ID,
        "error_rate": _fault_error_rate,
        "duration_s": duration_s,
    }


# ---------------------------------------------------------------------------
# Middleware: track inflight, latency, status for application traffic
# ---------------------------------------------------------------------------

@app.middleware("http")
async def track_request(request: Request, call_next):
    global _inflight, _ewma_ms, _total_requests, _total_errors

    path = request.url.path
    if path in ("/metrics", "/health", "/fault", "/slow"):
        return await call_next(request)

    # Bookkeeping starts before the injected delay, so a degraded server also
    # looks degraded in its own metrics (rt_ewma_ms up, inflight up). A fault
    # that the router cannot observe teaches nothing: the whole point of the
    # degraded scenario is that the state features reveal it.
    _inflight += 1
    started = time.perf_counter()
    status = 500
    try:
        if _slow_active():
            await asyncio.sleep(_slow_ms / 1000.0)

        # Injected faults pass through the same bookkeeping as real requests,
        # so error_rate_5s reflects them. Without this the feature is a
        # constant zero and a router can never learn to avoid a failing server.
        if _fault_active():
            response = JSONResponse(
                status_code=503,
                content={"id": BACKEND_ID, "error": "injected fault"},
            )
        else:
            response = await call_next(request)
        status = response.status_code
        return response
    finally:
        latency_ms = (time.perf_counter() - started) * 1000.0
        _inflight = max(0, _inflight - 1)
        if _inflight == 0 and _total_requests == 0:
            _ewma_ms = latency_ms
        else:
            _ewma_ms = _ewma_alpha * latency_ms + (1 - _ewma_alpha) * _ewma_ms
        _events.append((time.time(), latency_ms, status))
        _total_requests += 1
        if status >= 500:
            _total_errors += 1


def _burn(iterations: int) -> int:
    total = 0
    for i in range(iterations):
        total += i * i
    return total


# CPU work is bounded by a semaphore whose width equals the container's CPU
# quota. Requests beyond that wait asynchronously, which is what a fixed
# worker pool does in a real server and is the contention a router should
# route around.
#
# Threads, not processes: at most CPU_WORKERS of them ever hold the GIL, so
# the event loop still gets scheduled and keeps answering /metrics. Unbounded
# `to_thread` calls would instead starve the loop, and the queue-depth signal
# a router depends on would read a flat zero.
CPU_WORKERS = max(1, int(round(CPU_LIMIT)))
_cpu_slots: Optional[asyncio.Semaphore] = None
_cpu_queued = 0


def _get_cpu_slots() -> asyncio.Semaphore:
    global _cpu_slots
    if _cpu_slots is None:
        _cpu_slots = asyncio.Semaphore(CPU_WORKERS)
    return _cpu_slots


async def _run_cpu(iterations: int) -> int:
    global _cpu_queued
    slots = _get_cpu_slots()
    _cpu_queued += 1
    try:
        async with slots:
            return await asyncio.to_thread(_burn, iterations)
    finally:
        _cpu_queued -= 1


def cpu_queue_depth() -> int:
    """Requests waiting for a CPU worker slot."""
    return max(0, _cpu_queued - CPU_WORKERS)


# ---------------------------------------------------------------------------
# Application endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint used by Docker and the balancer."""
    return {"status": "healthy", "id": BACKEND_ID, "cpu_limit": CPU_LIMIT}


@app.get("/ping")
async def ping():
    """Light request - returns immediately."""
    return {"id": BACKEND_ID, "type": "ping", "result": "pong"}


@app.get("/cpu")
async def cpu():
    """CPU-bound request doing constant work, limited by the cgroup quota."""
    result = await _run_cpu(CPU_WORK_ITERATIONS)
    return {
        "id": BACKEND_ID,
        "type": "cpu",
        "computation": result,
        "cpu_limit": CPU_LIMIT,
        "workers": CPU_WORKERS,
        "queue_depth": cpu_queue_depth(),
    }


@app.get("/io")
async def io():
    """I/O-bound request whose wait grows with concurrency on this backend.

    Real downstream contention (a shared database, say) makes latency depend
    on how many requests this server is already handling, which is exactly
    the signal a load-aware router should react to.
    """
    concurrency = max(1, _inflight)
    # Square-root growth: 1 request -> base, 4 -> 2x, 9 -> 3x, capped at 6x
    contention = min(6.0, math.sqrt(concurrency))
    jitter = random.uniform(0.9, 1.1)
    wait_ms = IO_BASE_MS * contention * jitter / max(CPU_LIMIT ** 0.25, 0.6)
    await asyncio.sleep(wait_ms / 1000.0)
    return {
        "id": BACKEND_ID,
        "type": "io",
        "waited_ms": round(wait_ms, 2),
        "inflight": concurrency,
        "cpu_limit": CPU_LIMIT,
    }


@app.get("/payload")
@app.post("/payload")
async def payload(request: Request):
    """Heavy response - returns a large JSON payload."""
    item_count = int(os.environ.get("PAYLOAD_ITEMS", "2000"))
    data = {"id": BACKEND_ID, "items": list(range(item_count))}
    return JSONResponse(content=data)


_slow_ms = 0
_slow_until = 0.0


def _slow_active() -> bool:
    return _slow_ms > 0 and time.time() < _slow_until


@app.get("/slow")
async def slow(ms: int = 500, duration_s: float = 30.0):
    """
    Make this backend deliberately slow for a while.

    This used to sleep for the duration of a single request, which is useless
    as fault injection: it delayed one call and returned, so the router never
    saw a sustained difference between servers and error_rate_5s stayed zero.
    The sleep is now applied to every request inside the window, and the
    control call itself returns immediately.
    """
    global _slow_ms, _slow_until
    _slow_ms = max(0, ms)
    _slow_until = time.time() + max(0.0, duration_s)
    return {"id": BACKEND_ID, "slow_ms": _slow_ms, "duration_s": duration_s}


@app.get("/metrics")
async def metrics():
    """Internal metrics endpoint consumed by the balancer every second."""
    window = _window_events()
    latencies = [ev[1] for ev in window]
    errors = sum(1 for ev in window if ev[2] >= 500)

    body = {
        "backend_id": BACKEND_ID,
        "timestamp": time.time(),
        "uptime_s": round(time.time() - _started_at, 1),
        "cpu_pct": round(_cpu_pct_sample(), 2),
        "mem_pct": round(_read_mem_pct(), 2),
        "load_avg": round(_read_load_avg(), 3),
        "net_io_bytes_per_s": round(_net_io_bytes_per_s(), 1),
        "net_io_bytes_recv": 0.0,
        "net_io_bytes_sent": 0.0,
        "inflight": _inflight,
        "queue_depth": cpu_queue_depth(),
        "req_rate_5s": round(len(window) / WINDOW_S, 3),
        "rt_ewma_ms": round(_ewma_ms, 3),
        "rt_p95_5s_ms": round(_percentile(latencies, 95), 3),
        "error_rate_5s": round(errors / len(window), 4) if window else 0.0,
        "cpu_limit": CPU_LIMIT,
        "cpu_workers": CPU_WORKERS,
        "mem_limit_mb": MEM_LIMIT_MB,
        "total_requests": _total_requests,
        "total_errors": _total_errors,
    }
    response = JSONResponse(content=body)
    response.headers["cache-control"] = "no-store"
    return response


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request, path: str):
    """Catch-all for unknown routes."""
    result = await _run_cpu(CPU_WORK_ITERATIONS // 8)
    return {"id": BACKEND_ID, "path": path, "result": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

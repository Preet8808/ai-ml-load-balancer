import os
import time
import pytest
import asyncio
import pyarrow.parquet as pq
from balancer.interfaces import ServerInfo, ServerMetrics
from balancer.collector import MetricsCollector
from balancer.logger import RequestLogger

@pytest.mark.asyncio
async def test_request_logger(tmp_path):
    log_dir = str(tmp_path / "logs")
    logger = RequestLogger(log_dir=log_dir, run_id="test_run", buffer_size=2, flush_interval=0.1)
    await logger.start()
    await logger.start_run("test_run")

    entry1 = {
        "timestamp": time.time(),
        "req_type": "light",
        "payload_kb": 0.0,
        "server_id": "backend1",
        "srv_cpu_pct": 10.0,
        "srv_mem_pct": 20.0,
        "inflight": 1,
        "req_rate_5s": 5.0,
        "rt_ewma_ms": 15.0,
        "rt_p95_5s_ms": 25.0,
        "err_rate_5s": 0.0,
        "cpu_limit": 1.0,
        "cpu_delta_5s": 0.5,
        "response_ms": 12.0,
        "status": 200
    }

    await logger.log(entry1)
    await logger.log(entry1)  # Triggers buffer flush
    await asyncio.sleep(0.2)
    await logger.stop()

    files = os.listdir(log_dir)
    assert len(files) == 1
    assert files[0].endswith(".parquet")

    # Read back using pyarrow
    table = pq.read_table(os.path.join(log_dir, files[0]))
    assert table.num_rows == 2
    assert "response_ms" in table.column_names


@pytest.mark.asyncio
async def test_run_rotation_separates_run_ids(tmp_path):
    """
    Each run must land in its own readable file with its own run_id.

    A single writer for the whole process produced corrupt copies and gave
    every row the same run_id, which silently invalidates run-grouped CV.
    """
    log_dir = str(tmp_path / "logs")
    logger = RequestLogger(log_dir=log_dir, buffer_size=2, flush_interval=0.1)
    await logger.start()

    def make_entry(run_ts, server):
        return {
            "timestamp": run_ts, "req_type": "cpu", "payload_kb": 0.0,
            "server_id": server, "srv_cpu_pct": 10.0, "srv_mem_pct": 20.0,
            "inflight": 1, "req_rate_5s": 5.0, "rt_ewma_ms": 15.0,
            "rt_p95_5s_ms": 25.0, "err_rate_5s": 0.0, "cpu_limit": 1.0,
            "cpu_delta_5s": 0.5, "response_ms": 12.0, "status": 200,
        }

    await logger.start_run("run_a")
    for i in range(4):
        await logger.log(make_entry(1000.0 + i, "backend1"))

    # Rotating seals run_a and starts run_b
    info = await logger.start_run("run_b")
    assert info["previous"]["run_id"] == "run_a"
    assert info["previous"]["rows"] == 4

    for i in range(6):
        await logger.log(make_entry(2000.0 + i, "backend2"))

    await logger.stop()

    files = sorted(f for f in os.listdir(log_dir) if f.endswith(".parquet"))
    assert len(files) == 2

    run_ids = {}
    for name in files:
        table = pq.read_table(os.path.join(log_dir, name))
        ids = set(table.column("run_id").to_pylist())
        assert len(ids) == 1, "each file must hold exactly one run_id"
        run_ids[table.column("run_id").to_pylist()[0]] = table.num_rows

    assert run_ids == {"run_a": 4, "run_b": 6}


@pytest.mark.asyncio
async def test_sealed_file_is_immediately_readable(tmp_path):
    """A sealed file must be valid on disk without waiting for shutdown."""
    log_dir = str(tmp_path / "logs")
    logger = RequestLogger(log_dir=log_dir, buffer_size=100, flush_interval=100)
    await logger.start()
    await logger.start_run("run_seal")

    for i in range(3):
        await logger.log({
            "timestamp": 3000.0 + i, "req_type": "io", "payload_kb": 0.0,
            "server_id": "backend4", "srv_cpu_pct": 80.0, "srv_mem_pct": 30.0,
            "inflight": 2, "req_rate_5s": 7.0, "rt_ewma_ms": 40.0,
            "rt_p95_5s_ms": 60.0, "err_rate_5s": 0.0, "cpu_limit": 0.5,
            "cpu_delta_5s": 1.0, "response_ms": 41.0, "status": 200,
        })

    # Data is still buffered, not on disk yet
    sealed = await logger.seal()
    assert sealed["rows"] == 3

    path = sealed["path"]
    assert pq.read_table(path).num_rows == 3

    await logger.stop()


@pytest.mark.asyncio
async def test_requests_outside_a_run_are_dropped_not_mislabelled(tmp_path):
    log_dir = str(tmp_path / "logs")
    logger = RequestLogger(log_dir=log_dir, buffer_size=1, flush_interval=100)
    await logger.start()

    await logger.log({
        "timestamp": 1.0, "req_type": "light", "payload_kb": 0.0,
        "server_id": "backend1", "srv_cpu_pct": 0.0, "srv_mem_pct": 0.0,
        "inflight": 0, "req_rate_5s": 0.0, "rt_ewma_ms": 0.0,
        "rt_p95_5s_ms": 0.0, "err_rate_5s": 0.0, "cpu_limit": 1.0,
        "cpu_delta_5s": 0.0, "response_ms": 1.0, "status": 200,
    })

    assert logger.status()["dropped"] == 1
    await logger.stop()

@pytest.mark.asyncio
async def test_collector_health_state_machine():
    from balancer.interfaces import ServerMetrics

    metrics = ServerMetrics(
        server_id="s1", timestamp=time.time(),
        cpu_pct=10.0, mem_pct=20.0, load_avg=0.5,
        net_io_bytes_per_s=1000.0, inflight_requests=0,
        req_rate_5s=5.0, rt_ewma_ms=15.0, rt_p95_5s_ms=25.0,
        error_rate_5s=0.0, cpu_limit=1.0, cpu_delta_5s=0.0,
        healthy=True, stale=False,
    )
    server = ServerInfo(
        server_id="s1",
        host="localhost",
        port=9999,  # Unreachable port
        weight=1.0,
        cpu_limit=1.0,
        mem_limit_mb=512,
        metrics=metrics,
    )
    assert server.healthy is True

    collector = MetricsCollector([server], poll_interval=0.1)

    # 1 failure
    await collector._handle_failure(server)
    assert server.health_check_failures == 1
    assert server.metrics.healthy is True

    # 2 failures
    await collector._handle_failure(server)
    assert server.health_check_failures == 2
    assert server.metrics.healthy is True

    # 3 failures -> metrics marked unhealthy
    await collector._handle_failure(server)
    assert server.health_check_failures == 3
    assert server.metrics.healthy is False
    assert server.healthy is False

    await collector.stop()

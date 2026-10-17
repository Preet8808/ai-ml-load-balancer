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

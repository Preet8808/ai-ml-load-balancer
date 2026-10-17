"""Tests for Phase 3: Dataset generation components."""

import sys
sys.path.insert(0, 'C:/Users/DELL/projects/ai_ml_load_balancer')

import asyncio
import os
import time
import pandas as pd
import pyarrow.parquet as pq
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from bench.probe_mode import ProbeRunner, ProbeResult
from bench.data_quality import (
    load_all_parquet,
    check_schema,
    check_nans,
    compute_stats,
    run_quality_check,
    EXPECTED_COLUMNS,
)


# ---------------------------------------------------------------------------
# ProbeResult dataclass
# ---------------------------------------------------------------------------

def test_probe_result_fields():
    r = ProbeResult(
        timestamp=1000.0,
        server_id="s1",
        response_ms=12.5,
        status=200,
        cpu_pct=30.0,
        mem_pct=40.0,
        inflight=2,
        cpu_limit=1.0,
    )
    assert r.server_id == "s1"
    assert r.response_ms == 12.5
    assert r.status == 200


# ---------------------------------------------------------------------------
# ProbeRunner — mocked HTTP
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_probe_runner_collects_results(tmp_path):
    """ProbeRunner should collect one result per server per round."""
    servers = [
        {"server_id": "s1", "host": "localhost", "port": 8001, "cpu_limit": 2.0},
        {"server_id": "s2", "host": "localhost", "port": 8002, "cpu_limit": 1.0},
    ]
    runner = ProbeRunner(servers, output_dir=str(tmp_path))

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"cpu_pct": 10.0, "mem_pct": 20.0, "inflight": 0}
    mock_client.get = AsyncMock(return_value=mock_response)

    runner._client = mock_client

    results = await runner.run_probe(rounds=3, interval=0.01)

    assert len(results) == 6  # 3 rounds × 2 servers
    assert all(r["status"] == 200 for r in results)
    assert all(r["response_ms"] >= 0 for r in results)

    await runner.close()


@pytest.mark.asyncio
async def test_probe_runner_handles_failure(tmp_path):
    """ProbeRunner should record timeout penalty on connection failure."""
    servers = [
        {"server_id": "s1", "host": "localhost", "port": 9999, "cpu_limit": 1.0},
    ]
    runner = ProbeRunner(servers, output_dir=str(tmp_path))

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
    runner._client = mock_client

    results = await runner.run_probe(rounds=1, interval=0.01)

    assert len(results) == 1
    assert results[0]["status"] == 0
    assert results[0]["response_ms"] == 5000.0  # timeout penalty

    await runner.close()


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------

def test_check_schema_pass(tmp_path):
    """Schema check passes when all expected columns are present."""
    df = pd.DataFrame({col: [0.0] for col in EXPECTED_COLUMNS})
    result = check_schema(df)
    assert result["schema_ok"] is True
    assert result["missing_columns"] == []


def test_check_schema_fail_missing_columns():
    """Schema check fails when columns are missing."""
    df = pd.DataFrame({"timestamp": [0.0], "server_id": ["s1"]})
    result = check_schema(df)
    assert result["schema_ok"] is False
    assert len(result["missing_columns"]) > 0


def test_check_nans_pass():
    """NaN check passes when no NaNs."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    result = check_nans(df)
    assert result["has_nans"] is False


def test_check_nans_fail():
    """NaN check detects NaN values."""
    df = pd.DataFrame({"a": [1.0, float("nan")], "b": [3.0, 4.0]})
    result = check_nans(df)
    assert result["has_nans"] is True
    assert "a" in result["columns_with_nans"]


def test_compute_stats_basic():
    """Compute stats returns expected fields."""
    df = pd.DataFrame({
        "timestamp": [1.0, 2.0, 3.0],
        "run_id": ["r1", "r1", "r2"],
        "req_type": ["light", "cpu", "light"],
        "payload_kb": [0.0, 2.0, 0.0],
        "server_id": ["s1", "s2", "s1"],
        "srv_cpu_pct": [10.0, 20.0, 30.0],
        "srv_mem_pct": [20.0, 30.0, 40.0],
        "inflight": [0, 1, 2],
        "req_rate_5s": [5.0, 10.0, 15.0],
        "rt_ewma_ms": [10.0, 20.0, 30.0],
        "rt_p95_5s_ms": [20.0, 40.0, 60.0],
        "err_rate_5s": [0.0, 0.0, 0.0],
        "cpu_limit": [1.0, 2.0, 1.0],
        "cpu_delta_5s": [0.0, 1.0, -1.0],
        "response_ms": [10.0, 25.0, 15.0],
        "status": [200, 200, 200],
    })
    stats = compute_stats(df)
    assert stats["total_rows"] == 3
    assert stats["unique_servers"] == 2
    assert "light" in stats["req_type_counts"]
    assert stats["response_ms"]["mean"] == pytest.approx(16.67, rel=0.01)


def test_load_all_parquet_empty(tmp_path):
    """Loading from empty directory returns empty DataFrame."""
    df = load_all_parquet(str(tmp_path))
    assert df.empty


def test_load_all_parquet_roundtrip(tmp_path):
    """Loading Parquet files returns concatenated data."""
    df1 = pd.DataFrame({"timestamp": [1.0], "server_id": ["s1"], "response_ms": [10.0]})
    df2 = pd.DataFrame({"timestamp": [2.0], "server_id": ["s2"], "response_ms": [20.0]})
    df1.to_parquet(str(tmp_path / "f1.parquet"), index=False)
    df2.to_parquet(str(tmp_path / "f2.parquet"), index=False)

    loaded = load_all_parquet(str(tmp_path))
    assert len(loaded) == 2
    assert set(loaded["server_id"]) == {"s1", "s2"}


def test_expected_columns_count():
    """Verify we have exactly 16 columns matching the plan's schema."""
    assert len(EXPECTED_COLUMNS) == 16

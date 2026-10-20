"""
Tests for the dashboard's demo-control proxy.

The dashboard buttons only call these admin endpoints. Keeping the parameter
mapping here tested means a UI change cannot silently invert the slow/fault
settings, and the backend control ports stay private.
"""

import pytest
from fastapi import HTTPException

from balancer import main
from balancer.interfaces import ServerInfo


def _backend3() -> ServerInfo:
    return ServerInfo(server_id="backend3", host="backend3", port=8000)


@pytest.fixture
def captured(monkeypatch):
    calls = []

    async def fake_control(target, path, params):
        calls.append((target.server_id, path, params))
        return {"path": path, "request": params, "response": {}}

    monkeypatch.setattr(main, "SERVERS", [_backend3()])
    monkeypatch.setattr(main, "_backend_control", fake_control)
    return calls


@pytest.mark.asyncio
async def test_degrade_sets_slow_and_fault_for_the_same_window(captured):
    result = await main.demo_degrade(
        server_id="backend3", duration_s=45.0, delay_ms=400, error_rate=0.35
    )

    assert result["status"] == "ok"
    assert [path for _, path, _ in captured] == ["/slow", "/fault"]
    assert captured[0][2] == {"ms": 400, "duration_s": 45.0}
    assert captured[1][2] == {"error_rate": 0.35, "duration_s": 45.0}


@pytest.mark.asyncio
async def test_fail_is_an_error_fault_without_a_delay(captured):
    await main.demo_fail(server_id="backend3", duration_s=12.0)

    assert captured[0][1:] == ("/slow", {"ms": 0, "duration_s": 0})
    assert captured[1][1:] == (
        "/fault",
        {"error_rate": 1.0, "duration_s": 12.0},
    )


@pytest.mark.asyncio
async def test_clear_resets_both_controls(captured):
    await main.demo_clear(server_id="backend3")

    assert captured[0][1:] == ("/slow", {"ms": 0, "duration_s": 0})
    assert captured[1][1:] == (
        "/fault",
        {"error_rate": 0.0, "duration_s": 0},
    )


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "SERVERS", [_backend3()])

    with pytest.raises(HTTPException) as exc:
        main._demo_server("backend9")

    assert exc.value.status_code == 404

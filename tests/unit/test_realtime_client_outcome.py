"""Unit tests for BroadcastService outcome derivation.

The regression these lock in: a publish that coding-realtime REJECTED used to
be indistinguishable from one it accepted. The caller is fire-and-forget by
design, so nothing propagated — a 404 (the ownership check refusing to resolve
the container) left no trace at all. ``ok`` must be DERIVED from the response,
a 404 must be separately identifiable, and every non-2xx must be logged with
its status code and container name.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

import src.clients.realtime_client as realtime_mod
from coding_client.clients import ServiceError
from src.clients.realtime_client import (
    BroadcastEventType,
    BroadcastResult,
    BroadcastService,
)


@pytest.fixture(autouse=True)
def _reset_realtime_singleton():
    realtime_mod._http_client = None
    realtime_mod._instance = None
    yield
    realtime_mod._http_client = None
    realtime_mod._instance = None


def _stub_client(monkeypatch, *, result=None, error: Exception | None = None) -> AsyncMock:
    """Replace the module's HTTP client with a stub."""
    client = AsyncMock()
    if error is not None:
        client.broadcast.side_effect = error
    else:
        client.broadcast.return_value = result
    monkeypatch.setattr(realtime_mod, "_get_http_client", lambda: client)
    return client


class TestBroadcastOutcomeIsDerived:
    @pytest.mark.asyncio
    async def test_404_is_flagged_not_found_and_logged_at_error(self, monkeypatch, caplog):
        """A 404 is the ownership check rejecting the event. It must be
        distinguishable from any other failure and must be loud."""
        _stub_client(monkeypatch, error=ServiceError(404, "Container not found"))
        svc = BroadcastService()

        with caplog.at_level(logging.DEBUG):
            result = await svc.broadcast(
                BroadcastEventType.CONTAINER_STARTED,
                {"container_name": "codegraph-repo", "status": "creating"},
                tenant_id="tenant-9",
            )

        assert result.ok is False
        assert result.status_code == 404
        assert result.not_found is True

        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, [r.getMessage() for r in caplog.records]
        msg = errors[0].getMessage()
        assert "404" in msg
        assert "codegraph-repo" in msg

    @pytest.mark.asyncio
    async def test_non_2xx_is_logged_with_status_and_container(self, monkeypatch, caplog):
        _stub_client(monkeypatch, error=ServiceError(503, "upstream down"))
        svc = BroadcastService()

        with caplog.at_level(logging.DEBUG):
            result = await svc.broadcast(
                BroadcastEventType.CONTAINER_HEALTH,
                {"container_name": "codegraph-repo"},
                tenant_id="tenant-9",
            )

        assert result.ok is False
        assert result.status_code == 503
        assert result.not_found is False

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, [r.getMessage() for r in caplog.records]
        msg = warnings[0].getMessage()
        assert "503" in msg
        assert "codegraph-repo" in msg

    @pytest.mark.asyncio
    async def test_transport_exception_is_logged_and_never_raises(self, monkeypatch, caplog):
        """Fire-and-forget contract, TRANSPORT half: coding-realtime being
        unreachable must not break the caller's operation. (The other half —
        programming errors DO propagate — is pinned in
        ``test_realtime_client_contract.py``.)"""
        _stub_client(monkeypatch, error=OSError("connection refused"))
        svc = BroadcastService()

        with caplog.at_level(logging.DEBUG):
            result = await svc.broadcast(
                BroadcastEventType.CONTAINER_STOPPED,
                {"container_name": "codegraph-repo"},
                tenant_id="tenant-9",
            )

        assert result.ok is False
        assert result.status_code is None
        assert result.not_found is False
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings
        assert "codegraph-repo" in warnings[0].getMessage()

    @pytest.mark.asyncio
    async def test_success_derives_delivery_from_the_response_body(self, monkeypatch):
        _stub_client(monkeypatch, result={"delivered": True, "subscriber_count": 3})
        svc = BroadcastService()

        result = await svc.broadcast(
            BroadcastEventType.CONTAINER_STARTED,
            {"container_name": "codegraph-repo"},
            tenant_id="tenant-9",
        )

        assert result == BroadcastResult(
            ok=True, status_code=200, delivered=True, subscriber_count=3
        )

    @pytest.mark.asyncio
    async def test_accepted_but_undelivered_is_ok_but_reported(self, monkeypatch):
        """delivered=false is a successful publish nobody was listening for —
        not a failure, but it must be visible rather than assumed-success."""
        _stub_client(monkeypatch, result={"delivered": False, "subscriber_count": 0})
        svc = BroadcastService()

        result = await svc.broadcast(
            BroadcastEventType.CONTAINER_STARTED,
            {"container_name": "codegraph-repo"},
            tenant_id="tenant-9",
        )

        assert result.ok is True
        assert result.delivered is False
        assert result.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_unparseable_body_does_not_fabricate_delivery(self, monkeypatch):
        """No body fields -> unknown, not an invented True."""
        _stub_client(monkeypatch, result="ok")
        svc = BroadcastService()

        result = await svc.broadcast(
            BroadcastEventType.ANALYSIS_PROGRESS,
            {"container_name": "codegraph-repo"},
            tenant_id="tenant-9",
        )

        assert result.ok is True
        assert result.delivered is None
        assert result.subscriber_count is None

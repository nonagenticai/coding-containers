"""Contract tests for the container-lifecycle broadcast path.

Three stacked defects made this path 100% dead in production while looking
healthy, because each one was swallowed:

1. ``client.broadcast(json={...})`` — coding-client 1.7.x's ``broadcast`` is
   keyword-only (``event_type`` / ``data`` / ``priority`` / ``headers``), so the
   call raised ``TypeError`` before a byte left the pod.
2. A blanket ``except Exception`` turned that ``TypeError`` into a log line
   nobody read.
3. The local ``BroadcastEventType`` declared values coding-realtime's enum does
   not contain, so fixing (1) alone would only have converted a silent
   ``TypeError`` into a silent ``422``.

Plus: the call carried no ``X-Tenant-Id``, so it only reached coding-realtime
through that service's pod-CIDR trust fallback, which is being removed.

Every test here is written so it FAILS on the pre-fix code, and the
programming-error tests use the real ``BroadcastService.broadcast`` against a
stub client whose signature is the real client's — so a signature regression
cannot pass vacuously.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

import src.clients.realtime_client as realtime_mod
from coding_client.auth.tenant import TENANT_ID_HEADER
from coding_client.clients import CodingRealtimeClient, ServiceError
from src.clients.realtime_client import (
    BroadcastEventType,
    BroadcastService,
    BroadcastTenantMissingError,
)

#: The event types coding-realtime's ``BroadcastEventType``
#: (coding-realtime/src/services/websocket_broadcast.py) actually accepts on
#: ``POST /api/broadcast/publish``. Anything outside this set is answered
#: ``422 Invalid event_type``; verified live against the running service.
CODING_REALTIME_CONTAINER_EVENT_TYPES = frozenset(
    {
        "container_started",
        "container_stopped",
        "container_health",
        "container_resource_update",
        "analysis_progress",
    }
)


@pytest.fixture(autouse=True)
def _reset_realtime_singleton():
    realtime_mod._http_client = None
    realtime_mod._instance = None
    yield
    realtime_mod._http_client = None
    realtime_mod._instance = None


def _stub_client(monkeypatch, *, result: Any = None, error: Exception | None = None) -> AsyncMock:
    """A stub whose ``broadcast`` has the REAL client's signature.

    ``spec=`` on the method means a call with a kwarg the real
    ``CodingRealtimeClient.broadcast`` does not accept raises ``TypeError``
    exactly as it does in the pod — which is what makes the wrong-signature test
    a real test rather than a mock that accepts anything.
    """
    client = AsyncMock()
    client.broadcast = AsyncMock(spec=CodingRealtimeClient.broadcast)
    if error is not None:
        client.broadcast.side_effect = error
    else:
        client.broadcast.return_value = result
    monkeypatch.setattr(realtime_mod, "_get_http_client", lambda: client)
    return client


class TestEventTypesMatchCodingRealtime:
    """Defect 3: an event type this service can emit but coding-realtime rejects
    is a silent 422 — the failure mode fixing the signature alone would have
    produced."""

    def test_every_emittable_event_type_is_accepted_by_coding_realtime(self):
        emitted = {member.value for member in BroadcastEventType}
        unknown = emitted - CODING_REALTIME_CONTAINER_EVENT_TYPES
        assert not unknown, (
            f"{sorted(unknown)} are not in coding-realtime's BroadcastEventType; "
            "publishing them returns 422 Invalid event_type and the event is dropped. "
            "Either remove them here or add them to coding-realtime FIRST."
        )

    @pytest.mark.parametrize(
        "orphan", ["container_status", "container_created", "container_removed"]
    )
    def test_the_three_drifted_values_are_gone(self, orphan):
        """These were declared locally, are in no coding-realtime enum, and 422
        on the wire. Re-adding one must fail this test, not production."""
        assert orphan not in {member.value for member in BroadcastEventType}


class TestCallSignatureIsTheRealOne:
    """Defect 1: the shipped call used a kwarg the client does not take."""

    def test_installed_client_broadcast_does_not_accept_json(self):
        params = inspect.signature(CodingRealtimeClient.broadcast).parameters
        assert "json" not in params
        for required in ("event_type", "data", "priority", "headers"):
            assert required in params

    @pytest.mark.asyncio
    async def test_broadcast_calls_the_client_with_the_supported_kwargs(self, monkeypatch):
        client = _stub_client(monkeypatch, result={"delivered": True, "subscriber_count": 1})

        result = await BroadcastService().broadcast(
            BroadcastEventType.CONTAINER_STARTED,
            {"container_name": "codegraph-repo", "status": "creating"},
            tenant_id="tenant-7",
        )

        assert result.ok is True
        kwargs = client.broadcast.await_args.kwargs
        assert "json" not in kwargs
        assert kwargs["event_type"] == "container_started"
        assert kwargs["data"]["container_name"] == "codegraph-repo"


class TestProgrammingErrorsAreNotSwallowed:
    """Defect 2: the bare ``except (ServiceError, Exception)`` is why nobody
    noticed. Transport failures stay fire-and-forget; bugs propagate."""

    @pytest.mark.asyncio
    async def test_wrong_signature_typeerror_propagates(self, monkeypatch):
        """Simulates the shipped bug: the client rejects the call outright."""
        _stub_client(
            monkeypatch,
            error=TypeError(
                "CodingRealtimeClient.broadcast() got an unexpected keyword argument 'json'"
            ),
        )

        with pytest.raises(TypeError):
            await BroadcastService().broadcast(
                BroadcastEventType.CONTAINER_STARTED,
                {"container_name": "codegraph-repo"},
                tenant_id="tenant-7",
            )

    @pytest.mark.asyncio
    async def test_unexpected_bug_propagates(self, monkeypatch):
        _stub_client(monkeypatch, error=AttributeError("boom"))

        with pytest.raises(AttributeError):
            await BroadcastService().broadcast(
                BroadcastEventType.CONTAINER_HEALTH,
                {"container_name": "codegraph-repo"},
                tenant_id="tenant-7",
            )

    @pytest.mark.asyncio
    async def test_transport_failure_is_still_swallowed_but_logged_at_error(
        self, monkeypatch, caplog
    ):
        """Negative control for the two tests above: the fire-and-forget
        guarantee for real delivery failures is intact, and loud."""
        _stub_client(monkeypatch, error=httpx.ConnectError("no route to host"))

        with caplog.at_level(logging.DEBUG):
            result = await BroadcastService().broadcast(
                BroadcastEventType.CONTAINER_STARTED,
                {"container_name": "codegraph-repo"},
                tenant_id="tenant-7",
            )

        assert result.ok is False
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, [r.getMessage() for r in caplog.records]
        assert "codegraph-repo" in errors[0].getMessage()
        assert errors[0].exc_info is not None, "the exception itself must be logged"

    @pytest.mark.asyncio
    async def test_service_error_is_still_swallowed(self, monkeypatch):
        """A non-2xx from coding-realtime is reported, not raised."""
        _stub_client(monkeypatch, error=ServiceError(503, "upstream down"))

        result = await BroadcastService().broadcast(
            BroadcastEventType.CONTAINER_STARTED,
            {"container_name": "codegraph-repo"},
            tenant_id="tenant-7",
        )
        assert result.ok is False
        assert result.status_code == 503


class TestTenantIsRequiredAndSent:
    """Defect 4: no ``X-Tenant-Id`` meant the call only got through on
    coding-realtime's source-IP trust, which is being turned off."""

    @pytest.mark.asyncio
    async def test_owning_tenant_is_sent_as_the_header(self, monkeypatch):
        client = _stub_client(monkeypatch, result={"delivered": False, "subscriber_count": 0})

        await BroadcastService().broadcast(
            BroadcastEventType.CONTAINER_STARTED,
            {"container_name": "codegraph-repo"},
            tenant_id="0195f48b-d000-7000-8000-000000000003",
        )

        headers = client.broadcast.await_args.kwargs["headers"]
        assert headers[TENANT_ID_HEADER] == "0195f48b-d000-7000-8000-000000000003"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("missing", [None, ""])
    async def test_missing_tenant_raises_and_sends_nothing(self, monkeypatch, missing):
        """The important half: a tenant-less broadcast must NOT quietly go out
        header-less (the old behaviour), and must NOT be defaulted."""
        client = _stub_client(monkeypatch, result={"delivered": True})

        with pytest.raises(BroadcastTenantMissingError):
            await BroadcastService().broadcast(
                BroadcastEventType.CONTAINER_STARTED,
                {"container_name": "codegraph-repo"},
                tenant_id=missing,
            )

        client.broadcast.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tenant_is_keyword_only_so_it_cannot_be_passed_by_accident(self):
        params = inspect.signature(BroadcastService.broadcast).parameters
        assert params["tenant_id"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["tenant_id"].default is inspect.Parameter.empty, (
            "tenant_id must have NO default: a default is what let every call site "
            "omit it and silently publish without a tenant header."
        )

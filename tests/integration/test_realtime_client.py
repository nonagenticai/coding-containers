"""Integration tests for realtime_client wired to coding-realtime service."""

import json

import pytest
import respx
from httpx import Response

import src.clients.realtime_client as realtime_mod
from src.clients.realtime_client import (
    BroadcastEventType,
    BroadcastPriority,
    BroadcastService,
    get_broadcast_service,
)

# NOTE: deliberately NOT marked `integration`. Every test here drives the client
# against respx mocks and needs no live service; the marker excluded the whole
# file from CI (pyproject addopts carry `-m 'not integration'`).
# tests/integration/test_e2e_endpoints.py keeps its markers — those do need a
# running stack.

pytestmark = pytest.mark.asyncio

#: The OWNING tenant of the container an event is about. Required — the
#: broadcast is authorised against it, so it is never defaulted.
TEST_TENANT = "0195f48b-d000-7000-8000-000000000003"


@pytest.fixture(autouse=True)
def _reset_realtime_singleton():
    """Reset module-level singletons between tests."""
    realtime_mod._http_client = None
    realtime_mod._instance = None
    yield
    realtime_mod._http_client = None
    realtime_mod._instance = None


async def test_broadcast_sends_correct_payload(mock_realtime_service):
    """broadcast POSTs a FLAT body to /api/broadcast/publish (D9): the container
    is pulled from data['container_name'], no more {"json": ...} double-wrap, and
    it hits the publish route, not the WebSocket-only /subscribe."""
    svc = BroadcastService()
    await svc.broadcast(
        BroadcastEventType.CONTAINER_STARTED,
        {"container_name": "test-ctr"},
        tenant_id=TEST_TENANT,
    )

    broadcast_calls = [
        c for c in mock_realtime_service.calls if "/api/broadcast/publish" in str(c.request.url)
    ]
    assert len(broadcast_calls) == 1
    payload = json.loads(broadcast_calls[0].request.content)
    # Flat body — NOT wrapped in {"json": ...}.
    assert "json" not in payload
    assert payload["event_type"] == "container_started"
    assert payload["data"] == {"container_name": "test-ctr"}
    assert payload["priority"] == "normal"  # default priority
    assert payload["container_name"] == "test-ctr"  # resolved from data


async def test_broadcast_forwards_tenant_header(mock_realtime_service):
    """The provisioning tenant is forwarded as X-Tenant-Id so the publish passes
    the server's BOLA check."""
    svc = BroadcastService()
    await svc.broadcast(
        BroadcastEventType.CONTAINER_STARTED,
        {"container_name": "c1"},
        tenant_id="tenant-42",
    )
    call = next(
        c for c in mock_realtime_service.calls if "/api/broadcast/publish" in str(c.request.url)
    )
    assert call.request.headers.get("X-Tenant-Id") == "tenant-42"


async def test_broadcast_with_high_priority(mock_realtime_service):
    """broadcast should send the specified priority."""
    svc = BroadcastService()
    await svc.broadcast(
        BroadcastEventType.CONTAINER_HEALTH,
        {"status": "running", "container_name": "c1"},
        BroadcastPriority.HIGH,
        tenant_id=TEST_TENANT,
    )

    broadcast_calls = [
        c for c in mock_realtime_service.calls if "/api/broadcast/publish" in str(c.request.url)
    ]
    assert len(broadcast_calls) == 1
    payload = json.loads(broadcast_calls[0].request.content)
    assert "json" not in payload
    assert payload["priority"] == "high"
    assert payload["event_type"] == "container_health"


async def test_broadcast_does_not_raise_on_service_unavailable():
    """broadcast should silently handle connection errors (fire-and-forget)."""
    svc = BroadcastService()
    # No mock = connection refused, should not raise
    result = await svc.broadcast(
        BroadcastEventType.CONTAINER_STOPPED,
        {"container_name": "gone"},
        tenant_id=TEST_TENANT,
    )
    # Fire-and-forget holds for TRANSPORT failures, and the outcome is reported
    # rather than assumed.
    assert result.ok is False


async def test_broadcast_does_not_raise_on_http_error():
    """broadcast should silently handle HTTP 500 errors."""
    with respx.mock(base_url="http://coding-realtime:8694", assert_all_called=False) as mock:
        mock.post("/api/broadcast/publish").mock(return_value=Response(500, json={"error": "boom"}))
        svc = BroadcastService()
        result = await svc.broadcast(
            BroadcastEventType.ANALYSIS_PROGRESS,
            {"container_name": "c1", "pct": 50},
            tenant_id=TEST_TENANT,
        )
    assert result.ok is False
    assert result.status_code == 500


async def test_broadcast_event_type_values():
    """BroadcastEventType values must match coding-realtime's enum.

    This test used to assert container_status / container_created /
    container_removed — three values coding-realtime has never accepted (live:
    `422 Invalid event_type`). It was pinning the drift that made this path
    undeliverable, so it now asserts the values that are genuinely valid.
    """
    assert BroadcastEventType.CONTAINER_STARTED == "container_started"
    assert BroadcastEventType.CONTAINER_STOPPED == "container_stopped"
    assert BroadcastEventType.CONTAINER_HEALTH == "container_health"
    assert BroadcastEventType.ANALYSIS_PROGRESS == "analysis_progress"
    assert not hasattr(BroadcastEventType, "CONTAINER_STATUS")
    assert not hasattr(BroadcastEventType, "CONTAINER_CREATED")
    assert not hasattr(BroadcastEventType, "CONTAINER_REMOVED")


async def test_broadcast_priority_values():
    """BroadcastPriority values should match expected strings."""
    assert BroadcastPriority.LOW == "low"
    assert BroadcastPriority.NORMAL == "normal"
    assert BroadcastPriority.HIGH == "high"


def test_get_broadcast_service_returns_singleton():
    """get_broadcast_service should return the same instance."""
    s1 = get_broadcast_service()
    s2 = get_broadcast_service()
    assert s1 is s2

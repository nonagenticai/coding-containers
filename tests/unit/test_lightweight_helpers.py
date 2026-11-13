from __future__ import annotations

import httpx
import pytest

import src.clients.realtime_client as realtime_client_module
from src.clients.github_client import GitHubTokenManager
from src.clients.realtime_client import (
    BroadcastEventType,
    BroadcastPriority,
    BroadcastService,
)
from src.utils.cache_utils import generate_container_cache_key, get_container_cache_prefix


def test_generate_container_cache_key_is_stable_and_scoped():
    key = generate_container_cache_key(
        "repo-123",
        "quality-scan",
        tenant_id="tenant-a",
        language="python",
        path="/workspace",
    )

    assert key.startswith("tenant-a:repo-123:quality-scan:")
    assert len(key.split(":")[-1]) == 8
    assert get_container_cache_prefix("tenant-a", "repo-123") == "tenant-a:repo-123:"


def test_github_token_manager_reads_owner_env_and_holds_no_pat_reader(monkeypatch):
    manager = GitHubTokenManager()

    monkeypatch.setenv("GITHUB_OWNER", "owner-1")
    assert manager.get_system_owner() == "owner-1"

    monkeypatch.delenv("GITHUB_OWNER", raising=False)
    monkeypatch.setenv("GITHUB_SYSTEM_OWNER", "system-owner")
    assert manager.get_system_owner() == "system-owner"

    # The org PAT readers are gone: no code path can reach for an env PAT.
    assert not hasattr(manager, "get_token")
    assert not hasattr(manager, "get_system_token")


@pytest.mark.asyncio
async def test_github_token_manager_store_user_token_targets_the_secrets_service():
    """The store goes to the secrets service, and a failure RAISES.

    REWRITTEN. The previous version stubbed ``CodingGitHubClient.create_webhook``
    and asserted ``None`` on failure — it encoded the wrong peer (coding-github's
    inbound webhook receiver) as the contract, so it stayed green while no token
    was ever stored. It also could not have caught the move, because the stub was
    narrower than the real call.
    """
    import httpx
    import respx

    from src.config.settings import get_settings

    base = get_settings().secrets_api_url.rstrip("/")
    manager = GitHubTokenManager()

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(f"{base}/secrets").mock(
            return_value=httpx.Response(201, json={"secret_id": "sec-real"})
        )
        assert await manager.store_user_token("u1", "tok", "c1", "tenant-1") == "sec-real"
        assert route.called

    with respx.mock(assert_all_called=True) as mock:
        mock.post(f"{base}/secrets").mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(httpx.HTTPError):
            await manager.store_user_token("u1", "tok", "c1", "tenant-1")


@pytest.mark.asyncio
async def test_broadcast_service_never_raises(monkeypatch):
    service = BroadcastService()
    payloads: list[dict] = []

    class SuccessClient:
        async def broadcast(self, **kwargs):
            # New flat contract (D9): fields are passed directly, not wrapped in
            # a `json=` kwarg. Capture the whole call for assertion.
            payloads.append(kwargs)

    class FailingClient:
        async def broadcast(self, **kwargs):
            # A REAL transport failure. It used to raise RuntimeError, which only
            # "worked" because the handler caught bare Exception — the same
            # blanket catch that hid the wrong-signature TypeError for weeks.
            raise httpx.ConnectError("realtime unavailable")

    monkeypatch.setattr(realtime_client_module, "_http_client", None)
    monkeypatch.setattr(realtime_client_module, "CodingRealtimeClient", SuccessClient)
    await service.broadcast(
        BroadcastEventType.CONTAINER_STARTED,
        {"container_name": "repo-123"},
        BroadcastPriority.HIGH,
        tenant_id="tenant-9",
    )
    assert payloads == [
        {
            "event_type": "container_started",
            "data": {"container_name": "repo-123"},
            "priority": "high",
            "container_name": "repo-123",  # resolved from data
            "headers": {"X-Tenant-Id": "tenant-9"},
        }
    ]

    monkeypatch.setattr(realtime_client_module, "_http_client", None)
    monkeypatch.setattr(realtime_client_module, "CodingRealtimeClient", FailingClient)
    result = await service.broadcast(
        BroadcastEventType.CONTAINER_HEALTH,
        {"container_name": "repo-123", "status": "running"},
        tenant_id="tenant-9",
    )
    assert result.ok is False

"""Integration tests for github_client.

Token STORAGE is driven against the secrets service (respx-mocked), not
coding-github. coding-github is still the peer for the App clone-credential
lookup, which lives in its own test module.
"""

import httpx
import pytest
import respx
from httpx import Response

import src.clients.github_client as github_mod
from src.clients.github_client import GitHubTokenManager, get_github_token_manager
from src.config.settings import get_settings
from tests.integration.conftest import GITHUB_WEBHOOK_PATH

_SECRETS_BASE = get_settings().secrets_api_url.rstrip("/")

# NOTE: deliberately NOT marked `integration`. Every test here drives the client
# against respx mocks and needs no live service; the marker excluded the whole
# file from CI (pyproject addopts carry `-m 'not integration'`).
# tests/integration/test_e2e_endpoints.py keeps its markers — those do need a
# running stack.

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_github_singleton():
    """Reset module-level singleton between tests."""
    github_mod._instance = None
    yield
    github_mod._instance = None


async def test_store_user_token_returns_the_secrets_service_secret_id() -> None:
    """The manager returns the secrets service's own ``secret_id``.

    Not a coding-github "token id" — that concept never existed outside the old
    docstring. coding-github has no store-user-token endpoint; its own
    ``GitHubTokenManager.store_user_token`` is process-local in memory and is
    never mounted on a router.
    """
    with respx.mock(base_url=_SECRETS_BASE, assert_all_called=True) as mock:
        mock.post("/secrets").mock(return_value=Response(201, json={"secret_id": "sec-abc"}))
        mgr = GitHubTokenManager()
        assert await mgr.store_user_token("user1", "tok-abc", "container1", "tenant-1") == "sec-abc"


async def test_store_user_token_no_longer_touches_the_github_webhook_receiver() -> None:
    """The raw PAT must not go to ``/api/repositories/webhooks/webhook``.

    That path is coding-github's INBOUND GitHub webhook *delivery receiver*. Any
    request to it here is the defect returning. respx raises on an unmocked host,
    so mocking ONLY the secrets host is itself the assertion — plus the explicit
    sweep over the recorded calls.
    """
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(f"{_SECRETS_BASE}/secrets").mock(
            return_value=Response(201, json={"secret_id": "sec-1"})
        )
        mgr = GitHubTokenManager()
        await mgr.store_user_token("u", "ghp_secret", "ctr", "t")

        assert route.called
        for call in mock.calls:
            assert GITHUB_WEBHOOK_PATH not in str(call.request.url), (
                f"the PAT was sent to the webhook receiver again: {call.request.url}"
            )


async def test_store_user_token_raises_on_403_instead_of_reporting_success() -> None:
    """``403 RBAC: access denied`` is the LIVE failure this path ate for its whole life.

    Until gitops #2448 added coding-containers to ``secrets-write-access``, every
    write got exactly this from the secrets waypoint. The old code caught it,
    returned ``None``, and the provisioners carried on. It must now propagate.
    """
    with respx.mock(base_url=_SECRETS_BASE, assert_all_called=True) as mock:
        mock.post("/secrets").mock(return_value=Response(403, text="RBAC: access denied"))
        mgr = GitHubTokenManager()
        with pytest.raises(httpx.HTTPStatusError):
            await mgr.store_user_token("bob", "tok-err", "ctr-6", "t")


async def test_store_user_token_raises_when_the_service_is_unreachable() -> None:
    """Connection refused is not a store. No fabricated ``token-alice-ctr-5``, and no ``None``."""
    with respx.mock(base_url=_SECRETS_BASE, assert_all_called=True) as mock:
        mock.post("/secrets").mock(side_effect=httpx.ConnectError("refused"))
        mgr = GitHubTokenManager()
        with pytest.raises(httpx.HTTPError):
            await mgr.store_user_token("alice", "tok-xyz", "ctr-5", "t")


async def test_store_user_token_raises_when_a_2xx_carries_no_secret_id() -> None:
    """A 201 with no id cannot be resolved back, so it is a failure, not a success."""
    with respx.mock(base_url=_SECRETS_BASE, assert_all_called=True) as mock:
        mock.post("/secrets").mock(return_value=Response(201, json={"status": "ok"}))
        mgr = GitHubTokenManager()
        with pytest.raises(RuntimeError, match="no secret_id"):
            await mgr.store_user_token("carol", "tok-1", "ctr-7", "t")


# ---------------------------------------------------------------------------
# resolve_provision_secret_config_id — the precedence all three provisioners use
# ---------------------------------------------------------------------------


async def test_precedence_an_existing_secret_config_id_wins_and_stores_nothing() -> None:
    """A caller-supplied id means the row already exists (fr wrote it).

    ``crud_write`` resolved ``access_token`` OUT of that row on the way in, so
    storing it again would put the same secret under a second identifier and
    double the rotation surface. Nothing must go on the wire.
    """
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{_SECRETS_BASE}/secrets")
        mgr = GitHubTokenManager()
        result = await mgr.resolve_provision_secret_config_id(
            access_token="ghp_resolved_from_that_very_row",
            existing_secret_config_id="cfg-from-fr",
            user_id="u",
            container_id="ctr",
            tenant_id="t",
        )

    assert result == "cfg-from-fr"
    assert not route.called, "re-stored a token that was already in the secrets service"


async def test_precedence_an_inline_pat_is_stored_and_its_new_id_returned() -> None:
    with respx.mock(base_url=_SECRETS_BASE, assert_all_called=True) as mock:
        mock.post("/secrets").mock(return_value=Response(201, json={"secret_id": "sec-new"}))
        mgr = GitHubTokenManager()
        result = await mgr.resolve_provision_secret_config_id(
            access_token="ghp_inline",
            existing_secret_config_id=None,
            user_id="u",
            container_id="ctr",
            tenant_id="t",
        )

    assert result == "sec-new"


@pytest.mark.parametrize("access_token", [None, "", "   "])
async def test_precedence_no_token_and_no_id_stores_nothing_and_returns_none(
    access_token: str | None,
) -> None:
    """Nothing to persist. Whitespace is not a PAT — it must not reach the service."""
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{_SECRETS_BASE}/secrets")
        mgr = GitHubTokenManager()
        result = await mgr.resolve_provision_secret_config_id(
            access_token=access_token,
            existing_secret_config_id=None,
            user_id="u",
            container_id="ctr",
            tenant_id="t",
        )

    assert result is None
    assert not route.called


async def test_precedence_propagates_a_store_failure_rather_than_returning_none() -> None:
    """A provision that cannot persist the user's token must not report success.

    This is the whole shape of the original defect: the store failed, ``None``
    came back, and ``ContainerInfo`` was built anyway.
    """
    with respx.mock(base_url=_SECRETS_BASE, assert_all_called=True) as mock:
        mock.post("/secrets").mock(return_value=Response(500, text="boom"))
        mgr = GitHubTokenManager()
        with pytest.raises(httpx.HTTPStatusError):
            await mgr.resolve_provision_secret_config_id(
                access_token="ghp_inline",
                existing_secret_config_id=None,
                user_id="u",
                container_id="ctr",
                tenant_id="t",
            )


def test_get_system_owner_reads_env(monkeypatch):
    """get_system_owner should return GITHUB_OWNER from environment."""
    monkeypatch.setenv("GITHUB_OWNER", "nonagenticai")
    mgr = GitHubTokenManager()
    assert mgr.get_system_owner() == "nonagenticai"


def test_get_system_owner_falls_back_to_system_owner(monkeypatch):
    """get_system_owner prefers GITHUB_OWNER, falls back to GITHUB_SYSTEM_OWNER."""
    monkeypatch.delenv("GITHUB_OWNER", raising=False)
    monkeypatch.setenv("GITHUB_SYSTEM_OWNER", "fallback-owner")
    mgr = GitHubTokenManager()
    assert mgr.get_system_owner() == "fallback-owner"


def test_get_github_token_manager_returns_singleton():
    """get_github_token_manager should return the same instance."""
    m1 = get_github_token_manager()
    m2 = get_github_token_manager()
    assert m1 is m2

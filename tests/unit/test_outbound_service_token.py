"""Every outbound coding-* sibling call must carry this service's own licence.

Today those calls send only a bare ``X-Tenant-Id`` and are accepted purely
because the callees still trust that header. The moment a callee sets
``TRUST_TENANT_HEADER=false`` a bare header proves nothing and every one of these
calls 401s — silently, because each of these paths is wrapped in a broad
``except`` that degrades to a plausible-looking result ("no graph to delete",
"self-heal will cover", "clone token unavailable").

So these tests assert the two halves that must BOTH hold on the wire:

1. ``Authorization: Bearer <CODING_SERVICE_TOKEN>`` — who *we* are.
2. ``X-Tenant-Id: <customer>`` — who we are acting *for*, unchanged.

Asserting (2) is not redundant. The tempting shortcut is to put the credential
into ``X-Tenant-Id`` (some fleet clients overload that header), which collapses
every customer into coding-containers' own tenant because the callee then
resolves the tenant from the token. A test that only checked "a credential went
out" would pass on exactly that bug.

The negative case matters as much: with no token configured there must be NO
``Authorization`` header at all — an empty ``Bearer `` is a malformed credential,
not "no credential", and it burns an introspection round-trip on the fleet's
hottest auth path.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

import src.clients.realtime_client as realtime_mod
import src.config.settings as settings_mod
from src.clients.realtime_client import BroadcastEventType, BroadcastService
from src.models.container_models import ContainerInfo, ContainerStatus

_TOKEN = "svc-token-abc"  # noqa: S105
_TENANT = "0195f48b-d000-7000-8000-000000000003"

_ANALYSIS_URL = "http://coding-analysis.test:8688"
_GITHUB_URL = "http://coding-github.test:8692"


@pytest.fixture(autouse=True)
def _reset_settings_singleton(monkeypatch):
    """Point the clients at fake siblings and rebuild the settings singleton.

    ``CODING_SERVICE_TOKEN`` is explicitly DELETED here rather than assumed
    absent: a developer (or a CI runner) with the real token exported would
    otherwise turn the "no token ⇒ no header" tests green for the wrong reason.
    Each positive test sets it itself.
    """
    monkeypatch.delenv("CODING_SERVICE_TOKEN", raising=False)
    monkeypatch.setenv("CODING_CONTAINERS_ENVIRONMENT", "test")
    monkeypatch.setenv("CODING_ANALYSIS_URL", _ANALYSIS_URL)
    monkeypatch.setenv("CODING_GITHUB_URL", _GITHUB_URL)
    monkeypatch.setenv("CODING_CONTAINERS_INTERNAL_SERVICE_TOKEN", "shhh")  # noqa: S105
    settings_mod._settings_instance = None
    realtime_mod._http_client = None
    yield
    settings_mod._settings_instance = None
    realtime_mod._http_client = None


def _make_container(name: str = "repo-a") -> ContainerInfo:
    return ContainerInfo(
        container_name=name,
        repo_url="https://github.com/owner/repo",
        branch="main",
        api_endpoint="",
        internal_endpoint="",
        public_endpoint="",
        codebase_path="/workspace",
        status=ContainerStatus.RUNNING,
        description="",
    )


# ---------------------------------------------------------------------------
# src/clients/_tenant.py — the SDK re-export itself
# ---------------------------------------------------------------------------


class TestTenantModuleIsTheSdkImplementation:
    """The local module must BE the SDK's, not a second copy of it.

    coding-client 1.11.0 promoted this module into ``coding_client.auth.outbound``.
    A local re-implementation would compile, pass its own tests, and quietly lack
    ``service_token=`` — which is the entire feature. Identity (``is``) is the
    assertion, because a behaviourally-similar fork is exactly the failure mode.
    """

    def test_the_helpers_are_the_sdk_objects_not_local_copies(self):
        from coding_client.auth import outbound as sdk
        from src.clients import _tenant

        assert _tenant.tenant_headers is sdk.tenant_headers
        assert _tenant.optional_tenant_headers is sdk.optional_tenant_headers
        assert _tenant.TenantMissingError is sdk.TenantMissingError
        assert _tenant.TENANT_ID_HEADER is sdk.TENANT_ID_HEADER

    def test_both_builders_accept_a_service_token(self):
        """The capability the old local implementation did not have."""
        from src.clients._tenant import optional_tenant_headers, tenant_headers

        scoped = tenant_headers("t1", operation="probe", service_token=_TOKEN)
        assert scoped == {"X-Tenant-Id": "t1", "Authorization": f"Bearer {_TOKEN}"}

        optional = optional_tenant_headers(None, service_token=_TOKEN)
        assert optional == {"Authorization": f"Bearer {_TOKEN}"}

    def test_a_missing_tenant_is_an_error_never_a_default(self):
        """The rule the old docstring existed to state; it survives the move."""
        from src.clients._tenant import TenantMissingError, tenant_headers

        with pytest.raises(TenantMissingError):
            tenant_headers(None, operation="probe", service_token=_TOKEN)
        with pytest.raises(TenantMissingError):
            tenant_headers("", operation="probe", service_token=_TOKEN)

    def test_service_token_is_read_from_the_environment_per_call(self, monkeypatch):
        """Per call, so a rotated secret needs no pod restart — and so a token
        exported after import is still seen."""
        from src.clients._tenant import service_token

        assert service_token() is None
        monkeypatch.setenv("CODING_SERVICE_TOKEN", _TOKEN)
        assert service_token() == _TOKEN
        monkeypatch.setenv("CODING_SERVICE_TOKEN", "")
        assert service_token() is None


# ---------------------------------------------------------------------------
# src/core/graph_operations.py -> coding-analysis (DELETE graph)
# ---------------------------------------------------------------------------


class TestGraphDeletionSendsTheLicence:
    _ROUTE = f"{_ANALYSIS_URL}/api/analysis/graphs/repa"

    @staticmethod
    def _ok() -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "container_name": "repa",
                "deleted": True,
                "nodes_deleted": 1,
                "edges_deleted": 1,
            },
        )

    @respx.mock
    async def test_licence_and_customer_tenant_both_go_out(self, monkeypatch):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", _TOKEN)
        from src.core.graph_operations import GraphOperations

        route = respx.delete(self._ROUTE).mock(return_value=self._ok())
        result = await GraphOperations().delete_container_graph("repa", tenant_id=_TENANT)

        assert result["success"] is True
        request = route.calls.last.request
        assert request.headers.get("authorization") == f"Bearer {_TOKEN}"
        # The tenant is the container OWNER's and must survive alongside the
        # credential — a token in X-Tenant-Id would delete against the wrong
        # tenant's graph namespace.
        assert request.headers["x-tenant-id"] == _TENANT

    @respx.mock
    async def test_no_token_means_no_authorization_header_at_all(self):
        from src.core.graph_operations import GraphOperations

        route = respx.delete(self._ROUTE).mock(return_value=self._ok())
        await GraphOperations().delete_container_graph("repa", tenant_id=_TENANT)

        request = route.calls.last.request
        assert "authorization" not in request.headers
        assert request.headers["x-tenant-id"] == _TENANT


# ---------------------------------------------------------------------------
# src/services/docker_service.py -> coding-analysis (eager deep build, DETACHED)
# ---------------------------------------------------------------------------


class TestEagerGraphBuildSendsTheLicence:
    """The detached (``asyncio.create_task``) build.

    There is no inbound request left to borrow a credential from by the time this
    runs, which is exactly why ``Authorization`` must be this service's OWN
    licence rather than a forwarded copy of anyone's.
    """

    @staticmethod
    def _session_capturing(captured: dict[str, Any]) -> MagicMock:
        resp = MagicMock()
        resp.status = 200
        resp.json = AsyncMock(
            return_value={"result": {"metadata": {"graph_nodes": 1, "graph_edges": 1}}}
        )
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)

        def _post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers")
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        session.post = _post
        return session

    async def _run(self, monkeypatch) -> dict[str, Any]:
        from src.services.docker_service import DockerService

        svc = DockerService.__new__(DockerService)
        svc.registry = MagicMock()
        svc._await_container_running = AsyncMock(return_value=True)
        captured: dict[str, Any] = {}

        with patch("aiohttp.ClientSession", return_value=self._session_capturing(captured)):
            await svc._async_provision_graph_build(_make_container(), _TENANT)
        return captured

    async def test_licence_and_customer_tenant_both_go_out(self, monkeypatch):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", _TOKEN)
        captured = await self._run(monkeypatch)

        assert captured["headers"].get("Authorization") == f"Bearer {_TOKEN}"
        assert captured["headers"]["X-Tenant-Id"] == _TENANT
        # Content-Type is not collateral damage of building the identity headers.
        assert captured["headers"]["Content-Type"] == "application/json"

    async def test_no_token_means_no_authorization_header_at_all(self, monkeypatch):
        captured = await self._run(monkeypatch)

        assert "Authorization" not in captured["headers"]
        assert captured["headers"]["X-Tenant-Id"] == _TENANT


# ---------------------------------------------------------------------------
# src/clients/realtime_client.py -> coding-realtime (lifecycle broadcast)
# ---------------------------------------------------------------------------


class TestBroadcastSendsTheLicence:
    @staticmethod
    def _stub(monkeypatch) -> AsyncMock:
        client = AsyncMock()
        client.broadcast = AsyncMock(return_value={"published": True})
        monkeypatch.setattr(realtime_mod, "_get_http_client", lambda: client)
        return client

    async def test_licence_and_owning_tenant_both_go_out(self, monkeypatch):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", _TOKEN)
        client = self._stub(monkeypatch)

        await BroadcastService().broadcast(
            BroadcastEventType.CONTAINER_STARTED,
            {"container_name": "repa"},
            tenant_id=_TENANT,
        )

        headers = client.broadcast.await_args.kwargs["headers"]
        assert headers.get("Authorization") == f"Bearer {_TOKEN}"
        # coding-realtime resolves the container against X-Tenant-Id (BOLA). A
        # token in that slot would make the ownership check pass against
        # coding-containers' own tenant instead of the container's owner.
        assert headers["X-Tenant-Id"] == _TENANT

    async def test_no_token_means_no_authorization_header_at_all(self, monkeypatch):
        client = self._stub(monkeypatch)

        await BroadcastService().broadcast(
            BroadcastEventType.CONTAINER_STARTED,
            {"container_name": "repa"},
            tenant_id=_TENANT,
        )

        headers = client.broadcast.await_args.kwargs["headers"]
        assert "Authorization" not in headers
        assert headers["X-Tenant-Id"] == _TENANT


# ---------------------------------------------------------------------------
# src/clients/github_client.py -> coding-github (App clone token)
# ---------------------------------------------------------------------------


class TestCloneTokenLookupSendsTheLicence:
    _ROUTE = f"{_GITHUB_URL}/api/internal/github/clone-token"
    _REPO = "https://github.com/owner/repo"

    @staticmethod
    def _ok() -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "token": "ghs_installtoken",
                "expires_at": "2026-07-14T12:00:00Z",
                "visibility": "private",
                "source": "app",
            },
        )

    @respx.mock
    async def test_the_licence_is_added_alongside_not_instead_of_the_internal_token(
        self, monkeypatch
    ):
        """Both credentials, not one instead of the other — and no tenant assertion.

        ``X-Internal-Token`` is what coding-github's ``require_internal_token``
        actually authorises on — dropping it 403s every clone-token lookup and
        makes every private-repo provision fail. The licence is additive: it says
        *which* internal caller this is.

        The ``x-tenant-id`` assertion this used to require is gone on purpose.
        Sending the licence turned a previously-ignored header into an evaluated
        on-behalf assertion, and coding-github's tenant middleware — which runs in
        front of ``/api/internal``, contrary to that module's own docstring —
        refuses an assertion this subject may not vouch for. Identity yes,
        assertion no.
        """
        monkeypatch.setenv("CODING_SERVICE_TOKEN", _TOKEN)
        from src.clients.github_client import GitHubTokenManager

        route = respx.post(self._ROUTE).mock(return_value=self._ok())
        cred = await GitHubTokenManager().get_app_clone_credential(self._REPO, tenant_id=_TENANT)

        assert cred.ok is True
        request = route.calls.last.request
        assert request.headers.get("authorization") == f"Bearer {_TOKEN}"
        assert request.headers["x-internal-token"] == "shhh"
        assert "x-tenant-id" not in request.headers
        assert json.loads(request.read()) == {"repo_url": self._REPO, "tenant_id": _TENANT}

    @respx.mock
    async def test_no_token_still_sends_the_internal_token(self):
        from src.clients.github_client import GitHubTokenManager

        route = respx.post(self._ROUTE).mock(return_value=self._ok())
        await GitHubTokenManager().get_app_clone_credential(self._REPO, tenant_id=_TENANT)

        request = route.calls.last.request
        assert "authorization" not in request.headers
        assert request.headers["x-internal-token"] == "shhh"

    @respx.mock
    async def test_a_tenantless_lookup_still_proves_who_is_calling(self, monkeypatch):
        """Proving who we are is not a claim about whose data we want.

        The admin path genuinely has no tenant; the header is omitted rather than
        fabricated, but the credential still goes out.
        """
        monkeypatch.setenv("CODING_SERVICE_TOKEN", _TOKEN)
        from src.clients.github_client import GitHubTokenManager

        route = respx.post(self._ROUTE).mock(return_value=self._ok())
        await GitHubTokenManager().get_app_clone_credential(self._REPO, tenant_id=None)

        request = route.calls.last.request
        assert request.headers.get("authorization") == f"Bearer {_TOKEN}"
        assert "x-tenant-id" not in request.headers

"""Tenant-bound GitHub credentials: a customer repo never gets the system PAT.

Invariant (plans/public-github-app-usable.md): a token for a repo outside the
platform org is issued ONLY via the public App, ONLY for an installation bound
to the requesting tenant. coding-containers' share of that:

* the tenant travels to coding-github's clone-token route in the JSON BODY
  (never as ``X-Tenant-Id`` — the tenant middleware 403s that assertion);
* for a NON-platform owner the system PAT is never a clone credential (K8s and
  Docker provisioners) nor a push credential (git_push) — with no BYO token and
  no tenant-bound App token the operation fails closed;
* platform owners get coding-github's App token or a typed 412 — the org PAT
  is never a fallback, and a platform request with no tenant is refused.

Every "no fallback" test sets a system PAT, so a regression that re-opens the
fallback has a PAT to reach for and turns the test red.
"""

from __future__ import annotations

import json
import logging
import os
from unittest.mock import MagicMock, create_autospec, patch

import httpx
import pytest
import respx

import src.config.settings as settings_mod
from src.clients.github_client import (
    CloneCredential,
    GitHubTokenManager,
    is_platform_owner,
    repo_owner_from_url,
)
from src.core import kubernetes_container_provisioner as kcp
from src.core.container_spec import GitHubRepoRequest
from tests.unit.github_api_fence import no_outbound_network

_PAT = "ghp_SYSTEM_PAT_must_never_touch_a_customer_repo"  # pragma: allowlist secret
_BASE = "http://coding-github.test:8692"
_URL = f"{_BASE}/api/internal/github/clone-token"
_CUSTOMER_REPO = "https://github.com/acme-corp/private-app"
_PLATFORM_REPO = "https://github.com/nonagenticai/internal-app"


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    monkeypatch.setenv("CODING_CONTAINERS_ENVIRONMENT", "test")
    monkeypatch.setenv("CODING_GITHUB_URL", _BASE)
    monkeypatch.setenv("CODING_CONTAINERS_INTERNAL_SERVICE_TOKEN", "shhh")  # noqa: S105
    monkeypatch.delenv("CODING_CONTAINERS_PLATFORM_GITHUB_OWNERS", raising=False)
    settings_mod._settings_instance = None
    yield
    settings_mod._settings_instance = None


def _token_manager(credential: CloneCredential, system_token: str | None = _PAT) -> MagicMock:
    mgr = create_autospec(GitHubTokenManager, instance=True)
    mgr.get_system_owner.return_value = "nonagenticai"
    mgr.resolve_provision_secret_config_id.return_value = None
    mgr.get_app_clone_credential.return_value = credential
    return mgr


_NOT_INSTALLED = CloneCredential(
    ok=True, token=None, visibility="private", source="none", reason="not_installed"
)
_APP_TOKEN = CloneCredential(ok=True, token="ghs_tenantbound", visibility="private", source="app")

#: Every null-token answer a PLATFORM repo can get, incl. no reason at all, a
#: public visibility, and a failed call. Each must fail closed — never the PAT.
_PLATFORM_NULL_TOKENS = [
    _NOT_INSTALLED,
    CloneCredential(ok=True, token=None, visibility="private", reason="tenant_required"),
    CloneCredential(ok=True, token=None, visibility="private", reason="not_configured"),
    CloneCredential(ok=True, token=None, visibility="private", reason="mint_failed"),
    CloneCredential(ok=True, token=None, visibility="private", reason=None),
    CloneCredential(ok=True, token=None, visibility="public", reason=None),
    CloneCredential(ok=False, error="coding-github unreachable"),
]


def _null_id(credential: CloneCredential) -> str:
    return f"{credential.visibility}-{credential.reason or credential.error or 'none'}"


# ---------------------------------------------------------------------------
# owner helpers
# ---------------------------------------------------------------------------


class TestOwnerHelpers:
    @pytest.mark.parametrize(
        ("url", "owner"),
        [
            ("https://github.com/acme-corp/repo", "acme-corp"),
            ("https://github.com/acme-corp/repo.git", "acme-corp"),
            ("git@github.com:acme-corp/repo.git", "acme-corp"),
            ("https://github.com/acme-corp", ""),
            ("", ""),
            (None, ""),
            # The owner is the CANONICAL one or nothing: a first-segment parse of
            # these says "nonagenticai" (platform PAT) while git reaches acme-corp.
            ("https://github.com/nonagenticai/../acme-corp/repo", ""),
            ("https://github.com/nonagenticai/%2e%2e/acme-corp/repo", ""),
            ("git@github.com:nonagenticai/../acme-corp/repo.git", ""),
            ("https://www.github.com/nonagenticai/repo", ""),
        ],
    )
    def test_repo_owner_from_url(self, url, owner):
        assert repo_owner_from_url(url) == owner

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://github.com/acme-corp/repo", ("acme-corp", "repo")),
            ("https://GITHUB.COM/acme-corp/repo.git/", ("acme-corp", "repo")),
            ("git@github.com:acme-corp/repo.git", ("acme-corp", "repo")),
            ("ssh://git@github.com:22/acme-corp/repo.git", ("acme-corp", "repo")),
            ("https://github.com/nonagenticai/../acme-corp/repo", None),
            ("https://github.com/nonagenticai/%2e%2e/acme-corp/repo", None),
            ("https://github.com/nonagenticai/%252e%252e/acme-corp/repo", None),
            ("https://github.com/acme-corp/./repo", None),
            ("https://github.com/acme-corp/re..po", None),
            ("https://github.com//repo", None),
            ("https://github.com/acme-corp/repo/extra", None),
            ("https://github.com/acme-corp%2frepo/x", None),
            ("https://github.com/acme-corp/repo%0a", None),
            ("https://www.github.com/acme-corp/repo", None),
            ("https://github.com.evil.example/acme-corp/repo", None),
            ("https://user@github.com/acme-corp/repo", None),
            ("https://github.com:8443/acme-corp/repo", None),
            ("http://github.com/acme-corp/repo", None),
            ("https://github.com/acme-corp/repo?x=1", None),
            ("git@github.com:/acme-corp/repo", None),
            ("evil@github.com:acme-corp/repo", None),
            # An encoded "/" must not become a segment boundary git never sees.
            ("https://github.com/nonagenticai%2Fr", None),
            ("https://github.com/nonagenticai%2fr.git", None),
            ("git@github.com:nonagenticai%2Fr", None),
        ],
    )
    def test_canonical_github_repo(self, url, expected):
        from src.clients.github_client import canonical_github_repo

        assert canonical_github_repo(url) == expected

    def test_default_platform_owner_is_nonagenticai_case_insensitive(self):
        assert is_platform_owner("nonagenticai")
        assert is_platform_owner("NonAgenticAI")
        assert not is_platform_owner("acme-corp")

    def test_unparseable_owner_is_not_platform(self):
        assert not is_platform_owner("")
        assert not is_platform_owner(None)

    def test_setting_is_honoured(self, monkeypatch):
        monkeypatch.setenv("CODING_CONTAINERS_PLATFORM_GITHUB_OWNERS", "nonagenticai, specralabs")
        settings_mod._settings_instance = None
        assert is_platform_owner("specralabs")
        assert not is_platform_owner("acme-corp")


class TestRequireTenantForPlatformRepo:
    """``require_tenant_for_platform_repo`` — no implicit platform tenant, ever."""

    @pytest.mark.parametrize("absent", [None, "", "   "])
    def test_platform_owner_without_a_tenant_is_refused_and_logged(self, absent, caplog):
        with (
            caplog.at_level(logging.WARNING, logger=kcp.logger.name),
            pytest.raises(kcp.GitCredentialUnavailableError) as exc_info,
        ):
            kcp.require_tenant_for_platform_repo(
                absent, owner="nonagenticai", repo_url=_PLATFORM_REPO, context="unit"
            )
        assert exc_info.value.reason == "platform_tenant_missing"
        assert "tenant required for platform repo" in str(exc_info.value.detail)
        assert "event=platform_tenant_missing_refused" in caplog.text

    def test_a_real_tenant_is_returned_unchanged(self):
        for owner in ("nonagenticai", "acme-corp"):
            assert (
                kcp.require_tenant_for_platform_repo(
                    "tenant-of-record", owner=owner, repo_url=_PLATFORM_REPO, context="unit"
                )
                == "tenant-of-record"
            )

    def test_a_customer_owner_with_no_tenant_passes_through_as_none(self, caplog):
        with caplog.at_level(logging.WARNING, logger=kcp.logger.name):
            for owner in ("acme-corp", "", None):
                assert (
                    kcp.require_tenant_for_platform_repo(
                        None, owner=owner, repo_url=_CUSTOMER_REPO, context="unit"
                    )
                    is None
                )
        assert "platform_tenant_missing_refused" not in caplog.text


# ---------------------------------------------------------------------------
# clone-token wire format
# ---------------------------------------------------------------------------


class TestCloneTokenCarriesTenantInBody:
    @respx.mock
    async def test_tenant_id_is_in_the_body_and_not_a_header(self):
        route = respx.post(_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "token": None,
                    "visibility": "private",
                    "source": "none",
                    "reason": "not_installed",
                },
            )
        )
        cred = await GitHubTokenManager().get_app_clone_credential(
            _CUSTOMER_REPO, tenant_id="tenant-7"
        )
        sent = route.calls.last.request
        assert json.loads(sent.read()) == {"repo_url": _CUSTOMER_REPO, "tenant_id": "tenant-7"}
        assert "X-Tenant-Id" not in sent.headers
        assert cred.reason == "not_installed"
        assert cred.token is None

    @respx.mock
    async def test_push_scope_is_sent_in_the_body_with_the_tenant(self):
        route = respx.post(_URL).mock(
            return_value=httpx.Response(
                200,
                json={"token": "ghs_push", "visibility": "private", "source": "app"},
            )
        )
        cred = await GitHubTokenManager().get_app_clone_credential(
            _CUSTOMER_REPO, tenant_id="tenant-7", scope="push"
        )
        sent = route.calls.last.request
        assert json.loads(sent.read()) == {
            "repo_url": _CUSTOMER_REPO,
            "tenant_id": "tenant-7",
            "scope": "push",
        }
        assert "X-Tenant-Id" not in sent.headers
        assert sent.headers["X-Internal-Token"] == "shhh"
        assert cred.token == "ghs_push"

    @respx.mock
    async def test_absent_tenant_is_omitted_not_invented(self):
        route = respx.post(_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "token": None,
                    "visibility": "private",
                    "source": "none",
                    "reason": "tenant_required",
                },
            )
        )
        cred = await GitHubTokenManager().get_app_clone_credential(_CUSTOMER_REPO, tenant_id=None)
        assert json.loads(route.calls.last.request.read()) == {"repo_url": _CUSTOMER_REPO}
        assert cred.reason == "tenant_required"


# ---------------------------------------------------------------------------
# K8s provisioner
# ---------------------------------------------------------------------------


def _k8s_provisioner():
    with patch("kubernetes.config.load_kube_config"):
        prov = kcp.K8sContainerProvisioner(namespace="test-ns", in_cluster=False)
    prov.apps_v1 = MagicMock()
    prov.core_v1 = MagicMock()
    return prov


async def _k8s_provision(mgr, request):
    prov = _k8s_provisioner()
    settings = MagicMock()
    settings.main_api_endpoint = "http://localhost:8687"
    with (
        patch.object(kcp, "get_github_token_manager", return_value=mgr),
        patch.object(kcp, "get_settings", return_value=settings),
        patch.object(prov, "_create_deployment") as deploy_mock,
        patch.object(kcp, "ContainerMetadataRegistry"),
    ):
        await prov.provision_from_github(request)
    return deploy_mock


class TestK8sCustomerClone:
    async def test_customer_owner_never_falls_back_to_the_pat(self):
        mgr = _token_manager(_NOT_INSTALLED)
        request = GitHubRepoRequest(repo_url=_CUSTOMER_REPO, branch="main", tenant_id="tenant-7")
        prov = _k8s_provisioner()
        settings = MagicMock()
        settings.main_api_endpoint = "http://localhost:8687"
        with (
            patch.object(kcp, "get_github_token_manager", return_value=mgr),
            patch.object(kcp, "get_settings", return_value=settings),
            patch.object(prov, "_create_deployment") as deploy_mock,
            patch.object(kcp, "ContainerMetadataRegistry"),
            pytest.raises(kcp.GitCredentialUnavailableError) as exc_info,
        ):
            await prov.provision_from_github(request)

        assert exc_info.value.status_code == 412
        assert exc_info.value.reason == "github_app_not_installed"
        assert "github_app_not_installed for acme-corp" in str(exc_info.value.detail)
        deploy_mock.assert_not_called()
        assert mgr.get_app_clone_credential.await_args.kwargs["tenant_id"] == "tenant-7"

    async def test_customer_owner_clones_with_the_tenant_bound_app_token(self):
        mgr = _token_manager(_APP_TOKEN)
        request = GitHubRepoRequest(repo_url=_CUSTOMER_REPO, branch="main", tenant_id="tenant-7")
        deploy_mock = await _k8s_provision(mgr, request)
        assert deploy_mock.call_args.kwargs["github_token"] == "ghs_tenantbound"

    async def test_customer_public_repo_clones_with_no_credential_not_the_pat(self):
        mgr = _token_manager(
            CloneCredential(ok=True, token=None, visibility="public", source="none")
        )
        request = GitHubRepoRequest(repo_url=_CUSTOMER_REPO, branch="main", tenant_id="tenant-7")
        deploy_mock = await _k8s_provision(mgr, request)
        assert deploy_mock.call_args.kwargs["github_token"] is None
        assert deploy_mock.call_args.kwargs["repo_visibility"] == "public"

    @pytest.mark.parametrize("credential", _PLATFORM_NULL_TOKENS, ids=_null_id)
    async def test_platform_owner_with_no_app_token_fails_closed_never_the_pat(self, credential):
        mgr = _token_manager(credential)
        request = GitHubRepoRequest(repo_url=_PLATFORM_REPO, branch="main", tenant_id="int-1")
        with pytest.raises(kcp.GitCredentialUnavailableError) as exc_info:
            await _k8s_provision(mgr, request)
        assert exc_info.value.status_code == 412
        assert exc_info.value.reason == "platform_credential_unavailable"
        assert f"reason={credential.reason or credential.error or 'none'}" in str(
            exc_info.value.detail
        )
        assert not hasattr(mgr, "get_system_token")

    async def test_platform_owner_clones_with_the_app_token(self):
        mgr = _token_manager(_APP_TOKEN)
        request = GitHubRepoRequest(repo_url=_PLATFORM_REPO, branch="main", tenant_id="int-1")
        deploy_mock = await _k8s_provision(mgr, request)
        assert deploy_mock.call_args.kwargs["github_token"] == "ghs_tenantbound"
        assert not hasattr(mgr, "get_system_token")

    async def test_platform_owner_without_a_tenant_is_refused_not_substituted(self, caplog):
        """The body-only ``provision_from_github`` path: no tenant, no clone."""
        mgr = _token_manager(_APP_TOKEN)
        request = GitHubRepoRequest(repo_url=_PLATFORM_REPO, branch="main")
        assert request.tenant_id is None, "fixture must model the absent body field"
        prov = _k8s_provisioner()
        settings = MagicMock()
        settings.main_api_endpoint = "http://localhost:8687"
        with (
            caplog.at_level(logging.WARNING, logger=kcp.logger.name),
            patch.object(kcp, "get_github_token_manager", return_value=mgr),
            patch.object(kcp, "get_settings", return_value=settings),
            patch.object(prov, "_create_deployment") as deploy_mock,
            patch.object(kcp, "ContainerMetadataRegistry") as registry_cls,
            pytest.raises(kcp.GitCredentialUnavailableError) as exc_info,
        ):
            await prov.provision_from_github(request)
        assert exc_info.value.reason == "platform_tenant_missing"
        assert "event=platform_tenant_missing_refused" in caplog.text
        mgr.get_app_clone_credential.assert_not_called()
        assert not hasattr(mgr, "get_system_token")
        deploy_mock.assert_not_called()
        registry_cls.return_value.upsert.assert_not_called()

    async def test_customer_owner_without_a_tenant_still_sends_nothing(self):
        """Negative control for the K8s path — the refusal stops at platform."""
        mgr = _token_manager(_NOT_INSTALLED)
        request = GitHubRepoRequest(repo_url=_CUSTOMER_REPO, branch="main")
        with pytest.raises(kcp.GitCredentialUnavailableError) as exc_info:
            await _k8s_provision(mgr, request)
        assert exc_info.value.reason == "github_app_not_installed"
        assert mgr.get_app_clone_credential.await_args.kwargs["tenant_id"] is None


class TestK8sReprovisionTenantPersistence:
    """The PERSISTED consequence of ``ContainerManager``'s re-provision path.

    ``ContainerManager._reprovision`` (container_manager.py, around line 844)
    computes ``row_tenant = platform_clone_tenant_id(repo_owner)`` for a
    tenantless platform-repo row and calls
    ``provisioner.provision_container(tenant_id=row_tenant, ...)``. Unlike the
    two ``provision_from_github`` entry points above — which deliberately keep
    the clone-credential tenant separate from ``request.tenant_id`` — there is
    only ONE ``tenant_id`` parameter on ``provision_container``, so that single
    value becomes ``GitHubRepoRequest.tenant_id`` and flows into BOTH the
    ownership stamp handed to ``_create_deployment`` AND the persisted
    ``ContainerInfo.tenant_id``.

    These tests run the REAL ``K8sContainerProvisioner.provision_from_github``
    (production code, unlike the ContainerManager-level tests in
    test_container_manager.py where the whole provisioner is a mock) and only
    stub the K8s API layer (``_create_deployment``) and the
    ``ContainerMetadataRegistry`` CLASS — nothing here touches a real cluster
    or a real JSON file.
    """

    async def test_customer_owner_without_a_tenant_never_reaches_the_registry(self):
        """Negative control: the refusal happens before any persistence at all."""
        mgr = _token_manager(_NOT_INSTALLED)
        request = GitHubRepoRequest(repo_url=_CUSTOMER_REPO, branch="main")
        prov = _k8s_provisioner()
        settings = MagicMock()
        settings.main_api_endpoint = "http://localhost:8687"
        with (
            patch.object(kcp, "get_github_token_manager", return_value=mgr),
            patch.object(kcp, "get_settings", return_value=settings),
            patch.object(prov, "_create_deployment") as deploy_mock,
            patch.object(kcp, "ContainerMetadataRegistry") as registry_cls,
            pytest.raises(kcp.GitCredentialUnavailableError),
        ):
            await prov.provision_from_github(request)
        deploy_mock.assert_not_called()
        registry_cls.return_value.upsert.assert_not_called()

    async def test_a_named_tenant_on_a_platform_repo_is_persisted_unchanged(self):
        """Negative control: a row that already HAS a tenant is not touched.

        A real ``request.tenant_id`` on a platform repo must be stamped and
        persisted exactly as given.
        """
        own_tenant = "0195f48b-d000-7000-8000-00000000beef"
        mgr = _token_manager(_APP_TOKEN)
        request = GitHubRepoRequest(repo_url=_PLATFORM_REPO, branch="main", tenant_id=own_tenant)
        prov = _k8s_provisioner()
        settings = MagicMock()
        settings.main_api_endpoint = "http://localhost:8687"
        with (
            patch.object(kcp, "get_github_token_manager", return_value=mgr),
            patch.object(kcp, "get_settings", return_value=settings),
            patch.object(prov, "_create_deployment") as deploy_mock,
            patch.object(kcp, "ContainerMetadataRegistry") as registry_cls,
        ):
            info = await prov.provision_from_github(request)

        assert deploy_mock.call_args.kwargs["tenant_id"] == own_tenant
        persisted = registry_cls.return_value.upsert.call_args.args[1]
        assert persisted.tenant_id == own_tenant
        assert info.tenant_id == own_tenant


# ---------------------------------------------------------------------------
# Docker provisioner
# ---------------------------------------------------------------------------


def _docker_provisioner():
    with patch.dict(os.environ, {"KUBERNETES_IN_CLUSTER": "false"}, clear=False):
        from src.core.github_container_provisioner import GitHubContainerProvisioner

        manager = MagicMock()
        manager.image_name = "codegraph:latest"
        manager._find_available_port.return_value = 9999
        manager.create_container.return_value = MagicMock(
            container_name="c",
            api_endpoint="http://x",
            internal_endpoint="http://x",
            public_endpoint="http://x",
            status="running",
        )
        return GitHubContainerProvisioner(manager)


class TestDockerCustomerClone:
    async def _run(self, mgr, request):
        prov = _docker_provisioner()
        with (
            patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": _PAT}, clear=False),
            patch(
                "src.core.github_container_provisioner.get_github_token_manager",
                return_value=mgr,
            ),
            patch("src.core.github_container_provisioner.get_docker_service"),
            patch("src.core.github_container_provisioner.ContainerInfo"),
            patch.object(prov, "_clone_repo_into_shared_volume") as clone_mock,
            # The PyGithub pre-flight is covered separately below; never hit GitHub.
            patch.object(prov, "_validate_github_repository_access"),
        ):
            try:
                await prov.provision_from_github(request)
            except kcp.GitCredentialUnavailableError as exc:
                return prov, clone_mock, exc
        return prov, clone_mock, None

    async def test_customer_owner_never_falls_back_to_the_pat(self):
        mgr = _token_manager(_NOT_INSTALLED)
        request = GitHubRepoRequest(repo_url=_CUSTOMER_REPO, branch="main", tenant_id="tenant-7")
        prov, clone_mock, exc = await self._run(mgr, request)

        assert exc is not None, "a customer repo with no App install must fail closed"
        assert exc.reason == "github_app_not_installed"
        clone_mock.assert_not_called()
        prov.container_manager.create_container.assert_not_called()
        assert not hasattr(mgr, "get_system_token")
        assert mgr.get_app_clone_credential.await_args.kwargs["tenant_id"] == "tenant-7"

    async def test_customer_owner_uses_the_tenant_bound_app_token(self):
        mgr = _token_manager(_APP_TOKEN)
        request = GitHubRepoRequest(repo_url=_CUSTOMER_REPO, branch="main", tenant_id="tenant-7")
        prov, clone_mock, exc = await self._run(mgr, request)

        assert exc is None
        assert clone_mock.call_args.kwargs["token"] == "ghs_tenantbound"
        spec = prov.container_manager.create_container.call_args.args[0]
        # No credential of ANY kind in the long-lived workspace env.
        assert "GITHUB_ACCESS_TOKEN" not in spec.environment
        assert "ghs_tenantbound" not in spec.environment.values()
        assert _PAT not in spec.environment.values()

    @pytest.mark.parametrize("credential", _PLATFORM_NULL_TOKENS, ids=_null_id)
    async def test_platform_owner_with_no_app_token_fails_closed_never_the_pat(self, credential):
        mgr = _token_manager(credential)
        request = GitHubRepoRequest(repo_url=_PLATFORM_REPO, branch="main", tenant_id="int-1")
        prov, clone_mock, exc = await self._run(mgr, request)

        assert exc is not None and exc.reason == "platform_credential_unavailable"
        assert exc.status_code == 412
        clone_mock.assert_not_called()
        prov.container_manager.create_container.assert_not_called()
        assert not hasattr(mgr, "get_system_token")
        assert mgr.get_app_clone_credential.await_args.kwargs["repo_url"] == _PLATFORM_REPO

    async def test_platform_owner_without_a_tenant_is_refused_not_substituted(self, caplog):
        mgr = _token_manager(_APP_TOKEN)
        request = GitHubRepoRequest(repo_url=_PLATFORM_REPO, branch="main")
        assert request.tenant_id is None, "fixture must model the absent body field"
        with caplog.at_level(logging.WARNING, logger=kcp.logger.name):
            prov, clone_mock, exc = await self._run(mgr, request)

        assert exc is not None and exc.reason == "platform_tenant_missing"
        assert "event=platform_tenant_missing_refused" in caplog.text
        mgr.get_app_clone_credential.assert_not_called()
        assert not hasattr(mgr, "get_system_token")
        clone_mock.assert_not_called()
        prov.container_manager.create_container.assert_not_called()

    async def test_customer_owner_without_a_tenant_still_sends_nothing(self):
        """The negative control: the substitution must NOT reach a customer repo.

        A customer repo with no tenant keeps its own ``github_app_not_installed``
        refusal and is never handed a platform identity.
        """
        mgr = _token_manager(_NOT_INSTALLED)
        request = GitHubRepoRequest(repo_url=_CUSTOMER_REPO, branch="main")
        _prov, _clone_mock, exc = await self._run(mgr, request)

        assert exc is not None and exc.reason == "github_app_not_installed"
        assert mgr.get_app_clone_credential.await_args.kwargs["tenant_id"] is None

    def test_preflight_never_uses_the_pat_on_a_customer_repo(self):
        prov = _docker_provisioner()
        request = GitHubRepoRequest(repo_url=_CUSTOMER_REPO, branch="main")
        with (
            patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": _PAT}, clear=False),
            no_outbound_network() as fence,
        ):
            prov._validate_github_repository_access(request)
        assert fence.attempts == []


def test_preflight_logs_nothing_derived_from_the_token(caplog):
    """Neither a prefix nor the length of the PAT may reach a log line."""
    import logging

    prov = _docker_provisioner()
    request = GitHubRepoRequest(repo_url=_PLATFORM_REPO, branch="main")
    with (
        caplog.at_level(logging.DEBUG),
        patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": _PAT}, clear=False),
        no_outbound_network(),
    ):
        prov._validate_github_repository_access(request)
    text = caplog.text
    n = len(_PAT)
    assert _PAT not in text
    assert _PAT[:4] not in text  # "ghp_" — any prefix of the credential
    # The length in a structured context only. A bare "47" substring also
    # matched MagicMock ids (``id='4471...'``) and made this test flaky.
    import re

    length_ctx = re.compile(
        rf"(?i)(?:\blength\b|\blen\b|\bchars?\b|\bcharacters\b)\W{{0,3}}{n}\b"
        rf"|\b{n}\s*(?:chars?|characters)\b|len\({n}\)|len={n}\b"
    )
    assert not length_ctx.search(text), length_ctx.search(text)
    assert "token length" not in text.lower()
    assert "token prefix" not in text.lower()


def test_preflight_length_detector_catches_the_old_log_lines():
    """Control: the structured detector above fires on what used to be logged."""
    import re

    n = len(_PAT)
    length_ctx = re.compile(
        rf"(?i)(?:\blength\b|\blen\b|\bchars?\b|\bcharacters\b)\W{{0,3}}{n}\b"
        rf"|\b{n}\s*(?:chars?|characters)\b|len\({n}\)|len={n}\b"
    )
    for line in (f"Token length: {n}", f"token len={n}", f"({n} chars)"):
        assert length_ctx.search(line), line
    assert not length_ctx.search("<MagicMock id='4471234567'>")


# ---------------------------------------------------------------------------
# Canonical-only: owner, credential and clone URL all come from one pair
# ---------------------------------------------------------------------------

# git resolves these to victim/private; a first-segment parse says "nonagenticai".
_TRAVERSAL = "https://github.com/nonagenticai/x/../../victim/private"
_NON_CANONICAL = [
    _TRAVERSAL,
    "https://github.com/nonagenticai/%2e%2e/victim/private",
    "https://github.com/nonagenticai%2Fr",
]


def _unvalidated(repo_url: str) -> GitHubRepoRequest:
    """A request whose repo_url skipped the pydantic pattern (internal callers)."""
    return GitHubRepoRequest.model_construct(
        repo_url=repo_url,
        branch="main",
        access_token=None,
        tenant_id=None,
        user_id=None,
        container_name=None,
        description=None,
        history_cutoff=False,
        secret_config_id=None,
    )


class TestDockerProvisionerIsCanonicalOnly:
    @pytest.mark.parametrize("url", _NON_CANONICAL)
    async def test_non_canonical_url_refused_before_credential_or_preflight(self, url):
        mgr = _token_manager(_APP_TOKEN)
        prov = _docker_provisioner()
        with (
            patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": _PAT}, clear=False),
            patch(
                "src.core.github_container_provisioner.get_github_token_manager",
                return_value=mgr,
            ),
            patch("src.core.github_container_provisioner.get_docker_service"),
            no_outbound_network() as fence,
            patch.object(prov, "_clone_repo_into_shared_volume") as clone_mock,
            pytest.raises(ValueError, match="canonical"),
        ):
            await prov.provision_from_github(_unvalidated(url))
        assert not hasattr(mgr, "get_system_token")
        mgr.get_app_clone_credential.assert_not_awaited()
        assert fence.attempts == []
        clone_mock.assert_not_called()
        prov.container_manager.create_container.assert_not_called()

    @pytest.mark.parametrize("url", _NON_CANONICAL)
    def test_preflight_refuses_non_canonical_url_without_touching_the_pat(self, url):
        prov = _docker_provisioner()
        with (
            patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": _PAT}, clear=False),
            no_outbound_network() as fence,
            pytest.raises(ValueError, match="canonical"),
        ):
            prov._validate_github_repository_access(_unvalidated(url))
        assert fence.attempts == []

    async def test_clone_and_container_get_the_canonical_url(self):
        mgr = _token_manager(_APP_TOKEN)
        prov = _docker_provisioner()
        request = GitHubRepoRequest(
            repo_url=f"{_PLATFORM_REPO}.git/", branch="main", tenant_id="int-1"
        )
        with (
            patch(
                "src.core.github_container_provisioner.get_github_token_manager",
                return_value=mgr,
            ),
            patch("src.core.github_container_provisioner.get_docker_service"),
            patch("src.core.github_container_provisioner.ContainerInfo"),
            patch.object(prov, "_clone_repo_into_shared_volume") as clone_mock,
            patch.object(prov, "_validate_github_repository_access"),
        ):
            await prov.provision_from_github(request)
        assert clone_mock.call_args.kwargs["repo_url"] == _PLATFORM_REPO
        spec = prov.container_manager.create_container.call_args.args[0]
        assert spec.environment["REPO_URL"] == _PLATFORM_REPO

    def test_shared_volume_clone_command_uses_only_the_canonical_pair(self):
        prov = _docker_provisioner()
        prov.container_manager._resolve_shared_volume_name.return_value = "vol"
        prov._clone_repo_into_shared_volume(
            container_name="c1",
            repo_url="https://GITHUB.com/nonagenticai/internal-app.git/",
            branch="main",
            token="tok",  # noqa: S106
        )
        run = prov.container_manager.client.containers.run.call_args.kwargs
        cmd = run["command"][-1]
        assert " https://github.com/nonagenticai/internal-app " in cmd
        assert "tok@" not in cmd
        assert run["environment"] == {"GIT_TOKEN": "tok"}

    @pytest.mark.parametrize("url", _NON_CANONICAL)
    def test_shared_volume_clone_refuses_non_canonical(self, url):
        prov = _docker_provisioner()
        with pytest.raises(ValueError, match="canonical"):
            prov._clone_repo_into_shared_volume(
                container_name="c1",
                repo_url=url,
                branch="main",
                token=_PAT,
            )
        prov.container_manager.client.containers.run.assert_not_called()


class TestK8sProvisionerIsCanonicalOnly:
    @pytest.mark.parametrize("url", _NON_CANONICAL)
    async def test_non_canonical_url_refused_before_any_credential(self, url):
        mgr = _token_manager(_APP_TOKEN)
        prov = _k8s_provisioner()
        with (
            patch.object(kcp, "get_github_token_manager", return_value=mgr),
            patch.object(prov, "_create_deployment") as deploy_mock,
            patch.object(kcp, "ContainerMetadataRegistry"),
            pytest.raises(ValueError, match="canonical"),
        ):
            await prov.provision_from_github(_unvalidated(url))
        assert not hasattr(mgr, "get_system_token")
        mgr.get_app_clone_credential.assert_not_awaited()
        deploy_mock.assert_not_called()

    async def test_deployment_and_app_lookup_get_the_canonical_url(self):
        mgr = _token_manager(_APP_TOKEN)
        request = GitHubRepoRequest(
            repo_url="https://github.com/acme-corp/private-app.git/",
            branch="main",
            tenant_id="tenant-7",
        )
        deploy_mock = await _k8s_provision(mgr, request)
        assert deploy_mock.call_args.kwargs["repo_url"] == _CUSTOMER_REPO
        assert mgr.get_app_clone_credential.await_args.kwargs["repo_url"] == _CUSTOMER_REPO

    @pytest.mark.parametrize("url", _NON_CANONICAL)
    def test_create_deployment_refuses_non_canonical(self, url):
        prov = _k8s_provisioner()
        with pytest.raises(ValueError, match="canonical"):
            prov._create_deployment(name="d1", repo_url=url, branch="main", github_token=_PAT)
        prov.apps_v1.create_namespaced_deployment.assert_not_called()
        prov.core_v1.create_namespaced_secret.assert_not_called()


# ---------------------------------------------------------------------------
# coding-github #107: ``platform_repo_forbidden`` is a refusal, never a PAT cue
# ---------------------------------------------------------------------------

_FORBIDDEN = CloneCredential(
    ok=True, token=None, visibility="private", source="none", reason="platform_repo_forbidden"
)


class TestPlatformRepoForbiddenNeverFallsBackToThePat:
    """Every test here has the system PAT available, so a regression that answers
    the refusal with it has a credential to reach for and turns the test red."""

    async def test_k8s_provision_refuses(self):
        mgr = _token_manager(_FORBIDDEN)
        request = GitHubRepoRequest(repo_url=_PLATFORM_REPO, branch="main", tenant_id="cust-1")
        prov = _k8s_provisioner()
        settings = MagicMock()
        settings.main_api_endpoint = "http://localhost:8687"
        with (
            patch.object(kcp, "get_github_token_manager", return_value=mgr),
            patch.object(kcp, "get_settings", return_value=settings),
            patch.object(prov, "_create_deployment") as deploy_mock,
            patch.object(kcp, "ContainerMetadataRegistry"),
            pytest.raises(kcp.GitCredentialUnavailableError) as exc_info,
        ):
            await prov.provision_from_github(request)
        assert exc_info.value.status_code == 403
        assert exc_info.value.reason == "platform_repo_forbidden"
        deploy_mock.assert_not_called()

    async def test_docker_provision_refuses(self):
        mgr = _token_manager(_FORBIDDEN)
        request = GitHubRepoRequest(repo_url=_PLATFORM_REPO, branch="main", tenant_id="cust-1")
        prov, clone_mock, exc = await TestDockerCustomerClone()._run(mgr, request)
        assert exc is not None and exc.reason == "platform_repo_forbidden"
        assert exc.status_code == 403
        clone_mock.assert_not_called()
        prov.container_manager.create_container.assert_not_called()
        assert not hasattr(mgr, "get_system_token")

    async def test_docker_lifecycle_refuses(self):
        from src.services.docker_services.container_lifecycle import ContainerLifecycleManager

        mgr = _token_manager(_FORBIDDEN)
        request = GitHubRepoRequest(repo_url=_PLATFORM_REPO, branch="main", tenant_id="cust-1")
        with pytest.raises(kcp.GitCredentialUnavailableError) as exc_info:
            await ContainerLifecycleManager._resolve_clone_token(
                request, "nonagenticai", "cust-1", mgr
            )
        assert exc_info.value.reason == "platform_repo_forbidden"
        assert not hasattr(mgr, "get_system_token")

    @pytest.mark.parametrize("credential", _PLATFORM_NULL_TOKENS, ids=_null_id)
    async def test_docker_lifecycle_platform_owner_never_gets_the_pat(self, credential):
        from src.services.docker_services.container_lifecycle import ContainerLifecycleManager

        mgr = _token_manager(credential)
        request = GitHubRepoRequest(repo_url=_PLATFORM_REPO, branch="main", tenant_id="int-1")
        with pytest.raises(kcp.GitCredentialUnavailableError) as exc_info:
            await ContainerLifecycleManager._resolve_clone_token(
                request, "nonagenticai", "int-1", mgr
            )
        assert exc_info.value.reason == "platform_credential_unavailable"
        assert not hasattr(mgr, "get_system_token")

    async def test_docker_lifecycle_platform_owner_uses_the_app_token(self):
        from src.services.docker_services.container_lifecycle import ContainerLifecycleManager

        mgr = _token_manager(_APP_TOKEN)
        request = GitHubRepoRequest(repo_url=_PLATFORM_REPO, branch="main", tenant_id="int-1")
        token = await ContainerLifecycleManager._resolve_clone_token(
            request, "nonagenticai", "int-1", mgr
        )
        assert token == "ghs_tenantbound"
        assert not hasattr(mgr, "get_system_token")

    async def test_docker_lifecycle_customer_byo_is_unaffected(self):
        from src.services.docker_services.container_lifecycle import ContainerLifecycleManager

        mgr = _token_manager(_NOT_INSTALLED)
        request = GitHubRepoRequest(
            repo_url=_CUSTOMER_REPO, branch="main", tenant_id="t-7", access_token="ghp_byo"
        )
        token = await ContainerLifecycleManager._resolve_clone_token(
            request, "acme-corp", "t-7", mgr
        )
        assert token == "ghp_byo"  # pragma: allowlist secret
        mgr.get_app_clone_credential.assert_not_called()
        assert not hasattr(mgr, "get_system_token")


class TestCloneTokenGetsTheCanonicalUrl:
    """coding-github is sent ``https://github.com/<owner>/<repo>``, never the raw string."""

    _RAW = "https://github.com/acme-corp/private-app.git/"
    _CANON = "https://github.com/acme-corp/private-app"

    async def test_docker_provisioner_customer_path(self):
        from src.core.github_container_provisioner import GitHubContainerProvisioner

        mgr = _token_manager(_APP_TOKEN)
        request = GitHubRepoRequest(repo_url=self._RAW, branch="main", tenant_id="tenant-7")
        await GitHubContainerProvisioner._customer_clone_token(request, "acme-corp", mgr)
        assert mgr.get_app_clone_credential.await_args.kwargs["repo_url"] == self._CANON

    async def test_docker_lifecycle_path(self):
        from src.services.docker_services.container_lifecycle import ContainerLifecycleManager

        mgr = _token_manager(_APP_TOKEN)
        request = GitHubRepoRequest(repo_url=self._RAW, branch="main", tenant_id="tenant-7")
        await ContainerLifecycleManager._resolve_clone_token(request, "acme-corp", "tenant-7", mgr)
        assert mgr.get_app_clone_credential.await_args.kwargs["repo_url"] == self._CANON

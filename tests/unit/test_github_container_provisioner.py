"""Tests for core/github_container_provisioner.py."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.container_spec import GitHubRepoRequest
from src.models.container_models import ContainerInfo, ContainerStatus
from tests.unit.github_api_fence import no_outbound_network


def _token_manager(
    system_token: str | None = "ghp_systemtoken",
    system_owner: str | None = "system-owner",
    secret_config_id: str | None = None,
) -> MagicMock:
    """Autospec'd stub of the REAL GitHubTokenManager.

    ``create_autospec`` rather than a bare ``MagicMock``: a signature-agnostic
    double accepts any call and so cannot fail, and this fleet has shipped
    fabricated passes that way. It also wires the async methods as AsyncMock.
    """
    from unittest.mock import create_autospec

    from src.clients.github_client import GitHubTokenManager

    mgr = create_autospec(GitHubTokenManager, instance=True)
    mgr.get_system_owner.return_value = system_owner
    mgr.store_user_token.return_value = secret_config_id
    mgr.resolve_provision_secret_config_id.return_value = secret_config_id
    return mgr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(**overrides) -> GitHubRepoRequest:
    defaults = {
        "repo_url": "https://github.com/owner/repo",
        "branch": "main",
        "access_token": "ghp_testtoken123",
    }
    defaults.update(overrides)
    return GitHubRepoRequest(**defaults)


def _make_container_info(**overrides) -> ContainerInfo:
    defaults = {
        "container_name": "codegraph-repo-abc123",
        "api_endpoint": "http://localhost:9999",
        "internal_endpoint": "http://localhost:9999",
        "public_endpoint": "http://localhost:9999",
        "code_graph_id": "codegraph-repo-abc123",
        "codebase_path": "/app/provisioned_containers/codegraph-repo-abc123/repo",
        "repo_url": "https://github.com/owner/repo",
        "branch": "main",
        "status": ContainerStatus.RUNNING,
    }
    defaults.update(overrides)
    return ContainerInfo(**defaults)


# ---------------------------------------------------------------------------
# _extract_repo_name
# ---------------------------------------------------------------------------


class TestExtractRepoName:
    """Tests for GitHubContainerProvisioner._extract_repo_name."""

    def _make_provisioner(self):
        with patch.dict(os.environ, {"KUBERNETES_IN_CLUSTER": "false"}, clear=False):
            from src.core.github_container_provisioner import GitHubContainerProvisioner

            manager = MagicMock()
            return GitHubContainerProvisioner(manager)

    def test_simple_url(self):
        prov = self._make_provisioner()
        assert prov._extract_repo_name("https://github.com/owner/my-repo") == "my-repo"

    def test_url_with_git_suffix(self):
        prov = self._make_provisioner()
        assert prov._extract_repo_name("https://github.com/owner/my-repo.git") == "my-repo"

    def test_url_with_trailing_slash(self):
        prov = self._make_provisioner()
        # trailing slash results in empty last part; fallback is the empty string
        result = prov._extract_repo_name("https://github.com/owner/my-repo/")
        # After strip("/") the trailing slash is removed, so path_parts[-1] == "my-repo"
        assert result == "my-repo"


# ---------------------------------------------------------------------------
# validate_repository
# ---------------------------------------------------------------------------


class TestValidateRepository:
    """Tests for GitHubContainerProvisioner.validate_repository."""

    def _make_provisioner(self):
        with patch.dict(os.environ, {"KUBERNETES_IN_CLUSTER": "false"}, clear=False):
            from src.core.github_container_provisioner import GitHubContainerProvisioner

            manager = MagicMock()
            return GitHubContainerProvisioner(manager)

    @patch(
        "src.core.github_container_provisioner.GitHubContainerProvisioner._validate_github_repository_access"
    )
    def test_returns_true_on_success(self, mock_validate):
        mock_validate.return_value = None
        prov = self._make_provisioner()
        request = _make_request()
        assert prov.validate_repository(request) is True

    @patch(
        "src.core.github_container_provisioner.GitHubContainerProvisioner._validate_github_repository_access"
    )
    def test_returns_false_on_value_error(self, mock_validate):
        mock_validate.side_effect = ValueError("bad repo")
        prov = self._make_provisioner()
        request = _make_request()
        assert prov.validate_repository(request) is False


# ---------------------------------------------------------------------------
# _validate_github_repository_access
# ---------------------------------------------------------------------------


class TestValidateGithubRepositoryAccess:
    """The pre-flight is structural only: it never calls GitHub, BYO token or not.

    This pod has no egress to api.github.com (only coding-github does), so the
    PyGithub probe it used to run with a BYO token failed or hung every BYO
    provision. The clone, which runs where GitHub is reachable, is the authority.
    """

    def _make_provisioner(self):
        with patch.dict(os.environ, {"KUBERNETES_IN_CLUSTER": "false"}, clear=False):
            from src.core.github_container_provisioner import GitHubContainerProvisioner

            manager = MagicMock()
            return GitHubContainerProvisioner(manager)

    def test_non_github_url_raises(self):
        prov = self._make_provisioner()
        request = _make_request()
        # Bypass pydantic validation on repo_url by setting the internal value
        object.__setattr__(request, "repo_url", "https://gitlab.com/owner/repo")
        with no_outbound_network() as fence, pytest.raises(ValueError, match="canonical"):
            prov._validate_github_repository_access(request)
        assert fence.attempts == []

    def test_invalid_path_format_raises(self):
        prov = self._make_provisioner()
        request = _make_request()
        object.__setattr__(request, "repo_url", "https://github.com/singlepart")
        with pytest.raises(ValueError, match="canonical"):
            prov._validate_github_repository_access(request)

    @pytest.mark.parametrize("branch", ["main", "develop"])
    def test_byo_token_preflight_makes_no_network_call(self, branch):
        """The regression: with a BYO token the pre-flight reached for api.github.com."""
        prov = self._make_provisioner()
        request = _make_request(branch=branch)
        assert request.access_token
        with no_outbound_network() as fence:
            prov._validate_github_repository_access(request)
        assert fence.attempts == []

    def test_byo_token_preflight_logs_nothing_derived_from_the_token(self, caplog):
        import logging

        token = "byo-sentinel-token-not-a-pat-7f3a9c2e1b"
        prov = self._make_provisioner()
        request = _make_request(access_token=token)
        with caplog.at_level(logging.DEBUG), no_outbound_network():
            prov._validate_github_repository_access(request)
        assert caplog.records, "the pre-flight logs what it decided"
        for window in (token, token[:8], token[-8:], token[4:12]):
            assert window not in caplog.text

    @pytest.mark.parametrize(
        "repo_url",
        ["https://github.com/nonagenticai/repo", "https://github.com/customer-org/repo"],
    )
    def test_no_byo_token_preflight_never_reads_env_pat(self, repo_url, monkeypatch):
        """No BYO token: no env PAT is read and no anonymous probe runs either."""
        sentinel = "ghp_SENTINEL_must_never_be_used_000000000000"
        for key in ("GITHUB_ACCESS_TOKEN", "GITHUB_TOKEN", "GITHUB_SYSTEM_TOKEN"):
            monkeypatch.setenv(key, sentinel)

        prov = self._make_provisioner()
        request = _make_request(repo_url=repo_url, access_token=None)
        with no_outbound_network() as fence:
            prov._validate_github_repository_access(request)
        assert fence.attempts == []


@pytest.mark.asyncio
async def test_k8s_byo_provision_reaches_the_clone_without_any_github_call():
    """End to end through ``provision_from_github`` (validator NOT patched): a BYO
    provision in cluster mode is handed to the K8s provisioner, whose init
    container clones with that token. Before the fix it never got there."""
    with patch.dict(os.environ, {"KUBERNETES_IN_CLUSTER": "false"}, clear=False):
        from src.core.github_container_provisioner import GitHubContainerProvisioner

        prov = GitHubContainerProvisioner(MagicMock())
    k8s = MagicMock()
    k8s.provision_from_github = AsyncMock(return_value=_make_container_info())
    prov.use_k8s = True
    prov.k8s_provisioner = k8s
    request = _make_request()
    with no_outbound_network() as fence:
        info = await prov.provision_from_github(request)
    assert fence.attempts == []
    k8s.provision_from_github.assert_awaited_once_with(request)
    assert info.container_name == "codegraph-repo-abc123"


# ---------------------------------------------------------------------------
# _clone_repo_into_shared_volume
# ---------------------------------------------------------------------------


class TestCloneRepoIntoSharedVolume:
    """Tests for the Docker-based clone method."""

    def _make_provisioner(self):
        with patch.dict(os.environ, {"KUBERNETES_IN_CLUSTER": "false"}, clear=False):
            from src.core.github_container_provisioner import GitHubContainerProvisioner

            manager = MagicMock()
            manager.client = MagicMock()
            manager._resolve_shared_volume_name.return_value = "shared-vol"
            return GitHubContainerProvisioner(manager)

    def test_clone_with_token_and_branch(self):
        prov = self._make_provisioner()
        prov._clone_repo_into_shared_volume(
            container_name="test-ctr",
            repo_url="https://github.com/owner/repo",
            branch="develop",
            token="ghp_tok123",
        )
        prov.container_manager.client.containers.run.assert_called_once()
        call_kwargs = prov.container_manager.client.containers.run.call_args
        cmd = call_kwargs[1]["command"][1]  # The shell command string
        assert "--branch develop" in cmd
        # The token is in the clone container's env ONLY — never argv, never the URL.
        assert "ghp_tok123" not in cmd
        assert " https://github.com/owner/repo " in cmd
        assert call_kwargs[1]["environment"] == {"GIT_TOKEN": "ghp_tok123"}
        assert "credential.https://github.com.helper=" in cmd

    def test_clone_without_token(self):
        prov = self._make_provisioner()
        prov._clone_repo_into_shared_volume(
            container_name="test-ctr",
            repo_url="https://github.com/owner/repo",
            branch=None,
            token=None,
        )
        call_kwargs = prov.container_manager.client.containers.run.call_args
        cmd = call_kwargs[1]["command"][1]
        assert "--branch" not in cmd
        assert "https://github.com/owner/repo" in cmd

    def test_clone_failure_raises_value_error(self):
        prov = self._make_provisioner()
        prov.container_manager.client.containers.run.side_effect = OSError("docker fail")
        with pytest.raises(ValueError, match="Failed to clone"):
            prov._clone_repo_into_shared_volume(
                container_name="test-ctr",
                repo_url="https://github.com/owner/repo",
                branch="main",
                token=None,
            )


# ---------------------------------------------------------------------------
# provision_from_github
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProvisionFromGithub:
    """Tests for the main provisioning flow."""

    def _make_provisioner(self):
        with patch.dict(os.environ, {"KUBERNETES_IN_CLUSTER": "false"}, clear=False):
            from src.core.github_container_provisioner import GitHubContainerProvisioner

            manager = MagicMock()
            manager.image_name = "codegraph:latest"
            manager._find_available_port.return_value = 9999
            manager.client = MagicMock()
            manager._resolve_shared_volume_name.return_value = "shared-vol"
            manager.create_container.return_value = _make_container_info()
            return GitHubContainerProvisioner(manager)

    @patch("src.core.github_container_provisioner.get_docker_service")
    @patch("src.core.github_container_provisioner.get_github_token_manager")
    @patch(
        "src.core.github_container_provisioner.GitHubContainerProvisioner._validate_github_repository_access"
    )
    async def test_successful_provision(self, mock_validate, mock_token_mgr_fn, mock_docker_svc_fn):
        mock_token_mgr = _token_manager(secret_config_id="sec-id-123")
        mock_token_mgr_fn.return_value = mock_token_mgr

        mock_docker_svc = MagicMock()
        mock_docker_svc.registry.load.return_value = {}
        mock_docker_svc_fn.return_value = mock_docker_svc

        prov = self._make_provisioner()
        request = _make_request(user_id="user-1")
        result = await prov.provision_from_github(request)

        assert isinstance(result, ContainerInfo)
        assert result.repo_url == "https://github.com/owner/repo"
        assert result.user_id == "user-1"
        assert result.original_repo_owner == "owner"
        assert result.original_repo_name == "repo"
        assert result.system_fork_owner == "system-owner"
        # The secrets-service id must land on the field that is READ BACK. This
        # path used to write it only to ``user_token_id``, which nothing consults.
        assert result.secret_config_id == "sec-id-123"
        assert result.user_token_id == "sec-id-123"
        # Atomic single-field HSET, not a whole-registry replace — see
        # tests/unit/test_registry_sibling_replica_safety.py.
        mock_docker_svc.registry.upsert.assert_called_once()
        mock_docker_svc.registry.save.assert_not_called()

    @patch("src.core.github_container_provisioner.get_docker_service")
    @patch("src.core.github_container_provisioner.get_github_token_manager")
    @patch(
        "src.core.github_container_provisioner.GitHubContainerProvisioner._validate_github_repository_access"
    )
    async def test_provision_failure_cleans_up(
        self, mock_validate, mock_token_mgr_fn, mock_docker_svc_fn
    ):
        """When provisioning fails after container creation, cleanup is attempted."""
        mock_token_mgr = _token_manager(secret_config_id="sec-id-123")
        mock_token_mgr_fn.return_value = mock_token_mgr

        prov = self._make_provisioner()
        # Make clone fail after container is created
        prov.container_manager.client.containers.run.side_effect = OSError("clone fail")

        request = _make_request()
        with pytest.raises(ValueError, match="Failed to clone"):
            await prov.provision_from_github(request)

    @patch("src.core.github_container_provisioner.get_docker_service")
    @patch("src.core.github_container_provisioner.get_github_token_manager")
    @patch(
        "src.core.github_container_provisioner.GitHubContainerProvisioner._validate_github_repository_access"
    )
    async def test_k8s_delegation(self, mock_validate, mock_token_mgr_fn, mock_docker_svc_fn):
        """When use_k8s is True, delegates to k8s_provisioner."""
        with patch.dict(os.environ, {"KUBERNETES_IN_CLUSTER": "true"}, clear=False):
            from src.core.github_container_provisioner import GitHubContainerProvisioner

            manager = MagicMock()
            k8s_prov = MagicMock()
            expected = _make_container_info()
            k8s_prov.provision_from_github = AsyncMock(return_value=expected)

            with patch(
                "src.core.github_container_provisioner.get_k8s_container_provisioner",
                return_value=k8s_prov,
            ):
                provisioner = GitHubContainerProvisioner(manager)

            request = _make_request()
            result = await provisioner.provision_from_github(request)
            assert result == expected
            k8s_prov.provision_from_github.assert_awaited_once_with(request)


# ---------------------------------------------------------------------------
# get_github_container_provisioner (module-level singleton)
# ---------------------------------------------------------------------------


class TestGetGithubContainerProvisioner:
    def test_returns_instance(self):
        with patch("src.core.container_manager_core.get_container_manager") as mock_cm:
            mock_cm.return_value = MagicMock()
            import src.core.github_container_provisioner as mod

            mod._github_provisioner = None
            result = mod.get_github_container_provisioner()
            assert isinstance(result, mod.GitHubContainerProvisioner)
            # Reset global
            mod._github_provisioner = None

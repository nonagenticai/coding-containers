"""Tests for services/docker_services/container_lifecycle.py."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import docker.errors
import pytest

from src.clients.realtime_client import BroadcastTenantMissingError
from src.core.container_metadata_registry import ContainerMetadataRegistry
from src.models.container_models import (
    ContainerInfo,
    ContainerSpec,
    ContainerStatus,
    GitHubRepoRequest,
)
from src.services.docker_services.container_lifecycle import ContainerLifecycleManager


def _token_manager(
    system_token: str | None = "sys-token-123",
    system_owner: str | None = "nonagenticai",
    secret_config_id: str | None = None,
) -> MagicMock:
    """Autospec'd stub of the REAL GitHubTokenManager.

    ``create_autospec`` rather than a bare ``MagicMock``: a signature-agnostic
    double accepts any call and therefore cannot fail, and this fleet has already
    shipped fabricated passes that way. It also gives the async methods AsyncMock
    automatically, which a plain MagicMock does not — an un-awaitable return is
    how these tests broke when the store moved to the secrets service.
    """
    from unittest.mock import create_autospec

    from src.clients.github_client import GitHubTokenManager

    mgr = create_autospec(GitHubTokenManager, instance=True)
    mgr.get_system_owner.return_value = system_owner
    mgr.store_user_token.return_value = secret_config_id
    mgr.resolve_provision_secret_config_id.return_value = secret_config_id
    # Fixtures clone ``owner/...`` — a CUSTOMER owner — so the Docker path now
    # asks coding-github. Default to an honest "public, no token" answer; the
    # credential-ladder tests below override it.
    from src.clients.github_client import CloneCredential

    mgr.get_app_clone_credential.return_value = CloneCredential(
        ok=True, token=None, visibility="public", source="none"
    )
    return mgr


#: The OWNING tenant of a provisioned container. In production it comes from the
#: authenticated caller (``User.effective_tenant_id`` via
#: ``get_current_user_with_tenant`` on ``POST /api/containers/``), is passed to
#: ``DockerService.create_container(tenant_id=...)``, and is forwarded here.
#: It is required, never defaulted.
TEST_TENANT = "0195f48b-d000-7000-8000-000000000003"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(
    client: MagicMock | None = None,
    registry: MagicMock | None = None,
) -> ContainerLifecycleManager:
    """Build a ContainerLifecycleManager with mocked deps."""
    if registry is None:
        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.load.return_value = {}
        registry.save = MagicMock()
    mgr = ContainerLifecycleManager(
        docker_client=client,
        registry=registry,
        project_name="test-project",
    )
    return mgr


def _make_container_info(name: str = "test-container", **overrides) -> ContainerInfo:
    defaults = dict(
        container_name=name,
        api_endpoint="http://localhost:8687/api/containers/test-container",
        internal_endpoint="",
        public_endpoint="",
        code_graph_id="",
        codebase_path="/app/provisioned_containers/test-container/repo",
        repo_url="https://github.com/owner/repo",
        branch="main",
        status=ContainerStatus.RUNNING,
    )
    defaults.update(overrides)
    return ContainerInfo(**defaults)


def _docker_not_found_error():
    """Return a docker.errors.NotFound-like exception."""
    import docker.errors

    resp = MagicMock()
    resp.status_code = 404
    resp.reason = "Not Found"
    return docker.errors.NotFound("Not Found", response=resp)


# ===========================================================================
# _validate_container_name
# ===========================================================================


class TestValidateContainerName:
    def test_valid_name_unchanged(self):
        mgr = _make_manager()
        assert mgr._validate_container_name("codegraph-my-repo") == "codegraph-my-repo"

    def test_short_name_gets_padded(self):
        mgr = _make_manager()
        assert mgr._validate_container_name("ab") == "codegraph-container"

    def test_empty_name_gets_padded(self):
        mgr = _make_manager()
        assert mgr._validate_container_name("") == "codegraph-container"

    def test_invalid_chars_stripped(self):
        mgr = _make_manager()
        result = mgr._validate_container_name("valid@name#here")
        # Invalid chars replaced with dash; starts alphanumeric so no prefix
        assert "@" not in result
        assert "#" not in result

    def test_leading_non_alnum_gets_prefix(self):
        mgr = _make_manager()
        result = mgr._validate_container_name("-leading-dash")
        assert result.startswith("codegraph-")

    def test_dots_and_underscores_allowed(self):
        mgr = _make_manager()
        assert mgr._validate_container_name("my_container.name") == "my_container.name"

    def test_long_name_truncated_to_128(self):
        """Docker container names must not exceed 128 characters."""
        mgr = _make_manager()
        long_name = "a" * 200
        result = mgr._validate_container_name(long_name)
        assert len(result) <= 128

    def test_exactly_128_chars_unchanged(self):
        mgr = _make_manager()
        name = "a" * 128
        assert mgr._validate_container_name(name) == name

    def test_exactly_129_chars_truncated(self):
        mgr = _make_manager()
        name = "a" * 129
        result = mgr._validate_container_name(name)
        assert len(result) == 128

    def test_sanitized_then_truncated(self):
        """Sanitization + truncation applied together: invalid chars replaced, then truncated."""
        mgr = _make_manager()
        name = "valid@" * 50  # 300 chars, lots of invalid @
        result = mgr._validate_container_name(name)
        assert "@" not in result
        assert len(result) <= 128


# ===========================================================================
# _extract_repo_name
# ===========================================================================


class TestExtractRepoName:
    def test_simple_url(self):
        mgr = _make_manager()
        assert mgr._extract_repo_name("https://github.com/owner/my-repo") == "my-repo"

    def test_with_git_suffix(self):
        mgr = _make_manager()
        assert mgr._extract_repo_name("https://github.com/owner/my-repo.git") == "my-repo"

    def test_nested_path(self):
        mgr = _make_manager()
        assert mgr._extract_repo_name("https://github.com/org/sub/deep-repo") == "deep-repo"


# ===========================================================================
# _find_available_port
# ===========================================================================


class TestFindAvailablePort:
    def test_skips_used_ports(self):
        client = MagicMock()
        c1 = MagicMock()
        c1.ports = {"8080/tcp": [{"HostPort": "8700"}]}
        client.containers.list.return_value = [c1]
        mgr = _make_manager(client=client)

        with patch("socket.socket") as mock_sock_cls:
            sock_instance = MagicMock()
            mock_sock_cls.return_value.__enter__ = MagicMock(return_value=sock_instance)
            mock_sock_cls.return_value.__exit__ = MagicMock(return_value=False)
            sock_instance.connect_ex.return_value = 1  # port free
            port = mgr._find_available_port()

        # Should skip 8700 (used by container) and return 8701
        assert port == 8701

    def test_skips_bound_ports(self):
        client = MagicMock()
        client.containers.list.return_value = []
        mgr = _make_manager(client=client)

        call_count = 0

        with patch("socket.socket") as mock_sock_cls:
            sock_instance = MagicMock()
            mock_sock_cls.return_value.__enter__ = MagicMock(return_value=sock_instance)
            mock_sock_cls.return_value.__exit__ = MagicMock(return_value=False)

            def connect_side_effect(addr):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    return 0  # port in use
                return 1  # port free

            sock_instance.connect_ex.side_effect = connect_side_effect
            port = mgr._find_available_port()

        assert port == 8702

    def test_no_client_returns_default(self):
        mgr = _make_manager(client=None)
        with patch("socket.socket") as mock_sock_cls:
            sock_instance = MagicMock()
            mock_sock_cls.return_value.__enter__ = MagicMock(return_value=sock_instance)
            mock_sock_cls.return_value.__exit__ = MagicMock(return_value=False)
            sock_instance.connect_ex.return_value = 1
            port = mgr._find_available_port()
        assert port == 8700


# ===========================================================================
# _get_container_status
# ===========================================================================


class TestGetContainerStatus:
    def test_running_and_healthy(self):
        client = MagicMock()
        container = MagicMock()
        container.status = "running"
        exec_result = MagicMock()
        exec_result.exit_code = 0
        container.exec_run.return_value = exec_result
        client.containers.get.return_value = container
        mgr = _make_manager(client=client)
        assert mgr._get_container_status("c1") == ContainerStatus.RUNNING

    def test_running_and_unhealthy(self):
        client = MagicMock()
        container = MagicMock()
        container.status = "running"
        exec_result = MagicMock()
        exec_result.exit_code = 1
        container.exec_run.return_value = exec_result
        client.containers.get.return_value = container
        mgr = _make_manager(client=client)
        assert mgr._get_container_status("c1") == ContainerStatus.ERROR

    def test_running_exec_raises_still_running(self):
        import docker.errors

        client = MagicMock()
        container = MagicMock()
        container.status = "running"
        container.exec_run.side_effect = docker.errors.DockerException("exec error")
        client.containers.get.return_value = container
        mgr = _make_manager(client=client)
        assert mgr._get_container_status("c1") == ContainerStatus.RUNNING

    def test_stopped(self):
        client = MagicMock()
        container = MagicMock()
        container.status = "exited"
        client.containers.get.return_value = container
        mgr = _make_manager(client=client)
        assert mgr._get_container_status("c1") == ContainerStatus.STOPPED

    def test_creating(self):
        client = MagicMock()
        container = MagicMock()
        container.status = "created"
        client.containers.get.return_value = container
        mgr = _make_manager(client=client)
        assert mgr._get_container_status("c1") == ContainerStatus.CREATING

    def test_unknown_status(self):
        client = MagicMock()
        container = MagicMock()
        container.status = "paused"
        client.containers.get.return_value = container
        mgr = _make_manager(client=client)
        assert mgr._get_container_status("c1") == ContainerStatus.ERROR

    def test_not_found(self):

        client = MagicMock()
        client.containers.get.side_effect = _docker_not_found_error()
        mgr = _make_manager(client=client)
        assert mgr._get_container_status("missing") == ContainerStatus.ERROR

    def test_no_client(self):
        mgr = _make_manager(client=None)
        assert mgr._get_container_status("c1") == ContainerStatus.ERROR


# ===========================================================================
# _validate_container_prerequisites
# ===========================================================================


class TestValidateContainerPrerequisites:
    def test_all_pass(self):
        client = MagicMock()
        client.images.get.return_value = MagicMock()
        client.networks.get.return_value = MagicMock()
        mgr = _make_manager(client=client)
        spec = ContainerSpec(name="test", image="test:latest", port=8700)

        with patch("os.path.exists", return_value=True), patch("os.access", return_value=True):
            result = mgr._validate_container_prerequisites(spec)

        assert result["docker_available"] is True
        assert result["image_exists"] is True
        assert result["network_exists"] is True
        assert result["socket_accessible"] is True

    def test_no_client(self):
        mgr = _make_manager(client=None)
        spec = ContainerSpec(name="test", image="test:latest", port=8700)
        result = mgr._validate_container_prerequisites(spec)
        assert result["docker_available"] is False
        assert result["image_exists"] is False

    def test_missing_image(self):
        import docker.errors

        client = MagicMock()
        client.images.get.side_effect = docker.errors.DockerException("not found")
        client.networks.get.return_value = MagicMock()
        mgr = _make_manager(client=client)
        spec = ContainerSpec(name="test", image="missing:latest", port=8700)

        with patch("os.path.exists", return_value=True), patch("os.access", return_value=True):
            result = mgr._validate_container_prerequisites(spec)

        assert result["docker_available"] is True
        assert result["image_exists"] is False

    def test_missing_network(self):
        import docker.errors

        client = MagicMock()
        client.images.get.return_value = MagicMock()
        client.networks.get.side_effect = docker.errors.DockerException("not found")
        mgr = _make_manager(client=client)
        spec = ContainerSpec(name="test", image="test:latest", port=8700)

        with patch("os.path.exists", return_value=True), patch("os.access", return_value=True):
            result = mgr._validate_container_prerequisites(spec)

        assert result["network_exists"] is False

    def test_socket_not_accessible(self):
        client = MagicMock()
        client.images.get.return_value = MagicMock()
        client.networks.get.return_value = MagicMock()
        mgr = _make_manager(client=client)
        spec = ContainerSpec(name="test", image="test:latest", port=8700)

        with patch("os.path.exists", return_value=False):
            result = mgr._validate_container_prerequisites(spec)

        assert result["socket_accessible"] is False


# ===========================================================================
# start_container
# ===========================================================================


class TestStartContainer:
    @pytest.mark.asyncio
    async def test_success(self):
        client = MagicMock()
        container = MagicMock()
        client.containers.get.return_value = container
        mgr = _make_manager(client=client)

        result = await mgr.start_container("docker-123")
        client.containers.get.assert_called_once_with("docker-123")
        container.start.assert_called_once()
        assert result is None  # start_container returns None on success

    @pytest.mark.asyncio
    async def test_no_client_raises(self):
        mgr = _make_manager(client=None)
        with pytest.raises(Exception, match="Docker not available"):
            await mgr.start_container("docker-123")

    @pytest.mark.asyncio
    async def test_docker_error_raises(self):
        client = MagicMock()
        client.containers.get.side_effect = docker.errors.DockerException("docker boom")
        mgr = _make_manager(client=client)

        with pytest.raises(docker.errors.DockerException, match="docker boom"):
            await mgr.start_container("docker-123")


# ===========================================================================
# stop_container
# ===========================================================================


class TestStopContainer:
    @pytest.mark.asyncio
    async def test_success(self):
        client = MagicMock()
        container = MagicMock()
        client.containers.get.return_value = container
        mgr = _make_manager(client=client)

        await mgr.stop_container("docker-123")
        container.stop.assert_called_once_with(timeout=10)

    @pytest.mark.asyncio
    async def test_not_found_graceful(self):
        client = MagicMock()
        client.containers.get.side_effect = _docker_not_found_error()
        mgr = _make_manager(client=client)
        # Should not raise
        await mgr.stop_container("docker-123")

    @pytest.mark.asyncio
    async def test_no_client_warning_only(self):
        mgr = _make_manager(client=None)
        # Should not raise, just log warning
        await mgr.stop_container("docker-123")


# ===========================================================================
# remove_container_by_name
# ===========================================================================


class TestRemoveContainerByName:
    @pytest.mark.asyncio
    async def test_success(self):
        client = MagicMock()
        container = MagicMock()
        client.containers.get.return_value = container

        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.remove.return_value = True

        mgr = _make_manager(client=client, registry=registry)
        result = await mgr.remove_container_by_name("my-container")

        assert result is True
        container.stop.assert_called_once()
        container.remove.assert_called_once()
        # Single-field HDEL for exactly this container. A whole-registry save()
        # from a stale snapshot would also delete siblings — see
        # tests/unit/test_registry_sibling_replica_safety.py.
        registry.remove.assert_called_once_with("my-container")
        registry.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_not_found_in_docker(self):
        client = MagicMock()
        client.containers.get.side_effect = _docker_not_found_error()

        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.load.return_value = {"my-container": _make_container_info("my-container")}
        registry.save = MagicMock()

        mgr = _make_manager(client=client, registry=registry)
        result = await mgr.remove_container_by_name("my-container")
        assert result is True

    @pytest.mark.asyncio
    async def test_not_in_registry(self):
        client = MagicMock()
        container = MagicMock()
        client.containers.get.return_value = container

        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.remove.return_value = False  # nothing was there to delete

        mgr = _make_manager(client=client, registry=registry)
        result = await mgr.remove_container_by_name("ghost")
        assert result is True
        # The HDEL is still issued (it is idempotent and reports what it did);
        # what must never happen is a whole-registry replace.
        registry.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_client(self):
        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.load.return_value = {}
        mgr = _make_manager(client=None, registry=registry)
        result = await mgr.remove_container_by_name("any")
        assert result is True


# ===========================================================================
# remove_container_by_id
# ===========================================================================


class TestRemoveContainerById:
    @pytest.mark.asyncio
    async def test_success(self):
        client = MagicMock()
        container = MagicMock()
        client.containers.get.return_value = container
        mgr = _make_manager(client=client)

        await mgr.remove_container_by_id("abc123")
        container.remove.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_not_found(self):
        client = MagicMock()
        client.containers.get.side_effect = _docker_not_found_error()
        mgr = _make_manager(client=client)
        # Should not raise
        await mgr.remove_container_by_id("abc123")

    @pytest.mark.asyncio
    async def test_no_client(self):
        mgr = _make_manager(client=None)
        # Should not raise
        await mgr.remove_container_by_id("abc123")


# ===========================================================================
# remove_container(ContainerInfo)
# ===========================================================================


class TestRemoveContainerInfo:
    @pytest.mark.asyncio
    async def test_with_docker_client(self):
        client = MagicMock()
        docker_container = MagicMock()
        client.containers.get.return_value = docker_container

        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.load.return_value = {"myc": _make_container_info("myc")}
        registry.save = MagicMock()

        mgr = _make_manager(client=client, registry=registry)
        info = _make_container_info("myc")
        result = await mgr.remove_container(info)

        assert result is True
        docker_container.stop.assert_called_once_with(timeout=10)
        docker_container.remove.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_without_docker_client(self):
        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.remove.return_value = True

        mgr = _make_manager(client=None, registry=registry)
        info = _make_container_info("myc")
        result = await mgr.remove_container(info)

        assert result is True
        registry.remove.assert_called_once_with("myc")
        registry.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_docker_not_found(self):
        client = MagicMock()
        client.containers.get.side_effect = _docker_not_found_error()

        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.load.return_value = {"myc": _make_container_info("myc")}
        registry.save = MagicMock()

        mgr = _make_manager(client=client, registry=registry)
        info = _make_container_info("myc")
        result = await mgr.remove_container(info)
        assert result is True

    @pytest.mark.asyncio
    async def test_registry_cleanup_happens(self):
        client = MagicMock()
        docker_container = MagicMock()
        client.containers.get.return_value = docker_container

        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.remove.return_value = True

        mgr = _make_manager(client=client, registry=registry)
        info = _make_container_info("myc")
        result = await mgr.remove_container(info)

        assert result is True
        # Only "myc" is targeted. This test always MEANT "a sibling survives"
        # (it used to assert `"other" in saved_data`); expressing it as a
        # single-field HDEL is what actually makes that true under concurrency,
        # because a whole-registry save() prunes whatever the snapshot missed.
        registry.remove.assert_called_once_with("myc")
        registry.save.assert_not_called()


# ===========================================================================
# create_container
# ===========================================================================


class TestCreateContainer:
    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    @patch("src.services.docker_services.container_lifecycle.get_container_manager")
    @patch("src.services.docker_services.container_lifecycle.get_github_token_manager")
    @patch("src.services.docker_services.container_lifecycle.get_settings")
    async def test_success_generates_name_from_repo(
        self, mock_settings, mock_token_mgr_fn, mock_cm_fn, mock_broadcast_fn
    ):
        settings = MagicMock()
        settings.main_api_endpoint = "http://localhost:8687"
        mock_settings.return_value = settings

        token_mgr = _token_manager(system_token="sys-token-123")
        mock_token_mgr_fn.return_value = token_mgr

        cm = MagicMock()
        cm._resolve_shared_volume_name.return_value = "shared-vol"
        mock_cm_fn.return_value = cm

        broadcast = AsyncMock()
        mock_broadcast_fn.return_value = broadcast

        client = MagicMock()
        client.containers.run = MagicMock()  # clone container

        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.load.return_value = {}
        registry.save = MagicMock()

        mgr = _make_manager(client=client, registry=registry)

        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/my-repo",
            branch="main",
        )
        result = await mgr.create_container(request, tenant_id=TEST_TENANT)

        assert result.container_name == "codegraph-my-repo"
        assert result.status == ContainerStatus.RUNNING
        assert result.repo_url == "https://github.com/owner/my-repo"
        # Two saves: the provisional CREATING row (written before the "creating"
        # broadcast so the event is deliverable) and the RUNNING promotion.
        # Two atomic upserts: the provisional CREATING row, then RUNNING.
        assert registry.upsert.call_count == 2
        client.containers.run.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    @patch("src.services.docker_services.container_lifecycle.get_container_manager")
    @patch("src.services.docker_services.container_lifecycle.get_github_token_manager")
    @patch("src.services.docker_services.container_lifecycle.get_settings")
    async def test_pr_name_generation(
        self, mock_settings, mock_token_mgr_fn, mock_cm_fn, mock_broadcast_fn
    ):
        settings = MagicMock()
        settings.main_api_endpoint = "http://localhost:8687"
        mock_settings.return_value = settings

        token_mgr = _token_manager(system_token=None)
        mock_token_mgr_fn.return_value = token_mgr

        cm = MagicMock()
        cm._resolve_shared_volume_name.return_value = "shared-vol"
        mock_cm_fn.return_value = cm

        broadcast = AsyncMock()
        mock_broadcast_fn.return_value = broadcast

        client = MagicMock()

        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.load.return_value = {}
        registry.save = MagicMock()

        mgr = _make_manager(client=client, registry=registry)

        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/my-repo",
            branch="main",
            provision_type="pull_request",
            pr_number=42,
        )
        result = await mgr.create_container(request, tenant_id=TEST_TENANT)

        assert result.container_name == "codegraph-my-repo-pr-42"
        assert result.pr_number == 42

    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    @patch("src.services.docker_services.container_lifecycle.get_container_manager")
    @patch("src.services.docker_services.container_lifecycle.get_github_token_manager")
    @patch("src.services.docker_services.container_lifecycle.get_settings")
    async def test_token_storage(
        self, mock_settings, mock_token_mgr_fn, mock_cm_fn, mock_broadcast_fn
    ):
        settings = MagicMock()
        settings.main_api_endpoint = "http://localhost:8687"
        mock_settings.return_value = settings

        token_mgr = _token_manager(system_token="sys-tok", secret_config_id="sec-id-999")
        mock_token_mgr_fn.return_value = token_mgr

        cm = MagicMock()
        cm._resolve_shared_volume_name.return_value = "shared-vol"
        mock_cm_fn.return_value = cm

        broadcast = AsyncMock()
        mock_broadcast_fn.return_value = broadcast

        client = MagicMock()
        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.load.return_value = {}
        registry.save = MagicMock()

        mgr = _make_manager(client=client, registry=registry)
        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo",
            branch="main",
            access_token="ghp_user_token_abc",
            user_id="user-123",
        )
        result = await mgr.create_container(request, tenant_id=TEST_TENANT)

        token_mgr.resolve_provision_secret_config_id.assert_awaited_once()
        # The id the secrets service minted must land on BOTH fields.
        # ``secret_config_id`` is the one that is READ BACK (secret_resolver,
        # git_push, crud_write); persisting only ``user_token_id`` — which nothing
        # in this service consults — is the dead end this replaced.
        assert result.secret_config_id == "sec-id-999"
        assert result.user_token_id == "sec-id-999"
        assert result.github_access_token is None  # deprecated field

    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    @patch("src.services.docker_services.container_lifecycle.get_container_manager")
    @patch("src.services.docker_services.container_lifecycle.get_github_token_manager")
    @patch("src.services.docker_services.container_lifecycle.get_settings")
    async def test_broadcast_events_attempted(
        self, mock_settings, mock_token_mgr_fn, mock_cm_fn, mock_broadcast_fn
    ):
        settings = MagicMock()
        settings.main_api_endpoint = "http://localhost:8687"
        mock_settings.return_value = settings

        token_mgr = _token_manager(system_token=None)
        mock_token_mgr_fn.return_value = token_mgr

        cm = MagicMock()
        cm._resolve_shared_volume_name.return_value = "v"
        mock_cm_fn.return_value = cm

        broadcast = AsyncMock()
        mock_broadcast_fn.return_value = broadcast

        client = MagicMock()
        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.load.return_value = {}
        registry.save = MagicMock()

        mgr = _make_manager(client=client, registry=registry)
        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo",
            branch="main",
        )
        await mgr.create_container(request, tenant_id=TEST_TENANT)

        # get_broadcast_service is called twice: once for "creating", once for "running".
        # The actual .broadcast() may fail due to missing enum attributes (caught internally),
        # but the service is always fetched.
        assert mock_broadcast_fn.call_count >= 2

    @pytest.mark.asyncio
    async def test_no_client_raises(self):
        mgr = _make_manager(client=None)
        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo",
            branch="main",
        )
        with pytest.raises(
            Exception, match="Docker client not available - cannot create container"
        ):
            await mgr.create_container(request, tenant_id=TEST_TENANT)

    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    @patch("src.services.docker_services.container_lifecycle.get_container_manager")
    @patch("src.services.docker_services.container_lifecycle.get_github_token_manager")
    @patch("src.services.docker_services.container_lifecycle.get_settings")
    async def test_docker_error_triggers_cleanup(
        self, mock_settings, mock_token_mgr_fn, mock_cm_fn, mock_broadcast_fn
    ):
        settings = MagicMock()
        settings.main_api_endpoint = "http://localhost:8687"
        mock_settings.return_value = settings

        token_mgr = _token_manager(system_token=None)
        mock_token_mgr_fn.return_value = token_mgr

        cm = MagicMock()
        cm._resolve_shared_volume_name.return_value = "v"
        mock_cm_fn.return_value = cm

        broadcast = AsyncMock()
        mock_broadcast_fn.return_value = broadcast

        client = MagicMock()
        client.containers.run.side_effect = docker.errors.DockerException("docker run failed")

        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.load.return_value = {}
        registry.save = MagicMock()

        mgr = _make_manager(client=client, registry=registry)
        # Patch remove_container_by_name to track cleanup
        mgr.remove_container_by_name = AsyncMock(return_value=True)

        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo",
            branch="main",
        )
        with pytest.raises(RuntimeError, match="Container creation failed"):
            await mgr.create_container(request, tenant_id=TEST_TENANT)

        mgr.remove_container_by_name.assert_awaited_once()


# ===========================================================================
# create_container — "creating" broadcast must be deliverable
#
# coding-realtime's publish endpoint resolves the container for the calling
# tenant BEFORE publishing (validate_and_get_container_for_tenant) and maps
# every miss to a 404. So the "creating" event is only deliverable if the
# registry row already exists when it is emitted. These tests assert the
# ORDERING, not merely that a broadcast happened.
# ===========================================================================


class _FakeRegistry:
    """In-memory stand-in for ContainerMetadataRegistry.

    Real store semantics (not a MagicMock) so a test can ask the question that
    actually matters: *at the moment the event was emitted, was the container
    resolvable?*
    """

    def __init__(self) -> None:
        self.store: dict[str, ContainerInfo] = {}
        self.save_count = 0

    def load(self) -> dict[str, ContainerInfo]:
        return dict(self.store)

    def save(self, data: dict[str, ContainerInfo]) -> None:
        # Mirrors the real replace semantics: fields absent from `data` are
        # pruned. Production code must NOT reach here for single-entry
        # mutations — see tests/unit/test_registry_sibling_replica_safety.py.
        self.save_count += 1
        self.store = dict(data)

    def upsert(self, name: str, info: ContainerInfo) -> None:
        self.save_count += 1
        self.store[name] = info

    def remove(self, name: str) -> bool:
        return self.store.pop(name, None) is not None

    def get_container(self, name: str) -> ContainerInfo | None:
        return self.store.get(name)


def _lifecycle_mocks(mock_settings, mock_token_mgr_fn, mock_cm_fn):
    settings = MagicMock()
    settings.main_api_endpoint = "http://localhost:8687"
    mock_settings.return_value = settings

    token_mgr = _token_manager(system_token=None)
    mock_token_mgr_fn.return_value = token_mgr

    cm = MagicMock()
    cm._resolve_shared_volume_name.return_value = "shared-vol"
    mock_cm_fn.return_value = cm


class TestCreatingEventIsDeliverable:
    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    @patch("src.services.docker_services.container_lifecycle.get_container_manager")
    @patch("src.services.docker_services.container_lifecycle.get_github_token_manager")
    @patch("src.services.docker_services.container_lifecycle.get_settings")
    async def test_container_is_registered_before_creating_is_broadcast(
        self, mock_settings, mock_token_mgr_fn, mock_cm_fn, mock_broadcast_fn
    ):
        """The registry write MUST precede the 'creating' broadcast.

        Asserts on ordering: the recorded operation log has the CREATING
        registry save strictly before the creating broadcast. Broadcast first
        (the old order) means coding-realtime 404s the event and drops it.
        """
        _lifecycle_mocks(mock_settings, mock_token_mgr_fn, mock_cm_fn)

        registry = _FakeRegistry()
        ops: list[str] = []

        original_upsert = registry.upsert

        def _recording_upsert(name, info):
            original_upsert(name, info)
            if name == "codegraph-repo":
                ops.append(f"registry_save:{info.status.value}")

        registry.upsert = _recording_upsert  # type: ignore[method-assign]

        async def _recording_broadcast(**kwargs):
            ops.append(f"broadcast:{kwargs['data']['status']}")
            return SimpleNamespace(ok=True, status_code=200, not_found=False, error=None)

        broadcast = MagicMock()
        broadcast.broadcast = _recording_broadcast
        mock_broadcast_fn.return_value = broadcast

        mgr = _make_manager(client=MagicMock(), registry=registry)
        await mgr.create_container(
            GitHubRepoRequest(repo_url="https://github.com/owner/repo", branch="main"),
            tenant_id=TEST_TENANT,
        )

        assert ops[0] == "registry_save:creating", ops
        assert ops[1] == "broadcast:creating", ops
        assert ops.index("registry_save:creating") < ops.index("broadcast:creating")
        # And the RUNNING promotion still happens afterwards.
        assert "registry_save:running" in ops
        assert ops.index("registry_save:running") < ops.index("broadcast:running")

    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    @patch("src.services.docker_services.container_lifecycle.get_container_manager")
    @patch("src.services.docker_services.container_lifecycle.get_github_token_manager")
    @patch("src.services.docker_services.container_lifecycle.get_settings")
    async def test_container_is_resolvable_for_its_tenant_when_creating_fires(
        self, mock_settings, mock_token_mgr_fn, mock_cm_fn, mock_broadcast_fn
    ):
        """At broadcast time the container must be in the registry, under the
        broadcasting tenant — that is exactly what the publish-side ownership
        check looks up before it will accept the event."""
        _lifecycle_mocks(mock_settings, mock_token_mgr_fn, mock_cm_fn)

        registry = _FakeRegistry()
        seen: list[tuple[str, bool, str | None]] = []

        async def _resolving_broadcast(**kwargs):
            name = kwargs["data"]["container_name"]
            entry = registry.store.get(name)
            seen.append(
                (kwargs["data"]["status"], entry is not None, entry.tenant_id if entry else None)
            )
            return SimpleNamespace(ok=True, status_code=200, not_found=False, error=None)

        broadcast = MagicMock()
        broadcast.broadcast = _resolving_broadcast
        mock_broadcast_fn.return_value = broadcast

        mgr = _make_manager(client=MagicMock(), registry=registry)
        await mgr.create_container(
            GitHubRepoRequest(
                repo_url="https://github.com/owner/repo",
                branch="main",
                tenant_id="tenant-77",
            )
        )

        # First event emitted is "creating", and it WAS resolvable, for the
        # right tenant. Pre-fix this was (creating, False, None) -> hard 404.
        assert seen[0] == ("creating", True, "tenant-77")

    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    @patch("src.services.docker_services.container_lifecycle.get_container_manager")
    @patch("src.services.docker_services.container_lifecycle.get_github_token_manager")
    @patch("src.services.docker_services.container_lifecycle.get_settings")
    async def test_failed_create_leaves_no_provisional_registry_row(
        self, mock_settings, mock_token_mgr_fn, mock_cm_fn, mock_broadcast_fn
    ):
        """Registering early must not leak a half-registered container: the
        clone blows up after the provisional row is written, and the row is
        gone by the time the error surfaces."""
        _lifecycle_mocks(mock_settings, mock_token_mgr_fn, mock_cm_fn)

        broadcast = AsyncMock()
        mock_broadcast_fn.return_value = broadcast

        client = MagicMock()
        client.containers.run.side_effect = docker.errors.DockerException("clone blew up")

        registry = _FakeRegistry()
        mgr = _make_manager(client=client, registry=registry)

        with pytest.raises(RuntimeError, match="Container creation failed"):
            await mgr.create_container(
                GitHubRepoRequest(repo_url="https://github.com/owner/repo", branch="main"),
                tenant_id=TEST_TENANT,
            )

        assert registry.store == {}, f"leaked provisional row: {registry.store}"

    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    @patch("src.services.docker_services.container_lifecycle.get_container_manager")
    @patch("src.services.docker_services.container_lifecycle.get_github_token_manager")
    @patch("src.services.docker_services.container_lifecycle.get_settings")
    async def test_undeclared_failure_also_drops_the_provisional_row(
        self, mock_settings, mock_token_mgr_fn, mock_cm_fn, mock_broadcast_fn
    ):
        """A failure type the typed except clause does not cover (here: an
        OSError from the clone) must still roll the provisional row back."""
        _lifecycle_mocks(mock_settings, mock_token_mgr_fn, mock_cm_fn)

        broadcast = AsyncMock()
        mock_broadcast_fn.return_value = broadcast

        client = MagicMock()
        client.containers.run.side_effect = OSError("socket gone")

        registry = _FakeRegistry()
        mgr = _make_manager(client=client, registry=registry)

        with pytest.raises(OSError, match="socket gone"):
            await mgr.create_container(
                GitHubRepoRequest(repo_url="https://github.com/owner/repo", branch="main"),
                tenant_id=TEST_TENANT,
            )

        assert registry.store == {}, f"leaked provisional row: {registry.store}"

    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    @patch("src.services.docker_services.container_lifecycle.get_container_manager")
    @patch("src.services.docker_services.container_lifecycle.get_github_token_manager")
    @patch("src.services.docker_services.container_lifecycle.get_settings")
    async def test_rejected_creating_broadcast_is_logged_and_create_still_succeeds(
        self, mock_settings, mock_token_mgr_fn, mock_cm_fn, mock_broadcast_fn, caplog
    ):
        """A rejected publish is surfaced (ok is derived from the result, not
        assumed) but is best-effort: it must not fail the create."""
        _lifecycle_mocks(mock_settings, mock_token_mgr_fn, mock_cm_fn)

        broadcast = MagicMock()
        broadcast.broadcast = AsyncMock(
            return_value=SimpleNamespace(
                ok=False, status_code=404, not_found=True, error="container not found"
            )
        )
        mock_broadcast_fn.return_value = broadcast

        registry = _FakeRegistry()
        mgr = _make_manager(client=MagicMock(), registry=registry)

        with caplog.at_level(logging.WARNING):
            result = await mgr.create_container(
                GitHubRepoRequest(repo_url="https://github.com/owner/repo", branch="main"),
                tenant_id=TEST_TENANT,
            )

        assert result.status == ContainerStatus.RUNNING  # create not broken
        assert any(
            "NOT accepted" in r.message and "404" in r.getMessage() for r in caplog.records
        ), [r.getMessage() for r in caplog.records]


# ===========================================================================
# _verify_container_running
# ===========================================================================


class TestVerifyContainerRunning:
    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    async def test_running_and_healthy(self, mock_broadcast_fn):
        broadcast = AsyncMock()
        mock_broadcast_fn.return_value = broadcast

        client = MagicMock()
        container = MagicMock()
        container.status = "running"
        exec_result = MagicMock()
        exec_result.exit_code = 0
        container.exec_run.return_value = exec_result
        client.containers.get.return_value = container

        mgr = _make_manager(client=client)
        result = await mgr._verify_container_running("d1", "c1", timeout=5, tenant_id=TEST_TENANT)
        assert result is True

    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    async def test_running_and_unhealthy(self, mock_broadcast_fn):
        broadcast = AsyncMock()
        mock_broadcast_fn.return_value = broadcast

        client = MagicMock()
        container = MagicMock()
        container.status = "running"
        exec_result = MagicMock()
        exec_result.exit_code = 1
        container.exec_run.return_value = exec_result
        client.containers.get.return_value = container

        mgr = _make_manager(client=client)
        result = await mgr._verify_container_running("d1", "c1", timeout=5, tenant_id=TEST_TENANT)
        # Still returns True - container is running
        assert result is True

    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    async def test_exited_returns_false(self, mock_broadcast_fn):
        broadcast = AsyncMock()
        mock_broadcast_fn.return_value = broadcast

        client = MagicMock()
        container = MagicMock()
        container.status = "exited"
        container.logs.return_value = b"some logs"
        client.containers.get.return_value = container

        mgr = _make_manager(client=client)
        result = await mgr._verify_container_running("d1", "c1", timeout=5, tenant_id=TEST_TENANT)
        assert result is False

    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.time")
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    async def test_timeout_returns_false(self, mock_broadcast_fn, mock_time):
        broadcast = AsyncMock()
        mock_broadcast_fn.return_value = broadcast

        # Simulate time advancing past timeout
        mock_time.time.side_effect = [0.0, 100.0]
        mock_time.sleep = MagicMock()

        client = MagicMock()
        container = MagicMock()
        container.status = "creating"
        client.containers.get.return_value = container

        mgr = _make_manager(client=client)
        result = await mgr._verify_container_running("d1", "c1", timeout=5, tenant_id=TEST_TENANT)
        assert result is False


# ===========================================================================
# Branch name sanitization in create_container
# ===========================================================================


class TestBranchNameSanitization:
    """Verify that shell-injectable branch names are rejected."""

    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    @patch("src.services.docker_services.container_lifecycle.get_container_manager")
    @patch("src.services.docker_services.container_lifecycle.get_github_token_manager")
    @patch("src.services.docker_services.container_lifecycle.get_settings")
    async def test_valid_branch_names_accepted(
        self, mock_settings, mock_token_mgr_fn, mock_cm_fn, mock_broadcast_fn
    ):
        """Normal branch names like 'main', 'feature/foo', 'v1.0' should work."""
        settings = MagicMock()
        settings.main_api_endpoint = "http://localhost:8687"
        mock_settings.return_value = settings

        token_mgr = _token_manager(system_token=None)
        mock_token_mgr_fn.return_value = token_mgr

        cm = MagicMock()
        cm._resolve_shared_volume_name.return_value = "shared-vol"
        mock_cm_fn.return_value = cm

        broadcast = AsyncMock()
        mock_broadcast_fn.return_value = broadcast

        client = MagicMock()
        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.load.return_value = {}
        registry.save = MagicMock()

        mgr = _make_manager(client=client, registry=registry)

        for branch in ["main", "develop", "feature/new-thing", "release/v1.0.3", "fix-123"]:
            request = GitHubRepoRequest(
                repo_url="https://github.com/owner/my-repo",
                branch=branch,
            )
            result = await mgr.create_container(request, tenant_id=TEST_TENANT)
            assert result.status == ContainerStatus.RUNNING

    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    @patch("src.services.docker_services.container_lifecycle.get_container_manager")
    @patch("src.services.docker_services.container_lifecycle.get_github_token_manager")
    @patch("src.services.docker_services.container_lifecycle.get_settings")
    async def test_shell_injection_branch_rejected(
        self, mock_settings, mock_token_mgr_fn, mock_cm_fn, mock_broadcast_fn
    ):
        """Branch names with shell metacharacters must be rejected."""
        settings = MagicMock()
        settings.main_api_endpoint = "http://localhost:8687"
        mock_settings.return_value = settings

        token_mgr = _token_manager(system_token="sys-tok")
        mock_token_mgr_fn.return_value = token_mgr

        cm = MagicMock()
        cm._resolve_shared_volume_name.return_value = "shared-vol"
        mock_cm_fn.return_value = cm

        broadcast = AsyncMock()
        mock_broadcast_fn.return_value = broadcast

        client = MagicMock()
        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.load.return_value = {}
        registry.save = MagicMock()

        mgr = _make_manager(client=client, registry=registry)
        # Patch remove_container_by_name since creation will fail and trigger cleanup
        mgr.remove_container_by_name = AsyncMock(return_value=True)

        # These branch names contain shell injection characters
        # The model validation (pattern) should reject them, but even if
        # somehow bypassed, the lifecycle manager should catch them.
        # We use the internal method directly here to test the runtime check.
        for bad_branch in ["main; rm -rf /", "$(whoami)", "main`id`", "main|cat /etc/passwd"]:
            # Create request bypassing pydantic validation (simulating an internal caller)
            request = MagicMock(spec=GitHubRepoRequest)
            request.repo_url = "https://github.com/owner/repo"
            request.branch = bad_branch
            request.container_name = None
            request.provision_type = "branch"
            request.pr_number = None
            request.access_token = None
            request.user_id = None
            request.description = None
            # A real GitHubRepoRequest always carries tenant_id (defaults None);
            # the spec'd mock does not auto-expose pydantic fields, so set it.
            request.tenant_id = None

            with pytest.raises(Exception, match="(Invalid branch name|Container creation failed)"):
                await mgr.create_container(request, tenant_id=TEST_TENANT)


# ===========================================================================
# _find_available_port error message
# ===========================================================================


class TestFindAvailablePortExhausted:
    def test_all_ports_exhausted_error_message(self):
        """When all ports are in use, the error message should show correct range."""
        client = MagicMock()
        client.containers.list.return_value = []
        mgr = _make_manager(client=client)

        with patch("socket.socket") as mock_sock_cls:
            sock_instance = MagicMock()
            mock_sock_cls.return_value.__enter__ = MagicMock(return_value=sock_instance)
            mock_sock_cls.return_value.__exit__ = MagicMock(return_value=False)
            # All ports in use
            sock_instance.connect_ex.return_value = 0

            with pytest.raises(RuntimeError, match="8700-8799"):
                mgr._find_available_port()


# ===========================================================================
# remove_container with docker errors
# ===========================================================================


class TestRemoveContainerErrorPaths:
    @pytest.mark.asyncio
    async def test_remove_container_docker_error_still_cleans_registry(self):
        """remove_container cleans registry even when Docker removal fails."""
        client = MagicMock()
        docker_container = MagicMock()
        docker_container.stop.side_effect = docker.errors.DockerException("stop failed")
        client.containers.get.return_value = docker_container

        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.remove.return_value = True

        mgr = _make_manager(client=client, registry=registry)
        info = _make_container_info("myc")
        result = await mgr.remove_container(info)

        assert result is True
        registry.remove.assert_called_once_with("myc")

    @pytest.mark.asyncio
    async def test_remove_container_no_docker_registry_error(self):
        """remove_container without Docker returns False when registry fails."""
        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.remove.side_effect = OSError("corrupt registry")

        mgr = _make_manager(client=None, registry=registry)
        info = _make_container_info("myc")
        result = await mgr.remove_container(info)

        assert result is False

    @pytest.mark.asyncio
    async def test_remove_container_with_docker_full_exception(self):
        """remove_container returns False on unexpected exception."""
        client = MagicMock()
        docker_container = MagicMock()
        docker_container.stop.return_value = None
        docker_container.remove.return_value = None
        client.containers.get.return_value = docker_container

        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.remove.side_effect = OSError("registry exploded")

        mgr = _make_manager(client=client, registry=registry)
        info = _make_container_info("myc")
        result = await mgr.remove_container(info)

        assert result is False


# ===========================================================================
# Model-level branch validation (GitHubRepoRequest)
# ===========================================================================


class TestGitHubRepoRequestBranchValidation:
    """Verify that the branch field pattern rejects shell injection at the model level."""

    def test_valid_branches(self):
        for branch in ["main", "develop", "feature/new-thing", "release/v1.0.3", "fix_123"]:
            req = GitHubRepoRequest(
                repo_url="https://github.com/owner/my-repo",
                branch=branch,
            )
            assert req.branch == branch

    def test_shell_injection_rejected(self):
        from pydantic import ValidationError

        for bad_branch in ["main; rm -rf /", "$(whoami)", "main`id`", "main|cat"]:
            with pytest.raises(ValidationError):
                GitHubRepoRequest(
                    repo_url="https://github.com/owner/my-repo",
                    branch=bad_branch,
                )

    def test_branch_max_length(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GitHubRepoRequest(
                repo_url="https://github.com/owner/my-repo",
                branch="a" * 257,
            )

    def test_branch_256_chars_ok(self):
        req = GitHubRepoRequest(
            repo_url="https://github.com/owner/my-repo",
            branch="a" * 256,
        )
        assert len(req.branch) == 256

    def test_branch_with_dots_and_dashes(self):
        req = GitHubRepoRequest(
            repo_url="https://github.com/owner/my-repo",
            branch="release/v2.0.0-rc.1",
        )
        assert req.branch == "release/v2.0.0-rc.1"

    def test_spaces_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GitHubRepoRequest(
                repo_url="https://github.com/owner/my-repo",
                branch="main branch",
            )


# ---------------------------------------------------------------------------
# _verify_container_running - additional coverage (lines 515-519, 549-556, 561-563)
# ---------------------------------------------------------------------------


class TestVerifyContainerRunningExtended:
    @pytest.mark.asyncio
    async def test_health_check_exception_returns_true(self):
        """Health check error returns True (container running, health not critical) (lines 515-519)."""
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.exec_run.side_effect = docker.errors.DockerException("health check error")
        mock_container.reload.return_value = None
        mock_client.containers.get.return_value = mock_container

        mgr = _make_manager(client=mock_client)
        result = await mgr._verify_container_running(
            "abc123", "test-ctr", timeout=5, tenant_id=TEST_TENANT
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_container_other_status_waits(self):
        """Container in 'creating' status waits then times out (lines 549-556)."""
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.status = "creating"
        mock_container.reload.return_value = None
        mock_client.containers.get.return_value = mock_container

        mgr = _make_manager(client=mock_client)
        with patch("time.sleep"):
            result = await mgr._verify_container_running(
                "abc123", "test-ctr", timeout=1, tenant_id=TEST_TENANT
            )
        assert result is False

    @pytest.mark.asyncio
    async def test_container_check_exception(self):
        """Exception during container check waits and retries (lines 554-556)."""
        mock_client = MagicMock()
        mock_client.containers.get.side_effect = docker.errors.DockerException("docker error")

        mgr = _make_manager(client=mock_client)
        with patch("time.sleep"):
            result = await mgr._verify_container_running(
                "abc123", "test-ctr", timeout=1, tenant_id=TEST_TENANT
            )
        assert result is False

    @pytest.mark.asyncio
    async def test_outer_exception_returns_false(self):
        """Outer exception handler returns False (lines 561-563)."""
        mgr = _make_manager(client=None)
        result = await mgr._verify_container_running(
            "abc123", "test-ctr", timeout=1, tenant_id=TEST_TENANT
        )
        assert result is False


# ---------------------------------------------------------------------------
# _get_container_status - additional status branches (lines 594-596)
# ---------------------------------------------------------------------------


class TestGetContainerStatusExtended:
    def test_unknown_status(self):
        """Unknown container status returns ERROR (line 588-589)."""
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.status = "unknown_state"
        mock_client.containers.get.return_value = mock_container

        mgr = _make_manager(client=mock_client)
        result = mgr._get_container_status("test-ctr")
        assert result == ContainerStatus.ERROR

    def test_docker_exception_returns_error(self):
        """Docker exception returns ERROR (lines 594-596)."""
        mock_client = MagicMock()
        mock_client.containers.get.side_effect = docker.errors.DockerException("docker error")

        mgr = _make_manager(client=mock_client)
        result = mgr._get_container_status("test-ctr")
        assert result == ContainerStatus.ERROR


# ---------------------------------------------------------------------------
# _find_available_port - edge cases (lines 642-645, 654-655)
# ---------------------------------------------------------------------------


class TestFindAvailablePortExtended:
    def test_docker_port_check_failure(self):
        """Docker container port listing fails gracefully (lines 644-645)."""
        mock_client = MagicMock()
        mock_client.containers.list.side_effect = docker.errors.DockerException("docker error")

        mgr = _make_manager(client=mock_client)
        with patch("socket.socket") as mock_socket:
            mock_sock = MagicMock()
            mock_sock.__enter__ = MagicMock(return_value=mock_sock)
            mock_sock.__exit__ = MagicMock(return_value=False)
            mock_sock.connect_ex.return_value = 1  # Port available
            mock_socket.return_value = mock_sock

            port = mgr._find_available_port()
            assert port >= 8700

    def test_socket_check_exception(self):
        """Socket check exception is caught (line 654-655)."""
        mock_client = MagicMock()
        mock_client.containers.list.return_value = []

        mgr = _make_manager(client=mock_client)
        with patch("socket.socket") as mock_socket:
            # First call raises, second call succeeds
            mock_sock_fail = MagicMock()
            mock_sock_fail.__enter__ = MagicMock(side_effect=OSError("cannot bind"))
            mock_sock_fail.__exit__ = MagicMock(return_value=False)

            mock_sock_ok = MagicMock()
            mock_sock_ok.__enter__ = MagicMock(return_value=mock_sock_ok)
            mock_sock_ok.__exit__ = MagicMock(return_value=False)
            mock_sock_ok.connect_ex.return_value = 1

            mock_socket.side_effect = [mock_sock_fail, mock_sock_ok] + [mock_sock_ok] * 100

            port = mgr._find_available_port()
            assert port >= 8700


class TestOwningTenantIsThreadedIntoEveryBroadcast:
    """The tenant is REQUIRED and comes from the authenticated caller.

    Pre-fix, ``create_container`` read only ``request.tenant_id`` — the request
    *body* field, which ``POST /api/containers/`` never populates (it resolves
    the tenant from the authenticated user and hands it to
    ``DockerService.create_container`` as a separate argument). So on this path
    the tenant was ``None``: ``ContainerInfo`` was registered unowned and every
    lifecycle broadcast went out with no ``X-Tenant-Id``, reaching
    coding-realtime only via its pod-CIDR trust fallback.
    """

    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    @patch("src.services.docker_services.container_lifecycle.get_container_manager")
    @patch("src.services.docker_services.container_lifecycle.get_github_token_manager")
    @patch("src.services.docker_services.container_lifecycle.get_settings")
    async def test_every_broadcast_carries_the_owning_tenant(
        self, mock_settings, mock_token_mgr_fn, mock_cm_fn, mock_broadcast_fn
    ):
        settings = MagicMock()
        settings.main_api_endpoint = "http://localhost:8687"
        mock_settings.return_value = settings

        token_mgr = _token_manager(system_token=None)
        mock_token_mgr_fn.return_value = token_mgr

        cm = MagicMock()
        cm._resolve_shared_volume_name.return_value = "v"
        mock_cm_fn.return_value = cm

        broadcast = AsyncMock()
        mock_broadcast_fn.return_value = broadcast

        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.load.return_value = {}
        registry.save = MagicMock()

        mgr = _make_manager(client=MagicMock(), registry=registry)
        # NOTE: the request body carries NO tenant. The tenant comes from the
        # caller argument alone — which is exactly the case that used to
        # broadcast header-less.
        request = GitHubRepoRequest(repo_url="https://github.com/owner/repo", branch="main")
        result = await mgr.create_container(request, tenant_id=TEST_TENANT)

        assert broadcast.broadcast.await_count >= 2
        for call in broadcast.broadcast.await_args_list:
            assert call.kwargs["tenant_id"] == TEST_TENANT, (
                "a lifecycle broadcast went out without the owning tenant"
            )
        # ...and the registered container is owned, so coding-realtime's
        # publish-side ownership check can resolve it.
        assert result.tenant_id == TEST_TENANT

    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    @patch("src.services.docker_services.container_lifecycle.get_github_token_manager")
    @patch("src.services.docker_services.container_lifecycle.get_settings")
    async def test_no_resolvable_tenant_raises_before_anything_is_created(
        self, mock_settings, mock_token_mgr_fn, mock_broadcast_fn
    ):
        """Negative control: a tenant-less provision must FAIL LOUDLY, not fall
        back to a default and not quietly broadcast without a header."""
        settings = MagicMock()
        settings.main_api_endpoint = "http://localhost:8687"
        mock_settings.return_value = settings
        broadcast = AsyncMock()
        mock_broadcast_fn.return_value = broadcast

        client = MagicMock()
        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.load.return_value = {}
        registry.save = MagicMock()

        mgr = _make_manager(client=client, registry=registry)
        request = GitHubRepoRequest(repo_url="https://github.com/owner/repo", branch="main")

        with pytest.raises(BroadcastTenantMissingError):
            await mgr.create_container(request, tenant_id=None)

        # Nothing was provisioned, registered, or published.
        client.containers.run.assert_not_called()
        registry.save.assert_not_called()
        broadcast.broadcast.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("src.services.docker_services.container_lifecycle.get_broadcast_service")
    @patch("src.services.docker_services.container_lifecycle.get_container_manager")
    @patch("src.services.docker_services.container_lifecycle.get_github_token_manager")
    @patch("src.services.docker_services.container_lifecycle.get_settings")
    async def test_request_body_tenant_is_the_fallback(
        self, mock_settings, mock_token_mgr_fn, mock_cm_fn, mock_broadcast_fn
    ):
        """An explicit body tenant still works (the k8s provisioner path uses
        it), it is just no longer the ONLY source."""
        settings = MagicMock()
        settings.main_api_endpoint = "http://localhost:8687"
        mock_settings.return_value = settings

        token_mgr = _token_manager(system_token=None)
        mock_token_mgr_fn.return_value = token_mgr

        cm = MagicMock()
        cm._resolve_shared_volume_name.return_value = "v"
        mock_cm_fn.return_value = cm

        broadcast = AsyncMock()
        mock_broadcast_fn.return_value = broadcast

        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.load.return_value = {}
        registry.save = MagicMock()

        mgr = _make_manager(client=MagicMock(), registry=registry)
        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo",
            branch="main",
            tenant_id=TEST_TENANT,
        )
        result = await mgr.create_container(request)

        assert result.tenant_id == TEST_TENANT
        assert broadcast.broadcast.await_args_list[0].kwargs["tenant_id"] == TEST_TENANT


# ===========================================================================
# Docker clone credential ladder (same policy as the K8s provisioner)
# ===========================================================================

_PAT = "ghp_SYSTEM_PAT_must_never_touch_a_customer_repo"  # pragma: allowlist secret


class TestDockerCloneCredentialLadder:
    """BYO -> App token (platform: else 412, never the PAT) -> customer public/none -> 412.

    Every customer case has the PAT configured, so a regression that re-opens
    the PAT fallback has a credential to reach for and turns red.
    """

    @staticmethod
    async def _create(request, credential, system_token=_PAT):
        token_mgr = _token_manager(system_token=system_token)
        token_mgr.get_app_clone_credential.return_value = credential
        client = MagicMock()
        registry = MagicMock(spec=ContainerMetadataRegistry)
        registry.load.return_value = {}
        mgr = _make_manager(client=client, registry=registry)
        settings = MagicMock()
        settings.main_api_endpoint = "http://localhost:8687"
        cm = MagicMock()
        cm._resolve_shared_volume_name.return_value = "v"
        with (
            patch(
                "src.services.docker_services.container_lifecycle.get_github_token_manager",
                return_value=token_mgr,
            ),
            patch(
                "src.services.docker_services.container_lifecycle.get_settings",
                return_value=settings,
            ),
            patch(
                "src.services.docker_services.container_lifecycle.get_container_manager",
                return_value=cm,
            ),
            patch(
                "src.services.docker_services.container_lifecycle.get_broadcast_service",
                return_value=AsyncMock(),
            ),
        ):
            try:
                await mgr.create_container(request, tenant_id=TEST_TENANT)
                exc = None
            except Exception as e:  # noqa: BLE001 - the test inspects it
                exc = e
        return token_mgr, client, registry, exc

    @staticmethod
    def _clone_call(client):
        client.containers.run.assert_called_once()
        kwargs = client.containers.run.call_args.kwargs
        return kwargs["environment"], kwargs["command"][-1]

    @pytest.mark.asyncio
    async def test_customer_owner_no_app_install_is_412_and_never_the_pat(self):
        from src.clients.github_client import CloneCredential
        from src.core.kubernetes_container_provisioner import GitCredentialUnavailableError

        request = GitHubRepoRequest(repo_url="https://github.com/acme-corp/app", branch="main")
        token_mgr, client, registry, exc = await self._create(
            request,
            CloneCredential(ok=True, token=None, visibility="private", reason="not_installed"),
        )
        assert isinstance(exc, GitCredentialUnavailableError)
        assert exc.status_code == 412
        assert exc.reason == "github_app_not_installed"
        client.containers.run.assert_not_called()
        registry.upsert.assert_not_called()
        assert not hasattr(token_mgr, "get_system_token")
        call = token_mgr.get_app_clone_credential.await_args
        assert call.kwargs["tenant_id"] == TEST_TENANT

    @pytest.mark.asyncio
    async def test_customer_owner_clones_with_the_tenant_bound_app_token(self):
        from src.clients.github_client import CloneCredential

        request = GitHubRepoRequest(repo_url="https://github.com/acme-corp/app", branch="main")
        token_mgr, client, _registry, exc = await self._create(
            request, CloneCredential(ok=True, token="ghs_app", visibility="private")
        )
        assert exc is None
        env, cmd = self._clone_call(client)
        assert env == {"GIT_TOKEN": "ghs_app"}
        # The token rides the clone container's env into a bound credential
        # helper; it never enters the URL git records as remote.origin.url.
        assert "credential.https://github.com.helper=" in cmd
        assert " https://github.com/acme-corp/app " in cmd
        assert "x-access-token:" not in cmd and "@github.com" not in cmd
        assert "ghs_app" not in cmd
        assert "remote set-url origin https://github.com/acme-corp/app" in cmd
        assert not hasattr(token_mgr, "get_system_token")

    @pytest.mark.asyncio
    async def test_customer_public_repo_clones_with_no_credential(self):
        from src.clients.github_client import CloneCredential

        request = GitHubRepoRequest(repo_url="https://github.com/acme-corp/app", branch="main")
        token_mgr, client, _registry, exc = await self._create(
            request, CloneCredential(ok=True, token=None, visibility="public")
        )
        assert exc is None
        env, cmd = self._clone_call(client)
        assert env == {}
        assert "git clone --depth 1 --branch 'main' https://github.com/acme-corp/app " in cmd
        assert not hasattr(token_mgr, "get_system_token")

    @pytest.mark.asyncio
    async def test_byo_token_wins_for_a_customer_owner(self):
        from src.clients.github_client import CloneCredential

        request = GitHubRepoRequest(
            repo_url="https://github.com/acme-corp/app",
            branch="main",
            access_token="byo-customer-token",
        )
        token_mgr, client, _registry, exc = await self._create(
            request, CloneCredential(ok=True, token=None, visibility="private")
        )
        assert exc is None
        env, _cmd = self._clone_call(client)
        assert env == {"GIT_TOKEN": "byo-customer-token"}
        token_mgr.get_app_clone_credential.assert_not_awaited()
        assert not hasattr(token_mgr, "get_system_token")

    @pytest.mark.asyncio
    async def test_platform_owner_never_gets_the_pat(self):
        from src.clients.github_client import CloneCredential

        request = GitHubRepoRequest(repo_url="https://github.com/nonagenticai/app", branch="main")
        token_mgr, client, _registry, exc = await self._create(
            request, CloneCredential(ok=True, token=None, visibility="private")
        )
        # "No App token" for a platform owner is a 412 — the PAT is not a rung.
        assert exc is not None and exc.reason == "platform_credential_unavailable"
        token_mgr.get_app_clone_credential.assert_awaited_once()
        assert not hasattr(token_mgr, "get_system_token")

    @pytest.mark.asyncio
    async def test_platform_repo_forbidden_is_never_answered_with_the_pat(self):
        from src.clients.github_client import CloneCredential

        request = GitHubRepoRequest(repo_url="https://github.com/nonagenticai/app", branch="main")
        token_mgr, client, _registry, exc = await self._create(
            request,
            CloneCredential(
                ok=True, token=None, visibility="private", reason="platform_repo_forbidden"
            ),
        )
        assert exc is not None and exc.reason == "platform_repo_forbidden"
        assert exc.status_code == 403
        assert not hasattr(token_mgr, "get_system_token")
        client.containers.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_clone_url_is_the_canonical_one_not_the_raw_request(self):
        from src.clients.github_client import CloneCredential

        request = GitHubRepoRequest(repo_url="https://github.com/acme-corp/app.git/", branch="main")
        _token_mgr, client, _registry, exc = await self._create(
            request, CloneCredential(ok=True, token="ghs_app", visibility="private")
        )
        assert exc is None
        _env, cmd = self._clone_call(client)
        assert " https://github.com/acme-corp/app " in cmd
        # (the credential helper's path match legitimately names `app.git`)
        assert "github.com/acme-corp/app.git" not in cmd

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/nonagenticai%2Fr",
            "https://github.com/nonagenticai%2fr.git",
            "https://github.com/nonagenticai/x/../../victim/private",
        ],
    )
    async def test_encoded_slash_and_traversal_are_refused_before_any_credential(self, url):
        from src.clients.github_client import CloneCredential

        request = GitHubRepoRequest.model_construct(
            **{
                **GitHubRepoRequest(
                    repo_url="https://github.com/nonagenticai/app", branch="main"
                ).model_dump(),
                "repo_url": url,
            }
        )
        token_mgr, client, registry, exc = await self._create(
            request, CloneCredential(ok=True, token="ghs_app", visibility="private")
        )
        assert exc is not None
        assert "not a canonical GitHub repository URL" in str(exc)
        client.containers.run.assert_not_called()
        registry.upsert.assert_not_called()
        assert not hasattr(token_mgr, "get_system_token")
        token_mgr.get_app_clone_credential.assert_not_awaited()

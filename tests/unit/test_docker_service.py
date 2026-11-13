"""Tests for services/docker_service.py."""

import asyncio
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import docker.errors
import pytest

from src.models.container_models import ContainerInfo, ContainerStatus


class TestDockerServiceInit:
    @patch("src.services.docker_service.DockerAPIIntegration")
    @patch("src.services.docker_service.ContainerLifecycleManager")
    @patch("src.services.docker_service.ImageManager")
    @patch("src.services.docker_service.NetworkVolumeManager")
    @patch("src.services.docker_service.ContainerMonitor")
    @patch("src.services.docker_service.ContainerMetadataRegistry")
    @patch("src.services.docker_service.get_settings")
    def test_init_without_docker(
        self,
        mock_settings,
        mock_registry,
        mock_monitor,
        mock_net,
        mock_img,
        mock_lifecycle,
        mock_api,
    ):
        settings = MagicMock()
        settings.docker_base_port = 8700
        settings.worker.internal_worker_port = 8100
        mock_settings.return_value = settings

        api_instance = MagicMock()
        api_instance.get_docker_client.return_value = MagicMock()
        api_instance.is_docker_available.return_value = False
        mock_api.return_value = api_instance

        from src.services.docker_service import DockerService

        svc = DockerService()
        assert svc.base_port == 8700

    @patch("src.services.docker_service.DockerAPIIntegration")
    @patch("src.services.docker_service.ContainerLifecycleManager")
    @patch("src.services.docker_service.ImageManager")
    @patch("src.services.docker_service.NetworkVolumeManager")
    @patch("src.services.docker_service.ContainerMonitor")
    @patch("src.services.docker_service.ContainerMetadataRegistry")
    @patch("src.services.docker_service.get_settings")
    def test_is_docker_available_false(
        self,
        mock_settings,
        mock_registry,
        mock_monitor,
        mock_net,
        mock_img,
        mock_lifecycle,
        mock_api,
    ):
        settings = MagicMock()
        settings.docker_base_port = 8700
        settings.worker.internal_worker_port = 8100
        mock_settings.return_value = settings

        api_instance = MagicMock()
        api_instance.get_docker_client.return_value = MagicMock()
        api_instance.is_docker_available.return_value = False
        mock_api.return_value = api_instance

        from src.services.docker_service import DockerService

        svc = DockerService()
        assert svc.is_docker_available() is False

    @patch("src.services.docker_service.DockerAPIIntegration")
    @patch("src.services.docker_service.ContainerLifecycleManager")
    @patch("src.services.docker_service.ImageManager")
    @patch("src.services.docker_service.NetworkVolumeManager")
    @patch("src.services.docker_service.ContainerMonitor")
    @patch("src.services.docker_service.ContainerMetadataRegistry")
    @patch("src.services.docker_service.get_settings")
    def test_is_docker_available_true(
        self,
        mock_settings,
        mock_registry,
        mock_monitor,
        mock_net,
        mock_img,
        mock_lifecycle,
        mock_api,
    ):
        settings = MagicMock()
        settings.docker_base_port = 8700
        settings.worker.internal_worker_port = 8100
        mock_settings.return_value = settings

        api_instance = MagicMock()
        api_instance.get_docker_client.return_value = MagicMock()
        api_instance.is_docker_available.return_value = True
        mock_api.return_value = api_instance

        from src.services.docker_service import DockerService

        svc = DockerService()
        assert svc.is_docker_available() is True


# ---------------------------------------------------------------------------
# Helper to build a patched DockerService without real Docker
# ---------------------------------------------------------------------------

_SERVICE_PATCHES = [
    "src.services.docker_service.DockerAPIIntegration",
    "src.services.docker_service.ContainerLifecycleManager",
    "src.services.docker_service.ImageManager",
    "src.services.docker_service.NetworkVolumeManager",
    "src.services.docker_service.ContainerMonitor",
    "src.services.docker_service.ContainerMetadataRegistry",
    "src.services.docker_service.get_settings",
]


def _build_service(
    docker_available: bool = True,
    registry_data: dict | None = None,
):
    """Create a DockerService with all sub-components mocked."""
    with (
        patch("src.services.docker_service.get_settings") as mock_settings,
        patch("src.services.docker_service.ContainerMetadataRegistry") as mock_registry_cls,
        patch("src.services.docker_service.DockerAPIIntegration") as mock_api_cls,
        patch("src.services.docker_service.ContainerLifecycleManager") as _mock_lifecycle_cls,
        patch("src.services.docker_service.ImageManager") as _mock_img_cls,
        patch("src.services.docker_service.NetworkVolumeManager") as _mock_net_cls,
        patch("src.services.docker_service.ContainerMonitor") as _mock_monitor_cls,
    ):
        settings = MagicMock()
        settings.docker_base_port = 8700
        settings.worker.internal_worker_port = 8100
        mock_settings.return_value = settings

        api_instance = MagicMock()
        api_instance.get_docker_client.return_value = MagicMock()
        api_instance.is_docker_available.return_value = docker_available
        mock_api_cls.return_value = api_instance

        registry_instance = MagicMock()
        registry_instance.load.return_value = registry_data or {}
        registry_instance.save = MagicMock()
        mock_registry_cls.return_value = registry_instance

        from src.services.docker_service import DockerService

        svc = DockerService()
        svc.registry = registry_instance
    return svc


def _make_info(name: str = "test-container", **kw) -> ContainerInfo:
    defaults = dict(
        container_name=name,
        api_endpoint=f"http://localhost:8687/api/containers/{name}",
        internal_endpoint="",
        public_endpoint="",
        code_graph_id="",
        codebase_path=f"/app/provisioned_containers/{name}/repo",
        repo_url="https://github.com/owner/repo",
        branch="main",
        status=ContainerStatus.RUNNING,
    )
    defaults.update(kw)
    return ContainerInfo(**defaults)


# ===========================================================================
# list_containers
# ===========================================================================


class TestListContainers:
    def test_empty_list(self):
        svc = _build_service(registry_data={})
        result = svc.list_containers()
        assert result == []

    def test_filtered_by_tenant(self):
        c1 = _make_info("c1", tenant_id="tenant-a")
        c2 = _make_info("c2", tenant_id="tenant-b")
        svc = _build_service(registry_data={"c1": c1, "c2": c2})
        result = svc.list_containers(tenant_id="tenant-a")
        assert len(result) == 1
        assert result[0].container_name == "c1"

    def test_no_tenant_returns_all(self):
        c1 = _make_info("c1", tenant_id="t1")
        c2 = _make_info("c2", tenant_id="t2")
        svc = _build_service(registry_data={"c1": c1, "c2": c2})
        result = svc.list_containers(tenant_id=None)
        assert len(result) == 2


# ===========================================================================
# find_container_by_name
# ===========================================================================


class TestFindContainerByName:
    @pytest.mark.asyncio
    async def test_found(self):
        c1 = _make_info("c1")
        svc = _build_service(registry_data={"c1": c1})
        result = await svc.find_container_by_name("c1")
        assert result is not None
        assert result.container_name == "c1"

    @pytest.mark.asyncio
    async def test_not_found(self):
        svc = _build_service(registry_data={})
        # No registry entry AND no live Deployment to fall back to.
        mock_k8s = MagicMock()
        mock_k8s.is_available.return_value = True
        mock_k8s.reconstruct_container_info.return_value = None
        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            return_value=mock_k8s,
        ):
            result = await svc.find_container_by_name("ghost")
        assert result is None

    # =======================================================================
    # REGISTRY-MISS rebuild — the FOURTH site of the tenant gate.
    #
    # ⚠️ Until these tests existed, NOTHING in this file returned a container
    # from ``reconstruct_container_info`` — every case stubbed it to ``None``.
    # So the rebuild branch's tenant gate was never executed by any test, and
    # the site stayed vacuous through three rounds of review while the PR's own
    # route table marked ``/install-dependencies`` and ``/dependency-status``
    # "scoped". A stub that always returns None does not exercise the branch it
    # guards; it hides it.
    #
    # ⛔ This is not a read-only disclosure. ``container_dependencies.py:145``
    # and ``:243`` gate on this function, and past it ``ContainerInstaller``
    # calls ``get_container(name)`` with NO tenant argument, so package-manager
    # commands run inside the foreign pod.
    # =======================================================================

    @staticmethod
    def _k8s_returning(rebuilt):
        mock_k8s = MagicMock()
        mock_k8s.is_available.return_value = True
        mock_k8s.reconstruct_container_info.return_value = rebuilt
        return mock_k8s

    @pytest.mark.asyncio
    async def test_registry_miss_unrecoverable_owner_is_refused(self, caplog):
        """⭐ THE BLOCKER. A rebuilt workspace with NO recoverable owner must not
        be served to a tenant. ``tenant_id=None`` on a REBUILT container means
        ``_recover_tenant_id`` found neither the ``coding-job/tenant-id``
        annotation nor the ``tenant-id`` label, i.e. the owner is UNKNOWN — not
        ownerless — because the label is written only for tenants matching
        ``_K8S_LABEL_VALUE_RE``."""
        svc = _build_service(registry_data={})
        rebuilt = _make_info("aa-delivery-chat-bridge", tenant_id=None)
        with (
            caplog.at_level(logging.WARNING, logger="src.services.docker_service"),
            patch(
                "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
                return_value=self._k8s_returning(rebuilt),
            ),
        ):
            result = await svc.find_container_by_name(
                "aa-delivery-chat-bridge", tenant_id="0195f48b-d000-7000-8000-000000000003"
            )
        assert result is None, (
            "a workspace with no recoverable owner was served to a foreign tenant on the "
            "/install-dependencies + /dependency-status gate"
        )
        # The refusal must be VISIBLE. It returns the same ``None`` as a genuine
        # absence, so without a log line an operator sees a 404 against a plainly
        # Running Deployment with nothing explaining it. Asserted rather than
        # merely written, so the signal cannot be deleted silently later.
        assert any(
            r.levelname == "WARNING" and "UNRECOVERABLE" in r.getMessage() for r in caplog.records
        ), f"the fail-closed refusal emitted no WARNING: {[r.getMessage() for r in caplog.records]}"

    @pytest.mark.asyncio
    async def test_registry_miss_foreign_owner_is_refused(self):
        """CONTROL. The recovered-owner case was already correct — it is the
        positive half of the executable proof that isolated the blocker: same
        caller, same workspace, 404 when the owner IS recoverable and served
        when it is not."""
        svc = _build_service(registry_data={})
        rebuilt = _make_info(
            "aa-delivery-chat-bridge", tenant_id="0195f48b-d000-7000-8000-000000000001"
        )
        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            return_value=self._k8s_returning(rebuilt),
        ):
            result = await svc.find_container_by_name(
                "aa-delivery-chat-bridge", tenant_id="0195f48b-d000-7000-8000-000000000003"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_registry_miss_owner_still_reads_its_own(self):
        """POSITIVE CONTROL. Both refusals above are satisfiable by a gate that
        refuses every rebuilt container, which would break the registry-miss
        fallback outright — its whole purpose is keeping the OWNER working while
        the registry momentarily loses the row."""
        svc = _build_service(registry_data={})
        owner = "0195f48b-d000-7000-8000-000000000001"
        rebuilt = _make_info("aa-delivery-chat-bridge", tenant_id=owner)
        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            return_value=self._k8s_returning(rebuilt),
        ):
            result = await svc.find_container_by_name("aa-delivery-chat-bridge", tenant_id=owner)
        assert result is rebuilt

    @pytest.mark.asyncio
    async def test_registry_miss_untenanted_caller_still_gets_unknown_owner(self):
        """POSITIVE CONTROL. The refusal belongs to the TENANT gate, not to the
        rebuild. An untenanted/admin lookup must still resolve an
        unrecoverable-owner workspace, or the only role that could investigate
        it goes blind."""
        svc = _build_service(registry_data={})
        rebuilt = _make_info("aa-delivery-chat-bridge", tenant_id=None)
        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            return_value=self._k8s_returning(rebuilt),
        ):
            result = await svc.find_container_by_name("aa-delivery-chat-bridge")
        assert result is rebuilt

    @pytest.mark.asyncio
    async def test_registry_hit_null_tenant_still_fails_open(self):
        """⛔ CONTROL proving the change is scoped to the REBUILD branch. A
        registry ROW with no tenant is genuine legacy data this service wrote and
        keeps its documented fail-open. If this goes red, the fix stopped being
        targeted and became a fleet-wide gate flip."""
        legacy = _make_info("legacy-ws", tenant_id=None)
        svc = _build_service(registry_data={"legacy-ws": legacy})
        result = await svc.find_container_by_name("legacy-ws", tenant_id="any-tenant")
        assert result is not None
        assert result.container_name == "legacy-ws"

    @pytest.mark.asyncio
    async def test_tenant_validation_blocks(self):
        c1 = _make_info("c1", tenant_id="tenant-a")
        svc = _build_service(registry_data={"c1": c1})
        result = await svc.find_container_by_name("c1", tenant_id="tenant-b")
        assert result is None

    @pytest.mark.asyncio
    async def test_tenant_validation_allows(self):
        c1 = _make_info("c1", tenant_id="tenant-a")
        svc = _build_service(registry_data={"c1": c1})
        result = await svc.find_container_by_name("c1", tenant_id="tenant-a")
        assert result is not None
        assert result.container_name == "c1"
        assert result.tenant_id == "tenant-a"


# ===========================================================================
# find_existing_container
# ===========================================================================


class TestFindExistingContainer:
    @pytest.mark.asyncio
    async def test_match_by_url(self):
        c1 = _make_info("c1", repo_url="https://github.com/owner/repo")
        svc = _build_service(registry_data={"c1": c1})
        result = await svc.find_existing_container("https://github.com/owner/repo")
        assert result is not None
        assert result.container_name == "c1"

    @pytest.mark.asyncio
    async def test_match_by_url_with_git_suffix(self):
        c1 = _make_info("c1", repo_url="https://github.com/owner/repo")
        svc = _build_service(registry_data={"c1": c1})
        result = await svc.find_existing_container("https://github.com/owner/repo.git")
        assert result is not None
        assert result.container_name == "c1"

    @pytest.mark.asyncio
    async def test_no_match(self):
        c1 = _make_info("c1", repo_url="https://github.com/owner/other")
        svc = _build_service(registry_data={"c1": c1})
        result = await svc.find_existing_container("https://github.com/owner/repo")
        assert result is None

    @pytest.mark.asyncio
    async def test_k8s_running_not_error_when_absent_from_docker(self):
        """A K8s-provisioned match (no Docker container) must report RUNNING,
        not ERROR — the regression that blocked project_setup."""
        c1 = _make_info("c1", repo_url="https://github.com/owner/repo")
        svc = _build_service(registry_data={"c1": c1})
        svc.lifecycle_manager._get_container_status = MagicMock(
            side_effect=docker.errors.DockerException("no docker")
        )
        mock_k8s = MagicMock()
        mock_k8s.is_available.return_value = True
        mock_k8s.get_container_status = AsyncMock(return_value={"status": "running"})
        with (
            patch.dict(os.environ, {"KUBERNETES_IN_CLUSTER": "true"}),
            patch(
                "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
                return_value=mock_k8s,
            ),
        ):
            result = await svc.find_existing_container("https://github.com/owner/repo")
        assert result is not None
        assert result.status == ContainerStatus.RUNNING

    @pytest.mark.asyncio
    async def test_k8s_running_when_docker_getter_returns_error(self):
        """The real regression: lifecycle_manager._get_container_status RETURNS
        ERROR (it does not raise) for an unknown container, so a Docker-first
        approach never reaches the K8s query. In-cluster, K8s must be consulted
        FIRST and report RUNNING."""
        c1 = _make_info("c1", repo_url="https://github.com/owner/repo")
        svc = _build_service(registry_data={"c1": c1})
        # Docker getter returns ERROR (does NOT raise) — must NOT win.
        svc.lifecycle_manager._get_container_status = MagicMock(return_value=ContainerStatus.ERROR)
        mock_k8s = MagicMock()
        mock_k8s.is_available.return_value = True
        mock_k8s.get_container_status = AsyncMock(return_value={"status": "running"})
        with (
            patch.dict(os.environ, {"KUBERNETES_IN_CLUSTER": "true"}),
            patch(
                "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
                return_value=mock_k8s,
            ),
        ):
            result = await svc.find_container_by_name("c1")
        assert result is not None
        assert result.status == ContainerStatus.RUNNING

    @pytest.mark.asyncio
    async def test_k8s_starting_maps_creating(self):
        c1 = _make_info("c1", repo_url="https://github.com/owner/repo")
        svc = _build_service(registry_data={"c1": c1})
        svc.lifecycle_manager._get_container_status = MagicMock(
            side_effect=docker.errors.DockerException("no docker")
        )
        mock_k8s = MagicMock()
        mock_k8s.is_available.return_value = True
        mock_k8s.get_container_status = AsyncMock(return_value={"status": "starting"})
        with (
            patch.dict(os.environ, {"KUBERNETES_IN_CLUSTER": "true"}),
            patch(
                "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
                return_value=mock_k8s,
            ),
        ):
            result = await svc.find_existing_container("https://github.com/owner/repo")
        assert result.status == ContainerStatus.CREATING

    @pytest.mark.asyncio
    async def test_error_when_neither_docker_nor_k8s(self):
        c1 = _make_info("c1", repo_url="https://github.com/owner/repo")
        svc = _build_service(registry_data={"c1": c1})
        svc.lifecycle_manager._get_container_status = MagicMock(
            side_effect=docker.errors.DockerException("no docker")
        )
        with (
            patch.dict(os.environ, {"KUBERNETES_IN_CLUSTER": "true"}),
            patch(
                "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
                side_effect=ImportError("no k8s"),
            ),
        ):
            result = await svc.find_existing_container("https://github.com/owner/repo")
        assert result.status == ContainerStatus.ERROR


# ===========================================================================
# delete_container
# ===========================================================================


class TestDeleteContainer:
    @pytest.mark.asyncio
    async def test_success(self):
        c1 = _make_info("c1")
        svc = _build_service(registry_data={"c1": c1})
        svc.lifecycle_manager.remove_container_by_name = AsyncMock(return_value=True)
        result = await svc.delete_container("c1")
        assert result is True

    @pytest.mark.asyncio
    async def test_not_found(self):
        svc = _build_service(registry_data={})
        svc.lifecycle_manager.remove_container_by_name = AsyncMock(
            side_effect=docker.errors.DockerException("not found")
        )
        result = await svc.delete_container("ghost")
        assert result is False

    @pytest.mark.asyncio
    async def test_tenant_mismatch(self):
        c1 = _make_info("c1", tenant_id="tenant-a")
        svc = _build_service(registry_data={"c1": c1})
        result = await svc.delete_container("c1", tenant_id="tenant-b")
        assert result is False


# ===========================================================================
# get_language_container_image
# ===========================================================================


class TestGetLanguageContainerImage:
    def test_delegates_to_image_manager(self):
        """Verify the facade method delegates to image_manager correctly."""
        svc = _build_service()
        svc.image_manager.get_language_container_image.return_value = "python:3.11"

        result = svc.get_language_container_image("python")
        assert result == "python:3.11"
        svc.image_manager.get_language_container_image.assert_called_once_with("python")

    def test_passes_language_argument_through(self):
        """Verify that the language argument is passed unchanged to the image manager."""
        svc = _build_service()
        svc.image_manager.get_language_container_image.return_value = "any:image"

        svc.get_language_container_image("rust")
        svc.image_manager.get_language_container_image.assert_called_once_with("rust")

    def test_returns_whatever_image_manager_returns(self):
        """Facade should return image manager result without transformation."""
        svc = _build_service()
        svc.image_manager.get_language_container_image.return_value = "custom:v2.5"
        assert svc.get_language_container_image("go") == "custom:v2.5"


# ===========================================================================
# is_docker_available / retry_docker_connection
# ===========================================================================


class TestDockerAvailability:
    def test_is_docker_available_true(self):
        svc = _build_service(docker_available=True)
        assert svc.is_docker_available() is True

    def test_is_docker_available_false(self):
        svc = _build_service(docker_available=False)
        assert svc.is_docker_available() is False

    def test_retry_docker_connection_delegates(self):
        svc = _build_service(docker_available=False)
        svc.api_integration.retry_docker_connection = MagicMock(return_value=True)
        result = svc.retry_docker_connection()
        assert result is True
        svc.api_integration.retry_docker_connection.assert_called_once()

    def test_retry_docker_connection_returns_false(self):
        svc = _build_service(docker_available=False)
        svc.api_integration.retry_docker_connection = MagicMock(return_value=False)
        result = svc.retry_docker_connection()
        assert result is False


# ===========================================================================
# _normalize_repo_url
# ===========================================================================


class TestNormalizeRepoUrl:
    def test_strips_git_suffix(self):
        svc = _build_service()
        assert (
            svc._normalize_repo_url("https://github.com/owner/repo.git")
            == "https://github.com/owner/repo"
        )

    def test_strips_trailing_slash(self):
        svc = _build_service()
        assert (
            svc._normalize_repo_url("https://github.com/owner/repo/")
            == "https://github.com/owner/repo"
        )

    def test_https_url_unchanged(self):
        svc = _build_service()
        assert (
            svc._normalize_repo_url("https://github.com/owner/repo")
            == "https://github.com/owner/repo"
        )

    def test_ssh_url_converted(self):
        svc = _build_service()
        result = svc._normalize_repo_url("git@github.com:owner/repo.git")
        assert result == "https://github.com/owner/repo"

    def test_bare_path_gets_prefix(self):
        svc = _build_service()
        result = svc._normalize_repo_url("owner/repo")
        assert result == "https://github.com/owner/repo"

    def test_ssh_url_without_git_suffix(self):
        svc = _build_service()
        result = svc._normalize_repo_url("git@github.com:owner/repo")
        assert result == "https://github.com/owner/repo"

    def test_strips_both_git_suffix_and_trailing_slash(self):
        svc = _build_service()
        # .git takes precedence over trailing slash in this URL
        result = svc._normalize_repo_url("https://github.com/owner/repo.git")
        assert result == "https://github.com/owner/repo"

    def test_idempotent_normalization(self):
        """Normalizing an already-normalized URL should return the same result."""
        svc = _build_service()
        normalized = "https://github.com/owner/repo"
        assert svc._normalize_repo_url(normalized) == normalized
        # Double-normalize should also be stable
        assert svc._normalize_repo_url(svc._normalize_repo_url("owner/repo")) == normalized


# ===========================================================================
# create_container
# ===========================================================================


class TestCreateContainer:
    @pytest.mark.asyncio
    async def test_returns_existing_container(self):
        """create_container returns early when an existing container matches."""
        from src.models.container_models import GitHubRepoRequest

        c1 = _make_info("c1", repo_url="https://github.com/owner/repo")
        svc = _build_service(registry_data={"c1": c1})
        request = GitHubRepoRequest(repo_url="https://github.com/owner/repo", branch="main")
        result = await svc.create_container(request, tenant_id=None)
        assert result.container_name == "c1"

    @pytest.mark.asyncio
    async def test_history_cutoff_refuses_repo_url_dedupe_with_409(self):
        """history_cutoff on the repo_url dedupe path is a 409, never a silent reuse:
        the existing workspace's history was never cut, and cutting it now would
        rewrite git history another caller may hold."""
        from fastapi import HTTPException

        from src.models.container_models import GitHubRepoRequest

        c1 = _make_info("c1", repo_url="https://github.com/owner/repo")
        svc = _build_service(registry_data={"c1": c1})
        sha = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo", branch=sha, history_cutoff=True
        )
        with pytest.raises(HTTPException) as exc_info:
            await svc.create_container(request, tenant_id=None)
        assert exc_info.value.status_code == 409
        assert "history_cutoff refused" in exc_info.value.detail
        assert "c1" in exc_info.value.detail
        assert "https://github.com/owner/repo" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_history_cutoff_refuses_name_reuse_with_409(self):
        """Same refusal on the pinned-name reuse path; the usable existing container
        is left alone (no delete, no provision)."""
        from fastapi import HTTPException

        from src.models.container_models import GitHubRepoRequest

        existing = _make_info("it-flow1", repo_url="https://github.com/owner/repo")
        svc = _build_service(registry_data={"it-flow1": existing})
        svc._resolve_live_status = AsyncMock(return_value=ContainerStatus.RUNNING)
        svc.delete_container = AsyncMock(side_effect=AssertionError("must not tear down"))
        sha = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo",
            branch=sha,
            container_name="it-flow1",
            history_cutoff=True,
        )
        with pytest.raises(HTTPException) as exc_info:
            await svc.create_container(request, tenant_id=None)
        assert exc_info.value.status_code == 409
        assert "it-flow1" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_history_cutoff_is_threaded_to_k8s_provision(self):
        """A fresh provision passes the flag through to k8s.provision_container —
        the rebuild there DROPS any kwarg it is not handed."""
        from src.models.container_models import GitHubRepoRequest

        svc = _build_service(docker_available=True, registry_data={})
        # A FRESH provision: neither lookup finds anything, so the request reaches
        # the k8s provisioner (the registry MagicMock would otherwise "find" iso-1
        # in a terminal state and take the recreate path).
        svc.find_container_by_name = AsyncMock(return_value=None)
        svc.find_existing_container = AsyncMock(return_value=None)
        sha = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
        fresh = _make_info("iso-1", branch=sha, history_cutoff_applied=True)
        mock_k8s = MagicMock()
        mock_k8s.is_available.return_value = True
        mock_k8s.provision_container = AsyncMock(return_value=fresh)
        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo",
            branch=sha,
            container_name="iso-1",
            history_cutoff=True,
        )
        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            return_value=mock_k8s,
        ):
            result = await svc.create_container(request, tenant_id=None)
        assert mock_k8s.provision_container.await_args.kwargs["history_cutoff"] is True
        assert result.history_cutoff_applied is True

    @pytest.mark.asyncio
    async def test_history_cutoff_refused_on_docker_fallback(self):
        """No k8s → the Docker path would clone a branch tip with nothing cut → 400."""
        from fastapi import HTTPException

        from src.models.container_models import GitHubRepoRequest

        svc = _build_service(docker_available=True, registry_data={})
        svc.lifecycle_manager.create_container = AsyncMock(
            side_effect=AssertionError("docker path must not be reached")
        )
        sha = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo",
            branch=sha,
            container_name="iso-2",
            history_cutoff=True,
        )
        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
                return_value=None,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await svc.create_container(request, tenant_id=None)
        assert exc_info.value.status_code == 400
        assert "kubernetes provisioner" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_reuses_container_by_name_when_pinned(self):
        """When container_name is pinned, create reuses the same-named container
        (idempotent) rather than repo-based dedup or re-provisioning."""
        from src.models.container_models import GitHubRepoRequest

        existing = _make_info("it-flow1", repo_url="https://github.com/owner/repo")
        svc = _build_service(registry_data={"it-flow1": existing})
        svc._resolve_live_status = AsyncMock(return_value=ContainerStatus.RUNNING)
        svc.find_existing_container = AsyncMock(
            side_effect=AssertionError("must not repo-dedup when a name is pinned")
        )
        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo",
            branch="main",
            container_name="it-flow1",
        )
        result = await svc.create_container(request, tenant_id=None)
        assert result.container_name == "it-flow1"

    @pytest.mark.asyncio
    async def test_reuses_creating_container_by_name(self):
        """A still-provisioning (CREATING) same-named container is reused, NOT
        recreated — preserves the readiness-poll case the idempotency guard protects."""
        from src.models.container_models import GitHubRepoRequest

        existing = _make_info("it-flow1", repo_url="https://github.com/owner/repo")
        svc = _build_service(registry_data={"it-flow1": existing})
        svc._resolve_live_status = AsyncMock(return_value=ContainerStatus.CREATING)
        svc.delete_container = AsyncMock(
            side_effect=AssertionError("must not delete a still-provisioning container")
        )
        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo",
            branch="main",
            container_name="it-flow1",
        )
        result = await svc.create_container(request, tenant_id=None)
        assert result.container_name == "it-flow1"

    @pytest.mark.asyncio
    async def test_recreates_errored_container_by_name(self):
        """A same-named container in a terminal ERROR state (stale from a prior
        run under the same deterministic name) is torn down and re-provisioned
        fresh, instead of being reused (which returns container_status=error and
        blocks the card forever)."""
        from src.models.container_models import GitHubRepoRequest

        stale = _make_info("it-flow1", repo_url="https://github.com/owner/repo")
        svc = _build_service(docker_available=True, registry_data={"it-flow1": stale})
        svc._resolve_live_status = AsyncMock(return_value=ContainerStatus.ERROR)
        svc.delete_container = AsyncMock(return_value=True)

        fresh = _make_info("it-flow1")
        mock_k8s = MagicMock()
        mock_k8s.is_available.return_value = True
        mock_k8s.provision_container = AsyncMock(return_value=fresh)
        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo",
            branch="main",
            container_name="it-flow1",
        )
        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            return_value=mock_k8s,
        ):
            result = await svc.create_container(request, tenant_id=None)

        # stale record torn down, then a FRESH provision under the same name
        svc.delete_container.assert_awaited_once_with("it-flow1", tenant_id=None)
        mock_k8s.provision_container.assert_awaited_once()
        assert result.container_name == "it-flow1"

    @pytest.mark.asyncio
    async def test_provisions_when_named_container_absent(self):
        """A pinned name that isn't in the registry provisions under that exact name."""
        from src.models.container_models import GitHubRepoRequest

        svc = _build_service(docker_available=True, registry_data={})
        k8s_container = _make_info("it-flow1")
        mock_k8s = MagicMock()
        mock_k8s.is_available.return_value = True
        mock_k8s.reconstruct_container_info.return_value = None  # no existing deployment
        mock_k8s.provision_container = AsyncMock(return_value=k8s_container)
        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/newrepo",
            branch="main",
            container_name="it-flow1",
        )
        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            return_value=mock_k8s,
        ):
            result = await svc.create_container(request)
        assert result.container_name == "it-flow1"
        mock_k8s.provision_container.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_docker(self):
        """create_container falls back to Docker lifecycle_manager when K8s unavailable."""
        from src.models.container_models import GitHubRepoRequest

        svc = _build_service(docker_available=True, registry_data={})
        created = _make_info("new-repo-12345678")
        svc.lifecycle_manager.create_container = AsyncMock(return_value=created)

        request = GitHubRepoRequest(repo_url="https://github.com/owner/newrepo", branch="main")

        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            side_effect=ImportError("no k8s"),
        ):
            result = await svc.create_container(request)

        assert result.container_name == "new-repo-12345678"
        svc.lifecycle_manager.create_container.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_docker_fallback_forwards_the_authenticated_tenant(self):
        """The tenant must reach the lifecycle manager EXPLICITLY.

        The route resolves it from the authenticated caller
        (``User.effective_tenant_id``) and hands it to this method; it is not in
        the request body. Before this fix the docker fallback called
        ``lifecycle_manager.create_container(request)`` with no tenant at all, so
        the lifecycle manager fell back to ``request.tenant_id`` (``None``) and
        every lifecycle broadcast went out with no ``X-Tenant-Id`` header —
        reaching coding-realtime only through its pod-CIDR trust fallback.
        """
        from src.models.container_models import GitHubRepoRequest

        tenant = "0195f48b-d000-7000-8000-000000000003"
        svc = _build_service(docker_available=True, registry_data={})
        created = _make_info("new-repo-12345678")
        svc.lifecycle_manager.create_container = AsyncMock(return_value=created)

        # NOTE: the body carries NO tenant — the caller argument is the only source.
        request = GitHubRepoRequest(repo_url="https://github.com/owner/newrepo", branch="main")

        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            side_effect=ImportError("no k8s"),
        ):
            await svc.create_container(request, tenant_id=tenant)

        assert svc.lifecycle_manager.create_container.await_args.kwargs["tenant_id"] == tenant

    @pytest.mark.asyncio
    async def test_create_container_falls_back_to_docker_on_k8s_config_exception(self):
        """create_container falls back to Docker when k8s raises ConfigException.

        Regression: K8sContainerProvisioner.__init__ raises
        kubernetes.config.config_exception.ConfigException when no kubeconfig
        is reachable (and not in-cluster). That exception was not in the
        create_container k8s-attempt except tuple, so the fall-back branch
        was unreachable and a 500 leaked to the caller. This test pins the
        expected behaviour: ConfigException → Docker fall-back, not raise.
        """
        from kubernetes.config.config_exception import ConfigException

        from src.models.container_models import GitHubRepoRequest

        svc = _build_service(docker_available=True, registry_data={})
        created = _make_info("new-repo-12345678")
        svc.lifecycle_manager.create_container = AsyncMock(return_value=created)

        request = GitHubRepoRequest(repo_url="https://github.com/owner/newrepo", branch="main")

        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            side_effect=ConfigException("no kubeconfig and not in-cluster"),
        ):
            result = await svc.create_container(request)

        assert result.container_name == "new-repo-12345678"
        svc.lifecycle_manager.create_container.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_no_docker_and_no_k8s(self):
        """create_container raises when neither K8s nor Docker is available."""
        from src.models.container_models import GitHubRepoRequest

        svc = _build_service(docker_available=False, registry_data={})

        request = GitHubRepoRequest(repo_url="https://github.com/owner/newrepo", branch="main")

        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
                side_effect=ImportError("no k8s"),
            ),
            pytest.raises(Exception, match="Neither Kubernetes nor Docker"),
        ):
            await svc.create_container(request)

    @pytest.mark.asyncio
    async def test_k8s_provisioning_success(self):
        """create_container uses K8s provisioning when available."""
        from src.models.container_models import GitHubRepoRequest

        svc = _build_service(docker_available=True, registry_data={})
        k8s_container = _make_info("k8s-container")
        mock_k8s = MagicMock()
        mock_k8s.is_available.return_value = True
        mock_k8s.provision_container = AsyncMock(return_value=k8s_container)

        request = GitHubRepoRequest(repo_url="https://github.com/owner/newrepo", branch="main")

        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            return_value=mock_k8s,
        ):
            result = await svc.create_container(request)

        assert result.container_name == "k8s-container"

    @pytest.mark.asyncio
    async def test_tenant_filtering_on_existing(self):
        """create_container respects tenant_id when checking existing containers."""
        from src.models.container_models import GitHubRepoRequest

        c1 = _make_info("c1", repo_url="https://github.com/owner/repo", tenant_id="t-a")
        svc = _build_service(docker_available=True, registry_data={"c1": c1})
        created = _make_info("new-repo-12345678")
        svc.lifecycle_manager.create_container = AsyncMock(return_value=created)

        request = GitHubRepoRequest(repo_url="https://github.com/owner/repo", branch="main")

        # Different tenant - should not find the existing container
        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            side_effect=ImportError("no k8s"),
        ):
            await svc.create_container(request, tenant_id="t-b")
        # Because find_existing_container won't match, it should create a new one
        svc.lifecycle_manager.create_container.assert_awaited_once()


# ===========================================================================
# delete_container (extended)
# ===========================================================================


class TestDeleteContainerExtended:
    @pytest.mark.asyncio
    async def test_tenant_match_deletes(self):
        """delete_container succeeds when tenant matches."""
        c1 = _make_info("c1", tenant_id="tenant-a")
        svc = _build_service(registry_data={"c1": c1})
        svc.lifecycle_manager.remove_container_by_name = AsyncMock(return_value=True)
        result = await svc.delete_container("c1", tenant_id="tenant-a")
        assert result is True
        svc.lifecycle_manager.remove_container_by_name.assert_called_once_with("c1")

    @pytest.mark.asyncio
    async def test_no_tenant_skips_validation(self):
        """delete_container without tenant_id skips tenant validation."""
        c1 = _make_info("c1", tenant_id="tenant-a")
        svc = _build_service(registry_data={"c1": c1})
        svc.lifecycle_manager.remove_container_by_name = AsyncMock(return_value=True)
        result = await svc.delete_container("c1", tenant_id=None)
        assert result is True

    @pytest.mark.asyncio
    async def test_lifecycle_exception_returns_false(self):
        """delete_container returns False when lifecycle_manager throws."""
        c1 = _make_info("c1")
        svc = _build_service(registry_data={"c1": c1})
        svc.lifecycle_manager.remove_container_by_name = AsyncMock(side_effect=RuntimeError("boom"))
        result = await svc.delete_container("c1")
        assert result is False


# ===========================================================================
# File and Directory Operations (delegated to container_monitor)
# ===========================================================================


class TestReadContainerFile:
    @pytest.mark.asyncio
    async def test_delegates_to_monitor(self):
        svc = _build_service()
        svc.container_monitor.read_container_file = AsyncMock(
            return_value={"content": "hello", "size": 5}
        )
        result = await svc.read_container_file("c1", "/app/main.py", offset=0, length=100)
        assert result["content"] == "hello"
        svc.container_monitor.read_container_file.assert_awaited_once_with(
            "c1", "/app/main.py", 0, 100
        )


class TestListContainerDirectory:
    @pytest.mark.asyncio
    async def test_delegates_to_monitor(self):
        svc = _build_service()
        svc.container_monitor.list_container_directory = AsyncMock(
            return_value=[{"name": "file.py", "type": "file"}]
        )
        result = await svc.list_container_directory("c1", path="/app", recursive=True)
        assert len(result) == 1
        svc.container_monitor.list_container_directory.assert_awaited_once_with("c1", "/app", True)


class TestGetContainerRepositoryTree:
    @pytest.mark.asyncio
    async def test_delegates_to_monitor(self):
        svc = _build_service()
        tree = {"name": "repo", "children": []}
        svc.container_monitor.get_container_repository_tree = AsyncMock(return_value=tree)
        result = await svc.get_container_repository_tree("c1")
        assert result["name"] == "repo"
        svc.container_monitor.get_container_repository_tree.assert_awaited_once_with("c1")


class TestSearchContainerFiles:
    @pytest.mark.asyncio
    async def test_delegates_to_monitor(self):
        svc = _build_service()
        svc.container_monitor.search_container_files = AsyncMock(
            return_value=[{"path": "/app/test.py"}]
        )
        result = await svc.search_container_files("c1", "*.py", max_results=50)
        assert len(result) == 1
        svc.container_monitor.search_container_files.assert_awaited_once_with("c1", "*.py", 50)


class TestSearchContainerContent:
    @pytest.mark.asyncio
    async def test_delegates_to_monitor(self):
        svc = _build_service()
        svc.container_monitor.search_container_content = AsyncMock(
            return_value=[{"path": "/app/main.py", "line": 10, "match": "def foo"}]
        )
        result = await svc.search_container_content(
            "c1", "def foo", file_pattern="*.py", case_sensitive=True, max_results=20
        )
        assert len(result) == 1
        svc.container_monitor.search_container_content.assert_awaited_once_with(
            "c1", "def foo", "*.py", True, 20
        )


# ===========================================================================
# create_test_container
# ===========================================================================


class TestCreateTestContainer:
    @pytest.mark.asyncio
    async def test_creates_container(self):
        svc = _build_service(docker_available=True)
        mock_container = MagicMock()
        mock_container.id = "container123"
        client = svc.api_integration.get_docker_client()
        client.containers.create.return_value = mock_container

        config = {"image": "python:3.11-slim", "language": "python"}
        result = await svc.create_test_container(config)
        assert result == "container123"
        client.containers.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_when_docker_unavailable(self):
        svc = _build_service(docker_available=False)
        with pytest.raises(Exception, match="Docker not available"):
            await svc.create_test_container({"image": "python:3.11"})

    @pytest.mark.asyncio
    async def test_propagates_docker_error(self):
        svc = _build_service(docker_available=True)
        client = svc.api_integration.get_docker_client()
        client.containers.create.side_effect = RuntimeError("image not found")
        with pytest.raises(RuntimeError, match="image not found"):
            await svc.create_test_container({"image": "nonexistent:latest"})

    @pytest.mark.asyncio
    async def test_security_config_applied(self):
        svc = _build_service(docker_available=True)
        mock_container = MagicMock()
        mock_container.id = "sec123"
        client = svc.api_integration.get_docker_client()
        client.containers.create.return_value = mock_container

        config = {
            "image": "python:3.11-slim",
            "mem_limit": "256m",
            "cpu_quota": 25000,
            "network_disabled": False,
        }
        await svc.create_test_container(config)
        call_kwargs = client.containers.create.call_args
        assert call_kwargs.kwargs["mem_limit"] == "256m"
        assert call_kwargs.kwargs["cpu_quota"] == 25000
        assert call_kwargs.kwargs["network_disabled"] is False


# ===========================================================================
# exec_in_container
# ===========================================================================


class TestExecInContainer:
    @pytest.mark.asyncio
    async def test_delegates_to_monitor(self):
        svc = _build_service()
        svc.container_monitor.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "OK", "stderr": ""}
        )
        result = await svc.exec_in_container(
            "abc123", ["echo", "hello"], timeout=10, environment={"A": "1"}, user="root"
        )
        assert result["exit_code"] == 0
        svc.container_monitor.exec_in_container.assert_awaited_once_with(
            "abc123", ["echo", "hello"], 10, {"A": "1"}, "root"
        )


# ===========================================================================
# create_file_in_container
# ===========================================================================


class TestCreateFileInContainer:
    @pytest.mark.asyncio
    async def test_delegates_to_monitor(self):
        svc = _build_service()
        svc.container_monitor.create_file_in_container = AsyncMock(return_value=True)
        result = await svc.create_file_in_container("abc", "/tmp/test.py", "print(1)", mode="w")
        assert result is True
        svc.container_monitor.create_file_in_container.assert_awaited_once_with(
            "abc", "/tmp/test.py", "print(1)", "w"
        )


# ===========================================================================
# install_tools_in_container
# ===========================================================================


class TestInstallToolsInContainer:
    @pytest.mark.asyncio
    async def test_delegates_to_monitor(self):
        svc = _build_service()
        svc.container_monitor.install_tools_in_container = AsyncMock(return_value=True)
        result = await svc.install_tools_in_container("abc", ["pytest", "ruff"], language="python")
        assert result is True
        svc.container_monitor.install_tools_in_container.assert_awaited_once_with(
            "abc", ["pytest", "ruff"], "python"
        )


# ===========================================================================
# get_container_stats
# ===========================================================================


class TestGetContainerStats:
    @pytest.mark.asyncio
    async def test_delegates_to_monitor(self):
        svc = _build_service()
        stats = {"cpu_percent": 5.2, "memory_mb": 128}
        svc.container_monitor.get_container_stats = AsyncMock(return_value=stats)
        result = await svc.get_container_stats("abc123")
        assert result["cpu_percent"] == 5.2
        svc.container_monitor.get_container_stats.assert_awaited_once_with("abc123")


# ===========================================================================
# _sync_repo_to_shared_volume
# ===========================================================================


class TestSyncRepoToSharedVolume:
    def test_successful_sync(self):
        svc = _build_service()
        mock_container = MagicMock()
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.output = b""
        mock_container.exec_run.return_value = mock_result

        svc._sync_repo_to_shared_volume(mock_container, "c1", "/app/data/repo")
        assert mock_container.exec_run.call_count == 4  # cleanup, mkdir, cp, verify

    def test_cleanup_failure_raises(self):
        svc = _build_service()
        mock_container = MagicMock()
        fail_result = MagicMock()
        fail_result.exit_code = 1
        fail_result.output = b"permission denied"
        mock_container.exec_run.return_value = fail_result

        with pytest.raises(Exception, match="Failed to cleanup"):
            svc._sync_repo_to_shared_volume(mock_container, "c1", "/app/data/repo")

    def test_copy_failure_raises(self):
        svc = _build_service()
        mock_container = MagicMock()

        ok = MagicMock()
        ok.exit_code = 0
        ok.output = b""

        fail = MagicMock()
        fail.exit_code = 1
        fail.output = b"no space left"

        # cleanup OK, mkdir OK, copy FAIL
        mock_container.exec_run.side_effect = [ok, ok, fail]

        with pytest.raises(Exception, match="Failed to copy"):
            svc._sync_repo_to_shared_volume(mock_container, "c1", "/app/data/repo")

    def test_verify_failure_raises(self):
        svc = _build_service()
        mock_container = MagicMock()

        ok = MagicMock()
        ok.exit_code = 0
        ok.output = b""

        fail = MagicMock()
        fail.exit_code = 1
        fail.output = b"path missing"

        # cleanup OK, mkdir OK, copy OK, verify FAIL
        mock_container.exec_run.side_effect = [ok, ok, ok, fail]

        with pytest.raises(Exception, match="missing after copy"):
            svc._sync_repo_to_shared_volume(mock_container, "c1", "/app/data/repo")


# ===========================================================================
# _install_dependencies_in_container
# ===========================================================================


class TestInstallDependenciesInContainer:
    @pytest.mark.asyncio
    async def test_success(self):
        svc = _build_service()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.installed_packages = ["flask", "pytest"]
        mock_result.package_managers_used = ["pip"]
        mock_result.logs = ["installed flask", "installed pytest"]
        mock_result.errors = []

        with patch("src.core.container_installer.ContainerInstaller") as mock_cls:
            mock_installer = MagicMock()
            mock_installer.install_dependencies_in_container = AsyncMock(return_value=mock_result)
            mock_cls.return_value = mock_installer

            await svc._install_dependencies_in_container("c1", "/app/repo")

    @pytest.mark.asyncio
    async def test_failure_is_non_critical(self):
        """Dependency installation failure does not raise."""
        svc = _build_service()

        with patch(
            "src.core.container_installer.ContainerInstaller",
            side_effect=ImportError("no installer"),
        ):
            # Should not raise
            await svc._install_dependencies_in_container("c1", "/app/repo")

    @pytest.mark.asyncio
    async def test_partial_failure_logs_warnings(self):
        svc = _build_service()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.installed_packages = []
        mock_result.package_managers_used = []
        mock_result.logs = []
        mock_result.errors = ["pip install failed"]

        with patch("src.core.container_installer.ContainerInstaller") as mock_cls:
            mock_installer = MagicMock()
            mock_installer.install_dependencies_in_container = AsyncMock(return_value=mock_result)
            mock_cls.return_value = mock_installer

            # Should not raise despite errors
            await svc._install_dependencies_in_container("c1", "/app/repo")

    @pytest.mark.asyncio
    async def test_no_packages_installed(self):
        """Covers the 'no dependencies to install' log path (line 727)."""
        svc = _build_service()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.installed_packages = []
        mock_result.package_managers_used = []
        mock_result.logs = []
        mock_result.errors = []

        with patch("src.core.container_installer.ContainerInstaller") as mock_cls:
            mock_installer = MagicMock()
            mock_installer.install_dependencies_in_container = AsyncMock(return_value=mock_result)
            mock_cls.return_value = mock_installer

            await svc._install_dependencies_in_container("c1", "/app/repo")


# ===========================================================================
# list_containers error path
# ===========================================================================


class TestListContainersError:
    def test_list_containers_exception_returns_empty(self):
        """list_containers returns empty list on exception."""
        svc = _build_service()
        svc.registry.load.side_effect = RuntimeError("registry broken")
        result = svc.list_containers()
        assert result == []

    def test_list_containers_status_error(self):
        """list_containers sets ERROR status when _get_container_status raises."""
        c1 = _make_info("c1")
        svc = _build_service(registry_data={"c1": c1})
        svc.lifecycle_manager._get_container_status.side_effect = RuntimeError("docker down")
        result = svc.list_containers()
        assert len(result) == 1
        assert result[0].status == ContainerStatus.ERROR


# ===========================================================================
# find_container_by_name status error path
# ===========================================================================


class TestFindContainerByNameStatusError:
    @pytest.mark.asyncio
    async def test_status_check_failure_sets_error(self):
        """find_container_by_name sets ERROR when status check fails."""
        c1 = _make_info("c1")
        svc = _build_service(registry_data={"c1": c1})
        svc.lifecycle_manager._get_container_status.side_effect = RuntimeError("docker down")
        result = await svc.find_container_by_name("c1")
        assert result is not None
        assert result.status == ContainerStatus.ERROR

    @pytest.mark.asyncio
    async def test_find_existing_container_status_error(self):
        """find_existing_container sets ERROR when status check fails."""
        c1 = _make_info("c1", repo_url="https://github.com/owner/repo")
        svc = _build_service(registry_data={"c1": c1})
        svc.lifecycle_manager._get_container_status.side_effect = RuntimeError("docker down")
        result = await svc.find_existing_container("https://github.com/owner/repo")
        assert result is not None
        assert result.status == ContainerStatus.ERROR

    @pytest.mark.asyncio
    async def test_find_existing_container_tenant_filtering(self):
        """find_existing_container skips containers from different tenant."""
        c1 = _make_info("c1", repo_url="https://github.com/owner/repo", tenant_id="t-a")
        svc = _build_service(registry_data={"c1": c1})
        result = await svc.find_existing_container("https://github.com/owner/repo", tenant_id="t-b")
        assert result is None


# ===========================================================================
# get_docker_service factory
# ===========================================================================


class TestGetDockerServiceNew:
    def test_get_docker_service_returns_instance(self):
        with (
            patch("src.services.docker_service.get_settings") as mock_settings,
            patch("src.services.docker_service.ContainerMetadataRegistry"),
            patch("src.services.docker_service.DockerAPIIntegration") as mock_api_cls,
            patch("src.services.docker_service.ContainerLifecycleManager"),
            patch("src.services.docker_service.ImageManager"),
            patch("src.services.docker_service.NetworkVolumeManager"),
            patch("src.services.docker_service.ContainerMonitor"),
        ):
            settings = MagicMock()
            settings.docker_base_port = 8700
            settings.worker.internal_worker_port = 8100
            mock_settings.return_value = settings
            api_instance = MagicMock()
            api_instance.get_docker_client.return_value = MagicMock()
            api_instance.is_docker_available.return_value = False
            mock_api_cls.return_value = api_instance

            from src.services.docker_service import get_docker_service

            svc = get_docker_service()
            assert svc is not None
            assert svc.base_port == 8700


# ===========================================================================
# find_existing_container (extended with tenant)
# ===========================================================================


class TestFindExistingContainerExtended:
    @pytest.mark.asyncio
    async def test_tenant_filter_blocks(self):
        """find_existing_container skips containers that don't match tenant."""
        c1 = _make_info("c1", repo_url="https://github.com/owner/repo", tenant_id="tenant-a")
        svc = _build_service(registry_data={"c1": c1})
        result = await svc.find_existing_container(
            "https://github.com/owner/repo", tenant_id="tenant-b"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_tenant_filter_allows(self):
        """find_existing_container returns container when tenant matches."""
        c1 = _make_info("c1", repo_url="https://github.com/owner/repo", tenant_id="tenant-a")
        svc = _build_service(registry_data={"c1": c1})
        result = await svc.find_existing_container(
            "https://github.com/owner/repo", tenant_id="tenant-a"
        )
        assert result is not None
        assert result.container_name == "c1"

    @pytest.mark.asyncio
    async def test_status_error_on_lifecycle_failure(self):
        """find_existing_container sets ERROR status when status check fails."""
        c1 = _make_info("c1", repo_url="https://github.com/owner/repo")
        svc = _build_service(registry_data={"c1": c1})
        svc.lifecycle_manager._get_container_status.side_effect = RuntimeError("docker died")

        result = await svc.find_existing_container("https://github.com/owner/repo")
        assert result is not None
        assert result.status == ContainerStatus.ERROR


# ===========================================================================
# list_containers (extended)
# ===========================================================================


class TestListContainersExtended:
    def test_status_updates_from_lifecycle(self):
        """list_containers updates status from lifecycle manager."""
        c1 = _make_info("c1", status=ContainerStatus.CREATING)
        svc = _build_service(registry_data={"c1": c1})
        svc.lifecycle_manager._get_container_status.return_value = ContainerStatus.RUNNING
        result = svc.list_containers()
        assert len(result) == 1
        assert result[0].status == ContainerStatus.RUNNING

    def test_status_error_on_lifecycle_failure(self):
        """list_containers sets ERROR when status check raises."""
        c1 = _make_info("c1")
        svc = _build_service(registry_data={"c1": c1})
        svc.lifecycle_manager._get_container_status.side_effect = RuntimeError("fail")
        result = svc.list_containers()
        assert result[0].status == ContainerStatus.ERROR

    def test_exception_returns_empty(self):
        """list_containers returns empty list on registry load failure."""
        svc = _build_service(registry_data={})
        svc.registry.load.side_effect = RuntimeError("corrupt")
        result = svc.list_containers()
        assert result == []


# ===========================================================================
# find_container_by_name (extended)
# ===========================================================================


class TestFindContainerByNameExtended:
    @pytest.mark.asyncio
    async def test_status_error_on_lifecycle_failure(self):
        """find_container_by_name sets ERROR when status check fails."""
        c1 = _make_info("c1")
        svc = _build_service(registry_data={"c1": c1})
        svc.lifecycle_manager._get_container_status.side_effect = RuntimeError("docker gone")

        result = await svc.find_container_by_name("c1")
        assert result is not None
        assert result.status == ContainerStatus.ERROR


# ===========================================================================
# get_docker_service
# ===========================================================================


class TestGetDockerService:
    def test_returns_docker_service_instance(self):
        """get_docker_service returns a DockerService."""
        from src.services.docker_service import get_docker_service

        with (
            patch("src.services.docker_service.get_settings") as mock_settings,
            patch("src.services.docker_service.ContainerMetadataRegistry"),
            patch("src.services.docker_service.DockerAPIIntegration") as mock_api_cls,
            patch("src.services.docker_service.ContainerLifecycleManager"),
            patch("src.services.docker_service.ImageManager"),
            patch("src.services.docker_service.NetworkVolumeManager"),
            patch("src.services.docker_service.ContainerMonitor"),
        ):
            settings = MagicMock()
            settings.docker_base_port = 8700
            settings.worker.internal_worker_port = 8100
            mock_settings.return_value = settings

            api_instance = MagicMock()
            api_instance.get_docker_client.return_value = MagicMock()
            api_instance.is_docker_available.return_value = False
            mock_api_cls.return_value = api_instance

            svc = get_docker_service()
            assert svc is not None
            assert hasattr(svc, "base_port")


# ===========================================================================
# _install_dependencies_in_container (extended)
# ===========================================================================


class TestInstallDependenciesExtended:
    @pytest.mark.asyncio
    async def test_success_with_logs(self):
        """_install_dependencies_in_container logs when packages are installed."""
        svc = _build_service()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.installed_packages = ["flask"]
        mock_result.package_managers_used = ["pip"]
        mock_result.logs = ["installed flask", "installed pytest", "installed ruff", "extra log"]
        mock_result.errors = []

        with patch("src.core.container_installer.ContainerInstaller") as mock_cls:
            mock_installer = MagicMock()
            mock_installer.install_dependencies_in_container = AsyncMock(return_value=mock_result)
            mock_cls.return_value = mock_installer

            await svc._install_dependencies_in_container("c1", "/app/repo")

    @pytest.mark.asyncio
    async def test_success_no_packages(self):
        """_install_dependencies_in_container logs when no packages needed."""
        svc = _build_service()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.installed_packages = []
        mock_result.package_managers_used = []
        mock_result.logs = []
        mock_result.errors = []

        with patch("src.core.container_installer.ContainerInstaller") as mock_cls:
            mock_installer = MagicMock()
            mock_installer.install_dependencies_in_container = AsyncMock(return_value=mock_result)
            mock_cls.return_value = mock_installer

            await svc._install_dependencies_in_container("c1", "/app/repo")

    @pytest.mark.asyncio
    async def test_install_failure_with_errors(self):
        """_install_dependencies_in_container logs errors on failure."""
        svc = _build_service()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.installed_packages = []
        mock_result.package_managers_used = []
        mock_result.logs = []
        mock_result.errors = ["pip install failed", "missing gcc"]

        with patch("src.core.container_installer.ContainerInstaller") as mock_cls:
            mock_installer = MagicMock()
            mock_installer.install_dependencies_in_container = AsyncMock(return_value=mock_result)
            mock_cls.return_value = mock_installer

            await svc._install_dependencies_in_container("c1", "/app/repo")

    @pytest.mark.asyncio
    async def test_install_raises_exception(self):
        """_install_dependencies_in_container handles exceptions gracefully."""
        svc = _build_service()

        with patch(
            "src.core.container_installer.ContainerInstaller", side_effect=RuntimeError("boom")
        ):
            # Should not raise - non-critical
            await svc._install_dependencies_in_container("c1", "/app/repo")


# ===========================================================================
# _sync_repo_to_shared_volume - additional coverage
# ===========================================================================


class TestSyncRepoToSharedVolumeCleanup:
    def test_cleanup_fails(self):
        svc = _build_service(docker_available=True)
        mock_container = MagicMock()
        fail = MagicMock()
        fail.exit_code = 1
        fail.output = b"permission denied"
        mock_container.exec_run.return_value = fail

        with pytest.raises(Exception, match="Failed to cleanup"):
            svc._sync_repo_to_shared_volume(mock_container, "c1", "/app/data/repo")

    def test_mkdir_fails(self):
        svc = _build_service(docker_available=True)
        mock_container = MagicMock()
        ok = MagicMock()
        ok.exit_code = 0
        ok.output = b""
        fail = MagicMock()
        fail.exit_code = 1
        fail.output = b"cannot create"
        mock_container.exec_run.side_effect = [ok, fail]

        with pytest.raises(Exception, match="Failed to create"):
            svc._sync_repo_to_shared_volume(mock_container, "c1", "/app/data/repo")

    def test_copy_fails(self):
        svc = _build_service(docker_available=True)
        mock_container = MagicMock()
        ok = MagicMock()
        ok.exit_code = 0
        ok.output = b""
        fail = MagicMock()
        fail.exit_code = 1
        fail.output = b"copy failed"
        mock_container.exec_run.side_effect = [ok, ok, fail]

        with pytest.raises(Exception, match="Failed to copy"):
            svc._sync_repo_to_shared_volume(mock_container, "c1", "/app/data/repo")

    def test_verify_fails(self):
        svc = _build_service(docker_available=True)
        mock_container = MagicMock()
        ok = MagicMock()
        ok.exit_code = 0
        ok.output = b""
        fail = MagicMock()
        fail.exit_code = 1
        fail.output = b"missing"
        mock_container.exec_run.side_effect = [ok, ok, ok, fail]

        with pytest.raises(Exception, match="missing after copy"):
            svc._sync_repo_to_shared_volume(mock_container, "c1", "/app/data/repo")


# ===========================================================================
# _normalize_repo_url
# ===========================================================================


class TestNormalizeRepoUrlEdgeCases:
    def test_trailing_slash(self):
        svc = _build_service()
        assert svc._normalize_repo_url("https://github.com/a/b/") == "https://github.com/a/b"

    def test_ssh_url(self):
        svc = _build_service()
        result = svc._normalize_repo_url("git@github.com:org/repo.git")
        assert result == "https://github.com/org/repo"

    def test_ssh_url_no_git(self):
        svc = _build_service()
        result = svc._normalize_repo_url("git@github.com:org/repo")
        assert result == "https://github.com/org/repo"

    def test_https_with_git(self):
        svc = _build_service()
        result = svc._normalize_repo_url("https://github.com/org/repo.git")
        assert result == "https://github.com/org/repo"

    def test_bare_owner_repo(self):
        svc = _build_service()
        result = svc._normalize_repo_url("org/repo")
        assert result == "https://github.com/org/repo"


class TestAdmissionCap:
    """Phase C-1 in-flight admission cap in DockerService._enforce_inflight_cap."""

    @staticmethod
    def _settings(**kw):
        from types import SimpleNamespace

        base = dict(
            max_inflight_jobs=0,
            admission_reject_mode="reject",
            admission_wait_timeout_seconds=0,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    @pytest.mark.asyncio
    async def test_cap_disabled_skips_count(self):
        svc = _build_service()
        k8s = MagicMock()
        k8s.count_active_jobs = MagicMock(return_value=999)
        with patch(
            "src.services.docker_service.get_settings",
            return_value=self._settings(max_inflight_jobs=0),
        ):
            await svc._enforce_inflight_cap(k8s)
        k8s.count_active_jobs.assert_not_called()

    @pytest.mark.asyncio
    async def test_under_cap_allows(self):
        svc = _build_service()
        k8s = MagicMock()
        k8s.count_active_jobs = MagicMock(return_value=3)
        with patch(
            "src.services.docker_service.get_settings",
            return_value=self._settings(max_inflight_jobs=5),
        ):
            await svc._enforce_inflight_cap(k8s)  # no raise

    @pytest.mark.asyncio
    async def test_at_cap_rejects_429(self):
        from fastapi import HTTPException

        svc = _build_service()
        k8s = MagicMock()
        k8s.count_active_jobs = MagicMock(return_value=5)
        with patch(
            "src.services.docker_service.get_settings",
            return_value=self._settings(max_inflight_jobs=5),
        ):
            with pytest.raises(HTTPException) as exc:
                await svc._enforce_inflight_cap(k8s)
        assert exc.value.status_code == 429

    @pytest.mark.asyncio
    async def test_count_failure_fails_open(self):
        svc = _build_service()
        k8s = MagicMock()
        k8s.count_active_jobs = MagicMock(side_effect=RuntimeError("api down"))
        with patch(
            "src.services.docker_service.get_settings",
            return_value=self._settings(max_inflight_jobs=1),
        ):
            await svc._enforce_inflight_cap(k8s)  # fails open → no raise

    @pytest.mark.asyncio
    async def test_non_int_cap_from_mock_settings_skips(self):
        """A MagicMock settings (non-int cap) must early-return, not crash."""
        svc = _build_service()
        k8s = MagicMock()
        k8s.count_active_jobs = MagicMock(return_value=10)
        with patch("src.services.docker_service.get_settings", return_value=MagicMock()):
            await svc._enforce_inflight_cap(k8s)
        k8s.count_active_jobs.assert_not_called()

    @pytest.mark.asyncio
    async def test_wait_mode_admits_when_count_drops(self):
        svc = _build_service()
        k8s = MagicMock()
        # First reads over cap, then drops under → admit without raising.
        k8s.count_active_jobs = MagicMock(side_effect=[5, 5, 1])
        with (
            patch(
                "src.services.docker_service.get_settings",
                return_value=self._settings(
                    max_inflight_jobs=5,
                    admission_reject_mode="wait",
                    admission_wait_timeout_seconds=30,
                ),
            ),
            patch("src.services.docker_service.asyncio.sleep", new=AsyncMock()),
        ):
            await svc._enforce_inflight_cap(k8s)  # no raise
        assert k8s.count_active_jobs.call_count >= 2

    @pytest.mark.asyncio
    async def test_create_container_k8s_path_enforces_cap(self):
        """create_container's k8s branch invokes the cap and 429s over it."""
        from fastapi import HTTPException

        from src.models.container_models import GitHubRepoRequest

        svc = _build_service(docker_available=True, registry_data={})
        mock_k8s = MagicMock()
        mock_k8s.is_available.return_value = True
        mock_k8s.reconstruct_container_info.return_value = None  # no existing deployment
        mock_k8s.count_active_jobs = MagicMock(return_value=10)
        mock_k8s.provision_container = AsyncMock()
        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/newrepo",
            branch="main",
            container_name="capped-job",
        )
        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
                return_value=mock_k8s,
            ),
            patch(
                "src.services.docker_service.get_settings",
                return_value=self._settings(max_inflight_jobs=1),
            ),
        ):
            with pytest.raises(HTTPException) as exc:
                await svc.create_container(request)
        assert exc.value.status_code == 429
        mock_k8s.provision_container.assert_not_awaited()


# ===========================================================================
# Eager provision-time code-graph build (trigger_provision_graph_build /
# _async_provision_graph_build)
# ===========================================================================


class TestProvisionGraphBuild:
    """Tests for the eager, non-blocking graph build wired into provisioning."""

    @staticmethod
    def _post_session(resp):
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session.post.return_value.__aenter__ = AsyncMock(return_value=resp)
        session.post.return_value.__aexit__ = AsyncMock(return_value=False)
        return session

    @pytest.mark.asyncio
    async def test_trigger_skips_without_tenant(self):
        """No tenant -> cannot authenticate the build -> no task scheduled."""
        svc = _build_service(registry_data={})
        container = _make_info("c1")

        svc.trigger_provision_graph_build(container, None)

        assert len(svc._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_trigger_schedules_build_with_tenant(self):
        svc = _build_service(registry_data={})
        container = _make_info("c1")

        with patch.object(
            svc, "_async_provision_graph_build", new_callable=AsyncMock
        ) as mock_build:
            svc.trigger_provision_graph_build(container, "t1")
            assert len(svc._background_tasks) == 1
            await asyncio.gather(*list(svc._background_tasks))

        mock_build.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_build_success_marks_initialized(self):
        svc = _build_service(registry_data={})
        container = _make_info("c1")
        svc._await_container_running = AsyncMock(return_value=True)

        resp = MagicMock()
        resp.status = 200
        resp.json = AsyncMock(
            return_value={"result": {"metadata": {"graph_nodes": 5, "graph_edges": 2}}}
        )
        session = self._post_session(resp)

        with (
            patch("aiohttp.ClientSession", return_value=session),
            patch("src.services.docker_service.get_settings") as ms,
        ):
            ms.return_value.main_api_endpoint = "http://gw"
            await svc._async_provision_graph_build(container, "t1")

        assert container.graph_initialized is True
        svc.registry.upsert.assert_called()

    @pytest.mark.asyncio
    async def test_async_build_non_200_does_not_mark(self):
        svc = _build_service(registry_data={})
        container = _make_info("c1")
        svc._await_container_running = AsyncMock(return_value=True)

        resp = MagicMock()
        resp.status = 500
        resp.text = AsyncMock(return_value="boom")
        session = self._post_session(resp)

        with (
            patch("aiohttp.ClientSession", return_value=session),
            patch("src.services.docker_service.get_settings") as ms,
        ):
            ms.return_value.main_api_endpoint = "http://gw"
            await svc._async_provision_graph_build(container, "t1")

        assert container.graph_initialized is False

    @pytest.mark.asyncio
    async def test_analyze_goes_direct_to_analysis_not_through_the_gateway(self):
        """Regression pin for the mesh-denied eager build (2026-08-02).

        The build used to POST to ``{main_api_endpoint}/api/analysis/analyze`` —
        i.e. through coding-gateway, a north-south edge whose Istio
        AuthorizationPolicies do NOT admit the coding-containers namespace. Every
        eager build was reset at the mesh (`[Errno 104] Connection reset by
        peer`), so ``graph_initialized`` was never true for any k8s-provisioned
        container. Verified in-cluster from the real workload identity:
        containers -> gateway = ConnectionReset, containers -> analysis = 200.

        Asserting the actual POST URL, because the pre-existing tests here set
        only ``main_api_endpoint`` on a MagicMock and would pass against EITHER
        destination.
        """
        svc = _build_service(registry_data={})
        container = _make_info("c1")
        svc._await_container_running = AsyncMock(return_value=True)

        resp = MagicMock()
        resp.status = 200
        resp.json = AsyncMock(
            return_value={"result": {"metadata": {"graph_nodes": 5, "graph_edges": 2}}}
        )
        session = self._post_session(resp)

        with (
            patch("aiohttp.ClientSession", return_value=session),
            patch("src.services.docker_service.get_settings") as ms,
        ):
            ms.return_value.main_api_endpoint = "http://coding-gateway.coding-gateway:8687"
            ms.return_value.coding_analysis_url = "http://coding-analysis.coding-analysis:8688"
            await svc._async_provision_graph_build(container, "t1")

        posted_url = session.post.call_args[0][0]
        assert posted_url == "http://coding-analysis.coding-analysis:8688/api/analysis/analyze"
        # The gateway must not appear anywhere in the target.
        assert "coding-gateway" not in posted_url

    @pytest.mark.asyncio
    async def test_build_skipped_when_analysis_url_unset(self):
        """Unset analysis URL must skip cleanly, not POST to a bare path."""
        svc = _build_service(registry_data={})
        container = _make_info("c1")
        svc._await_container_running = AsyncMock(return_value=True)

        session = self._post_session(MagicMock())

        with (
            patch("aiohttp.ClientSession", return_value=session),
            patch("src.services.docker_service.get_settings") as ms,
        ):
            ms.return_value.main_api_endpoint = "http://gw"
            ms.return_value.coding_analysis_url = ""
            await svc._async_provision_graph_build(container, "t1")

        session.post.assert_not_called()
        assert container.graph_initialized is False

    def test_analysis_url_default_points_at_coding_analysis_in_cluster(self):
        """The default must work with no gitops change — that is what makes the
        fix take effect on deploy."""
        from src.config.settings import Settings

        assert (
            Settings().coding_analysis_url
            == "http://coding-analysis.coding-analysis.svc.cluster.local:8688"
        )

    @pytest.mark.asyncio
    async def test_async_build_skips_when_no_endpoint(self):
        svc = _build_service(registry_data={})
        container = _make_info("c1")

        with (
            patch("aiohttp.ClientSession") as client_session,
            patch("src.services.docker_service.get_settings") as ms,
        ):
            ms.return_value.main_api_endpoint = ""
            await svc._async_provision_graph_build(container, "t1")
            client_session.assert_not_called()

        assert container.graph_initialized is False

    @pytest.mark.asyncio
    async def test_async_build_swallows_client_error(self):
        svc = _build_service(registry_data={})
        container = _make_info("c1")

        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session.post.return_value.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientError("upstream down")
        )
        session.post.return_value.__aexit__ = AsyncMock(return_value=False)
        svc._await_container_running = AsyncMock(return_value=True)

        with (
            patch("aiohttp.ClientSession", return_value=session),
            patch("src.services.docker_service.get_settings") as ms,
        ):
            ms.return_value.main_api_endpoint = "http://gw"
            # Must not raise.
            await svc._async_provision_graph_build(container, "t1")

        assert container.graph_initialized is False

    @pytest.mark.asyncio
    async def test_async_build_skips_when_not_running(self):
        """Workspace-readiness gate (A1): if the container never reaches Running
        the eager build is skipped — no analyze POST, graph not marked — so a
        mid-clone workspace can't be scanned into a degraded components=1 graph."""
        svc = _build_service(registry_data={})
        container = _make_info("c1")
        svc._await_container_running = AsyncMock(return_value=False)

        with (
            patch("aiohttp.ClientSession") as client_session,
            patch("src.services.docker_service.get_settings") as ms,
        ):
            ms.return_value.main_api_endpoint = "http://gw"
            await svc._async_provision_graph_build(container, "t1")
            client_session.assert_not_called()

        assert container.graph_initialized is False

    @pytest.mark.asyncio
    async def test_await_container_running_true_when_running(self):
        svc = _build_service(registry_data={})
        svc._resolve_live_status = AsyncMock(return_value=ContainerStatus.RUNNING)
        assert await svc._await_container_running("c1") is True
        svc._resolve_live_status.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_await_container_running_false_on_timeout(self):
        svc = _build_service(registry_data={})
        svc._resolve_live_status = AsyncMock(return_value=ContainerStatus.CREATING)
        # Zero budget → the deadline has passed after the first (non-Running)
        # probe, so it returns False without sleeping.
        with patch("src.services.docker_service.EAGER_BUILD_READINESS_TIMEOUT_S", 0.0):
            assert await svc._await_container_running("c1") is False

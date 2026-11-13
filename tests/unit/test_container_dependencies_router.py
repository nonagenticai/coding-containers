"""Tests for src/routers/container_dependencies.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.container_models import ContainerInfo, ContainerStatus

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_container(name: str = "test-ctr") -> ContainerInfo:
    return ContainerInfo(
        container_name=name,
        api_endpoint="http://localhost:8687",
        internal_endpoint="http://test-ctr:8687",
        public_endpoint="http://localhost:8687",
        codebase_path="/repo",
        repo_url="https://github.com/owner/repo",
        branch="main",
        status=ContainerStatus.RUNNING,
    )


def _make_install_result(
    success: bool = True,
    installed: list[str] | None = None,
    pkg_managers: list[str] | None = None,
    errors: list[str] | None = None,
    install_time: float = 1.5,
) -> MagicMock:
    """Build a mock InstallationResult."""
    result = MagicMock()
    result.success = success
    result.installed_packages = installed or ["pkg1", "pkg2"]
    result.package_managers_used = pkg_managers or ["pip"]
    result.errors = errors or []
    result.installation_time = install_time
    return result


# ---------------------------------------------------------------------------
# POST /{container_name}/install-dependencies
# ---------------------------------------------------------------------------


class TestInstallContainerDependencies:
    @patch("src.routers.container_dependencies.ContainerInstaller")
    @patch("src.routers.container_dependencies.get_docker_service")
    async def test_successful_install(self, mock_get_docker, mock_installer_cls, client):
        # Docker finds the container
        docker_svc = AsyncMock()
        docker_svc.find_container_by_name = AsyncMock(return_value=_make_container())
        mock_get_docker.return_value = docker_svc

        # Installer returns success
        installer = AsyncMock()
        installer.install_dependencies_in_container = AsyncMock(return_value=_make_install_result())
        mock_installer_cls.return_value = installer

        resp = await client.post(
            "/api/containers/test-ctr/install-dependencies",
            json={"workspace_path": "/app/data/repo", "force_reinstall": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["status"] == "success"
        assert data["installed_packages"] == 2

    @patch("src.routers.container_dependencies.ContainerInstaller")
    @patch("src.routers.container_dependencies.get_docker_service")
    async def test_install_with_errors(self, mock_get_docker, mock_installer_cls, client):
        docker_svc = AsyncMock()
        docker_svc.find_container_by_name = AsyncMock(return_value=_make_container())
        mock_get_docker.return_value = docker_svc

        installer = AsyncMock()
        installer.install_dependencies_in_container = AsyncMock(
            return_value=_make_install_result(
                success=False,
                installed=[],
                errors=["Failed to install numpy", "Guidance: Install gcc first"],
            )
        )
        mock_installer_cls.return_value = installer

        resp = await client.post(
            "/api/containers/test-ctr/install-dependencies",
            json={},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert data["status"] == "error"
        assert "Install gcc first" in data["guidance"]

    @patch("src.routers.container_dependencies.get_docker_service")
    async def test_container_not_found_returns_404(self, mock_get_docker, client):
        docker_svc = AsyncMock()
        docker_svc.find_container_by_name = AsyncMock(return_value=None)
        mock_get_docker.return_value = docker_svc

        resp = await client.post(
            "/api/containers/missing/install-dependencies",
            json={},
        )
        assert resp.status_code == 404

    @patch("src.routers.container_dependencies.ContainerInstaller")
    @patch("src.routers.container_dependencies.get_docker_service")
    async def test_system_dependency_error(self, mock_get_docker, mock_installer_cls, client):
        from src.core.container_installer import SystemDependencyError

        docker_svc = AsyncMock()
        docker_svc.find_container_by_name = AsyncMock(return_value=_make_container())
        mock_get_docker.return_value = docker_svc

        installer = AsyncMock()
        err = SystemDependencyError(
            "Missing libxml2",
            missing_packages=["libxml2-dev"],
            guidance="apt-get install libxml2-dev",
        )
        installer.install_dependencies_in_container = AsyncMock(side_effect=err)
        mock_installer_cls.return_value = installer

        resp = await client.post(
            "/api/containers/test-ctr/install-dependencies",
            json={},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "libxml2" in data["guidance"]

    @patch("src.routers.container_dependencies.ContainerInstaller")
    @patch("src.routers.container_dependencies.get_docker_service")
    async def test_generic_exception_returns_500(self, mock_get_docker, mock_installer_cls, client):
        docker_svc = AsyncMock()
        docker_svc.find_container_by_name = AsyncMock(return_value=_make_container())
        mock_get_docker.return_value = docker_svc

        installer = AsyncMock()
        installer.install_dependencies_in_container = AsyncMock(
            side_effect=RuntimeError("unexpected error")
        )
        mock_installer_cls.return_value = installer

        resp = await client.post(
            "/api/containers/test-ctr/install-dependencies",
            json={},
        )
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# GET /{container_name}/dependency-status
# ---------------------------------------------------------------------------


class TestGetDependencyStatus:
    @patch("src.routers.container_dependencies.ContainerInstaller")
    @patch("src.routers.container_dependencies.get_docker_service")
    async def test_successful_status(self, mock_get_docker, mock_installer_cls, client):
        docker_svc = AsyncMock()
        docker_svc.find_container_by_name = AsyncMock(return_value=_make_container())
        mock_get_docker.return_value = docker_svc

        installer = AsyncMock()
        installer.analyze_project = AsyncMock(
            return_value={
                "project_type": "python",
                "detected_files": [
                    {
                        "path": "/workspace/requirements.txt",
                        "type": "python",
                        "manager": "pip",
                        "file_name": "requirements.txt",
                    }
                ],
                "installation_plan": ["Install python dependencies using pip"],
                "total_files": 1,
            }
        )
        mock_installer_cls.return_value = installer

        resp = await client.get("/api/containers/test-ctr/dependency-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["project_type"] == "python"
        assert data["total_dependency_files"] == 1

    @patch("src.routers.container_dependencies.ContainerInstaller")
    @patch("src.routers.container_dependencies.get_docker_service")
    async def test_the_real_analysis_document_validates(
        self, mock_get_docker, mock_installer_cls, client
    ):
        """D3: the route must accept what ``analyze_project`` PRODUCES. The fake
        above used to hand-write a dict-shaped plan; the producer returns a list,
        and the live workspace answered 500 on response validation."""
        from src.core.container_installer import ContainerInstaller

        real = ContainerInstaller("test-ctr")
        doc = real._analysis_document(
            [
                real._classify_dependency_file("/workspace/pyproject.toml"),
                real._classify_dependency_file("/workspace/package.json"),
            ],
            {"python", "javascript"},
        )
        docker_svc = AsyncMock()
        docker_svc.find_container_by_name = AsyncMock(return_value=_make_container())
        mock_get_docker.return_value = docker_svc
        installer = AsyncMock()
        installer.analyze_project = AsyncMock(return_value=doc)
        mock_installer_cls.return_value = installer

        resp = await client.get("/api/containers/test-ctr/dependency-status")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["project_type"] == "mixed"
        assert data["detected_files"][0]["file_name"] == "pyproject.toml"
        assert len(data["installation_plan"]) == 2
        assert data["total_dependency_files"] == 2

    @patch("src.routers.container_dependencies.get_docker_service")
    async def test_container_not_found_returns_404(self, mock_get_docker, client):
        docker_svc = AsyncMock()
        docker_svc.find_container_by_name = AsyncMock(return_value=None)
        mock_get_docker.return_value = docker_svc

        resp = await client.get("/api/containers/missing/dependency-status")
        assert resp.status_code == 404

    @patch("src.routers.container_dependencies.ContainerInstaller")
    @patch("src.routers.container_dependencies.get_docker_service")
    async def test_generic_exception_returns_500(self, mock_get_docker, mock_installer_cls, client):
        docker_svc = AsyncMock()
        docker_svc.find_container_by_name = AsyncMock(return_value=_make_container())
        mock_get_docker.return_value = docker_svc

        installer = AsyncMock()
        installer.analyze_project = AsyncMock(side_effect=RuntimeError("oops"))
        mock_installer_cls.return_value = installer

        resp = await client.get("/api/containers/test-ctr/dependency-status")
        assert resp.status_code == 500

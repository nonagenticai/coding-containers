"""Tests for health and root endpoints, plus container health router."""

from unittest.mock import MagicMock, patch

import pytest

from src.models.container_models import ContainerStatus

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Basic service health / root endpoints (existing)
# ---------------------------------------------------------------------------


async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "coding-containers"
    assert "version" in data


async def test_root_returns_service_info(client):
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "coding-containers"
    assert "version" in data
    assert "docs" in data
    assert "health" in data


async def test_openapi_json_available(client):
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "paths" in data
    assert "/health" in data["paths"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_container(
    name: str = "test-ctr",
    status: ContainerStatus = ContainerStatus.RUNNING,
    language: str | None = "python",
    codebase_path: str | None = "/repo",
) -> MagicMock:
    """Build a mock container with attributes matching what the router reads."""
    ctr = MagicMock()
    ctr.container_name = name
    ctr.status = status
    ctr.language = language
    ctr.codebase_path = codebase_path or "/repo"
    ctr.created_at = "2025-01-01T00:00:00"
    return ctr


# ---------------------------------------------------------------------------
# GET /api/health/container/{name} — get_container_health
# ---------------------------------------------------------------------------


class TestGetContainerHealth:
    @patch("src.routers.health.get_container_manager")
    async def test_running_container_healthy(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_container.return_value = _make_container(status=ContainerStatus.RUNNING)
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/container/test-ctr")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-ctr"
        assert data["healthy"] is True

    @patch("src.routers.health.get_container_manager")
    async def test_error_container_unhealthy(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_container.return_value = _make_container(status=ContainerStatus.ERROR)
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/container/test-ctr")
        assert resp.status_code == 200
        data = resp.json()
        assert data["healthy"] is False

    @patch("src.routers.health.get_container_manager")
    async def test_not_found_returns_404(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_container.return_value = None
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/container/missing")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "HTTP_404"
        assert "missing" in data["error"]["message"]

    @patch("src.routers.health.get_container_manager")
    async def test_value_error_returns_404(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_container.side_effect = ValueError("not found")
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/container/bad-name")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "HTTP_404"
        assert "bad-name" in data["error"]["message"]


# ---------------------------------------------------------------------------
# GET /api/health/containers/unhealthy — list_unhealthy_containers
# ---------------------------------------------------------------------------


class TestListUnhealthyContainers:
    @patch("src.routers.health.get_container_manager")
    async def test_returns_only_error_containers(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.list_containers.return_value = [
            _make_container(name="ok", status=ContainerStatus.RUNNING),
            _make_container(name="bad", status=ContainerStatus.ERROR),
        ]
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/containers/unhealthy")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["unhealthy"][0]["name"] == "bad"

    @patch("src.routers.health.get_container_manager")
    async def test_no_unhealthy(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.list_containers.return_value = [
            _make_container(name="ok", status=ContainerStatus.RUNNING),
        ]
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/containers/unhealthy")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["unhealthy"] == []


# ---------------------------------------------------------------------------
# GET /api/health/container/{name}/ready — container_readiness
# ---------------------------------------------------------------------------


class TestContainerReadiness:
    @patch("src.routers.health.get_container_manager")
    async def test_running_is_ready(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_container.return_value = _make_container(status=ContainerStatus.RUNNING)
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/container/test-ctr/ready")
        assert resp.status_code == 200
        assert resp.json()["healthy"] is True

    @patch("src.routers.health.get_container_manager")
    async def test_stopped_returns_503(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_container.return_value = _make_container(status=ContainerStatus.STOPPED)
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/container/test-ctr/ready")
        assert resp.status_code == 503

    @patch("src.routers.health.get_container_manager")
    async def test_not_found_returns_404(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_container.return_value = None
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/container/missing/ready")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "HTTP_404"
        assert "missing" in data["error"]["message"]

    @patch("src.routers.health.get_container_manager")
    async def test_value_error_returns_404(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_container.side_effect = ValueError("nope")
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/container/bad/ready")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "HTTP_404"
        assert "bad" in data["error"]["message"]


# ---------------------------------------------------------------------------
# GET /api/health/container/{name}/live — container_liveness
# ---------------------------------------------------------------------------


class TestContainerLiveness:
    @patch("src.routers.health.get_container_manager")
    async def test_running_is_alive(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_container.return_value = _make_container(status=ContainerStatus.RUNNING)
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/container/test-ctr/live")
        assert resp.status_code == 200
        assert resp.json()["healthy"] is True

    @patch("src.routers.health.get_container_manager")
    async def test_creating_is_alive(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_container.return_value = _make_container(status=ContainerStatus.CREATING)
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/container/test-ctr/live")
        assert resp.status_code == 200

    @patch("src.routers.health.get_container_manager")
    async def test_stopped_returns_503(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_container.return_value = _make_container(status=ContainerStatus.STOPPED)
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/container/test-ctr/live")
        assert resp.status_code == 503

    @patch("src.routers.health.get_container_manager")
    async def test_error_returns_503(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_container.return_value = _make_container(status=ContainerStatus.ERROR)
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/container/test-ctr/live")
        assert resp.status_code == 503

    @patch("src.routers.health.get_container_manager")
    async def test_not_found_returns_404(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_container.return_value = None
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/container/missing/live")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "HTTP_404"
        assert "missing" in data["error"]["message"]


# ---------------------------------------------------------------------------
# GET /api/health/container/{name}/statistics — container_statistics
# ---------------------------------------------------------------------------


class TestContainerStatistics:
    @patch("src.routers.health.get_container_manager")
    async def test_returns_statistics(self, mock_get_mgr, client):
        mgr = MagicMock()
        ctr = _make_container(status=ContainerStatus.RUNNING, language="python")
        mgr.get_container.return_value = ctr
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/container/test-ctr/statistics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-ctr"
        assert data["status"] == "running"
        assert data["language"] == "python"

    @patch("src.routers.health.get_container_manager")
    async def test_not_found_returns_404(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_container.return_value = None
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/container/missing/statistics")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "HTTP_404"
        assert "missing" in data["error"]["message"]

    @patch("src.routers.health.get_container_manager")
    async def test_value_error_returns_404(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_container.side_effect = ValueError("nope")
        mock_get_mgr.return_value = mgr

        resp = await client.get("/api/health/container/bad/statistics")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "HTTP_404"
        assert "bad" in data["error"]["message"]

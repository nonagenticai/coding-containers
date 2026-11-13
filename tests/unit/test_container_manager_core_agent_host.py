"""Tests for agent-host delegation methods in core/container_manager_core.py.

Covers:
- _create_container_via_agent_host
- _remove_container_via_agent_host
- exec_in_container_via_agent_host
- create_container with Docker (full path)
- _resolve_shared_volume_name (Docker-available branches)
- _find_available_port / _get_used_docker_ports with Docker
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from src.models.container_models import ContainerSpec, ContainerStatus


def _make_spec(**overrides) -> ContainerSpec:
    defaults = dict(name="test-container", image="codegraph-api:latest", port=8700)
    defaults.update(overrides)
    return ContainerSpec(**defaults)


# ---------------------------------------------------------------------------
# _create_container_via_agent_host
# ---------------------------------------------------------------------------


class TestCreateContainerViaAgentHost:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", "http://agent:8080")
    @respx.mock
    def test_success(self):
        respx.post("http://agent:8080/containers").mock(
            return_value=httpx.Response(
                201,
                json={
                    "container_id": "cid-123",
                    "container_name": "test-container",
                },
            )
        )

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        spec = _make_spec()
        info = mgr.create_container(spec)
        assert info.status == ContainerStatus.RUNNING
        assert "Managed by agent-host" in info.description
        assert mgr.has_agent_host_id("test-container")

    @patch("src.core.container_manager_core.AGENT_HOST_URL", "http://agent:8080")
    @respx.mock
    def test_non_201_raises(self):
        respx.post("http://agent:8080/containers").mock(
            return_value=httpx.Response(500, text="internal error")
        )

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        spec = _make_spec()
        with pytest.raises(RuntimeError, match="agent-host create failed"):
            mgr.create_container(spec)

    @patch("src.core.container_manager_core.AGENT_HOST_URL", "http://agent:8080")
    @respx.mock
    def test_network_error_raises(self):
        respx.post("http://agent:8080/containers").mock(
            side_effect=httpx.ConnectError("connection refused")
        )

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        spec = _make_spec()
        with pytest.raises(httpx.ConnectError):
            mgr.create_container(spec)


# ---------------------------------------------------------------------------
# _remove_container_via_agent_host
# ---------------------------------------------------------------------------


class TestRemoveContainerViaAgentHost:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", "http://agent:8080")
    @respx.mock
    def test_remove_success(self):
        respx.delete("http://agent:8080/containers/cid-123").mock(return_value=httpx.Response(200))

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        mgr._agent_host_ids["my-c"] = "cid-123"
        result = mgr.remove_container("my-c")
        assert result is True
        assert "my-c" not in mgr._agent_host_ids

    @patch("src.core.container_manager_core.AGENT_HOST_URL", "http://agent:8080")
    @respx.mock
    def test_remove_404_still_true(self):
        respx.delete("http://agent:8080/containers/cid-123").mock(return_value=httpx.Response(404))

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        mgr._agent_host_ids["my-c"] = "cid-123"
        assert mgr.remove_container("my-c") is True

    @patch("src.core.container_manager_core.AGENT_HOST_URL", "http://agent:8080")
    @respx.mock
    def test_remove_500_returns_false(self):
        respx.delete("http://agent:8080/containers/cid-123").mock(
            return_value=httpx.Response(500, text="fail")
        )

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        mgr._agent_host_ids["my-c"] = "cid-123"
        assert mgr.remove_container("my-c") is False

    @patch("src.core.container_manager_core.AGENT_HOST_URL", "http://agent:8080")
    def test_remove_no_id_returns_false(self):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.remove_container("unknown") is False

    @patch("src.core.container_manager_core.AGENT_HOST_URL", "http://agent:8080")
    @respx.mock
    def test_remove_network_error_returns_false(self):
        respx.delete("http://agent:8080/containers/cid-123").mock(
            side_effect=httpx.ConnectError("refused")
        )

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        mgr._agent_host_ids["my-c"] = "cid-123"
        assert mgr.remove_container("my-c") is False


# ---------------------------------------------------------------------------
# exec_in_container_via_agent_host
# ---------------------------------------------------------------------------


class TestExecInContainerViaAgentHost:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", "http://agent:8080")
    @respx.mock
    def test_exec_success(self):
        respx.post("http://agent:8080/containers/cid-123/exec").mock(
            return_value=httpx.Response(
                200,
                json={
                    "exit_code": 0,
                    "stdout": "hello",
                    "stderr": "",
                    "execution_time": 0.5,
                },
            )
        )

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        mgr._agent_host_ids["my-c"] = "cid-123"
        result = mgr.exec_in_container_via_agent_host("my-c", ["echo", "hello"])
        assert result["exit_code"] == 0
        assert result["stdout"] == "hello"
        assert result["duration"] == 0.5

    @patch("src.core.container_manager_core.AGENT_HOST_URL", "http://agent:8080")
    def test_exec_no_container_id_raises(self):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        with pytest.raises(RuntimeError, match="No agent-host container ID"):
            mgr.exec_in_container_via_agent_host("unknown", ["ls"])

    @patch("src.core.container_manager_core.AGENT_HOST_URL", "http://agent:8080")
    @respx.mock
    def test_exec_non_200_raises(self):
        respx.post("http://agent:8080/containers/cid-123/exec").mock(
            return_value=httpx.Response(500, text="exec failed")
        )

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        mgr._agent_host_ids["my-c"] = "cid-123"
        with pytest.raises(RuntimeError, match="agent-host exec failed"):
            mgr.exec_in_container_via_agent_host("my-c", ["ls"])

    @patch("src.core.container_manager_core.AGENT_HOST_URL", "http://agent:8080")
    @respx.mock
    def test_exec_missing_exit_code_raises(self):
        respx.post("http://agent:8080/containers/cid-123/exec").mock(
            return_value=httpx.Response(200, json={"stdout": "output"})
        )

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        mgr._agent_host_ids["my-c"] = "cid-123"
        with pytest.raises(RuntimeError, match="missing 'exit_code'"):
            mgr.exec_in_container_via_agent_host("my-c", ["ls"])


# ---------------------------------------------------------------------------
# create_container with Docker (full path through the method)
# ---------------------------------------------------------------------------


class TestCreateContainerDocker:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_create_success(self, mock_docker):
        """Test the full create_container path with Docker available."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.volumes.get.side_effect = Exception("not found")

        # Mock volume creation
        not_found = type("NotFound", (Exception,), {})
        mock_docker.errors.NotFound = not_found
        mock_docker.errors.DockerException = Exception
        mock_docker.errors.APIError = type("APIError", (Exception,), {})
        mock_client.volumes.get.side_effect = not_found("no vol")
        mock_client.volumes.create.return_value = MagicMock()

        # Mock container run
        mock_client.containers.run.return_value = MagicMock()

        # Mock docker.types.Mount
        mock_docker.types.Mount = MagicMock()

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        spec = _make_spec()
        info = mgr.create_container(spec)
        assert info.status == ContainerStatus.RUNNING
        assert info.container_name == "test-container"

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_create_with_spec_volumes(self, mock_docker):
        """Test create_container mounts spec.volumes."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        not_found = type("NotFound", (Exception,), {})
        mock_docker.errors.NotFound = not_found
        mock_docker.errors.DockerException = Exception
        mock_docker.errors.APIError = type("APIError", (Exception,), {})
        mock_client.volumes.get.side_effect = not_found("no vol")
        mock_client.volumes.create.return_value = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_docker.types.Mount = MagicMock()

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        spec = _make_spec(volumes={"/tmp": {"bind": "/app/tmp", "mode": "rw"}})
        info = mgr.create_container(spec)
        assert info.status == ContainerStatus.RUNNING

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_create_docker_run_fails_raises(self, mock_docker):
        """Test create_container when docker run raises."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        not_found = type("NotFound", (Exception,), {})
        mock_docker.errors.NotFound = not_found
        mock_docker.errors.DockerException = Exception
        mock_docker.errors.APIError = type("APIError", (Exception,), {})
        mock_client.volumes.get.side_effect = not_found("no vol")
        mock_client.volumes.create.return_value = MagicMock()
        mock_client.containers.run.side_effect = Exception("run failed")
        mock_docker.types.Mount = MagicMock()

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        spec = _make_spec()
        with pytest.raises(Exception, match="run failed"):
            mgr.create_container(spec)


# ---------------------------------------------------------------------------
# _resolve_shared_volume_name with Docker available
# ---------------------------------------------------------------------------


class TestResolveSharedVolumeNameDocker:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_uses_existing_volume(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.volumes.get.return_value = MagicMock()

        not_found = type("NotFound", (Exception,), {})
        mock_docker.errors.NotFound = not_found
        mock_docker.errors.DockerException = Exception
        mock_docker.errors.APIError = type("APIError", (Exception,), {})

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        name = mgr._resolve_shared_volume_name()
        assert isinstance(name, str)
        assert len(name) > 0

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_creates_volume_when_not_found(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        not_found = type("NotFound", (Exception,), {})
        api_error = type("APIError", (Exception,), {})
        mock_docker.errors.NotFound = not_found
        mock_docker.errors.DockerException = Exception
        mock_docker.errors.APIError = api_error

        # All get calls raise NotFound
        mock_client.volumes.get.side_effect = not_found("nope")
        # First create succeeds
        mock_client.volumes.create.return_value = MagicMock()

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        name = mgr._resolve_shared_volume_name()
        assert name is not None
        mock_client.volumes.create.assert_called()

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_cached_volume_name(self, mock_docker):
        """Once resolved, name is cached."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.volumes.get.return_value = MagicMock()

        not_found = type("NotFound", (Exception,), {})
        mock_docker.errors.NotFound = not_found
        mock_docker.errors.DockerException = Exception
        mock_docker.errors.APIError = type("APIError", (Exception,), {})

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        name1 = mgr._resolve_shared_volume_name()
        name2 = mgr._resolve_shared_volume_name()
        assert name1 == name2

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_all_create_fail_uses_fallback(self, mock_docker):
        """When all volume create attempts fail, falls back to last candidate."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        not_found = type("NotFound", (Exception,), {})
        api_error = type("APIError", (Exception,), {})
        mock_docker.errors.NotFound = not_found
        mock_docker.errors.DockerException = Exception
        mock_docker.errors.APIError = api_error

        mock_client.volumes.get.side_effect = not_found("nope")
        mock_client.volumes.create.side_effect = api_error("cannot create")

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        name = mgr._resolve_shared_volume_name()
        # Should fallback to last candidate
        assert name == mgr._shared_volume_candidates[-1]


# ---------------------------------------------------------------------------
# _find_available_port / _get_used_docker_ports with Docker
# ---------------------------------------------------------------------------


class TestFindAvailablePortDocker:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_find_port_with_docker(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True
        mock_docker.errors.DockerException = Exception

        # No running containers
        mock_client.containers.list.return_value = []

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        port = mgr._find_available_port()
        assert port >= mgr.base_port

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_get_used_ports_with_containers(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True
        mock_docker.errors.DockerException = Exception

        container = MagicMock()
        container.ports = {"8700/tcp": [{"HostPort": "8700"}]}
        container.name = "test-c"
        mock_client.containers.list.return_value = [container]

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        used = mgr._get_used_docker_ports()
        assert 8700 in used

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_get_used_ports_docker_exception(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        docker_exc = type("DockerException", (Exception,), {})
        mock_docker.errors.DockerException = docker_exc
        mock_client.containers.list.side_effect = docker_exc("list failed")

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        used = mgr._get_used_docker_ports()
        assert used == set()


# ---------------------------------------------------------------------------
# remove_container with Docker - volume removal and shared volume cleanup
# ---------------------------------------------------------------------------


class TestRemoveContainerDockerExtended:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_remove_with_volume_success(self, mock_docker):
        """Test removing container where volume exists and is removed."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        not_found = type("NotFound", (Exception,), {})
        mock_docker.errors.NotFound = not_found
        mock_docker.errors.DockerException = type("DockerException", (Exception,), {})
        mock_docker.errors.APIError = type("APIError", (Exception,), {})

        container = MagicMock()
        mock_volume = MagicMock()
        mock_client.containers.get.return_value = container
        mock_client.volumes.get.return_value = mock_volume
        mock_client.containers.run.return_value = None

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        result = mgr.remove_container("test-c")
        assert result is True
        mock_volume.remove.assert_called_once()

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_remove_shared_volume_cleanup_fails(self, mock_docker):
        """Shared volume cleanup failure is non-fatal."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        not_found = type("NotFound", (Exception,), {})
        mock_docker.errors.NotFound = not_found
        mock_docker.errors.DockerException = type("DockerException", (Exception,), {})
        mock_docker.errors.APIError = type("APIError", (Exception,), {})

        container = MagicMock()
        mock_client.containers.get.return_value = container
        mock_client.volumes.get.side_effect = not_found("no vol")
        mock_client.containers.run.side_effect = RuntimeError("cleanup failed")

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        result = mgr.remove_container("test-c")
        assert result is True  # cleanup failure is non-fatal

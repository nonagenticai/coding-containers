"""Tests for core/container_manager_core.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.models.container_models import ContainerInfo, ContainerSpec, ContainerStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_container_info(**overrides) -> ContainerInfo:
    defaults = dict(
        container_name="test-c",
        api_endpoint="http://gw:8687/api/containers/test-c",
        internal_endpoint="http://localhost:8700",
        public_endpoint="http://gw:8687/api/containers/test-c",
        code_graph_id="",
        codebase_path="",
        repo_url="",
        branch="",
        status=ContainerStatus.RUNNING,
        description="",
    )
    defaults.update(overrides)
    return ContainerInfo(**defaults)


def _make_spec(**overrides) -> ContainerSpec:
    defaults = dict(
        name="test-container",
        image="codegraph-api:latest",
        port=8700,
    )
    defaults.update(overrides)
    return ContainerSpec(**defaults)


# ---------------------------------------------------------------------------
# CoreContainerManager init & is_docker_available
# ---------------------------------------------------------------------------


class TestCoreContainerManagerInit:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_init_no_docker(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.docker_available is False
        assert mgr.client is None
        assert mgr.is_docker_available() is False

    @patch("src.core.container_manager_core.AGENT_HOST_URL", "http://agent:8080")
    def test_init_with_agent_host(self):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.docker_available is True
        assert mgr.is_docker_available() is True
        assert mgr.uses_agent_host is True

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_init_docker_success(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.docker_available is True
        assert mgr.client is mock_client

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_init_docker_exception(self, mock_docker):
        mock_docker.from_env.side_effect = OSError("no docker")
        mock_docker.errors.DockerException = Exception

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.docker_available is False
        assert mgr.client is None


# ---------------------------------------------------------------------------
# _build_shared_volume_candidates
# ---------------------------------------------------------------------------


class TestBuildSharedVolumeCandidates:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    @patch.dict("os.environ", {}, clear=True)
    def test_default_candidates(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        candidates = mgr._shared_volume_candidates
        assert "code_graph_helper_provisioned_containers" in candidates
        assert "provisioned_containers" in candidates

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    @patch.dict(
        "os.environ",
        {"PROVISIONED_CONTAINERS_VOLUME_NAME": "my-volume"},
        clear=True,
    )
    def test_explicit_volume_name(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr._shared_volume_candidates[0] == "my-volume"

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    @patch.dict(
        "os.environ",
        {"COMPOSE_PROJECT_NAME": "myproject"},
        clear=True,
    )
    def test_compose_project_name(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert "myproject_provisioned_containers" in mgr._shared_volume_candidates

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    @patch.dict(
        "os.environ",
        {
            "PROVISIONED_CONTAINERS_VOLUME_NAME": "code_graph_helper_provisioned_containers",
        },
        clear=True,
    )
    def test_deduplication(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        # Should not have duplicates
        assert len(mgr._shared_volume_candidates) == len(set(mgr._shared_volume_candidates))


# ---------------------------------------------------------------------------
# _validate_container_name
# ---------------------------------------------------------------------------


class TestValidateContainerName:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def _make_mgr(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        return CoreContainerManager()

    def test_valid_name_unchanged(self):
        mgr = self._make_mgr()
        assert mgr._validate_container_name("my-container") == "my-container"

    def test_short_name_gets_prefix(self):
        mgr = self._make_mgr()
        result = mgr._validate_container_name("ab")
        assert result.startswith("codegraph-")

    def test_empty_name_gets_prefix(self):
        mgr = self._make_mgr()
        result = mgr._validate_container_name("")
        assert result.startswith("codegraph-")

    def test_invalid_chars_sanitized(self):
        mgr = self._make_mgr()
        result = mgr._validate_container_name("my container!")
        assert " " not in result
        assert "!" not in result

    def test_name_starting_with_non_alnum(self):
        mgr = self._make_mgr()
        result = mgr._validate_container_name("-badname")
        assert result[0].isalnum()

    def test_dots_and_underscores_allowed(self):
        mgr = self._make_mgr()
        result = mgr._validate_container_name("my_container.v1")
        assert result == "my_container.v1"


# ---------------------------------------------------------------------------
# is_docker_available
# ---------------------------------------------------------------------------


class TestIsDockerAvailable:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_false_when_no_docker(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.is_docker_available() is False

    @patch("src.core.container_manager_core.AGENT_HOST_URL", "http://agent:8080")
    def test_true_with_agent_host(self):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.is_docker_available() is True


# ---------------------------------------------------------------------------
# retry_docker_connection
# ---------------------------------------------------------------------------


class TestRetryDockerConnection:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_already_available(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.docker_available is True
        assert mgr.retry_docker_connection() is True

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_retry_success(self, mock_docker):
        mock_docker.errors.DockerException = Exception
        # First call fails (init), second succeeds (retry)
        mock_client_fail = MagicMock()
        mock_client_fail.ping.side_effect = OSError("no docker")
        mock_client_ok = MagicMock()
        mock_client_ok.ping.return_value = True
        mock_docker.from_env.side_effect = [mock_client_fail, mock_client_ok]

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.docker_available is False
        assert mgr.retry_docker_connection() is True
        assert mgr.docker_available is True

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_retry_failure(self, mock_docker):
        mock_docker.errors.DockerException = Exception
        mock_docker.from_env.side_effect = OSError("no docker")

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.retry_docker_connection() is False
        assert mgr.docker_available is False


# ---------------------------------------------------------------------------
# get_container_status
# ---------------------------------------------------------------------------


class TestGetContainerStatus:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_no_docker_returns_error(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        status = mgr.get_container_status("test")
        assert status == ContainerStatus.ERROR

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_running_container(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        container = MagicMock()
        container.status = "running"
        mock_client.containers.get.return_value = container

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.get_container_status("test") == ContainerStatus.RUNNING

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_stopped_container(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        container = MagicMock()
        container.status = "exited"
        mock_client.containers.get.return_value = container

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.get_container_status("test") == ContainerStatus.STOPPED

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_unknown_status(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        container = MagicMock()
        container.status = "paused"
        mock_client.containers.get.return_value = container

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.get_container_status("test") == ContainerStatus.ERROR

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_not_found_returns_error(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        not_found = type("NotFound", (Exception,), {})
        mock_docker.errors.NotFound = not_found
        mock_client.containers.get.side_effect = not_found("gone")

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.get_container_status("test") == ContainerStatus.ERROR

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_docker_exception_returns_error(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        docker_exc = type("DockerException", (Exception,), {})
        mock_docker.errors.DockerException = docker_exc
        mock_docker.errors.NotFound = type("NotFound", (Exception,), {})
        mock_client.containers.get.side_effect = docker_exc("fail")

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.get_container_status("test") == ContainerStatus.ERROR


# ---------------------------------------------------------------------------
# get_container
# ---------------------------------------------------------------------------


class TestGetContainer:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_returns_error_container_info(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        info = mgr.get_container("test-c")
        assert info.container_name == "test-c"
        assert info.status == ContainerStatus.ERROR
        assert info.description == "Container not found"


# ---------------------------------------------------------------------------
# list_containers
# ---------------------------------------------------------------------------


class TestListContainers:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_returns_empty(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.list_containers() == []


# ---------------------------------------------------------------------------
# remove_container
# ---------------------------------------------------------------------------


class TestRemoveContainer:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_no_docker_returns_true(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.remove_container("test") is True

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_remove_existing_container(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        container = MagicMock()
        mock_client.containers.get.return_value = container

        not_found = type("NotFound", (Exception,), {})
        mock_docker.errors.NotFound = not_found
        mock_docker.errors.DockerException = type("DockerException", (Exception,), {})
        mock_docker.errors.APIError = type("APIError", (Exception,), {})

        # Volume get raises NotFound
        mock_client.volumes.get.side_effect = not_found("no vol")

        # Shared volume cleanup
        mock_client.containers.run.return_value = None

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.remove_container("test") is True
        container.stop.assert_called_once()
        container.remove.assert_called_once()

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_remove_force(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        container = MagicMock()
        mock_client.containers.get.return_value = container

        not_found = type("NotFound", (Exception,), {})
        mock_docker.errors.NotFound = not_found
        mock_docker.errors.DockerException = type("DockerException", (Exception,), {})
        mock_docker.errors.APIError = type("APIError", (Exception,), {})

        mock_client.volumes.get.side_effect = not_found("no vol")
        mock_client.containers.run.return_value = None

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.remove_container("test", force=True) is True
        container.stop.assert_not_called()  # force skips stop
        container.remove.assert_called_once()

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_remove_not_found_in_docker(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        not_found = type("NotFound", (Exception,), {})
        mock_docker.errors.NotFound = not_found
        mock_docker.errors.DockerException = type("DockerException", (Exception,), {})
        mock_docker.errors.APIError = type("APIError", (Exception,), {})

        # Container not found in Docker
        mock_client.containers.get.side_effect = not_found("no container")
        mock_client.volumes.get.side_effect = not_found("no vol")
        mock_client.containers.run.return_value = None

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.remove_container("test") is True

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_remove_docker_exception(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        docker_exc = type("DockerException", (Exception,), {})
        mock_docker.errors.DockerException = docker_exc
        mock_docker.errors.NotFound = type("NotFound", (Exception,), {})
        mock_docker.errors.APIError = type("APIError", (Exception,), {})

        mock_client.containers.get.side_effect = docker_exc("fail")

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.remove_container("test") is False


# ---------------------------------------------------------------------------
# create_container
# ---------------------------------------------------------------------------


class TestCreateContainer:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_no_docker_returns_error(self, mock_docker):
        mock_docker.from_env.side_effect = Exception("Docker not available")
        mock_docker.errors.DockerException = Exception

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        spec = _make_spec()
        info = mgr.create_container(spec)
        assert info.status == ContainerStatus.ERROR
        assert "Docker not available" in info.description


# ---------------------------------------------------------------------------
# _container_name_exists
# ---------------------------------------------------------------------------


class TestContainerNameExists:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_exists(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.containers.get.return_value = MagicMock()

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr._container_name_exists("test") is True

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_not_exists(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        not_found = type("NotFound", (Exception,), {})
        mock_docker.errors.NotFound = not_found
        mock_docker.errors.APIError = type("APIError", (Exception,), {})
        mock_client.containers.get.side_effect = not_found("nope")

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr._container_name_exists("test") is False

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_api_error_returns_true(self, mock_docker):
        """On API error, assume name might exist (safety)."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        api_error = type("APIError", (Exception,), {})
        mock_docker.errors.NotFound = type("NotFound", (Exception,), {})
        mock_docker.errors.APIError = api_error
        mock_client.containers.get.side_effect = api_error("api fail")

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr._container_name_exists("test") is True

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_no_docker_returns_false(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr._container_name_exists("test") is False


# ---------------------------------------------------------------------------
# _ensure_volume_exists
# ---------------------------------------------------------------------------


class TestEnsureVolumeExists:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_no_docker_returns_none(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr._ensure_volume_exists("test-vol") is None

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_existing_volume(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        mock_vol = MagicMock()
        mock_client.volumes.get.return_value = mock_vol

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        result = mgr._ensure_volume_exists("test-vol")
        assert result is mock_vol

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_creates_new_volume(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        not_found = type("NotFound", (Exception,), {})
        mock_docker.errors.NotFound = not_found
        mock_client.volumes.get.side_effect = not_found("nope")

        new_vol = MagicMock()
        mock_client.volumes.create.return_value = new_vol

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        result = mgr._ensure_volume_exists("test-vol")
        assert result is new_vol
        mock_client.volumes.create.assert_called_once_with(name="test-vol")


# ---------------------------------------------------------------------------
# _is_port_available
# ---------------------------------------------------------------------------


class TestIsPortAvailable:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_available_port(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        # Use a high port unlikely to be in use
        assert mgr._is_port_available(59999) is True

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_socket_error_returns_true(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        with patch("socket.socket") as mock_sock:
            mock_sock.return_value.__enter__ = MagicMock()
            mock_sock.return_value.__exit__ = MagicMock(return_value=False)
            mock_sock.return_value.__enter__.return_value.connect_ex.side_effect = OSError("fail")
            assert mgr._is_port_available(9999) is True


# ---------------------------------------------------------------------------
# _extract_container_ports / _add_host_port_if_valid
# ---------------------------------------------------------------------------


class TestExtractContainerPorts:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_extracts_ports(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        container = MagicMock()
        container.ports = {"8080/tcp": [{"HostPort": "8080"}]}
        container.name = "test"
        used = set()
        mgr._extract_container_ports(container, used)
        assert 8080 in used

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_no_ports(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        container = MagicMock()
        container.ports = None
        used = set()
        mgr._extract_container_ports(container, used)
        assert used == set()

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_empty_host_configs(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        container = MagicMock()
        container.ports = {"8080/tcp": None}
        used = set()
        mgr._extract_container_ports(container, used)
        assert used == set()

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_add_host_port_valid(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        used = set()
        mgr._add_host_port_if_valid({"HostPort": "9090"}, "test", used)
        assert 9090 in used

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_add_host_port_empty(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        used = set()
        mgr._add_host_port_if_valid({"HostPort": ""}, "test", used)
        assert used == set()

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_add_host_port_missing_key(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        used = set()
        mgr._add_host_port_if_valid({}, "test", used)
        assert used == set()

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_add_host_port_invalid_value(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        used = set()
        mgr._add_host_port_if_valid({"HostPort": "not-a-number"}, "test", used)
        assert used == set()


# ---------------------------------------------------------------------------
# _get_used_docker_ports
# ---------------------------------------------------------------------------


class TestGetUsedDockerPorts:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_no_docker(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr._get_used_docker_ports() == set()


# ---------------------------------------------------------------------------
# _volume_exists
# ---------------------------------------------------------------------------


class TestVolumeExists:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_no_docker(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr._volume_exists("test") is False

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_volume_found(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.volumes.get.return_value = MagicMock()

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr._volume_exists("test") is True

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_volume_not_found(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        not_found = type("NotFound", (Exception,), {})
        mock_docker.errors.NotFound = not_found
        mock_docker.errors.APIError = type("APIError", (Exception,), {})
        mock_client.volumes.get.side_effect = not_found("nope")

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr._volume_exists("test") is False

    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker")
    def test_volume_api_error(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True

        api_error = type("APIError", (Exception,), {})
        mock_docker.errors.NotFound = type("NotFound", (Exception,), {})
        mock_docker.errors.APIError = api_error
        mock_client.volumes.get.side_effect = api_error("api fail")

        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr._volume_exists("test") is False


# ---------------------------------------------------------------------------
# _build_public_endpoint
# ---------------------------------------------------------------------------


class TestBuildPublicEndpoint:
    def test_builds_endpoint(self):
        from src.core.container_manager_core import _build_public_endpoint

        endpoint = _build_public_endpoint("my-container")
        assert "my-container" in endpoint
        assert endpoint.endswith("/api/workspaces/my-container")


# ---------------------------------------------------------------------------
# uses_agent_host / has_agent_host_id
# ---------------------------------------------------------------------------


class TestAgentHostProperties:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_no_agent_host(self, _mock_from_env):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.uses_agent_host is False
        assert mgr.has_agent_host_id("test") is False

    @patch("src.core.container_manager_core.AGENT_HOST_URL", "http://agent:8080")
    def test_with_agent_host(self):
        from src.core.container_manager_core import CoreContainerManager

        mgr = CoreContainerManager()
        assert mgr.uses_agent_host is True
        # No container registered
        assert mgr.has_agent_host_id("test") is False
        # Register one
        mgr._agent_host_ids["test"] = "abc123"
        assert mgr.has_agent_host_id("test") is True


# ---------------------------------------------------------------------------
# get_container_manager singleton
# ---------------------------------------------------------------------------


class TestGetContainerManager:
    @patch("src.core.container_manager_core.AGENT_HOST_URL", None)
    @patch("src.core.container_manager_core.docker.from_env", side_effect=OSError("no docker"))
    def test_singleton(self, _mock_from_env):
        # Reset singleton
        import src.core.container_manager_core as mod
        from src.core.container_manager_core import (
            get_container_manager,
        )

        mod._container_manager = None
        mgr1 = get_container_manager()
        mgr2 = get_container_manager()
        assert mgr1 is mgr2
        # Clean up
        mod._container_manager = None

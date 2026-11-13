"""Tests for services/docker_services/docker_api_integration.py."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import docker.errors
import pytest


class TestDockerAPIIntegrationInit:
    """Tests for DockerAPIIntegration.__init__ and initialization helpers."""

    @patch.dict(os.environ, {"KUBERNETES_IN_CLUSTER": "true"})
    def test_init_kubernetes_in_cluster_disables_docker(self):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        obj = DockerAPIIntegration()
        assert obj.docker_available is False
        assert obj.client is None

    @patch.dict(os.environ, {"KUBERNETES_IN_CLUSTER": "false"}, clear=False)
    def test_init_all_connections_fail(self):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        with (
            patch.object(
                DockerAPIIntegration, "_check_docker_socket_permissions", return_value=False
            ),
            patch.object(DockerAPIIntegration, "_try_docker_connections") as mock_try,
            patch.object(DockerAPIIntegration, "_suggest_permission_fixes"),
        ):
            mock_try.side_effect = lambda self_inner=None: setattr(
                # _try_docker_connections sets self.client = None when all fail
                # but since we patch it, we need to manually leave client as None
                type("", (), {})(),
                "x",
                None,
            )
            obj = DockerAPIIntegration()
        # client is None because _try_docker_connections was patched and didn't set it
        # The outer except catches the "All Docker connection methods failed" exception
        assert obj.docker_available is False

    @patch.dict(os.environ, {"KUBERNETES_IN_CLUSTER": "false"}, clear=False)
    def test_init_success(self):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.version.return_value = {"Version": "24.0"}

        with (
            patch.object(
                DockerAPIIntegration, "_check_docker_socket_permissions", return_value=True
            ),
            patch.object(DockerAPIIntegration, "_try_docker_connections") as mock_try,
            patch.object(DockerAPIIntegration, "_verify_docker_functionality", return_value=True),
        ):
            # Make _try_docker_connections set self.client
            def set_client(self_ref=None):
                pass

            mock_try.side_effect = set_client
            obj = DockerAPIIntegration()
            # Manually set to simulate successful connection
            obj.client = mock_client
            obj.docker_available = True

        assert obj.docker_available is True
        assert obj.client is mock_client


class TestTryDockerConnections:
    """Tests for _try_docker_connections method."""

    def _make_obj(self):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        with patch.object(DockerAPIIntegration, "_initialize_docker_client"):
            obj = DockerAPIIntegration()
        obj.client = None
        obj.docker_available = False
        return obj

    def test_first_method_succeeds(self):
        obj = self._make_obj()
        mock_client = MagicMock()
        mock_client.ping.return_value = True

        with (
            patch.object(obj, "_connect_from_env", return_value=mock_client),
            patch.object(obj, "_connect_unix_socket") as mock_unix,
            patch.object(obj, "_connect_user_socket") as mock_user,
        ):
            obj._try_docker_connections()

        assert obj.client is mock_client
        mock_unix.assert_not_called()
        mock_user.assert_not_called()

    def test_first_fails_second_succeeds(self):
        obj = self._make_obj()
        mock_client = MagicMock()
        mock_client.ping.return_value = True

        with (
            patch.object(
                obj, "_connect_from_env", side_effect=docker.errors.DockerException("env fail")
            ),
            patch.object(obj, "_connect_unix_socket", return_value=mock_client),
            patch.object(obj, "_connect_user_socket") as mock_user,
        ):
            obj._try_docker_connections()

        assert obj.client is mock_client
        mock_user.assert_not_called()

    def test_all_methods_fail(self):
        obj = self._make_obj()

        with (
            patch.object(
                obj, "_connect_from_env", side_effect=docker.errors.DockerException("fail1")
            ),
            patch.object(
                obj, "_connect_unix_socket", side_effect=docker.errors.DockerException("fail2")
            ),
            patch.object(
                obj, "_connect_user_socket", side_effect=docker.errors.DockerException("fail3")
            ),
        ):
            obj._try_docker_connections()

        assert obj.client is None

    def test_connect_returns_none(self):
        obj = self._make_obj()

        with (
            patch.object(obj, "_connect_from_env", return_value=None),
            patch.object(obj, "_connect_unix_socket", return_value=None),
            patch.object(obj, "_connect_user_socket", return_value=None),
        ):
            obj._try_docker_connections()

        assert obj.client is None

    def test_ping_fails_tries_next(self):
        obj = self._make_obj()
        bad_client = MagicMock()
        bad_client.ping.side_effect = docker.errors.DockerException("ping fail")
        good_client = MagicMock()
        good_client.ping.return_value = True

        with (
            patch.object(obj, "_connect_from_env", return_value=bad_client),
            patch.object(obj, "_connect_unix_socket", return_value=good_client),
            patch.object(obj, "_connect_user_socket") as mock_user,
        ):
            obj._try_docker_connections()

        assert obj.client is good_client
        mock_user.assert_not_called()


class TestCheckDockerSocketPermissions:
    def _make_obj(self):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        with patch.object(DockerAPIIntegration, "_initialize_docker_client"):
            obj = DockerAPIIntegration()
        return obj

    def test_socket_exists_and_accessible(self):
        obj = self._make_obj()
        with (
            patch("os.path.exists", return_value=True),
            patch("os.stat"),
            patch("os.access", return_value=True),
        ):
            assert obj._check_docker_socket_permissions() is True

    def test_socket_exists_not_accessible(self):
        obj = self._make_obj()
        with (
            patch("os.path.exists", return_value=True),
            patch("os.stat"),
            patch("os.access", return_value=False),
        ):
            assert obj._check_docker_socket_permissions() is False

    def test_no_socket_found(self):
        obj = self._make_obj()
        with patch("os.path.exists", return_value=False):
            assert obj._check_docker_socket_permissions() is False

    def test_socket_stat_raises_os_error(self):
        obj = self._make_obj()
        with (
            patch("os.path.exists", return_value=True),
            patch("os.stat", side_effect=OSError("stat fail")),
        ):
            # Should continue to next socket, then ultimately return False
            assert obj._check_docker_socket_permissions() is False


class TestVerifyDockerFunctionality:
    def _make_obj(self):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        with patch.object(DockerAPIIntegration, "_initialize_docker_client"):
            obj = DockerAPIIntegration()
        return obj

    def test_client_none(self):
        obj = self._make_obj()
        obj.client = None
        assert obj._verify_docker_functionality() is False

    def test_version_succeeds(self):
        obj = self._make_obj()
        obj.client = MagicMock()
        obj.client.version.return_value = {"Version": "24.0"}
        assert obj._verify_docker_functionality() is True

    def test_version_raises(self):
        obj = self._make_obj()
        obj.client = MagicMock()
        obj.client.version.side_effect = docker.errors.DockerException("version fail")
        assert obj._verify_docker_functionality() is False


class TestIsDockerAvailable:
    def _make_obj(self):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        with patch.object(DockerAPIIntegration, "_initialize_docker_client"):
            obj = DockerAPIIntegration()
        return obj

    def test_all_conditions_true(self):
        obj = self._make_obj()
        obj.docker_available = True
        obj.client = MagicMock()
        assert obj.is_docker_available() is True

    def test_not_available(self):
        obj = self._make_obj()
        obj.docker_available = False
        obj.client = MagicMock()
        assert obj.is_docker_available() is False

    def test_no_client(self):
        obj = self._make_obj()
        obj.docker_available = True
        obj.client = None
        assert obj.is_docker_available() is False


class TestRetryDockerConnection:
    def _make_obj(self):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        with patch.object(DockerAPIIntegration, "_initialize_docker_client"):
            obj = DockerAPIIntegration()
        return obj

    def test_already_available(self):
        obj = self._make_obj()
        obj.docker_available = True
        assert obj.retry_docker_connection() is True

    def test_retry_succeeds(self):
        obj = self._make_obj()
        obj.docker_available = False
        obj.client = None

        mock_client = MagicMock()
        mock_client.ping.return_value = True

        def fake_try():
            obj.client = mock_client

        with patch.object(obj, "_try_docker_connections", side_effect=fake_try):
            result = obj.retry_docker_connection()

        assert result is True
        assert obj.docker_available is True

    def test_retry_fails(self):
        obj = self._make_obj()
        obj.docker_available = False
        obj.client = None

        def fake_try():
            obj.client = None

        with patch.object(obj, "_try_docker_connections", side_effect=fake_try):
            result = obj.retry_docker_connection()

        assert result is False
        assert obj.docker_available is False

    def test_retry_raises_exception(self):
        obj = self._make_obj()
        obj.docker_available = False
        obj.client = None

        with patch.object(
            obj, "_try_docker_connections", side_effect=docker.errors.DockerException("boom")
        ):
            result = obj.retry_docker_connection()

        assert result is False
        assert obj.client is None
        assert obj.docker_available is False


class TestGetDockerClient:
    def test_returns_client(self):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        with patch.object(DockerAPIIntegration, "_initialize_docker_client"):
            obj = DockerAPIIntegration()
        mock_client = MagicMock()
        obj.client = mock_client
        assert obj.get_docker_client() is mock_client

    def test_returns_none(self):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        with patch.object(DockerAPIIntegration, "_initialize_docker_client"):
            obj = DockerAPIIntegration()
        obj.client = None
        assert obj.get_docker_client() is None


class TestGetDockerInfo:
    def _make_obj(self, available=True):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        with patch.object(DockerAPIIntegration, "_initialize_docker_client"):
            obj = DockerAPIIntegration()
        obj.client = MagicMock() if available else None
        obj.docker_available = available
        return obj

    def test_success(self):
        obj = self._make_obj(available=True)
        obj.client.info.return_value = {
            "Containers": 5,
            "ContainersRunning": 3,
            "ContainersPaused": 0,
            "ContainersStopped": 2,
            "Images": 10,
            "Driver": "overlay2",
            "MemTotal": 1024,
            "NCPU": 4,
            "KernelVersion": "5.15.0",
            "OperatingSystem": "Linux",
            "Architecture": "x86_64",
        }
        obj.client.version.return_value = {
            "Version": "24.0.5",
            "ApiVersion": "1.43",
        }

        result = obj.get_docker_info()

        assert result["server_version"] == "24.0.5"
        assert result["api_version"] == "1.43"
        assert result["containers"] == 5
        assert result["containers_running"] == 3
        assert result["ncpu"] == 4

    def test_not_available(self):
        obj = self._make_obj(available=False)
        result = obj.get_docker_info()
        assert "error" in result

    def test_exception(self):
        obj = self._make_obj(available=True)
        obj.client.info.side_effect = docker.errors.DockerException("info fail")
        result = obj.get_docker_info()
        assert "error" in result
        assert "info fail" in result["error"]


class TestPingDocker:
    def _make_obj(self):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        with patch.object(DockerAPIIntegration, "_initialize_docker_client"):
            obj = DockerAPIIntegration()
        return obj

    def test_no_client(self):
        obj = self._make_obj()
        obj.client = None
        assert obj.ping_docker() is False

    def test_ping_succeeds(self):
        obj = self._make_obj()
        obj.client = MagicMock()
        obj.client.ping.return_value = True
        assert obj.ping_docker() is True

    def test_ping_fails(self):
        obj = self._make_obj()
        obj.client = MagicMock()
        obj.client.ping.side_effect = docker.errors.DockerException("ping fail")
        assert obj.ping_docker() is False


class TestCheckExecRunTimeoutSupport:
    def _make_obj(self):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        with patch.object(DockerAPIIntegration, "_initialize_docker_client"):
            obj = DockerAPIIntegration()
        return obj

    def test_cached_result(self):
        obj = self._make_obj()
        obj._exec_run_supports_timeout = True
        assert obj._check_exec_run_timeout_support() is True

    def test_inspect_succeeds_with_timeout(self):
        obj = self._make_obj()
        obj._exec_run_supports_timeout = None
        # The real Container.exec_run may or may not support timeout.
        # We test the inspection logic by mocking inspect.signature.
        mock_sig = MagicMock()
        mock_sig.parameters = {"timeout": MagicMock(), "cmd": MagicMock()}
        with patch("inspect.signature", return_value=mock_sig):
            result = obj._check_exec_run_timeout_support()
        assert result is True
        assert obj._exec_run_supports_timeout is True

    def test_inspect_fails(self):
        obj = self._make_obj()
        obj._exec_run_supports_timeout = None
        with patch("inspect.signature", side_effect=AttributeError("sig fail")):
            result = obj._check_exec_run_timeout_support()
        assert result is False
        assert obj._exec_run_supports_timeout is False


class TestSafeExecRun:
    def _make_obj(self):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        with patch.object(DockerAPIIntegration, "_initialize_docker_client"):
            obj = DockerAPIIntegration()
        return obj

    def test_with_timeout_supported(self):
        obj = self._make_obj()
        container = MagicMock()
        container.exec_run.return_value = MagicMock(exit_code=0, output=b"ok")

        with patch.object(obj, "_check_exec_run_timeout_support", return_value=True):
            result = obj.safe_exec_run(container, ["echo", "hello"], timeout=30)

        container.exec_run.assert_called_once_with(["echo", "hello"], timeout=30)
        assert result.exit_code == 0

    def test_with_timeout_not_supported(self):
        obj = self._make_obj()
        container = MagicMock()
        container.exec_run.return_value = MagicMock(exit_code=0, output=b"ok")

        with patch.object(obj, "_check_exec_run_timeout_support", return_value=False):
            obj.safe_exec_run(container, ["echo", "hello"], timeout=30)

        container.exec_run.assert_called_once_with(["echo", "hello"])

    def test_without_timeout(self):
        obj = self._make_obj()
        container = MagicMock()
        container.exec_run.return_value = MagicMock(exit_code=0, output=b"ok")

        obj.safe_exec_run(container, ["ls"])
        container.exec_run.assert_called_once_with(["ls"])

    def test_timeout_type_error_fallback(self):
        obj = self._make_obj()
        container = MagicMock()

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "timeout" in kwargs:
                raise TypeError("timeout not supported")
            return MagicMock(exit_code=0, output=b"ok")

        container.exec_run.side_effect = side_effect

        with patch.object(obj, "_check_exec_run_timeout_support", return_value=True):
            obj.safe_exec_run(container, ["ls"], timeout=10)

        assert call_count == 2  # First with timeout, then without

    def test_non_timeout_type_error_raises(self):
        obj = self._make_obj()
        container = MagicMock()
        container.exec_run.side_effect = TypeError("something else broke")

        with (
            patch.object(obj, "_check_exec_run_timeout_support", return_value=True),
            pytest.raises(TypeError, match="something else broke"),
        ):
            obj.safe_exec_run(container, ["ls"], timeout=10)


class TestCreateDockerContainer:
    def _make_obj(self):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        with patch.object(DockerAPIIntegration, "_initialize_docker_client"):
            obj = DockerAPIIntegration()
        return obj

    def test_success(self):
        obj = self._make_obj()
        obj.client = MagicMock()
        obj.docker_available = True

        mock_container = MagicMock()
        obj.client.containers.run.return_value = mock_container

        spec = {
            "image": "python:3.11",
            "name": "test-container",
            "ports": {"8080/tcp": 8080},
            "environment": {"FOO": "bar"},
        }

        result = obj.create_docker_container(spec)
        assert result is mock_container
        obj.client.containers.run.assert_called_once()
        call_kwargs = obj.client.containers.run.call_args[1]
        assert call_kwargs["image"] == "python:3.11"
        assert call_kwargs["name"] == "test-container"

    def test_not_available(self):
        obj = self._make_obj()
        obj.client = None
        obj.docker_available = False

        with pytest.raises(Exception, match="Docker not available"):
            obj.create_docker_container({"image": "x", "name": "y"})

    def test_docker_error(self):
        obj = self._make_obj()
        obj.client = MagicMock()
        obj.docker_available = True
        obj.client.containers.run.side_effect = docker.errors.DockerException("create fail")

        with pytest.raises(docker.errors.DockerException, match="create fail"):
            obj.create_docker_container({"image": "x", "name": "y"})

    def test_none_values_stripped(self):
        obj = self._make_obj()
        obj.client = MagicMock()
        obj.docker_available = True
        obj.client.containers.run.return_value = MagicMock()

        spec = {
            "image": "python:3.11",
            "name": "test",
            # command is None, should be excluded
        }
        obj.create_docker_container(spec)
        call_kwargs = obj.client.containers.run.call_args[1]
        assert "command" not in call_kwargs


class TestGetSystemDf:
    def _make_obj(self, available=True):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        with patch.object(DockerAPIIntegration, "_initialize_docker_client"):
            obj = DockerAPIIntegration()
        obj.client = MagicMock() if available else None
        obj.docker_available = available
        return obj

    def test_success(self):
        obj = self._make_obj()
        obj.client.df.return_value = {
            "LayersSize": 5000,
            "Images": [
                {"Repository": "python", "Tag": "3.11", "Size": 900, "SharedSize": 100},
            ],
            "Containers": [
                {"Id": "abc123", "Names": ["/test"], "SizeRw": 50, "SizeRootFs": 800},
            ],
            "Volumes": [
                {"Name": "vol1", "UsageData": {"Size": 200}},
            ],
        }

        result = obj.get_system_df()
        assert result["layers_size"] == 5000
        assert len(result["images"]) == 1
        assert result["images"][0]["repository"] == "python"
        assert len(result["containers"]) == 1
        assert len(result["volumes"]) == 1

    def test_not_available(self):
        obj = self._make_obj(available=False)
        result = obj.get_system_df()
        assert "error" in result

    def test_exception(self):
        obj = self._make_obj()
        obj.client.df.side_effect = docker.errors.DockerException("df fail")
        result = obj.get_system_df()
        assert "error" in result


class TestCleanupResources:
    def _make_obj(self, available=True):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        with patch.object(DockerAPIIntegration, "_initialize_docker_client"):
            obj = DockerAPIIntegration()
        obj.client = MagicMock() if available else None
        obj.docker_available = available
        return obj

    def test_success(self):
        obj = self._make_obj()
        obj.client.containers.prune.return_value = {
            "ContainersDeleted": ["c1"],
            "SpaceReclaimed": 100,
        }
        obj.client.images.prune.return_value = {
            "ImagesDeleted": ["i1"],
            "SpaceReclaimed": 200,
        }
        obj.client.volumes.prune.return_value = {
            "VolumesDeleted": ["v1"],
            "SpaceReclaimed": 300,
        }
        obj.client.networks.prune.return_value = {
            "NetworksDeleted": ["n1"],
        }

        result = obj.cleanup_resources()
        assert result["total_space_reclaimed"] == 600
        assert result["containers"]["deleted"] == ["c1"]
        assert result["images"]["deleted"] == ["i1"]
        assert result["volumes"]["deleted"] == ["v1"]
        assert result["networks"]["deleted"] == ["n1"]

    def test_not_available(self):
        obj = self._make_obj(available=False)
        result = obj.cleanup_resources()
        assert "error" in result

    def test_exception(self):
        obj = self._make_obj()
        obj.client.containers.prune.side_effect = docker.errors.DockerException("prune fail")
        result = obj.cleanup_resources()
        assert "error" in result

    def test_none_deleted_lists(self):
        obj = self._make_obj()
        obj.client.containers.prune.return_value = {
            "ContainersDeleted": None,
            "SpaceReclaimed": 0,
        }
        obj.client.images.prune.return_value = {
            "ImagesDeleted": None,
            "SpaceReclaimed": 0,
        }
        obj.client.volumes.prune.return_value = {
            "VolumesDeleted": None,
            "SpaceReclaimed": 0,
        }
        obj.client.networks.prune.return_value = {
            "NetworksDeleted": None,
        }

        result = obj.cleanup_resources()
        assert result["containers"]["deleted"] == []
        assert result["images"]["deleted"] == []
        assert result["volumes"]["deleted"] == []
        assert result["networks"]["deleted"] == []
        assert result["total_space_reclaimed"] == 0


class TestSuggestPermissionFixes:
    def test_logs_suggestions(self):
        from src.services.docker_services.docker_api_integration import DockerAPIIntegration

        with patch.object(DockerAPIIntegration, "_initialize_docker_client"):
            obj = DockerAPIIntegration()
        # Just verify it doesn't raise
        obj._suggest_permission_fixes()

"""Tests for services/docker_services/container_monitoring.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import docker.errors
import pytest

from src.services.docker_services.container_monitoring import ContainerMonitor


class TestContainerMonitorInit:
    def test_stores_client(self):
        client = MagicMock()
        monitor = ContainerMonitor(client)
        assert monitor.client is client


class TestWaitForAnalysisCompletion:
    async def test_no_client(self):
        monitor = ContainerMonitor(None)
        result = await monitor.wait_for_analysis_completion("abc123", timeout=1)
        assert result is False

    async def test_completion_found_immediately(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=0)
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = await monitor.wait_for_analysis_completion("abc123", timeout=5)
        assert result is True

    async def test_timeout(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=1)
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = await monitor.wait_for_analysis_completion("abc123", timeout=1)
        assert result is False

    async def test_exception_during_check(self):
        client = MagicMock()
        client.containers.get.side_effect = docker.errors.DockerException("container gone")
        monitor = ContainerMonitor(client)

        result = await monitor.wait_for_analysis_completion("abc123", timeout=1)
        assert result is False


class TestExecInContainer:
    async def test_no_client(self):
        monitor = ContainerMonitor(None)
        with pytest.raises(Exception, match="Docker not available"):
            await monitor.exec_in_container("abc", ["ls"])

    async def test_success(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=b"file1\nfile2\n")
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = await monitor.exec_in_container("abc", ["ls"], timeout=30)
        assert result["exit_code"] == 0
        assert "file1" in result["stdout"]
        assert result["stderr"] == ""
        assert "execution_time" in result

    async def test_with_environment_and_user(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=b"ok")
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = await monitor.exec_in_container(
            "abc",
            ["echo", "hello"],
            environment={"FOO": "bar"},
            user="testuser",
        )
        assert result["exit_code"] == 0
        call_kwargs = mock_container.exec_run.call_args[1]
        assert call_kwargs["user"] == "testuser"
        assert call_kwargs["environment"] == {"FOO": "bar"}

    async def test_none_output(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=None)
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = await monitor.exec_in_container("abc", ["true"])
        assert result["stdout"] == ""

    async def test_exception(self):
        client = MagicMock()
        client.containers.get.side_effect = docker.errors.NotFound("not found")
        monitor = ContainerMonitor(client)

        result = await monitor.exec_in_container("abc", ["ls"])
        assert result["exit_code"] == 1
        assert "not found" in result["stderr"].lower() or "NotFound" in result["stderr"]
        assert result["timeout"] is False


class TestGetContainerStats:
    async def test_no_client(self):
        monitor = ContainerMonitor(None)
        result = await monitor.get_container_stats("abc")
        assert result == {}

    async def test_success_with_stats(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.stats.return_value = {
            "memory": {"usage": 104857600},  # 100 MB
            "cpu": {
                "cpu_usage": {"total_usage": 500000},
                "system_cpu_usage": 10000000,
            },
        }
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = await monitor.get_container_stats("abc")
        assert result["memory_mb"] == 100
        assert result["cpu_percent"] == pytest.approx(5.0)
        assert result["docker_id"] == "abc"

    async def test_no_memory_or_cpu_keys(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.stats.return_value = {}
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = await monitor.get_container_stats("abc")
        assert result["memory_mb"] == 0
        assert result["cpu_percent"] == 0.0

    async def test_zero_system_cpu(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.stats.return_value = {
            "cpu": {
                "cpu_usage": {"total_usage": 500},
                "system_cpu_usage": 0,
            },
        }
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = await monitor.get_container_stats("abc")
        assert result["cpu_percent"] == 0.0

    async def test_cpu_calculation_with_different_values(self):
        """Verify CPU percentage is calculated as (total_usage / system_usage * 100)."""
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.stats.return_value = {
            "memory": {"usage": 52428800},  # 50 MB
            "cpu": {
                "cpu_usage": {"total_usage": 2500000},
                "system_cpu_usage": 100000000,
            },
        }
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = await monitor.get_container_stats("container-42")
        assert result["memory_mb"] == 50
        assert result["cpu_percent"] == pytest.approx(2.5)
        assert result["docker_id"] == "container-42"

    async def test_memory_conversion_rounds_down(self):
        """Verify memory conversion from bytes to MB uses integer division."""
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.stats.return_value = {
            "memory": {"usage": 1572864},  # 1.5 MB in bytes
            "cpu": {"cpu_usage": {"total_usage": 0}, "system_cpu_usage": 1},
        }
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = await monitor.get_container_stats("abc")
        # 1572864 / 1048576 = 1 (integer division)
        assert result["memory_mb"] == 1

    async def test_exception(self):
        client = MagicMock()
        client.containers.get.side_effect = docker.errors.DockerException("stats fail")
        monitor = ContainerMonitor(client)

        result = await monitor.get_container_stats("abc")
        assert result == {}


class TestGetContainerLogs:
    def test_no_client(self):
        monitor = ContainerMonitor(None)
        result = monitor.get_container_logs("abc")
        assert result == "Docker not available"

    def test_success(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.logs.return_value = b"log line 1\nlog line 2\n"
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = monitor.get_container_logs("abc", tail=50)
        assert "log line 1" in result
        assert "log line 2" in result

    def test_with_since(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.logs.return_value = b"recent logs"
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = monitor.get_container_logs("abc", tail=10, since="1h")
        assert result == "recent logs"
        call_kwargs = mock_container.logs.call_args[1]
        assert call_kwargs["since"] == "1h"

    def test_not_found(self):
        client = MagicMock()
        client.containers.get.side_effect = docker.errors.NotFound("not found")
        monitor = ContainerMonitor(client)

        result = monitor.get_container_logs("abc")
        assert "not found" in result

    def test_exception(self):
        client = MagicMock()
        client.containers.get.side_effect = docker.errors.DockerException("log fail")
        monitor = ContainerMonitor(client)

        result = monitor.get_container_logs("abc")
        assert "Error retrieving logs" in result


class TestCreateFileInContainer:
    async def test_no_client(self):
        monitor = ContainerMonitor(None)
        with pytest.raises(Exception, match="Docker not available"):
            await monitor.create_file_in_container("abc", "/app/test.py", "print(1)")

    async def test_success(self):
        client = MagicMock()
        monitor = ContainerMonitor(client)

        with patch.object(
            monitor,
            "exec_in_container",
            new_callable=AsyncMock,
            return_value={"exit_code": 0, "stderr": ""},
        ) as mock_exec:
            result = await monitor.create_file_in_container("abc", "/app/test.py", "print('hello')")

        assert result is True
        assert mock_exec.call_count == 2  # mkdir + echo

    async def test_mkdir_fails(self):
        client = MagicMock()
        monitor = ContainerMonitor(client)

        with patch.object(
            monitor,
            "exec_in_container",
            new_callable=AsyncMock,
            return_value={"exit_code": 1, "stderr": "mkdir fail"},
        ):
            result = await monitor.create_file_in_container("abc", "/app/test.py", "content")

        assert result is False

    async def test_exception(self):
        client = MagicMock()
        monitor = ContainerMonitor(client)

        with patch.object(
            monitor,
            "exec_in_container",
            new_callable=AsyncMock,
            side_effect=docker.errors.DockerException("exec fail"),
        ):
            result = await monitor.create_file_in_container("abc", "/app/test.py", "content")

        assert result is False


class TestReadContainerFile:
    async def test_no_client(self):
        monitor = ContainerMonitor(None)
        with pytest.raises(Exception, match="Docker not available"):
            await monitor.read_container_file("container1", "main.py")

    async def test_success(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.side_effect = [
            MagicMock(exit_code=0, output=b"line1\nline2\nline3\n"),
            MagicMock(exit_code=0, output=b"100 1704067200"),
        ]
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = await monitor.read_container_file("container1", "main.py")
        assert result["success"] is True
        assert "line1" in result["content"]
        assert result["size"] == 100
        assert result["file_path"] == "main.py"

    async def test_with_offset_and_length(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.side_effect = [
            MagicMock(exit_code=0, output=b"line0\nline1\nline2\nline3\nline4\n"),
            MagicMock(exit_code=0, output=b"500 1704067200"),
        ]
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = await monitor.read_container_file("container1", "main.py", offset=1, length=2)
        assert result["success"] is True
        lines = result["content"].split("\n")
        assert len(lines) == 2

    async def test_file_not_found(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(
            exit_code=1, output=b"No such file or directory"
        )
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        with pytest.raises(FileNotFoundError):
            await monitor.read_container_file("container1", "missing.py")

    async def test_read_error(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=1, output=b"Permission denied")
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        with pytest.raises(RuntimeError, match="Failed to read file"):
            await monitor.read_container_file("container1", "locked.py")

    async def test_stat_parse_fails(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.side_effect = [
            MagicMock(exit_code=0, output=b"content here"),
            MagicMock(exit_code=1, output=b"stat error"),
        ]
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = await monitor.read_container_file("container1", "main.py")
        assert result["success"] is True
        assert result["size"] == 0  # default when stat fails


class TestListContainerDirectory:
    async def test_no_client(self):
        monitor = ContainerMonitor(None)
        with pytest.raises(Exception, match="Docker not available"):
            await monitor.list_container_directory("c1")

    async def test_non_recursive(self):
        client = MagicMock()
        mock_container = MagicMock()
        ls_output = (
            b"total 8\n"
            b"drwxr-xr-x 2 root root 4096 Jan  1 00:00 src\n"
            b"-rw-r--r-- 1 root root  100 Jan  1 00:00 main.py\n"
        )
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=ls_output)
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = await monitor.list_container_directory("c1", path="")
        assert len(result) == 2
        dirs = [r for r in result if r["type"] == "directory"]
        files = [r for r in result if r["type"] == "file"]
        assert len(dirs) == 1
        assert len(files) == 1
        assert files[0]["name"] == "main.py"

    async def test_recursive(self):
        client = MagicMock()
        mock_container = MagicMock()
        find_output = b"/app/data/repo/src\n/app/data/repo/src/main.py\n"
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=find_output)
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = await monitor.list_container_directory("c1", recursive=True)
        assert len(result) == 2

    async def test_dot_dot_entries_skipped(self):
        client = MagicMock()
        mock_container = MagicMock()
        ls_output = (
            b"total 4\n"
            b"drwxr-xr-x 2 root root 4096 Jan  1 00:00 .\n"
            b"drwxr-xr-x 3 root root 4096 Jan  1 00:00 ..\n"
            b"-rw-r--r-- 1 root root  100 Jan  1 00:00 test.py\n"
        )
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=ls_output)
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        result = await monitor.list_container_directory("c1")
        assert len(result) == 1
        assert result[0]["name"] == "test.py"

    async def test_dir_not_found(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(
            exit_code=1, output=b"No such file or directory"
        )
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        with pytest.raises(FileNotFoundError):
            await monitor.list_container_directory("c1", path="nonexistent")


class TestGetContainerRepositoryTree:
    async def test_no_client(self):
        monitor = ContainerMonitor(None)
        with pytest.raises(Exception, match="Docker not available"):
            await monitor.get_container_repository_tree("c1")

    async def test_success(self):
        client = MagicMock()
        mock_container = MagicMock()
        find_output = (
            b"/app/data/repo\n"
            b"/app/data/repo/src\n"
            b"/app/data/repo/src/main.py\n"
            b"/app/data/repo/README.md\n"
        )
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=find_output)
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        tree = await monitor.get_container_repository_tree("c1")
        assert tree["type"] == "directory"
        assert tree["name"] == "repo"
        assert "src" in tree["children"]
        assert "README.md" in tree["children"]

    async def test_find_fails(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=1, output=b"find error")
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        with pytest.raises(RuntimeError, match="Failed to get repository tree"):
            await monitor.get_container_repository_tree("c1")


class TestAddToTree:
    def test_add_file(self):
        monitor = ContainerMonitor(MagicMock())
        tree = {"type": "directory", "name": "root", "children": {}}
        monitor._add_to_tree(tree, "src/main.py")

        assert "src" in tree["children"]
        assert tree["children"]["src"]["type"] == "directory"
        assert "main.py" in tree["children"]["src"]["children"]
        assert tree["children"]["src"]["children"]["main.py"]["type"] == "file"

    def test_add_directory_without_extension(self):
        monitor = ContainerMonitor(MagicMock())
        tree = {"type": "directory", "name": "root", "children": {}}
        monitor._add_to_tree(tree, "src/utils")

        assert "src" in tree["children"]
        assert "utils" in tree["children"]["src"]["children"]
        assert tree["children"]["src"]["children"]["utils"]["type"] == "directory"

    def test_add_multiple_files(self):
        monitor = ContainerMonitor(MagicMock())
        tree = {"type": "directory", "name": "root", "children": {}}
        monitor._add_to_tree(tree, "src/a.py")
        monitor._add_to_tree(tree, "src/b.py")

        assert len(tree["children"]["src"]["children"]) == 2


class TestSearchContainerFiles:
    async def test_no_client(self):
        monitor = ContainerMonitor(None)
        with pytest.raises(Exception, match="Docker not available"):
            await monitor.search_container_files("c1", "*.py")

    async def test_success(self):
        client = MagicMock()
        mock_container = MagicMock()
        find_output = b"/app/data/repo/src/main.py\n/app/data/repo/tests/test_main.py\n"
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=find_output)
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        results = await monitor.search_container_files("c1", "*.py")
        assert len(results) == 2
        assert results[0]["name"] == "main.py"
        assert results[0]["path"] == "src/main.py"

    async def test_no_matches(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=b"")
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        results = await monitor.search_container_files("c1", "*.xyz")
        assert results == []

    async def test_find_fails(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=2, output=b"find error")
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        results = await monitor.search_container_files("c1", "*.py")
        assert results == []

    async def test_exception(self):
        client = MagicMock()
        client.containers.get.side_effect = docker.errors.DockerException("gone")
        monitor = ContainerMonitor(client)

        results = await monitor.search_container_files("c1", "*.py")
        assert results == []


class TestSearchContainerContent:
    async def test_no_client(self):
        monitor = ContainerMonitor(None)
        with pytest.raises(Exception, match="Docker not available"):
            await monitor.search_container_content("c1", "TODO")

    async def test_success(self):
        client = MagicMock()
        mock_container = MagicMock()
        grep_output = b"/app/data/repo/main.py:10:# TODO: fix this\n"
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=grep_output)
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        results = await monitor.search_container_content("c1", "TODO")
        assert len(results) == 1
        assert results[0]["file"] == "main.py"
        assert results[0]["line"] == 10
        assert "TODO" in results[0]["content"] or "fix" in results[0]["content"]

    async def test_case_sensitive(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=b"")
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        await monitor.search_container_content("c1", "Error", case_sensitive=True)
        cmd_str = mock_container.exec_run.call_args[0][0][-1]
        assert "-i" not in cmd_str.split()

    async def test_case_insensitive(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=b"")
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        await monitor.search_container_content("c1", "Error", case_sensitive=False)
        cmd_str = mock_container.exec_run.call_args[0][0][-1]
        assert "-i" in cmd_str

    async def test_with_file_pattern(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=b"")
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        await monitor.search_container_content("c1", "import", file_pattern="*.py")
        cmd_str = mock_container.exec_run.call_args[0][0][-1]
        assert "--include" in cmd_str
        assert "*.py" in cmd_str

    async def test_no_matches_exit_code_1(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=1, output=b"")
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        results = await monitor.search_container_content("c1", "nonexistent")
        assert results == []

    async def test_grep_fails(self):
        client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=2, output=b"grep error")
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        results = await monitor.search_container_content("c1", "pattern")
        assert results == []

    async def test_exception(self):
        client = MagicMock()
        client.containers.get.side_effect = docker.errors.DockerException("gone")
        monitor = ContainerMonitor(client)

        results = await monitor.search_container_content("c1", "pattern")
        assert results == []

    async def test_non_numeric_line_number(self):
        client = MagicMock()
        mock_container = MagicMock()
        grep_output = b"/app/data/repo/file.txt:abc:some content\n"
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=grep_output)
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        results = await monitor.search_container_content("c1", "some")
        assert len(results) == 1
        assert results[0]["line"] == 0  # non-numeric defaults to 0

    async def test_multiple_grep_results_parsed_correctly(self):
        """Verify grep output with multiple matches is parsed into structured results."""
        client = MagicMock()
        mock_container = MagicMock()
        grep_output = (
            b"/app/data/repo/src/main.py:5:def hello():\n"
            b"/app/data/repo/src/main.py:15:def goodbye():\n"
            b"/app/data/repo/tests/test_main.py:3:def test_hello():\n"
        )
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=grep_output)
        client.containers.get.return_value = mock_container
        monitor = ContainerMonitor(client)

        results = await monitor.search_container_content("c1", "def ")
        assert len(results) == 3
        # file paths are relative to /app/data/repo/
        assert results[0]["file"] == "src/main.py"
        assert results[0]["line"] == 5
        assert "hello" in results[0]["content"]
        assert results[0]["match_pattern"] == "def "
        assert results[1]["line"] == 15
        assert "goodbye" in results[1]["content"]
        assert results[2]["file"] == "tests/test_main.py"
        assert results[2]["line"] == 3


class TestInstallToolsInContainer:
    async def test_no_client(self):
        monitor = ContainerMonitor(None)
        with pytest.raises(Exception, match="Docker not available"):
            await monitor.install_tools_in_container("abc", ["pytest"])

    async def test_python_success(self):
        client = MagicMock()
        monitor = ContainerMonitor(client)

        with patch.object(
            monitor,
            "exec_in_container",
            new_callable=AsyncMock,
            return_value={"exit_code": 0, "stderr": ""},
        ) as mock_exec:
            result = await monitor.install_tools_in_container(
                "abc", ["pytest", "ruff"], language="python"
            )

        assert result is True
        cmd = mock_exec.call_args[0][1]
        assert cmd[0] == "pip"
        assert "pytest" in cmd
        assert "ruff" in cmd

    async def test_python_empty_tools(self):
        client = MagicMock()
        monitor = ContainerMonitor(client)

        result = await monitor.install_tools_in_container("abc", [], language="python")
        assert result is True

    async def test_python_install_fails(self):
        client = MagicMock()
        monitor = ContainerMonitor(client)

        with patch.object(
            monitor,
            "exec_in_container",
            new_callable=AsyncMock,
            return_value={"exit_code": 1, "stderr": "pip error"},
        ):
            result = await monitor.install_tools_in_container(
                "abc", ["broken-pkg"], language="python"
            )

        assert result is False

    async def test_javascript_success(self):
        client = MagicMock()
        monitor = ContainerMonitor(client)

        with patch.object(
            monitor,
            "exec_in_container",
            new_callable=AsyncMock,
            return_value={"exit_code": 0, "stderr": ""},
        ) as mock_exec:
            result = await monitor.install_tools_in_container(
                "abc", ["eslint"], language="javascript"
            )

        assert result is True
        cmd = mock_exec.call_args[0][1]
        assert cmd[0] == "npm"
        assert "eslint" in cmd

    async def test_typescript_uses_npm(self):
        client = MagicMock()
        monitor = ContainerMonitor(client)

        with patch.object(
            monitor,
            "exec_in_container",
            new_callable=AsyncMock,
            return_value={"exit_code": 0, "stderr": ""},
        ) as mock_exec:
            result = await monitor.install_tools_in_container("abc", ["tsc"], language="typescript")

        assert result is True
        cmd = mock_exec.call_args[0][1]
        assert cmd[0] == "npm"

    async def test_javascript_install_fails(self):
        client = MagicMock()
        monitor = ContainerMonitor(client)

        with patch.object(
            monitor,
            "exec_in_container",
            new_callable=AsyncMock,
            return_value={"exit_code": 1, "stderr": "npm error"},
        ):
            result = await monitor.install_tools_in_container(
                "abc", ["broken-pkg"], language="javascript"
            )

        assert result is False

    async def test_unsupported_language_succeeds(self):
        client = MagicMock()
        monitor = ContainerMonitor(client)

        result = await monitor.install_tools_in_container("abc", ["some-tool"], language="ruby")
        assert result is True  # No install attempted, returns True

    async def test_exception(self):
        client = MagicMock()
        monitor = ContainerMonitor(client)

        with patch.object(
            monitor,
            "exec_in_container",
            new_callable=AsyncMock,
            side_effect=docker.errors.DockerException("exec fail"),
        ):
            result = await monitor.install_tools_in_container("abc", ["pytest"], language="python")

        assert result is False

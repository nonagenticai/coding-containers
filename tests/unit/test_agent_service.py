"""Tests for services/agent_service.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.agent_models import AgentExecutionRequest
from src.services.agent_service import AgentService


@pytest.fixture
def agent_service(mock_docker_service):
    return AgentService(mock_docker_service)


class TestBuildCliCommand:
    def test_claude_basic(self, agent_service):
        cmd = agent_service._build_cli_command("claude", "fix bug", None, None)
        assert cmd == ["claude", "-p", "fix bug", "--output-format", "json"]

    def test_claude_with_model(self, agent_service):
        cmd = agent_service._build_cli_command("claude", "fix", None, "opus")
        assert "--model" in cmd
        assert "opus" in cmd

    def test_claude_with_system_prompt(self, agent_service):
        cmd = agent_service._build_cli_command("claude", "fix", "you are helpful", None)
        assert "--system-prompt" in cmd
        assert "you are helpful" in cmd

    def test_codex_basic(self, agent_service):
        cmd = agent_service._build_cli_command("codex", "fix bug", None, None)
        assert cmd[0] == "codex"
        assert "fix bug" in cmd

    def test_codex_with_system_prompt(self, agent_service):
        cmd = agent_service._build_cli_command("codex", "fix", "be concise", None)
        assert any("be concise" in arg for arg in cmd)

    def test_unknown_cli_raises(self, agent_service):
        with pytest.raises(ValueError, match="Unknown CLI"):
            agent_service._build_cli_command("unknown", "prompt", None, None)


@pytest.mark.asyncio
class TestExecuteAgent:
    @patch("src.core.container_metadata_registry.ContainerMetadataRegistry")
    async def test_execute_success(self, mock_registry_cls, agent_service, mock_docker_service):
        mock_registry = MagicMock()
        mock_registry.load.return_value = {}
        mock_registry_cls.return_value = mock_registry

        mock_docker_service.exec_in_container = AsyncMock(
            return_value={"stdout": "done", "stderr": "", "exit_code": 0}
        )

        request = AgentExecutionRequest(cli="claude", prompt="fix bug")
        result = await agent_service.execute_agent("test-container", request)

        assert result.success is True
        assert result.exit_code == 0
        assert result.container_name == "test-container"

    @patch("src.core.container_metadata_registry.ContainerMetadataRegistry")
    async def test_execute_timeout(self, mock_registry_cls, agent_service, mock_docker_service):
        mock_registry = MagicMock()
        mock_registry.load.return_value = {}
        mock_registry_cls.return_value = mock_registry

        mock_docker_service.exec_in_container = AsyncMock(side_effect=TimeoutError("timed out"))

        request = AgentExecutionRequest(cli="claude", prompt="fix bug", timeout=10)
        result = await agent_service.execute_agent("test-container", request)

        assert result.success is False
        assert "timed out" in (result.error or "").lower()

    @patch("src.core.container_metadata_registry.ContainerMetadataRegistry")
    async def test_execute_unknown_cli(self, mock_registry_cls, agent_service):
        mock_registry = MagicMock()
        mock_registry.load.return_value = {}
        mock_registry_cls.return_value = mock_registry

        request = AgentExecutionRequest(cli="unknown_cli", prompt="fix bug")
        result = await agent_service.execute_agent("test-container", request)

        assert result.success is False
        assert result.exit_code == -1

    @patch("src.core.container_metadata_registry.ContainerMetadataRegistry")
    async def test_execute_k8s_path(self, mock_registry_cls, agent_service):
        """Execute agent via K8s exec path (lines 67-72)."""
        mock_registry = MagicMock()
        container_meta = MagicMock()
        container_meta.internal_endpoint = "http://deploy.ns.svc.cluster.local:8687"
        mock_registry.load.return_value = {"test-container": container_meta}
        mock_registry_cls.return_value = mock_registry

        mock_k8s = MagicMock()
        mock_k8s.exec_in_pod.return_value = {
            "stdout": "done via k8s",
            "stderr": "",
            "exit_code": 0,
        }

        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            return_value=mock_k8s,
        ):
            request = AgentExecutionRequest(cli="claude", prompt="fix bug")
            result = await agent_service.execute_agent("test-container", request)

        assert result.success is True
        assert "done via k8s" in result.output

    @patch("src.core.container_metadata_registry.ContainerMetadataRegistry")
    async def test_execute_claude_json_output(
        self, mock_registry_cls, agent_service, mock_docker_service
    ):
        """Claude JSON output parsing (lines 100-104)."""
        import json

        mock_registry = MagicMock()
        mock_registry.load.return_value = {}
        mock_registry_cls.return_value = mock_registry

        json_output = json.dumps(
            {
                "result": "Bug fixed successfully",
                "session_id": "sess-123",
                "total_cost_usd": 0.05,
                "duration_ms": 3000,
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
        )

        mock_docker_service.exec_in_container = AsyncMock(
            return_value={"stdout": json_output, "stderr": "", "exit_code": 0}
        )

        request = AgentExecutionRequest(cli="claude", prompt="fix bug")
        result = await agent_service.execute_agent("test-container", request)

        assert result.success is True
        assert result.output == "Bug fixed successfully"
        assert result.session_id == "sess-123"
        assert result.cost_usd == 0.05
        assert result.duration_ms == 3000
        assert result.usage == {"input_tokens": 100, "output_tokens": 50}

    @patch("src.core.container_metadata_registry.ContainerMetadataRegistry")
    async def test_execute_error_with_output(
        self, mock_registry_cls, agent_service, mock_docker_service
    ):
        """Error output combined with stdout (line 111)."""
        mock_registry = MagicMock()
        mock_registry.load.return_value = {}
        mock_registry_cls.return_value = mock_registry

        mock_docker_service.exec_in_container = AsyncMock(
            return_value={
                "stdout": "partial output",
                "stderr": "something went wrong",
                "exit_code": 1,
            }
        )

        request = AgentExecutionRequest(cli="codex", prompt="fix bug")
        result = await agent_service.execute_agent("test-container", request)

        assert result.success is False
        assert "partial output" in result.output
        assert "STDERR:" in result.output
        assert "something went wrong" in result.output

    @patch("src.core.container_metadata_registry.ContainerMetadataRegistry")
    async def test_execute_error_only_stderr(
        self, mock_registry_cls, agent_service, mock_docker_service
    ):
        """Error with no stdout uses stderr as output."""
        mock_registry = MagicMock()
        mock_registry.load.return_value = {}
        mock_registry_cls.return_value = mock_registry

        mock_docker_service.exec_in_container = AsyncMock(
            return_value={
                "stdout": "",
                "stderr": "fatal error",
                "exit_code": 1,
            }
        )

        request = AgentExecutionRequest(cli="codex", prompt="fix bug")
        result = await agent_service.execute_agent("test-container", request)

        assert result.success is False
        assert result.output == "fatal error"


class TestBuildCliCommandExtended:
    def test_codex_with_model(self, agent_service):
        """Codex command with model (line 186)."""
        cmd = agent_service._build_cli_command("codex", "fix", None, "gpt-4")
        assert "--model" in cmd
        assert "gpt-4" in cmd

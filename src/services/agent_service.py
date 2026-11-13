"""Agent execution service for code_graph_helper.

Executes CLI agents (Claude Code, Codex) in repository containers via docker exec,
leveraging existing DockerService infrastructure.
"""

import asyncio
import json
import logging
import time

import docker.errors

from src.core.log_safety import summarize_command
from src.models.agent_models import AgentExecutionRequest, AgentExecutionResponse
from src.services.docker_service import DockerService

logger = logging.getLogger(__name__)


class AgentService:
    """Service for executing CLI agents in containers."""

    def __init__(self, docker_service: DockerService):
        """Initialize agent service.

        Args:
            docker_service: Existing DockerService instance
        """
        self.docker_service = docker_service

    async def execute_agent(
        self, container_name: str, request: AgentExecutionRequest
    ) -> AgentExecutionResponse:
        """Execute CLI agent in container using docker exec.

        Args:
            container_name: Container name where agent should execute
            request: Agent execution request

        Returns:
            AgentExecutionResponse with results
        """
        start_time = time.time()

        try:
            # Build CLI command
            command = self._build_cli_command(
                cli=request.cli,
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                model=request.model,
            )

            logger.info(f"Executing {request.cli} agent in container {container_name[:12]}")

            # Check if this is a K8s-provisioned container
            from src.core.container_metadata_registry import ContainerMetadataRegistry

            registry = ContainerMetadataRegistry()
            registry_data = registry.load()
            container_meta = registry_data.get(container_name)

            runtime_endpoint = (
                getattr(container_meta, "internal_endpoint", "")
                or getattr(container_meta, "api_endpoint", "")
                or ""
            )
            if container_meta and ".svc.cluster.local" in runtime_endpoint:
                # K8s pod exec path
                from src.core.kubernetes_container_provisioner import (
                    get_k8s_container_provisioner,
                )

                k8s = get_k8s_container_provisioner()
                # Offload blocking k8s exec to a thread -- keeps the event loop
                # free so the /health liveness probe never times out mid-exec.
                result = await asyncio.to_thread(
                    k8s.exec_in_pod,
                    deployment_name=container_name,
                    command=command,
                    workdir="/repo",
                    timeout=request.timeout,
                )
            else:
                # Docker exec path
                result = await self.docker_service.exec_in_container(
                    docker_id=container_name, command=command, timeout=request.timeout
                )

            execution_time = time.time() - start_time
            raw_output = result.get("stdout", "")
            error = result.get("stderr", "")
            exit_code = result.get("exit_code", -1)

            # Parse JSON output from Claude Code (with --output-format json)
            session_id = None
            cost_usd = None
            duration_ms = None
            usage = None
            output = raw_output

            if request.cli == "claude" and raw_output:
                try:
                    parsed = json.loads(raw_output)
                    # Extract actual result from JSON
                    output = parsed.get("result", raw_output)
                    session_id = parsed.get("session_id")
                    cost_usd = parsed.get("total_cost_usd")
                    duration_ms = parsed.get("duration_ms")
                    usage = parsed.get("usage")
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Claude output as JSON, using raw output")
                    output = raw_output

            # Combine stdout and stderr if both present (for error cases)
            if error and exit_code != 0:
                output = f"{output}\n\nSTDERR:\n{error}" if output else error

            logger.info(
                f"Agent execution completed. "
                f"Exit code: {exit_code}, Duration: {execution_time:.2f}s, "
                f"Session: {session_id}"
            )

            return AgentExecutionResponse(
                success=exit_code == 0,
                output=output,
                error=error if exit_code != 0 else None,
                exit_code=exit_code,
                execution_time=execution_time,
                container_name=container_name,
                session_id=session_id,
                cost_usd=cost_usd,
                duration_ms=duration_ms,
                usage=usage,
            )

        except TimeoutError:
            execution_time = time.time() - start_time
            logger.warning(
                f"Agent execution timed out after {request.timeout}s in container {container_name[:12]}"
            )
            return AgentExecutionResponse(
                success=False,
                output="",
                error=f"Execution timed out after {request.timeout}s",
                exit_code=-1,
                execution_time=execution_time,
                container_name=container_name,
            )
        except (docker.errors.DockerException, RuntimeError, OSError, ValueError, ImportError) as e:
            execution_time = time.time() - start_time
            logger.exception(f"Agent execution error in container {container_name[:12]}: {e}")
            return AgentExecutionResponse(
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
                execution_time=execution_time,
                container_name=container_name,
            )

    def _build_cli_command(
        self, cli: str, prompt: str, system_prompt: str | None, model: str | None
    ) -> list[str]:
        """Build CLI command array.

        Args:
            cli: CLI name ('claude' or 'codex')
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Optional model specification

        Returns:
            Command as list of strings

        Raises:
            ValueError: If CLI is unknown
        """
        if cli == "claude":
            # Use -p with --output-format json for structured output
            # This gives us session_id, usage, cost, and result
            cmd = ["claude", "-p", prompt, "--output-format", "json"]
            if model:
                cmd.extend(["--model", model])
            if system_prompt:
                cmd.extend(["--system-prompt", system_prompt])

        elif cli == "codex":
            cmd = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox"]
            if model:
                cmd.extend(["--model", model])
            # Codex: prepend system prompt to task
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            cmd.append(full_prompt)

        else:
            raise ValueError(f"Unknown CLI: {cli}. Supported: 'claude', 'codex'")

        # ⛔ Previously `' '.join(cmd[:3])`, called "out of scope" on the theory
        # that a three-element slice is a program and two sub-commands. It is
        # not: the `claude` branch above builds ["claude", "-p", prompt, ...],
        # so element 2 of that slice is the ENTIRE PROMPT — which is
        # caller-supplied free text and routinely carries a credential
        # ("push the branch using token ghs_..."). The `codex` branch appends
        # the prompt last, so the slice hid it there and disclosed it here:
        # the same expression leaked or not depending on argv order, which is
        # precisely why a slice is not a redaction.
        logger.debug("Built CLI command: %s", summarize_command(cmd))
        return cmd

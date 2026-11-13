"""Agent execution endpoints for code_graph_helper.

Provides API endpoints for executing CLI agents (Claude Code, Codex) in
repository containers. Called by worker_management for delegated execution.
"""

import logging
from typing import Annotated

import docker.errors
from fastapi import APIRouter, Depends, Request

from coding_client.auth.tenant import require_tenant
from coding_client.models.responses import (
    internal_error,
    not_found,
)
from src.models.agent_models import (
    AgentCapabilitiesResponse,
    AgentCapabilityStatus,
    AgentExecutionRequest,
    AgentExecutionResponse,
)
from src.routers.tenant_guard import authorize_workspace
from src.runtime import ExecutionProfile, resolve_runtime
from src.services.agent_service import AgentService
from src.services.docker_service import DockerService, get_docker_service

logger = logging.getLogger(__name__)

#: CLIs ``agent-capabilities`` probes for, and the per-probe exec bound.
AGENT_CLIS = ("claude", "codex")
CAPABILITY_PROBE_TIMEOUT_S = 10

router = APIRouter()


def get_agent_service(
    docker_service: Annotated[DockerService, Depends(get_docker_service)],
) -> AgentService:
    """Dependency to get AgentService instance.

    Args:
        docker_service: Injected DockerService

    Returns:
        AgentService instance
    """
    return AgentService(docker_service)


@router.post(
    "/{container_name}/execute-agent",
    response_model=AgentExecutionResponse,
    operation_id="execute_workspace_agent",
    summary="Execute CLI agent in workspace",
    description="Execute Claude Code or Codex CLI in a repository container",
    responses={
        200: {
            "description": "Agent execution completed (check success field for result)",
            "model": AgentExecutionResponse,
        },
        404: {"description": "Container not found or not running"},
        500: {"description": "Internal server error"},
    },
)
async def execute_agent(
    container_name: str,
    request: AgentExecutionRequest,
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
    http_request: Request,
    tenant_id: str = Depends(require_tenant),
) -> AgentExecutionResponse:
    """Execute CLI agent in container.

    This endpoint is called by the worker service to execute agents
    in repository containers. Authentication is handled via license tokens.

    Args:
        container_name: Container name where agent should execute
        request: Agent execution request with CLI type, prompt, etc.
        agent_service: Injected AgentService
        http_request: FastAPI request (may have user context from auth middleware)

    Returns:
        AgentExecutionResponse with execution results

    Raises:
        HTTPException: 404 if container not found, 500 on unexpected errors
    """
    # Log request (user context available from auth middleware if configured)
    user = getattr(http_request.state, "user", None)
    user_id = getattr(user, "preferred_username", None) or "unknown" if user else "unknown"

    logger.info(
        f"Agent execution request from service account: {user_id}, "
        f"Container: {container_name[:12]}..., CLI: {request.cli}"
    )

    # ⭐ AUTHORIZATION. Must precede the dispatch, not merely the response: every
    # backend below selects the target by NAME alone.
    authorize_workspace(container_name, tenant_id, route="execute-agent")

    try:
        result = await agent_service.execute_agent(container_name, request)

        # Log result summary
        status_str = "SUCCESS" if result.success else "FAILED"
        logger.info(
            f"Agent execution {status_str}. "
            f"Container: {container_name[:12]}..., "
            f"Exit code: {result.exit_code}, "
            f"Duration: {result.execution_time:.2f}s"
        )

        return result

    except ValueError as e:
        # Container not found or not running
        logger.warning(f"Container validation failed: {e}")
        raise not_found("Container", container_name)
    except (docker.errors.DockerException, RuntimeError, OSError) as e:
        # Unexpected error
        logger.exception(f"Unexpected error during agent execution: {e}")
        raise internal_error("Internal server error during agent execution")


@router.get(
    "/{container_name}/agent-capabilities",
    response_model=AgentCapabilitiesResponse,
    operation_id="get_workspace_agent_capabilities",
    summary="Check workspace agent capabilities",
    description="Verify which CLI tools are available in container",
    responses={
        200: {
            "description": "Container capabilities retrieved",
            "model": AgentCapabilitiesResponse,
        },
    },
)
async def check_agent_capabilities(
    container_name: str,
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
    tenant_id: str = Depends(require_tenant),
) -> AgentCapabilitiesResponse:
    """Check if container has CLI tools available.

    Args:
        container_name: Container name to check
        agent_service: Injected AgentService

    Returns:
        Dictionary with available CLIs

    Note:
        Currently returns optimistic availability.
        Future: Can add actual checks by running `claude --version` etc.
    """
    logger.info(f"Agent capabilities check for container {container_name[:12]}...")

    # ⭐ AUTHORIZATION, before any exec reaches the workspace. This route runs
    # `which claude` / `which codex` INSIDE the named container, so an ungated
    # call is both a cross-tenant exec and an existence oracle for the name.
    info = authorize_workspace(container_name, tenant_id, route="agent-capabilities")

    # Through the workspace runtime (Kubernetes in the cluster), never Docker
    # directly. This used to call ``docker_service.exec_in_container``, which in
    # the cluster raises a bare ``Exception("Docker not available")`` — outside
    # every ``except`` below — so the live route answered 500 on every workspace.
    try:
        from src.core.container_manager_core import get_container_manager as get_core_manager

        # ``info`` is the record the gate above authorized: no second lookup.
        runtime = resolve_runtime(info, core=get_core_manager())
    except (RuntimeError, ValueError, KeyError, OSError, ImportError) as e:
        logger.warning(f"Agent capabilities: no runtime for {container_name[:12]}...: {e}")
        return AgentCapabilitiesResponse(
            container_name=container_name,
            capabilities={
                cli: AgentCapabilityStatus(available=False, note=f"Service error: {e}")
                for cli in AGENT_CLIS
            },
        )

    capabilities = {}
    for cli in AGENT_CLIS:
        try:
            result = await runtime.run_once(
                info,
                ["which", cli],
                timeout=CAPABILITY_PROBE_TIMEOUT_S,
                execution_profile=ExecutionProfile.SERVER_AUTHORED,
            )
            found = result.exit_code == 0
            capabilities[cli] = AgentCapabilityStatus(
                available=found,
                note="Verified via container check" if found else "Not found in container",
            )
        except Exception as e:  # noqa: BLE001 - a probe answers "unknown", never a 500
            logger.debug(f"Could not check {cli} in {container_name[:12]}...: {e}")
            capabilities[cli] = AgentCapabilityStatus(available=False, note=f"Check failed: {e}")

    return AgentCapabilitiesResponse(container_name=container_name, capabilities=capabilities)

"""Container write routes: create / delete-one / delete-all."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid7

import docker.errors
from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from coding_client.auth.models import User
from coding_client.auth.tenant import (
    get_current_user_with_tenant,
    require_tenant,
)
from coding_client.exceptions import BaseAPIException, InternalServerException
from coding_client.models.responses import internal_error, not_found
from src.core.toolchain import MemoryBudgetError
from src.models.container_api_models import (
    ContainerOperation,
    ContainerOperationResponse,
)
from src.models.container_models import ContainerInfo, GitHubRepoRequest
from src.models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error406Response,
    Error412Response,
    Error422Response,
    Error429Response,
    Error500Response,
)

from ._common import get_manager

logger = logging.getLogger(__name__)
router = APIRouter()


def memory_budget_refusal(e: MemoryBudgetError) -> BaseAPIException:
    """409 ``WORKSPACE_MEMORY_BUDGET_EXCEEDED`` naming the budget."""
    return BaseAPIException(
        status_code=409,
        message=str(e),
        error_code="WORKSPACE_MEMORY_BUDGET_EXCEEDED",
        details={
            "error_code": "WORKSPACE_MEMORY_BUDGET_EXCEEDED",
            "need_mi": e.need_mi,
            "max_mi": e.max_mi,
            "reserve_mi": e.reserve_mi,
            "toolchain": e.toolchain,
        },
    )


@router.post(
    "/",
    response_model=ContainerInfo,
    operation_id="create_workspace",
    summary="Create Workspace",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Container created successfully", "model": ContainerInfo},
        400: {"model": Error400Response, "description": "Invalid request"},
        412: {
            "model": Error412Response,
            "description": (
                "No git credential available and the repository is not public "
                "(GitHub App not installed on the owner, and no access token supplied)"
            ),
        },
        409: {
            "description": (
                "history_cutoff refused: the workspace already exists and the cutoff "
                "would rewrite its shared git history; or WORKSPACE_MEMORY_BUDGET_EXCEEDED: "
                "expected_services x service_memory_mi + expected_dependencies + the tooling "
                "reserve do not fit one workspace pod (8Gi). Not checked when an "
                "existing workspace is reused (its hints are ignored)"
            ),
        },
        429: {"model": Error429Response, "description": "Container capacity reached"},
        500: {"model": Error500Response, "description": "Container creation failed"},
    },
)
async def create_container(
    request: GitHubRepoRequest,
    user: User = Depends(get_current_user_with_tenant),
) -> dict[str, Any]:
    """Create a new container from a GitHub repository."""
    from src.models.container_models import GitHubRepoRequest as GHRepoRequest
    from src.services.docker_service import get_docker_service
    from src.services.secret_resolver import resolve_git_token

    try:
        # BYO token: if the caller passed a secret_config_id but no inline PAT,
        # resolve the per-tenant GIT_TOKEN from the secrets service so the clone
        # init container can reach PRIVATE repos. Best-effort — a miss leaves
        # access_token unset and public repos still provision normally.
        if request.secret_config_id and not request.access_token:
            resolved = await resolve_git_token(request.secret_config_id, user.effective_tenant_id)
            if resolved:
                request.access_token = resolved
                logger.info("Populated access_token from secret_config_id for clone auth.")

        docker_svc = get_docker_service()
        container = await docker_svc.create_container(
            request
            if isinstance(request, GHRepoRequest)
            else GHRepoRequest(**request.model_dump()),
            tenant_id=user.effective_tenant_id,
        )
        return container.model_dump()
    except HTTPException:
        raise
    except MemoryBudgetError as e:
        # Raised by the create path only for a NEW workspace: a reused one keeps its
        # pod and its hints are ignored (py2rust gap #4 full).
        raise memory_budget_refusal(e) from e
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as e:
        logger.error("Failed to create container: %s", e, exc_info=True)
        raise InternalServerException(
            message=f"Failed to create container: {e}",
            error_code="CONTAINER_CREATE_ERROR",
        ) from e


@router.delete(
    "/all",
    response_model=ContainerOperationResponse,
    operation_id="delete_all_workspaces",
    summary="Delete All Workspaces",
    responses={
        200: {"description": "All containers successfully deleted"},
        400: {"model": Error400Response, "description": "Invalid request parameters"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Resource not found"},
        406: {"model": Error406Response, "description": "Not acceptable"},
        422: {"model": Error422Response, "description": "Validation error"},
        429: {"model": Error429Response, "description": "Too many requests"},
        500: {"model": Error500Response, "description": "Error during bulk deletion"},
    },
)
async def delete_all_containers(
    tenant_id: str = Depends(require_tenant),
) -> ContainerOperationResponse:
    """Delete all containers for the current tenant."""
    try:
        container_manager = get_manager()
        containers = container_manager.list_containers(tenant_id=tenant_id)
        if not containers:
            return ContainerOperationResponse(
                message="No containers to delete",
                operation=ContainerOperation.DELETE,
                container_name="all",
                request_id=str(uuid7()),
            )
        removed_containers = []
        failed_containers = []
        for container in containers:
            try:
                result = await container_manager.remove_container(
                    container.container_name, force=True
                )
                if result["status"] == "success":
                    removed_containers.append(container.container_name)
                else:
                    failed_containers.append(
                        {
                            "name": container.container_name,
                            "error": result.get("message", "Unknown error"),
                        }
                    )
            except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as e:
                failed_containers.append({"name": container.container_name, "error": str(e)})
                logger.error("Failed to remove container %s: %s", container.container_name, e)
        message_parts = []
        if removed_containers:
            message_parts.append(
                f"Successfully removed {len(removed_containers)} container(s): {', '.join(removed_containers)}"
            )
        if failed_containers:
            message_parts.append(f"Failed to remove {len(failed_containers)} container(s)")
        if not message_parts:
            message = "No containers were removed"
        else:
            message = ". ".join(message_parts)
        if failed_containers and (not removed_containers):
            raise internal_error(f"Bulk deletion failed. {message}")
        return ContainerOperationResponse(
            message=message,
            operation=ContainerOperation.DELETE,
            container_name="all",
            request_id=str(uuid7()),
            details=(
                {
                    "removed": removed_containers,
                    "failed": failed_containers,
                    "total_processed": len(containers),
                }
                if removed_containers or failed_containers
                else None
            ),
        )
    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as e:
        logger.error("Unexpected error during bulk container deletion: %s", e)
        raise InternalServerException(message=f"Failed to delete containers: {str(e)}") from e


@router.delete(
    "/{container_name}",
    response_model=ContainerOperationResponse,
    operation_id="delete_workspace",
    summary="Remove Workspace",
    responses={
        200: {"description": "Container successfully removed"},
        400: {"model": Error400Response, "description": "Invalid request parameters"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Container not found"},
        406: {"model": Error406Response, "description": "Not acceptable"},
        422: {"model": Error422Response, "description": "Validation error"},
        429: {"model": Error429Response, "description": "Too many requests"},
        500: {"model": Error500Response, "description": "Error removing container"},
    },
)
async def remove_container(
    user: User = Depends(get_current_user_with_tenant),
    container_name: str = Path(..., description="Container name to remove"),
    force: bool = Query(False, description="Force removal even if container is running"),
) -> ContainerOperationResponse:
    """Remove a container and clean up its resources."""
    try:
        container_manager = get_manager()

        tenant_id = None if user.is_admin() else user.effective_tenant_id
        try:
            container = container_manager.get_container(container_name, tenant_id=tenant_id)
        except ValueError:
            container = None

        if not container:
            raise not_found("Container", container_name)

        result = await container_manager.remove_container(
            container_name, force=force, tenant_id=tenant_id
        )
        if result["status"] == "success":
            # Surface the graph-deletion outcome. `remove_container` already
            # computes it (and `_delete_graph` distinguishes "deleted N nodes"
            # from "already absent" from "could not delete — will be orphaned"),
            # but until now it was dropped on the floor here, so the ONLY place
            # that evidence existed was the pod log. That is exactly backwards
            # for the failure this route had: the pre-#89 teardown reported a
            # 200 while the graph survived, and a caller had no way to tell.
            # `graph_deletion` is absent from `result` only when there was no
            # graph identity to delete, so `details` stays None in that case
            # rather than asserting an empty deletion happened.
            graph_deletion = result.get("graph_deletion")
            return ContainerOperationResponse(
                message=f"Container '{container_name}' successfully removed",
                operation=ContainerOperation.DELETE,
                container_name=container_name,
                request_id=str(uuid7()),
                details={"graph_deletion": graph_deletion} if graph_deletion else None,
            )
        raise internal_error(f"Failed to remove container '{container_name}': {result['message']}")
    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as e:
        raise internal_error(f"Failed to remove container: {str(e)}") from e

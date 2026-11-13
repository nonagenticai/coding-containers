"""Container read routes: list / get-by-name / get-status / GET-all 405 stub."""

from __future__ import annotations

import logging
from uuid import uuid7

import docker.errors
from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from coding_client.auth.models import User
from coding_client.auth.tenant import (
    get_current_user_with_tenant,
    require_tenant,
)
from coding_client.exceptions import InternalServerException
from coding_client.models.responses import (
    bad_request,
    error_response,
    internal_error,
    not_found,
)
from src.models.container_api_models import (
    ContainerDetailResponse,
    ContainerListResponse,
    ContainerStatusResponse,
)
from src.models.container_models import ContainerInfo
from src.models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error405Response,
    Error406Response,
    Error422Response,
    Error429Response,
    Error500Response,
)

from ._common import get_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/",
    response_model=ContainerListResponse,
    operation_id="list_workspaces",
    summary="List All Workspaces",
    responses={
        200: {"description": "Successfully retrieved container list"},
        400: {"model": Error400Response, "description": "Invalid request parameters"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Resource not found"},
        406: {"model": Error406Response, "description": "Not acceptable"},
        422: {"model": Error422Response, "description": "Validation error"},
        429: {"model": Error429Response, "description": "Too many requests"},
        500: {"model": Error500Response, "description": "Error retrieving containers"},
    },
)
async def list_containers(
    status_filter: str | None = Query(None, description="Filter by container status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum containers to return"),
    user: User = Depends(get_current_user_with_tenant),
) -> ContainerListResponse:
    """List all containers with optional status filtering."""
    try:
        container_manager = get_manager()

        is_admin = user.is_admin()
        tenant_id = None if is_admin else user.effective_tenant_id
        if is_admin:
            logger.info(
                "Admin bypass: user '%s' listing all containers without tenant restriction",
                user.preferred_username or user.sub,
            )
        containers = container_manager.list_containers(tenant_id=tenant_id)

        if status_filter is not None:
            if not status_filter.strip():
                raise bad_request("status_filter cannot be empty string")
            valid_statuses = ["running", "stopped", "error", "creating", "removing"]
            if status_filter.lower() not in valid_statuses:
                raise bad_request(
                    f"Invalid status_filter: '{status_filter}'. Valid values are: {', '.join(valid_statuses)}"
                )
            containers = [c for c in containers if c.status.value.lower() == status_filter.lower()]
        containers = containers[:limit]
        safe_containers = [
            ContainerInfo(
                **{k: v for k, v in container.model_dump().items() if k != "github_access_token"}
            )
            for container in containers
        ]
        return ContainerListResponse(
            message=f"Found {len(containers)} containers",
            containers=safe_containers,
            total=len(containers),
            request_id=str(uuid7()),
        )
    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError, KeyError) as e:
        logger.error("Failed to list containers: %s", e)
        raise InternalServerException(
            message=f"Failed to list containers: {str(e)}",
            error_code="CONTAINER_LIST_ERROR",
        ) from e


@router.get(
    "/all",
    operation_id="get_all_not_allowed",
    summary="Method Not Allowed",
    description="GET method not supported for bulk operations",
    responses={
        405: {
            "model": Error405Response,
            "description": "Method not allowed - GET is not supported for bulk operations",
        }
    },
    include_in_schema=False,
)
async def get_all_containers_not_allowed(
    tenant_id: str = Depends(require_tenant),
) -> None:
    """Handle GET requests to /all with proper 405 response."""
    raise error_response(
        status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
        message="Method GET not allowed for this endpoint",
        error_code="METHOD_NOT_ALLOWED",
    )


@router.get(
    "/{container_name}",
    response_model=ContainerDetailResponse,
    operation_id="get_workspace",
    summary="Get Workspace Details",
    responses={
        200: {"description": "Successfully retrieved container details"},
        400: {"model": Error400Response, "description": "Invalid request parameters"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Container not found"},
        406: {"model": Error406Response, "description": "Not acceptable"},
        422: {"model": Error422Response, "description": "Validation error"},
        429: {"model": Error429Response, "description": "Too many requests"},
        500: {"model": Error500Response, "description": "Error retrieving container"},
    },
)
async def get_container_details(
    container_name: str = Path(..., description="Container name"),
    user: User = Depends(get_current_user_with_tenant),
) -> ContainerDetailResponse:
    """Get detailed information about a specific container."""
    try:
        container_manager = get_manager()

        is_admin = user.is_admin()
        tenant_id = None if is_admin else user.effective_tenant_id
        if is_admin:
            logger.info(
                "Admin bypass: user '%s' accessing container '%s' without tenant restriction",
                user.preferred_username or user.sub,
                container_name,
            )
        try:
            container = container_manager.get_container(container_name, tenant_id=tenant_id)
        except ValueError:
            container = None

        if not container:
            raise not_found("Container", container_name)

        safe_container = ContainerInfo(
            **{k: v for k, v in container.model_dump().items() if k != "github_access_token"}
        )
        return ContainerDetailResponse(
            message=f"Container '{container_name}' details retrieved",
            container=safe_container,
            request_id=str(uuid7()),
        )
    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError, KeyError) as e:
        raise internal_error(f"Failed to get container details: {str(e)}") from e


@router.get(
    "/{container_name}/status",
    response_model=ContainerStatusResponse,
    operation_id="get_workspace_status",
    summary="Get Workspace Status",
    responses={
        200: {"description": "Successfully retrieved container status"},
        400: {"model": Error400Response, "description": "Invalid request parameters"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Container not found"},
        406: {"model": Error406Response, "description": "Not acceptable"},
        422: {"model": Error422Response, "description": "Validation error"},
        429: {"model": Error429Response, "description": "Too many requests"},
        500: {"model": Error500Response, "description": "Error retrieving container status"},
    },
)
async def get_container_status(
    container_name: str = Path(..., description="Container name"),
    user: User = Depends(get_current_user_with_tenant),
) -> ContainerStatusResponse:
    """Get real-time status for a container."""
    try:
        container_manager = get_manager()

        # ⛔ Do NOT re-add a ``tenant_id=None`` retry here. One lived at this spot
        # from #66 until it was measured in production, and the reasoning that
        # justified it — "/status is a non-sensitive liveness read" — was wrong
        # about WHAT the leak costs. What /status discloses is not the body, it
        # is EXISTENCE AND READINESS of a name: measured 2026-09-17, a caller in
        # tenant …0003 got 200 OK for ``aa-delivery-chat-bridge`` (Deployment
        # label ``tenant-id: …0001``), then 404 seven seconds later on /exec,
        # because every sensitive route still gated. So the fallback did not buy
        # the readiness poll a working workspace; it bought it a FALSE "ready",
        # after which the conveyor adopted a name-colliding foreign workspace
        # instead of provisioning its own and the card finished with a null exit
        # code. A gate that opens the cheapest door and shuts every expensive one
        # converts an honest 404 into silent cross-tenant aliasing.
        #
        # The legitimate cross-tenant reader is served by the SAME mechanism the
        # rest of this module uses and nothing new: ``user.is_admin()`` below
        # widens to ``tenant_id=None``. An internal readiness poller that must
        # see every tenant authenticates as admin. See ``get_container_details``
        # above — this handler is now identical to it in scoping.
        #
        # ⛔ This block used to end "one that must not, provisions its own
        # workspace on the 404". Do NOT put that back: on a NAME COLLISION it
        # prescribes a cross-tenant WRITE, which is worse than the read it
        # replaces. ``POST /`` is not name-gated, and the provisioner's 409 path
        # reconciles the EXISTING Deployment in place with the caller's body —
        # re-stamping ``tenant-id`` to the caller and rolling a fresh pod. So a
        # caller that "provisions its own workspace" on a 404 caused by a
        # collision takes the other tenant's workspace over and destroys its
        # contents, silently. That takeover is now refused at the provisioner
        # (see ``_create_deployment``'s 409 branch); the correct response to a
        # 404 here is a DIFFERENT name, or admin credentials if the caller
        # genuinely needs fleet-wide reach.
        tenant_id = None if user.is_admin() else user.effective_tenant_id
        if user.is_admin():
            logger.info(
                "Admin bypass: user '%s' reading status of container '%s' "
                "without tenant restriction",
                user.preferred_username or user.sub,
                container_name,
            )
        try:
            container = container_manager.get_container(container_name, tenant_id=tenant_id)
        except ValueError:
            container = None

        if not container:
            raise not_found("Container", container_name)
        container_status = await container_manager.get_container_status(container_name)
        return ContainerStatusResponse(
            message=f"Status for container '{container_name}'",
            container_status=container_status,
            request_id=str(uuid7()),
        )
    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as e:
        raise internal_error(f"Failed to get container status: {str(e)}") from e

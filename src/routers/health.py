"""Health endpoints for container health monitoring.

The gateway proxies /api/health/container/{name} and /api/health/containers/unhealthy
here to check individual container health and list unhealthy containers.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from coding_client.auth.tenant import require_tenant
from coding_client.models.responses import not_found
from src.core.container_manager import get_container_manager
from src.models.container_models import ContainerStatus

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ContainerHealthResponse(BaseModel):
    """Health status for a single container."""

    name: str
    status: str
    healthy: bool


class UnhealthyContainersResponse(BaseModel):
    """List of containers with ERROR status."""

    unhealthy: list[ContainerHealthResponse]
    count: int


class ContainerStatisticsResponse(BaseModel):
    """Statistics for a single container."""

    name: str
    status: str
    language: str | None = None
    codebase_path: str | None = None
    created_at: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/container/{name}",
    response_model=ContainerHealthResponse,
    operation_id="get_container_health",
    summary="Get container health",
)
async def get_container_health(
    name: str,
    tenant_id: str = Depends(require_tenant),
):
    """Return health info for a single container."""
    manager = get_container_manager()
    try:
        container = manager.get_container(name, tenant_id=tenant_id)
    except ValueError:
        raise not_found("Container", name)

    if container is None:
        raise not_found("Container", name)

    return ContainerHealthResponse(
        name=container.container_name,
        status=str(container.status),
        healthy=container.status != ContainerStatus.ERROR,
    )


@router.get(
    "/containers/unhealthy",
    response_model=UnhealthyContainersResponse,
    operation_id="list_unhealthy_containers",
    summary="List unhealthy containers",
)
async def list_unhealthy_containers(
    tenant_id: str = Depends(require_tenant),
):
    """Return all containers with ERROR status."""
    manager = get_container_manager()
    containers = manager.list_containers(tenant_id=tenant_id)
    unhealthy = [
        ContainerHealthResponse(
            name=c.container_name,
            status=str(c.status),
            healthy=False,
        )
        for c in containers
        if c.status == ContainerStatus.ERROR
    ]
    return UnhealthyContainersResponse(unhealthy=unhealthy, count=len(unhealthy))


@router.get(
    "/container/{name}/ready",
    response_model=ContainerHealthResponse,
    operation_id="check_container_readiness",
    summary="Check container readiness",
)
async def container_readiness(
    name: str,
    tenant_id: str = Depends(require_tenant),
):
    """Check if a container is ready to serve requests."""
    manager = get_container_manager()
    try:
        container = manager.get_container(name, tenant_id=tenant_id)
    except ValueError:
        raise not_found("Container", name)

    if container is None:
        raise not_found("Container", name)

    ready = container.status == ContainerStatus.RUNNING
    if not ready:
        raise HTTPException(
            status_code=503,
            detail=f"Container '{name}' is not ready (status: {container.status})",
        )
    return ContainerHealthResponse(
        name=container.container_name,
        status=str(container.status),
        healthy=True,
    )


@router.get(
    "/container/{name}/live",
    response_model=ContainerHealthResponse,
    operation_id="check_container_liveness",
    summary="Check container liveness",
)
async def container_liveness(
    name: str,
    tenant_id: str = Depends(require_tenant),
):
    """Check if a container is alive (exists and not in ERROR state)."""
    manager = get_container_manager()
    try:
        container = manager.get_container(name, tenant_id=tenant_id)
    except ValueError:
        raise not_found("Container", name)

    if container is None:
        raise not_found("Container", name)

    alive = container.status not in (ContainerStatus.STOPPED, ContainerStatus.ERROR)
    if not alive:
        raise HTTPException(
            status_code=503,
            detail=f"Container '{name}' is not alive (status: {container.status})",
        )
    return ContainerHealthResponse(
        name=container.container_name,
        status=str(container.status),
        healthy=True,
    )


@router.get(
    "/container/{name}/statistics",
    response_model=ContainerStatisticsResponse,
    operation_id="get_container_statistics",
    summary="Get container statistics",
)
async def container_statistics(
    name: str,
    tenant_id: str = Depends(require_tenant),
):
    """Return statistics and metadata for a container."""
    manager = get_container_manager()
    try:
        container = manager.get_container(name, tenant_id=tenant_id)
    except ValueError:
        raise not_found("Container", name)

    if container is None:
        raise not_found("Container", name)

    return ContainerStatisticsResponse(
        name=container.container_name,
        status=str(container.status),
        language=getattr(container, "language", None),
        codebase_path=getattr(container, "codebase_path", None),
        created_at=str(getattr(container, "created_at", None))
        if hasattr(container, "created_at")
        else None,
    )

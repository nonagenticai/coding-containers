"""
Container Dependencies Management Endpoints

Handles dependency installation and management for containerized repositories.
"""

from __future__ import annotations

import logging

import docker.errors
from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field

from coding_client.auth.tenant import require_tenant
from coding_client.models.responses import (
    internal_error,
    not_found,
)
from src.core.container_installer import ContainerInstaller, SystemDependencyError
from src.models.common_api_models import ResponseStatus
from src.models.error_models import (
    Error400Response,
    Error404Response,
    Error500Response,
)
from src.services.docker_service import get_docker_service

logger = logging.getLogger(__name__)
router = APIRouter()


class DependencyInstallRequest(BaseModel):
    """Request model for dependency installation."""

    workspace_path: str = Field(
        default="/app/data/repo",
        description="Path to repository in container. Common paths: /app/data/repo (default), /workspace, /app/test_workspace",
    )
    force_reinstall: bool = Field(
        default=False, description="Force reinstallation of all dependencies"
    )


class DependencyInstallResponse(BaseModel):
    """Response model for dependency installation."""

    status: ResponseStatus
    message: str
    success: bool
    installed_packages: int = Field(description="Number of packages installed")
    package_managers_used: list[str] = Field(description="Package managers used")
    errors: list[str] = Field(default_factory=list, description="Installation errors")
    installation_time: float = Field(description="Time taken in seconds")
    guidance: str = Field(default="", description="Guidance for fixing errors")


class DetectedDependencyFile(BaseModel):
    """One dependency file found in the workspace."""

    path: str = Field(..., description="Absolute path in the workspace")
    type: str = Field(..., description="Ecosystem: python | javascript")
    manager: str = Field(..., description="Package manager implied by the file")
    file_name: str = Field(..., description="Base name of the file")


class DependencyStatusResponse(BaseModel):
    """Response model for dependency status analysis."""

    status: ResponseStatus
    message: str
    project_type: str = Field(..., description="Detected project type")
    # These are the shapes ``ContainerInstaller.analyze_project`` produces. They
    # used to be declared ``list[str]`` / ``dict``, so every real analysis with a
    # plan 500'd on response validation (``installation_plan: Input should be a
    # valid dictionary``), and a test feeding the route a hand-made dict hid it.
    detected_files: list[DetectedDependencyFile] = Field(
        default_factory=list, description="Dependency-related files found in the repository"
    )
    installation_plan: list[str] = Field(
        default_factory=list, description="Planned package manager actions, one per file"
    )
    total_dependency_files: int = Field(..., ge=0, description="Total dependency files detected")


@router.post(
    "/{container_name}/install-dependencies",
    response_model=DependencyInstallResponse,
    operation_id="install_workspace_dependencies",
    summary="Install Dependencies",
    description="""
    Install dependencies for a containerized repository.

    This endpoint analyzes the repository's dependency files (requirements.txt,
    package.json, etc.) and installs all required packages.

    **Features:**
    - Automatic detection of Python and JavaScript dependencies
    - Support for multiple package managers (pip, poetry, npm, yarn, etc.)
    - Clear error messages for missing system dependencies
    - Detailed guidance on fixing installation issues

    **Workspace Path:**
    The `workspace_path` parameter specifies where the repository is located inside the container.
    Common paths based on container configuration:
    - `/app/data/repo` - Default path where repositories are cloned
    - `/workspace` - Working directory for test execution containers
    - `/app/test_workspace` - Alternative test workspace location
    - `/app/analysis_workspace` - For code analysis operations

    To find the correct path for your container:
    1. Check the container's volume mappings
    2. Use `docker exec <container_name> ls /app/data` to explore the structure
    3. Look for where your source code is mounted

    **Example:**
    ```json
    {
        "workspace_path": "/app/data/repo",
        "force_reinstall": false
    }
    ```

    **Common Issues Resolved:**
    - Missing system libraries (libxml2, libssl, etc.)
    - Build tools not found (gcc, cmake, etc.)
    - Python headers missing for C extensions
    - Node.js/npm not installed
    """,
    responses={
        200: {"description": "Dependencies installed successfully"},
        400: {
            "model": Error400Response,
            "description": "Invalid request parameters",
        },
        404: {
            "model": Error404Response,
            "description": "Container not found",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error",
        },
    },
)
async def install_container_dependencies(
    container_name: str = Path(..., description="Container name or ID"),
    request: DependencyInstallRequest = DependencyInstallRequest(),
    tenant_id: str = Depends(require_tenant),
) -> DependencyInstallResponse:
    """Install dependencies in a containerized repository."""
    try:
        # Verify container exists
        docker_service = get_docker_service()

        # Check if container exists
        container = await docker_service.find_container_by_name(container_name, tenant_id=tenant_id)
        if not container:
            raise not_found("Container", container_name)

        # Create installer and run installation
        installer = ContainerInstaller(container_name)

        logger.info(
            f"Starting dependency installation for container {container_name} at {request.workspace_path}"
        )

        result = await installer.install_dependencies_in_container(
            container_name, request.workspace_path
        )

        # Prepare response
        response = DependencyInstallResponse(
            status=ResponseStatus.SUCCESS if result.success else ResponseStatus.ERROR,
            message=(
                f"Successfully installed {len(result.installed_packages)} packages"
                if result.success
                else "Dependency installation completed with errors"
            ),
            success=result.success,
            installed_packages=len(result.installed_packages),
            package_managers_used=result.package_managers_used,
            errors=result.errors[:10],  # Limit to first 10 errors
            installation_time=result.installation_time,
            guidance="",
        )

        # Add guidance if there were errors
        if result.errors and not result.success:
            # Extract guidance from errors
            for error in result.errors:
                if "Guidance:" in error:
                    response.guidance = error.split("Guidance:")[-1].strip()
                    break

        return response

    except HTTPException:
        raise
    except SystemDependencyError as e:
        # Provide specific guidance for system dependency errors
        return DependencyInstallResponse(
            status=ResponseStatus.ERROR,
            message=str(e),
            success=False,
            installed_packages=0,
            package_managers_used=[],
            errors=[str(e)],
            installation_time=0.0,
            guidance=e.guidance,
        )
    except (docker.errors.DockerException, RuntimeError, OSError, ValueError) as e:
        logger.error(
            f"Failed to install dependencies in container {container_name}: {e}",
            exc_info=True,
        )
        raise internal_error(f"Failed to install dependencies: {str(e)}") from e


@router.get(
    "/{container_name}/dependency-status",
    response_model=DependencyStatusResponse,
    operation_id="get_workspace_dependency_status",
    summary="Get Dependency Status",
    description="""
    Check the dependency installation status for a container.

    Returns information about:
    - Detected dependency files
    - Installed package managers
    - Potential missing dependencies
    - Recommendations for installation
    """,
    responses={
        200: {"description": "Dependency status retrieved"},
        404: {
            "model": Error404Response,
            "description": "Container not found",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error",
        },
    },
)
async def get_dependency_status(
    container_name: str = Path(..., description="Container name or ID"),
    tenant_id: str = Depends(require_tenant),
) -> DependencyStatusResponse:
    """Get dependency status for a container."""
    try:
        # Verify container exists
        docker_service = get_docker_service()

        container = await docker_service.find_container_by_name(container_name, tenant_id=tenant_id)
        if not container:
            raise not_found("Container", container_name)

        # Create installer and analyze project
        installer = ContainerInstaller(container_name)

        analysis = await installer.analyze_project("/app/data/repo")

        return DependencyStatusResponse(
            status=ResponseStatus.SUCCESS,
            message="Dependency analysis completed",
            project_type=analysis["project_type"],
            detected_files=analysis["detected_files"],
            installation_plan=analysis["installation_plan"],
            total_dependency_files=analysis["total_files"],
        )

    except HTTPException:
        raise
    except (docker.errors.DockerException, RuntimeError, OSError, ValueError, KeyError) as e:
        logger.error(
            f"Failed to get dependency status for container {container_name}: {e}",
            exc_info=True,
        )
        raise internal_error(f"Failed to get dependency status: {str(e)}") from e


# Vulture whitelist for FastAPI route handlers
_vulture_whitelist = [
    "install_container_dependencies",
    "get_dependency_status",
]

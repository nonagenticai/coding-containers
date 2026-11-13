"""Container filesystem routes: mkdir + touch (create-only operations)."""

from __future__ import annotations

import logging
import os
import shlex
from uuid import uuid7

import docker.errors
from fastapi import APIRouter, Body, Depends, HTTPException, Path

from coding_client.auth.models import User
from coding_client.auth.tenant import get_current_user_with_tenant
from coding_client.models.responses import (
    bad_request,
    internal_error,
    not_found,
)
from src.models.container_api_models import (
    CreateDirectoryRequest,
    CreateDirectoryResponse,
    CreateFileRequest,
    CreateFileResponse,
)
from src.models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error406Response,
    Error409Response,
    Error422Response,
    Error429Response,
    Error500Response,
)

from ._common import (
    get_manager,
    refuse_write_into_secondary_repos,
)
from .files_manage import _repository_root

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/{container_name}/mkdir",
    response_model=CreateDirectoryResponse,
    operation_id="create_workspace_directory",
    summary="Create Directory in Workspace",
    responses={
        200: {"description": "Directory created successfully"},
        409: {
            "model": Error409Response,
            "description": "SECONDARY_REPO_READ_ONLY: the path is under /workspace/.repos/ (read-only sources)",
        },
        400: {"model": Error400Response, "description": "Invalid request parameters"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Container not found"},
        406: {"model": Error406Response, "description": "Not acceptable"},
        422: {"model": Error422Response, "description": "Validation error"},
        429: {"model": Error429Response, "description": "Too many requests"},
        500: {"model": Error500Response, "description": "Error creating directory"},
    },
)
async def create_directory(
    container_name: str = Path(..., description="Container name"),
    request: CreateDirectoryRequest = Body(...),
    user: User = Depends(get_current_user_with_tenant),
) -> CreateDirectoryResponse:
    """Create a directory inside a container's file system."""
    try:
        container_manager = get_manager()

        tenant_id = None if user.is_admin() else user.effective_tenant_id
        try:
            container = container_manager.get_container(container_name, tenant_id=tenant_id)
        except ValueError:
            container = None

        if not container:
            raise not_found("Container", container_name)

        if container.status.value.lower() != "running":
            raise bad_request(
                f"Container '{container_name}' is not running (current status: {container.status.value})"
            )

        dir_path = request.path.strip()
        if not dir_path:
            raise bad_request("Invalid directory path: path must not be empty")

        normalized = os.path.normpath(dir_path)

        if any(
            [
                ".." in dir_path,
                normalized.startswith(".."),
                "\x00" in dir_path,
                "\n" in dir_path,
                "\r" in dir_path,
            ]
        ):
            raise bad_request("Invalid path: potential path traversal")

        if any(c in dir_path for c in [";", "|", "&", "`", "$"]):
            raise bad_request("Invalid path: shell metacharacters not allowed")
        refuse_write_into_secondary_repos(
            dir_path, workspace_root=_repository_root(container), container_name=container_name
        )

        command = f"mkdir -p {shlex.quote(dir_path)}"
        result = await container_manager.execute_command(
            container_name, command, tenant_id=tenant_id
        )

        if result.get("exit_code", 1) != 0:
            error_msg = result.get("stderr", "Unknown error")
            raise internal_error(f"Failed to create directory: {error_msg}")

        logger.info(
            "Successfully created directory '%s' in container '%s'",
            dir_path,
            container_name,
        )

        return CreateDirectoryResponse(
            message="Directory created successfully",
            path=dir_path,
            container_name=container_name,
            request_id=str(uuid7()),
        )

    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as e:
        logger.error("Failed to create directory in container '%s': %s", container_name, e)
        raise internal_error(f"Failed to create directory: {str(e)}") from e


@router.post(
    "/{container_name}/touch",
    response_model=CreateFileResponse,
    operation_id="create_workspace_file",
    summary="Create Empty File in Workspace",
    responses={
        200: {"description": "File created successfully"},
        409: {
            "model": Error409Response,
            "description": "SECONDARY_REPO_READ_ONLY: the path is under /workspace/.repos/ (read-only sources)",
        },
        400: {"model": Error400Response, "description": "Invalid request parameters"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Container not found"},
        406: {"model": Error406Response, "description": "Not acceptable"},
        422: {"model": Error422Response, "description": "Validation error"},
        429: {"model": Error429Response, "description": "Too many requests"},
        500: {"model": Error500Response, "description": "Error creating file"},
    },
)
async def create_file(
    container_name: str = Path(..., description="Container name"),
    request: CreateFileRequest = Body(...),
    user: User = Depends(get_current_user_with_tenant),
) -> CreateFileResponse:
    """Create an empty file inside a container's file system."""
    try:
        container_manager = get_manager()

        tenant_id = None if user.is_admin() else user.effective_tenant_id
        try:
            container = container_manager.get_container(container_name, tenant_id=tenant_id)
        except ValueError:
            container = None

        if not container:
            raise not_found("Container", container_name)

        if container.status.value.lower() != "running":
            raise bad_request(
                f"Container '{container_name}' is not running (current status: {container.status.value})"
            )

        dir_path = request.path.strip()
        filename = request.filename.strip()
        extension = request.extension.strip()

        if not dir_path or not filename or not extension:
            raise bad_request("Path, filename, and extension must not be empty")

        normalized = os.path.normpath(dir_path)

        for path_component in [dir_path, filename, extension]:
            if any(
                [
                    ".." in path_component,
                    "\x00" in path_component,
                    "\n" in path_component,
                    "\r" in path_component,
                ]
            ):
                raise bad_request("Invalid path: potential path traversal")

        if normalized.startswith(".."):
            raise bad_request("Invalid path: potential path traversal")

        for path_component in [dir_path, filename, extension]:
            if any(c in path_component for c in [";", "|", "&", "`", "$", "(", ")", "<", ">"]):
                raise bad_request("Invalid path: shell metacharacters not allowed")

        if extension.startswith("."):
            extension = extension[1:]

        full_path = f"{dir_path.rstrip('/')}/{filename}.{extension}"
        refuse_write_into_secondary_repos(
            full_path, workspace_root=_repository_root(container), container_name=container_name
        )

        command = f"mkdir -p {shlex.quote(dir_path)} && touch {shlex.quote(full_path)}"
        result = await container_manager.execute_command(
            container_name, command, tenant_id=tenant_id
        )

        if result.get("exit_code", 1) != 0:
            error_msg = result.get("stderr", "Unknown error")
            raise internal_error(f"Failed to create file: {error_msg}")

        logger.info(
            "Successfully created file '%s' in container '%s'",
            full_path,
            container_name,
        )

        return CreateFileResponse(
            message="File created successfully",
            path=full_path,
            container_name=container_name,
            request_id=str(uuid7()),
        )

    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as e:
        logger.error("Failed to create file in container '%s': %s", container_name, e)
        raise internal_error(f"Failed to create file: {str(e)}") from e

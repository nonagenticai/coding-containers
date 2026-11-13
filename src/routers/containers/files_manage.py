"""Container filesystem routes for moving and deleting repository paths."""

from __future__ import annotations

import logging
import shlex
from uuid import uuid7

import docker.errors
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query

from coding_client.auth.models import User
from coding_client.auth.tenant import get_current_user_with_tenant
from coding_client.models.responses import (
    bad_request,
    conflict,
    internal_error,
    not_found,
)
from src.models.container_api_models import (
    DeletePathResponse,
    MovePathRequest,
    MovePathResponse,
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
    _canonical_workspace_path,
    _normalize_workspace_path,
    get_manager,
    refuse_write_into_secondary_repos,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_PATH_NOT_FOUND = 40
_PATH_OUTSIDE_WORKSPACE = 41
_DESTINATION_PARENT_NOT_FOUND = 42
_DESTINATION_EXISTS = 43
_DIRECTORY_NOT_EMPTY = 44


def _parent_containment_guard(variable: str, *, failure_code: int) -> str:
    """Build a shell guard that resolves a path's parent below /workspace."""
    parent_variable = f"{variable}_parent"
    return (
        f'{parent_variable}=$(realpath "$(dirname -- "${variable}")") '
        f"|| exit {failure_code}; "
        f'case "${parent_variable}" in "$workspace"|"$workspace"/*) ;; '
        f"*) exit {_PATH_OUTSIDE_WORKSPACE} ;; esac"
    )


def _move_command(
    source_path: str,
    destination_path: str,
    *,
    overwrite: bool,
    workspace_root: str = "/workspace",
    move_binary: str = "mv",
) -> str:
    source = shlex.quote(source_path)
    destination = shlex.quote(destination_path)
    workspace = shlex.quote(workspace_root)
    move = shlex.quote(move_binary)
    destination_check = (
        ""
        if overwrite
        else (
            f'if [ -e "$destination" ] || [ -L "$destination" ]; '
            f"then exit {_DESTINATION_EXISTS}; fi; "
        )
    )
    force_flag = "-f " if overwrite else ""
    return "; ".join(
        [
            f"workspace=$(realpath {workspace}) || exit 1",
            f"source={source}",
            f"destination={destination}",
            (f'if [ ! -e "$source" ] && [ ! -L "$source" ]; then exit {_PATH_NOT_FOUND}; fi'),
            _parent_containment_guard("source", failure_code=_PATH_OUTSIDE_WORKSPACE),
            _parent_containment_guard("destination", failure_code=_DESTINATION_PARENT_NOT_FOUND),
            f'{destination_check}{move} {force_flag}-T -- "$source" "$destination"',
        ]
    )


def _delete_command(path: str, *, recursive: bool, workspace_root: str = "/workspace") -> str:
    target = shlex.quote(path)
    workspace = shlex.quote(workspace_root)
    if recursive:
        delete_operation = 'rm -rf -- "$target"'
    else:
        delete_operation = (
            'if [ -d "$target" ] && [ ! -L "$target" ]; then '
            'if [ -n "$(find "$target" -mindepth 1 -maxdepth 1 -print -quit)" ]; '
            f"then exit {_DIRECTORY_NOT_EMPTY}; fi; "
            'rmdir -- "$target"; else rm -- "$target"; fi'
        )

    return "; ".join(
        [
            f"workspace=$(realpath {workspace}) || exit 1",
            f"target={target}",
            (f'if [ ! -e "$target" ] && [ ! -L "$target" ]; then exit {_PATH_NOT_FOUND}; fi'),
            _parent_containment_guard("target", failure_code=_PATH_OUTSIDE_WORKSPACE),
            delete_operation,
        ]
    )


def _tenant_id_for(user: User) -> str | None:
    return None if user.is_admin() else user.effective_tenant_id


def _get_running_container(container_name: str, user: User):
    container_manager = get_manager()
    tenant_id = _tenant_id_for(user)
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
    return container_manager, tenant_id, container


def _repository_root(container) -> str:
    """Get the repo root registered for either a Docker container or K8s pod."""
    codebase_path = getattr(container, "codebase_path", None)
    if isinstance(codebase_path, str) and codebase_path.strip():
        normalized = codebase_path.strip()
        if normalized.startswith("/") and normalized != "/":
            return normalized
    return "/workspace"


@router.post(
    "/{container_name}/move",
    response_model=MovePathResponse,
    operation_id="move_workspace_path",
    summary="Move or Rename File or Directory in Workspace",
    responses={
        200: {"description": "File or directory moved successfully"},
        400: {"model": Error400Response, "description": "Invalid path or container state"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Container or source path not found"},
        406: {"model": Error406Response, "description": "Not acceptable"},
        409: {
            "model": Error409Response,
            "description": "Destination already exists, or SECONDARY_REPO_READ_ONLY: the path is under /workspace/.repos/ (read-only sources)",
        },
        422: {"model": Error422Response, "description": "Validation error"},
        429: {"model": Error429Response, "description": "Too many requests"},
        500: {"model": Error500Response, "description": "Error moving path"},
    },
)
async def move_path(
    container_name: str = Path(..., description="Container name"),
    request: MovePathRequest = Body(...),
    user: User = Depends(get_current_user_with_tenant),
) -> MovePathResponse:
    """Move or rename a file, symlink, or complete directory tree."""
    try:
        container_manager, tenant_id, container = _get_running_container(container_name, user)
        workspace_root = _repository_root(container)
        try:
            source_path = _normalize_workspace_path(
                request.source_path, workspace_root=workspace_root
            )
            destination_path = _normalize_workspace_path(
                request.destination_path, workspace_root=workspace_root
            )
        except ValueError as exc:
            raise bad_request(f"Invalid path: {exc}") from exc

        for side in (source_path, destination_path):
            refuse_write_into_secondary_repos(
                side, workspace_root=workspace_root, container_name=container_name
            )

        if source_path == destination_path:
            raise bad_request("Source and destination paths must be different")
        if destination_path.startswith(f"{source_path}/"):
            raise bad_request("A path cannot be moved inside itself")

        result = await container_manager.execute_command(
            container_name,
            _move_command(
                source_path,
                destination_path,
                overwrite=request.overwrite,
                workspace_root=workspace_root,
            ),
            tenant_id=tenant_id,
        )
        exit_code = result.get("exit_code", 1)
        canonical_source = _canonical_workspace_path(source_path, workspace_root=workspace_root)
        canonical_destination = _canonical_workspace_path(
            destination_path, workspace_root=workspace_root
        )
        if exit_code == _PATH_NOT_FOUND:
            raise not_found("Path", canonical_source)
        if exit_code == _DESTINATION_PARENT_NOT_FOUND:
            raise not_found("Destination parent", canonical_destination.rsplit("/", 1)[0])
        if exit_code == _DESTINATION_EXISTS:
            raise conflict(
                "Destination path",
                f"Destination '{canonical_destination}' already exists; "
                "set overwrite=true to replace it",
            )
        if exit_code == _PATH_OUTSIDE_WORKSPACE:
            raise bad_request("Invalid path: resolved parent is outside /workspace")
        if exit_code != 0:
            raise internal_error(f"Failed to move path: {result.get('stderr', 'Unknown error')}")

        logger.info(
            "Moved path '%s' to '%s' in container '%s'",
            source_path,
            destination_path,
            container_name,
        )
        return MovePathResponse(
            message="Path moved successfully",
            source_path=canonical_source,
            destination_path=canonical_destination,
            container_name=container_name,
            request_id=str(uuid7()),
        )
    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as exc:
        logger.error("Failed to move path in container '%s': %s", container_name, exc)
        raise internal_error(f"Failed to move path: {exc}") from exc


@router.delete(
    "/{container_name}/files",
    response_model=DeletePathResponse,
    operation_id="delete_workspace_path",
    summary="Delete File or Directory in Workspace",
    responses={
        200: {"description": "File or directory deleted successfully"},
        400: {"model": Error400Response, "description": "Invalid path or container state"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Container or path not found"},
        406: {"model": Error406Response, "description": "Not acceptable"},
        409: {
            "model": Error409Response,
            "description": "Directory is not empty, or SECONDARY_REPO_READ_ONLY: the path is under /workspace/.repos/ (read-only sources)",
        },
        422: {"model": Error422Response, "description": "Validation error"},
        429: {"model": Error429Response, "description": "Too many requests"},
        500: {"model": Error500Response, "description": "Error deleting path"},
    },
)
async def delete_path(
    container_name: str = Path(..., description="Container name"),
    path: str = Query(..., min_length=1, max_length=4096, description="File or directory path"),
    recursive: bool = Query(
        False, description="Required to delete a non-empty directory and all of its contents"
    ),
    user: User = Depends(get_current_user_with_tenant),
) -> DeletePathResponse:
    """Delete a file, symlink, empty directory, or recursively a directory tree."""
    try:
        container_manager, tenant_id, container = _get_running_container(container_name, user)
        workspace_root = _repository_root(container)
        try:
            normalized_path = _normalize_workspace_path(path, workspace_root=workspace_root)
        except ValueError as exc:
            raise bad_request(f"Invalid path: {exc}") from exc
        refuse_write_into_secondary_repos(
            normalized_path, workspace_root=workspace_root, container_name=container_name
        )

        result = await container_manager.execute_command(
            container_name,
            _delete_command(normalized_path, recursive=recursive, workspace_root=workspace_root),
            tenant_id=tenant_id,
        )
        exit_code = result.get("exit_code", 1)
        canonical_path = _canonical_workspace_path(normalized_path, workspace_root=workspace_root)
        if exit_code == _PATH_NOT_FOUND:
            raise not_found("Path", canonical_path)
        if exit_code == _DIRECTORY_NOT_EMPTY:
            raise conflict(
                "Directory",
                f"Directory '{canonical_path}' is not empty; set recursive=true to delete it",
            )
        if exit_code == _PATH_OUTSIDE_WORKSPACE:
            raise bad_request("Invalid path: resolved parent is outside /workspace")
        if exit_code != 0:
            raise internal_error(f"Failed to delete path: {result.get('stderr', 'Unknown error')}")

        logger.info("Deleted path '%s' in container '%s'", normalized_path, container_name)
        return DeletePathResponse(
            message="Path deleted successfully",
            path=canonical_path,
            container_name=container_name,
            recursive=recursive,
            request_id=str(uuid7()),
        )
    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as exc:
        logger.error("Failed to delete path in container '%s': %s", container_name, exc)
        raise internal_error(f"Failed to delete path: {exc}") from exc

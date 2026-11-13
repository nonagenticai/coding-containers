"""POST /api/workspaces/{name}/package-upgrade — typed, per-manager package upgrade.

Unlike ``/exec``, this route does NOT accept raw shell tokens. The body is a
typed request — manager + mode + packages — and the per-manager strategy
in ``services.package_upgrade.strategies`` constructs the command server-side.
This keeps the freeform ``/exec`` whitelist narrow while still letting
``coding-security`` orchestrate package upgrades across every supported
language.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, ConfigDict, Field

from coding_client.auth.tenant import require_tenant
from coding_client.models.package_manager import LanguagePackageManager
from coding_client.models.responses import bad_request, internal_error, not_found
from src.core.container_manager import get_container_manager
from src.services.package_upgrade import (
    AddSpec,
    MajorUpgradeSpec,
    ModifyAction,
    PackageSpec,
    UpgradeMode,
    run_modify,
    run_upgrade,
)
from src.services.package_upgrade.runner import (
    ManagerNotSupportedError,
    PathTraversalError,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class PackageUpgradeRequest(BaseModel):
    manager: LanguagePackageManager = Field(
        ...,
        description="Package manager to dispatch against. Determines both the command "
        "to run and the manifest files diffed before/after.",
    )
    mode: UpgradeMode = Field(
        ...,
        description="``in-range`` keeps the existing constraint and bumps the lockfile. "
        "``cross-major`` rewrites the constraint in the manifest THEN bumps the lock.",
    )
    packages: list[PackageSpec] = Field(
        default_factory=list,
        description="Required when mode=in-range. Ignored when mode=cross-major.",
    )
    major_specs: list[MajorUpgradeSpec] = Field(
        default_factory=list,
        description="Required when mode=cross-major. Ignored when mode=in-range.",
    )
    project_path: str = Field(
        default=".",
        max_length=512,
        description="Path inside the repo root (``/workspace``) where the package "
        "manager should run. Must be relative; ``..`` traversal is rejected.",
    )
    dry_run: bool = Field(
        default=False,
        description="When supported by the manager, runs the upgrade in a no-write mode.",
    )
    timeout_seconds: int = Field(default=300, ge=10, le=1800, description="Per-command timeout.")

    model_config = ConfigDict(extra="forbid")


class PackageUpgradeResponse(BaseModel):
    manager: LanguagePackageManager
    mode: UpgradeMode
    project_path: str
    dry_run: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    commands_executed: list[list[str]]
    manifest_files_before: dict[str, str]
    manifest_files_after: dict[str, str]

    model_config = ConfigDict(extra="forbid")


@router.post(
    "/{container_name}/package-upgrade",
    response_model=PackageUpgradeResponse,
    operation_id="package_upgrade_in_workspace",
    summary="Run a typed package upgrade in a workspace",
    description=(
        "Run an in-range or cross-major package upgrade inside a container using the "
        "correct package manager. Builds the command server-side from typed input — "
        "shell metacharacters in package names / versions are rejected at the request "
        "boundary. Returns pre/post manifest contents so the caller can compute a diff "
        "or open a PR. Companion route on coding-security: "
        "``POST /api/dependencies/{container_name}/upgrade(-major)``."
    ),
    responses={
        400: {"description": "Invalid request (mode/packages mismatch, path traversal)."},
        401: {"description": "Authentication required."},
        404: {"description": "Container not found or not visible to this tenant."},
        500: {"description": "Upgrade execution failed."},
    },
)
async def package_upgrade_in_container(
    container_name: str = Path(..., description="Name of the container"),
    request: PackageUpgradeRequest | None = None,
    tenant_id: str = Depends(require_tenant),
) -> PackageUpgradeResponse:
    if request is None:
        raise bad_request("Request body required")

    if request.mode is UpgradeMode.IN_RANGE and not request.packages:
        raise bad_request("packages must not be empty when mode='in-range'")
    if request.mode is UpgradeMode.CROSS_MAJOR and not request.major_specs:
        raise bad_request("major_specs must not be empty when mode='cross-major'")

    container_manager = get_container_manager()
    try:
        container_info = container_manager.get_container(container_name, tenant_id=tenant_id)
    except ValueError, RuntimeError:
        container_info = None
    if container_info is None:
        raise not_found("Container", container_name)

    try:
        result = await run_upgrade(
            container_name=container_name,
            tenant_id=tenant_id,
            manager=request.manager,
            mode=request.mode,
            packages=request.packages,
            major_specs=request.major_specs,
            project_path=request.project_path,
            dry_run=request.dry_run,
            timeout_seconds=request.timeout_seconds,
        )
    except PathTraversalError as exc:
        raise bad_request(str(exc)) from exc
    except ManagerNotSupportedError as exc:
        raise bad_request(str(exc)) from exc
    except HTTPException:
        raise
    except (RuntimeError, OSError, ValueError) as exc:
        logger.error(
            "package_upgrade failed: container=%s manager=%s err=%s",
            container_name,
            request.manager.value,
            exc,
        )
        raise internal_error(f"Package upgrade failed: {exc}") from exc

    return PackageUpgradeResponse(
        manager=request.manager,
        mode=request.mode,
        project_path=request.project_path,
        dry_run=request.dry_run,
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_seconds=result.duration_seconds,
        commands_executed=result.commands_executed,
        manifest_files_before=result.manifest_files_before,
        manifest_files_after=result.manifest_files_after,
    )


class PackageModifyRequest(BaseModel):
    manager: LanguagePackageManager = Field(
        ...,
        description="Package manager to dispatch against.",
    )
    action: ModifyAction = Field(
        ...,
        description="``add`` introduces new deps, ``remove`` drops them, ``replace`` "
        "runs remove then add atomically (one ``commit`` upstream on coding-security).",
    )
    add_specs: list[AddSpec] = Field(
        default_factory=list,
        description="Required for ``add`` and ``replace``. Ignored for ``remove``.",
    )
    remove_names: list[str] = Field(
        default_factory=list,
        description="Required for ``remove`` and ``replace``. Ignored for ``add``.",
        max_length=20,
    )
    project_path: str = Field(default=".", max_length=512)
    dry_run: bool = False
    timeout_seconds: int = Field(default=600, ge=10, le=1800)

    model_config = ConfigDict(extra="forbid")


class PackageModifyResponse(BaseModel):
    manager: LanguagePackageManager
    action: ModifyAction
    project_path: str
    dry_run: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    commands_executed: list[list[str]]
    manifest_files_before: dict[str, str]
    manifest_files_after: dict[str, str]

    model_config = ConfigDict(extra="forbid")


@router.post(
    "/{container_name}/package-modify",
    response_model=PackageModifyResponse,
    operation_id="package_modify_in_workspace",
    summary="Run an add / remove / replace operation on a workspace's dependency set",
    description=(
        "Add brand-new dependencies, remove existing ones, or atomically replace one "
        "for another (e.g. ``asyncpg`` → ``psycopg``). Like ``/package-upgrade``, the "
        "command is constructed server-side from typed input — no raw shell tokens. "
        "``replace`` runs remove then add and stops on the first non-zero exit so the "
        "coding-security orchestrator can refuse to commit a half-migration."
    ),
    responses={
        400: {"description": "Invalid request (action/specs mismatch, path traversal)."},
        401: {"description": "Authentication required."},
        404: {"description": "Container not found or not visible to this tenant."},
        500: {"description": "Modify execution failed."},
    },
)
async def package_modify_in_container(
    container_name: str = Path(..., description="Name of the container"),
    request: PackageModifyRequest | None = None,
    tenant_id: str = Depends(require_tenant),
) -> PackageModifyResponse:
    if request is None:
        raise bad_request("Request body required")

    if request.action is ModifyAction.ADD and not request.add_specs:
        raise bad_request("add_specs must not be empty when action='add'")
    if request.action is ModifyAction.REMOVE and not request.remove_names:
        raise bad_request("remove_names must not be empty when action='remove'")
    if request.action is ModifyAction.REPLACE and (
        not request.add_specs or not request.remove_names
    ):
        raise bad_request("both add_specs and remove_names required when action='replace'")

    container_manager = get_container_manager()
    try:
        container_info = container_manager.get_container(container_name, tenant_id=tenant_id)
    except ValueError, RuntimeError:
        container_info = None
    if container_info is None:
        raise not_found("Container", container_name)

    try:
        result = await run_modify(
            container_name=container_name,
            tenant_id=tenant_id,
            manager=request.manager,
            action=request.action,
            add_specs=request.add_specs,
            remove_names=request.remove_names,
            project_path=request.project_path,
            dry_run=request.dry_run,
            timeout_seconds=request.timeout_seconds,
        )
    except PathTraversalError as exc:
        raise bad_request(str(exc)) from exc
    except ManagerNotSupportedError as exc:
        raise bad_request(str(exc)) from exc
    except HTTPException:
        raise
    except (RuntimeError, OSError, ValueError) as exc:
        logger.error(
            "package_modify failed: container=%s manager=%s action=%s err=%s",
            container_name,
            request.manager.value,
            request.action.value,
            exc,
        )
        raise internal_error(f"Package modify failed: {exc}") from exc

    return PackageModifyResponse(
        manager=request.manager,
        action=request.action,
        project_path=request.project_path,
        dry_run=request.dry_run,
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_seconds=result.duration_seconds,
        commands_executed=result.commands_executed,
        manifest_files_before=result.manifest_files_before,
        manifest_files_after=result.manifest_files_after,
    )

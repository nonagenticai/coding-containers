"""Async workspace exec: ``POST {ws}/exec/async`` (202 + job) and ``GET {ws}/exec/jobs/{id}``.

Why it exists: the sync ``exec_in_workspace`` holds the HTTP request open and is
capped at 300 s, which breaks any test or build longer than five minutes. The
conveyor (n8n) instead parks a cell on a Wait node and is resumed by the
completion callback POSTed here — no polling (George, 2026-10-11).

Same gates as the sync route, reused, not re-implemented: ``require_tenant``,
the command allowlist + dangerous-pattern check at the boundary, the
tenant-scoped workspace lookup, and ``run_workspace_command`` (runtime allowlist
under ``CALLER_SUPPLIED``) for the execution itself.

Mounted on the canonical ``/api/workspaces`` surface only (never the legacy
``/api/containers`` alias) — see :mod:`src.routers.workspaces`.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Path, Response, status
from pydantic import BaseModel, ConfigDict, Field

from coding_client.auth.tenant import require_tenant
from coding_client.models.responses import (
    bad_request,
    conflict,
    not_found,
    service_unavailable,
    validation_error,
)
from src.core.container_manager import get_container_manager
from src.core.log_safety import summarize_command
from src.core.toolchain import toolchain_of
from src.routers.container_exec import EXEC_ENV_FIELD
from src.runtime import ANY_TOOLCHAIN, check_dangerous_patterns, validate_command
from src.runtime.policy import exec_env_violation
from src.services.exec_jobs.callback import callback_url_allowed
from src.services.exec_jobs.runner import ExecJobRunner, get_exec_job_runner, status_path
from src.services.exec_jobs.store import ACTIVE

logger = logging.getLogger(__name__)
router = APIRouter()

#: Retry-After (seconds) for a replay that lands while the holder's callback is in flight.
REPLAY_RETRY_AFTER_S = 2

#: Safety ceiling only, not a time budget (George 2026-10-11: no time budgets).
#: It bounds how long one job can hold a runner slot and a lease if a command
#: never exits; a legitimate build or test suite should never approach it.
EXEC_ASYNC_MAX_TIMEOUT_SECONDS = 4 * 60 * 60


class ExecAsyncRequest(BaseModel):
    """Body of ``exec_in_workspace_async``: the sync request plus ``callback_url``."""

    command: list[str] = Field(..., description="Command to execute as list of arguments")
    workdir: str = Field(default="/workspace", description="Working directory")
    timeout: int = Field(
        default=3600,
        ge=1,
        le=EXEC_ASYNC_MAX_TIMEOUT_SECONDS,
        description=(
            "Wall-clock limit in seconds (1-14400). Not a budget: 14400 (4 h) is a "
            "safety ceiling against a command that never exits."
        ),
        # Distinct n8n parameter name: the node generator flattens body fields per
        # node, and merged with exec_in_workspace.timeout (le=300) one of the two
        # ops would inherit the other's range and default.
        json_schema_extra={"x-n8n-param-name": "asyncExecTimeout"},
    )
    callback_url: str | None = Field(
        default=None,
        max_length=2048,
        description=(
            "Optional URL POSTed once with the final job document when the job ends "
            "(completed or failed — including orphaned/interrupted). Must sit under a "
            "configured EXEC_CALLBACK_URL_PREFIXES prefix (e.g. an n8n Wait-node "
            "resume URL), else 422. Also the idempotency key per tenant: the key is "
            "held while its job is queued/running OR its callback is still pending "
            "(being sent or scheduled for retry), and released once the callback has "
            "its final answer. A replayed POST while it is held returns the same job "
            "(202, same job_id), never a second run; while that job's callback is "
            "being delivered right now it answers 503 with Retry-After (retry then); "
            "the same URL held by a job on another workspace is a 409. Looping cells "
            "that reuse one resume URL should append a unique per-iteration suffix."
        ),
        json_schema_extra={"x-n8n-param-name": "execCallbackUrl"},
    )
    env: dict[str, str] | None = EXEC_ENV_FIELD

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "command": ["python", "-m", "pytest", "-q"],
                "workdir": "/workspace",
                "timeout": 3600,
                "callback_url": "https://example.com/webhook-waiting/4711",
            }
        }
    )


class ExecJobLinks(BaseModel):
    status: str = Field(..., description="Path of the job's status route")


class ExecJobAccepted(BaseModel):
    """202 body: the job exists and is leased to a replica."""

    job_id: str
    container_name: str
    status: str = Field(
        ...,
        description="'queued' for a new job; the existing job's status on an idempotent replay",
    )
    links: ExecJobLinks


class ExecJobStatus(BaseModel):
    """Job document; the completion callback POSTs the same fields."""

    job_id: str
    container_name: str
    status: str = Field(..., description="queued | running | completed | failed")
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    truncated: bool = Field(False, description="True when either stream was clipped")
    bytes_omitted: int = Field(0, description="Bytes dropped across both streams")
    timed_out: bool = False
    duration: float | None = None
    error: str | None = Field(
        None,
        description=(
            "Why a job failed without a normal exit: 'timeout: …', 'orphaned: …' "
            "(the running replica died), 'interrupted: …' (it shut down), or the "
            "execution error"
        ),
    )
    callback_state: str = Field(
        ...,
        description=(
            "none | sending | retrying (not accepted yet, resent with capped backoff "
            "until it is) | delivered | refused | disallowed"
        ),
    )
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    links: ExecJobLinks


def _runner_or_503() -> ExecJobRunner:
    runner = get_exec_job_runner()
    if runner is None:
        raise service_unavailable("Async exec is not available: no job store is configured")
    return runner


def _status_from_row(row: dict[str, Any]) -> ExecJobStatus:
    return ExecJobStatus(
        job_id=row["job_id"],
        container_name=row["container_name"],
        status=row["status"],
        exit_code=row["exit_code"],
        stdout=row["stdout"],
        stderr=row["stderr"],
        stdout_truncated=row["stdout_truncated"],
        stderr_truncated=row["stderr_truncated"],
        truncated=bool(row["stdout_truncated"] or row["stderr_truncated"]),
        bytes_omitted=row["bytes_omitted"],
        timed_out=row["timed_out"],
        duration=row["duration_seconds"],
        error=row["error"],
        callback_state=row["callback_state"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        links=ExecJobLinks(status=status_path(row["container_name"], row["job_id"])),
    )


@router.post(
    "/{container_name}/exec/async",
    response_model=ExecJobAccepted,
    operation_id="exec_in_workspace_async",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Execute command in workspace asynchronously (202 + completion callback)",
    description="""
    Start a command in the workspace and return immediately with a job id.

    The job's final document is POSTed once to `callback_url` (when given) as
    soon as it ends — completed (any exit code) or failed (timeout, execution
    error, or the running replica died: `orphaned` / `interrupted`). No polling
    is needed; `GET …/exec/jobs/{job_id}` reads the same document.

    Same security controls as `exec_in_workspace`: tenant isolation (a foreign
    workspace is a 404), the command allowlist and dangerous-pattern check.
    `timeout` may be up to 14400 s (a safety ceiling, not a budget).
    """,
    responses={
        202: {"description": "Job accepted"},
        400: {"description": "Invalid or disallowed command"},
        401: {"description": "Authentication required"},
        404: {"description": "Workspace not found or not owned by the caller's tenant"},
        409: {
            "description": "callback_url is held by another workspace's job (active, or "
            "its callback still pending)"
        },
        422: {"description": "callback_url not under an allowed prefix"},
        503: {
            "description": "Job store unavailable, or (with Retry-After) the previous job "
            "for this callback_url is delivering its callback right now"
        },
    },
)
async def exec_in_workspace_async(
    response: Response,
    request: ExecAsyncRequest,
    container_name: str = Path(..., description="Name of the workspace"),
    tenant_id: str = Depends(require_tenant),
) -> ExecJobAccepted:
    if not request.command:
        raise bad_request("Command list cannot be empty")
    # ANY_TOOLCHAIN before the lookup (fast 400 for what no workspace may run);
    # the workspace's own toolchain is checked right after the lookup below.
    if not validate_command(request.command, toolchain=ANY_TOOLCHAIN):
        raise bad_request(
            f"Command '{request.command[0]}' not allowed. Only whitelisted commands can be executed."
        )
    if not check_dangerous_patterns(request.command):
        raise bad_request("Command contains dangerous patterns and is not allowed")
    env_refusal = exec_env_violation(request.env, request.command)
    if env_refusal:
        raise bad_request(f"env refused: {env_refusal}")
    if request.callback_url is not None and not callback_url_allowed(request.callback_url):
        # Refused before anything is persisted: the control pod POSTs this URL
        # from inside the cluster, so only configured prefixes are accepted.
        raise validation_error(
            "callback_url is not under an allowed callback prefix", field="callback_url"
        )
    runner = _runner_or_503()

    # Tenant-scoped ownership check up front, so a foreign or missing workspace
    # is an honest synchronous 404 rather than a job that fails later.
    try:
        info = get_container_manager().get_container(container_name, tenant_id=tenant_id)
    except ValueError, RuntimeError:
        info = None
    if info is None:
        raise not_found("Container", container_name)
    if not validate_command(request.command, toolchain=toolchain_of(info)):
        raise bad_request(
            f"Command '{request.command[0]}' not allowed. Only whitelisted commands can be executed."
        )

    job_id = str(uuid.uuid7())
    try:
        job, created = await runner.store.create_job(
            job_id=job_id,
            tenant_id=tenant_id,
            container_name=container_name,
            command_summary=summarize_command(request.command),
            workdir=request.workdir,
            timeout_seconds=request.timeout,
            callback_url=request.callback_url,
            owner=runner.owner,
            lease_seconds=runner.lease_seconds,
        )
        if created:
            runner.submit(job, list(request.command), request.env)
    except Exception as exc:
        logger.exception("Async exec: could not create job for %s", container_name)
        raise service_unavailable("Async exec job store unavailable") from exc

    if not created:
        if job["container_name"] != container_name:
            raise conflict(
                "Exec job",
                "callback_url is already used by an active job on another workspace",
            )
        if job["status"] not in ACTIVE and job["callback_state"] == "sending":
            # The holder's callback is being delivered right now, outcome unknown:
            # it may resume the cell (then this replay is moot) or fail retryably
            # (then this replay gets the holder back). Never a second run.
            exc = service_unavailable(
                "callback_url's previous job is delivering its callback; retry shortly"
            )
            exc.headers = {"Retry-After": str(REPLAY_RETRY_AFTER_S)}
            raise exc
        logger.info(
            "Async exec replay: tenant=%s, container=%s, existing job=%s returned",
            tenant_id,
            container_name,
            job["job_id"],
        )
        link = status_path(container_name, job["job_id"])
        response.headers["Location"] = link
        return ExecJobAccepted(
            job_id=job["job_id"],
            container_name=container_name,
            status=job["status"],
            links=ExecJobLinks(status=link),
        )

    logger.info(
        "Async exec accepted: tenant=%s, container=%s, job=%s, command=%s, timeout=%d, "
        "callback=%s, env=%s",
        tenant_id,
        container_name,
        job_id,
        summarize_command(request.command),
        request.timeout,
        request.callback_url is not None,
        sorted(request.env) if request.env else [],
    )
    link = status_path(container_name, job_id)
    response.headers["Location"] = link
    return ExecJobAccepted(
        job_id=job_id,
        container_name=container_name,
        status="queued",
        links=ExecJobLinks(status=link),
    )


@router.get(
    "/{container_name}/exec/jobs/{job_id}",
    response_model=ExecJobStatus,
    operation_id="get_exec_job_status",
    summary="Get async exec job status",
    description=(
        "Read an async exec job. Tenant- and workspace-scoped: a job of another "
        "tenant or another workspace is a 404, indistinguishable from none."
    ),
    responses={
        404: {"description": "No such job for this tenant and workspace"},
        503: {"description": "Job store unavailable"},
    },
)
async def get_exec_job_status(
    container_name: str = Path(..., description="Name of the workspace"),
    job_id: str = Path(..., description="Job id returned by exec_in_workspace_async"),
    tenant_id: str = Depends(require_tenant),
) -> ExecJobStatus:
    runner = _runner_or_503()
    try:
        canonical = str(uuid.UUID(job_id))
    except ValueError:
        raise not_found("Exec job", job_id) from None
    try:
        row = await runner.store.get_job(
            job_id=canonical, tenant_id=tenant_id, container_name=container_name
        )
    except Exception as exc:
        logger.exception("Async exec: status read failed for %s", job_id)
        raise service_unavailable("Async exec job store unavailable") from exc
    if row is None:
        raise not_found("Exec job", job_id)
    return _status_from_row(row)

"""
V2 Container Exec API Endpoint

Secure command execution in containers with RBAC protection.
Provides ability to execute debugging commands inside containers.
"""

from __future__ import annotations

import logging
from typing import Any

from docker.errors import DockerException
from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel, ConfigDict, Field

from coding_client.auth.tenant import require_tenant
from coding_client.exceptions import BaseAPIException
from coding_client.models.responses import (
    bad_request,
    error_response,
    internal_error,
    not_found,
    service_unavailable,
)
from src.config.settings import get_settings
from src.core.container_manager import get_container_manager
from src.core.kubernetes_container_provisioner import (
    ContainerDeploymentNotFoundError,
    WorkspaceCloneFailedError,
)
from src.core.log_safety import summarize_command
from src.runtime import (
    ANY_TOOLCHAIN,
    CommandNotPermittedError,
    ExecutionProfile,
    RunOnceResult,
    RuntimeKind,
    UnsupportedCapabilityError,
    check_dangerous_patterns,
    resolve_runtime,
    validate_command,
)
from src.runtime.policy import (
    EXEC_ENV_MAX_TOTAL_BYTES,
    EXEC_ENV_MAX_VALUE_BYTES,
    EXEC_ENV_MAX_VARS,
    env_rule_summary,
    exec_env_violation,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# The identity every workspace command runs as. Not a request parameter: the
# service decides, and the Kubernetes pod security context pins the same
# unprivileged user (WORKSPACE_UID in kubernetes_container_provisioner). Named
# rather than inlined so the Docker and agent-host tiers cannot drift apart.
WORKSPACE_EXEC_USER = "coder"


def _clip(raw: bytes | None, limit: int) -> tuple[str, int]:
    """Decode one output stream, bounded to ``limit`` bytes.

    Returns ``(text, bytes_omitted)``. Decoding with ``errors="replace"`` before
    measuring would let a stream of invalid bytes grow on decode, so the limit is
    applied to the raw bytes and the tail dropped with ``errors="ignore"`` so a
    multi-byte character straddling the boundary vanishes instead of becoming a
    replacement character.
    """
    if not raw:
        return "", 0
    if len(raw) <= limit:
        return raw.decode("utf-8", errors="replace"), 0
    return raw[:limit].decode("utf-8", errors="ignore"), len(raw) - limit


def _container_not_provisioned(container_name: str) -> HTTPException:
    """404 for a container whose backing Deployment/pod is absent.

    Same ``CONTAINER_NOT_FOUND`` error code as ``not_found("Container", …)`` so
    downstream consumers key off a single code, but with a clearer message
    explaining the container must be provisioned before exec-based operations
    (e.g. dependency reports) can run.
    """
    return error_response(
        status_code=status.HTTP_404_NOT_FOUND,
        message=(
            f"Container '{container_name}' is not provisioned or running — "
            "provision it before running a dependency report"
        ),
        error_code="CONTAINER_NOT_FOUND",
    )


# ============================================================================
# MODELS
# ============================================================================


#: The ``env`` body field, shared by ``exec_in_workspace`` and
#: ``exec_in_workspace_async`` so the two cannot drift.
EXEC_ENV_FIELD: Any = Field(
    default=None,
    description=(
        "Extra environment for THIS command only, as {NAME: VALUE}. Applied without a "
        "shell (`/usr/bin/env NAME=VALUE <command>` as argv on Kubernetes, the native "
        "exec env on Docker), so a value is always data: `$(…)`, `;`, quotes and "
        "newlines are never interpreted. Names must match ^[A-Z_][A-Z0-9_]*$; at most "
        f"{EXEC_ENV_MAX_VARS} variables, {EXEC_ENV_MAX_VALUE_BYTES} bytes per value, "
        f"{EXEC_ENV_MAX_TOTAL_BYTES} bytes in total. {env_rule_summary()} The same rule "
        "applies to `env NAME=VALUE` inside `command` (equivalent, and still working) and to "
        "a background service's env. Agent-host workspaces have no per-command environment: "
        "any env there is refused (400 on the sync route, a failed job on the async one). "
        "A loopback Redis/Postgres for the repo's code: run `ws-redis start` (every image) "
        "or `ws-pg start` (python+rust); each prints its URL to pass here."
    ),
    json_schema_extra={
        "x-n8n-param-name": "execEnv",
        "propertyNames": {"pattern": "^[A-Z_][A-Z0-9_]*$"},
        "maxProperties": EXEC_ENV_MAX_VARS,
        "example": {"DATABASE_URL": "postgresql://app@127.0.0.1:5432/app", "RUST_LOG": "info"},
    },
)


class ExecRequest(BaseModel):
    """Request model for workspace command execution.

    The identity is NOT a request parameter. This model used to carry
    ``user: str = Field(default="root")``, published in the OpenAPI example, and
    passed straight to Docker exec — so the public API invited a caller to name
    the identity their command ran as, defaulting to root. The Kubernetes path
    never forwarded it, so in production it was inert, which is the only reason
    it was never exploited rather than a reason to keep it.

    Identity is the service's decision now: workspaces run as the unprivileged
    ``coder`` user pinned by the pod security context. Removing the field is not
    a breaking change — no caller in the fleet ever set it (verified across
    coding-client, coding-analysis, coding-repo-jobs, coding-security and
    coding-testing) and pydantic ignores unknown keys here, so an old caller
    still sending it gets a 200.

    Attributes:
        command: List of command arguments (e.g., ["ls", "-la"])
        workdir: Working directory inside the workspace (default: /workspace)
        timeout: Command execution timeout in seconds (default: 30)
    """

    command: list[str] = Field(
        ...,
        description="Command to execute as list of arguments",
    )
    workdir: str = Field(
        default="/workspace",
        description="Working directory for command execution",
    )
    timeout: int = Field(default=30, ge=1, le=300, description="Command timeout in seconds (1-300)")
    env: dict[str, str] | None = EXEC_ENV_FIELD

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "command": ["python", "--version"],
                "workdir": "/workspace",
                "timeout": 30,
            }
        }
    )


class ExecResponse(BaseModel):
    """Response model for workspace command execution.

    ``truncated`` and ``bytes_omitted`` are not optional extras. Output used to
    be unbounded on every backend, so a large or binary result was buffered
    whole in the service and serialised into this body. Now that it is capped,
    a clipped result must be distinguishable from a complete one — otherwise the
    cap trades an availability bug for a correctness one, and a caller parsing
    clipped output would reach a confidently wrong conclusion.

    Attributes:
        exit_code: Command exit code (0 = success)
        stdout: Standard output from command
        stderr: Standard error output from command
        duration: Command execution duration in seconds
        truncated: Whether either stream hit the output budget
        bytes_omitted: Total bytes dropped across both streams
    """

    exit_code: int = Field(
        ...,
        description="Command exit code (0 for success)",
    )
    stdout: str = Field(
        default="",
        description="Command standard output",
    )
    stderr: str = Field(
        default="",
        description="Command standard error output",
    )
    duration: float = Field(
        default=0.0,
        description="Command execution duration in seconds",
    )
    truncated: bool = Field(
        default=False,
        description="True when output exceeded the budget and was clipped",
    )
    bytes_omitted: int = Field(
        default=0,
        description="Bytes dropped across stdout and stderr when truncated",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "exit_code": 0,
                "stdout": "Python 3.9.0",
                "stderr": "",
                "duration": 0.15,
            }
        }
    )


# ============================================================================
# COMMAND VALIDATION
# ============================================================================


# validate_command and check_dangerous_patterns now live in src/runtime/policy.py
# so the SAME allowlist can be enforced in two places: here at the API boundary,
# where a bad command is a fast 400 before any workspace is touched, and again
# inside the runtime adapters for anything marked CALLER_SUPPLIED. The boundary
# check is a property of this call path; the adapter check is a property of the
# backend, and holds for whoever reaches it next.


# ============================================================================
# SHARED EXEC PATH
# ============================================================================


async def run_workspace_command(
    container_name: str,
    tenant_id: str,
    argv: list[str],
    *,
    cwd: str | None = None,
    timeout: int = 30,
    env: dict[str, str] | None = None,
) -> tuple[RuntimeKind, RunOnceResult]:
    """Run caller-supplied ``argv`` once in a tenant-owned workspace.

    The one exec path for routes that run a command in a workspace: tenant-scoped
    lookup, ``resolve_runtime`` (Kubernetes / agent-host / Docker, chosen once),
    then ``run_once`` under ``ExecutionProfile.CALLER_SUPPLIED`` so the runtime
    allowlist applies. ``cwd=None`` means the workspace's repository root.
    ``env`` is the caller's per-command environment; the runtime re-judges it
    under the same rule as the boundary (``exec_env_violation``).

    Raises the same HTTP errors the exec route has always answered with:
    404 (absent or foreign workspace, or Deployment gone), 400 (command refused
    by the runtime policy, unsupported capability), 503 (no usable runtime) and
    500 (execution transport error). Callers that report failures in a body
    rather than a status catch ``HTTPException``.
    """
    container_manager = get_container_manager()

    # Get container with tenant validation
    try:
        container_info = container_manager.get_container(container_name, tenant_id=tenant_id)
        logger.debug("Found container: %s for tenant: %s", container_name, tenant_id)
    except (ValueError, RuntimeError):  # fmt: skip
        logger.warning("Container not found: %s", container_name)
        raise not_found("Container", container_name)

    if container_info is None:
        logger.warning("Container not found: %s", container_name)
        raise not_found("Container", container_name)

    if cwd is None:
        # Same rule the file routes use (codebase_path, else /workspace).
        from src.routers.containers.files_manage import _repository_root

        cwd = _repository_root(container_info)

    # One backend, chosen once. This used to be a three-tier if/elif here —
    # Kubernetes when the endpoint looked in-cluster, else agent-host, else
    # Docker — while ContainerManager.execute_command made the same choice
    # by the opposite rule. Selection now lives in src.runtime.resolve so
    # both entry points reach the same backend for the same workspace.
    logger.debug("Executing: %s in container %s", summarize_command(argv), container_name)

    try:
        # The manager core supplies the Docker client and agent-host
        # registry; the provisioner is resolved lazily inside resolve_runtime
        # so the Kubernetes path costs nothing on a Docker-only host.
        from src.core.container_manager_core import (
            get_container_manager as get_core_manager,
        )

        runtime = resolve_runtime(container_info, core=get_core_manager())
    except RuntimeError as e:
        # 503, not 500: the workspace's backend is down, which is a
        # statement about this service's dependencies rather than about the
        # caller's request. The pre-refactor Docker tier already answered
        # 503 when the daemon was unreachable; keeping that means a client's
        # retry logic still works.
        logger.error("No runtime for container %s: %s", container_name, e)
        raise service_unavailable(f"Command execution failed: {e}") from e

    try:
        result = await runtime.run_once(
            container_info,
            argv,
            cwd=cwd,
            env=env or None,
            timeout=timeout,
            output_limit=get_settings().exec_max_output_bytes,
            # This argv came from the caller, so the allowlist applies —
            # stated here rather than left implicit in which function the
            # request happened to enter.
            execution_profile=ExecutionProfile.CALLER_SUPPLIED,
        )
    except ContainerDeploymentNotFoundError as e:
        # The workspace's Deployment is gone (never provisioned, or
        # deprovisioned). A 404, not a 500 — returning 500 makes
        # coding-gateway collapse it into a misleading "service
        # unavailable" instead of an honest "container missing".
        logger.warning("Exec on missing container: container=%s, error=%s", container_name, str(e))
        raise _container_not_provisioned(container_name) from e
    except WorkspaceCloneFailedError as e:
        # The workspace's clone failed (bad/insufficient BYO token, missing
        # branch, deleted repo). The clone is the only authority on repository
        # access, so this is the caller's precondition, not a server fault.
        logger.warning("Exec on workspace whose clone failed: container=%s", container_name)
        raise BaseAPIException(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            message=str(e),
            error_code="WORKSPACE_CLONE_FAILED",
            # The envelope renders ``code`` as HTTP_412, so the typed code is
            # repeated in ``details`` where a client can branch on it.
            details={
                "error_code": "WORKSPACE_CLONE_FAILED",
                "container_name": container_name,
                "reason": e.failure,
            },
        ) from e
    except CommandNotPermittedError as e:
        # The runtime layer refused a command. Same 400 the exec boundary
        # gives, so a caller sees one consistent answer.
        logger.warning("Runtime policy refused a command for %s: %s", container_name, e)
        raise bad_request(str(e)) from e
    except UnsupportedCapabilityError as e:
        # Asked for something this backend cannot do. A client error, and
        # named, rather than a 500 that hides which capability was missing.
        logger.warning("Unsupported exec capability for %s: %s", container_name, e)
        raise bad_request(str(e)) from e
    except (DockerException, RuntimeError, OSError, ValueError, KeyError) as e:
        logger.error("Execution error: container=%s, error=%s", container_name, str(e))
        raise internal_error(f"Command execution failed: {str(e)}") from e

    return runtime.kind, result


# ============================================================================
# API ENDPOINTS
# ============================================================================


@router.post(
    "/{container_name}/exec",
    response_model=ExecResponse,
    operation_id="exec_in_workspace",
    status_code=status.HTTP_200_OK,
    summary="Execute command in workspace",
    description="""
    Execute a command inside a running container for debugging purposes.

    Authenticated, tenant-scoped: the caller may only exec into a container
    owned by their own tenant. Only whitelisted commands are allowed.
    All executions are logged for audit purposes.

    **Security Controls:**
    - Tenant isolation: container must belong to the caller's tenant
      (BOLA-protected; a foreign container returns 404, not its contents)
    - Whitelist: Only approved commands allowed
    - Timeout: Maximum 300 seconds per command
    - Pattern matching: Dangerous commands blocked

    **Allowed Commands:**
    File operations: ls, cat, pwd, echo, find, head, tail, grep
    Process inspection: ps, top, df, du
    Languages: python, python3, node, npm, pip, java, go
    Git: git (read-only operations)
    Text processing: sed, awk, grep, cut, sort
    System: uname, whoami, env, date
    """,
    responses={
        200: {"description": "Command executed successfully"},
        400: {"description": "Invalid command or parameters"},
        401: {"description": "Authentication required"},
        404: {"description": "Container not found or not owned by the caller's tenant"},
        500: {"description": "Command execution failed"},
    },
)
async def exec_in_container(
    container_name: str = Path(
        ...,
        description="Name of the container",
    ),
    request: ExecRequest | None = None,
    tenant_id: str = Depends(require_tenant),
) -> ExecResponse:
    """Execute a command inside a container.

    Tenant-scoped: ``require_tenant`` resolves the caller's tenant and the
    container is fetched with that tenant (BOLA-protected), so a caller can
    only exec into a container they own. Commands are whitelist- and
    dangerous-pattern-checked.

    NOTE: there is no additional role gate. A role guard
    (``require_role("access-coding")``) was considered but would 401 the
    in-cluster service-to-service callers that authenticate with only an
    ``X-Tenant-Id`` header (no JWT, hence no ``request.state.user``), e.g.
    coding-analysis's quality/architecture flows. Adding one requires those
    service tokens to carry the role first — tracked as future hardening.

    Args:
        container_name: Name of the container
        request: Execution request with command details

    Returns:
        ExecResponse with exit code, stdout, stderr, and duration

    Raises:
        HTTPException: 400 if command invalid/dangerous
        HTTPException: 404 if container not found or not owned by the tenant
        HTTPException: 500 if execution fails
    """
    if request is None:
        raise bad_request("Request body required")

    # Validate command
    if not request.command:
        raise bad_request("Command list cannot be empty")

    # Check if command is whitelisted. ANY_TOOLCHAIN: the workspace (and so its
    # toolchain) is not looked up yet; a command no toolchain admits is still a
    # fast 400 here, and the runtime adapter re-checks against the workspace's
    # real toolchain (a python+rust-only command in a python workspace → 400).
    if not validate_command(request.command, toolchain=ANY_TOOLCHAIN):
        raise bad_request(
            f"Command '{request.command[0]}' not allowed. Only whitelisted commands can be executed."
        )

    # Check for dangerous patterns
    if not check_dangerous_patterns(request.command):
        raise bad_request("Command contains dangerous patterns and is not allowed")

    env_refusal = exec_env_violation(request.env, request.command)
    if env_refusal:
        raise bad_request(f"env refused: {env_refusal}")

    # Log execution request for audit. env NAMES only: a value is caller data
    # (a database URL with its password), and log lines go to a retained Loki.
    logger.info(
        "Exec request: tenant=%s, container=%s, command=%s, timeout=%d, env=%s",
        tenant_id,
        container_name,
        summarize_command(request.command),
        request.timeout,
        sorted(request.env) if request.env else [],
    )

    try:
        runtime_kind, result = await run_workspace_command(
            container_name,
            tenant_id,
            request.command,
            cwd=request.workdir,
            timeout=request.timeout,
            env=request.env,
        )

        # ⛔ This was the FOURTH command render in this file and the only one
        # the first pass missed — while lines 292 and 320 were being fixed.
        # It was also the worst of the four: `" ".join(request.command)` is the
        # WHOLE argv, not even the `[:3]` slice the other sites had, at INFO,
        # on the SUCCESS path. Both boundary gates admit the payload
        # (validate_command and check_dangerous_patterns both pass for
        # `git push https://x-access-token:<token>@github.com/o/r`), so a
        # caller-supplied credential went straight to a 30-day Loki with a 200.
        logger.info(
            "Exec success (%s): container=%s, command=%s, exit_code=%d, duration=%.2fs",
            runtime_kind.value,
            container_name,
            summarize_command(request.command),
            result.exit_code,
            result.duration,
        )
        if result.truncated:
            logger.warning(
                "Exec output truncated for %s: %d bytes omitted",
                container_name,
                result.bytes_omitted,
            )
        return ExecResponse(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration=result.duration,
            truncated=result.truncated,
            bytes_omitted=result.bytes_omitted,
        )

    except HTTPException:
        raise
    except ContainerDeploymentNotFoundError as e:
        # Defense-in-depth: if the typed not-found ever escapes a tier handler,
        # it must still surface as an honest 404, never a generic 500.
        logger.warning("Exec on missing container: container=%s, error=%s", container_name, str(e))
        raise _container_not_provisioned(container_name) from e
    except (DockerException, RuntimeError, OSError, ValueError, KeyError) as e:
        logger.error("Unexpected error during exec: %s", str(e))
        raise internal_error("An unexpected error occurred during command execution")

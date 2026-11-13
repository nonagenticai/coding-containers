"""Workspace background services: start a provider, probe it on loopback, stop it.

``POST   {ws}/services``               start (returns when ready, or failed + logs)
``GET    {ws}/services``               list
``GET    {ws}/services/{id}``          status + log tail
``DELETE {ws}/services/{id}``          stop (process-group kill)
``POST   {ws}/services/{id}/probe``    one HTTP request to the service, in the pod

py2rust gap #4 full: to RECORD a source system (many thin services that call each
other and share Redis), the workspace runs N source services, each from its own
``/workspace/.repos/<owner>__<repo>`` with its own port/env/command/readiness, next
to their dependencies. A caller starts N services with N ``POST {ws}/services``
(and ``DELETE`` the started ones if one fails). Dependencies are started by exec:
``ws-redis start`` (every image) or ``ws-pg start`` (python+rust) prints the loopback
URL to hand to services via ``env``. Starts are bounded by a live-service count cap;
memory by the pod's own limit, sized at create from the hints (``src.core.toolchain``).

WHY. The P7 ``integration_check`` card verifies a consumer's call against its
provider. Both repositories are checked out in ONE workspace pod (the consumer as
the workspace, the provider via ``POST {ws}/repos``), the provider runs as a local
background process on ``127.0.0.1:<port>``, and the probe runs in the same pod.
No cross-pod networking exists or is added (George 2026-10-12).

The consumer-side probe can equally be the existing ``exec_in_workspace`` /
``exec_in_workspace_async`` with ``workdir`` = the consumer checkout and a
``python3``/``node`` argv that calls ``http://127.0.0.1:<port>``. ``probe`` exists
because the runtime image has no ``curl`` and the card should not have to author
Python to make one HTTP request.

Security:

* tenant isolation is the workspace's: every route resolves the workspace with the
  caller's tenant (a foreign workspace is a 404), and service state lives only in
  that workspace's own pod — there is no shared registry to leak across tenants;
* the command passes the SAME allowlist + dangerous-pattern check as ``exec``; env
  names are validated and values are never logged or echoed; the port is
  1024-65535, never 8000, and must be free in the pod;
* nothing is interpolated into a shell: the pod program receives the validated
  spec as one base64 argv word;
* an ACCIDENT guard, not a boundary: at readiness, the declared port is checked
  once in ``/proc/net/tcp*`` and a provider bound beyond loopback (0.0.0.0) is
  stopped and reported ``not_loopback``. It does not watch other ports or later
  rebinds, and it adds no isolation that did not exist: the pre-existing coding-jobs
  NetworkPolicy already admits pod-to-pod traffic in that namespace, and ``exec``
  can already run a 0.0.0.0 listener. Isolation between workspaces is that policy's
  job, not this check's;
* output is capped (8 MiB log kept in the pod, 8 KiB tail returned, 64 KiB probe body);
* lifetime: the supervisor stops a service at ``max_lifetime_seconds`` (a safety
  ceiling, default and maximum 4 h — not a work budget), and deleting the
  workspace kills its pod and with it every service.

Mounted on the canonical ``/api/workspaces`` surface only.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Body, Depends, Path, status
from pydantic import BaseModel, ConfigDict, Field

from coding_client.auth.tenant import require_tenant
from coding_client.models.responses import (
    bad_request,
    error_response,
    internal_error,
    not_found,
)
from src.core.container_manager import get_container_manager
from src.core.toolchain import max_services_of, toolchain_of
from src.runtime.policy import env_rule_summary
from src.services.workspace_services.service import (
    DEFAULT_MAX_LIFETIME_SECONDS,
    LOG_TAIL_BYTES,
    PROBE_BODY_CAP,
    SERVICE_ROOT,
    STOP_GRACE_SECONDS,
    ServiceSpecError,
    build_launch_spec,
    pod_program_argv,
    validate_http_path,
    validate_service_id,
)

logger = logging.getLogger(__name__)
router = APIRouter()

#: Seconds the launch exec may take beyond ``ready_timeout`` (stop grace + startup).
_LAUNCH_EXEC_SLACK_S = STOP_GRACE_SECONDS + 20
_SHORT_EXEC_S = STOP_GRACE_SECONDS + 15
_PROBE_HEADER_LIMIT = 32
_HTTP_METHODS = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")


# ============================================================================
# MODELS
# ============================================================================


class ServiceReadiness(BaseModel):
    """When the service counts as started: an HTTP answer (< 500) or an open TCP port."""

    kind: Literal["http", "tcp"] = Field(default="http")
    path: str | None = Field(
        default="/",
        max_length=512,
        description="Request path for kind=http (GET; any status < 500 is ready)",
    )


class StartServiceRequest(BaseModel):
    command: list[str] = Field(
        ...,
        min_length=1,
        description="Provider argv; same allowlist as exec (python3, node, uv, npm, ...)",
        json_schema_extra={"x-n8n-param-name": "serviceCommand"},
    )
    workdir: str = Field(
        default="/workspace",
        description=(
            "Directory the provider runs in: /workspace, or a secondary repository "
            "path such as /workspace/.repos/<owner>__<repo>"
        ),
        json_schema_extra={"x-n8n-param-name": "serviceWorkdir"},
    )
    port: int = Field(
        ...,
        description=(
            "Port the provider listens on, on 127.0.0.1 ONLY (1024-65535, never 8000, "
            "must be free in the workspace). Pass it to the provider in its argv/env."
        ),
        json_schema_extra={"x-n8n-param-name": "servicePort"},
    )
    env: dict[str, str] | None = Field(
        default=None,
        description=(
            "Extra environment for the service (values are never logged or echoed). "
            "A loopback Redis/Postgres is started by exec: `ws-redis start` (every image) or "
            "`ws-pg start` (python+rust) prints its URL; pass it here, e.g. REDIS_URL / "
            "DATABASE_URL. Same rule as the exec env field: " + env_rule_summary()
        ),
        json_schema_extra={"x-n8n-param-name": "serviceEnv"},
    )
    readiness: ServiceReadiness = Field(
        default_factory=ServiceReadiness,
        json_schema_extra={"x-n8n-param-name": "serviceReadiness"},
    )
    ready_timeout: int = Field(
        default=60,
        ge=1,
        # 240 + the launch slack (25 s) = 265 s: under n8n's 300 s axios default and the
        # gateway's exec read budget (330 s), so the verdict always reaches the caller.
        le=240,
        description="Seconds to wait for readiness before the start is reported failed (1-240)",
        json_schema_extra={"x-n8n-param-name": "serviceReadyTimeout"},
    )
    max_lifetime_seconds: int = Field(
        default=DEFAULT_MAX_LIFETIME_SECONDS,
        ge=30,
        le=DEFAULT_MAX_LIFETIME_SECONDS,
        description=(
            "Safety ceiling, not a budget: the service is stopped this long after it "
            "started if nobody stopped it (default and maximum 4 h)"
        ),
        json_schema_extra={"x-n8n-param-name": "serviceMaxLifetime"},
    )
    name: str | None = Field(
        default=None,
        max_length=64,
        description="Optional label, e.g. the source repository (A-Z a-z 0-9 . _ -, max 64)",
        json_schema_extra={"x-n8n-param-name": "serviceName"},
    )
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "command": ["python3", "-m", "http.server", "18080", "--bind", "127.0.0.1"],
                "workdir": "/workspace/.repos/octocat__Hello-World",
                "port": 18080,
                "readiness": {"kind": "http", "path": "/"},
            }
        }
    )


class ServiceInfo(BaseModel):
    service_id: str
    state: str = Field(
        description="starting | running | exited | stopped | expired | failed | lost"
    )
    port: int | None = None
    workdir: str | None = None
    program: str | None = Field(default=None, description="Basename of argv[0]")
    pid: int | None = None
    started_at: float | None = None
    ended_at: float | None = None
    exit_code: int | None = None
    max_lifetime: int | None = None
    log_bytes: int | None = None
    log_dropped: int | None = None
    forced: bool | None = None
    name: str | None = None


class StartServiceResponse(BaseModel):
    container_name: str
    service_id: str
    ready: bool
    result: str = Field(
        description=(
            "ready | failed (see reason) | port_in_use | too_many_services | workdir_missing"
        )
    )
    reason: str | None = Field(
        default=None, description="exited_before_ready | ready_timeout | not_loopback"
    )
    url: str | None = Field(default=None, description="http://127.0.0.1:<port> when ready")
    http_status: int | None = None
    bind: str | None = Field(
        default=None, description="loopback | unverified (exposed services are refused)"
    )
    service: ServiceInfo | None = None
    log_tail: str = ""


class ServiceStatusResponse(BaseModel):
    container_name: str
    service: ServiceInfo
    log_tail: str = ""


class ServiceListResponse(BaseModel):
    container_name: str
    services: list[ServiceInfo]


class ProbeServiceRequest(BaseModel):
    method: str = Field(default="GET", json_schema_extra={"x-n8n-param-name": "probeMethod"})
    path: str = Field(default="/", json_schema_extra={"x-n8n-param-name": "probePath"})
    headers: dict[str, str] | None = Field(
        default=None, json_schema_extra={"x-n8n-param-name": "probeHeaders"}
    )
    body: str | None = Field(
        default=None,
        max_length=32 * 1024,
        description="Request body (UTF-8 text, at most 32 KiB)",
        json_schema_extra={"x-n8n-param-name": "probeBody"},
    )
    timeout: int = Field(
        default=30, ge=1, le=120, json_schema_extra={"x-n8n-param-name": "probeTimeout"}
    )


class ProbeServiceResponse(BaseModel):
    container_name: str
    service_id: str
    ok: bool = Field(description="True when an HTTP response was received (any status)")
    status: int | None = None
    reason: str | None = None
    headers: list[list[str]] = Field(default_factory=list)
    body: str = ""
    body_truncated: bool = False
    duration: float | None = None
    error: str | None = None


# ============================================================================
# HELPERS
# ============================================================================


def _workspace(container_name: str, tenant_id: str) -> Any:
    """The tenant's own running workspace, else 404 (foreign = absent)."""
    manager = get_container_manager()
    try:
        container = manager.get_container(container_name, tenant_id=tenant_id)
    except (ValueError, RuntimeError):  # fmt: skip
        container = None
    if container is None:
        raise not_found("Workspace", container_name)
    return container


def _workdir(container: Any, workdir: str) -> str:
    from src.routers.containers._common import _normalize_workspace_read_path

    root = getattr(container, "codebase_path", None) or "/workspace"
    try:
        return _normalize_workspace_read_path(workdir, workspace_root=root)
    except ValueError as e:
        raise bad_request(f"Invalid workdir: {e}") from e


async def _run_pod_program(
    container_name: str, tenant_id: str, action: str, spec: dict[str, Any], timeout: int
) -> dict[str, Any]:
    """Run one pod-program action; its single JSON line is the answer."""
    try:
        argv = pod_program_argv(action, spec)
    except ServiceSpecError as e:
        raise bad_request(str(e)) from e
    result = await get_container_manager().execute_command(
        container_name, argv, tenant_id=tenant_id, timeout=timeout
    )
    stdout = (result.get("stdout") or "").strip()
    line = stdout.splitlines()[-1] if stdout else ""
    try:
        doc = json.loads(line) if line else None
    except ValueError:
        doc = None
    if not isinstance(doc, dict):
        # stderr may echo the program; never the caller's env (it travels base64).
        logger.error(
            "workspace service %s on %s: no answer (exit=%s)",
            action,
            container_name,
            result.get("exit_code"),
        )
        raise internal_error(f"Workspace service {action} produced no answer")
    if doc.get("result") == "error":
        raise internal_error(f"Workspace service {action} failed: {doc.get('error')}")
    return doc


def _info(raw: Any, service_id: str | None = None) -> ServiceInfo:
    raw = raw if isinstance(raw, dict) else {}
    fields = {k: raw.get(k) for k in ServiceInfo.model_fields if k in raw}
    fields["service_id"] = service_id or str(raw.get("service_id") or "")
    fields["state"] = str(raw.get("state") or "unknown")
    try:
        return ServiceInfo.model_validate(fields)
    except ValueError:
        return ServiceInfo(service_id=fields["service_id"], state=fields["state"])


def _service_not_found(service_id: str) -> Exception:
    return not_found("Service", service_id)


# ============================================================================
# ROUTES
# ============================================================================


def _launch_spec(
    container: Any, request: StartServiceRequest, service_id: str, *, ready_timeout: int
) -> dict[str, Any]:
    workdir = _workdir(container, request.workdir)
    try:
        return build_launch_spec(
            service_id=service_id,
            argv=request.command,
            workdir=workdir,
            port=request.port,
            env=request.env,
            readiness_kind=request.readiness.kind,
            readiness_path=request.readiness.path,
            ready_timeout=ready_timeout,
            max_lifetime=request.max_lifetime_seconds,
            root=SERVICE_ROOT,
            toolchain=toolchain_of(container),
            name=request.name,
            max_services=max_services_of(container),
        )
    except ServiceSpecError as e:
        raise bad_request(str(e)) from e


def _refuse_admission(container_name: str, doc: dict[str, Any], port: int | None) -> None:
    """Raise the HTTP error for a launch the pod program did not admit."""
    result = str(doc.get("result"))
    if result == "port_in_use":
        raise error_response(
            status_code=status.HTTP_409_CONFLICT,
            message=f"port {doc.get('port', port)} is already in use in workspace {container_name}",
            error_code="SERVICE_PORT_IN_USE",
            details={"error_code": "SERVICE_PORT_IN_USE"},
        )
    if result == "too_many_services":
        raise error_response(
            status_code=status.HTTP_409_CONFLICT,
            message=(
                f"workspace {container_name} already runs {doc.get('running')} services "
                f"(at most {doc.get('max_services')}); stop one first"
            ),
            error_code="TOO_MANY_SERVICES",
            details={"error_code": "TOO_MANY_SERVICES"},
        )
    if result == "workdir_missing":
        raise bad_request(
            f"workdir {doc.get('workdir')} does not exist in workspace {container_name}"
        )


def _start_response(
    container_name: str, service_id: str, port: int, doc: dict[str, Any]
) -> StartServiceResponse:
    result = str(doc.get("result"))
    ready = result == "ready"
    return StartServiceResponse(
        container_name=container_name,
        service_id=service_id,
        ready=ready,
        result=result,
        reason=doc.get("reason"),
        url=f"http://127.0.0.1:{port}" if ready else None,
        http_status=doc.get("http_status"),
        bind=doc.get("bind"),
        service=_info(doc.get("service"), service_id),
        log_tail=str(doc.get("log_tail") or ""),
    )


_ADMISSION_409 = "Port in use, or too many services"


@router.post(
    "/{container_name}/services",
    response_model=StartServiceResponse,
    operation_id="start_workspace_service",
    status_code=status.HTTP_200_OK,
    summary="Start a background service in the workspace",
    responses={
        400: {"description": "Invalid command, port, env, workdir or readiness"},
        404: {"description": "Workspace not found or not owned by the caller's tenant"},
        409: {"description": _ADMISSION_409},
    },
)
async def start_workspace_service(
    container_name: str = Path(..., description="Workspace name"),
    request: StartServiceRequest = Body(...),
    tenant_id: str = Depends(require_tenant),
) -> StartServiceResponse:
    """Start ``command`` detached in ``workdir``; return once it is ready or has failed.

    A failed start (exited, never ready, bound beyond loopback) is a 200 with
    ``ready=false``, the reason and the captured log tail: for the integration
    check that is evidence about the provider, not a transport error.
    """
    container = _workspace(container_name, tenant_id)
    service_id = f"svc-{uuid.uuid4().hex[:16]}"
    spec = _launch_spec(container, request, service_id, ready_timeout=request.ready_timeout)
    logger.info(
        "start service: tenant=%s workspace=%s service=%s program=%s port=%d env_keys=%s",
        tenant_id,
        container_name,
        service_id,
        request.command[0].rsplit("/", 1)[-1],
        request.port,
        sorted((request.env or {}).keys()),
    )
    doc = await _run_pod_program(
        container_name,
        tenant_id,
        "launch",
        spec,
        timeout=request.ready_timeout + _LAUNCH_EXEC_SLACK_S,
    )
    _refuse_admission(container_name, doc, request.port)
    logger.info(
        "start service: workspace=%s service=%s result=%s reason=%s",
        container_name,
        service_id,
        doc.get("result"),
        doc.get("reason"),
    )
    return _start_response(container_name, service_id, request.port, doc)


@router.get(
    "/{container_name}/services",
    response_model=ServiceListResponse,
    operation_id="list_workspace_services",
    summary="List background services in the workspace",
)
async def list_workspace_services(
    container_name: str = Path(..., description="Workspace name"),
    tenant_id: str = Depends(require_tenant),
) -> ServiceListResponse:
    _workspace(container_name, tenant_id)
    doc = await _run_pod_program(
        container_name, tenant_id, "list", {"root": SERVICE_ROOT}, timeout=_SHORT_EXEC_S
    )
    services = [_info(s) for s in doc.get("services") or [] if isinstance(s, dict)]
    return ServiceListResponse(container_name=container_name, services=services)


@router.get(
    "/{container_name}/services/{service_id}",
    response_model=ServiceStatusResponse,
    operation_id="get_workspace_service",
    summary="Status and log tail of a workspace background service",
)
async def get_workspace_service(
    container_name: str = Path(..., description="Workspace name"),
    service_id: str = Path(..., description="Service id (svc-<16 hex>)"),
    tenant_id: str = Depends(require_tenant),
) -> ServiceStatusResponse:
    try:
        validate_service_id(service_id)
    except ServiceSpecError as e:
        raise bad_request(str(e)) from e
    _workspace(container_name, tenant_id)
    doc = await _run_pod_program(
        container_name,
        tenant_id,
        "status",
        {"root": SERVICE_ROOT, "service_id": service_id, "tail_bytes": LOG_TAIL_BYTES},
        timeout=_SHORT_EXEC_S,
    )
    if doc.get("result") == "not_found":
        raise _service_not_found(service_id)
    return ServiceStatusResponse(
        container_name=container_name,
        service=_info(doc.get("service"), service_id),
        log_tail=str(doc.get("log_tail") or ""),
    )


@router.delete(
    "/{container_name}/services/{service_id}",
    response_model=ServiceStatusResponse,
    operation_id="stop_workspace_service",
    summary="Stop a workspace background service",
)
async def stop_workspace_service(
    container_name: str = Path(..., description="Workspace name"),
    service_id: str = Path(..., description="Service id (svc-<16 hex>)"),
    tenant_id: str = Depends(require_tenant),
) -> ServiceStatusResponse:
    """SIGTERM the service's process group, SIGKILL after a grace; idempotent."""
    try:
        validate_service_id(service_id)
    except ServiceSpecError as e:
        raise bad_request(str(e)) from e
    _workspace(container_name, tenant_id)
    doc = await _run_pod_program(
        container_name,
        tenant_id,
        "stop",
        {
            "root": SERVICE_ROOT,
            "service_id": service_id,
            "tail_bytes": LOG_TAIL_BYTES,
            "stop_grace": STOP_GRACE_SECONDS,
        },
        timeout=_SHORT_EXEC_S,
    )
    if doc.get("result") == "not_found":
        raise _service_not_found(service_id)
    logger.info("stop service: workspace=%s service=%s", container_name, service_id)
    return ServiceStatusResponse(
        container_name=container_name,
        service=_info(doc.get("service"), service_id),
        log_tail=str(doc.get("log_tail") or ""),
    )


#: RFC 7230 ``token`` (field-name) charset, ASCII only.
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}$")


def _latin1(value: str) -> bool:
    """http.client encodes header values as latin-1; anything else would fail in the pod."""
    try:
        value.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return True


def _probe_headers(headers: dict[str, str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in (headers or {}).items():
        if len(out) >= _PROBE_HEADER_LIMIT:
            raise bad_request(f"at most {_PROBE_HEADER_LIMIT} headers")
        if not _HEADER_NAME_RE.match(key or ""):
            raise bad_request(f"invalid header name: {key!r}")
        if key.lower() in ("host", "content-length", "transfer-encoding", "connection"):
            raise bad_request(f"header {key} is set by the probe and may not be supplied")
        if len(value) > 4096 or any(c in value for c in "\r\n\x00") or not _latin1(value):
            raise bad_request(f"invalid value for header {key}")
        out[key] = value
    return out


@router.post(
    "/{container_name}/services/{service_id}/probe",
    response_model=ProbeServiceResponse,
    operation_id="probe_workspace_service",
    summary="Send one HTTP request to a workspace background service",
)
async def probe_workspace_service(
    container_name: str = Path(..., description="Workspace name"),
    service_id: str = Path(..., description="Service id (svc-<16 hex>)"),
    request: ProbeServiceRequest = Body(...),
    tenant_id: str = Depends(require_tenant),
) -> ProbeServiceResponse:
    """One request to ``http://127.0.0.1:<service port><path>``, made inside the pod.

    ``ok`` means a response arrived (any status — a 404 from the provider is
    evidence, not a failure of the probe); the body is capped at 64 KiB.
    """
    try:
        validate_service_id(service_id)
        path = validate_http_path(request.path)
    except ServiceSpecError as e:
        raise bad_request(str(e)) from e
    method = request.method.upper()
    if method not in _HTTP_METHODS:
        raise bad_request(f"method must be one of {', '.join(_HTTP_METHODS)}")
    headers = _probe_headers(request.headers)
    _workspace(container_name, tenant_id)
    spec = {
        "root": SERVICE_ROOT,
        "service_id": service_id,
        "tail_bytes": 0,
        "method": method,
        "path": path,
        "headers": headers,
        "body": base64.b64encode(request.body.encode()).decode() if request.body else None,
        "timeout": request.timeout,
        "body_cap": PROBE_BODY_CAP,
    }
    doc = await _run_pod_program(
        container_name, tenant_id, "probe", spec, timeout=request.timeout + _SHORT_EXEC_S
    )
    result = doc.get("result")
    if result == "not_found":
        raise _service_not_found(service_id)
    if result == "not_running":
        state = (doc.get("service") or {}).get("state")
        raise error_response(
            status_code=status.HTTP_409_CONFLICT,
            message=f"service {service_id} is not running (state={state})",
            error_code="SERVICE_NOT_RUNNING",
            details={"error_code": "SERVICE_NOT_RUNNING"},
        )
    if result == "ok":
        return ProbeServiceResponse(
            container_name=container_name,
            service_id=service_id,
            ok=True,
            status=doc.get("status"),
            reason=doc.get("reason"),
            headers=[[str(k), str(v)] for k, v in doc.get("headers") or []],
            body=str(doc.get("body") or ""),
            body_truncated=bool(doc.get("body_truncated")),
            duration=doc.get("duration"),
        )
    return ProbeServiceResponse(
        container_name=container_name,
        service_id=service_id,
        ok=False,
        error=str(doc.get("error") or "probe failed"),
        duration=doc.get("duration"),
    )

"""Validation and spec building for workspace background services.

Everything a caller supplies is validated HERE, server-side, before a byte reaches
the pod; the pod program (:mod:`.pod_program`) then receives one base64 JSON spec
as a single argv word. No caller string is ever interpolated into a shell: the
exec argv is ``["python3", "-I", "-c", <program source>, <action>, <spec>]`` and
the provisioner quotes it with ``shlex.join``.
"""

from __future__ import annotations

import base64
import json
import re
from functools import cache
from pathlib import Path
from typing import Any

from src.core.toolchain import MAX_SERVICES_PER_WORKSPACE
from src.runtime import check_dangerous_patterns, validate_command
from src.runtime.policy import env_assignment_refusal

#: Where per-service state, pid and log files live in the pod. ``/tmp`` is the
#: pod's own emptyDir (never inside the repository, so no git route sees it).
SERVICE_ROOT = "/tmp/.coding-services"  # noqa: S108  # nosec B108

#: How many services may be live in one workspace at once: a plain count cap
#: (``MAX_SERVICES_PER_WORKSPACE`` = 4, or ``MAX_EXPECTED_SERVICES`` for a workspace
#: sized by a services hint — ``toolchain.max_services_of``). Memory is not
#: admitted in the pod: the pod is sized at create from the hints and its memory
#: limit is the bound (an OOM kill).

#: SAFETY CEILING, not a work budget (George 2026-10-11: no time budgets). The
#: supervisor stops a service that is still running this long after its start, so a
#: forgotten provider cannot hold the pod's CPU/memory for the workspace's whole
#: life. Same ceiling as async exec. Deleting the workspace stops everything sooner.
DEFAULT_MAX_LIFETIME_SECONDS = 4 * 60 * 60

#: Bytes of service output kept in the pod; the rest is counted and dropped.
LOG_CAP_BYTES = 8 * 1024 * 1024
#: Bytes of log tail returned in a response.
LOG_TAIL_BYTES = 8 * 1024
#: Seconds between SIGTERM and SIGKILL of the service's process group.
STOP_GRACE_SECONDS = 5
#: Probe response body bytes returned.
PROBE_BODY_CAP = 64 * 1024
#: Largest spec handed to the pod (one argv word; Linux caps one at 128 KiB).
MAX_SPEC_BYTES = 64 * 1024

#: Never a provider port: 8000 is reserved estate-wide (operator rule).
RESERVED_PORTS = frozenset({8000})
MIN_PORT = 1024
MAX_PORT = 65535

_ID_RE = re.compile(r"^svc-[0-9a-f]{16}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_MAX_ENV_VARS = 32
_MAX_ENV_VALUE = 4096
_MAX_ARGV = 64
_MAX_ARG_CHARS = 4096

_PROGRAM_PATH = Path(__file__).with_name("pod_program.py")


class ServiceSpecError(ValueError):
    """A caller-supplied service field is invalid (→ HTTP 400)."""


@cache
def pod_program_source() -> str:
    """Source of :mod:`.pod_program`, run in the pod with ``python3 -I -c``."""
    return _PROGRAM_PATH.read_text(encoding="utf-8")


def pod_program_argv(action: str, spec: dict[str, Any]) -> list[str]:
    """The exec argv for one pod-program action. ``spec`` travels as one base64 word."""
    raw = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    if len(raw) > MAX_SPEC_BYTES:
        raise ServiceSpecError(f"request too large ({len(raw)} bytes > {MAX_SPEC_BYTES})")
    return [
        "python3",
        "-I",
        "-c",
        pod_program_source(),
        action,
        base64.b64encode(raw).decode("ascii"),
    ]


def validate_service_id(service_id: str) -> str:
    if not _ID_RE.match(service_id or ""):
        raise ServiceSpecError("service_id must look like svc-<16 hex>")
    return service_id


def validate_port(port: int) -> int:
    if not isinstance(port, int) or isinstance(port, bool):
        raise ServiceSpecError("port must be an integer")
    if port in RESERVED_PORTS:
        raise ServiceSpecError(f"port {port} is reserved and may never be used")
    if not MIN_PORT <= port <= MAX_PORT:
        raise ServiceSpecError(f"port must be {MIN_PORT}-{MAX_PORT}")
    return port


def validate_env(env: dict[str, str] | None) -> dict[str, str]:
    """Validated env; values are never logged or echoed (they may be secrets)."""
    if not env:
        return {}
    if len(env) > _MAX_ENV_VARS:
        raise ServiceSpecError(f"at most {_MAX_ENV_VARS} environment variables")
    out: dict[str, str] = {}
    for key, value in env.items():
        if not _ENV_KEY_RE.match(key):
            raise ServiceSpecError(f"invalid environment variable name: {key!r}")
        if not isinstance(value, str) or len(value) > _MAX_ENV_VALUE or "\x00" in value:
            raise ServiceSpecError(
                f"environment variable {key}: value must be a string of at most "
                f"{_MAX_ENV_VALUE} characters without NUL"
            )
        # ONE rule for every door a caller sets environment through (exec's env
        # field, ``env K=V`` argv, and this): src.runtime.policy.
        refusal = env_assignment_refusal(key)
        if refusal:
            raise ServiceSpecError(f"environment variable {refusal}")
        out[key] = value
    return out


def validate_name(name: str | None) -> str | None:
    """Optional caller label of a service (e.g. the source repository's name)."""
    if name is None:
        return None
    if not _NAME_RE.match(name):
        raise ServiceSpecError("name must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    return name


def validate_argv(argv: list[str], toolchain: str = "python") -> list[str]:
    """Exactly the exec route's policy (allowlist + dangerous patterns), plus size.

    ``toolchain`` is the WORKSPACE's (``src.core.toolchain.toolchain_of``), so a
    python+rust workspace's services may run ``ws-pg`` / cargo tools exactly as
    its exec route may; the default keeps the narrower python policy.
    """
    if not argv:
        raise ServiceSpecError("command must not be empty")
    if len(argv) > _MAX_ARGV or any(
        not isinstance(a, str) or len(a) > _MAX_ARG_CHARS or "\x00" in a for a in argv
    ):
        raise ServiceSpecError(
            f"command: at most {_MAX_ARGV} arguments of at most {_MAX_ARG_CHARS} characters"
        )
    if not validate_command(argv, toolchain=toolchain):
        raise ServiceSpecError(
            f"Command '{argv[0]}' not allowed. Only whitelisted commands can be executed."
        )
    if not check_dangerous_patterns(argv):
        raise ServiceSpecError("Command contains dangerous patterns and is not allowed")
    return list(argv)


_HTTP_PATH_RE = re.compile(r"^/[\x21-\x7e]{0,511}$")


def validate_http_path(path: str) -> str:
    """An origin-form request target: starts with ``/``, printable ASCII, no spaces."""
    if not _HTTP_PATH_RE.match(path or ""):
        raise ServiceSpecError(
            "path must start with '/' and contain only printable ASCII without spaces"
        )
    return path


def build_launch_spec(
    *,
    service_id: str,
    argv: list[str],
    workdir: str,
    port: int,
    env: dict[str, str] | None,
    readiness_kind: str,
    readiness_path: str | None,
    ready_timeout: int,
    max_lifetime: int,
    root: str = SERVICE_ROOT,
    toolchain: str = "python",
    name: str | None = None,
    max_services: int = MAX_SERVICES_PER_WORKSPACE,
) -> dict[str, Any]:
    if readiness_kind not in ("http", "tcp"):
        raise ServiceSpecError("readiness.kind must be 'http' or 'tcp'")
    readiness: dict[str, Any] = {"kind": readiness_kind}
    if readiness_kind == "http":
        readiness["path"] = validate_http_path(readiness_path or "/")
    return {
        "action": "launch",
        "service_id": validate_service_id(service_id),
        "root": root,
        "argv": validate_argv(argv, toolchain=toolchain),
        "workdir": workdir,
        "port": validate_port(port),
        "env": validate_env(env),
        "readiness": readiness,
        "ready_timeout": ready_timeout,
        "max_lifetime": max_lifetime,
        "max_services": max_services,
        "name": validate_name(name),
        "log_cap": LOG_CAP_BYTES,
        "tail_bytes": LOG_TAIL_BYTES,
        "stop_grace": STOP_GRACE_SECONDS,
    }

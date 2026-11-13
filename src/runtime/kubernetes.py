"""Kubernetes-backed workspace runtime — the production path."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from src.core.toolchain import toolchain_of

from .policy import enforce_command_policy
from .protocol import DEFAULT_OUTPUT_LIMIT
from .types import (
    ExecutionProfile,
    RunOnceResult,
    RuntimeCapabilities,
    RuntimeKind,
    UnsupportedCapabilityError,
)

logger = logging.getLogger(__name__)

__all__ = ["KubernetesRuntime"]

#: GNU coreutils `timeout` exits 124 when it kills the command. The provisioner
#: wraps every exec in `timeout N ...`, so this is how a wall-clock kill is told
#: apart from the command choosing to exit 124 itself — which it may, and which
#: is why `timed_out` is a hint rather than a guarantee.
TIMEOUT_EXIT_CODE = 124


class KubernetesRuntime:
    """Runs commands via `kubectl exec` against the workspace Deployment's pod."""

    kind = RuntimeKind.KUBERNETES

    def __init__(self, provisioner: Any) -> None:
        self._provisioner = provisioner

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            kind=self.kind,
            # The exec stream CAN carry stdin, but `exec_in_pod` opens it with
            # stdin=False and nothing wires a payload through yet. Advertised as
            # False because that is what is true today; claiming it and dropping
            # the bytes would run the command without its input and return a
            # plausible result for a question never asked.
            stdin=False,
            # Bounded inside the read loop, so the cap protects this process's
            # memory and not just the response body.
            incremental_output_limit=True,
        )

    async def run_once(
        self,
        workspace: Any,
        argv: Sequence[str],
        *,
        cwd: str = "/workspace",
        env: Mapping[str, str] | None = None,
        stdin: bytes | None = None,
        timeout: int = 30,
        output_limit: int = DEFAULT_OUTPUT_LIMIT,
        execution_profile: ExecutionProfile = ExecutionProfile.SERVER_AUTHORED,
    ) -> RunOnceResult:
        if stdin is not None:
            raise UnsupportedCapabilityError(self.kind, "stdin")

        # Second layer. The API boundary checks first and answers 400 there;
        # this holds for any future caller that reaches a backend another way.
        enforce_command_policy(argv, execution_profile, toolchain=toolchain_of(workspace), env=env)

        command = with_env(list(argv), env)
        name = _workspace_name(workspace)

        # exec_in_pod is a blocking k8s stream. Off the event loop or the
        # /health liveness probe times out and kubelet SIGKILLs a busy-but-
        # healthy pod mid-exec.
        result = await asyncio.to_thread(
            self._provisioner.exec_in_pod,
            deployment_name=name,
            command=command,
            workdir=cwd,
            timeout=timeout,
            max_output_bytes=output_limit,
        )
        return _to_result(result)


def _workspace_name(workspace: Any) -> str:
    """The Deployment name, whether given an object or a bare string."""
    if isinstance(workspace, str):
        return workspace
    name = getattr(workspace, "container_name", None)
    if not name:
        raise ValueError(f"workspace has no container_name: {workspace!r}")
    return str(name)


def with_env(argv: list[str], env: Mapping[str, str] | None) -> list[str]:
    """Prefix `/usr/bin/env K=V ...` when extra environment is requested.

    The Kubernetes exec API has no env parameter — unlike Docker's — so the only
    way to set one per-command is to run the command *through* env.

    Values are emitted RAW. An earlier version shell-quoted them here, with a
    docstring claiming that was needed "because the provisioner joins the list
    into a `sh -c` string" — which is exactly backwards. ``exec_in_pod`` runs the
    list through ``shlex.join``, so pre-quoting is quoted a SECOND time and the
    quotes arrive as data. Measured: ``env={"A": "b c"}`` reached the container
    as ``A='b c'``, literal single quotes included, so anything reading ``$A``
    got the wrong value and nothing failed loudly.

    Building an argv list and letting exactly one layer quote it is the whole
    contract; quoting defensively at both ends is how you get it wrong twice.
    """
    if not env:
        return argv
    for key in env:
        # A key that is empty or contains '=' produces a token env(1) cannot
        # parse as an assignment — it is taken as the command to RUN instead,
        # failing in a confusing way rather than an obvious one.
        if not key or "=" in key:
            raise ValueError(f"invalid environment variable name: {key!r}")
    prefix = ["/usr/bin/env"]
    prefix += [f"{key}={value}" for key, value in sorted(env.items())]
    return prefix + argv


def _to_result(raw: dict[str, Any]) -> RunOnceResult:
    exit_code = int(raw.get("exit_code", -1))
    omitted = int(raw.get("bytes_omitted", 0) or 0)
    truncated = bool(raw.get("truncated"))
    return RunOnceResult(
        exit_code=exit_code,
        stdout=raw.get("stdout", "") or "",
        stderr=raw.get("stderr", "") or "",
        duration=float(raw.get("duration", 0.0) or 0.0),
        # The provisioner reports one combined flag; attributing it to both
        # streams is the honest widening. Claiming it for stdout alone would
        # assert something not measured.
        stdout_truncated=truncated,
        stderr_truncated=truncated,
        bytes_omitted=omitted,
        timed_out=exit_code == TIMEOUT_EXIT_CODE,
    )

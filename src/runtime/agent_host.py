"""Agent-host workspace runtime — an HTTP exec shim on a remote host."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from src.core.toolchain import toolchain_of

from .policy import enforce_command_policy
from .protocol import DEFAULT_OUTPUT_LIMIT
from .types import (
    WORKSPACE_EXEC_USER,
    ExecutionProfile,
    RunOnceResult,
    RuntimeCapabilities,
    RuntimeKind,
    UnsupportedCapabilityError,
)

logger = logging.getLogger(__name__)

__all__ = ["AgentHostRuntime"]

TIMEOUT_EXIT_CODE = 124

# The identity comes from src.runtime.types so all three backends cite the same
# constant. The agent host's API still takes a `user` field, so something must
# be sent; it is fixed rather than exposed, and is deliberately not a `run_once`
# parameter — see the protocol docstring.


class AgentHostRuntime:
    """Runs commands by POSTing to the agent host's exec endpoint."""

    kind = RuntimeKind.AGENT_HOST

    def __init__(self, core: Any) -> None:
        self._core = core

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            kind=self.kind,
            # The agent host's exec body has no stdin field at all.
            stdin=False,
            # It returns a fully-buffered JSON body; the cap can only be applied
            # on receipt, so it bounds what a caller sees, not what was read.
            incremental_output_limit=False,
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
        if env:
            # No env field in the exec body. Refused rather than dropped: a
            # command that silently loses its environment usually still runs,
            # and returns a confidently wrong answer.
            raise UnsupportedCapabilityError(self.kind, "env")

        # Second layer. The API boundary checks first and answers 400 there;
        # this holds for any future caller that reaches a backend another way.
        enforce_command_policy(argv, execution_profile, toolchain=toolchain_of(workspace))

        name = _workspace_name(workspace)
        # exec_in_container_via_agent_host uses blocking httpx, so it goes off
        # the event loop like the other backends.
        raw = await asyncio.to_thread(
            self._core.exec_in_container_via_agent_host,
            name=name,
            command=list(argv),
            workdir=cwd,
            timeout=timeout,
            user=WORKSPACE_EXEC_USER,
        )

        stdout, out_omitted = _clip(raw.get("stdout") or "", output_limit)
        stderr, err_omitted = _clip(raw.get("stderr") or "", output_limit)
        exit_code = int(raw.get("exit_code", -1))
        return RunOnceResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration=float(raw.get("duration", 0.0) or 0.0),
            stdout_truncated=out_omitted > 0,
            stderr_truncated=err_omitted > 0,
            bytes_omitted=out_omitted + err_omitted,
            timed_out=exit_code == TIMEOUT_EXIT_CODE,
        )


def _workspace_name(workspace: Any) -> str:
    if isinstance(workspace, str):
        return workspace
    name = getattr(workspace, "container_name", None)
    if not name:
        raise ValueError(f"workspace has no container_name: {workspace!r}")
    return str(name)


def _clip(text: str, limit: int) -> tuple[str, int]:
    """Bound an already-decoded stream to `limit` BYTES.

    Measured in bytes, not characters: the budget protects the response body,
    and a character count understates a multi-byte stream by up to 4x.
    """
    if not text:
        return "", 0
    encoded = text.encode("utf-8", "replace")
    if len(encoded) <= limit:
        return text, 0
    kept = encoded[:limit].decode("utf-8", "ignore")
    return kept, len(encoded) - len(kept.encode("utf-8", "replace"))

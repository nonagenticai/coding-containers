"""Docker-backed workspace runtime — development and compose only.

Production is Kubernetes. This exists so the same code path works on a laptop,
and so the conformance suite has a second backend to prove the interface is
actually an interface rather than one implementation with a protocol bolted on.
"""

from __future__ import annotations

import asyncio
import logging
import time
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

__all__ = ["DockerRuntime"]

TIMEOUT_EXIT_CODE = 124


class DockerRuntime:
    """Runs commands via `docker exec` against a local container."""

    kind = RuntimeKind.DOCKER

    def __init__(self, client: Any) -> None:
        self._client = client

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            kind=self.kind,
            # `exec_run` has no stdin channel. Reaching it needs the low-level
            # exec_start(socket=True) API, which nothing here uses.
            stdin=False,
            # `exec_run` returns a fully-buffered result, so the cap can only be
            # applied afterwards. It bounds the response, not this process's
            # peak memory — declared rather than glossed, because a caller
            # choosing a backend for a huge output should know the difference.
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

        # Second layer. The API boundary checks first and answers 400 there;
        # this holds for any future caller that reaches a backend another way.
        enforce_command_policy(argv, execution_profile, toolchain=toolchain_of(workspace), env=env)

        name = _workspace_name(workspace)
        # `timeout` is enforced inside the container, as on the Kubernetes path,
        # so both backends fail the same way rather than one killing the command
        # and the other abandoning a still-running one.
        command = ["timeout", str(timeout), *argv]

        started = time.time()
        result = await asyncio.to_thread(self._exec, name, command, cwd, dict(env) if env else None)
        duration = time.time() - started

        raw_out, raw_err = _demux(result.output)
        stdout, out_omitted = _clip(raw_out, output_limit)
        stderr, err_omitted = _clip(raw_err, output_limit)
        exit_code = int(result.exit_code if result.exit_code is not None else -1)
        return RunOnceResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration=duration,
            stdout_truncated=out_omitted > 0,
            stderr_truncated=err_omitted > 0,
            bytes_omitted=out_omitted + err_omitted,
            timed_out=exit_code == TIMEOUT_EXIT_CODE,
        )

    def _exec(self, name: str, command: list[str], cwd: str, env: dict[str, str] | None) -> Any:
        # No `user=`: the identity comes from the image's USER, deliberately.
        # Kubernetes pins uid 1000 through the pod security context and the
        # runtime image declares the same user, so production agrees either way
        # (measured: codegraph-runtime runs as 1000/coder by default). Forcing
        # 1000 here would instead break a compose setup running some other
        # image, and this backend is dev-only. The invariant that matters is
        # pinned by test_the_identity_constants_agree.
        container = self._client.containers.get(name)
        return container.exec_run(
            cmd=command,
            workdir=cwd,
            environment=env,
            demux=True,
        )


def _workspace_name(workspace: Any) -> str:
    if isinstance(workspace, str):
        return workspace
    name = getattr(workspace, "container_name", None)
    if not name:
        raise ValueError(f"workspace has no container_name: {workspace!r}")
    return str(name)


def _demux(output: Any) -> tuple[bytes | None, bytes | None]:
    """Split `exec_run`'s output into (stdout, stderr).

    With `demux=True` docker-py returns a 2-tuple. Without it, a single blob
    with both streams interleaved — which is what `ContainerManager` used to
    request, and is the reason stderr silently arrived inside stdout on that
    path while the /exec router (which did pass demux=True) kept them apart.
    Unifying on demux is the fix; tolerating the flat shape as well means a
    docker-py version that ignores the flag degrades to the old merged
    behaviour instead of raising TypeError on a subscript.
    """
    if output is None:
        return None, None
    if isinstance(output, (bytes, bytearray)):
        return bytes(output), None
    try:
        return output[0], output[1]
    except TypeError, IndexError, KeyError:
        logger.warning("Unexpected exec_run output shape: %r", type(output))
        return None, None


def _clip(raw: bytes | None, limit: int) -> tuple[str, int]:
    """Decode one stream bounded to `limit` BYTES.

    Bounds the raw bytes before decoding: `errors="replace"` can grow a stream
    of invalid bytes on decode, so measuring after would measure the wrong
    thing. The tail is dropped with `errors="ignore"` so a multi-byte character
    straddling the boundary vanishes rather than becoming a replacement char.
    """
    if not raw:
        return "", 0
    if len(raw) <= limit:
        return raw.decode("utf-8", errors="replace"), 0
    return raw[:limit].decode("utf-8", errors="ignore"), len(raw) - limit

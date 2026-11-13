"""The one interface every workspace backend implements.

Before this existed, backend selection happened in two places that disagreed.
``src/routers/container_exec.py`` preferred Kubernetes whenever the workspace's
endpoint contained ``.svc.cluster.local``; ``src/core/container_manager.py``
preferred Docker and reached Kubernetes only when Docker was *absent*. In
cluster, with a working Docker socket, the same workspace therefore ran on a
different backend depending on which entry point the caller came through — and
only one of the two paths applied the command allowlist.

Neither precedence was wrong on its own. Having two was.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from .types import ExecutionProfile, RunOnceResult, RuntimeCapabilities, RuntimeKind

__all__ = ["WorkspaceRuntime"]

#: Per-stream byte budget applied when a caller does not specify one. Matches
#: the provisioner's internal ceiling; public routes pass something smaller.
DEFAULT_OUTPUT_LIMIT = 16 * 1024 * 1024


@runtime_checkable
class WorkspaceRuntime(Protocol):
    """Runs bounded, server- or caller-authored commands in one workspace.

    Deliberately narrow. ``run_once`` covers everything the service does today;
    long-lived process supervision (``start_process`` / ``stream_process_events``
    / ``write_process_stdin``) belongs to the async-process phase and is not
    stubbed here, because an unimplemented method on a Protocol is a promise
    callers will believe.

    There is no ``user`` parameter, and that is a decision rather than an
    omission: the Kubernetes exec API cannot honour one, so accepting it would
    mean silently ignoring it on the production backend. Identity comes from the
    pod security context.
    """

    kind: RuntimeKind

    def capabilities(self) -> RuntimeCapabilities:
        """What this backend supports. Cheap; safe to call per request."""
        ...

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
        """Run one command to completion and return its bounded output.

        Args:
            workspace: The ``ContainerInfo`` (or equivalent) identifying the
                target. Passed whole rather than as a name so a backend can read
                whichever address it actually needs.
            argv: Command as a list. Never a shell string — a single string is
                how an interpolated path with a space or a quote becomes two
                arguments.
            cwd: Working directory inside the workspace.
            env: Extra environment for this command only.
            stdin: Bytes to feed the command. Raises ``UnsupportedCapabilityError``
                on a backend whose ``capabilities().stdin`` is False.
            timeout: Wall-clock budget, enforced inside the workspace.
            output_limit: Per-stream byte budget. Exceeding it truncates and is
                reported on the result, never silently.
            execution_profile: Whether ``argv`` is caller-supplied and therefore
                subject to the command allowlist.

        Returns:
            A ``RunOnceResult``. A non-zero exit is a normal return, not an
            exception: the command failing is data the caller asked for.
            Exceptions are reserved for the backend being unable to run it.
        """
        ...

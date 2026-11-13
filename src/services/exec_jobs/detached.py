"""Detached async exec: the command outlives the coding-containers replica.

A Kubernetes workspace command runs in the WORKSPACE pod, not in this service, so
there is no reason for it to die with the replica that started it. Instead of one
blocking ``kubectl exec`` held open for the whole run (which a rollout cuts, and
the job then ends ``interrupted``/``orphaned`` with its outcome unknown), the job:

1. records its handle — the workspace pod name and a job directory in that
   pod's ``/tmp`` emptyDir — on the job row BEFORE launching;
2. launches a wrapper with ``setsid`` (``nohup`` if absent) through one short
   exec; the wrapper writes its pid, capped stdout/stderr, the byte counts it
   dropped, start/end times and finally the exit code (atomic rename) into the
   job directory;
3. is polled with short execs by whichever replica holds the job's lease. If
   that replica dies or shuts down, any surviving replica's reaper ADOPTS the
   row (one conditional UPDATE, one winner), keeps polling, completes the row
   and sends the callback through the existing single-sender hand-off.

Output never grows without bound in the pod: each stream is written through
``head -c CAP`` and the remainder is counted (``wc -c``) and discarded, so a job
directory holds at most ``2 * CAP`` bytes plus a few counters.

Only the Kubernetes runtime is detachable. Docker / agent-host workspaces keep
the blocking path (``DetachedBackend.locate`` returns ``None``).
"""

from __future__ import annotations

import asyncio
import enum
import logging
import shlex
from dataclasses import dataclass
from typing import Any, Protocol

from src.core.toolchain import toolchain_of
from src.runtime import ExecutionProfile, enforce_command_policy
from src.runtime.kubernetes import with_env

logger = logging.getLogger(__name__)

#: Root of every job directory in the workspace pod (the ``tmp`` emptyDir).
JOB_ROOT = "/tmp/.coding-exec-jobs"
#: How long the launch exec itself may take; the command is detached by then.
LAUNCH_EXEC_TIMEOUT_S = 30
#: Bound on one status / collect / cleanup exec.
PROBE_EXEC_TIMEOUT_S = 20
#: A missing job directory (or pid file) is only a verdict once the handle is at
#: least this old: until then the launch may still be in flight on the replica
#: that recorded the handle (it records BEFORE launching).
LAUNCH_GRACE_S = 60.0
TIMEOUT_EXIT_CODE = 124
WORKSPACE_GONE_ERROR = (
    "workspace_gone: the workspace pod running this detached command no longer "
    "exists (deleted, evicted or replaced); the command died with it and its "
    "outcome is unknown"
)
LOST_ERROR = (
    "lost: the detached command's wrapper is no longer running in the workspace "
    "pod and left no exit status (the workspace container restarted)"
)
NEVER_STARTED_ERROR = (
    "never_started: the replica that recorded this job stopped before the "
    "command was launched in the workspace pod; it did not run"
)


class WorkspaceGoneError(RuntimeError):
    """The pod named in a detached handle no longer exists or is terminating."""


class PollState(enum.StrEnum):
    RUNNING = "running"
    DONE = "done"
    MISSING = "missing"  # no job directory
    LOST = "lost"  # directory present, wrapper dead, no exit status


@dataclass(frozen=True)
class DetachedHandle:
    container_name: str
    pod: str
    job_dir: str


@dataclass(frozen=True)
class PollResult:
    state: PollState
    exit_code: int | None = None
    stdout_omitted: int = 0
    stderr_omitted: int = 0
    started: int | None = None  # pod clock, epoch seconds
    ended: int | None = None


@dataclass(frozen=True)
class DetachedOutcome:
    exit_code: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    bytes_omitted: int
    timed_out: bool
    duration: float | None


def job_dir(job_id: str) -> str:
    # job_id is a server-minted uuid7; quoted anyway wherever it reaches a shell.
    return f"{JOB_ROOT}/{job_id}"


def build_run_script(d: str, argv: list[str], cwd: str, timeout: int, cap: int) -> str:
    """The wrapper that runs IN the workspace pod (POSIX sh + coreutils).

    The streams go through FIFOs into ``head -c CAP`` and ``wc -c``: the command
    never blocks on a full pipe, at most CAP bytes per stream reach disk, and
    the number of dropped bytes is exact (``head -c`` never over-reads).
    ``exit_code`` is written last, via rename, so its presence means "done".
    """
    qd = shlex.quote(d)
    return "\n".join(
        [
            f"d={qd}",
            'echo $$ > "$d/pid"',
            'date +%s > "$d/started"',
            'mkfifo "$d/o" "$d/e"',
            f'{{ head -c {int(cap)} > "$d/stdout"; wc -c > "$d/stdout.omitted"; }} < "$d/o" &',
            "r1=$!",
            f'{{ head -c {int(cap)} > "$d/stderr"; wc -c > "$d/stderr.omitted"; }} < "$d/e" &',
            "r2=$!",
            'exec 3>"$d/o" 4>"$d/e"',
            f"( cd {shlex.quote(cwd)} && exec timeout {int(timeout)} {shlex.join(argv)} )"
            " >&3 2>&4 </dev/null",
            "rc=$?",
            "exec 3>&- 4>&-",
            'wait "$r1" "$r2"',
            'date +%s > "$d/ended"',
            'printf "%s\\n" "$rc" > "$d/exit_code.tmp" && mv "$d/exit_code.tmp" "$d/exit_code"',
            'rm -f "$d/o" "$d/e"',
            "",
        ]
    )


def build_launcher(d: str, run_script: str, env: dict[str, str] | None = None) -> list[str]:
    """argv for the short launch exec: write the wrapper, start it detached.

    ``env`` (the caller's per-command environment) is applied to THIS exec as
    ``/usr/bin/env K=V /bin/sh -c …`` argv — the same mechanism as the sync path —
    and inherited by the detached wrapper and the command. It is never written into
    ``run.sh``: the job directory lives for the job's whole life in the pod's
    emptyDir, and a value may be a credential (a database URL with its password).
    """
    qd = shlex.quote(d)
    # ``launched`` is printed only when every step before it succeeded: the
    # launch is judged by that marker, never by the exec's exit code alone.
    sh = (
        f"umask 077 && mkdir -p {qd} && printf '%s' {shlex.quote(run_script)} > {qd}/run.sh"
        " && { if command -v setsid >/dev/null 2>&1;"
        f" then setsid /bin/sh {qd}/run.sh </dev/null >/dev/null 2>&1 &"
        f" else nohup /bin/sh {qd}/run.sh </dev/null >/dev/null 2>&1 & fi; }}"
        " && echo launched"
    )
    return with_env(["/bin/sh", "-c", sh], env)


def build_poll(d: str) -> list[str]:
    """argv printing the job's state; the first line is the PollState."""
    qd = shlex.quote(d)
    sh = (
        f"d={qd}; "
        'meta() { cat "$d/exit_code"; cat "$d/stdout.omitted" 2>/dev/null || echo 0; '
        'cat "$d/stderr.omitted" 2>/dev/null || echo 0; '
        'cat "$d/started" 2>/dev/null || echo; cat "$d/ended" 2>/dev/null || echo; }; '
        'if [ -f "$d/exit_code" ]; then echo done; meta; '
        'elif [ ! -d "$d" ]; then echo missing; '
        'elif [ ! -f "$d/pid" ]; then echo running; '
        'elif kill -0 "$(cat "$d/pid")" 2>/dev/null; then echo running; '
        'elif [ -f "$d/exit_code" ]; then echo done; meta; '
        "else echo lost; fi"
    )
    return ["/bin/sh", "-c", sh]


def build_collect(d: str) -> list[str]:
    qd = shlex.quote(d)
    return ["/bin/sh", "-c", f"cat {qd}/stdout 2>/dev/null; cat {qd}/stderr >&2 2>/dev/null; true"]


def build_cleanup(d: str) -> list[str]:
    return ["rm", "-rf", "--", d]


def _int(text: str | None, default: int | None = 0) -> int | None:
    try:
        return int((text or "").strip())
    except ValueError:
        return default


def parse_poll(stdout: str) -> PollResult:
    lines = stdout.splitlines()
    head = lines[0].strip() if lines else ""
    if head == PollState.DONE.value:
        meta = [*lines[1:], "", "", "", "", ""]
        return PollResult(
            state=PollState.DONE,
            exit_code=_int(meta[0], None),
            stdout_omitted=_int(meta[1]) or 0,
            stderr_omitted=_int(meta[2]) or 0,
            started=_int(meta[3], None),
            ended=_int(meta[4], None),
        )
    try:
        return PollResult(state=PollState(head))
    except ValueError:
        raise RuntimeError(f"unparseable detached status: {head[:80]!r}") from None


class DetachedBackend(Protocol):
    async def locate(self, container_name: str, tenant_id: str) -> str | None: ...

    async def launch(
        self,
        handle: DetachedHandle,
        argv: list[str],
        *,
        cwd: str,
        timeout: int,
        cap: int,
        env: dict[str, str] | None = None,
    ) -> None: ...

    async def poll(self, handle: DetachedHandle) -> PollResult: ...

    async def collect(
        self, handle: DetachedHandle, status: PollResult, *, cap: int
    ) -> DetachedOutcome: ...

    async def cleanup(self, handle: DetachedHandle) -> None: ...


class K8sDetachedBackend:
    """Detached exec through the workspace pod's exec API."""

    def __init__(self, provisioner: Any = None, container_manager: Any = None) -> None:
        self._provisioner = provisioner
        self._manager = container_manager
        self._toolchain_by_pod: dict[str, str] = {}

    def _prov(self) -> Any:
        if self._provisioner is None:
            from src.core.kubernetes_container_provisioner import get_k8s_container_provisioner

            self._provisioner = get_k8s_container_provisioner()
        return self._provisioner

    async def locate(self, container_name: str, tenant_id: str) -> str | None:
        """The pod to launch in, or ``None`` when the workspace is not Kubernetes."""
        from src.core.container_manager import get_container_manager
        from src.core.container_manager_core import get_container_manager as get_core
        from src.runtime import RuntimeKind, resolve_runtime_kind

        manager = self._manager or get_container_manager()
        info = manager.get_container(container_name, tenant_id=tenant_id)
        if info is None:
            return None  # the blocking path answers the honest 404
        if resolve_runtime_kind(info, core=get_core()) is not RuntimeKind.KUBERNETES:
            return None
        pod = await asyncio.to_thread(self._prov().resolve_running_pod, container_name)
        if pod is not None:
            # Remembered for the launch() that follows on this executor, which
            # only gets a DetachedHandle (no ContainerInfo). A miss — e.g. a job
            # adopted after a restart — falls back to python, the NARROWER policy.
            self._toolchain_by_pod[pod] = toolchain_of(info)
        return pod

    async def _exec(
        self, handle: DetachedHandle, argv: list[str], *, timeout: int, cap: int | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "deployment_name": handle.container_name,
            "command": argv,
            "workdir": "/",
            "timeout": timeout,
            "pod_name": handle.pod,
        }
        if cap is not None:
            kwargs["max_output_bytes"] = cap
        result = await asyncio.to_thread(self._prov().exec_in_pod, **kwargs)
        if int(result.get("exit_code", -1)) == -1 and await asyncio.to_thread(
            self._prov().pod_is_gone, handle.pod
        ):
            raise WorkspaceGoneError(f"pod {handle.pod} of {handle.container_name} is gone")
        return result

    async def launch(
        self,
        handle: DetachedHandle,
        argv: list[str],
        *,
        cwd: str,
        timeout: int,
        cap: int,
        env: dict[str, str] | None = None,
    ) -> None:
        # The caller's argv AND env, under the same rule the sync route applies.
        enforce_command_policy(
            argv,
            ExecutionProfile.CALLER_SUPPLIED,
            toolchain=self._toolchain_by_pod.pop(handle.pod, "python"),
            env=env,
        )
        # env rides on the LAUNCH exec as `/usr/bin/env K=V` argv (same helper as the
        # sync path) and is inherited by the wrapper: never written into run.sh.
        script = build_run_script(handle.job_dir, list(argv), cwd, timeout, cap)
        result = await self._exec(
            handle, build_launcher(handle.job_dir, script, env), timeout=LAUNCH_EXEC_TIMEOUT_S
        )
        if int(result.get("exit_code", -1)) != 0 or "launched" not in (result.get("stdout") or ""):
            raise RuntimeError(
                f"detached launch failed (exit {result.get('exit_code')}): "
                f"{(result.get('stderr') or '').strip()[:300]}"
            )

    async def poll(self, handle: DetachedHandle) -> PollResult:
        result = await self._exec(handle, build_poll(handle.job_dir), timeout=PROBE_EXEC_TIMEOUT_S)
        if int(result.get("exit_code", -1)) != 0:
            raise RuntimeError(
                f"detached status read failed (exit {result.get('exit_code')}): "
                f"{(result.get('stderr') or '').strip()[:300]}"
            )
        return parse_poll(result.get("stdout") or "")

    async def collect(
        self, handle: DetachedHandle, status: PollResult, *, cap: int
    ) -> DetachedOutcome:
        result = await self._exec(
            handle, build_collect(handle.job_dir), timeout=PROBE_EXEC_TIMEOUT_S, cap=cap
        )
        if int(result.get("exit_code", -1)) != 0:
            raise RuntimeError(f"detached output read failed (exit {result.get('exit_code')})")
        return outcome_from(status, result.get("stdout") or "", result.get("stderr") or "")

    async def cleanup(self, handle: DetachedHandle) -> None:
        await self._exec(handle, build_cleanup(handle.job_dir), timeout=PROBE_EXEC_TIMEOUT_S)


def outcome_from(status: PollResult, stdout: str, stderr: str) -> DetachedOutcome:
    exit_code = status.exit_code if status.exit_code is not None else -1
    duration = (
        float(status.ended - status.started)
        if status.started is not None and status.ended is not None
        else None
    )
    return DetachedOutcome(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=status.stdout_omitted > 0,
        stderr_truncated=status.stderr_omitted > 0,
        bytes_omitted=status.stdout_omitted + status.stderr_omitted,
        timed_out=exit_code == TIMEOUT_EXIT_CODE,
        duration=duration,
    )

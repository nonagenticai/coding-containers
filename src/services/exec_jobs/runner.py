"""Runs async workspace exec jobs, renews their leases, reaps orphans, sends callbacks.

One ``ExecJobRunner`` per process (``owner`` = pod hostname + pid + nonce).

Invariant this module exists for: every job that reaches the store ends in a
terminal status AND, when it has a ``callback_url``, a callback attempt — even
when the replica running it dies mid-command. Three paths get it there:

1. normal: the owner finishes the job (``ExecJobStore.finish``) and POSTs;
2. graceful shutdown (SIGTERM during a rollout): ``shutdown`` fails every job
   this process still owns with ``interrupted`` and POSTs before exiting;
3. hard death (OOM, node loss): the lease lapses and any replica's reaper fails
   the job with ``orphaned`` and POSTs. A sender that died mid-POST leaves
   ``callback_state = 'sending'`` with a lapsed callback lease, which the reaper
   re-claims and re-sends.

Exactly one replica wins each terminal transition (conditional UPDATE), and only
the winner sends. The single at-least-once window is a sender dying after the
POST landed but before ``record_callback`` committed.

DETACHED jobs (Kubernetes workspaces, ``detached.py``) change paths 2 and 3: the
command runs detached in the workspace pod, so a replica that stops does not
take it down. Graceful shutdown RELEASES such a job (lapses its lease) instead of
failing it, and every reaper sweep ADOPTS lapsed detached jobs (one winner per
row) and re-attaches: it polls the job directory, completes the row from the
wrapper's exit code and output, and sends the callback through the same
single-sender hand-off. A job whose pod is gone ends ``workspace_gone``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from .callback import CallbackOutcome, retry_delay_seconds, send_callback
from .detached import (
    LAUNCH_GRACE_S,
    LOST_ERROR,
    NEVER_STARTED_ERROR,
    WORKSPACE_GONE_ERROR,
    DetachedBackend,
    DetachedHandle,
    PollState,
    WorkspaceGoneError,
    job_dir,
)
from .store import ExecJobStore

logger = logging.getLogger(__name__)

ORPHANED_ERROR = (
    "orphaned: the coding-containers replica running this job stopped renewing its "
    "lease (pod restart, rollout or node loss); the command's outcome is unknown"
)
INTERRUPTED_ERROR = (
    "interrupted: the coding-containers replica running this job shut down "
    "(rollout or restart) before the command finished; its outcome is unknown"
)
# Covers one send_callback round's worst case (3 x 10 s timeouts + 2 s + 4 s
# backoff = 36 s) with margin, so a live sender's claim never lapses under it.
CALLBACK_LEASE_SECONDS = 120
SHUTDOWN_CALLBACK_BUDGET_S = 20.0
#: Upper bound on :meth:`ExecJobRunner.shutdown`. See "The SIGTERM budget" below for
#: how it fits the pod's terminationGracePeriodSeconds.
SHUTDOWN_TIMEOUT_S = 22.0

# ── The SIGTERM budget, end to end ─────────────────────────────────────────
#
# On SIGTERM uvicorn (0.47 ``Server.shutdown``) stops accepting, closes idle
# connections, sleeps 0.1 s, then waits for in-flight requests for at most
# ``--timeout-graceful-shutdown`` seconds and CANCELS whatever is left. Only then
# does it run the lifespan shutdown (this runner's ``shutdown`` -> the
# ``interrupted`` callbacks). With no flag that wait is unbounded: one sync exec
# (up to 300 s) held the process until the kubelet's SIGKILL at 30 s, and the
# lifespan — and every ``interrupted`` callback — never ran.
#
# Worst case from SIGTERM to process exit (each term is a hard bound in code):
#   0.1   uvicorn's fixed sleep before the in-flight wait
# + 10    HTTP_GRACEFUL_SHUTDOWN_S   (Dockerfile CMD --timeout-graceful-shutdown)
# + 22    SHUTDOWN_TIMEOUT_S         (this runner; callbacks are inside it)
# + 5     src.db.session.POOL_CLOSE_TIMEOUT_S (close_pool)
# + 10    src.core.reaper._STOP_TIMEOUT_SECONDS (Deployment reaper; enabled in prod)
# = 47.1 s  <  TERMINATION_GRACE_PERIOD_S = 60  (12.9 s margin)
#
# The ``interrupted`` callbacks are done by 0.1 + 10 + 22 = 32.1 s. Under the
# Kubernetes DEFAULT grace of 30 s that is NOT guaranteed, so the Deployment must
# set ``terminationGracePeriodSeconds: 60`` in the deployment manifest.
# A cancelled sync exec's ``asyncio.to_thread`` worker is not interruptible, so
# the interpreter may then sit in the default-executor join until SIGKILL; that is
# after the lifespan, so nothing it owed is lost.
# tests/unit/test_shutdown_budget.py pins this arithmetic and the CMD flag.
HTTP_GRACEFUL_SHUTDOWN_S = 10
UVICORN_PRE_WAIT_S = 0.1
TERMINATION_GRACE_PERIOD_S = 60
#: Sub-budgets inside ``SHUTDOWN_TIMEOUT_S``: stopping the reaper must not eat the
#: time the ``interrupted`` callbacks need, nor must failing the owned rows.
SHUTDOWN_REAPER_STOP_S = 2.0
SHUTDOWN_FAIL_OWNED_S = 3.0
#: How often shutdown re-delivers cancel to a task that swallowed it.
SHUTDOWN_RECANCEL_S = 0.1

#: Detached jobs: how often the owner polls the job directory in the pod.
DETACHED_POLL_INTERVAL_S = 5.0
#: Past ``timeout`` the wrapper's own ``timeout`` has killed the command; a job
#: still without an exit status this long after that is failed, never awaited
#: forever.
DETACHED_DEADLINE_SLACK_S = 120.0

Executor = Callable[..., Awaitable[tuple[Any, Any]]]


def _default_executor() -> Executor:
    # Lazy: the shared exec path lives with the sync route; importing it at module
    # load would make this service module depend on the router package at import.
    from src.routers.container_exec import run_workspace_command

    return run_workspace_command


def make_owner_id() -> str:
    """A process-unique lease owner: hostname (pod name) + pid + nonce."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def status_path(container_name: str, job_id: str) -> str:
    return f"/api/workspaces/{container_name}/exec/jobs/{job_id}"


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def callback_body(row: dict[str, Any]) -> dict[str, Any]:
    """The POSTed completion document (same fields as the status route)."""
    return {
        "job_id": row["job_id"],
        "container_name": row["container_name"],
        "status": row["status"],
        "exit_code": row["exit_code"],
        "stdout": row["stdout"],
        "stderr": row["stderr"],
        "stdout_truncated": row["stdout_truncated"],
        "stderr_truncated": row["stderr_truncated"],
        "truncated": bool(row["stdout_truncated"] or row["stderr_truncated"]),
        "bytes_omitted": row["bytes_omitted"],
        "timed_out": row["timed_out"],
        "duration": row["duration_seconds"],
        "error": row["error"],
        "completed_at": _iso(row["completed_at"]),
        "links": {"status": status_path(row["container_name"], row["job_id"])},
    }


def _http_error_message(exc: HTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, dict):
        msg = detail.get("message") or detail.get("detail") or detail
        return f"HTTP {exc.status_code}: {msg}"
    return f"HTTP {exc.status_code}: {detail}"


class ExecJobRunner:
    """Owns this process's running jobs plus the shared reaper loop."""

    def __init__(
        self,
        store: ExecJobStore,
        *,
        owner: str,
        lease_seconds: int,
        max_concurrent: int,
        executor: Executor | None = None,
        detached: DetachedBackend | None = None,
        poll_interval: float = DETACHED_POLL_INTERVAL_S,
        output_cap: int | None = None,
    ) -> None:
        self.store = store
        self.owner = owner
        self.lease_seconds = lease_seconds
        self._sem = asyncio.Semaphore(max_concurrent)
        self._executor = executor
        self._detached = detached
        self._poll_interval = poll_interval
        self._output_cap = output_cap
        self._tasks: set[asyncio.Task[None]] = set()
        self._reaper: asyncio.Task[None] | None = None
        self._closing = False

    # ── submission ────────────────────────────────────────────────────

    def submit(
        self, job: dict[str, Any], argv: list[str], env: dict[str, str] | None = None
    ) -> asyncio.Task[None]:
        """Start running a freshly created (``queued``, leased-to-us) job.

        ``env`` (the request's per-command environment) is never stored on the job
        row — a value may be a credential. In this service it lives in memory, like
        ``argv``: it is needed only to LAUNCH, and an adopter never relaunches (a job
        not launched before its replica died ends ``never_started``). In the pod it is
        the launched process's environment (the launch exec's ``/usr/bin/env K=V``
        argv, inherited by the detached wrapper), never written into the job
        directory's ``run.sh``.
        """
        if self._closing:
            raise RuntimeError("runner is shutting down")
        task = asyncio.create_task(
            self._run(job, argv, env or None), name=f"exec-job-{job['job_id']}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _renew_loop(self, job_id: str) -> None:
        interval = max(1.0, self.lease_seconds / 3)
        while not self._closing:
            await asyncio.sleep(interval)
            if self._closing:
                # Shutdown may have just RELEASED this (detached) job's lease
                # for adoption; renewing now would undo that.
                return
            try:
                if not await self.store.renew_lease(
                    job_id=job_id, owner=self.owner, lease_seconds=self.lease_seconds
                ):
                    logger.warning("Exec job %s: lease lost (job finished elsewhere)", job_id)
                    return
            except Exception:
                # A DB blip must not stop renewing: if it outlasts the lease, the
                # reaper fails the job and calls back, which is the safe outcome.
                logger.exception("Exec job %s: lease renewal errored", job_id)

    async def _execute(
        self, job: dict[str, Any], argv: list[str], env: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Run the command; map every outcome onto terminal-row fields."""
        executor = self._executor or _default_executor()
        try:
            _kind, result = await executor(
                job["container_name"],
                job["tenant_id"],
                argv,
                cwd=job["workdir"],
                timeout=job["timeout_seconds"],
                env=env,
            )
        except HTTPException as exc:
            return {"status": "failed", "error": _http_error_message(exc)}
        except Exception as exc:
            logger.exception("Exec job %s: execution error", job["job_id"])
            return {"status": "failed", "error": f"execution error: {type(exc).__name__}: {exc}"}
        fields: dict[str, Any] = {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "stdout_truncated": result.stdout_truncated,
            "stderr_truncated": result.stderr_truncated,
            "bytes_omitted": result.bytes_omitted,
            "timed_out": result.timed_out,
            "duration_seconds": result.duration,
        }
        if result.timed_out:
            fields["status"] = "failed"
            fields["error"] = f"timeout: the command exceeded {job['timeout_seconds']}s"
        else:
            fields["status"] = "completed"
        return fields

    async def _run(
        self, job: dict[str, Any], argv: list[str], env: dict[str, str] | None = None
    ) -> None:
        job_id = job["job_id"]
        heartbeat = asyncio.create_task(self._renew_loop(job_id))
        row: dict[str, Any] | None = None
        try:
            async with self._sem:
                if not await self.store.mark_running(
                    job_id=job_id, owner=self.owner, lease_seconds=self.lease_seconds
                ):
                    logger.warning("Exec job %s: no longer queued for us; not run", job_id)
                    return
                handle = await self._launch_detached(job, argv, env)
                fields: dict[str, Any]
                if handle is None:
                    fields = await self._execute(job, argv, env)
                elif isinstance(handle, DetachedHandle):
                    fields = await self._follow(job, handle, recorded_at=time.time())
                else:
                    fields = handle  # locate/launch failed: terminal fields
            row = await self.store.finish(
                job_id=job_id,
                owner=self.owner,
                callback_lease_seconds=CALLBACK_LEASE_SECONDS,
                **fields,
            )
            if row is None:
                logger.warning("Exec job %s: already finished elsewhere; no callback", job_id)
        except asyncio.CancelledError:
            # shutdown() has already failed the row and sent its callback, or
            # (detached) released it for a surviving replica to adopt.
            raise
        except Exception:
            logger.exception("Exec job %s: runner error", job_id)
            with contextlib.suppress(Exception):
                row = await self.store.finish(
                    job_id=job_id,
                    owner=self.owner,
                    status="failed",
                    error="internal error in coding-containers while running the job",
                    callback_lease_seconds=CALLBACK_LEASE_SECONDS,
                )
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
        if row is not None:
            await self.deliver(row)
            await self._cleanup(row)

    # ── detached (Kubernetes) jobs ────────────────────────────────────

    def _cap(self) -> int:
        if self._output_cap is not None:
            return self._output_cap
        from src.config.settings import get_settings

        return int(get_settings().exec_max_output_bytes)

    async def _launch_detached(
        self, job: dict[str, Any], argv: list[str], env: dict[str, str] | None = None
    ) -> DetachedHandle | dict[str, Any] | None:
        """Launch detached when the workspace allows it.

        ``None``: not detachable (no backend, or not a Kubernetes workspace) — run
        it the blocking way. A ``dict``: terminal fields (the launch failed before
        or while starting the command). Otherwise the recorded handle.
        """
        if self._detached is None:
            return None
        job_id = job["job_id"]
        try:
            pod = await self._detached.locate(job["container_name"], job["tenant_id"])
        except HTTPException as exc:
            return {"status": "failed", "error": _http_error_message(exc)}
        except Exception as exc:
            logger.exception("Exec job %s: could not locate the workspace pod", job_id)
            return {"status": "failed", "error": f"execution error: {type(exc).__name__}: {exc}"}
        if pod is None:
            return None
        handle = DetachedHandle(job["container_name"], pod, job_dir(job_id))
        # Recorded BEFORE launching: if this replica dies between the two, the
        # adopter finds no job directory and (after LAUNCH_GRACE_S) reports
        # ``never_started`` rather than an unknown outcome.
        if not await self.store.record_detached(
            job_id=job_id, owner=self.owner, pod=handle.pod, job_dir=handle.job_dir
        ):
            raise RuntimeError("job is no longer ours to launch")
        try:
            await self._detached.launch(
                handle,
                argv,
                cwd=job["workdir"],
                timeout=int(job["timeout_seconds"]),
                cap=self._cap(),
                env=env,
            )
        except WorkspaceGoneError:
            return {"status": "failed", "error": WORKSPACE_GONE_ERROR}
        except HTTPException as exc:
            return {"status": "failed", "error": _http_error_message(exc)}
        except Exception as exc:
            logger.exception("Exec job %s: detached launch failed", job_id)
            return {"status": "failed", "error": f"execution error: {type(exc).__name__}: {exc}"}
        logger.info("Exec job %s: launched detached in pod %s", job_id, handle.pod)
        return handle

    async def _follow(
        self, job: dict[str, Any], handle: DetachedHandle, *, recorded_at: float
    ) -> dict[str, Any]:
        """Poll the job directory until the command ends; return terminal fields.

        Bounded: a pod that is gone ends it at once (``workspace_gone``), and
        nothing outlives ``started + timeout + DETACHED_DEADLINE_SLACK_S``.
        Transient exec errors are retried until that deadline.
        """
        assert self._detached is not None
        timeout = int(job["timeout_seconds"])
        started = _epoch(job.get("started_at")) or recorded_at
        deadline = started + timeout + DETACHED_DEADLINE_SLACK_S
        while True:
            try:
                status = await self._detached.poll(handle)
            except WorkspaceGoneError:
                return {"status": "failed", "error": WORKSPACE_GONE_ERROR}
            except Exception as exc:
                logger.warning("Exec job %s: status read failed: %s", job["job_id"], exc)
                status = None
            if status is not None:
                if status.state is PollState.DONE:
                    try:
                        out = await self._detached.collect(handle, status, cap=self._cap())
                    except WorkspaceGoneError:
                        return {"status": "failed", "error": WORKSPACE_GONE_ERROR}
                    except Exception as exc:
                        logger.warning("Exec job %s: output read failed: %s", job["job_id"], exc)
                    else:
                        return _detached_fields(out, timeout)
                elif status.state is PollState.LOST:
                    return {"status": "failed", "error": LOST_ERROR}
                elif (
                    status.state is PollState.MISSING
                    and time.time() - recorded_at >= LAUNCH_GRACE_S
                ):
                    return {"status": "failed", "error": NEVER_STARTED_ERROR}
            if time.time() > deadline:
                return {
                    "status": "failed",
                    "timed_out": True,
                    "error": (
                        f"timeout: no exit status from the workspace pod within "
                        f"{timeout}s + {int(DETACHED_DEADLINE_SLACK_S)}s"
                    ),
                }
            await asyncio.sleep(self._poll_interval)

    async def _reattach(self, row: dict[str, Any]) -> None:
        """Complete an ADOPTED detached job: poll, finish, call back once."""
        job_id = row["job_id"]
        heartbeat = asyncio.create_task(self._renew_loop(job_id))
        done: dict[str, Any] | None = None
        try:
            handle = DetachedHandle(row["container_name"], row["detached_pod"], row["detached_dir"])
            recorded_at = _epoch(row.get("detached_at")) or time.time()
            fields = await self._follow(row, handle, recorded_at=recorded_at)
            done = await self.store.finish(
                job_id=job_id,
                owner=self.owner,
                callback_lease_seconds=CALLBACK_LEASE_SECONDS,
                **fields,
            )
            if done is None:
                logger.warning("Exec job %s: adopted job finished elsewhere; no callback", job_id)
        except asyncio.CancelledError:
            raise  # shutdown released it again; the next adopter takes over
        except Exception:
            logger.exception("Exec job %s: re-attach error", job_id)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
        if done is not None:
            await self.deliver(done)
            await self._cleanup(done)

    async def _cleanup(self, row: dict[str, Any]) -> None:
        """Best effort: remove the job directory once the row is terminal."""
        if self._detached is None or not row.get("detached_pod") or not row.get("detached_dir"):
            return
        try:
            await self._detached.cleanup(
                DetachedHandle(row["container_name"], row["detached_pod"], row["detached_dir"])
            )
        except Exception as exc:
            logger.info("Exec job %s: job directory not removed: %s", row["job_id"], exc)

    # ── callbacks ─────────────────────────────────────────────────────

    async def deliver(self, row: dict[str, Any]) -> CallbackOutcome | None:
        """Send one round for a row this process holds the send claim on."""
        if row.get("callback_state") != "sending" or not row.get("callback_url"):
            return None
        outcome, attempts = await send_callback(
            row["callback_url"], callback_body(row), job_id=row["job_id"]
        )
        retry_in = None
        if outcome is CallbackOutcome.RETRY:
            retry_in = retry_delay_seconds(int(row.get("callback_attempts") or 0) + attempts)
        try:
            recorded = await self.store.record_callback(
                job_id=row["job_id"],
                owner=row["callback_owner"],
                outcome=outcome.value,
                attempts=attempts,
                retry_in_seconds=retry_in,
            )
            if not recorded:
                logger.warning("Exec job %s: callback claim moved before recording", row["job_id"])
            elif retry_in is not None:
                logger.warning(
                    "Exec job %s: callback not accepted yet; next round in %ss",
                    row["job_id"],
                    retry_in,
                )
        except Exception:
            # Unrecorded: the row stays 'sending' and is re-claimed once its
            # callback lease lapses, so the callback is still not lost.
            logger.exception("Exec job %s: could not record callback outcome", row["job_id"])
        return outcome

    # ── reaper ────────────────────────────────────────────────────────

    async def reap_once(self) -> int:
        """Adopt lapsed detached jobs, fail other lapsed-lease jobs, and (re)send
        due callbacks. Returns rows handled."""
        adopted: list[dict[str, Any]] = []
        if self._detached is not None and not self._closing:
            adopted = await self.store.adopt_detached(
                owner=self.owner, lease_seconds=self.lease_seconds
            )
            for row in adopted:
                logger.warning(
                    "Exec job %s on %s adopted (detached in pod %s; previous owner %s)",
                    row["job_id"],
                    row["container_name"],
                    row["detached_pod"],
                    row.get("previous_owner"),
                )
                task = asyncio.create_task(self._reattach(row), name=f"exec-job-{row['job_id']}")
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        reaped = await self.store.reap_expired(
            reaper=self.owner, error=ORPHANED_ERROR, callback_lease_seconds=CALLBACK_LEASE_SECONDS
        )
        for row in reaped:
            logger.warning(
                "Exec job %s on %s reaped as orphaned (owner was %s)",
                row["job_id"],
                row["container_name"],
                row.get("owner"),
            )
        due = await self.store.claim_due_callbacks(
            owner=self.owner, callback_lease_seconds=CALLBACK_LEASE_SECONDS
        )
        rows = [*reaped, *due]
        if rows:
            await asyncio.gather(*(self.deliver(r) for r in rows))
        return len(rows) + len(adopted)

    async def _reaper_loop(self, interval: float) -> None:
        while True:
            try:
                await self.reap_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Exec job reaper sweep failed")
            if self._closing:
                # A cancel that something below swallowed (see shutdown) must not
                # be followed by a full ``interval`` of sleep.
                return
            await asyncio.sleep(interval)

    def start_reaper(self, interval: float) -> None:
        """Start the sweep loop; its first sweep runs immediately (startup reap)."""
        if self._reaper is None:
            self._reaper = asyncio.create_task(self._reaper_loop(interval), name="exec-job-reaper")

    # ── shutdown ──────────────────────────────────────────────────────

    async def shutdown(self, timeout: float = SHUTDOWN_TIMEOUT_S) -> None:
        """Fail and call back every job this process still owns, then stop —
        within ``timeout`` seconds.

        Order: stop the reaper, fail owned jobs as ``interrupted``, send their
        callbacks (``SHUTDOWN_CALLBACK_BUDGET_S``), cancel the job tasks. Whatever
        is not delivered in time stays ``sending`` with a callback lease, which a
        surviving replica's reaper re-claims once it lapses.

        One ``cancel()`` is NOT enough: psycopg_pool's ``getconn`` catches
        ``CancelledError`` raised during its connection check (it puts the
        connection back and retries), so a cancel that lands there is swallowed
        and the task carries on (the reaper into its ``interval`` sleep). An
        unbounded ``await`` then holds the pod until SIGKILL, skipping the
        ``interrupted`` callbacks. So cancel is re-delivered until each task ends,
        and every phase is bounded by the overall deadline.
        """
        self._closing = True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        def left(cap: float | None = None) -> float:
            remaining = max(0.0, deadline - loop.time())
            return remaining if cap is None else min(cap, remaining)

        if self._reaper is not None:
            reaper, self._reaper = self._reaper, None
            await _cancel_until_done([reaper], loop.time() + left(SHUTDOWN_REAPER_STOP_S), "reaper")

        if self._detached is not None:
            # Detached commands keep running in their workspace pods: hand them
            # to a surviving replica (lease lapsed now) instead of failing them.
            releasing = asyncio.create_task(
                self.store.release_owned_detached(owner=self.owner),
                name="exec-jobs-release-detached",
            )
            if await _finish_within(releasing, left(SHUTDOWN_FAIL_OWNED_S), deadline):
                if releasing.exception() is not None:
                    # Not lost: their leases lapse on their own and they are adopted.
                    logger.error(
                        "Exec jobs: could not release detached jobs: %r", releasing.exception()
                    )
                elif releasing.result():
                    logger.info(
                        "Exec jobs: released %d detached job(s) for adoption", releasing.result()
                    )

        rows: list[dict[str, Any]] = []
        failing = asyncio.create_task(
            self.store.fail_owned_active(
                owner=self.owner,
                error=INTERRUPTED_ERROR,
                callback_lease_seconds=CALLBACK_LEASE_SECONDS,
            ),
            name="exec-jobs-fail-owned",
        )
        if await _finish_within(failing, left(SHUTDOWN_FAIL_OWNED_S), deadline):
            if failing.exception() is None:
                rows = failing.result()
            else:
                # The reaper on a surviving replica fails these once the leases lapse.
                logger.error(
                    "Exec jobs: could not fail owned jobs at shutdown: %r", failing.exception()
                )
        else:
            logger.error("Exec jobs: failing owned jobs did not finish in budget")

        if rows:
            sending = asyncio.create_task(
                _gather(*(self.deliver(r) for r in rows)), name="exec-jobs-shutdown-callbacks"
            )
            # Leave a sliver of the deadline for cancelling the job tasks below.
            budget = left(SHUTDOWN_CALLBACK_BUDGET_S) - SHUTDOWN_RECANCEL_S * 5
            if not await _finish_within(sending, max(0.0, budget), deadline):
                # Left 'sending' with a lease: another replica's reaper re-sends.
                logger.error("Exec jobs: shutdown callbacks did not finish in budget")
            elif sending.exception() is not None:
                logger.error("Exec jobs: shutdown callbacks errored: %r", sending.exception())

        await _cancel_until_done(list(self._tasks), deadline, "job")


def _epoch(value: Any) -> float | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.timestamp()
    return None


def _detached_fields(out: Any, timeout: int) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "exit_code": out.exit_code,
        "stdout": out.stdout,
        "stderr": out.stderr,
        "stdout_truncated": out.stdout_truncated,
        "stderr_truncated": out.stderr_truncated,
        "bytes_omitted": out.bytes_omitted,
        "timed_out": out.timed_out,
        "duration_seconds": out.duration,
    }
    if out.timed_out:
        fields["status"] = "failed"
        fields["error"] = f"timeout: the command exceeded {timeout}s"
    else:
        fields["status"] = "completed"
    return fields


async def _gather(*aws: Awaitable[Any]) -> None:
    await asyncio.gather(*aws)


async def _finish_within(task: asyncio.Task[Any], budget: float, deadline: float) -> bool:
    """Wait up to ``budget`` s for ``task``; if it is still running, cancel it
    (re-delivered until it ends, bounded by ``deadline``). True if it finished on
    its own; its outcome is then the caller's to read."""
    done, _ = await asyncio.wait({task}, timeout=budget)
    if done:
        return True
    await _cancel_until_done([task], deadline, "shutdown step")
    return False


async def _cancel_until_done(tasks: list[asyncio.Task[Any]], deadline: float, what: str) -> None:
    """Cancel ``tasks``, re-cancelling every ``SHUTDOWN_RECANCEL_S`` until each has
    ended or ``deadline`` (loop time) passes; then give up with a warning."""
    loop = asyncio.get_running_loop()
    pending = {t for t in tasks if not t.done()}
    while pending:
        remaining = deadline - loop.time()
        if remaining <= 0:
            logger.warning(
                "Exec jobs: %d %s task(s) did not stop before the shutdown deadline; "
                "abandoning them",
                len(pending),
                what,
            )
            break
        for task in pending:
            task.cancel()
        _done, pending = await asyncio.wait(pending, timeout=min(SHUTDOWN_RECANCEL_S, remaining))
    for task in tasks:
        if task.done() and not task.cancelled():
            # Retrieve the outcome so an error is logged once, not as "exception
            # was never retrieved" at garbage collection.
            exc = task.exception()
            if exc is not None:
                logger.warning("Exec jobs: task %s ended with %r", task.get_name(), exc)


_runner: ExecJobRunner | None = None


def get_exec_job_runner() -> ExecJobRunner | None:
    """The process runner, or None when no job store is configured."""
    return _runner


def set_exec_job_runner(runner: ExecJobRunner | None) -> None:
    global _runner
    _runner = runner

# ruff: noqa: F811 - pytest fixtures imported from test_exec_jobs_pg are re-bound as parameters
"""Detached async exec against REAL Postgres: a job outlives the replica running it.

Same harness as test_exec_jobs_pg.py (real store, real migration, recorded
callbacks). The workspace pod is either ``FakePods`` (an in-memory pod whose
commands the test finishes, and which the test can delete) or ``LocalShell``,
which runs the REAL wrapper scripts with /bin/sh in a temp directory.

"A replica dies" is simulated the way it happens: its job task is cancelled
without ``shutdown`` (no release, no callback) and its lease is lapsed.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from src.db import session as db_session
from src.services.exec_jobs import detached as dx
from src.services.exec_jobs.runner import ORPHANED_ERROR, ExecJobRunner
from src.services.exec_jobs.store import ExecJobStore
from tests.unit.test_exec_jobs_pg import (  # noqa: F401 - fixtures
    DSN,
    _job,
    _row,
    migrated,
    recorder,
    store,
)
from tests.unit.test_exec_runner_shutdown import finishes_within

pytestmark = pytest.mark.skipif(not DSN, reason="needs TEST_DATABASE_URL / DATABASE_URL")

POD = "ws-1-7d9f-abcde"


class FakePods:
    """One workspace pod. Commands stay 'running' until the test finishes them."""

    def __init__(self) -> None:
        self.alive = {POD}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.launches: list[tuple[str, list[str], int]] = []
        self.envs: list[dict[str, str] | None] = []
        self.cleaned: list[str] = []
        self.polls = 0

    async def locate(self, container_name: str, tenant_id: str) -> str | None:
        return POD

    def _check(self, handle: dx.DetachedHandle) -> None:
        if handle.pod not in self.alive:
            raise dx.WorkspaceGoneError(handle.pod)

    async def launch(self, handle, argv, *, cwd, timeout, cap, env=None) -> None:
        self._check(handle)
        self.launches.append((handle.job_dir, list(argv), cap))
        self.envs.append(env)
        self.jobs[handle.job_dir] = {"state": dx.PollState.RUNNING}

    def finish(self, job_id: str, exit_code: int, stdout: str = "", stderr: str = "") -> None:
        self.jobs[dx.job_dir(job_id)] = {
            "state": dx.PollState.DONE,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
        }

    async def poll(self, handle) -> dx.PollResult:
        self.polls += 1
        self._check(handle)
        job = self.jobs.get(handle.job_dir)
        if job is None:
            return dx.PollResult(dx.PollState.MISSING)
        if job["state"] is not dx.PollState.DONE:
            return dx.PollResult(job["state"])
        return dx.PollResult(dx.PollState.DONE, exit_code=job["exit_code"], started=1, ended=3)

    async def collect(self, handle, status, *, cap) -> dx.DetachedOutcome:
        self._check(handle)
        job = self.jobs[handle.job_dir]
        return dx.outcome_from(status, job["stdout"], job["stderr"])

    async def cleanup(self, handle) -> None:
        self.cleaned.append(handle.job_dir)
        self.jobs.pop(handle.job_dir, None)


def _runner(owner: str, backend: Any, lease: int = 60) -> ExecJobRunner:
    return ExecJobRunner(
        ExecJobStore(),
        owner=owner,
        lease_seconds=lease,
        max_concurrent=4,
        detached=backend,
        poll_interval=0.02,
        output_cap=4096,
    )


def _lapse(job_id: str) -> None:
    with psycopg.connect(db_session.normalize_dsn(DSN), autocommit=True) as conn:
        conn.execute(
            "UPDATE exec_jobs SET lease_expires_at = now() - interval '1 second' WHERE id = %s",
            (job_id,),
        )


async def _until(predicate, seconds: float = 5.0) -> None:
    for _ in range(int(seconds / 0.02)):
        if await predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not reached")


async def _launched_then_killed(store, pods, job_owner: str = "pod-a"):
    """Replica A starts a detached job, then dies (no shutdown, lease lapsed)."""
    a = _runner(job_owner, pods)
    job = await _job(store, job_owner)
    task = a.submit(job, ["python", "-m", "pytest"])
    await _until(lambda: _has_launched(pods, job))
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    _lapse(job["job_id"])
    return job


async def _has_launched(pods, job) -> bool:
    return dx.job_dir(job["job_id"]) in pods.jobs


# ── normal run (no replica change) ────────────────────────────────────


async def test_detached_job_completes_on_its_own_replica(store, recorder):
    pods = FakePods()
    a = _runner("pod-a", pods)
    job = await _job(store, "pod-a")
    task = a.submit(job, ["python", "-m", "pytest"])
    await _until(lambda: _has_launched(pods, job))
    row = await _row(store, job)
    assert row["status"] == "running"
    assert row["detached_pod"] == POD and row["detached_dir"] == dx.job_dir(job["job_id"])
    assert pods.launches[0][2] == 4096  # the output cap reaches the wrapper
    pods.finish(job["job_id"], 0, stdout="ok\n")
    await asyncio.wait_for(task, 5)
    row = await _row(store, job)
    assert row["status"] == "completed" and row["exit_code"] == 0 and row["stdout"] == "ok\n"
    assert [b["status"] for _, b in recorder.posts] == ["completed"]
    assert pods.cleaned == [dx.job_dir(job["job_id"])]


async def test_the_request_env_reaches_the_detached_launch(store, recorder):
    pods = FakePods()
    a = _runner("pod-a", pods)
    job = await _job(store, "pod-a")
    env = {"APP_MARKER": "env-marker-7f3a"}
    task = a.submit(job, ["python", "-m", "pytest"], env)
    await _until(lambda: _has_launched(pods, job))
    assert pods.envs == [env]
    pods.finish(job["job_id"], 0)
    await asyncio.wait_for(task, 5)


async def test_non_kubernetes_workspace_keeps_the_blocking_path(store, recorder):
    pods = FakePods()

    async def no_pod(container_name, tenant_id):
        return None

    pods.locate = no_pod  # type: ignore[method-assign]
    calls = []

    async def executor(container_name, tenant_id, argv, *, cwd, timeout, env=None):
        calls.append(argv)
        from src.runtime.types import RunOnceResult

        return "docker", RunOnceResult(exit_code=0, stdout="x", stderr="", duration=0.1)

    runner = ExecJobRunner(
        ExecJobStore(),
        owner="pod-a",
        lease_seconds=60,
        max_concurrent=4,
        executor=executor,
        detached=pods,
    )
    job = await _job(store, "pod-a")
    await runner.submit(job, ["ls"])
    assert calls == [["ls"]] and pods.launches == []
    row = await _row(store, job)
    assert row["status"] == "completed" and row["detached_pod"] is None


# ── re-attach ─────────────────────────────────────────────────────────


async def test_survivor_reattaches_and_completes_with_the_real_outcome(store, recorder):
    pods = FakePods()
    job = await _launched_then_killed(store, pods)
    b = _runner("pod-b", pods)
    assert await b.reap_once() == 1  # adopted, not reaped as an orphan
    row = await _row(store, job)
    assert row["status"] == "running" and row["owner"] == "pod-b"
    pods.finish(job["job_id"], 3, stdout="3 failed\n", stderr="E\n")
    await _until(lambda: _status_is(store, job, "completed"))
    await _until(lambda: _posted(recorder, 1))
    row = await _row(store, job)
    assert row["exit_code"] == 3 and row["stdout"] == "3 failed\n" and row["error"] is None
    assert [(b["job_id"], b["status"], b["exit_code"]) for _, b in recorder.posts] == [
        (job["job_id"], "completed", 3)
    ]
    await b.shutdown()


async def test_command_that_finished_while_no_replica_was_alive(store, recorder):
    pods = FakePods()
    job = await _launched_then_killed(store, pods)
    pods.finish(job["job_id"], 7, stdout="done while nobody watched\n")  # nobody alive
    b = _runner("pod-b", pods)
    b.start_reaper(interval=3600)  # only the immediate startup sweep can fire
    try:
        await _until(lambda: _posted(recorder, 1))
    finally:
        await b.shutdown()
    row = await _row(store, job)
    assert row["status"] == "completed" and row["exit_code"] == 7
    assert recorder.posts[0][1]["stdout"] == "done while nobody watched\n"
    assert row["error"] != ORPHANED_ERROR


async def test_exactly_one_callback_across_two_surviving_replicas(store, recorder):
    pods = FakePods()
    jobs = [await _launched_then_killed(store, pods) for _ in range(4)]
    b, c = _runner("pod-b", pods), _runner("pod-c", pods)
    counts = await asyncio.gather(b.reap_once(), c.reap_once(), b.reap_once(), c.reap_once())
    assert sum(counts) == 4  # each job adopted by exactly one replica
    owners = {(await _row(store, j))["owner"] for j in jobs}
    assert owners <= {"pod-b", "pod-c"}
    for j in jobs:
        pods.finish(j["job_id"], 0, stdout=j["job_id"])
    await _until(lambda: _posted(recorder, 4))
    await asyncio.sleep(0.2)  # any duplicate would land now
    assert sorted(b["job_id"] for _, b in recorder.posts) == sorted(j["job_id"] for j in jobs)
    # The dead owner's late finish loses: still one callback each.
    late = await store.finish(
        job_id=jobs[0]["job_id"], owner="pod-a", status="completed", callback_lease_seconds=120
    )
    assert late is None and len(recorder.posts) == 4
    await asyncio.gather(b.shutdown(), c.shutdown())


class StalledView:
    """One replica's view of the shared ``FakePods``: its polls block on ``gate``.

    Models a replica that is alive but stalled (GC pause, CPU starvation, a hung
    exec) for longer than its lease, so another replica takes the job over."""

    def __init__(self, pods: FakePods) -> None:
        self.pods = pods
        self.gate = asyncio.Event()
        self.stalled = asyncio.Event()  # set once a poll is parked on the gate
        self.seen_after_gate: list[dx.PollState] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.pods, name)

    async def poll(self, handle) -> dx.PollResult:
        if not self.gate.is_set():
            self.stalled.set()
            await self.gate.wait()
        result = await self.pods.poll(handle)
        self.seen_after_gate.append(result.state)
        return result


class KeepingPods(FakePods):
    """Cleanup is recorded but leaves the job dir, so a late follower still finds
    the DONE status files (the race these tests are about: both see DONE)."""

    async def cleanup(self, handle) -> None:
        self.cleaned.append(handle.job_dir)


async def _drained(runner: ExecJobRunner) -> None:
    await asyncio.wait_for(asyncio.gather(*list(runner._tasks), return_exceptions=True), 5)


async def test_stalled_adopter_loses_finish_to_the_next_adopter_and_sends_nothing(
    store, recorder, caplog
):
    """B adopts, stalls past its lease, C adopts; both then see DONE. C wins
    ``finish`` and calls back; B's ``_reattach`` takes the ``done is None`` branch
    and must send NOTHING: exactly one callback in total."""
    import logging

    pods = KeepingPods()
    job = await _launched_then_killed(store, pods)
    b_view = StalledView(pods)
    b = _runner("pod-b", b_view)
    with caplog.at_level(logging.WARNING):
        assert await b.reap_once() == 1
    # The adoption log names the PREVIOUS owner (the dead one), not the adopter.
    assert "previous owner pod-a)" in caplog.text
    await asyncio.wait_for(b_view.stalled.wait(), 5)  # B is parked mid-poll
    _lapse(job["job_id"])  # ... for longer than its lease
    pods.finish(job["job_id"], 4, stdout="c saw it\n")
    c = _runner("pod-c", pods)
    assert await c.reap_once() == 1
    assert (await _row(store, job))["owner"] == "pod-c"
    await _until(lambda: _posted(recorder, 1))
    await _drained(c)
    b_view.gate.set()  # B wakes, sees DONE too, and tries to finish
    await _drained(b)
    assert b_view.seen_after_gate == [dx.PollState.DONE]  # B observed DONE after the takeover
    await asyncio.sleep(0.2)  # a duplicate would land now
    assert [(p["job_id"], p["status"], p["exit_code"]) for _, p in recorder.posts] == [
        (job["job_id"], "completed", 4)
    ]
    row = await _row(store, job)
    assert row["owner"] == "pod-c" and row["callback_owner"] == "pod-c"
    await asyncio.gather(b.shutdown(), c.shutdown())


async def test_original_owner_back_after_a_takeover_also_loses(store, recorder):
    """A never died: it stalled past its lease, B adopted and completed the job.
    When A wakes and sees DONE, its ``finish`` loses and it sends nothing."""
    pods = KeepingPods()
    a_view = StalledView(pods)
    a = _runner("pod-a", a_view)
    job = await _job(store, "pod-a")
    a_task = a.submit(job, ["python", "-m", "pytest"])
    await asyncio.wait_for(a_view.stalled.wait(), 5)  # launched, A parked mid-poll
    _lapse(job["job_id"])
    pods.finish(job["job_id"], 0, stdout="b saw it\n")
    b = _runner("pod-b", pods)
    assert await b.reap_once() == 1
    await _until(lambda: _posted(recorder, 1))
    await _drained(b)
    a_view.gate.set()
    await asyncio.wait_for(a_task, 5)
    assert a_view.seen_after_gate == [dx.PollState.DONE]
    await asyncio.sleep(0.2)
    assert [(p["job_id"], p["status"]) for _, p in recorder.posts] == [(job["job_id"], "completed")]
    row = await _row(store, job)
    assert row["owner"] == "pod-b" and row["callback_owner"] == "pod-b"
    await asyncio.gather(a.shutdown(), b.shutdown())


async def test_adoption_has_one_winner_while_another_replica_is_mid_adoption(store, recorder):
    """B holds the row (mid-adoption) while C sweeps: C must skip it, not wait for
    it and then take it too (two owners would mean two followers)."""
    pods = FakePods()
    job = await _launched_then_killed(store, pods)
    b_tx = await psycopg.AsyncConnection.connect(db_session.normalize_dsn(DSN))
    try:
        await b_tx.execute("SELECT id FROM exec_jobs WHERE id = %s FOR UPDATE", (job["job_id"],))
        c_rows = await asyncio.wait_for(
            ExecJobStore().adopt_detached(owner="pod-c", lease_seconds=60), 3
        )
        assert c_rows == []  # skipped, not blocked
        await b_tx.execute(
            "UPDATE exec_jobs SET owner = 'pod-b', "
            "lease_expires_at = now() + interval '60 seconds' WHERE id = %s",
            (job["job_id"],),
        )
        await b_tx.commit()
    finally:
        await b_tx.close()
    assert await ExecJobStore().adopt_detached(owner="pod-c", lease_seconds=60) == []
    assert (await _row(store, job))["owner"] == "pod-b"


async def test_workspace_pod_gone_is_a_typed_failure_not_a_hang(store, recorder):
    pods = FakePods()
    job = await _launched_then_killed(store, pods)
    pods.alive.clear()  # the workspace pod was deleted with the command in it
    b = _runner("pod-b", pods)
    assert await b.reap_once() == 1
    await _until(lambda: _posted(recorder, 1), seconds=5)
    row = await _row(store, job)
    assert row["status"] == "failed" and row["error"] == dx.WORKSPACE_GONE_ERROR
    assert [(b["status"], b["error"]) for _, b in recorder.posts] == [
        ("failed", dx.WORKSPACE_GONE_ERROR)
    ]
    assert await finishes_within(b.shutdown(), 5)


async def test_handle_recorded_but_never_launched_ends_never_started(store, recorder, monkeypatch):
    """The owner died between recording the handle and launching."""
    import src.services.exec_jobs.runner as runner_mod

    monkeypatch.setattr(runner_mod, "LAUNCH_GRACE_S", 0.2)
    pods = FakePods()
    job = await _job(store, "pod-a")
    assert await store.mark_running(job_id=job["job_id"], owner="pod-a", lease_seconds=60)
    assert await store.record_detached(
        job_id=job["job_id"], owner="pod-a", pod=POD, job_dir=dx.job_dir(job["job_id"])
    )
    _lapse(job["job_id"])
    b = _runner("pod-b", pods)
    assert await b.reap_once() == 1
    await _until(lambda: _posted(recorder, 1))
    assert (await _row(store, job))["error"] == dx.NEVER_STARTED_ERROR
    await b.shutdown()


# ── graceful shutdown hands over instead of failing ────────────────────


async def test_shutdown_releases_detached_jobs_for_immediate_adoption(store, recorder):
    pods = FakePods()
    a = _runner("pod-a", pods)
    job = await _job(store, "pod-a")
    a.submit(job, ["python", "-m", "pytest"])
    await _until(lambda: _has_launched(pods, job))
    assert await finishes_within(a.shutdown(), 10)
    row = await _row(store, job)
    assert row["status"] == "running" and recorder.posts == []  # no 'interrupted'
    assert row["lease_expires_at"] < datetime.now(UTC) + timedelta(seconds=1)
    b = _runner("pod-b", pods)
    assert await b.reap_once() == 1  # no lease wait
    pods.finish(job["job_id"], 0)
    await _until(lambda: _posted(recorder, 1))
    assert recorder.posts[0][1]["status"] == "completed"
    await b.shutdown()


async def test_queued_non_detached_jobs_are_still_interrupted_at_shutdown(store, recorder):
    """Control: release is for detached rows only."""
    from src.services.exec_jobs.runner import INTERRUPTED_ERROR

    pods = FakePods()
    a = _runner("pod-a", pods)
    queued = await _job(store, "pod-a")  # never submitted: no handle
    await a.shutdown()
    assert (await _row(store, queued))["error"] == INTERRUPTED_ERROR


# ── the real wrapper, end to end through the runner ─────────────────────


class LocalShell(FakePods):
    """Runs the REAL detached.py scripts with /bin/sh (the 'pod' is this host)."""

    async def launch(self, handle, argv, *, cwd, timeout, cap, env=None) -> None:
        self.envs.append(env)
        script = dx.build_run_script(handle.job_dir, argv, cwd, timeout, cap)
        out = await self._run(dx.build_launcher(handle.job_dir, script, env))
        assert "launched" in out.stdout, out.stderr
        self.jobs[handle.job_dir] = {"state": dx.PollState.RUNNING}

    async def poll(self, handle) -> dx.PollResult:
        return dx.parse_poll((await self._run(dx.build_poll(handle.job_dir))).stdout)

    async def collect(self, handle, status, *, cap) -> dx.DetachedOutcome:
        out = await self._run(dx.build_collect(handle.job_dir))
        return dx.outcome_from(status, out.stdout, out.stderr)

    async def cleanup(self, handle) -> None:
        await self._run(dx.build_cleanup(handle.job_dir))
        self.cleaned.append(handle.job_dir)

    @staticmethod
    async def _run(argv):
        return await asyncio.to_thread(
            subprocess.run, argv, capture_output=True, text=True, timeout=30, check=False
        )


@pytest.mark.skipif(not shutil.which("timeout"), reason="needs coreutils timeout")
async def test_real_wrapper_survives_the_replica_and_is_capped(
    store, recorder, tmp_path, monkeypatch
):
    monkeypatch.setattr(dx, "JOB_ROOT", str(tmp_path))
    pods = LocalShell()
    a = _runner("pod-a", pods)
    job = await _job(store, "pod-a")
    job["workdir"] = str(tmp_path)  # /workspace does not exist on this host
    task = a.submit(
        job, ["sh", "-c", "sleep 1; head -c 10000 /dev/zero | tr '\\0' x; echo bye >&2; exit 5"]
    )
    await _until(lambda: _has_launched(pods, job))
    task.cancel()  # replica A dies while the command is still sleeping
    await asyncio.gather(task, return_exceptions=True)
    _lapse(job["job_id"])
    b = _runner("pod-b", pods)
    assert await b.reap_once() == 1
    await _until(lambda: _posted(recorder, 1), seconds=10)
    row = await _row(store, job)
    assert row["status"] == "completed" and row["exit_code"] == 5
    assert row["stdout"] == "x" * 4096 and row["stdout_truncated"] is True  # output_cap
    assert row["bytes_omitted"] > 0 and row["stderr"] == "bye\n"
    job_path = tmp_path / job["job_id"]

    async def cleaned() -> bool:  # cleanup runs right after the callback
        return not job_path.exists()

    await _until(cleaned)
    await b.shutdown()


async def _status_is(store, job, status: str) -> bool:
    return (await _row(store, job))["status"] == status


async def _posted(recorder, n: int) -> bool:
    return len(recorder.posts) >= n


def test_job_ids_are_uuids_so_the_dir_is_safe() -> None:
    jid = str(uuid.uuid7())
    assert dx.job_dir(jid) == f"{dx.JOB_ROOT}/{jid}"

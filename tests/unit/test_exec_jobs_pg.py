"""Async exec jobs against REAL Postgres: run -> callback, failure, timeout, orphan
reaping, shutdown, lease renewal, tenant scoping and two-replica races.

Needs ``TEST_DATABASE_URL`` (or ``DATABASE_URL``): in CI the archetype's per-job
ephemeral postgres:18 sidecar (``has_postgres: true``); locally an ephemeral
``postgres:18`` container. Skipped when neither is set. The schema comes from
the real alembic migration, so these tests also prove the migration.

The executor (the k8s exec) and the HTTP client are fakes; everything between
them — the store's conditional transitions, leases, the reaper and the
single-sender callback hand-off — is the production code against Postgres.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import psycopg
import pytest

from coding_client.models.responses import not_found
from src.db import session as db_session
from src.runtime.types import RunOnceResult
from src.services.exec_jobs import callback as cb
from src.services.exec_jobs.runner import (
    INTERRUPTED_ERROR,
    ORPHANED_ERROR,
    ExecJobRunner,
)
from src.services.exec_jobs.store import ExecJobStore
from tests.unit.test_exec_runner_shutdown import finishes_within

DSN = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
pytestmark = pytest.mark.skipif(not DSN, reason="needs TEST_DATABASE_URL / DATABASE_URL")

PREFIX = "http://203.0.113.10:5678/webhook-waiting/"
GOOD = PREFIX + "4711"
TENANT = "tenant-a"
WS = "ws-1"


@pytest.fixture(scope="module")
def migrated():
    from alembic.config import Config

    from alembic import command

    # No config file: alembic.ini's fileConfig() would disable this process's
    # loggers. env.py reads DATABASE_URL, exactly as the initContainer does.
    cfg = Config()
    cfg.set_main_option("script_location", "alembic")
    old = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = DSN
    try:
        command.upgrade(cfg, "head")
        command.upgrade(cfg, "head")  # idempotent: a re-run of the initContainer is a no-op
    finally:
        if old is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = old


@pytest.fixture
async def store(migrated, monkeypatch):
    from src.config.settings import get_settings

    monkeypatch.setattr(get_settings(), "pgbouncer_url", DSN)
    with psycopg.connect(db_session.normalize_dsn(DSN), autocommit=True) as conn:
        conn.execute("TRUNCATE exec_jobs")
    yield ExecJobStore()
    await db_session.close_pool()


class Recorder:
    """Stands in for httpx.AsyncClient across every sender; records POST bodies."""

    def __init__(self, status: int = 200, delay: float = 0.0):
        self.status = status
        self.delay = delay
        self.posts: list[tuple[str, dict]] = []
        # Scripted answers consumed first: an int status, (status, text) or an
        # exception to raise; then ``status`` for every later POST.
        self.script: list = []

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.posts.append((url, json))
        r = self.script.pop(0) if self.script else self.status
        if isinstance(r, Exception):
            raise r
        status, text = r if isinstance(r, tuple) else (r, "")
        return SimpleNamespace(status_code=status, text=text)


@pytest.fixture
def recorder():
    rec = Recorder()
    with (
        # NB asyncio.sleep is NOT patched here: ``cb.asyncio`` is the global module,
        # and the runner's lease renewal sleeps too. 2xx/4xx are never retried.
        patch.object(cb.httpx, "AsyncClient", new=rec),
        patch.object(
            cb, "get_settings", return_value=SimpleNamespace(exec_callback_prefixes=(PREFIX,))
        ),
    ):
        yield rec


def _result(**kw) -> RunOnceResult:
    base = {"exit_code": 0, "stdout": "ok\n", "stderr": "", "duration": 0.2}
    base.update(kw)
    return RunOnceResult(**base)


def _executor(result=None, exc: Exception | None = None, gate: asyncio.Event | None = None):
    calls: list[dict] = []
    envs: list[dict | None] = []

    async def run(container_name, tenant_id, argv, *, cwd, timeout, env=None):
        envs.append(env)
        calls.append(
            {"container": container_name, "tenant": tenant_id, "argv": argv, "timeout": timeout}
        )
        if gate is not None:
            await gate.wait()
        if exc is not None:
            raise exc
        return "kubernetes", result or _result()

    run.calls = calls  # type: ignore[attr-defined]
    run.envs = envs  # type: ignore[attr-defined]
    return run


def _runner(store, owner: str, executor=None, lease: int = 60) -> ExecJobRunner:
    return ExecJobRunner(
        store, owner=owner, lease_seconds=lease, max_concurrent=4, executor=executor
    )


async def _job(store, owner: str, *, lease: int = 60, callback: str | None = GOOD, **kw):
    if callback == GOOD:
        callback = f"{PREFIX}{uuid.uuid4().hex}"  # one resume URL per cell, as in n8n
    res = await store.create_job(
        job_id=str(uuid.uuid7()),
        tenant_id=kw.get("tenant_id", TENANT),
        container_name=kw.get("container_name", WS),
        command_summary="python -m pytest",
        workdir="/workspace",
        timeout_seconds=kw.get("timeout", 3600),
        callback_url=callback,
        owner=owner,
        lease_seconds=lease,
    )
    row, created = res if isinstance(res, tuple) else (res, True)
    assert created
    return row


async def _row(store, job) -> dict:
    row = await store.get_job(
        job_id=job["job_id"], tenant_id=job["tenant_id"], container_name=job["container_name"]
    )
    assert row is not None
    return row


# ── normal lifecycle ──────────────────────────────────────────────────


async def test_queue_run_complete_and_callback_once(store, recorder):
    ex = _executor(
        _result(exit_code=3, stdout="o", stderr="e", stderr_truncated=True, bytes_omitted=11)
    )
    runner = _runner(store, "pod-a", ex)
    job = await _job(store, "pod-a")
    assert job["status"] == "queued" and job["lease_expires_at"] is not None
    await runner.submit(job, ["python", "-m", "pytest"])

    row = await _row(store, job)
    assert row["status"] == "completed"  # a non-zero exit is still a completed run
    assert row["exit_code"] == 3 and row["stderr_truncated"] is True
    assert row["bytes_omitted"] == 11 and row["started_at"] is not None
    assert row["callback_state"] == "delivered" and row["callback_attempts"] == 1
    assert row["lease_expires_at"] is None
    assert ex.calls == [
        {"container": WS, "tenant": TENANT, "argv": ["python", "-m", "pytest"], "timeout": 3600}
    ]
    assert len(recorder.posts) == 1
    url, body = recorder.posts[0]
    assert url == job["callback_url"] and url.startswith(PREFIX)
    assert body["job_id"] == job["job_id"] and body["container_name"] == WS
    assert body["status"] == "completed" and body["exit_code"] == 3
    assert body["stdout"] == "o" and body["stderr"] == "e"
    assert body["truncated"] is True and body["bytes_omitted"] == 11
    assert body["error"] is None and body["completed_at"]
    assert body["links"]["status"] == f"/api/workspaces/{WS}/exec/jobs/{job['job_id']}"


async def test_submit_hands_the_request_env_to_the_executor(store, recorder):
    """The async exec's env reaches the command (and is never put on the job row)."""
    ex = _executor()
    runner = _runner(store, "pod-a", ex)
    job = await _job(store, "pod-a")
    env = {"APP_MARKER": "env-marker-7f3a"}
    await runner.submit(job, ["python", "-m", "pytest"], env)
    assert ex.envs == [env]
    row = await _row(store, job)
    assert row["status"] == "completed"
    assert "env-marker-7f3a" not in repr(dict(row))


async def test_no_callback_url_means_no_post(store, recorder):
    runner = _runner(store, "pod-a", _executor())
    job = await _job(store, "pod-a", callback=None)
    await runner.submit(job, ["ls"])
    row = await _row(store, job)
    assert row["status"] == "completed" and row["callback_state"] == "none"
    assert recorder.posts == []


async def test_execution_failure_fails_the_job_and_calls_back(store, recorder):
    runner = _runner(store, "pod-a", _executor(exc=not_found("Container", WS)))
    job = await _job(store, "pod-a")
    await runner.submit(job, ["ls"])
    row = await _row(store, job)
    assert row["status"] == "failed" and row["exit_code"] is None
    assert row["error"].startswith("HTTP 404")
    assert [b["status"] for _, b in recorder.posts] == ["failed"]


async def test_unexpected_executor_error_fails_the_job(store, recorder):
    runner = _runner(store, "pod-a", _executor(exc=OSError("stream reset")))
    job = await _job(store, "pod-a")
    await runner.submit(job, ["ls"])
    row = await _row(store, job)
    assert row["status"] == "failed" and "OSError: stream reset" in row["error"]
    assert len(recorder.posts) == 1


async def test_timeout_is_failed_with_output_and_calls_back(store, recorder):
    ex = _executor(_result(exit_code=124, timed_out=True, stdout="partial"))
    runner = _runner(store, "pod-a", ex)
    job = await _job(store, "pod-a", timeout=5)
    await runner.submit(job, ["python", "slow.py"])
    row = await _row(store, job)
    assert row["status"] == "failed" and row["timed_out"] is True
    assert row["exit_code"] == 124 and row["stdout"] == "partial"
    assert row["error"] == "timeout: the command exceeded 5s"
    assert [(b["status"], b["timed_out"]) for _, b in recorder.posts] == [("failed", True)]


async def test_refused_callback_is_recorded_not_retried(store, recorder):
    recorder.status = 404  # e.g. the Wait node is no longer waiting
    runner = _runner(store, "pod-a", _executor())
    job = await _job(store, "pod-a")
    await runner.submit(job, ["ls"])
    assert (await _row(store, job))["callback_state"] == "refused"
    assert len(recorder.posts) == 1


# ── orphans: the replica died ─────────────────────────────────────────


async def test_orphaned_running_job_is_failed_and_called_back(store, recorder):
    dead = await _job(store, "pod-dead", lease=0)  # lease already lapsed
    assert await store.mark_running(job_id=dead["job_id"], owner="pod-dead", lease_seconds=0)
    survivor = _runner(store, "pod-b")
    assert await survivor.reap_once() == 1
    row = await _row(store, dead)
    assert row["status"] == "failed" and row["error"] == ORPHANED_ERROR
    assert row["callback_state"] == "delivered"
    assert [(b["job_id"], b["status"], b["error"]) for _, b in recorder.posts] == [
        (dead["job_id"], "failed", ORPHANED_ERROR)
    ]
    assert await survivor.reap_once() == 0  # nothing left; no second callback
    assert len(recorder.posts) == 1


async def test_orphaned_queued_job_is_reaped_too(store, recorder):
    """A pod that dies between the INSERT and starting the runner leaves 'queued'."""
    dead = await _job(store, "pod-dead", lease=0)
    await _runner(store, "pod-b").reap_once()
    assert (await _row(store, dead))["status"] == "failed"
    assert len(recorder.posts) == 1


async def test_live_lease_is_not_reaped(store, recorder):
    live = await _job(store, "pod-a", lease=60)
    assert await _runner(store, "pod-b").reap_once() == 0
    assert (await _row(store, live))["status"] == "queued"
    assert recorder.posts == []


async def test_startup_sweep_runs_immediately(store, recorder):
    dead = await _job(store, "previous-generation-pod", lease=0)
    runner = _runner(store, "new-pod")
    runner.start_reaper(interval=3600)  # only the immediate first sweep can fire
    try:
        for _ in range(100):
            if recorder.posts:
                break
            await asyncio.sleep(0.05)
    finally:
        await runner.shutdown()
    assert (await _row(store, dead))["status"] == "failed"
    assert [b["job_id"] for _, b in recorder.posts] == [dead["job_id"]]


async def test_late_finish_by_the_dead_owner_does_not_call_back_twice(store, recorder):
    """The reaped owner was only partitioned, not dead: its finish must lose."""
    job = await _job(store, "pod-a", lease=0)
    await _runner(store, "pod-b").reap_once()
    late = await store.finish(
        job_id=job["job_id"],
        owner="pod-a",
        status="completed",
        exit_code=0,
        callback_lease_seconds=120,
    )
    assert late is None
    assert (await _row(store, job))["error"] == ORPHANED_ERROR
    assert len(recorder.posts) == 1


async def test_renewed_lease_survives_a_concurrent_reaper(store, recorder):
    gate = asyncio.Event()
    runner = _runner(store, "pod-a", _executor(gate=gate), lease=3)  # renews every 1 s
    job = await _job(store, "pod-a", lease=3)
    task = runner.submit(job, ["python", "long.py"])
    other = _runner(store, "pod-b")
    for _ in range(5):  # 5 s > the 3 s lease: only renewal keeps it alive
        await asyncio.sleep(1)
        assert await other.reap_once() == 0
    gate.set()
    await task
    row = await _row(store, job)
    assert row["status"] == "completed" and row["error"] is None
    assert len(recorder.posts) == 1


# ── graceful shutdown (rollout) ───────────────────────────────────────


async def test_shutdown_fails_running_jobs_and_calls_back_once(store, recorder):
    gate = asyncio.Event()  # never set: the command outlives the pod
    runner = _runner(store, "pod-a", _executor(gate=gate))
    job = await _job(store, "pod-a")
    runner.submit(job, ["python", "long.py"])
    for _ in range(100):
        if (await _row(store, job))["status"] == "running":
            break
        await asyncio.sleep(0.02)
    await runner.shutdown()
    row = await _row(store, job)
    assert row["status"] == "failed" and row["error"] == INTERRUPTED_ERROR
    assert row["callback_state"] == "delivered"
    assert [(b["status"], b["error"]) for _, b in recorder.posts] == [("failed", INTERRUPTED_ERROR)]
    with pytest.raises(RuntimeError):
        runner.submit(job, ["ls"])


async def test_shutdown_survives_a_cancel_swallowed_by_the_real_pool(store, recorder):
    """psycopg_pool's getconn swallows a CancelledError raised during its
    connection check. Here the check is slowed for the reaper so the shutdown
    cancel lands inside it every time; the owned job must still be failed and
    called back, and shutdown must end well inside the pod's grace period."""
    gate = asyncio.Event()  # never set: the command outlives the pod
    runner = _runner(store, "pod-a", _executor(gate=gate))
    job = await _job(store, "pod-a")  # opens the real pool on this loop
    runner.submit(job, ["python", "long.py"])
    for _ in range(100):
        if (await _row(store, job))["status"] == "running":
            break
        await asyncio.sleep(0.02)
    pool = await db_session.get_pool()
    real_check = pool._check
    reaper_checking = asyncio.Event()

    async def slow_check(conn):
        await real_check(conn)
        if asyncio.current_task() is not runner._reaper:
            return
        reaper_checking.set()
        try:
            await asyncio.sleep(0.3)
        finally:
            reaper_checking.clear()

    pool._check = slow_check
    try:
        runner.start_reaper(interval=3600)
        await asyncio.wait_for(reaper_checking.wait(), 5)  # reaper inside getconn's check
        reaper = runner._reaper
        # Before the fix: the 3600 s sleep. Not wait_for (see finishes_within).
        assert await finishes_within(runner.shutdown(), 5)
        assert reaper is not None and reaper.done()
    finally:
        pool._check = real_check
    row = await _row(store, job)
    assert row["status"] == "failed" and row["error"] == INTERRUPTED_ERROR
    assert row["callback_state"] == "delivered"
    assert [(b["status"], b["error"]) for _, b in recorder.posts] == [("failed", INTERRUPTED_ERROR)]


# ── two replicas racing ───────────────────────────────────────────────


async def test_two_reapers_reap_each_orphan_exactly_once(store, recorder):
    dead = [await _job(store, "pod-dead", lease=0) for _ in range(6)]
    a, b = _runner(ExecJobStore(), "pod-a"), _runner(ExecJobStore(), "pod-b")
    counts = await asyncio.gather(a.reap_once(), b.reap_once(), a.reap_once(), b.reap_once())
    assert sum(counts) == 6
    posted = sorted(body["job_id"] for _, body in recorder.posts)
    assert posted == sorted(j["job_id"] for j in dead)  # each once, none twice
    for j in dead:
        assert (await _row(store, j))["callback_state"] == "delivered"


async def test_owner_finish_and_reaper_race_has_one_winner(store, recorder):
    for _ in range(10):
        job = await _job(store, "pod-a", lease=0)
        finished, reaped = await asyncio.gather(
            store.finish(
                job_id=job["job_id"],
                owner="pod-a",
                status="completed",
                exit_code=0,
                callback_lease_seconds=120,
            ),
            store.reap_expired(reaper="pod-b", error=ORPHANED_ERROR, callback_lease_seconds=120),
        )
        winners = (finished is not None) + len(reaped)
        assert winners == 1
        row = await _row(store, job)
        assert row["callback_owner"] == ("pod-a" if finished is not None else "pod-b")


async def test_stuck_callback_is_resent_by_exactly_one_replica(store, recorder):
    """The sender died after the terminal transition, before recording delivery."""
    job = await _job(store, "pod-a")
    row = await store.finish(
        job_id=job["job_id"],
        owner="pod-a",
        status="completed",
        exit_code=0,
        callback_lease_seconds=0,  # its send claim has already lapsed
    )
    assert row is not None and row["callback_state"] == "sending"
    a, b = _runner(ExecJobStore(), "pod-b"), _runner(ExecJobStore(), "pod-c")
    counts = await asyncio.gather(a.reap_once(), b.reap_once())
    assert sum(counts) == 1
    assert [body["job_id"] for _, body in recorder.posts] == [job["job_id"]]
    final = await _row(store, job)
    assert final["callback_state"] == "delivered"
    assert final["callback_owner"] in ("pod-b", "pod-c")


async def test_a_live_send_claim_is_not_stolen(store, recorder):
    job = await _job(store, "pod-a")
    await store.finish(
        job_id=job["job_id"],
        owner="pod-a",
        status="completed",
        exit_code=0,
        callback_lease_seconds=120,
    )
    assert await _runner(store, "pod-b").reap_once() == 0
    assert recorder.posts == []
    # and only the claim holder may record the outcome
    assert not await store.record_callback(job_id=job["job_id"], owner="pod-b", outcome="delivered")
    assert await store.record_callback(job_id=job["job_id"], owner="pod-a", outcome="delivered")


# ── tenant isolation of the read ──────────────────────────────────────


async def test_status_read_is_tenant_and_workspace_scoped(store):
    job = await _job(store, "pod-a")
    assert await store.get_job(job_id=job["job_id"], tenant_id=TENANT, container_name=WS)
    assert (
        await store.get_job(job_id=job["job_id"], tenant_id="tenant-b", container_name=WS) is None
    )
    assert (
        await store.get_job(job_id=job["job_id"], tenant_id=TENANT, container_name="ws-2") is None
    )


async def test_status_route_hides_another_tenants_job(store, app, client):
    from coding_client.auth.tenant import require_tenant
    from src.routers import exec_async

    job = await _job(store, "pod-a")
    runner = _runner(store, "pod-a")
    url = f"/api/workspaces/{WS}/exec/jobs/{job['job_id']}"
    with patch.object(exec_async, "get_exec_job_runner", return_value=runner):
        try:
            app.dependency_overrides[require_tenant] = lambda: TENANT
            mine = await client.get(url)
            app.dependency_overrides[require_tenant] = lambda: "tenant-b"
            theirs = await client.get(url)
        finally:
            app.dependency_overrides.pop(require_tenant, None)
    assert mine.status_code == 200 and mine.json()["status"] == "queued"
    assert theirs.status_code == 404


async def test_finish_rejects_a_non_terminal_status(store):
    job = await _job(store, "pod-a")
    with pytest.raises(ValueError):
        await store.finish(
            job_id=job["job_id"], owner="pod-a", status="running", callback_lease_seconds=1
        )


# ── D1/D2: retryable callbacks are durable and never given up ─────────

N8N_RUNNING = (409, '{"code":409,"message":"The execution \\"4711\\" is running already."}')
N8N_FINISHED = (409, '{"code":409,"message":"The execution \\"4711 has finished already."}')


def _sql(query: str, *params):
    with psycopg.connect(db_session.normalize_dsn(DSN), autocommit=True) as conn:
        cur = conn.execute(query, params)
        return cur.fetchall() if cur.description else []


def _make_due(job_id: str) -> None:
    """Time passing, without sleeping: the scheduled retry is now due."""
    _sql(
        "UPDATE exec_jobs SET callback_next_attempt_at = now() - interval '1 second' "
        "WHERE id = %s AND callback_state = 'retrying'",
        job_id,
    )


@pytest.fixture
def one_attempt_rounds(monkeypatch):
    """One POST per round, so a test drives rounds without the 2 s / 4 s sleeps."""
    monkeypatch.setattr(cb, "CALLBACK_ATTEMPTS", 1)


async def test_fast_job_running_already_is_retried_until_the_cell_parks(
    store, recorder, one_attempt_rounds
):
    """D1: the job ends before the cell parks on its Wait node; n8n answers 409
    "running already". The callback must survive that and land once it parks."""
    recorder.script = [N8N_RUNNING]
    runner = _runner(store, "pod-a", _executor(exc=not_found("Container", WS)))
    job = await _job(store, "pod-a")
    await runner.submit(job, ["ls"])
    row = await _row(store, job)
    assert row["status"] == "failed"
    assert row["callback_state"] == "retrying" and row["callback_attempts"] == 1
    assert row["callback_owner"] is None and row["callback_next_attempt_at"] is not None
    assert await runner.reap_once() == 0  # not due yet: no hot loop
    _make_due(job["job_id"])
    assert await runner.reap_once() == 1  # the cell has parked now: 200
    row = await _row(store, job)
    assert row["callback_state"] == "delivered" and row["callback_attempts"] == 2
    assert row["callback_next_attempt_at"] is None
    assert [b["status"] for _, b in recorder.posts] == ["failed", "failed"]


async def test_outage_longer_than_any_round_is_retried_across_pod_restarts(
    store, recorder, one_attempt_rounds
):
    """D2: n8n/tailnet down for many rounds; a different replica (the original pod
    is gone) picks the due retry up; delivered once n8n is back. Never given up."""
    down = httpx.ConnectError("connection refused")
    recorder.script = [down, (502, "bad gateway"), down, (503, ""), down, down, down, down]
    job = await _job(store, "pod-a")
    await _runner(store, "pod-a", _executor()).submit(job, ["ls"])
    survivors = [_runner(ExecJobStore(), f"pod-{n}") for n in "bcdefgh"]
    for runner in survivors:  # pod-a is gone; each round lands on another replica
        row = await _row(store, job)
        assert row["callback_state"] == "retrying"
        _make_due(job["job_id"])
        assert await runner.reap_once() == 1
    row = await _row(store, job)
    assert row["callback_state"] == "retrying" and row["callback_attempts"] == 8
    _make_due(job["job_id"])
    assert await _runner(store, "pod-z").reap_once() == 1  # n8n is back: 200
    row = await _row(store, job)
    assert row["callback_state"] == "delivered" and row["callback_attempts"] == 9
    assert row["callback_owner"] == "pod-z"
    assert len(recorder.posts) == 9
    assert await _runner(store, "pod-z").reap_once() == 0  # done: nothing resent


async def test_retry_backoff_is_scheduled_by_the_database_and_capped(
    store, recorder, one_attempt_rounds
):
    recorder.status = 503
    job = await _job(store, "pod-a")
    _sql("UPDATE exec_jobs SET callback_attempts = 3000 WHERE id = %s", job["job_id"])
    await _runner(store, "pod-a", _executor()).submit(job, ["ls"])
    ((delay,),) = _sql(
        "SELECT extract(epoch FROM callback_next_attempt_at - now()) FROM exec_jobs WHERE id = %s",
        job["job_id"],
    )
    assert 250 < float(delay) <= cb.RETRY_CAP_S


async def test_due_retry_is_sent_by_exactly_one_of_two_replicas(
    store, recorder, one_attempt_rounds
):
    jobs = [await _job(store, "pod-a") for _ in range(5)]
    runner = _runner(store, "pod-a", _executor())
    for job in jobs:
        recorder.script = [(503, "")]
        await runner.submit(job, ["ls"])
        _make_due(job["job_id"])
    sent_before = len(recorder.posts)
    a, b = _runner(ExecJobStore(), "pod-b"), _runner(ExecJobStore(), "pod-c")
    counts = await asyncio.gather(a.reap_once(), b.reap_once(), a.reap_once(), b.reap_once())
    assert sum(counts) == 5
    resent = sorted(body["job_id"] for _, body in recorder.posts[sent_before:])
    assert resent == sorted(j["job_id"] for j in jobs)  # each once, none twice
    for job in jobs:
        assert (await _row(store, job))["callback_state"] == "delivered"


async def test_finished_execution_is_final_and_never_resent(store, recorder, one_attempt_rounds):
    recorder.script = [N8N_FINISHED]
    job = await _job(store, "pod-a")
    await _runner(store, "pod-a", _executor()).submit(job, ["ls"])
    row = await _row(store, job)
    assert row["callback_state"] == "refused" and row["callback_next_attempt_at"] is None
    assert await _runner(store, "pod-b").reap_once() == 0
    assert len(recorder.posts) == 1


async def test_migration_revives_given_up_callbacks(store):
    """Rows a pre-0002 deploy gave up on (``undeliverable``) become due retries."""
    from alembic.config import Config

    from alembic import command

    cfg = Config()
    cfg.set_main_option("script_location", "alembic")
    old = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = DSN
    try:
        command.downgrade(cfg, "0001")
        job_id = str(uuid.uuid7())
        _sql(
            "INSERT INTO exec_jobs (id, tenant_id, container_name, command_summary, workdir, "
            "timeout_seconds, status, owner, callback_url, callback_state, callback_owner) "
            "VALUES (%s, 't', 'w', 'ls', '/', 1, 'failed', 'p', %s, 'undeliverable', 'p')",
            job_id,
            GOOD,
        )
        command.upgrade(cfg, "head")
    finally:
        if old is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = old
    ((state, due, owner),) = _sql(
        "SELECT callback_state, callback_next_attempt_at <= now(), callback_owner "
        "FROM exec_jobs WHERE id = %s",
        job_id,
    )
    assert (state, due, owner) == ("retrying", True, None)


# ── D5: callback_url is an idempotency key for active jobs ────────────


async def _create(store, *, tenant: str = TENANT, url: str = GOOD, ws: str = WS):
    return await store.create_job(
        job_id=str(uuid.uuid7()),
        tenant_id=tenant,
        container_name=ws,
        command_summary="ls",
        workdir="/workspace",
        timeout_seconds=60,
        callback_url=url,
        owner="pod-a",
        lease_seconds=60,
    )


async def test_retried_post_returns_the_active_job(store):
    first, created = await _create(store)
    assert created
    again, created_again = await _create(store)
    assert created_again is False and again["job_id"] == first["job_id"]
    other_tenant, created_other = await _create(store, tenant="tenant-b")
    assert created_other and other_tenant["job_id"] != first["job_id"]
    ((n,),) = _sql("SELECT count(*) FROM exec_jobs WHERE tenant_id = %s", TENANT)
    assert n == 1


async def _finish(store, job, owner: str = "pod-a"):
    row = await store.finish(
        job_id=job["job_id"], owner=owner, status="completed", callback_lease_seconds=60
    )
    assert row is not None and row["callback_state"] == "sending"
    return row


async def test_replay_after_finish_while_callback_retrying_returns_the_same_job(store):
    """Validator repro: terminal job, callback parked as ``retrying`` — the key is
    still held, so the replayed POST must NOT start a second run."""
    first, _ = await _create(store)
    await _finish(store, first)
    assert await store.record_callback(
        job_id=first["job_id"], owner="pod-a", outcome="retrying", retry_in_seconds=30
    )
    again, created = await _create(store)
    assert created is False and again["job_id"] == first["job_id"]
    assert (again["status"], again["callback_state"]) == ("completed", "retrying")
    ((n,),) = _sql("SELECT count(*) FROM exec_jobs WHERE tenant_id = %s", TENANT)
    assert n == 1


async def test_replay_while_callback_sending_returns_the_holder_not_a_new_job(store):
    """The route turns this holder (terminal, ``sending``) into 503 + Retry-After."""
    first, _ = await _create(store)
    await _finish(store, first)
    again, created = await _create(store)
    assert created is False and again["job_id"] == first["job_id"]
    assert (again["status"], again["callback_state"]) == ("completed", "sending")


@pytest.mark.parametrize("final", ["delivered", "refused", "disallowed"])
async def test_idempotency_key_frees_up_once_the_callback_is_final(store, final):
    """Loops reuse one resume URL: once the callback has its final answer, the
    same URL may start a new job."""
    first, _ = await _create(store)
    await _finish(store, first)
    assert await store.record_callback(job_id=first["job_id"], owner="pod-a", outcome=final)
    second, created = await _create(store)
    assert created and second["job_id"] != first["job_id"]


async def test_job_without_callback_url_never_holds_a_key(store):
    a, created_a = await _create(store, url=None)
    b, created_b = await _create(store, url=None)
    assert created_a and created_b and a["job_id"] != b["job_id"]


async def test_concurrent_replays_across_the_terminal_to_callback_window_create_one_job(store):
    """Replays racing the terminal transition and the retrying hand-off (two
    replicas' worth of connections) never start a second job."""
    first, _ = await _create(store)

    async def replays(n: int):
        return await asyncio.gather(*(_create(ExecJobStore()) for _ in range(n)))

    finished, *res1 = await asyncio.gather(
        _finish(store, first), *(_create(ExecJobStore()) for _ in range(6))
    )
    recorded, *res2 = await asyncio.gather(
        store.record_callback(
            job_id=first["job_id"], owner="pod-a", outcome="retrying", retry_in_seconds=30
        ),
        *(_create(ExecJobStore()) for _ in range(6)),
    )
    assert finished is not None and recorded is True
    res3 = await replays(6)
    for row, created in [*res1, *res2, *res3]:
        assert created is False and row["job_id"] == first["job_id"]
    ((n,),) = _sql("SELECT count(*) FROM exec_jobs WHERE tenant_id = %s", TENANT)
    assert n == 1


async def test_replays_racing_the_final_callback_create_at_most_one_new_job(store):
    """The holder releases the key while replays run: each replay either gets
    the old holder (still held) or the ONE new job — never two new jobs, and
    never a released row passed off as the holder."""
    first, _ = await _create(store)
    await _finish(store, first)
    delivered, *results = await asyncio.gather(
        store.record_callback(job_id=first["job_id"], owner="pod-a", outcome="delivered"),
        *(_create(ExecJobStore()) for _ in range(8)),
    )
    assert delivered is True
    new_ids = {row["job_id"] for row, created in results if created}
    assert len(new_ids) <= 1
    for row, created in results:
        if not created:
            assert row["job_id"] == first["job_id"] or row["job_id"] in new_ids
            if row["job_id"] == first["job_id"]:
                assert row["callback_state"] == "sending"  # read while it still held
    # A replay after the release always lands on a holder: the one new job.
    last, created = await _create(store)
    if new_ids:
        assert created is False and last["job_id"] in new_ids
    else:
        assert created
    ((n,),) = _sql(
        "SELECT count(*) FROM exec_jobs WHERE tenant_id = %s AND status IN ('queued','running')",
        TENANT,
    )
    assert n == 1


@pytest.mark.parametrize("final", ["delivered", "refused"])
async def test_read_back_never_returns_a_holder_that_released_between_insert_and_select(
    store, monkeypatch, final
):
    """Forces the interleave the racing test above only samples: the INSERT
    conflicts with the holder, then the holder's callback reaches its final state
    BEFORE the read-back SELECT runs. The read-back filters on ``_HOLDS_KEY``, so
    it finds nothing, the INSERT is retried and a NEW job is created — the
    released holder is never handed back as if it still held the key."""
    first, _ = await _create(store)
    await _finish(store, first)
    real_one = store._one
    queries: list[str] = []

    async def interleaving_one(sql, params):
        kind = sql.lstrip().split(None, 1)[0].lower()
        if kind == "update":  # the holder's own record_callback below
            return await real_one(sql, params)
        queries.append(kind)
        if kind == "select" and queries == ["insert", "select"]:
            # The holder's sender records the final outcome right here.
            assert await store.record_callback(job_id=first["job_id"], owner="pod-a", outcome=final)
        return await real_one(sql, params)

    monkeypatch.setattr(store, "_one", interleaving_one)
    row, created = await _create(store)
    monkeypatch.undo()

    assert queries == ["insert", "select", "insert"]  # the interleave really happened
    assert created is True
    assert row["job_id"] != first["job_id"]
    assert (row["status"], row["callback_state"]) == ("queued", "none")
    ((n,),) = _sql(
        "SELECT count(*) FROM exec_jobs WHERE tenant_id = %s AND callback_url = %s",
        TENANT,
        GOOD,
    )
    assert n == 2


async def test_database_itself_rejects_a_second_job_while_the_callback_is_pending(store):
    first, _ = await _create(store)
    await _finish(store, first)
    assert await store.record_callback(
        job_id=first["job_id"], owner="pod-a", outcome="retrying", retry_in_seconds=30
    )
    with pytest.raises(psycopg.errors.UniqueViolation):
        _sql(
            "INSERT INTO exec_jobs (id, tenant_id, container_name, command_summary, workdir, "
            "timeout_seconds, status, owner, callback_url) "
            "VALUES (%s, %s, %s, 'ls', '/', 1, 'queued', 'p', %s)",
            str(uuid.uuid7()),
            TENANT,
            WS,
            GOOD,
        )


async def test_migration_resolves_duplicate_holders_from_a_0001_database(store):
    """0001 had no index, so it can hold several jobs per key. 0002 keeps the
    oldest as holder and fails/settles the newer ones instead of aborting on a
    UniqueViolation."""
    from alembic.config import Config

    from alembic import command

    cfg = Config()
    cfg.set_main_option("script_location", "alembic")
    old = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = DSN
    ids = [str(uuid.uuid7()) for _ in range(4)]
    try:
        command.downgrade(cfg, "0001")
        rows = [  # (id, status, callback_state, age) — ids[0] is the oldest
            (ids[0], "running", "none", "3 minutes"),
            (ids[1], "queued", "none", "2 minutes"),
            (ids[2], "failed", "sending", "1 minute"),
            (ids[3], "completed", "delivered", "4 minutes"),  # holds nothing
        ]
        for job_id, st, cbs, age in rows:
            _sql(
                "INSERT INTO exec_jobs (id, tenant_id, container_name, command_summary, workdir, "
                "timeout_seconds, status, owner, callback_url, callback_state, callback_owner, "
                "created_at) VALUES (%s, 't', 'w', 'ls', '/', 1, %s, 'p', %s, %s, 'p', "
                "now() - %s::interval)",
                job_id,
                st,
                GOOD,
                cbs,
                age,
            )
        command.upgrade(cfg, "head")
    finally:
        if old is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = old
    got = {
        r[0]: r[1:]
        for r in _sql(
            "SELECT id::text, status, callback_state, error FROM exec_jobs WHERE tenant_id = 't'"
        )
    }
    assert got[ids[0]] == ("running", "none", None)  # oldest: stays the holder
    assert got[ids[1]] == ("failed", "refused", "duplicate_callback_url")
    assert got[ids[2]] == ("failed", "refused", None)  # its callback is never sent
    assert got[ids[3]] == ("completed", "delivered", None)


async def test_concurrent_duplicate_posts_create_one_job(store):
    results = await asyncio.gather(*(_create(ExecJobStore()) for _ in range(6)))
    assert sum(created for _, created in results) == 1
    assert len({row["job_id"] for row, _ in results}) == 1


async def test_database_itself_rejects_a_second_active_job_per_callback_url(store):
    first, _ = await _create(store)
    with pytest.raises(psycopg.errors.UniqueViolation):
        _sql(
            "INSERT INTO exec_jobs (id, tenant_id, container_name, command_summary, workdir, "
            "timeout_seconds, status, owner, callback_url) "
            "VALUES (%s, %s, %s, 'ls', '/', 1, 'queued', 'p', %s)",
            str(uuid.uuid7()),
            TENANT,
            WS,
            GOOD,
        )

"""ExecJobRunner.shutdown is bounded even when a cancel is swallowed below it.

psycopg_pool 3.3.1 ``AsyncConnectionPool.getconn`` catches ``CancelledError``
raised during its connection check (``CLIENT_EXCEPTIONS`` includes it), puts the
connection back and retries. A cancel that lands there is swallowed: the reaper
finishes its round and goes into ``asyncio.sleep(interval)``, and a job task
carries on running its command. A single ``cancel()`` + unbounded ``await`` then
holds the pod until SIGKILL and skips the ``interrupted`` callbacks. The
real-pool reproduction is in test_exec_jobs_pg.py; these pin the runner's
contract without a database. Same shape as coding-github
tests/unit/test_ci_wait_manager_shutdown.py.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.services.exec_jobs.runner import ExecJobRunner


async def _swallow_cancels(state: SwallowingStore) -> None:
    """Sleep "forever" like an in-flight connection check, swallowing up to
    ``state.swallow`` cancels (each swallowed cancel returns normally)."""
    while True:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            if state.swallow <= 0:
                raise
            state.swallow -= 1
            return


class SwallowingStore:
    """A store whose reaper query swallows the first N cancels, like getconn's check."""

    def __init__(self, swallow: int) -> None:
        self.swallow = swallow
        self.in_reap = asyncio.Event()
        self.reaps = 0
        self.claims = 0
        self.failed_owned: list[str] = []

    async def reap_expired(self, **_: Any) -> list[dict[str, Any]]:
        self.reaps += 1
        self.in_reap.set()
        await _swallow_cancels(self)
        return []

    async def claim_due_callbacks(self, **_: Any) -> list[dict[str, Any]]:
        self.claims += 1
        return []

    async def fail_owned_active(self, *, owner: str, **_: Any) -> list[dict[str, Any]]:
        self.failed_owned.append(owner)
        return []

    async def mark_running(self, **_: Any) -> bool:
        return True

    async def renew_lease(self, **_: Any) -> bool:
        return True

    async def finish(self, **_: Any) -> dict[str, Any] | None:
        return None


async def finishes_within(coro: Any, seconds: float) -> bool:
    """True if ``coro`` completes within ``seconds``; never cancels it to find out
    (a cancel can make a hung coroutine return and read as a pass)."""
    task = asyncio.ensure_future(coro)
    done, _ = await asyncio.wait({task}, timeout=seconds)
    if not done:
        task.cancel()
        await asyncio.wait({task}, timeout=2)
        return False
    task.result()
    return True


def _runner(store: SwallowingStore, executor: Any = None) -> ExecJobRunner:
    return ExecJobRunner(
        store,  # type: ignore[arg-type]
        owner="pod-a",
        lease_seconds=60,
        max_concurrent=4,
        executor=executor,
    )


async def test_a_swallowed_reaper_cancel_does_not_leave_shutdown_waiting_out_the_interval():
    store = SwallowingStore(swallow=1)
    runner = _runner(store)
    runner.start_reaper(interval=3600)
    await asyncio.wait_for(store.in_reap.wait(), 2)
    reaper = runner._reaper
    assert reaper is not None

    # Before the fix: blocked for the 3600 s sleep. Not wait_for: cancelling the
    # old shutdown() made it swallow its OWN cancel and return "normally".
    assert await finishes_within(runner.shutdown(), 2)

    assert reaper.done()
    assert store.swallow == 0  # the cancel really was swallowed once
    assert store.reaps == 1  # and no second sweep round started after it
    assert store.failed_owned == ["pod-a"]  # the interrupted pass still ran


async def test_shutdown_gives_up_on_a_reaper_that_never_stops_and_still_fails_owned_jobs(caplog):
    store = SwallowingStore(swallow=10**9)  # swallows every cancel, forever
    runner = _runner(store)

    async def stubborn(_interval: float) -> None:
        while True:
            await store.reap_expired()

    runner._reaper_loop = stubborn  # type: ignore[method-assign]
    runner.start_reaper(interval=3600)
    await asyncio.wait_for(store.in_reap.wait(), 2)
    reaper = runner._reaper
    assert reaper is not None

    loop = asyncio.get_running_loop()
    try:
        t0 = loop.time()
        with caplog.at_level(logging.WARNING, logger="src.services.exec_jobs.runner"):
            assert await finishes_within(runner.shutdown(), 6)
        elapsed = loop.time() - t0

        # The reaper gets its own 2 s sub-budget, not the whole deadline ...
        assert 1.9 <= elapsed < 3.5
        assert not reaper.done()
        assert "reaper task(s) did not stop" in caplog.text
        # ... so the owned jobs are still failed (and called back) after it.
        assert store.failed_owned == ["pod-a"]
    finally:
        store.swallow = 0  # let the test's loop tear down cleanly whatever happened
        reaper.cancel()
        await asyncio.wait({reaper}, timeout=2)


async def test_a_job_task_that_swallows_one_cancel_is_re_cancelled():
    """No reaper at all: pins the re-cancel loop over the spawned job tasks on its
    own (a cancel swallowed inside the command's getconn, then a long await)."""
    store = SwallowingStore(swallow=0)
    started = asyncio.Event()
    swallowed = asyncio.Event()

    async def executor(*_a: Any, **_k: Any) -> tuple[Any, Any]:
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            swallowed.set()  # swallowed once ...
        await asyncio.sleep(3600)  # ... then the command keeps "running"
        raise AssertionError("unreachable")

    runner = _runner(store, executor)
    task = runner.submit(
        {
            "job_id": "j1",
            "container_name": "ws-1",
            "tenant_id": "tenant-a",
            "workdir": "/workspace",
            "timeout_seconds": 3600,
        },
        ["python", "long.py"],
    )
    await asyncio.wait_for(started.wait(), 2)

    # Before the fix: ``task.cancel(); await task`` waited out the second sleep.
    assert await finishes_within(runner.shutdown(), 2)

    assert swallowed.is_set()
    assert task.done() and task.cancelled()


async def test_reaper_loop_returns_after_its_sweep_once_closing():
    """A swallowed cancel must not be followed by a full ``interval`` of sleep."""
    store = SwallowingStore(swallow=0)
    runner = _runner(store)
    runner._closing = True
    sweep = asyncio.ensure_future(runner._reaper_loop(3600))
    await asyncio.wait_for(store.in_reap.wait(), 2)
    store.swallow = 1
    sweep.cancel()  # swallowed by the stand-in check
    assert await finishes_within(sweep, 1)
    assert store.claims == 1

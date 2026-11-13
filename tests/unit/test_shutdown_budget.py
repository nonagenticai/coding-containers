"""The SIGTERM budget: Dockerfile CMD flag + lifespan stop bounds fit the grace period.

uvicorn runs the lifespan shutdown (exec runner -> ``interrupted`` callbacks) only
after in-flight requests finish or ``--timeout-graceful-shutdown`` expires. With no
flag a 300 s sync exec held the process until SIGKILL and the lifespan never ran.
These pin (1) the flag is on the CMD the image actually runs, (2) its value is the
one the budget is computed from, and (3) the sum of every bounded stop step fits
``TERMINATION_GRACE_PERIOD_S`` (the value the Deployment must carry in gitops).
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import uvicorn.server

from src.core import reaper
from src.db import session
from src.services.exec_jobs import runner

DOCKERFILE = Path(__file__).resolve().parents[2] / "Dockerfile"


def _cmd_argv() -> list[str]:
    """The exec-form CMD of the final stage (the last CMD wins)."""
    cmds = re.findall(r"^CMD\s+(\[.*\])\s*$", DOCKERFILE.read_text(), flags=re.MULTILINE)
    assert cmds, "Dockerfile has no exec-form CMD"
    argv = json.loads(cmds[-1])
    assert isinstance(argv, list)
    return [str(a) for a in argv]


def _flag_value(argv: list[str], flag: str) -> str | None:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def test_cmd_runs_uvicorn_with_the_budgeted_graceful_shutdown_timeout() -> None:
    argv = _cmd_argv()
    assert argv[0] == "uvicorn"
    value = _flag_value(argv, "--timeout-graceful-shutdown")
    assert value is not None, (
        "uvicorn CMD has no --timeout-graceful-shutdown: a long in-flight request "
        "holds the pod until SIGKILL and the lifespan shutdown never runs"
    )
    assert int(value) == runner.HTTP_GRACEFUL_SHUTDOWN_S


def test_lifespan_still_fits_the_termination_grace_period() -> None:
    worst_case = (
        runner.UVICORN_PRE_WAIT_S
        + runner.HTTP_GRACEFUL_SHUTDOWN_S
        + runner.SHUTDOWN_TIMEOUT_S
        + session.POOL_CLOSE_TIMEOUT_S
        + reaper._STOP_TIMEOUT_SECONDS
    )
    # At least 5 s of margin for the kubelet signal path and process teardown.
    assert worst_case + 5 <= runner.TERMINATION_GRACE_PERIOD_S, worst_case


def test_interrupted_callbacks_finish_before_the_grace_period() -> None:
    callbacks_done_by = (
        runner.UVICORN_PRE_WAIT_S + runner.HTTP_GRACEFUL_SHUTDOWN_S + runner.SHUTDOWN_TIMEOUT_S
    )
    assert callbacks_done_by < runner.TERMINATION_GRACE_PERIOD_S


def test_graceful_timeout_leaves_requests_less_than_the_sync_exec_cap() -> None:
    """The flag is only useful if it cuts a long request off; 0 would cut every one."""
    assert 0 < runner.HTTP_GRACEFUL_SHUTDOWN_S < 300


def test_uvicorn_still_waits_its_fixed_pre_wait_then_the_graceful_timeout() -> None:
    """The budget models uvicorn's Server.shutdown order; fail loudly if it changes."""
    src = inspect.getsource(uvicorn.server.Server.shutdown)
    assert f"asyncio.sleep({runner.UVICORN_PRE_WAIT_S})" in src
    assert "timeout_graceful_shutdown" in src
    assert src.index("timeout_graceful_shutdown") < src.index("lifespan.shutdown")

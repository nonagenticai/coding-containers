"""The detached wrapper scripts, EXECUTED by a real /bin/sh (not string-matched).

The scripts ``detached.py`` builds are run here exactly as the workspace pod runs
them: launch (detached), poll until done, collect. That proves the exit code,
the per-stream output cap and the exact dropped-byte count, the timeout, the
working directory, and the "missing" / "lost" verdicts — against coreutils, not
against an assumption about them. Skipped only where ``timeout``/``mkfifo`` are
absent (the workspace image is Debian and has both).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from src.services.exec_jobs import detached as dx

pytestmark = pytest.mark.skipif(
    not (shutil.which("timeout") and shutil.which("mkfifo")), reason="needs coreutils timeout"
)


def _gnu_head() -> bool:
    probe = subprocess.run(["head", "--version"], capture_output=True, text=True, check=False)  # noqa: S607
    return probe.returncode == 0 and "GNU" in probe.stdout


def _sh(argv: list[str], cwd: str = "/") -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=30, check=False)  # noqa: S603


def _launch(d: Path, argv: list[str], *, cwd: str, timeout: int = 30, cap: int = 1 << 20) -> None:
    script = dx.build_run_script(str(d), argv, cwd, timeout, cap)
    done = _sh(dx.build_launcher(str(d), script))
    assert done.returncode == 0 and "launched" in done.stdout, done.stderr


def _wait_done(d: Path, seconds: float = 15) -> dx.PollResult:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        status = dx.parse_poll(_sh(dx.build_poll(str(d))).stdout)
        if status.state is not dx.PollState.RUNNING:
            return status
        time.sleep(0.05)
    raise AssertionError("detached command did not finish")


def _collect(d: Path, status: dx.PollResult) -> dx.DetachedOutcome:
    out = _sh(dx.build_collect(str(d)))
    return dx.outcome_from(status, out.stdout, out.stderr)


def test_exit_code_streams_and_cwd(tmp_path: Path) -> None:
    work = tmp_path / "repo"
    work.mkdir()
    d = tmp_path / "job"
    _launch(d, ["sh", "-c", "pwd; echo err >&2; exit 7"], cwd=str(work))
    status = _wait_done(d)
    assert status.state is dx.PollState.DONE and status.exit_code == 7
    out = _collect(d, status)
    assert out.exit_code == 7 and not out.timed_out
    assert Path(out.stdout.strip()).resolve() == work.resolve()
    assert out.stderr == "err\n"
    assert out.bytes_omitted == 0 and not out.stdout_truncated
    assert out.duration is not None and out.duration >= 0


def test_output_is_capped_per_stream_and_the_drop_is_counted(tmp_path: Path) -> None:
    d = tmp_path / "job"
    _launch(
        d,
        [
            "sh",
            "-c",
            "head -c 50000 /dev/zero | tr '\\0' a; head -c 300 /dev/zero | tr '\\0' b >&2",
        ],
        cwd=str(tmp_path),
        cap=1000,
    )
    status = _wait_done(d)
    out = _collect(d, status)
    assert out.exit_code == 0
    assert out.stdout == "a" * 1000  # exactly the cap reached disk
    assert out.stdout_truncated and out.bytes_omitted == status.stdout_omitted
    if _gnu_head():
        # The workspace image (Debian coreutils): ``head -c`` never over-reads,
        # so the count is exact. BSD head reads ahead a block, so elsewhere the
        # count is only bounded.
        assert status.stdout_omitted == 49000
    else:
        assert 0 < status.stdout_omitted <= 49000
    assert out.stderr == "b" * 300 and not out.stderr_truncated
    assert (d / "stdout").stat().st_size == 1000  # the pod's /tmp never held more


def test_timeout_is_exit_124(tmp_path: Path) -> None:
    d = tmp_path / "job"
    _launch(d, ["sleep", "30"], cwd=str(tmp_path), timeout=1)
    out = _collect(d, _wait_done(d))
    assert out.exit_code == dx.TIMEOUT_EXIT_CODE and out.timed_out


def test_the_command_survives_its_launcher(tmp_path: Path) -> None:
    """The launch exec returns at once; the command finishes afterwards."""
    d = tmp_path / "job"
    started = time.monotonic()
    _launch(d, ["sh", "-c", "sleep 1; echo late"], cwd=str(tmp_path))
    assert time.monotonic() - started < 1.0
    assert dx.parse_poll(_sh(dx.build_poll(str(d))).stdout).state is dx.PollState.RUNNING
    out = _collect(d, _wait_done(d))
    assert out.stdout == "late\n"


def test_missing_directory_is_reported(tmp_path: Path) -> None:
    status = dx.parse_poll(_sh(dx.build_poll(str(tmp_path / "never"))).stdout)
    assert status.state is dx.PollState.MISSING


def test_dead_wrapper_without_exit_status_is_lost(tmp_path: Path) -> None:
    d = tmp_path / "job"
    _launch(d, ["sleep", "30"], cwd=str(tmp_path))
    for _ in range(100):
        if (d / "pid").exists() and (d / "pid").read_text().strip():
            break
        time.sleep(0.02)
    os.kill(int((d / "pid").read_text()), signal.SIGKILL)  # the container restarted
    for _ in range(100):
        status = dx.parse_poll(_sh(dx.build_poll(str(d))).stdout)
        if status.state is not dx.PollState.RUNNING:
            break
        time.sleep(0.05)
    assert status.state is dx.PollState.LOST


def test_a_failed_launch_never_prints_the_marker(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    done = _sh(dx.build_launcher(str(blocker / "job"), "echo hi"))
    assert "launched" not in done.stdout


def test_hostile_argv_and_cwd_are_quoted(tmp_path: Path) -> None:
    work = tmp_path / "it's here; touch PWNED"
    work.mkdir()
    d = tmp_path / "job"
    _launch(d, ["printf", "%s|", "$(touch PWNED)", "a b"], cwd=str(work))
    out = _collect(d, _wait_done(d))
    assert out.stdout == "$(touch PWNED)|a b|"
    assert not (work / "PWNED").exists() and not (tmp_path / "PWNED").exists()

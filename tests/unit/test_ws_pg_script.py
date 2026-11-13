"""`ws-pg psql` argument gate (py2rust A1 fix, #182).

Runs the real scripts/ws-pg under /bin/sh with stub `psql` / `pg_ctl` on PATH.
The stub psql records the argv and environment it was exec'd with, so each test
proves either that psql was NEVER run (refusals) or exactly what it received.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ws-pg"

STUB_PSQL = """#!/bin/sh
for a in "$@"; do printf '%s\\n' "$a"; done > "$WS_PG_TEST_ARGV"
env > "$WS_PG_TEST_ENV"
"""
STUB_PG_CTL = "#!/bin/sh\nexit 0\n"


@pytest.fixture
def ws(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("psql", STUB_PSQL), ("pg_ctl", STUB_PG_CTL)):
        stub = bin_dir / name
        stub.write_text(body)
        stub.chmod(0o755)
    data = tmp_path / "data"
    data.mkdir()
    (data / "PG_VERSION").write_text("18\n")
    argv_file = tmp_path / "argv"
    env_file = tmp_path / "env"

    def run(*args: str, extra_env: dict[str, str] | None = None):
        env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "WS_PG_DATA": str(data),
            "WS_PG_RUN": str(tmp_path / "run"),
            "WS_PG_PORT": "5432",
            "WS_PG_TEST_ARGV": str(argv_file),
            "WS_PG_TEST_ENV": str(env_file),
            **(extra_env or {}),
        }
        proc = subprocess.run(  # noqa: S603 - fixed script path, test-controlled argv
            ["/bin/sh", str(SCRIPT), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        argv = argv_file.read_text().splitlines() if argv_file.exists() else None
        child_env = env_file.read_text() if env_file.exists() else None
        return proc, argv, child_env

    return run


def test_fixed_flags_and_only_c_args_reach_psql(ws):
    proc, argv, _ = ws("psql", "parity_sou_ab12", "-c", "BEGIN READ ONLY", "-c", "SELECT 1")
    assert proc.returncode == 0, proc.stderr
    user = argv[argv.index("-U") + 1]
    assert argv == [
        "-X", "--no-psqlrc", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1",
        "-h", "127.0.0.1", "-p", "5432", "-U", user, "-d", "parity_sou_ab12",
        "-c", "BEGIN READ ONLY", "-c", "SELECT 1",
    ]  # fmt: skip


def test_libpq_redirect_env_is_unset(ws):
    hostile = {
        "PGHOST": "evil",
        "PGHOSTADDR": "1.2.3.4",
        "PGSERVICE": "evil",
        "PGSERVICEFILE": "/tmp/evil",
        "PGPASSFILE": "/tmp/evil",
        "PGOPTIONS": "-c search_path=evil",
    }
    proc, _, child_env = ws("psql", "ws", "-c", "SELECT 1", extra_env=hostile)
    assert proc.returncode == 0, proc.stderr
    names = {line.split("=", 1)[0] for line in child_env.splitlines()}
    assert names.isdisjoint(hostile)


# Every string the validator used against the old raw-psql admission, shaped as
# `ws-pg psql` arguments, plus the generic ways out of a -c-only grammar.
REFUSED = [
    ["ws", "-Xh", "evil"],
    ["ws", "-Xhevil"],
    ["ws", "--ho=evil"],
    ["ws", "-h", "evil", "-c", "SELECT 1"],
    ["ws", "-c", "SELECT 1", "-h", "evil"],
    ["ws", "-c", "SELECT 1", "-d", "service=evil"],
    ["ws", "-c", "\\connect postgresql://evil/x"],
    ["ws", "-c", "SELECT 1", "-c", "\\! id"],
    ["ws", "-c", "SELECT 1 \\g /tmp/x"],
    ["ws", "-c"],
    ["ws"],
    ["ws", "SELECT 1"],
    ["ws", "-f", "/tmp/x.sql"],
    ["ws", "--command=SELECT 1"],
    ["postgresql://u@127.0.0.1/x?hostaddr=1.2.3.4", "-c", "SELECT 1"],
    ["service=evil", "-c", "SELECT 1"],
    ["host=evil dbname=x", "-c", "SELECT 1"],
    ["-d", "ws", "-c", "SELECT 1"],
    ["Ws", "-c", "SELECT 1"],
    ["1ws", "-c", "SELECT 1"],
    ["w-s", "-c", "SELECT 1"],
    ["a" * 64, "-c", "SELECT 1"],
    ["", "-c", "SELECT 1"],
]


@pytest.mark.parametrize("args", REFUSED)
def test_bypass_arguments_are_refused_before_psql_runs(ws, args):
    proc, argv, _ = ws("psql", *args)
    assert proc.returncode != 0
    assert proc.stderr.startswith("ws-pg: ")
    assert argv is None  # psql was never exec'd


def test_63_char_name_is_accepted(ws):
    proc, argv, _ = ws("psql", "a" * 63, "-c", "SELECT 1")
    assert proc.returncode == 0, proc.stderr
    assert argv[argv.index("-d") + 1] == "a" * 63


def test_refuses_when_server_is_not_running(ws, tmp_path):
    os.remove(tmp_path / "data" / "PG_VERSION")
    proc, argv, _ = ws("psql", "ws", "-c", "SELECT 1")
    assert proc.returncode == 1 and "not running" in proc.stderr
    assert argv is None

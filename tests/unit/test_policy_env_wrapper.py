"""#183: ``env`` must not launder a non-allowlisted program past the allowlist.

``validate_command`` used to check only ``basename(argv[0])``; ``env`` is on the
base allowlist, so ``env <anything>`` was admitted. The wrapped command is now
validated recursively under the same toolchain.
"""

from __future__ import annotations

import pytest

from src.runtime.policy import ANY_TOOLCHAIN, validate_command
from src.services.workspace_services import service as svc

PG = "postgresql://u:p@127.0.0.1:5432/db"

ALLOW = [
    # the coding-testing py2rust A5a migration shapes
    (["env", f"DATABASE_URL={PG}", "uv", "run", "alembic", "upgrade", "head"], "python"),
    (["env", f"DATABASE_URL={PG}", "sqlx", "migrate", "run"], "python+rust"),
    (["env", "FOO=1", "cargo", "test"], "python+rust"),
    (["env", "FOO=1", "BAR=2", "cargo", "test"], "python"),
    (["/usr/bin/env", "FOO=1", "python3", "-m", "pytest"], "python"),
    (["env", "-i", "FOO=1", "ls"], "python"),
    (["env", "-u", "X", "ls"], "python"),
    (["env", "-uX", "--unset=Y", "--unset", "Z", "ls"], "python"),
    (["env", "--", "FOO=1", "ls"], "python"),
    (["env", "ws-pg", "psql", "app", "-c", "select 1"], "python+rust"),
    (["env", "FOO=1", "env", "BAR=2", "ls"], "python"),
]

REFUSE = [
    (["env", "bash", "-c", "id"], "python+rust"),
    (["env", "sh"], "python"),
    (["env", "psql", "-h", "evil", "db"], "python+rust"),
    (["env", f"DATABASE_URL={PG}", "psql", "db"], "python+rust"),
    (["env", "env", "bash"], "python"),
    (["env", "-i", "sh"], "python"),
    (["env"], "python"),
    (["env", "FOO=1"], "python"),
    (["env", "-i"], "python"),
    (["env", "--"], "python"),
    (["env", "-u"], "python"),
    (["env", "-u", "X", "curl", "https://example.com"], "python"),
    # rust-only tools stay rust-only through env
    (["env", "FOO=1", "rustc", "x.rs"], "python"),
    (["env", "ws-pg", "psql", "db"], "python"),
    # options that build a new argv or are not modelled: fail closed
    (["env", "-S", "bash -c id"], "python"),
    (["env", "--split-string=bash -c id"], "python"),
    (["env", "-Sbash"], "python"),
    (["env", "-C", "/", "ls"], "python"),
    (["env", "--chdir=/", "ls"], "python"),
    (["env", "-P", "/tmp", "ls"], "python"),
    (["env", "-iu", "X", "ls"], "python"),
    (["env", "--ignore-signal", "ls"], "python"),
    # assignments that change which code the wrapped program runs
    (["env", "PATH=/workspace/bin", "ls"], "python"),
    (["env", "LD_PRELOAD=/x.so", "ls"], "python"),
    (["env", "BASH_ENV=/x", "ls"], "python"),
]


@pytest.mark.parametrize(("argv", "toolchain"), ALLOW)
def test_env_wrapping_an_allowed_command_is_allowed(argv, toolchain):
    assert validate_command(argv, toolchain=toolchain) is True
    assert svc.validate_argv(argv, toolchain=toolchain) == argv


@pytest.mark.parametrize(("argv", "toolchain"), REFUSE)
def test_env_wrapping_a_refused_command_is_refused(argv, toolchain):
    assert validate_command(argv, toolchain=toolchain) is False
    with pytest.raises(svc.ServiceSpecError):
        svc.validate_argv(argv, toolchain=toolchain)


@pytest.mark.parametrize(("argv", "toolchain"), REFUSE)
def test_boundary_sentinel_still_refuses_non_toolchain_wraps(argv, toolchain):
    """The API boundary uses ANY_TOOLCHAIN; rust-only wraps pass there by design."""
    rust_only = any(t in ("rustc", "ws-pg") for t in argv)
    assert validate_command(argv, toolchain=ANY_TOOLCHAIN) is rust_only


def _old_validate_command(command: list[str]) -> bool:
    """The pre-#183 rule: basename of argv[0] only."""
    return command[0].rsplit("/", 1)[-1].lower() in {"env", "ls", "cargo"}


def test_negative_control_old_rule_admitted_env_bash():
    """The defect is real: the basename-only rule admitted `env bash`."""
    assert _old_validate_command(["env", "bash", "-c", "id"]) is True
    assert _old_validate_command(["bash", "-c", "id"]) is False
    assert validate_command(["env", "bash", "-c", "id"]) is False

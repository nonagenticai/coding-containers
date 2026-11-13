"""A path-qualified argv[0] is matched by its basename, exactly as on origin/main.

The pod is the security boundary: tenant code already runs arbitrary code in it, so a
directory allowlist on argv[0] refused legitimate callers (repo-local tool bins, venvs
of source checkouts, rustup's cargo) without protecting anything.
"""

from __future__ import annotations

import pytest

from src.runtime import ExecutionProfile, enforce_command_policy
from src.runtime.policy import validate_command
from src.services.workspace_services.service import validate_argv

#: The forms the directory allowlist refused and origin/main accepts.
PATH_QUALIFIED = [
    ["node_modules/.bin/eslint", "--format=json"],
    ["/workspace/node_modules/.bin/tsc", "--noEmit"],
    ["./.venv/bin/ruff", "check"],
    ["/workspace/.repos/a__b/.venv/bin/python", "-m", "pytest"],
    ["/opt/venv/bin/python", "-V"],
    ["/root/.cargo/bin/cargo", "test"],
]


@pytest.mark.parametrize("argv", PATH_QUALIFIED, ids=lambda a: a[0])
def test_path_qualified_argv0_passes_every_policy_layer(argv):
    assert validate_command(argv, toolchain="python+rust") is True
    enforce_command_policy(argv, ExecutionProfile.CALLER_SUPPLIED, "python+rust")
    assert validate_argv(argv, toolchain="python+rust") == argv


def test_negative_control_the_basename_still_has_to_be_allowlisted():
    assert validate_command(["/workspace/node_modules/.bin/bash", "-c", "id"]) is False

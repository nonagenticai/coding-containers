"""``POST/GET /api/workspaces/{c}/repos``: read-only secondary repositories.

Three layers:

* validation (repo / ref) — pure functions;
* the route with a recording fake: the credential is clone-scope only, under the
  caller's tenant, through the SAME gates as provisioning (a platform repository the
  tenant may not clone is a 403 and nothing runs; the org PAT and the workspace BYO
  token are never used for a platform repository; a public customer repository is
  cloned tokenless; a private one without a credential is a 412); the token travels
  only in the clone exec's env; a repeat at the same ref is reused without asking;
* the REAL generated shell command, real git, with only ``git fetch`` shimmed to a
  local source repository (no network): the checkout lands at the documented path at
  the fetched commit, has NO remote, no token anywhere in ``.git``, and the primary
  repository ignores ``/.repos/``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.routers.containers.git_push as gp
import src.routers.containers.secondary_repos as sr
from coding_client.auth.tenant import require_tenant
from src.clients.github_client import CloneCredential, GitHubTokenManager
from src.models.container_models import ContainerStatus

TENANT = "tenant-conveyor"
WS = "cv-ems-p4"
URL = f"/api/workspaces/{WS}/repos"
_TOKEN = "ghs_secondary_clone_token_only_in_env"  # noqa: S105  # pragma: allowlist secret
_PAT = "ghp_SYSTEM_PAT_must_never_be_used"  # noqa: S105  # pragma: allowlist secret
_HEAD = "d" * 40


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://github.com/octocat/Hello-World", ("octocat", "Hello-World")),
        ("https://github.com/nonagenticai/ems.git", ("nonagenticai", "ems")),
        ("octocat/Hello-World", ("octocat", "Hello-World")),
    ],
)
def test_parse_repo_accepts_canonical_forms(raw, expected):
    assert sr.parse_repo(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "octocat",
        "https://evil.example/o/r",
        "https://github.com/o/r/x",
        "https://github.com/nonagenticai/../acme/r",
        "https://x:y@github.com/o/r",
        "file:///etc",
        "o/r;id",
    ],
)
def test_parse_repo_refuses_everything_else(raw):
    with pytest.raises(ValueError):
        sr.parse_repo(raw)


@pytest.mark.parametrize("ref", [None, "", "main", "v1.2.3", "feature/x", "a" * 40])
def test_valid_refs(ref):
    assert sr.validate_ref(ref) == ((ref or "").strip() or "HEAD")


@pytest.mark.parametrize(
    "ref", ["-x", "--upload-pack=x", "a..b", "a b", "$(id)", "a:b", "main~1", "x" * 256]
)
def test_invalid_refs(ref):
    with pytest.raises(ValueError):
        sr.validate_ref(ref)


def test_secondary_path_is_inside_the_workspace_and_unambiguous():
    assert sr.secondary_path("octocat", "Hello-World") == "/workspace/.repos/octocat__Hello-World"


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def _token_manager(credential: CloneCredential) -> MagicMock:
    mgr = MagicMock(spec=GitHubTokenManager)
    mgr.get_app_clone_credential = AsyncMock(return_value=credential)
    return mgr


async def _call(
    app,
    client,
    body: dict,
    *,
    credential: CloneCredential | None = None,
    byo_token: str | None = None,
    existing: str = "",
    clone_result: dict | None = None,
    caller_tenant: str = TENANT,
):
    calls: list[tuple[str, dict | None]] = []

    async def _run(container_name, command, tenant_id=None, timeout=300, env=None):
        calls.append((command, env))
        assert tenant_id == caller_tenant
        if " fetch " in command:
            return clone_result or {"exit_code": 0, "stdout": f"head {_HEAD}\n", "stderr": ""}
        return {"exit_code": 0, "stdout": existing, "stderr": ""}

    container = SimpleNamespace(
        container_name=WS,
        status=ContainerStatus("running"),
        tenant_id=TENANT,
        repo_url="https://github.com/nonagenticai/ems",
        github_access_token=byo_token,
        secret_config_id=None,
    )

    def _get(name, tenant_id=None):
        if (name, tenant_id) == (WS, TENANT):
            return container
        raise ValueError("not found")

    manager = MagicMock()
    manager.get_container.side_effect = _get
    manager.execute_command = AsyncMock(side_effect=_run)
    tm = _token_manager(
        credential or CloneCredential(ok=True, token=_TOKEN, visibility="private", source="app")
    )
    app.dependency_overrides[require_tenant] = lambda: caller_tenant
    try:
        with (
            patch("src.routers.containers.get_container_manager", return_value=manager),
            patch.object(sr, "get_github_token_manager", return_value=tm),
            patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": _PAT, "GITHUB_TOKEN": _PAT}),
        ):
            resp = await client.post(URL, json=body)
    finally:
        app.dependency_overrides.pop(require_tenant, None)
    return resp, calls, tm


def _clones(calls):
    return [(c, e) for c, e in calls if " fetch " in c]


def _no_secret_in_commands(calls, *secrets):
    for command, _ in calls:
        for secret in secrets:
            assert secret not in command


async def test_platform_repo_the_tenant_may_clone_is_added_with_a_clone_token(app, client):
    resp, calls, tm = await _call(app, client, {"repo_url": "nonagenticai/ems", "ref": "main"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {
        "container_name": WS,
        "reused": False,
        "repo": "nonagenticai/ems",
        "repo_url": "https://github.com/nonagenticai/ems",
        "ref": "main",
        "path": "/workspace/.repos/nonagenticai__ems",
        "head_sha": _HEAD,
        "read_only": True,
    }
    call = tm.get_app_clone_credential.await_args
    assert call.args == ("https://github.com/nonagenticai/ems",)
    assert call.kwargs == {"tenant_id": TENANT, "scope": "clone"}
    ((command, env),) = _clones(calls)
    assert env == {gp._TOKEN_ENV: _TOKEN}
    assert "https://github.com/nonagenticai/ems main" in command
    _no_secret_in_commands(calls, _TOKEN, _PAT)
    assert "@github.com" not in command and "remote add" not in command


async def test_platform_repo_forbidden_for_the_tenant_is_403_and_nothing_runs(app, client):
    """Today's fact for the conveyor tenant and nonagenticai/fr: refused, no PAT fallback."""
    resp, calls, tm = await _call(
        app,
        client,
        {"repo_url": "https://github.com/nonagenticai/fr"},
        credential=CloneCredential(
            ok=True, token=None, visibility="private", reason="platform_repo_forbidden"
        ),
    )
    assert resp.status_code == 403, resp.text
    assert "platform_repo_forbidden" in resp.text
    assert tm.get_app_clone_credential.await_args.kwargs == {"tenant_id": TENANT, "scope": "clone"}
    assert _clones(calls) == []
    for _c, env in calls:
        assert not env


async def test_platform_repo_never_uses_the_workspace_byo_token(app, client):
    byo = "ghp_tenant_byo_pat_0000000000000000000"  # noqa: S105  # pragma: allowlist secret
    resp, calls, tm = await _call(
        app,
        client,
        {"repo_url": "nonagenticai/fr"},
        byo_token=byo,
        credential=CloneCredential(
            ok=True, token=None, visibility="private", reason="platform_repo_forbidden"
        ),
    )
    assert resp.status_code == 403, resp.text
    assert tm.get_app_clone_credential.await_count == 1
    assert _clones(calls) == []


async def test_platform_null_token_is_412_never_the_org_pat(app, client):
    resp, calls, _ = await _call(
        app,
        client,
        {"repo_url": "nonagenticai/ems"},
        credential=CloneCredential(
            ok=True, token=None, visibility="private", reason="not_installed"
        ),
    )
    assert resp.status_code == 412, resp.text
    assert _clones(calls) == []


async def test_public_customer_repo_is_cloned_without_a_credential(app, client):
    resp, calls, _ = await _call(
        app,
        client,
        {"repo_url": "https://github.com/octocat/Hello-World", "ref": "master"},
        credential=CloneCredential(
            ok=True, token=None, visibility="public", reason="not_installed"
        ),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["path"] == "/workspace/.repos/octocat__Hello-World"
    ((_, env),) = _clones(calls)
    assert env is None


@pytest.mark.parametrize("credential_ok", [True, False])
async def test_private_customer_repo_without_a_credential_is_412(app, client, credential_ok):
    resp, calls, _ = await _call(
        app,
        client,
        {"repo_url": "acme/private"},
        credential=CloneCredential(
            ok=credential_ok,
            token=None,
            visibility="private" if credential_ok else None,
            reason="not_installed" if credential_ok else None,
            error=None if credential_ok else "unreachable",
        ),
    )
    assert resp.status_code == 412, resp.text
    assert _clones(calls) == []
    for _c, env in calls:
        assert not env or _PAT not in env.values()


async def test_customer_repo_uses_the_tenant_byo_token_first(app, client):
    byo = "ghp_tenant_byo_pat_1111111111111111111"  # noqa: S105  # pragma: allowlist secret
    resp, calls, tm = await _call(app, client, {"repo_url": "acme/widget"}, byo_token=byo)
    assert resp.status_code == 200, resp.text
    assert tm.get_app_clone_credential.await_count == 0
    ((_, env),) = _clones(calls)
    assert env == {gp._TOKEN_ENV: byo}
    _no_secret_in_commands(calls, byo)


def _listing(repo: str, ref: str, head: str = _HEAD) -> str:
    owner, name = repo.split("/")
    return f"repo\t{owner}__{name}\thttps://github.com/{repo}\t{ref}\t{head}\n"


async def test_same_repo_at_same_sha_is_reused_after_the_gate(app, client):
    resp, calls, tm = await _call(
        app,
        client,
        {"repo_url": "nonagenticai/ems", "ref": _HEAD},
        existing=_listing("nonagenticai/ems", _HEAD),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["reused"] is True and resp.json()["head_sha"] == _HEAD
    # The access decision still ran (clone scope, caller's tenant) -- reuse is not a bypass.
    assert tm.get_app_clone_credential.await_args.kwargs == {"tenant_id": TENANT, "scope": "clone"}
    assert _clones(calls) == []


async def test_a_planted_checkout_of_a_forbidden_repo_is_403(app, client):
    """Workspace code writes .repos/nonagenticai__fr with a config claiming fr: no 200."""
    resp, calls, tm = await _call(
        app,
        client,
        {"repo_url": "nonagenticai/fr", "ref": _HEAD},
        existing=_listing("nonagenticai/fr", _HEAD),
        credential=CloneCredential(
            ok=True, token=None, visibility="private", reason="platform_repo_forbidden"
        ),
    )
    assert resp.status_code == 403, resp.text
    assert "platform_repo_forbidden" in resp.text
    assert tm.get_app_clone_credential.await_count == 1
    assert calls == []  # not even the inspect ran


async def test_revoked_access_does_not_get_the_old_checkout_back(app, client):
    resp, calls, _ = await _call(
        app,
        client,
        {"repo_url": "acme/private", "ref": _HEAD},
        existing=_listing("acme/private", _HEAD),
        credential=CloneCredential(
            ok=True, token=None, visibility="private", reason="not_installed"
        ),
    )
    assert resp.status_code == 412, resp.text
    assert calls == []


@pytest.mark.parametrize(
    ("body", "existing"),
    [
        # a branch ref may have moved: always re-fetched
        ({"repo_url": "nonagenticai/ems", "ref": "main"}, _listing("nonagenticai/ems", "main")),
        # recorded SHA ref, but HEAD is something else: treated as refresh
        (
            {"repo_url": "nonagenticai/ems", "ref": _HEAD},
            _listing("nonagenticai/ems", _HEAD, head="e" * 40),
        ),
        ({"repo_url": "nonagenticai/ems", "ref": "release"}, _listing("nonagenticai/ems", "main")),
        (
            {"repo_url": "nonagenticai/ems", "ref": _HEAD, "refresh": True},
            _listing("nonagenticai/ems", _HEAD),
        ),
    ],
)
async def test_unverifiable_or_refreshed_reuse_re_clones(app, client, body, existing):
    resp, calls, tm = await _call(app, client, body, existing=existing)
    assert resp.status_code == 200, resp.text
    assert resp.json()["reused"] is False
    assert len(_clones(calls)) == 1
    assert tm.get_app_clone_credential.await_count == 1


@pytest.mark.parametrize(
    "body",
    [
        {"repo_url": "https://evil.example/o/r"},
        {"repo_url": "nonagenticai/ems", "ref": "--upload-pack=touch /tmp/x"},
        {"repo_url": "nonagenticai/ems", "ref": "a..b"},
    ],
)
async def test_bad_requests_are_refused_before_any_credential_or_exec(app, client, body):
    resp, calls, tm = await _call(app, client, body)
    assert resp.status_code == 400, resp.text
    assert calls == [] and tm.get_app_clone_credential.await_count == 0


async def test_another_tenants_workspace_is_404(app, client):
    resp, calls, tm = await _call(
        app, client, {"repo_url": "octocat/Hello-World"}, caller_tenant="t-b"
    )
    assert resp.status_code == 404, resp.text
    assert calls == [] and tm.get_app_clone_credential.await_count == 0


async def test_a_failed_clone_never_echoes_the_token(app, client):
    resp, _, _ = await _call(
        app,
        client,
        {"repo_url": "nonagenticai/ems"},
        clone_result={"exit_code": 128, "stdout": "", "stderr": f"fatal: auth {_TOKEN} failed"},
    )
    assert resp.status_code == 500
    assert _TOKEN not in resp.text


async def test_the_list_reports_only_well_formed_checkouts(app, client):
    from src.routers.containers import secondary_repos

    listing = (
        f"repo\tnonagenticai__ems\thttps://github.com/nonagenticai/ems\tmain\t{_HEAD}\n"
        # directory name does not match the recorded URL: tampered, not reported
        f"repo\tacme__x\thttps://github.com/nonagenticai/fr\tmain\t{_HEAD}\n"
        "repo\tbroken\t\t\t\n"
    )
    repos = secondary_repos._parse_repos(listing)
    assert [r.repo for r in repos] == ["nonagenticai/ems"]


# ---------------------------------------------------------------------------
# The REAL command: real git, only `git fetch` shimmed to a local source repo
# ---------------------------------------------------------------------------

_GIT = shutil.which("git")


@pytest.fixture
def real(tmp_path, monkeypatch):
    if _GIT is None:
        pytest.skip("needs a real git binary")
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key, raising=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    home = tmp_path / "home"
    home.mkdir()
    env.update(HOME=str(home), GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0")
    ident = ["-c", "user.name=t", "-c", "user.email=t@t"]

    def git(repo: Path, *args: str) -> str:
        return subprocess.run(
            [_GIT, "-C", str(repo), *ident, *args],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        ).stdout.strip()

    src = tmp_path / "provider-src"
    subprocess.run([_GIT, "init", "-q", "-b", "main", str(src)], check=True, env=env)
    (src / "app.py").write_text("print('provider')\n")
    git(src, "add", "app.py")
    git(src, "commit", "-qm", "provider")
    head = git(src, "rev-parse", "HEAD")

    ws = tmp_path / "workspace"
    subprocess.run([_GIT, "init", "-q", "-b", "main", str(ws)], check=True, env=env)
    (ws / "consumer.py").write_text("print('consumer')\n")
    git(ws, "add", "consumer.py")
    git(ws, "commit", "-qm", "consumer")

    log = tmp_path / "log"
    log.mkdir()
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "git"
    shim.write_text(
        "#!/bin/sh\n"
        f'REAL="{_GIT}"\n'
        'for a in "$@"; do\n'
        '  case "$a" in\n'
        "    fetch)\n"
        f'      printf "%s\\n" "$@" > "{log}/fetch.argv"\n'
        f'      printf "%s" "${{GIT_PUSH_TOKEN-}}" > "{log}/fetch.token"\n'
        "      for last; do :; done\n"
        f'      exec env GIT_ALLOW_PROTOCOL=file "$REAL" fetch -q --no-tags "file://{src}" "$last" ;;\n'
        "    checkout)\n"
        f'      printf "%s" "${{GIT_PUSH_TOKEN-unset}}" > "{log}/checkout.token"; break ;;\n'
        "  esac\n"
        "done\n"
        'exec "$REAL" "$@"\n'
    )
    shim.chmod(0o755)
    monkeypatch.setattr(sr, "WORKSPACE_ROOT", str(ws))

    def run(command: str, token: str | None = _TOKEN) -> subprocess.CompletedProcess:
        run_env = {**env, "PATH": f"{shim_dir}:{env['PATH']}"}
        if token:
            run_env[gp._TOKEN_ENV] = token
        return subprocess.run(
            ["sh", "-c", command], capture_output=True, text=True, env=run_env, check=False
        )

    return SimpleNamespace(ws=ws, src=src, head=head, log=log, run=run, git=git)


def test_real_clone_lands_read_only_at_the_documented_path(real):
    r = real.run(sr._clone_command("acme-corp", "provider", "main"))
    assert r.returncode == 0, r.stderr
    assert sr._parse_head(r.stdout) == real.head
    dest = real.ws / ".repos" / "acme-corp__provider"
    assert (dest / "app.py").read_text() == "print('provider')\n"
    assert real.git(dest, "rev-parse", "HEAD") == real.head
    # Fetched by explicit URL with the token; checked out without it.
    argv = (real.log / "fetch.argv").read_text().splitlines()
    assert argv[-2:] == ["https://github.com/acme-corp/provider", "main"]
    assert "--depth=1" in argv
    assert (real.log / "fetch.token").read_text() == _TOKEN
    assert (real.log / "checkout.token").read_text() == "unset"
    # No remote to push to, and no token anywhere in the checkout's git dir.
    assert real.git(dest, "remote") == ""
    for path in (dest / ".git").rglob("*"):
        if path.is_file():
            assert _TOKEN.encode() not in path.read_bytes(), path
    # No scratch directory left behind.
    assert [p.name for p in (real.ws / ".repos").iterdir()] == ["acme-corp__provider"]
    # The primary repository does not see the secondary checkout.
    assert "/.repos/" in (real.ws / ".git" / "info" / "exclude").read_text().splitlines()
    assert real.git(real.ws, "status", "--porcelain") == ""


def test_real_inspect_and_list_read_back_the_checkout(real):
    assert real.run(sr._clone_command("acme-corp", "provider", "main")).returncode == 0
    inspected = sr._parse_repos(real.run(sr._inspect_command("acme-corp", "provider")).stdout)
    listed = sr._parse_repos(real.run(sr._list_command()).stdout)
    assert len(inspected) == 1 and inspected == listed
    repo = inspected[0]
    assert repo.repo == "acme-corp/provider" and repo.ref == "main"
    assert repo.head_sha == real.head


def test_real_reclone_replaces_and_exclude_is_not_duplicated(real):
    for _ in range(2):
        assert real.run(sr._clone_command("acme-corp", "provider", "main")).returncode == 0
    lines = (real.ws / ".git" / "info" / "exclude").read_text().splitlines()
    assert lines.count("/.repos/") == 1


def test_real_public_clone_runs_without_a_token(real):
    r = real.run(sr._clone_command("octocat", "Hello-World", "main"), token=None)
    assert r.returncode == 0, r.stderr
    assert (real.log / "fetch.token").read_text() == ""


def test_real_failed_fetch_leaves_nothing_behind(real):
    r = real.run(sr._clone_command("acme-corp", "provider", "no-such-branch"))
    assert r.returncode != 0
    assert not (real.ws / ".repos" / "acme-corp__provider").exists()
    assert list((real.ws / ".repos").iterdir()) == []


def test_real_command_is_valid_shell():
    for command in (
        sr._clone_command("acme-corp", "provider", "main"),
        sr._inspect_command("acme-corp", "provider"),
        sr._list_command(),
    ):
        assert subprocess.run(["sh", "-n", "-c", command], check=False).returncode == 0

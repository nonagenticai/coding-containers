"""The clone leaves no credential at rest in the workspace's ``.git`` (auditor P0, #138).

The init-container script used to clone ``https://x-access-token:$GIT_TOKEN@github.com/o/r``.
git records the clone URL verbatim as ``remote.origin.url``, so the token sat in
``/workspace/.git/config`` on the emptyDir the workspace (uid 1000) shares.

These tests run the REAL rendered init-container script with the REAL ``git``
against a local bare repository. ``https://github.com/owner/repo`` is redirected
to that bare repo by an ``insteadOf`` in a scratch GLOBAL config — so the URL
the script names, and therefore the URL git records, is exactly production's.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from src.core import git_clone_credentials as gcc
from src.core.kubernetes_container_provisioner import _build_workspace_clone_cmd

_GIT = shutil.which("git")
_TOKEN = "ghs_clone_token_must_never_be_at_rest"  # noqa: S105  # pragma: allowlist secret
_CANON = "https://github.com/owner/repo"

pytestmark = pytest.mark.skipif(_GIT is None, reason="needs a real git binary")


@pytest.fixture(autouse=True)
def _isolate_from_the_enclosing_git(monkeypatch):
    """Strip every inherited ``GIT_*`` variable before any real git runs.

    Under a pre-commit hook git exports ``GIT_DIR``/``GIT_INDEX_FILE`` for the
    repository being committed, and ``git -C <tmp>`` honours them over ``-C``.
    Measured 2026-09-22: without this, a scratch-repo write landed in the
    developer's real ``.git``. Copied from test_git_push_exfiltration.py.
    """
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key, raising=False)


def _run(args: list[str], env: dict[str, str], cwd: Path | None = None) -> str:
    return subprocess.run(
        args, check=True, capture_output=True, text=True, env=env, cwd=cwd, timeout=30
    ).stdout


@pytest.fixture
def origin(tmp_path):
    """A bare repo with two commits on main and a PR head ref, plus the env that
    makes ``https://github.com/owner/repo`` resolve to it."""
    home = tmp_path / "home"
    home.mkdir()
    glob = tmp_path / "global.gitconfig"
    bare = tmp_path / "bare.git"
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        HOME=str(home),
        GIT_CONFIG_NOSYSTEM="1",  # Apple git's osxkeychain helper lives in system scope
        GIT_CONFIG_GLOBAL=str(glob),
        GIT_TERMINAL_PROMPT="0",
    )
    seed = tmp_path / "seed"
    _run([_GIT, "init", "-q", "-b", "main", str(seed)], env)
    ident = ["-c", "user.name=t", "-c", "user.email=t@t"]
    for i in range(2):
        (seed / "f.txt").write_text(f"{i}\n")
        _run([_GIT, "-C", str(seed), "add", "f.txt"], env)
        _run([_GIT, "-C", str(seed), *ident, "commit", "-qm", f"c{i}"], env)
    first = _run([_GIT, "-C", str(seed), "rev-parse", "HEAD~1"], env).strip()
    _run([_GIT, "clone", "-q", "--bare", str(seed), str(bare)], env)
    _run([_GIT, "-C", str(bare), "update-ref", "refs/pull/7/head", first], env)
    # The second insteadOf also resolves the OLD token-in-URL form, so a
    # regression to it still clones — and is then caught by what it leaves behind.
    tokened = f"https://x-access-token:{_TOKEN}@github.com/owner/repo"
    glob.write_text(
        f'[url "file://{bare}"]\n\tinsteadOf = {_CANON}\n\tinsteadOf = {tokened}\n'
        '[protocol "file"]\n\tallow = always\n'
    )
    return env, first


def _clone(tmp_path: Path, env: dict[str, str], **kw) -> subprocess.CompletedProcess:
    dest = tmp_path / "ws"
    dest.mkdir(exist_ok=True)
    cmd = _build_workspace_clone_cmd("github.com", "/owner/repo", dest=str(dest), **kw)
    return subprocess.run(
        ["sh", "-c", cmd],
        capture_output=True,
        text=True,
        env={**env, "GIT_TOKEN": _TOKEN},
        timeout=60,
    )


def _files_with_token(dest: Path) -> list[str]:
    return [
        str(p.relative_to(dest))
        for p in dest.rglob("*")
        if p.is_file() and _TOKEN.encode() in p.read_bytes()
    ]


def _origin_url(dest: Path, env: dict[str, str]) -> str:
    return subprocess.run(
        [_GIT, "-C", str(dest), "config", "--get", "remote.origin.url"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    ).stdout.strip()


class TestRealCloneLeavesNoToken:
    @pytest.mark.parametrize(
        "kw",
        [{"branch": "main"}, {"branch": ""}, {"branch": None, "pr_number": 7}],
        ids=["branch", "default-branch", "pull-request"],
    )
    def test_clone_records_the_canonical_url_and_no_token_anywhere(self, tmp_path, origin, kw):
        env, _first = origin
        out = _clone(tmp_path, env, **kw)
        assert out.returncode == 0, out.stderr
        dest = tmp_path / "ws"
        assert (dest / "f.txt").exists()
        assert _origin_url(dest, env) == _CANON
        assert _TOKEN not in (dest / ".git" / "config").read_text()
        # config, FETCH_HEAD, packed-refs, logs/, and the checkout itself
        assert _files_with_token(dest) == []
        assert _TOKEN not in out.stdout + out.stderr

    def test_sha_checkout_and_history_cutoff_leave_no_token(self, tmp_path, origin):
        env, first = origin
        out = _clone(tmp_path, env, branch=first, history_cutoff=True)
        assert out.returncode == 0, out.stderr
        dest = tmp_path / "ws"
        assert _origin_url(dest, env) == ""  # cutoff removes origin; the scrub tolerates it
        assert _files_with_token(dest) == []


class TestScrubRepairsAndRefuses:
    """The post-clone fragment also runs over an EXISTING checkout (the idempotency
    guard skips the clone), so a volume cloned by an older build is repaired, and
    anything it cannot repair fails the init container."""

    def _legacy_checkout(self, tmp_path, env) -> Path:
        dest = tmp_path / "ws"
        _run([_GIT, "clone", "-q", _CANON, str(dest)], env)
        _run(
            [
                _GIT,
                "-C",
                str(dest),
                "remote",
                "set-url",
                "origin",
                f"https://x-access-token:{_TOKEN}@github.com/owner/repo",
            ],
            env,
        )
        (dest / "wip.txt").write_text("uncommitted work\n")
        return dest

    def test_a_tokened_origin_left_by_an_older_build_is_rewritten(self, tmp_path, origin):
        env, _ = origin
        dest = self._legacy_checkout(tmp_path, env)
        assert _TOKEN in (dest / ".git" / "config").read_text()  # precondition
        out = _clone(tmp_path, env, branch="main")
        assert out.returncode == 0, out.stderr
        assert _origin_url(dest, env) == _CANON
        assert _files_with_token(dest) == []
        assert (dest / "wip.txt").exists()

    def test_a_token_it_cannot_scrub_fails_the_init_container_without_deleting_work(
        self, tmp_path, origin
    ):
        env, _ = origin
        dest = self._legacy_checkout(tmp_path, env)
        _run([_GIT, "-C", str(dest), "config", "remote.upstream.url", f"https://{_TOKEN}@x/y"], env)
        out = _clone(tmp_path, env, branch="main")
        assert out.returncode == 1
        assert "a credential remains under .git" in out.stderr
        assert _TOKEN not in out.stderr
        # k8s: never wipe a valid checkout (it may hold uncommitted work)
        assert (dest / "wip.txt").exists()

    def test_any_userinfo_in_a_remote_url_fails_even_without_this_token(self, tmp_path, origin):
        """The URL check is independent of the content scan: a credential that is
        not ``$GIT_TOKEN`` (another secret, a stale token) is refused too."""
        env, _ = origin
        dest = self._legacy_checkout(tmp_path, env)
        other = "https://u:other@x/y"  # pragma: allowlist secret
        _run([_GIT, "-C", str(dest), "config", "remote.upstream.url", other], env)
        out = _clone(tmp_path, env, branch="main")
        assert out.returncode == 1
        assert "a credential remains under .git" in out.stderr

    def test_a_token_in_a_non_config_file_fails_the_init_container(self, tmp_path, origin):
        env, _ = origin
        out = _clone(tmp_path, env, branch="main")
        assert out.returncode == 0, out.stderr
        dest = tmp_path / "ws"
        (dest / ".git" / "FETCH_HEAD").write_text(f"abc\t\tbranch 'main' of https://{_TOKEN}@x\n")
        again = _clone(tmp_path, env, branch="main")
        assert again.returncode == 1
        assert "a credential remains under .git" in again.stderr

    def test_the_docker_variant_wipes_on_failure(self, tmp_path, origin):
        env, _ = origin
        dest = self._legacy_checkout(tmp_path, env)
        _run([_GIT, "-C", str(dest), "config", "remote.upstream.url", f"https://{_TOKEN}@x/y"], env)
        frag = gcc.scrub_and_verify(
            shlex.quote(str(dest)), _CANON, token_env="GIT_TOKEN", wipe_on_failure=True
        )
        out = subprocess.run(
            ["sh", "-c", frag], capture_output=True, text=True, env={**env, "GIT_TOKEN": _TOKEN}
        )
        assert out.returncode == 1
        assert list(dest.iterdir()) == []


class TestTheHelperIsTheOnlyPathForTheToken:
    """With the clone's ``-c`` block, git answers the token for the canonical repo only."""

    def _fill(self, env, host: str, path: str) -> subprocess.CompletedProcess:
        args = gcc.clone_credential_shell_args("github.com", "owner/repo")
        return subprocess.run(
            ["sh", "-c", f"git {args} credential fill"],
            input=f"protocol=https\nhost={host}\npath={path}\n",
            capture_output=True,
            text=True,
            env={**env, "GIT_TOKEN": _TOKEN, "GIT_ASKPASS": ""},
            timeout=20,
        )

    def test_canonical_repo_gets_the_token(self, origin):
        env, _ = origin
        out = self._fill(env, "github.com", "owner/repo")
        assert out.returncode == 0, out.stderr
        assert f"username=x-access-token\npassword={_TOKEN}" in out.stdout

    @pytest.mark.parametrize(
        ("host", "path"), [("evil.example", "owner/repo"), ("github.com", "owner/other")]
    )
    def test_anything_else_gets_nothing(self, origin, host, path):
        env, _ = origin
        out = self._fill(env, host, path)
        assert _TOKEN not in out.stdout + out.stderr
        assert out.returncode != 0

    def test_the_token_is_never_in_the_rendered_script(self):
        cmd = _build_workspace_clone_cmd("github.com", "/owner/repo", "main")
        assert "$GIT_TOKEN@" not in cmd
        assert "x-access-token:" not in cmd

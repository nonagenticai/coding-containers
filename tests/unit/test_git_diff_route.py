"""Tests for GET /api/workspaces/{name}/git/diff.

Most tests run the route's REAL command strings against a REAL git repository in a
temp dir: the fake container manager executes each command with ``sh -c`` after
rewriting the ``/workspace`` root to that dir. A mock that returns canned stdout
would pass for any command the route built, including a wrong one; this cannot.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.routers.containers import git_diff


def _user(*, is_admin=False, tenant_id="tenant-a"):
    user = MagicMock()
    user.sub = "u"
    user.preferred_username = "u"
    user.effective_tenant_id = tenant_id
    user.is_admin.return_value = is_admin
    return user


def _container(status_val="running"):
    from src.models.container_models import ContainerStatus

    info = MagicMock()
    info.status = ContainerStatus(status_val)
    return info


def _run(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture(autouse=True)
def _isolate_from_the_enclosing_git(monkeypatch):
    """Strip every inherited ``GIT_*`` variable before any real git runs.

    Under a pre-push/pre-commit hook git exports ``GIT_DIR`` for the repository
    being pushed, and a subprocess ``git`` honours it over its cwd. Measured
    2026-09-22 on this very file: without this, the scratch repo's ``git config``
    and ``git commit`` landed in the developer's real ``.git`` (shared by every
    worktree) and the "not a repository" test ran against the real repo.
    """
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _run(r, "init", "-q", "-b", "main")
    _run(r, "config", "user.email", "t@example.com")
    _run(r, "config", "user.name", "t")
    (r / "src").mkdir()
    (r / "src" / "a.py").write_text("def a():\n    return 1\n")
    (r / "README.md").write_text("hello\n")
    _run(r, "add", "-A")
    _run(r, "commit", "-q", "-m", "one")
    return r


class _LocalManager:
    """Container manager whose execute_command runs the command on a local repo."""

    def __init__(self, repo: Path, status_val: str = "running"):
        self.repo = repo
        self.status_val = status_val
        self.commands: list[str] = []
        self.tenants: list[str | None] = []

    def get_container(self, name, tenant_id=None):
        self.tenants.append(tenant_id)
        return _container(self.status_val)

    async def execute_command(self, container_name, command, tenant_id=None, timeout=300):
        self.commands.append(command)
        prefix = f"cd {git_diff._WORKSPACE_ROOT} && "
        assert command.startswith(prefix), command
        proc = subprocess.run(
            ["sh", "-c", f"cd '{self.repo}' && " + command[len(prefix) :]],
            capture_output=True,
            text=True,
        )
        return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


async def _get(app, client, manager, url, user=None):
    from coding_client.auth.tenant import get_current_user_with_tenant

    app.dependency_overrides[get_current_user_with_tenant] = lambda: user or _user()
    try:
        with patch("src.routers.containers.get_container_manager", return_value=manager):
            return await client.get(url)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
class TestGitDiffAgainstRealRepo:
    async def test_working_tree_diff_has_content_and_untracked(self, app, client, repo):
        (repo / "src" / "a.py").write_text("def a():\n    return 2\n")
        (repo / "new.txt").write_text("untracked\n")
        mgr = _LocalManager(repo)

        r = await _get(app, client, mgr, "/api/workspaces/ctr/git/diff")

        assert r.status_code == 200, r.text
        d = r.json()
        assert d["compared_to"] == "working_tree"
        assert d["head_sha"] is None
        assert d["base_sha"] == _run(repo, "rev-parse", "HEAD")
        assert d["files"] == [{"status": "M", "path": "src/a.py", "old_path": None}]
        assert "-    return 1" in d["diff"] and "+    return 2" in d["diff"]
        # git diff never shows untracked files; they must be published, not dropped.
        assert d["untracked_files"] == ["new.txt"]
        assert "new.txt" not in d["diff"]
        assert d["truncated"] is False
        assert d["diff_bytes"] == len(d["diff"].encode())
        # the caller's tenant was the one the container was resolved under
        assert mgr.tenants == ["tenant-a"]

    async def test_commit_to_commit_diff_with_rename(self, app, client, repo):
        base = _run(repo, "rev-parse", "HEAD")
        _run(repo, "mv", "README.md", "DOCS.md")
        (repo / "src" / "b.py").write_text("x = 1\n")
        _run(repo, "add", "-A")
        _run(repo, "commit", "-q", "-m", "two")
        # an uncommitted edit must NOT leak into a commit..commit diff
        (repo / "src" / "a.py").write_text("dirty\n")

        r = await _get(
            app, client, _LocalManager(repo), f"/api/workspaces/ctr/git/diff?base={base}&head=main"
        )

        assert r.status_code == 200, r.text
        d = r.json()
        assert d["compared_to"] == "commit"
        assert d["head_sha"] == _run(repo, "rev-parse", "main")
        assert {"status": "R", "path": "DOCS.md", "old_path": "README.md"} in d["files"]
        assert {"status": "A", "path": "src/b.py", "old_path": None} in d["files"]
        assert all(f["path"] != "src/a.py" for f in d["files"])
        assert "dirty" not in d["diff"]
        assert d["untracked_files"] == []

    async def test_pinned_workspace_diffs_from_the_pin_not_the_branch_tip(self, app, client, repo):
        """A workspace imported at a pinned commit reviews what IT changed.

        The pin is RM ``repositories[].sha``: the workspace is cloned at that commit
        while ``base`` stays the branch name. A two-point diff against the branch tip
        would report every commit between the pin and the tip as REMOVED -- work the
        card never did. The base side is the merge base, so only the card's own change
        shows (this is what the pull request will contain too).
        """
        pin = _run(repo, "rev-parse", "HEAD")
        (repo / "README.md").write_text("later on the branch\n")
        _run(repo, "commit", "-q", "-am", "two")
        assert _run(repo, "rev-parse", "main") != pin
        # The workspace: checked out at the pin, with the card's own change on top.
        _run(repo, "checkout", "-q", "--detach", pin)
        (repo / "src" / "a.py").write_text("def a():\n    return 2\n")

        r = await _get(app, client, _LocalManager(repo), "/api/workspaces/ctr/git/diff?base=main")

        assert r.status_code == 200, r.text
        d = r.json()
        assert d["base"] == "main"
        assert d["base_sha"] == pin, "the diff must start at the merge base, not the branch tip"
        assert d["files"] == [{"status": "M", "path": "src/a.py", "old_path": None}]
        assert "later on the branch" not in d["diff"]
        assert "+    return 2" in d["diff"]

    async def test_no_common_ancestor_falls_back_to_plain_two_point_diff(self, app, client, repo):
        """Unrelated histories have no merge base; the base ref must still be honoured."""
        base = _run(repo, "rev-parse", "main")
        _run(repo, "checkout", "-q", "--orphan", "other")
        _run(repo, "rm", "-rqf", ".")
        (repo / "only-on-other.txt").write_text("x\n")
        _run(repo, "add", "-A")
        _run(repo, "commit", "-q", "-m", "orphan")

        r = await _get(
            app, client, _LocalManager(repo), "/api/workspaces/ctr/git/diff?base=main&head=other"
        )

        assert r.status_code == 200, r.text
        d = r.json()
        assert d["base_sha"] == base
        assert {"status": "A", "path": "only-on-other.txt", "old_path": None} in d["files"]

    async def test_path_limits_diff(self, app, client, repo):
        (repo / "src" / "a.py").write_text("changed\n")
        (repo / "README.md").write_text("changed too\n")

        r = await _get(
            app, client, _LocalManager(repo), "/api/workspaces/ctr/git/diff?path=/workspace/src"
        )

        assert r.status_code == 200, r.text
        d = r.json()
        assert d["path"] == "src"
        assert [f["path"] for f in d["files"]] == ["src/a.py"]
        assert "README" not in d["diff"]

    async def test_no_changes_is_empty_not_error(self, app, client, repo):
        r = await _get(app, client, _LocalManager(repo), "/api/workspaces/ctr/git/diff")
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["diff"] == "" and d["files"] == [] and d["untracked_files"] == []

    async def test_unknown_ref_is_404_not_empty_diff(self, app, client, repo):
        r = await _get(
            app, client, _LocalManager(repo), "/api/workspaces/ctr/git/diff?base=no-such-branch"
        )
        assert r.status_code == 404, r.text

    async def test_not_a_repository_is_500_not_404(self, app, client, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        r = await _get(app, client, _LocalManager(empty), "/api/workspaces/ctr/git/diff")
        assert r.status_code == 500, r.text

    async def test_truncation_is_reported(self, app, client, repo):
        (repo / "src" / "a.py").write_text("y = 2\n" * 500)
        r = await _get(
            app, client, _LocalManager(repo), "/api/workspaces/ctr/git/diff?max_bytes=100"
        )
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["truncated"] is True
        assert len(d["diff"].encode()) <= 100
        assert d["diff_bytes"] > 100


@pytest.mark.asyncio
class TestGitDiffRefusals:
    @pytest.mark.parametrize(
        "query",
        [
            "base=--output=/tmp/pwn",  # option injection
            "base=-p",
            "base=main..HEAD",  # range syntax smuggled through one param
            "head=a...b",
            "base=HEAD%3Bid",  # ; shell metachar
            "base=HEAD%24(id)",  # $(...)
            "path=../etc/passwd",
            "path=src/../../etc",
            "path=/etc/passwd",
        ],
    )
    async def test_refused_with_400_and_no_command_run(self, app, client, repo, query):
        mgr = _LocalManager(repo)
        r = await _get(app, client, mgr, f"/api/workspaces/ctr/git/diff?{query}")
        assert r.status_code == 400, (query, r.text)
        assert mgr.commands == []

    async def test_container_not_found(self, app, client, repo):
        mgr = _LocalManager(repo)
        mgr.get_container = lambda name, tenant_id=None: None
        r = await _get(app, client, mgr, "/api/workspaces/ctr/git/diff")
        assert r.status_code == 404

    async def test_container_stopped(self, app, client, repo):
        r = await _get(app, client, _LocalManager(repo, "stopped"), "/api/workspaces/ctr/git/diff")
        assert r.status_code == 400


def test_route_is_published_as_get_observer(app):
    """GET => the gateway classifies it observer by default (operation_metadata)."""
    op = app.openapi()["paths"]["/api/workspaces/{container_name}/git/diff"]
    assert set(op) == {"get"}
    assert op["get"]["operationId"] == "get_workspace_git_diff"


class TestMutationControls:
    """Prove the real-repo tests can fail: break the route, assert they notice."""

    @pytest.mark.asyncio
    async def test_dropping_untracked_listing_is_detected(self, app, client, repo, monkeypatch):
        (repo / "new.txt").write_text("u\n")
        real = _LocalManager.execute_command

        async def no_ls_files(self, container_name, command, tenant_id=None, timeout=300):
            # The snapshot exec lists the untracked paths before ``add -A``; drop that.
            listing = "_g ls-files -z --others --exclude-standard"
            if listing in command:
                command = command.replace(listing, ":")
            return await real(self, container_name, command, tenant_id, timeout)

        monkeypatch.setattr(_LocalManager, "execute_command", no_ls_files)
        r = await _get(app, client, _LocalManager(repo), "/api/workspaces/ctr/git/diff")
        assert r.json()["untracked_files"] != ["new.txt"]

    @pytest.mark.asyncio
    async def test_unknown_ref_would_read_as_empty_without_the_404_branch(
        self, app, client, repo, monkeypatch
    ):
        async def lenient(manager, container_name, ref, tenant_id):
            return ref  # skip verification: pass the unresolved string straight on

        monkeypatch.setattr(git_diff, "_resolve_commit", lenient)
        r = await _get(
            app, client, _LocalManager(repo), "/api/workspaces/ctr/git/diff?base=no-such-branch"
        )
        # Without resolution the unknown ref reaches git diff, which fails -> 500:
        # the "unknown ref is 404, never empty/ambiguous" contract is gone.
        assert r.status_code == 500

    @pytest.mark.asyncio
    async def test_permissive_ref_pattern_lets_option_injection_reach_git(
        self, app, client, repo, monkeypatch
    ):
        import re

        monkeypatch.setattr(git_diff, "_REF_PATTERN", re.compile(r"^.{1,200}$"))
        mgr = _LocalManager(repo)
        r = await _get(app, client, mgr, "/api/workspaces/ctr/git/diff?base=-p")
        # the refusal test asserts 400 AND zero commands; under the mutant git runs
        assert mgr.commands != []
        assert r.status_code != 400

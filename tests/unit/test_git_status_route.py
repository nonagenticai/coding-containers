"""GET git/status against a real git repo.

The route used to strip the whole ``git status --porcelain`` output and slice
``[3:]`` per line, so the leading space of an unstaged FIRST entry (`` M a.py``)
was eaten and the path came back as ``.py``. These tests run the route's real
command against a scratch repo so the parser meets git's actual output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.test_git_diff_route import (  # noqa: F401  (autouse fixture)
    _get,
    _isolate_from_the_enclosing_git,
    _LocalManager,
    _run,
)

URL = "/api/containers/ctr/git/status"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _run(r, "init", "-q", "-b", "main")
    _run(r, "config", "user.email", "t@example.com")
    _run(r, "config", "user.name", "t")
    (r / "a.py").write_text("x = 1\n")
    (r / "b.py").write_text("y = 1\n")
    (r / "my file.txt").write_text("s\n")
    _run(r, "add", "-A")
    _run(r, "commit", "-q", "-m", "one")
    return r


async def _status(app, client, repo: Path) -> dict:
    mgr = _LocalManager(repo)
    r = await _get(app, client, mgr, URL)
    assert r.status_code == 200, r.text
    assert any("status" in c for c in mgr.commands)
    return r.json()


@pytest.mark.asyncio
class TestGitStatusAgainstRealRepo:
    async def test_first_entry_unstaged_modified_keeps_full_path(self, app, client, repo):
        (repo / "a.py").write_text("x = 2\n")
        d = await _status(app, client, repo)
        assert d["modified_files"] == ["a.py"]
        assert d["staged_changes"] == 0
        assert d["unstaged_changes"] == 1

    async def test_staged_modification(self, app, client, repo):
        (repo / "b.py").write_text("y = 2\n")
        _run(repo, "add", "b.py")
        d = await _status(app, client, repo)
        assert d["modified_files"] == ["b.py"]
        assert d["staged_changes"] == 1
        assert d["unstaged_changes"] == 0

    async def test_untracked(self, app, client, repo):
        (repo / "new.txt").write_text("u\n")
        d = await _status(app, client, repo)
        assert d["new_files"] == ["new.txt"]
        assert d["modified_files"] == []
        assert d["staged_changes"] == 0

    async def test_renamed_reports_new_path_modified_and_original_deleted(self, app, client, repo):
        _run(repo, "mv", "b.py", "c.py")
        d = await _status(app, client, repo)
        assert d["modified_files"] == ["c.py"]
        assert d["new_files"] == []
        assert d["deleted_files"] == ["b.py"]
        assert d["staged_changes"] == 1

    async def test_rename_out_of_scope_original_is_visible_to_a_scope_check(
        self, app, client, repo
    ):
        # n8n Check Worktree Scope unions modified ∪ new ∪ deleted and compares it to
        # qa_changed_files. Reporting only the new path let a commit that DELETES an
        # out-of-scope file pass a scope of ["in_scope.py"].
        (repo / "out_of_scope.py").write_text("def f():\n    return 1\n" * 20)
        _run(repo, "add", "out_of_scope.py")
        _run(repo, "commit", "-q", "-m", "two")
        _run(repo, "mv", "out_of_scope.py", "in_scope.py")
        d = await _status(app, client, repo)
        got = set(d["modified_files"]) | set(d["new_files"]) | set(d["deleted_files"])
        assert "out_of_scope.py" in d["deleted_files"]
        assert got - {"in_scope.py"} == {"out_of_scope.py"}  # -> scope_mismatch

    async def test_copy_keeps_original_out_of_deleted(self, app, client, repo):
        from src.routers.containers.git_status_commit import _parse_porcelain_z

        assert _parse_porcelain_z("C  new.py\0old.py\0 M a.py\0") == [
            ("C ", "new.py", "old.py"),
            (" M", "a.py", None),
        ]

    async def test_path_with_spaces(self, app, client, repo):
        (repo / "my file.txt").write_text("changed\n")
        (repo / "an other.txt").write_text("u\n")
        d = await _status(app, client, repo)
        assert d["modified_files"] == ["my file.txt"]
        assert d["new_files"] == ["an other.txt"]

    async def test_multiple_entries(self, app, client, repo):
        (repo / "a.py").write_text("x = 2\n")  # unstaged modify
        (repo / "b.py").unlink()  # unstaged delete
        _run(repo, "mv", "my file.txt", "moved file.txt")  # staged rename
        (repo / "staged.py").write_text("s\n")
        _run(repo, "add", "staged.py")  # staged add
        (repo / "zz.txt").write_text("u\n")  # untracked
        d = await _status(app, client, repo)
        assert sorted(d["modified_files"]) == ["a.py", "moved file.txt"]
        assert sorted(d["deleted_files"]) == ["b.py", "my file.txt"]
        assert sorted(d["new_files"]) == ["staged.py", "zz.txt"]
        assert d["staged_changes"] == 2
        # "??" has a non-space Y column, so untracked counts as unstaged (unchanged semantics)
        assert d["unstaged_changes"] == 3

    async def test_clean_tree(self, app, client, repo):
        d = await _status(app, client, repo)
        assert d["modified_files"] == d["new_files"] == d["deleted_files"] == []
        assert "no changes" in d["message"]

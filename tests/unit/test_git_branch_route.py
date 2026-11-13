"""Tests for POST /api/workspaces/{name}/git/branch.

Like ``test_git_restore_route.py`` (whose fake manager this reuses), these run the
route's REAL command strings against a REAL git repository in a temp dir. A mock with
canned stdout would pass for a route that ran ``checkout -B`` and silently reset a
branch - exactly the behaviour this route exists to replace.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.unit.test_git_restore_route import _LocalManager, _post, _run

_FORBIDDEN = (" checkout", " -B", "--force", " -f ", " reset", "--discard-changes", "--merge")


@pytest.fixture(autouse=True)
def _isolate_from_the_enclosing_git(monkeypatch):
    """Strip inherited ``GIT_*`` vars so a hook-exported GIT_DIR cannot redirect git
    into the developer's real repository (see test_git_diff_route.py)."""
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key, raising=False)
    # Hermetic identity. A submodule checkout is a fresh clone that inherits none of the
    # fixture's per-repo ``user.*``, so a commit inside it falls back to auto-detection:
    # macOS invents one, a CI container dies with "unable to auto-detect email address".
    # useConfigOnly forbids that fallback everywhere, so a missing identity reds locally too.
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "t")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "t@example.com")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.useConfigOnly")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """``main`` checked out; ``other`` exists one commit ahead with a different a.py."""
    r = tmp_path / "repo"
    r.mkdir()
    _run(r, "init", "-q", "-b", "main")
    _run(r, "config", "user.email", "t@example.com")
    _run(r, "config", "user.name", "t")
    (r / "src").mkdir()
    (r / "src" / "a.py").write_text("def a():\n    return 1\n")
    (r / "src" / "b.py").write_text("def b():\n    return 1\n")
    _run(r, "add", "-A")
    _run(r, "commit", "-q", "-m", "one")
    _run(r, "switch", "-q", "-c", "other")
    (r / "src" / "a.py").write_text("def a():\n    return 2\n")
    _run(r, "commit", "-qam", "other")
    _run(r, "switch", "-q", "main")
    return r


def _head(repo: Path) -> str:
    return _run(repo, "rev-parse", "HEAD")


def _current(repo: Path) -> str:
    return _run(repo, "rev-parse", "--abbrev-ref", "HEAD")


def _assert_never_forced(mgr: _LocalManager) -> None:
    for command in mgr.commands:
        # `[ -f PATH ]` is the shell's regular-file test (byte-exact recheck), not a flag.
        argv_text = command.replace("[ -f ", "[ ")
        assert not any(bad in argv_text for bad in _FORBIDDEN), command


async def _branch(app, client, mgr, branch, **kw):
    return await _post(app, client, mgr, {"branch": branch}, route="branch", **kw)


@pytest.mark.asyncio
class TestEnsureBranch:
    async def test_absent_branch_is_created_from_head(self, app, client, repo):
        main_sha = _head(repo)
        mgr = _LocalManager(repo)

        r = await _branch(app, client, mgr, "feat/new")

        assert r.status_code == 200, r.text
        d = r.json()
        assert d["created"] is True
        assert d["branch"] == "feat/new"
        assert d["previous_branch"] == "main"
        assert d["head_sha"] == main_sha
        assert _current(repo) == "feat/new"
        assert _run(repo, "rev-parse", "main") == main_sha  # main untouched
        _assert_never_forced(mgr)

    async def test_present_branch_is_switched_to_not_reset(self, app, client, repo):
        other_sha = _run(repo, "rev-parse", "other")
        mgr = _LocalManager(repo)

        r = await _branch(app, client, mgr, "other")

        assert r.status_code == 200, r.text
        d = r.json()
        assert d["created"] is False
        assert d["branch"] == "other"
        assert d["previous_branch"] == "main"
        # checkout -B would have moved `other` to main's commit and lost its commit
        assert d["head_sha"] == other_sha
        assert _run(repo, "rev-parse", "other") == other_sha
        assert _current(repo) == "other"
        assert (repo / "src" / "a.py").read_text() == "def a():\n    return 2\n"
        _assert_never_forced(mgr)

    async def test_repeat_request_is_idempotent(self, app, client, repo):
        first = await _branch(app, client, _LocalManager(repo), "feat/again")
        head = _head(repo)
        second = await _branch(app, client, _LocalManager(repo), "feat/again")

        assert first.status_code == 200 and second.status_code == 200, second.text
        assert first.json()["created"] is True
        d = second.json()
        assert d["created"] is False
        assert d["previous_branch"] == "feat/again"
        assert d["head_sha"] == head
        assert d["message"] == "Already on branch"

    @pytest.mark.parametrize("target", ["feat/new", "other"])
    async def test_uncommitted_work_survives(self, app, client, repo, target):
        # b.py is identical on main and other, so the switch may carry it over.
        (repo / "src" / "b.py").write_text("uncommitted work\n")
        (repo / "src" / "staged_new.py").write_text("staged work\n")
        _run(repo, "add", "src/staged_new.py")
        (repo / "src" / "untracked.py").write_text("untracked work\n")
        mgr = _LocalManager(repo)

        r = await _branch(app, client, mgr, target)

        assert r.status_code == 200, r.text
        assert _current(repo) == target
        assert (repo / "src" / "b.py").read_text() == "uncommitted work\n"
        assert (repo / "src" / "staged_new.py").read_text() == "staged work\n"
        assert (repo / "src" / "untracked.py").read_text() == "untracked work\n"
        assert _run(repo, "diff", "--cached", "--name-only") == "src/staged_new.py"
        assert _run(repo, "diff", "--name-only") == "src/b.py"
        _assert_never_forced(mgr)

    async def test_switch_that_would_overwrite_local_changes_is_409(self, app, client, repo):
        # a.py differs between main and other: switching would clobber the edit.
        (repo / "src" / "a.py").write_text("uncommitted edit\n")
        main_sha = _head(repo)
        other_sha = _run(repo, "rev-parse", "other")
        mgr = _LocalManager(repo)

        r = await _branch(app, client, mgr, "other")

        assert r.status_code == 409, r.text
        assert r.json()["error"]["message"].startswith("branch_switch_conflict")
        assert "src/a.py" in r.text
        # nothing changed: same branch, same commit, edit intact, other not moved
        assert _current(repo) == "main"
        assert _head(repo) == main_sha
        assert _run(repo, "rev-parse", "other") == other_sha
        assert (repo / "src" / "a.py").read_text() == "uncommitted edit\n"
        _assert_never_forced(mgr)

    async def test_untracked_file_that_would_be_overwritten_is_409(self, app, client, repo):
        (repo / "src" / "c.py").write_text("on other\n")
        _run(repo, "switch", "-q", "other")
        _run(repo, "add", "src/c.py")
        _run(repo, "commit", "-qm", "c")
        _run(repo, "switch", "-q", "main")
        (repo / "src" / "c.py").write_text("untracked work on main\n")

        r = await _branch(app, client, _LocalManager(repo), "other")

        assert r.status_code == 409, r.text
        assert r.json()["error"]["message"].startswith("branch_switch_conflict")
        assert _current(repo) == "main"
        assert (repo / "src" / "c.py").read_text() == "untracked work on main\n"

    async def test_detached_head_creates_from_the_detached_commit(self, app, client, repo):
        other_sha = _run(repo, "rev-parse", "other")
        _run(repo, "switch", "-q", "--detach", "other")
        (repo / "src" / "b.py").write_text("uncommitted work\n")
        mgr = _LocalManager(repo)

        r = await _branch(app, client, mgr, "from-detached")

        assert r.status_code == 200, r.text
        d = r.json()
        assert d["created"] is True
        assert d["previous_branch"] is None
        assert d["head_sha"] == other_sha
        assert _current(repo) == "from-detached"
        assert (repo / "src" / "b.py").read_text() == "uncommitted work\n"
        _assert_never_forced(mgr)

    async def test_detached_head_switches_to_existing_branch(self, app, client, repo):
        main_sha = _head(repo)
        _run(repo, "switch", "-q", "--detach", "other")

        r = await _branch(app, client, _LocalManager(repo), "main")

        assert r.status_code == 200, r.text
        d = r.json()
        assert d["created"] is False and d["previous_branch"] is None
        assert d["head_sha"] == main_sha
        assert _current(repo) == "main"

    @pytest.mark.parametrize(
        "name", ["-f", "-B", "a..b", "a b", "HEAD;id", "$(id)", "/x", "x/", "a.lock", "a\nb"]
    )
    async def test_invalid_name_refused_without_any_command(self, app, client, repo, name):
        mgr = _LocalManager(repo)
        before = _run(repo, "for-each-ref", "refs/heads")

        r = await _branch(app, client, mgr, name)

        assert r.status_code == 400, r.text
        assert mgr.commands == []
        assert _current(repo) == "main"
        assert _run(repo, "for-each-ref", "refs/heads") == before

    async def test_empty_name_is_422(self, app, client, repo):
        mgr = _LocalManager(repo)
        r = await _branch(app, client, mgr, "")
        assert r.status_code == 422, r.text
        assert mgr.commands == []

    @pytest.mark.parametrize("name", ["a//b", "feat///x"])
    async def test_name_git_rejects_is_400_before_any_mutation(self, app, client, repo, name):
        # Passes _validate_git_branch_name but not `git check-ref-format --branch`
        # (consecutive slashes), so it is refused by the probe, before any switch.
        mgr = _LocalManager(repo)
        before = _run(repo, "for-each-ref", "refs/heads")

        r = await _branch(app, client, mgr, name)

        assert r.status_code == 400, r.text
        assert "git rejects" in r.text
        assert not any(" switch " in c for c in mgr.commands)
        assert _current(repo) == "main"
        assert _run(repo, "for-each-ref", "refs/heads") == before

    async def test_repo_without_commits_is_400(self, app, client, tmp_path):
        r_dir = tmp_path / "empty"
        r_dir.mkdir()
        _run(r_dir, "init", "-q", "-b", "main")
        mgr = _LocalManager(r_dir)

        r = await _branch(app, client, mgr, "feat/x")

        assert r.status_code == 400, r.text
        assert not any(" switch " in c for c in mgr.commands)

    async def test_foreign_tenant_is_404_and_runs_nothing(self, app, client, repo):
        mgr = _LocalManager(repo, owner="tenant-a")
        r = await _branch(app, client, mgr, "feat/x", tenant_id="tenant-b")
        assert r.status_code == 404, r.text
        assert mgr.commands == []
        assert _current(repo) == "main"

    async def test_not_running_is_400(self, app, client, repo):
        mgr = _LocalManager(repo, status_val="stopped")
        r = await _branch(app, client, mgr, "feat/x")
        assert r.status_code == 400, r.text
        assert mgr.commands == []


def _drop_blob(repo: Path, rev_path: str) -> None:
    """Delete one loose object, as a corrupt or partially-fetched store would lack it."""
    sha = _run(repo, "rev-parse", rev_path)
    (repo / ".git" / "objects" / sha[:2] / sha[2:]).unlink()


@pytest.mark.asyncio
class TestIncompleteSwitchFailsClosed:
    """``git switch`` exits 0 when a blob of the target tree is missing: HEAD and the
    index move, the file is never written, and a later ``add -A`` commits a deletion."""

    async def test_missing_blob_is_500_not_a_silent_success(self, app, client, repo):
        _drop_blob(repo, "other:src/a.py")
        mgr = _LocalManager(repo)

        r = await _branch(app, client, mgr, "other")

        assert r.status_code == 500, r.text
        assert "branch_switch_incomplete" in r.text
        assert "src/a.py" in r.text
        _assert_never_forced(mgr)

    async def test_missing_blob_of_a_path_new_in_the_target_is_500(self, app, client, repo):
        _run(repo, "switch", "-q", "other")
        (repo / "src" / "c.py").write_text("only on other\n")
        _run(repo, "add", "src/c.py")
        _run(repo, "commit", "-qm", "c")
        _run(repo, "switch", "-q", "main")
        _drop_blob(repo, "other:src/c.py")

        r = await _branch(app, client, _LocalManager(repo), "other")

        assert r.status_code == 500, r.text
        assert "branch_switch_incomplete" in r.text
        assert "src/c.py" in r.text

    async def test_switch_carrying_an_existing_deletion_is_still_200(self, app, client, repo):
        # b.py is identical on both branches; the caller's own deletion carries over
        # and is not the switch's doing (a bare `git diff --quiet HEAD` would 500).
        (repo / "src" / "b.py").unlink()

        r = await _branch(app, client, _LocalManager(repo), "other")

        assert r.status_code == 200, r.text
        assert _current(repo) == "other"
        assert not (repo / "src" / "b.py").exists()

    async def test_unwritable_dir_leaving_stale_content_is_500(self, app, client, repo):
        # git prints "unable to unlink old 'src/a.py'" and exits 0 with HEAD on
        # `other`, but a.py keeps main's content: nothing is missing, it is STALE.
        src = repo / "src"
        src.chmod(0o555)
        try:
            r = await _branch(app, client, _LocalManager(repo), "other")
            stale = (src / "a.py").read_text()
        finally:
            src.chmod(0o755)

        if stale != "def a():\n    return 1\n":
            pytest.skip("directory permissions not enforced here (e.g. running as root)")
        assert _current(repo) == "other"
        assert r.status_code == 500, r.text
        assert "branch_switch_incomplete" in r.text
        assert "src/a.py" in r.text

    async def test_switch_carrying_an_existing_modification_is_still_200(self, app, client, repo):
        # b.py is identical on both branches; the caller's own edit carries over.
        (repo / "src" / "b.py").write_text("def b():\n    return 99\n")

        r = await _branch(app, client, _LocalManager(repo), "other")

        assert r.status_code == 200, r.text
        assert _current(repo) == "other"
        assert (repo / "src" / "b.py").read_text() == "def b():\n    return 99\n"

    async def test_create_carrying_an_existing_deletion_is_still_200(self, app, client, repo):
        (repo / "src" / "b.py").unlink()

        r = await _branch(app, client, _LocalManager(repo), "feat/new")

        assert r.status_code == 200, r.text
        assert r.json()["created"] is True
        assert _current(repo) == "feat/new"


@pytest.mark.asyncio
class TestByteExactRecheck:
    """``ls-files --modified`` compares through the clean conversion; paths whose
    worktree bytes equal the index blob exactly are not left incomplete."""

    async def test_crlf_blob_under_eol_crlf_is_200(self, app, client, tmp_path):
        # A blob committed WITH CRLF under `text eol=crlf`: checkout writes the blob's
        # bytes, but clean(CRLF->LF) no longer hashes to it, so -m flags x.bat.
        r = tmp_path / "crlf"
        r.mkdir()
        _run(r, "init", "-q", "-b", "main")
        _run(r, "config", "user.email", "t@example.com")
        _run(r, "config", "user.name", "t")
        (r / ".gitattributes").write_text("*.bat text eol=crlf\n")
        (r / "x.bat").write_bytes(b"a\r\nb\r\n")
        _run(r, "add", "-A")
        _run(r, "commit", "-q", "-m", "one")
        _run(r, "switch", "-q", "-c", "other")
        (tmp_path / "raw").write_bytes(b"a\r\nc\r\n")
        sha = _run(r, "hash-object", "-w", "--no-filters", str(tmp_path / "raw"))
        _run(r, "update-index", "--cacheinfo", f"100644,{sha},x.bat")
        _run(r, "commit", "-q", "-m", "crlf blob")
        _run(r, "switch", "-q", "-f", "main")
        assert _run(r, "ls-files", "-m") == ""

        resp = await _branch(app, client, _LocalManager(r), "other")

        assert resp.status_code == 200, resp.text
        assert _current(r) == "other"
        assert (r / "x.bat").read_bytes() == b"a\r\nc\r\n"
        assert _run(r, "ls-files", "-m") == "x.bat"  # the flag this recheck overrides

    @pytest.mark.parametrize("refusal", [True, False])
    async def test_submodule_pointer_change_is_200(
        self, app, client, repo, tmp_path, monkeypatch, refusal
    ):
        """Submodules are unsupported (409, shared prefix); with that refusal off, the
        byte-exact recheck still handles a pointer change (defence in depth)."""
        from src.routers.containers import _common

        monkeypatch.setattr(_common, "SUBMODULE_CHECK_ENABLED", refusal)
        sub = tmp_path / "sub"
        sub.mkdir()
        _run(sub, "init", "-q", "-b", "main")
        _run(sub, "config", "user.email", "t@example.com")
        _run(sub, "config", "user.name", "t")
        (sub / "s.txt").write_text("1\n")
        _run(sub, "add", "-A")
        _run(sub, "commit", "-q", "-m", "s1")
        _run(repo, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(sub), "sub")
        _run(repo, "commit", "-q", "-m", "add sub")
        _run(repo, "switch", "-q", "-c", "subnext")
        (repo / "sub" / "s.txt").write_text("2\n")
        _run(repo / "sub", "commit", "-qam", "s2")
        _run(repo, "add", "sub")
        _run(repo, "commit", "-q", "-m", "bump sub")
        _run(repo / "sub", "checkout", "-q", "HEAD~1")
        _run(repo, "switch", "-q", "main")
        assert _run(repo, "ls-files", "-m") == ""
        mgr = _LocalManager(repo)

        # main's pointer is s1, subnext's is s2; the submodule's own HEAD stays at s1.
        resp = await _branch(app, client, mgr, "subnext")

        if refusal:
            assert resp.status_code == 409, resp.text
            assert _common.WORKSPACE_SUBMODULES in resp.text
            assert _current(repo) == "main"
            return
        assert resp.status_code == 200, resp.text
        assert _current(repo) == "subnext"
        assert _run(repo, "ls-files", "-m") == "sub"  # the gitlink -m flags
        _assert_never_forced(mgr)


def _commit_all(repo: Path, message: str) -> None:
    _run(repo, "add", "-A")
    _run(repo, "commit", "-q", "-m", message)


@pytest.mark.asyncio
class TestRecheckCandidateShapes:
    """The recheck must stay valid shell and byte-exact for candidate sets holding no
    regular file, and for names ``--stdin-paths`` would C-unquote."""

    @pytest.fixture
    def linkrepo(self, repo: Path) -> Path:
        # `lnk` (in its own dir) points at a.py on main and at b.py on `other`.
        (repo / "d").mkdir()
        (repo / "d" / "lnk").symlink_to("../src/a.py")
        _commit_all(repo, "link")
        _run(repo, "switch", "-q", "other")
        _run(repo, "merge", "-q", "main", "-m", "m")
        (repo / "d" / "lnk").unlink()
        (repo / "d" / "lnk").symlink_to("../src/b.py")
        _commit_all(repo, "retarget")
        _run(repo, "switch", "-q", "main")
        # Only the link differs now, so it is the whole candidate set.
        (repo / "src" / "a.py").write_text("def a():\n    return 2\n")
        _commit_all(repo, "align a")
        assert _run(repo, "diff", "--name-only", "main", "other") == "d/lnk"
        return repo

    async def test_symlink_only_clean_target_change_is_200(self, app, client, linkrepo):
        r = await _branch(app, client, _LocalManager(linkrepo), "other")

        assert r.status_code == 200, r.text
        assert os.readlink(linkrepo / "d" / "lnk") == "../src/b.py"

    async def test_symlink_only_stale_is_500_with_missing_paths(self, app, client, linkrepo):
        d = linkrepo / "d"
        d.chmod(0o555)
        try:
            r = await _branch(app, client, _LocalManager(linkrepo), "other")
            stale = os.readlink(d / "lnk")
        finally:
            d.chmod(0o755)

        if stale != "../src/a.py":
            pytest.skip("directory permissions not enforced here (e.g. running as root)")
        assert _current(linkrepo) == "other"
        assert r.status_code == 500, r.text
        details = r.json()["error"]["details"]
        assert details.get("missing_paths") == ["d/lnk"], r.text
        assert details.get("verify_error") is None, r.text

    async def test_quote_leading_name_left_stale_is_500(self, app, client, repo):
        # Top-level `"q"` and `q` hold the same bytes on `other`: `--stdin-paths` would
        # C-unquote the line `"q"` to `q` and hash the wrong file clean.
        (repo / '"q"').write_text("old\n")
        (repo / "q").write_text("new\n")
        _commit_all(repo, "q files")
        _run(repo, "switch", "-q", "other")
        _run(repo, "merge", "-q", "main", "-m", "m")
        (repo / '"q"').write_text("new\n")
        _commit_all(repo, "q changes")
        _run(repo, "switch", "-q", "main")
        (repo / "src" / "a.py").write_text("def a():\n    return 2\n")
        _commit_all(repo, "align a")
        assert _run(repo, "diff", "--name-only", "-z", "main", "other").strip("\0") == '"q"'
        repo.chmod(0o555)
        try:
            r = await _branch(app, client, _LocalManager(repo), "other")
            stale = (repo / '"q"').read_text()
        finally:
            repo.chmod(0o755)

        if stale != "old\n":
            pytest.skip("directory permissions not enforced here (e.g. running as root)")
        assert r.status_code == 500, r.text
        assert r.json()["error"]["details"]["missing_paths"] == ['"q"'], r.text

    async def test_symlinks_false_plain_file_link_is_200(self, app, client, linkrepo):
        # core.symlinks=false: git writes the link as a plain file holding the target
        # and `ls-files -m` compares its content, so a clean switch flags nothing; a
        # flagged one is hashed raw by the recheck (not readlink) and matches the blob.
        _run(linkrepo, "config", "core.symlinks", "false")
        (linkrepo / "d" / "lnk").unlink()
        _run(linkrepo, "checkout", "--", "d/lnk")
        assert not (linkrepo / "d" / "lnk").is_symlink()

        r = await _branch(app, client, _LocalManager(linkrepo), "other")

        assert r.status_code == 200, r.text
        assert not (linkrepo / "d" / "lnk").is_symlink()
        assert (linkrepo / "d" / "lnk").read_text() == "../src/b.py"


class _MovedDespiteRefusal(_LocalManager):
    """The switch really runs (HEAD moves), but reports a conflict refusal. The
    route must not call that a 409 "nothing changed": only a re-measured,
    unmoved HEAD earns branch_switch_conflict."""

    async def execute_command(self, container_name, command, tenant_id=None, timeout=300):
        result = await super().execute_command(container_name, command, tenant_id, timeout)
        if " switch " in command:
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": "error: Your local changes to the following files would be overwritten by checkout",
            }
        return result


class TestConflictNeedsAnUnmovedHead:
    async def test_refusal_text_with_a_moved_head_is_500_not_409(self, app, client, repo):
        mgr = _MovedDespiteRefusal(repo)

        r = await _branch(app, client, mgr, "other")

        assert _current(repo) == "other"  # the switch did happen
        assert r.status_code == 500, r.text
        assert "branch_switch_conflict" not in r.text


# --- Linux MAX_ARG_STRLEN: the recheck is chunked; clipped listings fail closed ---

_MAX_ARG_STRLEN = 131072


class _ArgCappedManager(_LocalManager):
    """Enforces Linux's per-argument cap locally (macOS has none): a command whose
    ``sh -c`` string exceeds MAX_ARG_STRLEN fails like execve's E2BIG (exit 126)."""

    async def execute_command(self, container_name, command, tenant_id=None, timeout=300):
        if len(command.encode()) > _MAX_ARG_STRLEN:
            self.commands.append(command)
            return {"exit_code": 126, "stdout": "", "stderr": "Argument list too long"}
        return await super().execute_command(container_name, command, tenant_id, timeout)


def _recheck_commands(mgr: _LocalManager) -> list[str]:
    return [c for c in mgr.commands if "hash-object --no-filters" in c]


def _assert_under_arg_cap(mgr: _LocalManager) -> None:
    from src.routers.containers.git_status_commit import _RECHECK_ARG_BUDGET, _requoted_weight

    for c in mgr.commands:
        assert len(c.encode()) < _MAX_ARG_STRLEN
        assert len(c.encode()) <= 96 * 1024
        # also when a backend re-quotes it into an outer `sh -c` string
        assert _requoted_weight(c) + 2 <= _RECHECK_ARG_BUDGET


@pytest.fixture
def bigrepo(tmp_path: Path) -> tuple[Path, list[str]]:
    r = tmp_path / "big"
    r.mkdir()
    _run(r, "init", "-q", "-b", "main")
    _run(r, "config", "user.email", "t@example.com")
    _run(r, "config", "user.name", "t")
    d = r / "a_fairly_long_directory_name" / "and_a_nested_one"
    d.mkdir(parents=True)
    paths = []
    for i in range(3000):
        (d / f"file_{i:04d}.bat").write_bytes(f"line {i}\r\n".encode())
        paths.append(f"a_fairly_long_directory_name/and_a_nested_one/file_{i:04d}.bat")
    _commit_all(r, "3000 files")
    return r, paths


@pytest.mark.asyncio
class TestRecheckChunkedUnderArgMax:
    async def test_3000_clean_candidates_fit_and_come_back_empty(self, bigrepo):
        from src.routers.containers.git_status_commit import _still_incomplete

        r, paths = bigrepo
        mgr = _ArgCappedManager(r)

        result = await _still_incomplete(mgr, "ctr", "tenant-a", paths)

        assert result == []
        assert len(_recheck_commands(mgr)) > 1  # really chunked
        _assert_under_arg_cap(mgr)

    async def test_single_command_would_overflow_the_cap(self, bigrepo):
        # Control: the pre-chunking shape (one command for all) is over the cap.
        from src.routers.containers.git_status_commit import _byte_exact_recheck_command

        _, paths = bigrepo
        assert len(_byte_exact_recheck_command(paths, []).encode()) > _MAX_ARG_STRLEN

    async def test_stale_path_in_a_late_chunk_is_still_found(self, bigrepo):
        from src.routers.containers.git_status_commit import _still_incomplete

        r, paths = bigrepo
        (r / paths[-1]).write_bytes(b"stale\r\n")
        mgr = _ArgCappedManager(r)

        assert await _still_incomplete(mgr, "ctr", "tenant-a", paths) == [paths[-1]]
        _assert_under_arg_cap(mgr)

    async def test_mixed_symlink_and_regular_chunks(self, tmp_path):
        from src.routers.containers.git_status_commit import _recheck_chunks, _still_incomplete

        r = tmp_path / "mixed"
        r.mkdir()
        _run(r, "init", "-q", "-b", "main")
        _run(r, "config", "user.email", "t@example.com")
        _run(r, "config", "user.name", "t")
        (r / "files_dir_with_a_long_name").mkdir()
        (r / "links_dir_with_a_long_name").mkdir()
        regular, links = [], []
        for i in range(700):
            p = f"files_dir_with_a_long_name/regular_{i:04d}.txt"
            (r / p).write_text(f"{i}\n")
            regular.append(p)
        for i in range(700):
            p = f"links_dir_with_a_long_name/link_{i:04d}"
            (r / p).symlink_to(f"../files_dir_with_a_long_name/regular_{i:04d}.txt")
            links.append(p)
        _commit_all(r, "mixed")
        chunks = _recheck_chunks(regular, links)
        assert len(chunks) >= 3
        assert any(reg and sym for reg, sym in chunks)  # a chunk holding both kinds
        assert any(sym and not reg for reg, sym in chunks)  # a symlink-only chunk
        # one stale of each kind, in different chunks
        (r / regular[3]).write_text("stale\n")
        (r / links[-1]).unlink()
        (r / links[-1]).symlink_to("../elsewhere")
        mgr = _ArgCappedManager(r)

        result = await _still_incomplete(mgr, "ctr", "tenant-a", regular + links)

        assert result == sorted([regular[3], links[-1]])
        assert len(_recheck_commands(mgr)) == len(chunks)
        _assert_under_arg_cap(mgr)

    async def test_any_chunk_failing_fails_the_recheck(self, bigrepo):
        from src.routers.containers.git_status_commit import _still_incomplete

        r, paths = bigrepo

        class _SecondChunkFails(_LocalManager):
            seen = 0

            async def execute_command(self, container_name, command, tenant_id=None, timeout=300):
                res = await super().execute_command(container_name, command, tenant_id, timeout)
                if "hash-object --no-filters" in command:
                    type(self).seen += 1
                    if type(self).seen == 2:
                        return {**res, "exit_code": 126}
                return res

        assert await _still_incomplete(_SecondChunkFails(r), "ctr", "t", paths) is None

    async def test_truncated_chunk_output_fails_the_recheck(self, bigrepo):
        from src.routers.containers.git_status_commit import _still_incomplete

        r, paths = bigrepo

        class _Clipped(_LocalManager):
            async def execute_command(self, container_name, command, tenant_id=None, timeout=300):
                res = await super().execute_command(container_name, command, tenant_id, timeout)
                return {**res, "truncated": "hash-object --no-filters" in command}

        assert await _still_incomplete(_Clipped(r), "ctr", "t", paths[:5]) is None


class _TruncatedModifiedListing(_LocalManager):
    """``ls-files --modified`` comes back flagged truncated on call number ``which``
    (1 = before the switch, 2 = after)."""

    def __init__(self, repo: Path, which: int):
        super().__init__(repo)
        self.which = which
        self.listings = 0

    async def execute_command(self, container_name, command, tenant_id=None, timeout=300):
        res = await super().execute_command(container_name, command, tenant_id, timeout)
        if "ls-files -z --modified" in command:
            self.listings += 1
            if self.listings == self.which:
                return {**res, "truncated": True}
        return res


@pytest.mark.asyncio
class TestTruncatedListingFailsClosed:
    async def test_truncated_listing_after_switch_is_500_incomplete(self, app, client, repo):
        r = await _branch(app, client, _TruncatedModifiedListing(repo, 2), "other")

        assert _current(repo) == "other"
        assert r.status_code == 500, r.text
        details = r.json()["error"]["details"]
        assert details["error"] == "branch_switch_incomplete", r.text
        assert details["missing_paths"] is None
        assert "truncated" in details["verify_error"], r.text

    async def test_truncated_listing_before_switch_is_500_and_nothing_moves(
        self, app, client, repo
    ):
        head = _head(repo)
        mgr = _TruncatedModifiedListing(repo, 1)

        r = await _branch(app, client, mgr, "other")

        assert r.status_code == 500, r.text
        assert "truncated" in r.text
        assert _current(repo) == "main" and _head(repo) == head
        assert not any(" switch " in c for c in mgr.commands)

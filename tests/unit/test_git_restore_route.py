"""Tests for POST /api/workspaces/{name}/git/restore.

Like ``test_git_diff_route.py``, these run the route's REAL command strings against a
REAL git repository in a temp dir: the fake manager executes each command with
``sh -c`` after rewriting the ``/workspace`` root. A mock returning canned stdout would
pass for any command the route built, including one that restores nothing.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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


# The exact spelling of each mutation the route runs (``--literal-pathspecs <verb> ``).
# A bare " clean " substring would also match the canonical LFS literal
# ``git-lfs clean -- %f`` that every hardened read embeds.
_MUTATION_RE = re.compile(r"--literal-pathspecs (restore|clean|rm) ")


def _mutations(commands: list[str], *verbs: str) -> list[str]:
    """The commands that run one of ``verbs`` (default: any mutation)."""
    wanted = set(verbs or ("restore", "clean", "rm"))
    return [c for c in commands if any(m in wanted for m in _MUTATION_RE.findall(c))]


@pytest.fixture(autouse=True)
def _isolate_from_the_enclosing_git(monkeypatch):
    """Strip inherited ``GIT_*`` vars so a hook-exported GIT_DIR cannot redirect git
    into the developer's real repository (see test_git_diff_route.py)."""
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
    (r / "src" / "b.py").write_text("def b():\n    return 1\n")
    _run(r, "add", "-A")
    _run(r, "commit", "-q", "-m", "one")
    return r


class _LocalManager:
    """Honours ``tenant_id`` like ``ContainerManager.get_container``: a container owned
    by another tenant returns ``None`` (not-found and wrong-tenant are indistinguishable,
    BOLA protection); ``tenant_id=None`` is the admin path and skips the check."""

    def __init__(self, repo: Path, status_val: str = "running", owner: str = "tenant-a"):
        self.repo = repo
        self.status_val = status_val
        self.owner = owner
        self.commands: list[str] = []

    def get_container(self, name, tenant_id=None):
        if tenant_id is not None and tenant_id != self.owner:
            return None
        return _container(self.status_val)

    async def execute_command(self, container_name, command, tenant_id=None, timeout=300):
        self.commands.append(command)
        prefix = "cd /workspace && "
        assert command.startswith(prefix), command
        proc = subprocess.run(
            ["sh", "-c", f"cd '{self.repo}' && " + command[len(prefix) :]],
            capture_output=True,
            text=True,
        )
        return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


async def _post(app, client, manager, body, *, tenant_id="tenant-a", route="restore"):
    from coding_client.auth.tenant import get_current_user_with_tenant

    app.dependency_overrides[get_current_user_with_tenant] = lambda: _user(tenant_id=tenant_id)
    try:
        with patch("src.routers.containers.get_container_manager", return_value=manager):
            return await client.post(f"/api/workspaces/ctr/git/{route}", json=body)
    finally:
        app.dependency_overrides.clear()


def _bystanders(repo: Path) -> None:
    """Uncommitted work the route must never touch: a modified tracked file and an
    untracked file, neither of them requested."""
    (repo / "src" / "b.py").write_text("uncommitted work\n")
    (repo / "src" / "keep_untracked.py").write_text("untracked work\n")


def _assert_untouched(repo: Path, mgr: _LocalManager) -> None:
    assert (repo / "src" / "a.py").read_text() == "def a():\n    return 1\n"
    assert (repo / "src" / "b.py").read_text() == "uncommitted work\n"
    assert (repo / "src" / "keep_untracked.py").read_text() == "untracked work\n"
    assert _run(repo, "ls-files") == "src/a.py\nsrc/b.py"
    assert not _mutations(mgr.commands)


@pytest.mark.asyncio
class TestGitRestore:
    async def test_tracked_restore_reverts_worktree_and_index(self, app, client, repo):
        (repo / "src" / "a.py").write_text("candidate\n")
        _run(repo, "add", "src/a.py")  # staged too: --staged must undo it
        (repo / "src" / "b.py").write_text("other candidate\n")  # NOT requested

        r = await _post(app, client, _LocalManager(repo), {"files": ["src/a.py"]})

        assert r.status_code == 200, r.text
        d = r.json()
        assert d["restored"] == ["src/a.py"] and d["cleaned"] == []
        assert d["status_after"] == []
        assert (repo / "src" / "a.py").read_text() == "def a():\n    return 1\n"
        assert _run(repo, "diff", "--cached", "--name-only") == ""
        # an unrequested path is untouched
        assert (repo / "src" / "b.py").read_text() == "other candidate\n"

    async def test_restore_from_non_head_source(self, app, client, repo):
        first = _run(repo, "rev-parse", "HEAD")
        (repo / "src" / "a.py").write_text("second\n")
        _run(repo, "commit", "-qam", "two")

        r = await _post(app, client, _LocalManager(repo), {"files": ["src/a.py"], "source": first})

        assert r.status_code == 200, r.text
        assert (repo / "src" / "a.py").read_text() == "def a():\n    return 1\n"
        assert r.json()["status_after"] == ["M  src/a.py"]

    async def test_unknown_path_is_404_and_touches_nothing(self, app, client, repo):
        (repo / "src" / "a.py").write_text("candidate\n")
        mgr = _LocalManager(repo)

        r = await _post(app, client, mgr, {"files": ["src/a.py", "src/nope.py"]})

        assert r.status_code == 404, r.text
        assert "src/nope.py" in r.text
        # classification happens before ANY mutation: the known path was not restored
        assert (repo / "src" / "a.py").read_text() == "candidate\n"
        assert not _mutations(mgr.commands, "restore", "clean")

    async def test_untracked_without_flag_is_409(self, app, client, repo):
        (repo / "src" / "new.py").write_text("x = 1\n")

        r = await _post(app, client, _LocalManager(repo), {"files": ["src/new.py"]})

        assert r.status_code == 409, r.text
        assert "src/new.py" in r.text
        assert (repo / "src" / "new.py").exists()

    async def test_untracked_with_flag_is_cleaned_even_if_staged(self, app, client, repo):
        (repo / "src" / "new.py").write_text("x = 1\n")
        (repo / "src" / "staged_new.py").write_text("y = 1\n")
        _run(repo, "add", "src/staged_new.py")

        r = await _post(
            app,
            client,
            _LocalManager(repo),
            {"files": ["src/new.py", "src/staged_new.py"], "clean_untracked": True},
        )

        assert r.status_code == 200, r.text
        d = r.json()
        assert d["cleaned"] == ["src/new.py", "src/staged_new.py"]
        assert d["status_after"] == []
        assert not (repo / "src" / "new.py").exists()
        assert not (repo / "src" / "staged_new.py").exists()

    async def test_glob_shaped_path_is_literal_not_a_wildcard(self, app, client, repo):
        (repo / "src" / "a.py").write_text("candidate a\n")
        (repo / "src" / "b.py").write_text("candidate b\n")

        r = await _post(app, client, _LocalManager(repo), {"files": ["src/*.py"]})

        # no file is literally named "src/*.py": unknown, never "restore everything"
        assert r.status_code == 404, r.text
        assert (repo / "src" / "a.py").read_text() == "candidate a\n"

    async def test_glob_named_untracked_file_cleans_only_itself(self, app, client, repo):
        # A file LITERALLY named `*.py` next to an unlisted untracked file and an
        # edited+staged tracked one. Without literal pathspecs `git rm --cached` would
        # unstage every *.py and `git clean` would delete keep_untracked.py.
        (repo / "*.py").write_text("glob-named candidate\n")
        (repo / "keep_untracked.py").write_text("untracked work\n")
        (repo / "a.py").write_text("tracked\n")
        _run(repo, "add", "a.py")
        _run(repo, "commit", "-qm", "a")
        (repo / "a.py").write_text("staged edit\n")
        _run(repo, "add", "a.py")
        (repo / "a.py").write_text("unstaged edit\n")
        index_before = _index(repo)
        mgr = _LocalManager(repo)

        r = await _post(app, client, mgr, {"files": ["*.py"], "clean_untracked": True})

        assert r.status_code == 200, r.text
        assert r.json()["cleaned"] == ["*.py"]
        assert not (repo / "*.py").exists()
        assert (repo / "keep_untracked.py").read_text() == "untracked work\n"
        assert (repo / "a.py").read_text() == "unstaged edit\n"
        assert _index(repo) == index_before
        assert all("export GIT_LITERAL_PATHSPECS=1; " in c for c in mgr.commands)

    async def test_glob_named_tracked_file_restores_only_itself(self, app, client, repo):
        # `[ab].py` as a glob matches a.py and b.py; git restore must touch only the
        # file with that literal name.
        for name in ("[ab].py", "a.py", "b.py"):
            (repo / name).write_text(f"committed {name}\n")
        _run(repo, "add", "-A")
        _run(repo, "commit", "-qm", "glob")
        for name in ("[ab].py", "a.py", "b.py"):
            (repo / name).write_text(f"edited {name}\n")
        _run(repo, "add", "a.py")  # staged bystander edit
        index_before = _index(repo)

        r = await _post(app, client, _LocalManager(repo), {"files": ["[ab].py"]})

        assert r.status_code == 200, r.text
        assert r.json()["restored"] == ["[ab].py"]
        assert (repo / "[ab].py").read_text() == "committed [ab].py\n"
        assert (repo / "a.py").read_text() == "edited a.py\n"
        assert (repo / "b.py").read_text() == "edited b.py\n"
        # only the requested entry changed in the index (its stat data aside)
        assert [ln for ln in _index(repo).splitlines() if not ln.endswith("\t[ab].py")] == [
            ln for ln in index_before.splitlines() if not ln.endswith("\t[ab].py")
        ]

    async def test_glob_named_staged_new_file_unstaged_and_cleaned_alone(self, app, client, repo):
        # Staged-new `?.py` beside a staged-new `x.py`, an untracked `k.py` and an
        # edited+staged tracked `a.py`: every bystander matches `?.py` as a glob.
        (repo / "a.py").write_text("tracked\n")
        _run(repo, "add", "a.py")
        _run(repo, "commit", "-qm", "a")
        (repo / "a.py").write_text("staged edit\n")
        (repo / "?.py").write_text("glob-named staged candidate\n")
        (repo / "x.py").write_text("staged new work\n")
        _run(repo, "add", "a.py", "x.py")
        _run(repo, "--literal-pathspecs", "add", "?.py")
        (repo / "k.py").write_text("untracked work\n")
        index_before = [ln for ln in _index(repo).splitlines() if not ln.endswith("\t?.py")]

        r = await _post(
            app, client, _LocalManager(repo), {"files": ["?.py"], "clean_untracked": True}
        )

        assert r.status_code == 200, r.text
        assert r.json()["cleaned"] == ["?.py"]
        assert not (repo / "?.py").exists()
        assert _index(repo).splitlines() == index_before
        assert (repo / "a.py").read_text() == "staged edit\n"
        assert (repo / "x.py").read_text() == "staged new work\n"
        assert (repo / "k.py").read_text() == "untracked work\n"

    @pytest.mark.parametrize("path", [":bar", "src/:bar", ":!a.py", ":/a.py"])
    async def test_pathspec_magic_colon_segment_refused_without_any_command(
        self, app, client, repo, path
    ):
        # `git check-ignore` reads `:bar` as pathspec `bar`, so an IGNORED file named
        # `:bar` classified as cleanable and reached git clean (which skips it): 500.
        (repo / ".gitignore").write_text(":bar\n")
        (repo / ":bar").write_text("ignored\n")
        mgr = _LocalManager(repo)

        r = await _post(app, client, mgr, {"files": [path, "src/a.py"], "clean_untracked": True})

        assert r.status_code == 400, r.text
        assert "':'" in r.text
        assert mgr.commands == []
        assert (repo / ":bar").read_text() == "ignored\n"

    async def test_unknown_source_ref_is_404(self, app, client, repo):
        r = await _post(
            app, client, _LocalManager(repo), {"files": ["src/a.py"], "source": "no-such"}
        )
        assert r.status_code == 404, r.text

    @pytest.mark.parametrize("kind", ["tree_oid", "tag_to_tree", "blob_oid"])
    async def test_source_that_resolves_but_is_not_a_commit_is_404(self, app, client, repo, kind):
        # `rev-parse --verify <src>` alone accepts ANY object; `git restore --source=<tree>`
        # would then happily restore from it. Only a commit is a valid source.
        if kind == "tree_oid":
            source = _run(repo, "rev-parse", "HEAD^{tree}")
        elif kind == "blob_oid":
            source = _run(repo, "rev-parse", "HEAD:src/b.py")
        else:
            tree = _run(repo, "rev-parse", "HEAD^{tree}")
            _run(repo, "tag", "-a", "-m", "t", "treetag", tree)
            source = "treetag"
        (repo / "src" / "a.py").write_text("candidate\n")
        mgr = _LocalManager(repo)

        r = await _post(app, client, mgr, {"files": ["src/a.py"], "source": source})

        assert r.status_code == 404, r.text
        assert (repo / "src" / "a.py").read_text() == "candidate\n"
        assert not _mutations(mgr.commands)

    @pytest.mark.parametrize("source", ["-p", "main..HEAD", "HEAD~1", "a b", "HEAD;id", ""])
    async def test_bad_source_ref_is_422(self, app, client, repo, source):
        mgr = _LocalManager(repo)
        r = await _post(app, client, mgr, {"files": ["src/a.py"], "source": source})
        assert r.status_code == 422, r.text
        assert mgr.commands == []

    @pytest.mark.parametrize("path", ["-f", "src/-rf", "-x/y.py"])
    async def test_leading_dash_path_rejected(self, app, client, repo, path):
        mgr = _LocalManager(repo)
        r = await _post(app, client, mgr, {"files": [path]})
        assert r.status_code == 400, r.text
        assert mgr.commands == []

    @pytest.mark.parametrize("path", ["/etc/passwd", "../outside.py", "src/../../x", "a;rm"])
    async def test_escaping_or_metachar_path_rejected(self, app, client, repo, path):
        mgr = _LocalManager(repo)
        r = await _post(app, client, mgr, {"files": [path]})
        assert r.status_code == 400, r.text
        assert mgr.commands == []

    @pytest.mark.parametrize("path", [".", "src/.", "./src/a.py", "src/", "a//b", "./", ""])
    async def test_directory_spelling_rejected_and_nothing_touched(self, app, client, repo, path):
        _bystanders(repo)
        mgr = _LocalManager(repo)
        r = await _post(app, client, mgr, {"files": [path], "clean_untracked": True})
        assert r.status_code in (400, 422), r.text
        if path:  # "" is a 422 at the model layer or 400 here; either way no command
            assert r.status_code == 400, r.text
        assert mgr.commands == []
        _assert_untouched(repo, mgr)

    @pytest.mark.parametrize("clean_untracked", [False, True])
    async def test_tracked_directory_name_is_400(self, app, client, repo, clean_untracked):
        _bystanders(repo)
        mgr = _LocalManager(repo)
        r = await _post(app, client, mgr, {"files": ["src"], "clean_untracked": clean_untracked})
        assert r.status_code == 400, r.text
        assert "irector" in r.text
        _assert_untouched(repo, mgr)

    async def test_untracked_directory_name_is_400(self, app, client, repo):
        _bystanders(repo)
        (repo / "newdir").mkdir()
        (repo / "newdir" / "x.py").write_text("x\n")
        mgr = _LocalManager(repo)
        r = await _post(app, client, mgr, {"files": ["newdir"], "clean_untracked": True})
        assert r.status_code == 400, r.text
        assert (repo / "newdir" / "x.py").read_text() == "x\n"
        _assert_untouched(repo, mgr)

    async def test_auditor_destructive_dot_scenario_is_refused(self, app, client, repo):
        # The proven defect: "." classified as untracked, then `git rm -r --cached -- .`
        # + `git clean -f -d -- .` wiped every tracked file and all uncommitted work.
        _bystanders(repo)
        (repo / "src" / "a.py").write_text("candidate\n")
        mgr = _LocalManager(repo)
        r = await _post(app, client, mgr, {"files": ["src/a.py", "."], "clean_untracked": True})
        assert r.status_code == 400, r.text
        assert mgr.commands == []
        assert (repo / "src" / "a.py").read_text() == "candidate\n"
        assert (repo / "src" / "b.py").read_text() == "uncommitted work\n"
        assert (repo / "src" / "keep_untracked.py").read_text() == "untracked work\n"
        assert _run(repo, "ls-files") == "src/a.py\nsrc/b.py"

    async def test_foreign_tenant_gets_same_status_as_commit_route(self, app, client, repo):
        (repo / "src" / "a.py").write_text("candidate\n")
        mgr = _LocalManager(repo, owner="tenant-a")

        restore = await _post(app, client, mgr, {"files": ["src/a.py"]}, tenant_id="tenant-b")
        commit = await _post(
            app,
            client,
            mgr,
            {"files": ["src/a.py"], "message": "m"},
            tenant_id="tenant-b",
            route="commit",
        )

        assert restore.status_code == 404, restore.text
        assert restore.status_code == commit.status_code
        assert mgr.commands == []
        assert (repo / "src" / "a.py").read_text() == "candidate\n"

    async def test_owner_tenant_is_admitted(self, app, client, repo):
        # positive control for the tenant test: same manager, owning tenant -> 200
        (repo / "src" / "a.py").write_text("candidate\n")
        r = await _post(
            app,
            client,
            _LocalManager(repo, owner="tenant-a"),
            {"files": ["src/a.py"]},
            tenant_id="tenant-a",
        )
        assert r.status_code == 200, r.text

    async def test_empty_file_list_is_422(self, app, client, repo):
        r = await _post(app, client, _LocalManager(repo), {"files": []})
        assert r.status_code == 422, r.text

    async def test_not_running_is_400(self, app, client, repo):
        mgr = _LocalManager(repo, status_val="stopped")
        r = await _post(app, client, mgr, {"files": ["src/a.py"]})
        assert r.status_code == 400, r.text
        assert mgr.commands == []


def _index(repo: Path) -> str:
    return _run(repo, "ls-files", "-s")


@pytest.mark.asyncio
class TestGitRestoreRefusesPathsGitWouldSkipOrRedirect:
    """Every path is classified before anything is mutated. Paths ``git clean`` silently
    skips (ignored, inside a submodule / nested repo, behind a symlink, under .git) or
    that ``git restore`` would reach through a replaced parent must refuse the WHOLE
    request with 400 - never restore the tracked siblings and then 500."""

    async def test_ignored_file_refused_before_tracked_sibling_restored(self, app, client, repo):
        (repo / ".gitignore").write_text("*.log\n")
        _run(repo, "add", ".gitignore")
        _run(repo, "commit", "-qm", "ignore")
        (repo / "src" / "a.py").write_text("candidate\n")
        (repo / "x.log").write_text("log\n")
        mgr = _LocalManager(repo)

        r = await _post(app, client, mgr, {"files": ["src/a.py", "x.log"], "clean_untracked": True})

        assert r.status_code == 400, r.text
        assert "x.log" in r.text and "ignored" in r.text
        assert (repo / "src" / "a.py").read_text() == "candidate\n"
        assert (repo / "x.log").read_text() == "log\n"
        assert not _mutations(mgr.commands)

    async def test_force_staged_ignored_file_keeps_its_index_entry(self, app, client, repo):
        (repo / ".gitignore").write_text("*.log\n")
        _run(repo, "add", ".gitignore")
        _run(repo, "commit", "-qm", "ignore")
        (repo / "x.log").write_text("log\n")
        _run(repo, "add", "-f", "x.log")
        before = _index(repo)

        r = await _post(
            app, client, _LocalManager(repo), {"files": ["x.log"], "clean_untracked": True}
        )

        assert r.status_code == 400, r.text
        assert _index(repo) == before and "x.log" in before
        assert (repo / "x.log").exists()

    @pytest.mark.parametrize("refusal", [True, False])
    async def test_submodule_paths_refused(self, app, client, repo, tmp_path, monkeypatch, refusal):
        # Submodules are unsupported: the shared prefix refuses 409 first; with that
        # refusal off, the per-path classification still refuses 400 (defence in depth).
        from src.routers.containers import _common

        monkeypatch.setattr(_common, "SUBMODULE_CHECK_ENABLED", refusal)
        sub = tmp_path / "subsrc"
        sub.mkdir()
        _run(sub, "init", "-q", "-b", "main")
        _run(sub, "config", "user.email", "t@example.com")
        _run(sub, "config", "user.name", "t")
        (sub / "s.txt").write_text("s\n")
        _run(sub, "add", "-A")
        _run(sub, "commit", "-qm", "s")
        _run(repo, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(sub), "sm")
        _run(repo, "commit", "-qm", "sm")
        (repo / "sm" / "new.txt").write_text("n\n")
        (repo / "sm" / "s.txt").write_text("mod\n")
        (repo / "src" / "a.py").write_text("candidate\n")

        for files in (["src/a.py", "sm/new.txt"], ["src/a.py", "sm/s.txt"], ["sm"]):
            mgr = _LocalManager(repo)
            r = await _post(app, client, mgr, {"files": files, "clean_untracked": True})
            assert r.status_code == (409 if refusal else 400), (files, r.text)
            assert (_common.WORKSPACE_SUBMODULES in r.text) is refusal
            assert not _mutations(mgr.commands)
        assert (repo / "sm" / "new.txt").read_text() == "n\n"
        assert (repo / "sm" / "s.txt").read_text() == "mod\n"
        assert (repo / "src" / "a.py").read_text() == "candidate\n"

    @pytest.mark.parametrize("refusal", [True, False])
    async def test_gitlink_only_in_index_refused(self, app, client, repo, monkeypatch, refusal):
        from src.routers.containers import _common

        monkeypatch.setattr(_common, "SUBMODULE_CHECK_ENABLED", refusal)
        # A gitlink staged but not at source and with no .git on disk: only the index
        # check can see it.
        head = _run(repo, "rev-parse", "HEAD")
        _run(repo, "update-index", "--add", "--cacheinfo", f"160000,{head},gl")
        (repo / "gl").mkdir()
        (repo / "gl" / "f.txt").write_text("f\n")

        r = await _post(
            app, client, _LocalManager(repo), {"files": ["gl/f.txt"], "clean_untracked": True}
        )

        assert r.status_code == (409 if refusal else 400), r.text
        assert "submodule" in r.text
        assert (_common.WORKSPACE_SUBMODULES in r.text) is refusal
        assert (repo / "gl" / "f.txt").exists()

    async def test_nested_repo_file_refused(self, app, client, repo):
        nested = repo / "nested"
        nested.mkdir()
        _run(nested, "init", "-q")
        (nested / "f").write_text("x\n")

        r = await _post(
            app, client, _LocalManager(repo), {"files": ["nested/f"], "clean_untracked": True}
        )

        assert r.status_code == 400, r.text
        assert (nested / "f").read_text() == "x\n"

    async def test_untracked_symlinked_parent_refused(self, app, client, repo, tmp_path):
        out = tmp_path / "outside"
        out.mkdir()
        (out / "secret.txt").write_text("SECRET")
        os.symlink(out, repo / "link")

        r = await _post(
            app,
            client,
            _LocalManager(repo),
            {"files": ["link/secret.txt"], "clean_untracked": True},
        )

        assert r.status_code == 400, r.text
        assert (out / "secret.txt").read_text() == "SECRET"

    async def test_tracked_file_under_parent_replaced_by_symlink_refused(
        self, app, client, repo, tmp_path
    ):
        out = tmp_path / "outside"
        out.mkdir()
        (out / "a.py").write_text("OUTSIDE")
        (out / "other").write_text("o")
        import shutil

        shutil.rmtree(repo / "src")
        os.symlink(out, repo / "src")
        mgr = _LocalManager(repo)

        r = await _post(app, client, mgr, {"files": ["src/a.py"]})

        assert r.status_code == 400, r.text
        assert "symlink" in r.text
        assert os.path.islink(repo / "src")
        assert sorted(os.listdir(out)) == ["a.py", "other"]
        assert (out / "a.py").read_text() == "OUTSIDE"
        assert not _mutations(mgr.commands, "restore")

    async def test_tracked_file_under_parent_replaced_by_file_refused(self, app, client, repo):
        import shutil

        shutil.rmtree(repo / "src")
        (repo / "src").write_text("now a file\n")

        r = await _post(app, client, _LocalManager(repo), {"files": ["src/a.py"]})

        assert r.status_code == 400, r.text
        assert (repo / "src").read_text() == "now a file\n"

    @pytest.mark.parametrize(
        "path", [".git/config", ".git/HEAD", ".git/index", "sub/.git/x", ".GIT/config"]
    )
    async def test_dot_git_segment_refused_without_any_command(self, app, client, repo, path):
        mgr = _LocalManager(repo)
        r = await _post(app, client, mgr, {"files": [path], "clean_untracked": True})
        assert r.status_code == 400, r.text
        assert mgr.commands == []
        assert (repo / ".git" / "config").exists() and (repo / ".git" / "HEAD").exists()

    async def test_missing_parent_of_tracked_file_is_restored(self, app, client, repo):
        # positive control: a missing (not symlinked) parent is fine; git recreates it
        import shutil

        shutil.rmtree(repo / "src")
        r = await _post(app, client, _LocalManager(repo), {"files": ["src/a.py"]})
        assert r.status_code == 200, r.text
        assert (repo / "src" / "a.py").read_text() == "def a():\n    return 1\n"

    async def test_unicode_staged_new_file_is_cleaned(self, app, client, repo):
        (repo / "üni_new.py").write_text("x\n")
        _run(repo, "add", "üni_new.py")

        r = await _post(
            app, client, _LocalManager(repo), {"files": ["üni_new.py"], "clean_untracked": True}
        )

        assert r.status_code == 200, r.text
        assert r.json()["cleaned"] == ["üni_new.py"]
        assert not (repo / "üni_new.py").exists()
        assert _run(repo, "ls-files") == "src/a.py\nsrc/b.py"

    async def test_unicode_status_after_is_unquoted(self, app, client, repo):
        (repo / "ü.py").write_text("v1\n")
        _run(repo, "add", "ü.py")
        _run(repo, "commit", "-qm", "u")
        first = _run(repo, "rev-parse", "HEAD~1")
        r = await _post(
            app,
            client,
            _LocalManager(repo),
            {"files": ["ü.py"], "source": first, "clean_untracked": True},
        )
        assert r.status_code == 200, r.text
        assert r.json()["status_after"] == ["D  ü.py"]


def _needs_non_root() -> None:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores directory permissions; the failure cannot be provoked")


@pytest.mark.asyncio
class TestGitRestoreNeverPartiallyMutatesSilently:
    """Auditor reproducers: a refusal git would hit mid-way is refused UP FRONT (400,
    nothing mutated); a failure that cannot be predicted returns 500 with the
    re-measured state of every listed path, never a bare message."""

    async def test_staged_new_edited_after_staging_refused_before_any_mutation(
        self, app, client, repo
    ):
        # `git rm --cached` refuses "staged content different from both the file and the
        # HEAD" - previously AFTER the tracked sibling had already been restored.
        (repo / "src" / "a.py").write_text("candidate\n")
        (repo / "n.txt").write_text("v1\n")
        _run(repo, "add", "n.txt")
        (repo / "n.txt").write_text("v2\n")
        before = _index(repo)
        mgr = _LocalManager(repo)

        r = await _post(app, client, mgr, {"files": ["src/a.py", "n.txt"], "clean_untracked": True})

        assert r.status_code == 400, r.text
        assert "n.txt" in r.text and "staged" in r.text
        assert (repo / "src" / "a.py").read_text() == "candidate\n"
        assert (repo / "n.txt").read_text() == "v2\n"
        assert _index(repo) == before
        assert not _mutations(mgr.commands)

    async def test_restore_failing_partway_reports_measured_state(self, app, client, repo):
        _needs_non_root()
        (repo / "q").mkdir()
        (repo / "q" / "2").write_text("o2\n")
        _run(repo, "add", "q/2")
        _run(repo, "commit", "-qm", "q")
        (repo / "src" / "a.py").write_text("C1\n")
        (repo / "q" / "2").write_text("C2\n")
        _run(repo, "add", "-A")
        os.chmod(repo / "q", 0o555)
        try:
            r = await _post(app, client, _LocalManager(repo), {"files": ["src/a.py", "q/2"]})
        finally:
            os.chmod(repo / "q", 0o755)

        assert r.status_code == 500, r.text
        d = r.json()["error"]["details"]
        assert d["error"] == "partial_restore"
        assert d["restored"] == ["src/a.py"]
        assert d["cleaned"] == []
        assert [(f["path"], f["stage"]) for f in d["failed"]] == [("q/2", "restore")]
        assert d["failed"][0]["stderr_tail"]
        # status_after is the REAL post-failure status of every listed path
        real = subprocess.run(
            ["git", "-c", "core.quotePath=false", "status", "--porcelain", "--", "src/a.py", "q/2"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        assert d["status_after"] == real
        assert any(ln.endswith("q/2") for ln in real)
        assert (repo / "src" / "a.py").read_text() == "def a():\n    return 1\n"

    async def test_clean_failing_leaves_tracked_candidates_untouched(self, app, client, repo):
        # Clean runs BEFORE restore, so a clean-side failure never costs a tracked
        # candidate: the restore is reported as not attempted.
        _needs_non_root()
        (repo / "src" / "a.py").write_text("candidate\n")
        (repo / "ro").mkdir()
        (repo / "ro" / "u.txt").write_text("u\n")
        os.chmod(repo / "ro", 0o555)
        try:
            r = await _post(
                app,
                client,
                _LocalManager(repo),
                {"files": ["ro/u.txt", "src/a.py"], "clean_untracked": True},
            )
        finally:
            os.chmod(repo / "ro", 0o755)

        assert r.status_code == 500, r.text
        d = r.json()["error"]["details"]
        assert d["error"] == "partial_restore"
        assert d["restored"] == [] and d["cleaned"] == []
        assert {(f["path"], f["stage"]) for f in d["failed"]} == {
            ("ro/u.txt", "clean"),
            ("src/a.py", "restore"),
        }
        assert sorted(d["status_after"]) == [" M src/a.py", "?? ro/u.txt"]
        assert (repo / "src" / "a.py").read_text() == "candidate\n"
        assert (repo / "ro" / "u.txt").exists()

    async def test_parent_that_is_a_file_in_the_index_refused(self, app, client, tmp_path):
        # source has a/b; HEAD and the index have `a` as a FILE, deleted on disk. Restore
        # of a/b used to 200 and silently drop the UNLISTED index entry `a`.
        r_ = tmp_path / "r"
        r_.mkdir()
        _run(r_, "init", "-q", "-b", "main")
        _run(r_, "config", "user.email", "t@example.com")
        _run(r_, "config", "user.name", "t")
        (r_ / "a").mkdir()
        (r_ / "a" / "b").write_text("x\n")
        _run(r_, "add", "-A")
        _run(r_, "commit", "-qm", "1")
        old = _run(r_, "rev-parse", "HEAD")
        _run(r_, "rm", "-rq", "a")
        (r_ / "a").write_text("file a\n")
        _run(r_, "add", "a")
        _run(r_, "commit", "-qm", "2")
        (r_ / "a").unlink()
        before = _index(r_)
        mgr = _LocalManager(r_)

        r = await _post(app, client, mgr, {"files": ["a/b"], "source": old})

        assert r.status_code == 400, r.text
        assert "parent that is a file" in r.text
        assert _index(r_) == before
        assert not _mutations(mgr.commands, "restore")

    async def test_parent_that_is_a_file_only_in_the_index_refused(self, app, client, tmp_path):
        # HEAD (the source) has a/b, so `a` is a TREE at source and at HEAD; only the
        # INDEX has `a`, as a staged file (deleted on disk). Nothing but the index lookup
        # sees the conflict; restoring a/b would silently drop the unlisted entry `a`.
        r_ = tmp_path / "r"
        r_.mkdir()
        _run(r_, "init", "-q", "-b", "main")
        _run(r_, "config", "user.email", "t@example.com")
        _run(r_, "config", "user.name", "t")
        (r_ / "a").mkdir()
        (r_ / "a" / "b").write_text("x\n")
        _run(r_, "add", "-A")
        _run(r_, "commit", "-qm", "1")
        _run(r_, "rm", "-rq", "a")
        (r_ / "a").write_text("staged file a\n")
        _run(r_, "add", "a")
        (r_ / "a").unlink()
        before = _index(r_)
        assert before.endswith("\ta") and "a/b" not in before
        mgr = _LocalManager(r_)

        r = await _post(app, client, mgr, {"files": ["a/b"]})

        assert r.status_code == 400, r.text
        assert _index(r_) == before
        assert not _mutations(mgr.commands)

    async def test_tracked_file_replaced_on_disk_by_a_directory_refused(self, app, client, repo):
        # src/a.py is a BLOB at HEAD (the source) and in the index; only the DISK has it
        # as a directory, holding unlisted work. No tree exists anywhere in git, so only
        # the on-disk directory check can refuse; restore would delete unlisted.txt.
        (repo / "src" / "a.py").unlink()
        (repo / "src" / "a.py").mkdir()
        (repo / "src" / "a.py" / "unlisted.txt").write_text("unlisted work\n")
        before = _index(repo)
        mgr = _LocalManager(repo)

        r = await _post(app, client, mgr, {"files": ["src/a.py"], "clean_untracked": True})

        assert r.status_code == 400, r.text
        assert (repo / "src" / "a.py" / "unlisted.txt").read_text() == "unlisted work\n"
        assert _index(repo) == before
        assert not _mutations(mgr.commands)

    async def test_parent_that_is_a_file_only_at_source_refused(self, app, client, repo):
        # index/HEAD have src/ as a tree; the SOURCE has `src` as a blob.
        _run(repo, "rm", "-rq", "src")
        (repo / "src").write_text("blob\n")
        _run(repo, "add", "src")
        _run(repo, "commit", "-qm", "src is a file")
        blob_src = _run(repo, "rev-parse", "HEAD")
        _run(repo, "reset", "-q", "--hard", "HEAD~1")
        (repo / "src" / "a.py").write_text("candidate\n")
        r = await _post(
            app, client, _LocalManager(repo), {"files": ["src/a.py"], "source": blob_src}
        )
        assert r.status_code == 400, r.text
        assert (repo / "src" / "a.py").read_text() == "candidate\n"

    async def test_gitlink_at_source_only_refused(self, app, client, repo):
        # The gitlink exists ONLY in the source tree (not in HEAD, not in the index, no
        # .git on disk) and names a commit this repository does not have - the normal
        # submodule case, where `git cat-file -t src:gl` fails instead of saying commit.
        _run(repo, "update-index", "--add", "--cacheinfo", "160000," + "1" * 40 + ",gl")
        _run(repo, "commit", "-qm", "gitlink")
        with_gl = _run(repo, "rev-parse", "HEAD")
        _run(repo, "rm", "-q", "--cached", "gl")
        _run(repo, "commit", "-qm", "drop gitlink")
        (repo / "gl").mkdir()
        (repo / "gl" / "f.txt").write_text("f\n")
        mgr = _LocalManager(repo)

        r = await _post(
            app,
            client,
            mgr,
            {"files": ["gl/f.txt"], "source": with_gl, "clean_untracked": True},
        )

        assert r.status_code == 400, r.text
        assert "submodule" in r.text
        assert (repo / "gl" / "f.txt").read_text() == "f\n"
        assert not _mutations(mgr.commands)


class _RecordingManager(_LocalManager):
    """Runs nothing: answers each step with the verdicts a restorable request gets, so
    the route builds every command (classify, mutate, verify, status) for paths that
    cannot exist on this filesystem. Only the command SIZES are under test."""

    def __init__(self, repo: Path, verdict: str, n: int):
        super().__init__(repo)
        self.verdict = verdict
        self.n = n

    async def execute_command(self, container_name, command, tenant_id=None, timeout=300):
        self.commands.append(command)
        # Step markers first: every command's shared prefix also runs `rev-parse --verify`.
        if "; _cls " in command:
            return {"exit_code": 0, "stdout": f"{self.verdict}\n" * self.n, "stderr": ""}
        if "; _vt " in command or "; _vu " in command:
            return {"exit_code": 0, "stdout": "ok\n" * self.n, "stderr": ""}
        return {"exit_code": 0, "stdout": "", "stderr": ""}


def _worst_case_paths(shape: str) -> list[str]:
    """20 paths of the maximum length the model admits, in the two shapes that cost
    most: every character a quote (shlex.quote grows each to 5 bytes) or one-character
    segments (maximal depth - a per-prefix probe would grow quadratically)."""
    from src.models.container_api_models import RESTORE_MAX_FILES, RESTORE_MAX_PATH_CHARS

    paths = []
    for i in range(RESTORE_MAX_FILES):
        tail = f"f{i:02d}"
        if shape == "quotes":
            body = "'" * (RESTORE_MAX_PATH_CHARS - len(tail))
        else:
            body = "d/" * ((RESTORE_MAX_PATH_CHARS - len(tail)) // 2)
        p = (body + tail)[:RESTORE_MAX_PATH_CHARS]
        assert len(p) >= RESTORE_MAX_PATH_CHARS - 1
        paths.append(p)
    return paths


@pytest.mark.asyncio
class TestGitRestoreBoundsAndTrust:
    """Auditor round 4: request size is bounded against the kernel's single-argument
    limit, index flags that make git ignore the disk are refused, and success is
    measured by hashing the bytes, never by asking ``git diff``."""

    async def test_more_than_twenty_files_is_422(self, app, client, repo):
        mgr = _LocalManager(repo)
        r = await _post(app, client, mgr, {"files": [f"f{i}.py" for i in range(21)]})
        assert r.status_code == 422, r.text
        assert mgr.commands == []

    async def test_twenty_files_is_accepted(self, app, client, repo):
        r = await _post(app, client, _LocalManager(repo), {"files": ["src/a.py"] * 20})
        assert r.status_code == 200, r.text

    async def test_path_longer_than_512_chars_is_422(self, app, client, repo):
        mgr = _LocalManager(repo)
        r = await _post(app, client, mgr, {"files": ["a/" + "y" * 511]})
        assert r.status_code == 422, r.text
        assert mgr.commands == []

    @pytest.mark.parametrize("shape", ["quotes", "deep"])
    @pytest.mark.parametrize(("verdict", "clean"), [("T", False), ("U", True)])
    async def test_worst_case_commands_stay_far_below_max_arg_strlen(
        self, app, client, repo, shape, verdict, clean
    ):
        # Every step is ONE `sh -c` argument; Linux refuses any single argv string
        # over MAX_ARG_STRLEN = 131072 bytes (E2BIG), and no exec backend has stdin.
        paths = _worst_case_paths(shape)
        mgr = _RecordingManager(repo, verdict, len(paths))
        r = await _post(app, client, mgr, {"files": paths, "clean_untracked": clean})
        assert r.status_code == 200, r.text
        sizes = [len(c.encode()) for c in mgr.commands]
        assert len(sizes) >= 4  # ref, classify, >=1 mutation, verify, status
        assert max(sizes) < 100_000, sizes

    async def test_twenty_long_quoted_paths_restore_end_to_end(self, app, client, repo):
        # The same worst-case quoting, against a REAL repository: 10 tracked candidates
        # restored and 10 staged-new files cleaned in one request.
        q = "'" * 255
        tracked = [f"{q}/t{i:02d}" + "'" * 250 for i in range(10)]
        untracked = [f"{q}/u{i:02d}" + "'" * 250 for i in range(10)]
        (repo / q).mkdir()
        for p in tracked:
            (repo / p).write_text("orig\n")
        _run(repo, "add", "-A")
        _run(repo, "commit", "-qm", "quotes")
        for p in tracked:
            (repo / p).write_text("candidate\n")
        for p in untracked:
            (repo / p).write_text("new\n")
        _run(repo, "add", "-A")
        mgr = _LocalManager(repo)

        r = await _post(app, client, mgr, {"files": tracked + untracked, "clean_untracked": True})

        assert r.status_code == 200, r.text
        assert r.json()["restored"] == tracked and r.json()["cleaned"] == untracked
        assert all((repo / p).read_text() == "orig\n" for p in tracked)
        assert not any((repo / p).exists() for p in untracked)
        assert max(len(c.encode()) for c in mgr.commands) < 100_000

    @pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"])
    @pytest.mark.parametrize("clean", [False, True])
    async def test_index_flag_that_hides_the_disk_is_refused(self, app, client, repo, flag, clean):
        # skip-worktree: `git restore` leaves the candidate on disk and `git diff`
        # reports the path clean - previously a 200 "restored" over unreverted bytes.
        (repo / "src" / "a.py").write_text("candidate\n")
        _run(repo, "update-index", flag, "src/b.py")
        (repo / "src" / "b.py").write_text("CANDIDATE\n")
        before = _index(repo)
        mgr = _LocalManager(repo)

        r = await _post(
            app, client, mgr, {"files": ["src/a.py", "src/b.py"], "clean_untracked": clean}
        )

        assert r.status_code == 400, r.text
        assert "src/b.py" in r.text and "skip-worktree or assume-unchanged" in r.text
        assert "src/a.py has" not in r.text
        assert (repo / "src" / "a.py").read_text() == "candidate\n"
        assert (repo / "src" / "b.py").read_text() == "CANDIDATE\n"
        assert _index(repo) == before
        tag = subprocess.run(
            ["git", "ls-files", "-v", "src/b.py"], cwd=repo, capture_output=True, text=True
        ).stdout[0]
        assert tag == ("S" if flag == "--skip-worktree" else "h")
        assert not _mutations(mgr.commands)

    async def test_restore_that_leaves_the_candidate_on_disk_is_500_not_200(
        self, app, client, repo
    ):
        # A restore step that exits 0 but leaves the candidate bytes behind, with the
        # index flagged so `git diff` calls the path clean. Verification must hash the
        # file, so this is the honest partial_restore report, never "restored".
        (repo / "src" / "a.py").write_text("candidate\n")

        class _LyingRestore(_LocalManager):
            async def execute_command(self, container_name, command, tenant_id=None, timeout=300):
                result = await super().execute_command(container_name, command, tenant_id)
                if _mutations([command], "restore"):
                    _run(self.repo, "update-index", "--skip-worktree", "src/a.py")
                    (self.repo / "src" / "a.py").write_text("candidate\n")
                return result

        r = await _post(app, client, _LyingRestore(repo), {"files": ["src/a.py"]})

        assert r.status_code == 500, r.text
        d = r.json()["error"]["details"]
        assert d["error"] == "partial_restore"
        assert d["restored"] == []
        assert [(f["path"], f["stage"]) for f in d["failed"]] == [("src/a.py", "verify")]

    async def test_restore_that_stages_the_wrong_blob_is_500(self, app, client, repo):
        # Disk correct, INDEX wrong: the index blob is compared too.
        (repo / "src" / "a.py").write_text("candidate\n")
        other = _run(repo, "rev-parse", "HEAD:src/b.py")

        class _WrongIndex(_LocalManager):
            async def execute_command(self, container_name, command, tenant_id=None, timeout=300):
                result = await super().execute_command(container_name, command, tenant_id)
                if _mutations([command], "restore"):
                    _run(self.repo, "update-index", "--cacheinfo", f"100644,{other},src/a.py")
                return result

        r = await _post(app, client, _WrongIndex(repo), {"files": ["src/a.py"]})

        assert r.status_code == 500, r.text
        assert r.json()["error"]["details"]["restored"] == []

    async def test_tracked_symlink_is_verified_by_its_target(self, app, client, repo):
        os.symlink("a.py", repo / "src" / "link")
        _run(repo, "add", "src/link")
        _run(repo, "commit", "-qm", "link")
        (repo / "src" / "link").unlink()
        os.symlink("b.py", repo / "src" / "link")

        r = await _post(app, client, _LocalManager(repo), {"files": ["src/link"]})

        assert r.status_code == 200, r.text
        assert r.json()["restored"] == ["src/link"]
        assert os.readlink(repo / "src" / "link") == "a.py"

    async def test_path_with_staged_new_children_in_index_is_refused(self, app, client, repo):
        # `d` is a blob at HEAD; the index instead holds staged-new `d/x` (unlisted
        # work), and nothing is on disk. `git restore d` would silently DROP d/x from
        # the index - the X guard (index entries other than the path) must refuse.
        (repo / "d").write_text("blob\n")
        _run(repo, "add", "d")
        _run(repo, "commit", "-qm", "d")
        _run(repo, "rm", "-q", "--cached", "d")
        (repo / "d").unlink()
        (repo / "d").mkdir()
        (repo / "d" / "x").write_text("UNLISTED WORK\n")
        _run(repo, "add", "d/x")
        (repo / "d" / "x").unlink()
        (repo / "d").rmdir()
        before = _index(repo)
        assert "\td/x" in before
        mgr = _LocalManager(repo)

        r = await _post(app, client, mgr, {"files": ["d"]})

        assert r.status_code == 400, r.text
        assert "index entries other than itself" in r.text
        assert _index(repo) == before
        assert not _mutations(mgr.commands, "restore")

    @pytest.mark.parametrize("ignorecase", ["true", "false"])
    async def test_case_variant_of_a_tracked_file_leaves_it_untouched(
        self, app, client, repo, tmp_path, ignorecase
    ):
        # `a.py` requested, `A.py` tracked. On a case-insensitive filesystem (macOS
        # dev boxes) `a.py` looks like a lone untracked file; cleaning it is a no-op
        # git refuses to apply to a tracked path, so the result is the honest 500
        # partial_restore. On a case-sensitive one (Linux containers) it is simply
        # unknown: 404. Either way A.py - content and index entry - is untouched.
        _run(repo, "config", "core.ignorecase", ignorecase)
        (repo / "A.py").write_text("tracked\n")
        _run(repo, "add", "A.py")
        _run(repo, "commit", "-qm", "A")
        (repo / "A.py").write_text("edited\n")
        before = _index(repo)
        probe = tmp_path / "CaseProbe"
        probe.write_text("")
        case_insensitive_fs = (tmp_path / "caseprobe").exists()

        r = await _post(
            app, client, _LocalManager(repo), {"files": ["a.py"], "clean_untracked": True}
        )

        if case_insensitive_fs:
            assert r.status_code == 500, r.text
            d = r.json()["error"]["details"]
            assert d["error"] == "partial_restore"
            assert d["restored"] == [] and d["cleaned"] == []
            assert [(f["path"], f["stage"]) for f in d["failed"]] == [("a.py", "verify")]
        else:
            assert r.status_code == 404, r.text
        assert (repo / "A.py").read_text() == "edited\n"
        assert _index(repo) == before
        assert "A.py" in os.listdir(repo)


@pytest.mark.asyncio
class TestGitRestoreOracle:
    """The ``_mutations`` oracle must fire on each real mutation and on nothing else."""

    async def test_each_mutation_is_detected_and_reads_are_not(self, app, client, repo):
        from src.routers.containers._common import CANONICAL_LFS_FILTERS

        (repo / "src" / "a.py").write_text("candidate\n")
        (repo / "src" / "new.py").write_text("x = 1\n")
        mgr = _LocalManager(repo)

        r = await _post(
            app, client, mgr, {"files": ["src/a.py", "src/new.py"], "clean_untracked": True}
        )

        assert r.status_code == 200, r.text
        assert len(_mutations(mgr.commands, "rm")) == 1
        assert len(_mutations(mgr.commands, "clean")) == 1
        assert len(_mutations(mgr.commands, "restore")) == 1
        assert len(_mutations(mgr.commands)) == 3
        reads = [c for c in mgr.commands if c not in _mutations(mgr.commands)]
        assert len(reads) >= 4  # ref, classify, verify, status
        # Every hardened read embeds the canonical LFS literal: the bare substring the
        # old oracle used matches it, the precise one does not.
        assert any(CANONICAL_LFS_FILTERS["filter.lfs.clean"] in c for c in reads)
        assert all(" clean " in c for c in reads if "git-lfs clean" in c)
        assert not _mutations(reads)


def _marker_script(repo: Path, name: str) -> tuple[Path, Path]:
    marker = repo.parent / f"{name}.ran"
    script = repo.parent / f"{name}.sh"
    script.write_text(f'#!/bin/sh\ntouch "{marker}"\ncat >/dev/null 2>&1\nexit 1\n')
    script.chmod(0o755)
    return script, marker


@pytest.mark.asyncio
class TestGitRestoreDoesNotRunRepoCommands:
    """The classifier, the mutations and the verifier all run git with the workspace
    config's fsmonitor and filter commands overridden."""

    @pytest.fixture
    def armed(self, repo: Path) -> list[Path]:
        fsmon, fsmon_marker = _marker_script(repo, "fsmonitor")
        clean, clean_marker = _marker_script(repo, "clean")
        smudge, smudge_marker = _marker_script(repo, "smudge")
        (repo / ".git" / "info").mkdir(exist_ok=True)
        (repo / ".git" / "info" / "attributes").write_text("*.py filter=evil\n")
        _run(repo, "config", "core.fsmonitor", str(fsmon))
        _run(repo, "config", "filter.evil.clean", str(clean))
        _run(repo, "config", "filter.evil.smudge", str(smudge))
        # Arming control: plain git DOES run each of them.
        # Same content, same size, new mtime: git cannot decide from stat alone and MUST
        # run the clean filter to re-hash. A size-changing rewrite would NOT arm it (a
        # size mismatch is "modified" without reading content), leaving the control to
        # racy-git luck on b.py: clean ran only when the fixture's commit happened to
        # land in the same timestamp tick as the files. mtime is attacker-reachable
        # (any touch), and it is what the LFS read test already does.
        a = repo / "src" / "a.py"
        st = a.stat()
        os.utime(a, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
        subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True)
        (repo / "src" / "b.py").unlink()
        subprocess.run(["git", "checkout", "--", "src/b.py"], cwd=repo, capture_output=True)
        for marker in (fsmon_marker, clean_marker, smudge_marker):
            assert marker.exists(), f"fixture not armed: {marker.name}"
            marker.unlink()
        return [fsmon_marker, clean_marker, smudge_marker]

    async def test_restore_and_clean_run_no_fsmonitor_or_filter(self, app, client, repo, armed):
        (repo / "src" / "a.py").write_text("candidate\n")
        (repo / "src" / "new.py").write_text("x = 1\n")
        _run(repo, "-c", "core.fsmonitor=false", "-c", "filter.evil.clean=", "add", "src/new.py")
        mgr = _LocalManager(repo)

        r = await _post(
            app, client, mgr, {"files": ["src/a.py", "src/new.py"], "clean_untracked": True}
        )

        assert [m.name for m in armed if m.exists()] == []
        assert r.status_code == 200, r.text
        assert r.json()["restored"] == ["src/a.py"] and r.json()["cleaned"] == ["src/new.py"]
        assert (repo / "src" / "a.py").read_text() == "def a():\n    return 1\n"
        assert not (repo / "src" / "new.py").exists()
        assert len(_mutations(mgr.commands)) == 3  # every stage really ran

    async def test_unexpressible_filter_key_fails_closed_before_mutating(self, app, client, repo):
        script, marker = _marker_script(repo, "eq")
        (repo / ".git" / "info").mkdir(exist_ok=True)
        (repo / ".git" / "info" / "attributes").write_text("*.py filter=a=b\n")
        _run(repo, "config", "filter.a=b.clean", str(script))
        (repo / "src" / "a.py").write_text("candidate\n")
        mgr = _LocalManager(repo)

        r = await _post(app, client, mgr, {"files": ["src/a.py"]})

        # The first hardened read (the source-ref check) already refuses: fail closed.
        assert r.status_code in (404, 500), r.text
        assert not marker.exists()
        assert (repo / "src" / "a.py").read_text() == "candidate\n"
        assert not _mutations(mgr.commands)


# --- partial clone: a restore must never lazily fetch through a planted promisor remote ---


def _plant_promisor(repo: Path, kind: str) -> tuple[str, Path]:
    """Commit a second revision, delete the first revision's ``src/a.py`` blob and make
    the repo a partial clone whose promisor remote runs a command when git lazily
    fetches. ``kind``: ``ext`` (an ``ext::`` URL under ``protocol.allow=always``) or
    ``uploadpack`` (a local-path remote with a planted ``remote.origin.uploadpack``,
    which ``protocol.ext.allow=never`` and ``core.sshCommand=false`` do not stop).
    Returns the source rev holding the missing blob and the marker the command creates.
    """
    marker = repo.parent / f"lazy-{kind}.ran"
    first = _run(repo, "rev-parse", "HEAD")
    (repo / "src" / "a.py").write_text("second\n")
    _run(repo, "commit", "-q", "-am", "two")
    missing = _run(repo, "rev-parse", f"{first}:src/a.py")
    (repo / ".git" / "objects" / missing[:2] / missing[2:]).unlink()
    _run(repo, "config", "extensions.partialClone", "origin")
    _run(repo, "config", "remote.origin.promisor", "true")
    if kind == "ext":
        _run(repo, "config", "protocol.allow", "always")
        _run(repo, "config", "remote.origin.url", f"ext::sh -c touch% {marker}")
    else:
        upstream = repo.parent / "upstream"
        upstream.mkdir()
        _run(upstream, "init", "-q", "--bare")
        script = repo.parent / "uploadpack.sh"
        script.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 1\n')
        script.chmod(0o755)
        _run(repo, "config", "remote.origin.url", str(upstream))
        _run(repo, "config", "remote.origin.uploadpack", str(script))
    # Arming control: plain git restore lazily fetches through the planted remote.
    (repo / "src" / "a.py").write_text("candidate\n")
    subprocess.run(
        ["git", "restore", f"--source={first}", "--staged", "--worktree", "--", "src/a.py"],
        cwd=repo,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    assert marker.exists(), f"fixture not armed: git restore did not lazy-fetch ({kind})"
    marker.unlink()
    (repo / "src" / "a.py").write_text("candidate\n")
    return first, marker


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["ext", "uploadpack"])
async def test_restore_never_lazy_fetches_through_planted_promisor(app, client, repo, kind):
    source, marker = _plant_promisor(repo, kind)
    mgr = _LocalManager(repo)

    r = await _post(app, client, mgr, {"files": ["src/a.py"], "source": source})

    assert not marker.exists(), f"the restore lazily fetched through the {kind} remote"
    assert _mutations(mgr.commands, "restore")  # the restore really ran
    # Missing blob, no fetch: fails closed, never a false "restored".
    assert r.status_code == 500, r.text
    assert "src/a.py" not in (r.json().get("restored") or [])

"""A workspace that plants replace refs, grafts or alternates cannot fool a verify read.

coding-containers#156: code that runs inside the workspace (tests, dependency sync, a
worker's candidate) can plant ``refs/replace/*``, ``.git/info/grafts`` or
``objects/info/alternates``. Plain git READS then report a history, tree or blob other
than the one ``commit``/``push`` write, so a "verify, then act" check passes on content
that is not what ships. Every workspace git now runs under two independent layers
(``src/routers/containers/_common.py``):

* the ENV layer: ``GIT_NO_REPLACE_OBJECTS=1``, ``GIT_GRAFT_FILE=/dev/null``,
  ``-c core.commitGraph=false`` (``WORKSPACE_GIT_ENV`` / ``WORKSPACE_GIT_CONFIG``);
* the REFUSAL layer: the workspace is refused (409 ``workspace_git_poisoned``) before any
  read when it holds replace refs, grafts or alternates (``workspace_poison_check``).

Each attack below runs the route's REAL commands against a REAL repo, once per layer
setting. That IS the mutation matrix, run on every test run:

=============  ==================  ==========================================
layers on      expected            proves
=============  ==================  ==========================================
both           409 refused         the shipped configuration
refusal only   409 refused         the refusal alone stops it (env removed)
env only       the TRUE content    the env alone stops it (refusal removed)
neither        the FAKE content    the attack is real (arming control)
=============  ==================  ==========================================

``alternates`` has no env switch, so it is covered by the refusal only.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fastapi import HTTPException

import src.routers.containers.git_push as gp
from src.routers.containers import _common
from tests.unit.test_git_diff_route import (  # noqa: F401  (autouse fixture)
    _get,
    _isolate_from_the_enclosing_git,
    _LocalManager,
    _run,
)
from tests.unit.test_git_restore_route import _post

LAYERS = ["both", "refusal_only", "env_only", "neither"]


def _layers(monkeypatch, layer: str) -> None:
    """Remove the refusal and/or the env hardening, exactly as a regression would."""
    if layer in ("env_only", "neither"):
        monkeypatch.setattr(_common, "POISON_CHECK_ENABLED", False)
    if layer == "neither":
        # The object-integrity check (a third layer, #156 validator) also catches a
        # replaced BLOB (``cat-file`` returns the replacement bytes, which do not hash to
        # the id); it is off here so the arming control still shows the attack is real.
        monkeypatch.setattr(_common, "OBJECT_CHECK_ENABLED", False)
    if layer in ("refusal_only", "neither"):
        monkeypatch.setattr(
            _common,
            "WORKSPACE_GIT_ENV",
            tuple(
                e
                for e in _common.WORKSPACE_GIT_ENV
                if not e.startswith(("GIT_NO_REPLACE_OBJECTS=", "GIT_GRAFT_FILE="))
            ),
        )
        monkeypatch.setattr(
            _common,
            "WORKSPACE_GIT_CONFIG",
            tuple(c for c in _common.WORKSPACE_GIT_CONFIG if c != "core.commitGraph=false"),
        )


def _refused(r) -> bool:
    return r.status_code == 409 and "workspace_git_poisoned" in r.text


def _blob(repo: Path, content: str) -> str:
    return subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=repo,
        input=content,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit_with(repo: Path, files: dict[str, str], parent: str | None) -> str:
    """A commit (not on any branch) whose tree is HEAD's with ``files`` overridden."""
    idx = repo / ".git" / "scratch-index"
    env_idx = {"GIT_INDEX_FILE": str(idx)}
    subprocess.run(["git", "read-tree", "HEAD"], cwd=repo, check=True, env=_env(env_idx))
    for path, content in files.items():
        blob = _blob(repo, content)
        subprocess.run(
            ["git", "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}"],
            cwd=repo,
            check=True,
            env=_env(env_idx),
        )
    tree = subprocess.run(
        ["git", "write-tree"],
        cwd=repo,
        check=True,
        env=_env(env_idx),
        capture_output=True,
        text=True,
    ).stdout.strip()
    idx.unlink()
    args = ["git", "commit-tree", tree, "-m", "forged"] + (["-p", parent] if parent else [])
    return subprocess.run(args, cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _env(extra: dict[str, str]) -> dict[str, str]:
    import os

    return {**os.environ, **extra}


def _alternates(repo: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.git"
    subprocess.run(["git", "init", "-q", "--bare", str(other)], check=True)
    (repo / ".git" / "objects" / "info").mkdir(parents=True, exist_ok=True)
    (repo / ".git" / "objects" / "info" / "alternates").write_text(f"{other / 'objects'}\n")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _run(r, "init", "-q", "-b", "main")
    _run(r, "config", "user.email", "t@example.com")
    _run(r, "config", "user.name", "t")
    (r / "a.py").write_text("x = 1\n")
    _run(r, "add", "-A")
    _run(r, "commit", "-q", "-m", "one")
    return r


def _two_commits(repo: Path) -> tuple[str, str]:
    """HEAD~1 = benign a.py, HEAD = a.py with an EVIL line. Returns (c0, c1)."""
    c0 = _run(repo, "rev-parse", "HEAD")
    (repo / "a.py").write_text("x = 1\nEVIL = 1\n")
    _run(repo, "commit", "-qam", "candidate")
    return c0, _run(repo, "rev-parse", "HEAD")


# ---------------------------------------------------------------------------
# Site 1: GET git/diff (r.qa.diff.fix-unit -> q8 / q12 scope + line cap)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDiff:
    URL = "/api/workspaces/ctr/git/diff?base=HEAD~1&head=HEAD"

    @pytest.mark.parametrize("layer", LAYERS)
    async def test_replace_ref_cannot_hide_the_shipped_line(
        self, app, client, repo, monkeypatch, layer
    ):
        _two_commits(repo)
        evil = _run(repo, "rev-parse", "HEAD:a.py")
        _run(repo, "replace", evil, _blob(repo, "x = 1\nOK = 1\n"))
        # Arming control: plain git shows the replacement.
        assert "EVIL" not in _run(repo, "diff", "HEAD~1", "HEAD")
        _layers(monkeypatch, layer)

        r = await _get(app, client, _LocalManager(repo), self.URL)

        if layer in ("both", "refusal_only"):
            assert _refused(r), r.text
        elif layer == "env_only":
            assert r.status_code == 200, r.text
            assert "+EVIL = 1" in r.json()["diff"]
            assert r.json()["blobs"] == {"a.py": evil}
        else:
            assert r.status_code == 200, r.text
            assert "EVIL" not in r.json()["diff"]  # fooled

    @pytest.mark.parametrize("layer", LAYERS)
    async def test_graft_cannot_move_the_base(self, app, client, repo, monkeypatch, layer):
        c0, c1 = _two_commits(repo)
        same = _commit_with(repo, {"a.py": "x = 1\nEVIL = 1\n"}, c0)  # HEAD's own tree
        (repo / ".git" / "info").mkdir(exist_ok=True)
        (repo / ".git" / "info" / "grafts").write_text(f"{c1} {same}\n")
        assert _run(repo, "diff", "HEAD~1", "HEAD") == ""  # arming: base moved
        _layers(monkeypatch, layer)

        r = await _get(app, client, _LocalManager(repo), self.URL)

        if layer in ("both", "refusal_only"):
            assert _refused(r), r.text
        elif layer == "env_only":
            assert r.status_code == 200, r.text
            assert r.json()["base_sha"] == c0
            assert "+EVIL = 1" in r.json()["diff"]
        else:
            assert r.status_code == 200, r.text
            assert r.json()["diff"] == ""  # fooled: "no change"

    @pytest.mark.parametrize("layer", ["both", "refusal_only"])
    async def test_alternates_are_refused(self, app, client, repo, tmp_path, monkeypatch, layer):
        _two_commits(repo)
        _alternates(repo, tmp_path)
        _layers(monkeypatch, layer)
        r = await _get(app, client, _LocalManager(repo), self.URL)
        assert _refused(r), r.text
        assert "objects/info/alternates" in r.text

    async def test_a_minus_diff_attribute_cannot_blank_the_content(self, app, client, repo):
        (repo / ".git" / "info").mkdir(exist_ok=True)
        (repo / ".git" / "info" / "attributes").write_text("*.py -diff\n")
        (repo / "a.py").write_text("x = 1\nEVIL = 1\n")
        # Arming control: plain git reports a binary change and no content.
        plain = _run(repo, "diff", "HEAD")
        assert "Binary files" in plain and "EVIL" not in plain

        r = await _get(app, client, _LocalManager(repo), "/api/workspaces/ctr/git/diff")

        assert r.status_code == 200, r.text
        assert "+EVIL = 1" in r.json()["diff"]
        assert "Binary files" not in r.json()["diff"]

    async def test_blobs_are_what_git_add_stores(self, app, client, repo):
        """The working-tree ``blobs`` equal the ids the commit then records, for a
        modified, a new (untracked), a deleted and a renamed path."""
        (repo / "b.py").write_text("b = 1\n")
        _run(repo, "add", "b.py")
        _run(repo, "commit", "-qm", "b")
        (repo / "a.py").write_text("x = 2\n")
        (repo / "new.py").write_text("n = 1\n")
        _run(repo, "rm", "-q", "b.py")

        r = await _get(app, client, _LocalManager(repo), "/api/workspaces/ctr/git/diff")

        assert r.status_code == 200, r.text
        blobs = r.json()["blobs"]
        _run(repo, "add", "-A")
        _run(repo, "commit", "-qm", "c")
        assert blobs == {
            "a.py": _run(repo, "rev-parse", "HEAD:a.py"),
            "new.py": _run(repo, "rev-parse", "HEAD:new.py"),
            "b.py": None,
        }


# ---------------------------------------------------------------------------
# Site 2: GET git/status (s.done.commit.fix-unit "worktree changes == qa_changed_files")
# ---------------------------------------------------------------------------


def _paths_in_status(body: dict) -> set[str]:
    out: set[str] = set()
    for value in body.values():
        if isinstance(value, list):
            out |= {v for v in value if isinstance(v, str)}
    return out


@pytest.mark.asyncio
class TestStatus:
    URL = "/api/containers/ctr/git/status"

    @pytest.mark.parametrize("layer", LAYERS)
    async def test_replace_ref_cannot_hide_a_staged_extra_file(
        self, app, client, repo, monkeypatch, layer
    ):
        (repo / "extra.py").write_text("import os\n")
        _run(repo, "add", "extra.py")  # what `commit` will write
        tree = _run(repo, "write-tree")
        fake_head = _run(repo, "commit-tree", tree, "-m", "looks like HEAD")
        _run(repo, "replace", "HEAD", fake_head)
        assert _run(repo, "status", "--porcelain") == ""  # arming: status reads clean
        _layers(monkeypatch, layer)

        r = await _get(app, client, _LocalManager(repo), self.URL)

        if layer in ("both", "refusal_only"):
            assert _refused(r), r.text
        elif layer == "env_only":
            assert r.status_code == 200, r.text
            assert "extra.py" in _paths_in_status(r.json())
        else:
            assert r.status_code == 200, r.text
            assert "extra.py" not in _paths_in_status(r.json())  # fooled

    @pytest.mark.parametrize("layer", ["both", "refusal_only"])
    async def test_alternates_are_refused(self, app, client, repo, tmp_path, monkeypatch, layer):
        _alternates(repo, tmp_path)
        _layers(monkeypatch, layer)
        assert _refused(await _get(app, client, _LocalManager(repo), self.URL))

    @pytest.mark.parametrize("pinned", [True, False])
    async def test_show_untracked_files_no_cannot_hide_a_new_file(
        self, app, client, repo, monkeypatch, pinned
    ):
        """``add -A`` commits an untracked file whatever ``status.showUntrackedFiles``
        says, so the status read must not honour a workspace ``no``."""
        _run(repo, "config", "status.showUntrackedFiles", "no")
        (repo / "extra.py").write_text("import os\n")
        assert _run(repo, "status", "--porcelain") == ""  # arming
        if not pinned:  # mutation: drop the -c pin
            monkeypatch.setattr(
                _common,
                "WORKSPACE_GIT_CONFIG",
                tuple(c for c in _common.WORKSPACE_GIT_CONFIG if not c.startswith("status.show")),
            )
        r = await _get(app, client, _LocalManager(repo), self.URL)
        assert r.status_code == 200, r.text
        assert ("extra.py" in _paths_in_status(r.json())) is pinned


# ---------------------------------------------------------------------------
# Site 3: POST git/restore (r.qa.revert.fix-unit "restored to source")
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRestore:
    @pytest.mark.parametrize("layer", LAYERS)
    async def test_replace_ref_cannot_fake_a_restore(self, app, client, repo, monkeypatch, layer):
        # Replacing only the BLOB is caught by the verifier (it hashes the disk); a
        # replaced COMMIT moves the tree it lists, so the restore and its self-check
        # agree on a blob that is not at source.
        _run(repo, "replace", "HEAD", _commit_with(repo, {"a.py": "EVIL = 1\n"}, None))
        (repo / "a.py").write_text("candidate = 1\n")
        _layers(monkeypatch, layer)

        r = await _post(app, client, _LocalManager(repo), {"files": ["a.py"], "source": "HEAD"})

        on_disk = (repo / "a.py").read_text()
        if layer in ("both", "refusal_only"):
            assert _refused(r), r.text
            assert on_disk == "candidate = 1\n"  # nothing was touched
        elif layer == "env_only":
            assert r.status_code == 200, r.text
            assert on_disk == "x = 1\n"
        else:
            assert r.status_code == 200, r.text  # "restored" and self-verified ...
            assert on_disk == "EVIL = 1\n"  # ... to content that is not at source

    @pytest.mark.parametrize("layer", ["both", "refusal_only"])
    async def test_grafts_are_refused_before_any_mutation(
        self, app, client, repo, monkeypatch, layer
    ):
        """``source`` admits no ``~``/``^`` (422), so a graft cannot move it; a workspace
        holding one is still refused rather than trusted."""
        c0, c1 = _two_commits(repo)
        (repo / ".git" / "info").mkdir(exist_ok=True)
        (repo / ".git" / "info" / "grafts").write_text(f"{c1} {c0}\n")
        (repo / "a.py").write_text("candidate = 1\n")
        _layers(monkeypatch, layer)
        r = await _post(app, client, _LocalManager(repo), {"files": ["a.py"], "source": "HEAD"})
        assert _refused(r), r.text
        assert (repo / "a.py").read_text() == "candidate = 1\n"

    async def test_alternates_are_refused_before_any_mutation(self, app, client, repo, tmp_path):
        _alternates(repo, tmp_path)
        (repo / "a.py").write_text("candidate = 1\n")
        r = await _post(app, client, _LocalManager(repo), {"files": ["a.py"], "source": "HEAD"})
        assert _refused(r), r.text
        assert (repo / "a.py").read_text() == "candidate = 1\n"


# ---------------------------------------------------------------------------
# Site 4: POST git/branch (ensure_branch "switch is complete")
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEnsureBranch:
    @pytest.mark.parametrize("layer", LAYERS)
    async def test_replace_ref_cannot_swap_the_checked_out_tree(
        self, app, client, repo, monkeypatch, layer
    ):
        c0 = _run(repo, "rev-parse", "HEAD")
        _run(repo, "checkout", "-q", "-b", "feat")
        (repo / "a.py").write_text("feat = 'real'\n")
        _run(repo, "commit", "-qam", "feat")
        feat = _run(repo, "rev-parse", "HEAD")
        _run(repo, "checkout", "-q", "main")
        _run(repo, "replace", feat, _commit_with(repo, {"a.py": "feat = 'fake'\n"}, c0))
        _layers(monkeypatch, layer)

        r = await _post(app, client, _LocalManager(repo), {"branch": "feat"}, route="branch")

        on_disk = (repo / "a.py").read_text()
        if layer in ("both", "refusal_only"):
            assert _refused(r), r.text
            assert _run(repo, "symbolic-ref", "--short", "HEAD") == "main"
            assert on_disk == "x = 1\n"
        elif layer == "env_only":
            assert r.status_code == 200, r.text
            assert on_disk == "feat = 'real'\n"
        else:
            assert r.status_code == 200, r.text
            assert on_disk == "feat = 'fake'\n"  # "switch complete" on a fake tree


# ---------------------------------------------------------------------------
# Push: the CONTENT at the pushed tip must be the reviewed content
# ---------------------------------------------------------------------------


class _ShManager:
    """Runs a git_push command verbatim (it already cds to the patched ``_WORKDIR``)."""

    async def execute_command(self, container_name, command, tenant_id=None, timeout=300):
        proc = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
        return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


@pytest.mark.asyncio
class TestPushContent:
    async def _check(self, tip: str, reviewed: dict[str, str | None]) -> HTTPException | None:
        try:
            await gp._check_pushed_content(
                _common._PoisonGuardedManager(_ShManager()), "ctr", None, tip, ["a.py"], reviewed
            )
        except HTTPException as e:
            return e
        return None

    @pytest.mark.parametrize("layer", LAYERS)
    async def test_replace_ref_cannot_pass_other_content(self, repo, monkeypatch, layer):
        monkeypatch.setattr(gp, "_WORKDIR", str(repo))
        c0, tip = _two_commits(repo)
        reviewed = _blob(repo, "x = 1\nOK = 1\n")  # what the faked diff showed QA
        _run(repo, "replace", tip, _commit_with(repo, {"a.py": "x = 1\nOK = 1\n"}, c0))
        assert _run(repo, "rev-parse", f"{tip}:a.py") == reviewed  # arming
        _layers(monkeypatch, layer)

        err = await self._check(tip, {"a.py": reviewed})

        if layer in ("both", "refusal_only"):
            assert err is not None and err.status_code == 409
            assert err.detail["error_code"] == "workspace_git_poisoned"
        elif layer == "env_only":
            assert err is not None and err.status_code == 403
            assert err.detail["error_code"] == "push_content_mismatch"
        else:
            assert err is None  # fooled: the EVIL blob would ship

    async def test_the_reviewed_blob_passes_and_a_later_edit_does_not(
        self, app, client, repo, monkeypatch
    ):
        """End to end on an honest workspace: diff ``blobs`` -> commit -> push check."""
        monkeypatch.setattr(gp, "_WORKDIR", str(repo))
        (repo / "a.py").write_text("x = 2\n")
        r = await _get(app, client, _LocalManager(repo), "/api/workspaces/ctr/git/diff")
        reviewed = r.json()["blobs"]
        _run(repo, "commit", "-qam", "reviewed")
        assert await self._check(_run(repo, "rev-parse", "HEAD"), reviewed) is None

        (repo / "a.py").write_text("x = 2\nEVIL = 1\n")  # changed after review
        _run(repo, "commit", "-qam", "sneaked in")
        err = await self._check(_run(repo, "rev-parse", "HEAD"), reviewed)
        assert err is not None and err.detail["error_code"] == "push_content_mismatch"

    async def test_alternates_are_refused(self, repo, tmp_path, monkeypatch):
        monkeypatch.setattr(gp, "_WORKDIR", str(repo))
        _alternates(repo, tmp_path)
        err = await self._check(_run(repo, "rev-parse", "HEAD"), {"a.py": None})
        assert err is not None and err.detail["error_code"] == "workspace_git_poisoned"


def test_the_push_and_the_helpers_share_one_hardening_list():
    """git_push builds its env and ``-c`` list from ``_common``; neither may drift."""
    for entry in _common.WORKSPACE_GIT_ENV:
        assert entry in gp._in_workdir("true")
        assert entry in gp._push_command("o", "r", "b", "a" * 40, force=False)
    for pair in _common.WORKSPACE_GIT_CONFIG:
        assert pair in gp._hardened_git()
        assert pair in _common.workspace_git_read(["status"])
        assert pair in _common.workspace_git_mutate(["add", "-A"])
        assert pair in _common.workspace_git_function("_g")
    for required in ("GIT_NO_REPLACE_OBJECTS=1", "GIT_GRAFT_FILE=/dev/null"):
        assert required in _common.WORKSPACE_GIT_ENV
    assert "core.commitGraph=false" in _common.WORKSPACE_GIT_CONFIG

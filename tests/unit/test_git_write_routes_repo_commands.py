"""git/commit and git/branch must not run commands named by the workspace repo's config.

The workspace config is attacker-controllable, and each of these is a COMMAND git runs
on its own: ``core.fsmonitor`` (any index read), ``filter.<d>.clean`` (``git add``),
``filter.<d>.smudge`` (``git switch`` writing the worktree), a hook
(``pre-commit``/``commit-msg``/``post-commit``/``post-checkout``), and a partial
clone's promisor remote (``ext::<command>``), which git runs when it lazily fetches a
missing object. Every test first proves the planted command DOES fire under plain git
(the fixture is armed), then runs the route's REAL commands against the same repo and
asserts the marker was never written.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.unit.test_git_restore_route import _LocalManager, _post, _run


@pytest.fixture(autouse=True)
def _isolate_from_the_enclosing_git(monkeypatch):
    """Strip inherited ``GIT_*`` vars so a hook-exported GIT_DIR cannot redirect git
    into the developer's real repository (see test_git_diff_route.py)."""
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key, raising=False)


def _script(repo: Path, name: str, body: str = "cat >/dev/null 2>&1\nexit 0") -> Path:
    """An executable that touches ``<name>.ran`` next to the repo; returns the marker."""
    marker = repo.parent / f"{name}.ran"
    path = repo.parent / f"{name}.sh"
    path.write_text(f'#!/bin/sh\ntouch "{marker}"\n{body}\n')
    path.chmod(0o755)
    return marker


def _plain(repo: Path, *args: str) -> None:
    """Plain git (the arming control): no hardening, failures tolerated."""
    subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, stdin=subprocess.DEVNULL, timeout=60
    )


def _armed(marker: Path, what: str) -> None:
    assert marker.exists(), f"fixture not armed: {what}"
    marker.unlink()


def _plant_promisor(repo: Path, missing_blob: str) -> Path:
    """Delete ``missing_blob`` and make the repo a partial clone whose promisor remote
    is ``ext::`` touching a marker. Returns the marker."""
    marker = repo.parent / "lazy.ran"
    (repo / ".git" / "objects" / missing_blob[:2] / missing_blob[2:]).unlink()
    _run(repo, "config", "extensions.partialClone", "evil")
    _run(repo, "config", "remote.evil.promisor", "true")
    _run(repo, "config", "protocol.allow", "always")
    _run(repo, "config", "remote.evil.url", f"ext::sh -c touch% {marker}")
    return marker


# --------------------------------------------------------------------------- commit


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


async def _commit(app, client, mgr, files=("*",)):
    return await _post(
        app, client, mgr, {"message": "route commit", "files": list(files)}, route="commit"
    )


def _last_message(repo: Path) -> str:
    return _run(repo, "log", "-1", "--format=%s")


@pytest.mark.asyncio
class TestCommitRouteRunsNoRepoCommand:
    async def test_fsmonitor(self, app, client, repo):
        marker = _script(repo, "fsmonitor", "exit 1")
        _run(repo, "config", "core.fsmonitor", str(repo.parent / "fsmonitor.sh"))
        (repo / "src" / "a.py").write_text("changed\n")
        _plain(repo, "add", "-A")
        _armed(marker, "git add skipped fsmonitor")
        _plain(repo, "commit", "-qm", "arming")
        _armed(marker, "git commit skipped fsmonitor")
        _plain(repo, "reset", "-q", "--soft", "HEAD~1")
        marker.unlink(missing_ok=True)
        (repo / "src" / "b.py").write_text("changed too\n")

        r = await _commit(app, client, _LocalManager(repo))

        assert r.status_code == 200, r.text
        assert _last_message(repo) == "route commit"
        assert not marker.exists(), "the workspace fsmonitor ran"

    @pytest.mark.parametrize("files", [["*"], ["src/a.py"]], ids=["add-A", "add-paths"])
    async def test_clean_filter(self, app, client, repo, files):
        marker = _script(repo, "clean", "cat")
        (repo / ".git" / "info").mkdir(exist_ok=True)
        (repo / ".git" / "info" / "attributes").write_text("*.py filter=evil\n")
        _run(repo, "config", "filter.evil.clean", str(repo.parent / "clean.sh"))
        (repo / "src" / "a.py").write_text("arming\n")
        _plain(repo, "add", "src/a.py")
        _armed(marker, "git add skipped the clean filter")
        _plain(repo, "-c", "filter.evil.clean=", "reset", "-q")
        marker.unlink(missing_ok=True)
        (repo / "src" / "a.py").write_text("candidate\n")

        r = await _commit(app, client, _LocalManager(repo), files)

        assert r.status_code == 200, r.text
        assert not marker.exists(), "the workspace clean filter ran"
        assert _run(repo, "show", "HEAD:src/a.py") == "candidate"  # raw bytes committed

    async def test_hooks(self, app, client, repo):
        markers = []
        for hook in ("pre-commit", "commit-msg", "prepare-commit-msg", "post-commit"):
            marker = _script(repo, hook)
            path = repo / ".git" / "hooks" / hook
            path.write_text((repo.parent / f"{hook}.sh").read_text())
            path.chmod(0o755)
            markers.append(marker)
        _plain(repo, "commit", "-q", "--allow-empty", "-m", "arming")
        for marker in markers:
            _armed(marker, f"git commit skipped {marker.stem}")
        _run(repo, "reset", "-q", "--soft", "HEAD~1")
        (repo / "src" / "a.py").write_text("candidate\n")

        r = await _commit(app, client, _LocalManager(repo))

        assert r.status_code == 200, r.text
        assert _last_message(repo) == "route commit"
        assert [m.name for m in markers if m.exists()] == []

    async def test_lazy_fetch_through_a_planted_promisor(self, app, client, repo):
        """``commit`` prints a rename-detecting summary that reads the OLD blob."""
        (repo / "src" / "a.py").write_text("".join(f"line {i}\n" for i in range(50)))
        _run(repo, "commit", "-qam", "big")
        blob = _run(repo, "rev-parse", "HEAD:src/a.py")
        _run(repo, "mv", "src/a.py", "src/c.py")
        with (repo / "src" / "c.py").open("a") as fh:
            fh.write("line 50\n")
        _run(repo, "add", "-A")
        marker = _plant_promisor(repo, blob)
        # status (hardened already) must not need the blob, or the route 400s first.
        _run(repo, "config", "status.renames", "false")
        _plain(repo, "commit", "-m", "arming")  # not -q: the summary is the fetch
        _armed(marker, "git commit did not lazy-fetch")
        _run(repo, "reset", "-q", "--soft", "HEAD~1")

        mgr = _LocalManager(repo)

        await _commit(app, client, mgr)

        assert any(" commit " in c for c in mgr.commands)  # the commit really ran
        assert not marker.exists(), "the commit lazily fetched through the planted remote"


# --------------------------------------------------------------------------- branch


@pytest.fixture
def branchy(repo: Path) -> Path:
    """``main`` checked out; ``other`` one commit ahead with a different a.py."""
    _run(repo, "switch", "-q", "-c", "other")
    (repo / "src" / "a.py").write_text("def a():\n    return 2\n")
    _run(repo, "commit", "-qam", "other")
    _run(repo, "switch", "-q", "main")
    return repo


async def _branch(app, client, mgr, branch="other"):
    return await _post(app, client, mgr, {"branch": branch}, route="branch")


def _arm_switch(repo: Path, marker: Path, what: str) -> None:
    _plain(repo, "switch", "-q", "other")
    _armed(marker, f"git switch skipped {what}")
    _run(repo, "-c", "core.fsmonitor=false", "-c", "filter.evil.smudge=", "-c",
         "core.hooksPath=/dev/null", "switch", "-q", "main")  # fmt: skip
    assert _run(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    marker.unlink(missing_ok=True)


@pytest.mark.asyncio
class TestBranchRouteRunsNoRepoCommand:
    async def test_smudge_filter(self, app, client, branchy):
        repo = branchy
        marker = _script(repo, "smudge", "cat")
        (repo / ".git" / "info").mkdir(exist_ok=True)
        (repo / ".git" / "info" / "attributes").write_text("*.py filter=evil\n")
        _run(repo, "config", "filter.evil.smudge", str(repo.parent / "smudge.sh"))
        _arm_switch(repo, marker, "the smudge filter")

        r = await _branch(app, client, _LocalManager(repo))

        assert r.status_code == 200, r.text
        assert r.json()["branch"] == "other"
        assert not marker.exists(), "the workspace smudge filter ran"
        assert (repo / "src" / "a.py").read_text() == "def a():\n    return 2\n"

    async def test_fsmonitor(self, app, client, branchy):
        repo = branchy
        marker = _script(repo, "fsmonitor", "exit 1")
        _run(repo, "config", "core.fsmonitor", str(repo.parent / "fsmonitor.sh"))
        _arm_switch(repo, marker, "fsmonitor")

        r = await _branch(app, client, _LocalManager(repo))

        assert r.status_code == 200, r.text
        assert not marker.exists(), "the workspace fsmonitor ran"

    async def test_post_checkout_hook(self, app, client, branchy):
        repo = branchy
        marker = _script(repo, "post-checkout")
        hook = repo / ".git" / "hooks" / "post-checkout"
        hook.write_text((repo.parent / "post-checkout.sh").read_text())
        hook.chmod(0o755)
        _arm_switch(repo, marker, "post-checkout")

        r = await _branch(app, client, _LocalManager(repo))

        assert r.status_code == 200, r.text
        assert not marker.exists(), "the workspace post-checkout hook ran"

    async def test_lazy_fetch_through_a_planted_promisor_fails_closed(self, app, client, branchy):
        """A planted promisor remote is never fetched from, and the gap it leaves fails closed.

        git exits 0 here ("unable to read sha1 file", HEAD moved, the path left absent
        from the worktree), exactly as for any failed lazy fetch. The route must neither
        fetch through the planted remote nor report that half-switched tree as a 200.
        """
        repo = branchy
        marker = _plant_promisor(repo, _run(repo, "rev-parse", "other:src/a.py"))
        _plain(repo, "switch", "-q", "other")
        _armed(marker, "git switch did not lazy-fetch")
        # The failed fetch still moved HEAD; go back without fetching anything.
        subprocess.run(
            ["git", "switch", "-q", "--discard-changes", "main"],
            cwd=repo,
            check=True,
            capture_output=True,
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        )
        assert _run(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
        assert _run(repo, "status", "--porcelain") == ""
        mgr = _LocalManager(repo)

        r = await _branch(app, client, mgr)

        assert not marker.exists(), "the switch lazily fetched through the planted remote"
        assert any(" switch " in c for c in mgr.commands)  # the switch really ran
        # The switch left src/a.py absent: that is an incomplete switch, and it fails closed.
        assert r.status_code == 500, r.text
        details = r.json()["error"]["details"]
        assert details["error"] == "branch_switch_incomplete", r.text
        assert "src/a.py" in details["missing_paths"], r.text
        assert not (repo / "src" / "a.py").exists()

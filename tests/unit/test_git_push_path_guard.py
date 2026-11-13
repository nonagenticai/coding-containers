"""A push to a user's repository carries card deliverables only, never process data.

``push_workspace_to_fork`` used to push whatever the workspace committed, so a
harness that wrote its run artefacts (``.aa/`` state, logs, traces, probe dirs)
into the tree shipped them to the user's GitHub repo. ``src.core.push_path_guard``
is the single rule; the route lists every path the pushed commits touch and
refuses the whole push on any hit. Three layers are tested here:

* the rule itself (pure function, incl. a pin that fails if the denylist empties);
* the route: a refused push never resolves a credential, re-points origin or pushes;
* the REAL listing command against a scratch git repository, because a rule fed
  the wrong path list proves nothing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.routers.containers.git_push as gp
from src.core import push_path_guard as guard
from src.core.push_path_guard import (
    PATH_DENIED_PROCESS_DATA,
    PATH_INVALID,
    PATH_NOT_IN_TARGET_FILES,
    refused_paths,
)
from src.routers.containers import _common

# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def _codes(paths, target_files=None) -> dict[str, str]:
    return {r.path: r.error_code for r in refused_paths(paths, target_files)}


@pytest.mark.parametrize(
    "path",
    [
        ".aa/state.json",
        "x_probe/a.py",
        "run.log",
        "src/deep/worker.LOG",
        "pkg/aa_bugfix_probe/queries.py",
        ".claude/settings.json",
        "plans/next.md",
        "traces/span.json",
        "run-state/lease.json",
        "out/run.trace.json",
        # Case variants: matching is on the lowercased path (as in coding-github).
        ".AA/x",
        ".Claude/settings.json",
        "PLANS/next.md",
        "X_PROBE/a.py",
        "src/RUN.LOG",
        # .aa / .claude at any depth; bare root entries named like a denied dir.
        "pkg/.aa/state.json",
        "pkg/.claude/x",
        ".aa",
        ".claude",
        "plans",
        "Traces",
        "run-state",
    ],
)
def test_process_data_is_refused(path):
    assert _codes([path]) == {path: PATH_DENIED_PROCESS_DATA}


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../outside.py", "src/../../x.py", "./src/a.py", "a//b.py", "a\\b.py", ""],
)
def test_absolute_and_traversal_paths_are_refused(path):
    assert _codes([path]) == {path: PATH_INVALID}


def test_a_git_quoted_name_is_refused_not_guessed():
    assert _codes(['"we\\tird.log"']) == {'"we\\tird.log"': PATH_INVALID}


@pytest.mark.parametrize(
    "path",
    [
        "src/calc.py",
        "tests/test_calc.py",
        "src/plans/model.py",
        "docs/logging.md",
        "probe/a.py",
        "src/plans",
        "README.aa",
    ],
)
def test_deliverables_are_allowed(path):
    assert refused_paths([path]) == []


def test_outside_target_files_is_refused_when_given():
    got = _codes(["src/calc.py", "README.md", "tests/unit/t.py"], ["src/calc.py", "tests/"])
    assert got == {"README.md": PATH_NOT_IN_TARGET_FILES}


def test_target_files_absent_means_no_allowlist():
    assert refused_paths(["README.md"], None) == []


def test_empty_target_files_admits_nothing():
    assert _codes(["src/calc.py"], []) == {"src/calc.py": PATH_NOT_IN_TARGET_FILES}


def test_naming_process_data_as_a_target_does_not_admit_it():
    assert _codes(["run.log"], ["run.log"]) == {"run.log": PATH_DENIED_PROCESS_DATA}


def test_the_denylist_is_the_agreed_list_and_every_entry_bites():
    """Fails if the denylist (or any part of it) were emptied or narrowed.

    The list is coding-github's (``src/services/path_guard.py``) verbatim; this
    pins it, then proves each entry actually refuses a path, so an entry that is
    listed but matched by nothing turns this red too.
    """
    assert guard.PROCESS_DATA_DENYLIST == (
        ".aa/*",
        "*/.aa/*",
        ".claude/*",
        "*/.claude/*",
        "plans/*",
        "traces/*",
        "run-state/*",
        "*_probe/*",
        "*.log",
        "*.trace.json",
    )
    assert guard.PROCESS_DATA_ROOT_NAMES == {".aa", ".claude", "plans", "traces", "run-state"}
    samples = {
        ".aa/*": ".aa/state.json",
        "*/.aa/*": "pkg/.aa/state.json",
        ".claude/*": ".claude/x",
        "*/.claude/*": "pkg/.claude/x",
        "plans/*": "plans/p.md",
        "traces/*": "traces/t",
        "run-state/*": "run-state/s",
        "*_probe/*": "x_probe/a.py",
        "*.log": "run.log",
        "*.trace.json": "a.trace.json",
    }
    for entry, sample in samples.items():
        (refusal,) = refused_paths([sample])
        assert refusal.rule == entry, (entry, refusal)


@pytest.mark.parametrize(
    ("constant", "sample"),
    [
        ("PROCESS_DATA_DENYLIST", ".aa/state.json"),
        ("PROCESS_DATA_DENYLIST", "run.log"),
        ("PROCESS_DATA_ROOT_NAMES", "plans"),
    ],
)
def test_emptying_a_denylist_constant_lets_process_data_through(monkeypatch, constant, sample):
    """Arming control for the pin above: the rule really reads these constants."""
    assert refused_paths([sample]) != []
    monkeypatch.setattr(guard, constant, type(getattr(guard, constant))())
    assert refused_paths([sample]) == []


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


_TOKEN = "ghs_path_guard_test_token"  # noqa: S105  # pragma: allowlist secret


_APPROVED = "e" * 40


def _exec(listing: dict, tree: dict | None = None, tip_commit: dict | None = None):
    calls: list[str] = []

    async def _run(container_name, command, tenant_id=None, timeout=300, env=None):
        calls.append(command)
        if "rev-list --count" in command:
            return {"exit_code": 0, "stdout": "1\n"}
        if " cat-file commit " in command:
            return tip_commit or {
                "exit_code": 0,
                "stdout": f"tree {_APPROVED}\nparent {_BASE}\nauthor t\n\nmsg\n",
                "stderr": "",
            }
        if " log --no-renames " in command:
            return listing
        if " ls-tree -z --full-tree " in command and tree is not None:
            return tree
        if " push " in command and "GIT_CONFIG_NOSYSTEM" in command:
            return {"exit_code": 0, "stdout": "", "stderr": ""}
        if "config --get-regexp" in command:
            return {"exit_code": 1, "stdout": "", "stderr": ""}
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    return _run, calls


_TIP = "a" * 40
_BASE = "b" * 40


async def _push(
    app,
    client,
    monkeypatch,
    listing: dict,
    body: dict | None = None,
    base: str | BaseException | None = _BASE,
    tree: dict | None = None,
    tip_commit: dict | None = None,
):
    from coding_client.auth.tenant import get_current_user_with_tenant
    from src.models.container_models import ContainerStatus

    lookups: list[str] = []

    async def _default_branch(owner, repo, token):
        lookups.append(f"{owner}/{repo}")
        return gp._RepoInfo(default_branch="main", private=True)

    async def _base(owner, repo, branch, default_branch, token):
        lookups.append(f"base {owner}/{repo} {branch} {default_branch}")
        if isinstance(base, BaseException):
            raise base
        return base

    monkeypatch.setattr(gp, "_remote_repo_info", _default_branch)
    monkeypatch.setattr(gp, "_push_base_sha", _base)
    user = MagicMock()
    user.is_admin.return_value = True
    user.effective_tenant_id = "tenant-7"
    app.dependency_overrides[get_current_user_with_tenant] = lambda: user
    container = MagicMock()
    container.container_name = "ctr"
    container.status = ContainerStatus("running")
    container.tenant_id = "tenant-7"
    container.repo_url = "https://github.com/acme/widget"
    container.github_access_token = _TOKEN  # BYO: no coding-github call needed
    run, calls = _exec(listing, tree, tip_commit)
    manager = MagicMock()
    manager.get_container.return_value = container
    manager.execute_command = AsyncMock(side_effect=run)
    try:
        with patch("src.routers.containers.get_container_manager", return_value=manager):
            resp = await client.post(
                "/api/containers/ctr/git/push", json={"branch": "aa/fix-1", **(body or {})}
            )
    finally:
        app.dependency_overrides.clear()
    return resp, calls, lookups


def _pushed(calls: list[str]) -> bool:
    return any(" push " in c and "GIT_CONFIG_NOSYSTEM" in c for c in calls)


def _ok(*paths: str) -> dict:
    """What the listing prints: the resolved tip, then one path per line."""
    body = "".join(f"{p}\n" for p in paths)
    return {"exit_code": 0, "stdout": f"tip {_TIP}\n{body}", "stderr": ""}


@pytest.mark.parametrize("bad", [".aa/state.json", "x_probe/a.py", "run.log", ".AA/x"])
async def test_route_refuses_process_data_before_the_push(app, client, monkeypatch, bad):
    resp, calls, lookups = await _push(app, client, monkeypatch, _ok("src/calc.py", bad))
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["message"].startswith(f"{PATH_DENIED_PROCESS_DATA}:")
    assert bad in resp.json()["error"]["message"]
    assert not _pushed(calls)
    assert not any("git remote" in c for c in calls)
    # The base came from GitHub (default-branch then base lookup), and the listing
    # ranged from exactly that commit.
    assert lookups == ["acme/widget", "base acme/widget aa/fix-1 main"]
    (listing,) = [c for c in calls if " log --no-renames " in c]
    assert f"--not {_BASE}" in listing
    assert "--remotes" not in listing


async def test_route_pushes_the_checked_sha_not_the_branch_name(app, client, monkeypatch):
    resp, calls, _ = await _push(app, client, monkeypatch, _ok("src/calc.py"))
    assert resp.status_code == 200, resp.text
    (push,) = [c for c in calls if " push " in c and "GIT_CONFIG_NOSYSTEM" in c]
    assert f" {_TIP}:refs/heads/aa/fix-1" in push
    assert "refs/heads/aa/fix-1:refs/heads/aa/fix-1" not in push
    # The listing ran before origin was re-pointed, the push after both.
    listing = next(i for i, c in enumerate(calls) if " log --no-renames " in c)
    remove = next(i for i, c in enumerate(calls) if "remote remove origin" in c)
    assert listing < remove
    # origin/<branch> records the pushed SHA, not whatever the branch is now.
    (track,) = [c for c in calls if "update-ref" in c]
    assert f"refs/remotes/origin/aa/fix-1 {_TIP}" in track


async def test_unknown_base_is_a_409_not_a_full_history_listing(app, client, monkeypatch):
    resp, calls, _ = await _push(
        app, client, monkeypatch, {"exit_code": gp._BASE_UNKNOWN_EXIT, "stdout": "", "stderr": ""}
    )
    assert resp.status_code == 409, resp.text
    message = resp.json()["error"]["message"]
    assert message.startswith("push_base_unknown:")
    # Actionable: names the base commit and the fork, and says what to do when the
    # fork is ahead of the workspace or private.
    assert _BASE in message
    assert "https://github.com/acme/widget" in message
    assert "ahead" in message and "rebase" in message
    assert "private" in message and "credential" in message
    assert _TOKEN not in resp.text
    assert not _pushed(calls)
    assert not any("git remote" in c for c in calls)


async def test_unresolvable_base_refuses_before_any_listing(app, client, monkeypatch):
    resp, calls, _ = await _push(app, client, monkeypatch, _ok("src/calc.py"), base=None)
    assert resp.status_code == 502, resp.text
    message = resp.json()["error"]["message"]
    assert message.startswith("push_base_unresolved:")
    assert "could not be read" in message and "lookup failed" in message
    assert "empty repository" not in message
    assert not _pushed(calls)
    assert not any(" log --no-renames " in c for c in calls)


async def test_an_empty_repository_is_named_not_reported_as_unreadable(app, client, monkeypatch):
    resp, calls, _ = await _push(
        app, client, monkeypatch, _ok("src/calc.py"), base=gp._EmptyRepositoryError("acme/widget")
    )
    assert resp.status_code == 502, resp.text
    message = resp.json()["error"]["message"]
    assert message.startswith("push_base_unresolved:")
    assert "empty repository (no branches" in message
    assert "https://github.com/acme/widget" in message
    assert "could not be read" not in message
    assert not _pushed(calls)
    assert not any(" log --no-renames " in c for c in calls)


@pytest.mark.parametrize("stdout", ["src/calc.py\n", "tip HEAD\nsrc/calc.py\n", "tip abc\n", ""])
async def test_a_listing_without_a_resolved_tip_refuses(app, client, monkeypatch, stdout):
    resp, calls, _ = await _push(
        app, client, monkeypatch, {"exit_code": 0, "stdout": stdout, "stderr": ""}
    )
    assert resp.status_code == 409, resp.text
    assert not _pushed(calls)


async def test_route_refuses_outside_target_files_when_given(app, client, monkeypatch):
    resp, calls, _ = await _push(
        app,
        client,
        monkeypatch,
        _ok("src/calc.py", "README.md"),
        {"target_files": ["src/"], "target_blobs": {}, "approved_tree": _APPROVED},
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["message"].startswith(f"{PATH_NOT_IN_TARGET_FILES}:")
    assert not _pushed(calls)


_BLOB = "c" * 40


def _tree(*entries: tuple[str, str, str]) -> dict:
    """What ``ls-tree -z`` prints for ``(type, id, path)`` entries at the tip."""
    body = "".join(f"100644 {kind} {oid}\t{path}\0" for kind, oid, path in entries)
    return {"exit_code": 0, "stdout": body, "stderr": ""}


async def test_route_allows_inside_target_files(app, client, monkeypatch):
    resp, calls, _ = await _push(
        app,
        client,
        monkeypatch,
        _ok("src/calc.py"),
        {
            "target_files": ["src/calc.py"],
            "target_blobs": {"src/calc.py": _BLOB},
            "approved_tree": _APPROVED,
        },
        tree=_tree(("blob", _BLOB, "src/calc.py")),
    )
    assert resp.status_code == 200, resp.text
    assert _pushed(calls)
    # The content check read the tip the push sends, not the branch name.
    assert any(f"ls-tree -z --full-tree {_TIP} " in c for c in calls)


async def test_target_files_without_target_blobs_is_a_422(app, client, monkeypatch):
    resp, calls, _ = await _push(
        app, client, monkeypatch, _ok("src/calc.py"), {"target_files": ["src/calc.py"]}
    )
    assert resp.status_code == 422, resp.text
    assert "target_blobs is required" in resp.text
    assert not _pushed(calls)


async def test_route_refuses_content_that_differs_from_the_reviewed_blob(app, client, monkeypatch):
    resp, calls, _ = await _push(
        app,
        client,
        monkeypatch,
        _ok("src/calc.py"),
        {
            "target_files": ["src/calc.py"],
            "target_blobs": {"src/calc.py": _BLOB},
            "approved_tree": _APPROVED,
        },
        tree=_tree(("blob", "d" * 40, "src/calc.py")),
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["message"].startswith("push_content_mismatch:")
    assert not _pushed(calls)


async def test_route_refuses_a_pushed_path_with_no_reviewed_blob(app, client, monkeypatch):
    resp, calls, _ = await _push(
        app,
        client,
        monkeypatch,
        _ok("src/calc.py", "src/extra.py"),
        {
            "target_files": ["src/"],
            "target_blobs": {"src/calc.py": _BLOB},
            "approved_tree": _APPROVED,
        },
        tree=_tree(("blob", _BLOB, "src/calc.py"), ("blob", _BLOB, "src/extra.py")),
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["message"].startswith("push_content_unverified:")
    assert not _pushed(calls)


async def test_route_requires_a_reviewed_deletion_to_be_absent(app, client, monkeypatch):
    body = {
        "target_files": ["src/old.py"],
        "target_blobs": {"src/old.py": None},
        "approved_tree": _APPROVED,
    }
    ok, ok_calls, _ = await _push(app, client, monkeypatch, _ok("src/old.py"), body, tree=_tree())
    assert ok.status_code == 200, ok.text
    assert _pushed(ok_calls)
    bad, calls, _ = await _push(
        app, client, monkeypatch, _ok("src/old.py"), body, tree=_tree(("blob", _BLOB, "src/old.py"))
    )
    assert bad.status_code == 403, bad.text
    assert not _pushed(calls)


async def test_an_unreadable_tip_tree_refuses_the_push(app, client, monkeypatch):
    resp, calls, _ = await _push(
        app,
        client,
        monkeypatch,
        _ok("src/calc.py"),
        {
            "target_files": ["src/calc.py"],
            "target_blobs": {"src/calc.py": _BLOB},
            "approved_tree": _APPROVED,
        },
        tree={"exit_code": 128, "stdout": "", "stderr": "fatal"},
    )
    assert resp.status_code == 409, resp.text
    assert not _pushed(calls)


@pytest.mark.parametrize(
    "blobs",
    [{"src/": _BLOB}, {"/etc/passwd": _BLOB}, {"a/../b.py": _BLOB}, {"a.py": "HEAD"}],
)
async def test_unsafe_target_blobs_entry_is_a_422(app, client, monkeypatch, blobs):
    resp, calls, _ = await _push(
        app, client, monkeypatch, _ok(), {"target_blobs": blobs, "approved_tree": _APPROVED}
    )
    assert resp.status_code == 422, resp.text
    assert "target_blobs" in resp.text
    assert not _pushed(calls)


@pytest.mark.parametrize(
    "body",
    [
        {"target_files": ["src/calc.py"], "target_blobs": {"src/calc.py": _BLOB}},
        {"target_blobs": {"src/calc.py": _BLOB}},
    ],
)
async def test_a_content_check_without_approved_tree_is_a_422(app, client, monkeypatch, body):
    resp, calls, _ = await _push(app, client, monkeypatch, _ok("src/calc.py"), body)
    assert resp.status_code == 422, resp.text
    assert "approved_tree is required" in resp.text
    assert not _pushed(calls)


@pytest.mark.parametrize("bad", ["HEAD", "e" * 39, "E" * 40])
async def test_an_approved_tree_that_is_not_an_id_is_a_422(app, client, monkeypatch, bad):
    resp, calls, _ = await _push(
        app, client, monkeypatch, _ok("src/calc.py"), {"approved_tree": bad}
    )
    assert resp.status_code == 422, resp.text
    assert not _pushed(calls)


_ONE = {"target_files": ["src/calc.py"], "target_blobs": {"src/calc.py": _BLOB}}


@pytest.mark.parametrize(
    ("header", "status", "code"),
    [
        (f"tree {_APPROVED}\nparent {'f' * 40}\n", 403, "push_not_single_commit"),
        (f"tree {_APPROVED}\nparent {'f' * 40}\nparent {_BASE}\n", 403, "push_not_single_commit"),
        (f"tree {_APPROVED}\n", 403, "push_not_single_commit"),
        (f"tree {'d' * 40}\nparent {_BASE}\n", 403, "push_tree_mismatch"),
        ("garbage\n", 409, "push_tree_mismatch"),
    ],
)
async def test_route_refuses_anything_but_one_commit_of_the_approved_tree(
    app, client, monkeypatch, header, status, code
):
    resp, calls, _ = await _push(
        app,
        client,
        monkeypatch,
        _ok("src/calc.py"),
        {**_ONE, "approved_tree": _APPROVED},
        tree=_tree(("blob", _BLOB, "src/calc.py")),
        tip_commit={"exit_code": 0, "stdout": header + "\nmsg\n", "stderr": ""},
    )
    assert resp.status_code == status, resp.text
    assert resp.json()["error"]["message"].startswith(f"{code}:")
    assert not _pushed(calls)
    # The raw commit read names the tip the push would send, not the branch.
    assert any(f"cat-file commit {_TIP}" in c for c in calls)


async def test_an_unreadable_tip_commit_refuses_the_push(app, client, monkeypatch):
    resp, calls, _ = await _push(
        app,
        client,
        monkeypatch,
        _ok("src/calc.py"),
        {**_ONE, "approved_tree": _APPROVED},
        tree=_tree(("blob", _BLOB, "src/calc.py")),
        tip_commit={"exit_code": 128, "stdout": "", "stderr": "fatal"},
    )
    assert resp.status_code == 409, resp.text
    assert not _pushed(calls)


async def test_approved_tree_alone_checks_the_range_and_pushes(app, client, monkeypatch):
    resp, calls, _ = await _push(
        app, client, monkeypatch, _ok("src/calc.py"), {"approved_tree": _APPROVED}
    )
    assert resp.status_code == 200, resp.text
    assert _pushed(calls)
    check = next(i for i, c in enumerate(calls) if " cat-file commit " in c)
    push = next(i for i, c in enumerate(calls) if " push " in c and "GIT_CONFIG_NOSYSTEM" in c)
    assert check < push


async def test_a_failed_listing_refuses_the_push(app, client, monkeypatch):
    resp, calls, _ = await _push(
        app, client, monkeypatch, {"exit_code": 128, "stdout": "", "stderr": "bad ref"}
    )
    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["message"].startswith(f"{PATH_DENIED_PROCESS_DATA}:")
    assert not _pushed(calls)


@pytest.mark.parametrize("entry", ["/etc/passwd", "../x.py", "a/../b.py"])
async def test_unsafe_target_files_entry_is_a_422(app, client, monkeypatch, entry):
    resp, calls, _ = await _push(app, client, monkeypatch, _ok(), {"target_files": [entry]})
    assert resp.status_code == 422, resp.text
    assert not _pushed(calls)


# ---------------------------------------------------------------------------
# The real listing command
# ---------------------------------------------------------------------------

_GIT = shutil.which("git")


@pytest.fixture
def scratch_repo(tmp_path, monkeypatch):
    if _GIT is None:
        pytest.skip("needs a real git binary")
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(HOME=str(home), GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0")
    repo = tmp_path / "ws"
    subprocess.run([_GIT, "init", "-q", "-b", "main", str(repo)], check=True, env=env)
    monkeypatch.setattr(gp, "_WORKDIR", str(repo))

    def commit(files: dict[str, str | None], msg: str) -> None:
        for name, content in files.items():
            p = repo / name
            if content is None:
                subprocess.run([_GIT, "-C", str(repo), "rm", "-q", name], check=True, env=env)
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            subprocess.run([_GIT, "-C", str(repo), "add", name], check=True, env=env)
        ident = ["-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run([_GIT, "-C", str(repo), *ident, "commit", "-qm", msg], check=True, env=env)

    def git(*args: str) -> str:
        return subprocess.run(
            [_GIT, "-C", str(repo), *args], check=True, env=env, capture_output=True, text=True
        ).stdout.strip()

    def run_listing(branch: str, base: str, strip: str = "") -> subprocess.CompletedProcess:
        command = gp._pushed_paths_command(branch, base)
        if strip:  # mutation control: the same command with one hardening removed
            assert strip in command, (strip, command)
            command = command.replace(strip, "")
        return subprocess.run(
            ["sh", "-c", command],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    def listing(branch: str, base: str, strip: str = "") -> list[str]:
        r = run_listing(branch, base, strip)
        assert r.returncode == 0, r.stderr
        parsed = gp._parse_listing(r.stdout)
        assert parsed is not None, r.stdout
        tip, paths = parsed
        assert tip == git("rev-parse", f"refs/heads/{branch}")
        return sorted(set(paths))

    listing.raw = run_listing  # type: ignore[attr-defined]
    return commit, git, listing


def test_listing_covers_every_pushed_commit_not_just_the_net_diff(scratch_repo):
    commit, git, listing = scratch_repo
    commit({"base.py": "b", "plans/old.md": "upstream"}, "upstream base")
    base = git("rev-parse", "HEAD")  # what GitHub reports the remote has
    git("checkout", "-q", "-b", "aa/fix-1")
    commit({"src/calc.py": "1", "run.log": "noise"}, "work")
    commit({"run.log": None}, "tidy")  # deleted again — still travels in the pack

    got = listing("aa/fix-1", base)
    assert got == ["run.log", "src/calc.py"]  # base.py / plans/old.md are the remote's
    assert _codes(got) == {"run.log": PATH_DENIED_PROCESS_DATA}


def test_second_push_after_remote_remove_lists_only_the_new_commits(scratch_repo):
    """The route runs ``remote remove origin`` (deleting refs/remotes/origin/*) and
    never re-fetches. The range comes from GitHub's branch head, so the second
    push lists only what is new — a repo that legitimately holds ``plans/`` in
    its history is not refused for it again."""
    commit, git, listing = scratch_repo
    commit({"plans/roadmap.md": "the user's own file"}, "upstream")
    upstream = git("rev-parse", "HEAD")
    git("checkout", "-q", "-b", "aa/fix-1")
    commit({"src/a.py": "1"}, "first push")
    assert listing("aa/fix-1", upstream) == ["src/a.py"]
    first_push = git("rev-parse", "HEAD")  # now the remote branch head
    git("remote", "add", "origin", "https://github.com/acme/widget")
    git("update-ref", "refs/remotes/origin/aa/fix-1", first_push)
    git("remote", "remove", "origin")  # what the route does; no refs/remotes left
    commit({"src/b.py": "2"}, "second push")
    assert listing("aa/fix-1", first_push) == ["src/b.py"]


def test_a_forged_remote_tracking_ref_does_not_hide_a_denied_path(scratch_repo):
    commit, git, listing = scratch_repo
    commit({"base.py": "b"}, "upstream")
    base = git("rev-parse", "HEAD")
    git("checkout", "-q", "-b", "aa/fix-1")
    commit({".aa/state.json": "{}", "src/a.py": "1"}, "work")
    git("update-ref", "refs/remotes/x/y", "HEAD")  # forged "the remote has it"
    assert listing("aa/fix-1", base) == [".aa/state.json", "src/a.py"]


def test_replace_refs_do_not_hide_a_denied_path(scratch_repo, monkeypatch):
    commit, git, listing = scratch_repo
    commit({"base.py": "b"}, "upstream")
    base = git("rev-parse", "HEAD")
    git("checkout", "-q", "-b", "clean")
    commit({"src/a.py": "1"}, "clean")
    clean = git("rev-parse", "HEAD")
    git("checkout", "-q", "-b", "aa/fix-1", base)
    commit({".aa/state.json": "{}"}, "dirty")
    git("replace", "HEAD", clean)  # `git log` would now show the clean commit
    # Arming control: without GIT_NO_REPLACE_OBJECTS the replacement hides it.
    hidden = git("log", "--name-only", "--format=", "aa/fix-1", "--not", base)
    assert ".aa/state.json" not in hidden
    # Both layers on: the workspace is refused before anything is listed.
    r = listing.raw("aa/fix-1", base)
    assert r.returncode == _common.WORKSPACE_POISONED_EXIT, (r.returncode, r.stderr)
    assert "workspace_git_poisoned: replace refs present" in r.stderr
    assert r.stdout == ""
    # Refusal off: the env alone still lists the real commit.
    monkeypatch.setattr(_common, "POISON_CHECK_ENABLED", False)
    assert ".aa/state.json" in listing("aa/fix-1", base)
    # Both off: the listing is fooled (the attack is real).
    assert ".aa/state.json" not in listing("aa/fix-1", base, strip=" GIT_NO_REPLACE_OBJECTS=1")


def _hidden_then_tidied(scratch_repo) -> tuple[str, str, str]:
    """base -> dirty (adds .aa/state.json) -> tidy (deletes it). The net diff
    tidy..base is clean, so only a walk through ``dirty`` sees the denied path;
    a forged parent link tidy -> base hides it. Returns (base, dirty, tidy)."""
    commit, git, _ = scratch_repo
    commit({"base.py": "b"}, "upstream")
    base = git("rev-parse", "HEAD")
    git("checkout", "-q", "-b", "aa/fix-1")
    commit({".aa/state.json": "{}", "src/a.py": "1"}, "dirty")
    dirty = git("rev-parse", "HEAD")
    commit({".aa/state.json": None}, "tidy")
    return base, dirty, git("rev-parse", "HEAD")


def test_a_graft_file_does_not_hide_a_denied_path(scratch_repo, monkeypatch):
    commit, git, listing = scratch_repo
    base, _dirty, tidy = _hidden_then_tidied(scratch_repo)
    graft_file = os.path.join(gp._WORKDIR, ".git", "info", "grafts")
    os.makedirs(os.path.dirname(graft_file), exist_ok=True)
    with open(graft_file, "w") as fh:
        fh.write(f"{tidy} {base}\n")  # "tidy's parent is base": dirty vanishes
    # Arming control: plain `git log` honours info/grafts and loses the path.
    hidden = git("log", "--name-only", "--format=", "aa/fix-1", "--not", base)
    assert ".aa/state.json" not in hidden
    r = listing.raw("aa/fix-1", base)
    assert r.returncode == _common.WORKSPACE_POISONED_EXIT, (r.returncode, r.stderr)
    assert "workspace_git_poisoned: info/grafts present" in r.stderr
    # Refusal off: GIT_GRAFT_FILE=/dev/null alone still lists the real history.
    monkeypatch.setattr(_common, "POISON_CHECK_ENABLED", False)
    assert listing("aa/fix-1", base) == [".aa/state.json", "src/a.py"]
    # Mutation: without the graft neutralisation the listing is fooled too.
    assert listing("aa/fix-1", base, strip=" GIT_GRAFT_FILE=/dev/null") == ["src/a.py"]


def _forge_commit_graph_parent(path: str, child: str, parent: str) -> None:
    """Rewrite ``child``'s first parent in a commit-graph file to ``parent``.

    Format (gitformat-commit-graph): header ``CGPH``, version, hash version, chunk
    count; a chunk table of (id, offset); OIDF fan-out, OIDL sorted ids, CDAT
    records of (tree id, parent1 pos, parent2 pos, generation+date). git trusts
    CDAT's parent positions on read and does not verify the trailing checksum."""
    import struct

    os.chmod(path, 0o644)
    with open(path, "rb") as fh:
        data = bytearray(fh.read())
    assert data[:4] == b"CGPH", data[:4]
    hash_len = 20 if data[5] == 1 else 32
    chunks = {}
    for i in range(data[6] + 1):
        off = 8 + 12 * i
        chunks[bytes(data[off : off + 4])] = struct.unpack(">Q", data[off + 4 : off + 12])[0]
    fanout = chunks[b"OIDF"]
    (count,) = struct.unpack(">I", data[fanout + 255 * 4 : fanout + 256 * 4])
    lookup = chunks[b"OIDL"]
    ids = [bytes(data[lookup + i * hash_len : lookup + (i + 1) * hash_len]) for i in range(count)]
    record = chunks[b"CDAT"] + ids.index(bytes.fromhex(child)) * (hash_len + 16)
    struct.pack_into(">I", data, record + hash_len, ids.index(bytes.fromhex(parent)))
    with open(path, "wb") as fh:
        fh.write(data)


def test_a_forged_commit_graph_does_not_hide_a_denied_path(scratch_repo):
    commit, git, listing = scratch_repo
    base, _dirty, tidy = _hidden_then_tidied(scratch_repo)
    git("commit-graph", "write", "--reachable")
    graph = os.path.join(gp._WORKDIR, ".git", "objects", "info", "commit-graph")
    _forge_commit_graph_parent(graph, tidy, base)
    # Arming control: with the graph honoured (git's default), the path is hidden
    # even with replace refs and grafts already neutralised.
    hidden = subprocess.run(
        [_GIT, "-C", gp._WORKDIR, "log", "--name-only", "--format=", "aa/fix-1", "--not", base],
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1", "GIT_GRAFT_FILE": "/dev/null"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert ".aa/state.json" not in hidden, hidden
    assert listing("aa/fix-1", base) == [".aa/state.json", "src/a.py"]
    # Mutation: without core.commitGraph=false the listing is fooled too.
    assert listing("aa/fix-1", base, strip=" -c core.commitGraph=false") == ["src/a.py"]


def test_workspace_log_showroot_false_does_not_hide_a_root_commit(scratch_repo):
    commit, git, listing = scratch_repo
    commit({"base.py": "b"}, "upstream")
    base = git("rev-parse", "HEAD")
    git("checkout", "-q", "--orphan", "aa/fix-1")
    git("rm", "-q", "-r", "--cached", ".")
    commit({".aa/state.json": "{}"}, "new root")
    git("config", "log.showRoot", "false")
    # Arming control: the workspace's own config hides the root commit's files.
    assert git("log", "--name-only", "--format=", "aa/fix-1", "--not", base) == ""
    assert ".aa/state.json" in listing("aa/fix-1", base)


def test_case_variant_in_a_real_listing_is_refused(scratch_repo):
    commit, git, listing = scratch_repo
    commit({"base.py": "b"}, "upstream")
    base = git("rev-parse", "HEAD")
    git("checkout", "-q", "-b", "aa/fix-1")
    commit({".AA/x": "1"}, "work")
    got = listing("aa/fix-1", base)
    assert got == [".AA/x"]
    assert _codes(got) == {".AA/x": PATH_DENIED_PROCESS_DATA}


def test_unknown_base_exits_base_unknown_instead_of_listing_everything(scratch_repo):
    commit, git, listing = scratch_repo
    commit({"src/calc.py": "1"}, "one")
    base = git("rev-parse", "HEAD")
    assert listing.raw("main", base).returncode == 0  # control: a known base lists
    r = listing.raw("main", "c" * 40)
    assert r.returncode == gp._BASE_UNKNOWN_EXIT, (r.returncode, r.stdout, r.stderr)
    assert r.stdout == ""


def test_listing_of_a_missing_branch_fails_so_the_route_refuses(scratch_repo):
    commit, git, listing = scratch_repo
    commit({"src/calc.py": "1"}, "one")
    base = git("rev-parse", "HEAD")
    assert listing.raw("main", base).returncode == 0  # control: the same command works
    assert listing.raw("no-such-branch", base).returncode != 0


# ---------------------------------------------------------------------------
# The base comes from GitHub (read by coding-github), never from workspace refs
# ---------------------------------------------------------------------------
# This pod has no api.github.com egress: the heads arrive on coding-github's
# push-scope clone-token answer as ``branch_head`` / ``default_branch_head``.


def _facts(branch_head, default_branch_head=None, *, ok=True):
    from src.clients.github_client import CloneCredential

    return CloneCredential(
        ok=ok,
        token="t",
        visibility="private",
        private=True,
        default_branch="main",
        branch_head=branch_head,
        default_branch_head=default_branch_head,
    )


async def test_base_is_the_remote_branch_head_when_the_branch_exists():
    from src.clients.github_client import RefHead

    facts = _facts(RefHead("present", _BASE))
    assert await gp._push_base_sha("acme", "widget", "aa/fix-1", "main", facts) == _BASE


async def test_base_falls_back_to_the_default_branch_head_for_a_new_branch():
    from src.clients.github_client import RefHead

    facts = _facts(RefHead("absent"), RefHead("present", "d" * 40))
    assert await gp._push_base_sha("acme", "widget", "aa/fix-1", "main", facts) == "d" * 40


@pytest.mark.parametrize("which", ["branch", "default"])
async def test_an_empty_repository_is_reported_as_empty(which):
    from src.clients.github_client import RefHead

    if which == "branch":
        facts = _facts(RefHead("empty_repository"))
    else:
        facts = _facts(RefHead("absent"), RefHead("empty_repository"))
    with pytest.raises(gp._EmptyRepositoryError):
        await gp._push_base_sha("acme", "widget", "aa/fix-1", "main", facts)


@pytest.mark.parametrize(
    "kind", ["no-facts", "not-ok", "no-head", "unknown", "default-missing", "default-absent"]
)
async def test_an_unreadable_base_is_none(kind):
    from src.clients.github_client import RefHead

    facts = {
        "no-facts": None,
        "not-ok": _facts(RefHead("present", _BASE), ok=False),
        "no-head": _facts(None),
        "unknown": _facts(RefHead("unknown")),
        "default-missing": _facts(RefHead("absent"), None),
        "default-absent": _facts(RefHead("absent"), RefHead("absent")),
    }[kind]
    assert await gp._push_base_sha("acme", "widget", "aa/fix-1", "main", facts) is None


async def test_an_unknown_branch_head_is_not_absent_even_with_a_default_head():
    from src.clients.github_client import RefHead

    facts = _facts(RefHead("unknown"), RefHead("present", "d" * 40))
    assert await gp._push_base_sha("acme", "widget", "aa/fix-1", "main", facts) is None

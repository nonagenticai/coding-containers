"""The read-only git routes must not run commands named by the workspace repo's config.

``core.fsmonitor=<path>`` makes git run ``<path>`` whenever it reads the index
(``status``, ``ls-files``, a worktree ``diff``), and a ``diff=<driver>`` attribute
makes ``git diff`` run ``diff.<driver>.textconv`` / ``diff.<driver>.command``. The
workspace config is attacker-controllable, so GET git/status and GET git/diff must
override both. Each test first proves the planted command DOES fire under plain git
(the fixture is armed), then runs the route's real commands against the same repo and
asserts the marker was never written.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.unit.test_git_diff_route import (  # noqa: F401  (autouse fixture)
    _get,
    _isolate_from_the_enclosing_git,
    _LocalManager,
    _run,
)


def _hook(repo: Path, name: str) -> tuple[Path, Path]:
    """A script that writes a marker file when git runs it; returns (script, marker)."""
    marker = repo.parent / f"{name}.ran"
    script = repo.parent / f"{name}.sh"
    script.write_text(f'#!/bin/sh\ntouch "{marker}"\ncat >/dev/null 2>&1\nexit 1\n')
    script.chmod(0o755)
    return script, marker


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
    # A dirty tracked file and an untracked one, so every read has work to do.
    (r / "a.py").write_text("x = 2\n")
    (r / "new.txt").write_text("u\n")
    return r


@pytest.fixture
def fsmonitor_marker(repo: Path) -> Path:
    script, marker = _hook(repo, "fsmonitor")
    _run(repo, "config", "core.fsmonitor", str(script))
    # Arming control: plain git DOES run it for each read the routes perform.
    for args in (
        ["status", "--porcelain"],
        ["ls-files", "--others", "--exclude-standard"],
        ["diff", "HEAD"],
    ):
        marker.unlink(missing_ok=True)
        subprocess.run(["git", *args], cwd=repo, capture_output=True)
        assert marker.exists(), f"fixture not armed: git {' '.join(args)} skipped fsmonitor"
    marker.unlink()
    return marker


@pytest.fixture
def diff_driver_markers(repo: Path) -> tuple[Path, Path]:
    textconv, textconv_marker = _hook(repo, "textconv")
    external, external_marker = _hook(repo, "extdiff")
    (repo / ".gitattributes").write_text("*.py diff=evil\n")
    _run(repo, "config", "diff.evil.textconv", str(textconv))
    subprocess.run(["git", "diff", "HEAD"], cwd=repo, capture_output=True)
    assert textconv_marker.exists(), "fixture not armed: git diff skipped textconv"
    _run(repo, "config", "diff.evil.command", str(external))
    subprocess.run(["git", "diff", "HEAD"], cwd=repo, capture_output=True)
    assert external_marker.exists(), "fixture not armed: git diff skipped the external driver"
    textconv_marker.unlink()
    external_marker.unlink()
    return textconv_marker, external_marker


@pytest.mark.asyncio
class TestReadRoutesDoNotRunRepoCommands:
    async def test_status_does_not_run_fsmonitor(self, app, client, repo, fsmonitor_marker):
        mgr = _LocalManager(repo)
        r = await _get(app, client, mgr, "/api/containers/ctr/git/status")
        assert r.status_code == 200, r.text
        assert r.json()["new_files"] == ["new.txt"]  # git status really ran
        assert any("status" in c for c in mgr.commands)
        assert not fsmonitor_marker.exists()

    async def test_diff_does_not_run_fsmonitor(self, app, client, repo, fsmonitor_marker):
        mgr = _LocalManager(repo)
        r = await _get(app, client, mgr, "/api/workspaces/ctr/git/diff")
        assert r.status_code == 200, r.text
        d = r.json()
        assert [f["path"] for f in d["files"]] == ["a.py"]
        assert "new.txt" in d["untracked_files"]  # ls-files ran
        assert not fsmonitor_marker.exists()

    async def test_diff_does_not_run_textconv_or_external_driver(
        self, app, client, repo, diff_driver_markers
    ):
        textconv_marker, external_marker = diff_driver_markers
        mgr = _LocalManager(repo)
        r = await _get(app, client, mgr, "/api/workspaces/ctr/git/diff")
        assert r.status_code == 200, r.text
        assert "-x = 1" in r.json()["diff"]  # the real content, not a driver's output
        assert not textconv_marker.exists()
        assert not external_marker.exists()


def _stat_dirty(repo: Path) -> None:
    """Rewrite a.py so its stat info no longer matches the index: git must re-read it."""
    (repo / "a.py").write_text("x = 3\n")


def _arm_filter(repo: Path, kind: str, where: str) -> Path:
    """Plant ``filter.evil.<kind>`` and select it for *.py; return the marker path.

    ``where`` picks the attribute/config source: ``gitattributes`` (worktree file),
    ``info`` (``.git/info/attributes``) or ``include`` (filter defined in a file pulled
    in by ``include.path``). Proves plain git runs the filter during ``status``.
    """
    script, marker = _hook(repo, f"filter-{kind}-{where}")
    if kind == "process":
        # A long-running filter: marker on start, then fail the handshake.
        script.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 1\n')
    if where == "info":
        (repo / ".git" / "info").mkdir(exist_ok=True)
        (repo / ".git" / "info" / "attributes").write_text("*.py filter=evil\n")
    else:
        (repo / ".gitattributes").write_text("*.py filter=evil\n")
    if where == "include":
        inc = repo.parent / "evil.cfg"
        inc.write_text(f'[filter "evil"]\n\t{kind} = {script}\n')
        _run(repo, "config", "include.path", str(inc))
    else:
        _run(repo, "config", f"filter.evil.{kind}", str(script))
    _stat_dirty(repo)
    subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True)
    assert marker.exists(), f"fixture not armed: git status skipped the {kind} filter"
    marker.unlink()
    _stat_dirty(repo)
    return marker


_FILTER_CASES = [
    ("clean", "gitattributes"),
    ("process", "gitattributes"),
    ("clean", "info"),
    ("clean", "include"),
]


@pytest.mark.asyncio
class TestReadRoutesDoNotRunFilters:
    @pytest.mark.parametrize(("kind", "where"), _FILTER_CASES)
    async def test_status_does_not_run_filter(self, app, client, repo, kind, where):
        marker = _arm_filter(repo, kind, where)
        mgr = _LocalManager(repo)
        r = await _get(app, client, mgr, "/api/containers/ctr/git/status")
        assert not marker.exists(), "the workspace filter ran"
        assert r.status_code == 200, r.text
        assert "a.py" in r.json()["modified_files"]  # git status really ran
        assert not marker.exists()

    @pytest.mark.parametrize(("kind", "where"), _FILTER_CASES)
    async def test_diff_does_not_run_filter(self, app, client, repo, kind, where):
        marker = _arm_filter(repo, kind, where)
        mgr = _LocalManager(repo)
        r = await _get(app, client, mgr, "/api/workspaces/ctr/git/diff")
        assert not marker.exists(), "the workspace filter ran"
        assert r.status_code == 200, r.text
        assert "+x = 3" in r.json()["diff"]  # real bytes, not a filter's output
        assert not marker.exists()

    async def test_unexpressible_filter_key_is_refused(self, app, client, repo):
        # A subsection containing '=' cannot be blanked with `-c <key>=`: refuse the read.
        script, marker = _hook(repo, "filter-eq")
        (repo / ".gitattributes").write_text("*.py filter=a=b\n")
        _run(repo, "config", "filter.a=b.clean", str(script))
        _stat_dirty(repo)
        mgr = _LocalManager(repo)
        r = await _get(app, client, mgr, "/api/containers/ctr/git/status")
        assert r.status_code == 500, r.text
        assert "refusing git read" in r.text
        assert not marker.exists()


# --- git-lfs: canonical filters are kept, everything else under filter.lfs is blanked ---


@pytest.fixture
def no_ambient_git_config(monkeypatch, tmp_path: Path) -> None:
    """Only the scratch repo's config counts (a developer's global lfs config would not)."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "empty-global.cfg"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def _read_overrides(repo: Path) -> list[str]:
    """Run the real builder's script against ``repo`` and return the ``-c`` values it
    would pass to git (``exec git`` swapped for printing its argv). Runs under dash
    when present: the workspace image's /bin/sh."""
    from src.routers.containers._common import _WORKSPACE_ROOT, workspace_git_read

    cmd = workspace_git_read(["status", "--porcelain"])
    prefix = f"cd {_WORKSPACE_ROOT} && "
    assert cmd.startswith(prefix)
    body = cmd[len(prefix) :]
    assert body.count('exec git "$@"') == 1
    body = body.replace('exec git "$@"', "printf '%s\\n' \"$@\"")
    sh = "dash" if shutil.which("dash") else "sh"
    proc = subprocess.run([sh, "-c", body], cwd=repo, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    argv = proc.stdout.splitlines()
    return [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "-c"]


def _set_canonical_lfs(repo: Path) -> None:
    from src.routers.containers._common import CANONICAL_LFS_FILTERS

    _run(repo, "config", "filter.lfs.required", "true")
    for key, value in CANONICAL_LFS_FILTERS.items():
        _run(repo, "config", key, value)


class TestLfsFilterArgs:
    def test_canonical_lfs_values_are_not_blanked(self, repo, no_ambient_git_config):
        _set_canonical_lfs(repo)
        _run(repo, "config", "filter.evil.clean", "/bin/false")
        overrides = _read_overrides(repo)
        assert not [o for o in overrides if o.startswith("filter.lfs.")], overrides
        assert "filter.evil.clean=" in overrides  # the builder still blanks the rest

    @pytest.mark.parametrize(
        "value",
        [
            "/usr/bin/git-lfs clean -- %f",  # absolute path: not canonical, blanked
            "git-lfs clean -- %f; touch /tmp/x",  # canonical prefix only
            "git-lfs clean -- %f ",  # trailing space
            "",
        ],
    )
    def test_non_canonical_lfs_clean_is_blanked(self, repo, no_ambient_git_config, value):
        _set_canonical_lfs(repo)
        _run(repo, "config", "filter.lfs.clean", value)
        overrides = _read_overrides(repo)
        assert "filter.lfs.clean=" in overrides
        assert "filter.lfs.smudge=" not in overrides  # still canonical

    def test_last_value_decides_for_a_multivalued_lfs_key(self, repo, no_ambient_git_config):
        _set_canonical_lfs(repo)
        _run(repo, "config", "--add", "filter.lfs.process", "/tmp/evil")
        assert "filter.lfs.process=" in _read_overrides(repo)

    def test_differently_cased_subsection_is_not_canonical(self, repo, no_ambient_git_config):
        _run(repo, "config", "filter.LFS.clean", "git-lfs clean -- %f")
        assert "filter.LFS.clean=" in _read_overrides(repo)


@pytest.mark.asyncio
class TestLfsReads:
    async def test_status_on_a_real_lfs_repo_does_not_500(
        self, app, client, repo, no_ambient_git_config
    ):
        if not shutil.which("git-lfs"):
            pytest.skip("git-lfs not installed locally; covered by TestLfsFilterArgs")
        _run(repo, "lfs", "install", "--local")
        _run(repo, "lfs", "track", "*.bin")
        (repo / "f.bin").write_bytes(b"\x00binary\x01" * 100)
        _run(repo, "add", ".gitattributes", "f.bin")
        _run(repo, "commit", "-q", "-m", "lfs")
        os.utime(repo / "f.bin", (0, 0))  # stat-dirty: git must re-clean it
        # Arming control: with the lfs filter blanked (the previous behaviour) the
        # same read fails, so a 200 below is the fix and not an idle fixture.
        blanked = subprocess.run(
            ["git", "-c", "filter.lfs.clean=", "-c", "filter.lfs.process=", "status"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        assert blanked.returncode != 0 and "clean filter 'lfs' failed" in blanked.stderr
        os.utime(repo / "f.bin", (1, 1))
        mgr = _LocalManager(repo)
        r = await _get(app, client, mgr, "/api/containers/ctr/git/status")
        assert r.status_code == 200, r.text
        assert "a.py" in r.json()["modified_files"]
        assert "f.bin" not in r.json()["modified_files"]  # content unchanged

    async def test_planted_lfs_clean_is_neutralised(self, app, client, repo, no_ambient_git_config):
        script, marker = _hook(repo, "filter-lfs-clean")
        (repo / ".gitattributes").write_text("*.py filter=lfs\n")
        _run(repo, "config", "filter.lfs.clean", str(script))
        _stat_dirty(repo)
        subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True)
        assert marker.exists(), "fixture not armed: git status skipped filter.lfs.clean"
        marker.unlink()
        _stat_dirty(repo)
        mgr = _LocalManager(repo)
        r = await _get(app, client, mgr, "/api/containers/ctr/git/status")
        assert not marker.exists(), "the planted lfs filter ran"
        assert r.status_code == 200, r.text
        assert "a.py" in r.json()["modified_files"]


# --- submodules: their own config is never enumerated, so the read must not recurse ---


def _git_file(repo: Path, *args: str) -> None:
    """``git`` with the file transport allowed (submodule add/update of a local path)."""
    _run(repo, "-c", "protocol.file.allow=always", *args)


@pytest.fixture
def submodule_repo(repo: Path, tmp_path: Path) -> Path:
    """``repo`` with a submodule ``sub`` whose checked-out commit is NEWER than the
    recorded pointer AND whose worktree is dirty; ``submodule.sub.ignore=none``."""
    sub = tmp_path / "subsrc"
    sub.mkdir()
    _run(sub, "init", "-q", "-b", "main")
    _run(sub, "config", "user.email", "t@example.com")
    _run(sub, "config", "user.name", "t")
    (sub / "s.py").write_text("s = 1\n")
    _run(sub, "add", "-A")
    _run(sub, "commit", "-q", "-m", "s1")
    _git_file(repo, "submodule", "add", "-q", str(sub), "sub")
    _run(repo, "commit", "-q", "-m", "add sub")
    _run(repo, "config", "submodule.sub.ignore", "none")
    inner = repo / "sub"
    _run(inner, "config", "user.email", "t@example.com")
    _run(inner, "config", "user.name", "t")
    (inner / "s.py").write_text("s = 2\n")
    _run(inner, "commit", "-q", "-am", "s2")  # new commit: pointer moved
    return repo


def _stat_dirty_sub(repo: Path) -> None:
    (repo / "sub" / "s.py").write_text("s = 3\n")


@pytest.fixture
def submodule_filter_marker(submodule_repo: Path) -> Path:
    """A clean filter planted in the SUBMODULE's own config and attributes."""
    inner = submodule_repo / "sub"
    script, marker = _hook(submodule_repo, "sub-filter-clean")
    info = Path(_run(inner, "rev-parse", "--absolute-git-dir")) / "info"
    info.mkdir(exist_ok=True)
    (info / "attributes").write_text("*.py filter=evil\n")
    _run(inner, "config", "filter.evil.clean", str(script))
    # Arming control: plain git (fsmonitor already off, as the helper does) recurses
    # into the submodule and runs its filter for both reads the routes perform.
    for args in (["status", "--porcelain"], ["diff", "--no-ext-diff", "--no-textconv", "HEAD"]):
        marker.unlink(missing_ok=True)
        _stat_dirty_sub(submodule_repo)
        subprocess.run(
            ["git", "-c", "core.fsmonitor=false", *args], cwd=submodule_repo, capture_output=True
        )
        assert marker.exists(), f"fixture not armed: git {' '.join(args)} skipped the filter"
    marker.unlink()
    _stat_dirty_sub(submodule_repo)
    return marker


@pytest.fixture
def submodule_diff_driver_marker(submodule_repo: Path) -> Path:
    """``diff.submodule=diff`` in the superproject + ``diff.evil.command`` in the
    submodule: a diff over the moved pointer runs a nested diff inside the submodule
    that honours the driver despite ``--no-ext-diff``."""
    inner = submodule_repo / "sub"
    script, marker = _hook(submodule_repo, "sub-extdiff")
    (inner / ".gitattributes").write_text("*.py diff=evil\n")
    (inner / "s.py").write_text("s = 4\n")
    _run(inner, "add", "-A")
    _run(inner, "commit", "-q", "-m", "s4")
    _run(inner, "config", "diff.evil.command", str(script))
    _run(submodule_repo, "config", "diff.submodule", "diff")
    _run(submodule_repo, "add", "sub")
    _run(submodule_repo, "commit", "-q", "-m", "bump sub")
    subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD~1",
            "HEAD",
        ],
        cwd=submodule_repo,
        capture_output=True,
    )
    assert marker.exists(), "fixture not armed: nested submodule diff skipped the driver"
    marker.unlink()
    return marker


@pytest.mark.asyncio
class TestReadRoutesDoNotEnterSubmodules:
    """Defence in depth BELOW the submodule refusal (submodules are unsupported: the
    shared prefix refuses them 409 first, ``test_the_refusal_comes_first``). These run
    with the refusal off, proving the read flags alone never enter a submodule."""

    @pytest.fixture(autouse=True)
    def _refusal_off(self, monkeypatch):
        from src.routers.containers import _common

        monkeypatch.setattr(_common, "SUBMODULE_CHECK_ENABLED", False)

    async def test_the_refusal_comes_first(
        self, app, client, submodule_repo, submodule_filter_marker, monkeypatch
    ):
        from src.routers.containers import _common

        monkeypatch.setattr(_common, "SUBMODULE_CHECK_ENABLED", True)
        for url in ("/api/containers/ctr/git/status", "/api/workspaces/ctr/git/diff"):
            r = await _get(app, client, _LocalManager(submodule_repo), url)
            assert r.status_code == 409, r.text
            assert _common.WORKSPACE_SUBMODULES in r.text
        assert not submodule_filter_marker.exists(), "the submodule's filter ran"

    async def test_status_does_not_run_submodule_filter(
        self, app, client, submodule_repo, submodule_filter_marker
    ):
        mgr = _LocalManager(submodule_repo)
        r = await _get(app, client, mgr, "/api/containers/ctr/git/status")
        assert not submodule_filter_marker.exists(), "the submodule's filter ran"
        assert r.status_code == 200, r.text
        assert "a.py" in r.json()["modified_files"]  # git status really ran

    async def test_diff_does_not_run_submodule_filter(
        self, app, client, submodule_repo, submodule_filter_marker
    ):
        mgr = _LocalManager(submodule_repo)
        r = await _get(app, client, mgr, "/api/workspaces/ctr/git/diff")
        assert not submodule_filter_marker.exists(), "the submodule's filter ran"
        assert r.status_code == 200, r.text
        paths = [f["path"] for f in r.json()["files"]]
        assert "a.py" in paths and "sub" in paths  # the moved pointer is still reported

    @pytest.mark.parametrize(
        "url",
        [
            "/api/workspaces/ctr/git/diff?base=HEAD~1&head=HEAD",
            "/api/workspaces/ctr/git/diff?base=HEAD~1",
        ],
    )
    async def test_diff_does_not_run_submodule_diff_driver(
        self, app, client, submodule_repo, submodule_diff_driver_marker, url
    ):
        mgr = _LocalManager(submodule_repo)
        r = await _get(app, client, mgr, url)
        assert not submodule_diff_driver_marker.exists(), "the submodule's diff driver ran"
        assert r.status_code == 200, r.text
        assert "Subproject commit" in r.json()["diff"]


# --- partial clone: a missing object must not be lazily fetched through the remote ---


@pytest.mark.asyncio
async def test_diff_does_not_lazy_fetch_through_core_sshcommand(app, client, repo):
    script, marker = _hook(repo, "ssh")
    script.write_text(
        f'#!/bin/sh\ntouch "{marker}"\nexit 1\n'
    )  # no stdin read: fetch holds it open
    _run(repo, "commit", "-q", "-am", "two")
    missing = _run(repo, "rev-parse", "HEAD~1:a.py")
    (repo / ".git" / "objects" / missing[:2] / missing[2:]).unlink()
    _run(repo, "config", "extensions.partialClone", "origin")
    _run(repo, "config", "remote.origin.url", "ssh://lazy-fetch.invalid/x")
    _run(repo, "config", "remote.origin.promisor", "true")
    _run(repo, "config", "core.sshCommand", str(script))
    # Arming control: plain git lazily fetches the missing blob through the ssh command.
    subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "diff", "--no-ext-diff", "HEAD~1", "HEAD"],
        cwd=repo,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    assert marker.exists(), "fixture not armed: git diff did not lazy-fetch"
    marker.unlink()
    mgr = _LocalManager(repo)
    r = await _get(app, client, mgr, "/api/workspaces/ctr/git/diff?base=HEAD~1&head=HEAD")
    assert not marker.exists(), "the read lazily fetched through core.sshCommand"
    assert any("diff" in c for c in mgr.commands)  # the diff really ran
    assert r.status_code == 500, r.text  # missing object: fails closed, no fetch


class TestSubcommandFlagLookup:
    """``_SUBCOMMAND_FLAGS`` keys on the SUBCOMMAND, not on ``args[0]``: restore's status
    starts with ``-c core.quotePath=false --literal-pathspecs``."""

    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            (["status"], 0),
            (["-c", "core.quotePath=false", "--literal-pathspecs", "status"], 3),
            (["-C", "sub", "-c", "a=b", "diff", "HEAD"], 4),
            (["--git-dir", ".git", "--work-tree=.", "diff"], 3),
            (["-c", "status", "rev-parse"], 2),  # "status" here is -c's VALUE
            (["--literal-pathspecs"], None),
        ],
    )
    def test_subcommand_index(self, args, expected):
        from src.routers.containers._common import _subcommand_index

        assert _subcommand_index(args) == expected

    def test_leading_c_pair_still_gets_status_flags(self):
        from src.routers.containers._common import workspace_git_read

        cmd = workspace_git_read(
            ["-c", "core.quotePath=false", "--literal-pathspecs", "status", "--porcelain"]
        )
        assert (
            "-c core.quotePath=false --literal-pathspecs status --ignore-submodules=dirty "
            "--porcelain" in cmd
        )

    def test_leading_options_still_get_diff_flags(self):
        from src.routers.containers._common import workspace_git_read

        cmd = workspace_git_read(["-C", "x", "diff", "HEAD"])
        assert "-C x diff --ignore-submodules=dirty --submodule=short HEAD" in cmd

    def test_option_value_named_like_a_subcommand_is_not_one(self):
        from src.routers.containers._common import workspace_git_read

        cmd = workspace_git_read(["-c", "status", "rev-parse", "HEAD"])
        assert "--ignore-submodules" not in cmd

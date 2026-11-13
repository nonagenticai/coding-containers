"""GET /api/workspaces/{name}/files (confined read + line span) and /search.

The fake container manager runs each generated command with ``sh -c`` against a
REAL directory tree in tmp_path, whose path is registered as the container's
``codebase_path``. The confinement lives partly in shell (``realpath`` plus a
``case`` containment check), and only a real shell with real symlinks can show
whether it holds. A mock returning canned stdout would pass for any command.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.routers.containers import files_io

# The read path trusts the kernel's record of the OPENED fd (/proc/self/fd/3). Without
# /proc (macOS) the route refuses by design (43); these real-shell tests need Linux,
# which is what workspaces and CI run. They are skipped, loudly, anywhere else.
pytestmark = pytest.mark.skipif(
    not os.path.isdir("/proc/self/fd"), reason="needs Linux /proc/self/fd (CI runs it)"
)

OUTSIDE_MARKER = "MARKER-FROM-OUTSIDE-THE-WORKSPACE"


@pytest.fixture(autouse=True)
def _isolate_from_the_enclosing_git(monkeypatch):
    # Same reason as test_git_diff_route.py: a hook's GIT_DIR must not leak in.
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def tree(tmp_path: Path) -> dict[str, Path]:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text(f"{OUTSIDE_MARKER}\nneedle outside\n")

    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "a.py").write_text("line1\nline2 needle\nline3\nline4 NEEDLE\nline5\n")
    (ws / "src" / "dots.txt").write_text("a.b\naxb\n")
    (ws / ".git").mkdir()
    (ws / ".git" / "config").write_text("needle in git dir\n")
    (ws / "link_out").symlink_to(outside / "secret.txt")
    (ws / "dir_out").symlink_to(outside)
    (ws / "link_in").symlink_to(ws / "src" / "a.py")
    return {"ws": ws, "outside": outside}


class _LocalManager:
    def __init__(self, root: Path):
        self.root = root
        self.commands: list[str] = []
        self.tenants: list[str | None] = []

    def get_container(self, name, tenant_id=None):
        info = MagicMock()
        info.codebase_path = str(self.root)
        return info

    async def execute_command(self, container_name, command, tenant_id=None, timeout=300):
        self.commands.append(command)
        self.tenants.append(tenant_id)
        proc = subprocess.run(
            ["sh", "-c", command],
            capture_output=True,
            text=True,
            errors="replace",
            stdin=subprocess.DEVNULL,
        )
        return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


async def _get(app, client, mgr, url, params=None):
    from coding_client.auth.tenant import require_tenant

    app.dependency_overrides[require_tenant] = lambda: "tenant-a"
    try:
        with patch("src.routers.containers.get_container_manager", return_value=mgr):
            return await client.get(url, params=params)
    finally:
        app.dependency_overrides.clear()


FILES = "/api/workspaces/ctr/files"
SEARCH = "/api/workspaces/ctr/search"


# ---------------------------------------------------------------------------
# Confinement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFilesConfinement:
    @pytest.mark.parametrize(
        "raw_query",
        [
            "path=../outside/secret.txt",
            "path=src/../../outside/secret.txt",
            "path=/workspace/../etc/passwd",
            "path=/etc/passwd",  # absolute, not under the /workspace alias
            "path=%2e%2e/outside/secret.txt",  # encoded once
            "path=%252e%252e/outside/secret.txt",  # encoded twice
            "path=src%2F..%2F..%2Foutside",  # encoded slashes
            "path=..%5Coutside",  # backslash
        ],
    )
    async def test_lexical_refusal_is_400_and_runs_nothing(self, app, client, tree, raw_query):
        mgr = _LocalManager(tree["ws"])
        r = await _get(app, client, mgr, f"{FILES}?{raw_query}")
        assert r.status_code == 400, (raw_query, r.text)
        assert OUTSIDE_MARKER not in r.text
        assert mgr.commands == []

    @pytest.mark.parametrize("path", ["link_out", "dir_out/secret.txt", "dir_out"])
    async def test_symlink_escape_is_400(self, app, client, tree, path):
        mgr = _LocalManager(tree["ws"])
        r = await _get(app, client, mgr, FILES, {"path": path})
        assert r.status_code == 400, r.text
        assert "outside the repository workspace" in r.text
        assert OUTSIDE_MARKER not in r.text

    async def test_symlink_inside_is_followed_and_reported(self, app, client, tree):
        r = await _get(app, client, _LocalManager(tree["ws"]), FILES, {"path": "link_in"})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["resolved_path"] == "/workspace/src/a.py"
        assert d["content"].startswith("line1\n")

    async def test_missing_path_is_404(self, app, client, tree):
        r = await _get(app, client, _LocalManager(tree["ws"]), FILES, {"path": "src/ghost.py"})
        assert r.status_code == 404

    async def test_workspace_alias_and_relative_forms_agree(self, app, client, tree):
        mgr = _LocalManager(tree["ws"])
        a = (await _get(app, client, mgr, FILES, {"path": "/workspace/src/a.py"})).json()
        b = (await _get(app, client, mgr, FILES, {"path": "src/a.py"})).json()
        assert a["content"] == b["content"] and a["sha256"] == b["sha256"]

    async def test_directory_listing_of_root(self, app, client, tree):
        r = await _get(app, client, _LocalManager(tree["ws"]), FILES)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["type"] == "directory" and d["resolved_path"] == "/workspace"
        names = {e["name"].split(" -> ")[0] for e in d["entries"]}
        assert {"src", "link_out", ".git"} <= names
        assert "." not in names and ".." not in names


# ---------------------------------------------------------------------------
# readSpan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReadSpan:
    async def test_whole_file_carries_total_and_hash(self, app, client, tree):
        raw = (tree["ws"] / "src" / "a.py").read_text()
        r = await _get(app, client, _LocalManager(tree["ws"]), FILES, {"path": "src/a.py"})
        d = r.json()
        assert d["content"] == raw
        assert d["lines_total"] == 5
        assert d["sha256"] == hashlib.sha256(raw.encode()).hexdigest()
        assert d["start_line"] is None and d["end_line"] is None

    async def test_span_is_one_based_inclusive_and_hash_is_whole_file(self, app, client, tree):
        raw = (tree["ws"] / "src" / "a.py").read_text()
        r = await _get(
            app,
            client,
            _LocalManager(tree["ws"]),
            FILES,
            {"path": "src/a.py", "start": 2, "end": 3},
        )
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["content"] == "line2 needle\nline3\n"
        assert (d["start_line"], d["end_line"], d["lines_total"]) == (2, 3, 5)
        assert d["sha256"] == hashlib.sha256(raw.encode()).hexdigest()

    async def test_end_past_eof_is_clamped(self, app, client, tree):
        d = (
            await _get(
                app,
                client,
                _LocalManager(tree["ws"]),
                FILES,
                {"path": "src/a.py", "start": 4, "end": 99},
            )
        ).json()
        assert d["content"] == "line4 NEEDLE\nline5\n" and d["end_line"] == 5

    async def test_single_bound_forms(self, app, client, tree):
        mgr = _LocalManager(tree["ws"])
        only_start = (await _get(app, client, mgr, FILES, {"path": "src/a.py", "start": 5})).json()
        assert only_start["content"] == "line5\n" and only_start["end_line"] == 5
        only_end = (await _get(app, client, mgr, FILES, {"path": "src/a.py", "end": 1})).json()
        assert only_end["content"] == "line1\n" and only_end["start_line"] == 1

    @pytest.mark.parametrize(
        "params",
        [
            {"start": 6},  # beyond EOF
            {"start": 3, "end": 2},  # inverted
            {"start": 1, "end": files_io.MAX_SPAN_LINES + 1},  # over the cap
        ],
    )
    async def test_bad_span_is_400(self, app, client, tree, params):
        r = await _get(
            app, client, _LocalManager(tree["ws"]), FILES, {"path": "src/a.py", **params}
        )
        assert r.status_code == 400, r.text

    async def test_zero_start_is_422(self, app, client, tree):
        r = await _get(
            app, client, _LocalManager(tree["ws"]), FILES, {"path": "src/a.py", "start": 0}
        )
        assert r.status_code == 422

    async def test_span_on_directory_is_400(self, app, client, tree):
        r = await _get(
            app, client, _LocalManager(tree["ws"]), FILES, {"path": "src", "start": 1, "end": 2}
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSearch:
    async def test_literal_hits_with_file_and_line(self, app, client, tree):
        r = await _get(app, client, _LocalManager(tree["ws"]), SEARCH, {"pattern": "needle"})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["hits"] == [{"file": "src/a.py", "line": 2, "text": "line2 needle", "repo": None}]
        assert d["truncated"] is False and d["path"] == "/workspace"

    async def test_git_dir_and_escaping_symlinks_are_not_searched(self, app, client, tree):
        d = (
            await _get(app, client, _LocalManager(tree["ws"]), SEARCH, {"pattern": "needle"})
        ).json()
        files = {h["file"] for h in d["hits"]}
        assert not any(f.startswith(".git") for f in files)
        assert all("outside" not in h["text"] for h in d["hits"])

    async def test_literal_is_not_regex(self, app, client, tree):
        mgr = _LocalManager(tree["ws"])
        lit = (await _get(app, client, mgr, SEARCH, {"pattern": "a.b", "path": "src"})).json()
        assert [h["text"] for h in lit["hits"]] == ["a.b"]
        rx = (
            await _get(app, client, mgr, SEARCH, {"pattern": "a.b", "path": "src", "regex": True})
        ).json()
        assert sorted(h["text"] for h in rx["hits"]) == ["a.b", "axb"]

    async def test_case_insensitive(self, app, client, tree):
        d = (
            await _get(
                app,
                client,
                _LocalManager(tree["ws"]),
                SEARCH,
                {"pattern": "needle", "case_sensitive": False, "path": "src/a.py"},
            )
        ).json()
        assert [h["line"] for h in d["hits"]] == [2, 4]
        assert d["path"] == "/workspace/src/a.py"

    async def test_cap_reports_truncation(self, app, client, tree):
        d = (
            await _get(
                app,
                client,
                _LocalManager(tree["ws"]),
                SEARCH,
                {"pattern": "line", "path": "src/a.py", "max_results": 2},
            )
        ).json()
        assert d["total_returned"] == 2 and d["truncated"] is True

    async def test_no_match_is_empty_200(self, app, client, tree):
        r = await _get(app, client, _LocalManager(tree["ws"]), SEARCH, {"pattern": "zzz-none"})
        assert r.status_code == 200 and r.json()["hits"] == []

    async def test_grep_failure_is_500_never_empty(self, app, client, tree):
        r = await _get(
            app, client, _LocalManager(tree["ws"]), SEARCH, {"pattern": "(", "regex": True}
        )
        assert r.status_code == 500, r.text

    @pytest.mark.parametrize("path", ["../outside", "/etc", "dir_out", "link_out"])
    async def test_search_path_is_confined(self, app, client, tree, path):
        r = await _get(
            app, client, _LocalManager(tree["ws"]), SEARCH, {"pattern": "needle", "path": path}
        )
        assert r.status_code == 400, r.text
        assert OUTSIDE_MARKER not in r.text and "outside" not in r.json().get("hits", "")

    async def test_newline_in_pattern_is_400(self, app, client, tree):
        mgr = _LocalManager(tree["ws"])
        r = await _get(app, client, mgr, SEARCH, {"pattern": "a\nb"})
        assert r.status_code == 400 and mgr.commands == []


def test_new_routes_are_published_as_get():
    from src.main import app

    paths = app.openapi()["paths"]
    assert set(paths["/api/workspaces/{container_name}/search"]) == {"get"}
    assert paths["/api/workspaces/{container_name}/search"]["get"]["operationId"] == (
        "search_workspace"
    )
    params = {
        p.get("name") for p in paths["/api/workspaces/{container_name}/files"]["get"]["parameters"]
    }
    assert {"path", "start", "end"} <= params


# ---------------------------------------------------------------------------
# Mutation controls: break one fence, prove the tests above notice
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMutationControls:
    async def test_without_the_containment_case_a_symlink_escape_leaks(
        self, app, client, tree, monkeypatch
    ):
        real = files_io._read_path_command

        def no_case(candidate, workspace_root, start=None, end=None):
            cmd = real(candidate, workspace_root, start, end)
            start = cmd.index('case "$real"')
            end = cmd.index("esac", start) + len("esac; ")
            return cmd[:start] + cmd[end:]

        monkeypatch.setattr(files_io, "_read_path_command", no_case)
        r = await _get(app, client, _LocalManager(tree["ws"]), FILES, {"path": "link_out"})
        # the escape test asserts 400 and no OUTSIDE_MARKER; under the mutant both break
        assert r.status_code == 200 and OUTSIDE_MARKER in r.text

    async def test_without_the_lexical_guard_commands_run(self, app, client, tree, monkeypatch):
        import posixpath

        def permissive(path, *, workspace_root):
            return posixpath.normpath(posixpath.join(workspace_root, path.lstrip("/")))

        monkeypatch.setattr(files_io, "_normalize_workspace_read_path", permissive)
        mgr = _LocalManager(tree["ws"])
        await _get(app, client, mgr, FILES, {"path": "../outside/secret.txt"})
        # the lexical test asserts ZERO commands; under the mutant git/sh is reached
        assert mgr.commands != []

    async def test_following_symlinks_in_recursion_would_leak(self, app, client, tree, monkeypatch):
        real = files_io._search_command

        def follow(*a, **kw):
            # GNU grep -R follows every symlink; BSD grep (macOS) needs -S as well.
            bsd = (
                "BSD"
                in subprocess.run(["grep", "--version"], capture_output=True, text=True).stdout
            )
            return real(*a, **kw).replace("grep -rnHI", "grep -RnHI -S" if bsd else "grep -RnHI")

        monkeypatch.setattr(files_io, "_search_command", follow)
        d = (
            await _get(app, client, _LocalManager(tree["ws"]), SEARCH, {"pattern": "needle"})
        ).json()
        assert any("outside" in h["text"] for h in d["hits"])


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [FILES, SEARCH + "?pattern=needle"])
async def test_foreign_tenant_gets_404_and_nothing_runs(app, client, tree, url):
    """The stub models a REAL workspace owned by another tenant: it is returned for an
    UNSCOPED lookup (tenant_id=None) and hidden from any other tenant. A route that
    drops ``tenant_id=`` from get_container therefore gets the workspace and fails
    here. (The previous stub returned None for tenant_id=None too, so that mutant
    survived the audit.)"""
    mgr = _LocalManager(tree["ws"])
    real = mgr.get_container
    mgr.get_container = lambda name, tenant_id=None: real(name) if tenant_id is None else None
    r = await _get(app, client, mgr, url)  # _get authenticates as "tenant-a"
    assert r.status_code == 404, r.text
    assert mgr.commands == []


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [FILES, SEARCH + "?pattern=needle"])
async def test_owning_tenant_is_served_and_scoped(app, client, tree, url):
    """Positive control for the test above, and the exec is scoped to the caller."""
    mgr = _LocalManager(tree["ws"])
    real = mgr.get_container
    seen: list = []
    mgr.get_container = lambda name, tenant_id=None: (
        seen.append(tenant_id) or (real(name) if tenant_id == "tenant-a" else None)
    )
    r = await _get(app, client, mgr, url)
    assert r.status_code == 200, r.text
    assert seen == ["tenant-a"] and mgr.tenants == ["tenant-a"]


# ---------------------------------------------------------------------------
# Audit 5279880810 follow-ups
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pattern_that_looks_like_an_option_is_a_pattern(app, client, tree):
    """Kills the surviving mutant 'drop -e before the pattern'."""
    (tree["ws"] / "src" / "flags.txt").write_text("plain\n-v marks this line\n")
    d = (
        await _get(app, client, _LocalManager(tree["ws"]), SEARCH, {"pattern": "-v", "path": "src"})
    ).json()
    assert d["hits"] == [
        {"file": "src/flags.txt", "line": 2, "text": "-v marks this line", "repo": None}
    ]


@pytest.mark.asyncio
async def test_metadata_is_over_whole_file_bytes_past_the_cap(app, client, tree, monkeypatch):
    raw = b"".join(f"row {i}\n".encode() for i in range(1000))
    (tree["ws"] / "big.txt").write_bytes(raw)
    monkeypatch.setattr(files_io, "MAX_READ_BYTES", 100)
    d = (await _get(app, client, _LocalManager(tree["ws"]), FILES, {"path": "big.txt"})).json()
    assert d["truncated"] is True
    assert len(d["content"].encode()) <= 100 and raw.decode().startswith(d["content"])
    assert d["size_bytes"] == len(raw)
    assert d["lines_total"] == 1000
    assert d["sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.asyncio
async def test_metadata_is_over_bytes_for_non_utf8(app, client, tree):
    raw = b"\xff\xfe latin-1 \xe9\nsecond"  # no trailing newline
    (tree["ws"] / "bin.txt").write_bytes(raw)
    d = (await _get(app, client, _LocalManager(tree["ws"]), FILES, {"path": "bin.txt"})).json()
    assert d["sha256"] == hashlib.sha256(raw).hexdigest()
    assert d["size_bytes"] == len(raw)
    assert d["lines_total"] == 2
    assert d["truncated"] is False


@pytest.mark.asyncio
async def test_span_of_a_large_file_is_exact(app, client, tree, monkeypatch):
    (tree["ws"] / "big.txt").write_text("".join(f"row {i}\n" for i in range(1, 5001)))
    monkeypatch.setattr(files_io, "MAX_READ_BYTES", 100)  # whole-file cap must not bite spans
    d = (
        await _get(
            app,
            client,
            _LocalManager(tree["ws"]),
            FILES,
            {"path": "big.txt", "start": 4000, "end": 4002},
        )
    ).json()
    assert d["content"] == "row 4000\nrow 4001\nrow 4002\n"
    assert (d["start_line"], d["end_line"], d["lines_total"]) == (4000, 4002, 5000)
    assert d["truncated"] is False


# ---------------------------------------------------------------------------
# The TOCTOU race: a swap loop running during N requests
# ---------------------------------------------------------------------------

import ctypes  # noqa: E402
import threading  # noqa: E402

_RENAME_EXCHANGE = 2
_libc = ctypes.CDLL(None, use_errno=True)
_HAS_RENAMEAT2 = hasattr(_libc, "renameat2")


def _exchange(a: Path, b: Path) -> None:
    if _libc.renameat2(-100, bytes(a), -100, bytes(b), _RENAME_EXCHANGE) != 0:
        raise OSError(ctypes.get_errno(), "renameat2")


class _Swapper:
    """Atomically exchange ws/d (a real dir) with ws/d_evil (a symlink to outside)."""

    def __init__(self, ws: Path, outside: Path):
        (ws / "d").mkdir()
        (ws / "d" / "f").write_text("INSIDE\n")
        (outside / "f").write_text(f"{OUTSIDE_MARKER}\n")
        (ws / "d_evil").symlink_to(outside)
        self.a, self.b = ws / "d", ws / "d_evil"
        self.stop = threading.Event()
        self.swaps = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            _exchange(self.a, self.b)
            self.swaps += 1

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def _legacy_read_command(candidate, workspace_root, start=None, end=None):
    """The pre-audit shape: resolve+check the PATH, then re-walk the PATH to read it."""
    import shlex

    return "; ".join(
        [
            f"workspace=$(realpath -- {shlex.quote(workspace_root)}) || exit 1",
            f"candidate={shlex.quote(candidate)}",
            '[ -e "$candidate" ] || exit 40',
            'target=$(realpath -- "$candidate") || exit 40',
            'case "$target" in "$workspace"|"$workspace"/*) ;; *) exit 41 ;; esac',
            'rel="${target#"$workspace"}"',
            "printf 'FILE /%s\\nSIZE 0\\nLINES 0\\nSHA -\\n' \"${rel#/}\"",
            'cat -- "$candidate"',
        ]
    )


async def _race_reads(app, client, tree, n):
    mgr = _LocalManager(tree["ws"])
    counts = {"inside": 0, "outside": 0, "refused": 0, "other": 0}
    with _Swapper(tree["ws"], tree["outside"]) as sw:
        for _ in range(n):
            r = await _get(app, client, mgr, FILES, {"path": "d/f"})
            if r.status_code == 200 and OUTSIDE_MARKER in r.text:
                counts["outside"] += 1
            elif r.status_code == 200 and "INSIDE" in r.text:
                counts["inside"] += 1
            elif r.status_code in (400, 404):
                counts["refused"] += 1
            else:
                counts["other"] += 1
    return counts, sw.swaps


N_RACE = 400


@pytest.mark.skipif(not _HAS_RENAMEAT2, reason="needs renameat2 (glibc/Linux)")
@pytest.mark.asyncio
class TestSymlinkSwapRace:
    async def test_swap_loop_never_serves_outside_content(self, app, client, tree):
        counts, swaps = await _race_reads(app, client, tree, N_RACE)
        print(f"race /files: {counts} swaps={swaps}")
        assert counts["outside"] == 0, counts
        # the race was really exercised: both sides of the swap were observed
        assert swaps > N_RACE and counts["inside"] > 0 and counts["refused"] > 0, counts

    async def test_mutation_control_path_rewalk_leaks(self, app, client, tree, monkeypatch):
        monkeypatch.setattr(files_io, "_read_path_command", _legacy_read_command)
        counts, _ = await _race_reads(app, client, tree, N_RACE)
        print(f"race /files legacy: {counts}")
        assert counts["outside"] > 0, counts

    async def test_swap_loop_never_searches_outside(self, app, client, tree):
        mgr = _LocalManager(tree["ws"])
        leaked = served = 0
        with _Swapper(tree["ws"], tree["outside"]):
            for _ in range(N_RACE // 2):
                r = await _get(app, client, mgr, SEARCH, {"pattern": OUTSIDE_MARKER, "path": "d"})
                if r.status_code == 200:
                    served += 1
                    leaked += bool(r.json()["hits"])
        print(f"race /search: served={served} leaked={leaked}")
        assert leaked == 0 and served > 0

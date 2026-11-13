"""``ls-files -s -z`` over 256 KiB reaches the branch-switch recheck whole, and a
clipped listing fails closed.

The byte-exact recheck (#147) looks every flagged path up in ONE ``ls-files -s -z``
read. That read's command is tiny, but its OUTPUT scales with the index: a repo with
a few thousand long paths produces far more than one 96 KiB recheck chunk. If the
output were cut, the parsed index would lack the tail, and a recheck that silently
compared a prefix could pass. These tests run the route's real command strings
against a real repository (``_LocalManager``), prove the whole listing is parsed,
and prove that a clipped listing - flagged ``truncated`` by the exec layer, or cut
without any flag - never yields a 200.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.unit.test_git_branch_route import _ArgCappedManager, _branch, _commit_all, _current
from tests.unit.test_git_restore_route import _LocalManager, _run

_N_FILES = 4000
_CLIP_AT = 96 * 1024  # one recheck chunk's budget; the listing must not stop here
_INDEX_READ = "ls-files -s -z"


@pytest.fixture(autouse=True)
def _isolate_from_the_enclosing_git(monkeypatch):
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key, raising=False)
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "t")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "t@example.com")


def _bulk_paths() -> list[str]:
    base = "a_rather_long_directory_name_for_padding/and_another_nested_level_here"
    return [f"{base}/module_{i:05d}_with_a_long_descriptive_name.txt" for i in range(_N_FILES)]


@pytest.fixture
def hugerepo(tmp_path: Path) -> tuple[Path, list[str]]:
    """``main`` with 4000 long-path files plus ``zz/x.bat``, the index's LAST entry.

    ``other`` changes only ``zz/x.bat`` to a blob committed WITH CRLF under
    ``text eol=crlf``: after the switch ``ls-files -m`` flags it, so the route must
    look it up in the index listing - past the 96 KiB mark.
    """
    r = tmp_path / "huge"
    r.mkdir()
    _run(r, "init", "-q", "-b", "main")
    _run(r, "config", "user.email", "t@example.com")
    _run(r, "config", "user.name", "t")
    (r / ".gitattributes").write_text("*.bat text eol=crlf\n")
    paths = _bulk_paths()
    for i, p in enumerate(paths):
        f = r / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"{i}\n")
    (r / "zz").mkdir()
    (r / "zz" / "x.bat").write_bytes(b"a\r\nb\r\n")
    _commit_all(r, "bulk")
    _run(r, "switch", "-q", "-c", "other")
    (tmp_path / "raw").write_bytes(b"a\r\nc\r\n")
    sha = _run(r, "hash-object", "-w", "--no-filters", str(tmp_path / "raw"))
    _run(r, "update-index", "--cacheinfo", f"100644,{sha},zz/x.bat")
    _run(r, "commit", "-q", "-m", "crlf blob")
    _run(r, "switch", "-q", "-f", "main")
    assert _run(r, "ls-files", "-m") == ""
    return r, paths + ["zz/x.bat"]


class _ClippedIndexListing(_LocalManager):
    """``ls-files -s -z`` stdout cut at ``_CLIP_AT`` bytes; ``flag`` says whether the
    exec layer reports it (the real one does: ``truncated``/``bytes_omitted``)."""

    def __init__(self, repo: Path, *, flag: bool):
        super().__init__(repo)
        self.flag = flag
        self.clipped = 0

    async def execute_command(self, container_name, command, tenant_id=None, timeout=300):
        res = await super().execute_command(container_name, command, tenant_id, timeout)
        if _INDEX_READ in command:
            out = res["stdout"]
            assert len(out.encode()) > _CLIP_AT  # the clip really removes something
            self.clipped += 1
            res = {**res, "stdout": out.encode()[:_CLIP_AT].decode("utf-8", "ignore")}
            if self.flag:
                res["truncated"] = True
        return res


@pytest.mark.asyncio
class TestIndexListingIsWhole:
    async def test_listing_over_256_kib_parses_every_entry(self, hugerepo):
        from src.routers.containers.git_status_commit import (
            _index_entries_command,
            _parse_index_entries,
        )

        r, paths = hugerepo
        res = await _LocalManager(r).execute_command("ctr", _index_entries_command())

        assert res["exit_code"] == 0, res["stderr"]
        assert len(res["stdout"].encode()) > 256 * 1024
        index = _parse_index_entries(res["stdout"])
        tracked = _run(r, "ls-files").splitlines()
        assert len(tracked) == _N_FILES + 2  # + .gitattributes + zz/x.bat
        assert len(index) == len(tracked)
        assert set(index) == set(tracked)
        assert tracked[-1] == "zz/x.bat" and index["zz/x.bat"][0][2] == "0"

    async def test_recheck_over_every_path_finds_the_last_one(self, hugerepo):
        from src.routers.containers.git_status_commit import _still_incomplete

        r, paths = hugerepo
        # On `other` zz/x.bat's blob holds CRLF, so its raw bytes match the index
        # (on `main` the blob was normalised to LF and the bytes differ).
        _run(r, "switch", "-q", "other")
        assert await _still_incomplete(_ArgCappedManager(r), "ctr", "t", paths) == []
        (r / paths[-1]).write_bytes(b"stale\r\n")
        assert await _still_incomplete(_ArgCappedManager(r), "ctr", "t", paths) == [paths[-1]]

    async def test_switch_flagging_the_last_entry_is_200(self, app, client, hugerepo):
        r, _ = hugerepo
        mgr = _LocalManager(r)

        resp = await _branch(app, client, mgr, "other")

        assert resp.status_code == 200, resp.text
        assert _current(r) == "other"
        assert _run(r, "ls-files", "-m") == "zz/x.bat"  # really went through the recheck
        assert sum(_INDEX_READ in c for c in mgr.commands) == 1


@pytest.mark.asyncio
class TestClippedIndexListingFailsClosed:
    async def test_flagged_clip_fails_the_recheck(self, hugerepo):
        from src.routers.containers.git_status_commit import _still_incomplete

        r, paths = hugerepo
        mgr = _ClippedIndexListing(r, flag=True)
        assert await _still_incomplete(mgr, "ctr", "t", paths) is None
        assert mgr.clipped == 1

    async def test_silent_clip_leaves_the_tail_incomplete(self, hugerepo):
        # No flag at all: the entries past the cut are absent from the parsed index,
        # which the recheck treats as incomplete - never as matched.
        from src.routers.containers.git_status_commit import _still_incomplete

        r, paths = hugerepo
        mgr = _ClippedIndexListing(r, flag=False)
        result = await _still_incomplete(mgr, "ctr", "t", paths)
        assert mgr.clipped == 1
        assert result is not None and paths[-1] in result
        assert len(result) > len(paths) // 2  # the whole tail, not one path

    @pytest.mark.parametrize("flag", [True, False])
    async def test_switch_with_clipped_listing_is_500_incomplete(self, app, client, hugerepo, flag):
        r, _ = hugerepo
        mgr = _ClippedIndexListing(r, flag=flag)

        resp = await _branch(app, client, mgr, "other")

        assert mgr.clipped == 1
        assert _current(r) == "other"  # the switch happened; the 200 is what is refused
        assert resp.status_code == 500, resp.text
        details = resp.json()["error"]["details"]
        assert details["error"] == "branch_switch_incomplete", resp.text
        if flag:
            assert details["missing_paths"] is None
            assert "recheck" in details["verify_error"], resp.text
        else:
            assert details["missing_paths"] == ["zz/x.bat"]

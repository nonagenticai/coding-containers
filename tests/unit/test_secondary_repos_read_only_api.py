"""py2rust gaps #14 + attribution: the API side of read-only secondary repositories.

Every file route that
WRITES refuses a path at or under ``.repos/`` with a typed 409 before the workspace
is touched, while the same routes keep working everywhere else; and the routes that
LIST paths say which repository a path belongs to (``repo``), so a workspace holding
one Rust target and N Python sources is never read as one merged tree.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.routers.containers._common import (
    refuse_write_into_secondary_repos,
    secondary_repo_of_path,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/workspace/.repos/acme__svc/src/app.py", "acme/svc"),
        (".repos/acme__svc/src/app.py", "acme/svc"),
        ("./.repos/acme__svc", "acme/svc"),
        (".repos//acme__svc/x", "acme/svc"),
        ("/workspace/.repos/my-org__my_repo/x", "my-org/my_repo"),
        ("/workspace/.repos", None),
        ("/workspace/.repos/.clone.ab12cd34/x", None),
        ("/workspace/src/main.rs", None),
        ("/workspace/x/.repos/acme__svc/y", None),  # not the workspace's .repos
        ("/etc/passwd", None),
        ("../.repos/acme__svc", None),
        ("", None),
    ],
)
def test_secondary_repo_of_path(path, expected):
    assert secondary_repo_of_path(path) == expected


def test_attribution_honours_a_non_default_repository_root():
    root = "/srv/repo"
    assert secondary_repo_of_path("/srv/repo/.repos/a__b/f", workspace_root=root) == "a/b"
    assert secondary_repo_of_path("/workspace/.repos/a__b/f", workspace_root=root) == "a/b"


@pytest.mark.parametrize(
    "path",
    [
        ".repos",
        ".repos/",
        "/workspace/.repos",
        ".repos/acme__svc/app.py",
        "/workspace/.repos/acme__svc/new_dir",
        "./src/../.repos/acme__svc/x",
    ],
)
def test_writes_into_repos_are_refused(path):
    from coding_client.exceptions import BaseAPIException

    with pytest.raises(BaseAPIException) as info:
        refuse_write_into_secondary_repos(path, container_name="ws")
    assert info.value.status_code == 409


@pytest.mark.parametrize(
    "path", ["src/app.py", "/workspace/target/x", ".reposx/y", "/workspace/src/.repos/x"]
)
def test_writes_elsewhere_are_not_refused(path):
    refuse_write_into_secondary_repos(path, container_name="ws")


# ---------------------------------------------------------------------------
# Route level, against a REAL workspace directory: every write route
# ---------------------------------------------------------------------------
#
# ``execute_command`` runs the route's real command in a real shell in a tmp
# workspace with ``.repos/`` checkouts (``linked_ws``).
# Reads and writes open through ``/proc/self/fd`` (Linux, as in the pod), so the
# routes that must SUCCEED are skipped where it is missing; the refusals happen
# before the shell and run everywhere.

_HAS_PROC_FD = os.path.isdir("/proc/self/fd")
needs_proc = pytest.mark.skipif(not _HAS_PROC_FD, reason="needs Linux /proc/self/fd (pods have it)")


def _user():
    user = SimpleNamespace(effective_tenant_id="tenant-a")
    user.is_admin = lambda: False
    return user


def _tree(ws) -> dict[str, bytes]:
    return {
        str(p.relative_to(ws)): (p.read_bytes() if p.is_file() and not p.is_symlink() else b"")
        for p in sorted(ws.rglob("*"))
    }


WRITES_INTO_REPOS = [
    ("POST", "write-file", {"json": {"file_path": ".repos/acme__svc/pkg/app.py", "content": "y"}}),
    ("POST", "mkdir", {"json": {"path": ".repos/acme__svc/new"}}),
    (
        "POST",
        "touch",
        {"json": {"path": "/workspace/.repos/acme__svc", "filename": "n", "extension": "py"}},
    ),
    (
        "POST",
        "move",
        {"json": {"source_path": ".repos/acme__svc/pkg/app.py", "destination_path": "src/x.py"}},
    ),
    (
        "POST",
        "move",
        {"json": {"source_path": "src/a.py", "destination_path": ".repos/acme__svc/a.py"}},
    ),
    ("POST", "move", {"json": {"source_path": ".repos", "destination_path": "old-repos"}}),
    ("DELETE", "files", {"params": {"path": ".repos/acme__svc", "recursive": "true"}}),
    ("DELETE", "files", {"params": {"path": "/workspace/.repos", "recursive": "true"}}),
]


@pytest.mark.parametrize(("method", "op", "kwargs"), WRITES_INTO_REPOS)
async def test_write_routes_refuse_repos_with_a_typed_409_and_change_nothing(
    app, client, linked_ws, method, op, kwargs
):
    before = _tree(linked_ws)
    resp = await _call_real(app, client, linked_ws, method, f"/api/workspaces/ws/{op}", **kwargs)
    assert resp.status_code == 409, resp.text
    err = resp.json()["error"]
    assert err["details"]["error_code"] == "SECONDARY_REPO_READ_ONLY"
    assert "read-only" in err["message"].lower()
    assert _tree(linked_ws) == before


@needs_proc
async def test_negative_control_the_same_routes_still_write_outside_repos(app, client, linked_ws):
    resp = await _call_real(
        app,
        client,
        linked_ws,
        "POST",
        "/api/workspaces/ws/write-file",
        json={"file_path": "src/new.py", "content": "z = 1\n"},
    )
    assert resp.status_code == 200, resp.text
    assert (linked_ws / "src" / "new.py").read_text() == "z = 1\n"
    resp = await _call_real(
        app, client, linked_ws, "POST", "/api/workspaces/ws/mkdir", json={"path": "src/newdir"}
    )
    assert resp.status_code in (200, 201), resp.text
    assert (linked_ws / "src" / "newdir").is_dir()


# ---------------------------------------------------------------------------
# Attribution in listings (real directories)
# ---------------------------------------------------------------------------


async def _get_real(app, client, ws, url, params):
    from coding_client.auth.tenant import require_tenant

    app.dependency_overrides[require_tenant] = lambda: "tenant-a"
    try:
        with patch(
            "src.routers.containers.get_container_manager", return_value=_running_manager(ws)
        ):
            return await client.get(url, params=params)
    finally:
        app.dependency_overrides.clear()


@needs_proc
async def test_directory_listing_of_repos_attributes_each_entry(app, client, linked_ws):
    resp = await _get_real(app, client, linked_ws, "/api/workspaces/ws/files", {"path": ".repos"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["repo"] is None  # .repos itself belongs to no single repository
    entries = {e["name"]: e["repo"] for e in data["entries"] if e["name"] not in (".", "..")}
    assert entries == {"acme__svc": "acme/svc", "acme__billing": "acme/billing"}


@needs_proc
async def test_file_read_inside_a_source_names_its_repo(app, client, linked_ws):
    resp = await _get_real(
        app, client, linked_ws, "/api/workspaces/ws/files", {"path": ".repos/acme__svc/pkg/app.py"}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["repo"] == "acme/svc"


@needs_proc
async def test_primary_paths_attribute_to_null(app, client, linked_ws):
    resp = await _get_real(app, client, linked_ws, "/api/workspaces/ws/files", {"path": "src"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["repo"] is None
    assert all(e["repo"] is None for e in resp.json()["entries"])


@needs_proc
async def test_search_hits_name_the_repo_they_came_from(app, client, linked_ws):
    resp = await _get_real(
        app, client, linked_ws, "/api/workspaces/ws/search", {"pattern": "handler"}
    )
    assert resp.status_code == 200, resp.text
    assert sorted((h["file"], h["repo"]) for h in resp.json()["hits"]) == [
        (".repos/acme__billing/api.py", "acme/billing"),
        (".repos/acme__svc/pkg/routes.py", "acme/svc"),
        ("src/main.rs", None),
    ]


# ---------------------------------------------------------------------------
# Resolved paths: a symlink in the primary repository that leads into .repos/
# ---------------------------------------------------------------------------


@pytest.fixture
def linked_ws(tmp_path):
    """A workspace whose primary repo links into a source checkout under .repos/."""
    ws = tmp_path / "ws"
    repos = ws / ".repos"
    src = repos / "acme__svc"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "app.py").write_text("x = 1\n")
    (src / "pkg" / "routes.py").write_text("def handler():\n    pass\n")
    (repos / "acme__billing").mkdir()
    (repos / "acme__billing" / "api.py").write_text("def handler():\n    pass\n")
    (ws / "src").mkdir()
    (ws / "src" / "a.py").write_text("a = 1\n")
    (ws / "src" / "main.rs").write_text("fn handler() {}\n")
    (ws / "src" / "link").symlink_to("../.repos/acme__svc")
    (ws / "src" / "dangling").symlink_to("../.repos/acme__svc/new.py")
    (ws / "rootlink").symlink_to(".")
    return ws


def _running_manager(ws):
    """execute_command runs the route's REAL command in a real shell in ``ws``."""
    import subprocess

    container = SimpleNamespace(
        status=SimpleNamespace(value="running"), codebase_path=str(ws), tenant_id="tenant-a"
    )

    async def run(_name, command, **_kw):
        if isinstance(command, list):
            command = " ".join(command) if command[:2] != ["/bin/sh", "-c"] else command[2]
        proc = subprocess.run(  # noqa: S603
            ["/bin/sh", "-c", command],  # noqa: S607
            cwd=ws,
            capture_output=True,
            text=True,
            check=False,
        )
        return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    return SimpleNamespace(get_container=lambda *_a, **_k: container, execute_command=run)


async def _call_real(app, client, ws, method, url, **kwargs):
    from coding_client.auth.tenant import get_current_user_with_tenant

    app.dependency_overrides[get_current_user_with_tenant] = _user
    try:
        with patch(
            "src.routers.containers.get_container_manager", return_value=_running_manager(ws)
        ):
            return await client.request(method, url, **kwargs)
    finally:
        app.dependency_overrides.clear()


def _gnu_mv() -> bool:
    import subprocess

    return subprocess.run(["mv", "--version"], capture_output=True, check=False).returncode == 0


async def test_negative_control_deleting_the_link_itself_still_works(app, client, linked_ws):
    resp = await _call_real(
        app, client, linked_ws, "DELETE", "/api/workspaces/ws/files", params={"path": "src/link"}
    )
    assert resp.status_code == 200, resp.text
    assert not (linked_ws / "src" / "link").is_symlink()
    assert (linked_ws / ".repos" / "acme__svc" / "pkg" / "app.py").exists()


@pytest.mark.skipif(not _gnu_mv(), reason="the move command uses GNU mv -T (CI/pods have it)")
async def test_negative_control_moving_the_link_itself_still_works(app, client, linked_ws):
    resp = await _call_real(
        app,
        client,
        linked_ws,
        "POST",
        "/api/workspaces/ws/move",
        json={"source_path": "src/link", "destination_path": "src/link2"},
    )
    assert resp.status_code == 200, resp.text
    assert (linked_ws / "src" / "link2").is_symlink()

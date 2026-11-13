"""coding-containers#166 — write-file is confined to the repository workspace.

Two halves, mirroring the read path (``test_workspace_files_read_search.py``):

* the LEXICAL half runs in the router (``_normalize_workspace_read_path``): any absolute path
  outside the ``/workspace`` alias, any ``..`` segment and any control character is refused with
  a 400 BEFORE the container is touched;
* the SYMLINK half runs in the container (``_write_path_command``): the deepest existing
  ancestor is resolved before ``mkdir -p``, the parent again after it, a symlink/dir final
  component is refused, and the content is written through the ONE opened fd whose kernel
  record is containment-checked.

The real-shell tests run the generated command with ``sh`` against a temp workspace. They need
Linux ``/proc/self/fd`` (CI runs them). Each escape test has a MUTATION CONTROL: the same command
with exactly one containment line removed must escape, which proves that line is what stops it.
"""

from __future__ import annotations

import base64
import os
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.routers.containers.files_io import _write_path_command

_HAS_PROC_FD = os.path.isdir("/proc/self/fd")
needs_proc = pytest.mark.skipif(not _HAS_PROC_FD, reason="needs Linux /proc/self/fd (CI runs it)")


def _make_user():
    user = MagicMock()
    user.is_admin.return_value = False
    user.effective_tenant_id = "tenant-a"
    return user


def _container(codebase_path: str | None = None):
    container = MagicMock()
    container.status.value = "running"
    container.codebase_path = codebase_path
    return container


async def _post(app, client, file_path: str, exec_result=None, codebase_path=None):
    from coding_client.auth.tenant import get_current_user_with_tenant

    app.dependency_overrides[get_current_user_with_tenant] = _make_user
    with patch("src.routers.containers.get_container_manager") as mock_mgr:
        manager = MagicMock()
        manager.get_container.return_value = _container(codebase_path)
        manager.execute_command = AsyncMock(
            return_value=exec_result or {"exit_code": 0, "stdout": "", "stderr": ""}
        )
        mock_mgr.return_value = manager
        response = await client.post(
            "/api/containers/ctr/write-file",
            json={"file_path": file_path, "content": "x = 1\n"},
        )
    app.dependency_overrides.clear()
    return response, manager


# ---------------------------------------------------------------------------
# Router: the lexical half (no container exec happens for a refused path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/etc/x.py",  # absolute, outside the alias — the #166 hole
        "/app/src/main.py",  # absolute, outside the alias
        "/workspace/../etc/passwd",  # alias + traversal
        "src/../../etc/x",  # relative traversal
        "..",
        "/workspace",  # the root itself is a directory
        "src/x\x01.py",  # control character
    ],
)
async def test_refused_paths_never_reach_the_container(app, client, path):
    response, manager = await _post(app, client, path)
    # 422 = refused earlier by the request model (control characters); either way, no exec.
    assert response.status_code in (400, 422), response.text
    manager.execute_command.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "root", "expected"),
    [
        ("src/a.py", None, "/workspace/src/a.py"),
        ("/workspace/src/a.py", None, "/workspace/src/a.py"),
        ("a.py", "/srv/repo", "/srv/repo/a.py"),
        ("/workspace/a.py", "/srv/repo", "/srv/repo/a.py"),  # alias maps to the real root
    ],
)
async def test_accepted_paths_are_written_under_the_repository_root(
    app, client, path, root, expected
):
    response, manager = await _post(app, client, path, codebase_path=root)
    assert response.status_code == 200, response.text
    command = manager.execute_command.call_args.args[1]
    assert f"candidate={expected}" in command
    assert f"workspace=$(realpath -- {root or '/workspace'})" in command
    assert "readlink /proc/self/fd/3" in command


@pytest.mark.asyncio
@pytest.mark.parametrize(("code", "status"), [(41, 400), (42, 400), (43, 500), (1, 500)])
async def test_in_container_refusals_map_to_typed_statuses(app, client, code, status):
    response, _ = await _post(
        app, client, "src/a.py", exec_result={"exit_code": code, "stderr": "e"}
    )
    assert response.status_code == status, response.text


# ---------------------------------------------------------------------------
# Container: the symlink half, run for real with sh
# ---------------------------------------------------------------------------


def _run(command: str) -> subprocess.CompletedProcess:
    return subprocess.run(["sh", "-c", command], capture_output=True, text=True, check=False)


def _cmd(candidate, workspace, content: str = "hello\n") -> str:
    b64 = base64.b64encode(content.encode()).decode()
    return _write_path_command(str(candidate), str(workspace), b64)


def _drop(command: str, needle: str) -> str:
    """Mutation: remove exactly the one newline-joined step containing *needle*."""
    steps = command.split("\n")
    kept = [s for s in steps if needle not in s]
    assert len(kept) == len(steps) - 1, f"mutation must remove exactly one step: {needle!r}"
    mutated = "\n".join(kept)
    # The mutant must still PARSE, or a "refusal" would really be a syntax error.
    assert subprocess.run(["sh", "-n", "-c", mutated], check=False).returncode == 0
    return mutated


@pytest.fixture
def tree(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    return workspace, outside


@needs_proc
def test_writes_a_new_nested_file(tree):
    workspace, _ = tree
    target = workspace / "src" / "pkg" / "a.py"
    result = _run(_cmd(target, workspace, "print(1)\n"))
    assert result.returncode == 0, result.stderr
    assert target.read_text() == "print(1)\n"


@needs_proc
def test_overwrite_truncates(tree):
    workspace, _ = tree
    target = workspace / "a.txt"
    target.write_text("a much longer previous content\n")
    assert _run(_cmd(target, workspace, "short\n")).returncode == 0
    assert target.read_text() == "short\n"


@needs_proc
def test_directory_target_is_refused(tree):
    workspace, _ = tree
    (workspace / "d").mkdir()
    assert _run(_cmd(workspace / "d", workspace)).returncode == 42


@needs_proc
def test_symlinked_parent_escape_is_refused_and_writes_nothing(tree):
    workspace, outside = tree
    (workspace / "docs").symlink_to(outside)  # e.g. a symlink committed in the repo
    result = _run(_cmd(workspace / "docs" / "x.py", workspace))
    assert result.returncode == 41
    assert not (outside / "x.py").exists()


@needs_proc
def test_mutation_without_parent_check_the_symlinked_parent_escapes(tree):
    workspace, outside = tree
    (workspace / "docs").symlink_to(outside)
    command = _cmd(workspace / "docs" / "x.py", workspace)
    # Remove BOTH realpath gates that see the parent; the fd gate alone must still hold.
    mutated = _drop(_drop(command, "ra=$(realpath"), "rp=$(realpath")
    mutated = _drop(mutated, 'case "$ra"')
    mutated = _drop(mutated, 'case "$rp"')
    assert _run(mutated).returncode == 41, "the fd gate alone must refuse"
    assert (outside / "x.py").exists() is True  # opened (created empty) before the fd check…
    assert (outside / "x.py").read_text() == ""  # …but nothing was written through it
    mutated_no_fd = _drop(mutated, 'case "$real"')
    assert _run(mutated_no_fd).returncode == 0
    assert (outside / "x.py").read_text() == "hello\n", "with every gate removed it escapes"


@needs_proc
def test_missing_ancestor_chain_through_a_symlink_creates_nothing_outside(tree):
    workspace, outside = tree
    (workspace / "a").symlink_to(outside)
    result = _run(_cmd(workspace / "a" / "b" / "c" / "x.py", workspace))
    assert result.returncode == 41
    assert not (outside / "b").exists(), "mkdir -p must not run through the escaping link"


@needs_proc
def test_mutation_without_ancestor_check_mkdir_escapes(tree):
    workspace, outside = tree
    (workspace / "a").symlink_to(outside)
    command = _cmd(workspace / "a" / "b" / "c" / "x.py", workspace)
    mutated = _drop(command, 'case "$ra"')
    _run(mutated)
    assert (outside / "b").exists(), "without the pre-mkdir ancestor check, directories leak out"


@needs_proc
def test_symlink_final_component_is_refused(tree):
    workspace, outside = tree
    secret = outside / "secret"
    secret.write_text("keep\n")
    (workspace / "f.py").symlink_to(secret)
    assert _run(_cmd(workspace / "f.py", workspace)).returncode == 41
    assert secret.read_text() == "keep\n"


@needs_proc
def test_mutation_without_link_and_fd_checks_the_final_symlink_is_overwritten(tree):
    workspace, outside = tree
    secret = outside / "secret"
    secret.write_text("keep\n")
    (workspace / "f.py").symlink_to(secret)
    command = _cmd(workspace / "f.py", workspace)
    mutated = _drop(_drop(command, 'if [ -L "$candidate" ]'), 'case "$real"')
    assert _run(mutated).returncode == 0
    assert secret.read_text() == "hello\n", "both gates removed: the write follows the link out"

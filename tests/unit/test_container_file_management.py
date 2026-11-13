"""Tests for moving and deleting files and directories in repo containers."""

import shutil
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_GNU_MV = shutil.which("gmv") or (shutil.which("mv") if sys.platform == "linux" else None)


def _user(*, admin: bool = True, tenant_id: str = "tenant-1") -> MagicMock:
    user = MagicMock()
    user.effective_tenant_id = tenant_id
    user.is_admin.return_value = admin
    return user


def _container(*, status: str = "running", codebase_path: str = "/workspace") -> MagicMock:
    container = MagicMock()
    container.status.value = status
    container.codebase_path = codebase_path
    return container


def _manager(*, exit_code: int = 0, stderr: str = "") -> MagicMock:
    manager = MagicMock()
    manager.get_container.return_value = _container()
    manager.execute_command = AsyncMock(
        return_value={"exit_code": exit_code, "stdout": "", "stderr": stderr}
    )
    return manager


@pytest.fixture
def authorize(app):
    from coding_client.auth.tenant import get_current_user_with_tenant

    def apply(user=None):
        app.dependency_overrides[get_current_user_with_tenant] = lambda: user or _user()

    yield apply
    app.dependency_overrides.clear()


class TestNormalizeWorkspacePath:
    def test_normalizes_and_maps_docker_root(self):
        from src.routers.containers import _canonical_workspace_path, _normalize_workspace_path

        assert _normalize_workspace_path("src/../README.md") == "/workspace/README.md"
        root = "/app/provisioned_containers/ctr/repo"
        resolved = _normalize_workspace_path("/workspace/src/main.py", workspace_root=root)
        assert resolved == f"{root}/src/main.py"
        assert _canonical_workspace_path(resolved, workspace_root=root) == "/workspace/src/main.py"

    @pytest.mark.parametrize("path", ["/workspace", ".", "..", "/etc/passwd"])
    def test_rejects_workspace_root_and_paths_outside_it(self, path):
        from src.routers.containers import _normalize_workspace_path

        with pytest.raises(ValueError):
            _normalize_workspace_path(path)

    def test_rejects_empty_and_control_characters(self):
        from src.routers.containers import _normalize_workspace_path

        with pytest.raises(ValueError, match="empty"):
            _normalize_workspace_path("   ")
        with pytest.raises(ValueError, match="control"):
            _normalize_workspace_path("src/bad\nname")


class TestFilesystemCommands:
    @pytest.mark.skipif(_GNU_MV is None, reason="runtime command requires GNU mv -T")
    def test_moves_a_directory_tree(self, tmp_path):
        from src.routers.containers.files_manage import _move_command

        source = tmp_path / "old directory"
        source.mkdir()
        (source / "nested").mkdir()
        (source / "nested" / "file.txt").write_text("content")
        destination = tmp_path / "new directory"

        moved = subprocess.run(
            [
                "sh",
                "-c",
                _move_command(
                    str(source),
                    str(destination),
                    overwrite=False,
                    workspace_root=str(tmp_path),
                    move_binary=_GNU_MV,
                ),
            ],
            check=False,
        )
        assert moved.returncode == 0
        assert not source.exists()
        assert (destination / "nested" / "file.txt").read_text() == "content"

    def test_recursively_deletes_a_directory_tree(self, tmp_path):
        from src.routers.containers.files_manage import _delete_command

        directory = tmp_path / "generated"
        (directory / "nested").mkdir(parents=True)
        (directory / "nested" / "file.txt").write_text("content")

        deleted = subprocess.run(
            [
                "sh",
                "-c",
                _delete_command(str(directory), recursive=True, workspace_root=str(tmp_path)),
            ],
            check=False,
        )
        assert deleted.returncode == 0
        assert not directory.exists()

    def test_non_recursive_delete_refuses_non_empty_directory(self, tmp_path):
        from src.routers.containers.files_manage import _delete_command

        directory = tmp_path / "non-empty"
        directory.mkdir()
        (directory / "file.txt").write_text("content")
        deleted = subprocess.run(
            [
                "sh",
                "-c",
                _delete_command(str(directory), recursive=False, workspace_root=str(tmp_path)),
            ],
            check=False,
        )
        assert deleted.returncode == 44
        assert directory.exists()

    def test_delete_rejects_an_intermediate_symlink_escape(self, tmp_path):
        from src.routers.containers.files_manage import _delete_command

        workspace = tmp_path / "workspace"
        outside = tmp_path / "outside"
        workspace.mkdir()
        outside.mkdir()
        protected = outside / "protected.txt"
        protected.write_text("keep")
        (workspace / "escape").symlink_to(outside, target_is_directory=True)

        deleted = subprocess.run(
            [
                "sh",
                "-c",
                _delete_command(
                    str(workspace / "escape" / "protected.txt"),
                    recursive=True,
                    workspace_root=str(workspace),
                ),
            ],
            check=False,
        )
        assert deleted.returncode == 41
        assert protected.read_text() == "keep"

    def test_delete_can_remove_a_broken_symlink_itself(self, tmp_path):
        from src.routers.containers.files_manage import _delete_command

        link = tmp_path / "broken-link"
        link.symlink_to(tmp_path / "missing")
        deleted = subprocess.run(
            [
                "sh",
                "-c",
                _delete_command(str(link), recursive=False, workspace_root=str(tmp_path)),
            ],
            check=False,
        )
        assert deleted.returncode == 0
        assert not link.is_symlink()


@pytest.mark.asyncio
class TestMovePath:
    async def test_moves_directory_and_propagates_tenant(self, app, client, authorize):
        authorize(_user(admin=False, tenant_id="tenant-7"))
        manager = _manager()
        with patch("src.routers.containers.get_container_manager", return_value=manager):
            response = await client.post(
                "/api/containers/ctr/move",
                json={"source_path": "src/old", "destination_path": "src/new"},
            )

        assert response.status_code == 200
        assert response.json()["source_path"] == "/workspace/src/old"
        assert response.json()["destination_path"] == "/workspace/src/new"
        assert 'mv -T -- "$source" "$destination"' in manager.execute_command.await_args.args[1]
        assert manager.execute_command.await_args.kwargs["tenant_id"] == "tenant-7"

    async def test_overwrite_uses_force_flag(self, app, client, authorize):
        authorize()
        manager = _manager()
        with patch("src.routers.containers.get_container_manager", return_value=manager):
            response = await client.post(
                "/api/containers/ctr/move",
                json={"source_path": "a", "destination_path": "b", "overwrite": True},
            )

        assert response.status_code == 200
        command = manager.execute_command.await_args.args[1]
        assert 'mv -f -T -- "$source" "$destination"' in command
        assert "exit 43" not in command

    async def test_uses_registered_docker_repo_root(self, app, client, authorize):
        authorize()
        manager = _manager()
        manager.get_container.return_value = _container(
            codebase_path="/app/provisioned_containers/ctr/repo"
        )
        with patch("src.routers.containers.get_container_manager", return_value=manager):
            response = await client.post(
                "/api/containers/ctr/move",
                json={"source_path": "old", "destination_path": "/workspace/new"},
            )

        assert response.status_code == 200
        command = manager.execute_command.await_args.args[1]
        assert "source=/app/provisioned_containers/ctr/repo/old" in command
        assert "destination=/app/provisioned_containers/ctr/repo/new" in command

    @pytest.mark.parametrize(
        ("exit_code", "expected_status"),
        [(40, 404), (41, 400), (42, 404), (43, 409), (1, 500)],
    )
    async def test_maps_command_failures(self, app, client, authorize, exit_code, expected_status):
        authorize()
        manager = _manager(exit_code=exit_code, stderr="move failed")
        with patch("src.routers.containers.get_container_manager", return_value=manager):
            response = await client.post(
                "/api/containers/ctr/move",
                json={"source_path": "a", "destination_path": "b"},
            )
        assert response.status_code == expected_status

    @pytest.mark.parametrize(
        ("source", "destination"),
        [
            ("/workspace/a", "/workspace/a"),
            ("/workspace/a", "/workspace/a/child"),
            ("/etc/passwd", "/workspace/passwd"),
        ],
    )
    async def test_rejects_invalid_moves(self, app, client, authorize, source, destination):
        authorize()
        manager = _manager()
        with patch("src.routers.containers.get_container_manager", return_value=manager):
            response = await client.post(
                "/api/containers/ctr/move",
                json={"source_path": source, "destination_path": destination},
            )
        assert response.status_code == 400
        manager.execute_command.assert_not_awaited()

    async def test_container_must_exist_and_be_running(self, app, client, authorize):
        authorize()
        missing_manager = _manager()
        missing_manager.get_container.side_effect = ValueError("missing")
        with patch("src.routers.containers.get_container_manager", return_value=missing_manager):
            missing = await client.post(
                "/api/containers/ctr/move",
                json={"source_path": "a", "destination_path": "b"},
            )
        stopped_manager = _manager()
        stopped_manager.get_container.return_value = _container(status="stopped")
        with patch("src.routers.containers.get_container_manager", return_value=stopped_manager):
            stopped = await client.post(
                "/api/containers/ctr/move",
                json={"source_path": "a", "destination_path": "b"},
            )
        assert missing.status_code == 404
        assert stopped.status_code == 400


@pytest.mark.asyncio
class TestDeletePath:
    async def test_deletes_file(self, app, client, authorize):
        authorize()
        manager = _manager()
        with patch("src.routers.containers.get_container_manager", return_value=manager):
            response = await client.delete(
                "/api/containers/ctr/files", params={"path": "src/obsolete.py"}
            )
        assert response.status_code == 200
        assert response.json()["path"] == "/workspace/src/obsolete.py"
        assert response.json()["recursive"] is False

    async def test_recursively_deletes_non_empty_directory(self, app, client, authorize):
        authorize()
        manager = _manager()
        with patch("src.routers.containers.get_container_manager", return_value=manager):
            response = await client.delete(
                "/api/containers/ctr/files",
                params={"path": "/workspace/generated", "recursive": "true"},
            )
        assert response.status_code == 200
        assert response.json()["recursive"] is True
        assert 'rm -rf -- "$target"' in manager.execute_command.await_args.args[1]

    async def test_uses_registered_docker_repo_root(self, app, client, authorize):
        authorize()
        manager = _manager()
        manager.get_container.return_value = _container(
            codebase_path="/app/provisioned_containers/ctr/repo"
        )
        with patch("src.routers.containers.get_container_manager", return_value=manager):
            response = await client.delete(
                "/api/containers/ctr/files", params={"path": "/workspace/generated"}
            )
        assert response.status_code == 200
        assert (
            "target=/app/provisioned_containers/ctr/repo/generated"
            in (manager.execute_command.await_args.args[1])
        )

    @pytest.mark.parametrize(
        ("exit_code", "expected_status"),
        [(40, 404), (41, 400), (44, 409), (1, 500)],
    )
    async def test_maps_command_failures(self, app, client, authorize, exit_code, expected_status):
        authorize()
        manager = _manager(exit_code=exit_code, stderr="delete failed")
        with patch("src.routers.containers.get_container_manager", return_value=manager):
            response = await client.delete(
                "/api/containers/ctr/files", params={"path": "/workspace/old"}
            )
        assert response.status_code == expected_status

    @pytest.mark.parametrize("path", ["/workspace", "/workspace/../etc/passwd"])
    async def test_rejects_dangerous_paths(self, app, client, authorize, path):
        authorize()
        manager = _manager()
        with patch("src.routers.containers.get_container_manager", return_value=manager):
            response = await client.delete(
                "/api/containers/ctr/files", params={"path": path, "recursive": "true"}
            )
        assert response.status_code == 400
        manager.execute_command.assert_not_awaited()

    async def test_openapi_exposes_move_and_delete_operations(self, client):
        schema = (await client.get("/openapi.json")).json()
        move = schema["paths"]["/api/workspaces/{container_name}/move"]["post"]
        delete = schema["paths"]["/api/workspaces/{container_name}/files"]["delete"]
        assert move["operationId"] == "move_workspace_path"
        assert delete["operationId"] == "delete_workspace_path"

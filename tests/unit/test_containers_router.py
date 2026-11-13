"""Tests for containers router endpoints."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_mock_user(*, is_admin=True, tenant_id="test-tenant"):
    """Create a mock User object for auth bypass."""
    user = MagicMock()
    user.sub = "test-user"
    user.preferred_username = "testuser"
    user.effective_tenant_id = tenant_id
    user.is_admin.return_value = is_admin
    return user


#: The commit the fake pushed-path listing reports as the tip (what the push sends).
_PUSH_TIP = "a" * 40


def _push_exec(*, push=None, remote_add=None, unpushed="1\n"):
    """Command-aware fake for ``execute_command`` on the git/push route.

    Dispatches on what the command IS rather than its position, so the route can
    add or reorder steps without every test silently shifting a canned result
    onto the wrong command. The config guard answers exit 1 (git's "no match").
    """

    async def _run(container_name, command, tenant_id=None, timeout=300, env=None):
        if "rev-list --count" in command:
            return {"exit_code": 0, "stdout": unpushed}
        # The push exec re-runs the config guard as its first step, so it also
        # contains `config --get-regexp`: identify it before the guard exec.
        if " push " in command and "GIT_CONFIG_NOSYSTEM" in command:
            return push if push is not None else {"exit_code": 0, "stdout": "", "stderr": ""}
        if "config --get-regexp" in command:
            return {"exit_code": 1, "stdout": "", "stderr": ""}
        if " log --no-renames " in command:  # the pushed-path listing: tip, no paths
            return {"exit_code": 0, "stdout": f"tip {_PUSH_TIP}\n", "stderr": ""}
        if " remote add origin " in command and remote_add is not None:
            return remote_add
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    return _run


def _make_container_info(name="test-container", status_val="running", tenant_id="test-tenant"):
    """Build a ContainerInfo-like MagicMock with model_dump support."""
    from src.models.container_models import ContainerStatus

    info = MagicMock()
    info.container_name = name
    info.status = ContainerStatus(status_val)
    info.tenant_id = tenant_id
    info.model_dump.return_value = {
        "container_name": name,
        "api_endpoint": f"http://test/{name}",
        "internal_endpoint": "",
        "public_endpoint": "",
        "code_graph_id": "",
        "codebase_path": f"/app/data/{name}",
        "repo_url": "https://github.com/test/repo",
        "branch": "main",
        "status": status_val,
        "description": None,
        "pr_number": None,
        "provision_type": "branch",
        "github_access_token": None,
        "user_token_id": None,
        "user_id": None,
        "original_repo_owner": None,
        "original_repo_name": None,
        "system_fork_owner": None,
        "graph_initialized": False,
        "tenant_id": tenant_id,
    }
    return info


# ---- existing tests ----


@pytest.mark.asyncio
async def test_list_containers_returns_200(app, client):
    """GET /api/containers/ returns 200 with container list."""
    from coding_client.auth.tenant import get_current_user_with_tenant

    mock_user = _make_mock_user()
    app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

    with patch("src.routers.containers.get_container_manager") as mock_mgr:
        manager = MagicMock()
        manager.list_containers.return_value = []
        mock_mgr.return_value = manager

        response = await client.get("/api/containers/")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert "containers" in data


@pytest.mark.asyncio
async def test_list_workspaces_reuses_lifecycle_handler(app, client):
    """The canonical workspace route returns the same source-of-truth data."""
    from coding_client.auth.tenant import get_current_user_with_tenant

    mock_user = _make_mock_user()
    app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

    with patch("src.routers.containers.get_container_manager") as mock_mgr:
        manager = MagicMock()
        manager.list_containers.return_value = []
        mock_mgr.return_value = manager

        response = await client.get("/api/workspaces/")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["containers"] == []


def test_openapi_publishes_workspaces_and_hides_legacy_containers(app):
    """Only one lifecycle surface is published to generated clients."""
    schema = app.openapi()

    assert "/api/workspaces/" in schema["paths"]
    assert "/api/workspaces/{container_name}/exec" in schema["paths"]
    assert not any(path.startswith("/api/containers") for path in schema["paths"])

    operation_ids = {
        operation["operationId"]
        for path_item in schema["paths"].values()
        for operation in path_item.values()
        if isinstance(operation, dict) and "operationId" in operation
    }
    assert "create_workspace" in operation_ids
    assert "exec_in_workspace" in operation_ids
    assert "create_container" not in operation_ids


@pytest.mark.asyncio
async def test_container_status_not_found(app, client):
    """GET /api/containers/{name}/status returns 404 for nonexistent container."""
    from coding_client.auth.tenant import get_current_user_with_tenant

    mock_user = _make_mock_user()
    app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

    with patch("src.routers.containers.get_container_manager") as mock_mgr:
        manager = MagicMock()
        manager.get_container.side_effect = ValueError("Container 'nonexistent' not found")
        mock_mgr.return_value = manager

        response = await client.get("/api/containers/nonexistent/status")

    app.dependency_overrides.clear()
    assert response.status_code == 404


# ---- helper function tests ----


class TestSanitizeShellArg:
    def test_normal_value(self):
        from src.routers.containers import _sanitize_shell_arg

        assert _sanitize_shell_arg("hello") == "hello"

    def test_empty_raises(self):
        from src.routers.containers import _sanitize_shell_arg

        with pytest.raises(ValueError, match="Empty value"):
            _sanitize_shell_arg("")

    def test_special_chars_quoted(self):
        from src.routers.containers import _sanitize_shell_arg

        result = _sanitize_shell_arg("hello world")
        assert result == "'hello world'"

    def test_semicolon_quoted(self):
        from src.routers.containers import _sanitize_shell_arg

        result = _sanitize_shell_arg("hello;rm -rf /")
        assert ";" not in result or result.startswith("'")

    def test_single_quote_escaped(self):
        from src.routers.containers import _sanitize_shell_arg

        result = _sanitize_shell_arg("it's")
        # shlex.quote escapes single quotes
        assert "it" in result
        assert result != "it's"  # Must be escaped


class TestValidateGitBranchName:
    def test_valid_branch(self):
        from src.routers.containers import _validate_git_branch_name

        assert _validate_git_branch_name("main") is True
        assert _validate_git_branch_name("feature/new-thing") is True
        assert _validate_git_branch_name("release_1-0") is True

    def test_empty_branch(self):
        from src.routers.containers import _validate_git_branch_name

        assert _validate_git_branch_name("") is False

    def test_length_capped_at_255_like_coding_github(self):
        from src.routers.containers import _validate_git_branch_name

        assert _validate_git_branch_name("a" * 255) is True
        assert _validate_git_branch_name("a" * 256) is False

    def test_dot_dot_rejected(self):
        from src.routers.containers import _validate_git_branch_name

        assert _validate_git_branch_name("../evil") is False
        assert _validate_git_branch_name("a..b") is False

    def test_leading_slash_rejected(self):
        from src.routers.containers import _validate_git_branch_name

        assert _validate_git_branch_name("/leading") is False

    def test_trailing_slash_rejected(self):
        from src.routers.containers import _validate_git_branch_name

        assert _validate_git_branch_name("trailing/") is False

    def test_branch_with_hyphen(self):
        from src.routers.containers import _validate_git_branch_name

        assert _validate_git_branch_name("fix-bug-123") is True
        assert _validate_git_branch_name("feature/add-auth") is True

    def test_branch_with_underscore(self):
        from src.routers.containers import _validate_git_branch_name

        assert _validate_git_branch_name("feature_branch") is True

    def test_branch_starting_with_hyphen(self):
        from src.routers.containers import _validate_git_branch_name

        # Branches starting with - could be interpreted as git flags
        assert _validate_git_branch_name("-dangerous") is False

    def test_none_branch(self):
        from src.routers.containers import _validate_git_branch_name

        assert _validate_git_branch_name(None) is False


class TestValidateGitFilePath:
    def test_valid_path(self):
        from src.routers.containers import _validate_git_file_path

        assert _validate_git_file_path("src/main.py") is True

    def test_empty_path(self):
        from src.routers.containers import _validate_git_file_path

        assert _validate_git_file_path("") is False

    def test_semicolon_rejected(self):
        from src.routers.containers import _validate_git_file_path

        assert _validate_git_file_path("file; rm -rf /") is False

    def test_pipe_rejected(self):
        from src.routers.containers import _validate_git_file_path

        assert _validate_git_file_path("file | cat") is False

    def test_null_byte_rejected(self):
        from src.routers.containers import _validate_git_file_path

        assert _validate_git_file_path("file\x00evil") is False

    def test_backtick_rejected(self):
        from src.routers.containers import _validate_git_file_path

        assert _validate_git_file_path("file`whoami`") is False

    def test_dollar_rejected(self):
        from src.routers.containers import _validate_git_file_path

        assert _validate_git_file_path("file$(rm -rf /)") is False

    def test_newline_rejected(self):
        from src.routers.containers import _validate_git_file_path

        assert _validate_git_file_path("file\nevil") is False

    def test_angle_bracket_rejected(self):
        from src.routers.containers import _validate_git_file_path

        assert _validate_git_file_path("file > /etc/passwd") is False

    def test_path_with_spaces_allowed(self):
        from src.routers.containers import _validate_git_file_path

        # Spaces are valid in file paths
        assert _validate_git_file_path("my file.py") is True

    def test_path_with_dots_allowed(self):
        from src.routers.containers import _validate_git_file_path

        assert _validate_git_file_path("src/config.json") is True
        assert _validate_git_file_path(".gitignore") is True


class TestSanitizeErrorMessage:
    def test_empty_passthrough(self):
        from src.routers.containers import _sanitize_error_message

        assert _sanitize_error_message("") == ""

    def test_no_token_passthrough(self):
        from src.routers.containers import _sanitize_error_message

        assert _sanitize_error_message("some error") == "some error"

    def test_redacts_explicit_token(self):
        from src.routers.containers import _sanitize_error_message

        result = _sanitize_error_message("failed with token mysecret123", token="mysecret123")
        assert "mysecret123" not in result
        assert "[REDACTED]" in result

    def test_redacts_ghp_pattern(self):
        from src.routers.containers import _sanitize_error_message

        token = "ghp_" + "a" * 36
        result = _sanitize_error_message(f"error: {token}")
        assert token not in result
        assert "[REDACTED]" in result

    def test_redacts_github_pat_pattern(self):
        from src.routers.containers import _sanitize_error_message

        token = "github_pat_" + "a" * 30
        result = _sanitize_error_message(f"error: {token}")
        assert token not in result

    def test_redacts_url_with_token(self):
        from src.routers.containers import _sanitize_error_message

        result = _sanitize_error_message("push to https://user:token@github.com/repo failed")
        assert "user:token" not in result

    def test_redacts_gho_pattern(self):
        from src.routers.containers import _sanitize_error_message

        token = "gho_" + "a" * 36
        result = _sanitize_error_message(f"error: {token}")
        assert token not in result
        assert "[REDACTED]" in result

    def test_none_error_passthrough(self):
        from src.routers.containers import _sanitize_error_message

        # empty string check
        assert _sanitize_error_message("", token="secret") == ""

    def test_multiple_tokens_redacted(self):
        from src.routers.containers import _sanitize_error_message

        msg = f"first ghp_{'a' * 36} and second ghp_{'b' * 36}"
        result = _sanitize_error_message(msg)
        assert "ghp_" not in result


class TestFindContainerByName:
    def test_found(self):
        from src.routers.containers import _find_container_by_name

        manager = MagicMock()
        container = MagicMock()
        container.container_name = "my-container"
        manager.get_container.return_value = container

        result = _find_container_by_name(manager, "my-container")
        assert result is container

    def test_not_found(self):
        from src.routers.containers import _find_container_by_name

        manager = MagicMock()
        manager.get_container.side_effect = ValueError("not found")

        result = _find_container_by_name(manager, "missing")
        assert result is None

    def test_runtime_error(self):
        from src.routers.containers import _find_container_by_name

        manager = MagicMock()
        manager.get_container.side_effect = RuntimeError("docker crash")

        result = _find_container_by_name(manager, "broken")
        assert result is None


# ---- endpoint tests ----


@pytest.mark.asyncio
class TestListContainersEndpoint:
    async def test_admin_sees_all(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user(is_admin=True)
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        c1 = _make_container_info("c1")
        c2 = _make_container_info("c2")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.list_containers.return_value = [c1, c2]
            mock_mgr.return_value = manager
            response = await client.get("/api/containers/")

        app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2

    async def test_invalid_status_filter(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.list_containers.return_value = []
            mock_mgr.return_value = manager
            response = await client.get("/api/containers/?status_filter=bogus")

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_empty_status_filter(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.list_containers.return_value = []
            mock_mgr.return_value = manager
            response = await client.get("/api/containers/?status_filter=%20")

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_status_filter_running(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        running = _make_container_info("r1", "running")
        stopped = _make_container_info("s1", "stopped")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.list_containers.return_value = [running, stopped]
            mock_mgr.return_value = manager
            response = await client.get("/api/containers/?status_filter=running")

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert response.json()["total"] == 1


@pytest.mark.asyncio
class TestGetContainerDetails:
    async def test_found(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("mycontainer")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            mock_mgr.return_value = manager
            response = await client.get("/api/containers/mycontainer")

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert response.json()["container"]["container_name"] == "mycontainer"

    async def test_not_found(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.side_effect = ValueError("not found")
            mock_mgr.return_value = manager
            response = await client.get("/api/containers/nope")

        app.dependency_overrides.clear()
        assert response.status_code == 404


@pytest.mark.asyncio
class TestGetContainerStatus:
    async def test_found(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("mycontainer")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.get_container_status = AsyncMock(return_value="running")
            mock_mgr.return_value = manager
            response = await client.get("/api/containers/mycontainer/status")

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert response.json()["container_status"] == "running"

    async def test_foreign_tenant_reads_404_not_readiness(self, app, client):
        """A caller outside the owning tenant must see a foreign workspace as ABSENT.

        ⚠️ This test replaces ``test_cross_tenant_readiness_poll_reads_status``,
        which asserted the OPPOSITE (200) and was the regression guard for #66.
        It is inverted deliberately, because #66's premise was refuted in
        production on 2026-09-17:

            11:03:38 GET  /api/workspaces/aa-delivery-chat-bridge/status
                          tenant=…0003 -> 200 OK
            11:03:45 POST /api/workspaces/aa-delivery-chat-bridge/exec
                          tenant=…0003 -> 404 Container not found
            11:15:32 POST /api/workspaces/aa-delivery-chat-bridge/exec
                          tenant=…0001 -> exit_code=0

        The Deployment carries ``tenant-id: …0001``. #66 reasoned that /status is
        "non-sensitive" because its BODY carries no secrets — but the disclosure
        is the EXISTENCE and READINESS of a name, and the 200 was actionable: the
        caller in …0003 adopted a name-colliding workspace it could never exec
        into rather than provisioning its own, and the card finished with a null
        exit code. An open /status in front of a closed /exec is not a relaxed
        gate, it is cross-tenant aliasing.

        ``_get_container`` below returns the container for ``tenant_id=None``, so
        the test FAILS (200) the moment a ``tenant_id=None`` fallback is
        reintroduced at this route. That is the mutation control, standing.
        """
        from coding_client.auth.tenant import get_current_user_with_tenant

        # Non-admin caller on a DIFFERENT tenant than the container owner.
        mock_user = _make_mock_user(is_admin=False, tenant_id="caller-tenant-b")
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        owned = _make_container_info("cve-sweep-x", tenant_id="owner-tenant-a")

        def _get_container(name, tenant_id=None):
            # Strict gate: wrong tenant -> None. A tenant-agnostic read would hit
            # the container — which is exactly what must NOT happen here.
            return owned if tenant_id is None else None

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.side_effect = _get_container
            manager.get_container_status = AsyncMock(return_value="running")
            mock_mgr.return_value = manager
            response = await client.get("/api/containers/cve-sweep-x/status")

        app.dependency_overrides.clear()
        assert response.status_code == 404
        # Absent, not "forbidden" — a 403 would disclose that the name exists.
        assert "not found" in response.text.lower()
        # And the readiness read must never have been attempted for a foreign
        # workspace: a 404 that still probed the backend is a timing oracle.
        manager.get_container_status.assert_not_awaited()

    async def test_owning_tenant_still_reads_status(self, app, client):
        """Positive control for the test above: the OWNER gets 200, same route.

        Without this, the isolation assertion is satisfiable by a route that
        404s everybody — the failure mode that makes a green isolation test
        worthless.
        """
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user(is_admin=False, tenant_id="owner-tenant-a")
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        owned = _make_container_info("cve-sweep-x", tenant_id="owner-tenant-a")

        def _get_container(name, tenant_id=None):
            return owned if tenant_id in (None, "owner-tenant-a") else None

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.side_effect = _get_container
            manager.get_container_status = AsyncMock(return_value="running")
            mock_mgr.return_value = manager
            response = await client.get("/api/containers/cve-sweep-x/status")

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert response.json()["container_status"] == "running"

    async def test_admin_still_reads_foreign_status(self, app, client):
        """The escape hatch for a genuine fleet-wide readiness poller.

        Tightening this route removes the anonymous cross-tenant read, not the
        cross-tenant read. An internal poller that legitimately must see every
        tenant authenticates as admin and is widened to ``tenant_id=None`` by
        the SAME ``user.is_admin()`` branch ``/`` and ``/{name}`` already use —
        no second mechanism was invented for /status.
        """
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user(is_admin=True, tenant_id="caller-tenant-b")
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        owned = _make_container_info("cve-sweep-x", tenant_id="owner-tenant-a")

        def _get_container(name, tenant_id=None):
            return owned if tenant_id is None else None

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.side_effect = _get_container
            manager.get_container_status = AsyncMock(return_value="running")
            mock_mgr.return_value = manager
            response = await client.get("/api/containers/cve-sweep-x/status")

        app.dependency_overrides.clear()
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Fleet-wide tenant isolation over the workspace surface
# ---------------------------------------------------------------------------


# Every route below resolves the workspace through
# ``get_container(name, tenant_id=<caller>)`` and 404s on a falsy result, so one
# parametrisation covers the whole surface. /status was the ONLY member that
# opted out; the rest are asserted here so the next "just this one endpoint"
# exemption has to break a test that names all of them together.
_TENANT_SCOPED_WORKSPACE_ROUTES = [
    ("get", "/api/workspaces/{name}", None),
    ("get", "/api/workspaces/{name}/status", None),
    ("get", "/api/workspaces/{name}/git/status", None),
    ("get", "/api/workspaces/{name}/git/diff", None),
    ("post", "/api/workspaces/{name}/git/restore", {"files": ["x.py"]}),
    ("post", "/api/workspaces/{name}/git/branch", {"branch": "feat/x"}),
    ("post", "/api/workspaces/{name}/mkdir", {"path": "/workspace/x"}),
    (
        "post",
        "/api/workspaces/{name}/touch",
        {"path": "/workspace", "filename": "x", "extension": "txt"},
    ),
    (
        "post",
        "/api/workspaces/{name}/write-file",
        {"file_path": "/workspace/x.txt", "content": "x"},
    ),
    (
        "post",
        "/api/workspaces/{name}/move",
        {"source_path": "/workspace/a", "destination_path": "/workspace/b"},
    ),
    ("delete", "/api/workspaces/{name}", None),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "template", "body"), _TENANT_SCOPED_WORKSPACE_ROUTES)
async def test_foreign_tenant_sees_workspace_as_absent(app, client, method, template, body):
    """A workspace owned by tenant A reads 404 for tenant B on every scoped route.

    The manager stub hands the container back for ``tenant_id=None`` only, i.e.
    it models a real, healthy, EXISTING workspace owned by somebody else. A
    route that drops its tenant argument therefore gets a 200 here and fails —
    this is the mutation control for the whole table, not just for /status.
    """
    from coding_client.auth.tenant import get_current_user_with_tenant

    mock_user = _make_mock_user(is_admin=False, tenant_id="caller-tenant-b")
    app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

    owned = _make_container_info("foreign-ws", tenant_id="owner-tenant-a")

    def _get_container(name, tenant_id=None):
        return owned if tenant_id is None else None

    url = template.format(name="foreign-ws")
    with patch("src.routers.containers.get_container_manager") as mock_mgr:
        manager = MagicMock()
        manager.get_container.side_effect = _get_container
        manager.get_container_status = AsyncMock(return_value="running")
        manager.execute_command = AsyncMock(
            return_value={"success": True, "exit_code": 0, "stdout": "", "stderr": ""}
        )
        manager.remove_container = AsyncMock(return_value={"status": "success"})
        mock_mgr.return_value = manager
        response = await getattr(client, method)(url, **({"json": body} if body else {}))

    app.dependency_overrides.clear()
    assert response.status_code == 404, (
        f"{method.upper()} {url} disclosed a foreign tenant's workspace "
        f"(status {response.status_code})"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "template", "body"), _TENANT_SCOPED_WORKSPACE_ROUTES)
async def test_owning_tenant_is_not_404ed(app, client, method, template, body):
    """Positive control for the table above, in the same file and same shape.

    Proves each route's 404 is caused by the tenant mismatch and not by the
    stub, the URL, or a validation error that would 404 everybody.
    """
    from coding_client.auth.tenant import get_current_user_with_tenant

    mock_user = _make_mock_user(is_admin=False, tenant_id="owner-tenant-a")
    app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

    owned = _make_container_info("foreign-ws", tenant_id="owner-tenant-a")

    def _get_container(name, tenant_id=None):
        return owned if tenant_id in (None, "owner-tenant-a") else None

    url = template.format(name="foreign-ws")
    with patch("src.routers.containers.get_container_manager") as mock_mgr:
        manager = MagicMock()
        manager.get_container.side_effect = _get_container
        manager.get_container_status = AsyncMock(return_value="running")
        manager.execute_command = AsyncMock(
            return_value={"success": True, "exit_code": 0, "stdout": "", "stderr": ""}
        )
        manager.remove_container = AsyncMock(return_value={"status": "success"})
        mock_mgr.return_value = manager
        response = await getattr(client, method)(url, **({"json": body} if body else {}))

    app.dependency_overrides.clear()
    assert response.status_code != 404, (
        f"{method.upper()} {url} 404s its OWN tenant — the isolation assertion "
        f"above would pass vacuously"
    )


@pytest.mark.asyncio
class TestDeleteAllContainers:
    async def test_no_containers(self, app, client):
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.list_containers.return_value = []
            mock_mgr.return_value = manager
            response = await client.delete("/api/containers/all")

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert "No containers to delete" in response.json()["message"]

    async def test_successful_deletion(self, app, client):
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        c1 = _make_container_info("c1")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.list_containers.return_value = [c1]
            manager.remove_container = AsyncMock(return_value={"status": "success"})
            mock_mgr.return_value = manager
            response = await client.delete("/api/containers/all")

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert "Successfully removed" in response.json()["message"]

    async def test_partial_failure(self, app, client):
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        c1 = _make_container_info("c1")
        c2 = _make_container_info("c2")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.list_containers.return_value = [c1, c2]
            manager.remove_container = AsyncMock(
                side_effect=[
                    {"status": "success"},
                    {"status": "error", "message": "busy"},
                ]
            )
            mock_mgr.return_value = manager
            response = await client.delete("/api/containers/all")

        app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        assert "Failed to remove" in data["message"]

    async def test_all_fail(self, app, client):
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        c1 = _make_container_info("c1")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.list_containers.return_value = [c1]
            manager.remove_container.side_effect = RuntimeError("boom")
            mock_mgr.return_value = manager
            response = await client.delete("/api/containers/all")

        app.dependency_overrides.clear()
        assert response.status_code == 500


@pytest.mark.asyncio
class TestRemoveContainer:
    async def test_remove_success(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("killme")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.remove_container = AsyncMock(return_value={"status": "success"})
            mock_mgr.return_value = manager
            response = await client.delete("/api/containers/killme")

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert "successfully removed" in response.json()["message"]

    async def test_remove_surfaces_graph_deletion_in_details(self, app, client):
        """The DELETE body must carry the graph-deletion outcome, not just the log.

        `remove_container` returns `graph_deletion`, but the route used to drop
        it, so a caller could not distinguish "graph deleted" from "graph
        orphaned" — the pre-#89 teardown returned 200 while the graph survived.
        On the unfixed route `details` is None, so this fails on a real value
        assertion, not an exception.
        """
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        graph_deletion = {"success": True, "nodes_deleted": 133, "edges_deleted": 166}
        container = _make_container_info("graphy")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.remove_container = AsyncMock(
                return_value={"status": "success", "graph_deletion": graph_deletion}
            )
            mock_mgr.return_value = manager
            response = await client.delete("/api/containers/graphy")

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert response.json()["details"] == {"graph_deletion": graph_deletion}

    async def test_remove_omits_details_when_there_was_no_graph(self, app, client):
        """No graph identity => `details` stays None.

        Guards the other direction: reporting an empty deletion would assert
        that a graph was removed when none existed.
        """
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("graphless")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.remove_container = AsyncMock(return_value={"status": "success"})
            mock_mgr.return_value = manager
            response = await client.delete("/api/containers/graphless")

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert response.json()["details"] is None

    async def test_remove_not_found(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.side_effect = ValueError("not found")
            mock_mgr.return_value = manager
            response = await client.delete("/api/containers/missing")

        app.dependency_overrides.clear()
        assert response.status_code == 404

    async def test_remove_failure(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("stubborn")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.remove_container = AsyncMock(
                return_value={
                    "status": "error",
                    "message": "container in use",
                }
            )
            mock_mgr.return_value = manager
            response = await client.delete("/api/containers/stubborn")

        app.dependency_overrides.clear()
        assert response.status_code == 500


@pytest.mark.asyncio
class TestGetAllNotAllowed:
    async def test_get_all_405(self, app, client):
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        response = await client.get("/api/containers/all")

        app.dependency_overrides.clear()
        # The endpoint raises error_response with 405
        assert response.status_code == 405


# ===========================================================================
# mkdir endpoint
# ===========================================================================


@pytest.mark.asyncio
class TestCreateDirectory:
    async def test_success(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("mycontainer", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                return_value={"exit_code": 0, "stdout": "", "stderr": ""}
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/mycontainer/mkdir",
                json={"path": "/app/src/new_module"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert response.json()["path"] == "/app/src/new_module"

    async def test_container_not_found(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.side_effect = ValueError("not found")
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/nope/mkdir",
                json={"path": "/app/test"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 404

    async def test_container_not_running(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("stopped-ctr", "stopped")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/stopped-ctr/mkdir",
                json={"path": "/app/test"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_path_traversal_rejected(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/mkdir",
                json={"path": "/app/../../../etc"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_shell_metachar_rejected(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/mkdir",
                json={"path": "/app; rm -rf /"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_exec_failure(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                return_value={"exit_code": 1, "stderr": "permission denied"}
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/mkdir",
                json={"path": "/root/forbidden"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 500


# ===========================================================================
# touch (create file) endpoint
# ===========================================================================


@pytest.mark.asyncio
class TestCreateFile:
    async def test_success(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                return_value={"exit_code": 0, "stdout": "", "stderr": ""}
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/touch",
                json={"path": "/app/src", "filename": "main", "extension": "py"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert response.json()["path"] == "/app/src/main.py"

    async def test_container_not_found(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.side_effect = ValueError("not found")
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/nope/touch",
                json={"path": "/app/src", "filename": "main", "extension": "py"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 404

    async def test_container_stopped(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "stopped")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/touch",
                json={"path": "/app/src", "filename": "main", "extension": "py"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_path_traversal_in_filename(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/touch",
                json={"path": "/app/src", "filename": "../../../etc/passwd", "extension": "txt"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_extension_with_dot_stripped(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                return_value={"exit_code": 0, "stdout": "", "stderr": ""}
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/touch",
                json={"path": "/app/src", "filename": "test", "extension": ".py"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        # Extension dot is stripped: should be test.py not test..py
        assert response.json()["path"] == "/app/src/test.py"


# ===========================================================================
# write-file endpoint
# ===========================================================================


@pytest.mark.asyncio
class TestWriteFile:
    async def test_success(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                return_value={"exit_code": 0, "stdout": "", "stderr": ""}
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/write-file",
                json={"file_path": "/workspace/src/main.py", "content": "print('hello')"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        assert data["path"] == "/workspace/src/main.py"
        assert data["bytes_written"] == len(b"print('hello')")

    async def test_container_not_found(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.side_effect = ValueError("not found")
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/nope/write-file",
                json={"file_path": "/app/test.py", "content": "x=1"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 404

    async def test_container_not_running(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "stopped")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/write-file",
                json={"file_path": "/app/test.py", "content": "x=1"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_path_traversal_rejected(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/write-file",
                json={"file_path": "/app/../../../etc/shadow", "content": "bad"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_exec_failure(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                return_value={"exit_code": 1, "stderr": "disk full"}
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/write-file",
                json={"file_path": "/workspace/test.py", "content": "x=1"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 500


# ===========================================================================
# git/status endpoint
# ===========================================================================


@pytest.mark.asyncio
class TestGitStatus:
    async def test_success_with_changes(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                side_effect=[
                    # branch command
                    {"exit_code": 0, "stdout": "main\n"},
                    # git status --porcelain=v1 -z (NUL-terminated entries)
                    {"exit_code": 0, "stdout": "?? src/new.py\0M  src/main.py\0 D old.py\0"},
                ]
            )
            mock_mgr.return_value = manager
            response = await client.get("/api/containers/ctr/git/status")

        app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        assert data["branch"] == "main"
        assert "src/main.py" in data["modified_files"]
        assert "src/new.py" in data["new_files"]
        assert "old.py" in data["deleted_files"]

    async def test_clean_working_tree(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                side_effect=[
                    {"exit_code": 0, "stdout": "main\n"},
                    {"exit_code": 0, "stdout": "\n"},
                ]
            )
            mock_mgr.return_value = manager
            response = await client.get("/api/containers/ctr/git/status")

        app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        assert "no changes" in data["message"]

    async def test_container_not_found(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.side_effect = ValueError("not found")
            mock_mgr.return_value = manager
            response = await client.get("/api/containers/nope/git/status")

        app.dependency_overrides.clear()
        assert response.status_code == 404

    async def test_container_stopped(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "stopped")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            mock_mgr.return_value = manager
            response = await client.get("/api/containers/ctr/git/status")

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_git_command_failure(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                side_effect=[
                    {"exit_code": 0, "stdout": "main\n"},
                    {"exit_code": 1, "stderr": "not a git repo"},
                ]
            )
            mock_mgr.return_value = manager
            response = await client.get("/api/containers/ctr/git/status")

        app.dependency_overrides.clear()
        assert response.status_code == 500


# ===========================================================================
# git/commit endpoint
# ===========================================================================


@pytest.mark.asyncio
class TestGitCommit:
    async def test_success_all_files(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                side_effect=[
                    # branch
                    {"exit_code": 0, "stdout": "main\n"},
                    # status check (has changes)
                    {"exit_code": 0, "stdout": " M file.py\n"},
                    # git add -A
                    {"exit_code": 0, "stdout": ""},
                    # pre-commit formatting: staged names+status, formatter probe,
                    # the formatter itself, then re-staging the formatted files
                    {"exit_code": 0, "stdout": "M\tfile.py\n"},
                    {"exit_code": 0, "stdout": "pyproject.toml\npyproject.toml:tool.ruff\n"},
                    {"exit_code": 0, "stdout": ""},
                    {"exit_code": 0, "stdout": ""},
                    # git commit
                    {"exit_code": 0, "stdout": ""},
                    # git rev-parse --short HEAD
                    {"exit_code": 0, "stdout": "abc1234\n"},
                    # git diff-tree count
                    {"exit_code": 0, "stdout": "1\n"},
                ]
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/commit",
                json={"message": "test commit", "files": ["*"]},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        assert data["commit_sha"] == "abc1234"
        assert data["files_committed"] == 1

    async def test_commit_survives_a_failing_formatter(self, app, client):
        """A formatter that errors must NOT destroy a correct fix.

        The pre-commit formatting step exists because generated code was arriving
        functionally correct and unmergeable on style. It fails OPEN on purpose: the
        repo's CI stays the authority on style, and a missing or broken formatter must
        never cost us the commit. This pins that direction, because the failure mode of
        adding a gate is that the gate starts rejecting good work.
        """
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                side_effect=[
                    {"exit_code": 0, "stdout": "main\n"},
                    {"exit_code": 0, "stdout": " M file.py\n"},
                    {"exit_code": 0, "stdout": ""},
                    # staged names+status, then a formatter probe that finds ruff
                    {"exit_code": 0, "stdout": "M\tfile.py\n"},
                    {"exit_code": 0, "stdout": "pyproject.toml\npyproject.toml:tool.ruff\n"},
                    # the formatter itself EXPLODES
                    {"exit_code": 2, "stderr": "ruff: internal error"},
                    # git commit still runs
                    {"exit_code": 0, "stdout": ""},
                    {"exit_code": 0, "stdout": "abc1234\n"},
                    {"exit_code": 0, "stdout": "1\n"},
                ]
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/commit",
                json={"message": "test commit", "files": ["*"]},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert response.json()["commit_sha"] == "abc1234"

    async def test_commit_succeeds_when_no_formatter_is_detected(self, app, client):
        """A repo with no formatter config commits exactly as it did before."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                side_effect=[
                    {"exit_code": 0, "stdout": "main\n"},
                    {"exit_code": 0, "stdout": " M notes.rst\n"},
                    {"exit_code": 0, "stdout": ""},
                    # staged names+status, then a probe that finds NO config at all
                    {"exit_code": 0, "stdout": "M\tnotes.rst\n"},
                    {"exit_code": 0, "stdout": ""},
                    # straight to commit - no formatter call, no re-stage
                    {"exit_code": 0, "stdout": ""},
                    {"exit_code": 0, "stdout": "def5678\n"},
                    {"exit_code": 0, "stdout": "1\n"},
                ]
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/commit",
                json={"message": "test commit", "files": ["*"]},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert response.json()["commit_sha"] == "def5678"

    async def test_commit_survives_formatting_raising_an_unexpected_exception(self, app, client):
        """A formatting step that RAISES (not just a non-zero exit) must not abort the
        commit either. The narrow except clauses on execute_command/this endpoint do not
        cover every exception a real container backend can raise - a Kubernetes
        ApiException is the motivating case - and letting one propagate through the bare
        `await _format_staged_files(...)` call would abort the commit, contradicting the
        documented fail-open contract.
        """
        from coding_client.auth.tenant import get_current_user_with_tenant

        class _FakeApiError(Exception):
            """Stands in for an exception type none of this module's except tuples name."""

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with (
            patch("src.routers.containers.get_container_manager") as mock_mgr,
            patch(
                "src.routers.containers.git_status_commit._format_staged_files",
                AsyncMock(side_effect=_FakeApiError("kube api exploded")),
            ),
        ):
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                side_effect=[
                    {"exit_code": 0, "stdout": "main\n"},
                    {"exit_code": 0, "stdout": " M file.py\n"},
                    {"exit_code": 0, "stdout": ""},  # git add
                    # _format_staged_files is patched to raise directly, so no calls
                    # are consumed for it here.
                    {"exit_code": 0, "stdout": ""},  # git commit
                    {"exit_code": 0, "stdout": "abc9999\n"},
                    {"exit_code": 0, "stdout": "1\n"},
                ]
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/commit",
                json={"message": "test commit", "files": ["*"]},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert response.json()["commit_sha"] == "abc9999"

    async def test_no_changes_400(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                side_effect=[
                    # branch
                    {"exit_code": 0, "stdout": "main\n"},
                    # status check (no changes)
                    {"exit_code": 0, "stdout": "\n"},
                ]
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/commit",
                json={"message": "empty commit"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_container_not_found(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.side_effect = ValueError("not found")
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/nope/git/commit",
                json={"message": "test"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 404

    async def test_specific_files(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                side_effect=[
                    # branch
                    {"exit_code": 0, "stdout": "dev\n"},
                    # status check
                    {"exit_code": 0, "stdout": " M src/a.py\n"},
                    # git add specific files
                    {"exit_code": 0, "stdout": ""},
                    # pre-commit formatting: staged names+status, formatter probe,
                    # the formatter itself, then re-staging the formatted files
                    {"exit_code": 0, "stdout": "M\tsrc/a.py\n"},
                    {"exit_code": 0, "stdout": "pyproject.toml\npyproject.toml:tool.ruff\n"},
                    {"exit_code": 0, "stdout": ""},
                    {"exit_code": 0, "stdout": ""},
                    # git commit
                    {"exit_code": 0, "stdout": ""},
                    # sha
                    {"exit_code": 0, "stdout": "def5678\n"},
                    # count
                    {"exit_code": 0, "stdout": "1\n"},
                ]
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/commit",
                json={"message": "selective commit", "files": ["src/a.py"]},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert response.json()["branch"] == "dev"

    async def test_leading_dash_file_path_rejected_with_400(self, app, client):
        """`request.files` entries are caller-supplied, unlike the staged paths
        `_format_staged_files` discovers internally. `-f` passes
        `_validate_git_file_path` (no shell metacharacters) and `shlex.quote` leaves it
        UNQUOTED, so an unguarded `git add -f .env` would force-stage a gitignored
        secret. A caller that asks for a dashed path must get a 400, not a silent drop.
        """
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                side_effect=[
                    {"exit_code": 0, "stdout": "main\n"},
                    {"exit_code": 0, "stdout": " M .env\n"},
                ]
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/commit",
                json={"message": "test commit", "files": ["-f", ".env"]},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400
        # git add was never called with the dashed argument.
        for call in manager.execute_command.call_args_list:
            assert " add " not in call.args[1]

    async def test_with_custom_author(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                side_effect=[
                    {"exit_code": 0, "stdout": "main\n"},
                    {"exit_code": 0, "stdout": " M x.py\n"},
                    {"exit_code": 0, "stdout": ""},
                    # pre-commit formatting: staged names+status, formatter probe,
                    # the formatter itself, then re-staging the formatted files
                    {"exit_code": 0, "stdout": "M\tx.py\n"},
                    {"exit_code": 0, "stdout": "pyproject.toml\npyproject.toml:tool.ruff\n"},
                    {"exit_code": 0, "stdout": ""},
                    {"exit_code": 0, "stdout": ""},
                    {"exit_code": 0, "stdout": ""},
                    {"exit_code": 0, "stdout": "aaa1111\n"},
                    {"exit_code": 0, "stdout": "1\n"},
                ]
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/commit",
                json={
                    "message": "authored commit",
                    "author_name": "John",
                    "author_email": "john@example.com",
                },
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200


# ===========================================================================
# _format_staged_files (unit-level, direct calls)
# ===========================================================================


@pytest.mark.asyncio
class TestFormatStagedFiles:
    async def test_leading_dash_filename_is_skipped(self):
        """A staged path whose basename starts with '-' must never reach a formatter
        command, quoted or not - shlex.quote leaves a leading-dash argument like
        '--plugin=evil.js' unquoted, so it would otherwise be parsed as a flag.
        """
        from src.routers.containers.git_status_commit import _format_staged_files

        manager = MagicMock()
        manager.execute_command = AsyncMock(
            side_effect=[
                {"exit_code": 0, "stdout": "A\t--plugin=evil.js\nM\tnormal.js\n"},
                {"exit_code": 0, "stdout": "package.json\n"},
                {"exit_code": 0, "stdout": ""},  # prettier
                {"exit_code": 0, "stdout": ""},  # re-stage
            ]
        )

        outcome = await _format_staged_files(manager, "ctr", "tenant-1")

        assert outcome == "formatted with prettier"
        for call in manager.execute_command.call_args_list:
            assert "--plugin=evil.js" not in call.args[1]
        prettier_cmd = manager.execute_command.call_args_list[2].args[1]
        assert "normal.js" in prettier_cmd
        assert " -- " in prettier_cmd

    async def test_staged_deletion_is_excluded(self):
        """A staged deletion must be dropped before formatting/re-staging: formatting a
        deleted path errors, and `git add <deleted-path>` exits 128, either of which
        would otherwise poison the whole batch for one unrelated removal.
        """
        from src.routers.containers.git_status_commit import _format_staged_files

        manager = MagicMock()
        manager.execute_command = AsyncMock(
            side_effect=[
                {"exit_code": 0, "stdout": "D\tgone.py\nM\tkept.py\n"},
                {"exit_code": 0, "stdout": "pyproject.toml\npyproject.toml:tool.ruff\n"},
                {"exit_code": 0, "stdout": ""},  # ruff
                {"exit_code": 0, "stdout": ""},  # re-stage
            ]
        )

        outcome = await _format_staged_files(manager, "ctr", "tenant-1")

        assert outcome == "formatted with ruff"
        ruff_cmd = manager.execute_command.call_args_list[2].args[1]
        assert "gone.py" not in ruff_cmd
        assert "kept.py" in ruff_cmd
        restage_cmd = manager.execute_command.call_args_list[3].args[1]
        assert "gone.py" not in restage_cmd

    async def test_deletions_only_short_circuits_before_any_formatter_call(self):
        """When every staged change is a deletion there is nothing to format or
        re-stage, so the function must return without probing for a formatter at all.
        """
        from src.routers.containers.git_status_commit import _format_staged_files

        manager = MagicMock()
        manager.execute_command = AsyncMock(return_value={"exit_code": 0, "stdout": "D\tgone.py\n"})

        outcome = await _format_staged_files(manager, "ctr", "tenant-1")

        assert outcome == "no staged files to format (deletions only)"
        assert manager.execute_command.call_count == 1

    async def test_black_configured_repo_is_formatted_with_ruff_format(self):
        """black is no longer baked into the runtime image: a repo that configures
        `[tool.black]` (and not ruff) is formatted with black-compatible `ruff format`,
        carrying black's line-length over because ruff does not read `[tool.black]`.
        """
        from src.routers.containers.git_status_commit import _format_staged_files

        manager = MagicMock()
        manager.execute_command = AsyncMock(
            side_effect=[
                {"exit_code": 0, "stdout": "M\tpkg/mod.py\n"},
                {
                    "exit_code": 0,
                    "stdout": "pyproject.toml\npyproject.toml:tool.black\nblack-line-length:100\n",
                },
                {"exit_code": 0, "stdout": ""},  # ruff format
                {"exit_code": 0, "stdout": ""},  # re-stage
            ]
        )

        outcome = await _format_staged_files(manager, "ctr", "tenant-1")

        assert outcome == "formatted with ruff"
        fmt_cmd = manager.execute_command.call_args_list[2].args[1]
        assert "ruff format --config line-length=100 -- pkg/mod.py" in fmt_cmd
        assert "black" not in fmt_cmd

    async def test_black_configured_without_line_length_uses_ruff_default(self):
        """No line-length under `[tool.black]` means black's default (88), which is
        also ruff's - so no flag is passed. A non-numeric token is never interpolated.
        """
        from src.routers.containers.git_status_commit import _format_staged_files

        manager = MagicMock()
        manager.execute_command = AsyncMock(
            side_effect=[
                {"exit_code": 0, "stdout": "M\tmod.py\n"},
                {
                    "exit_code": 0,
                    "stdout": "pyproject.toml\npyproject.toml:tool.black\nblack-line-length:\n",
                },
                {"exit_code": 0, "stdout": ""},  # ruff format
                {"exit_code": 0, "stdout": ""},  # re-stage
            ]
        )

        outcome = await _format_staged_files(manager, "ctr", "tenant-1")

        assert outcome == "formatted with ruff"
        fmt_cmd = manager.execute_command.call_args_list[2].args[1]
        assert "cd /workspace && ruff format -- mod.py" in fmt_cmd

    async def test_ruff_config_wins_over_black_config(self):
        """A repo with both `[tool.ruff]` and `[tool.black]` is formatted with its own
        ruff config, never with black's line-length."""
        from src.routers.containers.git_status_commit import _format_staged_files

        manager = MagicMock()
        manager.execute_command = AsyncMock(
            side_effect=[
                {"exit_code": 0, "stdout": "M\tmod.py\n"},
                {
                    "exit_code": 0,
                    "stdout": "pyproject.toml\npyproject.toml:tool.ruff\n"
                    "pyproject.toml:tool.black\nblack-line-length:120\n",
                },
                {"exit_code": 0, "stdout": ""},  # ruff format
                {"exit_code": 0, "stdout": ""},  # re-stage
            ]
        )

        await _format_staged_files(manager, "ctr", "tenant-1")

        fmt_cmd = manager.execute_command.call_args_list[2].args[1]
        assert "cd /workspace && ruff format -- mod.py" in fmt_cmd
        assert "--config" not in fmt_cmd

    async def test_black_probe_runs_for_real_against_a_messy_pyproject(self, tmp_path):
        """The probe command is EXECUTED (sh + python3 tomllib), not mocked, against a
        `[tool.black]` with a trailing comment, the underscore spelling, string
        normalisation off and the magic trailing comma off. Then the resulting
        `ruff format` command is executed too, proving ruff accepts the overrides.
        """
        import subprocess

        from src.routers.containers.git_status_commit import _format_staged_files

        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname = 'x'\n\n"
            "[tool.black]\n"
            "line_length = 100  # was 88\n"
            "skip-string-normalization = true\n"
            "skip_magic_trailing_comma = true\n"
            "target-version = ['py311']\n\n"
            "[tool.other]\nline-length = 5\n"
        )
        src = "x = {'a': 1,}\ny = 'single quotes stay'\n"
        (tmp_path / "mod.py").write_text(src)

        def run_for_real(cmd: str) -> dict:
            proc = subprocess.run(
                ["sh", "-c", cmd.replace("/workspace", str(tmp_path))],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

        calls: list[str] = []

        async def execute(_container, cmd, tenant_id=None):
            calls.append(cmd)
            if len(calls) == 1:
                return {"exit_code": 0, "stdout": "M\tmod.py\n"}  # git diff --cached
            if len(calls) in (2, 3):
                return run_for_real(cmd)  # the probe, then ruff format
            return {"exit_code": 0, "stdout": ""}  # git add

        manager = MagicMock()
        manager.execute_command = AsyncMock(side_effect=execute)

        outcome = await _format_staged_files(manager, "ctr", "tenant-1")

        assert outcome == "formatted with ruff"
        fmt_cmd = calls[2]
        assert "--config line-length=100 " in fmt_cmd  # not 10088
        assert "--config 'format.quote-style=\"preserve\"'" in fmt_cmd
        assert "--config format.skip-magic-trailing-comma=true" in fmt_cmd
        # ruff actually ran with those overrides: quotes preserved, and the magic
        # trailing comma ignored (collapsed onto one line).
        assert (tmp_path / "mod.py").read_text() == "x = {'a': 1}\ny = 'single quotes stay'\n"

    async def test_black_probe_ignores_garbage_values(self, tmp_path):
        """Non-int line-length / non-bool flags / invalid TOML yield no overrides."""
        import subprocess

        from src.routers.containers.git_status_commit import _BLACK_PROBE_PY

        def probe(text: str) -> str:
            (tmp_path / "pyproject.toml").write_text(text)
            return subprocess.run(
                ["python3", "-I", "-c", _BLACK_PROBE_PY],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout

        assert (
            probe(
                '[tool.black]\nline-length = "100; rm -rf /"\nskip-string-normalization = "yes"\n'
            )
            == ""
        )
        assert probe("[tool.black]\nline-length = true\n") == ""
        assert probe("[tool.black]\nline-length = 0\n") == ""
        assert probe("[tool.black\nnot toml") == ""
        assert probe("[tool.black]\nline-length = 120\n") == "black-line-length:120\n"

    async def test_overrides_never_echo_probe_text(self):
        from src.routers.containers.git_status_commit import _black_ruff_overrides

        assert _black_ruff_overrides({"black-line-length:100;rm", "black-line-length:9999"}) == []
        assert _black_ruff_overrides({"black-line-length:٣"}) == []  # non-ASCII digit
        assert _black_ruff_overrides({"black-line-length:120"}) == ["line-length=120"]

    async def test_prettier_runs_for_web_files(self):
        """The prettier/web path - previously untested - actually executes for a
        staged web-source file when package.json is present.
        """
        from src.routers.containers.git_status_commit import _format_staged_files

        manager = MagicMock()
        manager.execute_command = AsyncMock(
            side_effect=[
                {"exit_code": 0, "stdout": "M\tsrc/app.ts\n"},
                {"exit_code": 0, "stdout": "package.json\n"},
                {"exit_code": 0, "stdout": ""},  # prettier
                {"exit_code": 0, "stdout": ""},  # re-stage
            ]
        )

        outcome = await _format_staged_files(manager, "ctr", "tenant-1")

        assert outcome == "formatted with prettier"
        prettier_cmd = manager.execute_command.call_args_list[2].args[1]
        assert "prettier" in prettier_cmd
        assert "src/app.ts" in prettier_cmd
        assert " -- " in prettier_cmd


# ===========================================================================
# files endpoint (directory listing / file reading)
# ===========================================================================


# TestGetContainerFiles moved to tests/unit/test_workspace_files_read_search.py: the /files
# protocol is now one confined command (resolve + containment + read).

# ===========================================================================
# Delete scenarios
# ===========================================================================


@pytest.mark.asyncio
class TestRemoveContainerExtended:
    async def test_force_removal(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.remove_container = AsyncMock(return_value={"status": "success"})
            mock_mgr.return_value = manager
            response = await client.delete("/api/containers/ctr?force=true")

        app.dependency_overrides.clear()
        assert response.status_code == 200

    async def test_internal_error_on_exception(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.remove_container.side_effect = RuntimeError("unexpected")
            mock_mgr.return_value = manager
            response = await client.delete("/api/containers/ctr")

        app.dependency_overrides.clear()
        assert response.status_code == 500


# ===========================================================================
# Status scenarios
# ===========================================================================


@pytest.mark.asyncio
class TestGetContainerStatusExtended:
    async def test_not_found(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.side_effect = ValueError("not found")
            mock_mgr.return_value = manager
            response = await client.get("/api/containers/ghost/status")

        app.dependency_overrides.clear()
        assert response.status_code == 404

    async def test_unexpected_error(self, app, client):
        """get_container_status returns 500 on unexpected error."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.get_container_status.side_effect = RuntimeError("docker gone")
            mock_mgr.return_value = manager
            response = await client.get("/api/containers/ctr/status")

        app.dependency_overrides.clear()
        assert response.status_code == 500


# ===========================================================================
# create_container endpoint
# ===========================================================================


@pytest.mark.asyncio
class TestCreateContainerEndpoint:
    async def test_success(self, app, client):
        """POST /api/containers/ creates a container."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("new-repo-abc12345")
        with patch("src.services.docker_service.get_docker_service") as mock_get_ds:
            mock_ds = MagicMock()
            mock_ds.create_container = AsyncMock(return_value=container)
            mock_get_ds.return_value = mock_ds

            response = await client.post(
                "/api/containers/",
                json={"repo_url": "https://github.com/owner/repo", "branch": "main"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 201

    async def test_internal_error(self, app, client):
        """POST /api/containers/ returns 500 on failure."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        with patch("src.services.docker_service.get_docker_service") as mock_get_ds:
            mock_ds = MagicMock()
            mock_ds.create_container = AsyncMock(side_effect=RuntimeError("Docker unavailable"))
            mock_get_ds.return_value = mock_ds

            response = await client.post(
                "/api/containers/",
                json={"repo_url": "https://github.com/owner/repo", "branch": "main"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 500


# ===========================================================================
# get_container_details error path
# ===========================================================================


@pytest.mark.asyncio
class TestGetContainerDetailsError:
    async def test_unexpected_error(self, app, client):
        """GET /api/containers/{name} returns 500 on unexpected error."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            mock_mgr.side_effect = RuntimeError("database crash")
            response = await client.get("/api/containers/some-ctr")

        app.dependency_overrides.clear()
        assert response.status_code == 500


# ===========================================================================
# list_containers error path
# ===========================================================================


@pytest.mark.asyncio
class TestListContainersError:
    async def test_unexpected_error(self, app, client):
        """GET /api/containers/ returns 500 on unexpected error."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            mock_mgr.side_effect = RuntimeError("registry corrupt")
            response = await client.get("/api/containers/")

        app.dependency_overrides.clear()
        assert response.status_code == 500


# ===========================================================================
# git/commit error paths
# ===========================================================================


@pytest.mark.asyncio
class TestGitCommitExtended:
    async def test_container_stopped(self, app, client):
        """git/commit returns 400 when container is not running."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "stopped")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/commit",
                json={"message": "test"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_stage_failure(self, app, client):
        """git/commit returns 500 when git add fails."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                side_effect=[
                    {"exit_code": 0, "stdout": "main\n"},
                    {"exit_code": 0, "stdout": " M file.py\n"},
                    {"exit_code": 1, "stderr": "failed to stage"},  # git add fails
                ]
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/commit",
                json={"message": "test commit", "files": ["*"]},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 500

    async def test_commit_failure(self, app, client):
        """git/commit returns 500 when git commit fails."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                side_effect=[
                    {"exit_code": 0, "stdout": "main\n"},
                    {"exit_code": 0, "stdout": " M file.py\n"},
                    {"exit_code": 0, "stdout": ""},  # git add OK
                    # pre-commit formatting: staged names+status, formatter probe,
                    # the formatter itself, then re-staging the formatted files
                    {"exit_code": 0, "stdout": "M\ty.py\n"},
                    {"exit_code": 0, "stdout": "pyproject.toml\npyproject.toml:tool.ruff\n"},
                    {"exit_code": 0, "stdout": ""},
                    {"exit_code": 0, "stdout": ""},
                    {"exit_code": 1, "stderr": "commit hook failed"},  # commit fails
                ]
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/commit",
                json={"message": "test commit", "files": ["*"]},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 500

    async def test_invalid_file_path_rejected(self, app, client):
        """git/commit rejects files with dangerous characters."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                side_effect=[
                    {"exit_code": 0, "stdout": "main\n"},
                    {"exit_code": 0, "stdout": " M file.py\n"},
                ]
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/commit",
                json={"message": "evil", "files": ["file.py; rm -rf /"]},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400


# ===========================================================================
# git/push endpoint
# ===========================================================================


@pytest.mark.asyncio
class TestGitPush:
    @pytest.fixture(autouse=True)
    def _default_branch_is_not_the_pushed_one(self, monkeypatch):
        """These tests exercise credential/target handling, not the default-branch
        refusal (tests/unit/test_git_push_default_branch.py owns that), and push
        ``main``. Report a default branch no test here pushes to, so the route's
        GitHub lookup never leaves the process."""
        from src.routers.containers import git_push

        async def _fake(owner, repo, token):
            return git_push._RepoInfo(default_branch="trunk", private=True)

        async def _base(owner, repo, branch, default_branch, token):
            return "b" * 40

        monkeypatch.setattr(git_push, "_remote_repo_info", _fake)
        monkeypatch.setattr(git_push, "_push_base_sha", _base)

    async def test_container_not_found(self, app, client):
        """git/push returns 404 when container not found."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.side_effect = ValueError("not found")
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/nope/git/push",
                json={"branch": "main"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 404

    async def test_container_stopped(self, app, client):
        """git/push returns 400 when container is not running."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "stopped")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/push",
                json={"branch": "main"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_no_commits_to_push(self, app, client):
        """git/push returns 400 when no commits to push."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(return_value={"exit_code": 0, "stdout": "0\n"})
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/push",
                json={"branch": "main"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_no_remote_url_and_no_repo_url(self, app, client):
        """git/push returns 400 when no remote URL provided and container has none."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        # Remove repo_url from the container mock
        container.repo_url = None
        del container.repo_url

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(return_value={"exit_code": 0, "stdout": "1\n"})
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/push",
                json={"branch": "main"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_untrusted_remote_url_rejected_before_token_written(self, app, client):
        """git/push rejects a non-allowlisted remote_url (S1) before resolving or
        writing the GitHub token, so the token cannot be exfiltrated to it."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        container.github_access_token = "ghp_testtoken123"
        container.repo_url = "https://github.com/owner/repo"

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            # Only the unpushed-count command should run; the request must be
            # rejected before any remote/credential-helper/token-file command.
            manager.execute_command = AsyncMock(return_value={"exit_code": 0, "stdout": "2\n"})
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/push",
                json={"branch": "main", "remote_url": "https://attacker.example.com/x.git"},
            )
            # Assert no token file was ever written and no credential helper configured.
            executed = [c.args[1] for c in manager.execute_command.call_args_list]
            assert not any("credential.helper" in cmd for cmd in executed)
            assert not any(".git_cred_" in cmd for cmd in executed)
            assert not any(" remote add " in cmd for cmd in executed)

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_no_github_token(self, app, client):
        """PLATFORM owner, no token anywhere: typed 412, never a PAT and never a 401."""
        import os

        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        container.github_access_token = None
        container.repo_url = "https://github.com/nonagenticai/repo"

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(return_value={"exit_code": 0, "stdout": "1\n"})
            mock_mgr.return_value = manager

            from src.clients.github_client import CloneCredential

            tm = self._push_token_manager(
                CloneCredential(ok=True, token=None, visibility="private", reason="not_installed")
            )
            with (
                patch.dict(os.environ, {}, clear=False),
                patch("src.routers.containers.git_push.get_github_token_manager", return_value=tm),
            ):
                os.environ.pop("GITHUB_ACCESS_TOKEN", None)
                response = await client.post(
                    "/api/containers/ctr/git/push",
                    json={"branch": "main"},
                )

        app.dependency_overrides.clear()
        assert response.status_code == 412, response.text
        assert response.json()["error"]["message"].startswith("platform_credential_unavailable")

    @staticmethod
    def _push_token_manager(credential):
        from src.clients.github_client import GitHubTokenManager

        mgr = MagicMock(spec=GitHubTokenManager)
        mgr.get_app_clone_credential = AsyncMock(return_value=credential)
        return mgr

    @pytest.mark.parametrize(
        ("reason", "error_code"),
        [
            ("not_installed", "github_app_not_installed"),
            ("tenant_required", "tenant_required"),
            ("mint_failed", "github_app_push_credential_unavailable"),
        ],
    )
    async def test_customer_owner_never_pushes_with_the_env_pat(
        self, app, client, reason, error_code
    ):
        """Customer owner, no BYO token, coding-github mints no push token: 412
        carrying coding-github's reason.

        The env PAT is SET, so a regression that re-opens the fallback has a
        credential to reach for and turns this red.
        """
        from coding_client.auth.tenant import get_current_user_with_tenant
        from src.clients.github_client import CloneCredential

        pat = "ghp_SYSTEM_PAT_must_never_touch_a_customer_repo"  # pragma: allowlist secret
        app.dependency_overrides[get_current_user_with_tenant] = lambda: _make_mock_user()
        container = _make_container_info("ctr", "running", tenant_id="tenant-7")
        container.github_access_token = None
        container.secret_config_id = None
        container.repo_url = "https://github.com/acme-corp/repo"
        tm = self._push_token_manager(
            CloneCredential(ok=True, token=None, visibility="private", reason=reason)
        )

        with (
            patch("src.routers.containers.get_container_manager") as mock_mgr,
            patch("src.routers.containers.git_push.get_github_token_manager", return_value=tm),
            patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": pat}, clear=False),
        ):
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(return_value={"exit_code": 0, "stdout": "1\n"})
            mock_mgr.return_value = manager
            response = await client.post("/api/containers/ctr/git/push", json={"branch": "main"})

        app.dependency_overrides.clear()
        assert response.status_code == 412
        # The app's error envelope surfaces only the message; it must name the
        # coding-github reason's error code and the owner.
        assert response.json()["error"]["message"].startswith(f"{error_code} for acme-corp")
        # The push credential was requested with scope=push for the container's tenant.
        call = tm.get_app_clone_credential.await_args
        assert call.kwargs["scope"] == "push"
        # Admin caller: the tenant comes from the container, not the caller.
        assert call.kwargs["tenant_id"] == "tenant-7"
        # Only the unpushed-count probe ran; the PAT was never written anywhere.
        assert manager.execute_command.await_count == 1
        assert all(pat not in str(c) for c in manager.execute_command.await_args_list)

    async def test_customer_owner_pushes_with_the_tenant_bound_app_token(self, app, client, caplog):
        from coding_client.auth.tenant import get_current_user_with_tenant
        from src.clients.github_client import CloneCredential

        pat = "ghp_SYSTEM_PAT_must_never_touch_a_customer_repo"  # pragma: allowlist secret
        app_token = "ghs_tenantbound_push"  # noqa: S105
        app.dependency_overrides[get_current_user_with_tenant] = lambda: _make_mock_user(
            is_admin=False, tenant_id="tenant-7"
        )
        container = _make_container_info("ctr", "running", tenant_id="tenant-7")
        container.github_access_token = None
        container.secret_config_id = None
        container.repo_url = "https://github.com/acme-corp/repo"
        tm = self._push_token_manager(
            CloneCredential(ok=True, token=app_token, visibility="private", source="app")
        )

        with (
            patch("src.routers.containers.get_container_manager") as mock_mgr,
            patch("src.routers.containers.git_push.get_github_token_manager", return_value=tm),
            patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": pat}, clear=False),
        ):
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(side_effect=_push_exec())
            mock_mgr.return_value = manager
            response = await client.post("/api/containers/ctr/git/push", json={"branch": "main"})

        app.dependency_overrides.clear()
        assert response.status_code == 200
        tm.get_app_clone_credential.assert_awaited_once_with(
            "https://github.com/acme-corp/repo", tenant_id="tenant-7", scope="push", branch="main"
        )
        executed = [str(c) for c in manager.execute_command.await_args_list]
        assert any(app_token in c for c in executed)
        assert all(pat not in c for c in executed)
        assert app_token not in caplog.text

    async def test_customer_owner_with_byo_token_still_pushes(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        app.dependency_overrides[get_current_user_with_tenant] = lambda: _make_mock_user()
        container = _make_container_info("ctr", "running")
        container.github_access_token = "ghp_customer_byo"
        container.repo_url = "https://github.com/acme-corp/repo"
        pat = "ghp_SENTINEL_env_pat_must_never_push"  # pragma: allowlist secret

        with (
            patch("src.routers.containers.get_container_manager") as mock_mgr,
            patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": pat}, clear=False),
        ):
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(side_effect=_push_exec())
            mock_mgr.return_value = manager
            response = await client.post("/api/containers/ctr/git/push", json={"branch": "main"})

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert any("ghp_customer_byo" in str(c) for c in manager.execute_command.await_args_list)
        assert all(pat not in str(c) for c in manager.execute_command.await_args_list)

    @pytest.mark.parametrize("reason", ["not_installed", "mint_failed", "tenant_required", None])
    async def test_platform_owner_never_pushes_with_an_env_pat(self, app, client, reason):
        """Platform owner, coding-github issues no App token (any reason, or none):
        typed 412. A sentinel is SET in every PAT env var, so a regression that
        re-opens an env fallback has a credential to reach for and turns this red."""
        from coding_client.auth.tenant import get_current_user_with_tenant
        from src.clients.github_client import CloneCredential

        pat = "ghp_SENTINEL_env_pat_must_never_push"  # pragma: allowlist secret
        app.dependency_overrides[get_current_user_with_tenant] = lambda: _make_mock_user()
        container = _make_container_info("ctr", "running", tenant_id="tenant-internal")
        container.github_access_token = None
        container.secret_config_id = None
        container.repo_url = "https://github.com/nonagenticai/repo"
        tm = self._push_token_manager(
            CloneCredential(ok=True, token=None, visibility="private", reason=reason)
        )
        env = {"GITHUB_ACCESS_TOKEN": pat, "GITHUB_TOKEN": pat, "GITHUB_SYSTEM_TOKEN": pat}

        with (
            patch("src.routers.containers.get_container_manager") as mock_mgr,
            patch("src.routers.containers.git_push.get_github_token_manager", return_value=tm),
            patch("src.routers.containers.git_push._remote_repo_info") as repo_info,
            patch.dict(os.environ, env, clear=False),
        ):
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(side_effect=_push_exec())
            mock_mgr.return_value = manager
            response = await client.post("/api/containers/ctr/git/push", json={"branch": "main"})

        app.dependency_overrides.clear()
        assert response.status_code == 412, response.text
        assert response.json()["error"]["message"].startswith(
            "platform_credential_unavailable for nonagenticai"
        )
        tm.get_app_clone_credential.assert_awaited_once_with(
            "https://github.com/nonagenticai/repo",
            tenant_id="tenant-internal",
            scope="push",
            branch="main",
        )
        # Refused before any GitHub read or push exec: only the unpushed-count probe ran.
        repo_info.assert_not_called()
        assert manager.execute_command.await_count == 1
        assert all(pat not in str(c) for c in manager.execute_command.await_args_list)

    async def test_platform_owner_tenantless_push_refused_locally(self, app, client, caplog):
        """Admin caller (tenant_id=None) + legacy tenantless row + PLATFORM repo:
        refused HERE — coding-github is never asked (no tenant=None forwarded),
        no GitHub read, no push exec, and the refusal is logged."""
        import logging

        from coding_client.auth.tenant import get_current_user_with_tenant
        from src.clients.github_client import CloneCredential

        app.dependency_overrides[get_current_user_with_tenant] = lambda: _make_mock_user()
        container = _make_container_info("ctr", "running", tenant_id=None)
        container.github_access_token = None
        container.secret_config_id = None
        container.repo_url = "https://github.com/nonagenticai/repo"
        tm = self._push_token_manager(
            CloneCredential(ok=True, token="ghs_must_not_be_minted", visibility="private")
        )

        with (
            patch("src.routers.containers.get_container_manager") as mock_mgr,
            patch("src.routers.containers.git_push.get_github_token_manager", return_value=tm),
            patch("src.routers.containers.git_push._remote_repo_info") as repo_info,
            caplog.at_level(logging.WARNING),
        ):
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(side_effect=_push_exec())
            mock_mgr.return_value = manager
            response = await client.post("/api/containers/ctr/git/push", json={"branch": "main"})

        app.dependency_overrides.clear()
        assert response.status_code == 412, response.text
        assert response.json()["error"]["message"].startswith("platform_tenant_missing")
        tm.get_app_clone_credential.assert_not_called()
        repo_info.assert_not_called()
        # Only the credential-less unpushed-count probe ran.
        assert manager.execute_command.await_count == 1
        assert "event=platform_tenant_missing_refused" in caplog.text
        assert "context=git_push" in caplog.text

    async def test_customer_owner_tenantless_push_unchanged(self, app, client, caplog):
        """Customer owner with tenant None is NOT refused by the platform guard:
        coding-github is still asked (tenant=None) and its reason surfaces."""
        import logging

        from coding_client.auth.tenant import get_current_user_with_tenant
        from src.clients.github_client import CloneCredential

        app.dependency_overrides[get_current_user_with_tenant] = lambda: _make_mock_user()
        container = _make_container_info("ctr", "running", tenant_id=None)
        container.github_access_token = None
        container.secret_config_id = None
        container.repo_url = "https://github.com/acme-corp/repo"
        tm = self._push_token_manager(
            CloneCredential(ok=True, token=None, visibility="private", reason="tenant_required")
        )

        with (
            patch("src.routers.containers.get_container_manager") as mock_mgr,
            patch("src.routers.containers.git_push.get_github_token_manager", return_value=tm),
            caplog.at_level(logging.WARNING),
        ):
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(return_value={"exit_code": 0, "stdout": "1\n"})
            mock_mgr.return_value = manager
            response = await client.post("/api/containers/ctr/git/push", json={"branch": "main"})

        app.dependency_overrides.clear()
        assert response.status_code == 412
        assert response.json()["error"]["message"].startswith("tenant_required for acme-corp")
        tm.get_app_clone_credential.assert_awaited_once_with(
            "https://github.com/acme-corp/repo", tenant_id=None, scope="push", branch="main"
        )
        assert "platform_tenant_missing_refused" not in caplog.text

    async def test_platform_owner_pushes_with_the_app_token_not_the_env_pat(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant
        from src.clients.github_client import CloneCredential

        pat = "ghp_SENTINEL_env_pat_must_never_push"  # pragma: allowlist secret
        app_token = "ghs_platform_app_push"  # noqa: S105
        app.dependency_overrides[get_current_user_with_tenant] = lambda: _make_mock_user()
        container = _make_container_info("ctr", "running", tenant_id="tenant-internal")
        container.github_access_token = None
        container.secret_config_id = None
        container.repo_url = "https://github.com/nonagenticai/repo"
        tm = self._push_token_manager(
            CloneCredential(ok=True, token=app_token, visibility="private", source="app")
        )

        with (
            patch("src.routers.containers.get_container_manager") as mock_mgr,
            patch("src.routers.containers.git_push.get_github_token_manager", return_value=tm),
            patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": pat}, clear=False),
        ):
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(side_effect=_push_exec())
            mock_mgr.return_value = manager
            response = await client.post("/api/containers/ctr/git/push", json={"branch": "main"})

        app.dependency_overrides.clear()
        assert response.status_code == 200, response.text
        calls = manager.execute_command.await_args_list
        push_calls = [c for c in calls if (c.kwargs.get("env") or {}).get("GIT_PUSH_TOKEN")]
        assert len(push_calls) == 1
        assert push_calls[0].kwargs["env"] == {"GIT_PUSH_TOKEN": app_token}
        assert all(pat not in str(c) for c in calls)

    @pytest.mark.parametrize("app_token", [None, "ghs_internal_app_token"])
    async def test_platform_repo_forbidden_never_falls_back_to_the_pat(
        self, app, client, app_token
    ):
        """coding-github #107 refuses this tenant the platform repo. The env PAT is
        SET, so a regression that answers the refusal with it turns this red."""
        from coding_client.auth.tenant import get_current_user_with_tenant
        from src.clients.github_client import CloneCredential

        pat = "ghp_SYSTEM_PAT_must_never_answer_a_refusal"  # pragma: allowlist secret
        app.dependency_overrides[get_current_user_with_tenant] = lambda: _make_mock_user(
            is_admin=False, tenant_id="customer-tenant"
        )
        container = _make_container_info("ctr", "running", tenant_id="customer-tenant")
        container.github_access_token = None
        container.secret_config_id = None
        container.repo_url = "https://github.com/nonagenticai/repo"
        tm = self._push_token_manager(
            CloneCredential(
                ok=True,
                token=app_token,
                visibility="private",
                reason="platform_repo_forbidden",
            )
        )

        with (
            patch("src.routers.containers.get_container_manager") as mock_mgr,
            patch("src.routers.containers.git_push.get_github_token_manager", return_value=tm),
            patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": pat}, clear=False),
        ):
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.has_container.return_value = True
            manager.execute_command = AsyncMock(side_effect=_push_exec())
            mock_mgr.return_value = manager
            response = await client.post("/api/containers/ctr/git/push", json={"branch": "main"})

        app.dependency_overrides.clear()
        assert response.status_code == 403, response.text
        assert response.json()["error"]["message"].startswith("platform_repo_forbidden")
        # Only the unpushed-count probe ran; no credential went anywhere.
        assert manager.execute_command.await_count == 1
        assert all(pat not in str(c) for c in manager.execute_command.await_args_list)

    # The credential ladder is SHARED with git/fetch (``resolve_repo_credential``), which
    # renders its own direction ("a fetch from") and verb. These pin the EXACT text a
    # push caller reads, so a refactor that leaks the fetch wording into a push goes red.
    @pytest.mark.parametrize(
        ("owner_repo", "tenant", "reason", "status", "message"),
        [
            (
                "acme-corp/repo",
                "tenant-7",
                "tenant_required",
                412,
                "tenant_required for acme-corp: a push to a customer repository needs the "
                "provisioning tenant, and none is known for this container. Re-provision "
                "it from the fr workspace with the nonagentic GitHub App installed on "
                "acme-corp, or provision it with a GitHub token as access_token.",
            ),
            (
                "acme-corp/repo",
                "tenant-7",
                "not_installed",
                412,
                "github_app_not_installed for acme-corp: the GitHub App is not installed "
                "for this tenant on that owner. Install the nonagentic GitHub App on "
                "acme-corp from the fr workspace, or provision this workspace with a "
                "GitHub token as access_token.",
            ),
            (
                "acme-corp/repo",
                "tenant-7",
                "mint_failed",
                412,
                "github_app_push_credential_unavailable for acme-corp: no BYO token is "
                "attached and coding-github issued no tenant-bound push credential "
                "(reason=mint_failed). Install the nonagentic GitHub App on acme-corp "
                "from the fr workspace, or provision this workspace with a GitHub token "
                "as access_token.",
            ),
            (
                "nonagenticai/repo",
                "customer-tenant",
                "platform_repo_forbidden",
                403,
                "platform_repo_forbidden for nonagenticai: platform repositories are "
                "internal-tenant only.",
            ),
        ],
    )
    async def test_push_credential_refusal_messages_are_pinned(
        self, app, client, owner_repo, tenant, reason, status, message
    ):
        from coding_client.auth.tenant import get_current_user_with_tenant
        from src.clients.github_client import CloneCredential

        app.dependency_overrides[get_current_user_with_tenant] = lambda: _make_mock_user(
            is_admin=False, tenant_id=tenant
        )
        container = _make_container_info("ctr", "running", tenant_id=tenant)
        container.github_access_token = None
        container.secret_config_id = None
        container.repo_url = f"https://github.com/{owner_repo}"
        tm = self._push_token_manager(
            CloneCredential(ok=True, token=None, visibility="private", reason=reason)
        )
        with (
            patch("src.routers.containers.get_container_manager") as mock_mgr,
            patch("src.routers.containers.git_push.get_github_token_manager", return_value=tm),
        ):
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.has_container.return_value = True
            manager.execute_command = AsyncMock(side_effect=_push_exec())
            mock_mgr.return_value = manager
            response = await client.post("/api/containers/ctr/git/push", json={"branch": "main"})

        app.dependency_overrides.clear()
        assert response.status_code == status, response.text
        assert response.json()["error"]["message"] == message

    # --- push target: canonical, and only the container's own origin ---------

    @pytest.mark.parametrize(
        ("remote_url", "status"),
        [
            # Dot segments: the raw first segment says "nonagenticai" (platform
            # PAT) while git resolves the path to acme-corp/other.
            ("https://github.com/nonagenticai/../acme-corp/other", 400),
            ("https://github.com/nonagenticai/%2e%2e/acme-corp/other", 400),
            ("https://github.com/nonagenticai/%2E%2E/acme-corp/other.git", 400),
            ("git@github.com:nonagenticai/../acme-corp/other.git", 400),
            ("https://github.com/nonagenticai/repo/extra", 400),
            ("https://www.github.com/acme-corp/repo", 400),
            ("https://x-access-token:t@github.com/acme-corp/repo", 400),
            # Canonical, but not this workspace's origin.
            ("https://github.com/nonagenticai/other-repo", 403),
            ("git@github.com:nonagenticai/other-repo.git", 403),
            ("https://github.com/acme-corp/other", 403),
        ],
    )
    async def test_push_target_other_than_origin_is_refused_before_any_credential(
        self, app, client, remote_url, status
    ):
        """The container was provisioned from acme-corp/repo. Every other target —
        or a string whose meaning to git could differ from its parse — is refused
        before a credential is resolved; the platform PAT is set so a regression
        that classifies ``nonagenticai`` from the raw string has it to reach for."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        pat = "ghp_SYSTEM_PAT_must_never_touch_a_customer_repo"  # pragma: allowlist secret
        app.dependency_overrides[get_current_user_with_tenant] = lambda: _make_mock_user()
        container = _make_container_info("ctr", "running", tenant_id="tenant-7")
        container.github_access_token = None
        container.secret_config_id = None
        container.repo_url = "https://github.com/acme-corp/repo"

        with (
            patch("src.routers.containers.get_container_manager") as mock_mgr,
            patch("src.routers.containers.git_push.get_github_token_manager") as tm_factory,
            patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": pat}, clear=False),
        ):
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(return_value={"exit_code": 0, "stdout": "1\n"})
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/push",
                json={"branch": "main", "remote_url": remote_url},
            )

        app.dependency_overrides.clear()
        assert response.status_code == status, response.text
        if status == 403:
            assert response.json()["error"]["message"].startswith("push_target_mismatch")
        # Only the unpushed-count probe ran: no remote, no token file, no helper.
        assert manager.execute_command.await_count == 1
        assert all(pat not in str(c) for c in manager.execute_command.await_args_list)
        tm_factory.assert_not_called()

    async def test_container_without_origin_refuses_an_explicit_remote(self, app, client):
        from coding_client.auth.tenant import get_current_user_with_tenant

        app.dependency_overrides[get_current_user_with_tenant] = lambda: _make_mock_user()
        container = _make_container_info("ctr", "running")
        container.github_access_token = "byo-token"
        container.repo_url = None

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(return_value={"exit_code": 0, "stdout": "1\n"})
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/push",
                json={"branch": "main", "remote_url": "https://github.com/acme-corp/repo"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 403
        assert manager.execute_command.await_count == 1

    @pytest.mark.parametrize(
        "remote_url",
        [
            "https://github.com/acme-corp/repo",
            "https://github.com/acme-corp/repo.git",
            "https://GitHub.com/ACME-corp/Repo/",
            "git@github.com:acme-corp/repo.git",
            "ssh://git@github.com/acme-corp/repo.git",
        ],
    )
    async def test_origin_in_any_canonical_form_pushes_to_the_rebuilt_url(
        self, app, client, remote_url
    ):
        """Same repo as the origin → allowed, and git is handed the URL rebuilt
        from (owner, repo), never the raw string."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        app.dependency_overrides[get_current_user_with_tenant] = lambda: _make_mock_user()
        container = _make_container_info("ctr", "running")
        container.github_access_token = "byo-token"
        container.repo_url = "https://github.com/acme-corp/repo"

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(side_effect=_push_exec())
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/push",
                json={"branch": "main", "remote_url": remote_url},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200, response.text
        owner, repo = ("ACME-corp", "Repo") if "GitHub.com" in remote_url else ("acme-corp", "repo")
        expected = f"https://github.com/{owner}/{repo}"
        remote_add = [
            c.args[1] for c in manager.execute_command.await_args_list if "remote add" in c.args[1]
        ]
        assert len(remote_add) == 1
        assert remote_add[0].endswith(f" remote add origin {expected} )")
        assert response.json()["remote_url"] == expected

    async def test_success(self, app, client):
        """git/push succeeds with valid token and commits."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        container.github_access_token = "ghp_testtoken123"
        container.repo_url = "https://github.com/owner/repo"

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(side_effect=_push_exec(unpushed="2\n"))
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/push",
                json={"branch": "feature/test"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        assert data["commits_pushed"] == 2
        # The push goes to the EXPLICIT canonical URL (never a remote name), with
        # system/global config switched off, and the token only in the exec env.
        calls = manager.execute_command.await_args_list
        # Every workspace git now exports the isolation env (coding-containers#156), so
        # the push exec is found by its explicit-URL push, not by GIT_CONFIG_NOSYSTEM.
        (push,) = [c for c in calls if " push --no-recurse-submodules https://" in c.args[1]]
        cmd = push.args[1]
        assert "GIT_CONFIG_NOSYSTEM=1" in cmd
        assert "GIT_CONFIG_GLOBAL=/dev/null" in cmd
        assert (
            f"push --no-recurse-submodules https://github.com/owner/repo "
            f"{_PUSH_TIP}:refs/heads/feature/test" in cmd
        )
        assert " push origin " not in cmd.split("; git -c", 1)[1]
        assert "-c credential.helper= " in cmd
        assert push.kwargs["env"] == {"GIT_PUSH_TOKEN": "ghp_testtoken123"}
        assert all("ghp_testtoken123" not in c.args[1] for c in calls)
        assert all("/tmp/" not in c.args[1] for c in calls)

    async def test_push_failure(self, app, client):
        """git/push returns 500 when git push fails."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        container.github_access_token = "ghp_testtoken123"
        container.repo_url = "https://github.com/owner/repo"

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                side_effect=_push_exec(
                    push={"exit_code": 1, "stderr": "rejected: non-fast-forward"}
                )
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/push",
                json={"branch": "main"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 500

    async def test_push_auth_failure(self, app, client):
        """git/push returns 401 when push fails with authentication error."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        container.github_access_token = "ghp_testtoken123"
        container.repo_url = "https://github.com/owner/repo"

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                side_effect=_push_exec(
                    push={"exit_code": 1, "stderr": "Authentication failed for remote"}
                )
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/push",
                json={"branch": "main"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 401

    async def test_invalid_branch_name(self, app, client):
        """git/push returns 400 for invalid branch name."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        container.github_access_token = "ghp_testtoken123"
        container.repo_url = "https://github.com/owner/repo"

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(side_effect=_push_exec())
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/push",
                json={"branch": "../../../etc/passwd"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_force_push(self, app, client):
        """git/push with force=True uses --force flag."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        container.github_access_token = "ghp_testtoken123"
        container.repo_url = "https://github.com/owner/repo"

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(side_effect=_push_exec())
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/push",
                json={"branch": "main", "force": True},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200

    async def test_ssh_remote_url(self, app, client):
        """git/push handles git@github.com:user/repo format."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        container.github_access_token = "ghp_testtoken123"
        container.repo_url = "git@github.com:owner/repo.git"

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(side_effect=_push_exec())
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/push",
                json={"branch": "main"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200

    async def test_remote_add_failure(self, app, client):
        """git/push returns 500 when git remote add fails."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        container.github_access_token = "ghp_testtoken123"
        container.repo_url = "https://github.com/owner/repo"

        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                side_effect=_push_exec(remote_add={"exit_code": 1, "stderr": "remote add failed"})
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/git/push",
                json={"branch": "main"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 500


# ===========================================================================
# files endpoint error paths
# ===========================================================================


# TestGetContainerFilesExtended moved to tests/unit/test_workspace_files_read_search.py: the /files
# protocol is now one confined command (resolve + containment + read).

# ===========================================================================
# Additional touch/mkdir/write-file error paths
# ===========================================================================


@pytest.mark.asyncio
class TestCreateDirectoryExtended:
    async def test_empty_path_after_strip(self, app, client):
        """mkdir returns 400 when path is empty after stripping."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/mkdir",
                json={"path": "   "},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400


@pytest.mark.asyncio
class TestCreateFileExtended:
    async def test_shell_metachar_in_extension(self, app, client):
        """touch returns 400 when extension contains shell metacharacters."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/touch",
                json={"path": "/app", "filename": "test", "extension": "py;evil"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_exec_failure(self, app, client):
        """touch returns 500 when exec fails."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(return_value={"exit_code": 1, "stderr": "no space"})
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/touch",
                json={"path": "/app", "filename": "test", "extension": "py"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 500


@pytest.mark.asyncio
class TestWriteFileExtended:
    async def test_write_without_parent_dir(self, app, client):
        """write-file works when file has no parent directory path."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            manager.execute_command = AsyncMock(
                return_value={"exit_code": 0, "stdout": "", "stderr": ""}
            )
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/write-file",
                json={"file_path": "test.py", "content": "x = 1"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200

    async def test_shell_metachar_in_path(self, app, client):
        """write-file returns 400 when path has shell metacharacters."""
        from coding_client.auth.tenant import get_current_user_with_tenant

        mock_user = _make_mock_user()
        app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user

        container = _make_container_info("ctr", "running")
        with patch("src.routers.containers.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container
            mock_mgr.return_value = manager
            response = await client.post(
                "/api/containers/ctr/write-file",
                json={"file_path": "/app/test$(evil).py", "content": "bad"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 400

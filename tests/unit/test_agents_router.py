"""Tests for routers/agents.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.agent_models import AgentExecutionResponse

pytestmark = pytest.mark.asyncio


def _runtime_patch(run_once):
    """Patch the runtime the capabilities route resolves; ``run_once`` is the spy."""
    import contextlib

    runtime = MagicMock()
    runtime.run_once = run_once
    stack = contextlib.ExitStack()
    stack.enter_context(patch("src.routers.agents.resolve_runtime", return_value=runtime))
    stack.enter_context(patch("src.core.container_manager_core.get_container_manager"))
    return stack


def _result(exit_code):
    from src.runtime.types import RunOnceResult

    return RunOnceResult(exit_code=exit_code, stdout="", stderr="", duration=0.01)


@pytest.fixture(autouse=True)
def _owns_workspace(grant_workspace_ownership):
    """Every test here predates #135's ownership gate and names a workspace it
    never provisioned. Ownership is granted so these keep testing the agent
    behaviour they were written for; the gate itself is tested in
    ``test_agent_exec_tenant_scope.py``, which does NOT use this fixture."""


def _make_mock_user():
    user = MagicMock()
    user.sub = "test-user"
    user.preferred_username = "testuser"
    user.effective_tenant_id = "test-tenant"
    user.is_admin.return_value = True
    return user


async def test_execute_agent_success(app, client):
    from coding_client.auth.tenant import get_current_user_with_tenant, require_tenant

    mock_user = _make_mock_user()
    app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user
    app.dependency_overrides[require_tenant] = lambda: "test-tenant"

    mock_response = AgentExecutionResponse(
        success=True,
        output="done",
        exit_code=0,
        execution_time=1.0,
        container_name="test-container",
    )

    with patch("src.routers.agents.get_docker_service") as mock_get_ds:
        mock_ds = MagicMock()
        mock_get_ds.return_value = mock_ds

        with patch(
            "src.services.agent_service.AgentService.execute_agent", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_response

            response = await client.post(
                "/api/containers/test-container/execute-agent",
                json={"cli": "claude", "prompt": "fix bug"},
            )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


async def test_check_capabilities(app, client):
    from coding_client.auth.tenant import get_current_user_with_tenant, require_tenant

    mock_user = _make_mock_user()
    app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user
    app.dependency_overrides[require_tenant] = lambda: "test-tenant"

    run_once = AsyncMock(return_value=_result(1))
    with _runtime_patch(run_once):
        response = await client.get("/api/containers/test-container/agent-capabilities")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert "capabilities" in data
    assert "claude" in data["capabilities"]
    assert "codex" in data["capabilities"]


async def test_execute_agent_container_not_found(app, client):
    """Execute agent returns 404 when container not found."""
    from coding_client.auth.tenant import get_current_user_with_tenant, require_tenant

    mock_user = _make_mock_user()
    app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user
    app.dependency_overrides[require_tenant] = lambda: "test-tenant"

    with patch("src.routers.agents.get_docker_service") as mock_get_ds:
        mock_ds = MagicMock()
        mock_get_ds.return_value = mock_ds

        with patch(
            "src.services.agent_service.AgentService.execute_agent", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.side_effect = ValueError("Container not found or not running")

            response = await client.post(
                "/api/containers/nonexistent/execute-agent",
                json={"cli": "claude", "prompt": "fix bug"},
            )

    app.dependency_overrides.clear()
    assert response.status_code == 404


async def test_execute_agent_internal_error(app, client):
    """Execute agent returns 500 on unexpected error."""
    from coding_client.auth.tenant import get_current_user_with_tenant, require_tenant

    mock_user = _make_mock_user()
    app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user
    app.dependency_overrides[require_tenant] = lambda: "test-tenant"

    with patch("src.routers.agents.get_docker_service") as mock_get_ds:
        mock_ds = MagicMock()
        mock_get_ds.return_value = mock_ds

        with patch(
            "src.services.agent_service.AgentService.execute_agent", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.side_effect = RuntimeError("Docker daemon crashed")

            response = await client.post(
                "/api/containers/test-container/execute-agent",
                json={"cli": "codex", "prompt": "add tests"},
            )

    app.dependency_overrides.clear()
    assert response.status_code == 500


async def test_execute_agent_failed_result(app, client):
    """Execute agent returns 200 with success=False when agent fails."""
    from coding_client.auth.tenant import get_current_user_with_tenant, require_tenant

    mock_user = _make_mock_user()
    app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user
    app.dependency_overrides[require_tenant] = lambda: "test-tenant"

    mock_response = AgentExecutionResponse(
        success=False,
        output="",
        error="Agent timed out",
        exit_code=1,
        execution_time=300.0,
        container_name="test-container",
    )

    with patch("src.routers.agents.get_docker_service") as mock_get_ds:
        mock_ds = MagicMock()
        mock_get_ds.return_value = mock_ds

        with patch(
            "src.services.agent_service.AgentService.execute_agent", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_response

            response = await client.post(
                "/api/containers/test-container/execute-agent",
                json={"cli": "claude", "prompt": "fix bug"},
            )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["exit_code"] == 1


async def test_execute_agent_no_user_state(app, client):
    """Execute agent works when request.state has no user attribute."""
    from coding_client.auth.tenant import get_current_user_with_tenant, require_tenant

    mock_user = _make_mock_user()
    app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user
    app.dependency_overrides[require_tenant] = lambda: "test-tenant"

    mock_response = AgentExecutionResponse(
        success=True,
        output="done",
        exit_code=0,
        execution_time=1.0,
        container_name="test-container",
    )

    with patch("src.routers.agents.get_docker_service") as mock_get_ds:
        mock_ds = MagicMock()
        mock_get_ds.return_value = mock_ds

        with patch(
            "src.services.agent_service.AgentService.execute_agent", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_response

            response = await client.post(
                "/api/containers/test-container/execute-agent",
                json={"cli": "claude", "prompt": "fix"},
            )

    app.dependency_overrides.clear()
    assert response.status_code == 200


async def test_check_capabilities_service_error(app, client):
    """Check capabilities returns fallback when service raises."""
    from coding_client.auth.tenant import get_current_user_with_tenant, require_tenant

    mock_user = _make_mock_user()
    app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user
    app.dependency_overrides[require_tenant] = lambda: "test-tenant"

    with (
        patch("src.routers.agents.resolve_runtime", side_effect=RuntimeError("no runtime")),
        patch("src.core.container_manager_core.get_container_manager"),
    ):
        response = await client.get("/api/containers/test-container/agent-capabilities")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert data["capabilities"]["claude"]["available"] is False
    assert data["capabilities"]["codex"]["available"] is False


async def test_check_capabilities_with_available_cli(app, client):
    """Capabilities check correctly reports available CLI tools."""
    from coding_client.auth.tenant import get_current_user_with_tenant, require_tenant

    mock_user = _make_mock_user()
    app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user
    app.dependency_overrides[require_tenant] = lambda: "test-tenant"

    run_once = AsyncMock(return_value=_result(0))
    with _runtime_patch(run_once):
        response = await client.get("/api/containers/test-container/agent-capabilities")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert data["capabilities"]["claude"]["available"] is True
    assert data["capabilities"]["codex"]["available"] is True


async def test_check_capabilities_per_cli_exception(app, client):
    """Capabilities check handles per-CLI check failure gracefully."""
    from coding_client.auth.tenant import get_current_user_with_tenant, require_tenant

    mock_user = _make_mock_user()
    app.dependency_overrides[get_current_user_with_tenant] = lambda: mock_user
    app.dependency_overrides[require_tenant] = lambda: "test-tenant"

    run_once = AsyncMock(side_effect=RuntimeError("container exec failed"))
    with _runtime_patch(run_once):
        response = await client.get("/api/containers/test-container/agent-capabilities")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    # Both should be unavailable due to per-CLI exception
    assert data["capabilities"]["claude"]["available"] is False
    assert data["capabilities"]["codex"]["available"] is False
    assert "Check failed" in data["capabilities"]["claude"]["note"]


async def test_check_capabilities_never_500s_on_a_bare_exception(app, client):
    """D4 (live 500 on cv-sweep-ems): the in-cluster Docker path raised a bare
    ``Exception("Docker not available")`` that escaped every handler. The route
    probes through the workspace runtime now, and a probe failure of ANY type is
    an "unavailable" answer, never a 500."""
    from coding_client.auth.tenant import get_current_user_with_tenant, require_tenant

    app.dependency_overrides[get_current_user_with_tenant] = lambda: _make_mock_user()
    app.dependency_overrides[require_tenant] = lambda: "test-tenant"
    run_once = AsyncMock(side_effect=Exception("Docker not available"))
    with (
        _runtime_patch(run_once),
        patch("src.routers.agents.get_docker_service", side_effect=AssertionError("docker")),
    ):
        response = await client.get("/api/workspaces/test-container/agent-capabilities")

    app.dependency_overrides.clear()
    assert response.status_code == 200, response.text
    caps = response.json()["capabilities"]
    assert caps["claude"]["available"] is False
    assert "Docker not available" in caps["claude"]["note"]
    argvs = [c.args[1] for c in run_once.await_args_list]
    assert argvs == [["which", "claude"], ["which", "codex"]]

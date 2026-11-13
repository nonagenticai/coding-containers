"""Tests for routers/container_exec.py."""

from unittest.mock import MagicMock, patch

import pytest

from src.routers.container_exec import check_dangerous_patterns, validate_command


class TestValidateCommand:
    def test_empty_command(self):
        assert validate_command([]) is False

    def test_allowed_command(self):
        assert validate_command(["ls", "-la"]) is True

    def test_python_allowed(self):
        assert validate_command(["python", "--version"]) is True

    def test_git_allowed(self):
        assert validate_command(["git", "status"]) is True

    def test_claude_allowed(self):
        assert validate_command(["claude", "-p", "fix"]) is True

    def test_disallowed_command(self):
        assert validate_command(["rm", "-rf", "/"]) is False

    def test_command_with_path(self):
        assert validate_command(["/usr/bin/python", "--version"]) is True

    def test_unknown_command(self):
        assert validate_command(["malicious_binary"]) is False

    def test_case_insensitive(self):
        assert validate_command(["LS", "-la"]) is True
        assert validate_command(["Python", "--version"]) is True

    def test_more_allowed_commands(self):
        assert validate_command(["cat", "file.txt"]) is True
        assert validate_command(["pwd"]) is True
        assert validate_command(["echo", "hello"]) is True
        assert validate_command(["node", "app.js"]) is True
        assert validate_command(["npm", "install"]) is True
        assert validate_command(["cargo", "build"]) is True
        assert validate_command(["codex", "run"]) is True
        assert validate_command(["whoami"]) is True
        assert validate_command(["uname", "-a"]) is True

    def test_bun_allowed_pnpm_still_refused(self):
        # bun is baked into codegraph-runtime (Dockerfile.runtime, 1.4.4) and
        # the k8s installer execs `bun install --frozen-lockfile` for the five
        # org repos that commit a text bun.lock (measured 2026-09-17).
        assert validate_command(["bun", "install", "--frozen-lockfile"]) is True
        assert validate_command(["bun", "run", "test"]) is True
        # Negative control: pnpm is NOT in the image and NOT on the allowlist;
        # if this ever passes the allowlist widened by more than bun.
        assert validate_command(["pnpm", "install"]) is False

    def test_code_analysis_tools_allowed(self):
        # coding-analysis / coding-testing exec these into the per-repo container.
        for tool in (
            ["ruff", "check", "/app/data/repo", "--output-format=json"],
            ["ruff", "format", "--check", "--diff", "/app/data/repo"],
            ["ty", "check", "--output-format", "gitlab", "/app/data/repo"],
            ["bandit", "-r", "/app/data/repo", "-f", "json"],
            ["radon", "cc", "/app/data/repo", "-j"],
            ["vulture", "/app/data/repo"],
            ["eslint", "--format=json", "/app/data/repo"],
            ["prettier", "--check", "/app/data/repo"],
            ["semgrep", "--config=auto", "/app/data/repo"],
        ):
            assert validate_command(tool) is True, tool

    def test_replaced_python_tools_refused(self):
        # mypy/pyright -> ty, pylint/flake8/isort/autoflake -> ruff check, black ->
        # ruff format. None of them is baked into codegraph-runtime any more, so the
        # allowlist must not admit them either (an admitted-but-absent binary exits
        # 127, which callers read as a tool result rather than a refusal).
        for tool in ("mypy", "pyright", "pylint", "flake8", "isort", "black", "autoflake"):
            assert validate_command([tool, "/app/data/repo"]) is False, tool

    def test_uv_allowed_so_a_workspace_can_be_made_runnable(self):
        """`uv sync --frozen` must be reachable through /exec.

        THE MEASUREMENT (2026-08-14, AA conveyor, enterprise_mcp_server):
        `verification_test` runs `/workspace/.venv/bin/python -c "import redis"`
        and the repo's own suite through that same venv. Both exited 127 --
        `/workspace/.venv/bin/python: No such file or directory`.

        The venv is absent because the k8s provisioner is a thin volume-holder
        (init container clones, main container sleeps) and installs nothing,
        and the only server-side installer is docker-only. The caller's ONLY
        route to an environment is this whitelist, and `uv` was not in it -- so
        a container could be provisioned and never made runnable. `uv sync
        --frozen` works when run directly; the caller simply could not say it.

        If this assertion fails, the verification hop of the AA conveyor is
        broken again in exactly that way. Do not delete it to make CI pass.
        """
        assert validate_command(["uv", "sync", "--frozen"]) is True
        assert validate_command(["uv", "add", "redis==8.1.0"]) is True
        # The path form matters: basename extraction must reach the same verdict.
        assert validate_command(["/usr/local/bin/uv", "sync", "--frozen"]) is True

    def test_shell_still_refused(self):
        """Allowing `uv` must not be read as opening a shell.

        `sh`/`bash` remain refused: a shell turns the whitelist into a
        suggestion, because command[0] stops describing what actually runs.
        """
        assert validate_command(["sh", "-c", "uv sync --frozen"]) is False
        assert validate_command(["bash", "-lc", "uv sync"]) is False
        assert validate_command(["/bin/sh", "-c", "echo hi"]) is False

    def test_path_extraction(self):
        assert validate_command(["/usr/local/bin/node", "app.js"]) is True
        assert validate_command(["/bin/cat", "file"]) is True


class TestCheckDangerousPatterns:
    def test_safe_command(self):
        assert check_dangerous_patterns(["ls", "-la"]) is True

    def test_rm_rf(self):
        assert check_dangerous_patterns(["rm", "-rf", "/"]) is False

    def test_reboot(self):
        assert check_dangerous_patterns(["reboot"]) is False

    def test_shutdown(self):
        assert check_dangerous_patterns(["shutdown", "-h", "now"]) is False

    def test_mkfs(self):
        assert check_dangerous_patterns(["mkfs", "/dev/sda"]) is False

    def test_normal_python(self):
        assert check_dangerous_patterns(["python", "test.py"]) is True

    def test_dd_dangerous(self):
        assert check_dangerous_patterns(["dd", "if=/dev/zero", "of=/dev/sda"]) is False

    def test_poweroff(self):
        assert check_dangerous_patterns(["poweroff"]) is False

    def test_kill_9(self):
        assert check_dangerous_patterns(["kill", "-9", "1234"]) is False

    # Program tokens, never substrings: a path or search term that merely
    # CONTAINS the word is a file, not a command (file-reference used to 503).
    def test_word_inside_a_path_is_not_a_command(self):
        assert check_dangerous_patterns(["find", ".", "-name", "shutdown_handler.py"]) is True
        assert check_dangerous_patterns(["cat", "src/reboot_policy.py"]) is True
        assert check_dangerous_patterns(["grep", "-n", "executor.shutdown", "src/"]) is True
        assert check_dangerous_patterns(["ls", "docs/mkfs-notes"]) is True
        assert check_dangerous_patterns(["cat", "mkfs.txt"]) is True

    def test_program_by_path_or_variant_is_still_refused(self):
        assert check_dangerous_patterns(["/sbin/shutdown", "-h", "now"]) is False
        assert check_dangerous_patterns(["mkfs.ext4", "/dev/sda1"]) is False
        assert check_dangerous_patterns(["HALT"]) is False

    def test_program_in_a_later_position_is_still_refused(self):
        assert check_dangerous_patterns(["find", ".", "-exec", "shutdown", ";"]) is False
        assert check_dangerous_patterns(["find", ".", "-exec", "rm", "-fr", "{}", ";"]) is False

    def test_flags_without_the_program_are_safe(self):
        assert check_dangerous_patterns(["ls", "-rf"]) is True
        assert check_dangerous_patterns(["rm", "file.txt"]) is True
        assert check_dangerous_patterns(["kill", "-15", "1234"]) is True


# ---- endpoint tests (lines 321-494) ----


@pytest.mark.asyncio
class TestExecInContainerEndpoint:
    """Test the exec_in_container endpoint via HTTP client."""

    async def test_no_request_body(self, app, client):
        """Exec with no request body returns 400."""
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        response = await client.post("/api/containers/mycontainer/exec")

        app.dependency_overrides.clear()
        # FastAPI returns 422 for missing body validation
        assert response.status_code in (400, 422)

    async def test_empty_command(self, app, client):
        """Exec with empty command list returns 400."""
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        response = await client.post(
            "/api/containers/mycontainer/exec",
            json={"command": [], "workdir": "/workspace", "timeout": 30},
        )

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_disallowed_command(self, app, client):
        """Exec with non-whitelisted command returns 400."""
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        response = await client.post(
            "/api/containers/mycontainer/exec",
            json={"command": ["rm", "-rf", "/"], "workdir": "/workspace", "timeout": 30},
        )

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_dangerous_pattern_rejected(self, app, client):
        """Exec with dangerous pattern returns 400."""
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        # "reboot" is whitelisted but dangerous pattern check catches it
        # Actually, reboot is NOT whitelisted, so it fails command validation first.
        # Let's test mkfs which also isn't whitelisted.
        # For dangerous pattern test, we need a command that IS whitelisted but
        # has a dangerous argument pattern. Actually the check works on the full
        # string. Let's use something that passes whitelist but fails pattern:
        # "kill -9" -- kill is not whitelisted either. So this test is about
        # commands that fail both checks -- the first check returns 400.
        response = await client.post(
            "/api/containers/mycontainer/exec",
            json={"command": ["reboot"], "workdir": "/workspace", "timeout": 30},
        )

        app.dependency_overrides.clear()
        assert response.status_code == 400

    async def test_k8s_exec_success(self, app, client):
        """Exec routes through K8s when container has svc.cluster.local endpoint."""
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        container_info = MagicMock()
        container_info.internal_endpoint = "http://mycontainer.default.svc.cluster.local:8080"
        container_info.api_endpoint = ""
        container_info.container_name = "mycontainer"

        with patch("src.routers.container_exec.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container_info
            mock_mgr.return_value = manager

            with patch(
                "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner"
            ) as mock_k8s:
                k8s_prov = MagicMock()
                k8s_prov.exec_in_pod.return_value = {
                    "exit_code": 0,
                    "stdout": "Python 3.12.0",
                    "stderr": "",
                    "duration": 0.1,
                }
                mock_k8s.return_value = k8s_prov

                response = await client.post(
                    "/api/containers/mycontainer/exec",
                    json={
                        "command": ["python", "--version"],
                        "workdir": "/workspace",
                        "timeout": 10,
                    },
                )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        assert data["exit_code"] == 0
        assert "Python" in data["stdout"]

    async def test_k8s_exec_error(self, app, client):
        """Exec via K8s that raises an exception returns 500."""
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        container_info = MagicMock()
        container_info.internal_endpoint = "http://mycontainer.default.svc.cluster.local:8080"
        container_info.api_endpoint = ""
        container_info.container_name = "mycontainer"

        with patch("src.routers.container_exec.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container_info
            mock_mgr.return_value = manager

            with patch(
                "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner"
            ) as mock_k8s:
                k8s_prov = MagicMock()
                k8s_prov.exec_in_pod.side_effect = RuntimeError("pod crashed")
                mock_k8s.return_value = k8s_prov

                response = await client.post(
                    "/api/containers/mycontainer/exec",
                    json={
                        "command": ["python", "--version"],
                        "workdir": "/workspace",
                        "timeout": 10,
                    },
                )

        app.dependency_overrides.clear()
        assert response.status_code == 500

    async def test_k8s_exec_missing_deployment_returns_404_not_500(self, app, client):
        """A deprovisioned/absent container Deployment must yield 404, not 500.

        Regression: the k8s resolver used to re-wrap the 404 as a bare
        RuntimeError, which the exec router turned into a 500 → coding-gateway
        and fr then collapsed that into a misleading "service unavailable".
        The typed ContainerDeploymentNotFoundError must map to an honest 404.
        """
        from coding_client.auth.tenant import require_tenant

        # Use the class object the ROUTER actually bound at import time. A sibling
        # test reimports the provisioner module (sys.modules.pop), which would
        # otherwise give a fresh, non-identical class → the router's `except`
        # would miss it. Referencing the router's own binding is order-robust.
        from src.routers.container_exec import ContainerDeploymentNotFoundError

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        container_info = MagicMock()
        container_info.internal_endpoint = "http://mycontainer.default.svc.cluster.local:8080"
        container_info.api_endpoint = ""
        container_info.container_name = "mycontainer"

        with patch("src.routers.container_exec.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container_info
            mock_mgr.return_value = manager

            with patch(
                "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner"
            ) as mock_k8s:
                k8s_prov = MagicMock()
                k8s_prov.exec_in_pod.side_effect = ContainerDeploymentNotFoundError(
                    "Deployment mycontainer not found: 404"
                )
                mock_k8s.return_value = k8s_prov

                response = await client.post(
                    "/api/containers/mycontainer/exec",
                    json={
                        "command": ["python", "--version"],
                        "workdir": "/workspace",
                        "timeout": 10,
                    },
                )

        app.dependency_overrides.clear()
        assert response.status_code == 404
        # coding_client's exception handler reshapes the HTTPException into the
        # {"error": {"code": "HTTP_404", "message": ...}} envelope.
        error = response.json()["error"]
        assert error["code"] == "HTTP_404"
        assert "not provisioned" in error["message"]

    async def test_k8s_exec_on_a_failed_clone_returns_typed_412(self, app, client):
        """The clone is the authority on repo access (no API pre-flight): a
        workspace whose clone failed answers 412 WORKSPACE_CLONE_FAILED, not 500."""
        from coding_client.auth.tenant import require_tenant
        from src.routers.container_exec import WorkspaceCloneFailedError

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"
        container_info = MagicMock()
        container_info.internal_endpoint = "http://mycontainer.default.svc.cluster.local:8080"
        container_info.api_endpoint = ""
        container_info.container_name = "mycontainer"

        with patch("src.routers.container_exec.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.return_value = container_info
            mock_mgr.return_value = manager
            with patch(
                "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner"
            ) as mock_k8s:
                k8s_prov = MagicMock()
                k8s_prov.exec_in_pod.side_effect = WorkspaceCloneFailedError(
                    "mycontainer", "mycontainer-abc", "git-clone: CrashLoopBackOff (back-off)"
                )
                mock_k8s.return_value = k8s_prov
                response = await client.post(
                    "/api/containers/mycontainer/exec",
                    json={"command": ["python", "--version"], "workdir": "/workspace"},
                )

        app.dependency_overrides.clear()
        assert response.status_code == 412, response.text
        error = response.json()["error"]
        # coding_client's envelope renders ``code`` as HTTP_<status>; the typed
        # reason travels in ``details`` (same as GIT_CREDENTIAL_UNAVAILABLE).
        assert error["code"] == "HTTP_412"
        assert error["details"]["error_code"] == "WORKSPACE_CLONE_FAILED"
        assert error["details"]["reason"].startswith("git-clone: CrashLoopBackOff")
        assert "cloning the repository failed" in error["message"]

    async def test_container_not_found(self, app, client):
        """Exec on non-existent container returns 404."""
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        with patch("src.routers.container_exec.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.side_effect = ValueError("not found")
            mock_mgr.return_value = manager

            response = await client.post(
                "/api/containers/missing/exec",
                json={"command": ["ls"], "workdir": "/workspace", "timeout": 10},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 404

    async def test_agent_host_exec_success(self, app, client):
        """Exec routes through agent-host when container is agent-host managed."""
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        container_info = MagicMock()
        container_info.internal_endpoint = ""
        container_info.api_endpoint = "http://localhost:8080"
        container_info.container_name = "mycontainer"

        core_mgr = MagicMock()
        core_mgr.uses_agent_host = True
        core_mgr.has_agent_host_id.return_value = True
        core_mgr.exec_in_container_via_agent_host.return_value = {
            "exit_code": 0,
            "stdout": "hello",
            "stderr": "",
            "duration": 0.05,
        }

        with (
            patch("src.routers.container_exec.get_container_manager") as mock_mgr,
            patch("src.core.container_manager_core.get_container_manager", return_value=core_mgr),
        ):
            manager = MagicMock()
            manager.get_container.return_value = container_info
            mock_mgr.return_value = manager

            response = await client.post(
                "/api/containers/mycontainer/exec",
                json={"command": ["echo", "hello"], "workdir": "/workspace", "timeout": 10},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        assert data["exit_code"] == 0
        assert data["stdout"] == "hello"

    async def test_agent_host_exec_error(self, app, client):
        """Agent-host exec failure returns 500."""
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        container_info = MagicMock()
        container_info.internal_endpoint = ""
        container_info.api_endpoint = "http://localhost:8080"
        container_info.container_name = "mycontainer"

        core_mgr = MagicMock()
        core_mgr.uses_agent_host = True
        core_mgr.has_agent_host_id.return_value = True
        core_mgr.exec_in_container_via_agent_host.side_effect = RuntimeError("agent-host down")

        with (
            patch("src.routers.container_exec.get_container_manager") as mock_mgr,
            patch("src.core.container_manager_core.get_container_manager", return_value=core_mgr),
        ):
            manager = MagicMock()
            manager.get_container.return_value = container_info
            mock_mgr.return_value = manager

            response = await client.post(
                "/api/containers/mycontainer/exec",
                json={"command": ["echo", "hello"], "workdir": "/workspace", "timeout": 10},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 500

    async def test_docker_tier3_exec_success(self, app, client):
        """Exec falls through to Docker Tier 3 when neither K8s nor agent-host."""
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        container_info = MagicMock()
        container_info.internal_endpoint = ""
        container_info.api_endpoint = "http://localhost:8080"
        container_info.container_name = "mycontainer"

        core_mgr = MagicMock()
        core_mgr.uses_agent_host = False

        mock_docker_container = MagicMock()
        mock_exec_result = MagicMock()
        mock_exec_result.exit_code = 0
        mock_exec_result.output = (b"hello\n", b"")
        mock_docker_container.exec_run.return_value = mock_exec_result

        mock_docker_client = MagicMock()
        mock_docker_client.containers.get.return_value = mock_docker_container
        # The route used to call docker.client.from_env() itself, once per
        # request and unvalidated. It now uses the manager core's shared client,
        # which is ping()-checked at construction — so the mock belongs here.
        core_mgr.client = mock_docker_client
        core_mgr.is_docker_available.return_value = True

        with (
            patch("src.routers.container_exec.get_container_manager") as mock_mgr,
            patch("src.core.container_manager_core.get_container_manager", return_value=core_mgr),
        ):
            manager = MagicMock()
            manager.get_container.return_value = container_info
            mock_mgr.return_value = manager

            response = await client.post(
                "/api/containers/mycontainer/exec",
                json={"command": ["echo", "hello"], "workdir": "/workspace", "timeout": 10},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        assert data["exit_code"] == 0
        assert "hello" in data["stdout"]

    async def test_docker_tier3_exec_no_output(self, app, client):
        """Docker Tier 3 handles None output gracefully."""
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        container_info = MagicMock()
        container_info.internal_endpoint = ""
        container_info.api_endpoint = "http://localhost:8080"
        container_info.container_name = "mycontainer"

        core_mgr = MagicMock()
        core_mgr.uses_agent_host = False

        mock_docker_container = MagicMock()
        mock_exec_result = MagicMock()
        mock_exec_result.exit_code = 0
        mock_exec_result.output = None
        mock_docker_container.exec_run.return_value = mock_exec_result

        mock_docker_client = MagicMock()
        mock_docker_client.containers.get.return_value = mock_docker_container
        # The route used to call docker.client.from_env() itself, once per
        # request and unvalidated. It now uses the manager core's shared client,
        # which is ping()-checked at construction — so the mock belongs here.
        core_mgr.client = mock_docker_client
        core_mgr.is_docker_available.return_value = True

        with (
            patch("src.routers.container_exec.get_container_manager") as mock_mgr,
            patch("src.core.container_manager_core.get_container_manager", return_value=core_mgr),
        ):
            manager = MagicMock()
            manager.get_container.return_value = container_info
            mock_mgr.return_value = manager

            response = await client.post(
                "/api/containers/mycontainer/exec",
                json={"command": ["ls"], "workdir": "/workspace", "timeout": 10},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == ""
        assert data["stderr"] == ""

    async def test_docker_tier3_exec_error(self, app, client):
        """Docker Tier 3 exec failure returns 500."""
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        container_info = MagicMock()
        container_info.internal_endpoint = ""
        container_info.api_endpoint = "http://localhost:8080"
        container_info.container_name = "mycontainer"

        core_mgr = MagicMock()
        core_mgr.uses_agent_host = False

        mock_docker_client = MagicMock()
        mock_docker_client.containers.get.side_effect = RuntimeError("container gone")

        with (
            patch("src.routers.container_exec.get_container_manager") as mock_mgr,
            patch("src.core.container_manager_core.get_container_manager", return_value=core_mgr),
            patch("docker.client.from_env", return_value=mock_docker_client),
        ):
            manager = MagicMock()
            manager.get_container.return_value = container_info
            mock_mgr.return_value = manager

            response = await client.post(
                "/api/containers/mycontainer/exec",
                json={"command": ["ls"], "workdir": "/workspace", "timeout": 10},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 500

    async def test_docker_not_available_503(self, app, client):
        """Docker unavailable returns 503 for Tier 3 exec."""
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        container_info = MagicMock()
        container_info.internal_endpoint = ""
        container_info.api_endpoint = "http://localhost:8080"
        container_info.container_name = "mycontainer"

        core_mgr = MagicMock()
        core_mgr.uses_agent_host = False
        # Docker being down is now expressed through the manager core's own
        # availability check rather than by from_env() raising inside the route:
        # the route no longer builds a client, so there is nothing there to
        # raise. Same 503 to the caller.
        core_mgr.is_docker_available.return_value = False
        core_mgr.client = None

        with (
            patch("src.routers.container_exec.get_container_manager") as mock_mgr,
            patch("src.core.container_manager_core.get_container_manager", return_value=core_mgr),
        ):
            manager = MagicMock()
            manager.get_container.return_value = container_info
            mock_mgr.return_value = manager

            response = await client.post(
                "/api/containers/mycontainer/exec",
                json={"command": ["ls"], "workdir": "/workspace", "timeout": 10},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 503

    async def test_container_not_found_runtime_error(self, app, client):
        """Exec on container not found via RuntimeError returns 404."""
        from coding_client.auth.tenant import require_tenant

        app.dependency_overrides[require_tenant] = lambda: "test-tenant"

        with patch("src.routers.container_exec.get_container_manager") as mock_mgr:
            manager = MagicMock()
            manager.get_container.side_effect = RuntimeError("not found")
            mock_mgr.return_value = manager

            response = await client.post(
                "/api/containers/missing/exec",
                json={"command": ["ls"], "workdir": "/workspace", "timeout": 10},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 404


class TestExecIdentityIsNotACallerChoice:
    """`user` was a published request field defaulting to "root"."""

    def test_exec_request_no_longer_accepts_a_user(self):
        from src.routers.container_exec import ExecRequest

        assert "user" not in ExecRequest.model_fields

    def test_an_old_caller_still_sending_user_is_not_rejected(self):
        """Removing it is not a breaking change — pydantic ignores the extra key.

        No caller in the fleet ever set it (coding-client, coding-analysis,
        coding-repo-jobs, coding-security, coding-testing all pass only
        command/workdir/timeout), and the Kubernetes path never forwarded it, so
        this is belt-and-braces rather than a migration.
        """
        from src.routers.container_exec import ExecRequest

        request = ExecRequest(command=["ls"], user="root")
        assert request.command == ["ls"]
        assert not hasattr(request, "user")

    def test_the_published_example_does_not_advertise_root(self):
        from src.routers.container_exec import ExecRequest

        example = ExecRequest.model_config["json_schema_extra"]["example"]
        assert "user" not in example


class TestExecResponseReportsTruncation:
    """A clipped result must be distinguishable from a complete one.

    Capping output without saying so trades an availability bug for a
    correctness one: a caller parsing clipped output reaches a confidently wrong
    conclusion instead of a slow one.
    """

    def test_defaults_are_untruncated(self):
        from src.routers.container_exec import ExecResponse

        response = ExecResponse(exit_code=0)
        assert response.truncated is False
        assert response.bytes_omitted == 0

    def test_truncation_is_carried_on_the_response(self):
        from src.routers.container_exec import ExecResponse

        response = ExecResponse(exit_code=0, stdout="clipped", truncated=True, bytes_omitted=4096)
        assert response.truncated is True
        assert response.bytes_omitted == 4096


class TestClipHelper:
    def test_short_output_is_untouched(self):
        from src.routers.container_exec import _clip

        text, omitted = _clip(b"hello", 1024)
        assert (text, omitted) == ("hello", 0)

    def test_none_is_empty(self):
        from src.routers.container_exec import _clip

        assert _clip(None, 1024) == ("", 0)

    def test_long_output_is_clipped_and_counted(self):
        from src.routers.container_exec import _clip

        text, omitted = _clip(b"x" * 5000, 1000)
        assert len(text) == 1000
        assert omitted == 4000

    def test_a_straddling_multibyte_character_is_dropped_not_mangled(self):
        from src.routers.container_exec import _clip

        # 4 bytes of a 2-character, 6-byte stream: the second € is cut in half.
        text, omitted = _clip("€€".encode(), 4)
        assert text == "€"
        assert omitted == 2

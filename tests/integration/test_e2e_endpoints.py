"""Live E2E endpoint tests for coding-containers.

Hit the REAL service running at http://localhost:8691 (ENABLE_AUTH=false).
No mocks, no ASGI transport -- plain HTTP via httpx.

Run with:
    uv run pytest tests/integration/test_e2e_endpoints.py -v -m integration
"""

import httpx
import pytest

BASE_URL = "http://localhost:8691"
FAKE_CONTAINER = "nonexistent-e2e-test-container"


@pytest.fixture(scope="module")
def client():
    """Shared httpx client for all tests in this module."""
    with httpx.Client(base_url=BASE_URL, timeout=10) as c:
        yield c


# ──────────────────────────────────────────────────────────────────────
# Root & Health
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestRootAndHealth:
    """Tests for the root and health endpoints."""

    def test_root_returns_service_info(self, client: httpx.Client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "coding-containers"
        assert "version" in data
        assert data["docs"] == "/docs"
        assert data["health"] == "/health"

    def test_health_returns_ok(self, client: httpx.Client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "coding-containers"
        assert "version" in data

    def test_openapi_json_available(self, client: httpx.Client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "paths" in data
        assert "/health" in data["paths"]

    def test_docs_page_available(self, client: httpx.Client):
        resp = client.get("/docs")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


# ──────────────────────────────────────────────────────────────────────
# Container CRUD
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestContainerList:
    """Tests for GET /api/containers/."""

    def test_list_containers_returns_200(self, client: httpx.Client):
        resp = client.get("/api/containers/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert isinstance(data["containers"], list)
        assert isinstance(data["total"], int)
        assert data["total"] == len(data["containers"])

    def test_list_containers_has_timestamp(self, client: httpx.Client):
        resp = client.get("/api/containers/")
        data = resp.json()
        assert "timestamp" in data
        assert "request_id" in data


@pytest.mark.integration
class TestContainerDetail:
    """Tests for GET /api/containers/{name} -- 404 for missing containers."""

    def test_get_nonexistent_container_returns_404(self, client: httpx.Client):
        resp = client.get(f"/api/containers/{FAKE_CONTAINER}")
        assert resp.status_code == 404
        data = resp.json()
        assert data["status"] == "error"
        assert data["error_code"] == "CONTAINER_NOT_FOUND"
        assert FAKE_CONTAINER in data["message"]

    def test_404_response_has_standard_error_fields(self, client: httpx.Client):
        resp = client.get(f"/api/containers/{FAKE_CONTAINER}")
        data = resp.json()
        for field in ("status", "message", "timestamp", "request_id", "path", "error_code"):
            assert field in data, f"Missing field: {field}"


@pytest.mark.integration
class TestContainerStatus:
    """Tests for GET /api/containers/{name}/status -- 404 for missing."""

    def test_status_nonexistent_returns_404(self, client: httpx.Client):
        resp = client.get(f"/api/containers/{FAKE_CONTAINER}/status")
        assert resp.status_code == 404
        data = resp.json()
        assert data["error_code"] == "CONTAINER_NOT_FOUND"


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestCodeValidation:
    """Tests for POST /api/validation/multi-line-code-snippet."""

    def test_valid_python_code(self, client: httpx.Client):
        resp = client.post(
            "/api/validation/multi-line-code-snippet",
            json={"code_snippet": "print('hello world')"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["static_analysis"]["valid"] is True
        assert data["static_analysis"]["overall_score"] == 1.0
        assert data["overall_confidence"] == 1.0
        assert data["validation_summary"] == "Static: PASS"

    def test_multiline_valid_code(self, client: httpx.Client):
        code = "import os\nimport sys\n\ndef greet(name):\n    return f'Hello {name}'\n\nprint(greet('world'))"
        resp = client.post(
            "/api/validation/multi-line-code-snippet",
            json={"code_snippet": code},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["static_analysis"]["passed_checks"] == 1
        assert data["static_analysis"]["total_checks"] == 1

    def test_syntax_error_detected(self, client: httpx.Client):
        resp = client.post(
            "/api/validation/multi-line-code-snippet",
            json={"code_snippet": "def foo(:\n  pass"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert data["static_analysis"]["valid"] is False
        assert len(data["static_analysis"]["issues"]) > 0
        issue = data["static_analysis"]["issues"][0]
        assert issue["type"] == "syntax"
        assert issue["severity"] == "error"
        assert data["overall_confidence"] == 0.0
        assert data["validation_summary"] == "Static: FAIL"
        assert "Fix syntax errors" in data["suggestions"][0]

    def test_incomplete_expression_syntax_error(self, client: httpx.Client):
        resp = client.post(
            "/api/validation/multi-line-code-snippet",
            json={"code_snippet": "x = 1\ny = x +\n"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert data["static_analysis"]["valid"] is False

    def test_missing_code_snippet_returns_422(self, client: httpx.Client):
        resp = client.post(
            "/api/validation/multi-line-code-snippet",
            json={},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error_code"] == "VALIDATION_ERROR"
        assert any(e["field"] == "body.code_snippet" for e in data["validation_errors"])

    def test_empty_code_snippet_returns_422(self, client: httpx.Client):
        resp = client.post(
            "/api/validation/multi-line-code-snippet",
            json={"code_snippet": ""},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error_code"] == "VALIDATION_ERROR"

    def test_execution_validation_without_docker(self, client: httpx.Client):
        """When include_execution_validation=True but no Docker, execution_analysis reflects that."""
        resp = client.post(
            "/api/validation/multi-line-code-snippet",
            json={
                "code_snippet": "x = 42",
                "include_execution_validation": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # Static analysis should still pass
        assert data["static_analysis"]["valid"] is True
        # Execution analysis is present (Docker not available = unsuccessful)
        if data["execution_analysis"] is not None:
            assert isinstance(data["execution_analysis"]["execution_successful"], bool)

    def test_validation_response_has_package_info(self, client: httpx.Client):
        resp = client.post(
            "/api/validation/multi-line-code-snippet",
            json={"code_snippet": "import os"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "package_info" in data
        assert "packages" in data["package_info"]
        assert "total_packages" in data["package_info"]


# ──────────────────────────────────────────────────────────────────────
# Health sub-router (/api/health/...)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestHealthSubRouter:
    """Tests for /api/health/* container health endpoints."""

    def test_unhealthy_containers_list(self, client: httpx.Client):
        resp = client.get("/api/health/containers/unhealthy")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["unhealthy"], list)
        assert isinstance(data["count"], int)
        assert data["count"] == len(data["unhealthy"])

    def test_container_health_404(self, client: httpx.Client):
        resp = client.get(f"/api/health/container/{FAKE_CONTAINER}")
        assert resp.status_code == 404
        data = resp.json()
        assert data["error_code"] == "NOT_FOUND"

    def test_container_readiness_404(self, client: httpx.Client):
        resp = client.get(f"/api/health/container/{FAKE_CONTAINER}/ready")
        assert resp.status_code == 404
        data = resp.json()
        assert data["error_code"] == "NOT_FOUND"

    def test_container_liveness_404(self, client: httpx.Client):
        resp = client.get(f"/api/health/container/{FAKE_CONTAINER}/live")
        assert resp.status_code == 404
        data = resp.json()
        assert data["error_code"] == "NOT_FOUND"

    def test_container_statistics_404(self, client: httpx.Client):
        resp = client.get(f"/api/health/container/{FAKE_CONTAINER}/statistics")
        assert resp.status_code == 404
        data = resp.json()
        assert data["error_code"] == "NOT_FOUND"


# ──────────────────────────────────────────────────────────────────────
# Container Exec
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestContainerExec:
    """Tests for POST /api/containers/{name}/exec -- command validation."""

    def test_exec_nonexistent_container_returns_404(self, client: httpx.Client):
        resp = client.post(
            f"/api/containers/{FAKE_CONTAINER}/exec",
            json={"command": ["ls", "-la"]},
        )
        assert resp.status_code == 404
        data = resp.json()
        assert data["error_code"] == "CONTAINER_NOT_FOUND"

    def test_exec_blocked_command_returns_400(self, client: httpx.Client):
        """rm is not in the command whitelist; should be rejected before container lookup."""
        resp = client.post(
            f"/api/containers/{FAKE_CONTAINER}/exec",
            json={"command": ["rm", "-rf", "/"]},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error_code"] == "BAD_REQUEST"
        assert "not allowed" in data["message"].lower()

    def test_exec_empty_command_returns_400(self, client: httpx.Client):
        resp = client.post(
            f"/api/containers/{FAKE_CONTAINER}/exec",
            json={"command": []},
        )
        assert resp.status_code == 400

    def test_exec_missing_body_returns_400(self, client: httpx.Client):
        resp = client.post(
            f"/api/containers/{FAKE_CONTAINER}/exec",
            json=None,
        )
        # Missing body should return 400 (bad request) from the handler
        assert resp.status_code == 400


# ──────────────────────────────────────────────────────────────────────
# Agent Capabilities
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestAgentCapabilities:
    """Tests for GET /api/containers/{name}/agent-capabilities."""

    def test_capabilities_returns_200_even_for_missing_container(self, client: httpx.Client):
        """Agent capabilities returns 200 with unavailable tools, not 404."""
        resp = client.get(f"/api/containers/{FAKE_CONTAINER}/agent-capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert data["container_name"] == FAKE_CONTAINER
        assert "claude" in data["capabilities"]
        assert "codex" in data["capabilities"]
        # Without Docker, both should be unavailable
        assert data["capabilities"]["claude"]["available"] is False
        assert data["capabilities"]["codex"]["available"] is False

    def test_capabilities_has_note_field(self, client: httpx.Client):
        resp = client.get(f"/api/containers/{FAKE_CONTAINER}/agent-capabilities")
        data = resp.json()
        for cli in ("claude", "codex"):
            assert "note" in data["capabilities"][cli]


# ──────────────────────────────────────────────────────────────────────
# Container Dependencies (404 for missing containers)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestContainerDependencies:
    """Tests for dependency endpoints -- 404 for missing containers."""

    def test_dependency_status_nonexistent_returns_404(self, client: httpx.Client):
        resp = client.get(f"/api/containers/{FAKE_CONTAINER}/dependency-status")
        assert resp.status_code == 404

    def test_install_dependencies_nonexistent_returns_404(self, client: httpx.Client):
        resp = client.post(
            f"/api/containers/{FAKE_CONTAINER}/install-dependencies",
            json={"workspace_path": "/app/data/repo"},
        )
        assert resp.status_code == 404


# ──────────────────────────────────────────────────────────────────────
# Container File & Git Operations (404 for missing containers)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestContainerFileOps:
    """Tests for file/git operations on non-existent containers."""

    def test_list_files_nonexistent_returns_404(self, client: httpx.Client):
        resp = client.get(f"/api/containers/{FAKE_CONTAINER}/files")
        assert resp.status_code == 404

    def test_git_status_nonexistent_returns_404(self, client: httpx.Client):
        resp = client.get(f"/api/containers/{FAKE_CONTAINER}/git/status")
        assert resp.status_code == 404

    def test_mkdir_nonexistent_returns_404(self, client: httpx.Client):
        resp = client.post(
            f"/api/containers/{FAKE_CONTAINER}/mkdir",
            json={"path": "/workspace/test"},
        )
        assert resp.status_code == 404

    def test_touch_nonexistent_returns_404(self, client: httpx.Client):
        resp = client.post(
            f"/api/containers/{FAKE_CONTAINER}/touch",
            json={"path": "/workspace", "filename": "test", "extension": "py"},
        )
        assert resp.status_code == 404

    def test_write_file_nonexistent_returns_404(self, client: httpx.Client):
        resp = client.post(
            f"/api/containers/{FAKE_CONTAINER}/write-file",
            json={"file_path": "/workspace/test.py", "content": "print('hello')"},
        )
        assert resp.status_code == 404

    def test_move_nonexistent_returns_404(self, client: httpx.Client):
        resp = client.post(
            f"/api/containers/{FAKE_CONTAINER}/move",
            json={"source_path": "old", "destination_path": "new"},
        )
        assert resp.status_code == 404

    def test_delete_path_nonexistent_returns_404(self, client: httpx.Client):
        resp = client.delete(
            f"/api/containers/{FAKE_CONTAINER}/files",
            params={"path": "obsolete"},
        )
        assert resp.status_code == 404

    def test_git_commit_nonexistent_returns_404(self, client: httpx.Client):
        resp = client.post(
            f"/api/containers/{FAKE_CONTAINER}/git/commit",
            json={"message": "test commit"},
        )
        assert resp.status_code == 404

    def test_git_push_nonexistent_returns_404(self, client: httpx.Client):
        resp = client.post(
            f"/api/containers/{FAKE_CONTAINER}/git/push",
            json={"branch": "main"},
        )
        assert resp.status_code == 404


# ──────────────────────────────────────────────────────────────────────
# Error Response Format Consistency
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestErrorResponseFormat:
    """Verify error responses follow a consistent structure."""

    @pytest.mark.parametrize(
        "method,path",
        [
            ("GET", f"/api/containers/{FAKE_CONTAINER}"),
            ("GET", f"/api/containers/{FAKE_CONTAINER}/status"),
            ("GET", f"/api/health/container/{FAKE_CONTAINER}"),
        ],
    )
    def test_404_error_format(self, client: httpx.Client, method: str, path: str):
        resp = client.request(method, path)
        assert resp.status_code == 404
        data = resp.json()
        assert data["status"] == "error"
        assert isinstance(data["message"], str)
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["request_id"], str)
        assert isinstance(data["path"], str)
        assert isinstance(data["error_code"], str)

    def test_method_not_allowed(self, client: httpx.Client):
        """PATCH on a GET-only endpoint should return 405."""
        resp = client.patch("/api/containers/")
        assert resp.status_code == 405

"""Tests for src/main.py — KAG API standard exception handlers, lifespan."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import docker.errors

# ---------------------------------------------------------------------------
# KAG API standard exception handlers (via coding-client)
# ---------------------------------------------------------------------------


class TestHTTPExceptionHandler:
    async def test_nonexistent_container_returns_404(self, client):
        """Requesting a nonexistent container should return 404 with standard error body."""
        resp = await client.get("/api/health/container/nonexistent-container-12345")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        error = data["error"]
        assert error["code"] == "HTTP_404"
        assert "message" in error

    async def test_404_error_has_timestamp_and_request_id(self, client):
        """404 errors should include timestamp and request_id."""
        resp = await client.get("/api/health/container/no-such-thing-xyz/statistics")
        assert resp.status_code == 404
        error = resp.json()["error"]
        assert "timestamp" in error
        assert "request_id" in error


# ---------------------------------------------------------------------------
# Validation exception handler
# ---------------------------------------------------------------------------


class TestValidationExceptionHandler:
    async def test_invalid_request_body_returns_422(self, client):
        """POST with invalid body triggers RequestValidationError handler."""
        resp = await client.post("/api/validation/multi-line-code-snippet", json={"bad": "data"})
        assert resp.status_code == 422
        data = resp.json()
        assert "error" in data
        error = data["error"]
        assert error["code"] == "HTTP_422"
        assert "details" in error

    async def test_missing_required_field_returns_422_with_details(self, client):
        """Missing required fields trigger validation error with field-level details."""
        resp = await client.post("/api/validation/multi-line-code-snippet", json={})
        assert resp.status_code == 422
        error = resp.json()["error"]
        assert len(error["details"]) > 0
        first_detail = error["details"][0]
        assert "field" in first_detail
        assert "message" in first_detail
        assert "type" in first_detail

    async def test_empty_code_snippet_returns_422(self, client):
        """An empty code_snippet should fail validation."""
        resp = await client.post(
            "/api/validation/multi-line-code-snippet",
            json={"code_snippet": ""},
        )
        assert resp.status_code == 422

    async def test_too_short_code_snippet_returns_422(self, client):
        """A too-short code_snippet should fail validation."""
        resp = await client.post(
            "/api/validation/multi-line-code-snippet",
            json={"code_snippet": "x"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Health endpoint (sanity check)
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    async def test_health_returns_200(self, client):
        """GET /health should return 200 with service info."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "service" in data
        assert "version" in data

    async def test_root_returns_200(self, client):
        """GET / should return 200 with service info and links."""
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "service" in data
        assert "version" in data
        assert data["docs"] == "/docs"
        assert data["health"] == "/health"


# ---------------------------------------------------------------------------
# Standard error response fields
# ---------------------------------------------------------------------------


class TestStandardErrorResponse:
    async def test_error_response_has_all_required_fields(self, client):
        """Error responses from KAG handlers must include code, message, timestamp, request_id."""
        resp = await client.get("/api/health/container/nonexistent-abc123")
        assert resp.status_code == 404
        error = resp.json()["error"]
        for field in ("code", "message", "timestamp", "request_id"):
            assert field in error, f"Missing field in error envelope: {field}"


# ---------------------------------------------------------------------------
# register_exception_handlers integration
# ---------------------------------------------------------------------------


class TestExceptionHandlersRegistered:
    def test_app_has_exception_handlers(self, app):
        """Verify that the app has exception handlers registered."""
        assert len(app.exception_handlers) > 0

    def test_app_has_422_handler(self, app):
        """App should have a handler for RequestValidationError (422)."""
        from fastapi.exceptions import RequestValidationError

        assert RequestValidationError in app.exception_handlers


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


class TestLifespan:
    @patch("src.main.RedisService", create=True)
    @patch("src.main.get_docker_service", create=True)
    async def test_lifespan_docker_available(self, mock_get_docker, mock_redis_cls):
        """Test lifespan completes when Docker is available."""
        from src.main import lifespan

        mock_docker_svc = MagicMock()
        mock_docker_svc.docker_available = True
        mock_get_docker.return_value = mock_docker_svc

        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.host = "localhost"
        mock_redis.port = 6379
        mock_redis_cls.return_value = mock_redis

        mock_app = MagicMock()

        with (
            patch("src.services.docker_service.get_docker_service", mock_get_docker),
            patch("src.core.redis_service.RedisService", mock_redis_cls),
        ):
            async with lifespan(mock_app):
                pass
        # Verify redis was closed during shutdown
        mock_redis.close.assert_called_once()

    @patch(
        "src.services.docker_service.get_docker_service",
        side_effect=docker.errors.DockerException("Docker fail"),
    )
    async def test_lifespan_docker_fails_is_nonfatal(self, mock_get_docker):
        """Test lifespan does not crash when Docker initialization fails."""
        from src.main import lifespan

        mock_app = MagicMock()
        # Should not raise — Docker failure is non-fatal
        async with lifespan(mock_app):
            pass

    async def test_lifespan_docker_not_available(self):
        """Cover line 48: Docker initialised but not available."""
        from redis.exceptions import RedisError

        from src.main import lifespan

        mock_docker_svc = MagicMock()
        mock_docker_svc.is_docker_available.return_value = False

        mock_app = MagicMock()
        with (
            patch(
                "src.services.docker_service.get_docker_service",
                return_value=mock_docker_svc,
            ),
            patch(
                "src.core.redis_service.RedisService",
                side_effect=RedisError("no redis"),
            ),
        ):
            async with lifespan(mock_app):
                pass

    async def test_lifespan_redis_close_error(self):
        """Cover lines 72-73: redis_svc.close() raises during shutdown."""
        from redis.exceptions import RedisError

        from src.main import lifespan

        mock_docker_svc = MagicMock()
        mock_docker_svc.is_docker_available.return_value = True

        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.host = "localhost"
        mock_redis.port = 6379
        mock_redis.close.side_effect = RedisError("close failed")

        mock_app = MagicMock()
        with (
            patch(
                "src.services.docker_service.get_docker_service",
                return_value=mock_docker_svc,
            ),
            patch("src.core.redis_service.RedisService", return_value=mock_redis),
        ):
            async with lifespan(mock_app):
                pass
        # Should not raise — close error is swallowed

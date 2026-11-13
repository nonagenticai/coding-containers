"""Shared test fixtures for coding-containers."""

from __future__ import annotations

import os

# Set test environment variables before any app imports
os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("ENABLE_DOCS", "true")
os.environ.setdefault("ENVIRONMENT", "test")
# Settings.environment is sourced via env_prefix=CODING_CONTAINERS_; force test
# so the production validator (which now requires main_api_endpoint) does not
# fire under pytest.
os.environ.setdefault("CODING_CONTAINERS_ENVIRONMENT", "test")
# coding-client >=1.4.0 requires CodingTestingClient to be given a base_url or
# CODING_TESTING_URL; the quality/health delegates construct it lazily. Point it
# at a loopback discard port so construction succeeds and the HTTP call is
# refused instantly (no DNS, no retry backoff) — the delegates swallow the
# error and degrade gracefully, which is what these unit tests exercise.
os.environ.setdefault("CODING_TESTING_URL", "http://127.0.0.1:9")
# coding-client >=1.7.0 defaults TRUSTED_CIDRS to EMPTY, because the old default
# contained the cluster's pod-network range and so let every pod in the cluster
# authenticate itself (see coding-client #33). TestClient dispatches from
# 127.0.0.1, and these suites exercise ENDPOINT behaviour, not the auth boundary
# — so without this they all 401 on the trusted-network branch that used to be on
# by default. Loopback ONLY: keep the pod CIDR out, so a test can never rely on
# the bypass this default was changed to remove.
os.environ.setdefault("TRUSTED_CIDRS", "127.0.0.1/32,::1/128")
# coding-client >=1.9.0 removes the built-in "n8n-dev" fallback for
# DEFAULT_TENANT_ID (see coding-client #36): unset now resolves to None and the
# trusted-network branch above REFUSES to mint a caller rather than inventing a
# tenant, logging "DEFAULT_TENANT_ID is not configured". The bypass armed on the
# line above is therefore only half a precondition without this one.
# Value mirrors production exactly — clusters/strato/services/coding-containers/
# configmap.yaml sets the same uuid7 — so these tests exercise the tenant the
# service actually acts as, not a fixture-local invention.
os.environ.setdefault("DEFAULT_TENANT_ID", "0195f48b-d000-7000-8000-000000000001")

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def app():
    """Create FastAPI app for testing."""
    os.environ["ENABLE_AUTH"] = "false"
    os.environ["ENABLE_DOCS"] = "true"
    os.environ["ENVIRONMENT"] = "test"

    from src.main import app

    return app


@pytest.fixture
async def client(app):
    """Async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_docker_service():
    """Mock Docker service that simulates no Docker available."""
    mock = MagicMock()
    mock.docker_available = False
    mock.is_docker_available.return_value = False
    mock.get_language_container_image.return_value = None
    mock.exec_in_container = AsyncMock(
        return_value={"exit_code": 0, "stdout": "", "stderr": "", "duration": 0.1}
    )
    mock.registry = MagicMock()
    mock.registry.load.return_value = {}
    return mock


@pytest.fixture
def mock_cache_manager():
    """Mock Redis cache manager."""
    mock = MagicMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_container_manager():
    """Mock ContainerManager."""
    mock = MagicMock()
    mock.list_containers.return_value = []
    mock.get_container.return_value = None
    mock.has_container.return_value = False
    mock.remove_container.return_value = {"status": "removed"}
    mock.execute_command = AsyncMock(return_value={"exit_code": 0, "stdout": "", "stderr": ""})
    return mock


@pytest.fixture
def grant_workspace_ownership():
    """Resolve every workspace to the calling tenant, for tests that predate #135.

    #135 added an ownership gate to the three routes that execute work in a
    caller-named workspace (``POST /{name}/execute-agent``,
    ``GET /{name}/agent-capabilities`` and
    ``POST /api/validation/multi-line-code-snippet``). Tests written before that
    gate existed never provisioned a workspace at all, so under the gate they
    correctly 404 — the routes now require the caller to OWN the target.

    This fixture restores their premise (the caller owns the workspace) without
    touching the gate itself, so those tests keep testing what they were written
    to test. ⛔ It must NEVER be used in a test that asserts anything ABOUT the
    gate: it makes ``authorize_workspace`` pass unconditionally, so a cross-tenant
    assertion guarded by it would be vacuous. The gate's own tests
    (``tests/unit/test_agent_exec_tenant_scope.py``) deliberately do not use it —
    they run the real ``ContainerManager`` and stub only the process boundary.
    """
    with patch("src.routers.tenant_guard.get_container_manager") as mock_get_manager:
        # Only the non-None-ness matters to the gate; the routes re-resolve the
        # workspace through their own service layer.
        mock_get_manager.return_value.get_container.return_value = MagicMock()
        yield mock_get_manager

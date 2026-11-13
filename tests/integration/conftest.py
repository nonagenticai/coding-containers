"""Integration test fixtures with HTTP mocks for downstream services."""

import os

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("ENABLE_DOCS", "true")
os.environ.setdefault("ENVIRONMENT", "test")

import pytest
import respx
from httpx import ASGITransport, AsyncClient, Response

# Every `coding_client` service client resolves its base URL from an env var at
# construction time and raises if it is unset. The mock_* fixtures below install
# respx routers keyed on a specific host, so the env var and the router host have
# to agree or the client either fails to construct or sends the request
# somewhere respx is not listening.
#
# This deliberately does NOT touch `tests/conftest.py`, which pins the service
# URLs to the discard port http://127.0.0.1:9 so that an unmocked call in
# tests/unit fails instantly and loudly instead of escaping the test host. That
# default is correct and stays; it is simply the wrong value for the integration
# suites, which exist to exercise the real request path.
_DOWNSTREAM_BASE_URLS = {
    "CODING_GITHUB_URL": "http://coding-github:8692",
    "CODING_REALTIME_URL": "http://coding-realtime:8694",
}

# Downstream routes these suites assert against, verified against the owning
# service's router tree. Shared with the test modules so a fixture and the
# assertion that reads its recorded calls can never drift apart again — that
# drift is exactly what let both of these sit broken while CI stayed green.
#
# coding-github  src/main.py:235   include_router(github_router, prefix="/api/repositories")
#                src/routers/github.py:72        prefix="/webhooks"
#                src/routers/github_webhooks.py:43  @router.post("/webhook")
GITHUB_WEBHOOK_PATH = "/api/repositories/webhooks/webhook"


@pytest.fixture(autouse=True)
def _pin_downstream_base_urls(monkeypatch):
    """Point each service client at the host its respx fixture intercepts.

    Scoped per test via monkeypatch rather than set at conftest import: a
    module-level `os.environ[...] = ...` here would leak into tests/unit for the
    whole session (conftest import order, not test selection, decides who sees
    it) and would silently disarm the discard-port default that tests/unit
    relies on.
    """
    for var, url in _DOWNSTREAM_BASE_URLS.items():
        monkeypatch.setenv(var, url)


@pytest.fixture(autouse=True)
def _reset_http_client_singletons():
    """Reset module-level HTTP client singletons between integration tests.

    Each test may run inside a different respx mock context, so the singleton
    created in a previous test must not leak into the next one. Both client
    modules cache a `_http_client` built from the env var at first use, so both
    have to be reset — otherwise a client constructed before
    `_pin_downstream_base_urls` ever ran keeps whatever base URL it was born
    with.
    """
    import src.clients.github_client as gh_mod
    import src.clients.realtime_client as rt_mod

    modules = (gh_mod, rt_mod)
    for mod in modules:
        mod._http_client = None
    yield
    for mod in modules:
        mod._http_client = None


@pytest.fixture
def mock_github_service():
    """Mock coding-github HTTP responses."""
    with respx.mock(base_url="http://coding-github:8692", assert_all_called=False) as mock:
        # Real route: coding-github mounts github_router at /api/repositories and
        # its webhooks sub-router at /webhooks, so CodingGitHubClient.create_webhook
        # posts to /api/repositories/webhooks/webhook. The old /api/webhooks route
        # here matched nothing (that prefix belongs to the pr-actions router).
        mock.post(GITHUB_WEBHOOK_PATH).mock(
            return_value=Response(200, json={"token_id": "tok-123"})
        )
        mock.get("/api/repos").mock(return_value=Response(200, json={"repos": []}))
        mock.get("/health").mock(return_value=Response(200, json={"status": "ok"}))
        yield mock


@pytest.fixture
def mock_realtime_service():
    """Mock coding-realtime HTTP responses."""
    with respx.mock(base_url="http://coding-realtime:8694", assert_all_called=False) as mock:
        mock.post("/api/broadcast/publish").mock(
            return_value=Response(200, json={"delivered": True, "subscriber_count": 1})
        )
        mock.post("/api/send").mock(return_value=Response(200, json={"sent": True}))
        mock.get("/health").mock(return_value=Response(200, json={"status": "ok"}))
        yield mock


@pytest.fixture
def app():
    """Create FastAPI app for integration testing."""
    os.environ["ENABLE_AUTH"] = "false"
    os.environ["ENABLE_DOCS"] = "true"
    os.environ["ENVIRONMENT"] = "test"
    from src.main import app

    return app


@pytest.fixture
async def client(app):
    """Async HTTP client for integration testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

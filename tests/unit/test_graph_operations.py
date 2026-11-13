"""Tests for graph cleanup used by container teardown.

⚠️ The suite this replaces asserted the two properties that WERE the defect:
that the call goes to ``{main_api_endpoint}/api/graphs/{name}`` (coding-gateway
— a north-south edge that resets every ``coding-*`` east-west call), and that an
unreachable gateway "succeeds" by deleting local sqlite files (monolith-era
state that does not exist in k8s, so it always deleted nothing and reported
``0 nodes, 0 edges`` as a success). Those tests were green the whole time real
containers' graphs were being orphaned in coding-analysis.

The assertions here are therefore deliberately about what was wrong: the URL,
the direction, and never claiming a deletion that did not happen.
"""

from __future__ import annotations

import httpx
import pytest

ANALYSIS_URL = "http://coding-analysis.test:8688"


def _client_returning(response_factory, recorder: dict | None = None):
    """Build a fake httpx.AsyncClient whose DELETE returns ``response_factory(url)``."""

    class _MockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def delete(self, url, headers=None):
            if recorder is not None:
                recorder["url"] = url
                recorder["headers"] = headers
            return response_factory(url)

    return lambda **kwargs: _MockClient()


@pytest.fixture
def analysis_settings(monkeypatch):
    monkeypatch.setenv("CODING_ANALYSIS_URL", ANALYSIS_URL)
    from src.config.settings import reload_settings

    reload_settings()
    yield
    reload_settings()


@pytest.mark.asyncio
async def test_deletes_via_coding_analysis_directly_not_the_gateway(monkeypatch, analysis_settings):
    """The URL is the whole regression: coding-analysis, /api/analysis/graphs/…"""
    from src.core.graph_operations import GraphOperations

    seen: dict = {}

    def _ok(url):
        return httpx.Response(
            200,
            request=httpx.Request("DELETE", url),
            json={
                "container_name": "repo-a",
                "deleted": True,
                "nodes_deleted": 4,
                "edges_deleted": 7,
            },
        )

    monkeypatch.setattr("src.core.graph_operations.httpx.AsyncClient", _client_returning(_ok, seen))

    result = await GraphOperations().delete_container_graph("repo-a", tenant_id="tenant-a")

    assert seen["url"] == f"{ANALYSIS_URL}/api/analysis/graphs/repo-a"
    assert "/api/graphs/" not in seen["url"]  # the gateway path that never existed
    assert seen["headers"] == {"X-Tenant-Id": "tenant-a"}
    assert result["success"] is True
    assert result["deleted_nodes"] == 4
    assert result["deleted_edges"] == 7


@pytest.mark.asyncio
async def test_unreachable_analysis_is_a_failure_not_a_local_sqlite_success(
    monkeypatch, analysis_settings
):
    """The other half of the regression: no fabricated success on a failed call."""
    from src.core.graph_operations import GraphOperations

    class _FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def delete(self, url, headers=None):
            raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(
        "src.core.graph_operations.httpx.AsyncClient", lambda **kwargs: _FailingClient()
    )

    result = await GraphOperations().delete_container_graph("repo-b", tenant_id="tenant-a")

    assert result["success"] is False
    assert result["deleted_nodes"] == 0
    assert "unreachable" in result["message"]


@pytest.mark.asyncio
async def test_404_is_already_absent_and_is_marked_as_such(monkeypatch, analysis_settings):
    from src.core.graph_operations import GraphOperations

    def _missing(url):
        return httpx.Response(404, request=httpx.Request("DELETE", url), json={})

    monkeypatch.setattr("src.core.graph_operations.httpx.AsyncClient", _client_returning(_missing))

    result = await GraphOperations().delete_container_graph("missing", tenant_id="t")

    assert result["success"] is True
    assert result["already_absent"] is True
    assert result["deleted_nodes"] == 0


@pytest.mark.asyncio
async def test_server_error_is_a_failure(monkeypatch, analysis_settings):
    from src.core.graph_operations import GraphOperations

    def _boom(url):
        return httpx.Response(500, request=httpx.Request("DELETE", url), json={"e": 1})

    monkeypatch.setattr("src.core.graph_operations.httpx.AsyncClient", _client_returning(_boom))

    result = await GraphOperations().delete_container_graph("repo-c", tenant_id="t")

    assert result["success"] is False
    assert "500" in result["message"]


@pytest.mark.asyncio
async def test_no_tenant_fails_closed_without_issuing_a_request(monkeypatch, analysis_settings):
    """The endpoint is tenant-scoped; firing a request that cannot work is not honest."""
    from src.core.graph_operations import GraphOperations

    called = {"n": 0}

    class _NeverClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def delete(self, url, headers=None):
            called["n"] += 1
            raise AssertionError("must not issue a request without a tenant")

    monkeypatch.setattr(
        "src.core.graph_operations.httpx.AsyncClient", lambda **kwargs: _NeverClient()
    )

    result = await GraphOperations().delete_container_graph("repo-d", tenant_id=None)

    assert result["success"] is False
    assert called["n"] == 0
    assert "tenant" in result["message"]


@pytest.mark.asyncio
async def test_unset_analysis_url_fails_closed(monkeypatch):
    from src.config.settings import reload_settings
    from src.core.graph_operations import GraphOperations

    monkeypatch.setenv("CODING_ANALYSIS_URL", "")
    reload_settings()
    try:
        result = await GraphOperations().delete_container_graph("repo-e", tenant_id="t")
    finally:
        reload_settings()

    assert result["success"] is False
    assert "coding_analysis_url" in result["message"]


@pytest.mark.asyncio
async def test_non_json_success_body_is_not_reported_as_a_deletion(monkeypatch, analysis_settings):
    from src.core.graph_operations import GraphOperations

    def _html(url):
        return httpx.Response(200, request=httpx.Request("DELETE", url), text="<html>nope")

    monkeypatch.setattr("src.core.graph_operations.httpx.AsyncClient", _client_returning(_html))

    result = await GraphOperations().delete_container_graph("repo-f", tenant_id="t")

    assert result["success"] is False


def test_get_graph_operations_singleton():
    import src.core.graph_operations as mod
    from src.core.graph_operations import get_graph_operations

    mod._graph_operations = None
    ops1 = get_graph_operations()
    ops2 = get_graph_operations()
    assert ops1 is ops2
    mod._graph_operations = None

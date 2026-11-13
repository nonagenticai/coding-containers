"""Async exec routes: boundary gates, 202 contract, tenant scoping, sync route unchanged.

Store and runner are fakes here; the store's SQL (transitions, races, tenant
scoping of the read) is exercised against real Postgres in test_exec_jobs_pg.py.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coding_client.auth.tenant import require_tenant
from src.routers import exec_async
from src.services.exec_jobs import callback as cb

PREFIX = "http://203.0.113.10:5678/webhook-waiting/"
TENANT = "tenant-a"
WS = "ws-1"


@pytest.fixture
def runner():
    store = MagicMock()
    store.create_job = AsyncMock(side_effect=lambda **kw: ({"job_id": kw["job_id"], **kw}, True))
    store.get_job = AsyncMock(return_value=None)
    r = SimpleNamespace(store=store, owner="pod-a:1:x", lease_seconds=60, submit=MagicMock())
    with patch.object(exec_async, "get_exec_job_runner", return_value=r):
        yield r


@pytest.fixture
def owned_workspace():
    """get_container answers only for (WS, TENANT) — any other tenant is a miss."""
    manager = MagicMock()

    def _get(name, tenant_id=None):
        return SimpleNamespace(container_name=name) if (name, tenant_id) == (WS, TENANT) else None

    manager.get_container.side_effect = _get
    with patch.object(exec_async, "get_container_manager", return_value=manager):
        yield manager


@pytest.fixture
def as_tenant(app):
    app.dependency_overrides[require_tenant] = lambda: TENANT
    yield
    app.dependency_overrides.pop(require_tenant, None)


@pytest.fixture
def allow_prefix():
    with patch.object(
        cb, "get_settings", return_value=SimpleNamespace(exec_callback_prefixes=(PREFIX,))
    ):
        yield


async def test_accepts_and_returns_202_with_job_link(client, as_tenant, runner, owned_workspace):
    body = {"command": ["python", "-m", "pytest", "-q"], "timeout": 3600}
    resp = await client.post(f"/api/workspaces/{WS}/exec/async", json=body)
    assert resp.status_code == 202, resp.text
    data = resp.json()
    job_id = data["job_id"]
    assert uuid.UUID(job_id).version == 7
    assert data["status"] == "queued"
    assert data["links"]["status"] == f"/api/workspaces/{WS}/exec/jobs/{job_id}"
    assert resp.headers["location"] == data["links"]["status"]
    kw = runner.store.create_job.await_args.kwargs
    assert kw["tenant_id"] == TENANT and kw["container_name"] == WS
    assert kw["timeout_seconds"] == 3600
    assert kw["owner"] == runner.owner
    # The argv itself is never persisted, only its log-safe summary.
    assert "command" not in kw and isinstance(kw["command_summary"], str)
    job, argv, env = runner.submit.call_args.args
    assert job["job_id"] == job_id and argv == body["command"]
    assert env is None


async def test_env_reaches_the_runner_and_is_never_persisted(
    client, as_tenant, runner, owned_workspace
):
    """py2rust gap #5: ``env`` rides to the launch in memory, like argv."""
    secret = "postgresql://app:s3cret@127.0.0.1:5432/app"  # pragma: allowlist secret
    body = {"command": ["cargo", "test"], "env": {"DATABASE_URL": secret, "RUST_LOG": "debug"}}
    resp = await client.post(f"/api/workspaces/{WS}/exec/async", json=body)
    assert resp.status_code == 202, resp.text
    _job, argv, env = runner.submit.call_args.args
    assert argv == ["cargo", "test"] and env == body["env"]
    # The job row carries neither the values nor the names.
    kw = runner.store.create_job.await_args.kwargs
    assert secret not in repr(kw) and "DATABASE_URL" not in repr(kw)


@pytest.mark.parametrize(
    "env",
    [
        {"LD_PRELOAD": "/workspace/x.so"},
        {"PATH": "/workspace/bin"},
        {"IFS": "x"},
        {"lower": "x"},
        {"A": "x" * 8193},
    ],
    ids=lambda e: next(iter(e)),
)
async def test_refused_env_is_a_400_before_anything_is_persisted(
    client, as_tenant, runner, owned_workspace, env
):
    resp = await client.post(
        f"/api/workspaces/{WS}/exec/async", json={"command": ["cargo", "test"], "env": env}
    )
    assert resp.status_code == 400, resp.text
    assert "env refused" in resp.text
    runner.store.create_job.assert_not_awaited()
    runner.submit.assert_not_called()


async def test_timeout_above_sync_cap_is_allowed_up_to_four_hours(
    client, as_tenant, runner, owned_workspace
):
    ok = await client.post(
        f"/api/workspaces/{WS}/exec/async", json={"command": ["ls"], "timeout": 14400}
    )
    assert ok.status_code == 202
    over = await client.post(
        f"/api/workspaces/{WS}/exec/async", json={"command": ["ls"], "timeout": 14401}
    )
    assert over.status_code == 422


@pytest.mark.parametrize(
    "command", [[], ["rm", "-rf", "/"], ["find", ".", "-exec", "shutdown", ";"]], ids=str
)
async def test_boundary_command_gates_refuse_before_persisting(
    client, as_tenant, runner, owned_workspace, command
):
    resp = await client.post(f"/api/workspaces/{WS}/exec/async", json={"command": command})
    assert resp.status_code == 400
    runner.store.create_job.assert_not_awaited()


async def test_disallowed_callback_is_422_before_persisting(
    client, as_tenant, runner, owned_workspace, allow_prefix
):
    resp = await client.post(
        f"/api/workspaces/{WS}/exec/async",
        json={"command": ["ls"], "callback_url": "http://203.0.113.10/x"},
    )
    assert resp.status_code == 422
    runner.store.create_job.assert_not_awaited()


async def test_allowed_callback_is_persisted(
    client, as_tenant, runner, owned_workspace, allow_prefix
):
    resp = await client.post(
        f"/api/workspaces/{WS}/exec/async",
        json={"command": ["ls"], "callback_url": PREFIX + "4711"},
    )
    assert resp.status_code == 202
    assert runner.store.create_job.await_args.kwargs["callback_url"] == PREFIX + "4711"


async def test_foreign_workspace_is_404(client, app, runner, owned_workspace):
    app.dependency_overrides[require_tenant] = lambda: "tenant-b"
    try:
        resp = await client.post(f"/api/workspaces/{WS}/exec/async", json={"command": ["ls"]})
    finally:
        app.dependency_overrides.pop(require_tenant, None)
    assert resp.status_code == 404
    runner.store.create_job.assert_not_awaited()


async def test_lookup_error_is_404(client, as_tenant, runner, owned_workspace):
    owned_workspace.get_container.side_effect = ValueError("not yours")
    resp = await client.post(f"/api/workspaces/{WS}/exec/async", json={"command": ["ls"]})
    assert resp.status_code == 404


async def test_no_job_store_is_503(client, as_tenant, owned_workspace):
    with patch.object(exec_async, "get_exec_job_runner", return_value=None):
        post = await client.post(f"/api/workspaces/{WS}/exec/async", json={"command": ["ls"]})
        get = await client.get(f"/api/workspaces/{WS}/exec/jobs/{uuid.uuid7()}")
    assert post.status_code == 503
    assert get.status_code == 503


async def test_store_failure_on_create_is_503(client, as_tenant, runner, owned_workspace):
    runner.store.create_job.side_effect = OSError("db down")
    resp = await client.post(f"/api/workspaces/{WS}/exec/async", json={"command": ["ls"]})
    assert resp.status_code == 503
    runner.submit.assert_not_called()


async def test_status_read_is_scoped_to_caller_tenant_and_workspace(client, as_tenant, runner):
    job_id = str(uuid.uuid7())
    now = datetime.now(UTC)
    runner.store.get_job.return_value = {
        "job_id": job_id,
        "container_name": WS,
        "status": "completed",
        "exit_code": 0,
        "stdout": "ok",
        "stderr": "",
        "stdout_truncated": False,
        "stderr_truncated": True,
        "bytes_omitted": 7,
        "timed_out": False,
        "duration_seconds": 1.5,
        "error": None,
        "callback_state": "delivered",
        "created_at": now,
        "started_at": now,
        "completed_at": now,
    }
    resp = await client.get(f"/api/workspaces/{WS}/exec/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed" and data["truncated"] is True
    assert data["bytes_omitted"] == 7 and data["duration"] == 1.5
    assert runner.store.get_job.await_args.kwargs == {
        "job_id": job_id,
        "tenant_id": TENANT,
        "container_name": WS,
    }


async def test_status_unknown_or_foreign_job_is_404(client, as_tenant, runner):
    resp = await client.get(f"/api/workspaces/{WS}/exec/jobs/{uuid.uuid7()}")
    assert resp.status_code == 404


async def test_status_non_uuid_is_404_without_a_query(client, as_tenant, runner):
    resp = await client.get(f"/api/workspaces/{WS}/exec/jobs/not-a-uuid")
    assert resp.status_code == 404
    runner.store.get_job.assert_not_awaited()


async def test_status_store_failure_is_503(client, as_tenant, runner):
    runner.store.get_job.side_effect = OSError("db down")
    resp = await client.get(f"/api/workspaces/{WS}/exec/jobs/{uuid.uuid7()}")
    assert resp.status_code == 503


# ── the sync route is unchanged ───────────────────────────────────────


async def test_sync_exec_route_keeps_its_300s_cap(client, as_tenant):
    resp = await client.post(f"/api/workspaces/{WS}/exec", json={"command": ["ls"], "timeout": 301})
    assert resp.status_code == 422


def test_sync_exec_contract_is_unchanged(app):
    app.openapi_schema = None
    spec = app.openapi()
    op = spec["paths"]["/api/workspaces/{container_name}/exec"]["post"]
    assert op["operationId"] == "exec_in_workspace"
    schema = op["requestBody"]["content"]["application/json"]["schema"]
    # The sync body is ``ExecRequest | None``, so its schema is an anyOf.
    ref = next(s["$ref"] for s in schema.get("anyOf", [schema]) if "$ref" in s).rsplit("/", 1)[-1]
    assert ref == "ExecRequest"
    props = spec["components"]["schemas"][ref]["properties"]
    # ``env`` is the one ADDITIVE field (py2rust gap #5); everything else is as it was.
    assert set(props) == {"command", "workdir", "timeout", "env"}
    assert props["env"]["x-n8n-param-name"] == "execEnv"
    assert props["timeout"]["maximum"] == 300
    assert "x-n8n-param-name" not in props["timeout"]


def test_callback_url_description_states_how_long_the_key_is_held(app):
    """Feeds the generated n8n node: the key outlives queued/running while the
    callback is pending, and each replay outcome is spelled out."""
    app.openapi_schema = None
    spec = app.openapi()
    op = spec["paths"]["/api/workspaces/{container_name}/exec/async"]["post"]
    ref = op["requestBody"]["content"]["application/json"]["schema"]["$ref"].rsplit("/", 1)[-1]
    desc = spec["components"]["schemas"][ref]["properties"]["callback_url"]["description"]
    for phrase in (
        "OR its callback is still pending",
        "same job_id",
        "503 with Retry-After",
        "another workspace is a 409",
        "unique per-iteration suffix",
    ):
        assert phrase in desc, phrase
    assert "callback still pending" in op["responses"]["409"]["description"]


def test_async_routes_are_not_on_the_legacy_alias(app):
    # fastapi>=0.137 no longer flattens included routers into `app.routes` — the
    # list holds a private `_IncludedRouter` wrapper with no `.path` at all, so
    # `{r.path for r in app.routes}` silently collapses to the handful of routes
    # declared directly on the app and this assertion can never see the router's
    # paths. Reuse the accessor the router-mount tests already rely on rather
    # than keeping a second copy of the same workaround.
    from tests.unit.test_workspaces_router_mounts import _route_contexts

    paths = {path for path, _methods in _route_contexts(app.routes)}
    assert "/api/workspaces/{container_name}/exec/async" in paths
    assert "/api/containers/{container_name}/exec/async" not in paths


async def test_retried_post_with_same_callback_url_returns_the_existing_job(
    client, as_tenant, runner, owned_workspace, allow_prefix
):
    """D5: a client retry (after a 503 / lost response) must not start a second job."""
    existing = str(uuid.uuid7())
    runner.store.create_job.side_effect = None
    runner.store.create_job.return_value = (
        {"job_id": existing, "container_name": WS, "status": "running"},
        False,
    )
    resp = await client.post(
        f"/api/workspaces/{WS}/exec/async",
        json={"command": ["ls"], "callback_url": PREFIX + "4711"},
    )
    assert resp.status_code == 202, resp.text
    assert resp.json()["job_id"] == existing and resp.json()["status"] == "running"
    assert resp.headers["location"] == f"/api/workspaces/{WS}/exec/jobs/{existing}"
    runner.submit.assert_not_called()


async def test_callback_url_held_by_another_workspace_is_409(
    client, as_tenant, runner, owned_workspace, allow_prefix
):
    runner.store.create_job.side_effect = None
    runner.store.create_job.return_value = (
        {"job_id": str(uuid.uuid7()), "container_name": "ws-other", "status": "queued"},
        False,
    )
    resp = await client.post(
        f"/api/workspaces/{WS}/exec/async",
        json={"command": ["ls"], "callback_url": PREFIX + "4711"},
    )
    assert resp.status_code == 409
    runner.submit.assert_not_called()


async def test_replay_while_previous_callback_is_retrying_returns_the_existing_job(
    client, as_tenant, runner, owned_workspace, allow_prefix
):
    existing = str(uuid.uuid7())
    runner.store.create_job.side_effect = None
    runner.store.create_job.return_value = (
        {
            "job_id": existing,
            "container_name": WS,
            "status": "completed",
            "callback_state": "retrying",
        },
        False,
    )
    resp = await client.post(
        f"/api/workspaces/{WS}/exec/async",
        json={"command": ["ls"], "callback_url": PREFIX + "4711"},
    )
    assert resp.status_code == 202, resp.text
    assert resp.json()["job_id"] == existing and resp.json()["status"] == "completed"
    runner.submit.assert_not_called()


async def test_replay_while_previous_callback_is_sending_is_503_with_retry_after(
    client, as_tenant, runner, owned_workspace, allow_prefix
):
    """Delivery in flight, outcome unknown: the caller must retry, never a 2nd run."""
    runner.store.create_job.side_effect = None
    runner.store.create_job.return_value = (
        {
            "job_id": str(uuid.uuid7()),
            "container_name": WS,
            "status": "failed",
            "callback_state": "sending",
        },
        False,
    )
    resp = await client.post(
        f"/api/workspaces/{WS}/exec/async",
        json={"command": ["ls"], "callback_url": PREFIX + "4711"},
    )
    assert resp.status_code == 503, resp.text
    assert resp.headers["retry-after"] == "2"
    runner.submit.assert_not_called()


async def test_sending_holder_on_another_workspace_is_still_409(
    client, as_tenant, runner, owned_workspace, allow_prefix
):
    runner.store.create_job.side_effect = None
    runner.store.create_job.return_value = (
        {
            "job_id": str(uuid.uuid7()),
            "container_name": "ws-other",
            "status": "completed",
            "callback_state": "sending",
        },
        False,
    )
    resp = await client.post(
        f"/api/workspaces/{WS}/exec/async",
        json={"command": ["ls"], "callback_url": PREFIX + "4711"},
    )
    assert resp.status_code == 409
    runner.submit.assert_not_called()

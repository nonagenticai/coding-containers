"""#135 — the three routes that bound a tenant and never used it, gated END TO END.

THE DEFECT
----------
``src/routers/agents.py`` bound ``tenant_id = Depends(require_tenant)`` on both
its routes and then never referenced it: measured on the pre-fix file,
``tenant_id`` occurred on exactly two lines in the whole module, ``:66`` and
``:134``, and BOTH were the binding. ``src/routers/validation.py`` had the same
shape (``:17`` the import, ``:172`` the binding). So the caller was
AUTHENTICATED and the target workspace was chosen by a caller-supplied NAME that
was never compared against the caller's tenant.

Nothing downstream could have caught it. ``AgentService.execute_agent`` reads the
registry row directly and dispatches ``k8s.exec_in_pod(deployment_name=...)`` or
``docker_service.exec_in_container(docker_id=...)``; the validation route reaches
``exec_in_container(docker_id=<caller-supplied name>)``. None of those three
take a tenant argument at all.

WHY THIS FILE STUBS ONLY THE PROCESS BOUNDARY
---------------------------------------------
#133 needed FOUR rounds for this same invariant, and every round was late for the
same reason: the tests stubbed the very path that held the gap. The sharpest case
is recorded in ``test_docker_service.py`` — nothing there had ever returned a
container from ``reconstruct_container_info``, so that branch's gate was never
executed by any test while the suite stayed green.

The pre-existing ``test_agents_router.py`` is an instance of exactly that: all 9
of its tests patch ``AgentService.execute_agent`` or hand the router a mock
docker service, so the router's own authorization could be absent — as it WAS —
with every one of them green. Its 28 ``tenant`` occurrences are bindings and
``dependency_overrides`` lines; not one asserts a cross-tenant denial. (That was
the open question in #135: "whether any test asserts cross-tenant denial here is
unknown." Measured here — none did.)

So these tests stub exactly two things, both genuine process boundaries:

    * the Kubernetes API client and the Redis registry (as #133's
      ``test_registry_miss_tenant_scope.py`` does), and
    * the exec sink itself — ``exec_in_container`` — which is kept as a SPY.

The router, ``ContainerManager.get_container``, the gate, ``reconstruct_container_info``
and ``_recover_tenant_id`` all run for real. Every negative test asserts that the
spy was NOT awaited, so a 404 produced by a route that never looked anything up
cannot pass as a refusal, and a 404 that still executed in the foreign workspace
cannot pass at all.

MUTATION CONTROLS — run, measured, and reported as run
------------------------------------------------------
Each gate was removed and this file re-run. BASELINE: **13 passed**
(re-measured 2026-10-11, after the validation route gained a second gate).

    M1  ``authorize_workspace`` call deleted from ``execute_agent``
        (agents.py ``POST /{name}/execute-agent``).
        MEASURED: **2 failed / 11 passed** — ``test_execute_agent_foreign_tenant_is_404``
        and ``test_execute_agent_unrecoverable_owner_is_denied``.

    M2  ``authorize_workspace`` call deleted from ``check_agent_capabilities``
        (agents.py ``GET /{name}/agent-capabilities``).
        MEASURED: **2 failed / 11 passed** — the two ``test_capabilities_*`` negatives.

    M3a ``authorize_workspace`` call deleted from ``_execute_in_container``
        (validation.py). MEASURED: **1 failed / 12 passed** — only
        ``test_execute_in_container_authorizes_itself``. ⚠️ Every ROUTE test
        stays green under M3a, because ``_resolve_imports`` now refuses first
        whenever a ``container_name`` is given; that is why the helper is also
        tested in isolation.

    M3b ``authorize_workspace`` call deleted from ``_resolve_imports``
        (validation.py). MEASURED: **1 failed / 12 passed** — only
        ``test_validation_resolver_only_foreign_tenant_is_404`` (it answers 200
        ``not_computed`` instead of 404). With execution requested the route's
        tests stay green: ``_execute_in_container``'s gate still refuses.

    ⭐ Each of M1/M2/M3a/M3b reddens ONLY its own gate's tests. That is the
    property that matters: each gate is held by its own call and none is riding
    on another's. A single shared mutation could not have shown this — separate
    ones were needed, which is precisely the lesson #133 paid four rounds to learn.

    M4  the ``if container is None`` fail-closed branch in
        ``tenant_guard.authorize_workspace`` replaced by ``if False:`` —
        resolution still happens, refusal does not.
        MEASURED: **8 failed / 5 passed**. The 8 are every negative test in this
        file; the 5 survivors are exactly the positive controls. That split is
        the signature of a gate that refuses for the right reason rather than a
        route that 404s everybody.

    M5  the ``logger.warning`` in that same branch downgraded to ``debug``,
        proving the ``caplog`` assertions are load-bearing rather than decoration.
        MEASURED: **5 failed / 8 passed** — the five ``_assert_refusal_logged``
        callers. The unrecoverable-owner tests correctly survive: their WARNING
        comes from ``ContainerManager.get_container`` itself, not from this
        branch, so they are guarded by a different mechanism and cannot prove
        this one.
"""

import contextlib
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Same two tenants as #133's trace, kept concrete so the PRs read as one story.
OWNER = "0195f48b-d000-7000-8000-000000000001"
FOREIGN = "0195f48b-d000-7000-8000-000000000003"
WORKSPACE = "aa-delivery-chat-bridge"

pytestmark = pytest.mark.asyncio


def _deployment(*, tenant_label, tenant_annotation=None):
    """A live managed Deployment as the k8s client would return it."""
    dep = MagicMock()
    labels = {"app": "codegraph-repo", "managed-by": "code-graph-helper"}
    if tenant_label:
        labels["tenant-id"] = tenant_label
    dep.metadata.labels = labels
    annotations = {"coding-job/repo-url": "https://github.com/nonagenticai/x"}
    if tenant_annotation:
        annotations["coding-job/tenant-id"] = tenant_annotation
    dep.metadata.annotations = annotations
    dep.status.ready_replicas = 1
    dep.spec.replicas = 1
    return dep


class _Harness:
    """Real ContainerManager + real provisioner + real gate; k8s, Redis and the
    exec sink are the only fakes.

    ``exec_sink`` is the spy that makes the negative tests meaningful: it stands
    exactly where ``docker exec`` would run caller-supplied code inside the named
    workspace. If a gate is missing, it gets awaited.
    """

    def __init__(self, stack, app, *, tenant_label=OWNER, tenant_annotation=None):
        from src.core import kubernetes_container_provisioner as kmod
        from src.core.container_manager import ContainerManager

        # ⚠️ Captured BEFORE any patching below. FastAPI keys
        # ``dependency_overrides`` on the ORIGINAL function object it captured at
        # import time, so this has to be the real function — importing it further
        # down, while ``patch("src.services.docker_service.get_docker_service")``
        # is active, binds the MagicMock instead and the override then matches
        # nothing. MEASURED: that ordering bug let both execute-agent positive
        # controls drive the REAL Docker daemon on this machine
        # ("404 ... No such container: aa-delivery-chat-bridge").
        from src.services.docker_service import get_docker_service as _real_dep

        stack.enter_context(patch("src.core.container_manager.get_core_manager"))
        stack.enter_context(patch("src.core.container_manager.get_github_container_provisioner"))
        mock_ds = stack.enter_context(patch("src.core.container_manager.get_docker_service"))
        stack.enter_context(patch("kubernetes.config.load_kube_config"))
        settings = stack.enter_context(patch.object(kmod, "get_settings"))
        settings.return_value.main_api_endpoint = "https://api.example.test"

        # THE REGISTRY MISS — routine, per get_container's own comment, and the
        # state in which the owner must be recovered from the Deployment.
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        self.provisioner = kmod.K8sContainerProvisioner(namespace="coding-jobs", in_cluster=False)
        self.provisioner.apps_v1 = MagicMock()
        self.provisioner.core_v1 = MagicMock()
        self.provisioner.apps_v1.read_namespaced_deployment.return_value = _deployment(
            tenant_label=tenant_label, tenant_annotation=tenant_annotation
        )

        self.manager = ContainerManager()
        self.manager._github.use_k8s = True
        self.manager._github.k8s_provisioner = self.provisioner

        # THE SPY. An AsyncMock so the real AgentService can await it, returning
        # the shape a successful `which claude` / agent run produces.
        self.exec_sink = AsyncMock(
            return_value={"exit_code": 0, "stdout": "ok", "stderr": "", "duration": 0.1}
        )
        self.docker_service = MagicMock()
        self.docker_service.exec_in_container = self.exec_sink
        self.docker_service.is_docker_available.return_value = True
        self.docker_service.registry = MagicMock()
        self.docker_service.registry.load.return_value = {}

        # The gate resolves through this; the manager underneath it is real.
        stack.enter_context(
            patch("src.routers.tenant_guard.get_container_manager", return_value=self.manager)
        )
        # The three dispatch entry points, all pointed at the one spy.
        stack.enter_context(
            patch("src.routers.agents.get_docker_service", return_value=self.docker_service)
        )
        stack.enter_context(
            patch(
                "src.services.docker_service.get_docker_service",
                return_value=self.docker_service,
            )
        )
        # The validation route execs through the runtime layer
        # (container_exec.run_workspace_command -> resolve_runtime -> run_once),
        # not docker_service. Its lookup runs against the SAME real manager;
        # only the backend's run_once is the spy.
        from src.runtime import RunOnceResult, RuntimeKind

        self.run_sink = AsyncMock(
            return_value=RunOnceResult(exit_code=0, stdout="ok", stderr="", duration=0.1)
        )
        runtime = MagicMock()
        runtime.kind = RuntimeKind.KUBERNETES
        runtime.run_once = self.run_sink
        stack.enter_context(
            patch("src.routers.container_exec.get_container_manager", return_value=self.manager)
        )
        stack.enter_context(
            patch("src.routers.container_exec.resolve_runtime", return_value=runtime)
        )
        # agent-capabilities probes through the runtime too (D4).
        stack.enter_context(patch("src.routers.agents.resolve_runtime", return_value=runtime))
        stack.enter_context(patch("src.core.container_manager_core.get_container_manager"))
        # ⚠️ AND the FastAPI dependency, which is NOT the same thing. AgentService
        # receives its docker service through ``Depends(get_docker_service)``;
        # patching the module attribute does not reach what FastAPI already
        # captured, so without this override the route runs against the real
        # Docker daemon and the spy is never awaited — a negative test would then
        # "pass" while proving nothing.
        app.dependency_overrides[_real_dep] = lambda: self.docker_service

    @property
    def issued(self):
        """What was ISSUED against the cluster, as the k8s client saw it."""
        return self.provisioner.apps_v1.read_namespaced_deployment.call_args


@contextlib.contextmanager
def _as_tenant(app, tenant_id):
    """Bind the caller's tenant exactly as ``require_tenant`` would."""
    from coding_client.auth.tenant import get_current_user_with_tenant, require_tenant

    user = MagicMock()
    user.sub = "test-user"
    user.preferred_username = "testuser"
    user.effective_tenant_id = tenant_id
    user.is_admin.return_value = False

    app.dependency_overrides[get_current_user_with_tenant] = lambda: user
    app.dependency_overrides[require_tenant] = lambda: tenant_id
    try:
        yield
    finally:
        app.dependency_overrides.clear()


def _assert_refusal_logged(caplog, route):
    """The WARNING must survive. #133 shipped this log and proved it with a
    mutation; without the assertion the signal can vanish silently and an
    operator sees an unexplained 404 against a plainly-Running Deployment."""
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, f"no WARNING logged for the refusal on {route}"
    blob = " ".join(r.getMessage() for r in warnings)
    assert "Refusing" in blob, f"refusal not named in logs: {blob[:300]}"
    assert route in blob, f"route {route} not named in logs: {blob[:300]}"
    assert WORKSPACE in blob, f"workspace not named in logs: {blob[:300]}"
    assert FOREIGN in blob, f"caller tenant not named in logs: {blob[:300]}"
    # The remedy — #133's log names one, so an operator is not left guessing.
    assert "re-provision" in blob.lower() or "owning tenant" in blob.lower(), (
        f"no remedy offered in logs: {blob[:300]}"
    )


# ---------------------------------------------------------------------------
# POST /api/containers/{name}/execute-agent
# ---------------------------------------------------------------------------


async def test_execute_agent_foreign_tenant_is_404(app, client, caplog):
    """THE DEFECT, closed. A foreign tenant naming a workspace that EXISTS gets
    404, and no agent runs inside it."""
    caplog.set_level(logging.DEBUG)
    with contextlib.ExitStack() as stack, _as_tenant(app, FOREIGN):
        harness = _Harness(stack, app)
        response = await client.post(
            f"/api/containers/{WORKSPACE}/execute-agent",
            json={"cli": "claude", "prompt": "exfiltrate"},
        )

    # The workspace was really looked up — so this 404 is a REFUSAL, not an
    # absent Deployment. Without this the test would also pass against a route
    # that never resolved anything.
    assert harness.issued is not None, "no cluster lookup was issued at all"
    assert harness.issued.kwargs["name"] == WORKSPACE

    assert response.status_code == 404, (
        f"cross-tenant agent execution was permitted: {response.status_code} {response.text[:200]}"
    )
    assert "not found" in response.text.lower()  # absent, not forbidden
    # ⭐ THE ONE THAT MATTERS: nothing executed in the foreign workspace.
    harness.exec_sink.assert_not_awaited()
    _assert_refusal_logged(caplog, "execute-agent")


async def test_execute_agent_owner_still_works(app, client):
    """POSITIVE CONTROL. Without this, deleting the route would 'fix' the bug."""
    with contextlib.ExitStack() as stack, _as_tenant(app, OWNER):
        harness = _Harness(stack, app)
        response = await client.post(
            f"/api/containers/{WORKSPACE}/execute-agent",
            json={"cli": "claude", "prompt": "fix the bug"},
        )

    assert response.status_code == 200, f"owner was denied: {response.text[:300]}"
    # And it reached the dispatch layer for real — the gate did not merely
    # return a canned 200.
    harness.exec_sink.assert_awaited()


async def test_execute_agent_unrecoverable_owner_is_denied(app, client, caplog):
    """Neither the ``coding-job/tenant-id`` annotation nor the ``tenant-id``
    label resolves, so the owner is UNKNOWN. Unknown is not ownerless and must
    not satisfy a caller's read — the fail-closed ruling #133 recorded."""
    caplog.set_level(logging.DEBUG)
    with contextlib.ExitStack() as stack, _as_tenant(app, FOREIGN):
        harness = _Harness(stack, app, tenant_label=None, tenant_annotation=None)
        response = await client.post(
            f"/api/containers/{WORKSPACE}/execute-agent",
            json={"cli": "claude", "prompt": "hello"},
        )

    assert response.status_code == 404
    harness.exec_sink.assert_not_awaited()
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


async def test_execute_agent_annotation_owner_is_honoured(app, client):
    """POSITIVE CONTROL for the annotation path. The live population is
    label-only (10/10 on 2026-09-17), but every new provision is annotated, so
    the annotation must be a working credential and not dead code."""
    with contextlib.ExitStack() as stack, _as_tenant(app, OWNER):
        harness = _Harness(stack, app, tenant_label=None, tenant_annotation=OWNER)
        response = await client.post(
            f"/api/containers/{WORKSPACE}/execute-agent",
            json={"cli": "claude", "prompt": "fix the bug"},
        )

    assert response.status_code == 200, f"annotation owner was denied: {response.text[:300]}"
    harness.exec_sink.assert_awaited()


# ---------------------------------------------------------------------------
# GET /api/containers/{name}/agent-capabilities
# ---------------------------------------------------------------------------


async def test_capabilities_foreign_tenant_is_404(app, client, caplog):
    """This route runs ``which claude`` INSIDE the named container, so an ungated
    call is both a cross-tenant exec and an existence oracle for the name."""
    caplog.set_level(logging.DEBUG)
    with contextlib.ExitStack() as stack, _as_tenant(app, FOREIGN):
        harness = _Harness(stack, app)
        response = await client.get(f"/api/containers/{WORKSPACE}/agent-capabilities")

    assert harness.issued is not None, "no cluster lookup was issued at all"
    assert response.status_code == 404, (
        f"cross-tenant capability probe was permitted: {response.status_code} {response.text[:200]}"
    )
    harness.exec_sink.assert_not_awaited()
    harness.run_sink.assert_not_awaited()
    _assert_refusal_logged(caplog, "agent-capabilities")


async def test_capabilities_owner_still_works(app, client):
    """POSITIVE CONTROL."""
    with contextlib.ExitStack() as stack, _as_tenant(app, OWNER):
        harness = _Harness(stack, app)
        response = await client.get(f"/api/containers/{WORKSPACE}/agent-capabilities")

    assert response.status_code == 200, f"owner was denied: {response.text[:300]}"
    assert "claude" in response.json()["capabilities"]
    harness.run_sink.assert_awaited()


async def test_capabilities_unrecoverable_owner_is_denied(app, client):
    with contextlib.ExitStack() as stack, _as_tenant(app, FOREIGN):
        harness = _Harness(stack, app, tenant_label=None, tenant_annotation=None)
        response = await client.get(f"/api/containers/{WORKSPACE}/agent-capabilities")

    assert response.status_code == 404
    harness.exec_sink.assert_not_awaited()
    harness.run_sink.assert_not_awaited()


# ---------------------------------------------------------------------------
# POST /api/validation/multi-line-code-snippet
# ---------------------------------------------------------------------------

_SNIPPET = "import os\nprint(os.environ)\n"


async def test_validation_foreign_tenant_is_404(app, client, caplog):
    """``_execute_in_container`` runs caller-supplied Python (and
    ``_resolve_imports`` runs the import resolver) in the caller-named
    workspace. A foreign tenant must not reach either."""
    caplog.set_level(logging.DEBUG)
    with contextlib.ExitStack() as stack, _as_tenant(app, FOREIGN):
        harness = _Harness(stack, app)
        response = await client.post(
            "/api/validation/multi-line-code-snippet",
            json={
                "code_snippet": _SNIPPET,
                "container_name": WORKSPACE,
                "include_execution_validation": True,
            },
        )

    assert harness.issued is not None, "no cluster lookup was issued at all"
    assert response.status_code == 404, (
        f"cross-tenant code execution was permitted: {response.status_code} {response.text[:200]}"
    )
    # ⛔ Specifically NOT a 200 carrying execution_successful:false — the helper's
    # own `except` would have produced that, and the caller would have kept its
    # foreign workspace name believing it merely failed to run.
    harness.exec_sink.assert_not_awaited()
    harness.run_sink.assert_not_awaited()
    _assert_refusal_logged(caplog, "multi-line-code-snippet")


async def test_validation_owner_still_executes(app, client):
    """POSITIVE CONTROL — the owner's snippet still runs in the owner's workspace."""
    with contextlib.ExitStack() as stack, _as_tenant(app, OWNER):
        harness = _Harness(stack, app)
        response = await client.post(
            "/api/validation/multi-line-code-snippet",
            json={
                "code_snippet": _SNIPPET,
                "container_name": WORKSPACE,
                "include_execution_validation": True,
            },
        )

    assert response.status_code == 200, f"owner was denied: {response.text[:300]}"
    assert response.json()["execution_analysis"]["execution_successful"] is True
    # Executed through the runtime layer, in the owner's workspace, as the
    # caller's own snippet — not through docker_service.
    harness.exec_sink.assert_not_awaited()
    argvs = [c.args[1] for c in harness.run_sink.await_args_list]
    assert ["python", "-c", _SNIPPET] in argvs, argvs
    assert harness.issued.kwargs["name"] == WORKSPACE


async def test_validation_unrecoverable_owner_is_denied(app, client):
    with contextlib.ExitStack() as stack, _as_tenant(app, FOREIGN):
        harness = _Harness(stack, app, tenant_label=None, tenant_annotation=None)
        response = await client.post(
            "/api/validation/multi-line-code-snippet",
            json={
                "code_snippet": _SNIPPET,
                "container_name": WORKSPACE,
                "include_execution_validation": True,
            },
        )

    assert response.status_code == 404
    harness.exec_sink.assert_not_awaited()
    harness.run_sink.assert_not_awaited()


async def test_validation_without_execution_is_unaffected(app, client):
    """POSITIVE CONTROL / blast radius. Static-only validation never names a
    workspace and never execs, so it must NOT have acquired a 404."""
    with contextlib.ExitStack() as stack, _as_tenant(app, FOREIGN):
        harness = _Harness(stack, app)
        response = await client.post(
            "/api/validation/multi-line-code-snippet",
            json={"code_snippet": _SNIPPET, "include_execution_validation": False},
        )

    assert response.status_code == 200, (
        f"static-only validation regressed to {response.status_code}: {response.text[:200]}"
    )
    assert response.json()["execution_analysis"] is None
    harness.run_sink.assert_not_awaited()
    harness.exec_sink.assert_not_awaited()


async def test_validation_resolver_only_foreign_tenant_is_404(app, client, caplog):
    """Import resolution alone (no execution) also execs in the named workspace,
    so ``_resolve_imports`` must refuse on its OWN gate. Without it the route's
    later lookup would 404 inside the resolver's ``except`` and answer 200
    ``not_computed`` — a foreign caller told nothing was wrong."""
    caplog.set_level(logging.DEBUG)
    with contextlib.ExitStack() as stack, _as_tenant(app, FOREIGN):
        harness = _Harness(stack, app)
        response = await client.post(
            "/api/validation/multi-line-code-snippet",
            json={
                "code_snippet": _SNIPPET,
                "container_name": WORKSPACE,
                "include_execution_validation": False,
            },
        )

    assert harness.issued is not None, "no cluster lookup was issued at all"
    assert response.status_code == 404, (
        f"cross-tenant import resolution was permitted: {response.status_code} "
        f"{response.text[:200]}"
    )
    harness.run_sink.assert_not_awaited()
    _assert_refusal_logged(caplog, "multi-line-code-snippet")


async def test_execute_in_container_authorizes_itself(app, caplog):
    """ISOLATED from the route. Through the route, ``_resolve_imports`` refuses
    first, so the route tests stay green even with this helper's own gate
    deleted (M3a). Calling the helper directly is the only thing that proves its
    gate exists — and that a refusal is a 404, not a body reporting an unrun
    snippet."""
    from fastapi import HTTPException

    from src.routers.validation import _execute_in_container

    caplog.set_level(logging.DEBUG)
    with contextlib.ExitStack() as stack:
        harness = _Harness(stack, app)
        with pytest.raises(HTTPException) as refused:
            await _execute_in_container(_SNIPPET, WORKSPACE, 10, FOREIGN)
        assert refused.value.status_code == 404
        harness.run_sink.assert_not_awaited()
        _assert_refusal_logged(caplog, "multi-line-code-snippet")

        # Positive control: the owner runs, through the same helper.
        result = await _execute_in_container(_SNIPPET, WORKSPACE, 10, OWNER)
    app.dependency_overrides.clear()

    assert result.execution_status == "computed"
    assert result.execution_successful is True
    assert [c.args[1] for c in harness.run_sink.await_args_list] == [["python", "-c", _SNIPPET]]

"""The registry-MISS path must scope by tenant — exercised END TO END.

WHY THIS FILE EXISTS SEPARATELY FROM ``test_containers_router.py``
------------------------------------------------------------------
#133 fixed ``GET /{name}/status`` by deleting a ``tenant_id=None`` retry in the
router, and shipped eight tests for it. Every one of those eight stubs
``manager.get_container`` — so not one of them can see what the real
``ContainerManager.get_container`` does, and the leak lived one frame below the
stub:

    reconstruct_container_info()  ->  ContainerInfo(..., tenant_id=None)
    get_container()               ->  if container_tenant is not None and ...

On a registry MISS — which ``get_container``'s own comment calls routine, since
the Redis registry "is transient and can momentarily lose a freshly-stored
entry" — ``container_tenant`` was ``None``, the inequality never evaluated, and
``/status`` returned **200 + running** to a foreign tenant. ``/exec`` takes the
identical path. Measured against the merged #133 head (fa3eafe) before this
change: ``STATUS: 200 ... "container_status":"running"``.

So these tests stub exactly ONE thing — the Kubernetes API client and the
registry, i.e. the process boundary — and let the router, the manager, the gate
and the reconstruct all run for real. A test that stubs the component under test
cannot fail for the right reason.

MUTATION CONTROLS — run, with their measured numbers
----------------------------------------------------
    M1  reconstruct_container_info: ``tenant_id = labels.get(...)`` -> ``None``,
        i.e. restore the leak at its source.
        MEASURED: 3 failed / 194 passed. Both ``test_foreign_*`` go red; every
        positive control stays green.

    M3  ``get_container_status``: restore the ``tenant_id=None`` retry #133
        deleted, at its ORIGINAL position inside the ``try``.
        MEASURED: 2 failed / 148 passed on ``test_containers_router.py`` (the
        same two tests #133 reported), and 1 failed / 4 passed here — so this
        file catches the router regression independently of that suite.
        ⚠️ Restoring it AFTER the ``try``/``except`` instead gives 4 failed, not
        2, because a ``ValueError`` path then gets a retry it never had. The
        control is only reproducible at the original position.

⚠️ The paragraph that stood here said restoring ``container_tenant is not None``
in ``ContainerManager.get_container`` is "NOT a control for this file", because
that fail-open "was deliberately NOT removed". It HAS now been removed, on the
reconstructed path only — see ``test_unrecoverable_owner_is_404_not_readable``,
which is the inversion of the test that paragraph cited as its justification.

THE UNLABELABLE-OWNER ROUND — a second leak through the same door
-----------------------------------------------------------------
Recovering the tenant from the ``tenant-id`` LABEL closed the leak only for
owners the label can hold. ``job_labels[TENANT_LABEL_KEY]`` is written solely
when the value matches ``_K8S_LABEL_VALUE_RE`` ("otherwise omit rather than fail
provisioning"), while ``effective_tenant_id`` is ``tenant_id or organization_id
or sub`` with NO grammar validation. So an OWNED workspace could be provisioned
carrying no ownership stamp at all, and "no label" was then read as "no owner":

    read   ``reconstruct_container_info`` -> None -> gate vacuous -> foreign 200
    write  the 409 ownership guard read the incumbent's claim out of that SAME
           label, so the owner presented as "no claim", the guard did not fire,
           and ``patch_namespaced_deployment`` re-stamped the workspace to the
           caller and rolled a fresh pod. Takeover, with the incumbent's work
           destroyed, returning 200 OK.

The fix stamps the tenant as the ``coding-job/tenant-id`` ANNOTATION as well
(annotations have no grammar), recovers annotation-then-label at BOTH sites, and
fails CLOSED when neither yields an owner.

    M-A  ``_recover_tenant_id`` -> label only (annotation recovery neutered).
         MEASURED: 3 failed / 451 passed. ⚠️ The annotation-only NEGATIVE tests
         survive this one, because the fail-closed branch denies them for a
         different reason — so the tests that discriminate here are the POSITIVE
         controls (``test_annotation_owner_still_reads_its_own_workspace``,
         ``test_annotation_wins_over_a_stale_label``,
         ``test_409_reconciles_when_annotation_owner_matches_the_caller``).
         Reported explicitly: a negative test guarded by two mechanisms cannot
         by itself prove either of them.

    M-B  both fail-CLOSED branches -> ``if False:`` (recovery left intact).
         MEASURED: 2 failed / 452 passed — the two unrecoverable-owner tests.

    M-C  M-A + M-B, i.e. the whole read-path fix removed.
         MEASURED: 5 failed / 449 passed.

    M-D  the provision-time annotation STAMP deleted.
         MEASURED: 2 failed / 452 passed.

    M-E  the EXACT pre-change write guard restored — label-only claim AND the
         original ``incumbent_tenant and tenant_id and ...`` condition, which
         allowed an unrecoverable incumbent through.
         MEASURED: 2 failed / 452 passed —
         ``test_409_refuses_takeover_when_owner_is_only_in_the_annotation`` and
         ``test_409_refuses_an_incumbent_whose_owner_is_unrecoverable``.
         ⚠️ M-C does NOT redden those two: splitting the guard into an
         unknown-owner branch plus a mismatch branch means disabling only the
         first leaves the second refusing ``None != caller``. M-E is the faithful
         write-path control; M-C is not a substitute for it.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The 2026-09-17 production pair, kept concrete so the trace in the PR and the
# trace in the test are the same two tenants.
OWNER = "0195f48b-d000-7000-8000-000000000001"
FOREIGN = "0195f48b-d000-7000-8000-000000000003"
WORKSPACE = "aa-delivery-chat-bridge"


def _mock_user(*, is_admin=False, tenant_id=FOREIGN):
    user = MagicMock()
    user.sub = "test-user"
    user.preferred_username = "testuser"
    user.effective_tenant_id = tenant_id
    user.is_admin.return_value = is_admin
    return user


def _deployment(*, tenant_label, tenant_annotation=None):
    """A live managed Deployment as the k8s client would return it.

    ``tenant_label`` and ``tenant_annotation`` are independent on purpose — the
    whole defect lives in the states where they DISAGREE about whether an owner
    exists. The live population is label-only (10 of 10, re-measured 2026-09-17), a
    grammar-invalid owner is annotation-only, and neither-present is the
    unknown-owner residual.
    """
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
    """Real ContainerManager + real provisioner; only k8s and Redis are fakes."""

    def __init__(self, stack, *, tenant_label, tenant_annotation=None):
        from src.core import kubernetes_container_provisioner as kmod
        from src.core.container_manager import ContainerManager

        stack.enter_context(patch("src.core.container_manager.get_core_manager"))
        stack.enter_context(patch("src.core.container_manager.get_github_container_provisioner"))
        mock_ds = stack.enter_context(patch("src.core.container_manager.get_docker_service"))
        stack.enter_context(patch("kubernetes.config.load_kube_config"))
        settings = stack.enter_context(patch.object(kmod, "get_settings"))
        settings.return_value.main_api_endpoint = "https://api.example.test"

        docker_svc = MagicMock()
        # THE REGISTRY MISS. Not an exotic state: the fallback below exists
        # precisely because this happens in normal operation.
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
        # Only the readiness read itself is stubbed, so the test can assert it was
        # never reached for a foreign caller (a 404 that still probes the backend
        # is a timing oracle).
        self.manager.get_container_status = AsyncMock(return_value="running")

    @property
    def issued(self):
        """What was ISSUED against the cluster, as the k8s client saw it."""
        return self.provisioner.apps_v1.read_namespaced_deployment.call_args


@pytest.mark.asyncio
async def test_foreign_tenant_gets_404_on_registry_miss(app, client):
    """The leak, closed: foreign tenant reads a reconstructed workspace as ABSENT."""
    import contextlib

    from coding_client.auth.tenant import get_current_user_with_tenant

    app.dependency_overrides[get_current_user_with_tenant] = lambda: _mock_user(tenant_id=FOREIGN)
    with contextlib.ExitStack() as stack:
        harness = _Harness(stack, tenant_label=OWNER)
        with patch("src.routers.containers.get_container_manager", return_value=harness.manager):
            response = await client.get(f"/api/containers/{WORKSPACE}/status")
    app.dependency_overrides.clear()

    # What was ISSUED: the lookup did reach the cluster and did name this
    # workspace — the 404 is a REFUSAL, not an absent Deployment. Without this
    # the test would also pass against a route that never looked anything up.
    assert harness.issued is not None, "no cluster lookup was issued at all"
    assert harness.issued.kwargs["name"] == WORKSPACE
    assert harness.issued.kwargs["namespace"] == "coding-jobs"

    # What was RETURNED to the foreign tenant.
    assert response.status_code == 404, (
        f"registry-miss path disclosed a foreign workspace: {response.status_code} "
        f"{response.text[:200]}"
    )
    # Absent, not forbidden — a 403 would confirm the name exists.
    assert "not found" in response.text.lower()
    # And readiness was never read on a foreign caller's behalf.
    harness.manager.get_container_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_owning_tenant_still_reads_status_on_registry_miss(app, client):
    """POSITIVE CONTROL. The whole point of the registry-miss fallback is that the
    OWNER keeps working while Redis is momentarily missing the row. If this goes
    red, the fix has traded a leak for an outage."""
    import contextlib

    from coding_client.auth.tenant import get_current_user_with_tenant

    app.dependency_overrides[get_current_user_with_tenant] = lambda: _mock_user(tenant_id=OWNER)
    with contextlib.ExitStack() as stack:
        harness = _Harness(stack, tenant_label=OWNER)
        with patch("src.routers.containers.get_container_manager", return_value=harness.manager):
            response = await client.get(f"/api/containers/{WORKSPACE}/status")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["container_status"] == "running"
    harness.manager.get_container_status.assert_awaited_once()


@pytest.mark.asyncio
async def test_admin_still_reads_foreign_status_on_registry_miss(app, client):
    """POSITIVE CONTROL. The documented escape hatch for a fleet-wide readiness
    poller is admin (``is_admin()`` -> ``tenant_id=None``, gate not reached). It
    must survive, because it is the ONLY sanctioned cross-tenant reader now that
    the comment no longer offers "provision your own workspace on the 404"."""
    import contextlib

    from coding_client.auth.tenant import get_current_user_with_tenant

    app.dependency_overrides[get_current_user_with_tenant] = lambda: _mock_user(
        is_admin=True, tenant_id=FOREIGN
    )
    with contextlib.ExitStack() as stack:
        harness = _Harness(stack, tenant_label=OWNER)
        with patch("src.routers.containers.get_container_manager", return_value=harness.manager):
            response = await client.get(f"/api/containers/{WORKSPACE}/status")
    app.dependency_overrides.clear()

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_unrecoverable_owner_is_404_not_readable(app, client):
    """⭐ THE READ FAIL-OPEN, closed. ⚠️ INVERTED — this test asserted **200**.

    It was ``test_ownerless_workspace_stays_readable_on_registry_miss`` and it
    read a missing ``tenant-id`` label as PROOF that the workspace is OWNERLESS,
    explicitly distinguishing that from "owner unknown". That distinction cannot
    be drawn from the Deployment: the label is written only when the tenant
    satisfies ``_K8S_LABEL_VALUE_RE``, and ``effective_tenant_id`` is
    ``tenant_id or organization_id or sub`` with no grammar validation. So an
    unlabelled Deployment is equally consistent with an OWNED workspace whose
    owner could not be expressed as a label — and for that workspace the old
    assertion WAS the leak it was standing next to: a foreign tenant got 200 and
    ``container_status: running``.

    Now that the annotation records every owner regardless of grammar, neither
    stamp resolving means UNKNOWN, and unknown must not satisfy a foreign
    caller's read.

    The inverted test's own objection — that failing closed strands the legacy
    population on ``health.py`` and ``container_exec.py``, which have no
    ``is_admin()`` widening — is answered by measurement rather than by argument:
    live ``coding-jobs`` on 2026-09-17 carried 10 Deployments, 10 of them labelled.
    The stranded population is EMPTY, and every provision from here on writes the
    annotation.
    """
    import contextlib

    from coding_client.auth.tenant import get_current_user_with_tenant

    app.dependency_overrides[get_current_user_with_tenant] = lambda: _mock_user(tenant_id=FOREIGN)
    with contextlib.ExitStack() as stack:
        harness = _Harness(stack, tenant_label=None, tenant_annotation=None)
        with patch("src.routers.containers.get_container_manager", return_value=harness.manager):
            response = await client.get(f"/api/containers/{WORKSPACE}/status")
    app.dependency_overrides.clear()

    assert response.status_code == 404, (
        f"a workspace with NO recoverable owner was served to a foreign tenant: "
        f"{response.status_code} {response.text[:200]}"
    )
    # Absent, not forbidden — a 403 would confirm the name exists.
    assert "not found" in response.text.lower()
    # And readiness was never read on that caller's behalf (no timing oracle).
    harness.manager.get_container_status.assert_not_awaited()
    # Non-vacuity: the refusal is a REFUSAL, not a missing Deployment.
    assert harness.issued.kwargs["name"] == WORKSPACE


@pytest.mark.asyncio
async def test_foreign_tenant_404s_when_owner_is_only_in_the_annotation(app, client):
    """⭐ THE DEFECT, on the read path. RED before this change.

    The owner here is grammar-invalid as a label value (``/`` and ``:``), so
    provisioning stamped the annotation and omitted the label — exactly the state
    ``:1082``'s "otherwise omit rather than fail provisioning" produces. Reading
    the label alone recovered ``None``, the gate went vacuous, and the foreign
    caller was served.
    """
    import contextlib

    from coding_client.auth.tenant import get_current_user_with_tenant

    unlabelable_owner = "org:acme/platform-team"

    app.dependency_overrides[get_current_user_with_tenant] = lambda: _mock_user(tenant_id=FOREIGN)
    with contextlib.ExitStack() as stack:
        harness = _Harness(stack, tenant_label=None, tenant_annotation=unlabelable_owner)
        with patch("src.routers.containers.get_container_manager", return_value=harness.manager):
            response = await client.get(f"/api/containers/{WORKSPACE}/status")
    app.dependency_overrides.clear()

    assert response.status_code == 404, (
        f"annotation-owned workspace disclosed to a foreign tenant: "
        f"{response.status_code} {response.text[:200]}"
    )
    harness.manager.get_container_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_annotation_owner_still_reads_its_own_workspace(app, client):
    """POSITIVE CONTROL for the two tests above.

    Both of them are satisfiable by a gate that 404s everybody once the label is
    gone — which is precisely the outage the previous design feared. This asserts
    the annotation is genuinely RESOLVING an owner, not merely failing closed.
    """
    import contextlib

    from coding_client.auth.tenant import get_current_user_with_tenant

    unlabelable_owner = "org:acme/platform-team"

    app.dependency_overrides[get_current_user_with_tenant] = lambda: _mock_user(
        tenant_id=unlabelable_owner
    )
    with contextlib.ExitStack() as stack:
        harness = _Harness(stack, tenant_label=None, tenant_annotation=unlabelable_owner)
        with patch("src.routers.containers.get_container_manager", return_value=harness.manager):
            response = await client.get(f"/api/containers/{WORKSPACE}/status")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["container_status"] == "running"


@pytest.mark.asyncio
async def test_label_only_deployment_still_resolves_backward_compat(app, client):
    """BACKWARD COMPATIBILITY, and it is the whole live population.

    Measured in ``coding-jobs`` on 2026-09-17: 10 of 10 Deployments carry the
    ``tenant-id`` LABEL, 0 of 9 carry the annotation. Recovering from the
    annotation INSTEAD OF the label — rather than BEFORE it — would resolve every
    one of them to "owner unknown", and the fail-closed branch would then take the
    entire live estate offline. Both directions asserted, so a fallback that
    merely denies cannot pass this.
    """
    import contextlib

    from coding_client.auth.tenant import get_current_user_with_tenant

    # The owner reads its own label-only workspace.
    app.dependency_overrides[get_current_user_with_tenant] = lambda: _mock_user(tenant_id=OWNER)
    with contextlib.ExitStack() as stack:
        harness = _Harness(stack, tenant_label=OWNER, tenant_annotation=None)
        with patch("src.routers.containers.get_container_manager", return_value=harness.manager):
            owner_response = await client.get(f"/api/containers/{WORKSPACE}/status")
    app.dependency_overrides.clear()

    assert owner_response.status_code == 200, (
        "the label-only population (10 of 10 live Deployments) stopped resolving"
    )
    assert owner_response.json()["container_status"] == "running"

    # And it is still scoped: a foreign tenant is refused on the same shape.
    app.dependency_overrides[get_current_user_with_tenant] = lambda: _mock_user(tenant_id=FOREIGN)
    with contextlib.ExitStack() as stack:
        harness = _Harness(stack, tenant_label=OWNER, tenant_annotation=None)
        with patch("src.routers.containers.get_container_manager", return_value=harness.manager):
            foreign_response = await client.get(f"/api/containers/{WORKSPACE}/status")
    app.dependency_overrides.clear()

    assert foreign_response.status_code == 404


@pytest.mark.asyncio
async def test_annotation_wins_over_a_stale_label(app, client):
    """Precedence, asserted rather than assumed.

    The two stamps can disagree — a Deployment patched by an older build, or a
    label left behind by a prior owner. The annotation is the primary record
    because it is the one that is always writable, so it must WIN. A recovery
    that ORs in the other order would serve the stale label's tenant.
    """
    import contextlib

    from coding_client.auth.tenant import get_current_user_with_tenant

    app.dependency_overrides[get_current_user_with_tenant] = lambda: _mock_user(tenant_id=FOREIGN)
    with contextlib.ExitStack() as stack:
        # Label says FOREIGN, annotation says OWNER. The caller is FOREIGN.
        harness = _Harness(stack, tenant_label=FOREIGN, tenant_annotation=OWNER)
        with patch("src.routers.containers.get_container_manager", return_value=harness.manager):
            response = await client.get(f"/api/containers/{WORKSPACE}/status")
    app.dependency_overrides.clear()

    assert response.status_code == 404, (
        "a stale tenant-id LABEL overrode the authoritative annotation"
    )


@pytest.mark.asyncio
async def test_exec_takes_the_identical_path(app, client):
    """/exec reaches the same gate through the same reconstruct, so it leaked too.

    Asserted separately rather than assumed: #133's route audit reasoned about
    /exec from the ROUTER's scoping and concluded it was already safe, which was
    true one frame up and false one frame down.
    """
    import contextlib

    from coding_client.auth.tenant import require_tenant

    # ⚠️ /exec depends on ``require_tenant``, NOT ``get_current_user_with_tenant``
    # like the /status routes. Overriding the wrong dependency left the real one
    # in place, the caller resolved to the ambient default tenant, and the gate
    # was never reached — the request sailed through to the exec backend and
    # returned 500. Override the dependency the handler actually declares.
    app.dependency_overrides[require_tenant] = lambda: FOREIGN
    with contextlib.ExitStack() as stack:
        harness = _Harness(stack, tenant_label=OWNER)
        # Real gate, real reconstruct; the manager's exec is stubbed so a pass
        # cannot come from exec merely erroring out.
        harness.manager.execute_command = AsyncMock(
            return_value={"exit_code": 0, "stdout": "", "stderr": ""}
        )
        with patch(
            "src.routers.container_exec.get_container_manager", return_value=harness.manager
        ):
            response = await client.post(
                f"/api/containers/{WORKSPACE}/exec", json={"command": ["ls", "-la"]}
            )
    app.dependency_overrides.clear()

    # ⚠️ Non-vacuity guard. ``command`` is a LIST and is validated (schema, then
    # whitelist, then dangerous-patterns) BEFORE the tenant gate is reached. The
    # first draft of this test sent ``"echo hi"`` and got a 422 — it would have
    # "passed" any assertion looser than an exact 404 while never once reaching
    # the code under test.
    assert response.status_code not in (400, 422), (
        f"request was rejected before the tenant gate: {response.status_code} {response.text[:200]}"
    )
    assert response.status_code == 404, (
        f"/exec disclosed a foreign workspace on the registry-miss path: "
        f"{response.status_code} {response.text[:200]}"
    )
    harness.manager.execute_command.assert_not_awaited()
    # The lookup WAS issued against the cluster for this name: the 404 is a
    # refusal, not a missing Deployment.
    assert harness.issued.kwargs["name"] == WORKSPACE

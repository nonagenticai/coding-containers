"""The workspace sizing hints survive every hop, modelled on test_toolchain.py's
TestCreatePath: the create request reaches the provisioner with all four hints, the
provisioner rebuilds its request with them, both sizing annotations are read back on
a registry miss, a re-provision without hints nulls the stale memory annotation, and
a sized workspace's service launch carries the larger live-service cap."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from kubernetes.client.rest import ApiException

from src.core import toolchain as tc
from src.models.container_models import GitHubRepoRequest
from tests.unit.test_docker_service import _build_service, _make_info
from tests.unit.test_toolchain import _create, _k8s, _prov, kcp

HINTS = {
    "expected_repos": 3,
    "expected_services": 8,
    "service_memory_mi": 320,
    "expected_dependencies": ["redis"],
}


async def test_docker_service_forwards_all_four_hints_to_k8s():
    svc = _build_service(docker_available=True, registry_data={})
    svc.find_container_by_name = AsyncMock(return_value=None)
    svc.find_existing_container = AsyncMock(return_value=None)
    k8s = _k8s(_make_info("rs-1"))
    req = GitHubRepoRequest(repo_url="https://github.com/o/r", container_name="rs-1", **HINTS)
    with patch(
        "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
        return_value=k8s,
    ):
        await svc.create_container(req, tenant_id=None)
    kw = k8s.provision_container.await_args.kwargs
    assert {k: kw[k] for k in HINTS} == HINTS


async def test_provision_container_rebuilds_the_request_with_all_four_hints(monkeypatch):
    prov = _prov(monkeypatch)
    prov.provision_from_github = AsyncMock(return_value="ok")
    await prov.provision_container(container_name="ws", repo_url="https://github.com/o/r", **HINTS)
    req = prov.provision_from_github.await_args.args[0]
    assert {k: getattr(req, k) for k in HINTS} == HINTS


def test_reconstruct_reads_both_sizing_annotations_back(monkeypatch):
    prov = _prov(monkeypatch)
    dep = MagicMock()
    dep.metadata.labels = {"managed-by": "code-graph-helper", "tenant-id": "tenant-a"}
    dep.metadata.annotations = {
        kcp.REPO_URL_ANNOTATION: "https://github.com/o/r",
        tc.DISK_BUDGET_ANNOTATION: "14",
        tc.MEMORY_LIMIT_ANNOTATION: "6656",
    }
    dep.status.ready_replicas = 1
    dep.spec.replicas = 1
    prov.apps_v1.read_namespaced_deployment.return_value = dep
    info = prov.reconstruct_container_info("ws")
    assert (info.disk_budget_gi, info.memory_limit_mi) == (14, 6656)


def test_reprovision_without_hints_nulls_the_stale_memory_annotation(monkeypatch):
    prov = _prov(monkeypatch)
    prov.apps_v1.create_namespaced_deployment.side_effect = ApiException(status=409)
    incumbent = MagicMock()
    incumbent.metadata.deletion_timestamp = None
    incumbent.metadata.labels = {"tenant-id": "tenant-a"}
    incumbent.metadata.annotations = {
        kcp.TENANT_ANNOTATION: "tenant-a",
        tc.MEMORY_LIMIT_ANNOTATION: "6656",
    }
    prov.apps_v1.read_namespaced_deployment.return_value = incumbent
    _create(prov)
    patched = prov.apps_v1.patch_namespaced_deployment.call_args.kwargs["body"]
    assert patched.metadata.annotations[tc.MEMORY_LIMIT_ANNOTATION] is None


def test_a_sized_workspace_launches_with_the_larger_service_cap():
    from src.routers import workspace_services as ws_router

    request = ws_router.StartServiceRequest(command=["python3", "-V"], port=18080)

    def cap(memory_limit_mi):
        ws = SimpleNamespace(codebase_path="/workspace", memory_limit_mi=memory_limit_mi)
        return ws_router._launch_spec(ws, request, "svc-0123456789abcdef", ready_timeout=5)[
            "max_services"
        ]

    assert cap(6656) == tc.MAX_EXPECTED_SERVICES == 56
    assert cap(None) == tc.MAX_SERVICES_PER_WORKSPACE == 4

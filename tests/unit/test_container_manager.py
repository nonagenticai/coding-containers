"""Tests for core/container_manager.py."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients.github_client import is_platform_owner, repo_owner_from_url

# ---------------------------------------------------------------------------
# Helper to build a fake ContainerInfo-like mock
# ---------------------------------------------------------------------------


def _make_container(
    name: str = "c1",
    tenant_id: str | None = None,
    code_graph_id: str = "",
    repo_url: str = "https://github.com/org/repo",
    status: str = "running",
    created_at: str = "",
    updated_at: str = "",
) -> MagicMock:
    c = MagicMock()
    c.container_name = name
    c.tenant_id = tenant_id
    c.code_graph_id = code_graph_id
    c.repo_url = repo_url
    c.status = status
    c.created_at = created_at
    c.updated_at = updated_at
    return c


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def container_manager():
    with (
        patch("src.core.container_manager.get_core_manager") as mock_core,
        patch("src.core.container_manager.get_github_container_provisioner") as mock_github,
        patch("src.core.container_manager.get_docker_service") as mock_ds,
    ):
        mock_core.return_value = MagicMock()
        mock_github.return_value = MagicMock()
        mock_ds.return_value = MagicMock()

        from src.core.container_manager import ContainerManager

        mgr = ContainerManager()
        # By default no live-Deployment fallback (registry miss => not found),
        # so existing tests keep their semantics. Fallback tests opt in by
        # setting reconstruct_container_info.return_value explicitly.
        mgr._github.k8s_provisioner.reconstruct_container_info.return_value = None
        # No live cluster Deployments by default, so list_containers() reflects
        # only the JSON registry and existing tests keep their semantics. The
        # reconcile tests opt in by returning names here.
        mgr._github.k8s_provisioner.list_managed_container_names.return_value = []
        # Stash mocks for convenient access in tests
        mgr._mock_core_factory = mock_core
        mgr._mock_github_factory = mock_github
        mgr._mock_ds_factory = mock_ds
        yield mgr


# ===========================================================================
# Original tests (kept as-is)
# ===========================================================================


class TestContainerManagerList:
    @patch("src.core.container_manager.get_docker_service")
    def test_list_empty(self, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        result = container_manager.list_containers()
        assert result == []


class TestContainerManagerListK8sReconcile:
    """list_containers() unions in live-cluster Deployments the ephemeral JSON
    registry has lost (pod redeploy/reschedule wipes the node-local registry
    while the Deployments persist)."""

    @patch("src.core.container_manager.get_docker_service")
    def test_cluster_only_name_is_reconciled(self, mock_ds, container_manager):
        # Registry knows only "reg-box"; the cluster also runs "live-box" that
        # the registry has lost. list_containers() must return BOTH.
        registry_box = _make_container(name="reg-box", tenant_id=None)
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"reg-box": registry_box}
        mock_ds.return_value = docker_svc

        rebuilt = _make_container(name="live-box", tenant_id=None)
        container_manager._github.use_k8s = True
        container_manager._github.k8s_provisioner.list_managed_container_names.return_value = [
            "reg-box",  # already in registry -> must NOT be duplicated/rebuilt
            "live-box",  # cluster-only -> must be reconstructed and included
        ]
        container_manager._github.k8s_provisioner.reconstruct_container_info.return_value = rebuilt

        result = container_manager.list_containers()

        names = {c.container_name for c in result}
        assert names == {"reg-box", "live-box"}
        # Registry entry wins on conflict: reconstruct only called for the
        # cluster-only name, never for the name already in the registry.
        container_manager._github.k8s_provisioner.reconstruct_container_info.assert_called_once_with(
            "live-box"
        )

    @patch("src.core.container_manager.get_docker_service")
    def test_cluster_only_name_with_unknown_owner_is_tenant_filtered_out(
        self, mock_ds, container_manager
    ):
        """⚠️ INVERTED. This asserted ``== ["live-box"]`` — that a reconstructed
        cluster-only entry with ``tenant_id=None`` "survives tenant filtering for
        any tenant", on the stated ground of "legacy/visible parity".

        A cluster-only entry is not legacy registry data; it is a live Deployment
        whose owner we failed to recover. Since the ``tenant-id`` label is written
        only for tenants matching ``_K8S_LABEL_VALUE_RE``, that failure is equally
        consistent with an OWNED workspace whose owner cannot be labelled — so
        listing it handed every tenant the name and readiness of a workspace that
        may not be theirs, which is the same disclosure #133 closed on /status.

        Registry rows keep the legacy fail-open (see
        ``test_registry_row_with_no_tenant_is_still_listed``); only the rebuilt
        ones are dropped.
        """
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        rebuilt = _make_container(name="live-box", tenant_id=None)
        container_manager._github.use_k8s = True
        container_manager._github.k8s_provisioner.list_managed_container_names.return_value = [
            "live-box"
        ]
        container_manager._github.k8s_provisioner.reconstruct_container_info.return_value = rebuilt

        result = container_manager.list_containers(tenant_id="tenant-abc")

        assert [c.container_name for c in result] == []

    @patch("src.core.container_manager.get_docker_service")
    def test_registry_row_with_no_tenant_is_still_listed(self, mock_ds, container_manager):
        """⛔ CONTROL proving the drop above is scoped to REBUILT entries.

        A registry row with no tenant is legacy data this service wrote and keeps
        its documented visibility. If this goes red, the list filter stopped
        distinguishing the two populations and became a fleet-wide change.
        """
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {
            "reg-box": _make_container(name="reg-box", tenant_id=None)
        }
        mock_ds.return_value = docker_svc

        container_manager._github.use_k8s = True
        container_manager._github.k8s_provisioner.list_managed_container_names.return_value = []

        result = container_manager.list_containers(tenant_id="tenant-abc")

        assert [c.container_name for c in result] == ["reg-box"]

    @patch("src.core.container_manager.get_docker_service")
    def test_untenanted_list_still_sees_unknown_owner_entries(self, mock_ds, container_manager):
        """POSITIVE CONTROL. The drop is part of the TENANT FILTER, not the union.

        An admin list (``tenant_id=None``, the ``is_admin()`` widening) must still
        see every live Deployment — otherwise an unrecoverable workspace becomes
        invisible to the only role that could investigate it, which is the
        "traded a leak for a blind spot" failure mode.
        """
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        rebuilt = _make_container(name="live-box", tenant_id=None)
        container_manager._github.use_k8s = True
        container_manager._github.k8s_provisioner.list_managed_container_names.return_value = [
            "live-box"
        ]
        container_manager._github.k8s_provisioner.reconstruct_container_info.return_value = rebuilt

        result = container_manager.list_containers()

        assert [c.container_name for c in result] == ["live-box"]

    @patch("src.core.container_manager.get_docker_service")
    def test_k8s_error_degrades_to_registry_only(self, mock_ds, container_manager):
        from kubernetes.client.rest import ApiException

        registry_box = _make_container(name="reg-box", tenant_id=None)
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"reg-box": registry_box}
        mock_ds.return_value = docker_svc

        container_manager._github.use_k8s = True
        container_manager._github.k8s_provisioner.list_managed_container_names.side_effect = (
            ApiException(status=500, reason="cluster unreachable")
        )

        # Must not raise; degrades to the registry-only list.
        result = container_manager.list_containers()

        assert [c.container_name for c in result] == ["reg-box"]

    @patch("src.core.container_manager.get_docker_service")
    def test_no_reconcile_when_k8s_disabled(self, mock_ds, container_manager):
        registry_box = _make_container(name="reg-box", tenant_id=None)
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"reg-box": registry_box}
        mock_ds.return_value = docker_svc

        container_manager._github.use_k8s = False

        result = container_manager.list_containers()

        assert [c.container_name for c in result] == ["reg-box"]
        container_manager._github.k8s_provisioner.list_managed_container_names.assert_not_called()


class TestContainerManagerGet:
    @patch("src.core.container_manager.get_docker_service")
    def test_get_found(self, mock_ds, container_manager):
        container_info = MagicMock()
        container_info.container_name = "test"
        container_info.tenant_id = None

        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"test": container_info}
        mock_ds.return_value = docker_svc

        result = container_manager.get_container("test")
        assert result is container_info
        assert result.container_name == "test"

    @patch("src.core.container_manager.get_docker_service")
    def test_get_not_found(self, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        with pytest.raises(ValueError, match="not found"):
            container_manager.get_container("nonexistent")


# ===========================================================================
# has_container
# ===========================================================================


class TestHasContainer:
    @patch("src.core.container_manager.get_docker_service")
    def test_in_registry(self, mock_ds, container_manager):
        c = _make_container("box")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"box": c}
        mock_ds.return_value = docker_svc

        assert container_manager.has_container("box") is True

    @patch("src.core.container_manager.get_docker_service")
    def test_not_in_registry(self, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        assert container_manager.has_container("nope") is False

    @patch("src.core.container_manager.get_docker_service")
    def test_tenant_match(self, mock_ds, container_manager):
        c = _make_container("box", tenant_id="t1")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"box": c}
        mock_ds.return_value = docker_svc

        assert container_manager.has_container("box", tenant_id="t1") is True

    @patch("src.core.container_manager.get_docker_service")
    def test_tenant_mismatch(self, mock_ds, container_manager):
        c = _make_container("box", tenant_id="t1")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"box": c}
        mock_ds.return_value = docker_svc

        assert container_manager.has_container("box", tenant_id="t2") is False

    @patch("src.core.container_manager.get_docker_service")
    def test_legacy_null_tenant_allowed(self, mock_ds, container_manager):
        c = _make_container("box", tenant_id=None)
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"box": c}
        mock_ds.return_value = docker_svc

        # Legacy containers (tenant_id=None) are accessible to any tenant
        assert container_manager.has_container("box", tenant_id="t1") is True


# ===========================================================================
# get_container (extended)
# ===========================================================================


class TestGetContainerExtended:
    @patch("src.core.container_manager.get_docker_service")
    def test_wrong_tenant_returns_none(self, mock_ds, container_manager):
        c = _make_container("box", tenant_id="owner-a")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"box": c}
        mock_ds.return_value = docker_svc

        result = container_manager.get_container("box", tenant_id="owner-b")
        assert result is None

    @patch("src.core.container_manager.get_docker_service")
    def test_legacy_tenant_allowed(self, mock_ds, container_manager):
        c = _make_container("box", tenant_id=None)
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"box": c}
        mock_ds.return_value = docker_svc

        result = container_manager.get_container("box", tenant_id="any")
        assert result is c

    @patch("src.core.container_manager.get_docker_service")
    def test_no_tenant_check_returns_container(self, mock_ds, container_manager):
        c = _make_container("box", tenant_id="owner-a")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"box": c}
        mock_ds.return_value = docker_svc

        result = container_manager.get_container("box")
        assert result is c


# ===========================================================================
# get_container_for_graph
# ===========================================================================


class TestGetContainerForGraph:
    @patch("src.core.container_manager.get_docker_service")
    def test_match_by_code_graph_id(self, mock_ds, container_manager):
        c = _make_container("box", code_graph_id="graph-1")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"box": c}
        mock_ds.return_value = docker_svc

        result = container_manager.get_container_for_graph("graph-1")
        assert result is c

    @patch("src.core.container_manager.get_docker_service")
    def test_no_match(self, mock_ds, container_manager):
        c = _make_container("box", code_graph_id="graph-1")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"box": c}
        mock_ds.return_value = docker_svc

        result = container_manager.get_container_for_graph("unknown")
        assert result is None

    @patch("src.core.container_manager.get_docker_service")
    def test_exception_handling(self, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.side_effect = ValueError("boom")
        mock_ds.return_value = docker_svc

        result = container_manager.get_container_for_graph("any")
        assert result is None


# ===========================================================================
# list_containers (extended)
# ===========================================================================


class TestListContainersExtended:
    @patch("src.core.container_manager.get_docker_service")
    def test_all_no_tenant_filter(self, mock_ds, container_manager):
        c1 = _make_container("a", tenant_id="t1")
        c2 = _make_container("b", tenant_id="t2")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"a": c1, "b": c2}
        mock_ds.return_value = docker_svc

        result = container_manager.list_containers()
        assert len(result) == 2

    @patch("src.core.container_manager.get_docker_service")
    def test_tenant_filter(self, mock_ds, container_manager):
        c1 = _make_container("a", tenant_id="t1")
        c2 = _make_container("b", tenant_id="t2")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"a": c1, "b": c2}
        mock_ds.return_value = docker_svc

        result = container_manager.list_containers(tenant_id="t1")
        assert len(result) == 1
        assert result[0].container_name == "a"

    @patch("src.core.container_manager.get_docker_service")
    def test_legacy_containers_included(self, mock_ds, container_manager):
        c_legacy = _make_container("old", tenant_id=None)
        c_owned = _make_container("mine", tenant_id="t1")
        c_other = _make_container("theirs", tenant_id="t2")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {
            "old": c_legacy,
            "mine": c_owned,
            "theirs": c_other,
        }
        mock_ds.return_value = docker_svc

        result = container_manager.list_containers(tenant_id="t1")
        names = [c.container_name for c in result]
        assert "old" in names
        assert "mine" in names
        assert "theirs" not in names


# ===========================================================================
# execute_command
# ===========================================================================


class TestExecuteCommand:
    @patch("src.core.container_manager.get_docker_service")
    async def test_tenant_validation_failure(self, mock_ds, container_manager):
        """Tenant mismatch -> 'not found' error (BOLA protection)."""
        c = _make_container("box", tenant_id="owner-a")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"box": c}
        mock_ds.return_value = docker_svc

        result = await container_manager.execute_command("box", "ls", tenant_id="hacker")
        assert result["success"] is False
        assert "not found" in result["stderr"]
        # Should NOT reveal that the container exists -- security check
        assert "owner-a" not in result["stderr"]
        assert "permission" not in result["stderr"].lower()

    @patch("src.core.container_manager.get_docker_service")
    async def test_tenant_validation_success_allows_exec(self, mock_ds, container_manager):
        """Matching tenant allows command execution."""
        c = _make_container("box", tenant_id="owner-a")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"box": c}
        mock_ds.return_value = docker_svc

        container_manager._core.is_docker_available.return_value = True
        container_manager._core.uses_agent_host = False
        mock_container = MagicMock()
        exec_result = MagicMock()
        exec_result.exit_code = 0
        exec_result.output = b"success\n"
        mock_container.exec_run.return_value = exec_result
        container_manager._core.client.containers.get.return_value = mock_container

        result = await container_manager.execute_command("box", "echo ok", tenant_id="owner-a")
        assert result["success"] is True
        assert result["stdout"] == "success\n"

    @patch("src.core.container_manager.get_docker_service")
    async def test_docker_unavailable(self, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        container_manager._core.is_docker_available.return_value = False
        # Explicit: a MagicMock attribute is truthy, and leaving this unset
        # resolves the workspace to the agent host instead of the path under
        # test.
        container_manager._core.uses_agent_host = False
        # Stub the provisioner unavailable so no backend can serve the request.
        mock_k8s = MagicMock()
        mock_k8s.is_available.return_value = False
        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            return_value=mock_k8s,
        ):
            result = await container_manager.execute_command("box", "ls")
        assert result["success"] is False
        assert "not available" in result["stderr"].lower()

    @patch("src.core.container_manager.get_docker_service")
    async def test_agent_host_delegation(self, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        container_manager._core.is_docker_available.return_value = True
        container_manager._core.uses_agent_host = True
        container_manager._core.exec_in_container_via_agent_host.return_value = {
            "exit_code": 0,
            "stdout": "ok",
            "stderr": "",
        }

        result = await container_manager.execute_command("box", "echo hi")
        assert result["success"] is True
        assert result["stdout"] == "ok"

    @patch("src.core.container_manager.get_docker_service")
    async def test_direct_docker_exec_success(self, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        container_manager._core.is_docker_available.return_value = True
        container_manager._core.uses_agent_host = False

        mock_container = MagicMock()
        exec_result = MagicMock()
        exec_result.exit_code = 0
        exec_result.output = b"hello\n"
        mock_container.exec_run.return_value = exec_result
        container_manager._core.client.containers.get.return_value = mock_container

        result = await container_manager.execute_command("box", "echo hello")
        assert result["success"] is True
        assert result["stdout"] == "hello\n"

    @patch("src.core.container_manager.get_docker_service")
    async def test_direct_docker_exec_failure(self, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        container_manager._core.is_docker_available.return_value = True
        container_manager._core.uses_agent_host = False

        mock_container = MagicMock()
        exec_result = MagicMock()
        exec_result.exit_code = 1
        exec_result.output = b"error msg"
        mock_container.exec_run.return_value = exec_result
        container_manager._core.client.containers.get.return_value = mock_container

        result = await container_manager.execute_command("box", "bad-cmd")
        assert result["success"] is False
        assert result["exit_code"] == 1

    @patch("src.core.container_manager.get_docker_service")
    async def test_client_none_error(self, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        container_manager._core.is_docker_available.return_value = True
        container_manager._core.uses_agent_host = False
        container_manager._core.client = None

        result = await container_manager.execute_command("box", "ls")
        assert result["success"] is False
        assert "not available" in result["stderr"]


# ===========================================================================
# remove_container
# ===========================================================================


class TestRemoveContainer:
    @patch("src.core.container_manager.get_docker_service")
    async def test_tenant_mismatch_returns_not_found(self, mock_ds, container_manager):
        """BOLA: wrong tenant sees 'not found' instead of 'permission denied'."""
        c = _make_container("box", tenant_id="owner-a")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"box": c}
        mock_ds.return_value = docker_svc

        result = await container_manager.remove_container("box", tenant_id="hacker")
        assert result["status"] == "error"
        assert "not found" in result["message"]
        # Security: error message must not reveal container exists or leak tenant info
        assert "owner-a" not in result["message"]
        assert "permission" not in result["message"].lower()

    @patch("src.core.container_manager.get_docker_service")
    async def test_k8s_mode_path(self, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        container_manager._github.use_k8s = True
        k8s_prov = MagicMock()
        k8s_prov.remove_container = AsyncMock()
        container_manager._github.k8s_provisioner = k8s_prov
        container_manager._core.remove_container.return_value = True

        result = await container_manager.remove_container("box")
        assert result["status"] == "success"

    @patch("src.core.container_manager.get_docker_service")
    async def test_registry_cleanup(self, mock_ds, container_manager):
        c = _make_container("box", code_graph_id="g1")
        docker_svc = MagicMock()
        registry = {"box": c}
        docker_svc.registry.load.return_value = registry
        mock_ds.return_value = docker_svc

        container_manager._github.use_k8s = False
        container_manager._github.k8s_provisioner = None
        container_manager._core.remove_container.return_value = True
        container_manager._graph_deletion_callback = None

        result = await container_manager.remove_container("box")
        assert result["status"] == "success"
        # Atomic single-field HDEL for exactly this container. A whole-registry
        # save() from a stale snapshot also deletes siblings — that is the live
        # defect pinned by tests/unit/test_registry_sibling_replica_safety.py.
        docker_svc.registry.remove.assert_called_once_with("box")
        docker_svc.registry.save.assert_not_called()

    @patch("src.core.container_manager.get_docker_service")
    async def test_graph_deletion_with_callback(self, mock_ds, container_manager):
        """When a graph_deletion_callback is set, it gets invoked."""
        c = _make_container("box", code_graph_id="g1")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"box": c}
        mock_ds.return_value = docker_svc

        container_manager._github.use_k8s = False
        container_manager._github.k8s_provisioner = None
        container_manager._core.remove_container.return_value = True

        mock_cb = AsyncMock(
            return_value={
                "success": True,
                "deleted_nodes": 5,
                "deleted_edges": 10,
                "paths_deleted": ["/tmp"],
            }
        )
        container_manager._graph_deletion_callback = mock_cb

        result = await container_manager.remove_container("box")
        assert result["status"] == "success"
        assert "graph_deletion" in result
        assert result["graph_deletion"]["status"] == "deleted"


class TestRemoveContainerOrdering:
    """The ordering + honesty invariants that were broken in production.

    Neither was covered before, which is how a 35-second DELETE that logged two
    false failures, fabricated a graph-deletion success, and orphaned the graph
    rows shipped and stayed shipped.
    """

    @patch("src.core.container_manager.get_docker_service")
    async def test_graph_is_deleted_before_the_registry_entry_is_dropped(
        self, mock_ds, container_manager
    ):
        """coding-analysis validates the container EXISTS for the tenant first.

        Delete the registry entry first and that validation 404s, orphaning the
        graph rows permanently — the delete endpoint is the only thing that can
        remove them and it needs the container to exist.
        """
        c = _make_container("c1", code_graph_id="g1", tenant_id="t1")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"c1": c}
        mock_ds.return_value = docker_svc

        order: list[str] = []
        docker_svc.registry.remove.side_effect = lambda *_a, **_k: order.append("registry_save")

        async def _cb(graph_id, force=False):
            order.append("graph_delete")
            return {"success": True, "deleted_nodes": 3, "deleted_edges": 2}

        container_manager._graph_deletion_callback = _cb
        container_manager._github.use_k8s = False
        container_manager._github.k8s_provisioner = None
        container_manager._core.remove_container.return_value = True

        await container_manager.remove_container("c1", tenant_id="t1")

        assert order == ["graph_delete", "registry_save"], order

    @patch("src.core.container_manager.get_docker_service")
    async def test_the_callers_tenant_reaches_the_graph_deletion(self, mock_ds, container_manager):
        """Without a tenant coding-analysis cannot delete anything."""
        # Same tenant on both sides — a mismatch is a BOLA rejection, tested above.
        c = _make_container("c1", code_graph_id="g1", tenant_id="t-from-caller")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"c1": c}
        mock_ds.return_value = docker_svc

        seen: dict = {}

        container_manager._github.use_k8s = False
        container_manager._github.k8s_provisioner = None
        container_manager._core.remove_container.return_value = True
        container_manager._graph_deletion_callback = None

        class _Ops:
            async def delete_container_graph(self, graph_id, tenant_id=None):
                seen["graph_id"] = graph_id
                seen["tenant_id"] = tenant_id
                return {"success": True, "deleted_nodes": 1, "deleted_edges": 1}

        with patch("src.core.graph_operations.get_graph_operations", return_value=_Ops()):
            await container_manager.remove_container("c1", tenant_id="t-from-caller")

        assert seen["graph_id"] == "g1"
        assert seen["tenant_id"] == "t-from-caller"

    @patch("src.core.container_manager.get_docker_service")
    async def test_a_failed_graph_deletion_is_never_reported_as_deleted(
        self, mock_ds, container_manager
    ):
        """A 5s timeout used to return {"success": True, ... "pending"}."""
        c = _make_container("c1", code_graph_id="g1")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"c1": c}
        mock_ds.return_value = docker_svc

        container_manager._github.use_k8s = False
        container_manager._github.k8s_provisioner = None
        container_manager._core.remove_container.return_value = True
        container_manager._graph_deletion_callback = AsyncMock(
            return_value={"success": False, "message": "coding-analysis unreachable"}
        )

        result = await container_manager.remove_container("c1")

        assert result["graph_deletion"]["status"] == "failed"
        assert result["graph_deletion"]["status"] != "deleted"

    @patch("src.core.container_manager.get_docker_service")
    async def test_k8s_teardown_is_awaited_not_scheduled_on_a_blocked_loop(
        self, mock_ds, container_manager
    ):
        """The removal must complete WITHIN the call, on the running loop.

        The old sync implementation used
        ``asyncio.run_coroutine_threadsafe(...).result(timeout=30)`` from the
        loop thread, so the coroutine could not run until the caller returned:
        a guaranteed 30s timeout on every delete.
        """
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        done: list[str] = []

        async def _remove(name):
            await asyncio.sleep(0)
            done.append(name)

        container_manager._github.use_k8s = True
        k8s_prov = MagicMock()
        k8s_prov.remove_container = _remove
        container_manager._github.k8s_provisioner = k8s_prov
        container_manager._core.remove_container.return_value = True

        result = await container_manager.remove_container("c1")

        assert done == ["c1"]
        assert result["status"] == "success"


# ===========================================================================
# get_all_graphs_summary_info
# ===========================================================================


class TestGetAllGraphsSummaryInfo:
    @patch("src.core.container_manager.get_docker_service")
    @patch("src.core.container_manager.get_settings")
    async def test_containers_with_code_graph_id(self, mock_settings, mock_ds, container_manager):
        c = _make_container("box", code_graph_id="g1")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"box": c}
        mock_ds.return_value = docker_svc

        settings = MagicMock()
        settings.get_database_manager.side_effect = NotImplementedError
        mock_settings.return_value = settings

        result = await container_manager.get_all_graphs_summary_info()
        assert len(result) == 1
        assert result[0]["id"] == "g1"
        assert result[0]["is_containerized"] is True

    @patch("src.core.container_manager.get_docker_service")
    @patch("src.core.container_manager.get_settings")
    async def test_standalone_db_scan(self, mock_settings, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake .db file
            (Path(tmpdir) / "standalone.db").touch()

            db_mgr = MagicMock()
            db_mgr.base_paths = [tmpdir]
            settings = MagicMock()
            settings.get_database_manager.return_value = db_mgr
            mock_settings.return_value = settings

            result = await container_manager.get_all_graphs_summary_info()
            assert len(result) == 1
            assert result[0]["id"] == "standalone"
            assert result[0]["is_containerized"] is False

    @patch("src.core.container_manager.get_docker_service")
    @patch("src.core.container_manager.get_settings")
    async def test_database_manager_not_available(self, mock_settings, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        settings = MagicMock()
        settings.get_database_manager.side_effect = NotImplementedError
        mock_settings.return_value = settings

        result = await container_manager.get_all_graphs_summary_info()
        assert result == []


# ===========================================================================
# get_graph_details_by_id
# ===========================================================================


class TestGetGraphDetailsById:
    @patch("src.core.container_manager.get_docker_service")
    @patch("src.core.container_manager.get_settings")
    async def test_found_in_container(self, mock_settings, mock_ds, container_manager):
        c = _make_container("box", code_graph_id="g1")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"box": c}
        mock_ds.return_value = docker_svc

        settings = MagicMock()
        mock_settings.return_value = settings

        result = await container_manager.get_graph_details_by_id("g1")
        assert result is not None
        assert result["id"] == "g1"
        assert result["is_containerized"] is True
        assert result["container_name"] == "box"
        assert result["repository_url"] == "https://github.com/org/repo"

    @patch("src.core.container_manager.get_docker_service")
    @patch("src.core.container_manager.get_settings")
    async def test_found_as_standalone_db(self, mock_settings, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        settings = MagicMock()
        db_mgr = MagicMock()
        db_mgr.find_existing_database.return_value = "/tmp/g2.db"
        settings.get_database_manager.return_value = db_mgr
        mock_settings.return_value = settings

        result = await container_manager.get_graph_details_by_id("g2")
        assert result is not None
        assert result["id"] == "g2"
        assert result["is_containerized"] is False
        assert result["container_name"] is None
        assert result["status"] == "available"

    @patch("src.core.container_manager.get_docker_service")
    @patch("src.core.container_manager.get_settings")
    async def test_not_found(self, mock_settings, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        settings = MagicMock()
        db_mgr = MagicMock()
        db_mgr.find_existing_database.return_value = None
        settings.get_database_manager.return_value = db_mgr
        mock_settings.return_value = settings

        result = await container_manager.get_graph_details_by_id("nope")
        assert result is None


# ===========================================================================
# _scan_database_path
# ===========================================================================


class TestScanDatabasePath:
    def test_finds_db_files(self, container_manager):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "alpha.db").touch()
            (Path(tmpdir) / "beta.db").touch()
            summaries: list[dict] = []
            container_manager._scan_database_path(tmpdir, summaries)
            ids = {s["id"] for s in summaries}
            assert ids == {"alpha", "beta"}

    def test_skips_duplicates(self, container_manager):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "alpha.db").touch()
            summaries: list[dict] = [{"id": "alpha"}]
            container_manager._scan_database_path(tmpdir, summaries)
            assert len(summaries) == 1

    def test_handles_missing_dir(self, container_manager):
        summaries: list[dict] = []
        container_manager._scan_database_path("/nonexistent/dir/xyz", summaries)
        assert summaries == []


# ===========================================================================
# client / base_port properties
# ===========================================================================


class TestManagerProperties:
    def test_client_property(self, container_manager):
        mock_client = MagicMock()
        container_manager._core.client = mock_client
        assert container_manager.client is mock_client

    def test_base_port_property(self, container_manager):
        container_manager._core.base_port = 9999
        assert container_manager.base_port == 9999


# ===========================================================================
# execute_command - additional coverage
# ===========================================================================


class TestExecuteCommandPaths:
    @patch("src.core.container_manager.get_docker_service")
    async def test_execute_command_tenant_not_found(self, mock_ds, container_manager):
        """execute_command returns error when tenant validation fails."""
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        result = await container_manager.execute_command("c1", "ls", tenant_id="tenant1")
        assert result["success"] is False
        assert result["exit_code"] == 1

    async def test_execute_command_docker_unavailable(self, container_manager):
        """execute_command returns error when docker is unavailable.

        The K8s provisioner MUST be stubbed unavailable, exactly as the sibling
        ``test_docker_unavailable`` above already does. When docker is
        unavailable ``execute_command`` falls back to
        ``get_k8s_container_provisioner()`` and calls ``is_available()``, which
        opens a real connection to the Kubernetes API. Unstubbed, this "unit"
        test performs live network I/O: it passes in CI only because the runner
        has no cluster to reach and fails fast, and it HANGS INDEFINITELY on a
        machine that can route to one. Stubbing makes the test assert the
        "neither docker nor k8s" path it is named for, independently of whatever
        cluster the developer happens to be able to see.
        """
        container_manager._core.is_docker_available.return_value = False
        # Must be explicit: on a MagicMock core this attribute is truthy, which
        # would resolve the workspace to the agent host and quietly test a
        # different path than the one this test is named for.
        container_manager._core.uses_agent_host = False
        mock_k8s = MagicMock()
        mock_k8s.is_available.return_value = False
        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            return_value=mock_k8s,
        ):
            result = await container_manager.execute_command("c1", "ls")
        assert result["success"] is False
        # The error now names the backend the workspace ACTUALLY resolved to,
        # rather than the old "Neither Docker nor Kubernetes is available".
        # With Docker down and no endpoint to go on, an unknown workspace
        # resolves to Kubernetes — which is what in-cluster execution has always
        # done, just decided explicitly instead of by fallthrough.
        assert "Kubernetes" in result["stderr"]
        assert "not available" in result["stderr"]
        # Non-vacuity: the provisioner really was consulted and really said no.
        mock_k8s.is_available.assert_called_once()

    async def test_execute_command_agent_host(self, container_manager):
        """execute_command delegates to agent host when configured."""
        container_manager._core.is_docker_available.return_value = True
        container_manager._core.uses_agent_host = True
        container_manager._core.exec_in_container_via_agent_host.return_value = {
            "exit_code": 0,
            "stdout": "hello",
            "stderr": "",
        }
        result = await container_manager.execute_command("c1", "echo hello")
        assert result["success"] is True
        assert result["stdout"] == "hello"

    async def test_execute_command_no_client(self, container_manager):
        """execute_command returns error when docker client is None."""
        container_manager._core.is_docker_available.return_value = True
        container_manager._core.uses_agent_host = False
        container_manager._core.client = None
        result = await container_manager.execute_command("c1", "ls")
        assert result["success"] is False
        assert "not available" in result["stderr"]

    async def test_execute_command_docker_exec_success(self, container_manager):
        """execute_command runs via docker exec."""
        container_manager._core.is_docker_available.return_value = True
        container_manager._core.uses_agent_host = False
        mock_container = MagicMock()
        mock_exec_result = MagicMock()
        mock_exec_result.exit_code = 0
        mock_exec_result.output = b"output data"
        mock_container.exec_run.return_value = mock_exec_result
        container_manager._core.client = MagicMock()
        container_manager._core.client.containers.get.return_value = mock_container

        result = await container_manager.execute_command("c1", "ls")
        assert result["success"] is True
        assert result["stdout"] == "output data"

    async def test_execute_command_docker_exec_failure(self, container_manager):
        """execute_command returns error on non-zero exit."""
        container_manager._core.is_docker_available.return_value = True
        container_manager._core.uses_agent_host = False
        mock_container = MagicMock()
        mock_exec_result = MagicMock()
        mock_exec_result.exit_code = 1
        mock_exec_result.output = b"error data"
        mock_container.exec_run.return_value = mock_exec_result
        container_manager._core.client = MagicMock()
        container_manager._core.client.containers.get.return_value = mock_container

        result = await container_manager.execute_command("c1", "bad-cmd")
        assert result["success"] is False
        assert result["exit_code"] == 1

    async def test_execute_command_docker_exception(self, container_manager):
        """execute_command catches errors from docker."""
        container_manager._core.is_docker_available.return_value = True
        container_manager._core.uses_agent_host = False
        container_manager._core.client = MagicMock()
        container_manager._core.client.containers.get.side_effect = RuntimeError(
            "no such container"
        )

        result = await container_manager.execute_command("c1", "ls")
        assert result["success"] is False
        assert "no such container" in result["stderr"]


# ===========================================================================
# remove_container - additional coverage
# ===========================================================================


class TestRemoveContainerExtended:
    @patch("src.core.container_manager.get_docker_service")
    async def test_remove_tenant_mismatch(self, mock_ds, container_manager):
        """remove_container returns error for tenant mismatch."""
        c = _make_container(name="c1", tenant_id="tenant-a")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"c1": c}
        mock_ds.return_value = docker_svc

        result = await container_manager.remove_container("c1", tenant_id="tenant-b")
        assert result["status"] == "error"
        assert "not found" in result["message"]

    @patch("src.core.container_manager.get_docker_service")
    async def test_remove_success_no_graph(self, mock_ds, container_manager):
        """remove_container succeeds without graph id."""
        c = _make_container(name="c1")
        c.code_graph_id = ""
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"c1": c}
        mock_ds.return_value = docker_svc
        container_manager._core.remove_container.return_value = True
        container_manager._github.use_k8s = False

        result = await container_manager.remove_container("c1")
        assert result["status"] == "success"

    @patch("src.core.container_manager.get_docker_service")
    async def test_remove_registry_cleanup_error(self, mock_ds, container_manager):
        """remove_container handles registry errors gracefully."""
        docker_svc = MagicMock()
        docker_svc.registry.load.side_effect = OSError("disk error")
        mock_ds.return_value = docker_svc
        container_manager._core.remove_container.return_value = True
        container_manager._github.use_k8s = False

        result = await container_manager.remove_container("c1")
        assert result["status"] == "success"


# ===========================================================================
# get_container_for_graph
# ===========================================================================


class TestGetContainerForGraphExtended:
    @patch("src.core.container_manager.get_docker_service")
    def test_match_by_graph_id(self, mock_ds, container_manager):
        c = _make_container(name="c1", code_graph_id="graph-1")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"c1": c}
        mock_ds.return_value = docker_svc

        result = container_manager.get_container_for_graph("graph-1")
        assert result is c

    @patch("src.core.container_manager.get_docker_service")
    def test_match_by_container_name_suffix(self, mock_ds, container_manager):
        c = _make_container(name="codegraph_myproject", code_graph_id="other")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"codegraph_myproject": c}
        mock_ds.return_value = docker_svc

        result = container_manager.get_container_for_graph("myproject")
        assert result is c

    @patch("src.core.container_manager.get_docker_service")
    def test_no_match(self, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        result = container_manager.get_container_for_graph("nope")
        assert result is None

    @patch("src.core.container_manager.get_docker_service")
    def test_error_handling(self, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.side_effect = ValueError("bad data")
        mock_ds.return_value = docker_svc

        result = container_manager.get_container_for_graph("nope")
        assert result is None


# ===========================================================================
# provision_from_github
# ===========================================================================


class TestProvisionFromGithub:
    @pytest.mark.asyncio
    async def test_delegates_to_github_provisioner(self, container_manager):
        mock_result = MagicMock()
        container_manager._github.provision_from_github = AsyncMock(return_value=mock_result)
        request = MagicMock()
        result = await container_manager.provision_from_github(request)
        assert result is mock_result
        container_manager._github.provision_from_github.assert_awaited_once_with(request)


# ===========================================================================
# get_container_status
# ===========================================================================


class TestGetContainerStatus:
    """The facade must consult the K8s provisioner in K8s mode rather than the
    Docker-only core (which reports ERROR for pods absent from the local daemon).
    """

    async def test_non_k8s_delegates_to_core(self, container_manager):
        from src.models.container_models import ContainerStatus

        container_manager._github.use_k8s = False
        container_manager._github.k8s_provisioner = None
        container_manager._core.get_container_status.return_value = ContainerStatus.RUNNING

        result = await container_manager.get_container_status("c1")
        assert result == ContainerStatus.RUNNING
        container_manager._core.get_container_status.assert_called_once_with("c1")

    async def test_k8s_running_maps_to_running(self, container_manager):
        from src.models.container_models import ContainerStatus

        container_manager._github.use_k8s = True
        k8s_prov = MagicMock()
        k8s_prov.get_container_status = AsyncMock(return_value={"status": "running"})
        container_manager._github.k8s_provisioner = k8s_prov

        result = await container_manager.get_container_status("box")
        assert result == ContainerStatus.RUNNING
        k8s_prov.get_container_status.assert_awaited_once_with("box")
        # Must NOT fall through to the Docker-only core getter
        container_manager._core.get_container_status.assert_not_called()

    @pytest.mark.parametrize("k8s_status", ["starting", "pending"])
    async def test_k8s_provisioning_maps_to_creating(self, container_manager, k8s_status):
        from src.models.container_models import ContainerStatus

        container_manager._github.use_k8s = True
        k8s_prov = MagicMock()
        k8s_prov.get_container_status = AsyncMock(return_value={"status": k8s_status})
        container_manager._github.k8s_provisioner = k8s_prov

        result = await container_manager.get_container_status("box")
        assert result == ContainerStatus.CREATING

    async def test_k8s_not_found_maps_to_error(self, container_manager):
        from src.models.container_models import ContainerStatus

        container_manager._github.use_k8s = True
        k8s_prov = MagicMock()
        k8s_prov.get_container_status = AsyncMock(return_value={"status": "not_found"})
        container_manager._github.k8s_provisioner = k8s_prov

        result = await container_manager.get_container_status("box")
        assert result == ContainerStatus.ERROR

    async def test_k8s_api_exception_maps_to_error(self, container_manager):
        from kubernetes.client.rest import ApiException

        from src.models.container_models import ContainerStatus

        container_manager._github.use_k8s = True
        k8s_prov = MagicMock()
        k8s_prov.get_container_status = AsyncMock(side_effect=ApiException(status=500))
        container_manager._github.k8s_provisioner = k8s_prov

        result = await container_manager.get_container_status("box")
        assert result == ContainerStatus.ERROR


# ===========================================================================
# _scan_database_path error path
# ===========================================================================


class TestScanDatabasePathError:
    def test_oserror_handled(self, container_manager):
        """_scan_database_path handles OSError when scanning."""
        summaries: list[dict] = []
        # Pass a path that will cause an OSError when glob is called
        with patch("src.core.container_manager.Path") as mock_path:
            mock_dir = MagicMock()
            mock_dir.exists.return_value = True
            mock_dir.glob.side_effect = OSError("permission denied")
            mock_path.return_value = mock_dir
            container_manager._scan_database_path("/some/path", summaries)
        assert summaries == []


# ===========================================================================
# remove_container - K8s paths
# ===========================================================================


class TestRemoveContainerK8sPaths:
    @patch("src.core.container_manager.get_docker_service")
    async def test_k8s_remove_with_running_loop(self, mock_ds, container_manager):
        """K8s removal path with running event loop (lines 222-226)."""
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        container_manager._github.use_k8s = True
        k8s_prov = MagicMock()
        k8s_prov.remove_container = AsyncMock()
        container_manager._github.k8s_provisioner = k8s_prov
        container_manager._core.remove_container.return_value = True

        result = await container_manager.remove_container("c1")
        assert result["status"] == "success"

    @patch("src.core.container_manager.get_docker_service")
    @pytest.mark.asyncio
    async def test_k8s_remove_exception(self, mock_ds, container_manager):
        """K8s removal failure is handled gracefully (lines 231-232).

        When there IS a running loop and run_coroutine_threadsafe fails, the
        outer ``except Exception`` branch is taken.
        """
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        container_manager._github.use_k8s = True
        k8s_prov = MagicMock()
        k8s_prov.remove_container = AsyncMock(side_effect=OSError("k8s down"))
        container_manager._github.k8s_provisioner = k8s_prov
        container_manager._core.remove_container.return_value = True

        # Inside an async test there IS a running loop, so get_running_loop()
        # succeeds.  run_coroutine_threadsafe runs the coroutine which raises
        # OSError, caught by ``except Exception``.
        result = await container_manager.remove_container("c1")
        assert result["status"] == "success"

    @patch("src.core.container_manager.get_docker_service")
    async def test_remove_with_graph_deletion_callback_no_loop(self, mock_ds, container_manager):
        """remove_container uses callback for graph deletion with no event loop."""
        c = _make_container("c1", code_graph_id="g1")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"c1": c}
        mock_ds.return_value = docker_svc

        container_manager._github.use_k8s = False
        container_manager._core.remove_container.return_value = True
        mock_cb = AsyncMock(
            return_value={
                "success": True,
                "deleted_nodes": 10,
                "deleted_edges": 5,
                "paths_deleted": ["/db"],
            }
        )
        container_manager._graph_deletion_callback = mock_cb

        result = await container_manager.remove_container("c1")
        assert result["status"] == "success"
        assert "graph_deletion" in result
        assert result["graph_deletion"]["status"] == "deleted"

    @patch("src.core.container_manager.get_docker_service")
    async def test_remove_with_graph_deletion_failure(self, mock_ds, container_manager):
        """remove_container handles graph deletion failure (lines 355-363)."""
        c = _make_container("c1", code_graph_id="g1")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"c1": c}
        mock_ds.return_value = docker_svc

        container_manager._github.use_k8s = False
        container_manager._core.remove_container.return_value = True
        mock_cb = AsyncMock(
            return_value={
                "success": False,
                "message": "graph not found",
                "paths_checked": ["/db"],
            }
        )
        container_manager._graph_deletion_callback = mock_cb

        result = await container_manager.remove_container("c1")
        assert "graph_deletion" in result
        assert result["graph_deletion"]["status"] == "failed"

    @patch("src.core.container_manager.get_docker_service")
    async def test_remove_graph_deletion_exception(self, mock_ds, container_manager):
        """remove_container handles graph deletion exception (lines 364-369)."""
        c = _make_container("c1", code_graph_id="g1")
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"c1": c}
        mock_ds.return_value = docker_svc

        container_manager._github.use_k8s = False
        container_manager._core.remove_container.return_value = True
        container_manager._graph_deletion_callback = AsyncMock(
            side_effect=RuntimeError("deletion exploded")
        )

        result = await container_manager.remove_container("c1")
        assert "graph_deletion" in result
        assert result["graph_deletion"]["status"] == "error"

    @patch("src.core.container_manager.get_docker_service")
    async def test_remove_container_not_in_registry(self, mock_ds, container_manager):
        """remove_container handles container not in registry (line 253)."""
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        container_manager._github.use_k8s = False
        container_manager._core.remove_container.return_value = True

        result = await container_manager.remove_container("ghost")
        assert result["status"] == "success"

    @patch("src.core.container_manager.get_docker_service")
    async def test_remove_no_code_graph_id_uses_name(self, mock_ds, container_manager):
        """When container has no code_graph_id, uses container name as graph_id (line 245)."""
        c = _make_container("c1")
        c.code_graph_id = None
        del c.code_graph_id  # Remove attribute entirely
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"c1": c}
        mock_ds.return_value = docker_svc

        container_manager._github.use_k8s = False
        container_manager._core.remove_container.return_value = True
        container_manager._graph_deletion_callback = AsyncMock(
            return_value={
                "success": True,
                "deleted_nodes": 0,
                "deleted_edges": 0,
                "paths_deleted": [],
            }
        )

        result = await container_manager.remove_container("c1")
        assert result["status"] == "success"


# ===========================================================================
# get_all_graphs_summary_info - error path
# ===========================================================================


class TestGetAllGraphsSummaryInfoError:
    @patch("src.core.container_manager.get_docker_service")
    @patch("src.core.container_manager.get_settings")
    @pytest.mark.asyncio
    async def test_error_returns_empty_list(self, mock_settings, mock_ds, container_manager):
        """get_all_graphs_summary_info returns [] on error (lines 625-627)."""
        mock_settings.side_effect = ValueError("settings broken")
        result = await container_manager.get_all_graphs_summary_info()
        assert result == []


# ===========================================================================
# get_graph_details_by_id - additional paths
# ===========================================================================


class TestGetGraphDetailsByIdExtended:
    @patch("src.core.container_manager.get_docker_service")
    @patch("src.core.container_manager.get_settings")
    @pytest.mark.asyncio
    async def test_database_manager_not_implemented(
        self, mock_settings, mock_ds, container_manager
    ):
        """When db manager raises NotImplementedError (lines 662-664)."""
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        settings = MagicMock()
        settings.get_database_manager.side_effect = NotImplementedError
        mock_settings.return_value = settings

        result = await container_manager.get_graph_details_by_id("g1")
        assert result is None

    @patch("src.core.container_manager.get_docker_service")
    @patch("src.core.container_manager.get_settings")
    @pytest.mark.asyncio
    async def test_error_returns_none(self, mock_settings, mock_ds, container_manager):
        """get_graph_details_by_id returns None on error (lines 681-683)."""
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        # Make get_settings raise so the outer except catches it
        mock_settings.side_effect = OSError("settings broken")

        result = await container_manager.get_graph_details_by_id("g1")
        assert result is None


# ===========================================================================
# get_container: registry-miss fallback to the live k8s Deployment
# (regression: a fresh container 404'd for ~10 min after a transient registry
# drop, blocking the tournament candidate from ever committing)
# ===========================================================================


class TestGetContainerK8sFallback:
    @patch("src.core.container_manager.get_docker_service")
    def test_registry_miss_falls_back_to_live_deployment(self, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}  # registry transiently empty
        mock_ds.return_value = docker_svc

        rebuilt = _make_container("tour-x-c0", tenant_id=None)
        container_manager._github.use_k8s = True
        container_manager._github.k8s_provisioner.reconstruct_container_info.return_value = rebuilt

        result = container_manager.get_container("tour-x-c0")
        assert result is rebuilt
        container_manager._github.k8s_provisioner.reconstruct_container_info.assert_called_once_with(
            "tour-x-c0"
        )

    @patch("src.core.container_manager.get_docker_service")
    def test_fallback_respects_tenant_scope(self, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        rebuilt = _make_container("tour-x-c0", tenant_id="owner-a")
        container_manager._github.use_k8s = True
        container_manager._github.k8s_provisioner.reconstruct_container_info.return_value = rebuilt

        # Wrong tenant -> None (BOLA), right tenant -> the container
        assert container_manager.get_container("tour-x-c0", tenant_id="owner-b") is None
        assert container_manager.get_container("tour-x-c0", tenant_id="owner-a") is rebuilt

    @patch("src.core.container_manager.get_docker_service")
    def test_registry_miss_unrecoverable_owner_is_denied_cross_tenant(
        self, mock_ds, container_manager
    ):
        # ⚠️ INVERTED. This was
        # ``test_registry_miss_ownerless_reconstruct_is_cross_tenant_readable``
        # and asserted ``result is rebuilt`` — that a reconstructed container with
        # ``tenant_id=None`` stays readable by ANY tenant. Its justification was
        # that such a container is OWNERLESS ("provisioned with no tenant at all,
        # so no label to recover") and that this "is NOT the same thing as owner
        # unknown".
        #
        # It is exactly the same thing, and that is the defect. The ``tenant-id``
        # label is written only when the value satisfies ``_K8S_LABEL_VALUE_RE``,
        # while ``effective_tenant_id`` is ``tenant_id or organization_id or sub``
        # with no grammar validation — so "no label" is equally consistent with an
        # OWNED workspace whose owner could not be expressed as a label.
        # ``reconstruct_container_info`` now recovers the owner from the
        # ``coding-job/tenant-id`` ANNOTATION first (no grammar restriction) and
        # falls back to the label, so a surviving ``None`` means UNKNOWN — and an
        # unknown owner must not satisfy a foreign caller's read.
        #
        # ⛔ Scoped to the RECONSTRUCTED path only. A registry-HIT row with
        # ``tenant_id=None`` is genuinely legacy data this service wrote, and
        # keeps its fail-open — see ``test_legacy_null_tenant_allowed``, which is
        # the control proving this change did not flip that gate wholesale.
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}  # forced registry miss
        mock_ds.return_value = docker_svc

        rebuilt = _make_container(
            "it-flow1-x",
            tenant_id=None,
            repo_url="https://github.com/nonagenticai/enterprise_mcp_server",
        )
        container_manager._github.use_k8s = True
        container_manager._github.k8s_provisioner.reconstruct_container_info.return_value = rebuilt

        result = container_manager.get_container(
            "it-flow1-x", tenant_id="0195f48b-d000-7000-8000-000000000003"
        )
        assert result is None, "a workspace with no recoverable owner was served cross-tenant"

    @patch("src.core.container_manager.get_docker_service")
    def test_registry_miss_recovered_owner_still_reads_its_own(self, mock_ds, container_manager):
        """POSITIVE CONTROL for the denial above.

        Without it that assertion is satisfiable by a gate that denies every
        reconstructed container, which would break the registry-miss fallback
        outright — the fallback exists so the OWNER keeps working while Redis is
        momentarily missing the row.
        """
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        owner = "0195f48b-d000-7000-8000-000000000001"
        rebuilt = _make_container(
            "it-flow1-x",
            tenant_id=owner,
            repo_url="https://github.com/nonagenticai/enterprise_mcp_server",
        )
        container_manager._github.use_k8s = True
        container_manager._github.k8s_provisioner.reconstruct_container_info.return_value = rebuilt

        result = container_manager.get_container("it-flow1-x", tenant_id=owner)
        assert result is rebuilt
        # The reconstructed container must carry a non-empty repo_url so the
        # coding-github apply-changes/fork guard (not container.repo_url -> 404)
        # passes on the registry-miss path — the winner-PR-loop blocker.
        assert result.repo_url == "https://github.com/nonagenticai/enterprise_mcp_server"

    @patch("src.core.container_manager.get_docker_service")
    def test_registry_hit_legacy_null_tenant_still_fails_open(self, mock_ds, container_manager):
        """⛔ CONTROL proving the fail-closed change is SCOPED to reconstruct.

        A registry HIT carrying ``tenant_id=None`` is legacy data written by this
        service, not an unresolvable k8s Deployment, and it deliberately keeps the
        documented fail-open. If this goes red, the change stopped being a
        targeted fix and became a fleet-wide tenant-gate flip.
        """
        legacy = _make_container("legacy-ws", tenant_id=None)
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"legacy-ws": legacy}
        mock_ds.return_value = docker_svc

        result = container_manager.get_container("legacy-ws", tenant_id="any-tenant")
        assert result is legacy
        # And reconstruct was never consulted — this is the HIT path.
        container_manager._github.k8s_provisioner.reconstruct_container_info.assert_not_called()

    @patch("src.core.container_manager.get_docker_service")
    def test_list_drops_cluster_only_entries_with_no_recoverable_owner(
        self, mock_ds, container_manager
    ):
        """THE THIRD SITE of the same defect, found while fixing the other two.

        ``list_containers`` unions live Deployments the registry is missing, then
        filters by tenant with an ``or tenant_id is None`` arm for "legacy"
        rows. A cluster-only entry whose owner is unrecoverable landed in that
        arm and was listed to EVERY tenant — the same name-and-readiness
        disclosure the /status fix closes, one route over.

        The unrecoverable entry is dropped at the rebuild loop, so the ``is None``
        arm now covers only genuine registry rows. Both halves asserted: the
        unknown-owner entry is absent AND the owner's own entry is present, so a
        filter that simply dropped every rebuilt entry cannot pass.
        """
        owner = "0195f48b-d000-7000-8000-000000000001"
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc
        container_manager._core.list_containers = MagicMock(return_value=[])

        unknown = _make_container("unstamped-ws", tenant_id=None)
        owned = _make_container("owned-ws", tenant_id=owner)
        container_manager._github.use_k8s = True
        k8s = container_manager._github.k8s_provisioner
        k8s.list_managed_container_names.return_value = ["unstamped-ws", "owned-ws"]
        k8s.reconstruct_container_info.side_effect = lambda n: {
            "unstamped-ws": unknown,
            "owned-ws": owned,
        }[n]

        names = [c.container_name for c in container_manager.list_containers(tenant_id=owner)]

        assert "unstamped-ws" not in names, (
            "a cluster-only workspace with no recoverable owner was listed cross-tenant"
        )
        assert "owned-ws" in names, "the rebuild union stopped working entirely"

    @patch("src.core.container_manager.get_docker_service")
    def test_no_deployment_still_raises(self, mock_ds, container_manager):
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {}
        mock_ds.return_value = docker_svc

        container_manager._github.use_k8s = True
        container_manager._github.k8s_provisioner.reconstruct_container_info.return_value = None

        with pytest.raises(ValueError, match="not found"):
            container_manager.get_container("ghost")


# ===========================================================================
# ensure_container — bounded re-provision of a REAPED Deployment
#
# The reaper deletes a Deployment on a 24h TTL with no in-use guard while the
# Redis metadata registry keeps the entry, so the recoverable state is
# "registry entry present, Deployment gone". These tests pin the three
# properties the fix has to hold: it re-provisions that state, it still fails
# loudly for anything it cannot provision, and it cannot create a second
# Deployment (or spin).
# ===========================================================================


_REAPED_TENANT = "0195f48b-d000-7000-8000-0000000000aa"


def _reaped_info(
    name: str = "it-flow1-bump-fastmcp-to-3-4-7", tenant_id: str | None = _REAPED_TENANT
) -> MagicMock:
    """Registry metadata that survived the reap of its Deployment.

    Stamped with a tenant by default: a tenantless PLATFORM row is refused
    (no implicit platform tenant), so only the tenant-binding tests model one.
    """
    c = _make_container(name=name, tenant_id=tenant_id)
    c.repo_url = "https://github.com/nonagenticai/coding-containers"
    c.branch = "main"
    c.secret_config_id = "sc-1"
    return c


@pytest.mark.asyncio
class TestEnsureContainerReprovision:
    @pytest.fixture(autouse=True)
    def _clear_guards(self):
        from src.core.container_manager import _reset_reprovision_guards

        _reset_reprovision_guards()
        yield
        _reset_reprovision_guards()

    @staticmethod
    def _arm(mgr, registry, *, statuses, provisioned=None):
        """Wire the manager: registry contents + a scripted k8s status sequence."""
        mgr._github.use_k8s = True
        k8s = mgr._github.k8s_provisioner
        k8s.reconstruct_container_info.return_value = None
        k8s.get_container_status = AsyncMock(side_effect=list(statuses))
        k8s.provision_container = AsyncMock(return_value=provisioned)
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = registry
        return k8s, docker_svc

    async def test_involuntarily_lost_deployment_is_reprovisioned(self, container_manager):
        """(i) Deployment gone + registry spec intact -> re-provision, usable info.

        ⚠️ RENAMED from ``test_reaped_name_is_reprovisioned``: this state is no
        longer the reaper's fingerprint. A reap now drops the registry row
        itself (``_forget_reaped_metadata``), so a *reaped* name reaches this
        method with NO spec and is refused by
        ``test_unknown_name_still_raises_and_provisions_nothing``. What is
        rescued here is INVOLUNTARY loss — a crash, a node failure, an
        out-of-band ``kubectl delete deploy`` — where the Deployment vanished
        but nobody decided the container was finished. The old name now reads
        as sanctioning exactly the resurrection the reaper fix prevents.
        """
        info = _reaped_info()
        fresh = _reaped_info()
        fresh.status = "creating"
        k8s, docker_svc = self._arm(
            container_manager,
            {info.container_name: info},
            statuses=[{"status": "not_found"}],
            provisioned=fresh,
        )
        with patch("src.core.container_manager.get_docker_service", return_value=docker_svc):
            result = await container_manager.ensure_container(info.container_name)

        assert result is fresh
        k8s.provision_container.assert_awaited_once()
        kwargs = k8s.provision_container.await_args.kwargs
        assert kwargs["container_name"] == info.container_name
        assert kwargs["repo_url"] == info.repo_url
        assert kwargs["branch"] == "main"
        assert kwargs["secret_config_id"] == "sc-1"

    async def test_unknown_name_still_raises_and_provisions_nothing(self, container_manager):
        """(ii) Registry miss AND no Deployment -> no spec exists anywhere: raise."""
        k8s, docker_svc = self._arm(container_manager, {}, statuses=[{"status": "not_found"}])
        with patch("src.core.container_manager.get_docker_service", return_value=docker_svc):
            with pytest.raises(ValueError, match="not found"):
                await container_manager.ensure_container("never-existed")
        k8s.provision_container.assert_not_awaited()

    async def test_spec_without_repo_url_still_raises(self, container_manager):
        """(ii) Known name, but nothing to provision FROM -> raise, never silently succeed."""
        info = _reaped_info("no-spec")
        info.repo_url = ""
        k8s, docker_svc = self._arm(
            container_manager,
            {"no-spec": info},
            statuses=[{"status": "not_found"}],
        )
        with patch("src.core.container_manager.get_docker_service", return_value=docker_svc):
            with pytest.raises(ValueError, match="not found"):
                await container_manager.ensure_container("no-spec")
        k8s.provision_container.assert_not_awaited()

    async def test_provisioning_error_propagates(self, container_manager):
        """(ii) A provisioner failure must surface, not be swallowed into success."""
        info = _reaped_info("boom")
        k8s, docker_svc = self._arm(
            container_manager, {"boom": info}, statuses=[{"status": "not_found"}]
        )
        k8s.provision_container = AsyncMock(side_effect=RuntimeError("no git credential"))
        with patch("src.core.container_manager.get_docker_service", return_value=docker_svc):
            with pytest.raises(RuntimeError, match="no git credential"):
                await container_manager.ensure_container("boom")

    async def test_live_deployment_short_circuits(self, container_manager):
        """(iii) Idempotency: a Deployment that exists is never re-provisioned."""
        info = _reaped_info("alive")
        k8s, docker_svc = self._arm(
            container_manager, {"alive": info}, statuses=[{"status": "running"}]
        )
        with patch("src.core.container_manager.get_docker_service", return_value=docker_svc):
            result = await container_manager.ensure_container("alive")
        assert result is info
        k8s.provision_container.assert_not_awaited()

    async def test_two_calls_provision_once(self, container_manager):
        """(iii) Idempotency: call twice, one Deployment.

        First call sees not_found and provisions; the second sees the
        Deployment it just created and short-circuits.
        """
        info = _reaped_info("twice")
        k8s, docker_svc = self._arm(
            container_manager,
            {"twice": info},
            statuses=[{"status": "not_found"}, {"status": "running"}],
            provisioned=info,
        )
        with patch("src.core.container_manager.get_docker_service", return_value=docker_svc):
            first = await container_manager.ensure_container("twice")
            second = await container_manager.ensure_container("twice")
        assert first is info and second is info
        k8s.provision_container.assert_awaited_once()

    async def test_cooldown_blocks_a_second_attempt(self, container_manager):
        """Bounded: two not_found calls in a row provision ONCE, then refuse loudly."""
        info = _reaped_info("cooling")
        k8s, docker_svc = self._arm(
            container_manager,
            {"cooling": info},
            statuses=[{"status": "not_found"}, {"status": "not_found"}],
            provisioned=info,
        )
        with patch("src.core.container_manager.get_docker_service", return_value=docker_svc):
            await container_manager.ensure_container("cooling")
            with pytest.raises(ValueError, match="cooling down"):
                await container_manager.ensure_container("cooling")
        k8s.provision_container.assert_awaited_once()

    async def test_tenant_mismatch_never_provisions(self, container_manager):
        """BOLA parity: a foreign container returns None and provisions nothing."""
        info = _reaped_info("theirs")
        info.tenant_id = "owner-a"
        k8s, docker_svc = self._arm(
            container_manager, {"theirs": info}, statuses=[{"status": "not_found"}]
        )
        with patch("src.core.container_manager.get_docker_service", return_value=docker_svc):
            assert await container_manager.ensure_container("theirs", tenant_id="hacker") is None
        k8s.provision_container.assert_not_awaited()


@pytest.mark.asyncio
class TestExecuteCommandReprovisionsReapedDeployment:
    """The live 404 path: GET /api/workspaces/{name}/files -> execute_command."""

    @pytest.fixture(autouse=True)
    def _clear_guards(self):
        from src.core.container_manager import _reset_reprovision_guards

        _reset_reprovision_guards()
        yield
        _reset_reprovision_guards()

    @staticmethod
    def _ok_result():
        r = MagicMock()
        r.succeeded = True
        r.stdout = "DIR"
        r.stderr = ""
        r.exit_code = 0
        r.truncated = False
        r.bytes_omitted = 0
        return r

    async def test_exec_recovers_after_one_reprovision(self, container_manager):
        from src.core.kubernetes_container_provisioner import (
            ContainerDeploymentNotFoundError,
        )

        info = _reaped_info("reaped-box")
        container_manager._github.use_k8s = True
        k8s = container_manager._github.k8s_provisioner
        k8s.reconstruct_container_info.return_value = None
        k8s.get_container_status = AsyncMock(return_value={"status": "not_found"})
        k8s.provision_container = AsyncMock(return_value=info)
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"reaped-box": info}

        runtime = MagicMock()
        runtime.run_once = AsyncMock(
            side_effect=[
                ContainerDeploymentNotFoundError("Deployment reaped-box not found: 404"),
                self._ok_result(),
            ]
        )
        with (
            patch("src.core.container_manager.get_docker_service", return_value=docker_svc),
            patch("src.core.container_manager.resolve_runtime", return_value=runtime),
        ):
            result = await container_manager.execute_command("reaped-box", "ls -la /workspace")

        assert result["success"] is True
        assert result["stdout"] == "DIR"
        k8s.provision_container.assert_awaited_once()
        assert runtime.run_once.await_count == 2

    async def test_a_404_cannot_spin(self, container_manager):
        """Bounded: still not-found after the re-provision -> report, do not loop."""
        from src.core.kubernetes_container_provisioner import (
            ContainerDeploymentNotFoundError,
        )

        info = _reaped_info("stuck-box")
        container_manager._github.use_k8s = True
        k8s = container_manager._github.k8s_provisioner
        k8s.reconstruct_container_info.return_value = None
        k8s.get_container_status = AsyncMock(return_value={"status": "not_found"})
        k8s.provision_container = AsyncMock(return_value=info)
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {"stuck-box": info}

        runtime = MagicMock()
        # FINITE side_effect, exactly the two calls asserted below. A THIRD
        # call exhausts the list and fails in microseconds, so an unbounded
        # retry loop reds this test instead of manifesting as a ~400s CI hang.
        # ⛔ ``asyncio.wait_for`` cannot do this job — measured: the retry spin
        # awaits only AsyncMocks, so it never yields to the event loop and a
        # wrapping timeout never fires (an unbounded-retry mutant hung >120s).
        runtime.run_once = AsyncMock(
            side_effect=[
                ContainerDeploymentNotFoundError("Deployment stuck-box not found: 404"),
                ContainerDeploymentNotFoundError("Deployment stuck-box not found: 404"),
            ]
        )
        with (
            patch("src.core.container_manager.get_docker_service", return_value=docker_svc),
            patch("src.core.container_manager.resolve_runtime", return_value=runtime),
        ):
            result = await container_manager.execute_command("stuck-box", "ls")

        # Still an honest failure whose message the files router maps to 404.
        assert result["success"] is False
        assert "not found" in result["stderr"].lower()
        # ONE provision, TWO run attempts. No loop.
        k8s.provision_container.assert_awaited_once()
        assert runtime.run_once.await_count == 2


# ===========================================================================
# ensure_container — the three guarantees its DOCSTRING asserts
#
# Derived from the promised properties, not from the code: an adversarial
# mutation round showed the prose guarantees "never two concurrently", "stamp
# BEFORE attempting, so a provision that RAISES still consumes the cooldown"
# and "every message keeps the words 'not found'" were each unpinned — the
# suite stayed green with all three deleted.
# ===========================================================================


@pytest.mark.asyncio
class TestEnsureContainerReprovisionStatedGuarantees:
    @pytest.fixture(autouse=True)
    def _clear_guards(self):
        from src.core.container_manager import _reset_reprovision_guards

        _reset_reprovision_guards()
        yield
        _reset_reprovision_guards()

    @staticmethod
    def _arm(mgr, registry, *, statuses, provision):
        """As TestEnsureContainerReprovision._arm, but the caller owns the mock."""
        mgr._github.use_k8s = True
        k8s = mgr._github.k8s_provisioner
        k8s.reconstruct_container_info.return_value = None
        k8s.get_container_status = AsyncMock(side_effect=list(statuses))
        k8s.provision_container = provision
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = registry
        return k8s, docker_svc

    async def test_concurrent_calls_provision_once_and_loser_raises(self, container_manager):
        """ "never two provisions concurrently for one name".

        ``REPROVISION_COOLDOWN_S`` is pinned to 0 so the cooldown CANNOT be the
        guard that stops the second call — only ``_REPROVISION_INFLIGHT`` can.
        The two genuinely overlap: the loser is released only once the winner is
        parked inside ``provision_container``. A second provision returns
        immediately rather than waiting on the same event, so losing the
        in-flight guard fails this test by assertion instead of deadlocking it.
        """
        info = _reaped_info("concurrent")
        fresh = _reaped_info("concurrent")
        fresh.status = "creating"
        winner_inside = asyncio.Event()
        let_winner_finish = asyncio.Event()
        entered = 0

        async def slow_provision(**_kwargs):
            nonlocal entered
            entered += 1
            if entered == 1:
                winner_inside.set()
                await let_winner_finish.wait()
            return fresh

        k8s, docker_svc = self._arm(
            container_manager,
            {"concurrent": info},
            statuses=[{"status": "not_found"}, {"status": "not_found"}],
            provision=AsyncMock(side_effect=slow_provision),
        )

        async def loser():
            await asyncio.wait_for(winner_inside.wait(), timeout=10)
            try:
                return await container_manager.ensure_container("concurrent")
            finally:
                let_winner_finish.set()

        with (
            patch("src.core.container_manager.get_docker_service", return_value=docker_svc),
            patch("src.core.container_manager.REPROVISION_COOLDOWN_S", 0.0),
        ):
            won, lost = await asyncio.wait_for(
                asyncio.gather(
                    container_manager.ensure_container("concurrent"),
                    loser(),
                    return_exceptions=True,
                ),
                timeout=20,
            )

        # The load-bearing assertion: ONE provision for two overlapping calls.
        k8s.provision_container.assert_awaited_once()
        assert won is fresh, f"winner did not get the provisioned container: {won!r}"
        assert isinstance(lost, ValueError), f"loser did not refuse: {lost!r}"
        assert "not found" in str(lost).lower(), f"loser's refusal is not a 404: {lost}"

    async def test_failed_provision_still_consumes_the_cooldown(self, container_manager):
        """ "Stamp BEFORE attempting, so a provision that RAISES still consumes
        the cooldown" — the property that bounds a PERMANENTLY failing name to
        one attempt per window instead of one attempt per request.
        """
        info = _reaped_info("always-fails")
        k8s, docker_svc = self._arm(
            container_manager,
            {"always-fails": info},
            statuses=[{"status": "not_found"}, {"status": "not_found"}],
            provision=AsyncMock(side_effect=RuntimeError("provisioner down")),
        )
        with patch("src.core.container_manager.get_docker_service", return_value=docker_svc):
            with pytest.raises(RuntimeError, match="provisioner down"):
                await container_manager.ensure_container("always-fails")

            # Captured rather than pytest.raises'd so that a SECOND provisioner
            # error reads as an assertion failure, not as a crash in the test.
            second: Exception | None = None
            try:
                await container_manager.ensure_container("always-fails")
            except Exception as exc:
                second = exc

        assert isinstance(second, ValueError) and "cooling down" in str(second), (
            f"a failed provision did not consume the cooldown; second attempt: {second!r}"
        )
        k8s.provision_container.assert_awaited_once()

    async def test_cooldown_refusal_keeps_the_words_not_found(self, container_manager):
        """ "Every message keeps the words 'not found'" — the cooldown half.

        ``src/api/routers/files_io.py`` maps with ``"not found" in
        stderr.lower()``, so dropping those two words turns this honest 404 into
        a 500. The no-``repo_url`` half is already asserted; this is the other.
        """
        info = _reaped_info("still-404")
        k8s, docker_svc = self._arm(
            container_manager,
            {"still-404": info},
            statuses=[{"status": "not_found"}, {"status": "not_found"}],
            provision=AsyncMock(return_value=info),
        )
        with patch("src.core.container_manager.get_docker_service", return_value=docker_svc):
            await container_manager.ensure_container("still-404")
            with pytest.raises(ValueError) as exc:
                await container_manager.ensure_container("still-404")

        assert "not found" in str(exc.value).lower(), (
            f"cooldown refusal no longer maps to a 404: {exc.value}"
        )
        k8s.provision_container.assert_awaited_once()


@pytest.mark.asyncio
class TestReprovisionTenantBinding:
    """The re-provision's clone credential is minted for the ROW's tenant only."""

    @pytest.fixture(autouse=True)
    def _clear_guards(self):
        from src.core.container_manager import _reset_reprovision_guards

        _reset_reprovision_guards()
        yield
        _reset_reprovision_guards()

    @staticmethod
    def _arm(mgr, info):
        mgr._github.use_k8s = True
        k8s = mgr._github.k8s_provisioner
        k8s.reconstruct_container_info.return_value = None
        k8s.get_container_status = AsyncMock(return_value={"status": "not_found"})
        k8s.provision_container = AsyncMock(return_value=info)
        docker_svc = MagicMock()
        docker_svc.registry.load.return_value = {info.container_name: info}
        return k8s, docker_svc

    async def test_customer_repo_without_a_row_tenant_is_refused(self, container_manager, caplog):
        info = _reaped_info("cust-no-tenant", tenant_id=None)
        info.repo_url = "https://github.com/acme-corp/private-app"
        k8s, docker_svc = self._arm(container_manager, info)
        with patch("src.core.container_manager.get_docker_service", return_value=docker_svc):
            with pytest.raises(ValueError, match="carries no tenant"):
                await container_manager.ensure_container("cust-no-tenant")
            # Negative control, persisted half: refused before provisioning, so
            # the registry row is untouched — still no tenant.
            row = container_manager.get_container("cust-no-tenant")
        k8s.provision_container.assert_not_awaited()
        assert row.tenant_id is None
        assert "cust-no-tenant" in caplog.text

    async def test_platform_repo_without_a_row_tenant_is_refused_not_substituted(
        self, container_manager, caplog
    ):
        """A legacy tenantless PLATFORM row is refused — no implicit platform tenant.

        It must be re-registered under its real tenant. The refusal is logged
        with a stable key so the (expected-zero) legacy population is countable.
        """
        info = _reaped_info("plat-no-tenant", tenant_id=None)
        assert is_platform_owner(repo_owner_from_url(info.repo_url))
        k8s, docker_svc = self._arm(container_manager, info)
        with (
            caplog.at_level("WARNING"),
            patch("src.core.container_manager.get_docker_service", return_value=docker_svc),
        ):
            with pytest.raises(ValueError, match="tenant required for platform repo"):
                await container_manager.ensure_container("plat-no-tenant")
            row = container_manager.get_container("plat-no-tenant")
        k8s.provision_container.assert_not_awaited()
        assert row.tenant_id is None, "a refusal must not stamp a tenant onto the row"
        assert any(
            "event=platform_tenant_missing_refused" in r.getMessage()
            and "plat-no-tenant" in r.getMessage()
            for r in caplog.records
        )

    async def test_row_tenant_is_sent_unchanged_on_a_platform_repo(self, container_manager):
        """A row that HAS a tenant keeps sending its own — on a PLATFORM repo."""
        info = _reaped_info("plat-with-tenant")
        info.tenant_id = "0195f48b-d000-7000-8000-00000000beef"
        assert is_platform_owner(repo_owner_from_url(info.repo_url)), (
            "fixture must be a PLATFORM repo or this test cannot catch the override"
        )
        k8s, docker_svc = self._arm(container_manager, info)
        with patch("src.core.container_manager.get_docker_service", return_value=docker_svc):
            await container_manager.ensure_container("plat-with-tenant", tenant_id=None)
        assert k8s.provision_container.await_args.kwargs["tenant_id"] == info.tenant_id

    async def test_row_tenant_wins_over_the_callers(self, container_manager):
        info = _reaped_info("has-tenant")
        info.tenant_id = "tenant-row"
        k8s, docker_svc = self._arm(container_manager, info)
        with patch("src.core.container_manager.get_docker_service", return_value=docker_svc):
            await container_manager.ensure_container("has-tenant", tenant_id=None)
        assert k8s.provision_container.await_args.kwargs["tenant_id"] == "tenant-row"

    @staticmethod
    def _arm_persisting(mgr, info):
        """Like ``_arm``, but the mocked provisioner also PERSISTS.

        At this layer ``k8s.provision_container`` is a full mock — the real
        ``K8sContainerProvisioner`` code (which builds a new ``ContainerInfo``
        with ``tenant_id=request.tenant_id`` and calls
        ``ContainerMetadataRegistry().upsert(...)``) never runs, so a test here
        can only assert the ARGUMENT handed to the mock unless the mock is given
        a side effect that performs the one write the real provisioner is
        proven to perform — see
        ``tests/unit/test_tenant_bound_clone.py::TestK8sReprovisionTenantPersistence``,
        which runs that real code and confirms this is exactly what it does.
        This side effect mutates the SAME ``info`` object the fake registry
        holds, so ``ContainerManager.get_container`` reads back the persisted
        value exactly as a caller would after a real re-provision.
        """

        async def _persist(*, tenant_id, **_kwargs):
            info.tenant_id = tenant_id
            return info

        k8s, docker_svc = TestReprovisionTenantBinding._arm(mgr, info)
        k8s.provision_container = AsyncMock(side_effect=_persist)
        return k8s, docker_svc

    async def test_a_tenanted_row_is_persisted_unchanged_by_reprovision(self, container_manager):
        """Negative control, persisted half: an existing tenant is not overwritten."""
        info = _reaped_info("plat-with-tenant-persist")
        info.tenant_id = "0195f48b-d000-7000-8000-00000000beef"
        k8s, docker_svc = self._arm_persisting(container_manager, info)
        with patch("src.core.container_manager.get_docker_service", return_value=docker_svc):
            await container_manager.ensure_container("plat-with-tenant-persist", tenant_id=None)
            row = container_manager.get_container(
                "plat-with-tenant-persist", tenant_id="0195f48b-d000-7000-8000-00000000beef"
            )
        assert row.tenant_id == "0195f48b-d000-7000-8000-00000000beef"

"""Tests for src/core/kubernetes_container_provisioner.py — K8sContainerProvisioner."""

from __future__ import annotations

import importlib
import json
import sys
import time as time_module
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: force-load the REAL module (conftest installs a stub).
# ---------------------------------------------------------------------------
_REAL_MOD_NAME = "src.core.kubernetes_container_provisioner"

# ⛔ REUSE the real module when it is already loaded; only pop + re-import when
# the sys.modules entry is a stub (detected by the absence of the symbols this
# file needs). An UNCONDITIONAL pop + re-import created a SECOND copy of the
# module, and therefore a second, non-identical
# ``ContainerDeploymentNotFoundError`` class object. Any production module that
# had already bound the first copy (e.g. ``src.core.container_manager``, pulled
# in at collection time by test_container_exec_router.py) then had its
# ``except ContainerDeploymentNotFoundError`` silently miss instances raised
# from the second copy — it only still caught them via the broader
# ``except RuntimeError`` further out, which turns a recoverable not-found into
# a reported failure. That is invisible in a single-file run and only appears in
# a full-tree run. test_reaper.py already guards this the same way; keep both.
_existing = sys.modules.get(_REAL_MOD_NAME)
if _existing is not None and hasattr(_existing, "K8sContainerProvisioner"):
    sys.modules[_REAL_MOD_NAME] = _existing
else:
    sys.modules.pop(_REAL_MOD_NAME, None)

# The real module does `from kubernetes import client, config` and
# `from kubernetes.stream import stream as k8s_stream` at top-level.
# Those are available because kubernetes is installed.
# It also does `from src.clients.github_client import get_github_token_manager`
# and `from src.config.settings import get_settings` — both of which will
# import fine because conftest has stubs for coding_client.

# Patch kubernetes.config so __init__ won't fail trying to load a real kubeconfig
with patch("kubernetes.config.load_kube_config"):
    _mod = importlib.import_module(_REAL_MOD_NAME)

K8sContainerProvisioner = _mod.K8sContainerProvisioner
get_k8s_container_provisioner = _mod.get_k8s_container_provisioner

# Aliases for tests that patch/monkeypatch module-level names.
MODULE = _REAL_MOD_NAME
kcp = _mod


# ---------------------------------------------------------------------------
# Helper: build a provisioner with mocked K8s API clients
# ---------------------------------------------------------------------------


def _make_provisioner(namespace: str = "test-ns") -> K8sContainerProvisioner:
    with patch("kubernetes.config.load_kube_config"):
        prov = K8sContainerProvisioner(namespace=namespace, in_cluster=False)
    prov.apps_v1 = MagicMock()
    prov.core_v1 = MagicMock()
    return prov


def _app_credential(**kwargs):
    """A CloneCredential from the coding-github clone-token endpoint.

    Default: the lookup did not succeed (e.g. CODING_GITHUB_URL unset) — the
    pre-App behaviour, so the ladder falls through to the env system token.
    """
    from src.clients.github_client import CloneCredential

    defaults = {"ok": False, "error": "coding_github_url_unset"}
    defaults.update(kwargs)
    return CloneCredential(**defaults)


#: An App token for the platform-owner provision tests: the org PAT is never a
#: clone credential, so a provision that reaches the clone needs one of these.
_APP_OK = {"ok": True, "token": "ghs_app", "visibility": "private", "source": "app"}


def _mock_token_manager(
    system_token: str | None = "ghp_sys",
    credential=None,
    secret_config_id: str | None = "sec-id",
) -> MagicMock:
    """Autospec'd stub of the REAL GitHubTokenManager.

    ``create_autospec`` rather than a bare ``MagicMock``: a signature-agnostic
    double accepts any call and therefore cannot fail, and this fleet has already
    shipped fabricated passes that way. Autospec makes a renamed method or a
    changed keyword a test failure, and gives the async methods AsyncMock
    automatically.
    """
    from unittest.mock import create_autospec

    from src.clients.github_client import GitHubTokenManager

    mgr = create_autospec(GitHubTokenManager, instance=True)
    mgr.get_system_owner.return_value = "sys-owner"
    mgr.store_user_token.return_value = "sec-id"
    mgr.resolve_provision_secret_config_id.return_value = secret_config_id
    mgr.get_app_clone_credential.return_value = (
        credential if credential is not None else _app_credential()
    )
    return mgr


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_explicit_namespace(self):
        prov = _make_provisioner(namespace="my-ns")
        assert prov.namespace == "my-ns"

    def test_in_cluster_config(self):
        with patch("kubernetes.config.load_incluster_config") as mock_ic:
            K8sContainerProvisioner(namespace="test", in_cluster=True)
            mock_ic.assert_called_once()

    def test_config_load_failure_raises(self):
        with patch(
            "kubernetes.config.load_kube_config",
            side_effect=Exception("no kubeconfig"),
        ):
            with pytest.raises(Exception, match="no kubeconfig"):
                K8sContainerProvisioner(namespace="test", in_cluster=False)


# ---------------------------------------------------------------------------
# _extract_repo_name
# ---------------------------------------------------------------------------


class TestExtractRepoName:
    def test_simple_url(self):
        prov = _make_provisioner()
        assert prov._extract_repo_name("https://github.com/owner/my-repo") == "my-repo"

    def test_url_with_git_suffix(self):
        prov = _make_provisioner()
        assert prov._extract_repo_name("https://github.com/owner/my-repo.git") == "my-repo"

    def test_underscores_replaced(self):
        prov = _make_provisioner()
        assert prov._extract_repo_name("https://github.com/owner/my_repo") == "my-repo"

    def test_uppercase_lowered(self):
        prov = _make_provisioner()
        assert prov._extract_repo_name("https://github.com/Owner/MyRepo") == "myrepo"


# ---------------------------------------------------------------------------
# _create_deployment
# ---------------------------------------------------------------------------


class TestCreateDeployment:
    def test_creates_deployment(self):
        prov = _make_provisioner()
        prov._create_deployment(
            name="test-deploy",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
        )
        prov.apps_v1.create_namespaced_deployment.assert_called_once()

    def test_reconciles_in_place_on_409_when_stable(self):
        """On 409 against a stable (not terminating) Deployment, patch in place.

        The old behavior deleted + slept + recreated, which raced the finalizer.
        The new contract: no delete, a single in-place patch (the changed
        provisioned-at pod-template annotation rolls a fresh pod).
        """
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()

        conflict = ApiException(status=409, reason="Conflict")
        prov.apps_v1.create_namespaced_deployment.side_effect = conflict

        existing = MagicMock()
        existing.metadata.deletion_timestamp = None
        prov.apps_v1.read_namespaced_deployment.return_value = existing

        prov._create_deployment(
            name="test-deploy",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
        )

        prov.apps_v1.patch_namespaced_deployment.assert_called_once()
        prov.apps_v1.delete_namespaced_deployment.assert_not_called()
        # Only the initial create attempt; no second create on the patch path.
        assert prov.apps_v1.create_namespaced_deployment.call_count == 1

    def _conflict_with_incumbent(self, prov, *, incumbent_labels, incumbent_annotations=None):
        # ⚠️ ``incumbent_annotations`` MUST be set explicitly, even to None. It was
        # previously left unset, so ``existing.metadata.annotations`` stayed an
        # auto-created MagicMock — truthy, and ``.get(...)`` on it returns another
        # truthy MagicMock. The moment the ownership gate started reading
        # annotations, that phantom read as a FOREIGN owner and refused the
        # OWNER's own reconcile. A MagicMock attribute is not "absent"; it is an
        # arbitrary value, and a harness that leaves a load-bearing field
        # auto-mocked tests the mock rather than the code.
        from kubernetes.client.rest import ApiException

        prov.apps_v1.create_namespaced_deployment.side_effect = ApiException(
            status=409, reason="Conflict"
        )
        existing = MagicMock()
        existing.metadata.deletion_timestamp = None
        existing.metadata.labels = incumbent_labels
        existing.metadata.annotations = incumbent_annotations
        prov.apps_v1.read_namespaced_deployment.return_value = existing
        return existing

    def test_409_refuses_to_take_over_another_tenants_deployment(self):
        """A name collision must NOT hand tenant B tenant A's workspace.

        This is the WRITE half of the cross-tenant defect. The PR comment on
        ``/status`` used to prescribe "the caller provisions its own workspace on
        the 404" as the remedy for a scoped 404 — but ``POST /`` is not
        name-gated, so on a COLLISION that remedy lands here, and this branch
        patches the incumbent with ``body=deployment`` built from the CALLER's
        labels. That re-stamps ``tenant-id`` to the caller and the changed
        provisioned-at pod-template annotation rolls a fresh pod (new emptyDir →
        re-clone), so the incumbent tenant's workspace is taken over AND its
        contents destroyed — silently, with a 200.

        The refusal says "not found" so the router maps it to 404 and the name is
        never confirmed to exist, matching the ``get_container`` non-disclosure
        rule.
        """
        prov = _make_provisioner()
        self._conflict_with_incumbent(
            prov, incumbent_labels={"managed-by": "code-graph-helper", "tenant-id": "tenant-a"}
        )

        with pytest.raises(ValueError, match="not found"):
            prov._create_deployment(
                name="shared-name",
                repo_url="https://github.com/owner/repo",
                branch="main",
                github_token="ghp_test",
                tenant_id="tenant-b",
            )

        # The takeover must not have happened at all — not patched, not deleted.
        prov.apps_v1.patch_namespaced_deployment.assert_not_called()
        prov.apps_v1.delete_namespaced_deployment.assert_not_called()

    def test_409_still_reconciles_for_the_owning_tenant(self):
        """POSITIVE CONTROL. Idempotent re-provision by the OWNER still patches.

        Without this, the guard above is satisfiable by refusing every 409, which
        would break the create-reuse path that the whole in-place reconcile exists
        to serve.
        """
        prov = _make_provisioner()
        self._conflict_with_incumbent(
            prov, incumbent_labels={"managed-by": "code-graph-helper", "tenant-id": "tenant-a"}
        )

        prov._create_deployment(
            name="shared-name",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
            tenant_id="tenant-a",
        )

        prov.apps_v1.patch_namespaced_deployment.assert_called_once()

    def test_409_refuses_an_incumbent_whose_owner_is_unrecoverable(self):
        """⚠️ INVERTED, deliberately. Was ``test_409_still_reconciles_an_unlabelled
        _incumbent``, which asserted the OPPOSITE — that an unlabelled incumbent
        is patched — on the reasoning that "the guard fires only on a MEASURED
        conflict: both sides carry a tenant and they differ", and that refusing
        would strand the ownerless population.

        That reasoning rested on a premise that is false: it read "no label" as
        PROOF of "no owner". The label is written only when the tenant satisfies
        ``_K8S_LABEL_VALUE_RE``; ``effective_tenant_id`` is
        ``tenant_id or organization_id or sub`` with no grammar validation. So an
        unlabelled Deployment is equally consistent with an OWNED workspace whose
        owner could not be expressed as a label — and for that workspace the old
        assertion was a cross-tenant TAKEOVER: ``body=deployment`` carries the
        caller's stamps, re-owning the workspace and rolling a fresh pod with a
        new emptyDir, destroying the incumbent's work and returning 200.

        Unknown-owner and ownerless are indistinguishable here and demand
        opposite answers, so the tie must break toward not destroying data. The
        availability cost was measured before it was accepted: live ``coding-jobs``
        on 2026-09-17 had 10 of 10 Deployments labelled, so nothing reachable today
        is stranded, and every provision from here on writes the annotation.
        """
        prov = _make_provisioner()
        self._conflict_with_incumbent(prov, incumbent_labels={"managed-by": "code-graph-helper"})

        with pytest.raises(ValueError, match="not found"):
            prov._create_deployment(
                name="legacy-name",
                repo_url="https://github.com/owner/repo",
                branch="main",
                github_token="ghp_test",
                tenant_id="tenant-b",
            )

        # The part that destroys data. Asserted explicitly: a refusal that still
        # patched would be a takeover that merely reported itself as an error.
        prov.apps_v1.patch_namespaced_deployment.assert_not_called()

    def test_409_refuses_takeover_when_owner_is_only_in_the_annotation(self):
        """⭐ THE DEFECT, on the write path. RED before this change.

        The incumbent's owner is grammar-invalid as a k8s label value — an
        ``organization_id``-shaped owner with a ``/`` and a ``:`` — so provisioning
        stamped the annotation and omitted the label. The guard used to read the
        claim out of ``labels`` ALONE, which is the very thing its own comment
        forbids three lines later for the CALLER's side: the incumbent presented
        as "no claim", the guard did not fire, and the patch re-stamped the
        workspace to the caller.
        """
        prov = _make_provisioner()
        self._conflict_with_incumbent(
            prov,
            incumbent_labels={"managed-by": "code-graph-helper"},  # unlabelable owner
            incumbent_annotations={"coding-job/tenant-id": "org:acme/platform-team"},
        )

        with pytest.raises(ValueError, match="not found"):
            prov._create_deployment(
                name="shared-name",
                repo_url="https://github.com/owner/repo",
                branch="main",
                github_token="ghp_test",
                tenant_id="tenant-b",
            )

        prov.apps_v1.patch_namespaced_deployment.assert_not_called()

    def test_409_reconciles_when_annotation_owner_matches_the_caller(self):
        """POSITIVE CONTROL for the test above.

        Without it, that assertion is satisfiable by a guard that refuses every
        annotation-owned incumbent — which would make grammar-invalid tenants
        unable to re-provision their own workspaces at all.
        """
        prov = _make_provisioner()
        self._conflict_with_incumbent(
            prov,
            incumbent_labels={"managed-by": "code-graph-helper"},
            incumbent_annotations={"coding-job/tenant-id": "org:acme/platform-team"},
        )

        prov._create_deployment(
            name="shared-name",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
            tenant_id="org:acme/platform-team",
        )

        prov.apps_v1.patch_namespaced_deployment.assert_called_once()

    def test_409_label_only_incumbent_still_resolves_backward_compat(self):
        """BACKWARD COMPATIBILITY, and it is not hypothetical.

        Measured live 2026-09-17 in ``coding-jobs``: 10 of 10 Deployments carry the
        ``tenant-id`` LABEL and 0 of 9 carry the annotation. If the annotation
        were read INSTEAD OF the label rather than BEFORE it, the entire existing
        population would resolve to "owner unknown" and this fail-closed change
        would take every live workspace offline. Both directions asserted here:
        the foreign caller is refused AND the owner is reconciled.
        """
        prov = _make_provisioner()
        self._conflict_with_incumbent(
            prov,
            incumbent_labels={"managed-by": "code-graph-helper", "tenant-id": "tenant-a"},
            incumbent_annotations={"coding-job/repo-url": "https://github.com/nonagenticai/x"},
        )

        with pytest.raises(ValueError, match="not found"):
            prov._create_deployment(
                name="shared-name",
                repo_url="https://github.com/owner/repo",
                branch="main",
                github_token="ghp_test",
                tenant_id="tenant-b",
            )
        prov.apps_v1.patch_namespaced_deployment.assert_not_called()

        prov._create_deployment(
            name="shared-name",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
            tenant_id="tenant-a",
        )
        prov.apps_v1.patch_namespaced_deployment.assert_called_once()

    def test_provision_stamps_the_tenant_as_an_annotation_when_unlabelable(self):
        """The other half: an owner that cannot be labelled must still be RECORDED.

        Without this stamp the recovery above has nothing to recover, and the
        workspace is born with no ownership record at all — which is the state
        that fails open on read and is taken over on write.
        """
        prov = _make_provisioner()
        unlabelable = "org:acme/platform-team"  # '/' and ':' both violate the label grammar

        prov._create_deployment(
            name="ws",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
            tenant_id=unlabelable,
        )

        body = prov.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]
        assert body.metadata.annotations["coding-job/tenant-id"] == unlabelable
        # Non-vacuity: the label genuinely could NOT carry this value, so the
        # annotation is the only record — if the label had it, this test would be
        # asserting a redundant stamp and would pass even with the fix reverted.
        assert "tenant-id" not in body.metadata.labels

    def test_provision_stamps_both_label_and_annotation_for_a_normal_tenant(self):
        """The label write is deliberately NOT weakened or removed — the existing
        population resolves by it, and a reader on an older build sees only it."""
        prov = _make_provisioner()

        prov._create_deployment(
            name="ws",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
            tenant_id="0195f48b-d000-7000-8000-000000000001",
        )

        body = prov.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]
        assert body.metadata.labels["tenant-id"] == "0195f48b-d000-7000-8000-000000000001"
        assert body.metadata.annotations["coding-job/tenant-id"] == (
            "0195f48b-d000-7000-8000-000000000001"
        )

    def test_recreates_on_409_when_terminating(self):
        """On 409 against a terminating Deployment, wait for deletion then create.

        Patching a terminating object is wrong (it is destined for deletion), so
        we poll until the GET 404s, then create fresh. No patch in this branch.
        """
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()

        conflict = ApiException(status=409, reason="Conflict")
        # First create 409s; the post-deletion create succeeds.
        prov.apps_v1.create_namespaced_deployment.side_effect = [conflict, None]

        terminating = MagicMock()
        terminating.metadata.deletion_timestamp = "2026-06-19T00:00:00Z"
        not_found = ApiException(status=404, reason="Not Found")
        # read() #1 → the deletion_timestamp check; read() #2 → _wait_until_deleted → 404.
        prov.apps_v1.read_namespaced_deployment.side_effect = [terminating, not_found]

        with patch.object(time_module, "sleep"):
            prov._create_deployment(
                name="test-deploy",
                repo_url="https://github.com/owner/repo",
                branch="main",
                github_token="ghp_test",
            )

        prov.apps_v1.patch_namespaced_deployment.assert_not_called()
        prov.apps_v1.delete_namespaced_deployment.assert_not_called()
        assert prov.apps_v1.create_namespaced_deployment.call_count == 2

    def test_spec_has_recreate_strategy_and_provisioned_at_annotation(self):
        """Reprovision correctness rests on Recreate + a changing template annotation."""
        prov = _make_provisioner()
        prov._create_deployment(
            name="recreate-pod",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
        )
        deployment = prov.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]
        assert deployment.spec.strategy.type == "Recreate"
        template_annotations = deployment.spec.template.metadata.annotations or {}
        assert "coding-job/provisioned-at" in template_annotations
        # Epoch string so any change rolls the pod-template hash → fresh pod.
        assert template_annotations["coding-job/provisioned-at"].isdigit()

    def test_stamps_durable_repo_metadata_annotations(self):
        """repo_url/branch/secret_config_id land on the Deployment meta as
        annotations so reconstruct_container_info can recover them on a
        registry-miss (URLs and refs can't be labels)."""
        prov = _make_provisioner()
        prov._create_deployment(
            name="meta-deploy",
            repo_url="https://github.com/nonagenticai/enterprise_mcp_server",
            branch="aa/arch-refactor-cycle-fixture",
            github_token="ghp_test",
            secret_config_id="sec-123",
        )
        deployment = prov.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]
        ann = deployment.metadata.annotations or {}
        assert ann["coding-job/repo-url"] == "https://github.com/nonagenticai/enterprise_mcp_server"
        assert ann["coding-job/branch"] == "aa/arch-refactor-cycle-fixture"
        assert ann["coding-job/secret-config-id"] == "sec-123"

    def test_omits_secret_config_annotation_when_absent(self):
        prov = _make_provisioner()
        prov._create_deployment(
            name="meta-deploy-nosecret",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
        )
        deployment = prov.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]
        ann = deployment.metadata.annotations or {}
        assert ann["coding-job/repo-url"] == "https://github.com/owner/repo"
        assert "coding-job/secret-config-id" not in ann

    def test_raises_on_other_api_errors(self):
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        prov.apps_v1.create_namespaced_deployment.side_effect = ApiException(
            status=500, reason="Internal"
        )

        with pytest.raises(ApiException):
            prov._create_deployment(
                name="test-deploy",
                repo_url="https://github.com/owner/repo",
                branch="main",
                github_token="ghp_test",
            )

    def test_pod_spec_is_thin_volume_holder(self):
        """The per-repo pod runs only a git-clone init + a sleep container.

        Regression test for the code_graph_helper retirement: no HTTP sidecar,
        no ports, no probes, no claude/codex credential mounts. Default image
        must be a Python-3 base (repo_scanner_modules writes a python -c
        script inside the pod and coding-security shells out `uv add`), not
        the retired code_graph_helper image.
        """
        prov = _make_provisioner()
        prov._create_deployment(
            name="thin-pod",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
        )
        deployment = prov.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]
        pod_spec = deployment.spec.template.spec

        assert len(pod_spec.containers) == 1
        analysis = pod_spec.containers[0]
        assert analysis.name == "analysis", "exec_in_pod targets container='analysis'"
        # Python-3 base, either upstream python:3.x-slim or our codegraph-runtime
        # (which COPYs uv onto python:3.14-slim).
        assert "python" in analysis.image or "codegraph-runtime" in analysis.image
        assert "code_graph_helper" not in analysis.image
        assert analysis.ports is None
        assert analysis.startup_probe is None
        assert analysis.liveness_probe is None
        assert analysis.readiness_probe is None
        assert analysis.command == ["/bin/sh", "-c"]
        assert analysis.args and "sleep" in analysis.args[0]

        mount_names = {m.name for m in (analysis.volume_mounts or [])}
        assert mount_names == {"repo-data", "tmp"}, (
            "claude-creds/codex-creds mounts were for the retired sidecar"
        )
        volume_names = {v.name for v in pod_spec.volumes}
        assert volume_names == {"repo-data", "tmp"}

    def test_pod_affinity_excludes_control_plane_and_keeps_arch_selector(self):
        """Pins the ⛔ control-plane exclusion added above the affinity= block.

        Asserts REQUIRED (not preferred) so a future refactor that downgrades
        this to a soft preference — which would look like a harmless
        tightening but guarantees nothing — fails here.

        The key/operator are asserted as literal strings, not "some affinity
        exists", because of this guard's specific failure mode: a typo'd key
        makes DoesNotExist true on every node (no node carries a misspelt
        label), so the predicate silently degrades to matching everywhere,
        including the control plane it exists to exclude.
        """
        prov = _make_provisioner()
        prov.node_arch = "amd64"  # exercise the pre-existing selector too

        prov._create_deployment(
            name="affinity-pod",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
        )
        deployment = prov.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]
        pod_spec = deployment.spec.template.spec

        # The pre-existing arch constraint must survive, not be replaced by
        # the new affinity.
        assert pod_spec.node_selector == {"kubernetes.io/arch": "amd64"}

        node_affinity = pod_spec.affinity.node_affinity
        required = node_affinity.required_during_scheduling_ignored_during_execution
        assert required is not None, "must be REQUIRED, not merely preferred"
        assert not node_affinity.preferred_during_scheduling_ignored_during_execution

        terms = required.node_selector_terms
        assert len(terms) == 1
        expressions = terms[0].match_expressions
        assert len(expressions) == 1
        assert expressions[0].key == "node-role.kubernetes.io/control-plane"
        assert expressions[0].operator == "DoesNotExist"


# ---------------------------------------------------------------------------
# _resolve_pod_from_deployment
# ---------------------------------------------------------------------------


class TestResolvePodFromDeployment:
    def test_returns_running_pod_name(self):
        prov = _make_provisioner()

        mock_dep = MagicMock()
        mock_dep.spec.selector.match_labels = {"app": "codegraph-repo", "repo": "test"}
        prov.apps_v1.read_namespaced_deployment.return_value = mock_dep

        mock_pod = MagicMock()
        mock_pod.status.phase = "Running"
        mock_pod.metadata.name = "test-pod-abc123"

        mock_pods = MagicMock()
        mock_pods.items = [mock_pod]
        prov.core_v1.list_namespaced_pod.return_value = mock_pods

        result = prov._resolve_pod_from_deployment("test-deploy")
        assert result == "test-pod-abc123"

    def test_waits_then_returns_pod_that_becomes_running(self):
        """The Recreate gap: first list has only a Pending pod, a later list
        has it Running — the resolver must bounded-wait and return it, not
        fail fast."""
        prov = _make_provisioner()

        mock_dep = MagicMock()
        mock_dep.spec.selector.match_labels = {"app": "codegraph-repo", "repo": "test"}
        prov.apps_v1.read_namespaced_deployment.return_value = mock_dep

        pending = MagicMock()
        pending.status.phase = "Pending"
        running = MagicMock()
        running.status.phase = "Running"
        running.metadata.name = "test-pod-late"

        first = MagicMock()
        first.items = [pending]
        second = MagicMock()
        second.items = [running]
        prov.core_v1.list_namespaced_pod.side_effect = [first, second]

        with patch(f"{MODULE}.time.sleep") as mock_sleep:
            result = prov._resolve_pod_from_deployment("test-deploy")

        assert result == "test-pod-late"
        assert prov.core_v1.list_namespaced_pod.call_count == 2
        mock_sleep.assert_called_once()

    def test_raises_after_bounded_wait_when_no_running_pods(self, monkeypatch):
        """No pod ever reaches Running → raise after a BOUNDED number of polls
        (proving it waited and retried, not a single fail-fast list)."""
        monkeypatch.setattr(kcp, "POD_RESOLVE_TIMEOUT_S", 3.0)
        monkeypatch.setattr(kcp, "POD_RESOLVE_INTERVAL_S", 1.0)
        prov = _make_provisioner()

        mock_dep = MagicMock()
        mock_dep.spec.selector.match_labels = {"app": "codegraph-repo"}
        prov.apps_v1.read_namespaced_deployment.return_value = mock_dep

        pending = MagicMock()
        pending.status.phase = "Pending"
        mock_pods = MagicMock()
        mock_pods.items = [pending]
        prov.core_v1.list_namespaced_pod.return_value = mock_pods

        with patch(f"{MODULE}.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="No running pod"):
                prov._resolve_pod_from_deployment("test-deploy")

        # 3s / 1s = 3 attempts → 3 lists, sleeping between (not after) each.
        assert prov.core_v1.list_namespaced_pod.call_count == 3
        assert mock_sleep.call_count == 2

    def test_raises_when_deployment_not_found(self):
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        prov.apps_v1.read_namespaced_deployment.side_effect = ApiException(
            status=404, reason="Not Found"
        )

        # A 404 must raise the DISTINGUISHABLE not-found subclass (so the exec
        # router can map it to an honest HTTP 404), while still being a
        # RuntimeError for legacy `except RuntimeError` sites.
        with patch(f"{MODULE}.time.sleep") as mock_sleep:
            with pytest.raises(kcp.ContainerDeploymentNotFoundError, match="not found"):
                prov._resolve_pod_from_deployment("missing-deploy")
        assert issubclass(kcp.ContainerDeploymentNotFoundError, RuntimeError)

        # Absent deployment must fast-fail — never poll for pods, never sleep.
        prov.core_v1.list_namespaced_pod.assert_not_called()
        mock_sleep.assert_not_called()

    def test_raises_generic_runtimeerror_on_non_404_api_error(self):
        """A non-404 API error (e.g. 500) is a real cluster failure — it must
        stay a generic RuntimeError, NOT the not-found subclass, so it keeps
        mapping to HTTP 500 upstream."""
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        prov.apps_v1.read_namespaced_deployment.side_effect = ApiException(
            status=500, reason="Internal Server Error"
        )

        with patch(f"{MODULE}.time.sleep"):
            with pytest.raises(RuntimeError) as exc_info:
                prov._resolve_pod_from_deployment("boom-deploy")
        assert not isinstance(exc_info.value, kcp.ContainerDeploymentNotFoundError)


# ---------------------------------------------------------------------------
# Cold-image-pull budget + terminal-state short-circuit
#
# Measured in-cluster 2026-08-02 (worker2, cold image cache): the
# codegraph-runtime pull alone took 1m35s and the pod reached Running ~125s
# after create, while ``create_container`` had already reported
# ``status: running``. The old 30s budget could not span that, so the first
# exec against a freshly provisioned container on a cold node failed with a
# 500. These pods are built from real attribute shapes (None / list of
# statuses), NOT bare MagicMocks — a MagicMock iterates as empty and would
# make the terminal-state scan pass vacuously.
# ---------------------------------------------------------------------------


def _pod(name, phase, *, init_waiting=None, waiting=None):
    """A pod stub with the attribute shapes the real V1PodStatus uses.

    ``init_waiting`` / ``waiting`` are ``(container_name, reason, message)``
    tuples describing a container parked in ``state.waiting``.
    """
    from types import SimpleNamespace

    def _status(spec):
        if spec is None:
            return None
        cname, reason, message = spec
        waiting_obj = SimpleNamespace(reason=reason, message=message)
        return [SimpleNamespace(name=cname, state=SimpleNamespace(waiting=waiting_obj))]

    return SimpleNamespace(
        metadata=SimpleNamespace(name=name),
        status=SimpleNamespace(
            phase=phase,
            init_container_statuses=_status(init_waiting),
            container_statuses=_status(waiting),
        ),
    )


class TestColdImagePullBudget:
    def test_default_budget_spans_a_measured_cold_pull(self):
        """Exact pin, not a floor. The observed cold start was ~125s; a budget
        that silently drifts back under it re-opens the exact 500 this fixes."""
        assert kcp.POD_RESOLVE_TIMEOUT_S == 180.0

    def test_slow_cold_start_past_the_old_30s_budget_still_resolves(self, monkeypatch):
        """A pod still pulling its image for 60 polls — past the old 30s
        budget — must resolve, not raise."""
        monkeypatch.setattr(kcp, "POD_RESOLVE_TIMEOUT_S", 180.0)
        monkeypatch.setattr(kcp, "POD_RESOLVE_INTERVAL_S", 1.0)
        prov = _make_provisioner()

        mock_dep = MagicMock()
        mock_dep.spec.selector.match_labels = {"app": "codegraph-repo"}
        prov.apps_v1.read_namespaced_deployment.return_value = mock_dep

        # 60 polls of "still pulling" (a NON-terminal reason), then Running.
        pulling = MagicMock()
        pulling.items = [_pod("p", "Pending", init_waiting=("git-clone", "PodInitializing", ""))]
        ready = MagicMock()
        ready.items = [_pod("p", "Running")]
        prov.core_v1.list_namespaced_pod.side_effect = [pulling] * 60 + [ready]

        with patch(f"{MODULE}.time.sleep"):
            assert prov._resolve_pod_from_deployment("test-deploy") == "p"
        assert prov.core_v1.list_namespaced_pod.call_count == 61

    def test_image_pull_backoff_short_circuits_instead_of_burning_the_budget(self):
        """k8s has given up on the pull — fail immediately with an actionable
        reason rather than waiting the full 180s to say 'no running pod'."""
        prov = _make_provisioner()

        mock_dep = MagicMock()
        mock_dep.spec.selector.match_labels = {"app": "codegraph-repo"}
        prov.apps_v1.read_namespaced_deployment.return_value = mock_dep

        pods = MagicMock()
        pods.items = [
            _pod(
                "bad-pod",
                "Pending",
                init_waiting=("git-clone", "ImagePullBackOff", "manifest unknown"),
            )
        ]
        prov.core_v1.list_namespaced_pod.return_value = pods

        with patch(f"{MODULE}.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="ImagePullBackOff"):
                prov._resolve_pod_from_deployment("test-deploy")

        # Bailed on the FIRST poll — no waiting at all.
        assert prov.core_v1.list_namespaced_pod.call_count == 1
        mock_sleep.assert_not_called()

    def test_crashloop_on_the_app_container_also_short_circuits(self):
        prov = _make_provisioner()
        mock_dep = MagicMock()
        mock_dep.spec.selector.match_labels = {"app": "codegraph-repo"}
        prov.apps_v1.read_namespaced_deployment.return_value = mock_dep

        pods = MagicMock()
        pods.items = [_pod("bad", "Pending", waiting=("analysis", "CrashLoopBackOff", "back-off"))]
        prov.core_v1.list_namespaced_pod.return_value = pods

        with patch(f"{MODULE}.time.sleep"):
            with pytest.raises(RuntimeError, match="CrashLoopBackOff"):
                prov._resolve_pod_from_deployment("test-deploy")

    def test_clone_crashloop_raises_the_typed_clone_failure(self):
        """A git-clone init container that keeps exiting non-zero (a BYO token that
        cannot read the repo, a missing branch) is the clone's verdict on access:
        a typed ``WorkspaceCloneFailedError``, not a generic RuntimeError."""
        prov = _make_provisioner()
        mock_dep = MagicMock()
        mock_dep.spec.selector.match_labels = {"app": "codegraph-repo"}
        prov.apps_v1.read_namespaced_deployment.return_value = mock_dep
        pods = MagicMock()
        pods.items = [
            _pod(
                "clone-pod",
                "Pending",
                init_waiting=("git-clone", "CrashLoopBackOff", "back-off 10s restarting"),
            )
        ]
        prov.core_v1.list_namespaced_pod.return_value = pods

        with patch(f"{MODULE}.time.sleep") as mock_sleep:
            with pytest.raises(kcp.WorkspaceCloneFailedError) as info:
                prov._resolve_pod_from_deployment("test-deploy")
        mock_sleep.assert_not_called()
        assert isinstance(info.value, RuntimeError)  # existing except sites still hold
        assert info.value.failure.startswith("git-clone: CrashLoopBackOff")
        assert info.value.deployment_name == "test-deploy"
        assert info.value.pod_name == "clone-pod"

    def test_clone_image_pull_failure_is_not_a_clone_failure(self):
        """Control: an image problem on the same container is ours, not the caller's."""
        prov = _make_provisioner()
        mock_dep = MagicMock()
        mock_dep.spec.selector.match_labels = {"app": "codegraph-repo"}
        prov.apps_v1.read_namespaced_deployment.return_value = mock_dep
        pods = MagicMock()
        pods.items = [_pod("p", "Pending", init_waiting=("git-clone", "ImagePullBackOff", ""))]
        prov.core_v1.list_namespaced_pod.return_value = pods
        with patch(f"{MODULE}.time.sleep"):
            with pytest.raises(RuntimeError) as info:
                prov._resolve_pod_from_deployment("test-deploy")
        assert not isinstance(info.value, kcp.WorkspaceCloneFailedError)

    def test_a_running_pod_wins_over_a_terminal_sibling(self):
        """False-positive control. During a Recreate the old pod can be in
        CrashLoopBackOff while the new one is already Running — the healthy pod
        must be returned, NOT an exception."""
        prov = _make_provisioner()
        mock_dep = MagicMock()
        mock_dep.spec.selector.match_labels = {"app": "codegraph-repo"}
        prov.apps_v1.read_namespaced_deployment.return_value = mock_dep

        pods = MagicMock()
        pods.items = [
            _pod("old", "Pending", waiting=("analysis", "CrashLoopBackOff", "back-off")),
            _pod("new", "Running"),
        ]
        prov.core_v1.list_namespaced_pod.return_value = pods

        with patch(f"{MODULE}.time.sleep"):
            assert prov._resolve_pod_from_deployment("test-deploy") == "new"

    def test_non_terminal_waiting_reasons_are_not_treated_as_failures(self):
        """ContainerCreating / PodInitializing / Pulling are the NORMAL cold-start
        reasons. Mistaking any of them for terminal would reintroduce the bug."""
        prov = _make_provisioner()
        for reason in ("ContainerCreating", "PodInitializing", "Pulling"):
            pod = _pod("p", "Pending", init_waiting=("git-clone", reason, ""))
            assert prov._terminal_pod_failure(pod) is None, reason

    def test_pod_with_no_container_statuses_is_not_terminal(self):
        """A just-scheduled pod has None for both status lists."""
        prov = _make_provisioner()
        assert prov._terminal_pod_failure(_pod("p", "Pending")) is None


# ---------------------------------------------------------------------------
# exec_in_pod
# ---------------------------------------------------------------------------


class TestExecInPod:
    def test_successful_execution(self):
        prov = _make_provisioner()

        with patch.object(prov, "_resolve_pod_from_deployment", return_value="test-pod"):
            with patch.object(_mod, "k8s_stream") as mock_stream:
                mock_resp = MagicMock()
                mock_resp.is_open.side_effect = [True, False]
                mock_resp.peek_stdout.return_value = True
                mock_resp.read_stdout.return_value = "hello world"
                mock_resp.peek_stderr.return_value = False
                mock_resp.returncode = 0

                mock_stream.return_value = mock_resp

                result = prov.exec_in_pod(
                    deployment_name="test-deploy",
                    command=["echo", "hello"],
                )

                assert result["exit_code"] == 0
                assert "hello world" in result["stdout"]
                assert "duration" in result

    def test_execution_failure_returns_error(self):
        prov = _make_provisioner()

        with patch.object(prov, "_resolve_pod_from_deployment", return_value="test-pod"):
            with patch.object(_mod, "k8s_stream") as mock_stream:
                mock_stream.side_effect = RuntimeError("stream failed")

                result = prov.exec_in_pod(
                    deployment_name="test-deploy",
                    command=["bad-cmd"],
                )

                assert result["exit_code"] == -1
                assert "stream failed" in result["stderr"]

    def test_masked_attributeerror_returns_error_dict(self):
        """kubernetes-client re-raises an ApiException(body=None) as AttributeError
        ('NoneType' has no attribute 'decode'). exec_in_pod must return the ordinary
        error dict with a NUMERIC exit code, not let the crash escape as a 500 with no
        exit code -- that is exactly what produced `dispatch_failed` on a live card."""
        prov = _make_provisioner()

        with patch.object(prov, "_resolve_pod_from_deployment", return_value="test-pod"):
            with patch.object(_mod, "k8s_stream") as mock_stream:
                mock_stream.side_effect = AttributeError(
                    "'NoneType' object has no attribute 'decode'"
                )

                result = prov.exec_in_pod(
                    deployment_name="test-deploy",
                    command=["python", "-m", "checks.whatever"],
                )

                assert result["exit_code"] == -1
                assert "decode" in result["stderr"]

    def test_exit_code_from_error_channel_success(self):
        prov = _make_provisioner()

        with patch.object(prov, "_resolve_pod_from_deployment", return_value="test-pod"):
            with patch.object(_mod, "k8s_stream") as mock_stream:
                mock_resp = MagicMock(spec=[])  # bare spec so hasattr(returncode) is False
                mock_resp.is_open = MagicMock(return_value=False)
                mock_resp.close = MagicMock()
                mock_resp.read_channel = MagicMock(return_value=json.dumps({"status": "Success"}))

                mock_stream.return_value = mock_resp

                result = prov.exec_in_pod(
                    deployment_name="test-deploy",
                    command=["echo", "test"],
                )

                assert result["exit_code"] == 0

    def test_exit_code_from_error_channel_with_cause(self):
        prov = _make_provisioner()

        with patch.object(prov, "_resolve_pod_from_deployment", return_value="test-pod"):
            with patch.object(_mod, "k8s_stream") as mock_stream:
                mock_resp = MagicMock(spec=[])
                mock_resp.is_open = MagicMock(return_value=False)
                mock_resp.close = MagicMock()
                err_data = {
                    "status": "Failure",
                    "details": {"causes": [{"message": "2"}]},
                }
                mock_resp.read_channel = MagicMock(return_value=json.dumps(err_data))

                mock_stream.return_value = mock_resp

                result = prov.exec_in_pod(
                    deployment_name="test-deploy",
                    command=["false"],
                )

                assert result["exit_code"] == 2


# ---------------------------------------------------------------------------
# _cleanup_resources
# ---------------------------------------------------------------------------


class TestCleanupResources:
    def test_deletes_deployment_and_service(self):
        prov = _make_provisioner()
        prov._cleanup_resources("test-deploy")

        prov.apps_v1.delete_namespaced_deployment.assert_called_once()
        prov.core_v1.delete_namespaced_service.assert_called_once()

    def test_ignores_404_on_deployment(self):
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        prov.apps_v1.delete_namespaced_deployment.side_effect = ApiException(
            status=404, reason="Not Found"
        )

        prov._cleanup_resources("test-deploy")
        prov.core_v1.delete_namespaced_service.assert_called_once()

    def test_ignores_404_on_service(self):
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        prov.core_v1.delete_namespaced_service.side_effect = ApiException(
            status=404, reason="Not Found"
        )

        prov._cleanup_resources("test-deploy")


# ---------------------------------------------------------------------------
# remove_container
# ---------------------------------------------------------------------------


class TestRemoveContainer:
    @pytest.mark.asyncio
    async def test_calls_cleanup(self):
        prov = _make_provisioner()
        with patch.object(prov, "_cleanup_resources") as mock_cleanup:
            await prov.remove_container("test-ctr")
            mock_cleanup.assert_called_once_with("test-ctr")


# ---------------------------------------------------------------------------
# get_container_status
# ---------------------------------------------------------------------------


class TestGetContainerStatus:
    @pytest.mark.asyncio
    async def test_running_status(self):
        prov = _make_provisioner()
        mock_dep = MagicMock()
        mock_dep.status.ready_replicas = 1
        mock_dep.spec.replicas = 1
        mock_dep.status.conditions = None
        prov.apps_v1.read_namespaced_deployment.return_value = mock_dep

        result = await prov.get_container_status("test-ctr")
        assert result["status"] == "running"
        assert result["ready_replicas"] == 1

    @pytest.mark.asyncio
    async def test_starting_status(self):
        prov = _make_provisioner()
        mock_dep = MagicMock()
        mock_dep.status.ready_replicas = 0
        mock_dep.spec.replicas = 1
        mock_dep.status.conditions = [MagicMock()]
        prov.apps_v1.read_namespaced_deployment.return_value = mock_dep

        result = await prov.get_container_status("test-ctr")
        assert result["status"] == "starting"

    @pytest.mark.asyncio
    async def test_pending_status(self):
        prov = _make_provisioner()
        mock_dep = MagicMock()
        mock_dep.status.ready_replicas = None
        mock_dep.spec.replicas = 1
        mock_dep.status.conditions = None
        prov.apps_v1.read_namespaced_deployment.return_value = mock_dep

        result = await prov.get_container_status("test-ctr")
        assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_not_found(self):
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        prov.apps_v1.read_namespaced_deployment.side_effect = ApiException(
            status=404, reason="Not Found"
        )

        result = await prov.get_container_status("missing")
        assert result["status"] == "not_found"


# ---------------------------------------------------------------------------
# list_containers
# ---------------------------------------------------------------------------


class TestListContainers:
    @pytest.mark.asyncio
    async def test_lists_deployments(self):
        prov = _make_provisioner()

        mock_dep = MagicMock()
        mock_dep.metadata.name = "ctr-1"
        mock_dep.status.ready_replicas = 1
        mock_dep.spec.replicas = 1
        mock_dep.metadata.creation_timestamp = MagicMock()
        mock_dep.metadata.creation_timestamp.isoformat.return_value = "2025-01-01T00:00:00Z"

        mock_list = MagicMock()
        mock_list.items = [mock_dep]
        prov.apps_v1.list_namespaced_deployment.return_value = mock_list

        result = await prov.list_containers()
        assert len(result) == 1
        assert result[0]["name"] == "ctr-1"
        assert result[0]["status"] == "running"

    @pytest.mark.asyncio
    async def test_starting_deployment(self):
        prov = _make_provisioner()

        mock_dep = MagicMock()
        mock_dep.metadata.name = "ctr-2"
        mock_dep.status.ready_replicas = 0
        mock_dep.spec.replicas = 1
        mock_dep.metadata.creation_timestamp = None

        mock_list = MagicMock()
        mock_list.items = [mock_dep]
        prov.apps_v1.list_namespaced_deployment.return_value = mock_list

        result = await prov.list_containers()
        assert result[0]["status"] == "starting"
        assert result[0]["created"] is None

    @pytest.mark.asyncio
    async def test_api_exception_raises(self):
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        prov.apps_v1.list_namespaced_deployment.side_effect = ApiException(
            status=500, reason="Internal"
        )

        with pytest.raises(ApiException):
            await prov.list_containers()


# ---------------------------------------------------------------------------
# get_container_repository_tree
# ---------------------------------------------------------------------------


class TestGetContainerRepositoryTree:
    @pytest.mark.asyncio
    async def test_builds_tree(self):
        prov = _make_provisioner()

        with patch.object(
            prov,
            "exec_in_pod",
            return_value={
                "exit_code": 0,
                "stdout": "/workspace\n/workspace/src\n/workspace/src/main.py\n/workspace/README.md\n",
                "stderr": "",
            },
        ):
            result = await prov.get_container_repository_tree("test-deploy")

        assert "root" in result
        tree = result["root"]
        assert tree["name"] == "repo"
        assert "src" in tree["children"]
        assert "README.md" in tree["children"]

    @pytest.mark.asyncio
    async def test_raises_on_failure(self):
        prov = _make_provisioner()

        with patch.object(
            prov,
            "exec_in_pod",
            return_value={"exit_code": 1, "stdout": "", "stderr": "find: error"},
        ):
            with pytest.raises(RuntimeError, match="Failed to get repository tree"):
                await prov.get_container_repository_tree("test-deploy")


# ---------------------------------------------------------------------------
# _add_to_tree (static method)
# ---------------------------------------------------------------------------


class TestAddToTree:
    def test_adds_file(self):
        tree = {"type": "directory", "name": "repo", "children": {}}
        K8sContainerProvisioner._add_to_tree(tree, "src/main.py")

        assert "src" in tree["children"]
        assert "main.py" in tree["children"]["src"]["children"]
        assert tree["children"]["src"]["children"]["main.py"]["type"] == "file"

    def test_adds_directory(self):
        tree = {"type": "directory", "name": "repo", "children": {}}
        K8sContainerProvisioner._add_to_tree(tree, "src/utils")

        assert "src" in tree["children"]
        assert "utils" in tree["children"]["src"]["children"]
        assert tree["children"]["src"]["children"]["utils"]["type"] == "directory"

    def test_nested_paths(self):
        tree = {"type": "directory", "name": "repo", "children": {}}
        K8sContainerProvisioner._add_to_tree(tree, "a/b/c/file.txt")

        node = tree["children"]["a"]["children"]["b"]["children"]["c"]["children"]
        assert "file.txt" in node
        assert node["file.txt"]["type"] == "file"


# ---------------------------------------------------------------------------
# get_k8s_container_provisioner singleton
# ---------------------------------------------------------------------------


class TestGetK8sContainerProvisionerSingleton:
    def test_returns_singleton(self):
        _mod._k8s_provisioner = None

        with patch("kubernetes.config.load_kube_config"):
            p1 = get_k8s_container_provisioner()
            p2 = get_k8s_container_provisioner()
            assert p1 is p2

        _mod._k8s_provisioner = None  # cleanup


# ---------------------------------------------------------------------------
# provision_from_github
# ---------------------------------------------------------------------------


@pytest.fixture
def owner_is_platform(monkeypatch):
    """Configure the placeholder org ``owner`` as a PLATFORM owner.

    These tests exercise the platform rung of the clone ladder (App token only;
    the system PAT is never a fallback). Customer-owner behaviour — where
    that rung must never fire — is covered in tests/unit/test_tenant_bound_clone.py.
    """
    import src.config.settings as settings_mod

    monkeypatch.setenv("CODING_CONTAINERS_PLATFORM_GITHUB_OWNERS", "nonagenticai,owner")
    settings_mod._settings_instance = None
    yield
    settings_mod._settings_instance = None


@pytest.mark.usefixtures("owner_is_platform")
class TestProvisionFromGithub:
    @pytest.mark.asyncio
    async def test_provision_success(self):
        from src.core.container_spec import GitHubRepoRequest

        prov = _make_provisioner()

        mock_token_mgr = _mock_token_manager(
            system_token="ghp_systemtoken", secret_config_id="sec-id-123"
        )

        mock_settings = MagicMock()
        mock_settings.main_api_endpoint = "http://localhost:8687"

        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/myrepo",
            branch="main",
            access_token="ghp_usertoken",
            user_id="user-1",
            description="Test provision",
        )

        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_github_token_manager",
                return_value=mock_token_mgr,
            ),
            patch(
                "src.core.kubernetes_container_provisioner.get_settings", return_value=mock_settings
            ),
            patch.object(prov, "_create_deployment"),
            patch(
                "src.core.kubernetes_container_provisioner.ContainerMetadataRegistry"
            ) as mock_reg_cls,
        ):
            mock_reg = MagicMock()
            mock_reg.load.return_value = {}
            mock_reg_cls.return_value = mock_reg

            result = await prov.provision_from_github(request)

        assert result.status.value == "running"
        assert result.repo_url == "https://github.com/owner/myrepo"
        assert result.user_id == "user-1"
        assert result.original_repo_owner == "owner"
        assert result.original_repo_name == "myrepo"
        assert result.system_fork_owner == "sys-owner"

    @pytest.mark.asyncio
    async def test_provision_prefers_user_token_over_system_for_clone(self):
        """User-supplied PAT must be what the init container clones with.

        Previously the K8s path ignored request.access_token and always used
        the shared system_token, which for private repos of users the
        system account doesn't have access to → git clone 401.
        """
        from src.core.container_spec import GitHubRepoRequest

        prov = _make_provisioner()

        mock_token_mgr = _mock_token_manager(system_token="ghp_systemtoken")

        mock_settings = MagicMock()
        mock_settings.main_api_endpoint = "http://localhost:8687"

        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/privaterepo",
            branch="main",
            access_token="ghp_usertoken",
        )

        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_github_token_manager",
                return_value=mock_token_mgr,
            ),
            patch(
                "src.core.kubernetes_container_provisioner.get_settings", return_value=mock_settings
            ),
            patch.object(prov, "_create_deployment") as deploy_mock,
            patch(
                "src.core.kubernetes_container_provisioner.ContainerMetadataRegistry"
            ) as mock_reg_cls,
        ):
            mock_reg = MagicMock()
            mock_reg.load.return_value = {}
            mock_reg_cls.return_value = mock_reg
            await prov.provision_from_github(request)

        kwargs = deploy_mock.call_args.kwargs
        assert kwargs["github_token"] == "ghp_usertoken"

    @pytest.mark.asyncio
    async def test_provision_never_falls_back_to_system_token_when_no_user_token(self):
        """No BYO token and no App token: 412, never the shared system token."""
        from src.core.container_spec import GitHubRepoRequest

        prov = _make_provisioner()

        mock_token_mgr = _mock_token_manager(
            system_token="ghp_systemfallback", secret_config_id=None
        )

        mock_settings = MagicMock()
        mock_settings.main_api_endpoint = "http://localhost:8687"

        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/pubrepo",
            branch="main",
            tenant_id="tenant-1",
        )

        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_github_token_manager",
                return_value=mock_token_mgr,
            ),
            patch(
                "src.core.kubernetes_container_provisioner.get_settings", return_value=mock_settings
            ),
            patch.object(prov, "_create_deployment") as deploy_mock,
            patch(
                "src.core.kubernetes_container_provisioner.ContainerMetadataRegistry"
            ) as mock_reg_cls,
        ):
            mock_reg = MagicMock()
            mock_reg.load.return_value = {}
            mock_reg_cls.return_value = mock_reg
            with pytest.raises(kcp.GitCredentialUnavailableError) as exc_info:
                await prov.provision_from_github(request)

        assert exc_info.value.reason == "platform_credential_unavailable"
        deploy_mock.assert_not_called()
        assert not hasattr(mock_token_mgr, "get_system_token")

    @pytest.mark.asyncio
    async def test_provision_without_user_token(self):
        from src.core.container_spec import GitHubRepoRequest

        prov = _make_provisioner()

        mock_token_mgr = _mock_token_manager(
            credential=_app_credential(**_APP_OK), secret_config_id=None
        )

        mock_settings = MagicMock()
        mock_settings.main_api_endpoint = "http://localhost:8687"

        request = GitHubRepoRequest(
            tenant_id="tenant-1",
            repo_url="https://github.com/owner/myrepo.git",
            branch="develop",
        )

        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_github_token_manager",
                return_value=mock_token_mgr,
            ),
            patch(
                "src.core.kubernetes_container_provisioner.get_settings", return_value=mock_settings
            ),
            patch.object(prov, "_create_deployment"),
            patch(
                "src.core.kubernetes_container_provisioner.ContainerMetadataRegistry"
            ) as mock_reg_cls,
        ):
            mock_reg = MagicMock()
            mock_reg.load.return_value = {}
            mock_reg_cls.return_value = mock_reg

            result = await prov.provision_from_github(request)

        assert result.branch == "develop"
        # No inline PAT and no caller-supplied id ⇒ nothing to persist. The
        # precedence method is still consulted (it owns that decision); what must
        # NOT happen is a store, and no id may be invented for ContainerInfo.
        mock_token_mgr.store_user_token.assert_not_called()
        assert result.secret_config_id is None
        assert result.user_token_id is None

    @pytest.mark.asyncio
    async def test_provision_failure_triggers_cleanup(self):
        from src.core.container_spec import GitHubRepoRequest

        prov = _make_provisioner()

        def _created_then_raise(*_args, scope=None, **_kwargs):
            # Models a create that RETURNED, then a later step failing.
            scope.owned = True
            raise RuntimeError("k8s down")

        mock_token_mgr = _mock_token_manager(credential=_app_credential(**_APP_OK))

        mock_settings = MagicMock()
        mock_settings.main_api_endpoint = "http://localhost:8687"

        request = GitHubRepoRequest(
            tenant_id="tenant-1",
            repo_url="https://github.com/owner/myrepo",
            branch="main",
        )

        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_github_token_manager",
                return_value=mock_token_mgr,
            ),
            patch(
                "src.core.kubernetes_container_provisioner.get_settings", return_value=mock_settings
            ),
            patch.object(prov, "_create_deployment", side_effect=_created_then_raise),
            patch.object(prov, "_cleanup_resources") as mock_cleanup,
        ):
            with pytest.raises(RuntimeError, match="k8s down"):
                await prov.provision_from_github(request)

            mock_cleanup.assert_called_once()

        # #160: the same failure WITHOUT a create having returned (scope never
        # owned) must not clean up — the name may belong to another tenant.
        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_github_token_manager",
                return_value=mock_token_mgr,
            ),
            patch(
                "src.core.kubernetes_container_provisioner.get_settings", return_value=mock_settings
            ),
            patch.object(prov, "_create_deployment", side_effect=RuntimeError("k8s down")),
            patch.object(prov, "_cleanup_resources") as mock_cleanup,
        ):
            with pytest.raises(RuntimeError, match="k8s down"):
                await prov.provision_from_github(request)

            mock_cleanup.assert_not_called()

    @pytest.mark.asyncio
    async def test_provision_registry_failure_non_fatal(self):
        from src.core.container_spec import GitHubRepoRequest

        prov = _make_provisioner()

        mock_token_mgr = _mock_token_manager(credential=_app_credential(**_APP_OK))

        mock_settings = MagicMock()
        mock_settings.main_api_endpoint = "http://localhost:8687"

        request = GitHubRepoRequest(
            tenant_id="tenant-1",
            repo_url="https://github.com/owner/myrepo",
            branch="main",
        )

        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_github_token_manager",
                return_value=mock_token_mgr,
            ),
            patch(
                "src.core.kubernetes_container_provisioner.get_settings", return_value=mock_settings
            ),
            patch.object(prov, "_create_deployment"),
            patch(
                "src.core.kubernetes_container_provisioner.ContainerMetadataRegistry"
            ) as mock_reg_cls,
        ):
            mock_reg = MagicMock()
            mock_reg.load.side_effect = RuntimeError("registry broken")
            mock_reg_cls.return_value = mock_reg

            # Should succeed even when registry fails
            result = await prov.provision_from_github(request)
            assert result is not None

    @pytest.mark.asyncio
    async def test_provision_honors_request_container_name(self):
        # Regression: provision used to unconditionally generate
        # codegraph-<repo>-<6hex>, silently dropping the caller-supplied
        # container_name. n8n cells + smoke harnesses depend on a
        # deterministic name for cross-step lookups.
        from src.core.container_spec import GitHubRepoRequest

        prov = _make_provisioner()

        mock_token_mgr = _mock_token_manager(
            credential=_app_credential(**_APP_OK), system_token="ghp_systemtoken"
        )

        mock_settings = MagicMock()
        mock_settings.main_api_endpoint = "http://localhost:8687"

        request = GitHubRepoRequest(
            tenant_id="tenant-1",
            repo_url="https://github.com/owner/myrepo",
            branch="main",
            container_name="enterprise-mcp-server-aa",
        )

        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_github_token_manager",
                return_value=mock_token_mgr,
            ),
            patch(
                "src.core.kubernetes_container_provisioner.get_settings", return_value=mock_settings
            ),
            patch.object(prov, "_create_deployment") as mock_create,
            patch(
                "src.core.kubernetes_container_provisioner.ContainerMetadataRegistry"
            ) as mock_reg_cls,
        ):
            mock_reg = MagicMock()
            mock_reg.load.return_value = {}
            mock_reg_cls.return_value = mock_reg

            result = await prov.provision_from_github(request)

        assert result.container_name == "enterprise-mcp-server-aa"
        # _create_deployment must be called with the supplied name, not a slug.
        kwargs = mock_create.call_args.kwargs
        assert kwargs["name"] == "enterprise-mcp-server-aa"

    @pytest.mark.parametrize("history_cutoff", [True, False])
    @pytest.mark.asyncio
    async def test_provision_threads_history_cutoff_and_discloses_it(self, history_cutoff):
        """The flag must reach _create_deployment (which builds the clone command) AND be
        disclosed on the returned ContainerInfo — a request that silently produced a plain
        SHA checkout would read as isolated while being nothing of the kind."""
        from src.core.container_spec import GitHubRepoRequest

        prov = _make_provisioner()
        mock_token_mgr = _mock_token_manager(
            credential=_app_credential(**_APP_OK), system_token="ghp_systemtoken"
        )
        mock_settings = MagicMock()
        mock_settings.main_api_endpoint = "http://localhost:8687"
        sha = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
        request = GitHubRepoRequest(
            tenant_id="tenant-1",
            repo_url="https://github.com/owner/myrepo",
            branch=sha,
            container_name="bench-iso",
            history_cutoff=history_cutoff,
        )

        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_github_token_manager",
                return_value=mock_token_mgr,
            ),
            patch(
                "src.core.kubernetes_container_provisioner.get_settings", return_value=mock_settings
            ),
            patch.object(prov, "_create_deployment") as mock_create,
            patch(
                "src.core.kubernetes_container_provisioner.ContainerMetadataRegistry"
            ) as mock_reg_cls,
        ):
            mock_reg = MagicMock()
            mock_reg.load.return_value = {}
            mock_reg_cls.return_value = mock_reg

            result = await prov.provision_from_github(request)

        kwargs = mock_create.call_args.kwargs
        assert kwargs["history_cutoff"] is history_cutoff
        assert kwargs["branch"] == sha
        assert result.history_cutoff_applied is history_cutoff

    @pytest.mark.asyncio
    async def test_provision_container_rebuild_carries_history_cutoff(self):
        """provision_container REBUILDS the request from kwargs; a kwarg it does not
        accept is dropped (that is how the PR checkout was lost). The flag must survive."""
        prov = _make_provisioner()
        sha = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
        with patch.object(prov, "provision_from_github") as mock_pfg:
            mock_pfg.return_value = MagicMock()
            await prov.provision_container(
                container_name="bench-iso",
                repo_url="https://github.com/owner/myrepo",
                branch=sha,
                history_cutoff=True,
            )
        rebuilt = mock_pfg.call_args.args[0]
        assert rebuilt.history_cutoff is True
        assert rebuilt.branch == sha


# ---------------------------------------------------------------------------
# provision_from_github rollback must only reclaim what THIS call created
# (coding-containers#160)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("owner_is_platform")
class TestProvisionRollbackNeverDeletesAForeignWorkspace:
    """``POST /`` is not name-gated: ``docker_service.find_container_by_name``
    returns ``None`` for a name owned by another tenant, so tenant B's request
    for tenant A's name falls through to ``provision_from_github``. The
    ownership refusal in ``_create_deployment`` is correct — but the error-path
    rollback used to ``_cleanup_resources(deployment_name)`` on ANY failure,
    deleting the incumbent's Deployment and git-token Secret by name.
    """

    @staticmethod
    def _request(**overrides):
        from src.core.container_spec import GitHubRepoRequest

        fields = {
            "repo_url": "https://github.com/owner/repo",
            "branch": "main",
            "container_name": "shared-name",
            "access_token": "ghp_byo",
            "tenant_id": "tenant-b",
        }
        fields.update(overrides)
        return GitHubRepoRequest(**fields)

    @staticmethod
    def _incumbent_owned_by_tenant_a(prov):
        from kubernetes.client.rest import ApiException

        prov.apps_v1.create_namespaced_deployment.side_effect = ApiException(
            status=409, reason="Conflict"
        )
        existing = MagicMock()
        existing.metadata.deletion_timestamp = None
        existing.metadata.labels = {"managed-by": "code-graph-helper", "tenant-id": "tenant-a"}
        existing.metadata.annotations = {"coding-job/tenant-id": "tenant-a"}
        prov.apps_v1.read_namespaced_deployment.return_value = existing

    async def _provision(self, prov, request, token_mgr):
        mock_settings = MagicMock()
        mock_settings.main_api_endpoint = "http://localhost:8687"
        with (
            patch(f"{MODULE}.get_github_token_manager", return_value=token_mgr),
            patch(f"{MODULE}.get_settings", return_value=mock_settings),
            patch(f"{MODULE}.ContainerMetadataRegistry"),
        ):
            return await prov.provision_from_github(request)

    @pytest.mark.asyncio
    async def test_ownership_refusal_does_not_delete_the_other_tenants_workspace(self):
        prov = _make_provisioner()
        self._incumbent_owned_by_tenant_a(prov)

        with pytest.raises(
            ValueError,
            match=r"^Container 'shared-name' not found or not available for provisioning$",
        ):
            await self._provision(prov, self._request(), _mock_token_manager())

        # The refusal path was genuinely reached (not an earlier failure).
        prov.apps_v1.read_namespaced_deployment.assert_called_once()
        prov.apps_v1.patch_namespaced_deployment.assert_not_called()
        # ...and tenant A's workspace survives it: nothing deleted by name.
        prov.apps_v1.delete_namespaced_deployment.assert_not_called()
        prov.core_v1.delete_namespaced_secret.assert_not_called()
        prov.core_v1.delete_namespaced_service.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_before_any_create_does_not_delete_a_same_named_workspace(self):
        """Same flaw, earlier trigger: a RuntimeError from storing the BYO PAT
        (secrets service down) fires before ``_create_deployment`` ever runs, so
        this call created nothing — yet the old rollback deleted ``shared-name``."""
        prov = _make_provisioner()
        token_mgr = _mock_token_manager()
        token_mgr.resolve_provision_secret_config_id.side_effect = RuntimeError("secrets down")

        with pytest.raises(RuntimeError, match="secrets down"):
            await self._provision(prov, self._request(), token_mgr)

        prov.apps_v1.create_namespaced_deployment.assert_not_called()
        prov.apps_v1.delete_namespaced_deployment.assert_not_called()
        prov.core_v1.delete_namespaced_secret.assert_not_called()

    @staticmethod
    def _assert_nothing_deleted(prov):
        prov.apps_v1.delete_namespaced_deployment.assert_not_called()
        prov.core_v1.delete_namespaced_secret.assert_not_called()
        prov.core_v1.delete_namespaced_service.assert_not_called()

    @staticmethod
    def _assert_rolled_back(prov):
        prov.apps_v1.delete_namespaced_deployment.assert_called_once()
        assert prov.apps_v1.delete_namespaced_deployment.call_args.kwargs["name"] == "shared-name"
        prov.core_v1.delete_namespaced_secret.assert_called_once()
        assert (
            prov.core_v1.delete_namespaced_secret.call_args.kwargs["name"]
            == "shared-name-git-token"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [500, 429, 422, 403])
    async def test_non_409_create_error_does_not_delete_a_foreign_workspace(self, status):
        """Admission (403), validation (422) and throttling (429) are decided
        BEFORE the apiserver's exists-check, and a 5xx says nothing about who owns
        the name — so tenant A's same-named Deployment can be live while B's
        create fails with any of these. The create never returned, so this call
        owns nothing and must delete nothing."""
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        self._incumbent_owned_by_tenant_a(prov)
        boom = ApiException(status=status, reason="nope")
        prov.apps_v1.create_namespaced_deployment.side_effect = boom

        with pytest.raises(ApiException) as excinfo:
            await self._provision(prov, self._request(), _mock_token_manager())

        assert excinfo.value is boom
        prov.apps_v1.create_namespaced_deployment.assert_called_once()
        prov.apps_v1.read_namespaced_deployment.assert_not_called()
        self._assert_nothing_deleted(prov)

    @pytest.mark.asyncio
    async def test_failed_ownership_read_after_409_does_not_delete_a_foreign_workspace(self):
        """409, then the ownership read itself fails (503): ownership was never
        proven, so the rollback must not reclaim the name."""
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        self._incumbent_owned_by_tenant_a(prov)
        boom = ApiException(status=503, reason="Service Unavailable")
        prov.apps_v1.read_namespaced_deployment.return_value = None
        prov.apps_v1.read_namespaced_deployment.side_effect = boom

        with pytest.raises(ApiException) as excinfo:
            await self._provision(prov, self._request(), _mock_token_manager())

        assert excinfo.value is boom
        prov.apps_v1.read_namespaced_deployment.assert_called_once()
        prov.apps_v1.patch_namespaced_deployment.assert_not_called()
        self._assert_nothing_deleted(prov)

    @pytest.mark.asyncio
    async def test_tenantless_caller_patch_failure_does_not_delete_the_incumbent(self):
        """A caller with NO tenant still reaches the patch (pre-existing gate
        behaviour), but it has proven no ownership — a patch failure must not
        reclaim tenant A's workspace by name."""
        prov = _make_provisioner()
        self._incumbent_owned_by_tenant_a(prov)
        prov.apps_v1.patch_namespaced_deployment.side_effect = RuntimeError("patch failed")

        with pytest.raises(RuntimeError, match="patch failed"):
            await self._provision(prov, self._request(tenant_id=None), _mock_token_manager())

        prov.apps_v1.patch_namespaced_deployment.assert_called_once()
        self._assert_nothing_deleted(prov)

    @pytest.mark.asyncio
    async def test_fresh_create_then_secret_write_failure_rolls_back(self):
        """POSITIVE CONTROL: no incumbent — this call's create RETURNED, then the
        git-token Secret write (the next step) fails. This call made the
        Deployment, so it is rolled back — the fix must not be 'never clean up'."""
        prov = _make_provisioner()
        prov.core_v1.create_namespaced_secret.side_effect = RuntimeError("etcd unavailable")

        with pytest.raises(RuntimeError, match="etcd unavailable"):
            await self._provision(prov, self._request(tenant_id="tenant-a"), _mock_token_manager())

        prov.apps_v1.create_namespaced_deployment.assert_called_once()
        prov.apps_v1.read_namespaced_deployment.assert_not_called()
        self._assert_rolled_back(prov)

    @pytest.mark.asyncio
    async def test_fresh_create_then_failure_after_create_deployment_rolls_back(self):
        """POSITIVE CONTROL: the failure is in ``provision_from_github`` AFTER
        ``_create_deployment`` returned (building the ContainerInfo)."""
        prov = _make_provisioner()

        with (
            patch(f"{MODULE}.ContainerInfo", side_effect=ValueError("bad info")),
            pytest.raises(ValueError, match="bad info"),
        ):
            await self._provision(prov, self._request(tenant_id="tenant-a"), _mock_token_manager())

        prov.core_v1.create_namespaced_secret.assert_called_once()
        self._assert_rolled_back(prov)

    @pytest.mark.asyncio
    async def test_owner_verified_on_409_then_patch_failure_rolls_back(self):
        """POSITIVE CONTROL, pre-existing behaviour kept: 409, the incumbent is
        PROVEN to be the caller's own (tenant-a == tenant-a), then the patch
        fails. origin/main rolled back on any failure; for a verified owner that
        is still the behaviour (it deletes the owner's OWN workspace)."""
        prov = _make_provisioner()
        self._incumbent_owned_by_tenant_a(prov)
        prov.apps_v1.patch_namespaced_deployment.side_effect = RuntimeError("patch failed")

        with pytest.raises(RuntimeError, match="patch failed"):
            await self._provision(prov, self._request(tenant_id="tenant-a"), _mock_token_manager())

        prov.apps_v1.patch_namespaced_deployment.assert_called_once()
        self._assert_rolled_back(prov)


# ---------------------------------------------------------------------------
# exec_in_pod - error channel parsing edge cases
# ---------------------------------------------------------------------------


class TestExecInPodErrorChannelParsing:
    def test_error_channel_invalid_json(self):
        prov = _make_provisioner()

        with patch.object(prov, "_resolve_pod_from_deployment", return_value="test-pod"):
            with patch.object(_mod, "k8s_stream") as mock_stream:
                mock_resp = MagicMock()
                del mock_resp.returncode
                mock_resp.is_open.return_value = False
                mock_resp.read_channel.return_value = "not json"

                mock_stream.return_value = mock_resp

                result = prov.exec_in_pod(
                    deployment_name="test-deploy",
                    command=["test"],
                )
                # Unparseable error channel -> true status unknown -> fail closed.
                # A non-zero command that wrote only to stdout must NOT be
                # reported as success (exit 0); callers trust exit_code.
                assert result["exit_code"] == 1

    def test_error_channel_with_stderr(self):
        """When error channel has invalid JSON and there's stderr output, exit_code should be 1."""
        prov = _make_provisioner()

        with patch.object(prov, "_resolve_pod_from_deployment", return_value="test-pod"):
            with patch.object(_mod, "k8s_stream") as mock_stream:
                # Use a plain MagicMock (not spec=[]) so update() works
                mock_resp = MagicMock()
                # Remove returncode attribute so hasattr check fails
                del mock_resp.returncode
                mock_resp.is_open.side_effect = [True, False]
                mock_resp.peek_stdout.return_value = False
                mock_resp.peek_stderr.return_value = True
                mock_resp.read_stderr.return_value = "some error"
                mock_resp.read_channel.return_value = "invalid json"

                mock_stream.return_value = mock_resp

                result = prov.exec_in_pod(
                    deployment_name="test-deploy",
                    command=["test"],
                )
                # Invalid JSON with stderr -> exit_code 1
                assert result["exit_code"] == 1
                assert "some error" in result["stderr"]

    def test_error_channel_empty(self):
        prov = _make_provisioner()

        with patch.object(prov, "_resolve_pod_from_deployment", return_value="test-pod"):
            with patch.object(_mod, "k8s_stream") as mock_stream:
                mock_resp = MagicMock()
                del mock_resp.returncode
                mock_resp.is_open.return_value = False
                mock_resp.read_channel.return_value = ""

                mock_stream.return_value = mock_resp

                result = prov.exec_in_pod(
                    deployment_name="test-deploy",
                    command=["echo"],
                )
                # Empty error channel -> exit_code stays -1
                assert result["exit_code"] == -1


# ---------------------------------------------------------------------------
# _cleanup_resources - non-404 errors
# ---------------------------------------------------------------------------


class TestCleanupResourcesErrors:
    def test_non_404_deployment_error_logged(self):
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        prov.apps_v1.delete_namespaced_deployment.side_effect = ApiException(
            status=500, reason="Internal"
        )

        # Should not raise - just logs the error
        prov._cleanup_resources("test-deploy")
        prov.core_v1.delete_namespaced_service.assert_called_once()

    def test_non_404_service_error_logged(self):
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        prov.core_v1.delete_namespaced_service.side_effect = ApiException(
            status=403, reason="Forbidden"
        )

        # Should not raise
        prov._cleanup_resources("test-deploy")


# ---------------------------------------------------------------------------
# __init__ with env vars
# ---------------------------------------------------------------------------


class TestInitFromEnv:
    def test_defaults_from_env(self, monkeypatch):
        monkeypatch.setenv("PROVISIONER_NAMESPACE", "custom-ns")
        monkeypatch.setenv("PROVISIONER_IMAGE", "custom:latest")
        monkeypatch.setenv("PROVISIONER_NODE_ARCH", "amd64")
        monkeypatch.setenv("KUBERNETES_IN_CLUSTER", "false")

        with patch("kubernetes.config.load_kube_config"):
            prov = K8sContainerProvisioner(in_cluster=None)

        assert prov.namespace == "custom-ns"
        assert prov.image == "custom:latest"
        assert prov.node_arch == "amd64"

    def test_in_cluster_env(self, monkeypatch):
        monkeypatch.setenv("KUBERNETES_IN_CLUSTER", "true")

        with patch("kubernetes.config.load_incluster_config"):
            prov = K8sContainerProvisioner(namespace="test", in_cluster=None)

        assert prov.namespace == "test"


# ---------------------------------------------------------------------------
# reconstruct_container_info — registry-miss fallback to the live Deployment
# ---------------------------------------------------------------------------


def _fake_deployment(
    *, labels: dict, annotations: dict | None = None, ready: int = 1, desired: int = 1
):
    dep = MagicMock()
    dep.metadata.labels = labels
    dep.metadata.annotations = annotations or {}
    dep.status.ready_replicas = ready
    dep.spec.replicas = desired
    return dep


class TestReconstructContainerInfo:
    def test_returns_running_container_info_for_managed_deployment(self):
        prov = _make_provisioner(namespace="cc")
        prov.apps_v1.read_namespaced_deployment.return_value = _fake_deployment(
            labels={"managed-by": "code-graph-helper", "tenant-id": "tenant-xyz"},
            annotations={
                "coding-job/repo-url": "https://github.com/nonagenticai/enterprise_mcp_server",
                "coding-job/branch": "aa/arch-refactor-cycle-fixture",
                "coding-job/secret-config-id": "sec-123",
            },
            ready=1,
            desired=1,
        )
        with patch.object(_mod, "get_settings") as gs:
            gs.return_value.main_api_endpoint = "https://api.example.test/"
            info = prov.reconstruct_container_info("tour-abc-c0")

        assert info is not None
        assert info.container_name == "tour-abc-c0"
        # ⚠️ INVERTED deliberately. This asserted ``info.tenant_id is None`` — the
        # Deployment carried ``tenant-id: tenant-xyz`` and the reconstruct threw it
        # away, justified as "parity with the registry-HIT path, which stores
        # tenant_id=None". The HIT path does not: ``provision_from_github`` writes a
        # real tenant onto the registry ContainerInfo. Discarding it here did not
        # make the ``get_container`` BOLA gate "not apply", it made the gate VACUOUS
        # (``container_tenant is not None and ...``), so a registry MISS served
        # ``200 running`` to a foreign tenant. Recovering the label is what turns
        # "owner unknown" back into "owner known".
        assert info.tenant_id == "tenant-xyz"
        assert info.status.value == "running"
        assert info.internal_endpoint == "http://tour-abc-c0.cc.svc.cluster.local:8687"
        # Repo metadata recovered from the durable Deployment annotations so the
        # reconstructed container is usable by apply-changes/fork (which require a
        # non-empty repo_url). This is the registry-miss winner-PR-loop fix.
        assert info.repo_url == "https://github.com/nonagenticai/enterprise_mcp_server"
        assert info.branch == "aa/arch-refactor-cycle-fixture"
        assert info.original_repo_owner == "nonagenticai"
        assert info.original_repo_name == "enterprise_mcp_server"
        assert info.secret_config_id == "sec-123"

    def test_repo_url_empty_when_no_annotations(self):
        """Backward-compat: a Deployment provisioned before this change carries no
        repo-url annotation → repo_url stays "" (prior behaviour), no crash."""
        prov = _make_provisioner(namespace="cc")
        prov.apps_v1.read_namespaced_deployment.return_value = _fake_deployment(
            labels={"managed-by": "code-graph-helper"}, annotations={}
        )
        with patch.object(_mod, "get_settings") as gs:
            gs.return_value.main_api_endpoint = "https://api.example.test"
            info = prov.reconstruct_container_info("legacy-c0")
        assert info is not None
        assert info.repo_url == ""
        assert info.branch == "main"
        assert info.original_repo_owner == ""
        assert info.original_repo_name == ""
        assert info.secret_config_id is None

    def test_creating_when_not_ready(self):
        prov = _make_provisioner(namespace="cc")
        prov.apps_v1.read_namespaced_deployment.return_value = _fake_deployment(
            labels={"managed-by": "code-graph-helper"}, ready=0, desired=1
        )
        with patch.object(_mod, "get_settings") as gs:
            gs.return_value.main_api_endpoint = "https://api.example.test"
            info = prov.reconstruct_container_info("tour-abc-c0")
        assert info is not None
        assert info.status.value == "creating"
        assert info.tenant_id is None

    def test_returns_none_on_404(self):
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        prov.apps_v1.read_namespaced_deployment.side_effect = ApiException(status=404)
        assert prov.reconstruct_container_info("missing") is None

    def test_returns_none_for_unmanaged_deployment(self):
        prov = _make_provisioner()
        prov.apps_v1.read_namespaced_deployment.return_value = _fake_deployment(
            labels={"app": "something-else"}
        )
        assert prov.reconstruct_container_info("not-ours") is None


# ---------------------------------------------------------------------------
# _build_workspace_clone_cmd — shell-injection + SHA-as-branch (A4)
# ---------------------------------------------------------------------------


#: The credential ``-c`` block every authenticated clone/fetch carries.
_CRED = kcp.clone_credential_shell_args("github.com", "owner/repo")


class TestBuildWorkspaceCloneCmd:
    def test_plain_branch_and_tokenless_url(self):
        # A clean branch name needs no shell-quoting (shlex.quote is minimal);
        # the regex is what rejects unsafe names. The URL carries NO credential:
        # git records it verbatim as remote.origin.url on the shared emptyDir.
        cmd = kcp._build_workspace_clone_cmd("github.com", "/owner/repo", "main")
        assert "clone --depth 1 --branch main https://github.com/owner/repo /workspace" in cmd
        assert "@github.com" not in cmd
        assert "/workspace" in cmd

    def test_token_reaches_git_only_through_the_bound_credential_helper(self):
        """GitHub documents the password slot for App tokens (ghs_...); the helper
        answers ``username=x-access-token`` + the token, and only for this repo."""
        cmd = kcp._build_workspace_clone_cmd("github.com", "/owner/repo", "main")
        assert f"git {_CRED} clone " in cmd
        assert "https://x-access-token:" not in cmd
        assert "https://$GIT_TOKEN@" not in cmd
        assert "username=x-access-token" in _CRED

    def test_branch_with_slash_allowed(self):
        cmd = kcp._build_workspace_clone_cmd(
            "github.com", "/nonagenticai/enterprise_mcp_server", "aa/arch-refactor-cycle-fixture"
        )
        assert "--branch aa/arch-refactor-cycle-fixture " in cmd

    def test_empty_branch_omits_branch_flag(self):
        cmd = kcp._build_workspace_clone_cmd("github.com", "/owner/repo", "")
        assert "--branch" not in cmd
        assert f"git {_CRED} clone --depth 1 https://github.com/owner/repo /workspace" in cmd

    def test_unauthenticated_public_clone_has_no_credential(self):
        cmd = kcp._build_workspace_clone_cmd(
            "github.com", "/owner/repo", "main", authenticated=False
        )
        assert "$GIT_TOKEN" not in cmd
        assert "x-access-token" not in cmd
        assert "git clone --depth 1 --branch main https://github.com/owner/repo /workspace" in cmd

    def test_unauthenticated_clone_still_enforces_charset_guards(self):
        # Fail-closed must not be conditional on the credential mode.
        with pytest.raises(ValueError):
            kcp._build_workspace_clone_cmd(
                "evil$(whoami).com", "/owner/repo", "main", authenticated=False
            )
        with pytest.raises(ValueError):
            kcp._build_workspace_clone_cmd(
                "github.com", "/owner/repo;rm -rf /", "main", authenticated=False
            )
        with pytest.raises(ValueError):
            kcp._build_workspace_clone_cmd(
                "github.com", "/owner/repo", "main; rm -rf /", authenticated=False
            )

    def test_commit_sha_is_checked_out_not_branched(self):
        sha = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
        cmd = kcp._build_workspace_clone_cmd("github.com", "/owner/repo", sha)
        # SHA can't be a --branch target → full clone + checkout
        assert "--branch" not in cmd
        assert "--depth 1" not in cmd
        # The checkout now declares the directory safe: the emptyDir mount point is root-owned
        # and `git -C` on it fails with "dubious ownership". Intent is unchanged — the SHA is
        # CHECKED OUT rather than passed to --branch; only the ownership exception is added.
        assert f"git -c safe.directory=/workspace -C /workspace checkout {sha}" in cmd

    def test_commit_sha_without_history_cutoff_is_unchanged(self):
        """The flag is opt-in: its default must leave the SHA command exactly as today."""
        sha = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
        default_cmd = kcp._build_workspace_clone_cmd("github.com", "/owner/repo", sha)
        explicit_cmd = kcp._build_workspace_clone_cmd(
            "github.com", "/owner/repo", sha, history_cutoff=False
        )
        assert explicit_cmd == default_cmd
        assert f"git -c safe.directory=/workspace -C /workspace checkout {sha}" in default_cmd
        for cutoff_step in (
            "for-each-ref",
            "update-ref -d",
            "remote remove origin",
            "reflog",
            "gc ",
        ):
            assert cutoff_step not in default_cmd

    def test_commit_sha_with_history_cutoff_cuts_after_checkout(self):
        """SHA + history_cutoff: every ref, the remote, the reflog and the unreachable
        objects go, in that order, AFTER the checkout and inside the same `&&` chain."""
        sha = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
        cmd = kcp._build_workspace_clone_cmd("github.com", "/owner/repo", sha, history_cutoff=True)
        git = "git -c safe.directory=/workspace -C /workspace"

        checkout = f"{git} checkout {sha}"
        steps = [
            checkout,
            f"{git} for-each-ref --format='%(refname)' refs/heads refs/remotes refs/tags",
            f"| xargs -r -n1 {git} update-ref -d",
            f"{git} remote remove origin",
            f"{git} reflog expire --expire=now --all",
            f"{git} gc --prune=now --quiet",
        ]
        positions = [cmd.index(step) for step in steps]  # raises if any step is absent
        assert positions == sorted(positions), "cutoff steps must follow the checkout in order"

        # Same `&&` chain: every step sits between the checkout and the closing `; }` of the
        # _idempotent block, and each boundary between steps is `&&` (the ref deletion is one
        # pipeline, which is why for-each-ref → xargs is joined by `|` rather than `&&`).
        chain_end = cmd.index("; }", positions[0])
        assert all(p < chain_end for p in positions)
        chain = cmd[positions[0] : chain_end]
        assert chain.count(" && ") == 4
        assert chain.count(" | ") == 1
        # A failed cutoff must fail the init container, not certify a half-cut clone.
        assert "|| { find /workspace -mindepth 1 -delete 2>/dev/null || true; exit 1; }" in cmd
        # Still a FULL clone: a shallow one may not even contain the SHA.
        assert "--depth" not in cmd

    def test_history_cutoff_rejects_branch_name(self):
        with pytest.raises(ValueError, match="history_cutoff requires branch to be a full 40-hex"):
            kcp._build_workspace_clone_cmd("github.com", "/owner/repo", "main", history_cutoff=True)

    def test_history_cutoff_rejects_abbreviated_sha(self):
        # `_GIT_SHA_RE` admits 7-40 hex for a plain checkout; the cutoff insists on all 40.
        with pytest.raises(ValueError, match="full 40-hex"):
            kcp._build_workspace_clone_cmd(
                "github.com", "/owner/repo", "0123456789ab", history_cutoff=True
            )

    def test_history_cutoff_rejects_pull_request(self):
        sha = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
        with pytest.raises(ValueError, match="pull-request"):
            kcp._build_workspace_clone_cmd(
                "github.com", "/owner/repo", sha, pr_number=7, history_cutoff=True
            )

    def test_branch_shell_metacharacters_rejected(self):
        with pytest.raises(ValueError):
            kcp._build_workspace_clone_cmd("github.com", "/owner/repo", "main; rm -rf /")

    def test_branch_command_substitution_rejected(self):
        with pytest.raises(ValueError):
            kcp._build_workspace_clone_cmd("github.com", "/owner/repo", "$(touch pwned)")

    def test_malicious_hostname_rejected(self):
        with pytest.raises(ValueError):
            kcp._build_workspace_clone_cmd("evil$(whoami).com", "/owner/repo", "main")

    def test_malicious_path_rejected(self):
        with pytest.raises(ValueError):
            kcp._build_workspace_clone_cmd("github.com", "/owner/repo;rm -rf /", "main")


# ---------------------------------------------------------------------------
# GitHubRepoRequest.history_cutoff — the request-model gate in front of the
# clone builder (there is no dedicated models test file; kept beside the
# builder tests it guards).
# ---------------------------------------------------------------------------


class TestHistoryCutoffRequestModel:
    SHA = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret

    def _req(self, **kw):
        from src.models.container_models import GitHubRepoRequest

        return GitHubRepoRequest(repo_url="https://github.com/owner/repo", **kw)

    def test_defaults_off_and_not_applied(self):
        from src.models.container_models import ContainerInfo, ContainerStatus

        assert self._req(branch="main").history_cutoff is False
        info = ContainerInfo(
            container_name="c",
            api_endpoint="http://x",
            codebase_path="/workspace",
            repo_url="https://github.com/owner/repo",
            branch="main",
            status=ContainerStatus.RUNNING,
        )
        assert info.history_cutoff_applied is False

    def test_accepts_full_sha(self):
        req = self._req(branch=self.SHA, history_cutoff=True)
        assert req.history_cutoff is True
        assert req.branch == self.SHA

    def test_accepts_uppercase_hex_sha(self):
        assert self._req(branch=self.SHA.upper(), history_cutoff=True).history_cutoff is True

    @pytest.mark.parametrize(
        "branch",
        [
            "main",
            "0123456789ab",
            "0123456789abcdef0123456789abcdef0123456",  # pragma: allowlist secret
            "release/1.2",
        ],
    )
    def test_rejects_non_full_sha_branch(self, branch):
        with pytest.raises(ValueError, match="full 40-hex commit SHA"):
            self._req(branch=branch, history_cutoff=True)

    def test_rejects_pull_request_provision(self):
        with pytest.raises(ValueError, match="pull-request"):
            self._req(
                branch=self.SHA, history_cutoff=True, pr_number=7, provision_type="pull_request"
            )

    def test_rejects_pr_url(self):
        # parse_pr_url rewrites pr_number/provision_type first; the cutoff validator
        # runs after it and must see the rewritten fields.
        from src.models.container_models import GitHubRepoRequest

        with pytest.raises(ValueError, match="pull-request"):
            GitHubRepoRequest(
                repo_url="https://github.com/owner/repo/pull/7",
                branch=self.SHA,
                history_cutoff=True,
            )

    def test_flag_off_does_not_constrain_branch(self):
        assert self._req(branch="main", history_cutoff=False).branch == "main"


# ---------------------------------------------------------------------------
# _create_git_token_secret — empty-token guard + masked-ApiException retry
# ---------------------------------------------------------------------------


class TestCreateGitTokenSecret:
    def test_empty_token_raises(self):
        prov = _make_provisioner()
        with pytest.raises(RuntimeError, match="empty token"):
            prov._create_git_token_secret("repo", "")
        prov.core_v1.create_namespaced_secret.assert_not_called()

    def test_happy_path_creates(self):
        prov = _make_provisioner()
        name = prov._create_git_token_secret("repo", "ghp_tok")
        assert name == "repo-git-token"
        prov.core_v1.create_namespaced_secret.assert_called_once()

    def test_409_replaces(self):
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        prov.core_v1.create_namespaced_secret.side_effect = ApiException(status=409)
        name = prov._create_git_token_secret("repo", "ghp_tok")
        assert name == "repo-git-token"
        prov.core_v1.replace_namespaced_secret.assert_called_once()

    def test_genuine_4xx_raises_without_retry(self):
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        prov.core_v1.create_namespaced_secret.side_effect = ApiException(status=403)
        with pytest.raises(ApiException):
            prov._create_git_token_secret("repo", "ghp_tok")
        assert prov.core_v1.create_namespaced_secret.call_count == 1

    def test_masked_attributeerror_retries_then_succeeds(self):
        prov = _make_provisioner()
        # kubernetes==36.0.0 masks a torn ApiException as `.decode` on None.
        prov.core_v1.create_namespaced_secret.side_effect = [
            AttributeError("'NoneType' object has no attribute 'decode'"),
            None,
        ]
        with patch("time.sleep"):
            name = prov._create_git_token_secret("repo", "ghp_tok")
        assert name == "repo-git-token"
        assert prov.core_v1.create_namespaced_secret.call_count == 2

    def test_persistent_masked_failure_raises_runtimeerror(self):
        prov = _make_provisioner()
        prov.core_v1.create_namespaced_secret.side_effect = AttributeError("decode")
        with patch("time.sleep"), pytest.raises(RuntimeError, match="after 3 attempts"):
            prov._create_git_token_secret("repo", "ghp_tok")
        assert prov.core_v1.create_namespaced_secret.call_count == 3


# ---------------------------------------------------------------------------
# App-backed clone credential + the real PUBLIC-repo path
#
# Before this change a project created with no BYO PAT had NO credential
# anywhere (secret_config_id resolves to nothing, GITHUB_TOKEN is unset in the
# cluster), so _create_git_token_secret's empty-token guard fired and every such
# provision was a hard 500 — public repo or not.
# ---------------------------------------------------------------------------


def _init_container_of(prov: K8sContainerProvisioner):
    body = prov.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]
    return body.spec.template.spec.init_containers[0]


class TestTokenNeverReachesTheWorkspaceContainer:
    """$GIT_TOKEN is an env var of the init container ONLY (auditor P0, #138)."""

    def test_only_the_init_container_gets_the_git_token(self):
        prov = _make_provisioner()
        prov._create_deployment(
            name="privrepo",
            repo_url="https://github.com/owner/privrepo",
            branch="main",
            github_token="ghs_apptoken",
            repo_visibility="private",
        )
        body = prov.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]
        pod = body.spec.template.spec
        assert [e.name for e in pod.init_containers[0].env] == ["GIT_TOKEN"]
        for ctr in pod.containers:
            names = [e.name for e in (ctr.env or [])]
            assert "GIT_TOKEN" not in names, ctr.name
            assert not ctr.env_from, ctr.name
            assert "ghs_apptoken" not in str(ctr.env)
        # ...and the token value is nowhere in the rendered pod spec at all.
        assert "ghs_apptoken" not in str(body)
        script = pod.init_containers[0].args[0]
        assert "set -x" not in script and "GIT_TRACE" not in script
        assert "remote set-url origin https://github.com/owner/privrepo" in script


class TestPublicRepoPath:
    def test_public_repo_without_token_creates_no_secret(self):
        prov = _make_provisioner()

        prov._create_deployment(
            name="pubrepo",
            repo_url="https://github.com/owner/pubrepo",
            branch="main",
            github_token=None,
            repo_visibility="public",
        )

        # No git-token Secret at all — not an empty one, not any one.
        prov.core_v1.create_namespaced_secret.assert_not_called()

        init = _init_container_of(prov)
        assert init.env == []
        assert "$GIT_TOKEN" not in init.args[0]
        assert "https://github.com/owner/pubrepo" in init.args[0]

    def test_public_repo_with_token_still_uses_the_authenticated_path(self):
        # A token beats visibility: an App/BYO token must still be used (rate
        # limits, and the same code path as private).
        prov = _make_provisioner()

        prov._create_deployment(
            name="pubrepo",
            repo_url="https://github.com/owner/pubrepo",
            branch="main",
            github_token="ghs_apptoken",
            repo_visibility="public",
        )

        prov.core_v1.create_namespaced_secret.assert_called_once()
        init = _init_container_of(prov)
        assert init.env[0].name == "GIT_TOKEN"
        assert "credential.https://github.com.helper=" in init.args[0]
        assert " https://github.com/owner/pubrepo /workspace" in init.args[0]

    def test_private_repo_without_token_raises_typed_412_not_500(self):
        prov = _make_provisioner()

        with pytest.raises(kcp.GitCredentialUnavailableError) as exc_info:
            prov._create_deployment(
                name="privrepo",
                repo_url="https://github.com/owner/privrepo",
                branch="main",
                github_token=None,
                repo_visibility="private",
            )

        exc = exc_info.value
        assert exc.status_code == 412
        assert exc.error_code == "GIT_CREDENTIAL_UNAVAILABLE"
        assert exc.reason == "private_repo_no_credential"
        assert exc.details["owner"] == "owner"
        message = exc.detail["message"]
        assert "private" in message
        assert "GitHub App is not installed on 'owner'" in message
        # NOT the old opaque RuntimeError/500 from the empty-token Secret guard.
        assert not isinstance(exc, RuntimeError)
        prov.core_v1.create_namespaced_secret.assert_not_called()
        prov.apps_v1.create_namespaced_deployment.assert_not_called()

    def test_unknown_visibility_without_token_fails_closed_with_412(self):
        # The clone-token lookup failed → visibility unknown. Refuse to guess:
        # an unauthenticated clone of a possibly-private repo just reproduces the
        # opaque init-container failure this change exists to remove.
        prov = _make_provisioner()

        with pytest.raises(kcp.GitCredentialUnavailableError) as exc_info:
            prov._create_deployment(
                name="unknownrepo",
                repo_url="https://github.com/owner/unknownrepo",
                branch="main",
                github_token=None,
                repo_visibility=None,
            )

        assert exc_info.value.status_code == 412
        assert exc_info.value.reason == "visibility_unknown"
        prov.core_v1.create_namespaced_secret.assert_not_called()


class TestAppCloneTokenLadder:
    @pytest.mark.asyncio
    async def test_no_pat_calls_clone_token_endpoint_and_secret_gets_ghs_token(self):
        """No BYO PAT → coding-github is asked, and its ghs_ token lands in the Secret."""
        from src.core.container_spec import GitHubRepoRequest

        prov = _make_provisioner()
        mock_token_mgr = _mock_token_manager(
            system_token=None,
            credential=_app_credential(
                ok=True, token="ghs_appinstall", visibility="private", source="app"
            ),
        )

        mock_settings = MagicMock()
        mock_settings.main_api_endpoint = "http://localhost:8687"

        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/privaterepo",
            branch="main",
            tenant_id="tenant-42",
        )

        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_github_token_manager",
                return_value=mock_token_mgr,
            ),
            patch(
                "src.core.kubernetes_container_provisioner.get_settings", return_value=mock_settings
            ),
            patch(
                "src.core.kubernetes_container_provisioner.ContainerMetadataRegistry"
            ) as mock_reg_cls,
        ):
            mock_reg_cls.return_value = MagicMock()
            await prov.provision_from_github(request)

        # The endpoint WAS consulted, tenant-scoped.
        mock_token_mgr.get_app_clone_credential.assert_awaited_once()
        call_kwargs = mock_token_mgr.get_app_clone_credential.await_args.kwargs
        assert call_kwargs["repo_url"] == "https://github.com/owner/privaterepo"
        assert call_kwargs["tenant_id"] == "tenant-42"

        # ...and its token is what the git-clone init container will clone with.
        secret = prov.core_v1.create_namespaced_secret.call_args.kwargs["body"]
        assert secret.string_data == {"GIT_TOKEN": "ghs_appinstall"}
        init = _init_container_of(prov)
        assert "credential.https://github.com.helper=" in init.args[0]
        assert " https://github.com/owner/privaterepo /workspace" in init.args[0]
        assert "@github.com" not in init.args[0]

    @pytest.mark.asyncio
    async def test_byo_pat_short_circuits_the_app_lookup(self):
        from src.core.container_spec import GitHubRepoRequest

        prov = _make_provisioner()
        mock_token_mgr = _mock_token_manager()

        mock_settings = MagicMock()
        mock_settings.main_api_endpoint = "http://localhost:8687"

        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo",
            branch="main",
            access_token="ghp_byo",
        )

        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_github_token_manager",
                return_value=mock_token_mgr,
            ),
            patch(
                "src.core.kubernetes_container_provisioner.get_settings", return_value=mock_settings
            ),
            patch.object(prov, "_create_deployment") as deploy_mock,
            patch("src.core.kubernetes_container_provisioner.ContainerMetadataRegistry"),
        ):
            await prov.provision_from_github(request)

        mock_token_mgr.get_app_clone_credential.assert_not_awaited()
        assert deploy_mock.call_args.kwargs["github_token"] == "ghp_byo"

    @pytest.mark.asyncio
    async def test_public_repo_no_token_anywhere_provisions_without_a_secret(self):
        """The end-to-end case that used to 500: no PAT, no App token, public repo."""
        from src.core.container_spec import GitHubRepoRequest

        prov = _make_provisioner()
        mock_token_mgr = _mock_token_manager(
            system_token=None,
            credential=_app_credential(ok=True, token=None, visibility="public", source="none"),
        )

        mock_settings = MagicMock()
        mock_settings.main_api_endpoint = "http://localhost:8687"

        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/pubrepo",
            branch="main",
        )

        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_github_token_manager",
                return_value=mock_token_mgr,
            ),
            patch(
                "src.core.kubernetes_container_provisioner.get_settings", return_value=mock_settings
            ),
            patch(
                "src.core.kubernetes_container_provisioner.ContainerMetadataRegistry"
            ) as mock_reg_cls,
        ):
            mock_reg_cls.return_value = MagicMock()
            result = await prov.provision_from_github(request)

        assert result.status.value == "running"
        prov.core_v1.create_namespaced_secret.assert_not_called()
        prov.apps_v1.create_namespaced_deployment.assert_called_once()
        assert "$GIT_TOKEN" not in _init_container_of(prov).args[0]

    @pytest.mark.asyncio
    async def test_private_repo_no_token_anywhere_surfaces_the_412(self):
        from src.core.container_spec import GitHubRepoRequest

        prov = _make_provisioner()
        mock_token_mgr = _mock_token_manager(
            system_token=None,
            credential=_app_credential(ok=True, token=None, visibility="private", source="none"),
        )

        mock_settings = MagicMock()
        mock_settings.main_api_endpoint = "http://localhost:8687"

        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/privaterepo",
            branch="main",
        )

        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_github_token_manager",
                return_value=mock_token_mgr,
            ),
            patch(
                "src.core.kubernetes_container_provisioner.get_settings", return_value=mock_settings
            ),
            patch("src.core.kubernetes_container_provisioner.ContainerMetadataRegistry"),
        ):
            with pytest.raises(kcp.GitCredentialUnavailableError) as exc_info:
                await prov.provision_from_github(request)

        # The 412 must ESCAPE provision_from_github's
        # `except (ApiException, OSError, RuntimeError, ValueError)` unchanged.
        assert exc_info.value.status_code == 412

    @pytest.mark.asyncio
    async def test_env_system_token_is_never_used_when_the_app_has_none(self):
        """PLATFORM owner: no App token is a 412; the env PAT is not a rung."""
        from src.core.container_spec import GitHubRepoRequest

        prov = _make_provisioner()
        mock_token_mgr = _mock_token_manager(
            system_token="ghp_systemfallback",
            credential=_app_credential(ok=True, token=None, visibility="private", source="none"),
        )

        mock_settings = MagicMock()
        mock_settings.main_api_endpoint = "http://localhost:8687"

        request = GitHubRepoRequest(
            repo_url="https://github.com/nonagenticai/repo", branch="main", tenant_id="tenant-1"
        )

        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_github_token_manager",
                return_value=mock_token_mgr,
            ),
            patch(
                "src.core.kubernetes_container_provisioner.get_settings", return_value=mock_settings
            ),
            patch.object(prov, "_create_deployment") as deploy_mock,
            patch("src.core.kubernetes_container_provisioner.ContainerMetadataRegistry"),
            pytest.raises(kcp.GitCredentialUnavailableError) as exc_info,
        ):
            await prov.provision_from_github(request)

        assert exc_info.value.status_code == 412
        assert exc_info.value.reason == "platform_credential_unavailable"
        deploy_mock.assert_not_called()
        assert not hasattr(mock_token_mgr, "get_system_token")


# ---------------------------------------------------------------------------
# git-token Secret lifecycle ownership
#
# The leak this closes: the git-token Secret was only ever reclaimed by the
# explicit remove_container -> _cleanup_resources teardown. Any provision that
# was abandoned, crashed, or was torn down out-of-band (`kubectl delete deploy`,
# the reaper) left its credential in the namespace forever — 60 unreferenced
# `*-git-token` Secrets had accumulated in coding-jobs (2026-06-30 .. 2026-07-16)
# against zero running Deployments.
#
# The fix is an ownerReference on the Deployment, stamped at Secret-CREATE time
# (the Deployment is now created first) so the credential is never visible to
# the apiserver in an unowned state.
# ---------------------------------------------------------------------------

_LIVE_UID = "11111111-2222-3333-4444-555555555555"


def _minimal_deployment_spec():
    """A schema-valid ``V1DeploymentSpec`` for stub apiserver responses.

    kubernetes 37 (2026-11) made ``spec`` a REQUIRED field on ``V1Deployment``, so
    a stub built from metadata alone now raises a pydantic ValidationError before
    the code under test ever runs. Nothing here inspects the pod template — the
    fixtures only need an object the current client will accept.
    """
    from kubernetes import client as k8s_client

    return k8s_client.V1DeploymentSpec(
        selector=k8s_client.V1LabelSelector(match_labels={"app": "owned-repo"}),
        template=k8s_client.V1PodTemplateSpec(
            metadata=k8s_client.V1ObjectMeta(labels={"app": "owned-repo"}),
            spec=k8s_client.V1PodSpec(containers=[]),
        ),
    )


def _live_deployment(name: str = "owned-repo", uid: str = _LIVE_UID):
    """What the apiserver returns from create/patch — carries a real uid."""
    from kubernetes import client as k8s_client

    return k8s_client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=k8s_client.V1ObjectMeta(name=name, namespace="test-ns", uid=uid),
        spec=_minimal_deployment_spec(),
    )


def _created_secret_body(prov: K8sContainerProvisioner):
    return prov.core_v1.create_namespaced_secret.call_args.kwargs["body"]


def _owner_refs(obj) -> list:
    meta = getattr(obj, "metadata", None)
    return list(getattr(meta, "owner_references", None) or [])


def _would_be_garbage_collected(obj, live_owner_uids: set[str]) -> bool:
    """Mirror of the k8s garbage collector's rule for a namespaced dependent.

    A dependent is collected once it has at least one ownerReference and NONE of
    those owners still exist (owners are matched by uid, which is why a stale or
    guessed uid is dangerous). An object with NO ownerReferences is never touched
    by the GC — that is exactly the leak, so this returns False for it.
    """
    refs = _owner_refs(obj)
    if not refs:
        return False
    return all(ref.uid not in live_owner_uids for ref in refs)


class TestGitTokenSecretOwnership:
    def test_secret_is_created_with_an_ownerreference_to_the_deployment(self):
        prov = _make_provisioner()
        prov.apps_v1.create_namespaced_deployment.return_value = _live_deployment()

        prov._create_deployment(
            name="owned-repo",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
        )

        refs = _owner_refs(_created_secret_body(prov))
        assert len(refs) == 1, "the git-token Secret must be owned by exactly one object"
        ref = refs[0]
        assert ref.api_version == "apps/v1"
        assert ref.kind == "Deployment"
        assert ref.name == "owned-repo"
        assert ref.uid == _LIVE_UID
        assert ref.controller is True
        # block_owner_deletion=True would need `update` on deployments/finalizers
        # under OwnerReferencesPermissionEnforcement; the Role does not grant it.
        assert ref.block_owner_deletion is False

    def test_uid_is_taken_from_the_live_object_not_the_submitted_body(self):
        """The submitted V1Deployment body has no uid — only the apiserver's
        response does. A constructed/guessed uid would make the GC delete the
        Secret immediately."""
        prov = _make_provisioner()
        prov.apps_v1.create_namespaced_deployment.return_value = _live_deployment(
            uid="99999999-8888-7777-6666-555555555555"
        )

        prov._create_deployment(
            name="owned-repo",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
        )

        submitted = prov.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]
        assert submitted.metadata.uid is None, "precondition: the request body carries no uid"
        assert (
            _owner_refs(_created_secret_body(prov))[0].uid == "99999999-8888-7777-6666-555555555555"
        )

    def test_owner_is_created_before_the_secret(self):
        """Ordering: the Deployment must exist before the credential is written,
        otherwise there is a window in which an unowned Secret sits in etcd."""
        with patch("kubernetes.config.load_kube_config"):
            prov = K8sContainerProvisioner(namespace="test-ns", in_cluster=False)
        recorder = MagicMock()
        prov.apps_v1 = recorder.apps_v1
        prov.core_v1 = recorder.core_v1
        recorder.apps_v1.create_namespaced_deployment.return_value = _live_deployment()

        prov._create_deployment(
            name="owned-repo",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
        )

        names = [c[0] for c in recorder.mock_calls]
        assert "apps_v1.create_namespaced_deployment" in names
        assert "core_v1.create_namespaced_secret" in names
        assert names.index("apps_v1.create_namespaced_deployment") < names.index(
            "core_v1.create_namespaced_secret"
        )

    def test_409_replace_keeps_the_ownerreference(self):
        """replace_namespaced_secret sends a whole object; a body without
        ownerReferences would WIPE the one we just set."""
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        prov.apps_v1.create_namespaced_deployment.return_value = _live_deployment()
        prov.core_v1.create_namespaced_secret.side_effect = ApiException(status=409)

        prov._create_deployment(
            name="owned-repo",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
        )

        replaced = prov.core_v1.replace_namespaced_secret.call_args.kwargs["body"]
        assert _owner_refs(replaced)[0].uid == _LIVE_UID

    def test_in_place_reconcile_owns_the_secret_with_the_patched_uid(self):
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        prov.apps_v1.create_namespaced_deployment.side_effect = ApiException(
            status=409, reason="Conflict"
        )
        existing = MagicMock()
        existing.metadata.deletion_timestamp = None
        prov.apps_v1.read_namespaced_deployment.return_value = existing
        prov.apps_v1.patch_namespaced_deployment.return_value = _live_deployment(
            uid="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        )

        prov._create_deployment(
            name="owned-repo",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
        )

        assert (
            _owner_refs(_created_secret_body(prov))[0].uid == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        )

    def test_recreate_after_termination_owns_the_secret_with_the_new_uid(self):
        """The new incarnation has a NEW uid; reusing the terminated one would
        make the GC delete the fresh Secret."""
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        conflict = ApiException(status=409, reason="Conflict")
        prov.apps_v1.create_namespaced_deployment.side_effect = [
            conflict,
            _live_deployment(uid="new-uid-after-recreate"),
        ]
        terminating = MagicMock()
        terminating.metadata.deletion_timestamp = "2026-07-30T00:00:00Z"
        not_found = ApiException(status=404, reason="Not Found")
        prov.apps_v1.read_namespaced_deployment.side_effect = [terminating, not_found]

        with patch.object(time_module, "sleep"):
            prov._create_deployment(
                name="owned-repo",
                repo_url="https://github.com/owner/repo",
                branch="main",
                github_token="ghp_test",
            )

        assert _owner_refs(_created_secret_body(prov))[0].uid == "new-uid-after-recreate"

    def test_missing_uid_degrades_to_no_owner_rather_than_a_bogus_one(self, caplog):
        """A response without a real uid must NOT produce a guessed ownerReference:
        a uid that never existed means immediate garbage collection of a live
        credential. Degrade to today's behaviour and warn."""
        from kubernetes import client as k8s_client

        prov = _make_provisioner()
        prov.apps_v1.create_namespaced_deployment.return_value = k8s_client.V1Deployment(
            metadata=k8s_client.V1ObjectMeta(name="owned-repo", namespace="test-ns"),
            spec=_minimal_deployment_spec(),
        )

        with caplog.at_level("WARNING"):
            prov._create_deployment(
                name="owned-repo",
                repo_url="https://github.com/owner/repo",
                branch="main",
                github_token="ghp_test",
            )

        assert _owner_refs(_created_secret_body(prov)) == []
        assert any("WITHOUT an ownerReference" in r.getMessage() for r in caplog.records)

    def test_public_repo_path_creates_no_secret_to_own(self):
        prov = _make_provisioner()
        prov.apps_v1.create_namespaced_deployment.return_value = _live_deployment()

        prov._create_deployment(
            name="pubrepo",
            repo_url="https://github.com/owner/pubrepo",
            branch="main",
            github_token=None,
            repo_visibility="public",
        )

        prov.core_v1.create_namespaced_secret.assert_not_called()


class TestAbandonedProvisionSecretIsCollectable:
    """The negative control: the abandonment path, where nothing calls
    remove_container. Before the fix the Secret survived forever."""

    @pytest.mark.asyncio
    async def test_abandoned_provision_leaves_a_collectable_secret(self):
        from src.core.container_spec import GitHubRepoRequest

        prov = _make_provisioner()
        prov.apps_v1.create_namespaced_deployment.return_value = _live_deployment(
            name="abandoned-repo"
        )

        mock_token_mgr = _mock_token_manager(system_token="ghp_sys")
        mock_settings = MagicMock()
        mock_settings.main_api_endpoint = "http://localhost:8687"

        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo",
            branch="main",
            container_name="abandoned-repo",
            access_token="ghp_byo",
        )

        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_github_token_manager",
                return_value=mock_token_mgr,
            ),
            patch(
                "src.core.kubernetes_container_provisioner.get_settings", return_value=mock_settings
            ),
            patch("src.core.kubernetes_container_provisioner.ContainerMetadataRegistry"),
        ):
            await prov.provision_from_github(request)

        secret = _created_secret_body(prov)

        # Nothing tore the provision down: no explicit Secret delete ever ran.
        prov.core_v1.delete_namespaced_secret.assert_not_called()
        prov.apps_v1.delete_namespaced_deployment.assert_not_called()

        # False-positive control: while the owner is alive the GC must NOT
        # collect the credential (otherwise this whole assertion is vacuous).
        assert _would_be_garbage_collected(secret, {_LIVE_UID}) is False

        # ...and once the owner is gone — crashed, evicted, reaped, or deleted
        # out of band — the GC reclaims it. On the pre-fix code the Secret had no
        # ownerReferences at all, so this is False and the credential leaks.
        assert _would_be_garbage_collected(secret, set()) is True

    @pytest.mark.asyncio
    async def test_explicit_teardown_still_deletes_the_secret(self):
        """The ownerReference is a safety net UNDER _cleanup_resources, not a
        replacement: the synchronous delete must survive."""
        prov = _make_provisioner()
        await prov.remove_container("abandoned-repo")

        prov.core_v1.delete_namespaced_secret.assert_called_once()
        assert (
            prov.core_v1.delete_namespaced_secret.call_args.kwargs["name"]
            == "abandoned-repo-git-token"
        )

    @pytest.mark.asyncio
    async def test_secret_create_failure_after_the_deployment_is_torn_down(self):
        """The one regression the create-owner-first ordering could introduce:
        a Secret write that fails now leaves a Deployment behind. It must not —
        provision_from_github's error path has to reclaim it."""
        from src.core.container_spec import GitHubRepoRequest

        prov = _make_provisioner()
        prov.apps_v1.create_namespaced_deployment.return_value = _live_deployment(
            name="doomed-repo"
        )
        prov.core_v1.create_namespaced_secret.side_effect = RuntimeError("etcd unavailable")

        mock_token_mgr = _mock_token_manager(system_token="ghp_sys")
        mock_settings = MagicMock()
        mock_settings.main_api_endpoint = "http://localhost:8687"

        request = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo",
            branch="main",
            container_name="doomed-repo",
            access_token="ghp_byo",
        )

        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_github_token_manager",
                return_value=mock_token_mgr,
            ),
            patch(
                "src.core.kubernetes_container_provisioner.get_settings", return_value=mock_settings
            ),
            patch("src.core.kubernetes_container_provisioner.ContainerMetadataRegistry"),
            patch("time.sleep"),
            pytest.raises(RuntimeError),
        ):
            await prov.provision_from_github(request)

        prov.apps_v1.delete_namespaced_deployment.assert_called_once()
        assert prov.apps_v1.delete_namespaced_deployment.call_args.kwargs["name"] == "doomed-repo"


# ---------------------------------------------------------------------------
# The OTHER 70 stranded Secrets, and why they are not a second creation path
#
# 70 unreferenced `*-git-token` Secrets also sit in the `coding-containers`
# namespace itself (2026-05-20T10:26:51Z .. 2026-06-30T11:28:29Z), against the 60
# in `coding-jobs` (2026-06-30T20:26:35Z .. 2026-07-16T16:28:35Z). They are NOT a
# different code path and not an older one: they are THIS path, running under the
# older `PROVISIONER_NAMESPACE=coding-containers` config. gitops bef516cf9
# (2026-06-30 16:37:55Z, "create dedicated coding-jobs namespace for untrusted
# analysis pods") flipped that ConfigMap value to `coding-jobs`, and the two
# Secret populations do not overlap by a single object across that instant.
#
# So the fix above already covers both — but only because the Deployment and the
# Secret are BOTH written to `self.namespace`. That coupling is load-bearing and
# was, until now, asserted nowhere:
#
#   * ownerReferences are namespace-local. A namespaced dependent may only be
#     owned by an owner in its OWN namespace. If the Secret were ever written to
#     a namespace other than its Deployment's, the garbage collector would
#     resolve the reference to a non-existent owner and delete the credential
#     IMMEDIATELY — turning this fix into a live-credential shredder, which is
#     strictly worse than the leak it closes.
#   * The historical trigger here was exactly a namespace migration. The next one
#     must not be able to silently decouple the two writes.
# ---------------------------------------------------------------------------

_HISTORICAL_NS = "coding-containers"  # where the 70 landed, pre-2026-06-30
_CURRENT_NS = "coding-jobs"  # where the 60 landed, post-flip
_CODE_DEFAULT_NS = "code-graph-helper"  # PROVISIONER_NAMESPACE fallback in __init__


def _live_deployment_in(namespace: str, name: str = "owned-repo", uid: str = _LIVE_UID):
    """Apiserver response for a Deployment created in an arbitrary namespace."""
    from kubernetes import client as k8s_client

    return k8s_client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=k8s_client.V1ObjectMeta(name=name, namespace=namespace, uid=uid),
        spec=_minimal_deployment_spec(),
    )


class TestOwnerReferenceIsNamespaceAgnostic:
    """Pins the invariant that makes the fix cover BOTH stranded populations."""

    @pytest.mark.parametrize(
        "namespace",
        [_HISTORICAL_NS, _CURRENT_NS, _CODE_DEFAULT_NS, "test-ns"],
    )
    def test_secret_is_owned_whatever_provisioner_namespace_is(self, namespace: str):
        prov = _make_provisioner(namespace=namespace)
        prov.apps_v1.create_namespaced_deployment.return_value = _live_deployment_in(namespace)

        prov._create_deployment(
            name="owned-repo",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
        )

        secret = _created_secret_body(prov)
        refs = _owner_refs(secret)
        assert len(refs) == 1, (
            f"a git-token Secret provisioned into {namespace!r} must still be owned; "
            "an unowned one is exactly what stranded 130 credentials"
        )
        assert refs[0].uid == _LIVE_UID
        assert refs[0].kind == "Deployment"

        # False-positive control: while the owner is alive the credential must
        # NOT be collectable, or the assertion below proves nothing.
        assert _would_be_garbage_collected(secret, {_LIVE_UID}) is False
        assert _would_be_garbage_collected(secret, set()) is True

    @pytest.mark.parametrize(
        "namespace",
        [_HISTORICAL_NS, _CURRENT_NS, _CODE_DEFAULT_NS, "test-ns"],
    )
    def test_secret_and_its_owner_land_in_the_same_namespace(self, namespace: str):
        """A cross-namespace ownerReference resolves to a MISSING owner, so the GC
        deletes the dependent at once. The Secret must follow its Deployment."""
        prov = _make_provisioner(namespace=namespace)
        prov.apps_v1.create_namespaced_deployment.return_value = _live_deployment_in(namespace)

        prov._create_deployment(
            name="owned-repo",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
        )

        deploy_ns = prov.apps_v1.create_namespaced_deployment.call_args.kwargs["namespace"]
        secret_ns = prov.core_v1.create_namespaced_secret.call_args.kwargs["namespace"]
        assert deploy_ns == namespace
        assert secret_ns == deploy_ns, (
            "the git-token Secret must be created in its owning Deployment's namespace; "
            "an ownerReference across namespaces is treated as a dangling owner and the "
            "live credential is garbage-collected immediately"
        )
        assert _created_secret_body(prov).metadata.namespace == deploy_ns


class TestWorkspacePodHardening:
    """The pod runs a customer's repository code; it must not run it as root.

    Before this landed, `.spec.securityContext` was `{}`, both containers ran as
    uid 0 with the runtime's default capability set, and there was no seccomp
    profile — measured on live pods in `coding-jobs`, freshly provisioned, so
    current behaviour rather than legacy drift.
    """

    def _pod_spec(self):
        prov = _make_provisioner()
        prov._create_deployment(
            name="hardened",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
        )
        deployment = prov.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]
        return deployment.spec.template.spec

    def test_pod_runs_as_a_fixed_unprivileged_identity(self):
        from src.core.kubernetes_container_provisioner import WORKSPACE_GID, WORKSPACE_UID

        sc = self._pod_spec().security_context
        assert sc is not None, "pod securityContext was {} — repository code ran as root"
        assert sc.run_as_non_root is True
        assert sc.run_as_user == WORKSPACE_UID
        assert sc.run_as_group == WORKSPACE_GID
        assert sc.seccomp_profile.type == "RuntimeDefault"

    def test_fsgroup_makes_the_emptydir_writable_by_that_identity(self):
        """Both volumes are emptyDir, so fsGroup is what grants group write.

        Without it the clone lands root-owned and the non-root main container
        cannot write .git — `git commit` and `git config --local` fail EACCES.
        """
        from src.core.kubernetes_container_provisioner import WORKSPACE_GID

        sc = self._pod_spec().security_context
        assert sc.fs_group == WORKSPACE_GID
        assert sc.fs_group_change_policy == "OnRootMismatch"

    def test_the_clone_runs_as_the_same_identity_as_the_workspace(self):
        """The init container inherits the POD context — that is the ownership fix.

        `_build_workspace_clone_cmd` ends with `chmod -R 755`, which grants group
        r-x and never w. A root clone plus fsGroup alone would still leave
        .git/objects unwritable, so the clone has to run AS the workspace user
        rather than merely alongside it. Inheriting the pod-level context is what
        makes that true, so assert no init container overrides it back to root.
        """
        pod_spec = self._pod_spec()
        assert pod_spec.init_containers, "expected the git-clone init container"
        for container in pod_spec.init_containers:
            override = getattr(container.security_context, "run_as_user", None)
            assert override in (None, pod_spec.security_context.run_as_user)

    def test_every_container_drops_capabilities_and_cannot_escalate(self):
        pod_spec = self._pod_spec()
        for container in list(pod_spec.init_containers or []) + list(pod_spec.containers or []):
            sc = container.security_context
            assert sc is not None, f"{container.name} had no securityContext"
            assert sc.allow_privilege_escalation is False, container.name
            assert sc.capabilities.drop == ["ALL"], container.name

    def test_home_is_explicit(self):
        """Left to passwd resolution, $HOME becomes "/" for a uid with no entry."""
        from src.core.kubernetes_container_provisioner import WORKSPACE_HOME

        analysis = next(c for c in self._pod_spec().containers if c.name == "analysis")
        env = {v.name: v.value for v in (analysis.env or [])}
        assert env.get("HOME") == WORKSPACE_HOME


class TestBoundedStream:
    """`stdout_data += resp.read_stdout()` had no ceiling before this."""

    def test_keeps_everything_under_the_limit(self):
        from src.core.kubernetes_container_provisioner import _BoundedStream

        buf = _BoundedStream(100)
        buf.write("hello ")
        buf.write("world")
        assert buf.value() == "hello world"
        assert buf.truncated is False
        assert buf.bytes_omitted == 0

    def test_clips_at_the_limit_and_counts_the_remainder(self):
        from src.core.kubernetes_container_provisioner import _BoundedStream

        buf = _BoundedStream(10)
        buf.write("a" * 25)
        assert len(buf.value()) == 10
        assert buf.truncated is True
        assert buf.bytes_omitted == 15

    def test_later_writes_are_dropped_not_buffered(self):
        """The point is bounding memory, not trimming after the fact."""
        from src.core.kubernetes_container_provisioner import _BoundedStream

        buf = _BoundedStream(5)
        buf.write("abcde")
        buf.write("f" * 1000)
        assert buf.value() == "abcde"
        assert buf.bytes_omitted == 1000

    def test_counts_bytes_not_characters(self):
        """A character count understates a multi-byte stream by up to 4x."""
        from src.core.kubernetes_container_provisioner import _BoundedStream

        buf = _BoundedStream(4)
        buf.write("€€")  # 3 bytes each
        assert buf.bytes_omitted == 3
        # The straddling character is dropped rather than emitted half-decoded.
        assert buf.value() == "€"

    def test_a_zero_limit_keeps_nothing_and_still_counts(self):
        from src.core.kubernetes_container_provisioner import _BoundedStream

        buf = _BoundedStream(0)
        buf.write("anything")
        assert buf.value() == ""
        assert buf.truncated is True
        assert buf.bytes_omitted == 8


class TestRepositoryTreeReportsIncompleteness:
    """A tree that lost its tail must say so.

    `exec_in_pod` is bounded now, so a repository large enough to hit the
    ceiling would otherwise return a subtree indistinguishable from the whole
    thing — the same silent-success shape as a directory listing that reports
    "no files" for a path it could not read.
    """

    @pytest.mark.asyncio
    async def test_complete_tree_is_marked_complete(self):
        prov = _make_provisioner()
        with patch.object(
            prov,
            "exec_in_pod",
            return_value={
                "exit_code": 0,
                "stdout": "/workspace/src\n/workspace/src/main.py\n",
                "stderr": "",
                "duration": 0.1,
                "truncated": False,
                "bytes_omitted": 0,
            },
        ):
            result = await prov.get_container_repository_tree("repo")
        assert result["truncated"] is False
        assert result["bytes_omitted"] == 0
        assert "src" in result["root"]["children"]

    @pytest.mark.asyncio
    async def test_truncated_tree_is_marked_truncated(self):
        prov = _make_provisioner()
        with patch.object(
            prov,
            "exec_in_pod",
            return_value={
                "exit_code": 0,
                "stdout": "/workspace/src\n",
                "stderr": "",
                "duration": 0.1,
                "truncated": True,
                "bytes_omitted": 9001,
            },
        ):
            result = await prov.get_container_repository_tree("repo")
        assert result["truncated"] is True
        assert result["bytes_omitted"] == 9001


class TestCloneCommandSurvivesANonRootWorkspace:
    """Regression guards for a production breakage.

    When workspace pods moved to uid 1000, the clone command still ended with
    `chmod -R 755 /workspace`. The emptyDir MOUNT POINT is root-owned (the
    kubelet creates it root:fsGroup), and chmod requires OWNERSHIP rather than
    write permission — so the non-root clone got EPERM on /workspace itself,
    exited non-zero, broke the `&&` chain, and every new workspace went
    Init:CrashLoopBackOff.

    Then it got worse: the retry hit the same emptyDir, now non-empty, so
    `git clone` refused it with "destination path already exists" forever. One
    transient failure made a workspace permanently unrecoverable.

    The whole suite passed throughout — the commands are built as strings and
    nothing asserted on their shape, so a command that could not run looked
    identical to one that could.
    """

    def _cmd(self, branch="main", **kw):
        from src.core.kubernetes_container_provisioner import _build_workspace_clone_cmd

        return _build_workspace_clone_cmd(
            "github.com", "/owner/repo", branch, authenticated=False, **kw
        )

    @pytest.mark.parametrize("branch", ["main", "a" * 40, ""])
    def test_it_never_chmods_the_mount_point_itself(self, branch):
        """`chmod -R 755 /workspace` is EPERM for a non-root clone."""
        cmd = self._cmd(branch=branch)
        assert "chmod -R 755 /workspace" not in cmd
        assert "chmod -R" not in cmd

    @pytest.mark.parametrize("branch", ["main", "a" * 40, ""])
    def test_there_is_no_chmod_at_all(self, branch):
        """The clone must not touch modes. Do not reinstate this.

        Applying 755 to every file made a fresh clone report ` M` on each
        non-executable one, because git tracks the executable bit — so any
        "is the worktree clean" check saw dirty immediately, on a checkout
        nobody had modified.

        It was never needed: git already writes 644 for data files and 755 for
        files committed executable, and the clone and the workspace run as the
        SAME uid, so nothing requires group/other bits to be widened.
        """
        cmd = self._cmd(branch=branch)
        assert "chmod" not in cmd, "the clone must not rewrite file modes"

    @pytest.mark.parametrize("branch", ["main", "a" * 40, ""])
    def test_a_completed_clone_is_left_alone(self, branch):
        """Init containers re-run when a pod sandbox is recreated.

        An emptyDir survives that, so an unconditional wipe would delete a
        checkout the workspace had been using, uncommitted work included.
        """
        cmd = self._cmd(branch=branch)
        assert "-d /workspace/.git" in cmd

    @pytest.mark.parametrize("branch", ["main", "a" * 40, ""])
    def test_a_partial_destination_is_cleared_before_retrying(self, branch):
        """Otherwise one failed clone poisons the volume permanently."""
        cmd = self._cmd(branch=branch)
        assert "-mindepth 1 -delete" in cmd

    @pytest.mark.parametrize("branch", ["main", "a" * 40, ""])
    def test_nothing_walks_the_checkout_mutating_it(self, branch):
        """The clone writes the checkout and then leaves it alone.

        An intermediate version of this fix used `find -exec chmod` and
        reintroduced the bug it was fixing: `find -exec` passes each symlink to
        chmod as an ARGUMENT, and chmod then follows it to the target. A repo
        shipping an absolute symlink to a file the workspace user does not own
        made chmod exit non-zero — broken `&&` chain, crashloop — and, worse,
        gave the command a write primitive pointing OUTSIDE the checkout.

        Removing the chmod entirely removes the whole class. The only permitted
        traversal is the guarded `-delete` that clears a PARTIAL destination,
        which by construction runs only when there is no valid clone present.

        There are now TWO such deletes, and the count alone is not the invariant —
        WHERE they sit is. Both are inside `if [ ! -d /workspace/.git ]`: one clears a
        partial destination BEFORE cloning, one clears it AFTER a failed chain so the
        retry genuinely redoes the work instead of skipping on a `.git` left by a
        clone whose fetch/checkout failed. Neither can run when a VALID clone exists,
        because the guard is false and the whole block is skipped. The assertion below
        checks that placement, not just the number.
        """
        # Authenticated too: that path adds the one READ-ONLY credential scan
        # (`find -type f -exec grep -lF`; find does not follow symlinks without -L
        # and grep writes nothing). No other -exec may appear on either path.
        authed = kcp._build_workspace_clone_cmd("github.com", "/owner/repo", branch)
        assert authed.count("-exec") == authed.count("-type f -exec grep -lF ") == 1
        assert authed.count("find ") == 3
        cmd = self._cmd(branch=branch)
        assert "-exec" not in cmd
        assert "chmod" not in cmd and "chmod" not in authed
        # Every mutating walk is a bounded `-mindepth 1 -delete`, never a bare recursive remove.
        assert cmd.count("find ") == cmd.count("find /workspace -mindepth 1 -delete") == 2
        assert authed.count("find /workspace -mindepth 1 -delete") == 2
        assert "rm -rf" not in cmd
        # …and every delete is inside the no-valid-clone guard.
        guard = "if [ ! -d /workspace/.git ]; then "
        assert cmd.startswith(guard)
        assert "find " not in cmd[: len(guard)], "no traversal may precede the guard"
        guarded, _, after = cmd.partition("; fi && ")
        assert guarded.count("-delete") == 2
        # What follows the guard is the credential scrub, which runs over a VALID
        # checkout too — so it must never delete (it may hold uncommitted work).
        assert after and "-delete" not in after and "rm " not in after

    def test_the_command_is_balanced_shell(self):
        """It is embedded in a pod spec and run by `sh -c`; it must parse."""
        import subprocess

        for branch in ("main", "a" * 40, "", "feature/x"):
            cmd = self._cmd(branch=branch)
            proc = subprocess.run(["sh", "-n", "-c", cmd], capture_output=True, text=True)
            assert proc.returncode == 0, f"unparseable for {branch!r}: {proc.stderr}"

    def test_injection_guards_still_reject_bad_input(self):
        """The rewrite must not have loosened validation."""
        from src.core.kubernetes_container_provisioner import _build_workspace_clone_cmd

        for bad_host in ("git hub.com", "github.com;rm -rf /", ""):
            with pytest.raises(ValueError):
                _build_workspace_clone_cmd(bad_host, "/o/r", "main", authenticated=False)
        with pytest.raises(ValueError):
            _build_workspace_clone_cmd("github.com", "/o/r", "main; rm -rf /", authenticated=False)


class TestGitWorksOnARootOwnedWorktree:
    """Third bug from the same root cause, and the one with the widest blast radius.

    The kubelet creates an emptyDir owned by root, and `fsGroup` sets only the
    GROUP — so `/workspace` stays uid 0 while everything the clone writes inside
    it is uid 1000. Git compares the WORKTREE directory's owner against the euid
    and refuses:

        fatal: detected dubious ownership in repository at '/workspace'

    Measured on a live pod: status, commit and push all failed with exit 128,
    while `touch /workspace/probe` succeeded — the directory is writable, it is
    just not OWNED. Writability and ownership are different things, which is the
    same distinction that broke `chmod` two commits ago.

    Nothing can chown the mount point: it needs root, and removing the last root
    container from this pod is the whole point of running as 1000. Git's own
    escape hatch for exactly this container case is `safe.directory`.

    Every prior test passed while this was broken, because they assert on the pod
    SPEC and nothing exercised git inside the resulting workspace.
    """

    def _analysis_env(self):
        prov = _make_provisioner()
        prov._create_deployment(
            name="gitprobe",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
        )
        deployment = prov.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]
        analysis = next(c for c in deployment.spec.template.spec.containers if c.name == "analysis")
        return {v.name: v.value for v in (analysis.env or [])}

    def test_safe_directory_is_configured(self):
        """Scoped to /workspace, never a wildcard, and carried by GIT_CONFIG_PARAMETERS
        so a caller's GIT_CONFIG_COUNT pairs stack on it instead of replacing it."""
        from src.core.kubernetes_container_provisioner import WORKSPACE_DIR

        env = self._analysis_env()
        assert env.get("GIT_CONFIG_PARAMETERS") == f"'safe.directory'='{WORKSPACE_DIR}'"
        assert "GIT_CONFIG_COUNT" not in env


# ---------------------------------------------------------------------------
# The workspace pod must be able to INSTALL from a deployment's private index.
# MEASURED in a live pod (codegraph-runtime, py3.14.7): the index resolves and
# serves HTTP 200, but TLS verification fails with UnknownIssuer, and nothing on
# the pod carried any UV_*/PIP_* setting.
# A cold `uv sync --frozen` of a 189-package lock: 11 s FAIL without this env,
# 6 s OK with it. Until this landed, no repo depending on a package published
# only to that index could be driven at all.
# ---------------------------------------------------------------------------


class TestWorkspaceReachesThePrivateIndex:
    def _analysis_env(self, **kwargs):
        prov = _make_provisioner()
        prov._create_deployment(
            name="idxprobe",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
            **kwargs,
        )
        deployment = prov.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]
        analysis = next(c for c in deployment.spec.template.spec.containers if c.name == "analysis")
        return {v.name: v.value for v in (analysis.env or [])}

    def test_trust_settings_are_present(self, monkeypatch):
        """Without these the resolver aborts: invalid peer certificate."""
        monkeypatch.setenv("CODING_PRIVATE_INDEX_HOST", "pkg.example.test")
        env = self._analysis_env()
        assert env.get("UV_INSECURE_HOST") == "pkg.example.test"
        assert env.get("PIP_TRUSTED_HOST") == "pkg.example.test"
        assert env.get("UV_NATIVE_TLS") == "1"

    def test_the_index_url_is_present_too(self, monkeypatch):
        """Trust alone is not enough -- an untrusted-but-unqueried host reads
        as `package not found`, which looks like a different bug entirely."""
        monkeypatch.setenv("CODING_PRIVATE_INDEX_HOST", "pkg.example.test")
        env = self._analysis_env()
        assert env.get("UV_EXTRA_INDEX_URL") == "https://pkg.example.test/simple/"
        assert env.get("PIP_EXTRA_INDEX_URL") == "https://pkg.example.test/simple/"

    def test_no_index_is_configured_by_default(self, monkeypatch):
        """A deployment without a private index injects nothing at all."""
        monkeypatch.delenv("CODING_PRIVATE_INDEX_HOST", raising=False)
        env = self._analysis_env()
        for key in (
            "UV_INSECURE_HOST",
            "PIP_TRUSTED_HOST",
            "UV_EXTRA_INDEX_URL",
            "PIP_EXTRA_INDEX_URL",
        ):
            assert key not in env, f"{key} injected without a configured index"

    def test_the_existing_git_and_npm_env_survives(self):
        """The new entries are spliced into the same list; a bad splice would
        drop what was already there and nothing else would notice."""
        env = self._analysis_env()
        assert "safe.directory" in env.get("GIT_CONFIG_PARAMETERS", "")
        assert env.get("NPM_CONFIG_USERCONFIG") == "/tmp/.npmrc"
        assert "HOME" in env

    def test_disabling_the_index_injects_nothing(self, monkeypatch):
        """CODING_PRIVATE_INDEX_HOST="" must add no vars, not empty ones --
        `https:///simple/` would break every install rather than skip the index."""
        monkeypatch.setenv("CODING_PRIVATE_INDEX_HOST", "")
        env = self._analysis_env()
        for key in (
            "UV_INSECURE_HOST",
            "PIP_TRUSTED_HOST",
            "UV_NATIVE_TLS",
            "UV_EXTRA_INDEX_URL",
            "PIP_EXTRA_INDEX_URL",
        ):
            assert key not in env, f"{key} injected while the mechanism is disabled"
        assert "safe.directory" in env.get("GIT_CONFIG_PARAMETERS", "")


# ---------------------------------------------------------------------------
# The Kubernetes provisioner must honour the PULL REQUEST it was asked to check
# out. MEASURED 2026-08-29: it silently discarded pr_number/provision_type.
# ---------------------------------------------------------------------------


class TestCloneCmdHonoursPullRequest:
    """`pr_number` was accepted, threaded, and then dropped on the floor.

    ContainerCreateRequest carries `pr_number` + `provision_type`; n8n's
    `Provision Verify Container` sends both; the Docker path honours them. This
    provisioner read NEITHER, so every k8s workspace cloned `branch or "main"`.

    It never crashed, which is why it survived. The AA `verification_test` hop
    provisions `verify-<card>` for an UNMERGED PR and re-runs the tests against
    it. Cloning main meant it verified the code the fix was meant to change:

      * for a bug fix -> unpassable. Measured: the probe reported
        `compute_total(2, 3) == -1` (i.e. `a - b`) while the PR under test had
        correctly changed it to `a + b`.
      * for a dependency chore -> WORSE THAN A FAILURE. main's suite passes
        without the bump, so the hop returned GREEN having tested nothing the
        card did. A vacuous pass is not a pass.
    """

    def test_pr_number_fetches_and_checks_out_the_pr_head(self):
        cmd = kcp._build_workspace_clone_cmd("github.com", "/owner/repo", None, pr_number=113)
        assert f"git {_CRED} clone --depth 1 " in cmd
        # the fetch authenticates through the same helper, not a URL
        assert f"git {_CRED} -c safe.directory=/workspace -C /workspace fetch --depth 1 " in cmd
        assert "fetch --depth 1 origin pull/113/head" in cmd
        assert "checkout FETCH_HEAD" in cmd

    def test_pr_number_wins_over_branch(self):
        """The PR is the thing under test; a base branch must not shadow it."""
        cmd = kcp._build_workspace_clone_cmd("github.com", "/owner/repo", "main", pr_number=113)
        assert "pull/113/head" in cmd
        assert "--branch main" not in cmd

    def test_no_pr_number_still_renders_a_plain_branch_clone(self):
        """NEGATIVE CONTROL: an existing caller must get the same CLONE it always got.

        ⚠️ This deliberately does NOT claim byte-identity with `origin/main` any more, because
        that claim became FALSE: `_idempotent` now wraps every path in a cleanup-and-exit-1 guard
        (see TestPartialCloneChain). The earlier version of this test asserted the new code equals
        ITSELF with `pr_number=None`, which is not a control at all — it passes no matter what the
        function does. What matters is asserted directly: same clone, no PR ref.
        """
        cmd = kcp._build_workspace_clone_cmd("github.com", "/owner/repo", "main")
        assert (
            f"git {_CRED} clone --depth 1 --branch main https://github.com/owner/repo /workspace"
            in cmd
        )
        assert "pull/" not in cmd
        assert "fetch" not in cmd

    def test_still_credentialled_and_idempotent_on_the_pr_path(self):
        """The PR path must not lose the guards the branch path has."""
        cmd = kcp._build_workspace_clone_cmd("github.com", "/owner/repo", None, pr_number=7)
        assert f"git {_CRED} clone" in cmd
        assert "@github.com" not in cmd
        assert "if [ ! -d /workspace/.git ]" in cmd

    def test_public_repo_pr_path_carries_no_credential(self):
        cmd = kcp._build_workspace_clone_cmd(
            "github.com", "/owner/repo", None, authenticated=False, pr_number=7
        )
        assert "GIT_TOKEN" not in cmd
        assert "pull/7/head" in cmd

    @pytest.mark.parametrize("bad", [0, -1, "113", 1.0, True])
    def test_rejects_a_pr_number_that_is_not_a_positive_int(self, bad):
        """Fail closed. `True` is an int in Python and would render `pull/True/head`."""
        with pytest.raises(ValueError, match="Invalid pr_number"):
            kcp._build_workspace_clone_cmd("github.com", "/owner/repo", None, pr_number=bad)


class TestCreateDeploymentThreadsPullRequest:
    def test_deployment_clone_cmd_checks_out_the_pr(self):
        prov = _make_provisioner()
        prov._create_deployment(
            name="verify-pod",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
            pr_number=113,
        )
        deployment = prov.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]
        init = deployment.spec.template.spec.init_containers[0]
        joined = " ".join(init.args or []) + " ".join(init.command or [])
        assert "pull/113/head" in joined, joined

    def test_without_a_pr_the_deployment_still_clones_the_branch(self):
        prov = _make_provisioner()
        prov._create_deployment(
            name="branch-pod",
            repo_url="https://github.com/owner/repo",
            branch="main",
            github_token="ghp_test",
        )
        deployment = prov.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]
        init = deployment.spec.template.spec.init_containers[0]
        joined = " ".join(init.args or []) + " ".join(init.command or [])
        assert "--branch main" in joined
        assert "pull/" not in joined


class TestPartialCloneChain:
    """A failed step in a multi-step chain must not leave a RETRYABLE GREEN.

    `_idempotent` guards on `[ ! -d /workspace/.git ]`. A multi-step chain can leave `.git`
    present and the work unfinished — `git clone` succeeds, then `fetch` or `checkout` fails. The
    next init-container attempt then sees `.git`, SKIPS the whole block, and the `if` exits 0, so
    the pod comes up Ready on the DEFAULT BRANCH with no error.

    That is the same vacuous-green the PR checkout exists to remove, re-entering through the retry
    path — and it applies to the SHA path too, which has always been multi-step.
    """

    def test_a_failure_empties_the_destination_and_exits_nonzero(self):
        cmd = kcp._build_workspace_clone_cmd("github.com", "/owner/repo", None, pr_number=113)
        assert "|| {" in cmd, "the chain must have a failure branch"
        assert "exit 1" in cmd, "a partial chain must fail LOUDLY, not skip on retry"
        # and the destination is emptied so a retry genuinely redoes the work
        assert cmd.count("find /workspace -mindepth 1 -delete") == 2
        assert cmd.partition("; fi && ")[0].count("-delete") == 2

    def test_the_guard_covers_the_sha_path_too(self):
        """The SHA path is clone-then-checkout — multi-step, same partial-state risk."""
        cmd = kcp._build_workspace_clone_cmd("github.com", "/owner/repo", "a" * 40)
        assert "checkout" in cmd and "exit 1" in cmd

    def test_the_branch_path_gets_it_as_well(self):
        cmd = kcp._build_workspace_clone_cmd("github.com", "/owner/repo", "main")
        assert "exit 1" in cmd


class TestProvisionContainerThreadsThePullRequest:
    """⛔ THE FIX WAS UNREACHABLE ON THE LIVE ROUTE UNTIL THIS.

    `POST /api/containers` -> `DockerService.create_container` -> `provision_container`, and
    `provision_container` REBUILDS a fresh `GitHubRepoRequest` from its own arguments. Any field
    it does not accept is dropped, so the caller's `pr_number` / `provision_type` — which
    `ContainerCreateRequest.parse_pr_url` populates from a PR url — never reached the clone
    builder. The builder was correct and could not fire.

    `parse_pr_url` also rewrites `repo_url` to the bare repo, so the PR is not recoverable
    downstream either: if it is dropped here it is gone.
    """

    @pytest.mark.asyncio
    async def test_the_pr_reaches_the_rebuilt_request(self):
        prov = _make_provisioner()
        captured = {}

        async def _capture(request):
            captured["req"] = request
            return MagicMock()

        prov.provision_from_github = _capture
        await prov.provision_container(
            container_name="verify-x",
            repo_url="https://github.com/owner/repo",
            branch="main",
            pr_number=113,
            provision_type="pull_request",
        )
        assert captured["req"].pr_number == 113
        assert captured["req"].provision_type == "pull_request"

    @pytest.mark.asyncio
    async def test_a_plain_branch_provision_carries_no_pr(self):
        prov = _make_provisioner()
        captured = {}

        async def _capture(request):
            captured["req"] = request
            return MagicMock()

        prov.provision_from_github = _capture
        await prov.provision_container(
            container_name="plain-x", repo_url="https://github.com/owner/repo", branch="main"
        )
        assert captured["req"].pr_number is None
        assert captured["req"].provision_type == "branch"

    @pytest.mark.asyncio
    async def test_an_unrecognised_provision_type_falls_back_to_branch(self):
        """Fail safe: an unexpected value must not become a Literal validation error at runtime."""
        prov = _make_provisioner()
        captured = {}

        async def _capture(request):
            captured["req"] = request
            return MagicMock()

        prov.provision_from_github = _capture
        await prov.provision_container(
            container_name="odd-x",
            repo_url="https://github.com/owner/repo",
            provision_type="nonsense",
        )
        assert captured["req"].provision_type == "branch"


class TestGitCNeedsSafeDirectory:
    """Every `git -C {dest}` must carry `-c safe.directory={dest}`.

    MEASURED 2026-08-31 in production. The emptyDir mount point is ROOT-owned (the kubelet
    creates it root:fsGroup) while the workspace runs as uid 1000. `git clone` into it succeeds —
    it writes files as 1000 — but any subsequent `git -C {dest} …` checks the ownership of the
    repo DIRECTORY and refuses:

        Cloning into '/workspace'...
        fatal: detected dubious ownership in repository at '/workspace'

    That crash-looped the `git-clone` init container on verify-1853413860308944792
    (Init:CrashLoopBackOff x14), which failed `Assert Verify Environment`, errored
    `AA Pattern - Done Finalize`, and left a card finalized with a PR whose next hop never minted.

    The single-command BRANCH path never runs `git -C`, so it was never affected — which is why
    ordinary provisioning kept working and only the PR/SHA paths broke.
    """

    def test_the_pr_path_declares_the_directory_safe(self):
        cmd = kcp._build_workspace_clone_cmd("github.com", "/o/r", None, pr_number=117)
        # every `git -C` invocation, not just the first
        assert cmd.count("git -C /workspace") == 0, "a bare `git -C` will hit dubious ownership"
        # fetch + checkout, plus the three in the post-clone credential scrub
        assert cmd.count(" -C /workspace") == 5
        assert cmd.count("-c safe.directory=/workspace -C /workspace") == 5

    def test_the_sha_path_declares_it_too(self):
        """The SHA path always had this bug; it was masked by the silent retry."""
        cmd = kcp._build_workspace_clone_cmd("github.com", "/o/r", "a" * 40)
        assert "git -C /workspace" not in cmd
        assert "git -c safe.directory=/workspace -C /workspace checkout" in cmd

    def test_the_branch_path_needs_no_safe_directory(self):
        """NEGATIVE CONTROL: it is a single `git clone`, so it never reads a repo it does not own.

        If this ever starts needing safe.directory, the branch path has grown a `git -C` and the
        reasoning above must be revisited rather than the flag sprinkled on.
        """
        cmd = kcp._build_workspace_clone_cmd("github.com", "/o/r", "main")
        clone_part, _, scrub = cmd.partition("; fi && ")
        assert "git -C" not in cmd
        assert "safe.directory" not in clone_part
        assert "clone --depth 1 --branch main " in clone_part
        # The post-clone scrub does read the repo, so IT declares the directory.
        assert " -C /workspace" in scrub
        assert scrub.count(" -C /workspace") == scrub.count("-c safe.directory=/workspace -C ")

    def test_no_chown_or_chmod_was_reached_for(self):
        """⛔ The documented fix for a root-owned mount here is NOT to change modes.

        Three production incidents came from touching modes on this mount (see the builder's
        docstring). `safe.directory` states the exception without mutating anything.
        """
        for kw in ({}, {"pr_number": 7}, {}):
            cmd = kcp._build_workspace_clone_cmd("github.com", "/o/r", "main", **kw)
            assert "chmod" not in cmd
            assert "chown" not in cmd

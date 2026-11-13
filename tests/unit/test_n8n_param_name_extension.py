"""``x-n8n-param-name`` reaches the published OpenAPI spec for restore and branch.

The n8n node generator (n8n_module_scripts generate-node.ts) flattens every body
field into one parameter namespace per node. Without a distinct name, restore's
``files`` merges with commitWorkspaceChanges.files and branch's ``branch`` merges
with createWorkspace.branch. The extension must survive into ``app.openapi()``,
which is what the aggregator and generator read. Asserting on the model alone
would not prove that.
"""

from __future__ import annotations

from typing import Any

import pytest


def _operation(spec: dict[str, Any], operation_id: str) -> dict[str, Any]:
    for path_item in spec["paths"].values():
        for op in path_item.values():
            if isinstance(op, dict) and op.get("operationId") == operation_id:
                return op
    raise AssertionError(f"operationId {operation_id!r} not in the spec")


def _request_schema(spec: dict[str, Any], operation_id: str) -> dict[str, Any]:
    op = _operation(spec, operation_id)
    ref = op["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    assert ref.startswith("#/components/schemas/"), ref
    return spec["components"]["schemas"][ref.rsplit("/", 1)[-1]]


@pytest.mark.parametrize(
    ("operation_id", "field", "expected"),
    [
        ("restore_workspace_files", "files", "restoreFiles"),
        ("ensure_workspace_branch", "branch", "targetBranch"),
        # coding-containers#156: the snapshot binding (a git/diff response's `tree`).
        ("commit_workspace_changes", "approved_tree", "commitApprovedTree"),
        ("commit_workspace_changes", "approved_base", "commitApprovedBase"),
        ("push_workspace_to_fork", "approved_tree", "pushApprovedTree"),
        ("push_workspace_to_fork", "target_blobs", "pushTargetBlobs"),
        # R3b: git/fetch body fields must not merge with push/create fields.
        ("fetch_workspace_ref", "remote_url", "fetchRemoteUrl"),
        ("fetch_workspace_ref", "refspec", "fetchRefspec"),
        ("fetch_workspace_ref", "depth", "fetchDepth"),
        # Async exec: timeout's range differs from the sync exec's (le=300).
        ("exec_in_workspace_async", "timeout", "asyncExecTimeout"),
        ("exec_in_workspace_async", "callback_url", "execCallbackUrl"),
        # P7 multi-repo probe: every body field would otherwise merge with
        # create_workspace / exec / fetch fields of the same name.
        ("add_workspace_repo", "repo_url", "secondaryRepoUrl"),
        ("add_workspace_repo", "ref", "secondaryRepoRef"),
        ("add_workspace_repo", "refresh", "secondaryRepoRefresh"),
        ("start_workspace_service", "command", "serviceCommand"),
        ("start_workspace_service", "workdir", "serviceWorkdir"),
        ("start_workspace_service", "port", "servicePort"),
        ("start_workspace_service", "env", "serviceEnv"),
        ("start_workspace_service", "readiness", "serviceReadiness"),
        ("start_workspace_service", "ready_timeout", "serviceReadyTimeout"),
        ("start_workspace_service", "max_lifetime_seconds", "serviceMaxLifetime"),
        ("probe_workspace_service", "method", "probeMethod"),
        ("probe_workspace_service", "path", "probePath"),
        ("probe_workspace_service", "headers", "probeHeaders"),
        ("probe_workspace_service", "body", "probeBody"),
        ("probe_workspace_service", "timeout", "probeTimeout"),
    ],
)
def test_request_field_carries_n8n_param_name(app, operation_id, field, expected) -> None:
    app.openapi_schema = None  # force a fresh build, never a cached spec
    spec = app.openapi()
    prop = _request_schema(spec, operation_id)["properties"][field]
    assert prop.get("x-n8n-param-name") == expected


def test_schema_names_are_the_request_models(app) -> None:
    """Pin the component names the generator resolves, as in the brief."""
    app.openapi_schema = None
    schemas = app.openapi()["components"]["schemas"]
    assert schemas["GitBranchRequest"]["properties"]["branch"]["x-n8n-param-name"] == (
        "targetBranch"
    )
    assert schemas["GitRestoreRequest"]["properties"]["files"]["x-n8n-param-name"] == (
        "restoreFiles"
    )


def test_unrelated_same_named_fields_are_untouched(app) -> None:
    """Control: commit's ``files`` must keep its plain name, or the rename is global."""
    app.openapi_schema = None
    spec = app.openapi()
    commit_files = _request_schema(spec, "commit_workspace_changes")["properties"]["files"]
    assert "x-n8n-param-name" not in commit_files

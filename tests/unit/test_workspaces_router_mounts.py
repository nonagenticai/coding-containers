"""Guards on how the workspace surface is assembled.

The canonical ``/api/workspaces`` API and the hidden ``/api/containers``
compatibility alias are built from the same five router objects (see
``src/routers/workspaces/legacy_compat.py``). That sharing is deliberate for the
routes that predate the workspace vocabulary, and a trap for every route added
after them: a new route dropped into one of those five routers is republished
onto the legacy surface automatically, with no diff to review that says so.

These tests make the invariant checkable instead of a comment. The snapshot in
particular is what lets the workspace IDs be declared explicitly on their
decorators rather than derived at import time by a ``str.replace`` loop over
``router.routes`` — the loop kept the two surfaces in sync by construction, so
removing it needs something else to notice if they drift.
"""

from __future__ import annotations

import pytest

LEGACY_PREFIX = "/api/containers"
CANONICAL_PREFIX = "/api/workspaces"

# The published (method, path) -> operationId contract. Generated clients key
# off these strings, coding-gateway's ``operation_metadata.py`` attaches role
# metadata to them, and a CI drift guard there fails on a mismatch, so a change
# here is a cross-repo change and should never be a silent one.
PUBLISHED_OPERATIONS = {
    ("DELETE", "/api/workspaces/all"): "delete_all_workspaces",
    ("DELETE", "/api/workspaces/{container_name}"): "delete_workspace",
    ("DELETE", "/api/workspaces/{container_name}/files"): "delete_workspace_path",
    ("DELETE", "/api/workspaces/{container_name}/services/{service_id}"): (
        "stop_workspace_service"
    ),
    ("GET", "/api/workspaces/"): "list_workspaces",
    # GET /api/workspaces/all is deliberately absent: it is an
    # include_in_schema=False guard that returns 405 so a GET against the
    # bulk-delete path fails loudly instead of matching /{container_name}.
    ("GET", "/api/workspaces/{container_name}"): "get_workspace",
    ("GET", "/api/workspaces/{container_name}/agent-capabilities"): (
        "get_workspace_agent_capabilities"
    ),
    ("GET", "/api/workspaces/{container_name}/dependency-status"): (
        "get_workspace_dependency_status"
    ),
    ("GET", "/api/workspaces/{container_name}/files"): "list_workspace_files",
    ("GET", "/api/workspaces/{container_name}/exec/jobs/{job_id}"): "get_exec_job_status",
    ("GET", "/api/workspaces/{container_name}/git/status"): "get_workspace_git_status",
    ("GET", "/api/workspaces/{container_name}/git/diff"): "get_workspace_git_diff",
    ("GET", "/api/workspaces/{container_name}/repos"): "list_workspace_repos",
    ("GET", "/api/workspaces/{container_name}/search"): "search_workspace",
    ("GET", "/api/workspaces/{container_name}/services"): "list_workspace_services",
    ("GET", "/api/workspaces/{container_name}/services/{service_id}"): "get_workspace_service",
    ("GET", "/api/workspaces/{container_name}/status"): "get_workspace_status",
    ("POST", "/api/workspaces/"): "create_workspace",
    ("POST", "/api/workspaces/{container_name}/exec"): "exec_in_workspace",
    ("POST", "/api/workspaces/{container_name}/exec/async"): "exec_in_workspace_async",
    ("POST", "/api/workspaces/{container_name}/execute-agent"): "execute_workspace_agent",
    ("POST", "/api/workspaces/{container_name}/git/branch"): "ensure_workspace_branch",
    ("POST", "/api/workspaces/{container_name}/git/commit"): "commit_workspace_changes",
    ("POST", "/api/workspaces/{container_name}/git/fetch"): "fetch_workspace_ref",
    ("POST", "/api/workspaces/{container_name}/git/push"): "push_workspace_to_fork",
    ("POST", "/api/workspaces/{container_name}/git/restore"): "restore_workspace_files",
    ("POST", "/api/workspaces/{container_name}/install-dependencies"): (
        "install_workspace_dependencies"
    ),
    ("POST", "/api/workspaces/{container_name}/mkdir"): "create_workspace_directory",
    ("POST", "/api/workspaces/{container_name}/move"): "move_workspace_path",
    ("POST", "/api/workspaces/{container_name}/package-modify"): "package_modify_in_workspace",
    ("POST", "/api/workspaces/{container_name}/package-upgrade"): "package_upgrade_in_workspace",
    ("POST", "/api/workspaces/{container_name}/repos"): "add_workspace_repo",
    ("POST", "/api/workspaces/{container_name}/services"): "start_workspace_service",
    ("POST", "/api/workspaces/{container_name}/services/{service_id}/probe"): (
        "probe_workspace_service"
    ),
    ("POST", "/api/workspaces/{container_name}/touch"): "create_workspace_file",
    ("POST", "/api/workspaces/{container_name}/write-file"): "write_workspace_file",
}


def _published_operations(app) -> dict[tuple[str, str], str]:
    schema = app.openapi()
    return {
        (method.upper(), path): operation["operationId"]
        for path, path_item in schema["paths"].items()
        for method, operation in path_item.items()
        if isinstance(operation, dict) and "operationId" in operation
    }


def _route_contexts(routes):
    """Yield (path, methods) for every route reachable from ``routes``.

    fastapi>=0.137 no longer clones routes on ``include_router()``: the app's
    ``.routes`` list holds a private ``_IncludedRouter`` wrapper (no ``.path``
    at all) around the shared, un-prefixed original route object
    (https://github.com/fastapi/fastapi/releases/tag/0.137.0). Reading
    ``route.path``/``route.methods`` directly off ``app.routes`` either
    crashes on the wrapper or, on the original route object, returns the
    route's own declared path rather than where it is actually mounted.
    ``fastapi.routing.iter_route_contexts`` (public, added 0.137.2) is the
    accessor that resolves either shape to the real, fully-prefixed path.
    """
    try:
        from fastapi.routing import iter_route_contexts
    except ImportError:  # fastapi < 0.137.2: app.routes is already flat
        for route in routes:
            if hasattr(route, "path"):
                yield route.path, getattr(route, "methods", None) or ()
        return
    for ctx in iter_route_contexts(routes):
        if ctx.path is not None:
            yield ctx.path, ctx.methods or ()


def _mounted_routes(app) -> set[tuple[str, str]]:
    mounted = set()
    for path, methods in _route_contexts(app.routes):
        for method in methods:
            if method in {"HEAD", "OPTIONS"}:
                continue
            mounted.add((method, path))
    return mounted


def test_published_workspace_operations_match_snapshot(app):
    """The canonical contract is pinned, so a rename cannot land unnoticed.

    Regenerate deliberately if a route is genuinely added or renamed, and expect
    to refresh ``coding-gateway/tests/data/coding_operation_index.json`` via its
    ``scripts/refresh_operation_index.py`` in the same unit of work.
    """
    published = _published_operations(app)
    workspace_ops = {
        key: value for key, value in published.items() if key[1].startswith(CANONICAL_PREFIX)
    }
    assert workspace_ops == PUBLISHED_OPERATIONS


def test_every_legacy_route_has_a_canonical_twin(app):
    """The alias may never outlive the surface it aliases."""
    mounted = _mounted_routes(app)
    legacy = {(method, path) for method, path in mounted if path.startswith(LEGACY_PREFIX)}
    assert legacy, "legacy alias disappeared; delete these guards deliberately, not by accident"

    missing = {
        (method, path)
        for method, path in legacy
        if (method, path.replace(LEGACY_PREFIX, CANONICAL_PREFIX, 1)) not in mounted
    }
    assert not missing, f"legacy routes with no canonical twin: {sorted(missing)}"


def test_legacy_alias_is_not_published(app):
    """``/api/containers`` stays callable and stays out of generated clients."""
    schema = app.openapi()
    assert not any(path.startswith(LEGACY_PREFIX) for path in schema["paths"])


def test_no_published_operation_is_container_named(app):
    """The published vocabulary is 'workspace'; the runtime is an implementation detail.

    Health and validation routers are mounted elsewhere and keep their own
    container-scoped names, so this is scoped to the workspace surface.
    """
    published = _published_operations(app)
    offenders = {
        key: value
        for key, value in published.items()
        if key[1].startswith(CANONICAL_PREFIX) and "container" in value
    }
    assert not offenders, f"container-named operation IDs on the canonical surface: {offenders}"


@pytest.mark.parametrize(
    "module",
    ["src.routers.workspaces", "src.routers.workspaces.legacy_compat"],
)
def test_workspaces_package_exposes_only_a_router(module):
    """Keeps the package a mounting point, not a place handlers accumulate."""
    imported = __import__(module, fromlist=["__all__"])
    assert imported.__all__ == ["router"]

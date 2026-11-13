"""Graph deletion for container lifecycle cleanup.

coding-analysis owns the code graph (Postgres). This module is the one place
coding-containers asks it to drop a container's graph when the container goes
away.

⚠️ TWO THINGS HERE WERE BROKEN FOR EVERY CONTAINER EVER DELETED — do not
reintroduce either:

1. **It went through coding-gateway.** ``DELETE {main_api_endpoint}/api/graphs/…``
   is a *north-south* edge whose AuthorizationPolicies admit only istio-system /
   the master IP / the UI namespaces — never a ``coding-*`` namespace. Every call
   was reset by the mesh (``[Errno 104] Connection reset by peer``), which reads
   like a network blip rather than a policy denial. Same defect class as the
   eager code-graph build (#87). Address coding-analysis DIRECTLY.

2. **The path did not exist anyway.** coding-analysis serves graphs under
   ``/api/analysis/graphs/{container_name}``; ``/api/graphs/{name}`` is not a
   prefix coding-gateway's ``PROXY_ROUTES`` carries either.

The local-sqlite fallback that covered for (1) is gone with it: it looked for
per-repo ``.db`` files that are monolith-era state, so in k8s it always found
nothing and returned ``0 nodes, 0 edges`` — reported by the caller as a
successful deletion. Live proof this mattered: after two probe containers were
provisioned and deleted, ``GET /api/analysis/graphs`` still listed both, 12
nodes / 10 edges each.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.clients._tenant import service_token, tenant_headers
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

_graph_operations = None

# Deleting a graph is two indexed DELETEs plus the container-ownership check.
_DELETE_TIMEOUT_SECONDS = 30.0


class GraphOperations:
    """Delete graph data associated with container lifecycle events."""

    async def delete_container_graph(
        self, container_name: str, tenant_id: str | None = None
    ) -> dict[str, Any]:
        """Ask coding-analysis to drop ``container_name``'s graph.

        ⚠️ ORDERING IS LOAD-BEARING: coding-analysis validates that the container
        exists *for this tenant* before it will delete the graph, so this must run
        BEFORE the container is removed from the registry. Called afterwards it
        gets a 404 ("Container not found") and the graph rows are orphaned
        forever — with no way to reach them again, since the delete endpoint is
        the only thing that removes them and it needs the container to exist.

        Never raises: returns ``success=False`` with a reason so the caller can
        report the truth instead of a fabricated "deleted 0 nodes".
        """
        settings = get_settings()
        base_url = (settings.coding_analysis_url or "").rstrip("/")
        if not base_url:
            logger.error(
                "coding_analysis_url is unset — cannot delete the graph for %s; "
                "its nodes/edges will be orphaned in coding-analysis.",
                container_name,
            )
            return {
                "success": False,
                "message": "coding_analysis_url is not configured",
                "graph_name": container_name,
                "deleted_nodes": 0,
                "deleted_edges": 0,
            }

        if not tenant_id:
            # The endpoint is tenant-scoped; without a tenant the call can only
            # 401/422. Say so rather than firing a request that cannot work.
            logger.error(
                "No tenant for graph deletion of %s — coding-analysis requires "
                "X-Tenant-Id; the graph will be orphaned.",
                container_name,
            )
            return {
                "success": False,
                "message": "no tenant_id available for graph deletion",
                "graph_name": container_name,
                "deleted_nodes": 0,
                "deleted_edges": 0,
            }

        url = f"{base_url}/api/analysis/graphs/{container_name}"
        # The tenant is the container OWNER's, not this service's: coding-analysis
        # resolves the graph against it. Under TRUST_TENANT_HEADER=false a bare
        # X-Tenant-Id proves nothing, so this service's own licence rides in
        # Authorization and coding-analysis honours the asserted tenant iff
        # coding-containers' subject is in its CODING_ONBEHALF_SUBJECTS.
        # tenant_headers (not optional_) because the tenant SCOPES the deletion —
        # a missing one must never be defaulted into deleting another tenant's
        # graph. The early return above already covers it, so the raise is a
        # backstop against that guard being removed.
        headers = tenant_headers(
            tenant_id,
            operation=f"graph deletion for {container_name}",
            service_token=service_token(),
        )
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(_DELETE_TIMEOUT_SECONDS, connect=5.0)
            ) as client:
                response = await client.delete(url, headers=headers)
        except httpx.HTTPError as exc:
            logger.error(
                "Graph deletion request to coding-analysis failed for %s: %s", container_name, exc
            )
            return {
                "success": False,
                "message": f"coding-analysis unreachable: {exc}",
                "graph_name": container_name,
                "deleted_nodes": 0,
                "deleted_edges": 0,
            }

        if response.status_code == 404:
            # coding-analysis 404s both "this container has no graph rows" and
            # "no such container for this tenant". Either way there is nothing
            # left to delete, so this is a genuine success — but it is recorded
            # distinctly so it can never be read as "we deleted a graph".
            logger.info("No graph to delete for %s (coding-analysis 404)", container_name)
            return {
                "success": True,
                "message": f"Graph {container_name} already absent",
                "graph_name": container_name,
                "deleted_nodes": 0,
                "deleted_edges": 0,
                "already_absent": True,
            }

        if response.status_code >= 400:
            logger.error(
                "coding-analysis refused graph deletion for %s: HTTP %s %s",
                container_name,
                response.status_code,
                response.text[:300],
            )
            return {
                "success": False,
                "message": f"coding-analysis returned HTTP {response.status_code}",
                "graph_name": container_name,
                "deleted_nodes": 0,
                "deleted_edges": 0,
            }

        try:
            payload = response.json()
        except ValueError as exc:
            logger.error(
                "coding-analysis returned a non-JSON graph-deletion body for %s: %s",
                container_name,
                exc,
            )
            return {
                "success": False,
                "message": "coding-analysis returned a non-JSON body",
                "graph_name": container_name,
                "deleted_nodes": 0,
                "deleted_edges": 0,
            }

        return {
            "success": True,
            "message": f"Graph {container_name} deleted",
            "graph_name": container_name,
            "deleted_nodes": payload.get("nodes_deleted", 0),
            "deleted_edges": payload.get("edges_deleted", 0),
        }


def get_graph_operations() -> GraphOperations:
    """Get singleton graph operations instance."""
    global _graph_operations
    if _graph_operations is None:
        _graph_operations = GraphOperations()
    return _graph_operations

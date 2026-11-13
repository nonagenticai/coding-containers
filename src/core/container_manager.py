"""
Container manager module for provisioning Docker containers with GitHub repositories.

This module provides a streamlined interface to the refactored container management system,
handling Docker container provisioning, lifecycle management, and integration with GitHub
repositories. It serves as a facade over the core container management functionality.

The container manager handles:
- Container provisioning from GitHub repositories
- Container lifecycle management (start, stop, remove)
- Container registry integration and persistence
- Container status monitoring and health checks
- Code graph metadata management
- Database path scanning and management

Key Features:
    - GitHub repository integration for automatic container provisioning
    - Persistent container registry with JSON-based storage
    - Container health monitoring and status reporting
    - Code graph association and metadata management
    - Database path discovery and scanning
    - Fallback handling for Docker unavailability

Classes:
    ContainerManager: Main facade class for container operations

Functions:
    get_container_manager: Singleton factory function for ContainerManager

Usage:
    >>> manager = get_container_manager()
    >>> container = manager.provision_from_github(github_request)
    >>> status = manager.get_container_status(container.container_name)
    >>> containers = manager.list_containers()

Note:
    This module acts as a facade over the core container management system,
    providing a simplified interface while delegating complex operations to
    specialized components like CoreContainerManager and GitHubContainerProvisioner.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from kubernetes.client.rest import ApiException

from src.clients.github_client import (
    is_platform_owner,
    repo_owner_from_url,
)
from src.config.settings import get_settings
from src.models.container_models import ContainerInfo, ContainerStatus
from src.runtime import ExecutionProfile, resolve_runtime
from src.services.docker_service import get_docker_service

from .container_manager_core import (
    get_container_manager as get_core_manager,  # TODO: resolve cross-service dependency - this module may need to be extracted too
)
from .container_spec import (
    GitHubRepoRequest,  # TODO: resolve cross-service dependency - this module may need to be extracted too
)
from .github_container_provisioner import (
    get_github_container_provisioner,  # TODO: resolve cross-service dependency - this module may need to be extracted too
)
from .kubernetes_container_provisioner import (
    PLATFORM_TENANT_MISSING,
    PLATFORM_TENANT_MISSING_EVENT,
    TENANT_ANNOTATION,
    TENANT_LABEL_KEY,
    ContainerDeploymentNotFoundError,
)
from .log_safety import redact_secrets, summarize_command

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bounded recovery for a REAPED Deployment (see ``ensure_container``)
# ---------------------------------------------------------------------------
# The reaper deletes a Deployment on a 24h TTL with no in-use guard
# (``src/core/reaper.py`` states this itself), while the metadata registry is a
# Redis hash the reaper never touches. So the recoverable state is exactly
# "registry entry present, Deployment gone", and a re-provision by the stored
# name is possible *because* the name is deterministic and the spec survived.
#
# ⛔ These guards are what stops a 404 from spinning. A reaped name is
# re-provisioned AT MOST ONCE per cooldown window, and never twice
# concurrently; everything else fails loudly with a "not found" message so the
# HTTP mapping stays an honest 404 rather than a silent success.
REPROVISION_COOLDOWN_S = 300.0
# Names with a provision in flight right now. Claimed and released with NO
# ``await`` between the check and the mutation, so on a single event loop this
# is atomic without a lock. Across replicas the bound is Kubernetes itself: a
# Deployment name is unique in the namespace and the provisioner's 409 path
# reconciles in place (patch) instead of creating a second object.
_REPROVISION_INFLIGHT: set[str] = set()
# name -> time.monotonic() of the last attempt (success or failure).
_REPROVISION_LAST_ATTEMPT: dict[str, float] = {}


def _reset_reprovision_guards() -> None:
    """Clear the re-provision guards. Test hook only."""
    _REPROVISION_INFLIGHT.clear()
    _REPROVISION_LAST_ATTEMPT.clear()


def _map_k8s_container_status(status: str | None) -> ContainerStatus:
    """Map the K8s provisioner's status dict value onto the ContainerStatus enum.

    The provisioner returns ``running`` once ready_replicas satisfy the desired
    count, ``starting``/``pending`` while the deployment is still rolling out, and
    ``not_found`` when the deployment is gone. "Still provisioning" must surface as
    CREATING (not ERROR) so the orchestration layer can keep polling instead of
    treating a healthy-but-not-ready container as a hard failure.
    """
    if status == "running":
        return ContainerStatus.RUNNING
    if status in ("starting", "pending"):
        return ContainerStatus.CREATING
    return ContainerStatus.ERROR


class ContainerManager:
    """Streamlined facade for container operations and lifecycle management.

    This class provides a unified interface for all container-related operations,
    serving as a facade over the core container management system. It handles
    container provisioning, lifecycle management, and integration with various
    services like GitHub repositories and code graph databases.

    The manager delegates complex operations to specialized components while
    providing a simple, consistent interface for container operations. It manages
    the relationship between containers and code graphs, handles container
    registry persistence, and provides fallback mechanisms for various failure
    scenarios.

    Attributes:
        _core (CoreContainerManager): Core container management functionality
        _github (GitHubContainerProvisioner): GitHub repository provisioning

    Responsibilities:
        - Container provisioning from GitHub repositories
        - Container lifecycle management (create, start, stop, remove)
        - Container registry management and persistence
        - Container status monitoring and health checks
        - Code graph metadata association
        - Database path discovery and scanning

    Example:
        >>> manager = ContainerManager()
        >>> # Provision from GitHub
        >>> request = GitHubRepoRequest(repo_url="https://github.com/user/repo")
        >>> container = manager.provision_from_github(request)
        >>>
        >>> # Monitor container status
        >>> status = manager.get_container_status(container.container_name)
        >>>
        >>> # List all containers
        >>> containers = manager.list_containers()
        >>>
        >>> # Remove container
        >>> result = manager.remove_container(container.container_name)
    """

    def __init__(self, graph_deletion_callback: Callable | None = None) -> None:
        """Initialize the container manager with core and GitHub components.

        Creates a new ContainerManager instance with initialized core
        container management and GitHub provisioning capabilities.
        This sets up the facade pattern over the specialized components.

        Args:
            graph_deletion_callback: Optional callback for graph deletion operations.
                                   If not provided, will use default graph service.

        Initialization Process:
            1. Creates or retrieves the core container manager
            2. Creates or retrieves the GitHub container provisioner
            3. Establishes the facade pattern over these components

        Side Effects:
            - Initializes Docker client connection (if available)
            - Sets up container registry access
            - Configures GitHub integration capabilities
        """
        self._core = get_core_manager()
        self._github = get_github_container_provisioner()
        self._graph_deletion_callback = graph_deletion_callback

    def _scan_database_path(self, base_path: str, graph_summaries: list[dict]) -> None:
        """Scan a database path for standalone graph files.

        Args:
            base_path: The base path to scan
            graph_summaries: List to append found graphs to
        """
        try:
            db_dir = Path(base_path)
            if not db_dir.exists():
                return
            for db_file in db_dir.glob("*.db"):
                graph_id = db_file.stem
                if any(g["id"] == graph_id for g in graph_summaries):
                    continue
                graph_info = {
                    "id": graph_id,
                    "graph_id": graph_id,
                    "name": graph_id,
                    "description": f"Standalone code graph: {graph_id}",
                    "language": "python",
                    "created_at": "",
                    "updated_at": "",
                    "tags": [],
                    "status": "available",
                    "is_containerized": False,
                    "container_name": None,
                    "repository_url": "",
                }
                graph_summaries.append(graph_info)
        except OSError as e:
            logger.debug("Error scanning %s: %s", base_path, e)

    @property
    def client(self):
        """Access to Docker client for testing compatibility."""
        return self._core.client

    @property
    def base_port(self):
        """Access to base port for testing compatibility."""
        return self._core.base_port

    async def provision_from_github(self, request: GitHubRepoRequest) -> ContainerInfo:
        """Provision a container from a GitHub repository using dual-token architecture.

        Args:
            request: The GitHub repository request

        Returns:
            ContainerInfo object for the provisioned container
        """
        return await self._github.provision_from_github(request)

    async def remove_container(
        self, name: str, force: bool = False, tenant_id: str | None = None
    ) -> dict[str, Any]:
        """Remove a container by name and its associated code graph.

        ⚠️ THIS IS ``async`` FOR A REASON. It used to be a sync method that
        called ``asyncio.run_coroutine_threadsafe(...).result(timeout=N)`` for
        both the k8s teardown and the graph deletion. Every caller is an ``async``
        route handler, so those ran ON the event-loop thread: ``.result()``
        blocked the very loop that had to execute the coroutine. Both waits were
        therefore *guaranteed* to time out — deterministically, not under load.
        Live evidence from a single DELETE: 30.0s to the k8s timeout, then 5.0s
        to the graph timeout, a 35.3s request, two logged "failures" that had not
        actually failed, and the scheduled coroutines finally running only after
        the handler returned. Do not make this sync again.

        ⚠️ ORDER: the graph is deleted BEFORE the registry entry. coding-analysis
        validates that the container exists for the tenant before dropping its
        graph, so doing it the other way round 404s and orphans the rows.

        Args:
            name: The container name to remove
            force: If True, force removal even if container is running
            tenant_id: If provided, validates tenant access before removal.
                      Returns error if container belongs to different tenant.

        Returns:
            Dict with status, message, and graph deletion details
        """
        docker_service = get_docker_service()
        try:
            registry_data = docker_service.registry.load()
        except (OSError, KeyError) as e:
            # An unreadable registry must not abort the teardown: the k8s
            # Deployment still has to go. Proceed with no metadata.
            logger.warning("Could not read the container registry for %s: %s", name, e)
            registry_data = {}
        container_info = registry_data.get(name)

        # Validate tenant access if tenant_id is provided
        if tenant_id is not None and container_info is not None:
            container_tenant = getattr(container_info, "tenant_id", None)
            if container_tenant is not None and container_tenant != tenant_id:
                # Return "not found" style error to prevent enumeration
                return {
                    "status": "error",
                    "message": f"Container {name} not found",
                }

        # Resolve the graph identity while the registry entry still exists.
        graph_id = getattr(container_info, "code_graph_id", None) or (
            name if container_info is not None else None
        )
        graph_tenant_id = tenant_id or getattr(container_info, "tenant_id", None)

        # 1. Graph first — coding-analysis needs the container to still exist.
        graph_deletion_info = await self._delete_graph(name, graph_id, graph_tenant_id)

        # 2. K8s teardown.
        if self._github.use_k8s and self._github.k8s_provisioner:
            try:
                await self._github.k8s_provisioner.remove_container(name)
                logger.info("Deleted K8s resources for %s", name)
            except (ApiException, OSError, RuntimeError, TimeoutError) as e:
                logger.warning("Failed to delete K8s resources for %s: %s", name, e)

        # 3. Docker/core teardown + registry.
        success = self._core.remove_container(name, force=force)
        try:
            # Atomic single-field HDEL. NOT load()+save(): save() HDELs every
            # field absent from its payload, so tearing one container down
            # through a whole-registry snapshot deletes any container a sibling
            # replica created since that snapshot was taken.
            if docker_service.registry.remove(name):
                logger.info("Removed container %s from registry", name)
            else:
                logger.debug("Container %s was not found in registry", name)
        except (OSError, KeyError) as e:
            logger.warning("Failed to clean up registry for container %s: %s", name, e)

        result: dict[str, Any] = {
            "status": "success" if success else "error",
            "message": f"Container {name} {('removed' if success else 'removal failed')}",
        }
        if graph_deletion_info:
            result["graph_deletion"] = graph_deletion_info
        return result

    async def _delete_graph(
        self, name: str, graph_id: str | None, graph_tenant_id: str | None
    ) -> dict[str, Any] | None:
        """Drop the container's code graph; report honestly what happened.

        Returns ``None`` only when there is no graph identity to delete. Never
        raises — a graph that cannot be deleted must not block the container
        teardown, but it must also never be reported as deleted.
        """
        if not graph_id:
            return None

        try:
            if self._graph_deletion_callback:
                result = await self._graph_deletion_callback(graph_id, force=True)
            else:
                # Local import to avoid a circular dependency at module scope.
                from .graph_operations import get_graph_operations

                result = await get_graph_operations().delete_container_graph(
                    graph_id, tenant_id=graph_tenant_id
                )
        except (httpx.HTTPError, OSError, RuntimeError, TimeoutError) as e:
            logger.error("Graph deletion for container %s failed: %s", name, e, exc_info=True)
            return {"status": "error", "error": str(e)}

        if result.get("success"):
            if result.get("already_absent"):
                logger.info("No graph existed for container %s", name)
                return {"status": "already_absent", "deleted_nodes": 0, "deleted_edges": 0}
            logger.info(
                "Deleted graph for container %s: %s nodes, %s edges",
                name,
                result.get("deleted_nodes", 0),
                result.get("deleted_edges", 0),
            )
            return {
                "status": "deleted",
                "deleted_nodes": result.get("deleted_nodes", 0),
                "deleted_edges": result.get("deleted_edges", 0),
            }

        # NOT a success. Previously a 5s timeout here returned
        # {"success": True, "message": "Graph deletion initiated in background"},
        # so an orphaned graph was logged as a clean deletion.
        logger.error(
            "Failed to delete graph for container %s: %s — its nodes/edges are now "
            "ORPHANED in coding-analysis (the delete endpoint needs the container to exist).",
            name,
            result.get("message"),
        )
        return {"status": "failed", "error": result.get("message")}

    def list_containers(self, tenant_id: str | None = None) -> list[ContainerInfo]:
        """List containers, optionally filtered by tenant.

        Args:
            tenant_id: If provided, only return containers belonging to this tenant.
                      If None, returns all containers (for admin use or legacy code).

        Returns:
            List of ContainerInfo objects
        """
        docker_service = get_docker_service()
        registry_data = docker_service.registry.load()
        containers = list(registry_data.values())

        # The JSON registry lives on ephemeral pod-local storage and is lost on
        # redeploy/reschedule while the k8s Deployments persist, so it can
        # under-report a still-running container (previously surfaced to callers
        # as "Available containers: []" for a live container). Mirror the durable
        # source of truth the get_container miss-path and the reaper already use:
        # enumerate live managed Deployments and union in any whose name the
        # registry is missing, rebuilding each via reconstruct_container_info.
        # Registry entries win on name conflict (they carry richer/authoritative
        # metadata incl. tenant_id). Guarded so a k8s hiccup degrades to the
        # JSON-only list rather than failing the whole list call.
        rebuilt_names: set[str] = set()
        if self._github.use_k8s and self._github.k8s_provisioner:
            try:
                known_names = {getattr(c, "container_name", None) for c in containers}
                live_names = self._github.k8s_provisioner.list_managed_container_names()
                for name in live_names:
                    if name in known_names:
                        continue
                    rebuilt = self._github.k8s_provisioner.reconstruct_container_info(name)
                    if rebuilt is not None:
                        # Remember which entries came from the cluster rather than
                        # the registry — the tenant filter below treats a missing
                        # tenant differently for the two, and only the name
                        # distinguishes them once they are in one list.
                        rebuilt_names.add(name)
                        containers.append(rebuilt)
            except (ApiException, OSError, RuntimeError) as e:
                logger.warning(
                    "K8s reconcile for list_containers failed; returning registry-only list: %s",
                    e,
                )

        if tenant_id is not None:
            # Filter by tenant - include containers with matching tenant_id.
            # Also include OWNERLESS containers (tenant is None) for backward
            # compatibility with rows and Deployments provisioned before tenants
            # existed, or with no tenant at all.
            #
            # ⚠️ This comment used to say "Reconstructed cluster-only entries carry
            # tenant_id=None (parity with reconstruct_container_info ...), so they
            # are treated as legacy/visible". That is no longer true and was the
            # bug: reconstruct_container_info now RECOVERS the tenant from the
            # Deployment, so a cluster-only entry owned by another tenant is
            # filtered out here rather than listed as "legacy".
            #
            # ⚠️ CORRECTED AGAIN. The line that replaced it — "None now means
            # ownerless, not owner-unknown" — was itself false, and in the
            # permissive direction. Recovery read the ``tenant-id`` LABEL alone,
            # and that label is written only for tenants matching
            # ``_K8S_LABEL_VALUE_RE``, so an OWNED workspace with a
            # grammar-invalid owner also arrived here as ``None`` and was listed
            # to every tenant. Recovery is now annotation-then-label, and the
            # unrecoverable ones are dropped at the rebuild loop above rather
            # than falling into the ``is None`` arm of this filter.
            #
            # The ``is None`` arm therefore now covers ONLY registry rows — real
            # legacy data this service wrote — which is the population it was
            # always meant to serve.
            containers = [
                c
                for c in containers
                if getattr(c, "tenant_id", None) == tenant_id
                or (
                    getattr(c, "tenant_id", None) is None
                    and getattr(c, "container_name", None) not in rebuilt_names
                )
            ]

        return containers

    async def get_container_status(self, name: str) -> ContainerStatus:
        """Get the status of a container.

        In K8s mode the container is a Deployment/Pod that does not exist in the
        local Docker daemon, so the Docker-only core getter would always report
        ERROR. Route to the K8s provisioner (which reads the deployment's
        ready_replicas) and map its status onto the ContainerStatus enum. Fall
        back to the Docker core getter when K8s mode is disabled.

        Args:
            name: The container name

        Returns:
            Container status information
        """
        if self._github.use_k8s and self._github.k8s_provisioner:
            try:
                result = await self._github.k8s_provisioner.get_container_status(name)
            except (ApiException, OSError, RuntimeError) as e:
                logger.warning("K8s status check failed for %s: %s", name, e)
                return ContainerStatus.ERROR
            return _map_k8s_container_status(result.get("status"))
        return self._core.get_container_status(name)

    async def execute_command(
        self,
        container_name: str,
        command: str | list[str],
        tenant_id: str | None = None,
        timeout: int = 300,
        env: dict[str, str] | None = None,
    ) -> dict:
        """Execute a command in a workspace.

        Args:
            container_name: The workspace name
            command: The command to execute. A string is wrapped in ``sh -c``;
                a list is passed through untouched.
            tenant_id: If provided, validates tenant access before execution.
                      Returns error if container belongs to different tenant.
            timeout: Execution timeout in seconds (default 300). Raise it for
                     long-running commands such as pushing a large repository.
            env: Extra environment for THIS command only (git_push passes the
                 push credential this way, so it never sits in a file). A
                 backend without per-command env (agent host) refuses it
                 explicitly; that surfaces as a failed result, never a silent
                 run without it.

        Returns:
            Dict with 'success' key and result information.

        Backend selection is delegated to ``src.runtime.resolve_runtime``. It
        used to be decided here by testing ``is_docker_available()`` first and
        reaching Kubernetes only when Docker was absent — the opposite
        precedence to the one ``/exec`` applied. In cluster with a working
        Docker socket the same workspace therefore ran on a different backend
        depending on which entry point the caller came through. The returned
        dict keeps its historic shape, so callers are unaffected.
        """
        try:
            # Validate tenant access if tenant_id is provided
            if tenant_id is not None:
                if not self.has_container(container_name, tenant_id=tenant_id):
                    return {
                        "success": False,
                        "stderr": f"Container '{container_name}' not found",
                        "stdout": "",
                        "exit_code": 1,
                    }

            argv = ["sh", "-c", command] if isinstance(command, str) else list(command)

            # Registry metadata is an OPTIMISATION here, not a precondition.
            # This method has always worked for a workspace the metadata
            # registry does not know about — a plain Docker container in
            # compose, for instance — so a lookup miss degrades to the bare
            # name, which resolves to the Docker backend exactly as before.
            # Requiring the lookup would turn "we have no metadata" into
            # "this container does not exist".
            try:
                workspace: Any = self.get_container(container_name) or container_name
            except ValueError, RuntimeError, KeyError:
                logger.debug(
                    "No registry entry for %s; resolving runtime from the name alone",
                    container_name,
                )
                workspace = container_name

            try:
                runtime = resolve_runtime(workspace, core=self._core)
            except RuntimeError as e:
                # Preserves the old "Neither Docker nor Kubernetes is available"
                # outcome: no backend could be reached, reported rather than
                # silently retried against a different one.
                logger.warning("No runtime for workspace %s: %s", container_name, e)
                return {
                    "success": False,
                    "stderr": str(e),
                    "stdout": "",
                    "exit_code": 1,
                }

            async def _run(ws: Any, rt: Any) -> Any:
                return await rt.run_once(
                    ws,
                    argv,
                    cwd="/workspace",
                    env=env,
                    timeout=timeout,
                    # Internal callers build their own argv (git plumbing, tree
                    # walks, manifest reads), so the allowlist does not apply.
                    # That was already true; it is now stated rather than
                    # implied by which function happened to be entered.
                    execution_profile=ExecutionProfile.SERVER_AUTHORED,
                )

            try:
                result = await _run(workspace, runtime)
            except ContainerDeploymentNotFoundError:
                # The Deployment is gone while its registry entry survives —
                # the reaper's signature (TTL delete, no in-use guard). Rebuild
                # it once from the stored spec and retry the command ONCE.
                #
                # ⛔ BOUNDED: exactly one ``ensure_container`` call and at most
                # two ``run_once`` calls per ``execute_command``. A second
                # not-found from the retry is NOT caught here — it falls to the
                # handler below and is reported — so a 404 cannot spin.
                logger.warning(
                    "Deployment for %s is missing; attempting one bounded re-provision",
                    container_name,
                )
                recovered = await self.ensure_container(container_name, tenant_id=tenant_id)
                if recovered is None:
                    # Tenant mismatch: the BOLA gate, never a provision.
                    raise
                workspace = recovered
                result = await _run(workspace, resolve_runtime(workspace, core=self._core))
            if result.truncated:
                logger.warning(
                    "Output truncated for %s: %d bytes omitted",
                    container_name,
                    result.bytes_omitted,
                )
            return {
                "success": result.succeeded,
                "stdout": result.stdout,
                # Historic shape: this API suppresses stderr on success.
                "stderr": result.stderr if not result.succeeded else "",
                "exit_code": result.exit_code,
                "truncated": result.truncated,
                "bytes_omitted": result.bytes_omitted,
            }
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            # ⛔ This used to interpolate the RAW `command` — not a slice, the
            # whole string — so the failure path disclosed strictly more than
            # the INFO line in the provisioner did. That is the worst possible
            # distribution for a secret: it fires only when a push has ERRORED,
            # i.e. exactly when the token is still live and about to be retried,
            # and it never appears in a happy-path log sample. Same projection,
            # applied to this render in its own right.
            logger.error(
                "Error executing command %s in container %s: %s",
                summarize_command(command),
                container_name,
                redact_secrets(str(e)),
            )
            return {"success": False, "stderr": str(e), "stdout": "", "exit_code": 1}

    def has_container(self, name: str, tenant_id: str | None = None) -> bool:
        """Check if a container exists and is accessible to tenant.

        Args:
            name: The container name
            tenant_id: If provided, also validates tenant access.
                      Returns False if container exists but belongs to different tenant.

        Returns:
            True if container exists and is accessible, False otherwise.
        """
        # Reuse get_container so existence checks inherit the same registry-miss
        # → live-Deployment fallback and tenant-scoping rules.
        try:
            return self.get_container(name, tenant_id=tenant_id) is not None
        except ValueError:
            return False

    def get_container(self, name: str, tenant_id: str | None = None) -> ContainerInfo | None:
        """Get a container by name, optionally validating tenant access.

        Args:
            name: The container name
            tenant_id: If provided, validates that container belongs to this tenant.
                      Returns None if container exists but belongs to different tenant.
                      If None, returns container without tenant check (for admin use).

        Returns:
            ContainerInfo if found and accessible, None otherwise.
            Note: Returns None for both "not found" and "wrong tenant" to prevent
            enumeration attacks (BOLA protection).

        Raises:
            ValueError: If container not found (for backward compatibility with
                       code that catches this). New code should check for None.
        """
        docker_service = get_docker_service()
        registry_data = docker_service.registry.load()

        container = registry_data.get(name)

        # Registry miss: the JSON registry is transient and can momentarily lose a
        # freshly-stored entry (an interleaved whole-dict save), which previously
        # made a healthy container 404 for its entire readiness window. Fall back
        # to the durable source of truth — the live k8s Deployment — and rebuild a
        # usable ContainerInfo so status/exec/dependency calls keep working.
        reconstructed = False
        if container is None:
            if self._github.use_k8s and self._github.k8s_provisioner:
                container = self._github.k8s_provisioner.reconstruct_container_info(name)
                reconstructed = container is not None
            if container is None:
                raise ValueError(f"Container '{name}' not found")

        # If tenant_id specified, validate access
        if tenant_id is not None:
            container_tenant = getattr(container, "tenant_id", None)
            if container_tenant is None and reconstructed:
                # ⭐ THE RESIDUAL, decided explicitly and failing CLOSED.
                #
                # On the RECONSTRUCTED path a ``None`` tenant does not mean
                # "legacy data" — it means ``_recover_tenant_id`` found NEITHER
                # the ``coding-job/tenant-id`` annotation NOR the ``tenant-id``
                # label, so the owner is UNKNOWN. Unknown is not ownerless: the
                # label is written only for grammar-valid tenants, so an owned
                # workspace whose owner cannot be expressed as a k8s label is
                # indistinguishable here from one that truly has no owner. We
                # cannot prove this caller owns it, so we must not confirm it
                # exists — this branch previously returned the row and a foreign
                # tenant got 200 + ``container_status: running``.
                #
                # Scoped to ``reconstructed`` on purpose: a registry-HIT row with
                # ``tenant_id=None`` is genuinely legacy data written by this
                # service, and keeps its documented fail-open below. Measured
                # cost of this branch: 0 of the 10 live Deployments in
                # ``coding-jobs`` resolve to unknown (10 of 10 carry the label).
                #
                # Logged, because this refusal is otherwise INVISIBLE. It returns
                # the same ``None`` as a genuine absence, so an operator sees a
                # 404 for every caller — admin included — against a Deployment
                # that is plainly Running, with nothing in the logs to say why.
                # WARNING, not INFO: it means an owned workspace may be
                # unreachable, which is a state someone has to act on.
                logger.warning(
                    "Refusing %s to tenant %s: workspace owner is UNRECOVERABLE (no %s "
                    "annotation and no %s label on the Deployment). This is a REFUSAL, not a "
                    "missing workspace — the Deployment exists. Re-provision it so the owner "
                    "is stamped, or query as an admin.",
                    name,
                    tenant_id,
                    TENANT_ANNOTATION,
                    TENANT_LABEL_KEY,
                )
                return None
            # Allow access if:
            # 1. Container belongs to the same tenant
            # 2. Container has no tenant (legacy REGISTRY data — see above)
            if container_tenant is not None and container_tenant != tenant_id:
                # Return None instead of raising to prevent enumeration
                return None

        return container

    async def ensure_container(
        self, name: str, tenant_id: str | None = None
    ) -> ContainerInfo | None:
        """Return usable info for ``name``, re-provisioning a REAPED Deployment.

        This is the THIRD fallback, not the first. ``get_container`` already
        tries (1) the Redis metadata registry and (2)
        ``k8s_provisioner.reconstruct_container_info(name)`` — the live
        Deployment. This method runs only once both of those have been
        consulted, and it re-provisions only in the one state neither can
        serve: **registry entry present, Deployment gone**.

        That state is the reaper's fingerprint. ``src/core/reaper.py`` says of
        itself that it has "no 'active in-use' guard: a Deployment older than
        ``reaper_ttl_seconds`` (24h) is reapable regardless of an in-flight
        exec", and ``_cleanup_resources`` deletes the Deployment, the legacy
        Service and the git-token Secret — but never the registry hash. So the
        spec (``repo_url`` / ``branch`` / ``secret_config_id``) survives the
        reap, and the container name is deterministic, which is what makes a
        re-provision by name reconstruct the SAME object rather than a sibling.

        ⚠️ It deliberately does NOT rescue the both-gone case. If the registry
        has no entry AND no Deployment exists, there is no ``repo_url`` anywhere
        in the system, so nothing can be provisioned — ``get_container`` raises
        ``ValueError`` and that propagates. Converting that into a silent
        success is the failure mode this method must not have.

        Guarantees:
            * **Idempotent** — a live Deployment short-circuits before any
              provision, and the provisioner's 409 path reconciles in place
              (patch), so a second call cannot create a second Deployment.
            * **Bounded** — one attempt per name per ``REPROVISION_COOLDOWN_S``,
              never two concurrently. No loop, no retry.
            * **Fails loudly** — a missing spec, a cooling-down name, an
              in-flight peer, or a provisioning error all raise. Every message
              keeps the words "not found" so the router's existing mapping
              answers 404 rather than 500.

        Returns:
            ContainerInfo, or None for a tenant mismatch (BOLA parity with
            ``get_container`` — a foreign container is never provisioned).

        Raises:
            ValueError: name unknown to both the registry and the cluster, or
                known but not re-provisionable.
        """
        # Raises ValueError when neither the registry nor the live Deployment
        # knows this name — the genuinely unprovisionable case.
        info = self.get_container(name, tenant_id=tenant_id)
        if info is None:
            return None

        if not (self._github.use_k8s and self._github.k8s_provisioner):
            return info

        provisioner = self._github.k8s_provisioner
        status = await provisioner.get_container_status(name)
        if (status or {}).get("status") != "not_found":
            # The Deployment is there. Nothing to do — this is the idempotency
            # short-circuit, and it is why a second call provisions nothing.
            return info

        repo_url = getattr(info, "repo_url", "") or ""
        if not repo_url:
            raise ValueError(
                f"Container '{name}' not found: its Deployment is gone and the "
                "stored metadata carries no repo_url, so it cannot be re-provisioned"
            )

        # The tenant a re-provision mints its clone credential for is the ROW's
        # tenant — never the caller's. The caller here may be an admin or an
        # internal path with ``tenant_id=None``, and substituting it (the old
        # ``row or caller``) sent ``tenant_id=None`` to coding-github for a row
        # stamped without a tenant. For a customer repo that is a request with no
        # tenant binding at all, so it is refused here.
        #
        # A PLATFORM repo whose row has no tenant is refused too: there is no
        # implicit platform tenant. Such a row predates tenant stamping and must
        # be re-registered under its real tenant; guessing one here would stamp
        # an owner nobody stated onto the workspace.
        row_tenant = getattr(info, "tenant_id", None) or None
        if row_tenant is None:
            repo_owner = repo_owner_from_url(repo_url)
            if not is_platform_owner(repo_owner):
                logger.error(
                    "Refusing to re-provision %s: its registry row has no tenant_id and "
                    "%s is not a platform repository, so no tenant-bound credential "
                    "can be requested",
                    name,
                    repo_url,
                )
                raise ValueError(
                    f"Container '{name}' not found: its registry row carries no tenant, "
                    "and a customer repository is never re-provisioned without one"
                )
            logger.warning(
                "event=%s container=%s repo_url=%s context=reprovision_legacy_row: "
                "registry row carries no tenant; refused (no implicit platform tenant)",
                PLATFORM_TENANT_MISSING_EVENT,
                name,
                repo_url,
            )
            raise ValueError(
                f"Container '{name}' not found: {PLATFORM_TENANT_MISSING}: tenant required "
                "for platform repo; re-register legacy container under its tenant"
            )

        if name in _REPROVISION_INFLIGHT:
            raise ValueError(f"Container '{name}' not found: a re-provision is already in flight")
        now = time.monotonic()
        last = _REPROVISION_LAST_ATTEMPT.get(name)
        if last is not None and (now - last) < REPROVISION_COOLDOWN_S:
            raise ValueError(
                f"Container '{name}' not found: re-provision attempted "
                f"{int(now - last)}s ago, cooling down for {int(REPROVISION_COOLDOWN_S)}s"
            )

        # Stamp BEFORE attempting, so a provision that raises still consumes the
        # cooldown. A failing name must not be retried on every request.
        _REPROVISION_LAST_ATTEMPT[name] = now
        _REPROVISION_INFLIGHT.add(name)
        try:
            logger.warning(
                "Re-provisioning reaped container %s from stored metadata (repo=%s)",
                name,
                repo_url,
            )
            provisioned = await provisioner.provision_container(
                container_name=name,
                repo_url=repo_url,
                branch=getattr(info, "branch", None) or "main",
                tenant_id=row_tenant,
                secret_config_id=getattr(info, "secret_config_id", None),
            )
        finally:
            _REPROVISION_INFLIGHT.discard(name)
        return provisioned or info

    def get_container_for_graph(
        self, code_graph_id: str, tenant_id: str | None = None
    ) -> ContainerInfo | None:
        """Get container information for a specific code graph ID.

        Args:
            code_graph_id: The code graph identifier
            tenant_id: If provided, only search within tenant's containers.

        Returns:
            ContainerInfo if found and accessible, None otherwise
        """
        try:
            containers = self.list_containers(tenant_id=tenant_id)
            for container in containers:
                if (
                    code_graph_id == container.code_graph_id
                    or code_graph_id in container.container_name
                    or container.container_name.endswith(f"_{code_graph_id}")
                ):
                    return container
            return None
        except (ValueError, KeyError) as e:
            logger.error(
                "Error getting container for graph %s: %s",
                code_graph_id,
                e,
                exc_info=True,
            )
            return None

    async def get_all_graphs_summary_info(self, tenant_id: str | None = None) -> list[dict]:
        """Get summary information for all available graphs.

        Args:
            tenant_id: If provided, only return graphs for this tenant's containers.

        Returns:
            List of graph summary dictionaries
        """
        try:
            settings = get_settings()
            graph_summaries = []
            containers = self.list_containers(tenant_id=tenant_id)
            for container in containers:
                if hasattr(container, "code_graph_id") and container.code_graph_id:
                    graph_info: dict[str, Any] = {
                        "id": container.code_graph_id,
                        "graph_id": container.code_graph_id,
                        "name": getattr(container, "container_name", container.code_graph_id),
                        "description": f"Code graph for {getattr(container, 'repo_url', 'repository')}",
                        "language": "python",
                        "created_at": getattr(container, "created_at", ""),
                        "updated_at": getattr(container, "updated_at", ""),
                        "tags": [],
                        "status": getattr(container, "status", "unknown"),
                        "is_containerized": True,
                        "container_name": getattr(container, "container_name", ""),
                        "repository_url": getattr(container, "repo_url", ""),
                    }
                    graph_summaries.append(graph_info)
            try:
                db_manager = settings.get_database_manager()
                for base_path in db_manager.base_paths:
                    self._scan_database_path(base_path, graph_summaries)
            except NotImplementedError:
                logger.debug("DatabaseManager not available, skipping standalone graph scan")
            return graph_summaries
        except (OSError, ValueError, KeyError) as e:
            logger.error("Error getting all graphs summary: %s", e, exc_info=True)
            return []

    async def get_graph_details_by_id(
        self, graph_id: str, tenant_id: str | None = None
    ) -> dict | None:
        """Get detailed information for a specific graph.

        Args:
            graph_id: The graph identifier
            tenant_id: If provided, only return if graph belongs to this tenant.

        Returns:
            Graph details dictionary or None if not found
        """
        try:
            container = self.get_container_for_graph(graph_id, tenant_id=tenant_id)
            if container:
                return {
                    "id": graph_id,
                    "graph_id": graph_id,
                    "name": getattr(container, "container_name", graph_id),
                    "description": f"Code graph for {getattr(container, 'repo_url', 'repository')}",
                    "language": "python",
                    "created_at": getattr(container, "created_at", ""),
                    "updated_at": getattr(container, "updated_at", ""),
                    "tags": [],
                    "status": getattr(container, "status", "unknown"),
                    "is_containerized": True,
                    "container_name": getattr(container, "container_name", ""),
                    "repository_url": getattr(container, "repo_url", ""),
                }
            settings = get_settings()
            try:
                db_manager = settings.get_database_manager()
                existing_db = db_manager.find_existing_database(graph_id)
            except NotImplementedError:
                logger.debug("DatabaseManager not available, skipping standalone DB lookup")
                existing_db = None
            if existing_db:
                return {
                    "id": graph_id,
                    "graph_id": graph_id,
                    "name": graph_id,
                    "description": f"Standalone code graph: {graph_id}",
                    "language": "python",
                    "created_at": "",
                    "updated_at": "",
                    "tags": [],
                    "status": "available",
                    "is_containerized": False,
                    "container_name": None,
                    "repository_url": "",
                }
            return None
        except (OSError, ValueError, KeyError) as e:
            logger.error("Error getting graph details for %s: %s", graph_id, e, exc_info=True)
            return None


_container_manager_instance: ContainerManager | None = None


def get_container_manager(
    graph_deletion_callback: Callable | None = None,
) -> ContainerManager:
    """Get the global container manager instance.

    Returns the singleton instance of ContainerManager, creating it if it
    doesn't exist. This ensures consistent container management across the
    entire application while avoiding the overhead of multiple instances.

    Args:
        graph_deletion_callback: Optional callback for graph deletion operations.
                               Only used on first initialization.

    Returns:
        ContainerManager: The singleton container manager instance configured
                         with core container management and GitHub provisioning
                         capabilities.

    Thread Safety:
        Uses module-level global variable for singleton storage. Python's GIL
        provides basic thread safety for the initialization check, ensuring
        only one instance is created even in multi-threaded environments.

    Example:
        >>> # Get the global container manager
        >>> manager = get_container_manager()
        >>>
        >>> # All subsequent calls return the same instance
        >>> same_manager = get_container_manager()
        >>> assert manager is same_manager

    Note:
        This function implements the singleton pattern for ContainerManager.
        The instance is created lazily on first access and reused for all
        subsequent calls, ensuring consistent state across the application.
    """
    global _container_manager_instance
    if _container_manager_instance is None:
        _container_manager_instance = ContainerManager(graph_deletion_callback)
    return _container_manager_instance

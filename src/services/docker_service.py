"""
Modular Docker Service - Refactored Version

This is the updated DockerService class that uses the new modular architecture.
The original docker_service.py has been split into focused modules for better
maintainability and separation of concerns.

This class now acts as a facade that coordinates between the specialized modules:
- DockerAPIIntegration: Client initialization and connection
- ContainerLifecycleManager: Container lifecycle operations
- ImageManager: Image operations
- NetworkVolumeManager: Network and volume operations
- ContainerMonitor: Monitoring and file operations
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any
from uuid import uuid7

import docker.errors
from fastapi import HTTPException
from kubernetes.config.config_exception import ConfigException

from src.clients._tenant import service_token, tenant_headers
from src.config.settings import get_settings
from src.core.container_metadata_registry import ContainerMetadataRegistry
from src.core.toolchain import (
    MemoryBudgetError,
)
from src.models.container_models import (
    ContainerInfo,
    ContainerStatus,
    GitHubRepoRequest,
)
from src.services.docker_services import (
    ContainerLifecycleManager,
    ContainerMonitor,
    DockerAPIIntegration,
    ImageManager,
    NetworkVolumeManager,
)

logger = logging.getLogger(__name__)

# Eager-build workspace-readiness gate. The eager graph build drives
# coding-analysis to scan /workspace; in k8s the git-clone is an INIT container,
# so the pod only reaches Running AFTER the clone populates /workspace. Gating
# the eager build on a live Running status prevents it from racing the clone and
# persisting a degraded (components=1) graph. Tunable for slow-scheduling
# clusters; on timeout the build is skipped (self-heal rebuilds on demand)
# rather than caching a degraded graph.
EAGER_BUILD_READINESS_TIMEOUT_S = float(os.getenv("EAGER_BUILD_READINESS_TIMEOUT_S", "60"))
EAGER_BUILD_READINESS_INTERVAL_S = float(os.getenv("EAGER_BUILD_READINESS_INTERVAL_S", "2"))


class DockerService:
    """
    Modular Docker Service that coordinates container operations through specialized modules.

    This service maintains the same public API as the original while using a modular
    architecture for better maintainability.
    """

    def __init__(self) -> None:
        """Initialize the modular Docker service."""
        settings = get_settings()
        self.base_port = settings.docker_base_port
        self.image_name = "code_graph_helper-api:latest"
        self.project_name = "code_graph_helper"
        self.internal_worker_port = settings.worker.internal_worker_port
        self.registry = ContainerMetadataRegistry()

        # Initialize the modular components
        self.api_integration = DockerAPIIntegration()
        self.lifecycle_manager = ContainerLifecycleManager(
            self.api_integration.get_docker_client(), self.registry, self.project_name
        )
        self.image_manager = ImageManager(self.api_integration.get_docker_client())
        self.network_volume_manager = NetworkVolumeManager(
            self.api_integration.get_docker_client(), self.project_name
        )
        self.container_monitor = ContainerMonitor(self.api_integration.get_docker_client())

        # Set up project infrastructure
        if self.is_docker_available():
            self.network_volume_manager.setup_project_infrastructure()

        # Track background tasks to prevent garbage collection
        self._background_tasks: set[asyncio.Task[Any]] = set()

    def is_docker_available(self) -> bool:
        """Check if Docker is available and accessible."""
        return self.api_integration.is_docker_available()

    def retry_docker_connection(self) -> bool:
        """Retry Docker connection."""
        return self.api_integration.retry_docker_connection()

    # Container Operations (delegated to ContainerLifecycleManager)

    async def create_container(
        self,
        request: GitHubRepoRequest,
        tenant_id: str | None = None,
    ) -> ContainerInfo:
        """Create a new container from a GitHub repository request."""
        try:
            get_settings()

            # When the caller pins a deterministic container_name (the AA
            # orchestration relies on it for cross-step lookups like
            # verify_container_ready), make create idempotent BY NAME: reuse the
            # same-named container if it exists, otherwise provision under that
            # exact name below. Reusing a different repo-matched container would
            # make verify_container_ready poll a name that was never provisioned,
            # and re-provisioning on every doing retry would replace the
            # Deployment mid-readiness (status flips to error/not_found). Only
            # fall back to repo-based dedup when no explicit name was requested.
            requested_name = getattr(request, "container_name", None)
            if requested_name:
                by_name = await self.find_container_by_name(
                    str(requested_name), tenant_id=tenant_id
                )
                if by_name is not None:
                    # Reuse ONLY a usable container, so n8n's idempotent doing-
                    # retries don't replace a Deployment mid-readiness. find_
                    # container_by_name resolves the LIVE status; a still-
                    # provisioning container is CREATING (not ERROR), so the
                    # readiness-poll case is preserved. But a container found in a
                    # terminal-unusable state (ERROR / STOPPED / REMOVING) is stale
                    # from a PRIOR run under the same deterministic name (the AA
                    # orchestration derives container_name from the project name):
                    # reusing it returns container_status=error, so the caller's
                    # verify_container_ready typed step fails and the card blocks
                    # forever. Tear the stale record/Deployment down and re-provision
                    # fresh under the same name below.
                    reusable = {ContainerStatus.RUNNING, ContainerStatus.CREATING}
                    if by_name.status in reusable:
                        self._refuse_reuse_under_history_cutoff(request, by_name)
                        self._refuse_reuse_without_requested_toolchain(request, by_name)
                        logger.info(
                            "Reusing existing container by name: %s (status=%s)",
                            requested_name,
                            by_name.status,
                        )
                        return by_name
                    logger.warning(
                        "Existing container '%s' is in terminal state '%s' (stale from a "
                        "prior run); recreating instead of reusing",
                        requested_name,
                        by_name.status,
                    )
                    deleted = await self.delete_container(str(requested_name), tenant_id=tenant_id)
                    if not deleted:
                        logger.warning(
                            "Could not fully tear down stale container '%s'; provisioning "
                            "fresh anyway",
                            requested_name,
                        )
            else:
                existing = await self.find_existing_container(
                    str(request.repo_url), tenant_id=tenant_id
                )
                if existing is not None and not self._toolchain_satisfied(request, existing):
                    # Repo-dedup is a convenience, not a contract: a python+rust
                    # request must not be handed a python workspace that merely
                    # clones the same repo. Provision a fresh one instead.
                    logger.info(
                        "Not reusing %s for %s: it lacks the requested python+rust toolchain",
                        existing.container_name,
                        request.repo_url,
                    )
                    existing = None
                if existing is not None:
                    self._refuse_reuse_under_history_cutoff(request, existing)
                    logger.info(
                        f"Found existing container for {request.repo_url}: {existing.container_name}"
                    )
                    return existing

            # Normalize repo URL
            normalized_url = self._normalize_repo_url(str(request.repo_url))

            # Honor caller-supplied container_name; fall back to slug only when
            # absent. Matches kubernetes_container_provisioner.provision_from_github
            # so the docker fallback and k8s path produce the same name shape.
            requested_name = getattr(request, "container_name", None)
            if requested_name:
                container_name = str(requested_name)
            else:
                repo_name = (
                    normalized_url.split("/")[-1] if "/" in normalized_url else normalized_url
                )
                container_name = re.sub(r"[^a-zA-Z0-9_.-]", "-", repo_name).lower()[:50]
                container_name = f"{container_name}-{str(uuid7())[:8]}"

            # Try Kubernetes provisioning first if enabled
            try:
                from src.core.kubernetes_container_provisioner import (
                    get_k8s_container_provisioner,
                )

                k8s = get_k8s_container_provisioner()
                if k8s and k8s.is_available():
                    # Admission cap (Phase C-1): bound concurrent job
                    # Deployments. Runs only on genuine new provisions — the
                    # name-reuse / repo-dedup short-circuits above return before
                    # this point, so n8n's idempotent retries never trip it.
                    await self._enforce_inflight_cap(k8s)
                    logger.info(f"Provisioning container via Kubernetes: {container_name}")
                    container = await k8s.provision_container(
                        container_name=container_name,
                        repo_url=normalized_url,
                        branch=request.branch,
                        access_token=request.access_token,
                        tenant_id=tenant_id,
                        secret_config_id=getattr(request, "secret_config_id", None),
                        # The PR is the thing under test. ContainerCreateRequest.parse_pr_url
                        # populates these from a PR url (and rewrites repo_url to the bare repo,
                        # so they are NOT recoverable downstream). Dropping them here is what made
                        # every k8s workspace clone the default branch.
                        pr_number=getattr(request, "pr_number", None),
                        provision_type=getattr(request, "provision_type", "branch"),
                        history_cutoff=getattr(request, "history_cutoff", False),
                        toolchain=getattr(request, "toolchain", "python"),
                        # Same DROP hazard as toolchain: not forwarded here, the
                        # disk sizing hint would silently vanish (gap #4-lite).
                        expected_repos=getattr(request, "expected_repos", None),
                        # And the memory sizing hints (py2rust gap #4 full).
                        expected_services=getattr(request, "expected_services", None),
                        service_memory_mi=getattr(request, "service_memory_mi", None),
                        expected_dependencies=getattr(request, "expected_dependencies", None),
                    )
                    if container:
                        if not getattr(container, "tenant_id", None):
                            container.tenant_id = tenant_id
                        self.trigger_provision_graph_build(container, tenant_id)
                        return container
            except (ImportError, RuntimeError, docker.errors.DockerException, ConfigException) as e:
                logger.debug(f"K8s provisioning not available, falling back to Docker: {e}")

            # Fall back to Docker provisioning
            if not self.is_docker_available():
                raise Exception(
                    "Neither Kubernetes nor Docker is available for container provisioning"
                )
            if getattr(request, "toolchain", "python") != "python":
                # The Docker-path workspaces run the python image only; silently
                # handing back a workspace without cargo would fail every Rust
                # step later with a confusing exit 127.
                raise HTTPException(
                    status_code=400,
                    detail="toolchain python+rust is only supported by the kubernetes provisioner",
                )
            if getattr(request, "history_cutoff", False):
                # The Docker-path clones (container_lifecycle /
                # github_container_provisioner) never check out a SHA, so a cutoff
                # there would silently be a plain clone of the default branch.
                raise HTTPException(
                    status_code=400,
                    detail="history_cutoff is only supported by the kubernetes provisioner",
                )

            # Delegate to lifecycle manager which handles port allocation,
            # shared-volume clone, and registry save.
            #
            # tenant_id is passed EXPLICITLY. The lifecycle manager used to read
            # only `request.tenant_id` — the request-body field, which the route
            # never populates (it resolves the tenant from the authenticated user
            # and hands it to this method as a separate argument). So on this
            # path the provisioning tenant was None: ContainerInfo was registered
            # unowned, and every lifecycle broadcast went out with no
            # X-Tenant-Id, reaching coding-realtime only via its pod-CIDR trust
            # fallback.
            container = await self.lifecycle_manager.create_container(request, tenant_id=tenant_id)

            # Belt-and-braces: the lifecycle manager now stamps the tenant itself,
            # so this only fires if it somehow returned an unowned record.
            if not getattr(container, "tenant_id", None):
                container.tenant_id = tenant_id
                self.registry.upsert(container.container_name, container)

            # Eager, non-blocking graph build (self-heal in architecture/verify is
            # the correctness guarantee; this just gets the graph ready earlier).
            self.trigger_provision_graph_build(container, tenant_id)

            return container

        except MemoryBudgetError:
            raise  # a refusal (409 at the route), not a failure to log as one
        except (docker.errors.DockerException, RuntimeError, ValueError) as e:
            logger.error(f"Failed to create container: {e}")
            raise

    @staticmethod
    def _refuse_reuse_under_history_cutoff(
        request: GitHubRepoRequest, existing: ContainerInfo
    ) -> None:
        """409 when a ``history_cutoff`` request would land on an EXISTING workspace.

        Both reuse short-circuits above return a container that was cloned by
        some earlier request — possibly at a different ref, possibly without the
        cutoff — and the cutoff rewrites that shared git history. Reusing would
        either hand back a workspace whose history was never cut (a vacuous
        "isolated" benchmark) or destroy refs another caller relies on. Like the
        admission cap's 429, ``HTTPException`` is none of the types
        ``create_container`` catches, so it reaches the route untouched.
        """
        if not getattr(request, "history_cutoff", False):
            return
        raise HTTPException(
            status_code=409,
            detail=(
                f"history_cutoff refused: workspace {existing.container_name} already exists "
                f"(provisioned for {existing.repo_url}); the cutoff rewrites shared git "
                "history. Provision a fresh workspace with a unique container_name."
            ),
        )

    @staticmethod
    def _toolchain_satisfied(request: GitHubRepoRequest, existing: ContainerInfo) -> bool:
        """A python request is satisfied by any workspace (python+rust is a superset);
        a python+rust request only by a python+rust workspace."""
        if getattr(request, "toolchain", "python") != "python+rust":
            return True
        return getattr(existing, "toolchain", "python") == "python+rust"

    @classmethod
    def _refuse_reuse_without_requested_toolchain(
        cls, request: GitHubRepoRequest, existing: ContainerInfo
    ) -> None:
        """409 when a named python+rust request would reuse a python workspace.

        The caller pinned the name, so silently provisioning elsewhere is not an
        option, and reusing would hand back a workspace with no cargo. A python
        request never trips this (unchanged behaviour).
        """
        if cls._toolchain_satisfied(request, existing):
            return
        raise HTTPException(
            status_code=409,
            detail=(
                f"toolchain refused: workspace {existing.container_name} already exists with "
                f"toolchain {getattr(existing, 'toolchain', 'python')!r}, but python+rust was "
                "requested. Delete it first or provision with a unique container_name."
            ),
        )

    async def _enforce_inflight_cap(self, k8s: Any) -> None:
        """Bound concurrent job Deployments (Phase C-1 admission cap).

        Disabled when ``max_inflight_jobs <= 0``. Fails OPEN: a counting
        failure allows the provision rather than blocking work. When at/over
        cap, raises ``HTTPException(429)`` in reject mode (the default), or in
        wait mode polls until under cap or the wait timeout then 429s.

        ``HTTPException`` deliberately escapes ``create_container``'s
        ``except (DockerException, RuntimeError, ValueError)`` (it is none of
        those) so it propagates to the route untouched and does not emit a
        misleading "Failed to create container" error log.
        """
        settings = get_settings()
        cap = settings.max_inflight_jobs
        if not isinstance(cap, int) or cap <= 0:
            return  # disabled (0/unlimited) or non-numeric (test mock)

        async def _count() -> int | None:
            try:
                return await asyncio.to_thread(k8s.count_active_jobs)
            except Exception as exc:  # fail open on any count error
                logger.warning("In-flight cap count failed, allowing provision: %s", exc)
                return None

        active = await _count()
        if active is None or active < cap:
            return

        wait_timeout = settings.admission_wait_timeout_seconds
        if (
            settings.admission_reject_mode == "wait"
            and isinstance(wait_timeout, int)
            and wait_timeout > 0
        ):
            deadline = time.monotonic() + wait_timeout
            while time.monotonic() < deadline:
                await asyncio.sleep(min(2.0, max(0.1, deadline - time.monotonic())))
                active = await _count()
                if active is None or active < cap:
                    return

        raise HTTPException(
            status_code=429,
            detail=f"Container capacity reached ({active}/{cap} in-flight jobs); retry later",
        )

    async def delete_container(self, container_name: str, tenant_id: str | None = None) -> bool:
        """Delete a container and clean up resources."""
        try:
            # Validate tenant access
            if tenant_id:
                container_info = await self.find_container_by_name(
                    container_name, tenant_id=tenant_id
                )
                if container_info is None:
                    logger.warning(
                        f"Container {container_name} not found or not accessible by tenant {tenant_id}"
                    )
                    return False

            return await self.lifecycle_manager.remove_container_by_name(container_name)
        except (docker.errors.DockerException, RuntimeError) as e:
            logger.error(f"Failed to delete container {container_name}: {e}")
            return False

    def list_containers(self, tenant_id: str | None = None) -> list[ContainerInfo]:
        """List all containers, optionally filtered by tenant."""
        try:
            containers = []
            registry_data = self.registry.load()

            for container_name, container_info in registry_data.items():
                # Filter by tenant if specified
                if tenant_id is not None:
                    container_tenant = getattr(container_info, "tenant_id", None)
                    if container_tenant is not None and container_tenant != tenant_id:
                        continue

                containers.append(container_info)

            # Update container statuses
            updated_containers = []
            for container in containers:
                try:
                    current_status = self.lifecycle_manager._get_container_status(
                        container.container_name
                    )
                    container.status = current_status
                except docker.errors.DockerException, RuntimeError:
                    container.status = ContainerStatus.ERROR
                updated_containers.append(container)

            return updated_containers

        except (docker.errors.DockerException, OSError, RuntimeError) as e:
            logger.error(f"Failed to list containers: {e}")
            return []

    async def _resolve_live_status(self, container_name: str) -> ContainerStatus:
        """Resolve a container's live status, K8s-aware.

        In-cluster the K8s Deployment is the source of truth: the local Docker
        daemon has no such container and ``lifecycle_manager._get_container_status``
        RETURNS ``ContainerStatus.ERROR`` (it does not raise) for an unknown
        container — so a Docker-first approach would always mislabel a healthy
        K8s container as ERROR. Query the K8s provisioner FIRST when
        KUBERNETES_IN_CLUSTER (mirroring ``ContainerManager.get_container_status``),
        mapping starting/pending → CREATING so "still provisioning" is distinct
        from a real failure, and only fall back to the Docker getter otherwise.
        """
        if os.environ.get("KUBERNETES_IN_CLUSTER", "false").lower() == "true":
            try:
                from src.core.kubernetes_container_provisioner import (
                    get_k8s_container_provisioner,
                )

                k8s = get_k8s_container_provisioner()
                if k8s and k8s.is_available():
                    result = await k8s.get_container_status(container_name)
                    k8s_status = (result or {}).get("status")
                    if k8s_status == "running":
                        return ContainerStatus.RUNNING
                    if k8s_status in ("starting", "pending"):
                        return ContainerStatus.CREATING
                    if k8s_status == "not_found":
                        return ContainerStatus.ERROR
            except Exception as e:  # noqa: BLE001 - status resolution must never raise
                logger.debug("K8s status resolution failed for %s: %s", container_name, e)
        # Docker path: non-cluster, or K8s did not resolve a status.
        try:
            return self.lifecycle_manager._get_container_status(container_name)
        except docker.errors.DockerException, RuntimeError:
            return ContainerStatus.ERROR

    async def _await_container_running(self, container_name: str) -> bool:
        """Bounded-poll the live status until the container is RUNNING.

        Returns ``True`` once Running — in k8s that means the git-clone init
        container has completed and ``/workspace`` is populated — or ``False``
        if ``EAGER_BUILD_READINESS_TIMEOUT_S`` elapses first. Never raises:
        ``_resolve_live_status`` already swallows resolution errors.
        """
        deadline = time.monotonic() + EAGER_BUILD_READINESS_TIMEOUT_S
        while True:
            if await self._resolve_live_status(container_name) == ContainerStatus.RUNNING:
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(EAGER_BUILD_READINESS_INTERVAL_S)

    async def find_container_by_name(
        self, container_name: str, tenant_id: str | None = None
    ) -> ContainerInfo | None:
        """Find container by container_name.

        Args:
            container_name: The container name to find
            tenant_id: If provided, validates tenant access before returning info.
                      Returns None if container belongs to different tenant.

        Returns:
            ContainerInfo if found and accessible, None otherwise
        """
        registry_data = self.registry.load()
        if container_name in registry_data:
            container_info = registry_data[container_name]

            # Validate tenant access if tenant_id is provided
            if tenant_id is not None:
                container_tenant = getattr(container_info, "tenant_id", None)
                if container_tenant is not None and container_tenant != tenant_id:
                    return None

            container_info.status = await self._resolve_live_status(container_name)
            return container_info

        # Registry miss: fall back to the live Deployment (durable source of
        # truth) so a transient registry drop doesn't 404 a healthy container.
        try:
            from src.core.kubernetes_container_provisioner import (
                TENANT_ANNOTATION,
                TENANT_LABEL_KEY,
                get_k8s_container_provisioner,
            )

            k8s = get_k8s_container_provisioner()
            if k8s and k8s.is_available():
                rebuilt = k8s.reconstruct_container_info(container_name)
                if rebuilt is not None:
                    container_tenant = getattr(rebuilt, "tenant_id", None)
                    if tenant_id is not None and container_tenant is None:
                        # ⭐ THE RESIDUAL, failing CLOSED — the FOURTH site of
                        # this gate, and the one that stayed vacuous longest.
                        #
                        # ``reconstruct_container_info`` recovers the owner from
                        # the ``coding-job/tenant-id`` annotation, then the
                        # ``tenant-id`` label. A surviving ``None`` means the
                        # owner is UNKNOWN, not "ownerless": the label is written
                        # only for tenants matching ``_K8S_LABEL_VALUE_RE``, so an
                        # OWNED workspace whose owner cannot be expressed as a
                        # label is indistinguishable here from one with no owner.
                        #
                        # ⛔ This is NOT a read-only disclosure. This is the gate
                        # for ``POST /{name}/install-dependencies`` and
                        # ``GET /{name}/dependency-status``
                        # (``container_dependencies.py:145`` and ``:243``). Past
                        # this point ``ContainerInstaller`` calls
                        # ``get_container(name)`` with NO tenant argument at all,
                        # so the gate is never re-applied and package-manager
                        # commands RUN INSIDE the foreign pod. Measured at the
                        # previous head: a recovered-owner workspace correctly
                        # 404s the same caller, while an unknown-owner one was
                        # served — the /status fix's own signature, inverted.
                        logger.warning(
                            "Refusing %s to tenant %s: workspace owner is UNRECOVERABLE "
                            "(no %s annotation and no %s label on the Deployment). This is a "
                            "REFUSAL, not a missing workspace — the Deployment exists. "
                            "Re-provision it so the owner is stamped, or query as an admin.",
                            container_name,
                            tenant_id,
                            TENANT_ANNOTATION,
                            TENANT_LABEL_KEY,
                        )
                        return None
                    if (
                        tenant_id is not None
                        and container_tenant is not None
                        and container_tenant != tenant_id
                    ):
                        return None
                    return rebuilt
        except (RuntimeError, OSError) as e:
            logger.debug("k8s fallback for %s failed: %s", container_name, e)
        return None

    async def find_existing_container(
        self, repo_url: str, tenant_id: str | None = None
    ) -> ContainerInfo | None:
        """Find existing container by repository URL.

        Args:
            repo_url: Repository URL to search for
            tenant_id: If provided, validates tenant access

        Returns:
            ContainerInfo if found, None otherwise
        """
        normalized_url = self._normalize_repo_url(repo_url)
        registry_data = self.registry.load()

        for container_name, container_info in registry_data.items():
            container_repo = getattr(container_info, "repo_url", None)
            if container_repo and self._normalize_repo_url(str(container_repo)) == normalized_url:
                # Validate tenant access
                if tenant_id is not None:
                    container_tenant = getattr(container_info, "tenant_id", None)
                    if container_tenant is not None and container_tenant != tenant_id:
                        continue

                container_info.status = await self._resolve_live_status(container_name)
                return container_info

        return None

    def trigger_provision_graph_build(
        self,
        container: ContainerInfo,
        tenant_id: str | None,
        *,
        force_rebuild: bool = False,
    ) -> None:
        """Fire-and-forget eager code-graph build for a freshly provisioned container.

        Non-blocking: provisioning returns immediately while the graph is built
        in the background, honouring "every provisioned repo gets a graph"
        without slowing container creation. Correctness does not depend on this
        finishing — the architecture/verify endpoints self-heal (build the graph
        on demand) if it hasn't run yet — so a failure here is logged, not fatal.
        """
        if not tenant_id:
            # The build endpoint is tenant-scoped (X-Tenant-Id required); without
            # a tenant we cannot authenticate the call. Self-heal will cover it.
            logger.warning(
                "Skipping eager graph build for %s: no tenant_id available",
                container.container_name,
            )
            return
        task = asyncio.create_task(
            self._async_provision_graph_build(container, tenant_id, force_rebuild=force_rebuild)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        logger.info(
            "Eager code-graph build started in background for %s",
            container.container_name,
        )

    async def _async_provision_graph_build(
        self,
        container: ContainerInfo,
        tenant_id: str,
        *,
        force_rebuild: bool = False,
    ) -> None:
        """Populate the code graph by driving the deep-analysis ingestion path.

        Calls coding-analysis ``POST /api/analysis/analyze`` (analysis_type=deep),
        which resolves source files, parses them, and persists nodes/edges. This
        is the *real* ingestion — unlike ``/api/graphs/initialize``, which only
        creates the repository root node.
        """
        import aiohttp

        settings = get_settings()
        # Address coding-analysis DIRECTLY (east-west). Routing this through
        # coding-gateway (``main_api_endpoint``) was reset at the mesh for every
        # single build — the gateway is a north-south edge that does not admit
        # the coding-containers namespace. See ``coding_analysis_url``.
        analysis_base = settings.coding_analysis_url.rstrip("/")
        if not analysis_base:
            logger.warning(
                "Skipping eager graph build for %s: coding_analysis_url not configured",
                container.container_name,
            )
            return

        # Workspace-readiness gate (A1): don't drive the deep build until the
        # container is live. In k8s "Running" means the git-clone init container
        # finished and /workspace is populated, so this stops the eager analyze
        # from scanning an empty workspace and caching a degraded components=1
        # graph. On timeout, skip — the architecture/verify self-heal rebuilds on
        # demand, which is strictly better than persisting a degraded graph.
        if not await self._await_container_running(container.container_name):
            logger.warning(
                "Skipping eager graph build for %s: not Running within %ss "
                "(workspace may be mid-clone); self-heal will cover",
                container.container_name,
                EAGER_BUILD_READINESS_TIMEOUT_S,
            )
            return

        analyze_url = f"{analysis_base}/api/analysis/analyze"
        payload = {
            "analysis_type": "deep",
            "container_name": container.container_name,
            "language": "python",
        }
        # Detached (asyncio.create_task) — there is no inbound request left to
        # borrow a credential from, so Authorization carries this service's OWN
        # licence and X-Tenant-Id stays the provisioning caller's tenant. Folding
        # the token into X-Tenant-Id would attribute every customer's graph to
        # coding-containers' own tenant.
        headers = {
            "Content-Type": "application/json",
            **tenant_headers(
                tenant_id,
                operation=f"eager graph build for {container.container_name}",
                service_token=service_token(),
            ),
        }
        # Deep analysis parses the whole repo; allow generous time since this runs
        # in the background and never blocks provisioning.
        timeout = aiohttp.ClientTimeout(total=600, connect=5, sock_read=300)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(analyze_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        body = await response.json()
                        meta = (body.get("result") or {}).get("metadata") or {}
                        logger.info(
                            "Eager code-graph build for %s: %s nodes, %s edges",
                            container.container_name,
                            meta.get("graph_nodes"),
                            meta.get("graph_edges"),
                        )
                        container.graph_initialized = True
                        self.registry.upsert(container.container_name, container)
                    else:
                        text = await response.text()
                        logger.warning(
                            "Eager code-graph build for %s returned %s: %s",
                            container.container_name,
                            response.status,
                            text[:500],
                        )
        except (aiohttp.ClientError, OSError, TimeoutError) as exc:
            logger.warning(
                "Eager code-graph build failed for %s (self-heal will cover): %s",
                container.container_name,
                exc,
            )

    # File and Directory Operations (delegated to ContainerMonitor)

    async def read_container_file(
        self,
        container_name: str,
        file_path: str,
        offset: int = 0,
        length: int | None = None,
    ) -> dict[str, Any]:
        """Read file from container using docker exec."""
        return await self.container_monitor.read_container_file(
            container_name, file_path, offset, length
        )

    async def list_container_directory(
        self, container_name: str, path: str = "", recursive: bool = False
    ) -> list[dict[str, Any]]:
        """List directory contents in container."""
        return await self.container_monitor.list_container_directory(
            container_name, path, recursive
        )

    async def get_container_repository_tree(self, container_name: str) -> dict[str, Any]:
        """Get complete repository tree structure."""
        return await self.container_monitor.get_container_repository_tree(container_name)

    async def search_container_files(
        self, container_name: str, pattern: str, max_results: int = 100
    ) -> list[dict[str, Any]]:
        """Search for files by name pattern in container."""
        return await self.container_monitor.search_container_files(
            container_name, pattern, max_results
        )

    async def search_container_content(
        self,
        container_name: str,
        pattern: str,
        file_pattern: str | None = None,
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Search for content pattern in container files."""
        return await self.container_monitor.search_container_content(
            container_name, pattern, file_pattern, case_sensitive, max_results
        )

    # Test Container Operations

    async def create_test_container(self, config: dict[str, Any]) -> str:
        """Create container specifically for test execution."""
        if not self.is_docker_available():
            raise Exception("Docker not available for test container creation")

        try:
            test_id = str(uuid7())[:8]
            container_name = f"test-{config.get('language', 'python')}-{test_id}"

            security_config = {
                "mem_limit": config.get("mem_limit", "512m"),
                "cpu_quota": config.get("cpu_quota", 50000),
                "network_disabled": config.get("network_disabled", True),
                "read_only": config.get("read_only", True),
                "user": config.get("user", "testuser"),
                "security_opt": config.get("security_opt", ["no-new-privileges:true"]),
                "tmpfs": config.get("tmpfs", {"/tmp": "noexec,nosuid,size=100m"}),
                "cap_drop": ["ALL"],
            }

            client = self.api_integration.get_docker_client()
            assert client is not None, "Docker client unavailable"
            container = client.containers.create(
                image=config["image"],
                name=container_name,
                working_dir=config.get("working_dir", "/app/test"),
                labels=config.get("labels", {}),
                **security_config,
            )

            logger.info(f"Created test container {container_name} with image {config['image']}")
            return container.id

        except docker.errors.DockerException as e:
            logger.error(f"Failed to create test container: {e}")
            raise

    async def exec_in_container(
        self,
        docker_id: str,
        command: list[str],
        timeout: int = 30,
        environment: dict[str, str] | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        """Execute command in container and return structured result."""
        return await self.container_monitor.exec_in_container(
            docker_id, command, timeout, environment, user
        )

    async def create_file_in_container(
        self, docker_id: str, file_path: str, content: str, mode: str = "w"
    ) -> bool:
        """Create file with content in container."""
        return await self.container_monitor.create_file_in_container(
            docker_id, file_path, content, mode
        )

    async def install_tools_in_container(
        self, docker_id: str, tools: list[str], language: str = "python"
    ) -> bool:
        """Install testing/quality tools in container."""
        return await self.container_monitor.install_tools_in_container(docker_id, tools, language)

    async def get_container_stats(self, docker_id: str) -> dict[str, Any]:
        """Get resource usage statistics for container."""
        return await self.container_monitor.get_container_stats(docker_id)

    # Image Operations (delegated to ImageManager)

    def get_language_container_image(self, language: str) -> str:
        """Get appropriate container image for language."""
        return self.image_manager.get_language_container_image(language)

    # Repository Operations

    def _sync_repo_to_shared_volume(
        self, docker_container, container_name: str, source_path: str
    ) -> None:
        """
        Mirror the freshly cloned repository into the shared volume that the API container can read.

        Args:
            docker_container: Active docker container object
            container_name: Name of the provisioned container
            source_path: Path where the repository was cloned inside the container
        """
        shared_repo_path = f"/app/provisioned_containers/{container_name}/repo"

        # Ensure clean target directory
        cleanup = docker_container.exec_run(["/bin/sh", "-c", f"rm -rf '{shared_repo_path}'"])
        if cleanup.exit_code != 0:
            output = cleanup.output.decode("utf-8", errors="ignore") if cleanup.output else ""
            raise Exception(f"Failed to cleanup shared repo path {shared_repo_path}: {output}")

        mkdir = docker_container.exec_run(["/bin/sh", "-c", f"mkdir -p '{shared_repo_path}'"])
        if mkdir.exit_code != 0:
            output = mkdir.output.decode("utf-8", errors="ignore") if mkdir.output else ""
            raise Exception(f"Failed to create shared repo path {shared_repo_path}: {output}")

        copy_cmd = f"cp -a {source_path.rstrip('/')}/. '{shared_repo_path}/'"
        copy = docker_container.exec_run(["/bin/sh", "-c", copy_cmd])
        if copy.exit_code != 0:
            output = copy.output.decode("utf-8", errors="ignore") if copy.output else ""
            raise Exception(
                f"Failed to copy repository into shared volume ({shared_repo_path}): {output}"
            )

        verify = docker_container.exec_run(["/bin/sh", "-c", f"test -d '{shared_repo_path}'"])
        if verify.exit_code != 0:
            output = verify.output.decode("utf-8", errors="ignore") if verify.output else ""
            raise Exception(
                f"Shared repository path {shared_repo_path} missing after copy: {output}"
            )

    async def _install_dependencies_in_container(
        self, container_name: str, workspace_path: str
    ) -> None:
        """Install dependencies in the container after cloning."""
        try:
            from src.core.container_installer import ContainerInstaller

            installer = ContainerInstaller()
            result = await installer.install_dependencies_in_container(
                container_name, workspace_path
            )

            if result.success:
                if len(result.installed_packages) > 0:
                    logger.info(
                        f"Dependencies installed successfully: {len(result.installed_packages)} packages using {', '.join(result.package_managers_used)}"
                    )
                else:
                    logger.info(
                        "Dependency check completed successfully: no dependencies to install"
                    )
                if result.logs:
                    for log_entry in result.logs[-3:]:
                        logger.info(f"  {log_entry}")
            else:
                logger.warning(f"Dependency installation had issues: {len(result.errors)} errors")
                for error in result.errors:
                    logger.warning(f"  {error}")

        except (ImportError, RuntimeError, OSError) as e:
            logger.warning(f"Dependency installation failed (non-critical): {str(e)}")

    def _normalize_repo_url(self, repo_url: str) -> str:
        """Normalize GitHub repo URL to standard format."""
        url = repo_url.rstrip("/")
        if url.startswith("git@github.com:"):
            ssh_part = url.replace("git@github.com:", "")
            if ssh_part.endswith(".git"):
                ssh_part = ssh_part[:-4]
            url = f"https://github.com/{ssh_part}"
        elif url.endswith(".git"):
            url = url[:-4]
        elif "/" in url and (not url.startswith("http")):
            url = f"https://github.com/{url}"
        return url


def get_docker_service() -> DockerService:
    """Get Docker service instance."""
    return DockerService()

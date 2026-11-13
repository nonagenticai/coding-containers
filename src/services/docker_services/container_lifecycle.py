"""
Container Lifecycle Management Module

Handles creation, starting, stopping, and removal of Docker containers.
Extracted from docker_service.py to improve modularity and maintainability.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse
from uuid import uuid7

import docker
import docker.errors
import docker.types
import httpx

from coding_client.utils.retry_helper import retry_docker
from src.clients.github_client import (
    APP_NOT_INSTALLED_ERROR,
    GITHUB_HOST,
    canonical_github_repo,
    canonical_github_url,
    get_github_token_manager,  # TODO: resolve cross-service dependency
    is_platform_owner,
)
from src.clients.realtime_client import (  # TODO: resolve cross-service dependency
    BroadcastEventType,
    BroadcastPriority,
    BroadcastTenantMissingError,
    get_broadcast_service,
)
from src.config.settings import get_settings
from src.core.container_manager_core import (
    get_container_manager,  # TODO: resolve cross-service dependency
)
from src.core.container_metadata_registry import ContainerMetadataRegistry
from src.core.git_clone_credentials import (
    CLONE_TOKEN_ENV,
    clone_credential_shell_args,
    scrub_and_verify,
)
from src.core.kubernetes_container_provisioner import (
    GitCredentialUnavailableError,
    platform_token_or_refuse,
    refuse_if_platform_repo_forbidden,
)
from src.models.container_models import (
    ContainerInfo,
    ContainerSpec,
    ContainerStatus,
    GitHubRepoRequest,
)

logger = logging.getLogger(__name__)


class ContainerLifecycleManager:
    """Manages the lifecycle of Docker containers."""

    def __init__(self, docker_client: Any, registry: ContainerMetadataRegistry, project_name: str):
        """
        Initialize the container lifecycle manager.

        Args:
            docker_client: Docker client instance
            registry: Container metadata registry
            project_name: Project name for container labeling
        """
        self.client = docker_client
        self.registry = registry
        self.project_name = project_name

    def _write_registry_entry(self, info: ContainerInfo) -> None:
        """Upsert a container into the metadata registry.

        Anything that resolves a container by name — including coding-realtime's
        publish-side ownership check — reads this registry, so a lifecycle event
        is only deliverable once this has run.
        """
        # Atomic single-field HSET. NOT load()+save(): save() has replace
        # semantics (it HDELs every field absent from its payload), so writing
        # one entry through a whole-registry snapshot deletes any container a
        # sibling replica created after that snapshot was taken.
        self.registry.upsert(info.container_name, info)

    def _discard_registry_entry(self, container_name: str) -> None:
        """Drop a registry row (best-effort) so a failed create leaks nothing."""
        try:
            # Atomic single-field HDEL — see _write_registry_entry.
            if self.registry.remove(container_name):
                logger.info(f"🧹 Dropped provisional registry entry for {container_name}")
        except (OSError, ValueError) as e:
            logger.warning(f"⚠️ Failed to drop provisional registry entry for {container_name}: {e}")

    async def create_container(
        self,
        request: GitHubRepoRequest,
        env_vars: dict[str, str] | None = None,
        *,
        tenant_id: str | None = None,
    ) -> ContainerInfo:
        """Create and start a Docker container for GitHub repository.

        Args:
            request: GitHub repository provisioning request
            env_vars: Optional environment variables from secret stores
            tenant_id: The OWNING tenant of the container being provisioned.

        Where the tenant comes from: ``POST /api/workspaces/`` resolves the
        authenticated caller via ``get_current_user_with_tenant`` and passes
        ``user.effective_tenant_id`` into ``DockerService.create_container``,
        which forwards it here. ``request.tenant_id`` is the *body* field and is
        only a fallback — it is whatever the client chose to send, so it is
        usually ``None``. That is why every broadcast on this path was
        header-less: the lifecycle manager only ever looked at
        ``request.tenant_id`` while the real tenant sat one frame up in
        ``DockerService``.

        Raises:
            BroadcastTenantMissingError: neither source yielded a tenant. Checked
                up front, before any Docker work or registry write, so the
                failure is loud and leaves nothing half-created. Defaulting it
                would stamp a wrong owner onto ``ContainerInfo`` and authorise
                every subsequent broadcast against a tenant that does not own the
                container.
        """
        owning_tenant = tenant_id or request.tenant_id
        if not owning_tenant:
            raise BroadcastTenantMissingError(
                "Refusing to provision a container with no owning tenant "
                f"(repo_url={request.repo_url!r}): the tenant must come from the "
                "authenticated caller (User.effective_tenant_id) or an explicit "
                "request.tenant_id. It is never defaulted — a guessed owner is a "
                "cross-tenant authorisation bug, and every lifecycle broadcast for "
                "this container would be published under it."
            )

        # This path's clone (`git clone --depth 1 --branch ...`, below) never
        # checks out a SHA, so a history cutoff here would be a plain shallow
        # clone with nothing cut. Refuse up front, before any port/registry work.
        if getattr(request, "history_cutoff", False):
            raise ValueError("history_cutoff is only supported by the kubernetes provisioner")

        # Name of the provisionally-registered container, if we got that far.
        # Used to roll the registry row back on failure paths that the typed
        # except clause below does not cover.
        provisional_name: str | None = None
        try:
            container_name = request.container_name
            if not container_name:
                repo_name = self._extract_repo_name(request.repo_url)
                if request.provision_type == "pull_request" and request.pr_number:
                    container_name = f"codegraph-{repo_name}-pr-{request.pr_number}"
                else:
                    container_name = f"codegraph-{repo_name}"

            container_name = self._validate_container_name(container_name)
            assert (
                container_name  # Type narrowing: guaranteed to be str after validation
            )
            port = self._find_available_port()
            logger.info(f"🔧 Allocated port {port} for container {container_name}")

            if not self.client:
                raise Exception("Docker client not available - cannot create container")

            logger.info(f"🚀 Starting container creation for {container_name}")

            # Dual-token architecture: extract repo info and store user token
            parsed_url = urlparse(str(request.repo_url))
            path_parts = parsed_url.path.strip("/").split("/")
            original_repo_owner = path_parts[0] if len(path_parts) > 0 else ""
            original_repo_name = path_parts[1].replace(".git", "") if len(path_parts) > 1 else ""

            # Get or generate user_id
            user_id = request.user_id or f"anonymous-{uuid7().hex[:8]}"

            # Get token manager
            token_manager = get_github_token_manager()
            system_owner = token_manager.get_system_owner()

            # Store an inline BYO PAT in the secrets service; an id supplied on the
            # request wins (the row already exists — see the helper's docstring).
            # A raise fails the provision rather than logging a success that never
            # happened, which is what the old coding-github webhook POST did.
            secret_config_id = await token_manager.resolve_provision_secret_config_id(
                access_token=request.access_token,
                existing_secret_config_id=getattr(request, "secret_config_id", None),
                user_id=user_id,
                container_id=container_name,
                # ``owning_tenant`` (resolved above, and required — line 138) is the
                # tenant this container belongs to. Using the raw ``request.tenant_id``
                # here would write the row under a different partition than the one
                # ``resolve_git_token`` later reads with, and the read would 404.
                tenant_id=owning_tenant,
            )

            # Clone credential — the SAME ladder as the K8s provisioner:
            #   1. BYO token (request.access_token; crud_write resolves it from
            #      secret_config_id on the way in) — always wins.
            #   2. Platform owner: coding-github's App token, else a typed 412.
            #      The org PAT is never used.
            #   3. Customer owner: a tenant-bound App token from coding-github
            #      (tenant in the body), else no credential for a repo coding-github
            #      positively called PUBLIC, else a typed 412. Never the PAT.
            # Resolved before the provisional registry row, so a refusal leaves
            # nothing half-created. This path used to clone EVERY owner with
            # get_system_token() and ignore the BYO token altogether.
            canonical_repo = canonical_github_repo(str(request.repo_url))
            if canonical_repo is None:
                raise ValueError(
                    f"repo_url is not a canonical GitHub repository URL: {request.repo_url!r}"
                )
            clone_token = await self._resolve_clone_token(
                request, canonical_repo[0], owning_tenant, token_manager
            )

            # These are derived purely from the container name + settings, so
            # they are known before the clone runs and can be stamped onto the
            # provisional CREATING registry row below.
            resolved_codebase_path_in_api_server = (
                f"/app/provisioned_containers/{container_name}/repo"
            )
            public_endpoint = (
                f"{get_settings().main_api_endpoint.rstrip('/')}/api/workspaces/{container_name}"
            )

            container_info = ContainerInfo(
                container_name=container_name,
                api_endpoint=public_endpoint,
                internal_endpoint="",
                public_endpoint=public_endpoint,
                code_graph_id="",
                codebase_path=resolved_codebase_path_in_api_server,
                repo_url=str(request.repo_url),
                branch=request.branch,
                status=ContainerStatus.CREATING,
                description=request.description or f"Container for {request.repo_url}",
                pr_number=(request.pr_number if request.provision_type == "pull_request" else None),
                provision_type=request.provision_type,
                github_access_token=None,  # Deprecated - no longer store plaintext tokens
                # Legacy alias of secret_config_id — see github_container_provisioner.
                user_token_id=secret_config_id,
                # The secrets-service reference the fork-push + upstream-PR paths
                # re-resolve the per-tenant GIT_TOKEN from. This row previously
                # dropped it entirely, so a Docker-provisioned container had no way
                # back to its own token.
                secret_config_id=secret_config_id,
                user_id=user_id,
                original_repo_owner=original_repo_owner,
                original_repo_name=original_repo_name,
                system_fork_owner=system_owner,
                graph_initialized=False,
                # Stamp the provisioning tenant onto the registered ContainerInfo
                # (its documented purpose) — registering None defeated tenant
                # scoping / BOLA for every downstream container lookup, incl. the
                # broadcast publish + WS subscribe validation.
                tenant_id=owning_tenant,
            )

            # Register the CREATING row BEFORE broadcasting "creating".
            # coding-realtime's publish endpoint resolves the container for the
            # calling tenant first (validate_and_get_container_for_tenant) and
            # maps every miss to a 404 — so a "creating" event emitted ahead of
            # the registry write was ALWAYS rejected and dropped, and the
            # fire-and-forget caller never saw it. The clone below is the slow
            # part this event exists to cover, so the row has to land first.
            # Both failure paths at the bottom of this method delete the row, so
            # a failed create never leaves a half-registered container behind.
            self._write_registry_entry(container_info)
            provisional_name = container_name
            logger.debug(f"📝 Registered provisional CREATING entry for {container_name}")

            # Emit container started event (status: creating)
            try:
                broadcast_service = get_broadcast_service()
                creating_result = await broadcast_service.broadcast(
                    event_type=BroadcastEventType.CONTAINER_STARTED,
                    data={
                        "container_name": container_name,
                        "status": "creating",
                        "repo_url": str(request.repo_url),
                        "branch": request.branch,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                    priority=BroadcastPriority.HIGH,
                    tenant_id=owning_tenant,
                )
                # ok is DERIVED from the publish response — never assumed.
                if creating_result.ok:
                    logger.debug(f"📡 Broadcast: Container {container_name} status=creating")
                else:
                    logger.warning(
                        "📡 Broadcast of status=creating for %s was NOT accepted "
                        "(status=%s, not_found=%s): %s",
                        container_name,
                        creating_result.status_code,
                        creating_result.not_found,
                        creating_result.error,
                    )
            except (OSError, httpx.HTTPError) as broadcast_error:
                # Best-effort by design: never fail the create on a TRANSPORT
                # error. Narrow on purpose — the previous blanket `except
                # Exception` also swallowed TypeError (wrong call signature) and
                # BroadcastTenantMissingError, i.e. exactly the bugs that made
                # this whole path silently dead.
                logger.warning(f"Failed to broadcast container started event: {broadcast_error}")

            # Clone repo into the shared volume using a short-lived alpine/git
            # container.  This mirrors the K8s init-container pattern so both
            # Docker-compose and K8s paths produce the same layout:
            #   /app/provisioned_containers/{name}/repo
            container_manager = get_container_manager()
            shared_volume_name = container_manager._resolve_shared_volume_name()

            # Clone URL is the canonical (owner, repo) — the same parse that chose
            # the credential — never the raw request string, and NEVER carries the
            # token (auditor P0, #138): git records the clone URL verbatim as
            # remote.origin.url, so `https://x-access-token:$GIT_TOKEN@...` left
            # the token at rest in the shared volume's repo/.git/config. The token
            # rides in the clone container's env only, read by a process-scoped,
            # host/path-bound credential helper; the scrub proves no copy remains.
            clone_url = canonical_github_url(*canonical_repo)
            clone_env: dict[str, str] = {}
            cred = ""
            if clone_token:
                cred = clone_credential_shell_args(GITHUB_HOST, "/".join(canonical_repo)) + " "
                clone_env[CLONE_TOKEN_ENV] = clone_token

            # Sanitize branch name to prevent shell injection (only allow
            # alphanumeric, dash, underscore, dot, and forward slash).
            sanitized_branch = ""
            if request.branch:
                import re as _re

                if not _re.match(r"^[a-zA-Z0-9_./-]+$", request.branch):
                    raise ValueError(
                        f"Invalid branch name: {request.branch!r} — "
                        "only alphanumeric, dash, underscore, dot, and slash allowed"
                    )
                sanitized_branch = request.branch

            branch_flag = f"--branch '{sanitized_branch}' " if sanitized_branch else ""
            dest = f"/provisioned_containers/{container_name}/repo"
            scrub = scrub_and_verify(
                dest,
                clone_url,
                token_env=CLONE_TOKEN_ENV if clone_token else None,
                wipe_on_failure=True,
            )
            clone_cmd = (
                f"rm -rf {dest} && "
                f"git {cred}clone --depth 1 {branch_flag}{clone_url} {dest} "
                f"&& {{ {scrub}; }} && chmod -R 755 {dest}"
            )

            logger.info(
                f"📦 Cloning {request.repo_url} (branch={request.branch}) into shared volume for {container_name}"
            )

            self.client.containers.run(
                image="alpine/git:latest",
                entrypoint="sh",
                command=["-c", clone_cmd],
                volumes={
                    shared_volume_name: {
                        "bind": "/provisioned_containers",
                        "mode": "rw",
                    }
                },
                environment=clone_env,
                remove=True,
                detach=False,
            )
            logger.info(f"✅ Repository cloned into shared volume for {container_name}")

            # Promote the provisional row: same identity/tenant, status RUNNING.
            updated_info = container_info.model_copy(update={"status": ContainerStatus.RUNNING})
            self._write_registry_entry(updated_info)

            logger.info(
                f"✅ Container {container_name} fully provisioned with API server path: {resolved_codebase_path_in_api_server}"
            )

            # Emit container running event
            try:
                broadcast_service = get_broadcast_service()
                await broadcast_service.broadcast(
                    event_type=BroadcastEventType.CONTAINER_STARTED,
                    data={
                        "container_name": container_name,
                        "status": "running",
                        "repo_url": str(request.repo_url),
                        "branch": request.branch,
                        "api_endpoint": updated_info.api_endpoint,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                    priority=BroadcastPriority.HIGH,
                    tenant_id=owning_tenant,
                )
                logger.debug(f"📡 Broadcast: Container {container_name} status=running")
            except (OSError, httpx.HTTPError) as broadcast_error:
                logger.warning(f"Failed to broadcast container running event: {broadcast_error}")

            return updated_info

        except (docker.errors.DockerException, ValueError, RuntimeError) as e:
            logger.error(
                f"❌ Failed to create container {(container_name if 'container_name' in locals() else 'unknown')}: {str(e)}",
                exc_info=True,
            )

            # Emit container error event
            if "container_name" in locals() and container_name is not None:
                try:
                    broadcast_service = get_broadcast_service()
                    await broadcast_service.broadcast(
                        event_type=BroadcastEventType.CONTAINER_STOPPED,
                        data={
                            "container_name": container_name,
                            "status": "error",
                            "error_message": str(e),
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                        priority=BroadcastPriority.CRITICAL,
                        tenant_id=owning_tenant,
                    )
                    logger.debug(f"📡 Broadcast: Container {container_name} status=error")
                except (OSError, httpx.HTTPError) as broadcast_error:
                    logger.warning(f"Failed to broadcast container error event: {broadcast_error}")

            if "container_name" in locals() and container_name is not None:
                logger.info(f"🧹 Attempting cleanup of failed container {container_name}")
                try:
                    await self.remove_container_by_name(container_name)
                    logger.info(f"✅ Cleaned up failed container {container_name}")
                except docker.errors.DockerException as cleanup_e:
                    logger.warning(f"⚠️ Failed to cleanup container {container_name}: {cleanup_e}")
                    # remove_container_by_name is what normally drops the
                    # registry row; if it blew up, drop the provisional row
                    # directly so a failed create leaks nothing.
                    self._discard_registry_entry(container_name)
            raise RuntimeError(f"Container creation failed: {str(e)}")

        except BaseException:
            # Failure modes the typed clause above does not cover (cancellation,
            # token-manager / network errors, ...). They keep propagating
            # unchanged, but must not leave the provisional CREATING row behind.
            if provisional_name is not None:
                self._discard_registry_entry(provisional_name)
            raise

    @retry_docker(max_attempts=3, min_wait=1.0, max_wait=5.0)
    async def start_container(self, docker_id: str) -> None:
        """Start a container by ID with retry logic."""
        if not self.client:
            raise Exception("Docker not available")
        try:
            container = self.client.containers.get(docker_id)
            container.start()
            logger.debug(f"Started container {docker_id}")
        except docker.errors.DockerException as e:
            logger.error(f"Failed to start container {docker_id}: {e}")
            raise

    @retry_docker(max_attempts=3, min_wait=1.0, max_wait=5.0)
    async def stop_container(self, docker_id: str, timeout: int = 10) -> None:
        """Stop a container by ID with retry logic."""
        if not self.client:
            logger.warning(f"Docker not available, cannot stop container {docker_id}")
            return
        try:
            container = self.client.containers.get(docker_id)
            container.stop(timeout=timeout)
            logger.debug(f"Stopped container {docker_id}")
        except docker.errors.NotFound:
            logger.info(f"Container {docker_id} not found for stopping - already stopped")
        except docker.errors.DockerException as e:
            logger.warning(f"Failed to stop container {docker_id}: {e}")

    @retry_docker(max_attempts=3, min_wait=1.0, max_wait=5.0)
    async def remove_container_by_name(self, container_name: str) -> bool:
        """Remove a container by its name and clean up resources with retry logic."""
        if not self.client:
            logger.info(f"Docker not available, skipping container removal: {container_name}")
            return True

        try:
            container = self.client.containers.get(container_name)
            logger.info(f"Stopping container {container_name}")
            container.stop()
            logger.info(f"Removing container {container_name}")
            container.remove()
        except docker.errors.NotFound:
            logger.info(f"Container {container_name} not found in Docker, nothing to remove")
        except docker.errors.DockerException as e:
            logger.error(f"Error removing Docker container {container_name}: {e}")

        # Remove from registry (atomic single-field HDEL — see _write_registry_entry).
        try:
            if self.registry.remove(container_name):
                logger.info(f"Removed {container_name} from registry")
            else:
                logger.info(f"Container {container_name} not found in registry")
        except (OSError, ValueError) as e:
            logger.error(f"Error removing {container_name} from registry: {e}")

        return True

    @retry_docker(max_attempts=3, min_wait=1.0, max_wait=5.0)
    async def remove_container_by_id(self, docker_id: str, force: bool = True) -> None:
        """Remove a container by ID with retry logic."""
        if not self.client:
            logger.warning(f"Docker not available, cannot remove container {docker_id}")
            return
        try:
            container = self.client.containers.get(docker_id)
            container.remove(force=force)
            logger.debug(f"Removed container {docker_id}")
        except docker.errors.NotFound:
            logger.info(f"Container {docker_id} not found for removal - already removed")
        except docker.errors.DockerException as e:
            logger.warning(f"Failed to remove container {docker_id}: {e}")

    async def remove_container(self, container: ContainerInfo) -> bool:
        """Remove existing container and cleanup."""
        if not self.client:
            logger.warning(
                f"Docker not available, removing {container.container_name} from registry only"
            )
            try:
                # Atomic single-field HDEL — see _write_registry_entry.
                if self.registry.remove(container.container_name):
                    logger.info(f"Removed {container.container_name} from registry")
                return True
            except (OSError, ValueError) as e:
                logger.error(f"Failed to remove container from registry: {e}")
                return False

        try:
            container_name = container.container_name
            try:
                docker_container = self.client.containers.get(container_name)
                docker_container.stop(timeout=10)
                docker_container.remove(force=True)
                logger.info(f"Docker container {container_name} stopped and removed")
            except docker.errors.NotFound:
                logger.info(f"Docker container {container_name} not found - already removed")
            except docker.errors.DockerException as e:
                logger.error(f"Error removing Docker container {container_name}: {e}")

            # Remove from registry (atomic single-field HDEL — see _write_registry_entry).
            if self.registry.remove(container_name):
                logger.info(f"Removed {container_name} from registry")

            return True
        except (docker.errors.DockerException, OSError) as e:
            logger.error(f"Failed to remove container {container.container_name}: {e}")
            return False

    @staticmethod
    async def _resolve_clone_token(
        request: GitHubRepoRequest,
        owner: str,
        tenant_id: str | None,
        token_manager: Any,
    ) -> str | None:
        """BYO, else an App token. A platform owner with no App token (any
        reason) is a typed 412 — the org PAT is never a fallback.

        Returns ``None`` only for a customer repo coding-github positively called
        public; raises ``GitCredentialUnavailableError`` (412) otherwise.
        """
        byo = (request.access_token or "").strip() or None
        if byo:
            return byo
        canonical = canonical_github_repo(str(request.repo_url))
        if canonical is None:
            raise ValueError(
                f"repo_url is not a canonical GitHub repository URL: {request.repo_url!r}"
            )
        # The CANONICAL url, never the raw request string, goes to coding-github.
        canonical_url = canonical_github_url(*canonical)
        # coding-github is asked even for a platform owner: it alone decides
        # whether this tenant may have a platform repo (``platform_repo_forbidden``,
        # coding-github #107), and that refusal must stop the ladder BEFORE the PAT.
        credential = await token_manager.get_app_clone_credential(
            repo_url=canonical_url, tenant_id=tenant_id
        )
        refuse_if_platform_repo_forbidden(
            credential, repo_url=canonical_url, owner=owner, tenant_id=tenant_id
        )
        if is_platform_owner(owner):
            return platform_token_or_refuse(
                credential, repo_url=canonical_url, owner=owner, tenant_id=tenant_id
            )
        if credential.token:
            return credential.token
        if credential.is_public:
            return None
        message = (
            f"{APP_NOT_INSTALLED_ERROR} for {owner}: no BYO token and no GitHub App "
            f"installation bound to tenant {tenant_id or '<none>'} covers "
            f"{request.repo_url} (coding-github reason="
            f"{credential.reason or credential.error or '-'}). Install the nonagentic "
            f"GitHub App on {owner} from the fr workspace, "
            "or pass a GitHub token as access_token on the provision request."
        )
        logger.error(message)
        raise GitCredentialUnavailableError(
            repo_url=str(request.repo_url),
            owner=owner,
            reason=APP_NOT_INSTALLED_ERROR,
            message=message,
        )

    def _validate_container_prerequisites(self, spec: ContainerSpec) -> dict[str, bool]:
        """Validate all prerequisites before creating container."""
        validation = {
            "docker_available": False,
            "image_exists": False,
            "network_exists": False,
            "socket_accessible": False,
        }

        try:
            if self.client:
                validation["docker_available"] = True
                logger.info("✅ Docker client is available and responsive")
            else:
                logger.error("❌ Docker client not available")
                return validation

            # Check if image exists
            try:
                self.client.images.get(spec.image)
                validation["image_exists"] = True
                logger.info(f"✅ Image {spec.image} exists")
            except docker.errors.DockerException as e:
                logger.error(f"❌ Image {spec.image} not found: {e}")
                logger.info("💡 To fix: docker-compose build api")

            # Check if network exists
            network_name = "workflow-system-network"
            try:
                self.client.networks.get(network_name)
                validation["network_exists"] = True
                logger.info(f"✅ Network {network_name} exists")
            except docker.errors.DockerException as e:
                logger.error(f"❌ Network {network_name} not found: {e}")
                logger.info(f"💡 To fix: docker network create {network_name}")

            # Check socket accessibility
            try:
                socket_path = "/var/run/docker.sock"
                if os.path.exists(socket_path) and os.access(socket_path, os.R_OK | os.W_OK):
                    validation["socket_accessible"] = True
                    logger.info("✅ Docker socket is accessible")
                else:
                    logger.error("❌ Docker socket access issues")
                    logger.info("💡 To fix: sudo chmod 666 /var/run/docker.sock")
            except OSError as e:
                logger.error(f"❌ Socket check failed: {e}")

        except docker.errors.DockerException as e:
            logger.error(f"❌ Prerequisites validation failed: {e}")

        return validation

    async def _verify_container_running(
        self,
        docker_id: str,
        container_name: str,
        timeout: int = 30,
        *,
        tenant_id: str,
    ) -> bool:
        """Verify container is actually running and responding.

        ``tenant_id`` is the container's OWNING tenant, required for the same
        reason as in :meth:`create_container`: the health events emitted here are
        published to coding-realtime and must be authorised against the tenant
        that owns the container. These three call sites passed no tenant at all,
        so they were riding coding-realtime's pod-CIDR trust fallback and would
        401 the moment it is turned off.
        """
        if not self.client:
            logger.error(
                f"❌ Docker client not available for container verification: {container_name}"
            )
            return False
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    container = self.client.containers.get(docker_id)
                    container.reload()

                    if container.status == "running":
                        logger.info(f"✅ Container {container_name} is running")
                        try:
                            # Use safe exec_run from monitoring module
                            result = container.exec_run(["echo", "health_check"])
                            if result.exit_code == 0:
                                logger.info(f"✅ Container {container_name} health check passed")
                                # Emit health event
                                try:
                                    broadcast_service = get_broadcast_service()
                                    await broadcast_service.broadcast(
                                        event_type=BroadcastEventType.CONTAINER_HEALTH,
                                        data={
                                            "container_name": container_name,
                                            "status": "running",
                                            "health": "healthy",
                                            "timestamp": datetime.now(UTC).isoformat(),
                                        },
                                        priority=BroadcastPriority.NORMAL,
                                        tenant_id=tenant_id,
                                    )
                                except (OSError, httpx.HTTPError) as broadcast_error:
                                    logger.debug(
                                        f"Failed to broadcast health event: {broadcast_error}"
                                    )
                                return True
                            else:
                                logger.warning(f"⚠️ Container {container_name} health check failed")
                                # Emit unhealthy event
                                try:
                                    broadcast_service = get_broadcast_service()
                                    await broadcast_service.broadcast(
                                        event_type=BroadcastEventType.CONTAINER_HEALTH,
                                        data={
                                            "container_name": container_name,
                                            "status": "running",
                                            "health": "unhealthy",
                                            "timestamp": datetime.now(UTC).isoformat(),
                                        },
                                        priority=BroadcastPriority.NORMAL,
                                        tenant_id=tenant_id,
                                    )
                                except (OSError, httpx.HTTPError) as broadcast_error:
                                    logger.debug(
                                        f"Failed to broadcast health event: {broadcast_error}"
                                    )
                        except docker.errors.DockerException as health_e:
                            logger.warning(
                                f"⚠️ Container {container_name} health check error: {health_e}"
                            )
                            return (
                                True  # Container is running, health check failure is not critical
                            )
                        return True

                    elif container.status in ["exited", "dead"]:
                        logger.error(
                            f"❌ Container {container_name} failed to start: {container.status}"
                        )
                        logs = container.logs(tail=50).decode("utf-8", errors="ignore")
                        logger.error(f"Container logs:\n{logs}")

                        # Emit health event for failed container
                        try:
                            broadcast_service = get_broadcast_service()
                            await broadcast_service.broadcast(
                                event_type=BroadcastEventType.CONTAINER_HEALTH,
                                data={
                                    "container_name": container_name,
                                    "status": container.status,
                                    "health": "unhealthy",
                                    "error": f"Container failed to start: {container.status}",
                                    "timestamp": datetime.now(UTC).isoformat(),
                                },
                                priority=BroadcastPriority.HIGH,
                                tenant_id=tenant_id,
                            )
                        except (OSError, httpx.HTTPError) as broadcast_error:
                            logger.debug(f"Failed to broadcast health event: {broadcast_error}")
                        return False
                    else:
                        logger.info(
                            f"⏳ Container {container_name} status: {container.status}, waiting..."
                        )
                        await asyncio.sleep(2)

                except docker.errors.DockerException as e:
                    logger.warning(f"Error checking container {container_name}: {e}")
                    await asyncio.sleep(2)

            logger.error(f"❌ Container {container_name} failed to start within {timeout}s")
            return False

        except docker.errors.DockerException as e:
            logger.error(f"❌ Container verification failed for {container_name}: {e}")
            return False

    def _get_container_status(self, container_name: str) -> ContainerStatus:
        """Get current container status from Docker with enhanced error handling."""
        if not self.client:
            return ContainerStatus.ERROR

        try:
            container = self.client.containers.get(container_name)
            if container.status == "running":
                try:
                    result = container.exec_run(["echo", "ping"], timeout=3)
                    if result.exit_code == 0:
                        return ContainerStatus.RUNNING
                    else:
                        logger.warning(f"Container {container_name} running but not responsive")
                        return ContainerStatus.ERROR
                except docker.errors.DockerException as exec_err:
                    logger.debug("Exec check failed, assuming running: %s", exec_err)
                    return ContainerStatus.RUNNING
            elif container.status in ["stopped", "exited"]:
                return ContainerStatus.STOPPED
            elif container.status in ["created", "restarting"]:
                return ContainerStatus.CREATING
            else:
                logger.warning(f"Unknown container status for {container_name}: {container.status}")
                return ContainerStatus.ERROR

        except docker.errors.NotFound:
            logger.debug(f"Container {container_name} not found in Docker")
            return ContainerStatus.ERROR
        except docker.errors.DockerException as e:
            logger.error(f"Error checking status for {container_name}: {e}")
            return ContainerStatus.ERROR

    def _extract_repo_name(self, repo_url: str) -> str:
        """Extract repository name from URL."""
        from urllib.parse import urlparse

        repo_url_str = str(repo_url)
        parsed_url = urlparse(repo_url_str)
        path_parts = parsed_url.path.strip("/").split("/")
        repo_name = path_parts[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        return repo_name

    def _validate_container_name(self, container_name: str) -> str:
        """Validate container name format (no conflict resolution - containers are replaced)."""
        import re

        if not container_name or len(container_name) < 3:
            container_name = "codegraph-container"
        if not re.match("^[a-zA-Z0-9][a-zA-Z0-9_.-]*$", container_name):
            container_name = re.sub("[^a-zA-Z0-9_.-]", "-", container_name)
            if not container_name[0].isalnum():
                container_name = f"codegraph-{container_name}"
        # Docker container names must not exceed 128 characters
        if len(container_name) > 128:
            container_name = container_name[:128]
        return container_name

    def _find_available_port(self) -> int:
        """Find an available host port starting from base_port."""
        import socket

        base_port = 8700
        used_ports = set()

        if self.client:
            try:
                for container in self.client.containers.list(all=True):
                    if container.ports:
                        for _internal_port, host_configs in container.ports.items():
                            if host_configs:
                                for host_config in host_configs:
                                    if host_config.get("HostPort"):
                                        try:
                                            used_ports.add(int(host_config["HostPort"]))
                                        except ValueError:
                                            pass
            except (docker.errors.DockerException, RuntimeError) as e:
                logger.warning(f"Could not check Docker ports: {e}")

        for port in range(base_port, base_port + 100):
            if port not in used_ports:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(0.1)
                        if s.connect_ex(("127.0.0.1", port)) != 0:
                            return port
                except OSError:
                    pass

        raise RuntimeError(f"No available ports in range {base_port}-{base_port + 99}")

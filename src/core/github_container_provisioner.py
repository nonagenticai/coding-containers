"""
GitHub-specific container provisioning.
"""

from __future__ import annotations

import logging
import os
import shlex
from urllib.parse import urlparse
from uuid import uuid7

import docker.errors

from src.clients.github_client import (
    APP_NOT_INSTALLED_ERROR,
    GITHUB_HOST,
    canonical_github_repo,
    canonical_github_url,
    get_github_token_manager,  # TODO: implement client
    is_platform_owner,
)
from src.core.container_manager_core import CoreContainerManager
from src.core.container_spec import ContainerSpec, GitHubRepoRequest
from src.core.git_clone_credentials import (
    CLONE_TOKEN_ENV,
    clone_credential_shell_args,
    scrub_and_verify,
)
from src.core.kubernetes_container_provisioner import (
    GitCredentialUnavailableError,
    get_k8s_container_provisioner,
    platform_token_or_refuse,
    refuse_if_platform_repo_forbidden,
    require_tenant_for_platform_repo,
)
from src.models.container_models import ContainerInfo
from src.services.docker_service import get_docker_service

logger = logging.getLogger(__name__)


def _require_canonical_repo(repo_url: str | None) -> tuple[str, str]:
    """``(owner, repo)`` of a canonical GitHub remote, else ``ValueError``.

    Every owner used to pick a credential and every clone URL in this module is
    derived from this pair, never from a raw ``urlparse`` of the request: git
    resolves ``..`` and ``%2e%2e`` before talking to GitHub, so a first-segment
    parse of ``https://github.com/nonagenticai/../../victim/private`` names the
    platform owner (system PAT) while the clone reaches ``victim/private``.
    """
    canonical = canonical_github_repo(repo_url)
    if canonical is None:
        raise ValueError(
            f"repo_url is not a canonical GitHub repository URL "
            f"(https://github.com/<owner>/<repo>): {repo_url!r}"
        )
    return canonical


class GitHubContainerProvisioner:
    """Handles GitHub repository provisioning for container analysis."""

    def __init__(self, container_manager: CoreContainerManager):
        self.container_manager = container_manager
        # Switch to K8s provisioning when running in-cluster
        self.use_k8s = os.environ.get("KUBERNETES_IN_CLUSTER", "false").lower() == "true"
        self.k8s_provisioner = get_k8s_container_provisioner() if self.use_k8s else None

    @staticmethod
    def _validate_clone_url(url: str) -> None:
        """Validate that a clone URL uses an allowed protocol.

        Only ``https://`` and ``git@github.com:`` SSH URLs are permitted to
        prevent protocol-based attacks (e.g. ``file://``, ``ssh://``).

        Raises:
            ValueError: If the URL uses a disallowed protocol.
        """
        if url.startswith("https://"):
            return
        if url.startswith("git@github.com:"):
            return
        raise ValueError(
            f"Disallowed git clone URL protocol: {url!r} -- "
            "only https:// and git@github.com: URLs are allowed"
        )

    def validate_repository(self, request: GitHubRepoRequest) -> bool:
        """
        Validate access to a GitHub repository without provisioning.

        Args:
            request: Container provision request containing GitHub details

        Returns:
            True if repository is accessible, False otherwise
        """
        try:
            self._validate_github_repository_access(request)
            return True
        except ValueError:
            return False

    async def provision_from_github(self, request: GitHubRepoRequest) -> ContainerInfo:
        """
        Provision a container from a GitHub repository using dual-token architecture.

        Args:
            request: Container provision request containing GitHub details

        Returns:
            ContainerInfo containing provision results and metadata
        """
        try:
            logger.info("Starting GitHub provisioning for %s", request.repo_url)

            # Refuse a non-canonical remote BEFORE any credential or pre-flight.
            # The owner that selects the credential and the URL that is cloned
            # both come from this one pair (see _require_canonical_repo).
            original_repo_owner, original_repo_name = _require_canonical_repo(request.repo_url)
            canonical_url = canonical_github_url(original_repo_owner, original_repo_name)

            # Validate GitHub repository access (using user token if provided)
            self._validate_github_repository_access(request)

            # Use Kubernetes-native provisioning path when enabled
            if self.use_k8s and self.k8s_provisioner:
                logger.info("K8s mode enabled - delegating provisioning to K8sContainerProvisioner")
                return await self.k8s_provisioner.provision_from_github(request)

            # The Docker-path clone below (`_clone_repo_into_shared_volume`)
            # passes `branch` to `--branch`/`-b` and never checks out a SHA, so a
            # history cutoff here would be a plain clone of
            # a branch tip with nothing cut. Refuse rather than silently degrade.
            if getattr(request, "history_cutoff", False):
                raise ValueError("history_cutoff is only supported by the kubernetes provisioner")

            # Get or generate user_id
            user_id = request.user_id or f"anonymous-{uuid7().hex[:8]}"

            # Get token manager
            token_manager = get_github_token_manager()
            system_owner = token_manager.get_system_owner()

            # Generate container name if not provided
            container_name = request.container_name
            if not container_name:
                repo_name = self._extract_repo_name(request.repo_url)
                container_name = f"codegraph-{repo_name}-{uuid7().hex[:6]}"

            # Store an inline BYO PAT in the secrets service and keep the id it
            # returns. ``request.secret_config_id`` wins when present: the token is
            # then ALREADY in the secrets service under fr's
            # ``project:{id}:github_token`` row (crud_write resolved
            # ``access_token`` from it on the way in), so re-storing it would write
            # a duplicate row holding the same secret under a second identifier.
            # A raise here fails the provision — correct: the caller asked for the
            # token to be persisted, and a container whose token was silently
            # dropped fails later at git with a message that blames git.
            secret_config_id = await token_manager.resolve_provision_secret_config_id(
                access_token=request.access_token,
                existing_secret_config_id=getattr(request, "secret_config_id", None),
                user_id=user_id,
                container_id=container_name,
                tenant_id=getattr(request, "tenant_id", None),
            )

            # Clone/container credential. A platform owner gets coding-github's
            # App token or a typed 412 — never the org PAT. A customer owner NEVER gets the PAT (tenant-bound invariant): only a
            # BYO token or a public-App token bound to the requesting tenant; a
            # provably-public repo may be cloned with no credential at all.
            if is_platform_owner(original_repo_owner):
                system_token = await self._platform_clone_token(
                    request, original_repo_owner, canonical_url, token_manager
                )
            else:
                system_token = await self._customer_clone_token(
                    request, original_repo_owner, token_manager
                )

            # Find available port
            # Find available port using the container manager's internal method
            # This is intentionally accessing a protected member of the container manager
            # as this class is part of the core system and needs access to internal utilities
            port = self.container_manager._find_available_port()

            # Create container specification using SYSTEM token
            spec = ContainerSpec(
                name=container_name,
                image=self.container_manager.image_name,
                port=port,
                environment={
                    "REPO_URL": canonical_url,
                    "BRANCH": request.branch,
                    # ⛔ NO credential in the long-lived workspace env (it used to
                    # carry GITHUB_ACCESS_TOKEN). The clone takes the token in its
                    # own short-lived container (below) and a push resolves one per
                    # exec (routers/containers/git_push.py); nothing in this repo
                    # reads it from the workspace, and an env var there is readable
                    # by every process the tenant runs, for the workspace's life.
                    "CONTAINER_NAME": container_name,  # So the container knows its own name
                },
                volumes={
                    # Mount CLI credentials from host for agent execution
                    # Matches worker_management pattern: live credentials, externally updated
                    os.path.expanduser("~/.claude"): {
                        "bind": "/root/.claude",
                        "mode": "rw",  # Read-write for debug logs
                    },
                    os.path.expanduser("~/.codex"): {
                        "bind": "/root/.codex",
                        "mode": "rw",  # Read-write for debug logs
                    },
                },
                command=None,
                working_dir="/app",
            )

            # Clone repo into shared volume (mirrors K8s init-container pattern)
            self._clone_repo_into_shared_volume(
                container_name=container_name,
                repo_url=canonical_url,
                branch=request.branch,
                token=system_token,
            )

            # Create the container
            container_info = self.container_manager.create_container(spec)

            # Update container info with GitHub-specific data using dual-token metadata
            updated_info = ContainerInfo(
                container_name=container_info.container_name,
                api_endpoint=container_info.api_endpoint,
                internal_endpoint=container_info.internal_endpoint,
                public_endpoint=container_info.public_endpoint,
                code_graph_id=container_name,  # Use container_name as graph ID for consistent mapping
                codebase_path=f"/app/provisioned_containers/{container_name}/repo",
                repo_url=request.repo_url,
                branch=request.branch,
                status=container_info.status,
                description=request.description or f"Container for {request.repo_url}",
                github_access_token=None,  # Deprecated - no longer store plaintext tokens
                # Same value in both fields, on purpose. ``secret_config_id`` is the
                # one that is READ (secret_resolver, git_push, crud_write,
                # coding-github's upstream-PR); ``user_token_id`` is the legacy alias
                # kept for API compatibility and is now at least truthful, where it
                # used to hold a coding-github id that was never issued.
                user_token_id=secret_config_id,
                user_id=user_id,
                original_repo_owner=original_repo_owner,
                original_repo_name=original_repo_name,
                system_fork_owner=system_owner,
                # Persist the secrets-service reference so push/upstream-PR can
                # re-resolve the per-tenant GIT_TOKEN (source of truth = secrets-service).
                secret_config_id=secret_config_id,
            )

            # Save updated info to registry. Atomic single-field HSET, NOT
            # load()+save(): save() HDELs every field absent from its payload,
            # so a whole-registry write from a stale snapshot deletes any
            # container a sibling replica created since that snapshot.
            docker_service = get_docker_service()
            docker_service.registry.upsert(container_name, updated_info)

            logger.info(
                "GitHub container provisioned successfully: %s (user=%s, system=%s)",
                container_name,
                user_id,
                system_owner,
            )
            return updated_info

        except (
            docker.errors.DockerException,
            OSError,
            ValueError,
            RuntimeError,
        ) as e:
            logger.error("Failed to provision from GitHub: %s", str(e))
            # Clean up container if it was created
            if "container_info" in locals():
                try:
                    # NOTE: this is CoreContainerManager.remove_container (sync,
                    # returns bool) — not the async facade in container_manager.py.
                    self.container_manager.remove_container(container_info.container_name)
                except (docker.errors.DockerException, OSError) as cleanup_error:
                    logger.error("Failed to cleanup container after error: %s", cleanup_error)
            raise

    @staticmethod
    async def _platform_clone_token(
        request: GitHubRepoRequest, owner: str, canonical_url: str, token_manager
    ) -> str | None:
        """BYO, else coding-github's App token, else a typed 412. Never the org PAT.

        Same ladder as the K8s provisioner. coding-github is asked even for a
        platform owner, because it is the only party that knows whether this
        tenant may have a platform repository at all (``platform_repo_forbidden``,
        coding-github #107). A null token — for ANY reason — fails closed.

        ⚠️ ``request.tenant_id`` is the request BODY field, which the route never
        populates. A platform request without it is REFUSED
        (``platform_tenant_missing``) rather than given an implicit platform
        tenant.
        """
        byo = (request.access_token or "").strip() or None
        if byo:
            return byo
        tenant_id = require_tenant_for_platform_repo(
            getattr(request, "tenant_id", None),
            owner=owner,
            repo_url=canonical_url,
            context="docker_provision",
        )
        credential = await token_manager.get_app_clone_credential(
            repo_url=canonical_url, tenant_id=tenant_id
        )
        refuse_if_platform_repo_forbidden(
            credential, repo_url=canonical_url, owner=owner, tenant_id=tenant_id
        )
        return platform_token_or_refuse(
            credential, repo_url=canonical_url, owner=owner, tenant_id=tenant_id
        )

    @staticmethod
    async def _customer_clone_token(
        request: GitHubRepoRequest, owner: str, token_manager
    ) -> str | None:
        """BYO token, else a tenant-bound App token, else None only if provably public.

        Raises ``GitCredentialUnavailableError`` (412, reason
        ``github_app_not_installed``) otherwise — never falls back to the PAT.
        """
        byo = (request.access_token or "").strip() or None
        if byo:
            return byo
        tenant_id = getattr(request, "tenant_id", None)
        # The CANONICAL url, never the raw request string: coding-github parses
        # what it is sent, and the raw form is exactly what canonicalisation exists
        # to keep away from every credential decision.
        canonical_url = canonical_github_url(*_require_canonical_repo(request.repo_url))
        credential = await token_manager.get_app_clone_credential(
            repo_url=canonical_url, tenant_id=tenant_id
        )
        refuse_if_platform_repo_forbidden(
            credential, repo_url=canonical_url, owner=owner, tenant_id=tenant_id
        )
        if credential.token:
            return credential.token
        if credential.is_public:
            return None
        message = (
            f"{APP_NOT_INSTALLED_ERROR} for {owner or '<unknown owner>'}: no BYO token and "
            f"no GitHub App installation bound to tenant {tenant_id or '<none>'} covers "
            f"{request.repo_url} (coding-github reason="
            f"{credential.reason or credential.error or '-'}). Install the nonagentic "
            f"GitHub App on {owner or '<unknown owner>'} from the fr workspace, "
            "or pass a GitHub token as access_token on the provision request."
        )
        logger.error(message)
        raise GitCredentialUnavailableError(
            repo_url=str(request.repo_url),
            owner=owner,
            reason=APP_NOT_INSTALLED_ERROR,
            message=message,
        )

    def _validate_github_repository_access(self, request: GitHubRepoRequest) -> None:
        """Structural pre-flight only: the repo URL must be canonical. No network.

        There is deliberately NO GitHub API probe here, BYO token or not:

        * This pod has no egress to api.github.com. The ``default-restrict``
          NetworkPolicy (gitops ``services/coding-containers/networkpolicy.yaml``)
          admits only in-namespace, kube-dns, the API server, the n8n callback
          host and the coding-* namespaces; github.com:443 is granted only to the
          ``app: codegraph-repo`` workspace pods. Only coding-github talks to
          GitHub (``git_push._remote_repo_info``, ``github_client.CloneCredential``).
          The PyGithub probe that used to run here with a BYO token therefore
          failed or hung every BYO provision before it reached the clone.
        * coding-github has no route that checks a repository with a
          CALLER-SUPPLIED token (``/get-repository`` resolves its own App
          credential), so it cannot be routed there either.

        The clone is the authority: it runs where GitHub IS reachable (the
        workspace pod's ``git-clone`` init container, or the Docker path's clone
        container) with exactly the credential the workspace will use. A clone
        that cannot read the repository surfaces as ``WorkspaceCloneFailedError``
        (exec answers a typed 412 ``WORKSPACE_CLONE_FAILED``); the Docker path's
        clone raises ``ValueError("Failed to clone repository: ...")``.

        No token-derived value is logged here (not a prefix, length or shape).

        Raises:
            ValueError: the repo URL is not a canonical GitHub repository URL.
        """
        owner, repo_name = _require_canonical_repo(request.repo_url)
        logger.info(
            "Pre-flight for %s/%s: canonical URL; repository access is proven by the "
            "clone, not an API probe (BYO token supplied: %s)",
            owner,
            repo_name,
            bool((request.access_token or "").strip()),
        )

    def _clone_repo_into_shared_volume(
        self,
        container_name: str,
        repo_url: str,
        branch: str | None,
        token: str | None,
    ) -> None:
        """Clone repo into the shared provisioned-containers volume.

        Uses a short-lived alpine/git container, mirroring the K8s
        init-container pattern so both Docker and K8s paths produce
        the same layout: /provisioned_containers/{name}/repo
        """
        shared_volume_name = self.container_manager._resolve_shared_volume_name()

        self._validate_clone_url(repo_url)

        # Built ONLY from the canonical pair, never the raw string's path.
        owner, repo = _require_canonical_repo(repo_url)
        clean_url = canonical_github_url(owner, repo)

        # ⛔ The token never enters the URL or the command line (auditor P0, #138).
        # It used to be spliced in as ``https://<token>@github.com/...``: that put
        # it in this container's argv AND, because git records the clone URL
        # verbatim as remote.origin.url, at rest in the shared volume's
        # ``repo/.git/config`` for the life of the workspace. It now travels only
        # in the short-lived clone container's env, read by a process-scoped,
        # host/path-bound credential helper; the scrub then proves no copy remains.
        dest = f"/provisioned_containers/{container_name}/repo"
        qdest = shlex.quote(dest)
        cred = clone_credential_shell_args(GITHUB_HOST, f"{owner}/{repo}") + " " if token else ""
        scrub = scrub_and_verify(
            qdest, clean_url, token_env=CLONE_TOKEN_ENV if token else None, wipe_on_failure=True
        )
        branch_flag = f"--branch {shlex.quote(branch)} " if branch else ""
        clone_cmd = (
            f"git {cred}clone --depth 1 {branch_flag}{clean_url} {qdest} && "
            f"{{ {scrub}; }} && chmod -R 755 {qdest}"
        )

        logger.info(
            "Cloning %s (branch=%s) into shared volume for %s", repo_url, branch, container_name
        )

        try:
            assert self.container_manager.client is not None, "Docker client unavailable"
            self.container_manager.client.containers.run(
                image="alpine/git:latest",
                entrypoint="sh",
                command=["-c", clone_cmd],
                volumes={
                    shared_volume_name: {
                        "bind": "/provisioned_containers",
                        "mode": "rw",
                    }
                },
                environment={CLONE_TOKEN_ENV: token} if token else {},
                remove=True,
                detach=False,
            )
            logger.info("Repository cloned successfully for %s", container_name)
        except (
            docker.errors.APIError,
            docker.errors.ContainerError,
            docker.errors.ImageNotFound,
            OSError,
        ) as e:
            logger.error("Failed to clone repository for %s: %s", container_name, e)
            raise ValueError(f"Failed to clone repository: {e}") from e

    def _extract_repo_name(self, repo_url: str) -> str:
        """Extract repository name from URL"""
        parsed_url = urlparse(str(repo_url))
        path_parts = parsed_url.path.strip("/").split("/")
        repo_name = path_parts[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        return repo_name


# Global instance
_github_provisioner = None


def get_github_container_provisioner() -> GitHubContainerProvisioner:
    """Get the global GitHub container provisioner instance"""
    global _github_provisioner
    if _github_provisioner is None:
        from src.core.container_manager_core import (
            get_container_manager,
        )

        _github_provisioner = GitHubContainerProvisioner(get_container_manager())
    return _github_provisioner

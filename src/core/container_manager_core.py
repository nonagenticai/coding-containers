"""Core container lifecycle management for Docker containers.

This module provides the fundamental container lifecycle management functionality
for the Code Graph Helper application. It handles Docker container creation,
monitoring, and management operations with proper error handling and resource
allocation.

The core container manager focuses on:
- Docker client initialization and connection management
- Container provisioning and lifecycle management
- Port allocation and network configuration
- Error handling and graceful degradation
- Resource cleanup and container termination

Key Features:
    - Automatic Docker client initialization with fallback handling
    - Port allocation for container services
    - Container status monitoring and reporting
    - Graceful degradation when Docker is unavailable
    - Connection retry mechanisms for resilience
    - Comprehensive logging for debugging and monitoring

Classes:
    CoreContainerManager: Main container management class for Docker operations

Usage:
    manager = CoreContainerManager()
    if manager.is_docker_available():
        container = manager.create_container(spec)

Note:
    This module requires Docker to be installed and accessible. When Docker
    is not available, the manager will report errors gracefully and disable
    container functionality rather than failing catastrophically.
"""

from __future__ import annotations

import logging
import os
import re
import socket
from typing import Any
from uuid import uuid7

import docker
import docker.errors
import docker.models.volumes
import docker.types
import httpx

from src.config.settings import get_settings
from src.core.container_spec import ContainerSpec, ContainerStatus
from src.models.container_models import ContainerInfo

# Agent-host delegation URL (when set, delegates container ops to agent-host service)
AGENT_HOST_URL = os.getenv("AGENT_HOST_URL")

logger = logging.getLogger(__name__)


def _build_public_endpoint(container_name: str) -> str:
    """Build the stable gateway-facing endpoint for a provisioned container."""
    base_url = get_settings().main_api_endpoint.rstrip("/")
    return f"{base_url}/api/workspaces/{container_name}"


class CoreContainerManager:
    """Core container lifecycle management for Docker containers.

    This class provides the fundamental container management functionality including
    Docker client initialization, container creation, monitoring, and resource
    management. It handles the low-level Docker operations required for the
    Code Graph Helper application.

    The manager is designed to be resilient and gracefully handle Docker
    unavailability by providing clear error reporting and fallback mechanisms.

    Attributes:
        base_port (int): Starting port number for container port allocation
        image_name (str): Default Docker image name for containers
        docker_available (bool): Flag indicating Docker client availability
        client (docker.DockerClient): Docker client instance for operations

    Example:
        >>> manager = CoreContainerManager()
        >>> if manager.is_docker_available():
        ...     spec = ContainerSpec(name="test-container")
        ...     container = manager.create_container(spec)
        ...     print(f"Container status: {container.status}")
    """

    def __init__(self) -> None:
        """Initialize the core container manager with Docker client setup.

        Initializes the Docker client with proper timeout handling and error
        recovery mechanisms. If Docker is not available, the manager will
        operate in a degraded mode with appropriate error reporting.

        The initialization process:
        1. Sets up connection timeouts for Docker operations
        2. Attempts to create and test Docker client connection
        3. Configures fallback behavior for Docker unavailability
        4. Logs initialization status for debugging

        Side Effects:
            - Sets socket timeout during Docker connection testing
            - Initializes Docker client if available
            - Configures docker_available flag based on connection success
            - Logs initialization status and any errors

        Note:
            Docker unavailability is handled gracefully - the manager will
            continue to operate but will return error responses for container
            operations when Docker is not accessible.
        """

        self.base_port = 8700  # Starting port for containers
        self.image_name = "codegraph-api:latest"  # Default image name
        self.docker_available = False
        self.client: docker.DockerClient | None = None
        self._shared_volume_name_cache: str | None = None
        self._shared_volume_candidates: list[str] = self._build_shared_volume_candidates()
        self._agent_host_url: str | None = AGENT_HOST_URL
        # Map container_name -> agent-host container_id for delegation
        self._agent_host_ids: dict[str, str] = {}

        if self._agent_host_url:
            logger.info(
                "CoreContainerManager: delegating to agent-host at %s",
                self._agent_host_url,
            )
            self.docker_available = True  # Mark as available since agent-host handles Docker
            return

        # Try to initialize Docker client with error handling
        try:
            # 5 second timeout for Docker operations
            socket.setdefaulttimeout(5.0)

            self.client = docker.from_env(timeout=5)
            # Test Docker connection with timeout
            self.client.ping()

            # Reset timeout after successful initialization
            socket.setdefaulttimeout(None)
            self.docker_available = True
            logger.info("ContainerManager: Docker client initialized successfully")
        except (docker.errors.DockerException, OSError, TimeoutError) as e:
            logger.warning(
                "ContainerManager: Failed to initialize Docker client: %s. "
                "Container provisioning will be disabled. "
                "Error details: %s: %s",
                e,
                type(e).__name__,
                str(e),
            )
            self.client = None
            self.docker_available = False

    def _build_shared_volume_candidates(self) -> list[str]:
        """Build ordered list of candidate shared volume names."""
        explicit = os.getenv("PROVISIONED_CONTAINERS_VOLUME_NAME")
        project_name = (
            os.getenv("COMPOSE_PROJECT_NAME")
            or os.getenv("CODE_GRAPH_PROJECT_NAME")
            or "code_graph_helper"
        )
        candidates: list[str] = []
        if explicit:
            candidates.append(explicit)
        candidates.append(f"{project_name}_provisioned_containers")
        candidates.append("code_graph_helper_provisioned_containers")
        candidates.append("provisioned_containers")

        # Deduplicate while preserving order
        seen: set[str] = set()
        ordered = []
        for candidate in candidates:
            if candidate and candidate not in seen:
                ordered.append(candidate)
                seen.add(candidate)
        return ordered

    def _volume_exists(self, volume_name: str) -> bool:
        """Check whether a docker volume exists."""
        if not self.is_docker_available() or self.client is None:
            return False
        try:
            self.client.volumes.get(volume_name)
            return True
        except docker.errors.NotFound:
            return False
        except docker.errors.APIError as exc:
            logger.warning("Volume existence check failed for %s: %s", volume_name, exc)
            return False

    def _resolve_shared_volume_name(self) -> str:
        """Resolve the shared provisioned container volume name."""
        if self._shared_volume_name_cache:
            return self._shared_volume_name_cache

        if self.is_docker_available() and self.client is not None:
            for candidate in self._shared_volume_candidates:
                if self._volume_exists(candidate):
                    logger.info("Using existing shared provisioned volume: %s", candidate)
                    self._shared_volume_name_cache = candidate
                    return candidate

            for candidate in self._shared_volume_candidates:
                try:
                    self.client.volumes.create(name=candidate)
                    logger.info("Created shared provisioned volume: %s", candidate)
                    self._shared_volume_name_cache = candidate
                    return candidate
                except docker.errors.APIError as exc:
                    logger.warning(
                        "Failed to create shared volume candidate %s: %s",
                        candidate,
                        exc,
                    )

        # Fallback to last candidate (legacy default)
        fallback = self._shared_volume_candidates[-1]
        logger.info("Falling back to shared volume name: %s", fallback)
        self._shared_volume_name_cache = fallback
        return fallback

    def is_docker_available(self) -> bool:
        """Check if Docker is available and accessible for container operations.

        Verifies that Docker is both available and the client connection is
        properly established. This method should be called before attempting
        any container operations to ensure they will succeed.

        Returns:
            bool: True if Docker is available and client is connected, False otherwise.

        Example:
            >>> manager = CoreContainerManager()
            >>> if manager.is_docker_available():
            ...     # Safe to perform container operations
            ...     container = manager.create_container(spec)
            ... else:
            ...     # Handle Docker unavailability gracefully
            ...     print("Docker not available")

        Note:
            This method performs a simple boolean check without attempting
            network operations. Use retry_docker_connection() if you need
            to attempt reconnection to Docker.
        """
        if self._agent_host_url:
            return True
        return self.docker_available and self.client is not None

    def retry_docker_connection(self) -> bool:
        """Attempt to reconnect to Docker if it wasn't available initially.

        Retries Docker connection establishment when it failed during initialization
        or became unavailable during operation. This method uses shorter timeouts
        than initial connection attempts to avoid blocking operations.

        Returns:
            bool: True if Docker connection was successfully established, False otherwise.

        Connection Process:
            1. Checks if Docker is already available (early return if true)
            2. Sets shorter timeout for retry operations
            3. Attempts to create new Docker client
            4. Tests connection with ping operation
            5. Updates availability flags based on success/failure

        Example:
            >>> manager = CoreContainerManager()
            >>> if not manager.is_docker_available():
            ...     if manager.retry_docker_connection():
            ...         print("Docker connection restored")
            ...     else:
            ...         print("Docker still unavailable")

        Note:
            This method is automatically called by container operations when
            Docker is not initially available, providing automatic recovery
            from temporary Docker connectivity issues.
        """
        if self.docker_available:
            return True

        try:
            socket.setdefaulttimeout(3.0)  # Shorter timeout for retries

            self.client = docker.from_env(timeout=3)
            self.client.ping()

            socket.setdefaulttimeout(None)
            self.docker_available = True
            logger.info("ContainerManager: Docker connection established on retry")
            return True
        except (docker.errors.DockerException, OSError, TimeoutError) as e:
            logger.warning("ContainerManager: Docker retry failed: %s", e)
            self.client = None
            self.docker_available = False
            return False

    def create_container(self, spec: ContainerSpec) -> ContainerInfo:
        """Create a new Docker container from the provided specification.

        Creates and configures a new Docker container based on the provided
        specification, including port allocation, environment setup, and
        network configuration. Handles Docker unavailability gracefully
        by attempting reconnection and providing appropriate error responses.

        Args:
            spec (ContainerSpec): Container specification containing:
                - name (str): Container name for identification
                - image (str): Docker image to use (optional, uses default if not specified)
                - ports (dict): Port mapping configuration
                - environment (dict): Environment variables
                - volumes (dict): Volume mount configuration

        Returns:
            ContainerInfo: Container information object containing:
                - code_graph_id (str): Unique identifier for the container
                - codebase_path (str): Path to the codebase within container
                - repo_url (str): Repository URL if applicable
                - branch (str): Git branch if applicable
                - container_name (str): Docker container name
                - api_endpoint (str): API endpoint URL for the container
                - status (ContainerStatus): Current container status
                - description (str): Container description or error message

        Container Creation Process:
            1. Verifies Docker availability (attempts reconnection if needed)
            2. Allocates available port for the container service
            3. Creates container with specified configuration
            4. Starts the container and monitors startup
            5. Returns container information with status

        Error Handling:
            - Docker unavailable: Returns error container info with ERROR status
            - Container creation failure: Returns error container info with details
            - Port allocation failure: Uses fallback port allocation strategy

        Example:
            >>> spec = ContainerSpec(
            ...     name="python-analyzer",
            ...     image="codegraph-api:latest"
            ... )
            >>> container = manager.create_container(spec)
            >>> if container.status == ContainerStatus.RUNNING:
            ...     print(f"Container started: {container.api_endpoint}")

        Note:
            This method performs blocking operations including Docker container
            creation and startup. Consider using async wrappers for non-blocking
            operation in async contexts.
        """
        # Delegate to agent-host if configured
        if self._agent_host_url:
            return self._create_container_via_agent_host(spec)

        if not self.is_docker_available():
            # Try to reconnect to Docker first
            if not self.retry_docker_connection():
                # Docker still not available - return failure
                logger.warning("ContainerManager: Docker not available, failing for %s", spec.name)
                return ContainerInfo(
                    code_graph_id="",
                    codebase_path="",
                    repo_url="",
                    branch="",
                    container_name=spec.name,
                    api_endpoint="",
                    internal_endpoint="",
                    public_endpoint="",
                    status=ContainerStatus.ERROR,
                    description=f"Docker not available for {spec.name}",
                )

        try:
            # Validate container name
            validated_name = self._validate_container_name(spec.name)

            # Ensure volume exists
            volume_name = f"{validated_name}-data"
            self._ensure_volume_exists(volume_name)

            # Ensure shared volume exists
            shared_volume_name = self._resolve_shared_volume_name()
            self._ensure_volume_exists(shared_volume_name)

            # Additional check to ensure client is available
            if self.client is None:
                raise RuntimeError("Docker client is not available")

            # Create container with shared volume subdirectory
            # Mount the container's individual volume to /app/data (internal to container)
            # And mount the shared volume to /app/provisioned_containers (so it can write to the shared location)
            mounts = [
                docker.types.Mount(
                    target="/app/data",
                    source=volume_name,
                    type="volume",
                    read_only=False,
                ),
                docker.types.Mount(
                    target="/app/provisioned_containers",
                    source=shared_volume_name,
                    type="volume",
                    read_only=False,
                ),
                # Mount Docker socket for Docker-in-Docker operations
                docker.types.Mount(
                    target="/var/run/docker.sock",
                    source="/var/run/docker.sock",
                    type="bind",
                    read_only=False,
                ),
            ]

            # Mount additional volumes from spec (e.g. CLI credentials)
            if spec.volumes:
                for host_path, mount_config in spec.volumes.items():
                    if os.path.exists(host_path):
                        mounts.append(
                            docker.types.Mount(
                                target=mount_config["bind"],
                                source=host_path,
                                type="bind",
                                read_only=(mount_config.get("mode", "rw") == "ro"),
                            )
                        )

            self.client.containers.run(
                image=spec.image,
                name=validated_name,
                ports={f"{spec.port}/tcp": spec.port},
                mounts=mounts,
                environment=spec.environment,
                command=spec.command,
                working_dir=spec.working_dir,
                detach=True,
                remove=False,
                network="workflow-system-network",
                # Run as root for Docker socket access (matches docker-compose.yml)
                user="0:0",
                # Enable privileged mode for Docker operations (matches docker-compose.yml)
                privileged=True,
                # Add labels to group with code_graph_helper containers
                labels={
                    "com.docker.compose.project": "code_graph_helper",
                    "com.docker.compose.service": "provisioned",
                    "code_graph_helper.type": "provisioned",
                    "code_graph_helper.managed": "true",
                },
                # Security: Enable read-only root filesystem to prevent malware persistence
                read_only=True,
                # Provide writable tmpfs mounts for temporary files
                tmpfs={
                    "/tmp": "rw,noexec,nosuid,size=100m",
                    "/var/tmp": "rw,noexec,nosuid,size=50m",
                },
            )

            # Create container info
            public_endpoint = _build_public_endpoint(validated_name)
            container_info = ContainerInfo(
                container_name=validated_name,
                api_endpoint=public_endpoint,
                internal_endpoint=f"http://localhost:{spec.port}",
                public_endpoint=public_endpoint,
                code_graph_id="",  # To be set by provisioner
                codebase_path="",  # To be set by provisioner
                repo_url="",  # To be set by provisioner
                branch="",  # To be set by provisioner
                status=ContainerStatus.RUNNING,
                description="",
            )

            logger.info("Container %s created successfully", validated_name)
            return container_info

        except (docker.errors.DockerException, OSError, RuntimeError) as e:
            logger.error("Failed to create container %s: %s", spec.name, e)
            raise

    def remove_container(self, name: str, force: bool = False) -> bool:
        """Remove a container and its associated resources

        Args:
            name: Container name to remove
            force: If True, force removal even if container is running

        Returns:
            bool: True if successful, False otherwise
        """
        # Delegate to agent-host if configured
        if self._agent_host_url:
            return self._remove_container_via_agent_host(name)

        container_existed_in_docker = False

        if not self.is_docker_available():
            # Docker not available - just remove from registry
            logger.info("Docker not available, removing %s from registry only", name)
            return True

        try:
            # Remove from Docker
            if self.client is not None:
                try:
                    container = self.client.containers.get(name)
                    container_existed_in_docker = True
                    if not force:
                        # Normal removal - stop gracefully first
                        container.stop(timeout=10)
                    container.remove(v=True, force=force)
                    logger.info("Container %s stopped and removed from Docker", name)
                except docker.errors.NotFound:
                    # Container not found in Docker - this is normal for orphaned registry entries
                    logger.debug("Container %s was not found in Docker (registry cleanup)", name)

            # Remove volume
            volume_name = f"{name}-data"
            if self.client is not None:
                try:
                    volume = self.client.volumes.get(volume_name)
                    volume.remove()
                    logger.info("Volume %s removed", volume_name)
                except docker.errors.NotFound:
                    logger.debug("Volume %s not found", volume_name)

                # Clean up shared volume subdirectory for this container
                # This is done by running a cleanup command in a temporary container
                # that mounts the shared volume
                try:
                    shared_volume_cleanup_cmd = f"rm -rf /provisioned_containers/{name}"
                    shared_volume_name = self._resolve_shared_volume_name()
                    self.client.containers.run(
                        image="alpine:latest",
                        command=["sh", "-c", shared_volume_cleanup_cmd],
                        volumes={
                            shared_volume_name: {
                                "bind": "/provisioned_containers",
                                "mode": "rw",
                            }
                        },
                        remove=True,
                        detach=False,
                    )
                    logger.info(
                        "Shared volume directory /provisioned_containers/%s cleaned up",
                        name,
                    )
                except (docker.errors.DockerException, OSError, RuntimeError) as e:
                    logger.warning("Failed to clean up shared volume directory for %s: %s", name, e)

            # Log appropriate message based on whether container existed
            if not container_existed_in_docker:
                logger.info(
                    "Cleaned up registry entry for container %s (container was not running)",
                    name,
                )

            return True

        except (docker.errors.DockerException, OSError) as e:
            logger.error("Failed to remove container %s: %s", name, e)
            return False

    def list_containers(self) -> list[ContainerInfo]:
        """List all managed containers"""
        return []  # This method is no longer used

    def get_container_status(self, name: str) -> ContainerStatus:
        """Get the current status of a container"""
        if not self.is_docker_available() or self.client is None:
            # Docker not available - check registry only
            return ContainerStatus.ERROR

        try:
            container = self.client.containers.get(name)
            if container.status == "running":
                return ContainerStatus.RUNNING
            if container.status in ["stopped", "exited"]:
                return ContainerStatus.STOPPED
            return ContainerStatus.ERROR
        except docker.errors.NotFound:
            return ContainerStatus.ERROR
        except (docker.errors.DockerException, OSError) as e:
            logger.error("Error checking container %s status: %s", name, e)
            return ContainerStatus.ERROR

    def get_container(self, name: str) -> ContainerInfo:
        """Get a container by name"""
        return ContainerInfo(
            code_graph_id="",
            codebase_path="",
            repo_url="",
            branch="",
            container_name=name,
            api_endpoint="",
            internal_endpoint="",
            public_endpoint="",
            status=ContainerStatus.ERROR,
            description="Container not found",
        )

    # -- agent-host delegation methods --

    def _create_container_via_agent_host(self, spec: ContainerSpec) -> ContainerInfo:
        """Create container via agent-host HTTP API."""
        url = self._agent_host_url
        assert url is not None
        validated_name = self._validate_container_name(spec.name)
        try:
            resp = httpx.post(
                f"{url}/containers",
                json={
                    "container_type": "repo",
                    "worker_name": validated_name,
                    "worker_id": validated_name,
                    "image": spec.image or self.image_name,
                    "timeout": 7200,
                    "environment": spec.environment or {},
                },
                timeout=30.0,
            )
            if resp.status_code != 201:
                raise RuntimeError(f"agent-host create failed: {resp.status_code} {resp.text}")
            data = resp.json()
            self._agent_host_ids[validated_name] = data["container_id"]
            logger.info(
                "Container %s created via agent-host (id=%s)",
                validated_name,
                data["container_id"],
            )
            return ContainerInfo(
                container_name=validated_name,
                api_endpoint=_build_public_endpoint(validated_name),
                internal_endpoint="",
                public_endpoint=_build_public_endpoint(validated_name),
                code_graph_id="",
                codebase_path="",
                repo_url="",
                branch="",
                status=ContainerStatus.RUNNING,
                description=f"Managed by agent-host: {data['container_name']}",
            )
        except (httpx.HTTPError, RuntimeError) as e:
            logger.error("agent-host create_container failed for %s: %s", spec.name, e)
            raise

    def _remove_container_via_agent_host(self, name: str) -> bool:
        """Remove container via agent-host HTTP API."""
        url = self._agent_host_url
        assert url is not None
        container_id = self._agent_host_ids.pop(name, None)
        if not container_id:
            # Try by name -- agent-host might track it
            logger.warning("No agent-host ID for %s, cannot remove from agent-host", name)
            return False
        try:
            resp = httpx.delete(f"{url}/containers/{container_id}", timeout=15.0)
            if resp.status_code == 200:
                logger.info("Container %s removed via agent-host", name)
                return True
            elif resp.status_code == 404:
                logger.debug("Container %s already gone from agent-host", name)
                return True
            else:
                logger.warning("agent-host remove failed for %s: %s", name, resp.text)
                return False
        except (httpx.HTTPError, RuntimeError) as e:
            logger.error("agent-host remove_container failed for %s: %s", name, e)
            return False

    def exec_in_container_via_agent_host(
        self,
        name: str,
        command: list[str],
        workdir: str = "/workspace",
        timeout: int = 300,
        user: str = "root",
    ) -> dict[str, Any]:
        """Execute command in container via agent-host HTTP API.

        Returns dict with keys: exit_code, stdout, stderr, duration.
        """
        url = self._agent_host_url
        assert url is not None
        container_id = self._agent_host_ids.get(name)
        if not container_id:
            raise RuntimeError(f"No agent-host container ID for {name}")
        try:
            resp = httpx.post(
                f"{url}/containers/{container_id}/exec",
                json={
                    "command": command,
                    "workdir": workdir,
                    "timeout": timeout,
                    "user": user,
                },
                timeout=float(timeout + 15),
            )
            if resp.status_code != 200:
                raise RuntimeError(f"agent-host exec failed: {resp.status_code} {resp.text}")
            data = resp.json()
            if "exit_code" not in data:
                raise RuntimeError(
                    f"agent-host exec response missing 'exit_code': {list(data.keys())}"
                )
            return {
                "exit_code": data["exit_code"],
                "stdout": data.get("stdout", ""),
                "stderr": data.get("stderr", ""),
                "duration": data.get("execution_time", 0),
            }
        except (httpx.HTTPError, RuntimeError) as e:
            logger.error("agent-host exec failed for %s: %s", name, e)
            raise

    @property
    def uses_agent_host(self) -> bool:
        """Whether this manager delegates to agent-host."""
        return self._agent_host_url is not None

    def has_agent_host_id(self, name: str) -> bool:
        """Check if a container has an agent-host container ID."""
        return name in self._agent_host_ids

    def _validate_container_name(self, container_name: str) -> str:
        """Basic Docker-compliant name validation."""
        if not container_name or len(container_name) < 3:
            container_name = f"codegraph-{uuid7().hex[:8]}"

        # Basic Docker name pattern
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$", container_name):
            # Basic sanitization
            container_name = re.sub(r"[^a-zA-Z0-9_.-]", "-", container_name)
            if not container_name[0].isalnum():
                container_name = f"codegraph-{container_name}"

        # No automatic counter suffixes - use the requested name
        # If name conflicts exist, the existing container will be replaced
        return container_name

    def _container_name_exists(self, container_name: str) -> bool:
        """Check if a container name already exists"""
        # Check Docker if available
        if self.is_docker_available() and self.client is not None:
            try:
                self.client.containers.get(container_name)
                return True
            except docker.errors.NotFound:
                return False
            except docker.errors.APIError:
                # If we can't check Docker, assume it might exist to be safe
                return True

        return False

    def _ensure_volume_exists(self, volume_name: str) -> docker.models.volumes.Volume | None:
        """Ensure Docker volume exists, create if necessary"""
        if not self.is_docker_available() or self.client is None:
            logger.info("Docker not available, skipping volume creation for: %s", volume_name)
            return None  # Skip in Docker unavailable mode

        try:
            volume = self.client.volumes.get(volume_name)
            logger.info("Using existing volume: %s", volume_name)
            return volume
        except docker.errors.NotFound:
            volume = self.client.volumes.create(name=volume_name)
            logger.info("Created new volume: %s", volume_name)
            return volume

    def _find_available_port(self) -> int:
        """Find an available host port starting from base_port"""
        used_host_ports = self._get_used_docker_ports()

        for port_candidate in range(self.base_port, self.base_port + 100):
            if port_candidate not in used_host_ports and self._is_port_available(port_candidate):
                logger.info("Found available port: %s", port_candidate)
                return port_candidate
            used_host_ports.add(port_candidate)

        raise RuntimeError(f"No available ports in range {self.base_port}-{self.base_port + 100}")

    def _get_used_docker_ports(self) -> set[int]:
        """Get set of ports currently used by Docker containers"""
        used_host_ports: set[int] = set()

        if not (self.is_docker_available() and self.client is not None):
            return used_host_ports

        try:
            for container in self.client.containers.list(all=True):
                self._extract_container_ports(container, used_host_ports)
        except docker.errors.DockerException as e:
            logger.warning("Could not list Docker containers to find available port: %s", e)

        return used_host_ports

    def _extract_container_ports(self, container: Any, used_host_ports: set[int]) -> None:
        """Extract port mappings from a container"""
        if not container.ports:
            return

        for _internal_port, host_configs in container.ports.items():
            if not host_configs:
                continue

            for host_config in host_configs:
                self._add_host_port_if_valid(host_config, container.name, used_host_ports)

    def _add_host_port_if_valid(
        self,
        host_config: dict[str, Any],
        container_name: str,
        used_host_ports: set[int],
    ) -> None:
        """Add host port to used set if valid"""
        if "HostPort" not in host_config or not host_config["HostPort"]:
            return

        try:
            used_host_ports.add(int(host_config["HostPort"]))
        except ValueError:
            logger.warning(
                "Could not parse host port: %s for container %s",
                host_config["HostPort"],
                container_name,
            )

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available using socket connection test"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                return s.connect_ex(("127.0.0.1", port)) != 0
        except OSError as e:
            logger.warning("Socket check failed for port %s: %s", port, e)
            # Assume port is available if socket check fails
            return True


# Global instance
_container_manager: CoreContainerManager | None = None


def get_container_manager() -> CoreContainerManager:
    """Get the global container manager instance"""
    global _container_manager
    if _container_manager is None:
        _container_manager = CoreContainerManager()
    return _container_manager

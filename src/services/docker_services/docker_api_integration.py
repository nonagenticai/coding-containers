"""
Docker API Integration Module

Handles Docker client initialization, connection management, and API interactions.
Extracted from docker_service.py to improve modularity and maintainability.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any

import docker
import docker.errors
import docker.types

from coding_client.utils.retry_helper import retry_docker

logger = logging.getLogger(__name__)


class DockerAPIIntegration:
    """Manages Docker API client and connection."""

    def __init__(self) -> None:
        """Initialize Docker API integration."""
        self.client: docker.DockerClient | None = None
        self.docker_available = False
        self._exec_run_supports_timeout: bool | None = None

        # Initialize Docker client
        self._initialize_docker_client()

    def _initialize_docker_client(self) -> None:
        """Initialize Docker client with fallback connection methods."""
        # Disable Docker entirely when running in-cluster Kubernetes mode
        if os.getenv("KUBERNETES_IN_CLUSTER", "false").lower() == "true":
            logger.info("Docker disabled because KUBERNETES_IN_CLUSTER=true")
            return

        try:
            # Check Docker socket permissions first
            if not self._check_docker_socket_permissions():
                logger.warning("DockerAPIIntegration: Docker socket permission issues detected")
                self._suggest_permission_fixes()

            # Try multiple connection methods
            self._try_docker_connections()

            if self.client:
                self.docker_available = True
                logger.info("DockerAPIIntegration: Docker client initialized successfully")

                # Verify functionality
                if not self._verify_docker_functionality():
                    logger.warning(
                        "DockerAPIIntegration: Docker client connected but functionality limited"
                    )
            else:
                raise docker.errors.DockerException("All Docker connection methods failed")

        except (docker.errors.DockerException, ConnectionError, OSError) as e:
            logger.warning(
                f"DockerAPIIntegration: Failed to initialize Docker client: {e}. "
                f"Error details: {type(e).__name__}: {str(e)}"
            )
            self.client = None
            self.docker_available = False
            self._suggest_permission_fixes()

    def _try_docker_connections(self) -> None:
        """Try multiple methods to connect to Docker daemon."""
        connection_methods = [
            ("from_env", self._connect_from_env),
            ("unix socket", self._connect_unix_socket),
            ("user socket", self._connect_user_socket),
        ]

        for method_name, connect_func in connection_methods:
            try:
                self.client = connect_func()
                if self.client:
                    self.client.ping()
                    logger.info(
                        f"DockerAPIIntegration: Docker client initialized via {method_name}"
                    )
                    return
            except (docker.errors.DockerException, ConnectionError, OSError) as e:
                logger.debug(f"{method_name} connection failed: {e}")

        # If we get here, all methods failed
        self.client = None

    @retry_docker(max_attempts=3, min_wait=1.0, max_wait=5.0)
    def _connect_from_env(self) -> docker.DockerClient:
        """Connect using docker.from_env()."""
        return docker.from_env()

    @retry_docker(max_attempts=3, min_wait=1.0, max_wait=5.0)
    def _connect_unix_socket(self) -> docker.DockerClient:
        """Connect using unix socket."""
        return docker.DockerClient(base_url="unix:///var/run/docker.sock")

    @retry_docker(max_attempts=3, min_wait=1.0, max_wait=5.0)
    def _connect_user_socket(self) -> docker.DockerClient:
        """Connect using user-specific socket."""
        user_socket = f"unix://{os.path.expanduser('~')}/.docker/run/docker.sock"
        return docker.DockerClient(base_url=user_socket)

    def _check_docker_socket_permissions(self) -> bool:
        """Check if we can access the Docker socket."""
        socket_paths = [
            "/var/run/docker.sock",
            f"{os.path.expanduser('~')}/.docker/run/docker.sock",
        ]

        for socket_path in socket_paths:
            if os.path.exists(socket_path):
                try:
                    os.stat(socket_path)
                    return os.access(socket_path, os.R_OK | os.W_OK)
                except (OSError, PermissionError):  # fmt: skip
                    continue

        return False

    def _verify_docker_functionality(self) -> bool:
        """Verify Docker client can perform basic operations."""
        try:
            if self.client is None:
                return False
            self.client.version()
            return True
        except (docker.errors.DockerException, ConnectionError, OSError) as e:
            logger.debug(f"Docker functionality verification failed: {e}")
            return False

    def _suggest_permission_fixes(self) -> None:
        """Log suggestions for fixing Docker permission issues."""
        logger.warning("=" * 60)
        logger.warning("DOCKER PERMISSION ISSUE DETECTED")
        logger.warning("=" * 60)
        logger.warning("To fix Docker permissions, run the comprehensive fix script:")
        logger.warning("  chmod +x fix_docker_permissions_comprehensive.sh")
        logger.warning("  ./fix_docker_permissions_comprehensive.sh")
        logger.warning("")
        logger.warning("Or manually fix with these commands:")
        logger.warning("  sudo usermod -aG docker $USER")
        logger.warning("  sudo systemctl restart docker")
        logger.warning("  # Then log out and back in, or run: newgrp docker")
        logger.warning("=" * 60)

    def is_docker_available(self) -> bool:
        """Check if Docker is available and accessible."""
        return self.docker_available and (self.client is not None)

    def retry_docker_connection(self) -> bool:
        """Attempt to reconnect to Docker if it wasn't available initially."""
        if self.docker_available:
            return True

        try:
            self.client = None
            self._try_docker_connections()

            if self.client:
                self.docker_available = True
                logger.info("DockerAPIIntegration: Docker connection established on retry")
                return True
            else:
                raise docker.errors.DockerException("All Docker connection methods failed on retry")

        except (docker.errors.DockerException, ConnectionError, OSError) as e:
            logger.warning(f"DockerAPIIntegration: Docker retry failed: {e}")
            self.client = None
            self.docker_available = False
            return False

    def get_docker_client(self) -> docker.DockerClient | None:
        """Get the Docker client instance."""
        return self.client

    def get_docker_info(self) -> dict:
        """Get Docker system information."""
        if not self.is_docker_available():
            return {"error": "Docker not available"}

        try:
            assert self.client is not None  # Type narrowing for ty
            info = self.client.info()
            version = self.client.version()

            return {
                "server_version": version.get("Version", "Unknown"),
                "api_version": version.get("ApiVersion", "Unknown"),
                "containers": info.get("Containers", 0),
                "containers_running": info.get("ContainersRunning", 0),
                "containers_paused": info.get("ContainersPaused", 0),
                "containers_stopped": info.get("ContainersStopped", 0),
                "images": info.get("Images", 0),
                "storage_driver": info.get("Driver", "Unknown"),
                "memory_total": info.get("MemTotal", 0),
                "ncpu": info.get("NCPU", 0),
                "kernel_version": info.get("KernelVersion", "Unknown"),
                "operating_system": info.get("OperatingSystem", "Unknown"),
                "architecture": info.get("Architecture", "Unknown"),
            }

        except docker.errors.DockerException as e:
            logger.error(f"Error getting Docker info: {e}")
            return {"error": str(e)}

    @retry_docker(max_attempts=2, min_wait=0.5, max_wait=2.0)
    def ping_docker(self) -> bool:
        """Ping Docker daemon to check connectivity."""
        if not self.client:
            return False

        try:
            self.client.ping()
            return True
        except (docker.errors.DockerException, ConnectionError, OSError) as e:
            logger.debug(f"Docker ping failed: {e}")
            return False

    def _check_exec_run_timeout_support(self) -> bool:
        """Check if the docker library version supports timeout parameter for exec_run."""
        if self._exec_run_supports_timeout is not None:
            return self._exec_run_supports_timeout

        try:
            from docker.models.containers import Container

            exec_run_signature = inspect.signature(Container.exec_run)
            self._exec_run_supports_timeout = "timeout" in exec_run_signature.parameters
            logger.debug(f"Docker exec_run timeout support: {self._exec_run_supports_timeout}")
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Could not check exec_run timeout support: {e}")
            self._exec_run_supports_timeout = False

        return self._exec_run_supports_timeout

    @retry_docker(max_attempts=3, min_wait=1.0, max_wait=5.0)
    def safe_exec_run(
        self,
        container: Any,
        command: list,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Safely execute command with timeout compatibility and retry logic.

        Args:
            container: Docker container object
            command: Command to execute
            timeout: Optional timeout in seconds
            **kwargs: Additional arguments for exec_run

        Returns:
            Execution result
        """
        try:
            if timeout is not None and self._check_exec_run_timeout_support():
                return container.exec_run(command, timeout=timeout, **kwargs)
            else:
                if timeout is not None:
                    logger.debug("Timeout parameter not supported, executing without timeout")
                return container.exec_run(command, **kwargs)
        except TypeError as e:
            if "timeout" in str(e):
                logger.warning(f"Timeout parameter not supported: {e}")
                return container.exec_run(command, **kwargs)
            else:
                raise

    @retry_docker(max_attempts=3, min_wait=2.0, max_wait=10.0)
    def create_docker_container(self, spec: dict) -> Any:
        """
        Create a Docker container using the provided specification with retry logic.

        Args:
            spec: Container specification dictionary

        Returns:
            Docker container object
        """
        if not self.is_docker_available():
            raise Exception("Docker not available for container creation")

        try:
            # Extract container creation parameters
            create_params = {
                "image": spec["image"],
                "name": spec["name"],
                "ports": spec.get("ports", {}),
                "environment": spec.get("environment", {}),
                "command": spec.get("command"),
                "working_dir": spec.get("working_dir"),
                "detach": spec.get("detach", True),
                "remove": spec.get("remove", False),
                "network": spec.get("network"),
                "labels": spec.get("labels", {}),
                "volumes": spec.get("volumes", {}),
                "mounts": spec.get("mounts", []),
                "cap_add": spec.get("cap_add", []),
                "privileged": spec.get("privileged", False),
                "user": spec.get("user"),
            }

            # Remove None values
            create_params = {k: v for k, v in create_params.items() if v is not None}

            # Create and start container
            assert self.client is not None  # Type narrowing for ty
            container = self.client.containers.run(**create_params)
            logger.info(f"Docker container {spec['name']} created successfully")

            return container

        except docker.errors.DockerException as e:
            logger.error(f"Error creating Docker container: {e}")
            raise

    def get_system_df(self) -> dict:
        """Get Docker system disk usage information."""
        if not self.is_docker_available():
            return {"error": "Docker not available"}

        try:
            assert self.client is not None  # Type narrowing for ty
            df_info = self.client.df()
            return {
                "layers_size": df_info.get("LayersSize", 0),
                "images": [
                    {
                        "repository": img.get("Repository", ""),
                        "tag": img.get("Tag", ""),
                        "size": img.get("Size", 0),
                        "shared_size": img.get("SharedSize", 0),
                    }
                    for img in df_info.get("Images", [])
                ],
                "containers": [
                    {
                        "id": cont.get("Id", ""),
                        "names": cont.get("Names", []),
                        "size_rw": cont.get("SizeRw", 0),
                        "size_root_fs": cont.get("SizeRootFs", 0),
                    }
                    for cont in df_info.get("Containers", [])
                ],
                "volumes": [
                    {
                        "name": vol.get("Name", ""),
                        "size": vol.get("UsageData", {}).get("Size", 0),
                    }
                    for vol in df_info.get("Volumes", [])
                ],
            }

        except docker.errors.DockerException as e:
            logger.error(f"Error getting Docker system df: {e}")
            return {"error": str(e)}

    @retry_docker(max_attempts=3, min_wait=1.0, max_wait=10.0)
    def cleanup_resources(self) -> dict:
        """Clean up unused Docker resources with retry logic."""
        if not self.is_docker_available():
            return {"error": "Docker not available"}

        try:
            assert self.client is not None  # Type narrowing for ty
            results = {}

            # Prune containers
            container_prune = self.client.containers.prune()
            results["containers"] = {
                "deleted": container_prune.get("ContainersDeleted") or [],
                "space_reclaimed": container_prune.get("SpaceReclaimed", 0),
            }

            # Prune images
            image_prune = self.client.images.prune()
            results["images"] = {
                "deleted": image_prune.get("ImagesDeleted") or [],
                "space_reclaimed": image_prune.get("SpaceReclaimed", 0),
            }

            # Prune volumes
            volume_prune = self.client.volumes.prune()
            results["volumes"] = {
                "deleted": volume_prune.get("VolumesDeleted") or [],
                "space_reclaimed": volume_prune.get("SpaceReclaimed", 0),
            }

            # Prune networks
            network_prune = self.client.networks.prune()
            results["networks"] = {"deleted": network_prune.get("NetworksDeleted") or []}

            total_space = (
                results["containers"]["space_reclaimed"]
                + results["images"]["space_reclaimed"]
                + results["volumes"]["space_reclaimed"]
            )

            results["total_space_reclaimed"] = total_space

            logger.info(f"Docker cleanup completed, reclaimed {total_space} bytes")
            return results

        except docker.errors.DockerException as e:
            logger.error(f"Error during Docker cleanup: {e}")
            return {"error": str(e)}

"""
Network and Volume Management Module

Handles Docker network and volume operations.
Extracted from docker_service.py to improve modularity and maintainability.
"""

from __future__ import annotations

import logging
from typing import Any

import docker
import docker.errors
import docker.types

logger = logging.getLogger(__name__)


class NetworkVolumeManager:
    """Manages Docker networks and volumes."""

    def __init__(self, docker_client: Any, project_name: str):
        """
        Initialize the network and volume manager.

        Args:
            docker_client: Docker client instance
            project_name: Project name for resource naming
        """
        self.client = docker_client
        self.project_name = project_name

    # Volume Management Methods

    def ensure_volume_exists(
        self,
        volume_name: str,
        driver: str = "local",
        labels: dict[str, str] | None = None,
    ) -> bool:
        """
        Ensure Docker volume exists, create if it doesn't.

        Args:
            volume_name: Name of the volume
            driver: Volume driver (default: local)
            labels: Optional labels for the volume

        Returns:
            bool: True if volume exists or was created successfully
        """
        if not self.client:
            logger.info(f"Docker not available, skipping volume creation for: {volume_name}")
            return False

        try:
            # Check if volume already exists
            self.client.volumes.get(volume_name)
            logger.info(f"Using existing volume: {volume_name}")
            return True

        except docker.errors.NotFound:
            # Volume doesn't exist, create it
            try:
                volume_labels = labels or {}
                volume_labels.update(
                    {
                        "com.docker.compose.project": self.project_name,
                        "created_by": "code_graph_helper",
                    }
                )

                self.client.volumes.create(name=volume_name, driver=driver, labels=volume_labels)
                logger.info(f"Created new volume: {volume_name}")
                return True

            except docker.errors.DockerException as e:
                logger.error(f"Failed to create volume {volume_name}: {e}")
                return False

        except docker.errors.DockerException as e:
            logger.error(f"Error checking volume {volume_name}: {e}")
            return False

    def remove_volume(self, volume_name: str, force: bool = False) -> bool:
        """
        Remove a Docker volume.

        Args:
            volume_name: Name of the volume to remove
            force: Force removal even if volume is in use

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.warning(f"Docker not available, cannot remove volume: {volume_name}")
            return False

        try:
            volume = self.client.volumes.get(volume_name)
            volume.remove(force=force)
            logger.info(f"Removed volume: {volume_name}")
            return True

        except docker.errors.NotFound:
            logger.info(f"Volume {volume_name} not found for removal - already removed")
            return True  # Consider this successful since volume is gone

        except docker.errors.APIError as e:
            if "volume is in use" in str(e).lower() and not force:
                logger.warning(f"Volume {volume_name} is in use and force=False")
                return False
            logger.error(f"Docker API error removing volume {volume_name}: {e}")
            return False

        except docker.errors.DockerException as e:
            logger.error(f"Error removing volume {volume_name}: {e}")
            return False

    def list_volumes(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        List Docker volumes.

        Args:
            filters: Optional filters for volume listing

        Returns:
            List of volume information dictionaries
        """
        if not self.client:
            logger.warning("Docker not available for volume listing")
            return []

        try:
            volumes = self.client.volumes.list(filters=filters)
            volume_list = []

            for volume in volumes:
                volume_info = {
                    "name": volume.name,
                    "driver": volume.attrs.get("Driver", ""),
                    "mountpoint": volume.attrs.get("Mountpoint", ""),
                    "created": volume.attrs.get("CreatedAt", ""),
                    "labels": volume.attrs.get("Labels") or {},
                    "scope": volume.attrs.get("Scope", ""),
                }
                volume_list.append(volume_info)

            logger.debug(f"Found {len(volume_list)} volumes")
            return volume_list

        except docker.errors.DockerException as e:
            logger.error(f"Error listing volumes: {e}")
            return []

    def get_volume_info(self, volume_name: str) -> dict[str, Any] | None:
        """
        Get detailed information about a Docker volume.

        Args:
            volume_name: Name of the volume

        Returns:
            Volume information dictionary or None if not found
        """
        if not self.client:
            logger.warning("Docker not available for volume info")
            return None

        try:
            volume = self.client.volumes.get(volume_name)
            return {
                "name": volume.name,
                "driver": volume.attrs.get("Driver", ""),
                "mountpoint": volume.attrs.get("Mountpoint", ""),
                "created": volume.attrs.get("CreatedAt", ""),
                "labels": volume.attrs.get("Labels") or {},
                "scope": volume.attrs.get("Scope", ""),
                "options": volume.attrs.get("Options") or {},
            }

        except docker.errors.NotFound:
            logger.debug(f"Volume {volume_name} not found")
            return None
        except docker.errors.DockerException as e:
            logger.error(f"Error getting volume info for {volume_name}: {e}")
            return None

    def prune_volumes(self, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Remove unused volumes to free up space.

        Args:
            filters: Optional filters for pruning

        Returns:
            Dictionary with pruning results
        """
        if not self.client:
            logger.warning("Docker not available for volume pruning")
            return {"volumes_deleted": [], "space_reclaimed": 0}

        try:
            logger.info("Pruning unused Docker volumes...")
            result = self.client.volumes.prune(filters=filters)

            volumes_deleted = result.get("VolumesDeleted") or []
            space_reclaimed = result.get("SpaceReclaimed", 0)

            logger.info(f"Pruned {len(volumes_deleted)} volumes, reclaimed {space_reclaimed} bytes")

            return {
                "volumes_deleted": volumes_deleted,
                "space_reclaimed": space_reclaimed,
            }

        except docker.errors.DockerException as e:
            logger.error(f"Error pruning volumes: {e}")
            return {"volumes_deleted": [], "space_reclaimed": 0}

    # Network Management Methods

    def ensure_network_exists(
        self,
        network_name: str,
        driver: str = "bridge",
        labels: dict[str, str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> bool:
        """
        Ensure Docker network exists, create if it doesn't.

        Args:
            network_name: Name of the network
            driver: Network driver (default: bridge)
            labels: Optional labels for the network
            options: Optional network configuration options

        Returns:
            bool: True if network exists or was created successfully
        """
        if not self.client:
            logger.info(f"Docker not available, skipping network creation for: {network_name}")
            return False

        try:
            # Check if network already exists
            self.client.networks.get(network_name)
            logger.info(f"Using existing network: {network_name}")
            return True

        except docker.errors.NotFound:
            # Network doesn't exist, create it
            try:
                network_labels = labels or {}
                network_labels.update(
                    {
                        "com.docker.compose.project": self.project_name,
                        "created_by": "code_graph_helper",
                    }
                )

                self.client.networks.create(
                    name=network_name,
                    driver=driver,
                    labels=network_labels,
                    options=options or {},
                )
                logger.info(f"Created new network: {network_name}")
                return True

            except docker.errors.DockerException as e:
                logger.error(f"Failed to create network {network_name}: {e}")
                return False

        except docker.errors.DockerException as e:
            logger.error(f"Error checking network {network_name}: {e}")
            return False

    def remove_network(self, network_name: str) -> bool:
        """
        Remove a Docker network.

        Args:
            network_name: Name of the network to remove

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.warning(f"Docker not available, cannot remove network: {network_name}")
            return False

        try:
            network = self.client.networks.get(network_name)
            network.remove()
            logger.info(f"Removed network: {network_name}")
            return True

        except docker.errors.NotFound:
            logger.info(f"Network {network_name} not found for removal - already removed")
            return True  # Consider this successful since network is gone

        except docker.errors.APIError as e:
            if "has active endpoints" in str(e).lower():
                logger.warning(f"Network {network_name} has active endpoints, cannot remove")
                return False
            logger.error(f"Docker API error removing network {network_name}: {e}")
            return False

        except docker.errors.DockerException as e:
            logger.error(f"Error removing network {network_name}: {e}")
            return False

    def list_networks(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        List Docker networks.

        Args:
            filters: Optional filters for network listing

        Returns:
            List of network information dictionaries
        """
        if not self.client:
            logger.warning("Docker not available for network listing")
            return []

        try:
            networks = self.client.networks.list(filters=filters)
            network_list = []

            for network in networks:
                network_info = {
                    "id": network.id,
                    "name": network.name,
                    "driver": network.attrs.get("Driver", ""),
                    "scope": network.attrs.get("Scope", ""),
                    "created": network.attrs.get("Created", ""),
                    "labels": network.attrs.get("Labels") or {},
                    "options": network.attrs.get("Options") or {},
                    "containers": list(network.attrs.get("Containers", {}).keys()),
                }
                network_list.append(network_info)

            logger.debug(f"Found {len(network_list)} networks")
            return network_list

        except docker.errors.DockerException as e:
            logger.error(f"Error listing networks: {e}")
            return []

    def get_network_info(self, network_name: str) -> dict[str, Any] | None:
        """
        Get detailed information about a Docker network.

        Args:
            network_name: Name of the network

        Returns:
            Network information dictionary or None if not found
        """
        if not self.client:
            logger.warning("Docker not available for network info")
            return None

        try:
            network = self.client.networks.get(network_name)
            return {
                "id": network.id,
                "name": network.name,
                "driver": network.attrs.get("Driver", ""),
                "scope": network.attrs.get("Scope", ""),
                "created": network.attrs.get("Created", ""),
                "labels": network.attrs.get("Labels") or {},
                "options": network.attrs.get("Options") or {},
                "ipam": network.attrs.get("IPAM", {}),
                "containers": network.attrs.get("Containers", {}),
                "internal": network.attrs.get("Internal", False),
                "attachable": network.attrs.get("Attachable", False),
            }

        except docker.errors.NotFound:
            logger.debug(f"Network {network_name} not found")
            return None
        except docker.errors.DockerException as e:
            logger.error(f"Error getting network info for {network_name}: {e}")
            return None

    def connect_container_to_network(
        self, container_id: str, network_name: str, aliases: list[str] | None = None
    ) -> bool:
        """
        Connect a container to a network.

        Args:
            container_id: Container ID or name
            network_name: Network name
            aliases: Optional network aliases for the container

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.error("Docker not available for network connection")
            return False

        try:
            network = self.client.networks.get(network_name)
            network.connect(container_id, aliases=aliases)
            logger.info(f"Connected container {container_id} to network {network_name}")
            return True

        except docker.errors.NotFound as e:
            logger.error(f"Network or container not found: {e}")
            return False
        except docker.errors.APIError as e:
            if "already exists in network" in str(e).lower():
                logger.info(f"Container {container_id} already connected to network {network_name}")
                return True
            logger.error(f"Docker API error connecting to network: {e}")
            return False
        except docker.errors.DockerException as e:
            logger.error(
                f"Error connecting container {container_id} to network {network_name}: {e}"
            )
            return False

    def disconnect_container_from_network(
        self, container_id: str, network_name: str, force: bool = False
    ) -> bool:
        """
        Disconnect a container from a network.

        Args:
            container_id: Container ID or name
            network_name: Network name
            force: Force disconnection

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.error("Docker not available for network disconnection")
            return False

        try:
            network = self.client.networks.get(network_name)
            network.disconnect(container_id, force=force)
            logger.info(f"Disconnected container {container_id} from network {network_name}")
            return True

        except docker.errors.NotFound as e:
            logger.warning(f"Network or container not found for disconnection: {e}")
            return True  # Consider successful since connection doesn't exist
        except docker.errors.APIError as e:
            logger.error(f"Docker API error disconnecting from network: {e}")
            return False
        except docker.errors.DockerException as e:
            logger.error(
                f"Error disconnecting container {container_id} from network {network_name}: {e}"
            )
            return False

    def prune_networks(self, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Remove unused networks.

        Args:
            filters: Optional filters for pruning

        Returns:
            Dictionary with pruning results
        """
        if not self.client:
            logger.warning("Docker not available for network pruning")
            return {"networks_deleted": []}

        try:
            logger.info("Pruning unused Docker networks...")
            result = self.client.networks.prune(filters=filters)

            networks_deleted = result.get("NetworksDeleted") or []

            logger.info(f"Pruned {len(networks_deleted)} networks")

            return {"networks_deleted": networks_deleted}

        except docker.errors.DockerException as e:
            logger.error(f"Error pruning networks: {e}")
            return {"networks_deleted": []}

    # Utility Methods

    def create_container_mounts(self, container_name: str) -> list[docker.types.Mount]:
        """
        Create standard mounts for a container.

        Args:
            container_name: Name of the container

        Returns:
            List of Docker mount objects
        """
        volume_name = f"{container_name}-data"
        shared_volume_name = f"{self.project_name}_shared_code_graphs"

        # Ensure volumes exist
        self.ensure_volume_exists(volume_name)
        self.ensure_volume_exists(shared_volume_name)

        return [
            docker.types.Mount(
                target="/app/data", source=volume_name, type="volume", read_only=False
            ),
            docker.types.Mount(
                target="/app/shared/code_graphs",
                source=shared_volume_name,
                type="volume",
                read_only=False,
            ),
        ]

    def setup_project_infrastructure(self) -> bool:
        """
        Set up basic network and volume infrastructure for the project.

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.warning("Docker not available for infrastructure setup")
            return False

        try:
            # Create project network
            network_name = "workflow-system-network"
            if not self.ensure_network_exists(network_name):
                logger.error(f"Failed to create project network: {network_name}")
                return False

            # Create shared volumes
            shared_volume = f"{self.project_name}_shared_code_graphs"
            if not self.ensure_volume_exists(shared_volume):
                logger.error(f"Failed to create shared volume: {shared_volume}")
                return False

            logger.info("Project infrastructure setup completed successfully")
            return True

        except docker.errors.DockerException as e:
            logger.error(f"Error setting up project infrastructure: {e}")
            return False

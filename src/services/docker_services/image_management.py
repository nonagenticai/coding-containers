"""
Image Management Module

Handles Docker image operations including pulling, building, and validation.
Extracted from docker_service.py to improve modularity and maintainability.
"""

from __future__ import annotations

import logging
from typing import Any

import docker
import docker.errors
import docker.types

logger = logging.getLogger(__name__)


class ImageManager:
    """Manages Docker images for container operations."""

    def __init__(self, docker_client: Any):
        """
        Initialize the image manager.

        Args:
            docker_client: Docker client instance
        """
        self.client = docker_client

    def image_exists(self, image_name: str) -> bool:
        """Check if a Docker image exists locally."""
        if not self.client:
            logger.warning("Docker client not available for image check")
            return False

        try:
            self.client.images.get(image_name)
            logger.debug(f"Image {image_name} exists locally")
            return True
        except docker.errors.ImageNotFound:
            logger.debug(f"Image {image_name} not found locally")
            return False
        except docker.errors.DockerException as e:
            logger.error(f"Error checking image {image_name}: {e}")
            return False

    async def pull_image(self, image_name: str, tag: str = "latest") -> bool:
        """
        Pull a Docker image from registry.

        Args:
            image_name: Name of the image to pull
            tag: Tag of the image (default: latest)

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.error("Docker client not available for image pull")
            return False

        try:
            full_image_name = f"{image_name}:{tag}"
            logger.info(f"Pulling image {full_image_name}...")

            # Pull the image
            self.client.images.pull(image_name, tag=tag)
            logger.info(f"Successfully pulled image {full_image_name}")
            return True

        except docker.errors.ImageNotFound:
            logger.error(f"Image {image_name}:{tag} not found in registry")
            return False
        except docker.errors.APIError as e:
            logger.error(f"Docker API error pulling image {image_name}:{tag}: {e}")
            return False
        except docker.errors.DockerException as e:
            logger.error(f"Unexpected error pulling image {image_name}:{tag}: {e}")
            return False

    async def build_image(
        self,
        path: str,
        tag: str,
        dockerfile: str = "Dockerfile",
        build_args: dict[str, str] | None = None,
    ) -> bool:
        """
        Build a Docker image from a Dockerfile.

        Args:
            path: Build context path
            tag: Tag for the built image
            dockerfile: Path to Dockerfile (relative to context)
            build_args: Build arguments

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.error("Docker client not available for image build")
            return False

        try:
            logger.info(f"Building image {tag} from {path}...")

            # Build the image
            image, build_logs = self.client.images.build(
                path=path,
                tag=tag,
                dockerfile=dockerfile,
                buildargs=build_args or {},
                rm=True,  # Remove intermediate containers
                forcerm=True,  # Always remove intermediate containers, even on build failure
            )

            # Log build output
            for log in build_logs:
                if "stream" in log:
                    logger.debug(f"Build: {log['stream'].strip()}")
                elif "error" in log:
                    logger.error(f"Build error: {log['error']}")

            logger.info(f"Successfully built image {tag}")
            return True

        except docker.errors.BuildError as e:
            logger.error(f"Build error for image {tag}: {e}")
            for log in e.build_log:
                if "stream" in log:
                    logger.error(f"Build log: {log['stream'].strip()}")
            return False
        except docker.errors.APIError as e:
            logger.error(f"Docker API error building image {tag}: {e}")
            return False
        except docker.errors.DockerException as e:
            logger.error(f"Unexpected error building image {tag}: {e}")
            return False

    def list_images(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        List Docker images.

        Args:
            filters: Optional filters for image listing

        Returns:
            List of image information dictionaries
        """
        if not self.client:
            logger.warning("Docker client not available for image listing")
            return []

        try:
            images = self.client.images.list(filters=filters)
            image_list = []

            for image in images:
                image_info = {
                    "id": image.id,
                    "tags": image.tags,
                    "created": image.attrs.get("Created", ""),
                    "size": image.attrs.get("Size", 0),
                    "labels": image.attrs.get("Config", {}).get("Labels") or {},
                }
                image_list.append(image_info)

            logger.debug(f"Found {len(image_list)} images")
            return image_list

        except docker.errors.DockerException as e:
            logger.error(f"Error listing images: {e}")
            return []

    async def remove_image(self, image_name: str, force: bool = False) -> bool:
        """
        Remove a Docker image.

        Args:
            image_name: Name or ID of the image to remove
            force: Force removal even if image is in use

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.error("Docker client not available for image removal")
            return False

        try:
            logger.info(f"Removing image {image_name}...")
            self.client.images.remove(image_name, force=force)
            logger.info(f"Successfully removed image {image_name}")
            return True

        except docker.errors.ImageNotFound:
            logger.warning(f"Image {image_name} not found for removal")
            return True  # Consider this successful since image is gone
        except docker.errors.APIError as e:
            logger.error(f"Docker API error removing image {image_name}: {e}")
            return False
        except docker.errors.DockerException as e:
            logger.error(f"Unexpected error removing image {image_name}: {e}")
            return False

    def get_image_info(self, image_name: str) -> dict[str, Any] | None:
        """
        Get detailed information about a Docker image.

        Args:
            image_name: Name or ID of the image

        Returns:
            Image information dictionary or None if not found
        """
        if not self.client:
            logger.warning("Docker client not available for image info")
            return None

        try:
            image = self.client.images.get(image_name)
            return {
                "id": image.id,
                "tags": image.tags,
                "created": image.attrs.get("Created", ""),
                "size": image.attrs.get("Size", 0),
                "architecture": image.attrs.get("Architecture", ""),
                "os": image.attrs.get("Os", ""),
                "config": image.attrs.get("Config", {}),
                "labels": image.attrs.get("Config", {}).get("Labels") or {},
            }

        except docker.errors.ImageNotFound:
            logger.debug(f"Image {image_name} not found")
            return None
        except docker.errors.DockerException as e:
            logger.error(f"Error getting image info for {image_name}: {e}")
            return None

    def get_language_container_image(self, language: str) -> str:
        """
        Get appropriate container image for programming language.

        Args:
            language: Programming language name

        Returns:
            Docker image name for the language
        """
        image_map = {
            "python": "code_graph_helper-api:latest",
            "javascript": "node:18-slim",
            "typescript": "node:18-slim",
            "java": "openjdk:17-slim",
            "go": "golang:1.20-alpine",
            "rust": "rust:1.70-slim",
            "cpp": "gcc:latest",
            "c": "gcc:latest",
            "ruby": "ruby:3.2-slim",
            "php": "php:8.2-cli",
            "csharp": "mcr.microsoft.com/dotnet/sdk:7.0",
            "scala": "hseeberger/scala-sbt:17.0.2_1.6.2_2.13.8",
        }

        return image_map.get(language.lower(), "code_graph_helper-api:latest")

    async def ensure_image_available(self, image_name: str, auto_pull: bool = True) -> bool:
        """
        Ensure a Docker image is available locally.

        Args:
            image_name: Name of the image
            auto_pull: Whether to automatically pull if not found locally

        Returns:
            bool: True if image is available, False otherwise
        """
        # Check if image exists locally
        if self.image_exists(image_name):
            return True

        if not auto_pull:
            logger.warning(f"Image {image_name} not available and auto_pull disabled")
            return False

        # Try to pull the image
        if ":" in image_name:
            name, tag = image_name.rsplit(":", 1)
        else:
            name, tag = image_name, "latest"

        return await self.pull_image(name, tag)

    def prune_images(self, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Remove unused images to free up space.

        Args:
            filters: Optional filters for pruning

        Returns:
            Dictionary with pruning results
        """
        if not self.client:
            logger.warning("Docker client not available for image pruning")
            return {"images_deleted": [], "space_reclaimed": 0}

        try:
            logger.info("Pruning unused Docker images...")
            result = self.client.images.prune(filters=filters)

            images_deleted = result.get("ImagesDeleted") or []
            space_reclaimed = result.get("SpaceReclaimed", 0)

            logger.info(f"Pruned {len(images_deleted)} images, reclaimed {space_reclaimed} bytes")

            return {
                "images_deleted": images_deleted,
                "space_reclaimed": space_reclaimed,
            }

        except docker.errors.DockerException as e:
            logger.error(f"Error pruning images: {e}")
            return {"images_deleted": [], "space_reclaimed": 0}

    def tag_image(self, image_name: str, repository: str, tag: str = "latest") -> bool:
        """
        Tag an existing image with a new name/tag.

        Args:
            image_name: Source image name or ID
            repository: Target repository name
            tag: Target tag

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.error("Docker client not available for image tagging")
            return False

        try:
            image = self.client.images.get(image_name)
            success = image.tag(repository, tag)

            if success:
                logger.info(f"Successfully tagged {image_name} as {repository}:{tag}")
            else:
                logger.error(f"Failed to tag {image_name} as {repository}:{tag}")

            return success

        except docker.errors.ImageNotFound:
            logger.error(f"Source image {image_name} not found for tagging")
            return False
        except docker.errors.DockerException as e:
            logger.error(f"Error tagging image {image_name}: {e}")
            return False

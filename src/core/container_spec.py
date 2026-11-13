"""Backward-compatibility re-exports for container spec models."""

from src.models.container_models import ContainerSpec, ContainerStatus, GitHubRepoRequest

__all__ = ["ContainerStatus", "GitHubRepoRequest", "ContainerSpec"]

"""Container Health History Service

This service provides health check history tracking for containers using Redis.
It maintains a rolling history of the last N health checks per container, allowing
for trend analysis and historical health monitoring.

Storage mechanism: Redis Lists (LPUSH/LTRIM pattern)
- Each container has a dedicated list key: container:{name}:health:history
- Health checks are pushed to the left (most recent first)
- List is automatically trimmed to maintain only the last N entries
- Each entry contains timestamp, status, healthy flag, and issues

Features:
    - Store last 100 health checks per container (configurable)
    - Efficient Redis list-based storage with automatic trimming
    - Timestamp-based history tracking
    - Issue tracking and status monitoring
    - Bulk history retrieval with pagination support

Example:
    >>> from container_health_history_service import ContainerHealthHistoryService
    >>> service = ContainerHealthHistoryService()
    >>> await service.record_health_check("my-container", "running", True, [])
    >>> history = await service.get_health_history("my-container", limit=10)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from redis.exceptions import RedisError

from src.core.redis_service import get_redis_service_sync

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_HISTORY_LIMIT = 100  # Maximum number of health checks to store per container
DEFAULT_QUERY_LIMIT = 50  # Default number of entries to return when querying


class HealthCheckRecord:
    """Represents a single health check record.

    Attributes:
        timestamp: ISO formatted UTC timestamp
        status: Container status (e.g., "running", "stopped", "error")
        healthy: Boolean flag indicating overall health
        issues: List of detected issues or warnings
        metadata: Optional additional metadata (e.g., response times, resource usage)
    """

    def __init__(
        self,
        timestamp: str,
        status: str,
        healthy: bool,
        issues: list[str],
        metadata: dict[str, Any] | None = None,
    ):
        self.timestamp = timestamp
        self.status = status
        self.healthy = healthy
        self.issues = issues
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "status": self.status,
            "healthy": self.healthy,
            "issues": self.issues,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HealthCheckRecord:
        """Create record from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            status=data["status"],
            healthy=data["healthy"],
            issues=data.get("issues", []),
            metadata=data.get("metadata", {}),
        )


class ContainerHealthHistoryService:
    """Service for tracking and retrieving container health check history.

    Uses Redis lists for efficient storage and retrieval of health check records.
    Each container has a dedicated list that maintains the last N health checks.

    Storage Strategy:
        - Redis List per container: container:{name}:health:history
        - LPUSH for adding new records (most recent first)
        - LTRIM to maintain size limit
        - LRANGE for retrieval
        - TTL on the list for automatic cleanup of inactive containers
    """

    def __init__(self, history_limit: int = DEFAULT_HISTORY_LIMIT):
        """Initialize the health history service.

        Args:
            history_limit: Maximum number of health checks to store per container
        """
        self.history_limit = history_limit
        self._redis = None

    @property
    def redis(self):
        """Lazy-loaded Redis service instance."""
        if self._redis is None:
            self._redis = get_redis_service_sync()
        return self._redis

    def _get_history_key(self, container_name: str) -> str:
        """Get Redis key for container health history.

        Args:
            container_name: Name of the container

        Returns:
            Redis key string
        """
        return f"container:{container_name}:health:history"

    async def record_health_check(
        self,
        container_name: str,
        status: str,
        healthy: bool,
        issues: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Record a health check result for a container.

        This method stores the health check in Redis and automatically maintains
        the history limit by trimming older entries.

        Args:
            container_name: Name of the container
            status: Current status (e.g., "running", "stopped", "error")
            healthy: Overall health flag
            issues: List of detected issues
            metadata: Optional additional metadata

        Returns:
            True if recorded successfully, False otherwise

        Example:
            >>> await service.record_health_check(
            ...     "my-container",
            ...     "running",
            ...     True,
            ...     [],
            ...     {"response_time_ms": 42}
            ... )
        """
        if not container_name:
            logger.error("Container name cannot be empty")
            return False

        try:
            # Create health check record
            record = HealthCheckRecord(
                timestamp=datetime.now(UTC).isoformat(),
                status=status,
                healthy=healthy,
                issues=issues or [],
                metadata=metadata,
            )

            # Serialize to JSON
            record_json = json.dumps(record.to_dict())

            # Get Redis key
            history_key = self._get_history_key(container_name)

            # Push to list (left push for most recent first)
            result = self.redis.lpush(history_key, record_json)

            if result is None:
                logger.error(f"Failed to push health check for {container_name}")
                return False

            # Trim list to maintain history limit
            self.redis.client.ltrim(history_key, 0, self.history_limit - 1)

            # Set expiration to 30 days for automatic cleanup
            self.redis.expire(history_key, 30 * 24 * 60 * 60)

            logger.debug(
                f"Recorded health check for {container_name}: "
                f"status={status}, healthy={healthy}, issues_count={len(issues)}"
            )

            return True

        except RedisError as e:
            logger.error(f"Error recording health check for {container_name}: {e}")
            return False

    async def get_health_history(
        self,
        container_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get health check history for a container.

        Retrieves the most recent health checks for the specified container,
        with support for pagination.

        Args:
            container_name: Name of the container
            limit: Maximum number of records to return (default: DEFAULT_QUERY_LIMIT)
            offset: Number of records to skip (for pagination)

        Returns:
            List of health check records (most recent first)

        Example:
            >>> # Get last 10 health checks
            >>> history = await service.get_health_history("my-container", limit=10)
            >>> for check in history:
            ...     print(f"{check['timestamp']}: {check['status']}")
        """
        if not container_name:
            logger.error("Container name cannot be empty")
            return []

        if limit is None:
            limit = DEFAULT_QUERY_LIMIT

        try:
            history_key = self._get_history_key(container_name)

            # Check if history exists
            if not self.redis.exists(history_key):
                logger.debug(f"No health history found for container: {container_name}")
                return []

            # Calculate range
            start = offset
            stop = offset + limit - 1

            # Get records from Redis list
            raw_records = self.redis.lrange(history_key, start, stop)

            if not raw_records:
                return []

            # Parse JSON records
            history = []
            for raw_record in raw_records:
                try:
                    if isinstance(raw_record, str):
                        record_data = json.loads(raw_record)
                    elif isinstance(raw_record, dict):
                        record_data = raw_record
                    else:
                        continue

                    history.append(record_data)

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse health record: {e}")
                    continue

            logger.debug(f"Retrieved {len(history)} health check records for {container_name}")

            return history

        except RedisError as e:
            logger.error(f"Error retrieving health history for {container_name}: {e}")
            return []

    async def get_history_count(self, container_name: str) -> int:
        """Get the total number of health checks stored for a container.

        Args:
            container_name: Name of the container

        Returns:
            Number of health checks in history
        """
        if not container_name:
            return 0

        try:
            history_key = self._get_history_key(container_name)
            count = self.redis.client.llen(history_key)
            return count if count is not None else 0

        except RedisError as e:
            logger.error(f"Error getting history count for {container_name}: {e}")
            return 0

    async def get_latest_health_check(self, container_name: str) -> dict[str, Any] | None:
        """Get the most recent health check for a container.

        Args:
            container_name: Name of the container

        Returns:
            Most recent health check record or None if not found
        """
        history = await self.get_health_history(container_name, limit=1)
        return history[0] if history else None

    async def clear_history(self, container_name: str) -> bool:
        """Clear all health check history for a container.

        Args:
            container_name: Name of the container

        Returns:
            True if cleared successfully, False otherwise
        """
        if not container_name:
            return False

        try:
            history_key = self._get_history_key(container_name)
            deleted = self.redis.delete(history_key)

            if deleted > 0:
                logger.info(f"Cleared health history for container: {container_name}")
                return True
            else:
                logger.debug(f"No history to clear for container: {container_name}")
                return True

        except RedisError as e:
            logger.error(f"Error clearing history for {container_name}: {e}")
            return False

    async def get_health_statistics(
        self, container_name: str, limit: int | None = None
    ) -> dict[str, Any]:
        """Get aggregated statistics from health check history.

        Analyzes the health check history to provide useful statistics such as:
        - Total checks
        - Healthy/unhealthy ratio
        - Most common issues
        - Uptime percentage

        Args:
            container_name: Name of the container
            limit: Number of recent checks to analyze (None for all)

        Returns:
            Dictionary with health statistics
        """
        history = await self.get_health_history(container_name, limit=limit or 1000)

        if not history:
            return {
                "total_checks": 0,
                "healthy_count": 0,
                "unhealthy_count": 0,
                "health_percentage": 0.0,
                "common_issues": [],
            }

        total_checks = len(history)
        healthy_count = sum(1 for check in history if check.get("healthy", False))
        unhealthy_count = total_checks - healthy_count

        # Collect all issues
        all_issues = []
        for check in history:
            all_issues.extend(check.get("issues", []))

        # Count issue frequency
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        # Sort issues by frequency
        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "container_name": container_name,
            "total_checks": total_checks,
            "healthy_count": healthy_count,
            "unhealthy_count": unhealthy_count,
            "health_percentage": (
                (healthy_count / total_checks * 100) if total_checks > 0 else 0.0
            ),
            "common_issues": [{"issue": issue, "count": count} for issue, count in common_issues],
            "first_check": history[-1].get("timestamp") if history else None,
            "last_check": history[0].get("timestamp") if history else None,
        }


# Global service instance
_health_history_service: ContainerHealthHistoryService | None = None


def get_health_history_service() -> ContainerHealthHistoryService:
    """Get the global health history service instance.

    Returns:
        ContainerHealthHistoryService instance
    """
    global _health_history_service
    if _health_history_service is None:
        _health_history_service = ContainerHealthHistoryService()
    return _health_history_service

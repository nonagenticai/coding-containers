"""Container Health Metrics Service

This module provides comprehensive health metrics for containerized code graphs,
including graph statistics, cache performance, test results, and error logs.

The service retrieves and aggregates metrics from multiple sources:
- Graph database (node/edge counts)
- Redis cache (hit/miss ratios)
- Test execution results
- Error logs from the last 24 hours
- Analysis timestamps

Key Features:
    - Graph node and edge counts
    - Last analysis timestamp tracking
    - Cache performance metrics per container
    - Test result statistics
    - Error log aggregation from last 24 hours
    - Comprehensive health status reporting
    - Health check result caching (30 seconds TTL) for improved performance

Caching:
    Health check results are cached for 30 seconds using the container cache utilities.
    This significantly reduces database and Redis queries for repeated health checks.
    Cache keys are generated using: generate_container_cache_key(container_name, "health_check")

Functions:
    get_container_health_metrics: Main function to retrieve all health metrics for a container

Example:
    >>> from container_health_service import get_container_health_metrics
    >>> metrics = await get_container_health_metrics("my-container")
    >>> print(f"Nodes: {metrics['graph_node_count']}")
    >>> print(f"Cache hit ratio: {metrics['cache_hit_ratio']}")
"""

from __future__ import annotations

import inspect
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import aiosqlite
import redis

from src.config.settings import get_settings
from src.core.redis_service import get_redis_service_sync
from src.utils.cache_manager import get_cache_manager
from src.utils.cache_utils import generate_container_cache_key

logger = logging.getLogger(__name__)


async def get_container_health_metrics(container_name: str, tenant_id: str) -> dict[str, Any]:
    """Get detailed health metrics for a specific container.

    Retrieves comprehensive metrics including:
    1. Graph node count - Total number of nodes in the code graph
    2. Graph edge count - Total number of edges (relationships) in the graph
    3. Last analysis timestamp - When the graph was last analyzed/updated
    4. Cache hit/miss ratio - Performance metrics for this container's cache
    5. Number of test results stored - Count of test execution results
    6. Error logs from last 24 hours - Recent errors and warnings

    Results are cached for 30 seconds to improve performance for repeated health checks.

    Args:
        container_name: Name of the container to get metrics for
        tenant_id: Tenant ID for multi-tenant isolation (required)

    Returns:
        Dictionary containing the following metrics:
        {
            "container_name": str,
            "timestamp": str (ISO format),
            "graph_node_count": int,
            "graph_edge_count": int,
            "last_analysis_timestamp": str (ISO format) or None,
            "cache_hit_ratio": float (0.0 to 1.0),
            "cache_hits": int,
            "cache_misses": int,
            "cache_total_requests": int,
            "test_results_count": int,
            "error_logs_24h": List[Dict[str, Any]],
            "error_count_24h": int,
            "status": str ("healthy", "degraded", "unhealthy"),
            "warnings": List[str]
        }

    Example:
        >>> metrics = await get_container_health_metrics("my-python-project", "tenant-123")
        >>> print(f"Graph has {metrics['graph_node_count']} nodes")
        >>> print(f"Cache hit ratio: {metrics['cache_hit_ratio']:.2%}")
        >>> print(f"Errors in last 24h: {metrics['error_count_24h']}")
    """
    if not container_name:
        raise ValueError("container_name cannot be empty")
    if not tenant_id:
        raise ValueError("tenant_id cannot be empty - required for tenant isolation")

    # Generate cache key using the cache utilities
    cache_key = generate_container_cache_key(container_name, "health_check", tenant_id=tenant_id)

    # Try to get cached result
    cache_manager = get_cache_manager()
    cached_result = await cache_manager.get(cache_key)

    if cached_result is not None:
        logger.info(f"Cache hit for container health check: {container_name}")
        return cached_result

    logger.info(f"Cache miss for container health check: {container_name}")
    logger.info(f"Fetching health metrics for container: {container_name}")

    metrics: dict[str, Any] = {
        "container_name": container_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "graph_node_count": 0,
        "graph_edge_count": 0,
        "last_analysis_timestamp": None,
        "cache_hit_ratio": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_total_requests": 0,
        "test_results_count": 0,
        "error_logs_24h": [],
        "error_count_24h": 0,
        "status": "healthy",
        "warnings": [],
    }

    # 1. Get graph node and edge counts
    graph_metrics = await _get_graph_metrics(container_name)
    metrics.update(graph_metrics)

    # 2. Get last analysis timestamp
    last_analysis = await _get_last_analysis_timestamp(container_name)
    metrics["last_analysis_timestamp"] = last_analysis

    # 3. Get cache hit/miss ratio
    cache_metrics = await _get_cache_metrics(container_name)
    metrics.update(cache_metrics)

    # 4. Get test results count
    test_count = await _get_test_results_count(container_name)
    metrics["test_results_count"] = test_count

    # 5. Get error logs from last 24 hours
    error_logs = await _get_error_logs_24h(container_name)
    metrics["error_logs_24h"] = error_logs
    metrics["error_count_24h"] = len(error_logs)

    # 6. Determine overall health status
    status, warnings = _determine_health_status(metrics)
    metrics["status"] = status
    metrics["warnings"] = warnings

    logger.info(
        f"Health metrics retrieved for {container_name}: "
        f"status={status}, nodes={metrics['graph_node_count']}, "
        f"edges={metrics['graph_edge_count']}, errors={metrics['error_count_24h']}"
    )

    # Cache the result for 30 seconds
    await cache_manager.set(cache_key, metrics, ttl=30)
    logger.debug(f"Cached health check results for {container_name} with TTL of 30 seconds")

    return metrics


async def _get_graph_metrics(container_name: str) -> dict[str, Any]:
    """Get graph node and edge counts from the database.

    Args:
        container_name: Name of the container

    Returns:
        Dict with graph_node_count and graph_edge_count
    """
    try:
        settings = get_settings()
        try:
            db_manager = settings.get_database_manager()
            db_path = db_manager.find_existing_database(container_name)
        except NotImplementedError:
            logger.debug("DatabaseManager not available, returning empty graph metrics")
            return {"graph_node_count": 0, "graph_edge_count": 0}

        if not db_path:
            logger.warning(f"No database found for container: {container_name}")
            return {"graph_node_count": 0, "graph_edge_count": 0}

        async with aiosqlite.connect(db_path, timeout=10.0) as conn:
            # Get node count
            cursor = await conn.execute("SELECT COUNT(*) FROM nodes")
            node_count_result = await cursor.fetchone()
            node_count = node_count_result[0] if node_count_result else 0
            await cursor.close()

            # Get edge count
            cursor = await conn.execute("SELECT COUNT(*) FROM edges")
            edge_count_result = await cursor.fetchone()
            edge_count = edge_count_result[0] if edge_count_result else 0
            await cursor.close()

            logger.debug(
                f"Graph metrics for {container_name}: nodes={node_count}, edges={edge_count}"
            )

            return {"graph_node_count": node_count, "graph_edge_count": edge_count}

    except (aiosqlite.Error, OSError, ValueError) as e:
        logger.error(f"Error getting graph metrics for {container_name}: {e}")
        return {"graph_node_count": 0, "graph_edge_count": 0}


async def _get_last_analysis_timestamp(container_name: str) -> str | None:
    """Get the timestamp of the last analysis for this container.

    Args:
        container_name: Name of the container

    Returns:
        ISO formatted timestamp string or None if not found
    """
    try:
        settings = get_settings()
        try:
            db_manager = settings.get_database_manager()
            db_path = db_manager.find_existing_database(container_name)
        except NotImplementedError:
            logger.debug("DatabaseManager not available, returning no analysis timestamp")
            return None

        if not db_path:
            return None

        async with aiosqlite.connect(db_path, timeout=10.0) as conn:
            # Get the most recent updated_at timestamp from nodes
            cursor = await conn.execute(
                "SELECT MAX(updated_at) FROM nodes WHERE updated_at IS NOT NULL"
            )
            result = await cursor.fetchone()
            await cursor.close()

            if result and result[0]:
                timestamp = result[0]
                logger.debug(f"Last analysis for {container_name}: {timestamp}")
                return timestamp

            return None

    except (aiosqlite.Error, OSError, ValueError) as e:
        logger.error(f"Error getting last analysis timestamp for {container_name}: {e}")
        return None


async def _get_cache_metrics(container_name: str) -> dict[str, Any]:
    """Get cache hit/miss metrics for this container from Redis.

    Args:
        container_name: Name of the container

    Returns:
        Dict with cache metrics including hit_ratio, hits, misses, total_requests
    """
    try:
        redis_service = get_redis_service_sync()

        # Redis keys for tracking cache metrics
        hits_key = f"container:{container_name}:cache:hits"
        misses_key = f"container:{container_name}:cache:misses"

        # Get hit and miss counts
        hits = redis_service.get(hits_key, default=0)
        misses = redis_service.get(misses_key, default=0)

        # Convert to int if they're strings
        if isinstance(hits, str):
            hits = int(hits) if hits.isdigit() else 0
        if isinstance(misses, str):
            misses = int(misses) if misses.isdigit() else 0

        total_requests = hits + misses
        hit_ratio = (hits / total_requests) if total_requests > 0 else 0.0

        logger.debug(
            f"Cache metrics for {container_name}: hits={hits}, misses={misses}, ratio={hit_ratio:.2%}"
        )

        return {
            "cache_hit_ratio": round(hit_ratio, 4),
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_total_requests": total_requests,
        }

    except (ConnectionError, ValueError, TypeError) as e:
        logger.error(f"Error getting cache metrics for {container_name}: {e}")
        return {
            "cache_hit_ratio": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_total_requests": 0,
        }


async def _get_test_results_count(container_name: str) -> int:
    """Get the count of test results stored for this container.

    Args:
        container_name: Name of the container

    Returns:
        Number of test results stored
    """
    try:
        redis_service = get_redis_service_sync()

        # Redis key for test results
        test_results_key = f"container:{container_name}:test_results"

        # Get the length of the list (test results are typically stored in a Redis list)
        # Try to get the list length
        try:
            count_result = redis_service.client.llen(test_results_key)
            if inspect.isawaitable(count_result):
                count = await count_result
            else:
                count = count_result
            logger.debug(f"Test results count for {container_name}: {count}")
            return int(count) if isinstance(count, int) else 0
        except (redis.RedisError, TypeError, ValueError) as exc:
            # Redis WRONGTYPE (ResponseError) or type coercion failures
            logger.debug(f"llen failed for {container_name}, checking key existence: {exc}")
            if redis_service.exists(test_results_key):
                return 1
            return 0

    except (ConnectionError, ValueError) as e:
        logger.error(f"Error getting test results count for {container_name}: {e}")
        return 0


async def _get_error_logs_24h(container_name: str) -> list[dict[str, Any]]:
    """Get error logs from the last 24 hours for this container.

    Args:
        container_name: Name of the container

    Returns:
        List of error log entries from the last 24 hours
    """
    try:
        redis_service = get_redis_service_sync()

        # Redis key for error logs
        error_logs_key = f"container:{container_name}:errors"

        # Get error logs (stored as a list)
        error_logs_raw = redis_service.lrange(error_logs_key, 0, -1)

        if not error_logs_raw:
            return []

        # Filter logs from the last 24 hours
        cutoff_time = datetime.now(UTC) - timedelta(hours=24)
        recent_errors = []

        for log_entry in error_logs_raw:
            if isinstance(log_entry, dict):
                log_dict = log_entry
            else:
                # If stored as JSON string, it should be already parsed by Redis service
                continue

            # Check timestamp
            if "timestamp" in log_dict:
                try:
                    log_time = datetime.fromisoformat(log_dict["timestamp"].replace("Z", "+00:00"))
                    if log_time >= cutoff_time:
                        recent_errors.append(log_dict)
                except (ValueError, TypeError):  # fmt: skip
                    # If timestamp parsing fails, include it anyway
                    recent_errors.append(log_dict)
            else:
                # If no timestamp, include it
                recent_errors.append(log_dict)

        logger.debug(f"Found {len(recent_errors)} errors in last 24h for {container_name}")
        return recent_errors

    except (ConnectionError, ValueError) as e:
        logger.error(f"Error getting error logs for {container_name}: {e}")
        return []


def _determine_health_status(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    """Determine overall health status based on metrics.

    Args:
        metrics: Dictionary of health metrics

    Returns:
        Tuple of (status, warnings) where status is "healthy", "degraded", or "unhealthy"
    """
    warnings = []
    status = "healthy"

    # Check for errors (check higher threshold first)
    if metrics["error_count_24h"] > 50:
        status = "unhealthy"
        warnings.append(f"Critical error count: {metrics['error_count_24h']} errors in last 24h")
    elif metrics["error_count_24h"] > 10:
        status = "degraded"
        warnings.append(f"High error count: {metrics['error_count_24h']} errors in last 24h")

    # Check cache performance
    if metrics["cache_total_requests"] > 100 and metrics["cache_hit_ratio"] < 0.5:
        if status == "healthy":
            status = "degraded"
        warnings.append(
            f"Low cache hit ratio: {metrics['cache_hit_ratio']:.2%} "
            f"({metrics['cache_hits']}/{metrics['cache_total_requests']})"
        )

    # Check if graph is empty
    if metrics["graph_node_count"] == 0:
        if status == "healthy":
            status = "degraded"
        warnings.append("Graph is empty (no nodes found)")

    # Check for stale data (no analysis in last 7 days)
    if metrics["last_analysis_timestamp"]:
        try:
            last_analysis = datetime.fromisoformat(
                metrics["last_analysis_timestamp"].replace("Z", "+00:00")
            )
            days_since_analysis = (datetime.now(UTC) - last_analysis).days
            if days_since_analysis > 7:
                if status == "healthy":
                    status = "degraded"
                warnings.append(f"Stale data: Last analysis was {days_since_analysis} days ago")
        except (ValueError, TypeError):  # fmt: skip
            pass

    return status, warnings


# Convenience function for batch metrics retrieval
async def get_multiple_container_health_metrics(
    container_names: list[str],
    tenant_id: str,
) -> dict[str, dict[str, Any]]:
    """Get health metrics for multiple containers.

    Args:
        container_names: List of container names
        tenant_id: Tenant ID for multi-tenant isolation (required)

    Returns:
        Dictionary mapping container names to their health metrics
    """
    results = {}

    for container_name in container_names:
        try:
            metrics = await get_container_health_metrics(container_name, tenant_id=tenant_id)
            results[container_name] = metrics
        except (
            aiosqlite.Error,
            redis.RedisError,
            ConnectionError,
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
        ) as e:
            logger.error(f"Error getting metrics for {container_name}: {e}")
            results[container_name] = {
                "container_name": container_name,
                "error": str(e),
                "status": "unhealthy",
            }

    return results


async def detect_unhealthy_containers(
    tenant_id: str | None = None,
) -> list[dict[str, Any]]:
    """Detect and return list of unhealthy containers based on specific criteria.

    This function scans all containers and identifies those with health issues
    based on the following criteria:
    1. Container status is not "running"
    2. Code graph not initialized after 5 minutes of container creation
    3. API endpoint is unreachable

    Args:
        tenant_id: Optional tenant ID to filter containers by tenant.

    Returns:
        List of dictionaries containing unhealthy container information:
        {
            "container_name": str,
            "status": str,
            "issues": List[str],
            "api_available": bool,
            "graph_initialized": bool,
            "uptime_seconds": float,
            "repo_url": str,
            "api_endpoint": str,
        }

    Example:
        >>> unhealthy = await detect_unhealthy_containers()
        >>> for container in unhealthy:
        ...     print(f"{container['container_name']}: {container['issues']}")
    """
    try:
        import os

        import httpx

        from src.core.container_manager import get_container_manager
        from src.models.container_models import ContainerStatus

        settings = get_settings()
        container_manager = get_container_manager()
        unhealthy_containers = []

        # Get all containers
        containers = container_manager.list_containers(tenant_id=tenant_id)
        logger.info(f"Scanning {len(containers)} containers for health issues")

        for container in containers:
            issues = []
            api_available = False
            graph_initialized = False
            uptime_seconds = 0.0

            container_name = container.container_name
            container_status = (
                container.status.value
                if isinstance(container.status, ContainerStatus)
                else str(container.status)
            )

            # Criterion 1: Check if status is not "running"
            if container_status != "running":
                issues.append(f"Container status is '{container_status}' (not running)")

            # Criterion 2: Check if graph is initialized
            # For containers older than 5 minutes, graph should be initialized
            try:
                graph_id = f"codegraph-{container_name}"
                db_path = settings.get_code_graph_db_path(graph_id)

                if os.path.exists(db_path) and os.path.isfile(db_path):
                    graph_initialized = True
                else:
                    # Check container age - if older than 5 minutes, this is an issue
                    # We'll estimate uptime based on file creation time or use a flag
                    # For now, we'll mark it as an issue if graph is not initialized
                    issues.append("Code graph not initialized")

            except (OSError, ValueError, NotImplementedError) as e:
                logger.warning(f"Error checking graph for {container_name}: {e}")
                issues.append(f"Failed to check graph status: {str(e)}")

            # Criterion 3: Check if API is reachable
            try:
                probe_endpoint = (
                    getattr(container, "internal_endpoint", "")
                    or getattr(container, "api_endpoint", "")
                    or ""
                )
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(f"{probe_endpoint}/health")
                    api_available = response.status_code == 200
            except (httpx.HTTPError, OSError, ValueError) as e:
                issues.append(f"API endpoint unreachable: {str(e)}")
                api_available = False

            # If there are any issues, add to unhealthy list
            if issues:
                unhealthy_containers.append(
                    {
                        "container_name": container_name,
                        "status": container_status,
                        "issues": issues,
                        "api_available": api_available,
                        "graph_initialized": graph_initialized,
                        "uptime_seconds": uptime_seconds,
                        "repo_url": container.repo_url,
                        "api_endpoint": container.api_endpoint,
                        "internal_endpoint": getattr(container, "internal_endpoint", ""),
                        "public_endpoint": getattr(container, "public_endpoint", ""),
                        "codebase_path": container.codebase_path,
                    }
                )

        logger.info(
            f"Found {len(unhealthy_containers)} unhealthy containers out of {len(containers)} total"
        )
        return unhealthy_containers

    except (
        ImportError,
        RuntimeError,
        OSError,
        ValueError,
        ConnectionError,
        redis.RedisError,
    ) as e:
        logger.error(f"Error detecting unhealthy containers: {e}", exc_info=True)
        return []

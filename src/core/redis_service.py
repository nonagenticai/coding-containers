"""Thin wrapper that re-exports the Redis service from the core module.

The full Redis service implementation lives in the monolith. This wrapper
provides the same interface for use within the coding-containers microservice.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from redis import ConnectionPool, Redis
from redis.exceptions import ConnectionError, RedisError

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class RedisService:
    """Service for managing Redis connections and caching operations.

    Provides methods for basic Redis operations with automatic connection
    management, error handling, and JSON serialization support.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db: int | None = None,
        password: str | None = None,
        decode_responses: bool = True,
        max_connections: int = 50,
        socket_connect_timeout: int = 5,
        socket_timeout: int = 5,
    ):
        """Initialize Redis service with connection parameters.

        Args:
            host: Redis host (defaults to settings.REDIS_HOST)
            port: Redis port (defaults to settings.REDIS_PORT)
            db: Redis database number (defaults to settings.REDIS_DB)
            password: Redis password (defaults to settings.REDIS_PASSWORD)
            decode_responses: Whether to decode responses to strings
            max_connections: Maximum number of connections in the pool
            socket_connect_timeout: Timeout for connection attempts
            socket_timeout: Timeout for socket operations
        """
        settings = get_settings()
        self.host = host or settings.redis.host
        self.port = port or settings.redis.port
        self.db = db or settings.redis.db
        self.password = password or settings.redis.password
        self.decode_responses = decode_responses

        # Create connection pool
        self.pool = ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=self.decode_responses,
            max_connections=max_connections,
            socket_connect_timeout=socket_connect_timeout,
            socket_timeout=socket_timeout,
        )

        self._client: Redis | None = None
        self._is_connected = False

    @property
    def client(self) -> Redis:
        """Get Redis client instance, creating one if necessary.

        Returns:
            Redis client instance

        Raises:
            ConnectionError: If unable to connect to Redis
        """
        if self._client is None:
            self._client = Redis(connection_pool=self.pool)
            try:
                # Test connection
                self._client.ping()
                self._is_connected = True
                logger.info(f"Connected to Redis at {self.host}:{self.port}")
            except ConnectionError as e:
                self._is_connected = False
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self._client

    def is_connected(self) -> bool:
        """Check if Redis connection is active.

        Returns:
            True if connected, False otherwise
        """
        try:
            self.client.ping()
            self._is_connected = True
            return True
        except (ConnectionError, RedisError):  # fmt: skip
            self._is_connected = False
            return False

    def ping(self) -> bool:
        """Ping Redis server to test connectivity.

        Returns:
            True if ping successful, False otherwise
        """
        try:
            result = self.client.ping()
            return result is True
        except (ConnectionError, RedisError):  # fmt: skip
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from Redis by key.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        try:
            value = self.client.get(key)
            if value is None:
                return default

            # Try to deserialize JSON
            if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass

            return value
        except RedisError as e:
            logger.error(f"Error getting key {key}: {e}")
            return default

    def set(
        self,
        key: str,
        value: Any,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
        xx: bool = False,
        ttl: int | None = None,
    ) -> bool:
        """Set value in Redis with optional expiration.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized if not string)
            ex: Expiration time in seconds
            px: Expiration time in milliseconds
            nx: Only set if key doesn't exist
            xx: Only set if key exists
            ttl: Alias for ex (expiration time in seconds)

        Returns:
            True if successful, False otherwise
        """
        try:
            # ttl is an alias for ex
            if ttl is not None and ex is None:
                ex = ttl

            # Serialize complex objects to JSON
            if not isinstance(value, (str, bytes, int, float)):
                value = json.dumps(value)

            result = self.client.set(key, value, ex=ex, px=px, nx=nx, xx=xx)
            return bool(result)
        except RedisError as e:
            logger.error(f"Error setting key {key}: {e}")
            return False

    def delete(self, *keys: str) -> int:
        """Delete one or more keys from Redis.

        Args:
            *keys: Keys to delete

        Returns:
            Number of keys deleted
        """
        try:
            result = self.client.delete(*keys)
            return result if isinstance(result, int) else 0
        except RedisError as e:
            logger.error(f"Error deleting keys {keys}: {e}")
            return 0

    def exists(self, *keys: str) -> int:
        """Check if one or more keys exist.

        Args:
            *keys: Keys to check

        Returns:
            Number of keys that exist
        """
        try:
            result = self.client.exists(*keys)
            return result if isinstance(result, int) else 0
        except RedisError as e:
            logger.error(f"Error checking existence of keys {keys}: {e}")
            return 0

    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration time for a key.

        Args:
            key: Cache key
            seconds: Expiration time in seconds

        Returns:
            True if expiration was set, False otherwise
        """
        try:
            return bool(self.client.expire(key, seconds))
        except RedisError as e:
            logger.error(f"Error setting expiration for key {key}: {e}")
            return False

    def ttl(self, key: str) -> int:
        """Get time-to-live for a key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds, -1 if no expiration, -2 if key doesn't exist
        """
        try:
            result = self.client.ttl(key)
            return result if isinstance(result, int) else -2
        except RedisError as e:
            logger.error(f"Error getting TTL for key {key}: {e}")
            return -2

    def incr(self, key: str, amount: int = 1) -> int | None:
        """Increment a counter.

        Args:
            key: Counter key
            amount: Amount to increment by

        Returns:
            New counter value or None on error
        """
        try:
            result = self.client.incr(key, amount)
            return result if isinstance(result, int) else None
        except RedisError as e:
            logger.error(f"Error incrementing key {key}: {e}")
            return None

    def decr(self, key: str, amount: int = 1) -> int | None:
        """Decrement a counter.

        Args:
            key: Counter key
            amount: Amount to decrement by

        Returns:
            New counter value or None on error
        """
        try:
            result = self.client.decr(key, amount)
            return result if isinstance(result, int) else None
        except RedisError as e:
            logger.error(f"Error decrementing key {key}: {e}")
            return None

    def hget(self, name: str, key: str) -> Any:
        """Get value from a hash.

        Args:
            name: Hash name
            key: Field key

        Returns:
            Field value or None
        """
        try:
            value = self.client.hget(name, key)
            if (
                value
                and isinstance(value, str)
                and (value.startswith("{") or value.startswith("["))
            ):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            return value
        except RedisError as e:
            logger.error(f"Error getting hash field {name}.{key}: {e}")
            return None

    def hset(self, name: str, key: str, value: Any) -> bool:
        """Set value in a hash.

        Args:
            name: Hash name
            key: Field key
            value: Field value

        Returns:
            True if successful, False otherwise
        """
        try:
            if not isinstance(value, (str, bytes, int, float)):
                value = json.dumps(value)
            self.client.hset(name, key, value)
            return True
        except RedisError as e:
            logger.error(f"Error setting hash field {name}.{key}: {e}")
            return False

    def hgetall(self, name: str) -> dict:
        """Get all fields and values from a hash.

        Args:
            name: Hash name

        Returns:
            Dictionary of field-value pairs
        """
        try:
            data = self.client.hgetall(name)
            if not isinstance(data, dict):
                return {}
            # Try to deserialize JSON values
            result: dict[Any, Any] = {}
            for key, value in data.items():
                if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        result[key] = value
                else:
                    result[key] = value
            return result
        except RedisError as e:
            logger.error(f"Error getting hash {name}: {e}")
            return {}

    def hdel(self, name: str, *keys: str) -> int:
        """Delete fields from a hash.

        Args:
            name: Hash name
            *keys: Field keys to delete

        Returns:
            Number of fields deleted
        """
        try:
            result = self.client.hdel(name, *keys)
            return result if isinstance(result, int) else 0
        except RedisError as e:
            logger.error(f"Error deleting hash fields {name}.{keys}: {e}")
            return 0

    def lpush(self, key: str, *values: Any) -> int | None:
        """Push values to the left of a list.

        Args:
            key: List key
            *values: Values to push

        Returns:
            New list length or None on error
        """
        try:
            # Serialize non-string values
            serialized_values: list[Any] = []
            for value in values:
                if not isinstance(value, (str, bytes, int, float)):
                    serialized_values.append(json.dumps(value))
                else:
                    serialized_values.append(value)
            result = self.client.lpush(key, *serialized_values)
            return result if isinstance(result, int) else None
        except RedisError as e:
            logger.error(f"Error pushing to list {key}: {e}")
            return None

    def rpush(self, key: str, *values: Any) -> int | None:
        """Push values to the right of a list.

        Args:
            key: List key
            *values: Values to push

        Returns:
            New list length or None on error
        """
        try:
            # Serialize non-string values
            serialized_values: list[Any] = []
            for value in values:
                if not isinstance(value, (str, bytes, int, float)):
                    serialized_values.append(json.dumps(value))
                else:
                    serialized_values.append(value)
            result = self.client.rpush(key, *serialized_values)
            return result if isinstance(result, int) else None
        except RedisError as e:
            logger.error(f"Error pushing to list {key}: {e}")
            return None

    def lrange(self, key: str, start: int, stop: int) -> list:
        """Get range of elements from a list.

        Args:
            key: List key
            start: Start index
            stop: Stop index (-1 for end)

        Returns:
            List of elements
        """
        try:
            values = self.client.lrange(key, start, stop)
            if not isinstance(values, list):
                return []
            # Try to deserialize JSON values
            result: list[Any] = []
            for value in values:
                if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                    try:
                        result.append(json.loads(value))
                    except json.JSONDecodeError:
                        result.append(value)
                else:
                    result.append(value)
            return result
        except RedisError as e:
            logger.error(f"Error getting list range {key}[{start}:{stop}]: {e}")
            return []

    def flushdb(self) -> bool:
        """Flush current database.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.flushdb()
            logger.warning(f"Flushed Redis database {self.db}")
            return True
        except RedisError as e:
            logger.error(f"Error flushing database: {e}")
            return False

    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._is_connected = False
            logger.info("Closed Redis connection")

    def __enter__(self) -> RedisService:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    # =========================================================================
    # Key scanning methods
    # =========================================================================

    def scan_keys(self, pattern: str = "*", count: int = 100) -> list[str]:
        """Scan for keys matching a pattern.

        Args:
            pattern: Pattern to match (e.g., "prefix:*")
            count: Hint for number of keys to return per iteration

        Returns:
            List of matching keys
        """
        try:
            keys: list[str] = []
            cursor: int = 0
            while True:
                scan_result = self.client.scan(cursor=cursor, match=pattern, count=count)
                if not isinstance(scan_result, tuple) or len(scan_result) != 2:
                    break
                next_cursor, batch = scan_result
                if not isinstance(next_cursor, int):
                    break
                cursor = next_cursor
                if isinstance(batch, list):
                    for item in batch:
                        if isinstance(item, str):
                            keys.append(item)
                        elif isinstance(item, bytes):
                            keys.append(item.decode("utf-8", errors="replace"))
                if cursor == 0:
                    break
            return keys
        except RedisError as e:
            logger.error(f"Error scanning keys with pattern {pattern}: {e}")
            return []

    # =========================================================================
    # Async wrapper methods
    # These methods wrap the synchronous methods using asyncio.to_thread
    # to allow awaiting them in async code without blocking the event loop.
    # =========================================================================

    async def async_ping(self) -> bool:
        """Async wrapper for ping().

        Returns:
            True if ping successful, False otherwise
        """
        return await asyncio.to_thread(self.ping)

    async def async_get(self, key: str, default: Any = None) -> Any:
        """Async wrapper for get().

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        return await asyncio.to_thread(self.get, key, default)

    async def async_set(
        self,
        key: str,
        value: Any,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
        xx: bool = False,
        ttl: int | None = None,
    ) -> bool:
        """Async wrapper for set().

        Args:
            key: Cache key
            value: Value to cache
            ex: Expiration time in seconds
            px: Expiration time in milliseconds
            nx: Only set if key doesn't exist
            xx: Only set if key exists
            ttl: Alias for ex (expiration time in seconds)

        Returns:
            True if successful, False otherwise
        """
        return await asyncio.to_thread(self.set, key, value, ex, px, nx, xx, ttl)

    async def async_delete(self, *keys: str) -> int:
        """Async wrapper for delete().

        Args:
            *keys: Keys to delete

        Returns:
            Number of keys deleted
        """
        return await asyncio.to_thread(self.delete, *keys)

    async def async_exists(self, *keys: str) -> int:
        """Async wrapper for exists().

        Args:
            *keys: Keys to check

        Returns:
            Number of keys that exist
        """
        return await asyncio.to_thread(self.exists, *keys)

    async def async_scan_keys(self, pattern: str = "*", count: int = 100) -> list[str]:
        """Async wrapper for scan_keys().

        Args:
            pattern: Pattern to match (e.g., "prefix:*")
            count: Hint for number of keys to return per iteration

        Returns:
            List of matching keys
        """
        return await asyncio.to_thread(self.scan_keys, pattern, count)


# Global Redis service instance
redis_service: RedisService | None = None


def init_redis_service(**kwargs) -> RedisService:
    """Initialize global Redis service with custom parameters.

    Args:
        **kwargs: Arguments to pass to RedisService constructor

    Returns:
        RedisService instance
    """
    global redis_service
    redis_service = RedisService(**kwargs)
    return redis_service


class RedisServiceContainer:
    """Container for managing the singleton Redis service instance.

    This class encapsulates the singleton pattern for Redis service management,
    avoiding the need for global statements while maintaining thread safety.
    """

    _redis_service: RedisService | None = None

    @classmethod
    async def get_service(cls) -> RedisService:
        """Get the global Redis service instance."""
        if cls._redis_service is None:
            cls._redis_service = RedisService()
            logger.info("Created global Redis service instance")
        return cls._redis_service

    @classmethod
    def reset(cls) -> None:
        """Reset the Redis service for testing purposes."""
        cls._redis_service = None


async def get_redis_service() -> RedisService:
    """Get the global Redis service instance (async version).

    This is the public API function that delegates to the container
    to avoid global statements while maintaining the same interface.

    Returns:
        RedisService instance
    """
    return await RedisServiceContainer.get_service()


def get_redis_service_sync() -> RedisService:
    """Get or create global Redis service instance (synchronous version).

    Returns:
        RedisService instance
    """
    global redis_service
    if redis_service is None:
        redis_service = RedisService()
    return redis_service

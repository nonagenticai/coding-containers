"""Cache manager for coding-containers backed by Redis."""

from __future__ import annotations

import json
import logging
from typing import Any

from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

_cache_manager = None


class RedisCacheManager:
    """Redis-backed cache manager. Falls back to no-op if Redis unavailable."""

    def __init__(self, redis_client: Any = None) -> None:
        self._redis = redis_client

    def _get_redis(self) -> Any:
        if self._redis is None:
            try:
                from src.core.redis_service import get_redis_service_sync

                svc = get_redis_service_sync()
                self._redis = svc.client
            except ImportError, RedisError, RedisConnectionError, OSError:
                return None
        return self._redis

    async def get(self, key: str) -> Any:
        r = self._get_redis()
        if r is None:
            return None
        try:
            val = r.get(key)
            if val is None:
                return None
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except json.JSONDecodeError, ValueError:
                    return val
            return val
        except RedisError:
            return None

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        r = self._get_redis()
        if r is None:
            return
        try:
            r.setex(key, ttl, json.dumps(value, default=str))
        except RedisError, TypeError, ValueError:
            pass

    async def delete(self, key: str) -> None:
        r = self._get_redis()
        if r is None:
            return
        try:
            r.delete(key)
        except RedisError:
            pass

    async def get_or_compute(self, key: str, compute_fn: Any, ttl: int = 300) -> Any:
        cached = await self.get(key)
        if cached is not None:
            return cached
        result = await compute_fn()
        await self.set(key, result, ttl=ttl)
        return result

    def get_stats(self) -> dict:
        r = self._get_redis()
        if r is None:
            return {"hits": 0, "misses": 0, "errors": 0}
        try:
            info = r.info("stats")
            return {
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "errors": 0,
            }
        except RedisError:
            return {"hits": 0, "misses": 0, "errors": 0}

    async def get_container_stats(self, container_name: str) -> dict:
        r = self._get_redis()
        if r is None:
            return {"hits": 0, "misses": 0, "size": 0}
        try:
            keys = r.keys(f"*{container_name}*")
            return {"hits": 0, "misses": 0, "size": len(keys)}
        except RedisError:
            return {"hits": 0, "misses": 0, "size": 0}

    async def clear_container_cache(self, container_name: str) -> int:
        r = self._get_redis()
        if r is None:
            return 0
        try:
            keys = r.keys(f"*{container_name}*")
            if keys:
                return r.delete(*keys)
            return 0
        except RedisError:
            return 0


def get_cache_manager() -> RedisCacheManager:
    """Get singleton cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = RedisCacheManager()
    return _cache_manager


def get_cache_manager_sync() -> RedisCacheManager:
    """Get singleton cache manager (sync version)."""
    return get_cache_manager()

"""Tests for src/utils/cache_manager.py — RedisCacheManager."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError

from src.utils.cache_manager import RedisCacheManager, get_cache_manager

# ---------------------------------------------------------------------------
# RedisCacheManager — construction
# ---------------------------------------------------------------------------


class TestRedisCacheManagerInit:
    def test_accepts_redis_client(self):
        redis = MagicMock()
        mgr = RedisCacheManager(redis_client=redis)
        assert mgr._redis is redis

    def test_defaults_to_none(self):
        mgr = RedisCacheManager()
        assert mgr._redis is None


# ---------------------------------------------------------------------------
# _get_redis
# ---------------------------------------------------------------------------


class TestGetRedis:
    def test_returns_existing_client(self):
        redis = MagicMock()
        mgr = RedisCacheManager(redis_client=redis)
        assert mgr._get_redis() is redis

    @patch("src.core.redis_service.get_redis_service_sync")
    def test_lazy_loads_redis(self, mock_redis_sync):
        mock_svc = MagicMock()
        mock_svc.client = MagicMock()
        mock_redis_sync.return_value = mock_svc

        mgr = RedisCacheManager()
        result = mgr._get_redis()
        assert result is mock_svc.client
        assert mgr._redis is mock_svc.client

    @patch(
        "src.core.redis_service.get_redis_service_sync",
        side_effect=RedisConnectionError("no redis"),
    )
    def test_returns_none_on_import_error(self, mock_redis_sync):
        mgr = RedisCacheManager()
        assert mgr._get_redis() is None


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGet:
    async def test_returns_none_when_no_redis(self):
        mgr = RedisCacheManager()
        with patch.object(mgr, "_get_redis", return_value=None):
            result = await mgr.get("key")
        assert result is None

    async def test_returns_parsed_json(self):
        redis = MagicMock()
        data = {"foo": "bar"}
        redis.get.return_value = json.dumps(data)
        mgr = RedisCacheManager(redis_client=redis)

        result = await mgr.get("key")
        assert result == data

    async def test_returns_raw_string_on_json_error(self):
        redis = MagicMock()
        redis.get.return_value = "plain-text"
        mgr = RedisCacheManager(redis_client=redis)

        result = await mgr.get("key")
        assert result == "plain-text"

    async def test_returns_none_when_key_missing(self):
        redis = MagicMock()
        redis.get.return_value = None
        mgr = RedisCacheManager(redis_client=redis)

        result = await mgr.get("missing")
        assert result is None

    async def test_returns_non_string_value_directly(self):
        redis = MagicMock()
        redis.get.return_value = 42
        mgr = RedisCacheManager(redis_client=redis)

        result = await mgr.get("key")
        assert result == 42

    async def test_returns_none_on_exception(self):
        redis = MagicMock()
        redis.get.side_effect = RedisError("boom")
        mgr = RedisCacheManager(redis_client=redis)

        result = await mgr.get("key")
        assert result is None


# ---------------------------------------------------------------------------
# set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSet:
    async def test_sets_value_with_ttl(self):
        redis = MagicMock()
        mgr = RedisCacheManager(redis_client=redis)

        await mgr.set("key", {"a": 1}, ttl=60)
        redis.setex.assert_called_once_with("key", 60, json.dumps({"a": 1}))

    async def test_default_ttl_is_300(self):
        redis = MagicMock()
        mgr = RedisCacheManager(redis_client=redis)

        await mgr.set("key", "val")
        redis.setex.assert_called_once_with("key", 300, json.dumps("val"))

    async def test_noop_when_no_redis(self):
        mgr = RedisCacheManager()
        with patch.object(mgr, "_get_redis", return_value=None):
            await mgr.set("key", "val")  # should not raise

    async def test_swallows_exception(self):
        redis = MagicMock()
        redis.setex.side_effect = RedisError("write fail")
        mgr = RedisCacheManager(redis_client=redis)

        await mgr.set("key", "val")  # should not raise


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDelete:
    async def test_deletes_key(self):
        redis = MagicMock()
        mgr = RedisCacheManager(redis_client=redis)

        await mgr.delete("key")
        redis.delete.assert_called_once_with("key")

    async def test_noop_when_no_redis(self):
        mgr = RedisCacheManager()
        with patch.object(mgr, "_get_redis", return_value=None):
            await mgr.delete("key")  # should not raise

    async def test_swallows_exception(self):
        redis = MagicMock()
        redis.delete.side_effect = RedisError("del fail")
        mgr = RedisCacheManager(redis_client=redis)

        await mgr.delete("key")  # should not raise


# ---------------------------------------------------------------------------
# get_or_compute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetOrCompute:
    async def test_returns_cached_value(self):
        redis = MagicMock()
        redis.get.return_value = json.dumps({"cached": True})
        mgr = RedisCacheManager(redis_client=redis)

        compute_fn = AsyncMock(return_value={"computed": True})
        result = await mgr.get_or_compute("key", compute_fn, ttl=60)

        assert result == {"cached": True}
        compute_fn.assert_not_called()

    async def test_computes_and_caches_on_miss(self):
        redis = MagicMock()
        redis.get.return_value = None
        mgr = RedisCacheManager(redis_client=redis)

        compute_fn = AsyncMock(return_value={"computed": True})
        result = await mgr.get_or_compute("key", compute_fn, ttl=120)

        assert result == {"computed": True}
        compute_fn.assert_awaited_once()
        redis.setex.assert_called_once_with("key", 120, json.dumps({"computed": True}))


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_returns_redis_info(self):
        redis = MagicMock()
        redis.info.return_value = {"keyspace_hits": 100, "keyspace_misses": 20}
        mgr = RedisCacheManager(redis_client=redis)

        stats = mgr.get_stats()
        assert stats == {"hits": 100, "misses": 20, "errors": 0}

    def test_returns_defaults_when_no_redis(self):
        mgr = RedisCacheManager()
        with patch.object(mgr, "_get_redis", return_value=None):
            stats = mgr.get_stats()
        assert stats == {"hits": 0, "misses": 0, "errors": 0}

    def test_returns_defaults_on_exception(self):
        redis = MagicMock()
        redis.info.side_effect = RedisError("info fail")
        mgr = RedisCacheManager(redis_client=redis)

        stats = mgr.get_stats()
        assert stats == {"hits": 0, "misses": 0, "errors": 0}


# ---------------------------------------------------------------------------
# get_container_stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetContainerStats:
    async def test_returns_key_count(self):
        redis = MagicMock()
        redis.keys.return_value = ["k1", "k2", "k3"]
        mgr = RedisCacheManager(redis_client=redis)

        stats = await mgr.get_container_stats("my-ctr")
        assert stats == {"hits": 0, "misses": 0, "size": 3}
        redis.keys.assert_called_once_with("*my-ctr*")

    async def test_returns_defaults_when_no_redis(self):
        mgr = RedisCacheManager()
        with patch.object(mgr, "_get_redis", return_value=None):
            stats = await mgr.get_container_stats("ctr")
        assert stats == {"hits": 0, "misses": 0, "size": 0}

    async def test_returns_defaults_on_exception(self):
        redis = MagicMock()
        redis.keys.side_effect = RedisError("keys fail")
        mgr = RedisCacheManager(redis_client=redis)

        stats = await mgr.get_container_stats("ctr")
        assert stats == {"hits": 0, "misses": 0, "size": 0}


# ---------------------------------------------------------------------------
# clear_container_cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestClearContainerCache:
    async def test_deletes_matching_keys(self):
        redis = MagicMock()
        redis.keys.return_value = ["k1", "k2"]
        redis.delete.return_value = 2
        mgr = RedisCacheManager(redis_client=redis)

        count = await mgr.clear_container_cache("my-ctr")
        assert count == 2
        redis.delete.assert_called_once_with("k1", "k2")

    async def test_returns_zero_when_no_keys(self):
        redis = MagicMock()
        redis.keys.return_value = []
        mgr = RedisCacheManager(redis_client=redis)

        count = await mgr.clear_container_cache("ctr")
        assert count == 0

    async def test_returns_zero_when_no_redis(self):
        mgr = RedisCacheManager()
        with patch.object(mgr, "_get_redis", return_value=None):
            count = await mgr.clear_container_cache("ctr")
        assert count == 0

    async def test_returns_zero_on_exception(self):
        redis = MagicMock()
        redis.keys.side_effect = RedisError("keys fail")
        mgr = RedisCacheManager(redis_client=redis)

        count = await mgr.clear_container_cache("ctr")
        assert count == 0


# ---------------------------------------------------------------------------
# get_cache_manager singleton
# ---------------------------------------------------------------------------


class TestGetCacheManager:
    def test_returns_singleton(self):
        import src.utils.cache_manager as cm_module

        cm_module._cache_manager = None
        m1 = get_cache_manager()
        m2 = get_cache_manager()
        assert m1 is m2
        assert isinstance(m1, RedisCacheManager)
        # Singleton constructed without explicit client starts with _redis=None
        assert m1._redis is None
        cm_module._cache_manager = None  # cleanup

    def test_get_cache_manager_sync_same_as_get(self):
        import src.utils.cache_manager as cm_module
        from src.utils.cache_manager import get_cache_manager_sync

        cm_module._cache_manager = None
        m1 = get_cache_manager()
        m2 = get_cache_manager_sync()
        assert m1 is m2
        cm_module._cache_manager = None  # cleanup

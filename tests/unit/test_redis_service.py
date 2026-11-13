"""Tests for core/redis_service.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from redis.exceptions import ConnectionError, RedisError

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def redis_service():
    """Create a RedisService instance with mocked settings and ConnectionPool."""
    with (
        patch("src.core.redis_service.get_settings") as mock_settings,
        patch("src.core.redis_service.ConnectionPool") as mock_pool_cls,
    ):
        settings = MagicMock()
        settings.redis.host = "test-host"
        settings.redis.port = 6379
        settings.redis.db = 0
        settings.redis.password = None
        mock_settings.return_value = settings

        mock_pool_cls.return_value = MagicMock()

        from src.core.redis_service import RedisService

        svc = RedisService()
        yield svc


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults_from_settings(self):
        with (
            patch("src.core.redis_service.get_settings") as mock_settings,
            patch("src.core.redis_service.ConnectionPool") as mock_pool_cls,
        ):
            settings = MagicMock()
            settings.redis.host = "my-host"
            settings.redis.port = 6380
            settings.redis.db = 2
            settings.redis.password = "secret"
            mock_settings.return_value = settings
            mock_pool_cls.return_value = MagicMock()

            from src.core.redis_service import RedisService

            svc = RedisService()
            assert svc.host == "my-host"
            assert svc.port == 6380
            assert svc.db == 2
            assert svc.password == "secret"
            mock_pool_cls.assert_called_once_with(
                host="my-host",
                port=6380,
                db=2,
                password="secret",
                decode_responses=True,
                max_connections=50,
                socket_connect_timeout=5,
                socket_timeout=5,
            )

    def test_explicit_params_override_settings(self):
        with (
            patch("src.core.redis_service.get_settings") as mock_settings,
            patch("src.core.redis_service.ConnectionPool") as mock_pool_cls,
        ):
            settings = MagicMock()
            settings.redis.host = "default-host"
            settings.redis.port = 6379
            settings.redis.db = 0
            settings.redis.password = None
            mock_settings.return_value = settings
            mock_pool_cls.return_value = MagicMock()

            from src.core.redis_service import RedisService

            svc = RedisService(host="override-host", port=7000, db=5)
            assert svc.host == "override-host"
            assert svc.port == 7000
            assert svc.db == 5


# ---------------------------------------------------------------------------
# client property
# ---------------------------------------------------------------------------


class TestClientProperty:
    @patch("src.core.redis_service.Redis")
    def test_lazy_init_ping_success(self, mock_redis_cls, redis_service):
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis_cls.return_value = mock_client

        client = redis_service.client
        assert client is mock_client
        mock_client.ping.assert_called_once()
        assert redis_service._is_connected is True

    @patch("src.core.redis_service.Redis")
    def test_connection_failure(self, mock_redis_cls, redis_service):
        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionError("refused")
        mock_redis_cls.return_value = mock_client

        with pytest.raises(ConnectionError):
            _ = redis_service.client
        assert redis_service._is_connected is False

    @patch("src.core.redis_service.Redis")
    def test_returns_cached_client(self, mock_redis_cls, redis_service):
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis_cls.return_value = mock_client

        c1 = redis_service.client
        c2 = redis_service.client
        assert c1 is c2
        # Redis() called only once
        mock_redis_cls.assert_called_once()


# ---------------------------------------------------------------------------
# is_connected
# ---------------------------------------------------------------------------


class TestIsConnected:
    def test_true_on_successful_ping(self, redis_service):
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        redis_service._client = mock_client

        assert redis_service.is_connected() is True
        assert redis_service._is_connected is True

    def test_false_on_connection_error(self, redis_service):
        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionError("gone")
        redis_service._client = mock_client

        assert redis_service.is_connected() is False
        assert redis_service._is_connected is False

    def test_false_on_redis_error(self, redis_service):
        mock_client = MagicMock()
        mock_client.ping.side_effect = RedisError("oops")
        redis_service._client = mock_client

        assert redis_service.is_connected() is False


# ---------------------------------------------------------------------------
# get / set
# ---------------------------------------------------------------------------


class TestGetSet:
    def test_get_returns_string(self, redis_service):
        mock_client = MagicMock()
        mock_client.get.return_value = "hello"
        redis_service._client = mock_client

        assert redis_service.get("key") == "hello"

    def test_get_returns_default_when_none(self, redis_service):
        mock_client = MagicMock()
        mock_client.get.return_value = None
        redis_service._client = mock_client

        assert redis_service.get("missing", default="fallback") == "fallback"

    def test_get_deserializes_json_dict(self, redis_service):
        payload = {"a": 1, "b": [2, 3]}
        mock_client = MagicMock()
        mock_client.get.return_value = json.dumps(payload)
        redis_service._client = mock_client

        assert redis_service.get("key") == payload

    def test_get_deserializes_json_list(self, redis_service):
        payload = [1, 2, 3]
        mock_client = MagicMock()
        mock_client.get.return_value = json.dumps(payload)
        redis_service._client = mock_client

        assert redis_service.get("key") == payload

    def test_get_returns_default_on_error(self, redis_service):
        mock_client = MagicMock()
        mock_client.get.side_effect = RedisError("err")
        redis_service._client = mock_client

        assert redis_service.get("key", default="safe") == "safe"

    def test_set_string_value(self, redis_service):
        mock_client = MagicMock()
        mock_client.set.return_value = True
        redis_service._client = mock_client

        assert redis_service.set("key", "value") is True
        mock_client.set.assert_called_once_with(
            "key", "value", ex=None, px=None, nx=False, xx=False
        )

    def test_set_dict_serializes_to_json(self, redis_service):
        mock_client = MagicMock()
        mock_client.set.return_value = True
        redis_service._client = mock_client

        data = {"nested": True}
        redis_service.set("key", data)
        call_args = mock_client.set.call_args
        assert call_args[0][1] == json.dumps(data)

    def test_set_with_ttl(self, redis_service):
        mock_client = MagicMock()
        mock_client.set.return_value = True
        redis_service._client = mock_client

        redis_service.set("key", "val", ttl=300)
        mock_client.set.assert_called_once_with("key", "val", ex=300, px=None, nx=False, xx=False)

    def test_set_with_nx(self, redis_service):
        mock_client = MagicMock()
        mock_client.set.return_value = True
        redis_service._client = mock_client

        redis_service.set("key", "val", nx=True)
        mock_client.set.assert_called_once_with("key", "val", ex=None, px=None, nx=True, xx=False)

    def test_set_with_xx(self, redis_service):
        mock_client = MagicMock()
        mock_client.set.return_value = True
        redis_service._client = mock_client

        redis_service.set("key", "val", xx=True)
        mock_client.set.assert_called_once_with("key", "val", ex=None, px=None, nx=False, xx=True)

    def test_set_returns_false_on_error(self, redis_service):
        mock_client = MagicMock()
        mock_client.set.side_effect = RedisError("err")
        redis_service._client = mock_client

        assert redis_service.set("key", "val") is False


# ---------------------------------------------------------------------------
# delete / exists / expire / ttl
# ---------------------------------------------------------------------------


class TestDeleteExistsExpireTTL:
    def test_delete_returns_count(self, redis_service):
        mock_client = MagicMock()
        mock_client.delete.return_value = 2
        redis_service._client = mock_client

        assert redis_service.delete("a", "b") == 2

    def test_delete_error_returns_zero(self, redis_service):
        mock_client = MagicMock()
        mock_client.delete.side_effect = RedisError("err")
        redis_service._client = mock_client

        assert redis_service.delete("a") == 0

    def test_exists_returns_count(self, redis_service):
        mock_client = MagicMock()
        mock_client.exists.return_value = 3
        redis_service._client = mock_client

        assert redis_service.exists("a", "b", "c") == 3

    def test_exists_error_returns_zero(self, redis_service):
        mock_client = MagicMock()
        mock_client.exists.side_effect = RedisError("err")
        redis_service._client = mock_client

        assert redis_service.exists("a") == 0

    def test_expire_success(self, redis_service):
        mock_client = MagicMock()
        mock_client.expire.return_value = True
        redis_service._client = mock_client

        assert redis_service.expire("key", 60) is True

    def test_expire_error(self, redis_service):
        mock_client = MagicMock()
        mock_client.expire.side_effect = RedisError("err")
        redis_service._client = mock_client

        assert redis_service.expire("key", 60) is False

    def test_ttl_returns_value(self, redis_service):
        mock_client = MagicMock()
        mock_client.ttl.return_value = 120
        redis_service._client = mock_client

        assert redis_service.ttl("key") == 120

    def test_ttl_error_returns_neg2(self, redis_service):
        mock_client = MagicMock()
        mock_client.ttl.side_effect = RedisError("err")
        redis_service._client = mock_client

        assert redis_service.ttl("key") == -2


# ---------------------------------------------------------------------------
# incr / decr
# ---------------------------------------------------------------------------


class TestIncrDecr:
    def test_incr_success(self, redis_service):
        mock_client = MagicMock()
        mock_client.incr.return_value = 5
        redis_service._client = mock_client

        assert redis_service.incr("counter") == 5

    def test_incr_with_amount(self, redis_service):
        mock_client = MagicMock()
        mock_client.incr.return_value = 10
        redis_service._client = mock_client

        assert redis_service.incr("counter", 5) == 10
        mock_client.incr.assert_called_once_with("counter", 5)

    def test_incr_error_returns_none(self, redis_service):
        mock_client = MagicMock()
        mock_client.incr.side_effect = RedisError("err")
        redis_service._client = mock_client

        assert redis_service.incr("counter") is None

    def test_decr_success(self, redis_service):
        mock_client = MagicMock()
        mock_client.decr.return_value = 3
        redis_service._client = mock_client

        assert redis_service.decr("counter") == 3

    def test_decr_with_amount(self, redis_service):
        mock_client = MagicMock()
        mock_client.decr.return_value = 0
        redis_service._client = mock_client

        assert redis_service.decr("counter", 3) == 0
        mock_client.decr.assert_called_once_with("counter", 3)

    def test_decr_error_returns_none(self, redis_service):
        mock_client = MagicMock()
        mock_client.decr.side_effect = RedisError("err")
        redis_service._client = mock_client

        assert redis_service.decr("counter") is None


# ---------------------------------------------------------------------------
# hget / hset / hgetall / hdel
# ---------------------------------------------------------------------------


class TestHashOperations:
    def test_hget_string(self, redis_service):
        mock_client = MagicMock()
        mock_client.hget.return_value = "value"
        redis_service._client = mock_client

        assert redis_service.hget("myhash", "field") == "value"

    def test_hget_json_dict(self, redis_service):
        payload = {"x": 1}
        mock_client = MagicMock()
        mock_client.hget.return_value = json.dumps(payload)
        redis_service._client = mock_client

        assert redis_service.hget("myhash", "field") == payload

    def test_hget_json_list(self, redis_service):
        payload = [1, 2]
        mock_client = MagicMock()
        mock_client.hget.return_value = json.dumps(payload)
        redis_service._client = mock_client

        assert redis_service.hget("myhash", "field") == payload

    def test_hget_none(self, redis_service):
        mock_client = MagicMock()
        mock_client.hget.return_value = None
        redis_service._client = mock_client

        assert redis_service.hget("myhash", "field") is None

    def test_hget_error_returns_none(self, redis_service):
        mock_client = MagicMock()
        mock_client.hget.side_effect = RedisError("err")
        redis_service._client = mock_client

        assert redis_service.hget("myhash", "field") is None

    def test_hset_string(self, redis_service):
        mock_client = MagicMock()
        redis_service._client = mock_client

        assert redis_service.hset("myhash", "field", "val") is True
        mock_client.hset.assert_called_once_with("myhash", "field", "val")

    def test_hset_dict_serialized(self, redis_service):
        mock_client = MagicMock()
        redis_service._client = mock_client
        data = {"nested": True}

        redis_service.hset("myhash", "field", data)
        mock_client.hset.assert_called_once_with("myhash", "field", json.dumps(data))

    def test_hset_error(self, redis_service):
        mock_client = MagicMock()
        mock_client.hset.side_effect = RedisError("err")
        redis_service._client = mock_client

        assert redis_service.hset("myhash", "field", "val") is False

    def test_hgetall_with_json_values(self, redis_service):
        mock_client = MagicMock()
        mock_client.hgetall.return_value = {
            "plain": "text",
            "obj": '{"a": 1}',
            "arr": "[1, 2]",
        }
        redis_service._client = mock_client

        result = redis_service.hgetall("myhash")
        assert result == {"plain": "text", "obj": {"a": 1}, "arr": [1, 2]}

    def test_hgetall_error(self, redis_service):
        mock_client = MagicMock()
        mock_client.hgetall.side_effect = RedisError("err")
        redis_service._client = mock_client

        assert redis_service.hgetall("myhash") == {}

    def test_hdel_returns_count(self, redis_service):
        mock_client = MagicMock()
        mock_client.hdel.return_value = 1
        redis_service._client = mock_client

        assert redis_service.hdel("myhash", "f1") == 1

    def test_hdel_error(self, redis_service):
        mock_client = MagicMock()
        mock_client.hdel.side_effect = RedisError("err")
        redis_service._client = mock_client

        assert redis_service.hdel("myhash", "f1") == 0


# ---------------------------------------------------------------------------
# lpush / rpush / lrange
# ---------------------------------------------------------------------------


class TestListOperations:
    def test_lpush_string(self, redis_service):
        mock_client = MagicMock()
        mock_client.lpush.return_value = 1
        redis_service._client = mock_client

        assert redis_service.lpush("mylist", "a") == 1

    def test_lpush_dict_serialized(self, redis_service):
        mock_client = MagicMock()
        mock_client.lpush.return_value = 1
        redis_service._client = mock_client

        data = {"k": "v"}
        redis_service.lpush("mylist", data)
        mock_client.lpush.assert_called_once_with("mylist", json.dumps(data))

    def test_lpush_error(self, redis_service):
        mock_client = MagicMock()
        mock_client.lpush.side_effect = RedisError("err")
        redis_service._client = mock_client

        assert redis_service.lpush("mylist", "a") is None

    def test_rpush_string(self, redis_service):
        mock_client = MagicMock()
        mock_client.rpush.return_value = 2
        redis_service._client = mock_client

        assert redis_service.rpush("mylist", "b") == 2

    def test_rpush_dict_serialized(self, redis_service):
        mock_client = MagicMock()
        mock_client.rpush.return_value = 1
        redis_service._client = mock_client

        data = {"k": "v"}
        redis_service.rpush("mylist", data)
        mock_client.rpush.assert_called_once_with("mylist", json.dumps(data))

    def test_rpush_error(self, redis_service):
        mock_client = MagicMock()
        mock_client.rpush.side_effect = RedisError("err")
        redis_service._client = mock_client

        assert redis_service.rpush("mylist", "a") is None

    def test_lrange_plain(self, redis_service):
        mock_client = MagicMock()
        mock_client.lrange.return_value = ["a", "b"]
        redis_service._client = mock_client

        assert redis_service.lrange("mylist", 0, -1) == ["a", "b"]

    def test_lrange_deserializes_json(self, redis_service):
        mock_client = MagicMock()
        mock_client.lrange.return_value = ['{"x": 1}', "plain"]
        redis_service._client = mock_client

        result = redis_service.lrange("mylist", 0, -1)
        assert result == [{"x": 1}, "plain"]

    def test_lrange_error(self, redis_service):
        mock_client = MagicMock()
        mock_client.lrange.side_effect = RedisError("err")
        redis_service._client = mock_client

        assert redis_service.lrange("mylist", 0, -1) == []


# ---------------------------------------------------------------------------
# scan_keys
# ---------------------------------------------------------------------------


class TestScanKeys:
    def test_single_iteration(self, redis_service):
        mock_client = MagicMock()
        mock_client.scan.return_value = (0, ["key:1", "key:2"])
        redis_service._client = mock_client

        assert redis_service.scan_keys("key:*") == ["key:1", "key:2"]

    def test_multi_iteration(self, redis_service):
        mock_client = MagicMock()
        mock_client.scan.side_effect = [
            (42, ["key:1"]),
            (0, ["key:2"]),
        ]
        redis_service._client = mock_client

        assert redis_service.scan_keys("key:*") == ["key:1", "key:2"]

    def test_error_returns_empty(self, redis_service):
        mock_client = MagicMock()
        mock_client.scan.side_effect = RedisError("err")
        redis_service._client = mock_client

        assert redis_service.scan_keys("*") == []


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_connection(self, redis_service):
        mock_client = MagicMock()
        redis_service._client = mock_client
        redis_service._is_connected = True

        redis_service.close()
        mock_client.close.assert_called_once()
        assert redis_service._client is None
        assert redis_service._is_connected is False

    def test_close_when_no_client(self, redis_service):
        redis_service._client = None
        redis_service.close()  # should not raise


# ---------------------------------------------------------------------------
# ping - true return path
# ---------------------------------------------------------------------------


class TestPingReturnValue:
    def test_ping_returns_true_when_result_is_true(self, redis_service):
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        redis_service._client = mock_client
        redis_service._is_connected = True
        assert redis_service.ping() is True

    def test_ping_returns_false_when_result_is_not_true(self, redis_service):
        mock_client = MagicMock()
        mock_client.ping.return_value = "PONG"  # not exactly True
        redis_service._client = mock_client
        redis_service._is_connected = True
        # "PONG" is not True -> returns False
        result = redis_service.ping()
        assert result is False


# ---------------------------------------------------------------------------
# get - JSON decode failure path
# ---------------------------------------------------------------------------


class TestGetJsonDecodeFail:
    def test_get_invalid_json_string_starting_with_brace(self, redis_service):
        """get returns the raw string when JSON decode fails (lines 142-143)."""
        mock_client = MagicMock()
        mock_client.get.return_value = "{not valid json"
        redis_service._client = mock_client
        redis_service._is_connected = True

        result = redis_service.get("key1")
        assert result == "{not valid json"

    def test_get_value_starting_with_bracket(self, redis_service):
        """get tries JSON decode for values starting with [."""
        mock_client = MagicMock()
        mock_client.get.return_value = "[invalid"
        redis_service._client = mock_client
        redis_service._is_connected = True

        result = redis_service.get("key1")
        assert result == "[invalid"


# ---------------------------------------------------------------------------
# get - operator precedence fix
# ---------------------------------------------------------------------------


class TestGetOperatorPrecedenceFix:
    """Verify the operator-precedence fix on the JSON-deserialization guard."""

    def test_bytes_value_does_not_trigger_json_decode(self, redis_service):
        """Non-string values (bytes) should be returned as-is, not passed to json.loads."""
        mock_client = MagicMock()
        mock_client.get.return_value = b"[1, 2, 3]"
        redis_service._client = mock_client
        redis_service._is_connected = True

        result = redis_service.get("key1")
        # bytes value should be returned as-is (not deserialized)
        assert result == b"[1, 2, 3]"

    def test_string_list_value_is_deserialized(self, redis_service):
        """String values starting with '[' should be JSON-deserialized."""
        mock_client = MagicMock()
        mock_client.get.return_value = "[1, 2, 3]"
        redis_service._client = mock_client
        redis_service._is_connected = True

        result = redis_service.get("key1")
        assert result == [1, 2, 3]

    def test_string_dict_value_is_deserialized(self, redis_service):
        """String values starting with '{' should be JSON-deserialized."""
        mock_client = MagicMock()
        mock_client.get.return_value = '{"key": "val"}'
        redis_service._client = mock_client
        redis_service._is_connected = True

        result = redis_service.get("key1")
        assert result == {"key": "val"}


# ---------------------------------------------------------------------------
# hget - JSON decode failure (lines 301-302)
# ---------------------------------------------------------------------------


class TestHgetJsonDecode:
    def test_hget_invalid_json(self, redis_service):
        mock_client = MagicMock()
        mock_client.hget.return_value = "{broken json"
        redis_service._client = mock_client
        redis_service._is_connected = True

        result = redis_service.hget("myhash", "field1")
        assert result == "{broken json"


# ---------------------------------------------------------------------------
# hgetall - JSON decode failure in values (lines 345-346)
# ---------------------------------------------------------------------------


class TestHgetallJsonDecode:
    def test_hgetall_mixed_values(self, redis_service):
        mock_client = MagicMock()
        mock_client.hgetall.return_value = {
            "good": '{"a": 1}',
            "bad": "{not json",
            "plain": "hello",
        }
        redis_service._client = mock_client
        redis_service._is_connected = True

        result = redis_service.hgetall("myhash")
        assert result["good"] == {"a": 1}
        assert result["bad"] == "{not json"
        assert result["plain"] == "hello"


# ---------------------------------------------------------------------------
# lrange - JSON decode path (lines 435-436)
# ---------------------------------------------------------------------------


class TestLrangeJsonDecode:
    def test_lrange_mixed_values(self, redis_service):
        mock_client = MagicMock()
        mock_client.lrange.return_value = [
            '{"x": 1}',
            "[1, 2]",
            "{bad",
            "plain",
        ]
        redis_service._client = mock_client
        redis_service._is_connected = True

        result = redis_service.lrange("mylist", 0, -1)
        assert result[0] == {"x": 1}
        assert result[1] == [1, 2]
        assert result[2] == "{bad"
        assert result[3] == "plain"


# ---------------------------------------------------------------------------
# flushdb (lines 450-456)
# ---------------------------------------------------------------------------


class TestFlushdb:
    def test_flushdb_success(self, redis_service):
        mock_client = MagicMock()
        redis_service._client = mock_client
        redis_service._is_connected = True

        assert redis_service.flushdb() is True
        mock_client.flushdb.assert_called_once()

    def test_flushdb_error(self, redis_service):
        mock_client = MagicMock()
        mock_client.flushdb.side_effect = RedisError("cannot flush")
        redis_service._client = mock_client
        redis_service._is_connected = True

        assert redis_service.flushdb() is False


# ---------------------------------------------------------------------------
# context manager (lines 468, 472)
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_enter_returns_self(self, redis_service):
        assert redis_service.__enter__() is redis_service

    def test_exit_closes(self, redis_service):
        mock_client = MagicMock()
        redis_service._client = mock_client
        redis_service.__exit__(None, None, None)
        mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# async wrappers (lines 513, 525, 551, 562, 573, 585)
# ---------------------------------------------------------------------------


class TestAsyncWrappers:
    @pytest.mark.asyncio
    async def test_async_ping(self, redis_service):
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        redis_service._client = mock_client
        redis_service._is_connected = True
        result = await redis_service.async_ping()
        assert result is True

    @pytest.mark.asyncio
    async def test_async_get(self, redis_service):
        mock_client = MagicMock()
        mock_client.get.return_value = "val"
        redis_service._client = mock_client
        redis_service._is_connected = True
        result = await redis_service.async_get("key")
        assert result == "val"

    @pytest.mark.asyncio
    async def test_async_set(self, redis_service):
        mock_client = MagicMock()
        mock_client.set.return_value = True
        redis_service._client = mock_client
        redis_service._is_connected = True
        result = await redis_service.async_set("key", "val")
        assert result is True

    @pytest.mark.asyncio
    async def test_async_delete(self, redis_service):
        mock_client = MagicMock()
        mock_client.delete.return_value = 1
        redis_service._client = mock_client
        redis_service._is_connected = True
        result = await redis_service.async_delete("key")
        assert result == 1

    @pytest.mark.asyncio
    async def test_async_exists(self, redis_service):
        mock_client = MagicMock()
        mock_client.exists.return_value = 1
        redis_service._client = mock_client
        redis_service._is_connected = True
        result = await redis_service.async_exists("key")
        assert result == 1

    @pytest.mark.asyncio
    async def test_async_scan_keys(self, redis_service):
        mock_client = MagicMock()
        mock_client.scan.return_value = (0, ["key1", "key2"])
        redis_service._client = mock_client
        redis_service._is_connected = True
        result = await redis_service.async_scan_keys("key*")
        assert result == ["key1", "key2"]


# ---------------------------------------------------------------------------
# init_redis_service / RedisServiceContainer / get_redis_service (lines 602-603, 618-621, 626, 638, 648-650)
# ---------------------------------------------------------------------------


class TestRedisServiceFactory:
    def test_init_redis_service(self):
        import src.core.redis_service as mod

        with (
            patch("src.core.redis_service.get_settings") as mock_settings,
            patch("src.core.redis_service.ConnectionPool"),
        ):
            settings = MagicMock()
            settings.redis.host = "h"
            settings.redis.port = 6379
            settings.redis.db = 0
            settings.redis.password = None
            mock_settings.return_value = settings

            svc = mod.init_redis_service()
            assert svc is not None
            assert mod.redis_service is svc

        mod.redis_service = None  # cleanup

    @pytest.mark.asyncio
    async def test_redis_service_container_get_service(self):
        from src.core.redis_service import RedisServiceContainer

        RedisServiceContainer.reset()

        with (
            patch("src.core.redis_service.get_settings") as mock_settings,
            patch("src.core.redis_service.ConnectionPool"),
        ):
            settings = MagicMock()
            settings.redis.host = "h"
            settings.redis.port = 6379
            settings.redis.db = 0
            settings.redis.password = None
            mock_settings.return_value = settings

            svc = await RedisServiceContainer.get_service()
            assert svc is not None

        RedisServiceContainer.reset()

    @pytest.mark.asyncio
    async def test_get_redis_service_async(self):
        from src.core.redis_service import RedisServiceContainer, get_redis_service

        RedisServiceContainer.reset()

        with (
            patch("src.core.redis_service.get_settings") as mock_settings,
            patch("src.core.redis_service.ConnectionPool"),
        ):
            settings = MagicMock()
            settings.redis.host = "h"
            settings.redis.port = 6379
            settings.redis.db = 0
            settings.redis.password = None
            mock_settings.return_value = settings

            svc = await get_redis_service()
            assert svc is not None

        RedisServiceContainer.reset()

    def test_get_redis_service_sync(self):
        import src.core.redis_service as mod

        mod.redis_service = None

        with (
            patch("src.core.redis_service.get_settings") as mock_settings,
            patch("src.core.redis_service.ConnectionPool"),
        ):
            settings = MagicMock()
            settings.redis.host = "h"
            settings.redis.port = 6379
            settings.redis.db = 0
            settings.redis.password = None
            mock_settings.return_value = settings

            svc = mod.get_redis_service_sync()
            assert svc is not None
            assert mod.redis_service is svc

        mod.redis_service = None

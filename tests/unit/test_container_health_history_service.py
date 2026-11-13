"""Tests for services/container_health_history_service.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from redis.exceptions import RedisError

from src.services.container_health_history_service import (
    ContainerHealthHistoryService,
    HealthCheckRecord,
    get_health_history_service,
)

# ===========================================================================
# HealthCheckRecord
# ===========================================================================


class TestHealthCheckRecord:
    def test_to_dict(self):
        record = HealthCheckRecord(
            timestamp="2026-01-01T00:00:00+00:00",
            status="running",
            healthy=True,
            issues=[],
            metadata={"cpu": 0.5},
        )
        d = record.to_dict()
        assert d["timestamp"] == "2026-01-01T00:00:00+00:00"
        assert d["status"] == "running"
        assert d["healthy"] is True
        assert d["issues"] == []
        assert d["metadata"] == {"cpu": 0.5}

    def test_to_dict_default_metadata(self):
        record = HealthCheckRecord(
            timestamp="2026-01-01T00:00:00+00:00",
            status="running",
            healthy=True,
            issues=["issue1"],
        )
        d = record.to_dict()
        assert d["metadata"] == {}

    def test_from_dict(self):
        data = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "status": "error",
            "healthy": False,
            "issues": ["disk full"],
            "metadata": {"disk_pct": 99},
        }
        record = HealthCheckRecord.from_dict(data)
        assert record.timestamp == "2026-01-01T00:00:00+00:00"
        assert record.status == "error"
        assert record.healthy is False
        assert record.issues == ["disk full"]
        assert record.metadata == {"disk_pct": 99}

    def test_from_dict_missing_optional_fields(self):
        data = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "status": "running",
            "healthy": True,
        }
        record = HealthCheckRecord.from_dict(data)
        assert record.issues == []
        assert record.metadata == {}

    def test_roundtrip(self):
        record = HealthCheckRecord(
            timestamp="2026-03-13T12:00:00+00:00",
            status="running",
            healthy=True,
            issues=["warning1"],
            metadata={"response_ms": 42},
        )
        d = record.to_dict()
        restored = HealthCheckRecord.from_dict(d)
        assert restored.to_dict() == d


# ===========================================================================
# ContainerHealthHistoryService._get_history_key
# ===========================================================================


class TestGetHistoryKey:
    def test_key_format(self):
        svc = ContainerHealthHistoryService()
        assert svc._get_history_key("my-ctr") == "container:my-ctr:health:history"


# ===========================================================================
# ContainerHealthHistoryService.record_health_check
# ===========================================================================


@pytest.mark.asyncio
class TestRecordHealthCheck:
    def _make_service(self, mock_redis=None):
        svc = ContainerHealthHistoryService(history_limit=10)
        if mock_redis is None:
            mock_redis = MagicMock()
            mock_redis.lpush.return_value = 1
            mock_redis.client.ltrim.return_value = True
            mock_redis.expire.return_value = True
        svc._redis = mock_redis
        return svc

    async def test_record_success(self):
        svc = self._make_service()
        result = await svc.record_health_check("ctr", "running", True, [])
        assert result is True
        svc._redis.lpush.assert_called_once()
        svc._redis.client.ltrim.assert_called_once()
        svc._redis.expire.assert_called_once()

    async def test_record_with_issues_and_metadata(self):
        svc = self._make_service()
        result = await svc.record_health_check(
            "ctr", "error", False, ["disk full"], metadata={"disk_pct": 99}
        )
        assert result is True
        # Verify the JSON payload contains the issues
        call_args = svc._redis.lpush.call_args
        json_str = call_args[0][1]
        parsed = json.loads(json_str)
        assert parsed["status"] == "error"
        assert parsed["healthy"] is False
        assert "disk full" in parsed["issues"]
        assert parsed["metadata"]["disk_pct"] == 99

    async def test_empty_container_name_returns_false(self):
        svc = self._make_service()
        result = await svc.record_health_check("", "running", True, [])
        assert result is False

    async def test_lpush_returns_none(self):
        mock_redis = MagicMock()
        mock_redis.lpush.return_value = None
        svc = self._make_service(mock_redis)
        result = await svc.record_health_check("ctr", "running", True, [])
        assert result is False

    async def test_exception_returns_false(self):
        mock_redis = MagicMock()
        mock_redis.lpush.side_effect = RedisError("redis down")
        svc = self._make_service(mock_redis)
        result = await svc.record_health_check("ctr", "running", True, [])
        assert result is False

    async def test_history_limit_respected(self):
        svc = self._make_service()
        svc.history_limit = 50
        await svc.record_health_check("ctr", "running", True, [])
        ltrim_call = svc._redis.client.ltrim.call_args
        assert ltrim_call[0][2] == 49  # 0 to limit-1


# ===========================================================================
# ContainerHealthHistoryService.get_health_history
# ===========================================================================


@pytest.mark.asyncio
class TestGetHealthHistory:
    def _make_service(self, mock_redis=None):
        svc = ContainerHealthHistoryService()
        if mock_redis is None:
            mock_redis = MagicMock()
        svc._redis = mock_redis
        return svc

    async def test_empty_container_name(self):
        svc = self._make_service()
        result = await svc.get_health_history("")
        assert result == []

    async def test_no_history_key(self):
        mock_redis = MagicMock()
        mock_redis.exists.return_value = False
        svc = self._make_service(mock_redis)
        result = await svc.get_health_history("ctr")
        assert result == []

    async def test_returns_parsed_records(self):
        record_data = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "status": "running",
            "healthy": True,
            "issues": [],
        }
        mock_redis = MagicMock()
        mock_redis.exists.return_value = True
        mock_redis.lrange.return_value = [json.dumps(record_data)]
        svc = self._make_service(mock_redis)

        result = await svc.get_health_history("ctr", limit=10)
        assert len(result) == 1
        assert result[0]["status"] == "running"

    async def test_handles_dict_entries(self):
        """When Redis returns already-parsed dicts."""
        record_data = {"timestamp": "2026-01-01T00:00:00", "status": "running", "healthy": True}
        mock_redis = MagicMock()
        mock_redis.exists.return_value = True
        mock_redis.lrange.return_value = [record_data]
        svc = self._make_service(mock_redis)

        result = await svc.get_health_history("ctr")
        assert len(result) == 1

    async def test_skips_invalid_json(self):
        mock_redis = MagicMock()
        mock_redis.exists.return_value = True
        mock_redis.lrange.return_value = ["not valid json{{{"]
        svc = self._make_service(mock_redis)

        result = await svc.get_health_history("ctr")
        assert result == []

    async def test_skips_non_string_non_dict(self):
        mock_redis = MagicMock()
        mock_redis.exists.return_value = True
        mock_redis.lrange.return_value = [42, None]
        svc = self._make_service(mock_redis)

        result = await svc.get_health_history("ctr")
        assert result == []

    async def test_empty_raw_records(self):
        mock_redis = MagicMock()
        mock_redis.exists.return_value = True
        mock_redis.lrange.return_value = []
        svc = self._make_service(mock_redis)

        result = await svc.get_health_history("ctr")
        assert result == []

    async def test_pagination(self):
        mock_redis = MagicMock()
        mock_redis.exists.return_value = True
        mock_redis.lrange.return_value = []
        svc = self._make_service(mock_redis)

        await svc.get_health_history("ctr", limit=5, offset=10)
        mock_redis.lrange.assert_called_once_with("container:ctr:health:history", 10, 14)

    async def test_default_limit(self):
        mock_redis = MagicMock()
        mock_redis.exists.return_value = True
        mock_redis.lrange.return_value = []
        svc = self._make_service(mock_redis)

        await svc.get_health_history("ctr")
        call_args = mock_redis.lrange.call_args
        # Default limit is DEFAULT_QUERY_LIMIT (50), so stop = 0 + 50 - 1 = 49
        assert call_args[0][2] == 49

    async def test_exception_returns_empty(self):
        mock_redis = MagicMock()
        mock_redis.exists.side_effect = RedisError("redis down")
        svc = self._make_service(mock_redis)

        result = await svc.get_health_history("ctr")
        assert result == []


# ===========================================================================
# ContainerHealthHistoryService.get_history_count
# ===========================================================================


@pytest.mark.asyncio
class TestGetHistoryCount:
    async def test_returns_count(self):
        svc = ContainerHealthHistoryService()
        mock_redis = MagicMock()
        mock_redis.client.llen.return_value = 42
        svc._redis = mock_redis

        count = await svc.get_history_count("ctr")
        assert count == 42

    async def test_empty_name_returns_zero(self):
        svc = ContainerHealthHistoryService()
        count = await svc.get_history_count("")
        assert count == 0

    async def test_none_count(self):
        svc = ContainerHealthHistoryService()
        mock_redis = MagicMock()
        mock_redis.client.llen.return_value = None
        svc._redis = mock_redis

        count = await svc.get_history_count("ctr")
        assert count == 0

    async def test_exception_returns_zero(self):
        svc = ContainerHealthHistoryService()
        mock_redis = MagicMock()
        mock_redis.client.llen.side_effect = RedisError("redis down")
        svc._redis = mock_redis

        count = await svc.get_history_count("ctr")
        assert count == 0


# ===========================================================================
# ContainerHealthHistoryService.get_latest_health_check
# ===========================================================================


@pytest.mark.asyncio
class TestGetLatestHealthCheck:
    async def test_returns_latest(self):
        svc = ContainerHealthHistoryService()
        mock_redis = MagicMock()
        mock_redis.exists.return_value = True
        record = {"timestamp": "2026-01-01T00:00:00", "status": "running", "healthy": True}
        mock_redis.lrange.return_value = [json.dumps(record)]
        svc._redis = mock_redis

        result = await svc.get_latest_health_check("ctr")
        assert result is not None
        assert result["status"] == "running"

    async def test_returns_none_when_empty(self):
        svc = ContainerHealthHistoryService()
        mock_redis = MagicMock()
        mock_redis.exists.return_value = False
        svc._redis = mock_redis

        result = await svc.get_latest_health_check("ctr")
        assert result is None


# ===========================================================================
# ContainerHealthHistoryService.clear_history
# ===========================================================================


@pytest.mark.asyncio
class TestClearHistory:
    async def test_clears_existing(self):
        svc = ContainerHealthHistoryService()
        mock_redis = MagicMock()
        mock_redis.delete.return_value = 1
        svc._redis = mock_redis

        result = await svc.clear_history("ctr")
        assert result is True

    async def test_clears_nonexistent(self):
        svc = ContainerHealthHistoryService()
        mock_redis = MagicMock()
        mock_redis.delete.return_value = 0
        svc._redis = mock_redis

        result = await svc.clear_history("ctr")
        assert result is True

    async def test_empty_name_returns_false(self):
        svc = ContainerHealthHistoryService()
        result = await svc.clear_history("")
        assert result is False

    async def test_exception_returns_false(self):
        svc = ContainerHealthHistoryService()
        mock_redis = MagicMock()
        mock_redis.delete.side_effect = RedisError("redis down")
        svc._redis = mock_redis

        result = await svc.clear_history("ctr")
        assert result is False


# ===========================================================================
# ContainerHealthHistoryService.get_health_statistics
# ===========================================================================


@pytest.mark.asyncio
class TestGetHealthStatistics:
    def _make_service_with_history(self, history):
        svc = ContainerHealthHistoryService()
        mock_redis = MagicMock()
        mock_redis.exists.return_value = True
        mock_redis.lrange.return_value = [json.dumps(h) for h in history]
        svc._redis = mock_redis
        return svc

    async def test_empty_history(self):
        svc = ContainerHealthHistoryService()
        mock_redis = MagicMock()
        mock_redis.exists.return_value = False
        svc._redis = mock_redis

        stats = await svc.get_health_statistics("ctr")
        assert stats["total_checks"] == 0
        assert stats["health_percentage"] == 0.0
        assert stats["common_issues"] == []

    async def test_all_healthy(self):
        history = [
            {
                "timestamp": f"2026-01-0{i}T00:00:00",
                "status": "running",
                "healthy": True,
                "issues": [],
            }
            for i in range(1, 4)
        ]
        svc = self._make_service_with_history(history)

        stats = await svc.get_health_statistics("ctr")
        assert stats["total_checks"] == 3
        assert stats["healthy_count"] == 3
        assert stats["unhealthy_count"] == 0
        assert stats["health_percentage"] == 100.0

    async def test_mixed_health(self):
        history = [
            {
                "timestamp": "2026-01-01T00:00:00",
                "status": "running",
                "healthy": True,
                "issues": [],
            },
            {
                "timestamp": "2026-01-02T00:00:00",
                "status": "error",
                "healthy": False,
                "issues": ["disk full"],
            },
            {
                "timestamp": "2026-01-03T00:00:00",
                "status": "error",
                "healthy": False,
                "issues": ["disk full", "OOM"],
            },
        ]
        svc = self._make_service_with_history(history)

        stats = await svc.get_health_statistics("ctr")
        assert stats["total_checks"] == 3
        assert stats["healthy_count"] == 1
        assert stats["unhealthy_count"] == 2
        assert stats["health_percentage"] == pytest.approx(33.333, rel=0.01)
        # "disk full" should appear twice, "OOM" once
        issues = {i["issue"]: i["count"] for i in stats["common_issues"]}
        assert issues["disk full"] == 2
        assert issues["OOM"] == 1

    async def test_first_and_last_check_timestamps(self):
        history = [
            {
                "timestamp": "2026-01-03T00:00:00",
                "status": "running",
                "healthy": True,
                "issues": [],
            },
            {
                "timestamp": "2026-01-02T00:00:00",
                "status": "running",
                "healthy": True,
                "issues": [],
            },
            {
                "timestamp": "2026-01-01T00:00:00",
                "status": "running",
                "healthy": True,
                "issues": [],
            },
        ]
        svc = self._make_service_with_history(history)

        stats = await svc.get_health_statistics("ctr")
        assert stats["last_check"] == "2026-01-03T00:00:00"
        assert stats["first_check"] == "2026-01-01T00:00:00"


# ===========================================================================
# get_health_history_service (singleton)
# ===========================================================================


class TestGetHealthHistoryService:
    def test_returns_instance(self):
        import src.services.container_health_history_service as mod

        mod._health_history_service = None
        svc = get_health_history_service()
        assert isinstance(svc, ContainerHealthHistoryService)
        # Reset
        mod._health_history_service = None

    def test_returns_same_instance(self):
        import src.services.container_health_history_service as mod

        mod._health_history_service = None
        svc1 = get_health_history_service()
        svc2 = get_health_history_service()
        assert svc1 is svc2
        mod._health_history_service = None


# ===========================================================================
# Lazy Redis property
# ===========================================================================


class TestRedisLazyProperty:
    @patch("src.services.container_health_history_service.get_redis_service_sync")
    def test_lazy_loads_redis(self, mock_redis_fn):
        mock_redis = MagicMock()
        mock_redis_fn.return_value = mock_redis

        svc = ContainerHealthHistoryService()
        assert svc._redis is None
        r = svc.redis
        assert r is mock_redis
        mock_redis_fn.assert_called_once()

    def test_uses_cached_redis(self):
        svc = ContainerHealthHistoryService()
        mock_redis = MagicMock()
        svc._redis = mock_redis
        assert svc.redis is mock_redis

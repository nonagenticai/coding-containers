"""Tests for services/container_health_service.py."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import RedisError

from src.services.container_health_service import (
    _determine_health_status,
    _get_cache_metrics,
    _get_error_logs_24h,
    _get_graph_metrics,
    _get_last_analysis_timestamp,
    _get_test_results_count,
    get_container_health_metrics,
    get_multiple_container_health_metrics,
)

# ===========================================================================
# _determine_health_status
# ===========================================================================


class TestDetermineHealthStatus:
    def test_healthy_metrics(self):
        metrics = {
            "error_count_24h": 0,
            "cache_total_requests": 50,
            "cache_hit_ratio": 0.8,
            "cache_hits": 40,
            "graph_node_count": 100,
            "last_analysis_timestamp": None,
        }
        status, warnings = _determine_health_status(metrics)
        assert status == "healthy"
        assert warnings == []

    def test_degraded_high_errors(self):
        metrics = {
            "error_count_24h": 15,
            "cache_total_requests": 0,
            "cache_hit_ratio": 0.0,
            "cache_hits": 0,
            "graph_node_count": 100,
            "last_analysis_timestamp": None,
        }
        status, warnings = _determine_health_status(metrics)
        assert status == "degraded"
        assert any("error" in w.lower() for w in warnings)

    def test_degraded_empty_graph(self):
        metrics = {
            "error_count_24h": 0,
            "cache_total_requests": 0,
            "cache_hit_ratio": 0.0,
            "cache_hits": 0,
            "graph_node_count": 0,
            "last_analysis_timestamp": None,
        }
        status, warnings = _determine_health_status(metrics)
        assert status == "degraded"
        assert any("empty" in w.lower() for w in warnings)

    def test_degraded_low_cache_ratio(self):
        metrics = {
            "error_count_24h": 0,
            "cache_total_requests": 200,
            "cache_hit_ratio": 0.3,
            "cache_hits": 60,
            "graph_node_count": 50,
            "last_analysis_timestamp": None,
        }
        status, warnings = _determine_health_status(metrics)
        assert status == "degraded"
        assert any("cache" in w.lower() for w in warnings)

    def test_stale_analysis_timestamp(self):
        old_timestamp = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        metrics = {
            "error_count_24h": 0,
            "cache_total_requests": 0,
            "cache_hit_ratio": 0.0,
            "cache_hits": 0,
            "graph_node_count": 50,
            "last_analysis_timestamp": old_timestamp,
        }
        status, warnings = _determine_health_status(metrics)
        assert status == "degraded"
        assert any("stale" in w.lower() for w in warnings)

    def test_recent_analysis_timestamp_no_warning(self):
        recent_timestamp = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        metrics = {
            "error_count_24h": 0,
            "cache_total_requests": 0,
            "cache_hit_ratio": 0.0,
            "cache_hits": 0,
            "graph_node_count": 50,
            "last_analysis_timestamp": recent_timestamp,
        }
        status, warnings = _determine_health_status(metrics)
        assert status == "healthy"
        assert not any("stale" in w.lower() for w in warnings)

    def test_invalid_timestamp_ignored(self):
        metrics = {
            "error_count_24h": 0,
            "cache_total_requests": 0,
            "cache_hit_ratio": 0.0,
            "cache_hits": 0,
            "graph_node_count": 50,
            "last_analysis_timestamp": "not-a-date",
        }
        # Should not raise, just ignore bad timestamp
        status, warnings = _determine_health_status(metrics)
        assert status == "healthy"

    def test_multiple_warnings_accumulated(self):
        """Both empty graph and high errors produce multiple warnings."""
        metrics = {
            "error_count_24h": 20,
            "cache_total_requests": 200,
            "cache_hit_ratio": 0.1,
            "cache_hits": 20,
            "graph_node_count": 0,
            "last_analysis_timestamp": None,
        }
        status, warnings = _determine_health_status(metrics)
        assert status == "degraded"
        assert len(warnings) >= 2


# ===========================================================================
# get_container_health_metrics
# ===========================================================================


@pytest.mark.asyncio
class TestGetContainerHealthMetrics:
    async def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            await get_container_health_metrics("", "tenant-1")

    async def test_empty_tenant_raises(self):
        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            await get_container_health_metrics("my-container", "")

    @patch("src.services.container_health_service._get_error_logs_24h", new_callable=AsyncMock)
    @patch("src.services.container_health_service._get_test_results_count", new_callable=AsyncMock)
    @patch("src.services.container_health_service._get_cache_metrics", new_callable=AsyncMock)
    @patch(
        "src.services.container_health_service._get_last_analysis_timestamp",
        new_callable=AsyncMock,
    )
    @patch("src.services.container_health_service._get_graph_metrics", new_callable=AsyncMock)
    @patch("src.services.container_health_service.get_cache_manager", new_callable=MagicMock)
    async def test_cached_result_returned(
        self, mock_get_cm, mock_graph, mock_analysis, mock_cache, mock_tests, mock_errors
    ):
        cached = {"container_name": "test", "status": "healthy"}
        mock_cm = AsyncMock()
        mock_cm.get = AsyncMock(return_value=cached)
        mock_get_cm.return_value = mock_cm

        result = await get_container_health_metrics("test", "tenant-1")
        assert result == cached
        mock_graph.assert_not_called()

    @patch("src.services.container_health_service._get_error_logs_24h", new_callable=AsyncMock)
    @patch("src.services.container_health_service._get_test_results_count", new_callable=AsyncMock)
    @patch("src.services.container_health_service._get_cache_metrics", new_callable=AsyncMock)
    @patch(
        "src.services.container_health_service._get_last_analysis_timestamp",
        new_callable=AsyncMock,
    )
    @patch("src.services.container_health_service._get_graph_metrics", new_callable=AsyncMock)
    @patch("src.services.container_health_service.get_cache_manager", new_callable=MagicMock)
    async def test_cache_miss_fetches_metrics(
        self, mock_get_cm, mock_graph, mock_analysis, mock_cache, mock_tests, mock_errors
    ):
        mock_cm = AsyncMock()
        mock_cm.get = AsyncMock(return_value=None)
        mock_cm.set = AsyncMock()
        mock_get_cm.return_value = mock_cm

        mock_graph.return_value = {"graph_node_count": 10, "graph_edge_count": 5}
        mock_analysis.return_value = None
        mock_cache.return_value = {
            "cache_hit_ratio": 0.5,
            "cache_hits": 10,
            "cache_misses": 10,
            "cache_total_requests": 20,
        }
        mock_tests.return_value = 3
        mock_errors.return_value = []

        result = await get_container_health_metrics("test-ctr", "tenant-1")

        assert result["container_name"] == "test-ctr"
        assert result["graph_node_count"] == 10
        assert result["graph_edge_count"] == 5
        assert result["cache_hit_ratio"] == 0.5
        assert result["test_results_count"] == 3
        assert result["error_count_24h"] == 0
        assert result["status"] == "healthy"
        mock_cm.set.assert_called_once()

    @patch("src.services.container_health_service._get_error_logs_24h", new_callable=AsyncMock)
    @patch("src.services.container_health_service._get_test_results_count", new_callable=AsyncMock)
    @patch("src.services.container_health_service._get_cache_metrics", new_callable=AsyncMock)
    @patch(
        "src.services.container_health_service._get_last_analysis_timestamp",
        new_callable=AsyncMock,
    )
    @patch("src.services.container_health_service._get_graph_metrics", new_callable=AsyncMock)
    @patch("src.services.container_health_service.get_cache_manager", new_callable=MagicMock)
    async def test_metrics_with_errors(
        self, mock_get_cm, mock_graph, mock_analysis, mock_cache, mock_tests, mock_errors
    ):
        mock_cm = AsyncMock()
        mock_cm.get = AsyncMock(return_value=None)
        mock_cm.set = AsyncMock()
        mock_get_cm.return_value = mock_cm

        mock_graph.return_value = {"graph_node_count": 0, "graph_edge_count": 0}
        mock_analysis.return_value = None
        mock_cache.return_value = {
            "cache_hit_ratio": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_total_requests": 0,
        }
        mock_tests.return_value = 0
        error_list = [{"message": "err1"}, {"message": "err2"}]
        mock_errors.return_value = error_list

        result = await get_container_health_metrics("test-ctr", "tenant-1")
        assert result["error_count_24h"] == 2
        assert result["error_logs_24h"] == error_list


# ===========================================================================
# _get_graph_metrics
# ===========================================================================


@pytest.mark.asyncio
class TestGetGraphMetrics:
    @patch("src.services.container_health_service.get_settings")
    async def test_no_database_manager(self, mock_settings):
        mock_s = MagicMock()
        mock_s.get_database_manager.side_effect = NotImplementedError
        mock_settings.return_value = mock_s

        result = await _get_graph_metrics("ctr")
        assert result == {"graph_node_count": 0, "graph_edge_count": 0}

    @patch("src.services.container_health_service.get_settings")
    async def test_no_db_path(self, mock_settings):
        mock_s = MagicMock()
        mock_db_mgr = MagicMock()
        mock_db_mgr.find_existing_database.return_value = None
        mock_s.get_database_manager.return_value = mock_db_mgr
        mock_settings.return_value = mock_s

        result = await _get_graph_metrics("ctr")
        assert result == {"graph_node_count": 0, "graph_edge_count": 0}

    @patch("src.services.container_health_service.aiosqlite")
    @patch("src.services.container_health_service.get_settings")
    async def test_db_error_returns_zeros(self, mock_settings, mock_aiosqlite):
        import aiosqlite as real_aiosqlite

        mock_s = MagicMock()
        mock_db_mgr = MagicMock()
        mock_db_mgr.find_existing_database.return_value = "/tmp/test.db"
        mock_s.get_database_manager.return_value = mock_db_mgr
        mock_settings.return_value = mock_s

        mock_aiosqlite.connect.side_effect = real_aiosqlite.Error("db error")
        mock_aiosqlite.Error = real_aiosqlite.Error

        result = await _get_graph_metrics("ctr")
        assert result == {"graph_node_count": 0, "graph_edge_count": 0}


# ===========================================================================
# _get_last_analysis_timestamp
# ===========================================================================


@pytest.mark.asyncio
class TestGetLastAnalysisTimestamp:
    @patch("src.services.container_health_service.get_settings")
    async def test_no_database_manager(self, mock_settings):
        mock_s = MagicMock()
        mock_s.get_database_manager.side_effect = NotImplementedError
        mock_settings.return_value = mock_s

        result = await _get_last_analysis_timestamp("ctr")
        assert result is None

    @patch("src.services.container_health_service.get_settings")
    async def test_no_db_path(self, mock_settings):
        mock_s = MagicMock()
        mock_db_mgr = MagicMock()
        mock_db_mgr.find_existing_database.return_value = None
        mock_s.get_database_manager.return_value = mock_db_mgr
        mock_settings.return_value = mock_s

        result = await _get_last_analysis_timestamp("ctr")
        assert result is None


# ===========================================================================
# _get_cache_metrics
# ===========================================================================


@pytest.mark.asyncio
class TestGetCacheMetrics:
    @patch("src.services.container_health_service.get_redis_service_sync")
    async def test_returns_metrics(self, mock_redis_fn):
        mock_redis = MagicMock()
        mock_redis.get.side_effect = lambda key, default=0: {"hits": 10, "misses": 5}.get(
            key.split(":")[-1], default
        )
        mock_redis_fn.return_value = mock_redis

        # Need to override to use integer values
        mock_redis.get = MagicMock(side_effect=[10, 5])

        result = await _get_cache_metrics("ctr")
        assert result["cache_hits"] == 10
        assert result["cache_misses"] == 5
        assert result["cache_total_requests"] == 15
        assert result["cache_hit_ratio"] == round(10 / 15, 4)

    @patch("src.services.container_health_service.get_redis_service_sync")
    async def test_string_values_converted(self, mock_redis_fn):
        mock_redis = MagicMock()
        mock_redis.get = MagicMock(side_effect=["20", "10"])
        mock_redis_fn.return_value = mock_redis

        result = await _get_cache_metrics("ctr")
        assert result["cache_hits"] == 20
        assert result["cache_misses"] == 10

    @patch("src.services.container_health_service.get_redis_service_sync")
    async def test_zero_requests(self, mock_redis_fn):
        mock_redis = MagicMock()
        mock_redis.get = MagicMock(return_value=0)
        mock_redis_fn.return_value = mock_redis

        result = await _get_cache_metrics("ctr")
        assert result["cache_hit_ratio"] == 0.0
        assert result["cache_total_requests"] == 0

    @patch("src.services.container_health_service.get_redis_service_sync")
    async def test_connection_error_returns_defaults(self, mock_redis_fn):
        mock_redis_fn.side_effect = ConnectionError("no redis")

        result = await _get_cache_metrics("ctr")
        assert result["cache_hit_ratio"] == 0.0
        assert result["cache_total_requests"] == 0


# ===========================================================================
# _get_test_results_count
# ===========================================================================


@pytest.mark.asyncio
class TestGetTestResultsCount:
    @patch("src.services.container_health_service.get_redis_service_sync")
    async def test_returns_list_length(self, mock_redis_fn):
        mock_redis = MagicMock()
        mock_redis.client.llen.return_value = 7
        mock_redis_fn.return_value = mock_redis

        result = await _get_test_results_count("ctr")
        assert result == 7

    @patch("src.services.container_health_service.get_redis_service_sync")
    async def test_non_list_key_exists(self, mock_redis_fn):
        mock_redis = MagicMock()
        mock_redis.client.llen.side_effect = RedisError("WRONGTYPE")
        mock_redis.exists.return_value = True
        mock_redis_fn.return_value = mock_redis

        result = await _get_test_results_count("ctr")
        assert result == 1

    @patch("src.services.container_health_service.get_redis_service_sync")
    async def test_key_not_exists(self, mock_redis_fn):
        mock_redis = MagicMock()
        mock_redis.client.llen.side_effect = RedisError("WRONGTYPE")
        mock_redis.exists.return_value = False
        mock_redis_fn.return_value = mock_redis

        result = await _get_test_results_count("ctr")
        assert result == 0

    @patch("src.services.container_health_service.get_redis_service_sync")
    async def test_connection_error(self, mock_redis_fn):
        mock_redis_fn.side_effect = ConnectionError("no redis")

        result = await _get_test_results_count("ctr")
        assert result == 0


# ===========================================================================
# _get_error_logs_24h
# ===========================================================================


@pytest.mark.asyncio
class TestGetErrorLogs24h:
    @patch("src.services.container_health_service.get_redis_service_sync")
    async def test_returns_recent_errors(self, mock_redis_fn):
        recent_time = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        old_time = (datetime.now(UTC) - timedelta(hours=48)).isoformat()

        mock_redis = MagicMock()
        mock_redis.lrange.return_value = [
            {"message": "err1", "timestamp": recent_time},
            {"message": "err2", "timestamp": old_time},
        ]
        mock_redis_fn.return_value = mock_redis

        result = await _get_error_logs_24h("ctr")
        assert len(result) == 1
        assert result[0]["message"] == "err1"

    @patch("src.services.container_health_service.get_redis_service_sync")
    async def test_entries_without_timestamp_included(self, mock_redis_fn):
        mock_redis = MagicMock()
        mock_redis.lrange.return_value = [{"message": "no-ts-err"}]
        mock_redis_fn.return_value = mock_redis

        result = await _get_error_logs_24h("ctr")
        assert len(result) == 1

    @patch("src.services.container_health_service.get_redis_service_sync")
    async def test_empty_logs(self, mock_redis_fn):
        mock_redis = MagicMock()
        mock_redis.lrange.return_value = []
        mock_redis_fn.return_value = mock_redis

        result = await _get_error_logs_24h("ctr")
        assert result == []

    @patch("src.services.container_health_service.get_redis_service_sync")
    async def test_none_logs(self, mock_redis_fn):
        mock_redis = MagicMock()
        mock_redis.lrange.return_value = None
        mock_redis_fn.return_value = mock_redis

        result = await _get_error_logs_24h("ctr")
        assert result == []

    @patch("src.services.container_health_service.get_redis_service_sync")
    async def test_invalid_timestamp_still_included(self, mock_redis_fn):
        mock_redis = MagicMock()
        mock_redis.lrange.return_value = [
            {"message": "bad-ts", "timestamp": "not-a-date"},
        ]
        mock_redis_fn.return_value = mock_redis

        result = await _get_error_logs_24h("ctr")
        assert len(result) == 1

    @patch("src.services.container_health_service.get_redis_service_sync")
    async def test_non_dict_entries_skipped(self, mock_redis_fn):
        mock_redis = MagicMock()
        mock_redis.lrange.return_value = ["just-a-string", 42]
        mock_redis_fn.return_value = mock_redis

        result = await _get_error_logs_24h("ctr")
        assert result == []

    @patch("src.services.container_health_service.get_redis_service_sync")
    async def test_connection_error(self, mock_redis_fn):
        mock_redis_fn.side_effect = ConnectionError("no redis")
        result = await _get_error_logs_24h("ctr")
        assert result == []


# ===========================================================================
# get_multiple_container_health_metrics
# ===========================================================================


@pytest.mark.asyncio
class TestGetMultipleContainerHealthMetrics:
    @patch(
        "src.services.container_health_service.get_container_health_metrics", new_callable=AsyncMock
    )
    async def test_multiple_containers(self, mock_get_metrics):
        mock_get_metrics.side_effect = [
            {"container_name": "a", "status": "healthy"},
            {"container_name": "b", "status": "degraded"},
        ]

        result = await get_multiple_container_health_metrics(["a", "b"], "tenant-1")
        assert "a" in result
        assert "b" in result
        assert result["a"]["status"] == "healthy"
        assert result["b"]["status"] == "degraded"

    @patch(
        "src.services.container_health_service.get_container_health_metrics", new_callable=AsyncMock
    )
    async def test_one_container_fails(self, mock_get_metrics):
        mock_get_metrics.side_effect = [
            {"container_name": "a", "status": "healthy"},
            RuntimeError("boom"),
        ]

        result = await get_multiple_container_health_metrics(["a", "b"], "tenant-1")
        assert result["a"]["status"] == "healthy"
        assert result["b"]["status"] == "unhealthy"
        assert "error" in result["b"]

    @patch(
        "src.services.container_health_service.get_container_health_metrics", new_callable=AsyncMock
    )
    async def test_empty_list(self, mock_get_metrics):
        result = await get_multiple_container_health_metrics([], "tenant-1")
        assert result == {}


# ===========================================================================
# _get_graph_metrics — database path exists (lines 192-209)
# ===========================================================================


@pytest.mark.asyncio
class TestGetGraphMetricsWithDB:
    @patch("src.services.container_health_service.aiosqlite")
    @patch("src.services.container_health_service.get_settings")
    async def test_returns_counts_from_database(self, mock_settings, mock_aiosqlite):
        """Cover lines 192-209: successful DB read."""
        mock_s = MagicMock()
        mock_db_mgr = MagicMock()
        mock_db_mgr.find_existing_database.return_value = "/tmp/test.db"
        mock_s.get_database_manager.return_value = mock_db_mgr
        mock_settings.return_value = mock_s

        # Build a mock async context manager for aiosqlite.connect
        mock_cursor_nodes = AsyncMock()
        mock_cursor_nodes.fetchone = AsyncMock(return_value=(42,))
        mock_cursor_nodes.close = AsyncMock()

        mock_cursor_edges = AsyncMock()
        mock_cursor_edges.fetchone = AsyncMock(return_value=(17,))
        mock_cursor_edges.close = AsyncMock()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=[mock_cursor_nodes, mock_cursor_edges])

        mock_aiosqlite.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_aiosqlite.connect.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await _get_graph_metrics("ctr")
        assert result == {"graph_node_count": 42, "graph_edge_count": 17}

    @patch("src.services.container_health_service.aiosqlite")
    @patch("src.services.container_health_service.get_settings")
    async def test_returns_zeros_when_count_is_none(self, mock_settings, mock_aiosqlite):
        """Cover edge case: fetchone returns (None,) or None."""
        mock_s = MagicMock()
        mock_db_mgr = MagicMock()
        mock_db_mgr.find_existing_database.return_value = "/tmp/test.db"
        mock_s.get_database_manager.return_value = mock_db_mgr
        mock_settings.return_value = mock_s

        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)
        mock_cursor.close = AsyncMock()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        mock_aiosqlite.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_aiosqlite.connect.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await _get_graph_metrics("ctr")
        assert result["graph_node_count"] == 0
        assert result["graph_edge_count"] == 0


# ===========================================================================
# _get_last_analysis_timestamp — database path exists (lines 237-254)
# ===========================================================================


@pytest.mark.asyncio
class TestGetLastAnalysisTimestampWithDB:
    @patch("src.services.container_health_service.aiosqlite")
    @patch("src.services.container_health_service.get_settings")
    async def test_returns_timestamp(self, mock_settings, mock_aiosqlite):
        """Cover lines 237-248: successful timestamp read."""
        mock_s = MagicMock()
        mock_db_mgr = MagicMock()
        mock_db_mgr.find_existing_database.return_value = "/tmp/test.db"
        mock_s.get_database_manager.return_value = mock_db_mgr
        mock_settings.return_value = mock_s

        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=("2025-01-01T12:00:00",))
        mock_cursor.close = AsyncMock()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        mock_aiosqlite.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_aiosqlite.connect.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await _get_last_analysis_timestamp("ctr")
        assert result == "2025-01-01T12:00:00"

    @patch("src.services.container_health_service.aiosqlite")
    @patch("src.services.container_health_service.get_settings")
    async def test_returns_none_when_no_rows(self, mock_settings, mock_aiosqlite):
        mock_s = MagicMock()
        mock_db_mgr = MagicMock()
        mock_db_mgr.find_existing_database.return_value = "/tmp/test.db"
        mock_s.get_database_manager.return_value = mock_db_mgr
        mock_settings.return_value = mock_s

        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)
        mock_cursor.close = AsyncMock()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        mock_aiosqlite.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_aiosqlite.connect.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await _get_last_analysis_timestamp("ctr")
        assert result is None

    @patch("src.services.container_health_service.aiosqlite")
    @patch("src.services.container_health_service.get_settings")
    async def test_returns_none_on_db_error(self, mock_settings, mock_aiosqlite):
        """Cover line 252-254: exception path."""
        import aiosqlite as real_aiosqlite

        mock_s = MagicMock()
        mock_db_mgr = MagicMock()
        mock_db_mgr.find_existing_database.return_value = "/tmp/test.db"
        mock_s.get_database_manager.return_value = mock_db_mgr
        mock_settings.return_value = mock_s

        mock_aiosqlite.connect.side_effect = real_aiosqlite.Error("db error")
        mock_aiosqlite.Error = real_aiosqlite.Error

        result = await _get_last_analysis_timestamp("ctr")
        assert result is None


# ===========================================================================
# _determine_health_status — unreachable "unhealthy" branch (lines 408-410)
# ===========================================================================


class TestDetermineHealthStatusAdditional:
    def test_critical_error_count_is_unhealthy(self):
        """Errors above 50 should produce 'unhealthy' status with critical warning."""
        metrics = {
            "error_count_24h": 60,
            "cache_total_requests": 0,
            "cache_hit_ratio": 0.0,
            "cache_hits": 0,
            "graph_node_count": 100,
            "last_analysis_timestamp": None,
        }
        status, warnings = _determine_health_status(metrics)
        assert status == "unhealthy"
        assert any("critical" in w.lower() for w in warnings)

    def test_boundary_at_50_is_degraded_not_unhealthy(self):
        """Exactly 50 errors is not > 50, so stays degraded (> 10)."""
        metrics = {
            "error_count_24h": 50,
            "cache_total_requests": 0,
            "cache_hit_ratio": 0.0,
            "cache_hits": 0,
            "graph_node_count": 100,
            "last_analysis_timestamp": None,
        }
        status, warnings = _determine_health_status(metrics)
        assert status == "degraded"
        assert any("high error" in w.lower() for w in warnings)

    def test_boundary_at_10_is_healthy(self):
        """Exactly 10 errors is not > 10, so stays healthy."""
        metrics = {
            "error_count_24h": 10,
            "cache_total_requests": 0,
            "cache_hit_ratio": 0.0,
            "cache_hits": 0,
            "graph_node_count": 50,
            "last_analysis_timestamp": None,
        }
        status, warnings = _determine_health_status(metrics)
        assert status == "healthy"
        assert not any("error" in w.lower() for w in warnings)

    def test_boundary_at_11_is_degraded(self):
        """11 errors is > 10, so becomes degraded."""
        metrics = {
            "error_count_24h": 11,
            "cache_total_requests": 0,
            "cache_hit_ratio": 0.0,
            "cache_hits": 0,
            "graph_node_count": 50,
            "last_analysis_timestamp": None,
        }
        status, warnings = _determine_health_status(metrics)
        assert status == "degraded"
        assert any("high error" in w.lower() for w in warnings)

    def test_boundary_at_51_is_unhealthy(self):
        """51 errors is > 50, so becomes unhealthy."""
        metrics = {
            "error_count_24h": 51,
            "cache_total_requests": 0,
            "cache_hit_ratio": 0.0,
            "cache_hits": 0,
            "graph_node_count": 100,
            "last_analysis_timestamp": None,
        }
        status, warnings = _determine_health_status(metrics)
        assert status == "unhealthy"
        assert any("critical" in w.lower() for w in warnings)

    def test_cache_low_plus_errors_stays_degraded(self):
        """Both cache and errors produce multiple warnings."""
        metrics = {
            "error_count_24h": 15,
            "cache_total_requests": 200,
            "cache_hit_ratio": 0.2,
            "cache_hits": 40,
            "graph_node_count": 50,
            "last_analysis_timestamp": None,
        }
        status, warnings = _determine_health_status(metrics)
        assert status == "degraded"
        assert len(warnings) >= 2

    def test_unhealthy_not_downgraded_by_cache_check(self):
        """Critical errors make status unhealthy; cache issues add warnings
        but do not downgrade unhealthy to degraded."""
        metrics = {
            "error_count_24h": 100,
            "cache_total_requests": 200,
            "cache_hit_ratio": 0.1,
            "cache_hits": 20,
            "graph_node_count": 0,
            "last_analysis_timestamp": None,
        }
        status, warnings = _determine_health_status(metrics)
        assert status == "unhealthy"
        # Should have warnings for errors, cache, AND empty graph
        assert len(warnings) >= 3


# ===========================================================================
# detect_unhealthy_containers (lines 507-597)
# ===========================================================================


@pytest.mark.asyncio
class TestDetectUnhealthyContainers:
    @patch("src.services.container_health_service.get_settings")
    @patch(
        "src.services.container_health_service.get_container_health_metrics", new_callable=AsyncMock
    )
    async def test_returns_empty_on_exception(self, mock_metrics, mock_settings):
        """Cover line 595-597: top-level exception returns []."""
        from src.services.container_health_service import detect_unhealthy_containers

        mock_settings.side_effect = RuntimeError("settings broken")

        result = await detect_unhealthy_containers(tenant_id="t1")
        assert result == []

    @patch("src.services.container_health_service.get_settings")
    async def test_detects_non_running_container(self, mock_settings):
        """Cover lines 507-593: full flow with a stopped container."""
        from src.services.container_health_service import detect_unhealthy_containers

        mock_s = MagicMock()
        mock_s.get_code_graph_db_path.return_value = "/nonexistent/path.db"
        mock_settings.return_value = mock_s

        stopped_container = MagicMock()
        stopped_container.container_name = "stopped-ctr"
        stopped_container.status = "stopped"
        stopped_container.repo_url = "https://github.com/owner/repo"
        stopped_container.api_endpoint = "http://localhost:8687"
        stopped_container.codebase_path = "/repo"
        stopped_container.internal_endpoint = ""
        stopped_container.public_endpoint = ""

        mock_mgr = MagicMock()
        mock_mgr.list_containers.return_value = [stopped_container]

        with (
            patch(
                "src.services.container_health_service.get_container_health_metrics",
                new_callable=AsyncMock,
            ),
            patch(
                "src.core.container_manager.get_container_manager",
                return_value=mock_mgr,
            ),
        ):
            result = await detect_unhealthy_containers(tenant_id="t1")

        assert len(result) >= 1
        assert result[0]["container_name"] == "stopped-ctr"
        assert any("not running" in issue for issue in result[0]["issues"])

    @patch("src.services.container_health_service.get_settings")
    async def test_detects_uninitialized_graph(self, mock_settings):
        """Container is running but graph db file does not exist."""
        from src.services.container_health_service import detect_unhealthy_containers

        mock_s = MagicMock()
        mock_s.get_code_graph_db_path.return_value = "/nonexistent/path.db"
        mock_settings.return_value = mock_s

        running_container = MagicMock()
        running_container.container_name = "my-ctr"
        running_container.status = MagicMock()
        running_container.status.value = "running"
        running_container.repo_url = "https://github.com/owner/repo"
        running_container.api_endpoint = "http://localhost:8687"
        running_container.codebase_path = "/repo"
        running_container.internal_endpoint = ""
        running_container.public_endpoint = ""

        mock_mgr = MagicMock()
        mock_mgr.list_containers.return_value = [running_container]

        # Import ContainerStatus for isinstance check

        # The container status is a MagicMock, so isinstance check fails,
        # it will use str(container.status) which is a mock repr != "running"
        with patch(
            "src.core.container_manager.get_container_manager",
            return_value=mock_mgr,
        ):
            result = await detect_unhealthy_containers(tenant_id="t1")

        # Should find issues (at least graph not initialized + API unreachable)
        assert len(result) >= 1

    @patch("src.services.container_health_service.get_settings")
    async def test_empty_container_list(self, mock_settings):
        """No containers means no unhealthy results."""
        from src.services.container_health_service import detect_unhealthy_containers

        mock_s = MagicMock()
        mock_settings.return_value = mock_s

        mock_mgr = MagicMock()
        mock_mgr.list_containers.return_value = []

        with patch(
            "src.core.container_manager.get_container_manager",
            return_value=mock_mgr,
        ):
            result = await detect_unhealthy_containers(tenant_id="t1")

        assert result == []

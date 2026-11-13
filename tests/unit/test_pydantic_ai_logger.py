"""Tests for services/agents/pydantic_ai_logger.py."""

from __future__ import annotations

import json
import logging
from datetime import datetime

import pytest

from src.models.pydantic_ai_models import (
    AgentConfiguration,
    AIAnalysisRequest,
    AIAnalysisResponse,
    PydanticAIAnalysisType,
    PydanticAITestFailureContext,
)
from src.services.agents.pydantic_ai_logger import (
    PydanticAILogEntry,
    PydanticAILogger,
    create_pydantic_ai_logger,
    get_pydantic_ai_logger,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ai_logger():
    """Create a fresh PydanticAILogger for each test."""
    return PydanticAILogger(logger_name="test_pydantic_ai")


@pytest.fixture
def agent_config():
    return AgentConfiguration(
        agent_id="test-agent-001",
        ai_model_provider="openai",
        ai_model_name="gpt-4",
        temperature=0.1,
        max_tokens=2000,
        enabled=True,
        timeout_seconds=30,
        max_retries=3,
    )


@pytest.fixture
def test_context():
    return PydanticAITestFailureContext(
        test_id="test-123",
        language="python",
        test_type="unit",
        exit_code=1,
        stdout="",
        stderr="SyntaxError: invalid syntax",
        execution_time_ms=500.0,
    )


@pytest.fixture
def analysis_request(test_context):
    return AIAnalysisRequest(
        context=test_context,
        analysis_types=[PydanticAIAnalysisType.FAILURE_ANALYSIS],
        ai_model_provider="openai",
        ai_model_name="gpt-4",
    )


@pytest.fixture
def success_response():
    return AIAnalysisResponse(
        request_id="req-001",
        success=True,
        total_analysis_time_ms=1500.0,
        ai_model_usage={"total_tokens": 500},
    )


@pytest.fixture
def failure_response():
    return AIAnalysisResponse(
        request_id="req-002",
        success=False,
        total_analysis_time_ms=200.0,
        error_message="Model timeout",
    )


# ---------------------------------------------------------------------------
# PydanticAILogEntry
# ---------------------------------------------------------------------------


class TestPydanticAILogEntry:
    def test_create_minimal_entry(self):
        entry = PydanticAILogEntry(
            level="INFO",
            operation="test_op",
            message="test message",
        )
        assert entry.level == "INFO"
        assert entry.operation == "test_op"
        assert entry.message == "test message"
        assert entry.agent_id is None
        assert entry.request_id is None
        assert entry.test_id is None
        assert entry.duration_ms is None
        assert entry.success is None
        assert entry.error_details is None
        assert entry.ai_model_info is None
        assert entry.usage_stats is None
        assert isinstance(entry.metadata, dict)
        assert isinstance(entry.timestamp, datetime)

    def test_create_full_entry(self):
        entry = PydanticAILogEntry(
            level="ERROR",
            operation="analysis",
            agent_id="agent-1",
            request_id="req-1",
            test_id="test-1",
            message="something failed",
            duration_ms=123.4,
            success=False,
            error_details={"error": "boom"},
            ai_model_info={"provider": "openai"},
            usage_stats={"tokens": 100},
            metadata={"key": "val"},
        )
        assert entry.agent_id == "agent-1"
        assert entry.error_details == {"error": "boom"}

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            PydanticAILogEntry(
                level="INFO",
                operation="test",
                message="msg",
                unknown_field="x",
            )


# ---------------------------------------------------------------------------
# PydanticAILogger initialisation
# ---------------------------------------------------------------------------


class TestLoggerInitialization:
    def test_default_logger_name(self):
        ai_logger = PydanticAILogger()
        assert ai_logger.logger is not None
        assert ai_logger.entries == []
        assert ai_logger.max_entries == 1000

    def test_custom_logger_name(self):
        ai_logger = PydanticAILogger(logger_name="custom_name")
        assert ai_logger.logger.name == "custom_name"

    def test_configure_logger_adds_handler(self):
        ai_logger = PydanticAILogger(logger_name="handler_test")
        assert len(ai_logger.logger.handlers) >= 1
        assert ai_logger.logger.level == logging.INFO

    def test_configure_logger_idempotent(self):
        ai_logger = PydanticAILogger(logger_name="idempotent_test")
        handler_count = len(ai_logger.logger.handlers)
        ai_logger._configure_logger()
        assert len(ai_logger.logger.handlers) == handler_count


# ---------------------------------------------------------------------------
# _add_entry
# ---------------------------------------------------------------------------


class TestAddEntry:
    def test_add_entry_appends(self, ai_logger):
        entry = PydanticAILogEntry(level="INFO", operation="op", message="msg")
        ai_logger._add_entry(entry)
        assert len(ai_logger.entries) == 1
        assert ai_logger.entries[0] is entry

    def test_add_entry_caps_at_max(self, ai_logger):
        ai_logger.max_entries = 5
        for i in range(10):
            entry = PydanticAILogEntry(level="INFO", operation="op", message=f"msg-{i}")
            ai_logger._add_entry(entry)
        assert len(ai_logger.entries) == 5
        assert ai_logger.entries[0].message == "msg-5"
        assert ai_logger.entries[-1].message == "msg-9"


# ---------------------------------------------------------------------------
# log_agent_initialization
# ---------------------------------------------------------------------------


class TestLogAgentInitialization:
    def test_success(self, ai_logger, agent_config):
        ai_logger.log_agent_initialization(
            agent_config=agent_config,
            success=True,
            duration_ms=42.0,
        )
        assert len(ai_logger.entries) == 1
        entry = ai_logger.entries[0]
        assert entry.level == "INFO"
        assert entry.operation == "agent_initialization"
        assert entry.agent_id == "test-agent-001"
        assert entry.success is True
        assert entry.duration_ms == 42.0
        assert "succeeded" in entry.message

    def test_failure(self, ai_logger, agent_config):
        ai_logger.log_agent_initialization(
            agent_config=agent_config,
            success=False,
            error_details={"error": "bad key"},
            duration_ms=10.0,
        )
        entry = ai_logger.entries[0]
        assert entry.level == "ERROR"
        assert entry.success is False
        assert "failed" in entry.message
        assert "bad key" in entry.message

    def test_failure_without_error_details(self, ai_logger, agent_config):
        ai_logger.log_agent_initialization(
            agent_config=agent_config,
            success=False,
        )
        entry = ai_logger.entries[0]
        assert entry.level == "ERROR"
        assert "failed" in entry.message

    def test_ai_model_info_populated(self, ai_logger, agent_config):
        ai_logger.log_agent_initialization(agent_config=agent_config, success=True)
        entry = ai_logger.entries[0]
        assert entry.ai_model_info["provider"] == "openai"
        assert entry.ai_model_info["model"] == "gpt-4"

    def test_metadata_populated(self, ai_logger, agent_config):
        ai_logger.log_agent_initialization(agent_config=agent_config, success=True)
        entry = ai_logger.entries[0]
        assert entry.metadata["enabled"] is True
        assert entry.metadata["max_retries"] == 3


# ---------------------------------------------------------------------------
# log_analysis_request
# ---------------------------------------------------------------------------


class TestLogAnalysisRequest:
    def test_basic_request(self, ai_logger, analysis_request):
        ai_logger.log_analysis_request(analysis_request, "req-100")
        assert len(ai_logger.entries) == 1
        entry = ai_logger.entries[0]
        assert entry.level == "INFO"
        assert entry.operation == "analysis_request"
        assert entry.request_id == "req-100"
        assert entry.test_id == "test-123"
        assert "req-100" in entry.message
        assert entry.ai_model_info["provider"] == "openai"
        assert entry.metadata["language"] == "python"
        assert entry.metadata["test_type"] == "unit"


# ---------------------------------------------------------------------------
# log_analysis_response
# ---------------------------------------------------------------------------


class TestLogAnalysisResponse:
    def test_success_response(self, ai_logger, success_response):
        ai_logger.log_analysis_response(
            response=success_response,
            duration_ms=1500.0,
            agent_id="agent-1",
        )
        entry = ai_logger.entries[0]
        assert entry.level == "INFO"
        assert entry.success is True
        assert "completed" in entry.message
        assert entry.agent_id == "agent-1"
        assert entry.duration_ms == 1500.0

    def test_failure_response(self, ai_logger, failure_response):
        ai_logger.log_analysis_response(
            response=failure_response,
            duration_ms=200.0,
        )
        entry = ai_logger.entries[0]
        assert entry.level == "ERROR"
        assert entry.success is False
        assert "failed" in entry.message
        assert "Model timeout" in entry.message
        assert entry.error_details["error_message"] == "Model timeout"


# ---------------------------------------------------------------------------
# log_model_interaction
# ---------------------------------------------------------------------------


class TestLogModelInteraction:
    def test_success_interaction(self, ai_logger):
        ai_logger.log_model_interaction(
            agent_id="agent-1",
            ai_model_provider="openai",
            ai_model_name="gpt-4",
            operation="analyze",
            success=True,
            duration_ms=500.0,
            tokens_used=350,
            cost_usd=0.02,
        )
        entry = ai_logger.entries[0]
        assert entry.level == "DEBUG"
        assert entry.success is True
        assert "succeeded" in entry.message
        assert entry.usage_stats["tokens_used"] == 350
        assert entry.usage_stats["cost_usd"] == 0.02

    def test_failure_interaction(self, ai_logger):
        ai_logger.log_model_interaction(
            agent_id="agent-1",
            ai_model_provider="openai",
            ai_model_name="gpt-4",
            operation="analyze",
            success=False,
            duration_ms=100.0,
            error_details={"error": "rate limited"},
        )
        entry = ai_logger.entries[0]
        assert entry.level == "WARNING"
        assert entry.success is False
        assert "failed" in entry.message
        assert "rate limited" in entry.message

    def test_no_usage_stats_when_no_tokens(self, ai_logger):
        ai_logger.log_model_interaction(
            agent_id="agent-1",
            ai_model_provider="openai",
            ai_model_name="gpt-4",
            operation="test",
            success=True,
            duration_ms=100.0,
        )
        entry = ai_logger.entries[0]
        assert entry.usage_stats is None

    def test_operation_prefix(self, ai_logger):
        ai_logger.log_model_interaction(
            agent_id="a",
            ai_model_provider="openai",
            ai_model_name="gpt-4",
            operation="embed",
            success=True,
            duration_ms=10.0,
        )
        assert ai_logger.entries[0].operation == "model_embed"


# ---------------------------------------------------------------------------
# log_error
# ---------------------------------------------------------------------------


class TestLogError:
    def test_basic_error(self, ai_logger):
        err = ValueError("something broke")
        ai_logger.log_error(
            operation="test_op",
            error=err,
            agent_id="agent-1",
            request_id="req-1",
            test_id="test-1",
            duration_ms=50.0,
        )
        entry = ai_logger.entries[0]
        assert entry.level == "ERROR"
        assert entry.success is False
        assert "something broke" in entry.message
        assert entry.error_details["error_type"] == "ValueError"
        assert entry.error_details["error_message"] == "something broke"

    def test_error_with_to_dict(self, ai_logger):
        class CustomError(Exception):
            def to_dict(self):
                return {"extra": "data"}

        err = CustomError("custom error")
        ai_logger.log_error(operation="op", error=err)
        entry = ai_logger.entries[0]
        assert entry.error_details.get("extra") == "data"
        assert entry.error_details["error_type"] == "CustomError"

    def test_error_with_additional_context(self, ai_logger):
        err = RuntimeError("oops")
        ai_logger.log_error(
            operation="op",
            error=err,
            additional_context={"container": "test-container"},
        )
        entry = ai_logger.entries[0]
        assert entry.error_details["container"] == "test-container"


# ---------------------------------------------------------------------------
# log_performance_metric
# ---------------------------------------------------------------------------


class TestLogPerformanceMetric:
    def test_basic_metric(self, ai_logger):
        ai_logger.log_performance_metric(
            metric_name="latency_ms",
            value=42.5,
        )
        entry = ai_logger.entries[0]
        assert entry.level == "DEBUG"
        assert entry.operation == "performance_metric"
        assert entry.metadata["metric_name"] == "latency_ms"
        assert entry.metadata["metric_value"] == 42.5

    def test_with_agent_and_operation(self, ai_logger):
        ai_logger.log_performance_metric(
            metric_name="tokens",
            value=500,
            agent_id="agent-1",
            operation="embedding",
            metadata={"batch": True},
        )
        entry = ai_logger.entries[0]
        assert entry.agent_id == "agent-1"
        assert entry.operation == "embedding"
        assert entry.metadata["batch"] is True


# ---------------------------------------------------------------------------
# log_health_check
# ---------------------------------------------------------------------------


class TestLogHealthCheck:
    def test_healthy(self, ai_logger):
        ai_logger.log_health_check(
            agent_id="agent-1",
            healthy=True,
            response_time_ms=25.0,
        )
        entry = ai_logger.entries[0]
        assert entry.level == "INFO"
        assert entry.success is True
        assert "passed" in entry.message

    def test_unhealthy(self, ai_logger):
        ai_logger.log_health_check(
            agent_id="agent-1",
            healthy=False,
            response_time_ms=5000.0,
            error_details={"error": "timeout"},
        )
        entry = ai_logger.entries[0]
        assert entry.level == "WARNING"
        assert entry.success is False
        assert "failed" in entry.message
        assert "timeout" in entry.message

    def test_unhealthy_no_error_details(self, ai_logger):
        ai_logger.log_health_check(agent_id="agent-1", healthy=False)
        entry = ai_logger.entries[0]
        assert "failed" in entry.message


# ---------------------------------------------------------------------------
# log_batch_analysis
# ---------------------------------------------------------------------------


class TestLogBatchAnalysis:
    def test_all_successful(self, ai_logger):
        ai_logger.log_batch_analysis(
            batch_size=10,
            successful_count=10,
            failed_count=0,
            total_duration_ms=5000.0,
            average_duration_ms=500.0,
        )
        entry = ai_logger.entries[0]
        assert entry.level == "INFO"
        assert entry.success is True
        assert entry.metadata["batch_size"] == 10
        assert entry.metadata["success_rate"] == 1.0

    def test_some_failures(self, ai_logger):
        ai_logger.log_batch_analysis(
            batch_size=10,
            successful_count=7,
            failed_count=3,
            total_duration_ms=8000.0,
            average_duration_ms=800.0,
        )
        entry = ai_logger.entries[0]
        assert entry.success is False
        assert entry.metadata["failed_count"] == 3
        assert entry.metadata["success_rate"] == 0.7

    def test_zero_batch(self, ai_logger):
        ai_logger.log_batch_analysis(
            batch_size=0,
            successful_count=0,
            failed_count=0,
            total_duration_ms=0.0,
            average_duration_ms=0.0,
        )
        entry = ai_logger.entries[0]
        assert entry.metadata["success_rate"] == 0


# ---------------------------------------------------------------------------
# Query / retrieval methods
# ---------------------------------------------------------------------------


class TestQueryMethods:
    def _populate(self, ai_logger):
        """Insert a variety of entries."""
        entries = [
            PydanticAILogEntry(
                level="INFO",
                operation="analysis_response",
                agent_id="a1",
                message="m1",
                success=True,
                duration_ms=100.0,
            ),
            PydanticAILogEntry(
                level="ERROR",
                operation="analysis_response",
                agent_id="a1",
                message="m2",
                success=False,
                duration_ms=200.0,
            ),
            PydanticAILogEntry(level="INFO", operation="health_check", agent_id="a2", message="m3"),
            PydanticAILogEntry(
                level="CRITICAL", operation="model_run", agent_id="a1", message="m4"
            ),
            PydanticAILogEntry(
                level="DEBUG",
                operation="analysis_response",
                agent_id="a2",
                message="m5",
                success=True,
                duration_ms=50.0,
            ),
        ]
        for e in entries:
            ai_logger._add_entry(e)

    def test_get_recent_entries(self, ai_logger):
        self._populate(ai_logger)
        recent = ai_logger.get_recent_entries(limit=3)
        assert len(recent) == 3
        assert recent[-1].message == "m5"

    def test_get_recent_entries_default(self, ai_logger):
        self._populate(ai_logger)
        recent = ai_logger.get_recent_entries()
        assert len(recent) == 5

    def test_get_entries_by_operation(self, ai_logger):
        self._populate(ai_logger)
        entries = ai_logger.get_entries_by_operation("analysis_response")
        assert len(entries) == 3

    def test_get_entries_by_operation_with_limit(self, ai_logger):
        self._populate(ai_logger)
        entries = ai_logger.get_entries_by_operation("analysis_response", limit=2)
        assert len(entries) == 2

    def test_get_entries_by_agent(self, ai_logger):
        self._populate(ai_logger)
        entries = ai_logger.get_entries_by_agent("a1")
        assert len(entries) == 3

    def test_get_entries_by_agent_with_limit(self, ai_logger):
        self._populate(ai_logger)
        entries = ai_logger.get_entries_by_agent("a1", limit=1)
        assert len(entries) == 1

    def test_get_error_entries(self, ai_logger):
        self._populate(ai_logger)
        errors = ai_logger.get_error_entries()
        assert len(errors) == 2
        levels = {e.level for e in errors}
        assert levels == {"ERROR", "CRITICAL"}


# ---------------------------------------------------------------------------
# get_performance_summary
# ---------------------------------------------------------------------------


class TestPerformanceSummary:
    def test_empty(self, ai_logger):
        summary = ai_logger.get_performance_summary()
        assert summary["total_analyses"] == 0
        assert summary["success_rate"] == 0.0

    def test_with_data(self, ai_logger):
        for success, dur in [(True, 100.0), (True, 200.0), (False, 50.0)]:
            ai_logger._add_entry(
                PydanticAILogEntry(
                    level="INFO",
                    operation="analysis_response",
                    message="m",
                    success=success,
                    duration_ms=dur,
                )
            )
        summary = ai_logger.get_performance_summary()
        assert summary["total_analyses"] == 3
        assert summary["successful_analyses"] == 2
        assert summary["failed_analyses"] == 1
        assert summary["min_duration_ms"] == 50.0
        assert summary["max_duration_ms"] == 200.0
        avg = (100.0 + 200.0 + 50.0) / 3
        assert abs(summary["average_duration_ms"] - avg) < 0.01
        assert abs(summary["success_rate"] - 2 / 3) < 0.01


# ---------------------------------------------------------------------------
# export_logs
# ---------------------------------------------------------------------------


class TestExportLogs:
    def test_export_json(self, ai_logger):
        ai_logger._add_entry(PydanticAILogEntry(level="INFO", operation="op", message="hello"))
        exported = ai_logger.export_logs(format="json")
        data = json.loads(exported)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["message"] == "hello"

    def test_export_json_with_limit(self, ai_logger):
        for i in range(5):
            ai_logger._add_entry(PydanticAILogEntry(level="INFO", operation="op", message=f"m{i}"))
        exported = ai_logger.export_logs(format="json", limit=2)
        data = json.loads(exported)
        assert len(data) == 2

    def test_export_csv(self, ai_logger):
        ai_logger._add_entry(PydanticAILogEntry(level="INFO", operation="op", message="hello"))
        exported = ai_logger.export_logs(format="csv")
        assert "hello" in exported
        assert "operation" in exported  # header row

    def test_export_csv_empty(self, ai_logger):
        exported = ai_logger.export_logs(format="csv")
        assert exported == ""

    def test_export_unsupported_format(self, ai_logger):
        with pytest.raises(ValueError, match="Unsupported format"):
            ai_logger.export_logs(format="xml")


# ---------------------------------------------------------------------------
# clear_logs
# ---------------------------------------------------------------------------


class TestClearLogs:
    def test_clear(self, ai_logger):
        ai_logger._add_entry(PydanticAILogEntry(level="INFO", operation="op", message="m"))
        assert len(ai_logger.entries) == 1
        ai_logger.clear_logs()
        assert len(ai_logger.entries) == 0


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


class TestModuleFunctions:
    def test_get_pydantic_ai_logger_singleton(self):
        import src.services.agents.pydantic_ai_logger as mod

        mod._pydantic_ai_logger = None
        logger1 = get_pydantic_ai_logger()
        logger2 = get_pydantic_ai_logger()
        assert logger1 is logger2
        mod._pydantic_ai_logger = None  # cleanup

    def test_create_pydantic_ai_logger(self):
        logger_instance = create_pydantic_ai_logger("my_logger")
        assert isinstance(logger_instance, PydanticAILogger)
        assert logger_instance.logger.name == "my_logger"

    def test_create_pydantic_ai_logger_default(self):
        logger_instance = create_pydantic_ai_logger()
        assert logger_instance.logger.name == "pydantic_ai"

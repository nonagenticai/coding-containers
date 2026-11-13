"""Comprehensive logging system for PydanticAI operations."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.models.pydantic_ai_models import (
    AgentConfiguration,
    AIAnalysisRequest,
    AIAnalysisResponse,
)


class PydanticAILogEntry(BaseModel):
    """Structured log entry for PydanticAI operations."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        protected_namespaces=(),
    )

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    level: str = Field(..., description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    operation: str = Field(..., description="Operation being performed")
    agent_id: str | None = Field(None, description="Agent ID if applicable")
    request_id: str | None = Field(None, description="Request ID for tracing")
    test_id: str | None = Field(None, description="Test ID being analyzed")
    message: str = Field(..., description="Log message")
    duration_ms: float | None = Field(None, description="Operation duration in milliseconds")
    success: bool | None = Field(None, description="Whether operation succeeded")
    error_details: dict[str, Any] | None = Field(None, description="Error details if applicable")
    ai_model_info: dict[str, Any] | None = Field(None, description="Model information")
    usage_stats: dict[str, Any] | None = Field(None, description="Usage statistics")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PydanticAILogger:
    """Specialized logger for PydanticAI operations."""

    def __init__(self, logger_name: str | None = None):
        """Initialize PydanticAI logger.

        Args:
            logger_name: Name of the logger instance (defaults to __name__)
        """
        self.logger = logging.getLogger(logger_name or __name__)
        self.entries: list[PydanticAILogEntry] = []
        self.max_entries = 1000  # Keep last 1000 entries in memory

        # Configure logger format
        self._configure_logger()

    def _configure_logger(self) -> None:
        """Configure the logger with appropriate formatting."""
        # Only configure if no handlers exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _add_entry(self, entry: PydanticAILogEntry) -> None:
        """Add log entry to internal storage."""
        self.entries.append(entry)

        # Maintain max entries limit
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    def log_agent_initialization(
        self,
        agent_config: AgentConfiguration,
        success: bool,
        error_details: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Log agent initialization.

        Args:
            agent_config: Agent configuration
            success: Whether initialization succeeded
            error_details: Error details if failed
            duration_ms: Initialization duration
        """
        level = "INFO" if success else "ERROR"
        message = (
            f"Agent initialization {'succeeded' if success else 'failed'}: {agent_config.agent_id}"
        )

        if not success and error_details:
            message += f" - {error_details.get('error', 'Unknown error')}"

        entry = PydanticAILogEntry(
            level=level,
            operation="agent_initialization",
            agent_id=agent_config.agent_id,
            message=message,
            duration_ms=duration_ms,
            success=success,
            error_details=error_details,
            ai_model_info={
                "provider": agent_config.ai_model_provider,
                "model": agent_config.ai_model_name,
                "temperature": agent_config.temperature,
                "max_tokens": agent_config.max_tokens,
            },
            metadata={
                "enabled": agent_config.enabled,
                "timeout_seconds": agent_config.timeout_seconds,
                "max_retries": agent_config.max_retries,
            },
        )

        self._add_entry(entry)
        self.logger.log(getattr(logging, level), message)

    def log_analysis_request(
        self,
        request: AIAnalysisRequest,
        request_id: str,
    ) -> None:
        """Log analysis request.

        Args:
            request: Analysis request
            request_id: Request ID for tracing
        """
        message = f"Analysis request initiated: {request_id} for test {request.context.test_id}"

        entry = PydanticAILogEntry(
            level="INFO",
            operation="analysis_request",
            request_id=request_id,
            test_id=request.context.test_id,
            message=message,
            ai_model_info={
                "provider": request.ai_model_provider,
                "model": request.ai_model_name,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            },
            metadata={
                "language": request.context.language,
                "test_type": request.context.test_type,
                "exit_code": request.context.exit_code,
                "execution_time_ms": request.context.execution_time_ms,
                "analysis_types": [t.value for t in request.analysis_types],
                "include_code_fixes": request.include_code_fixes,
                "include_performance_analysis": request.include_performance_analysis,
                "include_security_analysis": request.include_security_analysis,
            },
        )

        self._add_entry(entry)
        self.logger.info(message)

    def log_analysis_response(
        self,
        response: AIAnalysisResponse,
        duration_ms: float,
        agent_id: str | None = None,
    ) -> None:
        """Log analysis response.

        Args:
            response: Analysis response
            duration_ms: Response duration
            agent_id: Agent ID that performed analysis
        """
        level = "INFO" if response.success else "ERROR"
        message = f"Analysis {'completed' if response.success else 'failed'}: {response.request_id}"

        if not response.success:
            message += f" - {response.error_message}"

        entry = PydanticAILogEntry(
            level=level,
            operation="analysis_response",
            agent_id=agent_id,
            request_id=response.request_id,
            message=message,
            duration_ms=duration_ms,
            success=response.success,
            error_details=(
                {"error_message": response.error_message} if response.error_message else None
            ),
            usage_stats=response.ai_model_usage,
            metadata={
                "total_analysis_time_ms": response.total_analysis_time_ms,
                "has_failure_analysis": response.failure_analysis is not None,
                "code_quality_issues_count": len(response.code_quality_issues),
                "has_performance_analysis": response.performance_analysis is not None,
                "has_security_analysis": response.security_analysis is not None,
                "warnings_count": len(response.warnings),
            },
        )

        self._add_entry(entry)
        self.logger.log(getattr(logging, level), message)

    def log_model_interaction(
        self,
        agent_id: str,
        ai_model_provider: str,
        ai_model_name: str,
        operation: str,
        success: bool,
        duration_ms: float,
        tokens_used: int | None = None,
        cost_usd: float | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> None:
        """Log model interaction.

        Args:
            agent_id: Agent ID
            ai_model_provider: Model provider
            ai_model_name: Model name
            operation: Operation performed
            success: Whether interaction succeeded
            duration_ms: Interaction duration
            tokens_used: Tokens used
            cost_usd: Cost in USD
            error_details: Error details if failed
        """
        level = "DEBUG" if success else "WARNING"
        message = f"Model interaction {'succeeded' if success else 'failed'}: {agent_id} -> {ai_model_provider}:{ai_model_name}"

        if not success and error_details:
            message += f" - {error_details.get('error', 'Unknown error')}"

        entry = PydanticAILogEntry(
            level=level,
            operation=f"model_{operation}",
            agent_id=agent_id,
            message=message,
            duration_ms=duration_ms,
            success=success,
            error_details=error_details,
            ai_model_info={
                "provider": ai_model_provider,
                "model": ai_model_name,
            },
            usage_stats=(
                {
                    "tokens_used": tokens_used,
                    "cost_usd": cost_usd,
                }
                if tokens_used is not None or cost_usd is not None
                else None
            ),
        )

        self._add_entry(entry)
        self.logger.log(getattr(logging, level), message)

    def log_error(
        self,
        operation: str,
        error: Exception,
        agent_id: str | None = None,
        request_id: str | None = None,
        test_id: str | None = None,
        duration_ms: float | None = None,
        additional_context: dict[str, Any] | None = None,
    ) -> None:
        """Log error details.

        Args:
            operation: Operation that failed
            error: Exception that occurred
            agent_id: Agent ID if applicable
            request_id: Request ID if applicable
            test_id: Test ID if applicable
            duration_ms: Operation duration
            additional_context: Additional context
        """
        message = f"Error in {operation}: {str(error)}"

        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        to_dict_method = getattr(error, "to_dict", None)
        if callable(to_dict_method):
            extra = to_dict_method()
            if isinstance(extra, dict):
                error_details.update(extra)

        if additional_context:
            error_details.update(additional_context)

        entry = PydanticAILogEntry(
            level="ERROR",
            operation=operation,
            agent_id=agent_id,
            request_id=request_id,
            test_id=test_id,
            message=message,
            duration_ms=duration_ms,
            success=False,
            error_details=error_details,
        )

        self._add_entry(entry)
        self.logger.error(message, exc_info=True)

    def log_performance_metric(
        self,
        metric_name: str,
        value: int | float,
        agent_id: str | None = None,
        operation: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log performance metric.

        Args:
            metric_name: Metric name
            value: Metric value
            agent_id: Agent ID if applicable
            operation: Operation if applicable
            metadata: Additional metadata
        """
        message = f"Performance metric - {metric_name}: {value}"

        entry = PydanticAILogEntry(
            level="DEBUG",
            operation=operation or "performance_metric",
            agent_id=agent_id,
            message=message,
            metadata={
                "metric_name": metric_name,
                "metric_value": value,
                **(metadata or {}),
            },
        )

        self._add_entry(entry)
        self.logger.debug(message)

    def log_health_check(
        self,
        agent_id: str,
        healthy: bool,
        response_time_ms: float | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> None:
        """Log health check result.

        Args:
            agent_id: Agent ID
            healthy: Whether agent is healthy
            response_time_ms: Health check response time
            error_details: Error details if unhealthy
        """
        level = "INFO" if healthy else "WARNING"
        message = f"Health check {'passed' if healthy else 'failed'}: {agent_id}"

        if not healthy and error_details:
            message += f" - {error_details.get('error', 'Unknown error')}"

        entry = PydanticAILogEntry(
            level=level,
            operation="health_check",
            agent_id=agent_id,
            message=message,
            duration_ms=response_time_ms,
            success=healthy,
            error_details=error_details,
        )

        self._add_entry(entry)
        self.logger.log(getattr(logging, level), message)

    def log_batch_analysis(
        self,
        batch_size: int,
        successful_count: int,
        failed_count: int,
        total_duration_ms: float,
        average_duration_ms: float,
    ) -> None:
        """Log batch analysis results.

        Args:
            batch_size: Total batch size
            successful_count: Number of successful analyses
            failed_count: Number of failed analyses
            total_duration_ms: Total duration
            average_duration_ms: Average duration per analysis
        """
        message = f"Batch analysis completed: {batch_size} tests, {successful_count} successful, {failed_count} failed"

        entry = PydanticAILogEntry(
            level="INFO",
            operation="batch_analysis",
            message=message,
            duration_ms=total_duration_ms,
            success=failed_count == 0,
            metadata={
                "batch_size": batch_size,
                "successful_count": successful_count,
                "failed_count": failed_count,
                "success_rate": successful_count / batch_size if batch_size > 0 else 0,
                "average_duration_ms": average_duration_ms,
            },
        )

        self._add_entry(entry)
        self.logger.info(message)

    def get_recent_entries(self, limit: int = 100) -> list[PydanticAILogEntry]:
        """Get recent log entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of recent log entries
        """
        return self.entries[-limit:]

    def get_entries_by_operation(
        self, operation: str, limit: int = 100
    ) -> list[PydanticAILogEntry]:
        """Get log entries by operation.

        Args:
            operation: Operation to filter by
            limit: Maximum number of entries to return

        Returns:
            List of log entries for the operation
        """
        filtered_entries = [entry for entry in self.entries if entry.operation == operation]
        return filtered_entries[-limit:]

    def get_entries_by_agent(self, agent_id: str, limit: int = 100) -> list[PydanticAILogEntry]:
        """Get log entries by agent.

        Args:
            agent_id: Agent ID to filter by
            limit: Maximum number of entries to return

        Returns:
            List of log entries for the agent
        """
        filtered_entries = [entry for entry in self.entries if entry.agent_id == agent_id]
        return filtered_entries[-limit:]

    def get_error_entries(self, limit: int = 100) -> list[PydanticAILogEntry]:
        """Get error log entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of error log entries
        """
        error_entries = [entry for entry in self.entries if entry.level in ["ERROR", "CRITICAL"]]
        return error_entries[-limit:]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary from log entries.

        Returns:
            Performance summary dictionary
        """
        analysis_entries = [
            entry for entry in self.entries if entry.operation == "analysis_response"
        ]

        if not analysis_entries:
            return {
                "total_analyses": 0,
                "successful_analyses": 0,
                "failed_analyses": 0,
                "average_duration_ms": 0.0,
                "min_duration_ms": 0.0,
                "max_duration_ms": 0.0,
                "success_rate": 0.0,
            }

        successful = [entry for entry in analysis_entries if entry.success]
        failed = [entry for entry in analysis_entries if not entry.success]

        durations = [
            entry.duration_ms for entry in analysis_entries if entry.duration_ms is not None
        ]

        return {
            "total_analyses": len(analysis_entries),
            "successful_analyses": len(successful),
            "failed_analyses": len(failed),
            "average_duration_ms": (sum(durations) / len(durations) if durations else 0.0),
            "min_duration_ms": min(durations) if durations else 0.0,
            "max_duration_ms": max(durations) if durations else 0.0,
            "success_rate": (len(successful) / len(analysis_entries) if analysis_entries else 0.0),
        }

    def export_logs(self, format: str = "json", limit: int | None = None) -> str:
        """Export log entries.

        Args:
            format: Export format ("json" or "csv")
            limit: Maximum number of entries to export

        Returns:
            Exported log data as string
        """
        entries = self.entries[-limit:] if limit else self.entries

        if format == "json":
            return json.dumps([entry.model_dump() for entry in entries], indent=2, default=str)
        elif format == "csv":
            import csv
            import io

            output = io.StringIO()

            if entries:
                fieldnames = entries[0].model_dump().keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()

                for entry in entries:
                    row = entry.model_dump()
                    # Convert complex objects to strings
                    for key, value in row.items():
                        if isinstance(value, (dict, list)):
                            row[key] = json.dumps(value, default=str)
                    writer.writerow(row)

            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def clear_logs(self) -> None:
        """Clear all log entries."""
        self.entries.clear()
        self.logger.info("Log entries cleared")


# Global logger instance
_pydantic_ai_logger: PydanticAILogger | None = None


def get_pydantic_ai_logger() -> PydanticAILogger:
    """Get global PydanticAI logger instance."""
    global _pydantic_ai_logger
    if _pydantic_ai_logger is None:
        _pydantic_ai_logger = PydanticAILogger()
    return _pydantic_ai_logger


def create_pydantic_ai_logger(logger_name: str = "pydantic_ai") -> PydanticAILogger:
    """Create a new PydanticAI logger instance.

    Args:
        logger_name: Name of the logger

    Returns:
        New PydanticAI logger instance
    """
    return PydanticAILogger(logger_name)

"""PydanticAI models for intelligent test failure analysis."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


class PydanticAIAnalysisType(StrEnum):
    """Types of test failure analysis."""

    FAILURE_ANALYSIS = "failure_analysis"
    CODE_QUALITY = "code_quality"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DEPENDENCY = "dependency"
    RACE_CONDITION = "race_condition"
    MEMORY_LEAK = "memory_leak"
    TIMEOUT = "timeout"
    COMPILATION_ERROR = "compilation_error"
    RUNTIME_ERROR = "runtime_error"


class FailureSeverity(StrEnum):
    """Severity levels for test failures."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FailureCategory(StrEnum):
    """Categories of test failures."""

    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    LOGIC_ERROR = "logic_error"
    RUNTIME_ERROR = "runtime_error"
    ASSERTION_ERROR = "assertion_error"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"
    DEPENDENCY_ERROR = "dependency_error"
    CONFIGURATION_ERROR = "configuration_error"
    ENVIRONMENT_ERROR = "environment_error"
    UNKNOWN = "unknown"


class PydanticAITestFailureContext(BaseModel):
    """Context information for test failure analysis."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        frozen=False,
    )

    test_id: str = Field(..., description="Unique test execution ID")
    language: str = Field(..., description="Programming language being tested")
    test_type: str = Field(..., description="Type of test (unit, integration, etc.)")
    exit_code: int = Field(..., description="Test process exit code")
    stdout: str = Field(default="", description="Standard output from test execution")
    stderr: str = Field(default="", description="Standard error from test execution")
    execution_time_ms: float = Field(..., description="Test execution time in milliseconds")
    container_name: str | None = Field(None, description="Container name where test ran")
    source_code: str | None = Field(None, description="Source code being tested")
    test_code: str | None = Field(None, description="Test code that failed")
    dependencies: dict[str, str] = Field(default_factory=dict, description="Project dependencies")
    environment: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    resource_usage: dict[str, Any] | None = Field(None, description="Resource usage during test")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When context was created",
    )

    @field_validator("execution_time_ms")
    @classmethod
    def validate_execution_time(cls, v: float) -> float:
        """Validate execution time is non-negative."""
        if v < 0:
            raise ValueError("Execution time cannot be negative")
        return v

    @field_validator("exit_code")
    @classmethod
    def validate_exit_code(cls, v: int) -> int:
        """Validate exit code is reasonable."""
        if v < 0 or v > 255:
            logger.warning(f"Unusual exit code: {v}")
        return v


class FailureAnalysisResult(BaseModel):
    """Result of intelligent test failure analysis."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        protected_namespaces=(),
    )

    analysis_id: str = Field(..., description="Unique analysis result ID")
    test_id: str = Field(..., description="Test ID that was analyzed")
    analysis_type: PydanticAIAnalysisType = Field(..., description="Type of analysis performed")
    category: FailureCategory = Field(..., description="Category of failure")
    severity: FailureSeverity = Field(..., description="Severity of the failure")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in analysis (0-1)")
    root_cause: str = Field(..., description="Identified root cause of failure")
    symptoms: list[str] = Field(default_factory=list, description="Observable symptoms")
    potential_causes: list[str] = Field(
        default_factory=list, description="Potential causes considered"
    )
    recommendations: list[str] = Field(default_factory=list, description="Recommended actions")
    code_fixes: list[str] = Field(default_factory=list, description="Suggested code fixes")
    related_issues: list[str] = Field(
        default_factory=list, description="Related issues or patterns"
    )
    analysis_duration_ms: float = Field(..., description="Time taken for analysis")
    ai_model_provider: str = Field(..., description="AI model provider used")
    ai_model_name: str = Field(..., description="Specific model used for analysis")
    raw_response: str | None = Field(None, description="Raw AI response for debugging")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When analysis was completed",
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class CodeQualityIssue(BaseModel):
    """Code quality issue identified during analysis."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        frozen=False,
    )

    issue_id: str = Field(..., description="Unique issue identifier")
    category: str = Field(..., description="Issue category (style, bug, performance, etc.)")
    severity: FailureSeverity = Field(..., description="Issue severity")
    message: str = Field(..., description="Issue description")
    file_path: str | None = Field(None, description="File where issue was found")
    line_number: int | None = Field(None, description="Line number of issue")
    column_number: int | None = Field(None, description="Column number of issue")
    code_snippet: str | None = Field(None, description="Code snippet showing issue")
    suggested_fix: str | None = Field(None, description="Suggested fix for issue")
    fix_confidence: float | None = Field(None, ge=0.0, le=1.0, description="Confidence in fix")
    rule_id: str | None = Field(None, description="Rule ID that triggered issue")
    tool_name: str | None = Field(None, description="Tool that detected issue")

    @field_validator("line_number")
    @classmethod
    def validate_line_number(cls, v: int | None) -> int | None:
        """Validate line number is positive."""
        if v is not None and v <= 0:
            raise ValueError("Line number must be positive")
        return v


class PerformanceAnalysis(BaseModel):
    """Performance analysis results."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        frozen=False,
    )

    analysis_id: str = Field(..., description="Unique analysis identifier")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    memory_usage_mb: float | None = Field(None, description="Memory usage in MB")
    cpu_usage_percent: float | None = Field(None, description="CPU usage percentage")
    bottlenecks: list[str] = Field(default_factory=list, description="Identified bottlenecks")
    optimization_suggestions: list[str] = Field(
        default_factory=list, description="Optimization suggestions"
    )
    performance_score: float | None = Field(
        None, ge=0.0, le=10.0, description="Performance score (0-10)"
    )
    baseline_comparison: dict[str, Any] | None = Field(None, description="Comparison with baseline")

    @field_validator("performance_score")
    @classmethod
    def validate_performance_score(cls, v: float | None) -> float | None:
        """Validate performance score is between 0 and 10."""
        if v is not None and not 0.0 <= v <= 10.0:
            raise ValueError("Performance score must be between 0.0 and 10.0")
        return v


class SecurityAnalysis(BaseModel):
    """Security analysis results."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        frozen=False,
    )

    analysis_id: str = Field(..., description="Unique analysis identifier")
    vulnerabilities: list[str] = Field(
        default_factory=list, description="Identified vulnerabilities"
    )
    security_score: float | None = Field(None, ge=0.0, le=10.0, description="Security score (0-10)")
    risk_level: FailureSeverity = Field(..., description="Overall risk level")
    recommendations: list[str] = Field(default_factory=list, description="Security recommendations")
    cve_references: list[str] = Field(default_factory=list, description="Related CVE references")
    compliance_issues: list[str] = Field(default_factory=list, description="Compliance issues")

    @field_validator("security_score")
    @classmethod
    def validate_security_score(cls, v: float | None) -> float | None:
        """Validate security score is between 0 and 10."""
        if v is not None and not 0.0 <= v <= 10.0:
            raise ValueError("Security score must be between 0.0 and 10.0")
        return v


class AIAnalysisRequest(BaseModel):
    """Request for AI-powered test failure analysis."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        protected_namespaces=(),
    )

    context: PydanticAITestFailureContext = Field(..., description="Test failure context")
    analysis_types: list[PydanticAIAnalysisType] = Field(
        default_factory=list, description="Types of analysis to perform"
    )
    ai_model_provider: str = Field(default="openai", description="AI model provider to use")
    ai_model_name: str = Field(default="gpt-4", description="Specific model to use")
    max_tokens: int = Field(default=2000, description="Maximum tokens for AI response")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature")
    timeout_seconds: int = Field(default=30, description="Request timeout in seconds")
    include_code_fixes: bool = Field(default=True, description="Include code fix suggestions")
    include_performance_analysis: bool = Field(
        default=True, description="Include performance analysis"
    )
    include_security_analysis: bool = Field(default=True, description="Include security analysis")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional request metadata"
    )

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is between 0 and 2."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        """Validate max tokens is reasonable."""
        if v <= 0 or v > 32000:
            raise ValueError("Max tokens must be between 1 and 32000")
        return v


class AIAnalysisResponse(BaseModel):
    """Response from AI-powered test failure analysis."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        protected_namespaces=(),
    )

    request_id: str = Field(..., description="Unique request identifier")
    success: bool = Field(..., description="Whether analysis was successful")
    failure_analysis: FailureAnalysisResult | None = Field(
        None, description="Main failure analysis"
    )
    code_quality_issues: list[CodeQualityIssue] = Field(
        default_factory=list, description="Code quality issues"
    )
    performance_analysis: PerformanceAnalysis | None = Field(
        None, description="Performance analysis"
    )
    security_analysis: SecurityAnalysis | None = Field(None, description="Security analysis")
    total_analysis_time_ms: float = Field(..., description="Total analysis time")
    ai_model_usage: dict[str, Any] = Field(
        default_factory=dict, description="Model usage statistics"
    )
    error_message: str | None = Field(None, description="Error message if analysis failed")
    warnings: list[str] = Field(default_factory=list, description="Analysis warnings")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional response metadata"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When response was created",
    )


class AgentConfiguration(BaseModel):
    """Configuration for PydanticAI agents."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        protected_namespaces=(),
    )

    agent_id: str = Field(..., description="Unique agent identifier")
    ai_model_provider: str = Field(..., description="AI model provider")
    ai_model_name: str = Field(..., description="Model name")
    api_key: str | None = Field(None, description="API key for model provider")
    base_url: str | None = Field(None, description="Base URL for API")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout_seconds: int = Field(default=30, description="Request timeout")
    rate_limit_rpm: int = Field(default=100, description="Rate limit requests per minute")
    rate_limit_tpm: int = Field(default=10000, description="Rate limit tokens per minute")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=2000, description="Maximum tokens per request")
    system_prompt: str | None = Field(None, description="System prompt for agent")
    enabled: bool = Field(default=True, description="Whether agent is enabled")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional configuration")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is between 0 and 2."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v


class AgentMetrics(BaseModel):
    """Metrics for PydanticAI agent performance."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        frozen=False,
    )

    agent_id: str = Field(..., description="Agent identifier")
    total_requests: int = Field(default=0, description="Total requests processed")
    successful_requests: int = Field(default=0, description="Successful requests")
    failed_requests: int = Field(default=0, description="Failed requests")
    average_response_time_ms: float = Field(default=0.0, description="Average response time")
    total_tokens_used: int = Field(default=0, description="Total tokens used")
    total_cost_usd: float = Field(default=0.0, description="Total cost in USD")
    last_request_at: datetime | None = Field(None, description="Last request timestamp")
    error_rate: float = Field(default=0.0, description="Error rate (0-1)")
    uptime_percent: float = Field(default=100.0, description="Uptime percentage")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metrics")

    @field_validator("error_rate")
    @classmethod
    def validate_error_rate(cls, v: float) -> float:
        """Validate error rate is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Error rate must be between 0.0 and 1.0")
        return v

    @field_validator("uptime_percent")
    @classmethod
    def validate_uptime_percent(cls, v: float) -> float:
        """Validate uptime percentage is between 0 and 100."""
        if not 0.0 <= v <= 100.0:
            raise ValueError("Uptime percentage must be between 0.0 and 100.0")
        return v

"""Testing models for container-based test execution and quality analysis.

This module provides comprehensive Pydantic models for the testing system that supports
multi-language test execution in containerized environments. It includes models for
test requests, results, quality analysis, and container-based execution with security
controls and resource management.

The testing system supports:
- Multi-language test execution (Python, JavaScript, TypeScript, Go, Rust, Java, C#, C++)
- Quality analysis with automated tool integration
- Resource limits and security controls
- Container-based isolation and execution
- Comprehensive result reporting with metadata

Key Model Categories:
    Request Models: TestRequest, QualityRequest, QualityScanRequest
    Response Models: TestResult, QualityResult, ContainerTestResult
    Configuration Models: ResourceLimits, TestSecurityLevel, TestType
    Factory Functions: Helper functions for creating model instances

Example:
    ```python
    from testing_models import TestRequest, TestResult, TestSecurityLevel, TestType

    # Create a test request
    request = TestRequest(
        code='def add(a, b): return a + b',
        test_code='def test_add(): assert add(2, 3) == 5',
        language="python",
        test_type=TestType.UNIT,
        security_level=TestSecurityLevel.HIGH,
        run_quality_check=True,
        timeout=30
    )

    # Process test result
    result = TestResult(
        test_id="test-123",
        success=True,
        stdout="Test passed",
        stderr="",
        exit_code=0,
        execution_time=2.5,
        created_at=datetime.now(timezone.utc),
        language="python",
        security_level=TestSecurityLevel.HIGH
    )
    ```
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid7

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)


class TestType(StrEnum):
    __test__ = False
    """Test execution types."""

    UNIT = "unit"
    INTEGRATION = "integration"
    BENCHMARK = "benchmark"
    COVERAGE = "coverage"
    SECURITY = "security"
    DOC = "doc"
    CLIPPY = "clippy"
    FMT = "fmt"
    VET = "vet"
    RACE = "race"


class ResourceLimits(BaseModel):
    """Resource limits for test execution."""

    memory_mb: int | None = Field(None, ge=128, le=4096, description="Memory limit in MB")
    cpu_cores: float | None = Field(None, ge=0.1, le=4.0, description="CPU cores limit")
    timeout_seconds: int | None = Field(None, ge=1, le=3600, description="Timeout in seconds")


class TestSecurityLevel(StrEnum):
    __test__ = False
    """Security levels for test execution."""

    LOW = "low"  # Minimal security, maximum resources
    MEDIUM = "medium"  # Balanced security and performance
    HIGH = "high"  # Maximum security, minimal resources
    CRITICAL = "critical"  # Critical security level


# Backward compatibility alias
SecurityLevel = TestSecurityLevel


class TestRequest(BaseModel):
    """Request model for test execution."""

    __test__ = False  # Not a pytest test class

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
        ignored_types=(property,),  # Ignore computed fields in serialization
        json_schema_extra={
            "example": {
                "code": "def add(a, b):\n    return a + b",
                "test_code": "def test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0",
                "language": "python",
                "container_name": "my-project-tests",
                "test_type": "unit",
                "security_level": "high",
                "timeout": 30,
                "run_quality_check": True,
                "dependencies": {"pytest": "^7.0.0"},
                "resource_limits": {"memory_mb": 512, "cpu_cores": 1.0},
            }
        },
    )

    # Core test data
    code: str | None = Field(None, description="Code to test")
    test_code: str | None = Field(None, description="Test code to execute")
    language: str = Field(
        description="Programming language",
        json_schema_extra={
            "enum": [
                "python",
                "javascript",
                "typescript",
                "nodejs",
                "go",
                "rust",
                "java",
                "csharp",
                "cpp",
            ]
        },
    )

    # Execution parameters
    timeout: int | None = Field(None, ge=1, le=300, description="Test timeout in seconds")
    security_level: TestSecurityLevel = Field(
        default=TestSecurityLevel.HIGH,
        description="Security level (defaults to HIGH for security)",
    )
    with_hypothesis: bool = Field(default=False, description="Enable hypothesis testing (Python)")

    # File-based testing
    code_path: str | None = Field(None, description="Path to code file")
    code_file_path: str | None = Field(None, description="Alias for code file path")
    test_path: str | None = Field(None, description="Path to test file")
    test_file_path: str | None = Field(None, description="Alias for test file path")

    # Container preferences
    container_name: str | None = Field(None, description="Container name for test execution")

    # New fields for unification
    test_type: TestType | None = Field(TestType.UNIT, description="Type of test to run")
    dependencies: dict[str, str] = Field(default_factory=dict, description="Package dependencies")
    resource_limits: ResourceLimits | None = Field(None, description="Resource limits")

    # Quality integration
    run_quality_check: bool = Field(False, description="Run quality analysis after test")
    quality_tools: list[str] | None = Field(None, description="Specific quality tools to run")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate supported languages."""
        # Handle empty string
        if not v or v.strip() == "":
            raise ValueError("Language cannot be empty")

        # Extended to support all 9 languages
        supported = [
            "python",
            "javascript",
            "typescript",
            "nodejs",
            "go",
            "rust",
            "java",
            "csharp",
            "cpp",
        ]
        if v.lower() not in supported:
            raise ValueError(f"Language must be one of: {supported}")
        return v.lower()

    @model_validator(mode="after")
    def validate_test_request(self) -> TestRequest:
        """Ensure we have code and tests; default to scanning container root."""
        if not self.code and not self.code_path and not self.code_file_path:
            self.code_path = "."
        if not self.test_code and not self.test_path and not self.test_file_path:
            self.test_path = "."
        return self

    # Computed fields using Pydantic v2
    @computed_field
    def code_file_path_resolved(self) -> str | None:
        """Get code file path from either field."""
        return self.code_file_path or self.code_path

    @computed_field
    def test_file_path_resolved(self) -> str | None:
        """Get test file path from either field."""
        return self.test_file_path or self.test_path


class TestResult(BaseModel):
    """Result model for test execution."""

    __test__ = False  # Not a pytest test class

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        json_schema_serialization_defaults_required=True,
    )

    # Execution metadata
    test_id: str = Field(description="Unique test execution ID")
    success: bool = Field(description="Whether tests passed")

    # Output data
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error output")
    exit_code: int = Field(description="Process exit code")

    # Timing and context
    execution_time: float = Field(description="Execution time in seconds")
    created_at: datetime = Field(description="When test was executed")

    # Request context
    language: str = Field(description="Programming language used")
    security_level: TestSecurityLevel = Field(description="Security level used")

    # Optional additional data
    coverage_data: dict[str, Any] | None = Field(None, description="Code coverage data")
    test_metadata: dict[str, Any] | None = Field(None, description="Additional test metadata")

    # New fields for unification
    test_type: str | None = Field(None, description="Type of test executed")
    container_name: str | None = Field(None, description="Container used for execution")
    quality_report: QualityResult | None = Field(None, description="Quality analysis report")
    error: str | None = Field(None, description="Error message if test failed")

    @field_serializer("created_at")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        if v.tzinfo:
            return v.isoformat()
        return v.replace(tzinfo=UTC).isoformat()


class QualityRequest(BaseModel):
    """Request model for quality analysis."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
        ignored_types=(property,),  # Ignore computed fields in serialization
        json_schema_extra={
            "example": {
                "code": "import os\nimport sys\n\ndef process_data(data):\n    # TODO: implement processing\n    pass",
                "container_name": "my-python-project",
                "language": "python",
                "tools": ["ruff", "ty"],
                "fix_issues": True,
                "fail_on_issues": False,
                "custom_rules": {"max_line_length": 100, "allow_todos": False},
            }
        },
    )

    # Container scoping - REQUIRED for container-based operations
    container_name: str = Field(description="Container name for scoped quality analysis")

    # Code to analyze
    code: str | None = Field(None, description="Code content to analyze")
    code_content: str | None = Field(None, description="Alias for code content")
    code_path: str | None = Field(None, description="Path to code file")
    language: str | None = Field(None, description="Programming language")

    # Tool selection
    tools: list[str] | None = Field(None, description="Specific tools to run")

    # Analysis options
    fix_issues: bool = Field(default=False, description="Automatically apply resolves")
    auto_resolve: bool = Field(default=False, description="Automatically apply resolves (alias)")
    fail_on_issues: bool = Field(default=False, description="Fail if issues found")
    custom_rules: dict[str, Any] | None = Field(None, description="Custom rules configuration")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str | None) -> str | None:
        """Validate supported languages."""
        if v is None:
            return v
        # Handle empty string
        if v.strip() == "":
            raise ValueError("Language cannot be empty")
        supported = ["python", "javascript", "typescript", "nodejs"]
        if v.lower() not in supported:
            raise ValueError(f"Language must be one of: {supported}")
        return v.lower()

    @model_validator(mode="after")
    def validate_quality_request(self) -> QualityRequest:
        """Ensure we have code to analyze; default to scanning container root."""
        if not self.code and not self.code_content and not self.code_path:
            self.code_path = "."
        return self

    # Computed field using Pydantic v2
    @computed_field
    def code_content_resolved(self) -> str | None:
        """Get code content from either field."""
        return self.code_content or self.code


class QualityScanRequest(BaseModel):
    """Request model for container-wide quality scanning."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "container_name": "my-fastapi-backend",
                "scan_path": "/app/data",
                "recursive": True,
                "include_patterns": ["*.py", "*.js", "*.ts"],
                "exclude_patterns": [
                    "__pycache__",
                    "node_modules",
                    "*.min.js",
                ],
                "max_files": 500,
                "language": "python",
                "tools": ["ruff", "ty", "bandit"],
                "fix_issues": False,
                "parallel_analysis": True,
                "deep_analysis": True,
                "fail_on_critical": True,
                "min_quality_score": 80.0,
                "timeout": 1800,
            }
        },
    )

    # Container to scan
    container_name: str = Field(description="Container name to scan")

    # Scan parameters
    scan_path: str | None = Field(
        default="/app/data", description="Base path to scan within container"
    )
    recursive: bool = Field(default=True, description="Scan directories recursively")

    # File filtering
    include_patterns: list[str] = Field(
        default_factory=lambda: ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx"],
        description="File patterns to include in scan",
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "__pycache__",
            "*.pyc",
            ".git",
            "node_modules",
            "*.min.js",
        ],
        description="Patterns to exclude from scan",
    )
    max_files: int = Field(default=500, ge=1, le=2000, description="Maximum files to scan")

    # Language and tool selection
    language: str | None = Field(None, description="Primary language (auto-detect if None)")
    tools: list[str] | None = Field(
        None, description="Specific tools to run (all available if None)"
    )

    # Analysis options
    fix_issues: bool = Field(default=False, description="Attempt to apply resolves automatically")
    parallel_analysis: bool = Field(
        default=True, description="Run tools in parallel for performance"
    )
    deep_analysis: bool = Field(default=False, description="Enable comprehensive analysis mode")

    # Quality thresholds
    fail_on_critical: bool = Field(default=False, description="Fail scan if critical issues found")
    min_quality_score: float | None = Field(
        None, ge=0.0, le=100.0, description="Minimum acceptable quality score"
    )

    # Custom configuration
    custom_rules: dict[str, Any] | None = Field(None, description="Custom rules configuration")
    timeout: int | None = Field(default=3600, ge=30, le=7200, description="Scan timeout in seconds")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str | None) -> str | None:
        """Validate supported languages."""
        if v is None:
            return v
        # Handle empty string
        if v.strip() == "":
            raise ValueError("Language cannot be empty")
        supported = ["python", "javascript", "typescript", "nodejs"]
        if v.lower() not in supported:
            raise ValueError(f"Language must be one of: {supported}")
        return v.lower()


class QualityIssue(BaseModel):
    """Individual quality issue found by analysis tools."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid", str_strip_whitespace=True)

    tool: str = Field(description="Tool that found the issue")
    severity: str = Field(description="Issue severity (critical, error, warning, info)")
    message: str = Field(description="Issue description")
    file_path: str = Field(description="File path where issue was found")
    line_number: int | None = Field(None, description="Line number (if applicable)")
    column_number: int | None = Field(None, description="Column number (if applicable)")
    rule_id: str | None = Field(None, description="Rule ID that triggered issue")

    # Optional resolve suggestions
    suggested_fix: str | None = Field(None, description="Suggested resolve for the issue")
    fix_confidence: float | None = Field(None, ge=0.0, le=1.0, description="Confidence in resolve")


class QualityResult(BaseModel):
    """Result model for quality analysis."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    # Analysis metadata
    analysis_id: str = Field(description="Unique analysis ID")
    success: bool = Field(default=True, description="Whether analysis completed successfully")
    overall_score: float = Field(ge=0.0, le=100.0, description="Overall quality score (0-100)")
    total_issues: int = Field(ge=0, description="Total number of issues found")

    # Detailed results
    issues: list[QualityIssue] = Field(default=[], description="List of issues found")
    tools_results: dict[str, dict[str, Any]] = Field(default={}, description="Results by tool")

    # Execution context
    execution_time: float = Field(description="Analysis time in seconds")
    created_at: datetime = Field(description="When analysis was performed")
    language: str = Field(description="Programming language analyzed")

    # Error handling
    error: str | None = Field(None, description="Error message if analysis failed")

    @field_serializer("created_at")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        if v.tzinfo:
            return v.isoformat()
        return v.replace(tzinfo=UTC).isoformat()

    @model_validator(mode="after")
    def validate_quality_result(self) -> QualityResult:
        """Validate quality result consistency."""
        # Auto-correct missing error message for failed analysis
        if not self.success and (not self.error or self.error.strip() == ""):
            self.error = "Analysis failed without specific error details"

        if self.total_issues != len(self.issues):
            # Auto-correct for consistency
            self.total_issues = len(self.issues)
        return self

    @classmethod
    def create_failed_result(
        cls,
        analysis_id: str | None = None,
        error_message: str | None = None,
        language: str = "unknown",
        execution_time: float = 0.0,
    ) -> QualityResult:
        """Factory method to create failed QualityResult with safe error handling."""
        # Ensure analysis_id is provided
        if not analysis_id:
            analysis_id = str(uuid7())

        # Ensure error message is never None or empty
        if not error_message or error_message.strip() == "":
            error_message = "Quality analysis failed - no specific error details available"

        return cls(
            analysis_id=analysis_id,
            success=False,
            overall_score=0.0,
            total_issues=0,
            issues=[],
            tools_results={},
            execution_time=execution_time,
            created_at=datetime.now(UTC),
            language=language,
            error=error_message,
        )


class CommandResult(BaseModel):
    """Result of command execution in container."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    exit_code: int = Field(description="Command exit code")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error output")
    execution_time: float = Field(description="Execution time in seconds")
    timeout: bool = Field(default=False, description="Whether command timed out")


class ContainerTestResult(BaseModel):
    """Extended test result with container-specific information."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Base test result
    test_result: TestResult = Field(description="Core test execution result")

    # Container context
    container_name: str = Field(description="Container name used for execution")
    container_image: str = Field(description="Container image used")

    # Resource usage
    memory_used: int | None = Field(None, description="Peak memory usage in MB")
    cpu_time: float | None = Field(None, description="CPU time used in seconds")

    # Security context
    security_violations: list[str] = Field(default=[], description="Security violations detected")
    sandbox_effective: bool = Field(default=True, description="Whether sandbox was effective")


class TestExecutionStatus(StrEnum):
    __test__ = False
    """Status of test execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TestExecutionInfo(BaseModel):
    __test__ = False
    """Information about a test execution in progress."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    execution_id: str = Field(description="Unique execution ID")
    status: TestExecutionStatus = Field(description="Current execution status")
    started_at: datetime = Field(description="When execution started")
    updated_at: datetime = Field(description="Last status update")

    # Progress information
    progress_percentage: float | None = Field(
        None, ge=0.0, le=100.0, description="Execution progress"
    )
    current_step: str | None = Field(None, description="Current execution step")

    # Results (available when completed)
    result: TestResult | None = Field(None, description="Test result when completed")
    error_message: str | None = Field(None, description="Error message if failed")


# Factory functions for easy model creation
def create_test_request(
    code: str,
    language: str,
    container_name: str,
    test_code: str | None = None,
    security_level: TestSecurityLevel = TestSecurityLevel.MEDIUM,
    timeout: int | None = None,
    code_path: str | None = None,
    code_file_path: str | None = None,
    test_path: str | None = None,
    test_file_path: str | None = None,
    test_type: TestType | None = None,
    dependencies: dict[str, str] | None = None,
    resource_limits: ResourceLimits | None = None,
    run_quality_check: bool = False,
    quality_tools: list[str] | None = None,
) -> TestRequest:
    """Factory function to create test request."""
    return TestRequest(
        code=code,
        test_code=test_code,
        language=language,
        security_level=security_level,
        timeout=timeout,
        code_path=code_path,
        code_file_path=code_file_path,
        test_path=test_path,
        test_file_path=test_file_path,
        container_name=container_name,
        test_type=test_type,
        dependencies=dependencies or {},
        resource_limits=resource_limits,
        run_quality_check=run_quality_check,
        quality_tools=quality_tools,
    )


def create_quality_request(
    container_name: str,
    code: str | None = None,
    code_path: str | None = None,
    language: str | None = None,
    tools: list[str] | None = None,
) -> QualityRequest:
    """Factory function to create quality request.

    Note: This creates a QualityRequest (for single file analysis),
    not a QualityScanRequest (for container-wide scanning).
    """
    return QualityRequest(
        container_name=container_name,
        code=code,
        code_path=code_path,
        language=language,
        tools=tools,
    )


def create_test_result(
    success: bool,
    container_name: str,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    execution_time: float = 0.0,
    language: str = "python",
    security_level: TestSecurityLevel = TestSecurityLevel.MEDIUM,
    test_id: str | None = None,
    created_at: datetime | None = None,
    coverage_data: dict[str, Any] | None = None,
    test_metadata: dict[str, Any] | None = None,
    test_type: str | None = None,
    quality_report: QualityResult | None = None,
    error: str | None = None,
) -> TestResult:
    """Factory function to create test result."""
    return TestResult(
        test_id=test_id or str(uuid7()),
        success=success,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        execution_time=execution_time,
        created_at=created_at or datetime.now(UTC),
        coverage_data=coverage_data,
        test_metadata=test_metadata,
        language=language,
        security_level=security_level,
        test_type=test_type,
        container_name=container_name,
        quality_report=quality_report,
        error=error,
    )


# Rebuild model to ensure proper forward reference resolution
QualityResult.model_rebuild()

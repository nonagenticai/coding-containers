"""Security-specific Pydantic models for code security analysis and validation.

This module provides comprehensive data models for security analysis operations,
including code vulnerability detection, security violation reporting, and
security-aware code validation. The models support multi-level security
assessment and provide structured security analysis results.

The security analysis system enables:
- Code vulnerability detection with severity assessment
- Security violation reporting with remediation suggestions
- Multi-level security validation (LOW, MEDIUM, HIGH, CRITICAL)
- Context-aware security analysis with configurable rules
- Integration with security scanning tools and methodologies

Key Model Categories:
    Enumeration Models: SecurityLevel for security level classification
    Violation Models: SecurityViolation for detected security issues
    Request Models: SecurityValidationRequest, SecurityScanRequest, SecurityReportRequest
    Response Models: SecurityValidationResult, VulnerabilityScanResult

Security Features:
    - Pattern-based vulnerability detection
    - Severity-based issue classification
    - Remediation suggestion generation
    - Context-aware security analysis
    - Configurable security validation levels

Example:
    ```python
    from security_models import (
        SecurityLevel, SecurityViolation,
        SecurityValidationRequest, SecurityValidationResult
    )

    # Create security validation request
    request = SecurityValidationRequest(
        code='import os\\nos.system("rm -rf /")',
        language="python",
        security_level=SecurityLevel.HIGH,
        context={"environment": "production"}
    )

    # Security violation detection
    violation = SecurityViolation(
        type="command_injection",
        severity=SecurityLevel.CRITICAL,
        line_number=2,
        pattern_matched="os.system",
        description="Dangerous system command execution detected",
        suggested_fix="Use subprocess.run with shell=False instead"
    )

    # Security validation result
    result = SecurityValidationResult(
        is_safe=False,
        violations=[violation],
        risk_score=95.0,
        sandbox_config={"network_access": False, "file_system": "read_only"}
    )
    ```
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SecurityLevel(StrEnum):
    """Enumeration of security levels for code validation and risk assessment.

    Defines the security validation levels that determine the depth and strictness
    of security analysis performed on code. Each level applies different security
    rules and has different performance characteristics.

    Attributes:
        LOW: Basic security checks with minimal performance impact.
            Suitable for development environments and low-risk code.
        MEDIUM: Standard security validation with balanced performance.
            Appropriate for most production code and regular workflows.
        HIGH: Comprehensive security analysis with thorough validation.
            Recommended for security-sensitive applications and critical systems.
        CRITICAL: Maximum security scrutiny with intensive analysis.
            Required for high-security environments and critical infrastructure.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityViolation(BaseModel):
    """Model representing a security violation detected in code analysis.

    Represents a specific security issue found during code analysis, including
    details about the violation type, severity, location, and suggested remediation.
    This model enables structured reporting of security findings with actionable
    information for developers.

    Attributes:
        type (str): The category of security violation (e.g., "sql_injection",
            "command_injection", "xss", "hardcoded_credentials").
        severity (SecurityLevel): The severity level of the violation, determining
            the urgency and priority of remediation.
        line_number (Optional[int]): The line number in the source code where
            the violation was detected, if applicable.
        pattern_matched (str): The specific pattern, keyword, or code construct
            that triggered the security violation detection.
        description (str): A human-readable description of the security issue,
            explaining the potential risk and impact.
        suggested_fix (Optional[str]): A recommended approach or code change
            to remediate the security violation.
    """

    type: str = Field(..., description="Type of security violation")
    severity: SecurityLevel = Field(..., description="Severity level")
    line_number: int | None = Field(None, description="Line number where violation occurs")
    pattern_matched: str = Field(..., description="Pattern that triggered the violation")
    description: str = Field(..., description="Human-readable description")
    suggested_fix: str | None = Field(None, description="Suggested remediation")


class SecurityValidationRequest(BaseModel):
    """Request model for security validation of code snippets.

    Represents a request to perform security analysis on code snippets,
    with configurable security levels and contextual information to guide
    the analysis process.

    Attributes:
        code (str): The source code to be analyzed for security violations.
        language (str): The programming language of the code (e.g., "python",
            "javascript", "java").
        security_level (SecurityLevel): The required security validation level,
            defaults to MEDIUM for balanced performance and security.
        context (Optional[Dict[str, Any]]): Additional context information
            that may influence the security analysis, such as environment
            settings, deployment context, or specific security requirements.
        container_name (Optional[str]): Optional container name for context.
            Note: This field is accepted but not used in validation.
    """

    code: str = Field(..., description="Code to validate")
    language: str = Field(..., description="Programming language")
    security_level: SecurityLevel = Field(
        SecurityLevel.MEDIUM, description="Required security level"
    )
    context: dict[str, Any] | None = Field(None, description="Additional context for validation")
    container_name: str | None = Field(
        None, description="Optional container name (accepted but not used)"
    )


class SecurityValidationResult(BaseModel):
    """Result model for security validation outcomes.

    Represents the comprehensive results of security analysis performed on code,
    including safety assessment, detailed violations, risk scoring, and
    recommended security configurations.

    Attributes:
        is_safe (bool): Whether the code passes security validation at the
            requested security level. False indicates security violations
            were found that require attention.
        violations (List[SecurityViolation]): A list of security violations
            detected during analysis, empty if no violations were found.
        risk_score (float): Overall risk score from 0-100, where 0 represents
            no risk and 100 represents maximum risk. Higher scores indicate
            more severe security concerns.
        sandbox_config (Optional[Dict[str, Any]]): Recommended sandbox
            configuration for safe execution of the code, including resource
            limits, network restrictions, and file system access controls.
    """

    is_safe: bool = Field(..., description="Whether code passes security validation")
    violations: list[SecurityViolation] = Field(default_factory=list)
    risk_score: float = Field(..., ge=0, le=100, description="Overall risk score")
    sandbox_config: dict[str, Any] | None = Field(
        None, description="Recommended sandbox configuration"
    )


class SecurityScanRequest(BaseModel):
    """Request model for initiating a security scan on a container.

    Represents a request to perform security scanning on a specified container,
    with configurable scan types and dependency scanning options.

    Attributes:
        container_name (str): The name of the container to scan for security
            vulnerabilities.
        scan_type (Optional[str]): The type of security scan to perform.
            Defaults to "full" for comprehensive scanning. Other options may
            include "quick", "dependencies", or "custom" depending on the
            security scanning implementation.
        include_dependencies (bool): Whether to include dependency scanning
            as part of the security scan. Defaults to True to ensure
            comprehensive vulnerability detection in third-party packages.
    """

    container_name: str = Field(..., description="Name of the container to scan")
    scan_type: str | None = Field("full", description="Type of security scan to perform")
    include_dependencies: bool = Field(True, description="Whether to include dependency scanning")


class SecurityReportRequest(BaseModel):
    """Request model for generating a security report for a container.

    Represents a request to generate a formatted security report containing
    the results of security scans performed on a container.

    Attributes:
        container_name (str): The name of the container for which to generate
            the security report.
        format (str): The output format for the security report. Defaults to
            "json" for structured data output. Other options may include "html",
            "pdf", "text", or "xml" depending on the reporting implementation.
    """

    container_name: str = Field(..., description="Name of the container for security report")
    format: str = Field("json", description="Output format for the report")


class VulnerabilityScanResult(BaseModel):
    """Result model for vulnerability scanning operations.

    Represents the comprehensive results of a vulnerability scan performed on
    a container, including all detected vulnerabilities and timing information.

    Attributes:
        container_name (str): The name of the container that was scanned.
        vulnerabilities (List[Dict]): A list of detected vulnerabilities, where
            each dictionary contains vulnerability details such as CVE identifiers,
            severity levels, affected packages, and remediation information.
        scan_timestamp (datetime): The timestamp when the vulnerability scan
            was completed, providing traceability for security audit purposes.
    """

    container_name: str = Field(..., description="Name of the scanned container")
    vulnerabilities: list[dict] = Field(
        default_factory=list, description="List of detected vulnerabilities"
    )
    scan_timestamp: datetime = Field(..., description="Timestamp when the scan was completed")


class OSVPackageInfo(BaseModel):
    """Model representing package information for OSV.dev queries.

    Attributes:
        name (str): Package name.
        ecosystem (str): Package ecosystem (npm, PyPI, crates.io, Go, Maven, etc).
    """

    name: str = Field(..., description="Package name")
    ecosystem: str = Field(..., description="Package ecosystem")


class OSVVulnerability(BaseModel):
    """Model representing an OSV.dev vulnerability.

    Attributes:
        id (str): Unique vulnerability identifier.
        summary (str): Brief summary of the vulnerability.
        details (Optional[str]): Detailed description of the vulnerability.
        severity (Optional[str]): Severity level (CRITICAL, HIGH, MEDIUM, LOW).
        affected_versions (List[str]): List of affected package versions.
        fixed_versions (Optional[List[str]]): Versions where the issue is fixed.
        cve_ids (List[str]): Associated CVE identifiers.
        references (List[str]): Links to additional information.
        published_date (Optional[str]): Publication date of the vulnerability.
        package_name (str): Name of the affected package.
        ecosystem (str): Package ecosystem.
    """

    id: str = Field(..., description="Vulnerability identifier (e.g., GHSA-xxxx)")
    summary: str = Field(..., description="Vulnerability summary")
    details: str | None = Field(None, description="Detailed description")
    severity: str | None = Field(None, description="Severity level")
    affected_versions: list[str] = Field(default_factory=list, description="Affected versions")
    fixed_versions: list[str] | None = Field(None, description="Versions where issue is fixed")
    cve_ids: list[str] = Field(default_factory=list, description="Associated CVE IDs")
    references: list[str] = Field(default_factory=list, description="Reference URLs")
    published_date: str | None = Field(None, description="Publication date")
    package_name: str = Field(..., description="Affected package name")
    ecosystem: str = Field(..., description="Package ecosystem")


class PackageInfo(BaseModel):
    """Model representing package information for batch queries.

    Attributes:
        name (str): Package name.
        version (str): Package version.
        ecosystem (str): Package ecosystem.
    """

    name: str = Field(..., description="Package name")
    version: str = Field(..., description="Package version")
    ecosystem: str = Field(..., description="Package ecosystem")

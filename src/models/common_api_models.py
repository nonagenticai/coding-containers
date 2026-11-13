"""Common API Models for V2 API.

This module contains shared Pydantic models used across all V2 API endpoints.
These models provide a consistent structure for API responses, error handling,
pagination, and health monitoring throughout the application.

The models ensure:
    - Type-safe request/response handling with comprehensive validation
    - Standardized response formats across all endpoints
    - Consistent error reporting and status tracking
    - Reusable pagination and health check structures
    - Automatic timestamp generation and request ID tracking

Example:
    ```python
    from common_api_models import BaseResponse, ResponseStatus

    response = BaseResponse(
        status=ResponseStatus.SUCCESS,
        message="Operation completed successfully"
    )
    ```

Note:
    All models use Pydantic v2 with ConfigDict for configuration and
    include comprehensive JSON schema examples for OpenAPI documentation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid7

from pydantic import BaseModel, ConfigDict, Field, field_validator

from coding_client.types.constraint_types import SAFE_STRING_PATTERN


class ResponseStatus(StrEnum):
    """Enumeration of standard API response status values.

    This enum defines all possible status values that can be returned in API
    responses, providing a consistent vocabulary for operation states across
    the entire API surface.

    Attributes:
        SUCCESS: Operation completed successfully without errors
        ERROR: Operation failed with an error condition
        PENDING: Operation has been queued but not yet started
        RUNNING: Operation is currently executing
        COMPLETED: Long-running operation has finished successfully
        FAILED: Long-running operation has terminated with failure

    Note:
        SUCCESS/ERROR are typically used for synchronous operations,
        while PENDING/RUNNING/COMPLETED/FAILED are used for async operations.
    """

    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BaseResponse(BaseModel):
    """Base response model providing standard fields for all V2 API responses.

    This model serves as the foundation for all API responses, ensuring
    consistent structure and metadata across endpoints. It automatically
    generates timestamps and unique request IDs for tracking and debugging.

    Attributes:
        status (ResponseStatus): The operation status indicating success or failure
        message (str): Human-readable description of the operation result
        timestamp (datetime): UTC timestamp when the response was generated
        request_id (str): Unique identifier for request tracking and correlation

    Example:
        ```python
        response = BaseResponse(
            status=ResponseStatus.SUCCESS,
            message="User created successfully"
        )
        # timestamp and request_id are auto-generated
        ```

    Note:
        The timestamp is always in UTC timezone to ensure consistency.
        Request IDs follow the format "req_[8-char-hex]" for brevity.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Operation completed successfully",
                "timestamp": "2025-07-13T10:30:00Z",
                "request_id": "req_12345",
            }
        }
    )

    status: ResponseStatus = Field(..., description="Response status")
    message: str = Field(..., description="Human-readable message", pattern=SAFE_STRING_PATTERN)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Response timestamp",
    )
    request_id: str = Field(
        default_factory=lambda: f"req_{uuid7().hex[:8]}",
        min_length=1,
        description="Unique request identifier",
    )


class SuccessResponse(BaseResponse):
    """Standard success response format with optional data payload.

    This model extends BaseResponse for successful operations that may
    return additional data. It provides a consistent wrapper for all
    successful API operations.

    Attributes:
        status (ResponseStatus): Always set to SUCCESS for successful responses
        data (Optional[Dict[str, Any]]): Optional payload containing response data

    Example:
        ```python
        response = SuccessResponse(
            message="User retrieved successfully",
            data={"user_id": "123", "name": "John Doe", "email": "john@example.com"}
        )
        ```

    Note:
        The data field is optional and can contain any JSON-serializable structure.
        For paginated responses, use PaginatedResponse instead.
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    data: dict[str, Any] | None = Field(None, description="Response data")


class PaginationParams(BaseModel):
    """Standard pagination parameters for list endpoints.

    This model defines common pagination parameters used across all
    endpoints that return lists of items. It ensures consistent
    pagination behavior throughout the API.

    Attributes:
        page (int): Page number to retrieve (1-based indexing)
        limit (int): Number of items per page (default: 50, max: 1000)

    Example:
        ```python
        params = PaginationParams(page=2, limit=25)
        # Retrieves items 26-50 from the result set
        ```

    Note:
        Page numbers start at 1, not 0, for better user experience.
        The limit is capped at 1000 to prevent performance issues.
    """

    model_config = ConfigDict(json_schema_extra={"example": {"page": 1, "limit": 20}})

    page: int = Field(1, ge=1, description="Page number (1-based)")
    limit: int = Field(50, ge=1, le=1000, description="Items per page")


class PaginatedResponse(SuccessResponse):
    """Response format for paginated list endpoints.

    This model extends SuccessResponse to provide pagination metadata
    alongside the actual items. It ensures consistent pagination
    structure across all list-returning endpoints.

    Attributes:
        pagination (Dict[str, Any]): Metadata about pagination state
        items (List[Any]): The actual items for the current page

    The pagination dict must contain:
        - page: Current page number
        - limit: Items per page
        - total: Total number of items across all pages
        - pages: Total number of pages

    Example:
        ```python
        response = PaginatedResponse(
            message="Users retrieved successfully",
            pagination={"page": 2, "limit": 20, "total": 95, "pages": 5},
            items=[{"id": 21, "name": "User 21"}, ...]
        )
        ```

    Note:
        The items list contains only the items for the current page.
        Pagination metadata helps clients navigate through result sets.
    """

    pagination: dict[str, Any] = Field(..., description="Pagination metadata")
    items: list[Any] = Field(..., description="Response items")

    @field_validator("pagination")
    @classmethod
    def validate_pagination(cls, v):
        """Validate the pagination dictionary to ensure it contains required keys such as page, limit, total, and pages.

        Args:
            v (dict): The pagination dictionary to validate.

        Raises:
            ValueError: If any required keys are missing.
        """
        required_keys = {"page", "limit", "total", "pages"}
        if not all(key in v for key in required_keys):
            raise ValueError(f"Pagination must contain: {required_keys}")
        return v


class HealthStatus(StrEnum):
    """Enumeration of system and component health status values.

    This enum defines standardized health states for system components,
    enabling consistent health monitoring and alerting across the platform.

    Attributes:
        HEALTHY: Component is functioning normally with no issues
        DEGRADED: Component is functional but experiencing performance issues
        UNHEALTHY: Component is not functioning properly or is unavailable
        UNKNOWN: Component health cannot be determined (connectivity issues)

    Note:
        DEGRADED status indicates the component works but may have reduced
        performance or partial functionality. Use this for warning conditions.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth(BaseModel):
    """Health status information for individual system components.

    This model provides detailed health information for system components
    such as databases, external services, and internal subsystems.

    Attributes:
        name (str): Human-readable component name
        status (HealthStatus): Current health status of the component
        message (Optional[str]): Additional status description or error details
        last_check (datetime): Timestamp of the most recent health check
        metrics (Optional[Dict[str, Any]]): Component-specific health metrics

    Example:
        ```python
        db_health = ComponentHealth(
            name="Database",
            status=HealthStatus.HEALTHY,
            message="All connections active",
            metrics={"active_connections": 15, "response_time_ms": 12.5}
        )
        ```

    Note:
        The last_check timestamp is automatically set to current UTC time if not provided.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Database",
                "status": "healthy",
                "message": "All connections active",
                "last_check": "2025-07-13T10:30:00Z",
                "metrics": {"connections": 15, "response_time_ms": 12.5},
            }
        }
    )

    name: str = Field(..., min_length=1, pattern=SAFE_STRING_PATTERN, description="Component name")
    status: HealthStatus = Field(..., description="Health status")
    message: str | None = Field(None, pattern=SAFE_STRING_PATTERN, description="Status message")
    last_check: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last health check",
    )
    metrics: dict[str, Any] | None = Field(None, description="Component metrics")

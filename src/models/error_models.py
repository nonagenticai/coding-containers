"""Error Response Models for API v2."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from coding_client.types.constraint_types import SAFE_STRING_PATTERN
from src.models.common_api_models import ResponseStatus


class StandardErrorResponse(BaseModel):
    """Standard error response format for all API errors.

    This model defines the base structure for all error responses in the V2 API,
    ensuring consistent error reporting that matches OpenAPI schema requirements
    and passes schemathesis validation tests.

    Attributes:
        status (ResponseStatus): Always 'error' for error responses
        message (str): Human-readable error description for display to users
        timestamp (datetime): ISO 8601 timestamp of when the error occurred
        request_id (str): Unique identifier for request tracking and correlation
        path (str): API endpoint path where the error occurred
        error_code (Optional[str]): Machine-readable error code for programmatic handling
        details (Optional[Dict[str, Any]]): Additional context-specific error information

    Example:
        ```python
        error = StandardErrorResponse(
            message="Database connection failed",
            path="/api/users/123",
            request_id="req_550e8400",
            error_code="DB_CONNECTION_ERROR",
            details={"database": "users_db", "timeout": 30}
        )
        ```

    Note:
        The timestamp is automatically set to current UTC time if not provided.
        Error codes should follow UPPER_SNAKE_CASE naming convention.
    """

    status: ResponseStatus = Field(
        default=ResponseStatus.ERROR,
        description="Response status (always 'error' for error responses)",
    )
    message: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="Human-readable error message"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="ISO 8601 timestamp of when the error occurred",
    )
    request_id: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Unique request identifier for tracking",
    )
    path: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="API endpoint path where the error occurred",
    )
    error_code: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code for programmatic handling",
    )
    details: dict[str, Any] | None = Field(None, description="Additional error context and details")

    @field_serializer("timestamp")
    def serialize_timestamp(self, timestamp: datetime) -> str:
        """Serialize timestamp to ISO 8601 format with timezone information.

        This ensures all timestamps in error responses follow a consistent format
        that includes timezone information for proper interpretation by clients.

        Args:
            timestamp (datetime): The timestamp to serialize

        Returns:
            str: ISO 8601 formatted timestamp string with timezone

        Note:
            If the timestamp lacks timezone info, UTC is assumed.
        """
        if timestamp.tzinfo is None:
            # If no timezone info, assume UTC
            timestamp = timestamp.replace(tzinfo=UTC)
        return timestamp.isoformat()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "error",
                "message": "Resource not found",
                "timestamp": "2025-01-12T10:30:00.000Z",
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "path": "/api/workspaces/my-container",
                "error_code": "RESOURCE_NOT_FOUND",
                "details": {
                    "resource_type": "container",
                    "resource_id": "my-container",
                },
            }
        }
    )


class APIValidationErrorDetail(BaseModel):
    """Detailed information about a specific field validation error.

    This model provides granular information about validation failures,
    enabling clients to understand exactly what went wrong with each field
    and potentially provide inline error feedback to users.

    Attributes:
        field (str): The field path that failed validation (e.g., "user.email")
        message (str): Human-readable description of the validation failure
        type (str): Type of validation error (e.g., "format", "required", "range")
        input (Any): The actual value that failed validation (for debugging)

    Example:
        ```python
        error_detail = APIValidationErrorDetail(
            field="age",
            message="Age must be between 0 and 150",
            type="range_error",
            input=200
        )
        ```

    Note:
        The field path uses dot notation for nested fields (e.g., "address.zipcode").
        The input value is included to help with debugging but may be omitted
        for sensitive fields like passwords.
    """

    field: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Field that failed validation")
    message: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Validation error message")
    type: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Type of validation error")
    input: Any = Field(None, description="The invalid input value")


class ValidationErrorResponse(StandardErrorResponse):
    """Specialized error response for request validation failures.

    This model extends StandardErrorResponse to provide detailed information
    about validation errors, including field-level error details. It's used
    when request data fails Pydantic model validation.

    Attributes:
        status (ResponseStatus): Always 'error' for validation errors
        message (str): Default "Request validation failed"
        error_code (str): Default "VALIDATION_ERROR"
        validation_errors (List[APIValidationErrorDetail]): List of field-specific errors

    Example:
        ```python
        validation_error = ValidationErrorResponse(
            path="/api/users",
            request_id="req_123456",
            validation_errors=[
                APIValidationErrorDetail(
                    field="email",
                    message="Invalid email format",
                    type="format_error",
                    input="not-an-email"
                ),
                APIValidationErrorDetail(
                    field="age",
                    message="Age must be a positive integer",
                    type="type_error",
                    input=-5
                )
            ]
        )
        ```

    Note:
        This response type is typically returned with HTTP status code 422
        (Unprocessable Entity) to indicate validation failures.
    """

    status: ResponseStatus = Field(default=ResponseStatus.ERROR, description="Response status")
    message: str = Field(
        default="Request validation failed",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str = Field(
        default="VALIDATION_ERROR",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )
    validation_errors: list[APIValidationErrorDetail] = Field(
        ..., description="List of validation errors"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "error",
                "message": "Request validation failed",
                "timestamp": "2025-01-12T10:30:00.000Z",
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "path": "/api/analysis/analyze",
                "error_code": "VALIDATION_ERROR",
                "validation_errors": [
                    {
                        "field": "file_path",
                        "message": "Field required",
                        "type": "missing",
                        "input": None,
                    }
                ],
            }
        }
    )


# Predefined error responses for common HTTP status codes
class Error400Response(StandardErrorResponse):
    """Bad Request (400) error response model.

    Used when the request is malformed or contains invalid syntax that
    prevents the server from understanding it. This differs from validation
    errors (422) which occur when the request is well-formed but contains
    invalid data.

    Common scenarios:
        - Malformed JSON in request body
        - Invalid query parameter format
        - Missing required headers
        - Invalid HTTP method for endpoint
    """

    message: str = Field(
        default="Bad request",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str = Field(
        default="BAD_REQUEST",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error401Response(StandardErrorResponse):
    """Unauthorized (401) error response model.

    Used when authentication is required but missing or invalid. This indicates
    the client must authenticate itself to get the requested response.

    Common scenarios:
        - Missing authentication token/credentials
        - Expired authentication token
        - Invalid authentication credentials
        - Token lacks required scopes
    """

    message: str = Field(
        default="Authentication required",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str = Field(
        default="UNAUTHORIZED",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error403Response(StandardErrorResponse):
    """Forbidden (403) error response model.

    Used when the client is authenticated but lacks permission to access
    the requested resource. Unlike 401, re-authenticating will not help.

    Common scenarios:
        - User lacks required role or permission
        - Resource belongs to another user
        - Operation not allowed in current context
        - API key lacks required permissions
    """

    message: str = Field(
        default="Insufficient permissions",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str = Field(
        default="FORBIDDEN",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error404Response(StandardErrorResponse):
    """Not Found (404) error response model.

    Used when the requested resource does not exist or cannot be found.
    This may also be used to hide the existence of resources the client
    lacks permission to know about.

    Common scenarios:
        - Resource with given ID doesn't exist
        - Endpoint path is incorrect
        - Resource was deleted
        - Resource hidden due to permissions
    """

    message: str = Field(
        default="Resource not found",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str = Field(
        default="NOT_FOUND",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error405Response(StandardErrorResponse):
    """Method Not Allowed error response."""

    message: str = Field(
        default="Method not allowed",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str = Field(
        default="METHOD_NOT_ALLOWED",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error406Response(StandardErrorResponse):
    """Not Acceptable (406) error response model.

    Used when the server cannot produce a response matching the list of
    acceptable values defined in the request's Accept headers. This indicates
    a content negotiation failure between client and server.

    Common scenarios:
        - Client requests unsupported media type
        - Missing or invalid Accept header
        - Requested format not available
        - API version mismatch in Accept header
    """

    message: str = Field(
        default="Not acceptable",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str = Field(
        default="NOT_ACCEPTABLE",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error408Response(StandardErrorResponse):
    """Request Timeout (408) error response model.

    Used when the server didn't receive a complete request message within
    the time it was prepared to wait. This is typically used for long-running
    operations that exceed reasonable time limits.

    Common scenarios:
        - Long-running repository provisioning operations timeout
        - Database queries that exceed timeout limits
        - External API calls that don't respond in time
        - File processing operations that take too long

    Note:
        The client may repeat the request without modifications at any later time.
        Consider including guidance on retry timing in the response message.
    """

    message: str = Field(
        default="Request timeout",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str = Field(
        default="REQUEST_TIMEOUT",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error409Response(StandardErrorResponse):
    """Conflict error response."""

    message: str = Field(
        default="Resource conflict",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str = Field(
        default="CONFLICT",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error412Response(StandardErrorResponse):
    """Precondition Failed (412) error response.

    Raised when the caller must fix something outside the request before it can
    succeed. Today's only producer: a container provision for a repository with
    no usable git credential — the repo is private (or its visibility could not
    be determined) and neither a BYO PAT nor a GitHub App installation token is
    available. The fix is on the caller's side (install the nonagentic GitHub App
    on the owner from the fr workspace, or pass a GitHub token as ``access_token``
    on the provision request), so it is a precondition, not a server fault.
    """

    message: str = Field(
        default="Precondition failed",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str = Field(
        default="PRECONDITION_FAILED",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error422Response(ValidationErrorResponse):
    """Unprocessable Entity (422) error response for validation failures.

    Used when the request is well-formed but contains semantic errors.
    This is the standard response for Pydantic validation failures and
    indicates the request data doesn't meet business logic requirements.

    This class is an alias for ValidationErrorResponse to match HTTP
    status code naming conventions.
    """


class Error429Response(StandardErrorResponse):
    """Too Many Requests (429) error response model.

    Used when the client has sent too many requests in a given amount of time
    and has exceeded the rate limit. This is part of rate limiting to prevent
    abuse and ensure fair resource usage.

    Common scenarios:
        - API rate limit exceeded (per user/IP/key)
        - Bulk operation rate limiting
        - Resource-intensive endpoint throttling
        - DDoS protection activation

    Note:
        Should include rate limit headers when possible:
        - X-RateLimit-Limit: Request limit
        - X-RateLimit-Remaining: Remaining requests
        - X-RateLimit-Reset: Time when limit resets
        - Retry-After: Seconds to wait before retrying
    """

    message: str = Field(
        default="Too many requests",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str = Field(
        default="RATE_LIMIT_EXCEEDED",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error500Response(StandardErrorResponse):
    """Internal Server Error (500) response model.

    Used when an unexpected error occurs on the server side. These errors
    indicate bugs or infrastructure issues that prevent request processing.

    Common scenarios:
        - Unhandled exceptions in request processing
        - Database connection failures
        - External service unavailable
        - Configuration errors

    Note:
        Error details should be limited in production to avoid exposing
        sensitive information about the system internals.
    """

    message: str = Field(
        default="Internal server error",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str = Field(
        default="INTERNAL_ERROR",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error503Response(StandardErrorResponse):
    """Service Unavailable (503) error response model.

    Used when the service is temporarily unable to handle requests.
    This typically indicates a temporary condition that will be resolved.

    Common scenarios:
        - Service under maintenance
        - Temporary overload
        - Dependency services unavailable
        - Rate limiting (though 429 is preferred)

    Note:
        Consider including a Retry-After header when possible to indicate
        when the client should retry the request.
    """

    message: str = Field(
        default="Service temporarily unavailable",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str = Field(
        default="SERVICE_UNAVAILABLE",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


# Aliases for backwards compatibility
APIError = StandardErrorResponse
ValidationError = ValidationErrorResponse
NotFoundError = Error404Response
InternalServerError = Error500Response

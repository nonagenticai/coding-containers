"""Agent execution models for code_graph_helper.

Models for executing CLI agents (Claude Code, Codex) in repository containers
via delegated execution from worker_management.
"""

from pydantic import BaseModel, Field


class AgentExecutionRequest(BaseModel):
    """Request to execute CLI agent in a container."""

    cli: str = Field(..., description="CLI to use: 'claude' or 'codex'")
    prompt: str = Field(..., description="Task prompt for the agent")
    system_prompt: str | None = Field(default=None, description="System prompt for the agent")
    model: str | None = Field(default=None, description="Model to use (e.g., 'sonnet', 'opus')")
    workspace: str = Field(default="/workspace", description="Working directory in container")
    timeout: int = Field(default=300, description="Execution timeout in seconds")


class AgentExecutionResponse(BaseModel):
    """Response from agent execution."""

    success: bool
    output: str
    error: str | None = None
    exit_code: int
    execution_time: float
    container_name: str
    # Claude Code JSON output fields (from --output-format json)
    session_id: str | None = Field(
        default=None, description="Session ID for conversation continuity"
    )
    cost_usd: float | None = Field(default=None, description="Total cost in USD for this execution")
    duration_ms: int | None = Field(
        default=None, description="Execution duration in milliseconds (from CLI)"
    )
    usage: dict | None = Field(default=None, description="Token usage metrics")


class AgentCapabilityStatus(BaseModel):
    """Availability status for a single CLI agent inside a container."""

    available: bool = Field(..., description="Whether the CLI is available in the container")
    note: str = Field(..., description="Human-readable explanation of the availability result")


class AgentCapabilitiesResponse(BaseModel):
    """Response model for agent capability checks."""

    container_name: str = Field(..., description="Container name")
    capabilities: dict[str, AgentCapabilityStatus] = Field(
        ..., description="Capability map keyed by CLI name"
    )

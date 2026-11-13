"""One interface, and one place to choose, for running commands in a workspace."""

from __future__ import annotations

from .agent_host import AgentHostRuntime
from .docker import DockerRuntime
from .kubernetes import KubernetesRuntime
from .policy import (
    ANY_TOOLCHAIN,
    CommandNotPermittedError,
    check_dangerous_patterns,
    enforce_command_policy,
    validate_command,
)
from .protocol import DEFAULT_OUTPUT_LIMIT, WorkspaceRuntime
from .resolve import resolve_runtime, resolve_runtime_kind
from .types import (
    ExecutionProfile,
    RunOnceResult,
    RuntimeCapabilities,
    RuntimeKind,
    UnsupportedCapabilityError,
)

__all__ = [
    "ANY_TOOLCHAIN",
    "DEFAULT_OUTPUT_LIMIT",
    "AgentHostRuntime",
    "CommandNotPermittedError",
    "DockerRuntime",
    "ExecutionProfile",
    "KubernetesRuntime",
    "RunOnceResult",
    "RuntimeCapabilities",
    "RuntimeKind",
    "UnsupportedCapabilityError",
    "WorkspaceRuntime",
    "check_dangerous_patterns",
    "enforce_command_policy",
    "resolve_runtime",
    "resolve_runtime_kind",
    "validate_command",
]

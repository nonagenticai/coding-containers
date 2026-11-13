"""Module for defining container models used in the application."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core.toolchain import (
    MAX_EXPECTED_REPOS,
    MAX_EXPECTED_SERVICES,
    SERVICE_MEMORY_MAX_MI,
    SERVICE_MEMORY_MIN_MI,
)


class ContainerStatus(StrEnum):
    """Container status enumeration"""

    CREATING = "creating"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    REMOVING = "removing"


class AnalysisConfig(BaseModel):
    """Configuration for code analysis during provisioning."""

    include_tests: bool | None = Field(
        None,
        description="Whether to include test files in code graph analysis",
    )
    language_hint: str | None = Field(
        None,
        description="Primary programming language hint for better analysis",
    )
    exclude_patterns: list[str] | None = Field(
        None, description="Additional patterns to exclude from analysis"
    )
    include_patterns: list[str] | None = Field(
        None, description="Patterns to include despite other exclusions"
    )


class GitHubRepoRequest(BaseModel):
    """Request model for GitHub repository operations."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    repo_url: str = Field(
        ...,
        min_length=1,
        description="GitHub repository URL",
        pattern=r"^https://github\.com/[A-Za-z0-9\-_.]+/[A-Za-z0-9\-_.]+(?:/pull/\d+)?(?:\.git)?/?$",
        json_schema_extra={"example": "https://github.com/georgiedekker/enterprise_mcp_server"},
    )
    branch: str = Field(
        "main",
        min_length=1,
        max_length=256,
        description="Git branch to clone",
        pattern=r"^[a-zA-Z0-9_./-]+$",
        json_schema_extra={"example": "main"},
    )
    container_name: str | None = Field(
        None,
        description="Optional container name",
        pattern=r"^[a-zA-Z][a-zA-Z0-9\-_]{0,62}$",
        json_schema_extra={"example": "enterprise-mcp-server"},
    )
    description: str | None = Field(None, description="Optional description", max_length=500)
    access_token: str | None = Field(
        None,
        description="GitHub Personal Access Token for private repositories",
        json_schema_extra={"example": "<github-personal-access-token>"},
    )
    user_id: str | None = Field(
        None,
        description="User ID for multi-account workflow (defaults to anonymous if not provided)",
        pattern=r"^[a-zA-Z0-9\-_\.@]+$",
        max_length=128,
        json_schema_extra={"example": "user-123"},
    )
    force_rebuild: bool | None = Field(
        False, description="Force recreation if container already exists"
    )
    analysis_config: AnalysisConfig | None = Field(
        None, description="Configuration for code analysis during provisioning"
    )
    pr_number: int | None = Field(
        None, description="Pull request number to provision", ge=1, le=999999
    )
    provision_type: Literal["branch", "pull_request"] = Field(
        "branch", description="Type of provisioning"
    )
    secret_config_id: str | None = Field(
        None,
        description="Client ID for retrieving environment variables from secrets store",
        pattern=r"^[a-zA-Z0-9\-_\.]+$",
        max_length=128,
        json_schema_extra={"example": "repo-myorg-myproject"},
    )
    tenant_id: str | None = Field(
        None,
        description=(
            "Provisioning tenant; stamped onto the Deployment as a 'tenant-id' label and onto "
            "the ContainerInfo so tenant scoping survives even when the JSON registry entry is "
            "transiently lost and rebuilt from the live cluster."
        ),
        pattern=r"^[a-zA-Z0-9\-_\.]+$",
        max_length=128,
        json_schema_extra={"example": "0195f48b-d000-7000-8000-000000000003"},
    )
    history_cutoff: bool = Field(
        default=False,
        description=(
            "Benchmark isolation: after cloning at a commit SHA, delete every remote/tag ref, "
            "remove origin, expire reflog and gc so no later commit is reachable. Requires "
            "`branch` to be a full 40-hex SHA and a workspace that does not already exist."
        ),
    )
    toolchain: Literal["python", "python+rust"] = Field(
        default="python",
        description=(
            "Workspace toolchain. `python` (default) is the standard workspace. `python+rust` "
            "adds Rust 1.99.0 (cargo, clippy, rustfmt, cargo-nextest/audit/deny), sqlx-cli and "
            "a loopback Postgres 18 (`ws-pg start` / `ws-pg createdb <name>` print the database URL), with a larger resource "
            "profile. Kubernetes provisioner only."
        ),
    )
    expected_repos: int | None = Field(
        default=None,
        ge=0,
        le=MAX_EXPECTED_REPOS,
        description=(
            "How many secondary repositories (POST {ws}/repos) this workspace is expected "
            "to hold, e.g. the N Python sources of a py2rust rewrite. Sizes the "
            "workspace's disk at create (a running pod cannot be resized): +2Gi of "
            "ephemeral storage per repository on top of the toolchain's base (python 10Gi, "
            "python+rust 30Gi), capped at the namespace maximum of 40Gi. Omitted: python+rust "
            "workspaces are sized for 0 secondary repositories; python workspaces keep "
            "their unsized pod spec. Kubernetes provisioner only; ignored (and not checked) on "
            "reuse of an existing workspace."
        ),
        json_schema_extra={"x-n8n-param-name": "expectedRepos"},
    )
    expected_services: int | None = Field(
        default=None,
        ge=0,
        le=MAX_EXPECTED_SERVICES,
        description=(
            "How many background services (POST {ws}/services) this workspace is expected "
            "to run at once, e.g. the N source services of a system being recorded. Sizes "
            "the pod's memory/CPU at create (a running pod cannot be resized): tooling "
            "reserve (python 1Gi, python+rust 4Gi) + expected_dependencies + "
            "expected_services x service_memory_mi, at least the toolchain default, at "
            "most the namespace maximum of 8Gi. A set that cannot fit is a 409 "
            "WORKSPACE_MEMORY_BUDGET_EXCEEDED here, never an OOM-killed pod later. Omitted "
            "(and no expected_dependencies): the toolchain's resources are kept. "
            "Kubernetes provisioner only; ignored (and not checked) on "
            "reuse of an existing workspace."
        ),
        json_schema_extra={"x-n8n-param-name": "expectedServices"},
    )
    service_memory_mi: int | None = Field(
        default=None,
        ge=SERVICE_MEMORY_MIN_MI,
        le=SERVICE_MEMORY_MAX_MI,
        description=(
            "Memory hint per expected service in MiB (peak RSS incl. its children and "
            "supervisor; default 256). Used with expected_services."
        ),
        json_schema_extra={"x-n8n-param-name": "serviceMemoryMi"},
    )
    expected_dependencies: list[Literal["redis", "postgres"]] | None = Field(
        default=None,
        max_length=2,
        description=(
            "Dependency services the workspace will run (started by exec: `ws-redis start` / "
            "`ws-pg start`, each prints its loopback URL): "
            "redis (320 MiB) and/or postgres (512 MiB, python+rust only). Counted into "
            "the memory sizing like expected_services."
        ),
        json_schema_extra={"x-n8n-param-name": "expectedDependencies"},
    )

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, v):
        """Validate repo URL is not empty and follows GitHub URL format."""
        if not v or not v.strip():
            raise ValueError("Repository URL cannot be empty")

        v = v.strip()

        # Basic URL format validation for GitHub URLs

        # Check if it's a PR URL
        pr_url_pattern = re.compile(
            r"^https://github\.com/[A-Za-z0-9\-_.]+/[A-Za-z0-9\-_.]+/pull/\d+(?:/.*)?$"
        )

        if pr_url_pattern.match(v):
            # This is a PR URL - we'll handle parsing in the model validator
            return v

        # Standard repo URL pattern
        github_url_pattern = re.compile(
            r"^https://github\.com/[A-Za-z0-9\-_.]+/[A-Za-z0-9\-_.]+(?:\.git)?/?$"
        )

        # Additional check for obviously invalid URLs like single characters
        if len(v) < 10 or not v.startswith("https://"):
            raise ValueError(
                "Repository URL must be a valid GitHub URL (e.g., https://github.com/owner/repo)"
            )

        if not github_url_pattern.match(v):
            raise ValueError(
                "Repository URL must be a valid GitHub URL format (e.g., https://github.com/owner/repo)"
            )

        return v

    @field_validator("branch")
    @classmethod
    def validate_branch(cls, v):
        """Validate branch is not empty."""
        if not v or not v.strip():
            raise ValueError("Branch name cannot be empty")
        return v.strip()

    @field_validator("pr_number")
    @classmethod
    def validate_pr_number(cls, v):
        """Validate PR number if provided."""
        if v is not None and v <= 0:
            raise ValueError("PR number must be positive if provided")
        return v

    @model_validator(mode="after")
    def parse_pr_url(self):
        """Parse PR URL and extract repository URL and PR number if applicable."""

        # Check if repo_url is a PR URL
        pr_url_pattern = re.compile(
            r"^https://github\.com/([A-Za-z0-9\-_.]+)/([A-Za-z0-9\-_.]+)/pull/(\d+)(?:/.*)?$"
        )

        match = pr_url_pattern.match(self.repo_url)
        if match:
            # Extract components from PR URL
            owner = match.group(1)
            repo = match.group(2)
            pr_number = int(match.group(3))

            # Update fields
            self.repo_url = f"https://github.com/{owner}/{repo}"
            self.pr_number = pr_number
            self.provision_type = "pull_request"

        return self

    @model_validator(mode="after")
    def validate_dependencies_toolchain(self):
        """``ws-pg`` exists only in the python+rust image: refuse rather than size for it."""
        if "postgres" in (self.expected_dependencies or ()) and self.toolchain != "python+rust":
            raise ValueError("expected_dependencies 'postgres' requires toolchain 'python+rust'")
        return self

    @model_validator(mode="after")
    def validate_history_cutoff(self):
        """A history cutoff only makes sense at a pinned commit.

        The cutoff deletes every ref that could reach a later commit, so the
        checkout target must itself be a FULL 40-hex SHA — a branch name or an
        abbreviated SHA would leave the meaning of "later" undefined. It also
        cannot combine with a PR provision: the PR path checks out FETCH_HEAD,
        not the SHA in ``branch``. Runs AFTER ``parse_pr_url`` so a PR url that
        rewrote ``pr_number``/``provision_type`` is refused too.
        """
        if not self.history_cutoff:
            return self
        if not re.fullmatch(r"[0-9a-fA-F]{40}", self.branch or ""):
            raise ValueError("history_cutoff requires branch to be a full 40-hex commit SHA")
        if self.pr_number is not None or self.provision_type == "pull_request":
            raise ValueError(
                "history_cutoff cannot be combined with a pull-request provision "
                "(pr_number / provision_type='pull_request' / a PR url)"
            )
        return self


class ContainerInfo(BaseModel):
    """Information model for container instances."""

    container_name: (
        str  # Primary identifier - same as Docker container name, used to access code graphs
    )
    api_endpoint: str = Field(
        ...,
        description="Deprecated alias for the public/gateway-facing endpoint",
    )
    internal_endpoint: str = Field(
        default="",
        description="Direct runtime-reachable endpoint used for internal probes and exec routing",
    )
    public_endpoint: str = Field(
        default="",
        description="Client-facing or gateway-facing endpoint for API responses",
    )
    code_graph_id: str = ""  # Code graph identifier for this container
    codebase_path: str
    repo_url: str
    branch: str
    status: ContainerStatus
    description: str | None = None
    pr_number: int | None = None
    provision_type: str = "branch"
    github_access_token: str | None = Field(
        None,
        description="GitHub access token for this container's repository operations (deprecated - use user_token_id)",
    )
    user_token_id: str | None = Field(
        None,
        description="Reference to encrypted user GitHub token in SecureTokenManager",
    )
    user_id: str | None = Field(
        None,
        description="User ID who provisioned this container",
    )
    original_repo_owner: str | None = Field(
        None,
        description="Original repository owner (for creating upstream PRs)",
    )
    original_repo_name: str | None = Field(
        None,
        description="Original repository name (for creating upstream PRs)",
    )
    system_fork_owner: str | None = Field(
        None,
        description="System account owner where fork resides (e.g., 'nonagenticai')",
    )
    graph_initialized: bool = Field(
        False,
        description="Whether the code graph has been initialized for this container",
    )
    tenant_id: str | None = Field(
        None,
        description="Tenant identifier for multi-tenant data isolation",
    )
    secret_config_id: str | None = Field(
        None,
        description=(
            "Opaque secrets-service reference (the project's github_token_secret_id) "
            "used to re-resolve the per-tenant GIT_TOKEN for push/upstream-PR after "
            "provisioning. The raw token is never stored on the container; downstream "
            "fork-push and upstream-PR re-fetch it from secrets-service by this id + tenant."
        ),
    )
    history_cutoff_applied: bool = Field(
        False,
        description=(
            "True when the workspace was cloned at a commit SHA with history_cutoff: every "
            "remote/tag ref and the origin remote were deleted, the reflog expired and "
            "unreachable objects pruned, so no commit after that SHA is reachable from the clone."
        ),
    )
    toolchain: Literal["python", "python+rust"] = Field(
        "python",
        description="Workspace toolchain the workspace was provisioned with (see create_workspace).",
    )
    disk_budget_gi: int | None = Field(
        None,
        description=(
            "Size limit of /workspace in Gi (the repo-data volume), set when the workspace "
            "was sized (python+rust, or expected_repos given). Null: unsized."
        ),
    )
    memory_limit_mi: int | None = Field(
        None,
        description=(
            "Memory limit of the workspace container in MiB, set when the workspace was "
            "sized for services (expected_services / expected_dependencies). Null: the "
            "toolchain default (python 2048, python+rust 6144)."
        ),
    )

    @model_validator(mode="after")
    def sync_endpoint_fields(self):
        """Keep legacy api_endpoint aligned with explicit internal/public endpoints."""
        if not self.public_endpoint:
            self.public_endpoint = self.api_endpoint
        if not self.api_endpoint:
            self.api_endpoint = self.public_endpoint or self.internal_endpoint
        return self


class ContainerSpec(BaseModel):
    """Specification model for container setup."""

    name: str = Field(..., description="Container name")
    image: str = Field(..., description="Docker image to use")
    port: int = Field(..., description="Host port to bind to")
    command: str | None = Field(None, description="Command to run in container")
    working_dir: str | None = Field(None, description="Working directory")
    environment: dict[str, Any] = Field(default_factory=dict, description="Environment variables")
    volumes: dict[str, Any] = Field(default_factory=dict, description="Volume mounts")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        """Validate container name is not empty."""
        if not v or not v.strip():
            raise ValueError("Container name cannot be empty")
        return v.strip()

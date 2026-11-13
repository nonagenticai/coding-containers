"""Container Management API Models.

This module contains Pydantic models for Docker container lifecycle management
and monitoring operations in the V2 API. These models provide type-safe
representations for container operations, status tracking, and detailed
container information.

The models support:
    - Container listing and detailed inspection
    - Container lifecycle operations (start, stop, restart, delete)
    - Real-time status monitoring
    - Batch operations on multiple containers
    - Integration with external container registries

Example:
    ```python
    operation_request = ContainerOperationRequest(
        operation=ContainerOperation.RESTART,
        force=False
    )
    ```

Note:
    All container models integrate with the core container management system
    and provide validation for container names and operation parameters.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from coding_client.types.constraint_types import SAFE_STRING_PATTERN
from src.models.common_api_models import BaseResponse, ResponseStatus
from src.models.container_models import ContainerInfo, ContainerStatus


class ContainerListResponse(BaseResponse):
    """Response model for container listing operations.

    This model provides a comprehensive list of containers along with
    metadata about the total count for pagination and filtering.

    Attributes:
        status (ResponseStatus): Always SUCCESS for successful listing operations
        containers (List[ContainerInfo]): Detailed information for each container
        total (int): Total number of containers matching the query criteria

    Note:
        The containers list may be filtered or paginated based on request parameters.
        The total field represents the unfiltered count for pagination calculations.
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    containers: list[ContainerInfo] = Field(..., description="List of containers")
    total: int = Field(..., description="Total number of containers")


class ContainerDetailResponse(BaseResponse):
    """Response model for retrieving detailed information about a specific container.

    This model provides comprehensive details about a single container,
    including its current state, configuration, and runtime information.

    Attributes:
        status (ResponseStatus): Always SUCCESS for successful detail retrieval
        container (ContainerInfo): Complete container information and metadata

    Note:
        This response includes more detailed information than what's provided
        in container listing operations, such as environment variables and volume mounts.
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    container: ContainerInfo = Field(..., description="Container details")


class ContainerStatusResponse(BaseResponse):
    """Response model for container status queries.

    This model provides current status information for a container,
    optimized for frequent polling and monitoring use cases.

    Attributes:
        status (ResponseStatus): Always SUCCESS for successful status retrieval
        container_status (ContainerStatus): Current container state and health information

    Note:
        This is a lightweight response focused on status information,
        suitable for monitoring and health check scenarios.
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    container_status: ContainerStatus = Field(..., description="Current container status")


#: Shared description of the ``repo`` attribution field (py2rust: N source repos
#: live under .repos/ next to the target repository in one workspace).
_REPO_ATTRIBUTION = (
    "Secondary repository `<owner>/<repo>` this path belongs to (it lies under "
    "/workspace/.repos/<owner>__<repo>/, added read-only with POST {ws}/repos); "
    "null for the workspace's own repository."
)


class ContainerFileEntry(BaseModel):
    """Single file-system entry returned by the container file browser."""

    permissions: str = Field(..., description="POSIX permission string from ls output")
    size: str = Field(..., description="Entry size as reported by ls")
    modified: str = Field(..., description="Last modified timestamp string from ls output")
    name: str = Field(..., description="Entry name")
    type: Literal["directory", "file"] = Field(..., description="Entry type")
    repo: str | None = Field(None, description=_REPO_ATTRIBUTION)


class ContainerFilesResponse(BaseModel):
    """Response model for listing a directory or reading a file in a container."""

    path: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Requested container path")
    type: Literal["directory", "file"] = Field(..., description="Resolved path type")
    content: str | None = Field(None, description="File content when the path is a file")
    entries: list[ContainerFileEntry] | None = Field(
        None, description="Directory entries when the path is a directory"
    )
    resolved_path: str | None = Field(
        None,
        description=(
            "Canonical /workspace path after resolving symlinks. Always inside the "
            "workspace: anything that resolves outside it is refused with 400."
        ),
    )
    repo: str | None = Field(None, description=_REPO_ATTRIBUTION)
    lines_total: int | None = Field(None, ge=0, description="Line count of the WHOLE file")
    start_line: int | None = Field(
        None, ge=1, description="First line returned (1-based), set on a span read"
    )
    end_line: int | None = Field(
        None, ge=0, description="Last line returned (inclusive, clamped to EOF), span reads"
    )
    sha256: str | None = Field(
        None,
        description=(
            "SHA-256 of the WHOLE file's bytes (computed in the workspace, not of the "
            "span or of decoded text): pin a span to the file version it came from."
        ),
    )
    size_bytes: int | None = Field(None, ge=0, description="Size of the WHOLE file in bytes")
    truncated: bool | None = Field(
        None, description="True when `content` was capped and is a prefix of the file/span"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "path": "/workspace/src",
                "type": "directory",
                "content": None,
                "entries": [
                    {
                        "permissions": "-rw-r--r--",
                        "size": "1240",
                        "modified": "Mar 12 09:15",
                        "name": "main.py",
                        "type": "file",
                    }
                ],
            }
        }
    )


class ContainerOperation(StrEnum):
    """Enumeration of supported container lifecycle operations.

    This enum defines all container management operations that can be
    performed through the API, with validation to ensure only valid
    operations are requested.

    Attributes:
        START: Start a stopped container
        STOP: Stop a running container gracefully
        RESTART: Stop and then start a container
        DELETE: Remove a container permanently (requires confirmation)

    Note:
        DELETE operations are irreversible and will remove all container data.
        Use with caution and consider the 'force' parameter for proper handling.
    """

    START = "start"
    STOP = "stop"
    RESTART = "restart"
    DELETE = "delete"


class ContainerOperationRequest(BaseModel):
    """Request model for performing lifecycle operations on containers.

    This model defines the structure for requesting container operations,
    with validation to ensure safe operation execution.

    Attributes:
        operation (ContainerOperation): The specific operation to perform
        force (bool): Whether to force the operation if normal execution fails

    Example:
        ```python
        restart_request = ContainerOperationRequest(
            operation=ContainerOperation.RESTART,
            force=True  # Force restart even if container is unresponsive
        )
        ```

    Note:
        The 'force' parameter should be used carefully, especially with
        STOP and DELETE operations, as it may cause data loss.
    """

    model_config = ConfigDict(
        json_schema_extra={"example": {"operation": "restart", "force": False}}
    )

    operation: ContainerOperation = Field(..., description="Operation to perform")
    force: bool = Field(False, description="Force the operation if needed")


class ContainerOperationResponse(BaseResponse):
    """Response model for container operation results.

    This model provides confirmation and details about completed container
    operations, including the specific operation performed and target container.

    Attributes:
        status (ResponseStatus): SUCCESS if operation completed, ERROR if failed
        operation (ContainerOperation): The operation that was performed
        container_name (str): Name of the container that was operated on

    Note:
        Even if the response status is SUCCESS, check the message field for
        any warnings or additional information about the operation.
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    operation: ContainerOperation = Field(..., description="Operation performed")
    container_name: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="Target container name",
    )
    details: dict[str, Any] | None = Field(None, description="Additional operation details")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Container restarted successfully",
                "operation": "restart",
                "container_name": "my-web-app",
            }
        }
    )


class CreateDirectoryRequest(BaseModel):
    """Request model for creating directories inside a container.

    This model defines the structure for requesting directory creation
    within a provisioned container's file system. Supports nested path creation.

    Attributes:
        path (str): The directory path to create (supports nested directories)

    Example:
        ```python
        mkdir_request = CreateDirectoryRequest(
            path="/app/src/features/new_module"
        )
        ```

    Note:
        The path will be created with mkdir -p behavior, creating parent
        directories as needed. Default permissions will be applied.
    """

    model_config = ConfigDict(
        json_schema_extra={"example": {"path": "/app/src/features/new_module"}}
    )

    path: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="Directory path to create inside container",
        examples=["/app/src/features/new_module", "/app/data/cache"],
    )


class CreateDirectoryResponse(BaseResponse):
    """Response model for directory creation operations.

    This model provides confirmation and details about directory creation
    within a container, including the full path that was created.

    Attributes:
        status (ResponseStatus): SUCCESS if directory was created, ERROR if failed
        path (str): The full path of the created directory
        container_name (str): Name of the container where directory was created

    Note:
        A successful response indicates the directory exists and is accessible
        within the container's file system.
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    path: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Created directory path")
    container_name: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Container name")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Directory created successfully",
                "path": "/app/src/features/new_module",
                "container_name": "my-repo-container",
                "request_id": "mkdir_abc123",
            }
        }
    )


class CreateFileRequest(BaseModel):
    """Request model for creating empty files inside a container.

    This model defines the structure for requesting file creation within a
    provisioned container's file system. Accepts path, filename, and extension
    as separate parameters for clarity and validation.

    Attributes:
        path (str): The directory path where file will be created
        filename (str): The file name without extension
        extension (str): The file extension (without leading dot)

    Example:
        ```python
        touch_request = CreateFileRequest(
            path="/app/src/services",
            filename="auth_service",
            extension="py"
        )
        # Creates: /app/src/services/auth_service.py
        ```

    Note:
        Parent directories will be created automatically if they don't exist.
        The file will be created empty with default permissions (644).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "path": "/app/src/services",
                "filename": "auth_service",
                "extension": "py",
            }
        }
    )

    path: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="Directory path where file will be created",
        examples=["/app/src/services", "/app/tests/unit"],
    )
    filename: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="File name without extension",
        examples=["auth_service", "test_integration", "config"],
    )
    extension: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="File extension without leading dot",
        examples=["py", "js", "ts", "json", "yaml"],
    )


class CreateFileResponse(BaseResponse):
    """Response model for file creation operations.

    This model provides confirmation and details about file creation
    within a container, including the full path that was created.

    Attributes:
        status (ResponseStatus): SUCCESS if file was created, ERROR if failed
        path (str): The full path of the created file
        container_name (str): Name of the container where file was created

    Note:
        A successful response indicates the file exists and is accessible
        within the container's file system. The file will be empty.
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    path: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Created file path")
    container_name: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Container name")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "File created successfully",
                "path": "/app/src/services/auth_service.py",
                "container_name": "my-repo-container",
                "request_id": "touch_def456",
            }
        }
    )


class WriteFileRequest(BaseModel):
    """Request model for writing content to files inside a container.

    This model defines the structure for writing or updating file content within a
    provisioned container's file system. Supports both creating new files and
    overwriting existing ones.

    Attributes:
        file_path (str): Full path to the file (including directory, name, and extension)
        content (str): Content to write to the file

    Example:
        ```python
        write_request = WriteFileRequest(
            file_path="/app/src/services/auth_service.py",
            content="def authenticate(user, password):\\n    return True\\n"
        )
        ```

    Note:
        Parent directories will be created automatically if they don't exist.
        If the file already exists, its content will be overwritten.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_path": "/app/src/services/auth_service.py",
                "content": "def authenticate(user, password):\n    # Authentication logic here\n    return True\n",
            }
        }
    )

    file_path: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="Full path to the file",
        examples=["/app/src/main.py", "/app/config/settings.json"],
    )
    content: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Content to write to the file",
        examples=["def main():\n    pass\n", '{"key": "value"}'],
    )


class WriteFileResponse(BaseResponse):
    """Response model for file write operations.

    This model provides confirmation and details about file write operations
    within a container, including the full path and size of the written file.

    Attributes:
        status (ResponseStatus): SUCCESS if file was written, ERROR if failed
        path (str): The full path of the written file
        container_name (str): Name of the container where file was written
        bytes_written (int): Number of bytes written to the file

    Note:
        A successful response indicates the file exists with the specified content
        within the container's file system.
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    path: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Written file path")
    container_name: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Container name")
    bytes_written: int = Field(..., description="Number of bytes written", ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "File written successfully",
                "path": "/app/src/services/auth_service.py",
                "container_name": "my-repo-container",
                "bytes_written": 245,
                "request_id": "write_ghi789",
            }
        }
    )


class MovePathRequest(BaseModel):
    """Request to move or rename a file or directory in the repository workspace."""

    source_path: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        pattern=SAFE_STRING_PATTERN,
        description="File or directory to move, relative to /workspace or as an absolute path",
        examples=["src/old_name.py", "/workspace/src/old_module"],
    )
    destination_path: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        pattern=SAFE_STRING_PATTERN,
        description="Exact destination path, relative to /workspace or as an absolute path",
        examples=["src/new_name.py", "/workspace/src/new_module"],
    )
    overwrite: bool = Field(
        False,
        description="Replace an existing destination when the underlying move operation permits it",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source_path": "/workspace/src/old_module",
                "destination_path": "/workspace/src/new_module",
                "overwrite": False,
            }
        }
    )


class MovePathResponse(BaseResponse):
    """Confirmation that a file or directory was moved."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    source_path: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Original path")
    destination_path: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Resulting path")
    container_name: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Container name")


class DeletePathResponse(BaseResponse):
    """Confirmation that a file or directory was deleted."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    path: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Deleted path")
    container_name: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Container name")
    recursive: bool = Field(..., description="Whether recursive directory deletion was authorized")


class WorkspaceSearchHit(BaseModel):
    """One matching line from :class:`WorkspaceSearchResponse`."""

    file: str = Field(..., description="Path relative to the repository root")
    line: int = Field(..., ge=1, description="1-based line number")
    text: str = Field(..., description="The matching line")
    repo: str | None = Field(None, description=_REPO_ATTRIBUTION)


class WorkspaceSearchResponse(BaseModel):
    """Line hits for a literal or regex pattern, confined to the workspace."""

    pattern: str = Field(..., description="Pattern as requested")
    regex: bool = Field(..., description="True = extended regex, False = fixed string")
    path: str = Field(..., description="Canonical /workspace path that was searched")
    hits: list[WorkspaceSearchHit] = Field(default_factory=list)
    total_returned: int = Field(..., ge=0)
    truncated: bool = Field(
        ..., description="True when more matches exist than max_results; hits is a prefix"
    )


class GitDiffFile(BaseModel):
    """One changed path in a :class:`GitDiffResponse` (``git diff --name-status``)."""

    status: str = Field(..., description="git status letter: A, M, D, R, C, T, U")
    path: str = Field(..., description="Path relative to the repository root (new path on rename)")
    old_path: str | None = Field(None, description="Source path, set only for renames and copies")


#: A full object id (SHA-1 or SHA-256), lowercase.
_OBJECT_ID_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_BLOB_ID_RE = _OBJECT_ID_RE


class GitDiffResponse(BaseResponse):
    """Unified diff of a workspace between two commits, or a commit and the working tree.

    ``head_sha`` is ``None`` exactly when ``compared_to == "working_tree"``: the diff
    then covers staged AND unstaged edits to TRACKED files. ``git diff`` never shows
    untracked files, so they are listed separately in ``untracked_files`` rather than
    silently omitted -- an empty ``diff`` with a non-empty ``untracked_files`` is NOT
    "no changes".
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    container_name: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Container name")
    base: str = Field(..., description="Base ref as requested")
    base_sha: str = Field(
        ...,
        description=(
            "Commit the diff starts from: the base ref's commit, or its merge base with "
            "`head` when the two have diverged (a workspace imported at a pinned commit)"
        ),
    )
    head: str | None = Field(None, description="Head ref as requested; null = working tree")
    head_sha: str | None = Field(None, description="Commit the head ref resolved to")
    compared_to: Literal["commit", "working_tree"] = Field(
        ..., description="What the base was compared against"
    )
    path: str | None = Field(None, description="Pathspec the diff was limited to, if any")
    context_lines: int = Field(..., ge=0, description="Unified context lines (-U)")
    files: list[GitDiffFile] = Field(default_factory=list, description="Changed paths")
    untracked_files: list[str] = Field(
        default_factory=list,
        description="Untracked files (working-tree comparisons only); NOT part of `diff`",
    )
    diff: str = Field("", description="Unified diff text, possibly truncated")
    diff_bytes: int = Field(..., ge=0, description="Size of the FULL diff in UTF-8 bytes")
    truncated: bool = Field(False, description="True when `diff` was cut at max_bytes")
    blobs: dict[str, str | None] = Field(
        default_factory=dict,
        description=(
            "Blob id of each changed path's content on the compared side (the head "
            "commit, or the working tree as `git add` would store it), keyed by path; "
            "null = the path is absent there (deleted, or the old side of a rename). "
            "Covers every path in `files` (and each `old_path`) and every "
            "`untracked_files` entry. Read from `tree`, never from the live working tree. "
            "Pass it as the push's `target_blobs` (a secondary check under `tree`)."
        ),
    )
    tree: str | None = Field(
        None,
        description=(
            "Object id of the ONE immutable tree this response was computed from: for a "
            "working-tree comparison, a snapshot of the whole workspace as `git add -A` "
            "would commit it (created files included), frozen in a single exec; for a "
            "commit comparison, the head commit's tree. `diff`, `files`, "
            "`untracked_files` and `blobs` all describe this tree, never a later read of "
            "the working tree. Pass it as the commit's `approved_tree` (with `base_sha` as "
            "`approved_base`) so exactly this tree is committed, and as the push's "
            "`approved_tree` so exactly this tree is pushed."
        ),
    )


class GitStatusResponse(BaseResponse):
    """Response model for git status operations.

    This model provides information about the current state of the git repository
    in the container, including modified, added, and deleted files.

    Attributes:
        status (ResponseStatus): SUCCESS if git status retrieved successfully
        container_name (str): Name of the container
        modified_files (List[str]): List of modified files
        new_files (List[str]): List of new/untracked files
        deleted_files (List[str]): List of deleted files
        staged_changes (int): Number of staged changes
        unstaged_changes (int): Number of unstaged changes
        branch (str): Current git branch name

    Note:
        File paths are relative to the repository root.
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    container_name: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Container name")
    modified_files: list[str] = Field(default_factory=list, description="Modified files")
    new_files: list[str] = Field(default_factory=list, description="New/untracked files")
    deleted_files: list[str] = Field(default_factory=list, description="Deleted files")
    staged_changes: int = Field(0, description="Number of staged changes", ge=0)
    unstaged_changes: int = Field(0, description="Number of unstaged changes", ge=0)
    branch: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Current branch name")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Git status retrieved successfully",
                "container_name": "my-repo-container",
                "modified_files": ["src/main.py", "src/utils.py"],
                "new_files": ["src/auth_service.py"],
                "deleted_files": ["src/old_code.py"],
                "staged_changes": 0,
                "unstaged_changes": 4,
                "branch": "main",
                "request_id": "git_status_jkl012",
            }
        }
    )


class GitCommitRequest(BaseModel):
    """Request model for committing changes in a container's git repository.

    This model defines the structure for creating git commits from changes
    in the container's filesystem. Supports auto-detection of all changes
    or explicit file specification.

    Attributes:
        message (str): Commit message
        files (List[str]): List of file paths to commit, or ["*"] for all changes
        author_name (Optional[str]): Git author name (uses git config if not provided)
        author_email (Optional[str]): Git author email (uses git config if not provided)

    Example:
        ```python
        # Commit all changes
        commit_request = GitCommitRequest(
            message="Add authentication service",
            files=["*"]
        )

        # Commit specific files
        commit_request = GitCommitRequest(
            message="Update authentication logic",
            files=["src/auth_service.py", "src/utils.py"]
        )
        ```

    Note:
        Files are automatically staged before committing. Use ["*"] to commit all changes.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Add authentication service and update utils",
                "files": ["*"],
                "author_name": "Developer",
                "author_email": "dev@example.com",
            }
        }
    )

    message: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="Commit message",
        examples=["Add authentication service", "Fix bug in user validation"],
    )
    files: list[str] = Field(
        default=["*"],
        description="Files to commit (* for all changes)",
        examples=[["*"], ["src/main.py", "src/utils.py"]],
    )
    author_name: str | None = Field(
        None, pattern=SAFE_STRING_PATTERN, description="Git author name (optional)"
    )
    author_email: str | None = Field(
        None, pattern=SAFE_STRING_PATTERN, description="Git author email (optional)"
    )
    approved_tree: str | None = Field(
        None,
        description=(
            "Commit EXACTLY this tree (a `git/diff` response's `tree`) on top of "
            "`approved_base`, instead of re-staging the working tree: a file changed "
            "after the review cannot enter the commit. Requires `approved_base`; `files` "
            "must be left at ['*'] (the tree already says what is committed) and the "
            "pre-commit formatter does not run (it would change reviewed content). "
            "Refused (409 commit_base_moved) unless HEAD is still `approved_base`."
        ),
        examples=["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
        # Distinct n8n parameter names: the generator flattens body fields per node.
        json_schema_extra={"x-n8n-param-name": "commitApprovedTree"},
    )
    approved_base: str | None = Field(
        None,
        description=(
            "The commit the approved tree was reviewed against (the `git/diff` "
            "response's `base_sha`); the new commit's only parent. Required with "
            "`approved_tree`."
        ),
        json_schema_extra={"x-n8n-param-name": "commitApprovedBase"},
    )

    @field_validator("approved_tree", "approved_base")
    @classmethod
    def _full_object_ids(cls, value: str | None) -> str | None:
        if value is not None and not _OBJECT_ID_RE.fullmatch(value):
            raise ValueError("must be a full lowercase object id (40 or 64 hex characters)")
        return value

    @model_validator(mode="after")
    def _approved_tree_needs_base_and_all_files(self) -> GitCommitRequest:
        if (self.approved_tree is None) != (self.approved_base is None):
            raise ValueError("approved_tree and approved_base are given together or not at all")
        if self.approved_tree is not None and self.files != ["*"]:
            raise ValueError(
                "files must be ['*'] with approved_tree: the approved tree already names "
                "exactly what is committed"
            )
        return self


_RESTORE_SOURCE_PATTERN = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")

#: Upper bounds on a restore request. The route builds each step as ONE ``sh -c``
#: argument, and Linux caps a single argv string at MAX_ARG_STRLEN (131072 bytes);
#: the exec backends have no stdin channel to move the paths out of argv. Every
#: step names each path exactly once, and ``shlex.quote`` grows a path by at most
#: 5x (each ``'`` becomes ``'"'"'``; a 4-byte UTF-8 character is not expanded), so
#: the worst case is ~20 * (5 * 512) bytes plus a fixed prelude - about half the
#: kernel limit. ``test_worst_case_commands_stay_far_below_max_arg_strlen`` pins it.
#: The only consumer (the conveyor's QA) reverts 1-3 files per candidate.
RESTORE_MAX_FILES = 20
RESTORE_MAX_PATH_CHARS = 512


class GitRestoreRequest(BaseModel):
    """Revert named paths in a workspace to their content at ``source``.

    The typed replacement for a raw ``exec`` of ``git checkout -- <files>``: a losing
    candidate rewrite is reverted through a route that refuses unknown paths instead of
    silently doing nothing.

    ``source`` is restricted to ``[A-Za-z0-9._/-]`` with no leading ``-`` (it would be
    parsed as an option) and no ``..`` (no range syntax smuggled through one ref).
    Untracked paths are only removed when ``clean_untracked`` is true; otherwise the
    request is refused with 409 so a caller never deletes a file by accident.

    ``files`` is capped at ``RESTORE_MAX_FILES`` paths of ``RESTORE_MAX_PATH_CHARS``
    characters (422 above either) so every command the route builds stays far below
    the kernel's single-argument limit - see ``RESTORE_MAX_FILES``.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "files": ["src/auth_service.py"],
                "source": "HEAD",
                "clean_untracked": False,
            }
        }
    )

    files: list[Annotated[str, StringConstraints(max_length=RESTORE_MAX_PATH_CHARS)]] = Field(
        ...,
        min_length=1,
        max_length=RESTORE_MAX_FILES,
        description=(
            f"Paths (relative to the repository root) to restore; at most "
            f"{RESTORE_MAX_FILES} paths of at most {RESTORE_MAX_PATH_CHARS} characters"
        ),
        examples=[["src/main.py"]],
        # Distinct n8n parameter name: without it this merges with
        # commitWorkspaceChanges.files in the generated node's UI.
        json_schema_extra={"x-n8n-param-name": "restoreFiles"},
    )
    source: str = Field(
        "HEAD",
        description="Branch, tag or commit to restore from (default HEAD)",
        examples=["HEAD", "main", "a1b2c3d"],
    )
    clean_untracked: bool = Field(
        False,
        description="Delete listed paths that do not exist at `source` (else 409)",
    )

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value: str) -> str:
        if not _RESTORE_SOURCE_PATTERN.fullmatch(value) or value.startswith("-") or ".." in value:
            raise ValueError(
                "source must match [A-Za-z0-9._/-]+, must not start with '-' "
                "and must not contain '..'"
            )
        return value


class GitRestoreResponse(BaseResponse):
    """Outcome of a git restore: which paths were restored, which were deleted.

    ``status_after`` is ``git status --porcelain`` limited to the requested paths
    AFTER the operation; an empty list means every requested path is clean.
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    container_name: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Container name")
    source: str = Field(..., description="Ref the tracked paths were restored from")
    restored: list[str] = Field(default_factory=list, description="Tracked paths restored")
    cleaned: list[str] = Field(default_factory=list, description="Untracked paths deleted")
    status_after: list[str] = Field(
        default_factory=list, description="Porcelain status lines for the requested paths"
    )


class GitBranchRequest(BaseModel):
    """Ensure the workspace is on local branch ``branch`` (idempotent).

    The typed replacement for a raw ``exec`` of ``git checkout -B <branch>``, which
    silently resets an existing branch to HEAD. Here an existing branch is switched to
    (uncommitted work carries over, or the request is refused with 409), and an absent
    one is created from the current HEAD. Nothing is ever forced or reset.

    The name is checked by the route with ``_validate_git_branch_name`` (400) and then
    by ``git check-ref-format --branch`` before anything is mutated.
    """

    model_config = ConfigDict(json_schema_extra={"example": {"branch": "feat/auth-service"}})

    branch: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Local branch to switch to, created from HEAD if absent",
        examples=["feat/auth-service", "fix/issue-123"],
        # Distinct n8n parameter name: without it this merges with
        # createWorkspace.branch in the generated node's UI.
        json_schema_extra={"x-n8n-param-name": "targetBranch"},
    )


class GitBranchResponse(BaseResponse):
    """Outcome of ensuring a branch: which branch HEAD is on now, and whether it is new.

    ``branch`` and ``head_sha`` are MEASURED after the switch, never echoed from the
    request. ``previous_branch`` is ``None`` when HEAD was detached.
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    container_name: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Container name")
    branch: str = Field(..., description="Branch HEAD is on after the request")
    created: bool = Field(..., description="True if the branch did not exist and was created")
    head_sha: str = Field(..., description="Commit HEAD points at after the request")
    previous_branch: str | None = Field(
        None, description="Branch HEAD was on before the request (null if detached)"
    )


class GitCommitResponse(BaseResponse):
    """Response model for git commit operations.

    This model provides confirmation and details about git commit operations
    in the container, including commit hash and files committed.

    Attributes:
        status (ResponseStatus): SUCCESS if commit created, ERROR if failed
        container_name (str): Name of the container
        commit_sha (str): SHA hash of the created commit
        files_committed (int): Number of files included in the commit
        branch (str): Branch where commit was created

    Note:
        The commit is created locally in the container's git repository
        and is not automatically pushed to remote.
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    container_name: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Container name")
    commit_sha: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Commit SHA hash")
    files_committed: int = Field(..., description="Number of files committed", ge=0)
    branch: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Branch name")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Changes committed successfully",
                "container_name": "my-repo-container",
                "commit_sha": "a1b2c3d4e5f6",
                "files_committed": 3,
                "branch": "main",
                "request_id": "git_commit_mno345",
            }
        }
    )


#: Bounds on the optional ``GitPushRequest.target_files`` allowlist.
PUSH_MAX_TARGET_FILES = 500
PUSH_MAX_PATH_CHARS = 1024


class GitPushRequest(BaseModel):
    """Request model for pushing commits to GitHub fork.

    This model defines the structure for pushing local commits from the
    container's git repository to a GitHub fork.

    Attributes:
        branch (str): Branch name to push to
        remote_url (Optional[str]): Fork URL (auto-creates fork if not provided)
        force (bool): Whether to force push
        create_fork (bool): Auto-create fork if it doesn't exist

    Example:
        ```python
        # Push to fork (auto-create if needed)
        push_request = GitPushRequest(
            branch="feature/auth-service",
            create_fork=True
        )

        # Push to specific remote
        push_request = GitPushRequest(
            branch="feature/auth-service",
            remote_url="https://github.com/username/repo.git"
        )
        ```

    Note:
        Requires GitHub authentication token to be configured for the container.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "branch": "feature/auth-service",
                "remote_url": None,
                "force": False,
                "create_fork": True,
            }
        }
    )

    branch: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="Branch name to push",
        examples=["main", "feature/new-feature", "bugfix/issue-123"],
    )
    remote_url: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="Remote repository URL (auto-detected if not provided)",
    )
    force: bool = Field(False, description="Force push (use with caution)")
    create_fork: bool = Field(True, description="Auto-create fork if it doesn't exist")
    target_files: (
        list[Annotated[str, StringConstraints(min_length=1, max_length=PUSH_MAX_PATH_CHARS)]] | None
    ) = Field(
        None,
        max_length=PUSH_MAX_TARGET_FILES,
        description=(
            "Optional allowlist of repository-relative paths the pushed commits may "
            "touch; an entry ending in '/' admits everything under it. When set, a "
            "push touching any other path is refused (path_not_in_target_files). "
            "Process data (.aa/, *_probe/, *.log, ...) is refused whether or not "
            "this is set."
        ),
        examples=[["src/calc.py", "tests/"]],
        # Distinct n8n parameter name: the generator flattens body fields per node.
        json_schema_extra={"x-n8n-param-name": "pushTargetFiles"},
    )

    target_blobs: dict[str, str | None] | None = Field(
        None,
        max_length=PUSH_MAX_TARGET_FILES,
        description=(
            "Content the push must ship: repository-relative path -> the blob id that "
            "was reviewed (a `git/diff` response's `blobs`), null = the path must be "
            "absent. When set, EVERY path the pushed commits touch must be a key here, "
            "and its blob at the pushed tip must equal the value, or the push is "
            "refused (push_content_unverified / push_content_mismatch). REQUIRED when "
            "`target_files` is set: an allowlist of paths alone lets a workspace that "
            "faked the reviewed diff ship other content in an allowed file."
        ),
        examples=[{"src/calc.py": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}],
        json_schema_extra={"x-n8n-param-name": "pushTargetBlobs"},
    )

    approved_tree: str | None = Field(
        None,
        description=(
            "The reviewed tree (a `git/diff` response's `tree`). When set, the push "
            "sends EXACTLY ONE commit: its only parent must be the commit the remote "
            "already has (the base GitHub reports) and its tree must be this id, or the "
            "push is refused (push_not_single_commit / push_tree_mismatch). Tree-id "
            "equality binds every byte pushed, and an intermediate commit (evil content "
            "added, then reverted) can never ride along. REQUIRED when `target_files` or "
            "`target_blobs` is set."
        ),
        examples=["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
        json_schema_extra={"x-n8n-param-name": "pushApprovedTree"},
    )

    @field_validator("approved_tree")
    @classmethod
    def _approved_tree_is_an_id(cls, value: str | None) -> str | None:
        if value is not None and not _OBJECT_ID_RE.fullmatch(value):
            raise ValueError("approved_tree must be a full lowercase tree id")
        return value

    @field_validator("target_blobs")
    @classmethod
    def _target_blobs_are_paths_and_ids(
        cls, value: dict[str, str | None] | None
    ) -> dict[str, str | None] | None:
        """Keys are plain repository-relative FILE paths; values full blob ids or null."""
        if value is None:
            return None
        for path, blob in value.items():
            if (
                not path
                or len(path) > PUSH_MAX_PATH_CHARS
                or path.startswith("/")
                or path.endswith("/")
                or "\\" in path
                or "\x00" in path
                or any(seg in ("", ".", "..") for seg in path.split("/"))
            ):
                raise ValueError(f"target_blobs key is not a relative file path: {path!r}")
            if blob is not None and not _BLOB_ID_RE.fullmatch(blob):
                raise ValueError(f"target_blobs[{path!r}] is not a full lowercase blob id")
        return value

    @model_validator(mode="after")
    def _target_files_need_target_blobs(self) -> GitPushRequest:
        if self.target_files is not None and self.target_blobs is None:
            raise ValueError(
                "target_blobs is required when target_files is set: pass the reviewed "
                "diff's `blobs` so the push can check CONTENT, not only paths"
            )
        if (
            self.target_files is not None or self.target_blobs is not None
        ) and self.approved_tree is None:
            raise ValueError(
                "approved_tree is required when target_files or target_blobs is set: pass "
                "the reviewed diff's `tree` so the push is bound to ONE snapshot"
            )
        return self

    @field_validator("target_files")
    @classmethod
    def _target_files_are_relative(cls, value: list[str] | None) -> list[str] | None:
        """Each entry must be a plain repository-relative path (no '/', '..', '\\')."""
        if value is None:
            return None
        for entry in value:
            body = entry[:-1] if entry.endswith("/") else entry
            if (
                not body
                or body.startswith("/")
                or "\\" in body
                or "\x00" in body
                or any(seg in ("", ".", "..") for seg in body.split("/"))
            ):
                raise ValueError(f"target_files entry is not a relative repository path: {entry!r}")
        return value


class GitPushResponse(BaseResponse):
    """Response model for git push operations.

    This model provides confirmation and details about git push operations
    from the container to GitHub fork.

    Attributes:
        status (ResponseStatus): SUCCESS if pushed successfully, ERROR if failed
        container_name (str): Name of the container
        branch (str): Branch that was pushed
        remote_url (str): URL of the remote repository
        commits_pushed (int): Number of commits pushed

    Note:
        After successful push, commits are available on GitHub and can be
        used to create pull requests.
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    container_name: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Container name")
    branch: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Branch pushed")
    remote_url: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Remote repository URL")
    commits_pushed: int = Field(..., description="Number of commits pushed", ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Commits pushed successfully to fork",
                "container_name": "my-repo-container",
                "branch": "feature/auth-service",
                "remote_url": "https://github.com/username/repo.git",
                "commits_pushed": 2,
                "request_id": "git_push_pqr678",
            }
        }
    )


class GitFetchRequest(BaseModel):
    """Request model for ``POST /{workspace}/git/fetch``.

    The refspec is validated strictly by the route (``validate_fetch_refspec``), not
    here, so there is ONE rule and a refusal is a 400 naming it.
    """

    model_config = ConfigDict(
        json_schema_extra={"example": {"refspec": "pull/123/head:refs/remotes/pr/123", "depth": 50}}
    )

    remote_url: str | None = Field(
        None,
        max_length=512,
        description=(
            "Repository to fetch from; defaults to the workspace origin. Must be the "
            "workspace's own origin repository on an allowed host."
        ),
        json_schema_extra={"x-n8n-param-name": "fetchRemoteUrl"},
    )
    refspec: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description=(
            "[+]<src>[:<dst>], e.g. 'pull/123/head:refs/remotes/pr/123' or a branch "
            "name. <dst>, when given, must be under refs/remotes/<remote>/. Allowed "
            "characters: [A-Za-z0-9._/-]; no leading '-', no '..', no whitespace."
        ),
        examples=["pull/123/head:refs/remotes/pr/123", "feature/x"],
        json_schema_extra={"x-n8n-param-name": "fetchRefspec"},
    )
    depth: int | None = Field(
        None,
        ge=1,
        le=100000,
        description=(
            "Optional shallow-fetch depth. Allowed only when the workspace repository "
            "is already shallow (a provisioned clone --depth 1 is); on a full clone it "
            "is refused with 400 fetch_depth_on_full_clone, because fetching with a "
            "depth would make that clone shallow permanently. Omit it to fetch normally."
        ),
        json_schema_extra={"x-n8n-param-name": "fetchDepth"},
    )


class GitFetchResponse(BaseResponse):
    """Result of ``POST /{workspace}/git/fetch``."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    ok: bool = Field(True, description="True when the ref was fetched and resolved")
    container_name: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Workspace name")
    remote_url: str = Field(..., description="Canonical URL fetched from")
    fetched_ref: str = Field(
        ..., description="Local ref holding the fetched commit (the refspec's dst, else FETCH_HEAD)"
    )
    head_sha: str = Field(..., description="Commit id of fetched_ref (git rev-parse)")

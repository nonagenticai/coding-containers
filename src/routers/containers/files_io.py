"""Container filesystem routes: write-file + list/read files (I/O operations)."""

from __future__ import annotations

import base64
import logging
import posixpath
import re
import shlex
from uuid import uuid7

import docker.errors
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query

from coding_client.auth.models import User
from coding_client.auth.tenant import (
    get_current_user_with_tenant,
    require_tenant,
)
from coding_client.models.responses import (
    bad_request,
    internal_error,
    not_found,
)
from src.models.container_api_models import (
    ContainerFilesResponse,
    WorkspaceSearchHit,
    WorkspaceSearchResponse,
    WriteFileRequest,
    WriteFileResponse,
)
from src.models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error406Response,
    Error409Response,
    Error422Response,
    Error429Response,
    Error500Response,
)

from ._common import (
    _WORKSPACE_ROOT,
    _normalize_workspace_read_path,
    _validate_git_file_path,
    get_manager,
    refuse_write_into_secondary_repos,
    secondary_repo_of_path,
)
from .files_manage import _repository_root

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/{container_name}/write-file",
    response_model=WriteFileResponse,
    operation_id="write_workspace_file",
    summary="Write Content to File in Workspace",
    responses={
        200: {"description": "File written successfully"},
        409: {
            "model": Error409Response,
            "description": "SECONDARY_REPO_READ_ONLY: the path is under /workspace/.repos/ (read-only sources)",
        },
        400: {"model": Error400Response, "description": "Invalid request parameters"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Container not found"},
        406: {"model": Error406Response, "description": "Not acceptable"},
        422: {"model": Error422Response, "description": "Validation error"},
        429: {"model": Error429Response, "description": "Too many requests"},
        500: {"model": Error500Response, "description": "Error writing file"},
    },
)
async def write_file_content(
    container_name: str = Path(..., description="Container name"),
    request: WriteFileRequest = Body(...),
    user: User = Depends(get_current_user_with_tenant),
) -> WriteFileResponse:
    """Write content to a file inside a container's file system."""
    try:
        container_manager = get_manager()

        tenant_id = None if user.is_admin() else user.effective_tenant_id
        try:
            container = container_manager.get_container(container_name, tenant_id=tenant_id)
        except ValueError:
            container = None

        if not container:
            raise not_found("Container", container_name)

        if container.status.value.lower() != "running":
            raise bad_request(
                f"Container '{container_name}' is not running (current status: {container.status.value})"
            )

        file_path = request.file_path.strip()
        content = request.content

        if not file_path:
            raise bad_request("file_path must not be empty")

        if any(c in file_path for c in [";", "|", "&", "`", "$", "(", ")", "<", ">"]):
            raise bad_request("Invalid path: shell metacharacters not allowed")

        # coding-containers#166: a WRITE is confined exactly like a read. The path maps to the
        # repository root (relative, or under the /workspace alias); any other absolute path,
        # any `..` segment and any control character is refused here, before the container is
        # touched. The symlink half is decided IN the container by _write_path_command.
        workspace_root = _repository_root(container)
        try:
            candidate = _normalize_workspace_read_path(file_path, workspace_root=workspace_root)
        except ValueError as exc:
            raise bad_request(f"Invalid path: {exc}") from exc
        if posixpath.normpath(candidate) == posixpath.normpath(workspace_root):
            raise bad_request("Invalid path: the workspace root is a directory, not a file")
        refuse_write_into_secondary_repos(
            candidate, workspace_root=workspace_root, container_name=container_name
        )

        content_bytes = content.encode("utf-8")
        content_b64 = base64.b64encode(content_bytes).decode("ascii")

        result = await container_manager.execute_command(
            container_name,
            _write_path_command(candidate, workspace_root, content_b64),
            tenant_id=tenant_id,
        )

        exit_code = result.get("exit_code", 1)
        if exit_code != 0:
            if exit_code == _READ_OUTSIDE_WORKSPACE:
                raise bad_request("Invalid path: resolves outside the repository workspace")
            if exit_code == _READ_NOT_FILE_OR_DIR:
                raise bad_request("Invalid path: not a regular file")
            if exit_code == _READ_CANNOT_VERIFY:
                raise internal_error(
                    "Cannot verify the written path is inside the workspace (no /proc in the "
                    "workspace); refusing to write rather than trusting the path"
                )
            error_msg = result.get("stderr", "Unknown error")
            raise internal_error(f"Failed to write file: {error_msg}")

        logger.info(
            "Successfully wrote %d bytes to file '%s' in container '%s'",
            len(content_bytes),
            file_path,
            container_name,
        )

        return WriteFileResponse(
            message="File written successfully",
            path=file_path,
            container_name=container_name,
            bytes_written=len(content_bytes),
            request_id=str(uuid7()),
        )

    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as e:
        logger.error("Failed to write file in container '%s': %s", container_name, e)
        raise internal_error(f"Failed to write file: {str(e)}") from e


# In-container exit codes for the read path. Distinct from anything cat/ls return
# so "outside the workspace" can never be confused with "does not exist".
_READ_NOT_FOUND = 40
_READ_OUTSIDE_WORKSPACE = 41
_READ_NOT_FILE_OR_DIR = 42
_READ_CANNOT_VERIFY = 43

# A span read is for compiling exact context, not for dumping a file; a caller that
# needs more lines makes a second, explicit request.
MAX_SPAN_LINES = 2000
# Content is capped well below the exec transport's own output limit (16 MiB), so a
# large file comes back flagged ``truncated`` instead of silently clipped.
MAX_READ_BYTES = 8 * 1024 * 1024

# THE FD IS THE UNIT OF TRUST. The first version resolved the path with ``realpath``,
# checked containment, then ``cat``-ed the PATH again. Between those two walks a
# symlink could be swapped in (``mv --exchange`` in a loop: 1333 of 3000 reads served
# outside content labelled as a workspace file, audit 5279880810). Now the path is
# opened ONCE onto fd 3, the kernel's own record of what was opened
# (``readlink /proc/self/fd/3``) is containment-checked, and every later read goes
# through that same open file (``/proc/self/fd/3`` re-opens the SAME inode, it does
# not re-walk the path). Swapping the path after the open changes nothing we read.
# No /proc means we cannot verify, and that is a refusal (43), never a fallback.
_OPEN_AND_VERIFY = [
    "[ -d /proc/self/fd ] || exit {cannot}",
    '[ -e "$candidate" ] || exit {missing}',
    'exec 3<"$candidate" || exit {missing}',
    "real=$(readlink /proc/self/fd/3) || exit {cannot}",
    'case "$real" in "$workspace"|"$workspace"/*) ;; *) exit {outside} ;; esac',
    'rel="${{real#"$workspace"}}"',
]


def _open_and_verify(candidate: str, workspace_root: str) -> list[str]:
    codes = {
        "cannot": _READ_CANNOT_VERIFY,
        "missing": _READ_NOT_FOUND,
        "outside": _READ_OUTSIDE_WORKSPACE,
    }
    return [
        f"workspace=$(realpath -- {shlex.quote(workspace_root)}) || exit 1",
        f"candidate={shlex.quote(candidate)}",
        *(step.format(**codes) for step in _OPEN_AND_VERIFY),
    ]


def _write_path_command(candidate: str, workspace_root: str, content_b64: str) -> str:
    """Write *content_b64* (base64) to *candidate*, confined to the workspace root.

    The write mirrors the read path's FD discipline (coding-containers#166):
      1. before ``mkdir -p``, the deepest EXISTING ancestor is ``realpath``-resolved and must
         be inside the root, so no directory is ever created through a symlink that leaves it;
      2. after ``mkdir -p`` the parent is resolved again and checked;
      3. a symlink or directory as the final component is refused;
      4. the file is opened ONCE (``3<>``: creates it, does not truncate), the kernel's record
         of what was opened (``readlink /proc/self/fd/3``) must be inside the root, and the
         content is written through that same open file (``/proc/self/fd/3`` re-opens the SAME
         inode with truncation), so swapping the path after the check changes nothing written.
    Exits 41 outside, 42 not a regular file, 43 unverifiable (no /proc) — the read codes.
    """
    q = shlex.quote
    # Newline-joined: a step may itself contain "; " (case/if), so "\n" is the only
    # separator that never occurs inside a step (tests remove single steps by it).
    return "\n".join(
        [
            f"workspace=$(realpath -- {q(workspace_root)}) || exit 1",
            f"candidate={q(candidate)}",
            f"[ -d /proc/self/fd ] || exit {_READ_CANNOT_VERIFY}",
            'parent=$(dirname -- "$candidate")',
            'anc="$parent"; while [ ! -e "$anc" ] && [ ! -L "$anc" ]; do anc=$(dirname -- "$anc"); done',
            f'ra=$(realpath -- "$anc") || exit {_READ_OUTSIDE_WORKSPACE}',
            f'case "$ra" in "$workspace"|"$workspace"/*) ;; *) exit {_READ_OUTSIDE_WORKSPACE} ;; esac',
            'mkdir -p -- "$parent" || exit 1',
            f'rp=$(realpath -- "$parent") || exit {_READ_CANNOT_VERIFY}',
            f'case "$rp" in "$workspace"|"$workspace"/*) ;; *) exit {_READ_OUTSIDE_WORKSPACE} ;; esac',
            f'if [ -L "$candidate" ]; then exit {_READ_OUTSIDE_WORKSPACE}; fi',
            f'if [ -d "$candidate" ]; then exit {_READ_NOT_FILE_OR_DIR}; fi',
            'exec 3<>"$candidate" || exit 1',
            f"real=$(readlink /proc/self/fd/3) || exit {_READ_CANNOT_VERIFY}",
            f'case "$real" in "$workspace"/*) ;; *) exit {_READ_OUTSIDE_WORKSPACE} ;; esac',
            f"[ -f /proc/self/fd/3 ] || exit {_READ_NOT_FILE_OR_DIR}",
            f"printf %s {q(content_b64)} | base64 -d > /proc/self/fd/3",
        ]
    )


def _read_path_command(
    candidate: str,
    workspace_root: str,
    start: int | None = None,
    end: int | None = None,
) -> str:
    """Open *candidate* once, verify the OPENED file is inside the root, read via the fd.

    Output: ``DIR /<rel>`` + ``ls -la``, or ``FILE /<rel>``, ``SIZE <bytes>``,
    ``LINES <n>``, ``SHA <hex|->`` then the content (whole file capped at
    MAX_READ_BYTES, or ``sed``-selected span). SIZE/LINES/SHA are computed over the
    whole file's BYTES in the container, so they are right past the cap and for
    non-UTF-8 files. Exits 40 missing, 41 outside, 42 not file/dir, 43 unverifiable.
    """
    fd = "/proc/self/fd/3"
    if start is not None:
        assert end is not None
        body = f"sed -n '{int(start)},{int(end)}p' {fd} | head -c {MAX_READ_BYTES + 1}"
    else:
        body = f"head -c {MAX_READ_BYTES + 1} {fd}"
    return "; ".join(
        [
            *_open_and_verify(candidate, workspace_root),
            f"if [ -d {fd} ]; then printf 'DIR /%s\\n' \"${{rel#/}}\"; ls -la {fd}/; "
            f"elif [ -f {fd} ]; then "
            f"size=$(wc -c < {fd}); lines=$(wc -l < {fd}); "
            f'if [ "$size" -gt 0 ] && [ -n "$(tail -c 1 {fd})" ]; then lines=$((lines + 1)); fi; '
            f"if command -v sha256sum >/dev/null 2>&1; "
            f"then sha=$(sha256sum < {fd} | cut -d ' ' -f 1); else sha=-; fi; "
            f"printf 'FILE /%s\\nSIZE %s\\nLINES %s\\nSHA %s\\n' "
            f'"${{rel#/}}" "$size" "$lines" "$sha"; {body}; '
            f"else exit {_READ_NOT_FILE_OR_DIR}; fi",
        ]
    )


def _parse_ls(stdout: str) -> list[dict]:
    entries = []
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("total"):
            continue
        parts = line.split(None, 8)
        if len(parts) >= 9 and parts[8] not in {".", ".."}:
            entries.append(
                {
                    "permissions": parts[0],
                    "size": parts[4],
                    "modified": f"{parts[5]} {parts[6]} {parts[7]}",
                    "name": parts[8],
                    "type": "directory" if parts[0].startswith("d") else "file",
                }
            )
    return entries


def _raise_for_confined_exit(exit_code: int, path: str) -> None:
    if exit_code == _READ_NOT_FOUND:
        raise not_found("Path", path)
    if exit_code == _READ_OUTSIDE_WORKSPACE:
        raise bad_request("Invalid path: resolves outside the repository workspace")
    if exit_code == _READ_NOT_FILE_OR_DIR:
        raise bad_request("Invalid path: not a regular file or directory")
    if exit_code == _READ_CANNOT_VERIFY:
        raise internal_error(
            "Cannot verify the opened path is inside the workspace (no /proc in the "
            "workspace); refusing to read rather than trusting the path"
        )


@router.get(
    "/{container_name}/files",
    response_model=ContainerFilesResponse,
    operation_id="list_workspace_files",
    summary="List directory or read file (optionally a line span)",
    description=(
        "List a directory or read a file inside the repository workspace. `path` is "
        "relative to the repository root or under the `/workspace` alias; `..`, other "
        "absolute paths, encoded traversal and anything whose OPENED file lies outside "
        "the workspace (symlinks, including ones swapped in mid-request) are refused "
        "with 400. `start`/`end` (1-based, inclusive) return only that span, at most "
        f"{MAX_SPAN_LINES} lines. `size_bytes`, `lines_total` and `sha256` always "
        f"describe the whole file's bytes; whole-file content is capped at "
        f"{MAX_READ_BYTES} bytes and flagged `truncated`."
    ),
    responses={
        200: {"description": "File or directory listing returned", "model": ContainerFilesResponse},
        400: {"model": Error400Response, "description": "Path outside workspace or bad span"},
        404: {"model": Error404Response, "description": "Container or path not found"},
        500: {"model": Error500Response, "description": "Error accessing files"},
    },
)
async def get_container_files(
    container_name: str = Path(..., description="Container name"),
    path: str = Query(default="/workspace", description="File or directory path"),
    start: int | None = Query(None, ge=1, description="First line of a span (1-based)"),
    end: int | None = Query(None, ge=1, description="Last line of a span (inclusive)"),
    tenant_id: str = Depends(require_tenant),
):
    """List directory contents or read a file (or a line span) from a workspace."""
    try:
        container_manager = get_manager()

        if not _validate_git_file_path(path):
            raise bad_request("Invalid path: contains forbidden characters")
        try:
            container = container_manager.get_container(container_name, tenant_id=tenant_id)
        except ValueError:
            container = None
        if not container:
            raise not_found("Container", container_name)

        if end is not None and start is None:
            start = 1
        if start is not None:
            if end is None:
                end = start + MAX_SPAN_LINES - 1
            if end < start:
                raise bad_request(f"end ({end}) must be >= start ({start})")
            if end - start + 1 > MAX_SPAN_LINES:
                raise bad_request(
                    f"span {start}-{end} is {end - start + 1} lines; the maximum is "
                    f"{MAX_SPAN_LINES}. Request it in several spans."
                )

        workspace_root = _repository_root(container)
        try:
            candidate = _normalize_workspace_read_path(path, workspace_root=workspace_root)
        except ValueError as e:
            raise bad_request(f"Invalid path: {e}") from e

        result = await container_manager.execute_command(
            container_name,
            _read_path_command(candidate, workspace_root, start, end),
            tenant_id=tenant_id,
        )
        exit_code = result.get("exit_code", 1)
        _raise_for_confined_exit(exit_code, path)
        if exit_code != 0:
            stderr = result.get("stderr", "")
            if "not found" in stderr.lower():
                raise not_found("Container", container_name)
            raise internal_error(f"Failed to read path: {stderr}")

        stdout = result.get("stdout", "")
        header, _, body = stdout.partition("\n")
        kind, _, real_path = header.partition(" ")
        if kind not in {"DIR", "FILE"} or not real_path:
            raise internal_error("Failed to read path: unexpected response from workspace")
        # The shell printed the OPENED file's path relative to the resolved root.
        resolved = posixpath.normpath(f"{_WORKSPACE_ROOT}{real_path}")

        if kind == "DIR":
            if start is not None:
                raise bad_request("start/end apply to files; this path is a directory")
            entries = _parse_ls(body)
            for entry in entries:
                entry["repo"] = secondary_repo_of_path(posixpath.join(resolved, entry["name"]))
            return {
                "path": path,
                "type": "directory",
                "content": None,
                "entries": entries,
                "resolved_path": resolved,
                "repo": secondary_repo_of_path(resolved),
            }

        meta: dict[str, str] = {}
        for _ in range(3):
            line, _, body = body.partition("\n")
            key, _, value = line.partition(" ")
            meta[key] = value.strip()
        try:
            size_bytes = int(meta["SIZE"])
            lines_total = int(meta["LINES"])
        except (KeyError, ValueError) as e:
            raise internal_error("Failed to read path: malformed file metadata") from e
        sha = meta.get("SHA")

        content_bytes = len(body.encode("utf-8"))
        truncated = content_bytes > MAX_READ_BYTES
        if truncated:
            body = body.encode("utf-8")[:MAX_READ_BYTES].decode("utf-8", errors="ignore")
        payload = {
            "path": path,
            "type": "file",
            "content": body,
            "entries": None,
            "resolved_path": resolved,
            "repo": secondary_repo_of_path(resolved),
            "lines_total": lines_total,
            "size_bytes": size_bytes,
            "sha256": sha if sha and sha != "-" else None,
            "truncated": truncated if start is not None else size_bytes > MAX_READ_BYTES,
        }
        if start is not None:
            assert end is not None
            if start > lines_total and not (start == 1 and lines_total == 0):
                raise bad_request(
                    f"start ({start}) is beyond the end of the file (lines_total={lines_total})"
                )
            payload.update(start_line=start, end_line=min(end, lines_total))
        return payload

    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as e:
        logger.error("Failed to access files in container '%s': %s", container_name, e)
        raise internal_error(f"Failed to access files: {e}") from e


_SEARCH_MAX_RESULTS = 1000
_SEARCH_HIT = re.compile(r"^(?P<file>.*?):(?P<line>\d+):(?P<text>.*)$")
_FD_PATH = "/proc/self/fd/3"


def _search_command(
    candidate: str,
    workspace_root: str,
    *,
    pattern: str,
    regex: bool,
    case_sensitive: bool,
    limit: int,
) -> str:
    """Open and verify the search root exactly like a read, then grep THROUGH the fd.

    A directory root: ``cd /proc/self/fd/3`` pins the cwd to the directory that was
    opened and verified (the kernel resolves the fd, not the path), then
    ``grep -r ... .`` searches relative to it. ``-r`` (never ``-R``) does not follow
    symlinks met during recursion. A file root: grep reads ``/proc/self/fd/3``.
    ``-e`` always precedes the pattern, so a pattern like ``-v`` is a pattern and
    never an option. Output goes to a temp file so ``head`` caps it WITHOUT losing
    grep's exit status: 1 is "no match", >1 (incl. 124 timeout) is a failure.
    """
    flags = ["-rnHI", "--exclude-dir=.git", "-E" if regex else "-F"]
    if not case_sensitive:
        flags.append("-i")
    flag_str = " ".join(flags)
    pat = shlex.quote(pattern)
    return "; ".join(
        [
            *_open_and_verify(candidate, workspace_root),
            "out=$(mktemp) || exit 1",
            f"if [ -d {_FD_PATH} ]; then kind=DIR; cd {_FD_PATH} || exit {_READ_CANNOT_VERIFY}; "
            f'grep {flag_str} -e {pat} -- . > "$out"; rc=$?; '
            f"else kind=FILE; "
            f'grep {flag_str} -e {pat} -- {_FD_PATH} > "$out"; rc=$?; fi',
            'printf \'ROOT %s /%s\\n\' "$kind" "${rel#/}"',
            f'head -n {int(limit) + 1} "$out"',
            'rm -f "$out"',
            "exit $rc",
        ]
    )


@router.get(
    "/{container_name}/search",
    response_model=WorkspaceSearchResponse,
    operation_id="search_workspace",
    summary="Search workspace files for a literal or regex pattern",
    description=(
        "Line hits (`file`, `line`, `text`) for `pattern` under `path` (a file or a "
        "directory; default the whole repository). `regex=false` is a fixed-string "
        "match; `regex=true` is POSIX extended regex. `.git` and binary files are "
        "skipped and symlinks inside the tree are not followed. The search root is "
        "opened once and verified like `/files`. At most `max_results` hits are "
        "returned and `truncated` says whether more exist. A search that fails or "
        "times out is 500, never an empty hit list."
    ),
    responses={
        200: {"description": "Search completed", "model": WorkspaceSearchResponse},
        400: {"model": Error400Response, "description": "Invalid pattern or path"},
        404: {"model": Error404Response, "description": "Container or path not found"},
        500: {"model": Error500Response, "description": "Search failed"},
    },
)
async def search_workspace(
    container_name: str = Path(..., description="Container name"),
    pattern: str = Query(..., min_length=1, max_length=500, description="Search pattern"),
    regex: bool = Query(False, description="Treat pattern as POSIX extended regex"),
    path: str = Query(default="/workspace", description="File or directory to search"),
    case_sensitive: bool = Query(True, description="Case-sensitive match"),
    max_results: int = Query(100, ge=1, le=_SEARCH_MAX_RESULTS, description="Hit cap"),
    tenant_id: str = Depends(require_tenant),
) -> WorkspaceSearchResponse:
    """Typed grep over the caller's workspace."""
    try:
        container_manager = get_manager()

        if any(ord(ch) < 32 and ch != "\t" for ch in pattern) or "\x7f" in pattern:
            raise bad_request("pattern must not contain control characters or newlines")
        if not _validate_git_file_path(path):
            raise bad_request("Invalid path: contains forbidden characters")
        try:
            container = container_manager.get_container(container_name, tenant_id=tenant_id)
        except ValueError:
            container = None
        if not container:
            raise not_found("Container", container_name)

        workspace_root = _repository_root(container)
        try:
            candidate = _normalize_workspace_read_path(path, workspace_root=workspace_root)
        except ValueError as e:
            raise bad_request(f"Invalid path: {e}") from e

        result = await container_manager.execute_command(
            container_name,
            _search_command(
                candidate,
                workspace_root,
                pattern=pattern,
                regex=regex,
                case_sensitive=case_sensitive,
                limit=max_results,
            ),
            tenant_id=tenant_id,
            timeout=60,
        )
        exit_code = result.get("exit_code", 1)
        _raise_for_confined_exit(exit_code, path)
        if exit_code not in (0, 1):
            raise internal_error(
                f"Search failed (exit {exit_code}): {(result.get('stderr') or '').strip()[:300]}"
            )

        header, _, body = (result.get("stdout") or "").partition("\n")
        parts = header.split(" ", 2)
        if len(parts) != 3 or parts[0] != "ROOT" or parts[1] not in {"DIR", "FILE"}:
            raise internal_error("Search failed: unexpected response from workspace")
        kind, root_rel = parts[1], parts[2].strip("/")

        hits: list[WorkspaceSearchHit] = []
        for line in body.splitlines():
            m = _SEARCH_HIT.match(line)
            if not m:
                continue
            if kind == "FILE":
                file_path = root_rel
            else:
                inner = m.group("file").removeprefix("./")
                file_path = posixpath.join(root_rel, inner) if root_rel else inner
            hits.append(
                WorkspaceSearchHit(
                    file=file_path,
                    line=int(m.group("line")),
                    text=m.group("text"),
                    repo=secondary_repo_of_path(file_path),
                )
            )
        truncated = len(hits) > max_results
        hits = hits[:max_results]

        return WorkspaceSearchResponse(
            pattern=pattern,
            regex=regex,
            path=f"{_WORKSPACE_ROOT}/{root_rel}" if root_rel else _WORKSPACE_ROOT,
            hits=hits,
            total_returned=len(hits),
            truncated=truncated,
        )

    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as e:
        logger.error("Failed to search container '%s': %s", container_name, e)
        raise internal_error(f"Failed to search: {e}") from e

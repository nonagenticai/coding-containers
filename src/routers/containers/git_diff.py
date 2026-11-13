"""Container git route: typed, read-only diff.

WHY THIS EXISTS. Worker inference in the conveyor is single-turn with no tools, so the
workflow has to compile exact context (including "what changed") BEFORE the call. The
only way to get a diff out of a workspace was ``POST /exec`` with a hand-built shell
string: untyped, a producer by role, and indistinguishable from any other command.
``git/status`` lists changed paths but carries no content. This route is the typed
observer for the diff itself.

Three answers are kept apart on purpose, because an empty string cannot tell them apart:

- unknown ref           -> 404, never an empty diff
- git itself failed     -> 500, never an empty diff
- genuinely no changes  -> 200 with ``diff == ""`` and ``files == []``

and a working-tree comparison publishes ``untracked_files`` separately, because ``git
diff`` never shows untracked files and silently dropping them would read as "no change".

THE DIFF STARTS WHERE THE WORKSPACE STARTED. ``base`` may be a branch name while the
workspace was cloned at a pinned commit (RM ``repositories[].sha``); a two-point diff
against the branch tip would then report every commit between the pin and the tip as
REMOVED. The base side is therefore ``merge-base(base, head)`` -- what a pull request
shows -- which is the base commit itself whenever the base ref is an ancestor.

ONE SNAPSHOT (coding-containers#156). A working-tree comparison first freezes the whole
workspace into a tree object in ONE exec (``_snapshot_command``); ``diff``, ``files``,
``untracked_files`` and ``blobs`` are all read from that tree id, never again from the
mutable worktree, and the response returns it as ``tree`` for the commit and push routes
(``approved_tree``) to bind to. Every read exec runs its own poison and object-integrity
check (``_checked_git``).
"""

from __future__ import annotations

import logging
import posixpath
import re
import shlex
from uuid import uuid7

import docker.errors
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from coding_client.auth.models import User
from coding_client.auth.tenant import get_current_user_with_tenant
from coding_client.models.responses import bad_request, internal_error, not_found
from src.models.container_api_models import GitDiffFile, GitDiffResponse
from src.models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error422Response,
    Error500Response,
)

from ._common import (
    _WORKSPACE_ROOT,
    _with_subcommand_flags,
    get_git_manager,
    workspace_git_function,
    workspace_git_read,
    workspace_object_check,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# A ref we are willing to hand to git. Leading ``-`` is excluded so a ref can never be
# parsed as an option (``--output=...`` would write a file). ``..`` is excluded so the
# caller names two explicit endpoints instead of smuggling range syntax (``a..b``,
# ``a...b``) through one parameter. Everything is shlex-quoted as well; this pattern is
# the first fence, not the only one.
_REF_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._/~^-]{0,199}$")

DEFAULT_MAX_BYTES = 200_000


def _validate_ref(ref: str, name: str) -> str:
    value = ref.strip()
    if not _REF_PATTERN.fullmatch(value) or ".." in value:
        raise bad_request(
            f"Invalid {name} ref '{ref}': must be a branch, tag or commit "
            f"(no leading '-', no '..' range syntax, no reflog or shell syntax)"
        )
    return value


def _validate_diff_path(path: str) -> str:
    """Return *path* relative to the repository root, or raise 400.

    Absolute paths are accepted only under the ``/workspace`` alias. Anything that
    normalises outside the repository (``..`` segments, other absolute paths) is
    refused rather than rewritten.
    """
    stripped = path.strip()
    if not stripped:
        raise bad_request("path must not be empty")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in stripped):
        raise bad_request("path must not contain control characters")
    if ".." in stripped.split("/"):
        raise bad_request("Invalid path: path traversal is not permitted")
    if posixpath.isabs(stripped):
        normalized = posixpath.normpath(stripped)
        if normalized == _WORKSPACE_ROOT:
            return "."
        if not normalized.startswith(f"{_WORKSPACE_ROOT}/"):
            raise bad_request("path must be inside the repository workspace")
        return posixpath.relpath(normalized, _WORKSPACE_ROOT)
    return posixpath.normpath(stripped)


def _git(args: list[str]) -> str:
    """Build a hardened read-only workspace git command (see ``workspace_git_read``)."""
    return workspace_git_read(args)


async def _resolve_commit(manager, container_name: str, ref: str, tenant_id: str | None) -> str:
    result = await manager.execute_command(
        container_name,
        _git(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"]),
        tenant_id=tenant_id,
    )
    exit_code = result.get("exit_code", 1)
    sha = (result.get("stdout") or "").strip()
    # ``rev-parse --verify --quiet`` exits 1 with no output for an unknown ref, and
    # 128 when it cannot run at all (not a repository). Only the first is a 404.
    if exit_code == 1:
        raise not_found("Git ref", ref)
    if exit_code != 0 or not sha:
        raise internal_error(
            f"Failed to resolve ref '{ref}': {(result.get('stderr') or '').strip()[:300]}"
        )
    return sha


async def _diff_base_commit(
    manager, container_name: str, base_sha: str, head: str, tenant_id: str | None
) -> str:
    """The commit this diff actually starts from: ``git merge-base <base> <head>``.

    A review diff must show what the workspace changed FROM WHERE IT STARTED, not from
    wherever its base ref happens to point now. A project can pin its repository to a
    commit (RM ``repositories[].sha``): the workspace is then cloned at that commit
    while ``base`` stays the branch name, so a plain two-point diff against the branch
    tip reports every commit between the pin and the tip as REMOVED -- work the project
    never did, and not what its pull request will show. GitHub compares a PR from the
    merge base for the same reason.

    For an unpinned workspace the base ref is an ancestor of ``head``, the merge base
    equals the base commit, and nothing changes.

    No common ancestor, or a git failure, falls back to ``base_sha``: this route's
    contract stays "diff from what you asked for", never an empty diff.
    """
    result = await manager.execute_command(
        container_name, _git(["merge-base", base_sha, head]), tenant_id=tenant_id
    )
    first = ((result.get("stdout") or "").splitlines() or [""])[0].strip()
    if result.get("exit_code", 1) != 0 or not _OID_RE.match(first):
        return base_sha
    return first


def _parse_name_status(output: str) -> list[GitDiffFile]:
    files: list[GitDiffFile] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0][:1]
        if status in {"R", "C"} and len(parts) >= 3:
            files.append(GitDiffFile(status=status, path=parts[2], old_path=parts[1]))
        elif len(parts) >= 2:
            files.append(GitDiffFile(status=status, path=parts[1]))
    return files


def _truncate_utf8(text: str, max_bytes: int) -> tuple[str, int, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, len(encoded), False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), len(encoded), True


# One exec per chunk of paths; the whole script is one ``sh -c`` argument and must stay
# below Linux MAX_ARG_STRLEN (131072) even after a backend re-quotes it.
_BLOBS_ARG_BUDGET = 64 * 1024

_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _object_check(base_sha: str, tree: str, tracked: str) -> str:
    """``workspace_object_check`` over everything this route's decision reads: the base
    commit (whose tree lands in ``$_vbt``), ``tree``/``tracked`` and every object that
    differs between them and the base tree. Each read exec runs it FIRST, itself: an
    object rewritten in place after an earlier exec is refused here, not trusted."""
    trees = list(dict.fromkeys([tree, tracked]))
    pairs = [('"$_vbt"', t) for t in trees] + ([(tracked, tree)] if tracked != tree else [])
    return workspace_object_check(base_sha, trees, pairs)


def _checked_git(base_sha: str, tree: str, tracked: str, args: list[str]) -> str:
    """``workspace_git_read(args)``, preceded IN THE SAME EXEC by ``_object_check``. An
    ``args`` entry ``"$_vbt"`` is passed unquoted: the base tree id read from the
    VERIFIED base commit bytes, never ``<base>^{tree}`` (a second, unverified read)."""
    words = " ".join(a if a == '"$_vbt"' else shlex.quote(a) for a in _with_subcommand_flags(args))
    return (
        f"cd {_WORKSPACE_ROOT} && {workspace_git_function('_g')}"
        f"{_object_check(base_sha, tree, tracked)}_g {words}"
    )


def _head_tree_command(head_sha: str) -> str:
    """Print ``head_sha``'s tree id, read from its VERIFIED bytes (see
    ``workspace_object_check``); refuses a head commit that does not hash to its id."""
    return (
        f"cd {_WORKSPACE_ROOT} && {workspace_git_function('_g')}"
        f"{workspace_object_check(head_sha, [], [])}"
        'printf "%s\\n" "$_vbt"'
    )


# The snapshot exec's shell variables holding TREE and TRACKED.
_SNAP_TREE = '"$_a"'
_SNAP_TRACKED = '"$_b"'


def _snapshot_command(base_sha: str) -> str:
    """ONE exec that freezes the working tree into two tree objects.

    WHY (coding-containers#156 validator). Code that ran in the workspace can leave a
    background process behind (a test server, a dependency's watcher) that runs as the
    same uid and rewrites files between any two execs. When the patch text and the
    blob ids came from separate reads of the MUTABLE worktree, QA reviewed one content
    and ``blobs`` (then the commit, then the push check) carried another. Everything a
    later decision trusts is therefore derived from ONE immutable snapshot, named by
    object id:

    1. copy the real index to a temp index (``GIT_INDEX_FILE``), so the real index is
       never touched and ``add -A`` starts from exactly what the commit route's
       ``add -A`` starts from. ``cp -p``: git treats an entry as "racily clean" (and
       re-reads the file) only when the INDEX file is not newer than it; a copy with a
       fresh mtime made a same-second, same-size edit look unchanged (measured);
    2. ``ls-files -z --others --exclude-standard`` -> the untracked paths;
    3. ``add -A`` + ``write-tree`` -> ``TREE``: what a commit of this workspace holds;
    4. ``update-index -z --force-remove --stdin`` of the untracked paths +
       ``write-tree`` -> ``TRACKED``: ``TREE`` minus the created files, derived from
       the SAME index (no second worktree read), so ``diff`` keeps its contract
       (created files are listed in ``untracked_files``, never in ``diff``).

    5. ``workspace_object_check`` over both trees and ``base_sha``: ``write-tree`` keeps
       an object that already exists, so a planted in-place object would otherwise be
       part of the "snapshot".

    Prints ``TREE`` then ``TRACKED``, one per line. Every later read (name-status,
    patch, untracked listing, blobs) names these ids, never the worktree, and the
    commit/push routes bind to ``TREE`` (``approved_tree``). A file a background
    process swaps after this exec changes nothing that was reviewed; one swapped DURING
    it lands in ``TREE`` and is therefore in what QA reads. The temp files live in the
    git dir (always writable; the root filesystem may not be) and are removed on exit.
    Hardened like every workspace git (``workspace_git_function``): replace refs,
    grafts and alternates are refused before anything is read.
    """
    return (
        f"cd {_WORKSPACE_ROOT} && {workspace_git_function('_g')}"
        "_gd=$(_g rev-parse --git-dir) || exit 1; "
        '_t=$(mktemp "$_gd/snapshot-index.XXXXXX") || exit 1; '
        '_u=$(mktemp "$_gd/snapshot-untracked.XXXXXX") || { rm -f "$_t"; exit 1; }; '
        'trap \'rm -f "$_t" "$_u"\' EXIT; '
        "_i=$(_g rev-parse --git-path index) || exit 1; "
        'if [ -f "$_i" ]; then cp -p "$_i" "$_t" || exit 1; else rm -f "$_t"; fi; '
        'export GIT_INDEX_FILE="$_t"; '
        '_g ls-files -z --others --exclude-standard > "$_u" || exit 1; '
        "_g add -A || exit 1; "
        "_a=$(_g write-tree) || exit 1; "
        '_g update-index -z --force-remove --stdin < "$_u" || exit 1; '
        "_b=$(_g write-tree) || exit 1; "
        # A tree id ``write-tree`` computed is only trusted once its objects hash to their
        # ids: an object that already existed (planted in place) is kept, not rewritten.
        f"{_object_check(base_sha, _SNAP_TREE, _SNAP_TRACKED)}"
        'printf "%s\\n%s\\n" "$_a" "$_b"'
    )


def _parse_snapshot(stdout: str) -> tuple[str, str] | None:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if len(lines) != 2 or not all(_OID_RE.match(x) for x in lines):
        return None
    return lines[0], lines[1]


def _blobs_command(base_sha: str, tree: str, tracked: str, paths: list[str]) -> str:
    """Print ``<path>\\t<blob id or ->`` per path (NUL-terminated records, in order).

    Read from ``tree`` (the snapshot, or the head commit's tree), NEVER from the
    worktree: ``ls-tree`` of the exact entry; a blob entry's id, anything else (a
    gitlink, a tree) or no entry is ``-``, so a gitlink change always fails the push
    content check closed. Hardened like every workspace git (``workspace_git_function``):
    replace refs, grafts and alternates are refused before anything is read, in THIS
    exec (a background process can plant them after the snapshot exec ran), and so is
    an object that does not hash to its id (``_object_check``).
    """
    if not _OID_RE.match(tree):
        raise ValueError("blobs are read from a tree id")
    per_path = []
    for p in paths:
        q = shlex.quote(p)
        per_path.append(
            f"b=$(_g --literal-pathspecs ls-tree -z --full-tree {tree} -- {q} "
            f"| tr '\\0' '\\n' | P={q} awk '{{ i = index($0, \"\\t\"); "
            'if (substr($0, i + 1) == ENVIRON["P"]) { split(substr($0, 1, i - 1), a, " "); '
            'if (a[2] == "blob") print a[3]; exit } }\'); '
            f"printf '%s\\t%s\\0' {q} \"${{b:--}}\";"
        )
    return (
        f"cd {_WORKSPACE_ROOT} && {workspace_git_function('_g')}"
        f"{_object_check(base_sha, tree, tracked)}" + " ".join(per_path)
    )


async def _content_blobs(
    manager,
    container_name: str,
    tenant_id: str | None,
    ids: tuple[str, str, str],
    files: list[GitDiffFile],
    untracked: list[str],
) -> dict[str, str | None]:
    """``path -> blob id | None`` for every changed/untracked path, read from the tree in
    ``ids`` = ``(base_sha, tree, tracked)`` (see ``_blobs_command``). A failed or short read is a 500: an unread blob is never
    reported as absent."""
    paths: list[str] = []
    for f in files:
        paths.append(f.path)
        if f.old_path:
            paths.append(f.old_path)
    paths.extend(untracked)
    paths = list(dict.fromkeys(paths))
    out: dict[str, str | None] = {}
    chunk: list[str] = []
    chunks: list[list[str]] = []
    size = 0
    for p in paths:
        weight = 4 * len(p.encode("utf-8", "surrogateescape")) + 400
        if chunk and size + weight > _BLOBS_ARG_BUDGET:
            chunks.append(chunk)
            chunk, size = [], 0
        chunk.append(p)
        size += weight
    if chunk:
        chunks.append(chunk)
    for part in chunks:
        result = await manager.execute_command(
            container_name, _blobs_command(*ids, part), tenant_id=tenant_id
        )
        records = [r for r in (result.get("stdout") or "").split("\0") if r]
        if result.get("exit_code", 1) != 0 or len(records) != len(part):
            raise internal_error(
                f"Failed to read content blob ids: {(result.get('stderr') or '').strip()[:300]}"
            )
        for want, record in zip(part, records, strict=True):
            path, tab, blob = record.rpartition("\t")
            if (
                not tab
                or path != want
                or not (blob == "-" or re.fullmatch(r"[0-9a-f]{40,64}", blob))
            ):
                raise internal_error(f"Failed to read content blob ids: bad record {record!r}")
            out[path] = None if blob == "-" else blob
    return out


@router.get(
    "/{container_name}/git/diff",
    response_model=GitDiffResponse,
    operation_id="get_workspace_git_diff",
    summary="Get Git Diff",
    description=(
        "Unified diff between `base` and `head` commits, or between `base` and the "
        "working tree when `head` is omitted. Read-only. Unknown refs are 404, never "
        "an empty diff; untracked files are listed in `untracked_files`, not `diff`. "
        "The comparison starts at `merge-base(base, head)` — what a pull request shows "
        "— so a workspace imported at a pinned commit never reports the branch's later "
        "commits as removed; for an ancestor base this is the base commit itself."
    ),
    responses={
        200: {"description": "Diff computed"},
        400: {"model": Error400Response, "description": "Invalid ref or path"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Container or ref not found"},
        422: {"model": Error422Response, "description": "Validation error"},
        500: {"model": Error500Response, "description": "git failed"},
    },
)
async def get_git_diff(
    container_name: str = Path(..., description="Container name"),
    base: str = Query("HEAD", description="Base commit-ish (branch, tag, sha, HEAD~n)"),
    head: str | None = Query(
        None, description="Head commit-ish; omit to compare against the working tree"
    ),
    path: str | None = Query(None, description="Limit the diff to this repo-relative path"),
    context_lines: int = Query(3, ge=0, le=50, description="Unified context lines"),
    max_bytes: int = Query(
        DEFAULT_MAX_BYTES, ge=1, le=2_000_000, description="Truncate `diff` at this size"
    ),
    user: User = Depends(get_current_user_with_tenant),
) -> GitDiffResponse:
    """Return a typed diff of the caller's workspace."""
    try:
        container_manager = get_git_manager()

        tenant_id = None if user.is_admin() else user.effective_tenant_id
        try:
            container = container_manager.get_container(container_name, tenant_id=tenant_id)
        except ValueError:
            container = None
        if not container:
            raise not_found("Container", container_name)
        if container.status.value.lower() != "running":
            raise bad_request(
                f"Container '{container_name}' is not running "
                f"(current status: {container.status.value})"
            )

        base_ref = _validate_ref(base, "base")
        head_ref = _validate_ref(head, "head") if head is not None and head.strip() else None
        pathspec = _validate_diff_path(path) if path is not None and path.strip() else None

        base_sha = await _resolve_commit(container_manager, container_name, base_ref, tenant_id)
        head_sha = (
            await _resolve_commit(container_manager, container_name, head_ref, tenant_id)
            if head_ref
            else None
        )
        # The diff starts where the workspace STARTED, not where its base ref points
        # now — a workspace imported at a pinned commit must not report the commits
        # between that commit and the branch tip as removed. See ``_diff_base_commit``.
        diff_base_sha = await _diff_base_commit(
            container_manager, container_name, base_sha, head_sha or "HEAD", tenant_id
        )

        # ONE immutable snapshot binds everything below (see ``_snapshot_command``):
        # ``tree`` is what a commit of this workspace holds, ``tracked`` the same tree
        # without the created (untracked) files. Every later read names these ids, so a
        # background process rewriting the worktree between execs cannot make the diff,
        # the blobs and the committed content disagree. A head commit is already
        # immutable: its tree is both.
        if head_sha is None:
            snap = await container_manager.execute_command(
                container_name, _snapshot_command(diff_base_sha), tenant_id=tenant_id
            )
            parsed = (
                _parse_snapshot(snap.get("stdout") or "") if snap.get("exit_code", 1) == 0 else None
            )
            if parsed is None:
                raise internal_error(
                    "Failed to snapshot the working tree: "
                    f"{(snap.get('stderr') or '').strip()[:300]}"
                )
            tree, tracked = parsed
        else:
            tree_result = await container_manager.execute_command(
                container_name, _head_tree_command(head_sha), tenant_id=tenant_id
            )
            tree = (tree_result.get("stdout") or "").strip()
            if tree_result.get("exit_code", 1) != 0 or not _OID_RE.match(tree):
                raise internal_error(
                    f"Failed to resolve the tree of {head_sha}: "
                    f"{(tree_result.get('stderr') or '').strip()[:300]}"
                )
            tracked = tree

        # Resolved ids, not the caller's strings or the worktree, go into the diff
        # itself: nothing can move between resolution and diff, and the response names
        # exactly what was compared. The base side is the tree read from the VERIFIED
        # base commit bytes (``"$_vbt"``, see ``_checked_git``).
        ids = (diff_base_sha, tree, tracked)
        endpoints = ['"$_vbt"', tracked]
        tail = ["--", pathspec] if pathspec else ["--"]
        # ``--text``: a ``-diff``/``binary`` attribute (``.gitattributes`` or the
        # workspace-owned ``.git/info/attributes``) otherwise turns a text change into
        # "Binary files differ", so the content QA reviews and the line cap counts would
        # be blank while the real bytes still ship.
        common = ["--no-color", "--no-ext-diff", "--no-textconv", "--text"]

        name_status = await container_manager.execute_command(
            container_name,
            _checked_git(
                *ids,
                [
                    "-c",
                    "core.quotePath=false",
                    "diff",
                    *common,
                    "--name-status",
                    "-M",
                    *endpoints,
                    *tail,
                ],
            ),
            tenant_id=tenant_id,
        )
        if name_status.get("exit_code", 1) != 0:
            raise internal_error(
                f"git diff --name-status failed: {(name_status.get('stderr') or '').strip()[:300]}"
            )

        patch = await container_manager.execute_command(
            container_name,
            _checked_git(*ids, ["diff", *common, "-M", f"-U{context_lines}", *endpoints, *tail]),
            tenant_id=tenant_id,
        )
        if patch.get("exit_code", 1) != 0:
            raise internal_error(f"git diff failed: {(patch.get('stderr') or '').strip()[:300]}")

        # The created files: exactly the entries ``tree`` has and ``tracked`` lacks,
        # read tree-to-tree (never the worktree), limited by the same pathspec.
        untracked: list[str] = []
        if tracked != tree:
            others = await container_manager.execute_command(
                container_name,
                _checked_git(
                    *ids,
                    [
                        "diff",
                        "--no-color",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--no-renames",
                        "--name-only",
                        "-z",
                        tracked,
                        tree,
                        *tail,
                    ],
                ),
                tenant_id=tenant_id,
            )
            if others.get("exit_code", 1) != 0:
                raise internal_error(
                    f"git diff (created files) failed: {(others.get('stderr') or '').strip()[:300]}"
                )
            untracked = [p for p in (others.get("stdout") or "").split("\0") if p]

        diff_text, diff_bytes, truncated = _truncate_utf8(patch.get("stdout") or "", max_bytes)
        files = _parse_name_status(name_status.get("stdout") or "")
        blobs = await _content_blobs(
            container_manager, container_name, tenant_id, ids, files, untracked
        )

        return GitDiffResponse(
            message=(
                f"{len(files)} changed path(s)"
                + (f", {len(untracked)} untracked" if untracked else "")
                + (f" (diff truncated at {max_bytes} bytes)" if truncated else "")
            ),
            container_name=container_name,
            base=base_ref,
            base_sha=diff_base_sha,
            head=head_ref,
            head_sha=head_sha,
            compared_to="commit" if head_sha else "working_tree",
            path=pathspec,
            context_lines=context_lines,
            files=files,
            untracked_files=untracked,
            diff=diff_text,
            diff_bytes=diff_bytes,
            truncated=truncated,
            blobs=blobs,
            tree=tree,
            request_id=str(uuid7()),
        )

    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as e:
        logger.error("Failed to get git diff in container '%s': %s", container_name, e)
        raise internal_error(f"Failed to get git diff: {e}") from e

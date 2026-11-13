"""Container git routes: status + commit + restore + branch."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
from uuid import uuid7

import docker.errors
from fastapi import APIRouter, Body, Depends, HTTPException, Path

from coding_client.auth.models import User
from coding_client.auth.tenant import get_current_user_with_tenant
from coding_client.models.responses import (
    bad_request,
    internal_error,
    not_found,
)
from src.models.container_api_models import (
    GitBranchRequest,
    GitBranchResponse,
    GitCommitRequest,
    GitCommitResponse,
    GitRestoreRequest,
    GitRestoreResponse,
    GitStatusResponse,
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
    _sanitize_shell_arg,
    _validate_git_branch_name,
    _validate_git_file_path,
    get_git_manager,
    workspace_git_function,
    workspace_git_mutate,
    workspace_git_read,
    workspace_object_check,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _parse_porcelain_z(output: str) -> list[tuple[str, str, str | None]]:
    """Parse ``git status --porcelain=v1 -z`` into ``(XY, path, original)`` triples.

    Each entry is ``XY<space><path>`` terminated by NUL; paths are never quoted and
    may contain spaces or newlines. A rename/copy (``R``/``C`` in either column) is
    followed by a SEPARATE NUL-terminated entry holding the original path, returned
    as ``original`` (``None`` for every other entry). Nothing is stripped: the status columns are positional, and
    stripping the output (the old newline parser did) ate the leading space of an
    unstaged first entry and shifted its path by one character.
    """
    parts = output.split("\0")
    result: list[tuple[str, str, str | None]] = []
    i = 0
    while i < len(parts):
        entry = parts[i]
        i += 1
        if len(entry) < 4:  # the empty tail after the final NUL, or a stray fragment
            continue
        status_code, path = entry[:2], entry[3:]
        original = None
        if "R" in status_code or "C" in status_code:
            original = parts[i] if i < len(parts) else None
            i += 1
        result.append((status_code, path, original))
    return result


@router.get(
    "/{container_name}/git/status",
    response_model=GitStatusResponse,
    operation_id="get_workspace_git_status",
    summary="Get Git Status",
    responses={
        200: {"description": "Git status retrieved successfully"},
        400: {"model": Error400Response, "description": "Invalid request parameters"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Container not found"},
        406: {"model": Error406Response, "description": "Not acceptable"},
        422: {"model": Error422Response, "description": "Validation error"},
        429: {"model": Error429Response, "description": "Too many requests"},
        500: {"model": Error500Response, "description": "Error getting git status"},
    },
)
async def get_git_status(
    container_name: str = Path(..., description="Container name"),
    user: User = Depends(get_current_user_with_tenant),
) -> GitStatusResponse:
    """Get git status from container to see what files have changed."""
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
                f"Container '{container_name}' is not running (current status: {container.status.value})"
            )

        branch_result = await container_manager.execute_command(
            container_name,
            workspace_git_read(["rev-parse", "--abbrev-ref", "HEAD"]),
            tenant_id=tenant_id,
        )
        branch = branch_result.get("stdout", "unknown").strip()

        status_result = await container_manager.execute_command(
            container_name,
            workspace_git_read(["status", "--porcelain=v1", "-z"]),
            tenant_id=tenant_id,
        )

        if status_result.get("exit_code", 1) != 0:
            error_msg = status_result.get("stderr", "Unknown error")
            raise internal_error(f"Failed to get git status: {error_msg}")

        status_output = status_result.get("stdout", "")
        entries = _parse_porcelain_z(status_output)
        modified_files = []
        new_files = []
        deleted_files = []
        staged_changes = 0
        unstaged_changes = 0

        for status_code, file_path, original in entries:
            if status_code[0] != " " and status_code[0] != "?":
                staged_changes += 1

            if status_code[1] != " ":
                unstaged_changes += 1

            if "R" in status_code and original:
                # A rename also REMOVES its original path. Scope checks (n8n Check
                # Worktree Scope) must see it, or an out-of-scope file can be deleted
                # by a commit whose only reported path is in scope.
                deleted_files.append(original)

            if "D" in status_code:
                deleted_files.append(file_path)
            elif "?" in status_code or "A" in status_code:
                new_files.append(file_path)
            elif "M" in status_code or "R" in status_code or "C" in status_code:
                # The NEW path of a rename/copy (a copy leaves its original in place).
                modified_files.append(file_path)

        message = "Git status retrieved successfully"
        if not entries:
            message = "Git status retrieved successfully - no changes"

        logger.info(
            "Git status for container '%s': %d modified, %d new, %d deleted",
            container_name,
            len(modified_files),
            len(new_files),
            len(deleted_files),
        )

        return GitStatusResponse(
            message=message,
            container_name=container_name,
            modified_files=modified_files,
            new_files=new_files,
            deleted_files=deleted_files,
            staged_changes=staged_changes,
            unstaged_changes=unstaged_changes,
            branch=branch,
            request_id=str(uuid7()),
        )

    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as e:
        logger.error("Failed to get git status in container '%s': %s", container_name, e)
        raise internal_error(f"Failed to get git status: {str(e)}") from e


# Reads the TARGET repo's `[tool.black]` with tomllib (the runtime image's python3 is
# 3.14) and prints one fixed token per black setting `ruff format` must be told about.
# Both key spellings are accepted (black normalises `line_length` to `line-length`).
# Only an int line-length and literal `true` flags are reported; anything else is
# dropped. A regex/awk read of the file is NOT a substitute: `line-length = 100  # was
# 88` reads as 10088.
_BLACK_PROBE_PY = """
import tomllib
try:
    with open("pyproject.toml", "rb") as fh:
        black = tomllib.load(fh).get("tool", {}).get("black")
    if isinstance(black, dict):
        def get(key):
            for name in (key, key.replace("-", "_")):
                if name in black:
                    return black[name]
        n = get("line-length")
        if type(n) is int and 1 <= n <= 320:
            print("black-line-length:%d" % n)
        if get("skip-string-normalization") is True:
            print("black-skip-string-normalization")
        if get("skip-magic-trailing-comma") is True:
            print("black-skip-magic-trailing-comma")
except Exception:
    pass
"""


def _black_ruff_overrides(present: set[str]) -> list[str]:
    """`ruff format --config` overrides equivalent to the probed `[tool.black]` settings.

    Every override is a constant or an int re-rendered here, so nothing the target repo
    wrote reaches the shell. ruff's own line-length range is 1..320.
    """
    overrides: list[str] = []
    for tok in present:
        if tok.startswith("black-line-length:"):
            raw = tok.split(":", 1)[1]
            if raw.isascii() and raw.isdigit() and 1 <= int(raw) <= 320:
                overrides.append(f"line-length={int(raw)}")
                break
    if "black-skip-string-normalization" in present:
        overrides.append('format.quote-style="preserve"')
    if "black-skip-magic-trailing-comma" in present:
        overrides.append("format.skip-magic-trailing-comma=true")
    return overrides


async def _format_staged_files(
    container_manager, container_name: str, tenant_id: str | None
) -> str:
    """Run the TARGET repo's own formatter over the staged files, then re-stage them.

    WHY THIS EXISTS. The conveyor's generated fixes were functionally correct and still
    failed CI, because nothing between "write the fix" and "open the PR" ran the repo's
    own style gate. `verify_fix_green` runs the card's test command and nothing else, so
    a diff that satisfies the oracle can still be rejected by `ruff format --check`. That
    is not a property of one card: it recurs for every repo with a format gate, and the
    delivery sweep then refuses the merge — correctly, but permanently.

    ⛔ IT FAILS OPEN, DELIBERATELY - BUT ONLY WITH RESPECT TO FORMATTING FAILURES. A
    formatter that is absent, unconfigured or broken must never destroy a correct fix;
    the repo's CI remains the authority on style, and this only removes the most common
    reason a good change arrives unmergeable. This does NOT mean the request itself
    survives everything: an ``asyncio.CancelledError`` (the client disconnected, the
    server is shutting down, ...) still propagates out of the caller and aborts the
    commit, exactly as it must. Every skip and every failure is logged with its reason so
    a silent no-op is distinguishable from a silent success.

    ⚠️ Scope is the STAGED set only. Formatting the whole tree would put unrelated files
    in the conveyor's commit, which is how a narrow fix turns into an unreviewable diff.

    ⚠️ Deleted paths are excluded. ``git diff --cached --name-status`` (not
    ``--name-only``) is used so staged deletions can be told apart from paths that still
    exist: formatting a deleted path errors, and ``git add`` on a deleted path exits 128,
    which would poison the whole re-stage for one unrelated removal.

    ⚠️ A path whose basename starts with ``-`` is dropped outright, never just quoted.
    ``_sanitize_shell_arg`` is ``shlex.quote``, which leaves a leading-dash argument like
    ``--plugin=evil.js`` UNQUOTED (it contains no character ``shlex.quote`` needs to
    escape), so a staged file with that name would be parsed as a flag by the formatter
    it is handed to. Every command below also gets a ``--`` terminator as a second,
    independent guard, but the terminator alone is not trusted here.
    """
    staged = await container_manager.execute_command(
        container_name,
        workspace_git_read(["diff", "--cached", "--no-ext-diff", "--no-textconv", "--name-status"]),
        tenant_id=tenant_id,
    )
    entries = [ln for ln in staged.get("stdout", "").splitlines() if ln.strip()]
    if not entries:
        return "no staged files"

    names: list[str] = []
    for line in entries:
        columns = line.split("\t")
        status = columns[0].strip()
        if not status or status[0] == "D":
            continue  # deleted; nothing on disk to format or re-stage
        # A rename/copy reports "R100\told\tnew" (or "C100\t..."); the path that still
        # exists on disk is the last column.
        path = columns[-1].strip()
        if path:
            names.append(path)

    if not names:
        return "no staged files to format (deletions only)"

    dashed = [n for n in names if n.rsplit("/", 1)[-1].startswith("-")]
    if dashed:
        logger.warning(
            "skipping %d staged path(s) starting with '-' in '%s': %s",
            len(dashed),
            container_name,
            ", ".join(dashed),
        )

    # Only pass paths we would accept anywhere else in this module, and never one
    # whose basename starts with "-" (see docstring above).
    safe = [
        n for n in names if _validate_git_file_path(n) and not n.rsplit("/", 1)[-1].startswith("-")
    ]
    if not safe:
        return "no staged files to format (all skipped)"
    py = [n for n in safe if n.endswith(".py")]
    web = [n for n in safe if n.endswith((".js", ".jsx", ".ts", ".tsx", ".json", ".css", ".md"))]

    # Detection reads the TARGET repo's own files; there is no new config format here.
    # A single round trip also decides WHICH Python formatter is actually configured,
    # rather than treating "pyproject.toml exists" as "black is configured" - that read
    # made the black branch unreachable, since pyproject.toml is already covered by the
    # ruff-config check that runs first.
    probe = await container_manager.execute_command(
        container_name,
        "cd /workspace && ls -1 pyproject.toml .ruff.toml ruff.toml package.json 2>/dev/null; "
        "if [ -f pyproject.toml ]; then "
        "grep -q '^\\[tool\\.ruff' pyproject.toml && echo 'pyproject.toml:tool.ruff'; "
        "grep -q '^\\[tool\\.black' pyproject.toml && echo 'pyproject.toml:tool.black'; "
        f"python3 -I -c {shlex.quote(_BLACK_PROBE_PY)} 2>/dev/null; "
        "fi",
        tenant_id=tenant_id,
    )
    present = set(probe.get("stdout", "").split())
    ruff_configured = bool({".ruff.toml", "ruff.toml", "pyproject.toml:tool.ruff"} & present)
    black_configured = "pyproject.toml:tool.black" in present
    # A black-configured repo is formatted with `ruff format` (black-compatible; black is
    # no longer baked into the runtime image). ruff does not read `[tool.black]`, so the
    # settings that change its output wholesale are carried over as `--config`
    # overrides, built here from validated values only - never from probe text.
    black_overrides = _black_ruff_overrides(present)

    cmds: list[tuple[str, str]] = []
    if py:
        files = " ".join(_sanitize_shell_arg(f) for f in py)
        if ruff_configured:
            cmds.append(("ruff", f"cd /workspace && ruff format -- {files}"))
        elif black_configured:
            overrides = "".join(f" --config {_sanitize_shell_arg(o)}" for o in black_overrides)
            cmds.append(("ruff", f"cd /workspace && ruff format{overrides} -- {files}"))
        # else: neither is actually configured - skip rather than guess.
    if web and "package.json" in present:
        files = " ".join(_sanitize_shell_arg(f) for f in web)
        cmds.append(("prettier", f"cd /workspace && npx --no-install prettier --write -- {files}"))

    if not cmds:
        return "no formatter detected for the staged file types"

    ran: list[str] = []
    for tool, cmd in cmds:
        result = await container_manager.execute_command(container_name, cmd, tenant_id=tenant_id)
        if result.get("exit_code", 1) != 0:
            # Fail open: log the reason, keep the commit.
            logger.warning(
                "formatter %s did not run cleanly in '%s' (exit %s): %s",
                tool,
                container_name,
                result.get("exit_code"),
                (result.get("stderr") or "").strip()[:300],
            )
            continue
        ran.append(tool)

    if not ran:
        return "formatter(s) detected but none completed cleanly"

    readd = await container_manager.execute_command(
        container_name, workspace_git_mutate(["add", "--", *safe]), tenant_id=tenant_id
    )
    if readd.get("exit_code", 1) != 0:
        # The files are formatted on disk but not re-staged; the commit would carry the
        # UNFORMATTED content. Say so rather than reporting a format that did not land.
        logger.warning(
            "re-staging after formatting failed in '%s': %s",
            container_name,
            (readd.get("stderr") or "").strip()[:300],
        )
        return "formatted but re-staging failed; commit carries unformatted content"

    return "formatted with " + ", ".join(ran)


# ``git add`` and ``git commit`` go through ``workspace_git_mutate`` (fsmonitor off,
# non-canonical filters blanked, no lazy fetch: ``commit`` prints its summary through a
# rename-detecting diff that faulted a missing blob in through a planted promisor
# remote). Workspace hooks are off too, as on the push route: a ``pre-commit`` /
# ``commit-msg`` / ``post-commit`` hook is a command the workspace names. The
# route runs the repo's own FORMATTER itself (``_format_staged_files``) instead.
_HOOKS_OFF = "core.hooksPath=/dev/null"

# Exit codes of ``_approved_commit_command`` (outside git's 0/1/128/129 and the shared
# 96/97 refusal codes).
_APPROVED_BASE_MOVED_EXIT = 81
_APPROVED_TREE_UNKNOWN_EXIT = 82
_APPROVED_TREE_EMPTY_EXIT = 83
_COMMIT_ID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
# ``workspace_object_check`` leaves the verified base commit's tree id in this variable.
_VERIFIED_BASE_TREE = '"$_vbt"'


def _approved_commit_command(tree: str, base: str, message: str, author: list[str]) -> str:
    """ONE exec that commits EXACTLY ``tree`` with ``base`` as its only parent.

    The reviewed content is the snapshot tree the diff route froze (``tree``), so the
    commit never re-reads the working tree (``add -A`` would pick up whatever a
    background process wrote since the review). ``commit-tree`` builds the commit from
    the object ids alone; ``update-ref HEAD <new> <base>`` is a compare-and-swap, so a
    HEAD that moved since the review (another commit, a branch switch) refuses instead
    of stacking on top of it. The real index is then reset to ``tree`` (``read-tree``,
    worktree untouched), so a later edit shows as an ordinary unstaged change.

    Prints ``<full id> <short id> <changed path count>``. Exits
    ``_APPROVED_BASE_MOVED_EXIT`` (HEAD is not ``base``), ``_APPROVED_TREE_UNKNOWN_EXIT``
    (``tree`` is not a tree here), ``_APPROVED_TREE_EMPTY_EXIT`` (``tree`` is ``base``'s
    tree: nothing to commit). Hardened like every workspace git, with its OWN poison
    check: replace refs / grafts / alternates planted after the diff exec are refused
    here, before anything is written.
    """
    if not (_COMMIT_ID_RE.match(tree) and _COMMIT_ID_RE.match(base)):
        raise ValueError("approved tree and base must be full object ids")
    ident = " ".join(shlex.quote(a) for a in author)
    return (
        f"cd {_WORKSPACE_ROOT} && {workspace_git_function('_g')}"
        "h=$(_g rev-parse --verify --quiet 'HEAD^{commit}'); "
        f'[ "$h" = {base} ] || {{ echo "commit_base_moved: HEAD is ${{h:-unborn}}" >&2; '
        f"exit {_APPROVED_BASE_MOVED_EXIT}; }}; "
        f'[ "$(_g cat-file -t {tree} 2>/dev/null)" = tree ] || '
        f"exit {_APPROVED_TREE_UNKNOWN_EXIT}; "
        # Every object the commit is built from hashes to its id (the base commit, whose
        # tree lands in $_vbt, the approved tree and everything that differs between
        # them): an object rewritten in place since the review is refused here.
        f"{workspace_object_check(base, [tree], [(_VERIFIED_BASE_TREE, tree)])}"
        f'[ "$_vbt" != {tree} ] || exit {_APPROVED_TREE_EMPTY_EXIT}; '
        f"c=$(_g {ident} commit-tree {tree} -p {base} -m {shlex.quote(message)}) || exit 1; "
        f'_g update-ref -m "commit (approved tree)" HEAD "$c" {base} || '
        f"exit {_APPROVED_BASE_MOVED_EXIT}; "
        f"_g read-tree {tree} || exit 1; "
        '_s=$(_g rev-parse --short "$c") || exit 1; '
        '_n=$(_g diff-tree --no-commit-id --name-only -r "$c" | wc -l) || exit 1; '
        'printf "%s %s %s\\n" "$c" "$_s" "$_n"'
    )


async def _commit_approved_tree(
    manager, container_name: str, tenant_id: str | None, request: GitCommitRequest, branch: str
) -> GitCommitResponse:
    """The ``approved_tree`` path of ``commit_changes`` (see ``_approved_commit_command``)."""
    if request.approved_tree is None or request.approved_base is None:
        raise bad_request("approved_tree and approved_base are given together")
    author: list[str] = []
    if request.author_name and request.author_email:
        author = [
            "-c",
            f"user.name={request.author_name}",
            "-c",
            f"user.email={request.author_email}",
        ]
    result = await manager.execute_command(
        container_name,
        _approved_commit_command(
            request.approved_tree, request.approved_base, request.message, author
        ),
        tenant_id=tenant_id,
    )
    rc = result.get("exit_code", 1)
    if rc == _APPROVED_BASE_MOVED_EXIT:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "commit_base_moved",
                "message": (
                    "commit_base_moved: HEAD is no longer the approved base "
                    f"{request.approved_base}, so the approved tree was not committed. "
                    "Re-run the diff and the review."
                ),
            },
        )
    if rc == _APPROVED_TREE_UNKNOWN_EXIT:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "approved_tree_unknown",
                "message": (
                    f"approved_tree_unknown: {request.approved_tree} is not a tree in this "
                    "workspace; pass the `tree` of a git/diff response from this workspace."
                ),
            },
        )
    if rc == _APPROVED_TREE_EMPTY_EXIT:
        raise bad_request("No changes to commit - the approved tree is the base's tree")
    fields = (result.get("stdout") or "").split()
    if rc != 0 or len(fields) != 3 or not _COMMIT_ID_RE.match(fields[0]):
        raise internal_error(
            f"Failed to commit the approved tree: {(result.get('stderr') or '').strip()[:300]}"
        )
    logger.info(
        "Committed approved tree %s on %s in container '%s' (commit: %s)",
        request.approved_tree,
        request.approved_base,
        container_name,
        fields[0],
    )
    return GitCommitResponse(
        message="Approved tree committed",
        container_name=container_name,
        commit_sha=fields[1],
        files_committed=int(fields[2]),
        branch=branch,
        request_id=str(uuid7()),
    )


@router.post(
    "/{container_name}/git/commit",
    response_model=GitCommitResponse,
    operation_id="commit_workspace_changes",
    summary="Commit Changes",
    responses={
        200: {"description": "Changes committed successfully"},
        400: {"model": Error400Response, "description": "Invalid request or no changes"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Container not found"},
        406: {"model": Error406Response, "description": "Not acceptable"},
        422: {"model": Error422Response, "description": "Validation error"},
        429: {"model": Error429Response, "description": "Too many requests"},
        500: {"model": Error500Response, "description": "Error committing changes"},
    },
)
async def commit_changes(
    container_name: str = Path(..., description="Container name"),
    request: GitCommitRequest = Body(...),
    user: User = Depends(get_current_user_with_tenant),
) -> GitCommitResponse:
    """Commit changes to git in the container."""
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
                f"Container '{container_name}' is not running (current status: {container.status.value})"
            )

        branch_result = await container_manager.execute_command(
            container_name,
            workspace_git_read(["rev-parse", "--abbrev-ref", "HEAD"]),
            tenant_id=tenant_id,
        )
        branch = branch_result.get("stdout", "main").strip()

        if request.approved_tree is not None:
            return await _commit_approved_tree(
                container_manager, container_name, tenant_id, request, branch
            )

        status_check = await container_manager.execute_command(
            container_name, workspace_git_read(["status", "--porcelain"]), tenant_id=tenant_id
        )
        if not status_check.get("stdout", "").strip():
            raise bad_request("No changes to commit - working tree is clean")

        if request.files == ["*"] or "*" in request.files:
            add_result = await container_manager.execute_command(
                container_name, workspace_git_mutate(["add", "-A"]), tenant_id=tenant_id
            )
        else:
            for file_path in request.files:
                if not _validate_git_file_path(file_path):
                    raise bad_request("Invalid file path: contains forbidden characters")
                if file_path.rsplit("/", 1)[-1].startswith("-"):
                    # shlex.quote leaves a leading-dash token UNQUOTED (it has no
                    # character quote needs to escape), so e.g. "-f" would be parsed
                    # by `git add` as the --force flag rather than as a path - the
                    # caller asked for a dashed path, so refuse it, don't silently
                    # drop or reinterpret it.
                    raise bad_request(f"Invalid file path: must not start with '-': {file_path}")
            add_result = await container_manager.execute_command(
                container_name,
                workspace_git_mutate(["add", "--", *request.files]),
                tenant_id=tenant_id,
            )

        if add_result.get("exit_code", 1) != 0:
            error_msg = add_result.get("stderr", "Unknown error")
            raise internal_error(f"Failed to stage files: {error_msg}")

        try:
            format_outcome = await _format_staged_files(
                container_manager, container_name, tenant_id
            )
        except asyncio.CancelledError:
            # Cancellation must always propagate - the client is gone (or the server is
            # shutting down) and suppressing it here would break asyncio's cancellation
            # contract. This is a deliberate decision, not an oversight: "fail open" for
            # this helper means formatting failures never fail the commit, not that the
            # request itself survives cancellation.
            raise
        except Exception as e:  # noqa: BLE001 - deliberately broad: see _format_staged_files
            # A formatting convenience must never be able to fail a commit. The narrow
            # except clauses on execute_command/this endpoint do not cover everything a
            # container backend can raise (e.g. a Kubernetes ApiException), and letting
            # one of those propagate here would abort the commit - the opposite of the
            # fail-open contract this helper documents.
            format_outcome = f"formatting step raised {type(e).__name__}: {e}"
            logger.warning("pre-commit formatting in '%s' raised: %s", container_name, e)
        else:
            logger.info("pre-commit formatting in '%s': %s", container_name, format_outcome)

        author: list[str] = []
        if request.author_name and request.author_email:
            author = [
                "-c",
                f"user.name={request.author_name}",
                "-c",
                f"user.email={request.author_email}",
            ]
        commit_cmd = workspace_git_mutate(
            [*author, "-c", _HOOKS_OFF, "commit", "-m", request.message]
        )

        commit_result = await container_manager.execute_command(
            container_name, commit_cmd, tenant_id=tenant_id
        )

        if commit_result.get("exit_code", 1) != 0:
            error_msg = commit_result.get("stderr", "Unknown error")
            raise internal_error(f"Failed to create commit: {error_msg}")

        sha_result = await container_manager.execute_command(
            container_name,
            workspace_git_read(["rev-parse", "--short", "HEAD"]),
            tenant_id=tenant_id,
        )
        commit_sha = sha_result.get("stdout", "unknown").strip()

        files_result = await container_manager.execute_command(
            container_name,
            workspace_git_read(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"])
            + " | wc -l",
            tenant_id=tenant_id,
        )
        files_committed = int(files_result.get("stdout", "0").strip() or 0)

        logger.info(
            "Successfully committed %d files to container '%s' (commit: %s)",
            files_committed,
            container_name,
            commit_sha,
        )

        return GitCommitResponse(
            message="Changes committed successfully",
            container_name=container_name,
            commit_sha=commit_sha,
            files_committed=files_committed,
            branch=branch,
            request_id=str(uuid7()),
        )

    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as e:
        logger.error("Failed to commit changes in container '%s': %s", container_name, e)
        raise internal_error(f"Failed to commit changes: {str(e)}") from e


def _stderr_tail(result: dict, limit: int = 500) -> str:
    """Last *limit* characters of a command result's stderr (or a placeholder)."""
    err = (result.get("stderr") or "").strip()
    return err[-limit:] if err else f"exit code {result.get('exit_code', 'unknown')}"


def _validate_restore_path(file_path: str) -> str:
    """Return *file_path* if it names a single repo-relative FILE path, else raise 400.

    Same fences as ``commit_changes`` (shell metacharacters, leading dash), plus the
    ones that matter because ``git clean``/``git rm`` delete: every ``/``-separated
    segment must be non-empty and neither ``.`` nor ``..``. That rejects absolute
    paths, ``..`` escapes, and every spelling of a directory (``.``, ``src/.``,
    ``./x``, ``src/``, ``a//b``) - ``.`` would otherwise make ``git clean`` wipe the
    whole workspace. Whether a well-formed path is actually a directory is decided
    later, against the repository, at classification time.
    """
    if not _validate_git_file_path(file_path):
        raise bad_request(f"Invalid file path: contains forbidden characters: {file_path!r}")
    if file_path.startswith("-") or file_path.rsplit("/", 1)[-1].startswith("-"):
        # shlex.quote leaves a leading-dash token UNQUOTED; refuse, never reinterpret.
        raise bad_request(f"Invalid file path: must not start with '-': {file_path}")
    if any(seg in ("", ".", "..") for seg in file_path.split("/")):
        raise bad_request(
            "Invalid file path: must be a repository-relative file path with no empty, "
            f"'.' or '..' segments: {file_path}"
        )
    if any(seg.lower() == ".git" for seg in file_path.split("/")):
        # The repository's own metadata (or a nested repo's) is never a workspace file.
        raise bad_request(f"Invalid file path: must not contain a '.git' segment: {file_path}")
    if any(seg.startswith(":") for seg in file_path.split("/")):
        # A leading ':' is pathspec MAGIC (`:bar` reads as `bar`, `:!x` excludes) to
        # every git command that is not under --literal-pathspecs - and check-ignore
        # cannot be (it rejects the literal magic with 128). Refuse, never reinterpret.
        raise bad_request(f"Invalid file path: no segment may start with ':': {file_path}")
    return file_path


# Prefix of EVERY command the restore route runs, right after ``cd /workspace && ``.
# Defence in depth under the per-command ``--literal-pathspecs`` flags: a git command
# added (or a flag dropped) later still reads ``*.py`` / ``[ab].py`` as the literal
# file, never as a glob over unlisted paths. check-ignore alone opts out (it refuses
# the literal magic outright) and is fenced by the ':' refusal above instead.
_RESTORE_GIT_ENV = "export GIT_LITERAL_PATHSPECS=1; "


def _restore_git_read(args: list[str]) -> str:
    """``workspace_git_read`` with ``_RESTORE_GIT_ENV`` exported, like every restore command."""
    prefix = f"cd {_WORKSPACE_ROOT} && "
    return prefix + _RESTORE_GIT_ENV + workspace_git_read(args).removeprefix(prefix)


def _restore_git_mutate(args: list[str]) -> str:
    """``workspace_git_mutate`` with ``_RESTORE_GIT_ENV`` exported.

    The mutations get the same ``core.fsmonitor=false`` and filter blanking as the
    reads. fsmonitor is only a cache: without it git stats every path, same result.
    Blanking the filters is REQUIRED, not merely safe: ``restore --worktree`` runs the
    smudge filter and ``rm --cached`` the clean filter, both commands the workspace
    config names; and the verifier (``_vt``) hashes the disk with the filters blanked,
    so a restore that smudged through a non-canonical filter would fail verification.
    A non-LFS filtered path is therefore restored as the raw blob bytes, and a
    ``filter.<d>.required`` driver fails the step loudly (reported, never "restored").
    Canonical git-lfs filters are kept on both sides, so LFS paths still round-trip.
    """
    prefix = f"cd {_WORKSPACE_ROOT} && "
    return prefix + _RESTORE_GIT_ENV + workspace_git_mutate(args).removeprefix(prefix)


# The classify and verify scripts define ``_g`` (hardened ``git``) once, up front. Built
# per call (never frozen at import) so the shared hardening is read when the command is.
def _restore_script_head() -> str:
    return "cd /workspace && " + _RESTORE_GIT_ENV + workspace_git_function("_g")


# Verdicts from the per-path classifier. Only T and U are actionable; every other
# verdict refuses the WHOLE request before anything is mutated.
_RESTORE_REFUSALS = {
    "D": "is a directory (name individual files)",
    "L": "has a parent that is a symlink or not a directory",
    "S": "is inside a submodule or nested git repository",
    "C": "has a parent that is a file (not a directory) in the index, at source or in HEAD",
    "I": "is ignored by .gitignore/exclude rules (git clean would skip it)",
    "M": (
        "is staged with content that differs from the file on disk "
        "(git rm --cached would refuse; re-stage or unstage it first)"
    ),
    "X": "is not a single file (submodule, special file, or index entries other than itself)",
    "W": (
        "has skip-worktree or assume-unchanged set in the index "
        "(git would not look at the file on disk; clear the flag first)"
    ),
}

# Shell functions shared by the classify and verify commands. They are defined ONCE
# per command and then called once per path, so a command grows LINEARLY with the
# requested paths (one quoted mention each) - never with path depth. Every command
# is one ``sh -c`` argument and must stay far below Linux MAX_ARG_STRLEN (131072);
# the exec backends offer no stdin to move paths out of argv (see RESTORE_MAX_FILES).
#
# ``_exact PATH FIELD`` reads git -z output (NULs turned into newlines; the path
# validator forbids newlines) and prints FIELD of the metadata before the TAB of the
# ONE entry whose whole path equals PATH, so a prefix never matches its children.
# ``_meta PATH`` prints the full metadata of EVERY entry equal to PATH (all stages).
# ``ty REV PATH`` prints the object TYPE of the exact tree entry PATH at REV (blob /
# tree / commit) from ``ls-tree``, which reads the TREE and so also reports a gitlink
# whose commit is not in this repository. ``ix PATH`` prints the mode of the exact
# index entry PATH. ``_flag PATH`` prints the ``ls-files -v`` tag letter of PATH.
#
# ``_cls SRC PATH`` prints exactly one verdict letter. Every PARENT prefix is
# checked first, in order, so nothing below a symlink, a non-directory, a submodule
# (gitlink at SRC, in HEAD or in the index) or a nested repository (a ``.git``
# entry) is ever acted on: ``git restore`` would replace an unlisted symlinked
# directory with a real one, and ``git clean`` silently skips such paths. A parent
# that is a FILE in the index, at SRC or in HEAD is a type conflict: restoring the
# child would silently drop that unlisted index entry. A parent missing on disk is
# allowed - restoring a tracked file recreates it. Then, for the path itself:
# W = skip-worktree ('S'/'s') or assume-unchanged (any lowercase tag): git trusts the
# index over the disk for such an entry, so `restore` can leave the candidate in
# place and `diff` reports it clean; D = directory; X = index entries other than the
# path itself (a staged `d/x` under a requested `d`, which restore would silently
# drop), or a non-blob at SRC, or a special file; T = blob at SRC; M = staged with
# content that differs from disk (`git rm --cached` refuses it AFTER siblings
# mutated); I = ignored (clean would skip it; --no-index so a force-staged ignored
# file counts too); U = cleanable lone file; N = nowhere. check-ignore takes plain
# pathnames (it rejects --literal-pathspecs, GIT_LITERAL_PATHSPECS and ':(magic)'
# with 128), so it runs in a subshell with the env var unset - a leading ':' is
# refused at validation - and any exit other than 0/1 fails closed to X.
#
# ``_vt SRC PATH`` / ``_vu PATH`` print ok|no: the post-mutation check. They never ask
# ``git diff`` (which believes skip-worktree / assume-unchanged and stat caches):
# _vt requires the index to hold exactly one stage-0 entry equal to SRC's mode and
# blob, and the bytes on disk (``git hash-object``, or the link target for a symlink)
# to hash to SRC's blob with the matching file type; _vu requires the path absent on
# disk and in the index.
_RESTORE_SHELL_PRELUDE = (
    '_exact() { P="$1" awk -v F="$2" \'{ i = index($0, "\\t"); '
    'if (substr($0, i + 1) == ENVIRON["P"]) { split(substr($0, 1, i - 1), a, " "); '
    "print a[F]; exit } }'; }; "
    '_meta() { P="$1" awk \'{ i = index($0, "\\t"); '
    'if (substr($0, i + 1) == ENVIRON["P"]) print substr($0, 1, i - 1) }\'; }; '
    'ty() { _g --literal-pathspecs ls-tree -z "$1" -- "$2" 2>/dev/null '
    "| tr '\\0' '\\n' | _exact \"$2\" 2; }; "
    'ix() { _g --literal-pathspecs ls-files -s -z -- "$1" 2>/dev/null '
    "| tr '\\0' '\\n' | _exact \"$1\" 1; }; "
    '_flag() { _g --literal-pathspecs ls-files -v -z -- "$1" 2>/dev/null '
    "| tr '\\0' '\\n' | P=\"$1\" awk '{ if (substr($0, 3) == ENVIRON[\"P\"]) "
    "{ print substr($0, 1, 1); exit } }'; }; "
    "_cls() { "
    'src="$1"; f="$2"; rest="$f"; p=""; '
    "while :; do "
    'case "$rest" in */*) ;; *) break ;; esac; '
    'p="${p:+$p/}${rest%%/*}"; rest="${rest#*/}"; '
    'if [ -L "$p" ] || { [ -e "$p" ] && [ ! -d "$p" ]; }; then echo L; return; fi; '
    'if [ -e "$p/.git" ] || [ -L "$p/.git" ]; then echo S; return; fi; '
    'ps=$(ty "$src" "$p"); ph=$(ty HEAD "$p"); pi=$(ix "$p"); '
    'if [ "$ps" = commit ] || [ "$ph" = commit ] || [ "$pi" = 160000 ]; then echo S; return; fi; '
    'if [ -n "$pi" ] || [ "$ps" = blob ] || [ "$ph" = blob ]; then echo C; return; fi; '
    "done; "
    't=$(ty "$src" "$f"); '
    "i=$(_g --literal-pathspecs ls-files -z -- \"$f\" | tr '\\0' '\\n'); "
    # Letters spelled out: under a non-C locale a range like [a-z] also matches
    # upper-case tags (the plain "H" of an ordinary entry) in bash/zsh-as-sh.
    'case "$(_flag "$f")" in [abcdefghijklmnopqrstuvwxyz]|S) echo W; return ;; esac; '
    'if [ -d "$f" ] && [ ! -L "$f" ]; then echo D; '
    'elif [ "$t" = tree ]; then echo D; '
    'elif [ -n "$i" ] && [ "$i" != "$f" ]; then echo X; '
    'elif [ "$t" = blob ]; then echo T; '
    'elif [ -n "$t" ]; then echo X; '
    'elif [ -f "$f" ] || [ -L "$f" ]; then '
    'if [ -n "$i" ] && ! _g --literal-pathspecs diff --quiet --no-ext-diff '
    '--no-textconv -- "$f"; then echo M; '
    'else (unset GIT_LITERAL_PATHSPECS; _g check-ignore -q --no-index -- "$f"); rc=$?; '
    "if [ $rc -eq 0 ]; then echo I; elif [ $rc -eq 1 ]; then echo U; else echo X; fi; fi; "
    'elif [ -e "$f" ]; then echo X; '
    "else echo N; fi; }; "
    "_vt() { "
    's=$(_g --literal-pathspecs ls-tree -z "$1" -- "$2" 2>/dev/null '
    "| tr '\\0' '\\n' | _meta \"$2\"); "
    'x=$(_g --literal-pathspecs ls-files -s -z -- "$2" 2>/dev/null '
    "| tr '\\0' '\\n' | _meta \"$2\"); "
    'm="${s%% *}"; o="${s##* }"; w=""; '
    'case "$s" in *" blob "*) ;; *) echo no; return ;; esac; '
    'if [ "$x" != "$m $o 0" ]; then echo no; return; fi; '
    'if [ "$m" = 120000 ]; then [ -L "$2" ] && '
    'w=$(printf \'%s\' "$(readlink "$2")" | _g hash-object --stdin); '
    'elif [ -f "$2" ] && [ ! -L "$2" ]; then w=$(_g hash-object -- "$2"); fi; '
    'if [ -n "$w" ] && [ "$w" = "$o" ]; then echo ok; else echo no; fi; }; '
    "_vu() { "
    'if [ -e "$1" ] || [ -L "$1" ] || '
    '[ -n "$(_g --literal-pathspecs ls-files -z -- "$1")" ]; '
    "then echo no; else echo ok; fi; }; "
)


@router.post(
    "/{container_name}/git/restore",
    response_model=GitRestoreResponse,
    operation_id="restore_workspace_files",
    summary="Restore Files",
    responses={
        200: {"description": "Files restored"},
        400: {"model": Error400Response, "description": "Invalid path or container not running"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Container, source ref or path unknown"},
        409: {"model": Error409Response, "description": "Untracked paths without clean_untracked"},
        422: {"model": Error422Response, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Error restoring files"},
    },
)
async def restore_files(
    container_name: str = Path(..., description="Container name"),
    request: GitRestoreRequest = Body(...),
    user: User = Depends(get_current_user_with_tenant),
) -> GitRestoreResponse:
    """Revert named paths to their content at ``source`` (default HEAD).

    WHY THIS EXISTS. The conveyor's QA writes candidate rewrites into a workspace and
    must revert the losers. Doing that with a raw ``exec`` of ``git checkout -- <f>`` is
    untyped and silently succeeds on a path that does not exist, which reads exactly
    like "reverted". Here every requested path is classified FIRST:

    FILES ONLY, never directories. Every requested path is classified before
    anything is mutated; any refusal leaves the workspace untouched:

    - a blob at ``source`` (and not a directory on disk) -> restored (index AND worktree)
    - absent at ``source``, a regular file or symlink on disk, and at most its own
      single entry in the index               -> deleted only if ``clean_untracked``, else 409
    - a directory (tree at ``source`` or directory on disk), a ``.git`` segment, a
      parent that is a symlink or not a directory, a path inside a submodule or
      nested repository, an ignored file to be cleaned, or anything else
                                               -> 400 naming the paths
    - neither at ``source`` nor on disk        -> 404 naming the paths
    - an index entry with skip-worktree or assume-unchanged set
                                               -> 400 (git would not read the disk)

    After mutating, success is MEASURED, never inferred from exit codes or ``git
    diff``: a restored path's index entry and on-disk bytes must hash to the blob at
    ``source``; a cleaned path must be gone from disk and index. Any mismatch is a
    500 ``partial_restore`` naming the confirmed state of every path. At most
    ``RESTORE_MAX_FILES`` paths per request (422 above) - see that constant.

    All pathspecs run under ``--literal-pathspecs`` AND ``GIT_LITERAL_PATHSPECS=1`` so
    a path like ``*.py`` or ``[ab].py`` names that literal file and can never widen to
    unlisted paths; a segment starting with ``:`` (pathspec magic) is refused, and
    ``git rm --cached``/``git clean`` run without ``-r``/``-d`` so even a path that
    slipped past classification cannot recurse into a directory.
    """
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
                f"Container '{container_name}' is not running (current status: {container.status.value})"
            )

        files = list(dict.fromkeys(_validate_restore_path(f) for f in request.files))
        source = request.source
        q_source = _sanitize_shell_arg(source)

        ref_result = await container_manager.execute_command(
            container_name,
            _restore_git_read(["rev-parse", "--verify", "--quiet", f"{source}^{{commit}}"]),
            tenant_id=tenant_id,
        )
        if ref_result.get("exit_code", 1) != 0:
            raise not_found("Git ref", source)

        # Single pass: one round trip, exactly one verdict line per path, in order.
        # Nothing is mutated unless EVERY verdict is actionable (T or U).
        q_paths = {f: _sanitize_shell_arg(f) for f in files}
        classify = await container_manager.execute_command(
            container_name,
            _restore_script_head()
            + _RESTORE_SHELL_PRELUDE
            + "; ".join(f"_cls {q_source} {q_paths[f]}" for f in files),
            tenant_id=tenant_id,
        )
        kinds = [ln.strip() for ln in classify.get("stdout", "").splitlines() if ln.strip()]
        if classify.get("exit_code", 1) != 0 or len(kinds) != len(files):
            raise internal_error(
                f"Failed to classify paths: {classify.get('stderr', 'unexpected probe output')}"
            )
        if any(k not in ("T", "U", "N", *_RESTORE_REFUSALS) for k in kinds):
            raise internal_error(f"Failed to classify paths: unexpected probe output {kinds!r}")
        verdicts = list(zip(files, kinds, strict=True))

        refused = [f"{f} {_RESTORE_REFUSALS[k]}" for f, k in verdicts if k in _RESTORE_REFUSALS]
        if refused:
            raise bad_request("Paths not restorable: " + "; ".join(refused))
        tracked = [f for f, k in verdicts if k == "T"]
        untracked = [f for f, k in verdicts if k == "U"]
        unknown = [f for f, k in verdicts if k == "N"]

        if unknown:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Paths neither tracked at '{source}' nor present in the workspace: "
                    + ", ".join(unknown)
                ),
            )
        if untracked and not request.clean_untracked:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Paths not tracked at '{source}' (set clean_untracked=true to delete): "
                    + ", ".join(untracked)
                ),
            )

        # MUTATION ORDER. The clean side runs first, the restore last:
        # 1. `git rm --cached` validates EVERY path before it writes the index once, so
        #    it is all-or-nothing - if it refuses, nothing at all has been mutated (the
        #    classifier's M verdict already rules out the one refusal it can hit).
        # 2. `git clean` only deletes content the caller asked to have deleted.
        # 3. `git restore` overwrites, and can fail PARTWAY (permission, disk): running it
        #    last means a failure in 1-2 leaves every tracked candidate untouched.
        # After any failure the remaining steps are skipped and the real state of every
        # listed path is re-measured and returned, never a bare message.
        steps: list[tuple[str, list[str], str]] = []
        if untracked:
            # A path absent at source may still be in the INDEX (staged new file);
            # `git clean` never touches indexed paths, so unstage first.
            steps.append(
                (
                    "unstage",
                    untracked,
                    _restore_git_mutate(
                        ["--literal-pathspecs", "rm", "-q", "--cached", "--ignore-unmatch"]
                        + ["--", *untracked]
                    ),
                )
            )
            steps.append(
                (
                    "clean",
                    untracked,
                    _restore_git_mutate(
                        ["--literal-pathspecs", "clean", "-f", "-q", "--", *untracked]
                    ),
                )
            )
        if tracked:
            steps.append(
                (
                    "restore",
                    tracked,
                    _restore_git_mutate(
                        ["--literal-pathspecs", "restore", f"--source={source}"]
                        + ["--staged", "--worktree", "--", *tracked]
                    ),
                )
            )

        failed_step: tuple[str, str] | None = None  # (stage, stderr tail)
        attempted: dict[str, str] = {}  # path -> the last stage that ran for it
        for stage, paths, cmd in steps:
            result = await container_manager.execute_command(
                container_name, cmd, tenant_id=tenant_id
            )
            for f in paths:
                attempted[f] = stage
            if result.get("exit_code", 1) != 0:
                failed_step = (stage, _stderr_tail(result))
                break

        # Re-measure. A zero exit is not proof (`git clean` silently skips paths), and
        # neither is `git diff`, which trusts skip-worktree/assume-unchanged flags and
        # the stat cache: _vt/_vu compare the index entry and the bytes on disk with
        # the blob at source. restored/cleaned are derived ONLY from this check.
        verify_parts = [f"_vt {q_source} {q_paths[f]}" for f in tracked] + [
            f"_vu {q_paths[f]}" for f in untracked
        ]
        verify = await container_manager.execute_command(
            container_name,
            _restore_script_head() + _RESTORE_SHELL_PRELUDE + "; ".join(verify_parts),
            tenant_id=tenant_id,
        )
        marks = [ln.strip() for ln in verify.get("stdout", "").splitlines() if ln.strip()]
        if verify.get("exit_code", 1) != 0 or len(marks) != len(tracked) + len(untracked):
            marks = ["unknown"] * (len(tracked) + len(untracked))
        ok = dict(zip(tracked + untracked, marks, strict=True))

        status_result = await container_manager.execute_command(
            container_name,
            _restore_git_read(
                ["-c", "core.quotePath=false", "--literal-pathspecs", "status", "--porcelain"]
                + ["--", *files]
            ),
            tenant_id=tenant_id,
        )
        status_after: list[str] | None = None
        if status_result.get("exit_code", 1) == 0:
            status_after = [ln for ln in status_result.get("stdout", "").splitlines() if ln.strip()]

        restored = [f for f in tracked if f in attempted and ok[f] == "ok"]
        cleaned = [f for f in untracked if attempted.get(f) == "clean" and ok[f] == "ok"]
        failed: list[dict[str, str]] = []
        for f in tracked + untracked:
            if f in restored or f in cleaned:
                continue
            stage = attempted.get(f)
            if failed_step is not None and stage == failed_step[0]:
                failed.append({"path": f, "stage": failed_step[0], "stderr_tail": failed_step[1]})
            elif failed_step is not None and (
                stage is None or (f in untracked and stage != "clean")
            ):
                failed.append(
                    {
                        "path": f,
                        "stage": "restore" if f in tracked else "clean",
                        "stderr_tail": "not attempted: an earlier step failed",
                    }
                )
            else:
                failed.append(
                    {
                        "path": f,
                        "stage": "verify",
                        "stderr_tail": (
                            f"command exited 0 but the path is not in the requested state "
                            f"(verify: {ok[f]})"
                        ),
                    }
                )

        if failed or status_after is None:
            details: dict[str, object] = {
                "error": "partial_restore",
                "restored": restored,
                "cleaned": cleaned,
                "failed": failed,
                "status_after": status_after,
            }
            if status_after is None:
                details["status_error"] = _stderr_tail(status_result)
            logger.error(
                "Restore in container '%s' from '%s' partially applied: %s",
                container_name,
                source,
                details,
            )
            raise internal_error(
                "Restore partially applied: see details for the confirmed state of every path",
                details=details,
            )

        logger.info(
            "Restored %d and cleaned %d path(s) in container '%s' from '%s'",
            len(restored),
            len(cleaned),
            container_name,
            source,
        )

        return GitRestoreResponse(
            message="Files restored",
            container_name=container_name,
            source=source,
            restored=restored,
            cleaned=cleaned,
            status_after=status_after,
            request_id=str(uuid7()),
        )

    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as e:
        logger.error("Failed to restore files in container '%s': %s", container_name, e)
        raise internal_error(f"Failed to restore files: {str(e)}") from e


# The branch route's ref probes (rev-parse / symbolic-ref / check-ref-format /
# show-ref) run through ``_g`` from ``workspace_git_function``: the shared env and
# ``-c`` list (``safe.directory``, ``core.hooksPath=/dev/null``, fsmonitor off, no
# replace refs / grafts / system or global config) and the poison refusal, like every
# other workspace git. LC_ALL=C: the 409 classifier matches git's English refusal text
# (_BRANCH_CONFLICT_MARKERS); a translated message would turn every conflict into a 500.
def _branch_git_prelude() -> str:
    return f"{_BRANCH_SWITCH_ENV}{workspace_git_function('_g')}"


# ``git switch`` WRITES the worktree: it runs each checked-out path's smudge filter
# (workspace config), reads the index through fsmonitor, and lazily fetches a missing
# blob through a partial clone's promisor remote. So it goes through
# ``workspace_git_mutate`` (fsmonitor off, non-canonical filters blanked, no lazy
# fetch). ``safe.directory`` and ``hooksPath`` ride as ENVIRONMENT config
# (command scope, like ``-c``) so the helper's filter enumeration also runs under
# ``safe.directory``: without it, a workspace owned by another uid would enumerate
# no local filter while the switch itself, given ``-c safe.directory``, ran them.
_BRANCH_SWITCH_ENV = (
    "export LC_ALL=C GIT_CONFIG_COUNT=2 "
    "GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=/workspace "
    "GIT_CONFIG_KEY_1=core.hooksPath GIT_CONFIG_VALUE_1=/dev/null; "
)


def _branch_switch_command(args: list[str]) -> str:
    prefix = f"cd {_WORKSPACE_ROOT} && "
    return (
        prefix + _BRANCH_SWITCH_ENV + workspace_git_mutate(["switch", *args]).removeprefix(prefix)
    )


# Probe exit codes (anything else nonzero is an unexpected failure -> 500).
_BRANCH_UNBORN_EXIT = 3
_BRANCH_BADNAME_EXIT = 4

# stderr fragments of the refusals ``git switch`` makes BEFORE it writes anything:
# local changes / untracked files it would overwrite, an unmerged index, the branch
# checked out in another worktree, or a branch that appeared since the probe.
_BRANCH_CONFLICT_MARKERS = (
    "would be overwritten",
    "resolve your current index",
    "already used by worktree",
    "already checked out",
    "already exists",
)


def _kv_lines(stdout: str) -> dict[str, str]:
    """Parse ``key=value`` lines (first ``=`` splits; empty values kept)."""
    out: dict[str, str] = {}
    for line in (stdout or "").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key.strip()] = value.strip()
    return out


def _modified_paths_command() -> str:
    """NUL-separated tracked paths whose worktree file differs from the index
    (``ls-files --modified``: missing files AND stale or partly written content).

    ``git switch`` exits 0 when it cannot write a path of the target tree: a blob
    missing from the object store (``unable to read sha1 file``) leaves the file
    absent, and an unwritable directory (``unable to unlink old``) or a short write
    leaves the OLD content in place. Either way HEAD and the index move, and a later
    ``add -A`` commits the gap as a change. Comparing this set before and after the
    switch catches both from the tree itself, without depending on git's error
    wording. ``--modified`` may read file content, and so run clean filters, so it
    goes through ``workspace_git_mutate`` (filters blanked, fsmonitor off, no lazy
    fetch) with the same config env as the switch.
    """
    prefix = f"cd {_WORKSPACE_ROOT} && "
    return (
        prefix
        + _BRANCH_SWITCH_ENV
        + workspace_git_mutate(["ls-files", "-z", "--modified"]).removeprefix(prefix)
    )


def _index_entries_command() -> str:
    """``ls-files -s -z`` (``<mode> <sha> <stage>\\t<path>`` per index entry), hardened
    like every workspace read and with the switch's config env."""
    prefix = f"cd {_WORKSPACE_ROOT} && "
    return (
        prefix
        + _BRANCH_SWITCH_ENV
        + workspace_git_read(["ls-files", "-s", "-z"]).removeprefix(prefix)
    )


def _parse_index_entries(stdout: str) -> dict[str, list[tuple[str, str, str]]]:
    """``ls-files -s -z`` output -> ``{path: [(mode, sha, stage), ...]}``."""
    out: dict[str, list[tuple[str, str, str]]] = {}
    for record in (stdout or "").split("\0"):
        meta, tab, path = record.partition("\t")
        fields = meta.split(" ")
        if tab and len(fields) == 3:
            out.setdefault(path, []).append((fields[0], fields[1], fields[2]))
    return out


# Byte-exact recheck, one command per chunk of candidates (``_recheck_chunks``).
# Regular files are hashed in one ``hash-object --no-filters -- <paths>`` call per
# chunk (raw bytes: no eol conversion, no filter). Paths go as argv, never through ``--stdin-paths``, which C-unquotes a line
# starting with ``"`` (a file named ``"q"`` would be hashed as ``q``). The call is
# skipped when no candidate is a regular file on disk. A symlink is hashed as its link
# target (``hash-object`` would follow the link); under ``core.symlinks=false`` git
# checks a symlink out as a plain file holding the target, so such a file is hashed
# raw instead. Output: an ``r<k>`` line per candidate k that is a regular file on
# disk, an ``l<k> <sha>`` line per symlink candidate k, then ``--``, then one sha per
# ``r`` line in the same order. Paths are ``./``-prefixed so a leading ``-`` is never
# an option.
_RECHECK_SPLIT = "--"


# Linux caps ONE argv string at MAX_ARG_STRLEN = 131072 bytes, and the whole recheck
# travels as the single ``sh -c`` argument (a backend may re-quote it into another
# ``sh -c`` string, which turns every ``'`` into ``'"'"'``). A switch between trees
# differing in a few thousand blobs overflowed it (exit 126, a false 500 after HEAD
# had moved), and macOS has no per-argument limit, so nothing local saw it. The
# candidates are therefore split into chunks whose command stays under this budget,
# measured as the re-quoted length; one ``execute_command`` per chunk.
_RECHECK_ARG_BUDGET = 96 * 1024


def _requoted_weight(text: str) -> int:
    """Bytes ``text`` occupies once ``shlex.quote``-d into an outer ``sh -c`` string;
    additive over concatenated pieces (bar the 2 enclosing quotes)."""
    return len(text.encode("utf-8", "surrogateescape")) + 4 * text.count("'")


def _recheck_regular_mark(k: int, path: str) -> str:
    q = shlex.quote("./" + path)
    return f'{{ [ -f {q} ] && [ ! -L {q} ] && echo r{k} && set -- "$@" {q}; }};'


def _recheck_symlink_mark(k: int, path: str) -> str:
    q = shlex.quote("./" + path)
    return (
        f'if [ -L {q} ]; then echo "l{k} $(printf %s "$(readlink {q})" '
        '| _g hash-object --no-filters --stdin)"; '
        f'elif [ "$_s" = false ] && [ -f {q} ]; then '
        f'echo "l{k} $(_g hash-object --no-filters -- {q})"; fi;'
    )


def _recheck_frame(marks: list[str], with_symlinks: bool) -> str:
    symlink_mode = "_s=$(_g config --type=bool core.symlinks) || _s=true; " if with_symlinks else ""
    return (
        f"cd {_WORKSPACE_ROOT} && {_BRANCH_SWITCH_ENV}{workspace_git_function('_g')}"
        + symlink_mode
        + "set --; "
        + " ".join(marks)
        + f" echo {_RECHECK_SPLIT}; "
        + '[ "$#" -eq 0 ] || _g hash-object --no-filters -- "$@"'
    )


def _byte_exact_recheck_command(regular: list[str], symlinks: list[str]) -> str:
    """Shell command hashing the worktree bytes of ``regular`` (index key ``r<k>``) and
    the link targets of ``symlinks`` (``l<k>``); see ``_RECHECK_SPLIT``. ONE chunk:
    callers with many candidates go through ``_recheck_chunks``."""
    marks = [_recheck_regular_mark(k, p) for k, p in enumerate(regular)]
    marks += [_recheck_symlink_mark(k, p) for k, p in enumerate(symlinks)]
    return _recheck_frame(marks, bool(symlinks))


def _recheck_chunks(
    regular: list[str], symlinks: list[str], budget: int | None = None
) -> list[tuple[list[int], list[int]]]:
    """Split candidates into ``(regular indices, symlink indices)`` chunks whose
    ``_byte_exact_recheck_command`` stays within ``budget`` re-quoted bytes. Built
    incrementally: a chunk is closed BEFORE the next mark would push it over. A lone
    mark over the budget still gets its own chunk (a path is at most PATH_MAX = 4096
    bytes, so only pathological quoting could get there, and then that chunk's
    command fails and the recheck fails closed)."""
    budget = _RECHECK_ARG_BUDGET if budget is None else budget
    overhead = _requoted_weight(_recheck_frame([], True)) + 2
    chunks: list[tuple[list[int], list[int]]] = []
    reg: list[int] = []
    sym: list[int] = []
    used = overhead
    items = [(False, i) for i in range(len(regular))] + [(True, i) for i in range(len(symlinks))]
    for is_link, i in items:
        if is_link:
            mark = _recheck_symlink_mark(len(sym), symlinks[i])
        else:
            mark = _recheck_regular_mark(len(reg), regular[i])
        weight = _requoted_weight(mark) + 1  # + the joining space
        if (reg or sym) and used + weight > budget:
            chunks.append((reg, sym))
            reg, sym, used = [], [], overhead
            # indices restart per chunk, so the mark is rebuilt with k = 0
            mark = (
                _recheck_symlink_mark(0, symlinks[i])
                if is_link
                else _recheck_regular_mark(0, regular[i])
            )
            weight = _requoted_weight(mark) + 1
        (sym if is_link else reg).append(i)
        used += weight
    if reg or sym:
        chunks.append((reg, sym))
    return chunks


def _recheck_matches(
    stdout: str, regular: list[tuple[str, str]], symlinks: list[tuple[str, str]]
) -> set[str] | None:
    """Paths of one chunk whose bytes (or link target) hash to their index blob;
    ``None`` when the output is not the shape the command prints."""
    lines = stdout.split("\n")
    if _RECHECK_SPLIT not in lines:
        return None
    split = lines.index(_RECHECK_SPLIT)
    marks, hashes = lines[:split], [h for h in lines[split + 1 :] if h]
    reg_keys = [int(m[1:]) for m in marks if m.startswith("r")]
    if len(reg_keys) != len(hashes):
        return None
    matched: set[str] = set()
    for k, sha in zip(reg_keys, hashes, strict=True):
        if sha == regular[k][1]:
            matched.add(regular[k][0])
    for m in marks:
        if m.startswith("l"):
            key, _, sha = m[1:].partition(" ")
            if sha == symlinks[int(key)][1]:
                matched.add(symlinks[int(key)][0])
    return matched


async def _still_incomplete(
    container_manager, container_name: str, tenant_id, candidates: list[str]
) -> list[str] | None:
    """Of ``candidates`` (paths ``ls-files --modified`` newly flags after the switch),
    those the switch really left wrong: ``None`` when the recheck itself fails.

    ``--modified`` compares through the CLEAN conversion, which does not round-trip
    for e.g. a blob committed with CRLF under ``eol=crlf``: the file holds exactly the
    blob's bytes, yet is flagged. So a candidate is dropped when its worktree bytes
    hash to its stage-0 index blob WITHOUT filters (a symlink: its target). A gitlink
    (mode 160000) is dropped outright: a submodule pointer the switch moved without
    checking the submodule out is not worktree content of this repository. Everything
    else stays: a missing file, stale bytes, an unmerged or absent index entry.

    The recheck runs in chunks under ``_RECHECK_ARG_BUDGET``; any chunk failing, or any
    output coming back ``truncated``, fails the whole recheck (``None``).
    """
    if not candidates:
        return []
    idx_res = await container_manager.execute_command(
        container_name, _index_entries_command(), tenant_id=tenant_id
    )
    if idx_res.get("exit_code", 1) != 0 or idx_res.get("truncated"):
        return None
    index = _parse_index_entries(idx_res.get("stdout", ""))

    remaining: list[str] = []
    regular: list[tuple[str, str]] = []  # (path, index sha)
    symlinks: list[tuple[str, str]] = []
    for path in candidates:
        entries = index.get(path, [])
        if len(entries) == 1 and entries[0][0] == "160000":
            continue
        if len(entries) != 1 or entries[0][2] != "0" or "\n" in path:
            remaining.append(path)
        elif entries[0][0] in ("100644", "100755"):
            regular.append((path, entries[0][1]))
        elif entries[0][0] == "120000":
            symlinks.append((path, entries[0][1]))
        else:
            remaining.append(path)
    if not regular and not symlinks:
        return sorted(remaining)

    matched: set[str] = set()
    for reg_idx, sym_idx in _recheck_chunks([p for p, _ in regular], [p for p, _ in symlinks]):
        reg_chunk = [regular[i] for i in reg_idx]
        sym_chunk = [symlinks[i] for i in sym_idx]
        res = await container_manager.execute_command(
            container_name,
            _byte_exact_recheck_command([p for p, _ in reg_chunk], [p for p, _ in sym_chunk]),
            tenant_id=tenant_id,
        )
        if res.get("exit_code", 1) != 0 or res.get("truncated"):
            return None
        chunk_matched = _recheck_matches(res.get("stdout", ""), reg_chunk, sym_chunk)
        if chunk_matched is None:
            return None
        matched |= chunk_matched
    remaining.extend(p for p, _ in (*regular, *symlinks) if p not in matched)
    return sorted(remaining)


def _branch_state_command() -> str:
    """Print ``head=<sha>`` and ``branch=<short name or empty when detached>``."""
    return (
        f"{_branch_git_prelude()}"
        "h=$(_g rev-parse --verify --quiet 'HEAD^{commit}'); "
        "b=$(_g symbolic-ref -q --short HEAD) || b=; "
        'printf \'head=%s\\nbranch=%s\\n\' "$h" "$b"'
    )


@router.post(
    "/{container_name}/git/branch",
    response_model=GitBranchResponse,
    operation_id="ensure_workspace_branch",
    summary="Ensure Branch",
    responses={
        200: {"description": "HEAD is on the requested branch"},
        400: {"model": Error400Response, "description": "Invalid branch name or not running"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Container not found"},
        409: {
            "model": Error409Response,
            "description": "branch_switch_conflict: git refused the switch; nothing changed",
        },
        422: {"model": Error422Response, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Error switching branch"},
    },
)
async def ensure_branch(
    container_name: str = Path(..., description="Container name"),
    request: GitBranchRequest = Body(...),
    user: User = Depends(get_current_user_with_tenant),
) -> GitBranchResponse:
    """Put HEAD on local branch ``branch``; idempotent, never destructive.

    WHY THIS EXISTS. Callers used a raw ``exec`` of ``git checkout -B <branch>``, which
    RESETS an existing branch to the current HEAD and so silently drops its commits.

    - ``refs/heads/<branch>`` exists -> ``git switch --no-guess <branch>``. Uncommitted
      changes carry over; if git refuses (they would be overwritten, unmerged index,
      branch checked out in another worktree) the answer is 409
      ``branch_switch_conflict`` and nothing has changed - git checks before it writes,
      and the route re-measures HEAD to confirm it.
    - absent -> ``git switch --no-track -c <branch>`` from the current HEAD, including
      a detached one (``previous_branch`` is then ``null``).
    - Never ``checkout -B``, ``--force``, ``--discard-changes``, ``--merge`` or ``reset``.

    The name must pass ``_validate_git_branch_name`` (400, before any command runs)
    and ``git check-ref-format --branch`` (400, before any mutation). A workspace with
    no commit yet is a 400. ``branch`` and ``head_sha`` in the response are measured
    after the switch; a switch that exits 0 but leaves HEAD elsewhere is a 500. So is
    one that exits 0 but leaves a path of the new tree unwritten (a missing blob):
    500 ``branch_switch_incomplete``, measured as tracked paths modified in the
    worktree (``ls-files --modified``: missing or stale content) after the switch
    that were not already modified before it.
    """
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
                f"Container '{container_name}' is not running (current status: {container.status.value})"
            )

        branch = request.branch
        # Validated BEFORE the first command that interpolates it.
        if not _validate_git_branch_name(branch):
            raise bad_request("Invalid branch name: contains forbidden characters or patterns")
        q_branch = _sanitize_shell_arg(branch)
        q_ref = _sanitize_shell_arg(f"refs/heads/{branch}")

        probe = await container_manager.execute_command(
            container_name,
            "cd /workspace && "
            f"{_branch_git_prelude()}"
            "h=$(_g rev-parse --verify --quiet 'HEAD^{commit}') "
            f"|| exit {_BRANCH_UNBORN_EXIT}; "
            f"n=$(_g check-ref-format --branch {q_branch} 2>/dev/null) "
            f'&& [ "$n" = {q_branch} ] || exit {_BRANCH_BADNAME_EXIT}; '
            "b=$(_g symbolic-ref -q --short HEAD) || b=; "
            f"if _g show-ref --verify --quiet {q_ref}; then e=1; else e=0; fi; "
            'printf \'head=%s\\nexists=%s\\nprev=%s\\n\' "$h" "$e" "$b"',
            tenant_id=tenant_id,
        )
        probe_rc = probe.get("exit_code", 1)
        if probe_rc == _BRANCH_UNBORN_EXIT:
            raise bad_request("Workspace repository has no commit yet; nothing to branch from")
        if probe_rc == _BRANCH_BADNAME_EXIT:
            raise bad_request(f"Invalid branch name: git rejects {branch!r} as a branch name")
        before = _kv_lines(probe.get("stdout", ""))
        if probe_rc != 0 or not before.get("head") or before.get("exists") not in ("0", "1"):
            raise internal_error(f"Failed to inspect workspace branch: {_stderr_tail(probe)}")
        exists = before["exists"] == "1"
        previous_branch = before.get("prev") or None

        # Measured BEFORE the switch: modifications (incl. deletions) the caller already
        # had carry over and are not the switch's doing; only NEW ones mean it left the
        # tree incomplete or stale.
        modified_before_res = await container_manager.execute_command(
            container_name, _modified_paths_command(), tenant_id=tenant_id
        )
        if modified_before_res.get("exit_code", 1) != 0:
            raise internal_error(
                f"Failed to inspect workspace worktree: {_stderr_tail(modified_before_res)}"
            )
        # A clipped listing would drop pre-existing modifications from the baseline,
        # and the switch has not run yet: refuse now rather than guess.
        if modified_before_res.get("truncated"):
            raise internal_error(
                "Failed to inspect workspace worktree: modified-path listing was truncated",
                details={"error": "worktree_listing_truncated"},
            )
        modified_before = set(filter(None, modified_before_res.get("stdout", "").split("\0")))

        if exists:
            switch_cmd = _branch_switch_command(["--no-guess", branch])
        else:
            switch_cmd = _branch_switch_command(["--no-track", "-c", branch])
        switched = await container_manager.execute_command(
            container_name, switch_cmd, tenant_id=tenant_id
        )

        state = await container_manager.execute_command(
            container_name, f"cd /workspace && {_branch_state_command()}", tenant_id=tenant_id
        )
        after = _kv_lines(state.get("stdout", "")) if state.get("exit_code", 1) == 0 else {}
        head_after = after.get("head") or None
        branch_after = after.get("branch") or None

        if switched.get("exit_code", 1) != 0:
            stderr = _stderr_tail(switched)
            unchanged = head_after == before["head"] and branch_after == previous_branch
            if unchanged and any(m in stderr for m in _BRANCH_CONFLICT_MARKERS):
                logger.warning(
                    "branch_switch_conflict in container '%s' switching to '%s': %s",
                    container_name,
                    branch,
                    stderr,
                )
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error_code": "branch_switch_conflict",
                        "message": (
                            f"branch_switch_conflict: git refused to switch to '{branch}'; "
                            f"the workspace is unchanged. {stderr}"
                        ),
                        "branch": branch,
                        "current_branch": previous_branch,
                        "stderr_tail": stderr,
                    },
                )
            raise internal_error(
                f"Failed to switch to branch '{branch}': {stderr}",
                details={
                    "error": "branch_switch_failed",
                    "head_after": head_after,
                    "branch_after": branch_after,
                },
            )

        if branch_after != branch or not head_after:
            raise internal_error(
                f"git switch exited 0 but HEAD is not on '{branch}'",
                details={
                    "error": "branch_switch_unverified",
                    "head_after": head_after,
                    "branch_after": branch_after,
                    "state_error": None if after else _stderr_tail(state),
                },
            )

        # A switch can exit 0 with HEAD moved yet paths of the new tree never written
        # (a missing blob) or left with the old content (unwritable dir, short write).
        # Fail closed: a 200 here would let the next ``add -A`` commit those paths as
        # deletions or reverts.
        modified_after_res = await container_manager.execute_command(
            container_name, _modified_paths_command(), tenant_id=tenant_id
        )
        missing: list[str] | None = None
        verify_res = modified_after_res
        if modified_after_res.get("truncated"):
            # A clipped listing silently drops candidates: fail closed.
            verify_res = {"stderr": "modified-path listing after the switch was truncated"}
        elif modified_after_res.get("exit_code", 1) == 0:
            candidates = sorted(
                set(filter(None, modified_after_res.get("stdout", "").split("\0")))
                - modified_before
            )
            missing = await _still_incomplete(
                container_manager, container_name, tenant_id, candidates
            )
            if missing is None:
                verify_res = {"stderr": "byte-exact recheck of the flagged paths failed"}
        if missing is None or missing:
            raise internal_error(
                f"branch_switch_incomplete: HEAD is on '{branch}' but the worktree is "
                "missing or stale paths of its tree; do not commit from this workspace",
                details={
                    "error": "branch_switch_incomplete",
                    "head_after": head_after,
                    "branch_after": branch_after,
                    "missing_paths": missing,
                    "switch_stderr_tail": _stderr_tail(switched),
                    "verify_error": (_stderr_tail(verify_res) if missing is None else None),
                },
            )

        if not exists:
            message = "Branch created"
        elif previous_branch == branch:
            message = "Already on branch"
        else:
            message = "Switched to branch"
        logger.info(
            "%s '%s' in container '%s' (from %s, head %s)",
            message,
            branch,
            container_name,
            previous_branch or "detached HEAD",
            head_after,
        )

        return GitBranchResponse(
            message=message,
            container_name=container_name,
            branch=branch,  # == branch_after: re-measured and verified above
            created=not exists,
            head_sha=head_after,
            previous_branch=previous_branch,
            request_id=str(uuid7()),
        )

    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as e:
        logger.error("Failed to switch branch in container '%s': %s", container_name, e)
        raise internal_error(f"Failed to switch branch: {str(e)}") from e

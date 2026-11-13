"""Workspace git fetch route: ``POST /{container}/git/fetch``.

WHY. A workspace's ``origin`` is tokenless: ``GIT_TOKEN`` reaches the clone init
container only, and the long-lived container has no credential. So a private
repository's pull-request head (``pull/<n>/head``) or a branch pushed after the
clone could not be brought into the workspace at all, and computed QA cannot check
out the delivered PR head (measured at origin/main 1d1fd7f).

HOW. Modelled on ``git/push`` (``git_push.py``) and sharing its parts, not copies
of them:

* The credential is RE-RESOLVED on every call by ``resolve_repo_credential`` --
  the SAME ladder the push uses (BYO token / secret_config_id, then coding-github's
  tenant-bound App token; platform repositories need a stated tenant and honour
  ``platform_repo_forbidden``; NO org-PAT fallback), with ``scope="clone"``.
* The fetch target must be the workspace's own origin repository, on an
  allowlisted host, in canonical form -- the same three refusals as the push.
* The token reaches git ONLY as ``$GIT_PUSH_TOKEN`` in the environment of the ONE
  fetch exec, read by the host- and path-bound credential helper the push uses. It
  is never written into ``.git/config``, a remote URL, a file or argv, and never
  logged. It is ``unset`` before the fetched ref is resolved.
* The same config guard runs first (refusing redirect / credential-helper / proxy /
  include / promisor / submodule / trace2 config), the same transport pins apply
  (``https`` only, no submodule recursion, no lazy fetch), and the explicit URL is
  re-checked for an ``insteadOf`` rewrite inside the exec.

The refspec is validated here, strictly, before anything runs (see
``validate_fetch_refspec``): a destination may only be a remote-tracking ref, so a
fetch can never move a branch, a tag, ``HEAD`` or plant a ``refs/replace`` ref.

This module is mounted on the canonical ``/api/workspaces`` surface ONLY (from
``src.routers.workspaces``), never the legacy ``/api/containers`` alias.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from typing import NamedTuple

import docker.errors
from fastapi import APIRouter, Body, Depends, HTTPException, Path

from coding_client.auth.models import User
from coding_client.auth.tenant import get_current_user_with_tenant
from coding_client.models.responses import (
    bad_request,
    internal_error,
    not_found,
    unauthorized,
)
from src.clients.github_client import (
    GITHUB_HOST,
    canonical_github_repo,
    canonical_github_url,
    same_github_repo,
)
from src.models.container_api_models import GitFetchRequest, GitFetchResponse
from src.models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error422Response,
    Error500Response,
)

from . import git_push as _gp
from ._common import (
    _sanitize_error_message,
    _validate_git_push_remote,
    get_git_manager,
    workspace_git_config_args,
    workspace_git_env,
    workspace_poison_check,
)
from .git_push import (
    _CONFIG_EXIT,
    _REWRITE_EXIT,
    _TOKEN_ENV,
    _config_guard_command,
    _guard_keys,
    _push_exec_prelude,
    _push_git_config_args,
    resolve_repo_credential,
)

logger = logging.getLogger(__name__)
router = APIRouter()

#: Exit status of the fetch exec when the fetched ref does not resolve to a commit.
_HEAD_EXIT = 94

#: Prefix of the line the fetch exec prints with the fetched commit id.
_HEAD_PREFIX = "head "

#: Destination namespace a fetch may write. Remote-tracking refs only: never
#: ``refs/heads`` (moves a branch, or the checked-out one), ``refs/tags``,
#: ``refs/replace`` (rewrites every later read) or ``HEAD``.
_DST_PREFIX = "refs/remotes/"

#: One side of a refspec: starts alphanumeric (so never ``-option``, ``.``, ``/``),
#: then only ``[A-Za-z0-9._/-]`` -- no whitespace, ``:``, ``*``, ``^``, ``~``,
#: ``?``, ``[``, ``\\``, ``@{``, quotes or shell metacharacters.
_REF_SIDE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
_MAX_REFSPEC_CHARS = 255

_COMMIT_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class FetchRefspec(NamedTuple):
    """A validated refspec: ``[+]<src>[:<dst>]``."""

    force: bool
    src: str
    dst: str | None

    def render(self) -> str:
        spec = self.src if self.dst is None else f"{self.src}:{self.dst}"
        return f"+{spec}" if self.force else spec

    @property
    def fetched_ref(self) -> str:
        """The local ref that holds the fetched commit afterwards."""
        return self.dst if self.dst is not None else "FETCH_HEAD"


def _valid_ref_side(value: str) -> bool:
    if not _REF_SIDE_RE.fullmatch(value):
        return False
    if ".." in value or "//" in value or value.endswith(("/", ".")):
        return False
    return all(
        part and not part.startswith(".") and not part.endswith(".lock")
        for part in value.split("/")
    )


def validate_fetch_refspec(refspec: str) -> FetchRefspec | None:
    """Parse ``refspec`` strictly; ``None`` means refuse.

    Accepted: ``[+]<src>[:<dst>]`` where each side passes ``_valid_ref_side`` and a
    ``dst`` lies under ``refs/remotes/<x>/...``. Refused: a leading ``-`` (option
    injection), ``..`` (range / traversal), whitespace or any control character,
    more than one ``:``, globs, negative refspecs (``^``), an empty side, and any
    destination outside the remote-tracking namespace.
    """
    if not refspec or len(refspec) > _MAX_REFSPEC_CHARS:
        return None
    force = refspec.startswith("+")
    body = refspec[1:] if force else refspec
    src, sep, dst = body.partition(":")
    if not _valid_ref_side(src):
        return None
    if not sep:
        return FetchRefspec(force=force, src=src, dst=None)
    if not _valid_ref_side(dst):
        return None
    rest = dst[len(_DST_PREFIX) :] if dst.startswith(_DST_PREFIX) else ""
    if "/" not in rest:  # refs/remotes/<remote>/<name> at minimum
        return None
    return FetchRefspec(force=force, src=src, dst=dst)


def _fetch_command(owner: str, repo: str, spec: FetchRefspec, depth: int | None) -> str:
    """One exec: refuse a poisoned workspace, run the config guard, isolate config,
    re-verify URL resolution, fetch from the EXPLICIT URL, then (token unset)
    resolve the fetched ref and print ``head <sha>``.

    Same layering as ``git_push._push_command``; the residual is the same too
    (``$GIT_PUSH_TOKEN`` is in this exec's environment while the fetch runs).
    """
    url = f"https://{GITHUB_HOST}/{owner}/{repo}"
    q_url = shlex.quote(url)
    safe_dir = " ".join(
        shlex.quote(a)
        for a in ["-c", f"safe.directory={_gp._WORKDIR}", *workspace_git_config_args()]
    )
    git_c = " ".join(
        shlex.quote(a)
        for a in [
            *workspace_git_config_args(),
            *_push_git_config_args(owner, repo),
            # No background maintenance child inheriting the token.
            "-c",
            "gc.auto=0",
            "-c",
            "maintenance.auto=false",
        ]
    )
    depth_flag = f" --depth={int(depth)}" if depth is not None else ""
    q_ref = shlex.quote(f"{spec.fetched_ref}^{{commit}}")
    return (
        f"cd {shlex.quote(_gp._WORKDIR)} || exit 1; "
        f"{_push_exec_prelude()}"
        f"{workspace_poison_check()}"
        f"guard_keys=$({_config_guard_command()}); guard_rc=$?; "
        'if [ "$guard_rc" -ne 1 ] || [ -n "$guard_keys" ]; then '
        f'printf "%s\\n" "$guard_keys"; exit {_CONFIG_EXIT}; fi; '
        f"{workspace_git_env()}export GIT_ASKPASS= SSH_ASKPASS= GIT_ALLOW_PROTOCOL=https && "
        f"fetch_url=$(git {safe_dir} ls-remote --get-url {q_url}) && "
        f'test "$fetch_url" = {q_url} || '
        f"{{ echo 'fetch_url_rewritten: the fetch URL does not resolve to itself' >&2; "
        f"exit {_REWRITE_EXIT}; }}; "
        f"git {git_c} fetch --no-recurse-submodules --no-tags{depth_flag} "
        f"{q_url} {shlex.quote(spec.render())} || exit $?; "
        f"unset {_TOKEN_ENV}; "
        f"head=$(git {safe_dir} rev-parse --verify --quiet {q_ref}) || exit {_HEAD_EXIT}; "
        f'printf "{_HEAD_PREFIX}%s\\n" "$head"'
    )


#: ``git rev-parse --is-shallow-repository`` answers exactly one of these.
_SHALLOW_TRUE = "true"
_SHALLOW_FALSE = "false"


def _shallow_check_command() -> str:
    """Read-only exec: print whether the workspace repository is already shallow.

    WHY. ``git fetch --depth`` on a FULL clone writes ``.git/shallow`` and converts
    it to shallow permanently, which can break a later push, merge-base or log in
    that workspace. So ``depth`` is honoured only where the repository is already
    shallow (every provisioned workspace is a ``clone --depth 1``), and a full clone
    is refused before any credential is resolved. No token is in this exec.
    """
    safe_dir = " ".join(
        shlex.quote(a)
        for a in ["-c", f"safe.directory={_gp._WORKDIR}", *workspace_git_config_args()]
    )
    return (
        f"cd {shlex.quote(_gp._WORKDIR)} || exit 1; "
        f"{workspace_git_env()}"
        f"git {safe_dir} rev-parse --is-shallow-repository"
    )


def _parse_head(stdout: str) -> str | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_HEAD_PREFIX):
            sha = line[len(_HEAD_PREFIX) :].strip()
            return sha if _COMMIT_SHA_RE.match(sha) else None
    return None


def _fetch_config_refusal(container_name: str, keys: list[str]) -> HTTPException:
    logger.error(
        "fetch_config_refused for container %s: workspace git config sets %s",
        container_name,
        ", ".join(keys) or "<unreadable>",
    )
    return HTTPException(
        status_code=409,
        detail={
            "error_code": "fetch_config_refused",
            "message": (
                "fetch_config_refused: the workspace git configuration contains settings "
                "that could redirect the fetch or hand its credential to another program "
                f"({', '.join(keys) or 'git config could not be read'}). Remove them and retry."
            ),
            "keys": keys,
        },
    )


@router.post(
    "/{container_name}/git/fetch",
    response_model=GitFetchResponse,
    operation_id="fetch_workspace_ref",
    summary="Fetch a Ref into the Workspace",
    responses={
        200: {"description": "Ref fetched; head_sha is the fetched commit"},
        400: {"model": Error400Response, "description": "Invalid refspec or remote"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Repository not permitted"},
        404: {"model": Error404Response, "description": "Workspace not found"},
        422: {"model": Error422Response, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Error fetching"},
    },
)
async def fetch_workspace_ref(
    container_name: str = Path(..., description="Workspace name"),
    request: GitFetchRequest = Body(...),
    user: User = Depends(get_current_user_with_tenant),
) -> GitFetchResponse:
    """Fetch ``refspec`` from the workspace's origin repository with a per-call credential."""
    github_token: str | None = None
    try:
        manager = get_git_manager()
        tenant_id = None if user.is_admin() else user.effective_tenant_id
        try:
            container = manager.get_container(container_name, tenant_id=tenant_id)
        except ValueError:
            container = None
        if not container:
            raise not_found("Workspace", container_name)
        if container.status.value.lower() != "running":
            raise bad_request(
                f"Workspace '{container_name}' is not running "
                f"(current status: {container.status.value})"
            )

        # Validated BEFORE anything is resolved or run.
        spec = validate_fetch_refspec(request.refspec)
        if spec is None:
            raise bad_request(
                "Invalid refspec: expected [+]<src>[:refs/remotes/<remote>/<name>] using only "
                "[A-Za-z0-9._/-], not starting with '-' and without '..' or whitespace"
            )

        remote_url = request.remote_url or getattr(container, "repo_url", None) or ""
        if not remote_url:
            raise bad_request("No remote URL provided and workspace has no configured repository")
        if not _validate_git_push_remote(remote_url):
            raise bad_request(
                "remote_url host is not an allowed GitHub target; "
                "refusing to send credentials to an untrusted remote"
            )
        target = canonical_github_repo(remote_url)
        if target is None:
            raise bad_request(
                "remote_url is not a canonical GitHub repository URL "
                "(https://github.com/<owner>/<repo>, git@github.com:<owner>/<repo>)"
            )
        origin = canonical_github_repo(getattr(container, "repo_url", None))
        if not same_github_repo(target, origin):
            label = f"{target[0]}/{target[1]}"
            origin_label = f"{origin[0]}/{origin[1]}" if origin else "<none>"
            logger.warning(
                "fetch_target_mismatch for workspace %s: requested %s, provisioned from %s",
                container_name,
                label,
                origin_label,
            )
            raise HTTPException(
                status_code=403,
                detail={
                    "error_code": "fetch_target_mismatch",
                    "message": (
                        f"fetch_target_mismatch: {label} is not this workspace's origin "
                        f"repository ({origin_label}); a workspace fetches only from the "
                        "repository it was provisioned from (a fork's PR head is "
                        "reachable as pull/<n>/head on that repository)."
                    ),
                    "requested": label,
                    "origin": origin_label,
                },
            )
        owner, repo = target
        clean_url = canonical_github_url(owner, repo)

        # A depth on a FULL clone would make it shallow forever: refuse it here,
        # BEFORE any credential is resolved. Fail closed on an unreadable answer.
        if request.depth is not None:
            shallow = await manager.execute_command(
                container_name, _shallow_check_command(), tenant_id=tenant_id
            )
            answer = (shallow.get("stdout") or "").strip()
            if shallow.get("exit_code", 1) != 0 or answer not in (_SHALLOW_TRUE, _SHALLOW_FALSE):
                raise internal_error(
                    "Could not determine whether the workspace repository is shallow; "
                    "refusing a depth-limited fetch"
                )
            if answer == _SHALLOW_FALSE:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_code": "fetch_depth_on_full_clone",
                        "message": (
                            "fetch_depth_on_full_clone: this workspace holds a full "
                            "(non-shallow) clone, and fetching with depth would convert it "
                            "to a shallow clone permanently. Omit depth and retry."
                        ),
                    },
                )

        github_token = await resolve_repo_credential(
            container,
            tenant_id=tenant_id,
            owner=owner,
            clean_url=clean_url,
            container_name=container_name,
            scope="clone",
        )
        if not github_token:
            raise unauthorized("GitHub authentication not configured for this workspace")

        result = await manager.execute_command(
            container_name,
            _fetch_command(owner, repo, spec, request.depth),
            tenant_id=tenant_id,
            timeout=int(os.getenv("GIT_FETCH_TIMEOUT", "900")),
            env={_TOKEN_ENV: github_token},
        )
        rc = result.get("exit_code", 1)
        if rc == _CONFIG_EXIT:
            raise _fetch_config_refusal(container_name, _guard_keys(result))
        if rc == _REWRITE_EXIT:
            logger.error(
                "fetch_url_rewritten for workspace %s: %s did not resolve to itself",
                container_name,
                clean_url,
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": "fetch_url_rewritten",
                    "message": (
                        f"fetch_url_rewritten: {clean_url} is rewritten by the workspace "
                        "git configuration; the fetch was refused."
                    ),
                },
            )
        if rc == _HEAD_EXIT:
            raise internal_error(
                f"Fetched {spec.src} but {spec.fetched_ref} does not resolve to a commit"
            )
        if rc != 0:
            error_msg = _sanitize_error_message(
                result.get("stderr") or "Unknown error", github_token
            )
            if "authentication" in error_msg.lower() or "permission" in error_msg.lower():
                raise unauthorized(f"GitHub authentication failed: {error_msg}")
            raise internal_error(f"Failed to fetch {spec.src}: {error_msg}")

        head_sha = _parse_head(result.get("stdout") or "")
        if head_sha is None:
            raise internal_error(f"Fetched {spec.src} but its commit id could not be read")

        logger.info(
            "Fetched %s from %s into %s (%s) in workspace %s",
            spec.src,
            clean_url,
            spec.fetched_ref,
            head_sha,
            container_name,
        )
        return GitFetchResponse(
            ok=True,
            message=f"Fetched {spec.src}",
            container_name=container_name,
            remote_url=clean_url,
            fetched_ref=spec.fetched_ref,
            head_sha=head_sha,
        )
    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as e:
        safe = _sanitize_error_message(str(e), github_token)
        logger.error("Failed to fetch in workspace '%s': %s", container_name, safe)
        raise internal_error(f"Failed to fetch: {safe}") from e

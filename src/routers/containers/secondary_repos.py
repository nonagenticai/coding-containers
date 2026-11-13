"""Secondary (read-only) repositories in a workspace: ``POST``/``GET {ws}/repos``.

WHY. The P7 ``integration_check`` card probes a consumer against its provider in ONE
workspace pod (George 2026-10-12): the workspace holds the consumer, and the
provider (or any other repository the check needs) is added here, read-only, at
``/workspace/.repos/<owner>__<repo>``. A provider is then started from that path with
``POST {ws}/services``. No second pod, no cross-pod networking.

ACCESS — no new grant. The credential is resolved per call under the caller's
tenant with the same gates provisioning applies, clone scope only:

* PLATFORM owner (``nonagenticai``…): a tenant is required; coding-github is asked for
  a ``clone``-scope App token for THIS repository and this tenant; its
  ``platform_repo_forbidden`` is a 403 and stops; any other null token is a 412. The
  org PAT is never a credential, and — stricter than provisioning — neither is the
  workspace's BYO token: a platform repository is reachable only through
  coding-github's tenant gate.
* CUSTOMER owner: the workspace's BYO token (the tenant's own PAT), else a
  tenant-bound App token, else — only when coding-github positively says the
  repository is PUBLIC — a tokenless clone. Anything else is a 412.

READ-ONLY. A secondary checkout has no remote at all (it is fetched by explicit URL),
no push credential is ever resolved for it (only ``scope="clone"``), and the push
route only ever pushes ``/workspace`` to the workspace's own origin. ``/.repos/`` is
added to the workspace repository's ``.git/info/exclude`` so ``git add -A`` / status /
diff on the primary repository never see the secondary checkouts.

CREDENTIAL HANDLING (same shape as ``git/fetch``): the token reaches git only as
``$GIT_PUSH_TOKEN`` in the ONE clone exec, read by the host- and path-bound
credential helper; it is never in a URL, ``.git/config``, a command string or a log.
⚠️ On the Kubernetes backend per-command env is applied as ``/usr/bin/env K=V`` argv
(``runtime/kubernetes.py with_env``), so the token IS in that process's argv for the
instant before ``env`` execs ``sh`` -- readable by same-uid workspace code. The clone is
built in a FRESH ``mktemp -d`` directory under isolated config (no system/global
config, hooks off, https only, no submodule recursion), checked for a token left in
``.git`` (tripwire), the token is unset before checkout, and only then is the
checkout moved into place. Residual (as for push/fetch): the token is in that exec's
environment while the fetch runs, readable by same-uid workspace code via ``/proc``;
it is a short-lived ONE-repository contents:read token for a repository this tenant
was just granted read access to by the same gate.

Mounted on the canonical ``/api/workspaces`` surface only.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path
from pydantic import BaseModel, ConfigDict, Field

from coding_client.auth.tenant import require_tenant
from coding_client.models.responses import bad_request, internal_error, not_found
from src.clients.github_client import (
    APP_NOT_INSTALLED_ERROR,
    GITHUB_HOST,
    canonical_github_repo,
    canonical_github_url,
    get_github_token_manager,
    is_platform_owner,
)
from src.core.kubernetes_container_provisioner import (
    GitCredentialUnavailableError,
    platform_token_or_refuse,
    refuse_if_platform_repo_forbidden,
    require_tenant_for_platform_repo,
)

from ._common import (
    SECONDARY_REPOS_DIR,
    _sanitize_error_message,
    get_manager,
    workspace_git_config_args,
    workspace_git_env,
)
from .git_fetch import _valid_ref_side
from .git_push import (
    _REWRITE_EXIT,
    _TOKEN_ENV,
    _credential_error_message,
    _push_git_config_args,
)

logger = logging.getLogger(__name__)
router = APIRouter()

#: Read at call time (tests point them at a scratch directory).
WORKSPACE_ROOT = "/workspace"
_EXCLUDE_LINE = f"/{SECONDARY_REPOS_DIR}/"

#: Exit codes of the clone exec.
_HEAD_EXIT = 94
_TOKEN_LEFT_EXIT = 95
_HEAD_PREFIX = "head "
_REPO_PREFIX = "repo\t"
_COMMIT_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_DEFAULT_REF = "HEAD"
_MAX_REF_CHARS = 255


class AddRepoRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"repo_url": "https://github.com/octocat/Hello-World", "ref": "master"}
        }
    )

    repo_url: str = Field(
        ...,
        max_length=512,
        description="GitHub repository: https://github.com/<owner>/<repo> or <owner>/<repo>",
        json_schema_extra={"x-n8n-param-name": "secondaryRepoUrl"},
    )
    ref: str | None = Field(
        default=None,
        max_length=_MAX_REF_CHARS,
        description="Branch, tag or full commit SHA; default: the repository's default branch",
        json_schema_extra={"x-n8n-param-name": "secondaryRepoRef"},
    )
    refresh: bool = Field(
        default=False,
        description="Re-fetch even when this repository is already checked out at this ref",
        json_schema_extra={"x-n8n-param-name": "secondaryRepoRefresh"},
    )


class SecondaryRepo(BaseModel):
    repo: str = Field(description="<owner>/<repo>")
    repo_url: str
    ref: str
    path: str = Field(description="Absolute path in the workspace (use as exec/service workdir)")
    head_sha: str
    read_only: bool = True


class AddRepoResponse(SecondaryRepo):
    container_name: str
    reused: bool = Field(description="True when an existing checkout at this ref was returned")


class SecondaryRepoList(BaseModel):
    container_name: str
    repos: list[SecondaryRepo]


# ============================================================================
# Validation
# ============================================================================


def parse_repo(repo_url: str) -> tuple[str, str]:
    """``(owner, repo)`` from a canonical GitHub URL or ``owner/repo``; ValueError otherwise."""
    raw = (repo_url or "").strip()
    if raw and "://" not in raw and "@" not in raw and raw.count("/") == 1:
        raw = f"https://{GITHUB_HOST}/{raw}"
    target = canonical_github_repo(raw)
    if target is None:
        raise ValueError("repo_url must be https://github.com/<owner>/<repo> (or <owner>/<repo>)")
    return target


def validate_ref(ref: str | None) -> str:
    """A branch/tag name or a commit SHA, safe as one git argv word."""
    value = (ref or "").strip() or _DEFAULT_REF
    if len(value) > _MAX_REF_CHARS or not _valid_ref_side(value):
        raise ValueError(
            "ref must be a branch, tag or commit SHA using only [A-Za-z0-9._/-], "
            "not starting with '-' and without '..'"
        )
    return value


def secondary_path(owner: str, repo: str) -> str:
    """``/workspace/.repos/<owner>__<repo>`` (GitHub owners never contain ``_``)."""
    return f"{WORKSPACE_ROOT}/{SECONDARY_REPOS_DIR}/{owner}__{repo}"


# ============================================================================
# Commands (server-built; every value shell-quoted)
# ============================================================================


def _git(*extra: str) -> str:
    return " ".join(shlex.quote(a) for a in ["git", *workspace_git_config_args(), *extra])


def _describe_fragment(dest_var: str) -> str:
    """Print ``repo\\t<dir>\\t<url>\\t<ref>\\t<head>`` for the checkout in ``$<dest_var>``."""
    g = _git()
    return (
        f'if [ -d "${dest_var}/.git" ]; then '
        f'u=$(cd "${dest_var}" && {g} config --get coding.secondaryurl); '
        f'r=$(cd "${dest_var}" && {g} config --get coding.secondaryref); '
        f"h=$(cd \"${dest_var}\" && {g} rev-parse --verify --quiet 'HEAD^{{commit}}'); "
        f'printf "repo\\t%s\\t%s\\t%s\\t%s\\n" "${{{dest_var}##*/}}" "$u" "$r" "$h"; fi'
    )


def _inspect_command(owner: str, repo: str) -> str:
    """Read-only, no credential: describe the existing checkout, if any."""
    dest = shlex.quote(secondary_path(owner, repo))
    return f"{workspace_git_env()}d={dest}; {_describe_fragment('d')}"


def _list_command() -> str:
    root = shlex.quote(f"{WORKSPACE_ROOT}/{SECONDARY_REPOS_DIR}")
    return (
        f"{workspace_git_env()}for d in {root}/*; do "
        f'[ -L "$d" ] && continue; {_describe_fragment("d")}; done; true'
    )


def _clone_command(owner: str, repo: str, ref: str) -> str:
    """One exec: fresh dir, isolated config, fetch ref at depth 1, tripwire, checkout, move.

    ``$GIT_PUSH_TOKEN`` (absent for a public clone) is read only by the path-bound
    helper and is unset before checkout.
    """
    url = canonical_github_url(owner, repo)
    q_url = shlex.quote(url)
    q_ref = shlex.quote(ref)
    q_ws = shlex.quote(WORKSPACE_ROOT)
    q_root = shlex.quote(f"{WORKSPACE_ROOT}/{SECONDARY_REPOS_DIR}")
    q_dest = shlex.quote(secondary_path(owner, repo))
    q_exclude = shlex.quote(_EXCLUDE_LINE)
    git_plain = _git()
    git_c = " ".join(
        shlex.quote(a)
        for a in [
            "git",
            *workspace_git_config_args(),
            *_push_git_config_args(owner, repo),
            "-c",
            "gc.auto=0",
            "-c",
            "maintenance.auto=false",
        ]
    )
    return (
        f"cd {q_ws} || exit 1; "
        # Primary repository: never see the secondary checkouts (status / add -A / diff).
        "if [ -d .git ] && [ ! -L .git ]; then mkdir -p .git/info && "
        f"{{ grep -qxF {q_exclude} .git/info/exclude 2>/dev/null || "
        f"printf '%s\\n' {q_exclude} >> .git/info/exclude; }}; fi; "
        f"mkdir -p {q_root} || exit 1; "
        f"tmp=$(mktemp -d {q_root}/.clone.XXXXXXXX) || exit 1; "
        "trap 'rm -rf \"$tmp\"' EXIT; "
        f"{workspace_git_env()}"
        f'{git_plain} init -q --template= "$tmp" >/dev/null || exit 1; '
        'cd "$tmp" || exit 1; '
        f"export GIT_ASKPASS= SSH_ASKPASS= GIT_ALLOW_PROTOCOL=https; "
        f"fetch_url=$({git_plain} ls-remote --get-url {q_url}) && "
        f'test "$fetch_url" = {q_url} || '
        f"{{ echo 'fetch_url_rewritten: the clone URL does not resolve to itself' >&2; "
        f"exit {_REWRITE_EXIT}; }}; "
        f"{git_c} fetch --quiet --no-recurse-submodules --no-tags --depth=1 "
        f"{q_url} {q_ref} || exit $?; "
        f'if [ -n "${{{_TOKEN_ENV}:-}}" ] && '
        f'grep -rlF --exclude-dir=objects -e "${_TOKEN_ENV}" .git >/dev/null 2>&1; then '
        f"echo 'credential_left_in_git: refusing the checkout' >&2; exit {_TOKEN_LEFT_EXIT}; fi; "
        f"unset {_TOKEN_ENV}; export GIT_ALLOW_PROTOCOL=; "
        f"{git_plain} -c advice.detachedHead=false checkout -q --detach FETCH_HEAD || exit 1; "
        f"{git_plain} config coding.secondaryurl {q_url} && "
        f"{git_plain} config coding.secondaryref {q_ref} || exit 1; "
        f"head=$({git_plain} rev-parse --verify --quiet 'HEAD^{{commit}}') || exit {_HEAD_EXIT}; "
        f"cd {q_ws} || exit 1; "
        f'rm -rf {q_dest} && mv "$tmp" {q_dest} || exit 1; '
        "trap - EXIT; "
        f'printf "{_HEAD_PREFIX}%s\\n" "$head"'
    )


def _parse_head(stdout: str) -> str | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_HEAD_PREFIX):
            sha = line[len(_HEAD_PREFIX) :].strip()
            return sha if _COMMIT_SHA_RE.match(sha) else None
    return None


def _parse_repos(stdout: str) -> list[SecondaryRepo]:
    repos: list[SecondaryRepo] = []
    for line in stdout.splitlines():
        if not line.startswith(_REPO_PREFIX):
            continue
        parts = line.split("\t")
        if len(parts) != 5:
            continue
        _, dirname, url, ref, head = parts
        target = canonical_github_repo(url)
        if target is None or not _COMMIT_SHA_RE.match(head) or dirname != "__".join(target):
            continue  # not ours (or tampered with): never reported as a secondary repo
        owner, repo = target
        repos.append(
            SecondaryRepo(
                repo=f"{owner}/{repo}",
                repo_url=canonical_github_url(owner, repo),
                ref=ref,
                path=secondary_path(owner, repo),
                head_sha=head,
            )
        )
    return repos


# ============================================================================
# Credential (clone scope only — a push credential is never resolved here)
# ============================================================================


def _refusal(exc: GitCredentialUnavailableError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={
            "error_code": exc.reason,
            "message": _credential_error_message(exc),
            "reason": exc.reason,
        },
    )


async def resolve_secondary_credential(
    container: Any, *, tenant_id: str, owner: str, repo: str
) -> str | None:
    """The clone token for a secondary repository, ``None`` for a positively public one.

    Raises ``HTTPException``: 403 ``platform_repo_forbidden``, 412 for every other
    missing credential. See the module docstring for the ladder.
    """
    url = canonical_github_url(owner, repo)
    manager = get_github_token_manager()
    try:
        if is_platform_owner(owner):
            req_tenant = require_tenant_for_platform_repo(
                tenant_id, owner=owner, repo_url=url, context="secondary_repo"
            )
            credential = await manager.get_app_clone_credential(
                url, tenant_id=req_tenant, scope="clone"
            )
            refuse_if_platform_repo_forbidden(
                credential, repo_url=url, owner=owner, tenant_id=req_tenant
            )
            return platform_token_or_refuse(
                credential, repo_url=url, owner=owner, tenant_id=req_tenant
            )
    except GitCredentialUnavailableError as exc:
        raise _refusal(exc) from exc

    byo = getattr(container, "github_access_token", None)
    if not byo and getattr(container, "secret_config_id", None):
        from src.services.secret_resolver import resolve_git_token

        byo = await resolve_git_token(container.secret_config_id, tenant_id)
    if byo:
        return byo
    credential = await manager.get_app_clone_credential(url, tenant_id=tenant_id, scope="clone")
    if credential.token:
        return credential.token
    if credential.is_public:
        return None
    reason = credential.reason or credential.error or "unknown"
    raise HTTPException(
        status_code=412,
        detail={
            "error_code": APP_NOT_INSTALLED_ERROR if reason == "not_installed" else reason,
            "message": (
                f"no clone credential for {owner}/{repo}: no BYO token on this workspace "
                f"and coding-github issued no tenant-bound token (reason={reason}), and the "
                "repository is not public."
            ),
            "reason": reason,
        },
    )


# ============================================================================
# Routes
# ============================================================================


def _workspace(container_name: str, tenant_id: str) -> Any:
    try:
        container = get_manager().get_container(container_name, tenant_id=tenant_id)
    except (ValueError, RuntimeError):  # fmt: skip
        container = None
    if container is None:
        raise not_found("Workspace", container_name)
    status = getattr(getattr(container, "status", None), "value", "running")
    if str(status).lower() != "running":
        raise bad_request(f"Workspace '{container_name}' is not running (current status: {status})")
    return container


@router.post(
    "/{container_name}/repos",
    response_model=AddRepoResponse,
    operation_id="add_workspace_repo",
    summary="Add a read-only secondary repository to the workspace",
    responses={
        400: {"description": "Invalid repository or ref"},
        403: {"description": "Repository not permitted for this tenant"},
        404: {"description": "Workspace not found"},
        409: {"description": "The clone URL was rewritten"},
        412: {"description": "No clone credential for this repository"},
    },
)
async def add_workspace_repo(
    container_name: str = Path(..., description="Workspace name"),
    request: AddRepoRequest = Body(...),
    tenant_id: str = Depends(require_tenant),
) -> AddRepoResponse:
    """Check ``repo_url`` out at ``ref`` under ``/workspace/.repos/<owner>__<repo>``.

    Access is decided on EVERY call (coding-github clone-scope gate / public check),
    reuse included. Idempotent for a commit-SHA ref: a checkout whose HEAD is that SHA
    is returned as-is (``reused``) unless ``refresh`` is set; a branch/tag ref is
    re-fetched each time, since the remote may have moved.
    """
    try:
        owner, repo = parse_repo(request.repo_url)
        ref = validate_ref(request.ref)
    except ValueError as e:
        raise bad_request(str(e)) from e
    container = _workspace(container_name, tenant_id)
    manager = get_manager()
    url = canonical_github_url(owner, repo)

    # ⛔ The access decision runs FIRST, on every call -- including a reuse. A checkout
    # already on disk proves nothing: access may have been revoked since, and code in
    # the workspace can plant a directory whose config claims any repository (a
    # forbidden platform repo included), which would otherwise come back as a 200.
    token = await resolve_secondary_credential(
        container, tenant_id=tenant_id, owner=owner, repo=repo
    )
    if not request.refresh and _COMMIT_SHA_RE.match(ref):
        # Reuse only what is verifiable without the network: a commit-SHA ref whose
        # checked-out HEAD IS that SHA. A branch/tag/HEAD ref may have moved, so it is
        # always re-fetched (a depth-1 fetch), never returned stale.
        found = await manager.execute_command(
            container_name, _inspect_command(owner, repo), tenant_id=tenant_id, timeout=60
        )
        existing = _parse_repos(found.get("stdout") or "")
        if (
            existing
            and existing[0].repo_url == url
            and existing[0].ref == ref
            and existing[0].head_sha == ref
        ):
            return AddRepoResponse(
                container_name=container_name, reused=True, **existing[0].model_dump()
            )

    logger.info(
        "add secondary repo: tenant=%s workspace=%s repo=%s/%s ref=%s credential=%s",
        tenant_id,
        container_name,
        owner,
        repo,
        ref,
        "token" if token else "public",
    )
    result = await manager.execute_command(
        container_name,
        _clone_command(owner, repo, ref),
        tenant_id=tenant_id,
        timeout=int(os.getenv("GIT_FETCH_TIMEOUT", "900")),
        env={_TOKEN_ENV: token} if token else None,
    )
    rc = result.get("exit_code", 1)
    if rc != 0:
        message = _sanitize_error_message(str(result.get("stderr") or "unknown error"), token)
        if rc == _REWRITE_EXIT:
            raise HTTPException(
                status_code=409,
                detail={"error_code": "fetch_url_rewritten", "message": message},
            )
        if rc == _TOKEN_LEFT_EXIT:
            raise internal_error("credential_left_in_git: the checkout was refused")
        raise internal_error(f"Failed to check out {owner}/{repo}@{ref}: {message[:2000]}")
    head = _parse_head(result.get("stdout") or "")
    if head is None:
        raise internal_error(
            f"Checked out {owner}/{repo}@{ref} but its commit id could not be read"
        )
    return AddRepoResponse(
        container_name=container_name,
        reused=False,
        repo=f"{owner}/{repo}",
        repo_url=url,
        ref=ref,
        path=secondary_path(owner, repo),
        head_sha=head,
    )


@router.get(
    "/{container_name}/repos",
    response_model=SecondaryRepoList,
    operation_id="list_workspace_repos",
    summary="List the workspace's secondary repositories",
)
async def list_workspace_repos(
    container_name: str = Path(..., description="Workspace name"),
    tenant_id: str = Depends(require_tenant),
) -> SecondaryRepoList:
    _workspace(container_name, tenant_id)
    result = await get_manager().execute_command(
        container_name, _list_command(), tenant_id=tenant_id, timeout=60
    )
    return SecondaryRepoList(
        container_name=container_name, repos=_parse_repos(result.get("stdout") or "")
    )

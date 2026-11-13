"""
Kubernetes-based container provisioning for code analysis.

This module creates a Kubernetes Deployment per provisioned repository.
Each pod runs a git-clone init container that populates /workspace, then
a single long-lived sleep container (``python:3.14-slim``) that holds
the volume while coding-analysis reaches in via ``kubectl exec`` (see
``exec_in_pod``). The image needs ``/bin/sh`` + ``python3`` because
``repo_scanner_modules`` writes and runs a Python script inside the pod
to collect dependencies/imports.

The pre-split ``code_graph_helper`` sidecar that used to serve an HTTP
API on :8687 is gone — nothing listens inside the pod, and the per-pod
Kubernetes Service has been removed along with it. ``ContainerInfo.
internal_endpoint`` is still populated with a ``*.svc.cluster.local:8687``
string because ``container_exec.py`` uses that substring as a routing
sigil to pick the K8s exec path; the URL itself is not reachable.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shlex
import time
from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import urlparse
from uuid import uuid7

from fastapi import status as http_status
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.config.config_exception import ConfigException
from kubernetes.stream import stream as k8s_stream

from coding_client.exceptions import BaseAPIException
from src.clients.github_client import (
    APP_NOT_INSTALLED_ERROR,
    PLATFORM_REPO_FORBIDDEN,
    CloneCredential,
    canonical_github_repo,
    canonical_github_url,
    get_github_token_manager,  # TODO: resolve cross-service dependency
    is_platform_owner,
)
from src.config.settings import get_settings
from src.core.container_metadata_registry import ContainerMetadataRegistry, RemoveOutcome
from src.core.container_spec import (
    GitHubRepoRequest,  # TODO: resolve cross-service dependency - this module may need to be extracted too
)
from src.core.git_clone_credentials import (
    CLONE_TOKEN_ENV,
    clone_credential_shell_args,
    repo_path_from_url_path,
    scrub_and_verify,
)
from src.core.graph_operations import get_graph_operations
from src.core.log_safety import redact_secrets, summarize_command
from src.core.private_index import python_resolver_env
from src.core.toolchain import (
    DISK_BUDGET_ANNOTATION,
    MEMORY_LIMIT_ANNOTATION,
    PYTHON,
    TOOLCHAIN_ANNOTATION,
    DiskProfile,
    MemoryProfile,
    disk_profile,
    image_for,
    memory_profile,
    normalize_toolchain,
    resources_for,
    toolchain_from_annotations,
)
from src.models.container_models import ContainerInfo, ContainerStatus

logger = logging.getLogger(__name__)


class GitCredentialUnavailableError(BaseAPIException):
    """No clone credential, and the repo is not provably public → HTTP 412.

    The provisioner can only clone tokenlessly when coding-github positively
    reports ``visibility="public"``. Any other tokenless outcome — a private repo
    the GitHub App is not installed on, or a clone-token lookup that failed so the
    visibility is unknown — is a *precondition* the caller must fix (install the
    nonagentic GitHub App on the owner from the fr workspace, or pass a
    GitHub token as ``access_token`` on the provision request). It is not a server fault, so it is a typed
    412 with an actionable message rather than the old opaque 500 from
    ``_create_git_token_secret``'s empty-token guard.

    A ``BaseAPIException`` (→ ``HTTPException``), so it deliberately escapes the
    ``except (ApiException, OSError, RuntimeError, ValueError)`` blocks in
    ``provision_from_github`` and ``docker_service.create_container`` and reaches
    the router unchanged — the same mechanism the admission cap's 429 relies on.
    """

    def __init__(
        self,
        repo_url: str,
        owner: str,
        reason: str,
        message: str,
        status_code: int = http_status.HTTP_412_PRECONDITION_FAILED,
    ) -> None:
        super().__init__(
            status_code=status_code,
            message=message,
            error_code="GIT_CREDENTIAL_UNAVAILABLE",
            details={"repo_url": repo_url, "owner": owner, "reason": reason},
        )
        self.reason = reason


def refuse_if_platform_repo_forbidden(
    credential: CloneCredential | None,
    *,
    repo_url: str,
    owner: str,
    tenant_id: str | None,
) -> None:
    """Raise 403 when coding-github REFUSED this tenant a platform-owned repo.

    ``platform_repo_forbidden`` (coding-github #107) is a policy decision, not a
    missing credential. Every clone/push credential ladder in this service ends
    in "else the system PAT" for platform owners, and taking that rung here
    would hand the tenant the very repository coding-github just refused — with
    a strictly MORE powerful, non-expiring credential. So it is checked before
    any ladder runs, and it stops the ladder.
    """
    if credential is None or credential.reason != PLATFORM_REPO_FORBIDDEN:
        return
    message = (
        f"{PLATFORM_REPO_FORBIDDEN} for {owner or '<unknown owner>'}: coding-github refused "
        f"a credential for {repo_url} to tenant {tenant_id or '<none>'} (platform "
        "repositories are internal-tenant only). The system token is not a fallback."
    )
    logger.error(message)
    raise GitCredentialUnavailableError(
        repo_url=repo_url,
        owner=owner,
        reason=PLATFORM_REPO_FORBIDDEN,
        message=message,
        status_code=http_status.HTTP_403_FORBIDDEN,
    )


#: Reason on the 412 raised when a PLATFORM-owned repo gets no App token from
#: coding-github. There is no further rung: the org PAT is never a fallback.
PLATFORM_CREDENTIAL_UNAVAILABLE = "platform_credential_unavailable"
#: Reason on the refusal of a PLATFORM-owned repo request that carries no tenant.
PLATFORM_TENANT_MISSING = "platform_tenant_missing"
#: Stable log key for that refusal (grep/Loki: ``event=platform_tenant_missing_refused``).
PLATFORM_TENANT_MISSING_EVENT = "platform_tenant_missing_refused"


def require_tenant_for_platform_repo(
    tenant_id: str | None,
    *,
    owner: str | None,
    repo_url: str,
    context: str,
) -> str | None:
    """Return ``tenant_id``; REFUSE a platform-owned repo request that has none.

    A platform repository is only ever requested under the caller's own tenant.
    There is no implicit platform identity: a request (or a legacy registry row)
    with no tenant is refused, loudly, rather than being given the platform
    tenant — that substitution would bless a request whose tenancy nobody stated.
    A CUSTOMER owner with no tenant is returned unchanged (``None``); its own
    ladder refuses it with the customer-specific error.
    """
    if tenant_id is not None and str(tenant_id).strip():
        return tenant_id
    if not is_platform_owner(owner):
        return None
    logger.warning(
        "event=%s owner=%s repo_url=%s context=%s: a platform repository request "
        "carries no tenant; refused (no implicit platform tenant)",
        PLATFORM_TENANT_MISSING_EVENT,
        owner,
        repo_url,
        context,
    )
    raise GitCredentialUnavailableError(
        repo_url=repo_url,
        owner=owner or "",
        reason=PLATFORM_TENANT_MISSING,
        message=(
            f"{PLATFORM_TENANT_MISSING}: tenant required for platform repo {repo_url}; "
            "the request carries no tenant and none is ever implied. Send the caller's "
            "tenant, or re-register the legacy container under its tenant."
        ),
    )


def platform_token_or_refuse(
    credential: CloneCredential,
    *,
    repo_url: str,
    owner: str,
    tenant_id: str | None,
) -> str:
    """The App token coding-github issued for a PLATFORM repo, or a typed 412.

    Fail closed on EVERY null token, whatever coding-github's reason (including
    no reason at all): the org PAT (``get_system_token``) is never a fallback for
    a platform repository. It is org-wide and non-expiring, so reaching for it
    would hand the caller strictly more than coding-github was willing to issue.
    """
    if credential.token:
        return credential.token
    reason = credential.reason or credential.error or "none"
    message = (
        f"{PLATFORM_CREDENTIAL_UNAVAILABLE} for {owner or '<unknown owner>'}: coding-github "
        f"issued no App token for {repo_url} to tenant {tenant_id or '<none>'} "
        f"(coding-github reason={reason}). The org PAT is not a fallback for platform "
        "repositories; fix the App installation or the tenant binding."
    )
    logger.error(message)
    raise GitCredentialUnavailableError(
        repo_url=repo_url,
        owner=owner,
        reason=PLATFORM_CREDENTIAL_UNAVAILABLE,
        message=message,
    )


class ContainerDeploymentNotFoundError(RuntimeError):
    """Raised when a container's k8s Deployment is absent (HTTP 404 upstream).

    A ``RuntimeError`` subclass so existing ``except RuntimeError`` sites keep
    working, but a *distinguishable* type so callers that want honest not-found
    semantics (a deprovisioned container is a 404, not a 500 server error) can
    catch it specifically instead of collapsing it into a generic exec failure.
    """


#: The workspace pod's clone init container (see ``_create_deployment``).
CLONE_INIT_CONTAINER = "git-clone"


class WorkspaceCloneFailedError(RuntimeError):
    """The workspace pod's ``git-clone`` init container cannot complete.

    The clone is the authority on whether the workspace's credential can read the
    repository (there is no GitHub API pre-flight: this pod has no egress to
    api.github.com). A BYO token that cannot read the repo, a missing branch or a
    deleted repo all end here, with the pod parked in a back-off state. A
    ``RuntimeError`` subclass so every existing ``except RuntimeError`` keeps
    working; the exec route maps it to a typed 412 ``WORKSPACE_CLONE_FAILED``.
    The message carries the kubelet's waiting reason only, never a credential.
    """

    def __init__(self, deployment_name: str, pod_name: str, failure: str) -> None:
        super().__init__(
            f"Workspace {deployment_name} cannot start: cloning the repository failed "
            f"in pod {pod_name} ({failure}). Check that the token can read the "
            "repository and that the branch exists, then re-provision."
        )
        self.deployment_name = deployment_name
        self.pod_name = pod_name
        self.failure = failure


class WorkspaceNotAvailableError(ValueError):
    """``_create_deployment`` refused to touch an EXISTING Deployment it does not own.

    A ``ValueError`` subclass worded identically to a genuine absence, so every
    existing ``except ValueError`` and the non-disclosing router mapping are
    unchanged. The distinct type exists for one reader: ``provision_from_github``'s
    rollback, which must NOT ``_cleanup_resources`` a Deployment this call never
    created — that would delete another tenant's workspace by name (#160).
    """


@dataclass
class _RollbackScope:
    """Per-call proof that THIS provision may reclaim ``deployment_name``.

    ``owned`` flips True only once ``_create_deployment`` has either created the
    Deployment itself (create RETURNED) or verified the existing one belongs to
    the caller's tenant. Any failure before that — a non-409 create error (403
    admission, 422 validation, 429 throttle, 5xx), a failed ownership read, an
    ownership refusal — leaves it False, so the rollback never deletes a
    same-named Deployment or git-token Secret owned by someone else (#160).

    A per-call object, never an instance attribute: provisions run concurrently
    on one provisioner.
    """

    owned: bool = False


# ---------------------------------------------------------------------------
# Phase C-1 — job lifecycle labels (reaper + admission cap).
#
# The reaper and the admission cap act ONLY on Deployments that carry the
# positive opt-in label ``coding-job=true``. Pre-existing long-lived
# singletons (e.g. ``enterprise-mcp-server-aa``, ``fr-test-verification``)
# were created before this label existed, so they are excluded by
# construction; ``REAP_NEVER_NAMES`` is a belt-and-suspenders allowlist.
# ---------------------------------------------------------------------------
JOB_LABEL_KEY = "coding-job"
JOB_LABEL_VALUE = "true"
TTL_CLASS_LABEL_KEY = "ttl-class"
TTL_CLASS_DEFAULT = "default"
CREATED_AT_LABEL_KEY = "created-at"
# Stamped on every job Deployment so the registry-miss fallback in
# container_manager.get_container can recover the provisioning tenant straight
# from the live cluster (the JSON registry is transient/ephemeral).
TENANT_LABEL_KEY = "tenant-id"
# k8s label values: <=63 chars, [A-Za-z0-9] start/end, [A-Za-z0-9._-] inside.
_K8S_LABEL_VALUE_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]{0,61}[A-Za-z0-9])?$")
DONE_ANNOTATION_KEY = "coding-job/done"
# Durable repo metadata stamped as Deployment ANNOTATIONS (not labels — a repo
# URL and a branch ref both violate the k8s label-value grammar). The
# registry-miss fallback (reconstruct_container_info) recovers these from the
# live Deployment so a container whose transient JSON-registry row was evicted
# is still usable by apply-changes/fork (which require a non-empty repo_url).
REPO_URL_ANNOTATION = "coding-job/repo-url"
BRANCH_ANNOTATION = "coding-job/branch"
SECRET_CONFIG_ANNOTATION = "coding-job/secret-config-id"  # pragma: allowlist secret
# ⛔ The OWNING TENANT, stamped as an annotation IN ADDITION TO ``TENANT_LABEL_KEY``
# — for exactly the reason the three annotations above exist: the value may not be
# expressible as a k8s label.
#
# ``job_labels[TENANT_LABEL_KEY]`` is written only when the tenant satisfies
# ``_K8S_LABEL_VALUE_RE`` ("otherwise omit rather than fail provisioning"), but
# ``effective_tenant_id`` is ``tenant_id or organization_id or sub``
# (``coding_client/auth/models.py:51``) and NOTHING validates it against that
# grammar. A ``sub`` or an organization id with a ``/``, a ``:``, a ``@`` or more
# than 63 chars is a perfectly real owner that CANNOT be labelled — so an OWNED
# workspace could be provisioned carrying no ownership stamp at all, and then:
#
#   * read  — ``reconstruct_container_info`` recovered ``None``, the
#     ``container_tenant is not None`` gate went vacuous, and a foreign tenant got
#     200 + ``container_status: running``; and
#   * write — the 409 ownership gate read the incumbent's claim out of that same
#     label, so a grammar-invalid owner presented as "no claim", the guard did not
#     fire, and ``patch_namespaced_deployment`` re-stamped the workspace to the
#     CALLER and rolled a fresh pod (new emptyDir → re-clone). Cross-tenant
#     takeover that destroys the incumbent's work, returning 200 OK.
#
# Annotations have no grammar restriction, so this stamp is ALWAYS writable. It is
# the primary record of ownership; the label remains the backward-compatible
# fallback and is deliberately still written (see ``_recover_tenant_id``).
TENANT_ANNOTATION = "coding-job/tenant-id"

# ⛔ WHEN the credential in ``<name>-git-token`` was last minted, as epoch seconds.
# Stamped on the Secret at create time and re-stamped on every refresh.
#
# GitHub App installation tokens live ~50 minutes (coding-github mints with
# ``_TOKEN_TTL_SECONDS = 50 * 60``). The Secret is written ONCE at provisioning
# time, but the ``git-clone`` init container is free to run at ANY later moment:
# init containers re-run whenever the pod's sandbox is recreated (a node reboot,
# an eviction, a drain), and ``_idempotent`` re-runs the clone against the fresh
# emptyDir. After the token expires, every one of those replays is a clone with a
# dead credential, which GitHub answers with exactly
# ``Invalid username or token`` — and because it is a credential error the
# container backs off and retries forever, so the workspace can never recover.
#
# A Secret's own ``creation_timestamp`` cannot be the clock here: it does not move
# when the Secret is replaced, so a refreshed Secret would look permanently old and
# be re-minted on every sweep. Hence this explicit stamp.
GIT_TOKEN_REFRESHED_ANNOTATION = "coding-job/token-refreshed-at"


def _positive_int_annotation(annotations: Any, key: str) -> int | None:
    """A positive integer annotation, or None (absent / unreadable)."""
    raw = (annotations or {}).get(key) if isinstance(annotations, dict) else None
    try:
        value = int(raw) if raw is not None else None
    except TypeError, ValueError:
        return None
    return value if value is not None and value > 0 else None


def _body_annotations(body: Any) -> dict[str, Any]:
    """The annotations dict of a body THIS module built, guaranteed present.

    ⚠️ The loose ``dict[str, Any]`` return is load-bearing, not laziness. A
    strategic-merge patch deletes an annotation by sending ``null`` for that key
    (see the toolchain / memory-limit nulls where this is used). The generated
    model types annotations as ``dict[str, str]``, which cannot express ``None``.
    Casting keeps the documented behaviour instead of "fixing" it into
    ``pop()`` — omitting a key from a MERGE patch does not delete it on the
    server, it preserves it, which is the opposite of what those call sites need.
    """
    metadata = body.metadata
    if metadata is None:  # pragma: no cover — this module always builds metadata
        metadata = body.metadata = client.V1ObjectMeta()
    if metadata.annotations is None:
        metadata.annotations = {}
    return cast(dict[str, Any], metadata.annotations)


def _recover_tenant_id(labels: Any, annotations: Any) -> str | None:
    """Recover a Deployment's owning tenant. ANNOTATION first, label as fallback.

    The order is not cosmetic:

    * the **annotation** is stamped for every tenant regardless of grammar, so
      when present it is the complete answer;
    * the **label** is stamped only for grammar-valid tenants, but it is the ONLY
      stamp the pre-existing population carries — measured live in ``coding-jobs``
      on 2026-09-17: **10 of 10 Deployments labelled, 0 of 10 annotated**. Dropping
      the fallback would strand every workspace provisioned before this change,
      so it stays.

    ⛔ A ``None`` return means the owner is **UNKNOWN**, which is NOT the same as
    "ownerless" and must never be treated as it. From the Deployment alone the two
    are indistinguishable: a workspace provisioned with no tenant and a workspace
    whose grammar-invalid owner could not be labelled look identical. Both call
    sites therefore fail **CLOSED** on ``None`` — an unknown owner does not satisfy
    a foreign caller's read and does not let the 409 guard pass. The cost is
    bounded and was measured before it was accepted: 0 of the 10 live Deployments
    resolve to unknown, and every workspace provisioned from here on carries the
    annotation, so the unknown state is reachable only by a pre-change
    grammar-invalid provision — of which there are currently none.
    """
    # ``(x or {})`` rather than an isinstance check, to match the idiom used at
    # every other metadata read in this module.
    annotations = annotations or {}
    labels = labels or {}
    return annotations.get(TENANT_ANNOTATION) or labels.get(TENANT_LABEL_KEY) or None


# ---------------------------------------------------------------------------
# Init-container `git clone` command builder (shell-injection + SHA-as-branch).
#
# The clone runs in a shell inside the init container. The host/path are
# interpolated unquoted into shell (the URL, and the credential helper's
# literals), so they are charset-validated rather than shlex-quoted. The branch is
# a caller-supplied ref: a name is validated + shlex-quoted into `--branch`; a
# commit SHA cannot be used with `git clone --branch` (a SHA passed as a branch
# is itself a cause of a degraded/empty /workspace), so it is cloned then
# `checkout`ed. Mirrors the Docker path's guard (container_lifecycle.py) and
# closes the unquoted `--branch {branch}` hole on the k8s path.
#
# ⛔ THE TOKEN NEVER ENTERS THE URL (auditor P0, #138). It used to:
# `https://x-access-token:$GIT_TOKEN@host/path` — and git records the clone URL
# verbatim as remote.origin.url, so the token sat at rest in /workspace/.git/config
# on the emptyDir the workspace container (uid 1000, customer code) shares, for
# the life of the pod. Now the URL is the canonical tokenless one and the token
# reaches git through a process-scoped (`git -c`), host/path-bound `!`-credential
# helper that reads $GIT_TOKEN at call time (src/core/git_clone_credentials.py).
# After the clone, `scrub_and_verify` re-points origin at the canonical URL and
# FAILS the init container (no wipe: see scrub_and_verify) if any remote URL carries userinfo
# or any .git file outside objects/ contains the token. $GIT_TOKEN is set on the
# init container ONLY; the `analysis` container never receives it.
#
# CREDENTIAL SLOT: the helper answers `username=x-access-token` + the token as
# the PASSWORD. GitHub documents the username-slot form only for classic PATs;
# for GitHub App installation tokens (``ghs_...``) the password-slot form is the
# only documented one, and GitHub ignores the username whenever a valid token is
# in the password slot (classic PAT, fine-grained PAT, and ghs_ alike).
#
# PUBLIC repos (``authenticated=False``) get NO helper at all — there is no
# GIT_TOKEN env var in the init container.
# ---------------------------------------------------------------------------
_GIT_HOST_RE = re.compile(r"^[A-Za-z0-9.\-]+$")
_GIT_PATH_RE = re.compile(r"^[A-Za-z0-9._/~\-]*$")
_GIT_REF_RE = re.compile(r"^[A-Za-z0-9._/\-]+$")
_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def _idempotent(clone_cmd: str, qdest: str) -> str:
    """Make the clone survive being re-run against a dirty destination.

    An init container that fails is retried against the SAME emptyDir, and
    ``git clone`` refuses a non-empty directory — so one failed attempt used to
    poison the volume permanently: every retry died with "destination path
    already exists and is not an empty directory" and the pod sat in
    Init:CrashLoopBackOff forever. That is how a single transient error became
    an unrecoverable workspace, and it is exactly what a bad `chmod` in this
    command did in production.

    The guard is deliberately NOT an unconditional wipe. Init containers re-run
    when a pod's sandbox is recreated (a node reboot, say), and an emptyDir
    survives container restarts — so a blind `rm -rf` there would delete a
    checkout the workspace had already been using, including uncommitted work.
    Testing for ``.git`` distinguishes the two cases: a complete clone is left
    alone, and only a partial or garbage directory is cleared and retried.
    """
    # No echo, no nested quoting: this string is embedded in a pod spec and
    # passed to `sh -c`, and every quote in it is one more thing to get wrong.
    # ⛔ CLEAN UP A PARTIAL CHAIN, or the retry certifies the WRONG REF.
    # The guard tests `.git`, and a multi-step chain can leave `.git` present but the work
    # unfinished: `git clone` succeeds, then `fetch`/`checkout` fails. The next init-container
    # attempt then sees `.git`, SKIPS the whole block, and the `if` exits 0 — so the pod comes up
    # Ready on the DEFAULT BRANCH with no error. That is precisely the vacuous-green this module's
    # PR checkout exists to remove, re-entering through the retry path.
    # On failure the destination is emptied (so a retry genuinely redoes it) and a non-zero exit is
    # forced, so the init container fails LOUDLY instead of certifying the wrong tree.
    return (
        f"if [ ! -d {qdest}/.git ]; then "
        f"find {qdest} -mindepth 1 -delete 2>/dev/null || true; "
        f"{{ {clone_cmd}; }} || {{ find {qdest} -mindepth 1 -delete 2>/dev/null || true; exit 1; }}; "
        f"fi"
    )


# ⛔ EVERY `git -C {dest}` NEEDS `-c safe.directory={dest}` (measured 2026-08-31, production).
#
# The emptyDir mount point is ROOT-owned (the kubelet creates it root:fsGroup) while the workspace
# runs as uid 1000. `git clone` into it succeeds — it writes files as 1000 — but any subsequent
# `git -C {dest} …` checks the ownership of the repo DIRECTORY and refuses:
#
#     Cloning into '/workspace'...
#     fatal: detected dubious ownership in repository at '/workspace'
#
# That crash-looped `git-clone` on verify-1853413860308944792 (Init:CrashLoopBackOff x14), which
# failed `Assert Verify Environment`, errored `AA Pattern - Done Finalize`, and left a card
# finalized with a PR whose next hop never minted.
#
# The single-command BRANCH path is unaffected — it never runs `git -C`. The SHA path always had
# this bug; it was masked because a failed checkout left `.git` present, so the retry skipped the
# block and the pod came up on the DEFAULT BRANCH with exit 0 — the vacuous green the idempotency
# guard now refuses.
#
# ⛔ Do NOT "fix" this with chown/chmod. See the docstring below: three production incidents came
# from touching modes on this mount. `safe.directory` states the exception without mutating
# anything, and is scoped per-command rather than written into a global config that may not be
# writable.
def _build_workspace_clone_cmd(
    hostname: str,
    path: str,
    branch: str | None,
    dest: str = "/workspace",
    *,
    authenticated: bool = True,
    pr_number: int | None = None,
    history_cutoff: bool = False,
) -> str:
    """Build the init-container ``git clone`` shell command, safely.

    ``authenticated=False`` emits a credential-free URL for a repo coding-github
    positively reported as PUBLIC.

    ``history_cutoff=True`` is honoured ONLY on the commit-SHA path: after the
    checkout it deletes every ``refs/remotes`` and ``refs/tags`` ref, removes the
    ``origin`` remote, expires the reflog and prunes unreachable objects, so the
    clone holds no way to reach a commit after that SHA (benchmark isolation).
    It raises ``ValueError`` on a PR or branch-name ref: those paths do not pin a
    SHA, so "later than the checkout" has no meaning there and the request model
    already refuses the combination.

    Raises ``ValueError`` on a host/path/branch that fails its charset guard
    (provisioning fails closed rather than emitting an injectable command).

    **There is deliberately no ``chmod`` here. Do not add one back.** The clone
    carried a trailing ``chmod -R 755`` from the initial commit, with no comment
    saying why, and it caused three production incidents once workspaces stopped
    running as root:

    1. ``chmod`` needs OWNERSHIP, not write permission, and the emptyDir mount
       point is root-owned — EPERM, non-zero exit, broken ``&&`` chain,
       Init:CrashLoopBackOff.
    2. Rewritten as ``find -exec chmod``, it followed symlinks out of the
       checkout — EPERM again on any repo shipping an absolute symlink, plus a
       write primitive pointing outside the workspace.
    3. Applying 755 to every file made a fresh clone report ``M`` on each
       non-executable one, because git tracks the executable bit. Any "is the
       worktree clean" check saw dirty immediately.

    It was never needed. Measured: git already writes 644 for data files and 755
    for files committed executable, the clone and the workspace run as the SAME
    uid so nothing needs group/other bits widened, and a clone with no chmod at
    all is readable, traversable, runnable and reports a clean worktree.
    """
    if not _GIT_HOST_RE.match(hostname or ""):
        raise ValueError(f"Invalid repo hostname: {hostname!r}")
    if not _GIT_PATH_RE.match(path or ""):
        raise ValueError(f"Invalid repo path: {path!r}")

    # Tokenless on BOTH paths; the authenticated one gets the credential helper.
    auth_url = f"https://{hostname}{path}"
    cred = (
        clone_credential_shell_args(hostname, repo_path_from_url_path(path)) + " "
        if authenticated
        else ""
    )
    qdest = shlex.quote(dest)
    scrub = scrub_and_verify(
        qdest, auth_url, token_env=CLONE_TOKEN_ENV if authenticated else None, wipe_on_failure=False
    )
    ref = (branch or "").strip()

    # A PULL REQUEST is the thing under test, so it WINS over `branch`.
    #
    # MEASURED 2026-08-29. `pr_number` and `provision_type` were accepted by
    # ContainerCreateRequest, threaded all the way from n8n's `Provision Verify
    # Container` (provisionType="pull_request" + prNumber), honoured by the Docker
    # path (container_lifecycle / docker_service: `git fetch origin
    # pull/{n}/head`) — and then SILENTLY DROPPED here. This provisioner never
    # read either field, so every k8s workspace cloned `request.branch or "main"`.
    #
    # The consequence was not a crash, which is why it survived: the AA
    # `verification_test` hop provisions `verify-<card>` for an UNMERGED PR and
    # re-runs the tests. Cloning main meant it verified the code the fix was
    # supposed to change. For a bug fix that is unpassable — measured, the probe
    # reported `compute_total(2, 3) == -1` (i.e. `a - b`) against a PR that
    # correctly changed it to `a + b`. For a dependency chore it is worse than a
    # failure: main's suite passes without the bump, so the hop returned GREEN
    # while testing nothing the card had done.
    #
    # `--depth 1` on both the clone and the fetch keeps this as cheap as the
    # branch path. FETCH_HEAD is used rather than a local branch name because the
    # PR head needs no name here — nothing pushes from a verify workspace.
    if pr_number is not None:
        if not isinstance(pr_number, int) or isinstance(pr_number, bool) or pr_number <= 0:
            raise ValueError(f"Invalid pr_number: {pr_number!r} — must be a positive integer")
        if history_cutoff:
            raise ValueError(
                "history_cutoff requires a commit-SHA checkout; a pull-request provision "
                "checks out FETCH_HEAD and cannot be cut off"
            )
        return (
            _idempotent(
                f"git {cred}clone --depth 1 {auth_url} {qdest} && "
                f"git {cred}-c safe.directory={qdest} -C {qdest} "
                f"fetch --depth 1 origin pull/{pr_number}/head && "
                f"git -c safe.directory={qdest} -C {qdest} checkout FETCH_HEAD",
                qdest,
            )
            + f" && {scrub}"
        )

    # A commit SHA can't be a `--branch` target → clone then checkout.
    if ref and _GIT_SHA_RE.match(ref):
        git = f"git -c safe.directory={qdest} -C {qdest}"
        chain = f"git {cred}clone {auth_url} {qdest} && {git} checkout {shlex.quote(ref)}"
        if history_cutoff:
            if len(ref) != 40:
                raise ValueError(
                    "history_cutoff requires branch to be a full 40-hex commit SHA "
                    f"(got {len(ref)} hex chars)"
                )
            # Benchmark isolation: leave the clone with NO ref that can reach a
            # commit after `ref`. The checkout above detaches HEAD at the SHA, so
            # every named ref can go — including refs/heads: `git clone` creates a
            # local branch for the remote's HEAD pointing at the TIP, which is
            # normally later than the benchmark SHA, so deleting only remotes and
            # tags would leave `git log main` showing the future. `xargs -r` keeps
            # an empty ref list from running `update-ref -d` with no argument
            # (busybox xargs in alpine/git supports it). After the refs are gone,
            # the remote is removed (no `git fetch` target), the reflog expired
            # (it is the last thing keeping the old tips reachable) and `gc
            # --prune=now` drops the now-unreachable objects.
            chain += (
                f" && {git} for-each-ref --format='%(refname)' refs/heads refs/remotes refs/tags"
                f" | xargs -r -n1 {git} update-ref -d"
                f" && {git} remote remove origin"
                f" && {git} reflog expire --expire=now --all"
                f" && {git} gc --prune=now --quiet"
            )
        return _idempotent(chain, qdest) + f" && {scrub}"

    if history_cutoff:
        raise ValueError(
            "history_cutoff requires branch to be a full 40-hex commit SHA; "
            f"got a branch name: {ref!r}"
        )

    branch_flag = ""
    if ref:
        if not _GIT_REF_RE.match(ref):
            raise ValueError(
                f"Invalid branch name: {ref!r} — only alphanumeric, dash, "
                "underscore, dot, and slash allowed"
            )
        branch_flag = f"--branch {shlex.quote(ref)} "

    return (
        _idempotent(f"git {cred}clone --depth 1 {branch_flag}{auth_url} {qdest}", qdest)
        + f" && {scrub}"
    )


# ---------------------------------------------------------------------------
# Pod resolution bounded-wait (exec Recreate-gap guard).
#
# The job Deployment uses the Recreate strategy (see _create_deployment), so
# every (re)provision has a brief window with ZERO pods: the old pod is
# Terminating while the new one is still Pending/ContainerCreating. The old
# behaviour failed fast ("No running pod found") and offloaded all retry to
# callers — but the DOING refactoring pipeline execs ~7 commands in sequence
# and any one landing in that window hard-failed the step. Bounded-wait for a
# Running pod here so a just-provisioned container's exec survives the gap.
# Only Running pods are ever returned, so the Recreate stale-pod guard holds.
# Tunable for slow-scheduling clusters.
# ---------------------------------------------------------------------------
# A first provision onto a node with a cold image cache must pull the runtime
# image (~300 MB) before the pod can reach Running. Measured 2026-08-02 on
# worker2: 1m35s for the codegraph-runtime pull alone, ~125s from create to
# Running. The previous 30s budget could not span that, so the very first exec
# against a freshly provisioned container on a cold node failed with a 500 —
# while ``create_container`` had already reported ``status: running``. Terminal
# pull/crash states are detected and bailed on immediately (see
# ``_terminal_pod_failure``), so the larger budget only ever costs wall-clock on
# a pod that is genuinely still coming up.
POD_RESOLVE_TIMEOUT_S = float(os.getenv("POD_RESOLVE_TIMEOUT_S", "180"))

# The unprivileged identity every workspace runs as. Must match the `coder` user
# baked into Dockerfile.runtime — the image and the pod spec have to agree or the
# checkout is unwritable.
#
# These pods execute arbitrary code out of a customer's repository. Until this
# landed they ran as uid 0 with `.spec.securityContext == {}`: no dropped
# capabilities, no seccomp profile, nothing between repository code and the node
# but the container runtime's defaults.
WORKSPACE_UID = 1000
WORKSPACE_GID = 1000

# Where the unprivileged user's caches live. Set explicitly rather than left to
# passwd resolution, which varies by base image and silently becomes "/" for a
# uid with no passwd entry — at which point every tool that writes a cache fails
# in a different way.
WORKSPACE_DIR = "/workspace"


def git_safe_directory_parameters(path: str) -> str:
    """``GIT_CONFIG_PARAMETERS`` value marking ``path`` as a git safe.directory."""
    return f"'safe.directory'='{path}'"


#: Default image for provisioned workspace pods. LOCAL/dev only: a deployment
#: pins PROVISIONER_IMAGE by DIGEST (pods are IfNotPresent, so only a new digest
#: propagates — a re-pushed mutable tag would not).
DEFAULT_PROVISIONER_IMAGE = "ghcr.io/nonagenticai/codegraph-runtime:latest"


def npm_userconfig_startup() -> str:
    """Shell prefix that writes the optional scoped npm registry to ``/tmp/.npmrc``.

    A k8s env var NAME cannot carry npm's scoped-registry key (``@`` and ``:``
    are illegal), so the setting has to be a FILE written at container start.
    Empty when ``CODING_NPM_REGISTRY`` is unset: a public checkout resolves the
    scope from the public registry, and npm treats a missing userconfig as empty.
    ``strict-ssl=false`` is emitted only alongside a configured mirror, because
    a self-signed internal registry is the only reason to need it.
    """
    registry = os.environ.get("CODING_NPM_REGISTRY", "")
    if not registry:
        return ""
    scoped = shlex.quote(f"@nonagentic:registry={registry}")
    return f"printf '%s\\n' {scoped} 'strict-ssl=false' > /tmp/.npmrc 2>/dev/null || true; "


WORKSPACE_HOME = "/home/coder"

# Hard ceiling on what one exec may return, per stream. Internal server-authored
# commands (repository tree walks, `cat` of a single file) run against this; the
# public /exec route passes a smaller, configurable budget.
#
# Before this existed the reader accumulated `stdout_data += resp.read_stdout()`
# with no limit, so a command that printed a big — or binary — file was buffered
# whole in the service process and then serialised into a JSON response. There
# was nothing between a customer's `cat /dev/urandom` and the pod's memory limit.
EXEC_HARD_OUTPUT_BYTES = 16 * 1024 * 1024
POD_RESOLVE_INTERVAL_S = float(os.getenv("POD_RESOLVE_INTERVAL_S", "1"))

# Selector for the opt-in job Deployments the reaper/cap manage.
JOB_LABEL_SELECTOR = (
    f"app=codegraph-repo,managed-by=code-graph-helper,{JOB_LABEL_KEY}={JOB_LABEL_VALUE}"
)
# Wider selector (labelled + unlabelled legacy) used only in legacy reap mode.
ALL_MANAGED_SELECTOR = "app=codegraph-repo,managed-by=code-graph-helper"

# Names the reaper must never delete regardless of age/labels.
REAP_NEVER_NAMES: frozenset[str] = frozenset({"enterprise-mcp-server-aa", "fr-test-verification"})


class _BoundedStream:
    """Accumulates up to ``limit`` bytes of one output stream and counts the rest.

    Discards past the budget rather than buffering everything and trimming at the
    end: the point is to bound the service's memory, and a trim-afterwards
    implementation has already paid the cost it was meant to prevent.

    Measures in BYTES, not characters. The stream is decoded text, but the limit
    protects byte-oriented resources — process memory and the JSON response body
    — and for multi-byte content a character count understates both by up to 4x.
    """

    __slots__ = ("_chunks", "_limit", "_used", "bytes_omitted")

    def __init__(self, limit: int) -> None:
        self._chunks: list[str] = []
        self._limit = max(0, limit)
        self._used = 0
        self.bytes_omitted = 0

    def write(self, text: str) -> None:
        if not text:
            return
        encoded = text.encode("utf-8", "replace")
        remaining = self._limit - self._used
        if remaining <= 0:
            self.bytes_omitted += len(encoded)
            return
        if len(encoded) <= remaining:
            self._chunks.append(text)
            self._used += len(encoded)
            return
        # Partial chunk. Decode with errors="ignore" so a multi-byte character
        # straddling the boundary is dropped rather than emitted as a replacement
        # char, then count what actually survived — slicing bytes can keep fewer
        # than `remaining`, and the omitted count has to reflect that.
        kept = encoded[:remaining].decode("utf-8", "ignore")
        kept_bytes = len(kept.encode("utf-8", "replace"))
        if kept:
            self._chunks.append(kept)
            self._used += kept_bytes
        self.bytes_omitted += len(encoded) - kept_bytes

    @property
    def truncated(self) -> bool:
        return self.bytes_omitted > 0

    def value(self) -> str:
        return "".join(self._chunks)


class K8sContainerProvisioner:
    """Kubernetes-based container provisioner.

    Creates a Deployment per provisioned repository. The pod is a thin
    volume-holder: git-clone init container populates /workspace, then a
    busybox sleep container keeps the volume mounted so coding-analysis
    can ``kubectl exec`` into it on demand.
    """

    def __init__(
        self,
        namespace: str | None = None,
        in_cluster: bool | None = None,
    ):
        """
        Initialize the Kubernetes container provisioner.

        Args:
            namespace: Kubernetes namespace for provisioned containers
            in_cluster: Whether running inside a cluster (auto-detect if None)
        """
        self.namespace = namespace or os.environ.get("PROVISIONER_NAMESPACE", "code-graph-helper")
        # The per-repo pod is a thin volume-holder: nothing listens on any
        # port. coding-analysis runs scan scripts inside it via kubectl exec
        # (repo_scanner_modules writes a python3 -c dependency/import script
        # into /tmp and executes it), so the image needs /bin/sh + python3.
        # python:3.14-slim provides both on a ~45 MB base. Override via env
        # if a different toolchain is required (e.g. for non-Python repos).
        # codegraph-runtime adds `uv` on top of python:3.14-slim so cross-major
        # dependency upgrades (`uv add pkg==X.0.0`) can run inside the pod.
        # Without uv on PATH the strategy in coding-containers/src/services/
        # package_upgrade/strategies.py exits 127. Override via env if a
        # different toolchain is required (e.g. for non-Python repos).
        #
        # PROPAGATION: this floating :latest default is for LOCAL/dev only.
        # A deployment pins PROVISIONER_IMAGE by DIGEST
        # (codegraph-runtime:sha-<gitsha>@sha256:<digest>) so that the
        # provisioned pods' imagePullPolicy: IfNotPresent (below) actually
        # propagates a rebuilt image — a re-pushed mutable :latest would NOT.
        # The image-only CI workflow prints the pin line on every main build.
        self.image = os.environ.get("PROVISIONER_IMAGE", DEFAULT_PROVISIONER_IMAGE)
        self.node_arch = os.environ.get("PROVISIONER_NODE_ARCH")

        # Determine if running in-cluster
        if in_cluster is None:
            in_cluster = os.environ.get("KUBERNETES_IN_CLUSTER", "false").lower() == "true"

        try:
            if in_cluster:
                # Override K8s API host if configured — ztunnel intercepts the default
                # ClusterIP and breaks TLS since kube-system is not in the ambient mesh
                k8s_host = os.environ.get("K8S_API_HOST")
                k8s_port = os.environ.get("K8S_API_PORT", "6443")
                if k8s_host:
                    logger.info(
                        "Loading in-cluster config with explicit API host: %s:%s",
                        k8s_host,
                        k8s_port,
                    )
                    os.environ["KUBERNETES_SERVICE_HOST"] = k8s_host
                    os.environ["KUBERNETES_SERVICE_PORT"] = k8s_port
                else:
                    logger.info("Loading in-cluster Kubernetes configuration")
                config.load_incluster_config()
            else:
                logger.info("Loading local Kubernetes configuration")
                config.load_kube_config()
        except (ApiException, OSError, RuntimeError, ConfigException) as e:
            logger.error(f"Failed to load Kubernetes config: {e}")
            raise

        # Workaround for kubernetes==36.0.0: InClusterConfigLoader._set_config
        # writes the SA token to api_key['authorization'] but Configuration.
        # auth_settings() only reads api_key['BearerToken']. Without this
        # bridge, every request goes anonymously and the apiserver returns 401.
        # Remove when the lib pin bumps past the upstream fix.
        _cfg = client.Configuration.get_default_copy()
        if "authorization" in _cfg.api_key and "BearerToken" not in _cfg.api_key:
            _cfg.api_key["BearerToken"] = _cfg.api_key["authorization"]
            client.Configuration.set_default(_cfg)

        # Initialize API clients
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        self._available = True
        logger.info(f"K8sContainerProvisioner initialized for namespace: {self.namespace}")

    def is_available(self) -> bool:
        """Check if the K8s provisioner is operational."""
        return self._available

    async def provision_container(
        self,
        container_name: str,
        repo_url: str,
        branch: str | None = None,
        access_token: str | None = None,
        tenant_id: str | None = None,
        secret_config_id: str | None = None,
        pr_number: int | None = None,
        provision_type: str = "branch",
        history_cutoff: bool = False,
        toolchain: str = PYTHON,
        expected_repos: int | None = None,
        expected_services: int | None = None,
        service_memory_mi: int | None = None,
        expected_dependencies: list[Literal["redis", "postgres"]] | None = None,
    ) -> ContainerInfo:
        """Provision a container from a GitHub repo.

        This is the public API called by docker_service.py.
        Delegates to provision_from_github with a GitHubRepoRequest.

        ``secret_config_id`` is threaded through so the provisioned container
        retains the opaque secrets-service reference — downstream fork-push and
        upstream-PR re-resolve the per-tenant GIT_TOKEN by it (the in-memory
        dual-token store does not survive cross-service).
        """
        container_name = self._sanitize_k8s_name(container_name)
        # Narrowed to the model's Literal explicitly: callers hand us a plain `str`
        # (docker_service reads it off the request with getattr), and an unrecognised value must
        # fall back to "branch" rather than raise a validation error at provision time.
        _ptype: Literal["branch", "pull_request"] = (
            "pull_request" if provision_type == "pull_request" else "branch"
        )
        # ⛔ pr_number / provision_type MUST be carried here. This method REBUILDS the request
        # from its arguments, so any field it does not accept is DROPPED — the caller's
        # ContainerCreateRequest (which parse_pr_url populates from a PR url) never reaches
        # provision_from_github. That is how the PR checkout was lost on the live route while the
        # gate deeper down looked correct: the fix was in a frame production never reaches.
        request = GitHubRepoRequest(
            repo_url=repo_url,
            branch=branch or "main",
            container_name=container_name,
            access_token=access_token,
            secret_config_id=secret_config_id,
            tenant_id=tenant_id,
            pr_number=pr_number,
            provision_type=_ptype,
            # Same DROP hazard as pr_number above: this rebuild is the only way
            # the flag reaches provision_from_github.
            history_cutoff=history_cutoff,
            # Same DROP hazard again: a python+rust request rebuilt without it
            # would silently provision a python workspace.
            toolchain=normalize_toolchain(toolchain),
            # And again for the disk hint (py2rust gap #4-lite).
            expected_repos=expected_repos,
            # And the memory hints (gap #4 full).
            expected_services=expected_services,
            service_memory_mi=service_memory_mi,
            expected_dependencies=expected_dependencies,
        )
        return await self.provision_from_github(request)

    @staticmethod
    def _sanitize_k8s_name(name: str) -> str:
        """Sanitize a name to be DNS-compatible for Kubernetes resources.

        - Lowercase
        - Replace underscores and non-alphanumeric chars with hyphens
        - Collapse consecutive hyphens
        - Truncate to 63 characters
        - Strip leading/trailing hyphens
        """
        name = name.lower()
        name = re.sub(r"[^a-z0-9-]", "-", name)
        name = re.sub(r"-{2,}", "-", name)
        name = name[:63]
        name = name.strip("-")
        # Ensure non-empty
        if not name:
            name = "codegraph"
        return name

    def _extract_repo_name(self, repo_url: str) -> str:
        """Extract repository name from URL."""
        parsed_url = urlparse(str(repo_url))
        path_parts = parsed_url.path.strip("/").split("/")
        repo_name = path_parts[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        return repo_name.lower().replace("_", "-")

    @staticmethod
    def _owner_repo_from_url(repo_url: str) -> tuple[str, str]:
        """Parse (owner, repo) from a GitHub URL; ("", "") if not parseable.

        Mirrors the provision-time parse but guards an empty/short path so the
        registry-miss reconstruct path never raises on a Deployment that lacks
        the repo-url annotation.
        """
        if not repo_url:
            return "", ""
        path_parts = urlparse(str(repo_url)).path.strip("/").split("/")
        if len(path_parts) < 2 or not path_parts[0] or not path_parts[1]:
            return "", ""
        return path_parts[0], path_parts[1].replace(".git", "")

    async def provision_from_github(self, request: GitHubRepoRequest) -> ContainerInfo:
        """
        Provision a Kubernetes Deployment for a GitHub repository.

        Args:
            request: Container provision request containing GitHub details

        Returns:
            ContainerInfo containing provision results and metadata
        """
        # ⛔ Rollback runs only if THIS call created the Deployment or proved it
        # owns the existing one (``scope.owned``, set inside
        # ``_create_deployment``). ``deployment_name`` is caller-chosen and
        # ``POST /`` does not gate on it, so any failure before that proof — the
        # BYO-PAT store raising, a non-409 create error, a failed ownership read,
        # the ownership refusal — means a same-named Deployment may belong to
        # someone else. Cleaning up by name there deleted another tenant's
        # workspace and Secret (#160).
        # Memory sizing first, before anything is touched: raises MemoryBudgetError
        # (409 at the create route) when the hinted set cannot fit one pod.
        memory = memory_profile(
            normalize_toolchain(getattr(request, "toolchain", None)),
            getattr(request, "expected_services", None),
            getattr(request, "service_memory_mi", None),
            getattr(request, "expected_dependencies", None),
        )
        scope = _RollbackScope()
        try:
            logger.info("Starting K8s provisioning for %s", request.repo_url)

            # Refuse a non-canonical remote before any credential is touched.
            # The owner that picks the credential and the URL the init container
            # clones both come from this one pair — never a raw first-segment
            # parse, which git's own dot-segment / %2e%2e resolution defeats.
            canonical = canonical_github_repo(request.repo_url)
            if canonical is None:
                raise ValueError(
                    "repo_url is not a canonical GitHub repository URL "
                    f"(https://github.com/<owner>/<repo>): {request.repo_url!r}"
                )
            original_repo_owner, original_repo_name = canonical
            canonical_url = canonical_github_url(original_repo_owner, original_repo_name)

            # Get or generate user_id
            user_id = request.user_id or f"anonymous-{uuid7().hex[:8]}"

            # Get token manager
            token_manager = get_github_token_manager()
            system_owner = token_manager.get_system_owner()

            # Honor caller-supplied container_name (n8n cells + smoke harnesses
            # rely on deterministic names for cross-step lookups); fall back to
            # codegraph-<repo>-<6hex> only when absent.
            requested_name = request.container_name
            if requested_name:
                deployment_name = self._sanitize_k8s_name(requested_name)
            else:
                repo_name_clean = self._extract_repo_name(request.repo_url)
                unique_id = uuid7().hex[:6]
                deployment_name = self._sanitize_k8s_name(
                    f"codegraph-{repo_name_clean}-{unique_id}"
                )

            # Store an inline BYO PAT in the secrets service; an id supplied on the
            # request wins (the row already exists — see the helper's docstring).
            # A raise fails the provision, which is the point: this used to POST the
            # PAT at coding-github's webhook receiver, 4xx, and be swallowed.
            secret_config_id = await token_manager.resolve_provision_secret_config_id(
                access_token=request.access_token,
                existing_secret_config_id=getattr(request, "secret_config_id", None),
                user_id=user_id,
                container_id=deployment_name,
                tenant_id=getattr(request, "tenant_id", None),
            )

            # ----------------------------------------------------------------
            # Clone-credential ladder:
            #   1. request.access_token — BYO PAT (already resolved upstream
            #      from secrets-service when secret_config_id was passed). It has
            #      the tenant's actual repo access, so it always wins.
            #   2. coding-github clone-token — a short-lived GitHub App
            #      installation token (ghs_...). This is the rung that was
            #      missing: a project created without a PAT had NOTHING here and
            #      hard-500'd at the git-token Secret, public repo or not.
            #   3. None — no credential. Legal ONLY for a CUSTOMER repo
            #      coding-github positively called PUBLIC; anything else is a
            #      typed 412 below. A PLATFORM repo with no App token is a 412
            #      too (``platform_token_or_refuse``): the org PAT
            #      (GITHUB_TOKEN/GITHUB_SYSTEM_TOKEN) is never a clone credential.
            # The App is only consulted when there is no BYO PAT: an extra token
            # mint per provision buys nothing when we already hold a better one.
            # ----------------------------------------------------------------
            byo_token = (
                request.access_token.strip()
                if request.access_token and request.access_token.strip()
                else None
            )
            clone_token: str | None = byo_token
            credential: CloneCredential | None = None
            if clone_token is None:
                repo_owner = original_repo_owner
                # The CLONE-CREDENTIAL tenant is the request's own. A platform repo
                # asked for with NO tenant (e.g. ``provision_from_github``'s body
                # field left empty) is REFUSED here — never given an implicit
                # platform tenant. A customer owner keeps ``None`` and is refused
                # by its own ladder below.
                clone_tenant_id = require_tenant_for_platform_repo(
                    getattr(request, "tenant_id", None),
                    owner=repo_owner,
                    repo_url=canonical_url,
                    context="k8s_provision",
                )
                credential = await token_manager.get_app_clone_credential(
                    repo_url=canonical_url,
                    tenant_id=clone_tenant_id,
                )
                # Before any rung: a refusal is never answered with the PAT.
                refuse_if_platform_repo_forbidden(
                    credential,
                    repo_url=canonical_url,
                    owner=repo_owner,
                    tenant_id=clone_tenant_id,
                )
                if is_platform_owner(repo_owner):
                    clone_token = platform_token_or_refuse(
                        credential,
                        repo_url=canonical_url,
                        owner=repo_owner,
                        tenant_id=clone_tenant_id,
                    )
                else:
                    # Customer repo: the system PAT is NEVER a clone credential
                    # here (tenant-bound invariant). Only a tenant-bound App token
                    # may clone it; a positively-PUBLIC repo may still be cloned
                    # with no credential at all, since nothing is issued.
                    clone_token = credential.token or None
                    if clone_token is None and not credential.is_public:
                        message = (
                            f"{APP_NOT_INSTALLED_ERROR} for {repo_owner or '<unknown owner>'}: "
                            f"no BYO token and no GitHub App installation bound to tenant "
                            f"{getattr(request, 'tenant_id', None) or '<none>'} covers "
                            f"{request.repo_url} (coding-github reason="
                            f"{credential.reason or credential.error or '-'}). Install the "
                            f"nonagentic GitHub App on {repo_owner or '<unknown owner>'} from "
                            "the fr workspace, or pass a GitHub token as access_token on the "
                            "provision request."
                        )
                        logger.error("%s (deployment=%s)", message, deployment_name)
                        raise GitCredentialUnavailableError(
                            repo_url=str(request.repo_url),
                            owner=repo_owner,
                            reason=APP_NOT_INSTALLED_ERROR,
                            message=message,
                        )

            # Honour the PR the caller asked for. `provision_type` gates it so a
            # stray pr_number on a plain branch provision cannot silently redirect
            # the checkout — the two fields have to agree, exactly as the Docker
            # path requires.
            _pr = (
                getattr(request, "pr_number", None)
                if getattr(request, "provision_type", "branch") == "pull_request"
                else None
            )
            # Benchmark isolation flag. The model validator already requires a
            # full-SHA `branch` and no PR; the clone builder re-checks and raises
            # ValueError (→ the except below cleans up and re-raises) so the flag
            # can never be silently dropped into a plain clone.
            history_cutoff = bool(getattr(request, "history_cutoff", False))
            toolchain = normalize_toolchain(getattr(request, "toolchain", None))
            disk = disk_profile(toolchain, getattr(request, "expected_repos", None))
            self._create_deployment(
                name=deployment_name,
                repo_url=canonical_url,
                branch=request.branch or "main",
                github_token=clone_token,
                repo_visibility=credential.visibility if credential and credential.ok else None,
                tenant_id=getattr(request, "tenant_id", None),
                secret_config_id=secret_config_id,
                pr_number=_pr,
                history_cutoff=history_cutoff,
                toolchain=toolchain,
                disk=disk,
                memory=memory,
                scope=scope,
            )

            # internal_endpoint is a routing sigil for container_exec.py, which
            # dispatches to the K8s exec path when ``.svc.cluster.local`` is
            # present in it. No Service exists behind this URL.
            service_endpoint = f"http://{deployment_name}.{self.namespace}.svc.cluster.local:8687"
            public_endpoint = (
                f"{get_settings().main_api_endpoint.rstrip('/')}/api/workspaces/{deployment_name}"
            )

            # Build ContainerInfo response
            container_info = ContainerInfo(
                container_name=deployment_name,
                api_endpoint=public_endpoint,
                internal_endpoint=service_endpoint,
                public_endpoint=public_endpoint,
                code_graph_id=deployment_name,
                codebase_path="/workspace",
                repo_url=request.repo_url,
                branch=request.branch or "main",
                status=ContainerStatus.RUNNING,
                description=request.description or f"K8s Deployment for {request.repo_url}",
                github_access_token=None,  # Don't store plaintext
                # Legacy alias of secret_config_id — see github_container_provisioner.
                user_token_id=secret_config_id,
                user_id=user_id,
                original_repo_owner=original_repo_owner,
                original_repo_name=original_repo_name,
                system_fork_owner=system_owner,
                # Persist the opaque secrets-service reference (NOT the token) so
                # downstream fork-push + upstream-PR can re-resolve the per-tenant
                # GIT_TOKEN from secrets-service. The in-memory dual-token store does
                # not survive cross-service, so secrets-service is the source of truth.
                secret_config_id=secret_config_id,
                tenant_id=getattr(request, "tenant_id", None),
                # Disclose the cutoff on the response: the clone command above
                # was built with it, so a consumer can tell an isolated
                # workspace from an ordinary SHA checkout.
                history_cutoff_applied=history_cutoff,
                toolchain=toolchain,
                disk_budget_gi=disk.repo_data_gi if disk else None,
                memory_limit_mi=memory.limit_mi if memory else None,
            )

            # Store in registry for container management consistency. Use the
            # atomic, locked single-key upsert so a concurrent whole-dict save
            # (eager graph-init/build) can never drop this just-created entry.
            try:
                ContainerMetadataRegistry().upsert(deployment_name, container_info)
                logger.info("Container %s stored in registry", deployment_name)
            except (OSError, RuntimeError) as reg_error:
                logger.warning("Failed to store container in registry: %s", reg_error)
                # Non-fatal - container is still provisioned

            logger.info(
                "K8s provisioning completed: %s (user=%s, endpoint=%s)",
                deployment_name,
                user_id,
                service_endpoint,
            )
            return container_info

        except (ApiException, OSError, RuntimeError, ValueError) as e:
            logger.error("Failed to provision K8s resources: %s", str(e))
            # Clean up only what THIS call created or proved it owns. An
            # ownership refusal (WorkspaceNotAvailableError) is raised before
            # ``owned`` is ever set, so it needs no special case here.
            if scope.owned:
                try:
                    self._cleanup_resources(deployment_name)
                except (ApiException, OSError) as cleanup_error:
                    logger.error("Failed to cleanup after error: %s", cleanup_error)
            raise

    @staticmethod
    def _owner_reference(deployment: Any, name: str) -> Any | None:
        """Build a ``V1OwnerReference`` pointing at a LIVE Deployment object.

        The uid MUST come from the object the apiserver actually returned — a
        constructed/guessed uid is worse than none: a uid that never existed
        makes the garbage collector delete the dependent immediately, and a
        stale uid (from a previous incarnation of the same name) does the same.
        So this returns ``None`` — a safe degrade to today's behaviour — unless
        the live object carries a real string uid.

        ``block_owner_deletion`` is deliberately False: True is only honoured
        when the caller holds ``update`` on the owner's ``finalizers``
        subresource under the OwnerReferencesPermissionEnforcement admission
        plugin, and the provisioner's Role does not grant that. False needs no
        extra RBAC and still gives full cascading deletion.
        """
        uid = getattr(getattr(deployment, "metadata", None), "uid", None)
        if not isinstance(uid, str) or not uid:
            logger.warning(
                "No live uid for Deployment %s; git-token Secret will be created WITHOUT an "
                "ownerReference and can only be reclaimed by the explicit teardown path.",
                name,
            )
            return None
        return client.V1OwnerReference(
            api_version="apps/v1",
            kind="Deployment",
            name=name,
            uid=uid,
            controller=True,
            block_owner_deletion=False,
        )

    def _create_git_token_secret(
        self, name: str, github_token: str, owner: Any | None = None
    ) -> str:
        """Create a Kubernetes Secret containing the GitHub token.

        ``owner`` is the ownerReference of the Deployment this credential belongs
        to. It is stamped at CREATE time (and re-stamped on the 409/replace path,
        whose body would otherwise wipe an existing one) so the Secret is never
        visible to the apiserver in an unowned state — a crash between the two
        writes is exactly the abandonment case that leaked 60 Secrets into
        ``coding-jobs``.

        Returns:
            The name of the created Secret.
        """
        secret_name = f"{name}-git-token"
        if not github_token:
            # Never create a Secret with GIT_TOKEN=None: the k8s API rejects it
            # and (on kubernetes==36.0.0) the error surfaces as a masked
            # `.decode` AttributeError. Fail with a clear, actionable message.
            raise RuntimeError(
                f"Refusing to create git-token Secret {secret_name!r} with an empty token "
                "(secret_config_id did not resolve and no system token is set)."
            )
        secret = client.V1Secret(
            api_version="v1",
            kind="Secret",
            metadata=client.V1ObjectMeta(
                name=secret_name,
                namespace=self.namespace,
                labels={
                    "app": "codegraph-repo",
                    "repo": name,
                    "managed-by": "code-graph-helper",
                },
                annotations={GIT_TOKEN_REFRESHED_ANNOTATION: str(int(time.time()))},
                owner_references=[owner] if owner is not None else None,
            ),
            type="Opaque",
            string_data={"GIT_TOKEN": github_token},
        )

        # Bounded retry. Threaded exec now uses an isolated client (see
        # exec_in_pod), so this loop-side create no longer races it; but if any
        # torn response still surfaces as ApiException(body=None) -- which
        # kubernetes==36.0.0's own handler crashes on (`e.body.decode()` on None
        # -> AttributeError) -- or a 5xx, treat it as transient and retry.
        # Genuine 4xx (401/403/422) still raise immediately.
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                self.core_v1.create_namespaced_secret(namespace=self.namespace, body=secret)
                logger.info("Created Secret: %s", secret_name)
                return secret_name
            except ApiException as e:
                if e.status == 409:
                    self.core_v1.replace_namespaced_secret(
                        name=secret_name, namespace=self.namespace, body=secret
                    )
                    logger.info("Replaced Secret: %s", secret_name)
                    return secret_name
                if e.status and e.status < 500:
                    raise  # genuine client error -- retrying won't help
                last_exc = e
            except AttributeError as e:  # masked ApiException(body=None)
                logger.warning(
                    "Secret create hit masked ApiException for %s (attempt %d/3): %s",
                    secret_name,
                    attempt + 1,
                    e,
                )
                last_exc = e
            time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Secret create failed for {secret_name!r} after 3 attempts: {last_exc}")

    def _create_deployment(
        self,
        name: str,
        repo_url: str,
        branch: str,
        github_token: str | None,
        repo_visibility: str | None = None,
        tenant_id: str | None = None,
        secret_config_id: str | None = None,
        pr_number: int | None = None,
        history_cutoff: bool = False,
        toolchain: str = PYTHON,
        disk: DiskProfile | None = None,
        memory: MemoryProfile | None = None,
        *,
        scope: _RollbackScope | None = None,
    ) -> None:
        """Create a Kubernetes Deployment for the repository.

        ``scope.owned`` (when a scope is passed) is set True only once this call
        has created the Deployment (a create RETURNED) or verified the caller's
        tenant owns the existing one — never before. See ``_RollbackScope``.

        Two clone modes:

        * ``github_token`` present → authenticated clone. A git-token Secret is
          created and mounted into the git-clone init container as ``GIT_TOKEN``
          (the main workspace container never sees it; the clone takes seconds, so
          a ~1h App token is fine).
        * ``github_token`` absent AND ``repo_visibility == "public"`` → the real
          public-repo path: NO Secret, NO ``GIT_TOKEN`` env var, and an
          unauthenticated clone URL.

        Absent token + anything else (private, or an unknown visibility because
        the clone-token lookup failed) raises ``GitCredentialUnavailableError``
        (412) — an actionable precondition, not the old opaque 500.
        """
        # Host and path of the clone URL come from the canonical pair only.
        canonical = canonical_github_repo(repo_url)
        if canonical is None:
            raise ValueError(f"repo_url is not a canonical GitHub repository URL: {repo_url!r}")
        owner, _repo = canonical
        parsed_url = urlparse(canonical_github_url(*canonical))
        hostname = parsed_url.hostname or ""
        path = parsed_url.path or ""

        authenticated = bool(github_token)
        if not authenticated and repo_visibility != "public":
            if repo_visibility == "private":
                message = (
                    f"Cannot clone {repo_url}: repository is private, no credential "
                    f"available; the GitHub App is not installed on '{owner}' and no "
                    "access token was supplied. Install the nonagentic GitHub App on "
                    f"'{owner}' from the fr workspace, or pass a GitHub token as "
                    "access_token on the provision request."
                )
                reason = "private_repo_no_credential"
            else:
                # Do NOT name a single cause here. This text used to send the
                # operator straight at the X-Internal-Token shared secret; on
                # 2026-08-15 that secret was byte-identical on both sides and the
                # real refusal was a tenant assertion, three hops up. Point at the
                # log line that carries coding-github's OWN answer instead of
                # guessing on its behalf.
                message = (
                    f"Cannot clone {repo_url}: no credential available and the "
                    "repository's visibility could not be determined (the coding-github "
                    "clone-token lookup failed). Refusing to attempt an unauthenticated "
                    "clone of a repository that may be private. The preceding "
                    "'coding-github rejected the clone-token call' log line carries the "
                    "server's own reason and status — read it before changing anything; "
                    "a 403 there is as often an authorization refusal as a bad "
                    "CODING_GITHUB_URL or X-Internal-Token. Passing a GitHub token as "
                    "access_token on the provision request also unblocks it, at the cost "
                    "of skipping the App path."
                )
                reason = "visibility_unknown"
            logger.error("%s (deployment=%s)", message, name)
            raise GitCredentialUnavailableError(
                repo_url=str(repo_url), owner=owner, reason=reason, message=message
            )

        # Authenticated clones get a Secret holding the token; PUBLIC tokenless
        # clones get neither the Secret nor the env var (the Secret guard in
        # _create_git_token_secret is deliberately NOT relaxed — it still refuses
        # an empty token, which remains a correct guard on THIS path).
        #
        # ORDERING: the Secret is created AFTER the Deployment (see the tail of
        # this method), not here. Only the NAME is needed to build the env var,
        # and it is deterministic. Creating the owner first is what lets the
        # ownerReference be stamped at Secret-create time from the live uid, so
        # there is no window in which an un-owned credential exists in etcd.
        env_vars: list[Any] = []
        if authenticated:
            secret_name = f"{name}-git-token"
            env_vars.append(
                client.V1EnvVar(
                    name="GIT_TOKEN",
                    value_from=client.V1EnvVarSource(
                        secret_key_ref=client.V1SecretKeySelector(
                            name=secret_name,
                            key="GIT_TOKEN",
                        ),
                    ),
                )
            )
        else:
            logger.info(
                "Provisioning %s with an UNAUTHENTICATED clone of %s "
                "(coding-github reports the repository is public; no git-token Secret).",
                name,
                repo_url,
            )

        # The clone URL uses the $GIT_TOKEN env var instead of embedding the token
        # (and omits credentials entirely on the public path). Build the command
        # through the validated helper so a branch/SHA/host can't inject shell, and
        # a commit SHA is checked out rather than passed to `--branch` (which would
        # fail and leave /workspace degraded).
        clone_cmd = _build_workspace_clone_cmd(
            hostname,
            path,
            branch,
            authenticated=authenticated,
            pr_number=pr_number,
            history_cutoff=history_cutoff,
        )

        # Stamp the opt-in lifecycle labels so the reaper/admission cap can
        # find this Deployment by selector. created-at is epoch seconds (label
        # values must match [A-Za-z0-9._-]{0,63}); the reaper uses the freshest
        # of the server-set creation_timestamp and this label, so re-stamping it
        # on every (re)provision refreshes the TTL lease — important now that the
        # 409 path reconciles in place (patch) and no longer resets
        # creation_timestamp via delete-recreate.
        provisioned_at = str(int(time.time()))
        job_labels = {
            "app": "codegraph-repo",
            "repo": name,
            "managed-by": "code-graph-helper",
            JOB_LABEL_KEY: JOB_LABEL_VALUE,
            TTL_CLASS_LABEL_KEY: TTL_CLASS_DEFAULT,
            CREATED_AT_LABEL_KEY: provisioned_at,
        }
        # Stamp the provisioning tenant so a registry-miss fallback recovers it
        # from the cluster. Only when it satisfies the k8s label-value grammar
        # (UUID tenant ids do); otherwise omit rather than fail provisioning.
        if tenant_id and _K8S_LABEL_VALUE_RE.match(tenant_id):
            job_labels[TENANT_LABEL_KEY] = tenant_id

        # Stamp the durable repo metadata as annotations (label-grammar-unsafe
        # values: URL has '/' and ':', branch refs can have '/'). These are the
        # source of truth the registry-miss fallback reads back in
        # reconstruct_container_info so a reconstructed container still carries a
        # real repo_url/branch — required by apply-changes/fork.
        deployment_annotations: dict[str, str] = {
            REPO_URL_ANNOTATION: str(repo_url),
            BRANCH_ANNOTATION: branch,
        }
        # Stamp the owning tenant here TOO, unconditionally on truthiness — no
        # grammar test, because annotations have no grammar. This is what makes
        # the ownership record total: the label write three lines above silently
        # drops a grammar-invalid owner, and an unstamped owned workspace is a
        # read fail-open and a write takeover (see TENANT_ANNOTATION). The label
        # write is deliberately NOT removed — it is what the existing population
        # resolves by.
        if tenant_id:
            deployment_annotations[TENANT_ANNOTATION] = tenant_id
        if secret_config_id:
            deployment_annotations[SECRET_CONFIG_ANNOTATION] = secret_config_id

        # Toolchain (py2rust A1). ⛔ python adds NOTHING here — no annotation, the
        # same image, the same resources — so a python Deployment stays
        # byte-identical to the pre-toolchain spec (G1; pinned by
        # test_toolchain_python_spec_snapshot.py). Only python+rust is stamped,
        # which is also what the exec policy and reconstruct_container_info read.
        toolchain = normalize_toolchain(toolchain)
        main_image = image_for(toolchain, self.image)
        if toolchain != PYTHON:
            deployment_annotations[TOOLCHAIN_ANNOTATION] = toolchain
        main_resources = resources_for(toolchain) or {
            "requests": {"memory": "256Mi", "cpu": "10m"},
            "limits": {"memory": "2Gi", "cpu": "500m"},
        }
        # Disk (py2rust gap #4-lite). ⛔ ``disk`` is None for a python workspace
        # without an ``expected_repos`` hint, and then NOTHING below is added: no
        # ephemeral-storage keys, no sizeLimit, no annotation (G1).
        repo_data_volume = client.V1EmptyDirVolumeSource()
        if disk is not None:
            main_resources["requests"]["ephemeral-storage"] = f"{disk.request_gi}Gi"
            main_resources["limits"]["ephemeral-storage"] = f"{disk.limit_gi}Gi"
            # The volume gets its own bound so /workspace (target dir + N sources)
            # cannot crowd out /tmp and the writable layer under the container limit.
            repo_data_volume = client.V1EmptyDirVolumeSource(size_limit=f"{disk.repo_data_gi}Gi")
            deployment_annotations[DISK_BUDGET_ANNOTATION] = str(disk.repo_data_gi)
        # Memory/CPU (py2rust gap #4 full). ⛔ ``memory`` is None without a services
        # hint, and then nothing changes (G1). With one, the container limit holds
        # the tooling reserve + dependencies + N services (sized, never > 8Gi).
        if memory is not None:
            main_resources["requests"]["memory"] = f"{memory.request_mi}Mi"
            main_resources["limits"]["memory"] = f"{memory.limit_mi}Mi"
            main_resources["limits"]["cpu"] = f"{memory.cpu_limit_m}m"
            deployment_annotations[MEMORY_LIMIT_ANNOTATION] = str(memory.limit_mi)

        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(
                name=name,
                namespace=self.namespace,
                labels=job_labels,
                annotations=deployment_annotations,
            ),
            spec=client.V1DeploymentSpec(
                replicas=1,
                # Recreate, not the default RollingUpdate: at replicas=1 a rolling
                # update briefly runs two pods, and exec_in_pod picks the first
                # Running pod by label — so a reprovision window could exec the
                # OLD pod (stale /workspace from a prior branch). Recreate keeps at
                # most one volume-holder pod; the brief no-pod gap surfaces as a
                # clean "No running pod found" for the caller to retry.
                strategy=client.V1DeploymentStrategy(type="Recreate"),
                selector=client.V1LabelSelector(
                    match_labels={
                        "app": "codegraph-repo",
                        "repo": name,
                    },
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        # provisioned-at changes every provision so the pod-template
                        # hash changes → a new ReplicaSet → a NEW pod with a fresh
                        # emptyDir → the git-clone init container re-clones into an
                        # empty /workspace. This is what makes an in-place patch (no
                        # delete) still yield a clean workspace on reprovision.
                        annotations={"coding-job/provisioned-at": provisioned_at},
                        labels={
                            "app": "codegraph-repo",
                            "repo": name,
                            "istio.io/dataplane-mode": "ambient",
                            JOB_LABEL_KEY: JOB_LABEL_VALUE,
                            TTL_CLASS_LABEL_KEY: TTL_CLASS_DEFAULT,
                        },
                    ),
                    spec=client.V1PodSpec(
                        node_selector=(
                            {"kubernetes.io/arch": self.node_arch} if self.node_arch else None
                        ),
                        # ⛔ NEVER let a coding-job land on the k3s control plane.
                        #
                        # This is the precondition for the CPU ceiling raised below, and
                        # it must not be removed while that ceiling stands. A ceiling is
                        # only safe once placement is bounded: a 1-CPU git-clone that can
                        # schedule onto the control plane is a 1-CPU burst against the
                        # API server, and the symptom (slow kubectl, reconcile timeouts)
                        # looks nothing like its cause.
                        #
                        # ⭐ THE SMALL REQUEST IS WHAT ATTRACTS THEM, WHICH IS THE
                        # COUNTER-INTUITIVE PART. node_selector above is arch-only, so
                        # nothing constrained these pods to workers. The container requests
                        # 100m, and the scheduler's LeastAllocated scoring favours the
                        # LEAST-requested node — a lightly loaded control plane therefore
                        # actively ATTRACTS them. A low request does not make a pod
                        # harmless; it makes it mobile, and its LIMIT is then the only
                        # thing bounding what it consumes wherever it lands.
                        #
                        # ⚠️ Quota does not guard this: the namespace's cpu limits are far
                        # above a single provisioned pod's ceiling, so nothing else stops
                        # it. That is why this belongs in the pod spec.
                        #
                        # MEASURED, why this key and not a nodeSelector on worker labels:
                        # `node-role.kubernetes.io/control-plane` is present on the
                        # control plane and on NO other node. So DoesNotExist excludes
                        # exactly the control plane and nothing else — it is not an inert
                        # predicate that matches everything. Dedicated CI nodes
                        # additionally carry a taint these pods do not tolerate, so the
                        # effective set is the general workers. A positive nodeSelector on
                        # worker labels would instead have to be edited every time a node
                        # joins.
                        #
                        # ⚠️ The REQUEST deliberately stays at 100m — see the resources
                        # block below, which argues that explicitly. Raising it would buy
                        # the same placement guarantee by spending scheduled capacity the
                        # namespace quota has to pay for, and would still not exclude the
                        # control plane, only make it less attractive. Affinity states the
                        # invariant; a request only nudges it.
                        affinity=client.V1Affinity(
                            node_affinity=client.V1NodeAffinity(
                                required_during_scheduling_ignored_during_execution=client.V1NodeSelector(
                                    node_selector_terms=[
                                        client.V1NodeSelectorTerm(
                                            match_expressions=[
                                                client.V1NodeSelectorRequirement(
                                                    key="node-role.kubernetes.io/control-plane",
                                                    operator="DoesNotExist",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ),
                        ),
                        service_account_name="code-graph-helper-sa",
                        # Set at POD level so the git-clone init container inherits
                        # it, which is the whole trick. Both volumes are emptyDir,
                        # and _build_workspace_clone_cmd finishes with
                        # `chmod -R 755` — group gets r-x, never w. So a root clone
                        # plus fsGroup alone would leave .git/objects and .git/refs
                        # owned 0:1000 mode 644/755, and `git commit` / `git config
                        # --local` would fail EACCES for a non-root main container.
                        # Cloning AS 1000:1000 into an emptyDir the kubelet already
                        # chowned to gid 1000 lands every file 1000:1000, which makes
                        # the chmod harmless. It also removes the last root container
                        # in the pod, so no per-container runAsNonRoot=false
                        # exception is needed.
                        security_context=client.V1PodSecurityContext(
                            run_as_non_root=True,
                            run_as_user=WORKSPACE_UID,
                            run_as_group=WORKSPACE_GID,
                            fs_group=WORKSPACE_GID,
                            # OnRootMismatch: skip the recursive chown when the
                            # volume root already carries the right gid. On a large
                            # checkout Always costs seconds of pod startup for no
                            # change.
                            fs_group_change_policy="OnRootMismatch",
                            seccomp_profile=client.V1SeccompProfile(type="RuntimeDefault"),
                        ),
                        init_containers=[
                            client.V1Container(
                                name=CLONE_INIT_CONTAINER,
                                image="alpine/git:latest",
                                security_context=client.V1SecurityContext(
                                    allow_privilege_escalation=False,
                                    capabilities=client.V1Capabilities(drop=["ALL"]),
                                    # read_only_root_filesystem is deliberately NOT
                                    # set. The /exec allowlist admits pip, apt,
                                    # apt-get, npm and uv; those write caches under
                                    # $HOME and __pycache__ into site-packages,
                                    # neither of which the /tmp volume covers.
                                    # Enabling it needs a further emptyDir mounted at
                                    # WORKSPACE_HOME first.
                                ),
                                command=["/bin/sh", "-c"],
                                args=[clone_cmd],
                                env=env_vars,
                                volume_mounts=[
                                    client.V1VolumeMount(
                                        name="repo-data",
                                        mount_path="/workspace",
                                    ),
                                ],
                                resources=client.V1ResourceRequirements(
                                    # Larger repos can exceed 256Mi during clone.
                                    #
                                    # ⚠️ The CPU LIMIT is a CEILING, not a
                                    # reservation — the REQUEST is what costs
                                    # scheduled capacity and it deliberately stays
                                    # at 100m. Raising the ceiling changes nothing
                                    # for the common case and only lets a big clone
                                    # finish.
                                    #
                                    # MEASURED 2026-09-17, why 200m was wrong:
                                    # ``clone_cmd`` is not just a clone — it is
                                    # clone + checkout + ref delete + ``reflog
                                    # expire`` + ``gc --prune=now``, and gc is
                                    # CPU-bound. On a small repo that is 1-3 s and
                                    # the ceiling never binds (aa-delivery-*,
                                    # tour-*, verify-* pods all read 1-3 s). On a
                                    # full-history repo it does not: a
                                    # django/django clone burned 42.9 CPU-seconds,
                                    # which at a 200m ceiling CANNOT complete in
                                    # under 214 s however idle the node is. Observed
                                    # wall clock was 238 s with the container
                                    # throttled in ~100% of its CFS periods — i.e.
                                    # ~90% of that init was the ceiling, not work.
                                    # 102 such pods in 24 h tripped
                                    # ContainerChronicallyCPUThrottled (67 of 288
                                    # 5-minute buckets over 80% throttled).
                                    #
                                    # 1 CPU is sized to that measurement (42.9
                                    # CPU-seconds ⇒ ~45 s) and stays well inside the
                                    # coding-jobs LimitRange max of 4. Peak observed
                                    # concurrency is 5 clones, so the worst-case
                                    # burst is 5 cores on a 16-core node currently
                                    # running at ~5.8.
                                    requests={"memory": "128Mi", "cpu": "100m"},
                                    limits={"memory": "512Mi", "cpu": "1"},
                                ),
                            ),
                        ],
                        containers=[
                            # Volume-holder: sleeps forever so the pod stays up
                            # and coding-analysis can ``kubectl exec`` into
                            # /workspace. No ports, no probes, no HTTP server.
                            # Name stays ``analysis`` for the exec_in_pod
                            # contract (see ``exec_in_pod(container=...)``).
                            client.V1Container(
                                name="analysis",
                                image=main_image,
                                image_pull_policy="IfNotPresent",
                                security_context=client.V1SecurityContext(
                                    allow_privilege_escalation=False,
                                    capabilities=client.V1Capabilities(drop=["ALL"]),
                                    # read_only_root_filesystem is deliberately NOT
                                    # set. The /exec allowlist admits pip, apt,
                                    # apt-get, npm and uv; those write caches under
                                    # $HOME and __pycache__ into site-packages,
                                    # neither of which the /tmp volume covers.
                                    # Enabling it needs a further emptyDir mounted at
                                    # WORKSPACE_HOME first.
                                ),
                                command=["/bin/sh", "-c"],
                                # Node/TS `npm install` of a private @nonagentic dep
                                # needs npm pointed at the deployment's private registry
                                # for the @nonagentic scope. A k8s env var name can't
                                # hold the scoped-registry key (`@`/`:` are illegal), so
                                # write an npm userconfig FILE at startup and point npm
                                # at it via NPM_CONFIG_USERCONFIG (inherited by the
                                # exec'd npm). A self-signed private registry needs
                                # strict-ssl=false (same rationale as UV_INSECURE_HOST
                                # for uv); public deps still resolve from npmjs via the
                                # codegraph-repo public egress. Unset = no file written.
                                # No-op for Python repos.
                                env=[
                                    client.V1EnvVar(
                                        name="NPM_CONFIG_USERCONFIG",
                                        value="/tmp/.npmrc",
                                    ),
                                    # Deterministic regardless of how the image
                                    # resolves uid 1000 in /etc/passwd. /tmp is a
                                    # group-writable emptyDir thanks to fsGroup, so
                                    # the npm userconfig above keeps working.
                                    client.V1EnvVar(
                                        name="HOME",
                                        value=WORKSPACE_HOME,
                                    ),
                                    # A deployment's private Python index is served
                                    # over HTTPS with an internal CA no workspace pod
                                    # trusts. Without these, a repo whose uv.lock pins
                                    # a package from that index cannot install AT ALL:
                                    # measured in this image, a cold `uv sync
                                    # --frozen` of a 189-package lock fails in 11 s
                                    # on `invalid peer certificate: UnknownIssuer`,
                                    # and succeeds in 6 s with them set.
                                    #
                                    # This is the uv counterpart of the npm
                                    # strict-ssl=false above, and the same posture
                                    # container_installer.py already applies
                                    # per-exec — except that route (/install-
                                    # dependencies) has no caller, so until now
                                    # NOTHING put these on a workspace. Setting them
                                    # on the POD means every exec inherits them,
                                    # including a card's own test command.
                                    #
                                    # Empty dict when CODING_PRIVATE_INDEX_HOST="",
                                    # so disabling the mechanism adds no vars rather
                                    # than adding broken ones.
                                    *[
                                        client.V1EnvVar(name=_k, value=_v)
                                        for _k, _v in python_resolver_env().items()
                                    ],
                                    # Without this, EVERY git operation in a
                                    # workspace fails with "detected dubious
                                    # ownership in repository at '/workspace'"
                                    # — status, commit and push alike.
                                    #
                                    # The kubelet creates an emptyDir owned by
                                    # root, and fsGroup sets only the GROUP, so
                                    # the worktree directory stays uid 0 while
                                    # everything the clone wrote inside it is
                                    # uid 1000. Git compares the WORKTREE dir's
                                    # owner against the euid and refuses. Nothing
                                    # can chown it: root would be needed, and
                                    # removing the last root container from this
                                    # pod is the point of running as 1000.
                                    #
                                    # Passed as GIT_CONFIG_PARAMETERS (command
                                    # scope, the variable `git -c` itself uses)
                                    # rather than GIT_CONFIG_COUNT/KEY_n/VALUE_n: a
                                    # caller's own GIT_CONFIG_COUNT pairs then
                                    # STACK on it instead of replacing it.
                                    #
                                    # Passed as env rather than written
                                    # into the image's system config because
                                    # docker_service.py sets GIT_CONFIG_NOSYSTEM=1
                                    # on a sibling path — a system-config fix is
                                    # one env var away from being silently
                                    # disabled. Verified in git 2.47.3 that
                                    # safe.directory IS honoured from these
                                    # variables (it is deliberately ignored from
                                    # repo-local config, so this is not a given).
                                    #
                                    # Scoped to the one path, never "*".
                                    client.V1EnvVar(
                                        name="GIT_CONFIG_PARAMETERS",
                                        value=git_safe_directory_parameters(WORKSPACE_DIR),
                                    ),
                                ],
                                args=[
                                    npm_userconfig_startup() + "trap 'exit 0' TERM INT; "
                                    "while :; do sleep 86400 & wait $!; done"
                                ],
                                volume_mounts=[
                                    client.V1VolumeMount(
                                        name="repo-data",
                                        mount_path="/workspace",
                                    ),
                                    client.V1VolumeMount(
                                        name="tmp",
                                        mount_path="/tmp",
                                    ),
                                ],
                                resources=client.V1ResourceRequirements(
                                    # 256Mi OOM-kills real dependency installs: an
                                    # `npm install` of a TS/n8n devDependency tree is
                                    # SIGKILLed (exit 137 "Killed") mid-resolve, and
                                    # because only the exec'd child dies (not the
                                    # sleeping main proc) the pod never restarts, so
                                    # the OOM is invisible as a crash — it only shows
                                    # up as upgrade success=false. Python `uv` fit
                                    # under 256Mi (why Python drives worked and Node
                                    # broke). Burst the limit to 2Gi (request stays
                                    # low so scheduling is unaffected).
                                    # (The python values live in main_resources'
                                    # fallback above; python+rust gets the larger
                                    # RUST_RESOURCES profile from src/core/toolchain.py.)
                                    requests=main_resources["requests"],
                                    limits=main_resources["limits"],
                                ),
                            ),
                        ],
                        volumes=[
                            client.V1Volume(
                                name="repo-data",
                                empty_dir=repo_data_volume,
                            ),
                            client.V1Volume(
                                name="tmp",
                                empty_dir=client.V1EmptyDirVolumeSource(),
                            ),
                        ],
                    ),
                ),
            ),
        )

        live: Any | None = None
        try:
            live = self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment,
            )
            # The create RETURNED: this call made the object, so it may reclaim it.
            if scope is not None:
                scope.owned = True
            logger.info("Created Deployment: %s", name)
        except ApiException as e:
            if e.status != 409:
                logger.error("Failed to create Deployment: %s", e)
                raise
            # Already exists. Reconcile in place rather than delete-recreate: a
            # delete sets a finalizer and the object lingers until its pod
            # terminates, so the old "delete + sleep(2) + create" raced the
            # finalizer and the second create could 409 with "object is being
            # deleted". A strategic-merge patch never deletes, so that race class
            # is gone; the changed provisioned-at pod-template annotation rolls a
            # fresh pod (new emptyDir → re-clone). Mirrors the create-or-replace
            # pattern already used for the git-token Secret above.
            existing = self.apps_v1.read_namespaced_deployment(
                name=name,
                namespace=self.namespace,
            )
            # The client's major bump (kubernetes 36 -> 37) models `.metadata`
            # as Optional. The API server always returns it on an object we just
            # read, so defaulting to an empty V1ObjectMeta preserves behaviour.
            existing_metadata = existing.metadata or client.V1ObjectMeta()
            if existing_metadata.deletion_timestamp is not None:
                # Rare cross-operation race: a prior delete (remove_container or
                # the reaper) is still finalizing. Wait for it to finish, then
                # create fresh — patching a terminating object would just patch
                # something destined for deletion.
                logger.info(
                    "Deployment %s is terminating; waiting for deletion before recreate",
                    name,
                )
                self._wait_until_deleted(name, timeout=30)
                live = self.apps_v1.create_namespaced_deployment(
                    namespace=self.namespace,
                    body=deployment,
                )
                if scope is not None:
                    scope.owned = True
                logger.info("Recreated Deployment after deletion: %s", name)
            else:
                # ⛔ Ownership gate. Without it this branch is a cross-tenant
                # TAKEOVER: ``body=deployment`` carries the CALLER's labels, so
                # patching re-stamps ``tenant-id`` to the caller, and the changed
                # provisioned-at pod-template annotation rolls a fresh pod (new
                # emptyDir → re-clone), destroying the incumbent's work. Names are
                # caller-supplied and ``POST /`` does not gate on them, so any
                # tenant could reach any other tenant's workspace by naming it.
                #
                # Compare against the caller's RAW ``tenant_id``, not
                # ``job_labels[TENANT_LABEL_KEY]``: the label is only written when
                # the value satisfies the k8s label grammar, so reading the
                # claim back out of the label would let a grammar-invalid tenant
                # present as "no claim" and walk through the guard.
                #
                # ⛔ The INCUMBENT side had exactly that defect, which the comment
                # above forbade three lines later for the caller: it read the
                # incumbent's claim out of ``labels`` ALONE. A grammar-invalid
                # owner is never labelled, so the incumbent read as "no claim",
                # the guard did not fire, and the patch below re-stamped the
                # workspace to the caller. Recover from the annotation first.
                incumbent_tenant = _recover_tenant_id(
                    existing_metadata.labels, existing_metadata.annotations
                )
                if not incumbent_tenant:
                    # ⭐ THE RESIDUAL, decided explicitly and failing CLOSED.
                    # Neither stamp yielded an owner, so the owner is UNKNOWN —
                    # NOT "ownerless". The two are indistinguishable here, and
                    # they demand opposite answers: ownerless is safe to patch,
                    # unknown may be someone else's work. We cannot tell, so we
                    # must not destroy it. This branch previously ALLOWED the
                    # patch ("an unlabelled incumbent stays reconcilable"), which
                    # is precisely the door a grammar-invalid owner fell through.
                    #
                    # Availability cost, measured before accepting it: 0 of the
                    # 10 live Deployments in ``coding-jobs`` resolve to unknown
                    # (10/10 labelled), and every provision from here on writes the
                    # annotation — so nothing reachable today is stranded. A
                    # genuinely ownerless Deployment can still be replaced by
                    # deleting it, which is an explicit destructive act rather
                    # than a silent one.
                    #
                    # Logged, because the ValueError below is worded IDENTICALLY
                    # to a genuine absence (deliberately — the router maps it to a
                    # non-disclosing 404). Without this line a refused provision
                    # and a missing Deployment are indistinguishable to an
                    # operator reading the logs.
                    logger.warning(
                        "Refusing to reconcile Deployment %s for tenant %s: incumbent owner is "
                        "UNRECOVERABLE (no %s annotation and no %s label). Refusing rather than "
                        "patching, because patching would re-stamp the workspace to the caller "
                        "and roll a fresh pod, destroying the incumbent's work if it has one.",
                        name,
                        tenant_id,
                        TENANT_ANNOTATION,
                        TENANT_LABEL_KEY,
                    )
                    raise WorkspaceNotAvailableError(
                        f"Container '{name}' not found or not available for provisioning"
                    )
                if tenant_id and incumbent_tenant != tenant_id:
                    # "not found" so the router maps this to 404 and the name is
                    # not confirmed to exist — same non-disclosure rule the
                    # get_container BOLA gate follows.
                    raise WorkspaceNotAvailableError(
                        f"Container '{name}' not found or not available for provisioning"
                    )
                # Ownership PROVEN only for a caller whose tenant matches the
                # incumbent's. A tenant-less caller still reaches the patch below
                # (pre-existing behaviour) but has proven nothing, so its
                # failure must not reclaim the incumbent by name.
                if scope is not None and tenant_id and incumbent_tenant == tenant_id:
                    scope.owned = True
                # A strategic-merge patch MERGES annotations, so a python
                # reprovision of a former python+rust Deployment would keep the
                # toolchain stamp (and with it the wider exec allowlist) on a
                # python image. Null it explicitly — only on this path, so the
                # create body of a python workspace is unchanged (G1).
                if toolchain == PYTHON and TOOLCHAIN_ANNOTATION in (
                    existing_metadata.annotations or {}
                ):
                    _body_annotations(deployment)[TOOLCHAIN_ANNOTATION] = None
                # The disk sizing (gap #4-lite) is deliberately NOT nulled here: a
                # merge keeps the old sizeLimit and ephemeral limits TOGETHER, so a
                # stale-but-consistent bound remains (harmless, unlike a stale
                # toolchain stamp); a new hint replaces them.
                # The memory sizing is the opposite: without a hint the body sets
                # limits.memory back to the toolchain default, so the annotation that
                # reports the OLD sized limit (memory_limit_mi) must go with it.
                if memory is None and MEMORY_LIMIT_ANNOTATION in (
                    existing_metadata.annotations or {}
                ):
                    _body_annotations(deployment)[MEMORY_LIMIT_ANNOTATION] = None
                live = self.apps_v1.patch_namespaced_deployment(
                    name=name,
                    namespace=self.namespace,
                    body=deployment,
                )
                logger.info("Reconciled existing Deployment in place: %s", name)

        # The owner now exists, so the credential can be written already owned.
        # An abandoned/crashed provision, a reaper reap and an out-of-band
        # `kubectl delete deploy` all now cascade to this Secret via the k8s
        # garbage collector. _cleanup_resources still deletes it explicitly —
        # this is the safety net underneath that path, not a replacement.
        if authenticated:
            self._create_git_token_secret(
                name,
                github_token or "",
                owner=self._owner_reference(live, name),
            )

    def _wait_until_deleted(self, name: str, timeout: int = 30) -> None:
        """Block until a Deployment is fully gone (GET returns 404).

        Replaces the old blind ``time.sleep(2)``: polls the apiserver so the
        recreate only happens once the prior object (and its finalizer) is
        actually gone. Raises ``RuntimeError`` on timeout rather than risking a
        ``409 object is being deleted`` from a premature recreate.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self.apps_v1.read_namespaced_deployment(name=name, namespace=self.namespace)
            except ApiException as e:
                if e.status == 404:
                    return
                raise
            time.sleep(0.5)
        raise RuntimeError(
            f"Deployment {name} still present after {timeout}s; deletion did not complete"
        )

    @staticmethod
    def _terminal_pod_failure(pod: Any) -> str | None:
        """Return a reason string when *pod* is settled in a state it cannot leave.

        ``_resolve_pod_from_deployment`` waits up to ``POD_RESOLVE_TIMEOUT_S`` for a
        cold image pull. That budget must not be spent on a pod that Kubernetes has
        already given up on, so a pod parked in a back-off/config state short-circuits
        the wait with an actionable message instead of a generic timeout.

        Only settled states count. A first-time pull reports ``Pulling`` (and is not
        listed here), so a slow-but-healthy start is never mistaken for a failure.
        """
        settled = {
            "ImagePullBackOff",  # pull failed and k8s is now backing off
            "ErrImagePull",
            "InvalidImageName",
            "CreateContainerConfigError",  # missing Secret/ConfigMap key
            "CrashLoopBackOff",
        }
        statuses = list(pod.status.init_container_statuses or []) + list(
            pod.status.container_statuses or []
        )
        for st in statuses:
            waiting = getattr(st.state, "waiting", None) if st.state else None
            reason = getattr(waiting, "reason", None) if waiting else None
            if reason in settled:
                message = getattr(waiting, "message", None) or ""
                return f"{st.name}: {reason}{f' ({message})' if message else ''}"
        return None

    def resolve_running_pod(self, deployment_name: str) -> str:
        """Public alias of ``_resolve_pod_from_deployment`` (same bounded wait)."""
        return self._resolve_pod_from_deployment(deployment_name)

    def pod_is_gone(self, pod_name: str) -> bool:
        """True when ``pod_name`` no longer exists or is being deleted / has ended.

        A pod that is merely not Ready is NOT gone. ``False`` on any API error
        other than 404: an unreachable API server is not proof of absence.
        """
        try:
            pod = self.core_v1.read_namespaced_pod(name=pod_name, namespace=self.namespace)
        except ApiException as e:
            return e.status == 404
        if getattr(pod.metadata, "deletion_timestamp", None):
            return True
        return getattr(pod.status, "phase", None) in ("Succeeded", "Failed")

    def _resolve_pod_from_deployment(self, deployment_name: str) -> str:
        """Resolve a deployment name to a running pod name.

        The Deployment uses the Recreate strategy, so every (re)provision has a
        brief window with zero pods (the old pod Terminating, the new one not
        yet Running). Rather than fail fast and force every caller to implement
        retry, bounded-wait up to ``POD_RESOLVE_TIMEOUT_S`` for a
        ``phase=="Running"`` pod, re-listing every ``POD_RESOLVE_INTERVAL_S``.
        Only Running pods are ever returned, so the Recreate stale-pod guard is
        preserved — a Terminating old pod (phase != Running) is never handed
        back.

        Args:
            deployment_name: Name of the Kubernetes Deployment

        Returns:
            Name of a running pod belonging to the deployment

        Raises:
            ContainerDeploymentNotFoundError: If the Deployment is absent
                (fast-fail, no wait) — a not-found condition, not a server error.
            RuntimeError: On other API errors, or if no running pod appears
                within the bounded timeout.
        """
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
            )
        except ApiException as e:
            # 404 = the Deployment genuinely does not exist (container never
            # provisioned or already deprovisioned). Surface a distinguishable
            # not-found error so the exec router can return an honest HTTP 404
            # rather than a misleading 500. Other statuses stay a generic
            # RuntimeError (real cluster/API failures → 500).
            if e.status == 404:
                raise ContainerDeploymentNotFoundError(
                    f"Deployment {deployment_name} not found: {e.status}"
                ) from e
            raise RuntimeError(f"Deployment {deployment_name} not found: {e.status}") from e

        selector = deployment.spec.selector if deployment.spec else None
        match_labels = selector.match_labels if selector else None
        if not match_labels:
            # An EMPTY label selector lists EVERY pod in the namespace, so a
            # Deployment we cannot derive a selector from must fail here rather
            # than go on to resolve some other workload's pod.
            raise RuntimeError(f"Deployment {deployment_name} has no selector labels")
        label_selector = ",".join(f"{k}={v}" for k, v in match_labels.items())

        # Bounded poll: at least one attempt, then re-list until a Running pod
        # appears or the timeout elapses. Attempt-counted (not wall-clock) so
        # the wait is deterministic and unit-testable with time.sleep patched.
        attempts = max(1, int(POD_RESOLVE_TIMEOUT_S / POD_RESOLVE_INTERVAL_S))
        for attempt in range(attempts):
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=label_selector,
            )
            for pod in pods.items:
                pod_name = (pod.metadata.name if pod.metadata else None) or ""
                if pod.status is not None and pod.status.phase == "Running":
                    if attempt:
                        logger.info(
                            "Resolved pod %s for deployment %s after waiting %ds "
                            "for the Recreate provisioning gap / cold image pull",
                            pod_name,
                            deployment_name,
                            int(attempt * POD_RESOLVE_INTERVAL_S),
                        )
                    return pod_name

            # Do not burn the (now much larger) budget on a pod k8s has given up
            # on. Checked only after the Running scan above, so a pod that has
            # already recovered still wins.
            for pod in pods.items:
                pod_name = (pod.metadata.name if pod.metadata else None) or ""
                failure = self._terminal_pod_failure(pod)
                # The clone RAN and keeps exiting non-zero: the repository could
                # not be read with this credential/branch. Image or config errors
                # on the same container are ours, not the caller's: left generic.
                if failure and failure.startswith(f"{CLONE_INIT_CONTAINER}: CrashLoopBackOff"):
                    raise WorkspaceCloneFailedError(deployment_name, pod_name, failure)
                if failure:
                    raise RuntimeError(
                        f"Pod {pod_name} for deployment {deployment_name} cannot start -- {failure}"
                    )

            if attempt < attempts - 1:
                time.sleep(POD_RESOLVE_INTERVAL_S)

        raise RuntimeError(
            f"No running pod found for deployment {deployment_name} after "
            f"{int(POD_RESOLVE_TIMEOUT_S)}s"
        )

    def exec_in_pod(
        self,
        deployment_name: str,
        command: list[str],
        workdir: str = "/workspace",
        timeout: int = 300,
        container: str = "analysis",
        max_output_bytes: int | None = None,
        pod_name: str | None = None,
    ) -> dict[str, Any]:
        """Execute a command in a pod belonging to a deployment.

        Args:
            deployment_name: Name of the Kubernetes Deployment
            command: Command as list of strings
            workdir: Working directory inside the container
            timeout: Execution timeout in seconds
            container: Container name within the pod
            max_output_bytes: Per-stream byte budget. Defaults to
                ``EXEC_HARD_OUTPUT_BYTES``, which is a ceiling rather than a
                policy — callers exposing this to the outside pass something
                smaller.
            pod_name: Exec into exactly this pod instead of resolving one from
                the Deployment. Detached async exec pins the pod it launched
                in: a replacement pod has a fresh /tmp and none of the job's
                status files, so resolving afresh would read the wrong pod.

        Returns:
            Dict with exit_code, stdout, stderr, duration, truncated,
            bytes_omitted. ``truncated`` is always present so a caller cannot
            mistake a clipped result for a complete one by forgetting to look.
        """
        if pod_name is None:
            pod_name = self._resolve_pod_from_deployment(deployment_name)
        # NEVER log `command` (or a slice of it) directly. Every command reaches
        # here as ["sh", "-c", <whole script>], so `command[:3]` published the
        # entire script — git_push's `printf '%s' <token>` included — to a log
        # store that retains it. summarize_command keeps the programs and drops
        # the values. See src/core/log_safety.py.
        logger.info(
            "Executing in pod %s (deployment %s): %s",
            pod_name,
            deployment_name,
            summarize_command(command),
        )

        # Wrap command with cd to workdir and timeout
        shell_cmd = f"cd {shlex.quote(workdir)} && timeout {timeout} {shlex.join(command)}"
        exec_command = ["/bin/sh", "-c", shell_cmd]

        start_time = time.time()

        # Dedicated ApiClient for the exec. exec_in_pod runs in a WORKER THREAD
        # (its async callers wrap it in asyncio.to_thread) while REST calls such
        # as create_namespaced_secret run on the event loop. The kubernetes
        # ApiClient is NOT thread-safe -- sharing self.core_v1 across the
        # thread/loop boundary races on the client's mutable response state and
        # tears a concurrent create into ApiException(body=None), which the lib's
        # own handler then crashes on (`e.body.decode()` on None -> AttributeError
        # -> spurious 500). A fresh per-call client isolates the exec completely.
        exec_core_v1 = client.CoreV1Api(client.ApiClient())

        try:
            resp = k8s_stream(
                exec_core_v1.connect_get_namespaced_pod_exec,
                pod_name,
                self.namespace,
                container=container,
                command=exec_command,
                stderr=True,
                stdout=True,
                stdin=False,
                tty=False,
                _preload_content=False,
            )

            limit = EXEC_HARD_OUTPUT_BYTES if max_output_bytes is None else max_output_bytes
            stdout_buf = _BoundedStream(limit)
            stderr_buf = _BoundedStream(limit)

            # Read output with timeout. Reading continues past the budget even
            # though the excess is discarded: draining is what lets the command
            # finish and the error channel deliver the real exit code. Stopping
            # early would trade unbounded memory for an unknown exit status.
            deadline = start_time + timeout + 5
            while resp.is_open():
                resp.update(timeout=min(5, max(1, deadline - time.time())))
                if resp.peek_stdout():
                    stdout_buf.write(resp.read_stdout())
                if resp.peek_stderr():
                    stderr_buf.write(resp.read_stderr())
                if time.time() > deadline:
                    logger.warning("Exec timeout reached for pod %s", pod_name)
                    break

            stdout_data = stdout_buf.value()
            stderr_data = stderr_buf.value()
            truncated = stdout_buf.truncated or stderr_buf.truncated
            bytes_omitted = stdout_buf.bytes_omitted + stderr_buf.bytes_omitted
            if truncated:
                logger.warning(
                    "Exec output truncated for pod=%s: %d bytes omitted (limit %d/stream)",
                    pod_name,
                    bytes_omitted,
                    limit,
                )

            resp.close()
            duration = time.time() - start_time

            # Parse exit code from error channel
            exit_code = resp.returncode if hasattr(resp, "returncode") else -1

            # Try to get exit code from the error channel response
            if exit_code == -1:
                err = resp.read_channel(3)  # ERROR_CHANNEL
                if err:
                    try:
                        err_data = json.loads(err)
                        if err_data.get("status") == "Success":
                            exit_code = 0
                        else:
                            # Extract exit code from reason/message
                            exit_code = int(
                                err_data.get("details", {})
                                .get("causes", [{}])[0]
                                .get("message", "1")
                            )
                    except json.JSONDecodeError, ValueError, IndexError, KeyError:
                        # Error channel was present but unparseable: the command's
                        # true status is unknown. Fail closed — a non-zero command
                        # that wrote only to stdout (empty stderr) must NOT be
                        # reported as success (exit 0), or callers like
                        # get_container_repository_tree trust corrupt output.
                        logger.warning(
                            "Pod exec error-channel unparseable for pod=%s; "
                            "defaulting exit_code=1 (fail-closed)",
                            pod_name,
                        )
                        exit_code = 1

            logger.info(
                "Pod exec completed: pod=%s, exit_code=%d, duration=%.2fs",
                pod_name,
                exit_code,
                duration,
            )

            return {
                "exit_code": exit_code,
                "stdout": stdout_data,
                "stderr": stderr_data,
                "duration": duration,
                "truncated": truncated,
                "bytes_omitted": bytes_omitted,
            }

        except (ApiException, AttributeError, OSError, RuntimeError) as e:
            # AttributeError is here for ONE reason, and it is measured: kubernetes-client
            # 36.0.0's own ApiClient.__call_api re-raises an ApiException with
            #     e.body = e.body.decode("utf-8")
            # and crashes with AttributeError when body is None -- destroying e.status /
            # e.reason before anyone can read them. Without this line that crash escaped
            # exec_in_pod entirely, the route 500'd with NO exit code, and the conveyor's
            # step harness recorded `dispatch_failed` (card 1865491581142828786, 16:21:34Z,
            # request b69d4093). _create_git_token_secret already handles the same masked
            # case (the precedent is ~750 lines up); exec_in_pod never got it. The real
            # k8s error is unrecoverable by then, so this returns the ordinary error dict
            # with a NUMERIC exit code -- a measured failure the sweep can retry, not a
            # silent one.
            duration = time.time() - start_time
            # `str(e)` is free-form: a urllib3/websocket error renders the exec
            # URL, whose `?command=` query parameters carry the whole script.
            # No structural projection is possible on an arbitrary exception
            # message, so this is the best-effort denylist layer — deliberately
            # a fallback, not the guard the line above relies on.
            logger.error(
                "Pod exec failed: pod=%s, command=%s, error=%s%s",
                pod_name,
                summarize_command(command),
                redact_secrets(str(e)),
                " [masked ApiException(body=None) from kubernetes-client; the real "
                "k8s status/reason were destroyed inside the library]"
                if isinstance(e, AttributeError) and "decode" in str(e)
                else "",
            )
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "duration": duration,
                # Present on every path so a caller reading result["truncated"]
                # never has to guard for a missing key — a KeyError here would
                # turn a failed exec into a 500.
                "truncated": False,
                "bytes_omitted": 0,
            }
        finally:
            # Release the per-call ApiClient's connection pool (it is not reused).
            try:
                exec_core_v1.api_client.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass

    async def get_container_repository_tree(self, container_name: str) -> dict[str, Any]:
        """Get repository tree structure by executing find in a K8s pod.

        Args:
            container_name: Deployment name to exec into

        Returns:
            Tree structure dictionary with a ``root`` key, plus ``truncated``
            and ``bytes_omitted`` when the listing hit the output ceiling.
        """
        # Offload the blocking k8s exec to a thread so the event loop (and the
        # /health liveness probe) stays responsive.
        result = await asyncio.to_thread(
            self.exec_in_pod,
            deployment_name=container_name,
            command=["find", "/workspace", "-type", "f", "-o", "-type", "d"],
            workdir="/",
            timeout=30,
        )

        if result["exit_code"] != 0:
            raise RuntimeError(f"Failed to get repository tree: {result['stderr']}")

        # A tree that lost its tail must say so. This exec is bounded by
        # EXEC_HARD_OUTPUT_BYTES now, and a repository large enough to hit it
        # would otherwise return a subtree that looks like the whole thing —
        # the same silent-success shape as a listing that reports "no files"
        # for a directory it was never able to read.
        if result.get("truncated"):
            logger.warning(
                "Repository tree for %s is incomplete: %d bytes of find output omitted",
                container_name,
                result.get("bytes_omitted", 0),
            )

        tree: dict[str, Any] = {"type": "directory", "name": "repo", "children": {}}

        for line in result["stdout"].strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            path = line.replace("/workspace", "").strip("/")
            if path:
                self._add_to_tree(tree, path)

        return {
            "root": tree,
            "truncated": bool(result.get("truncated")),
            "bytes_omitted": int(result.get("bytes_omitted", 0)),
        }

    @staticmethod
    def _add_to_tree(tree: dict[str, Any], path: str) -> None:
        """Build tree structure from a file path."""
        parts = path.split("/")
        current = tree["children"]

        for i, part in enumerate(parts):
            if part not in current:
                is_file = i == len(parts) - 1 and "." in part
                current[part] = {
                    "type": "file" if is_file else "directory",
                    "name": part,
                    "children": {} if not is_file else None,
                }
            current = current[part]["children"] if current[part]["children"] is not None else {}

    def _cleanup_resources(self, name: str) -> None:
        """Clean up Deployment and Secret resources.

        Legacy per-pod Services (from before the code_graph_helper sidecar was
        retired) may still exist for old containers; delete them too so cleanup
        stays complete for historical provisions.

        The explicit Secret delete below is KEPT even though the Secret now
        carries an ownerReference on its Deployment: it reclaims the credential
        synchronously instead of waiting for the garbage collector, and it still
        covers pre-ownerReference Secrets. The ownerReference is the safety net
        for every path that never reaches this method (crash, abandonment,
        out-of-band `kubectl delete deploy`), not a replacement for it.
        """
        try:
            self.apps_v1.delete_namespaced_deployment(
                name=name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(propagation_policy="Background"),
            )
            logger.info("Deleted Deployment: %s", name)
        except ApiException as e:
            if e.status != 404:
                logger.error("Failed to delete Deployment: %s", e)

        # Best-effort cleanup of any legacy Service left over from pre-retirement
        # provisions. New provisions never create one; 404 here is expected.
        try:
            self.core_v1.delete_namespaced_service(
                name=name,
                namespace=self.namespace,
            )
            logger.info("Deleted legacy Service: %s", name)
        except ApiException as e:
            if e.status != 404:
                logger.error("Failed to delete legacy Service: %s", e)

        # Clean up the git token secret
        secret_name = f"{name}-git-token"
        try:
            self.core_v1.delete_namespaced_secret(
                name=secret_name,
                namespace=self.namespace,
            )
            logger.info("Deleted Secret: %s", secret_name)
        except ApiException as e:
            if e.status != 404:
                logger.error("Failed to delete Secret: %s", e)

    async def remove_container(self, container_name: str) -> None:
        """Remove a provisioned container (Deployment + Secret, plus any legacy Service)."""
        self._cleanup_resources(container_name)

    async def get_container_status(self, container_name: str) -> dict[str, Any]:
        """Get the status of a provisioned container."""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=container_name,
                namespace=self.namespace,
            )

            deployment_status = deployment.status
            ready_replicas = (
                deployment_status.ready_replicas if deployment_status is not None else None
            ) or 0
            desired_replicas = (deployment.spec.replicas if deployment.spec else None) or 1

            if ready_replicas >= desired_replicas:
                status = "running"
            elif deployment_status is not None and deployment_status.conditions:
                status = "starting"
            else:
                status = "pending"

            return {
                "name": container_name,
                "status": status,
                "ready_replicas": ready_replicas,
                "desired_replicas": desired_replicas,
                "endpoint": f"http://{container_name}.{self.namespace}.svc.cluster.local:8687",
            }
        except ApiException as e:
            if e.status == 404:
                return {"name": container_name, "status": "not_found"}
            raise

    def reconstruct_container_info(self, container_name: str) -> ContainerInfo | None:
        """Rebuild a minimal ContainerInfo from the live Deployment, or None.

        The JSON registry is transient and can momentarily lose a freshly-stored
        entry (an interleaved whole-dict save), which made a healthy container
        404 for the whole readiness window. The Deployment is the durable source
        of truth, so the registry-miss path in ``ContainerManager.get_container``
        falls back here: if a managed Deployment with this name exists, return a
        usable ContainerInfo (name, endpoints, live status, and the repo metadata
        recovered from the Deployment annotations stamped at provision time) so
        status/exec/dependency AND apply-changes/fork calls keep working — the
        last require a non-empty ``repo_url``.

        ``tenant_id`` is RECOVERED via ``_recover_tenant_id`` — the
        ``coding-job/tenant-id`` ANNOTATION first, the ``tenant-id`` label as the
        backward-compatible fallback. It used to be hardcoded ``None`` here "for
        parity with the registry-HIT path"; that was measured false and the
        consequence was a cross-tenant read. Reading the label alone then left a
        second, narrower hole: a grammar-invalid owner is never labelled, so an
        OWNED workspace still reconstructed as tenant-less. ``None`` now means the
        owner is UNKNOWN and the caller must fail CLOSED on it.
        Synchronous — the k8s client is blocking and this is called from sync
        ``get_container``.
        """
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=container_name,
                namespace=self.namespace,
            )
        except ApiException as e:
            if e.status == 404:
                return None
            logger.warning("reconstruct_container_info read failed for %s: %s", container_name, e)
            return None

        metadata = deployment.metadata
        labels = (metadata.labels or {}) if metadata else {}
        if labels.get("managed-by") != "code-graph-helper":
            return None  # not one of ours — don't fabricate access to it

        # Recover the durable repo metadata stamped at provision time. Pre-change
        # Deployments carry no annotations → repo_url stays "" (the prior
        # behaviour, no crash); they are transient and reaper-bound.
        annotations = (metadata.annotations or {}) if metadata else {}
        repo_url = annotations.get(REPO_URL_ANNOTATION, "")
        branch = annotations.get(BRANCH_ANNOTATION) or "main"
        secret_config_id = annotations.get(SECRET_CONFIG_ANNOTATION) or None
        original_repo_owner, original_repo_name = self._owner_repo_from_url(repo_url)

        # ⛔ Do NOT hardcode this back to ``None``. It was ``tenant_id=None`` until
        # this change, justified as "parity with the registry-HIT path, which stores
        # tenant_id=None for these containers (verified live)". Both halves of that
        # justification are wrong:
        #
        #   * The HIT path does NOT store None. ``provision_from_github`` writes
        #     ``tenant_id=getattr(request, "tenant_id", None)`` onto the registry
        #     ContainerInfo (:807) from the tenant the route stamped, and
        #     ``container_lifecycle`` raises ``BroadcastTenantMissingError`` rather
        #     than persist a tenant-less row. So "parity" made the MISS path
        #     strictly weaker than the HIT path, not equal to it.
        #   * ``ContainerManager.get_container`` gates on
        #     ``container_tenant is not None and container_tenant != tenant_id``,
        #     so a ``None`` here does not make the gate "not apply" — it makes the
        #     gate VACUOUS. Unknown was read as permitted.
        #
        # Measured consequence, on the same registry-MISS path ``get_container``'s
        # own comment calls routine: a caller in tenant …0003 got 200 OK and
        # ``container_status: running`` from ``GET /{name}/status`` for a Deployment
        # labelled ``tenant-id: …0001``. #133 closed that at the router for a
        # registry HIT; the MISS path kept serving it, so the same poll answered
        # 404 or 200 depending only on whether Redis had the row that second.
        #
        # The cluster has always carried the answer — we read ``metadata.labels``
        # four lines above this to check ``managed-by`` and then threw the tenant
        # away.
        #
        # ⚠️ This line used to read the LABEL ALONE, and the sentence that stood
        # here — "a ``None`` that survives this line now means the Deployment was
        # provisioned with NO tenant at all (ownerless), not 'owner unknown'" —
        # was FALSE, which is what made the remaining fail-open dangerous. The
        # label is written only for grammar-valid tenants, so an unlabelled
        # Deployment could equally be an OWNED one whose owner could not be
        # expressed as a label. Recovering from the annotation first closes that
        # gap; a ``None`` that survives now genuinely means UNKNOWN, and
        # ``ContainerManager.get_container`` fails closed on it.
        tenant_id = _recover_tenant_id(labels, annotations)

        ready = (deployment.status.ready_replicas or 0) if deployment.status else 0
        desired = (deployment.spec.replicas or 1) if deployment.spec else 1
        status = ContainerStatus.RUNNING if ready >= desired else ContainerStatus.CREATING

        service_endpoint = f"http://{container_name}.{self.namespace}.svc.cluster.local:8687"
        public_endpoint = (
            f"{get_settings().main_api_endpoint.rstrip('/')}/api/workspaces/{container_name}"
        )
        return ContainerInfo(
            container_name=container_name,
            api_endpoint=public_endpoint,
            internal_endpoint=service_endpoint,
            public_endpoint=public_endpoint,
            code_graph_id=container_name,
            codebase_path="/workspace",
            repo_url=repo_url,
            branch=branch,
            original_repo_owner=original_repo_owner,
            original_repo_name=original_repo_name,
            secret_config_id=secret_config_id,
            status=status,
            description="Reconstructed from live Deployment (registry entry missing)",
            # Recovered from the Deployment annotation, then label — see above.
            # ``None`` here means UNKNOWN owner, and get_container denies on it.
            tenant_id=tenant_id,
            # Absent annotation ⇒ python (every pre-toolchain Deployment), so the
            # exec policy fails toward the narrower allowlist.
            toolchain=toolchain_from_annotations(annotations),
            disk_budget_gi=_positive_int_annotation(annotations, DISK_BUDGET_ANNOTATION),
            memory_limit_mi=_positive_int_annotation(annotations, MEMORY_LIMIT_ANNOTATION),
        )

    async def list_containers(self) -> list[dict[str, Any]]:
        """List all provisioned containers."""
        try:
            deployments = self.apps_v1.list_namespaced_deployment(
                namespace=self.namespace,
                label_selector="managed-by=code-graph-helper",
            )

            result = []
            for dep in deployments.items:
                dep_metadata = dep.metadata or client.V1ObjectMeta()
                name = dep_metadata.name
                ready = (dep.status.ready_replicas if dep.status else None) or 0
                desired = (dep.spec.replicas if dep.spec else None) or 1

                result.append(
                    {
                        "name": name,
                        "status": "running" if ready >= desired else "starting",
                        "ready_replicas": ready,
                        "desired_replicas": desired,
                        "created": (
                            dep_metadata.creation_timestamp.isoformat()
                            if dep_metadata.creation_timestamp
                            else None
                        ),
                    }
                )

            return result
        except ApiException as e:
            logger.error("Failed to list containers: %s", e)
            raise

    def list_managed_container_names(self) -> list[str]:
        """Live Deployment names for managed containers (durable cluster truth).

        Synchronous sibling of the async ``list_containers`` above, callable from
        the sync ``ContainerManager.list_containers``. The JSON registry lives on
        ephemeral pod-local storage and is lost on redeploy/reschedule while the
        Deployments persist, so the registry can under-report a still-running
        container. Enumerating the cluster by the same ``managed-by`` label the
        reconstruct path guards on lets the manager union in those survivors and
        rebuild each via ``reconstruct_container_info``. The k8s client is
        blocking, matching ``reconstruct_container_info`` (also sync).
        """
        deployments = self.apps_v1.list_namespaced_deployment(
            namespace=self.namespace,
            label_selector="managed-by=code-graph-helper",
        )
        return [
            dep.metadata.name for dep in deployments.items if dep.metadata and dep.metadata.name
        ]

    # ------------------------------------------------------------------
    # Phase C-1 — lifecycle GC (reaper) + admission-cap counting.
    # These operate off the CLUSTER (label-selected Deployments), never the
    # ephemeral JSON registry, so they survive the /app/data wipe on pod
    # redeploy. Callers off-load these blocking calls to a worker thread.
    # ------------------------------------------------------------------
    def _list_job_deployments(self, *, include_unlabelled: bool = False) -> list[Any]:
        """List managed job Deployments.

        Opt-in set (``coding-job=true``) by default; the wider managed set
        (labelled + legacy unlabelled) when ``include_unlabelled`` is True.
        """
        selector = ALL_MANAGED_SELECTOR if include_unlabelled else JOB_LABEL_SELECTOR
        return list(
            self.apps_v1.list_namespaced_deployment(
                namespace=self.namespace,
                label_selector=selector,
            ).items
        )

    def count_active_jobs(self) -> int:
        """Count opt-in job Deployments in the namespace (live cluster truth)."""
        return len(self._list_job_deployments())

    def _deployment_age_seconds(self, dep: Any) -> float:
        """Age of a Deployment in seconds.

        Uses the *freshest* (most recent) of the server-set
        ``creation_timestamp`` and the ``created-at`` label epoch. The label is
        re-stamped on every (re)provision and is provisioner-controlled (never
        user-supplied), so honoring it lets an in-place reconcile (patch, which
        keeps the original ``creation_timestamp``) still refresh the TTL lease —
        matching the old delete-recreate behavior where ``creation_timestamp``
        reset. Unknown age or clock-skew (negative) is clamped to 0 → treated as
        fresh, never reaped on uncertainty.
        """
        now = time.time()
        epochs: list[float] = []
        ts = getattr(dep.metadata, "creation_timestamp", None)
        if ts is not None:
            try:
                epochs.append(ts.timestamp())
            except AttributeError, OSError, ValueError, TypeError:
                pass
        label_val = (dep.metadata.labels or {}).get(CREATED_AT_LABEL_KEY)
        if label_val is not None:
            try:
                epochs.append(float(int(label_val)))
            except TypeError, ValueError:
                pass
        if not epochs:
            return 0.0
        age = now - max(epochs)
        if age < 0:
            logger.warning(
                "Deployment %s reports negative age (%.1fs) — clock skew; treating as fresh",
                dep.metadata.name,
                age,
            )
            return 0.0
        return age

    def _is_reapable(
        self, dep: Any, *, ttl_seconds: int, exclude_names: frozenset[str]
    ) -> tuple[bool, str]:
        """Decide whether a Deployment may be reaped.

        Order: allowlist → explicit done-annotation fast-path → age TTL.
        """
        name = dep.metadata.name
        if name in exclude_names:
            return (False, "allowlisted")
        annotations = dep.metadata.annotations or {}
        if str(annotations.get(DONE_ANNOTATION_KEY, "")).lower() == "true":
            return (True, "done-annotation")
        if self._deployment_age_seconds(dep) >= ttl_seconds:
            return (True, "ttl-expired")
        return (False, "fresh")

    def _reap_deployments(
        self,
        deployments: list[Any],
        *,
        ttl_seconds: int,
        exclude_names: frozenset[str],
        dry_run: bool,
        counts: dict[str, Any],
    ) -> None:
        """Apply the reap decision to each Deployment, accumulating into counts.

        A per-Deployment delete failure is isolated (logged, counted) so one
        bad delete never aborts the rest of the sweep.
        """
        for dep in deployments:
            counts["scanned"] += 1
            name = dep.metadata.name
            reapable, reason = self._is_reapable(
                dep, ttl_seconds=ttl_seconds, exclude_names=exclude_names
            )
            if not reapable:
                counts["skipped"] += 1
                continue
            if dry_run:
                counts["would_reap"] += 1
                logger.info("[reaper dry-run] would reap %s (%s)", name, reason)
                continue
            # Graph FIRST, while the Deployment still exists: coding-analysis
            # validates the container for the tenant before dropping its graph
            # and 404s once it is gone, so this is the last moment the graph is
            # reachable. Best-effort — it never raises and never blocks the reap.
            self._delete_reaped_graph(dep, counts)
            try:
                self._cleanup_resources(name)
            except (ApiException, OSError, RuntimeError) as e:
                counts["errors"] += 1
                logger.error("[reaper] failed to reap %s: %s", name, e)
                # The Deployment may still exist. Leave the registry row alone:
                # it is the ONLY copy of this container's repo_url, and
                # ensure_container must still be able to recover from it.
                continue
            counts["reaped"] += 1
            counts["names_reaped"].append(name)
            logger.info("[reaper] reaped %s (%s)", name, reason)
            # Ordering is deliberate: the Deployment delete has SUCCEEDED and is
            # already counted before the registry row is touched, so a registry
            # failure can never mask or un-count a deletion that happened.
            self._forget_reaped_metadata(name, counts)

    @staticmethod
    def _delete_reaped_graph(dep: Any, counts: dict[str, Any]) -> None:
        """Drop a to-be-reaped workspace's code graph in coding-analysis.

        ⭐ WHY HERE. The only other graph-delete path is
        ``ContainerManager.remove_container`` (explicit DELETE). The reaper tears
        workspaces down without it, and coding-analysis's
        ``DELETE /api/analysis/graphs/{name}`` validates the container for the
        tenant FIRST — so once the Deployment is gone the endpoint 404s and the
        rows are orphaned forever (983 graphs / ~1.15M rows measured). The reaper
        is the one caller that KNOWS it is about to delete the workspace, so it
        deletes the graph at the source, before the Deployment goes. No absence
        is inferred anywhere: the name comes from the Deployment in hand.

        Same client as ``remove_container`` (``GraphOperations
        .delete_container_graph``): this service's licence in ``Authorization``
        plus the container OWNER's tenant in ``X-Tenant-Id``. The tenant is the
        registry row's (as in ``remove_container``), else the Deployment's own
        stamp (annotation, then label). That endpoint deletes ``edges`` and
        ``nodes`` only — ``dependency_report_snapshots`` are deliberately left.

        Best-effort: NEVER raises and never blocks the reap (a stuck reaper is
        worse than an orphan). Outcomes are counted; a failure also increments
        ``graph_errors`` (escalates the sweep log) and names the container.
        Called from the reaper's worker thread (``asyncio.to_thread``), where no
        loop is running, so ``asyncio.run`` is safe; on a loop thread it refuses
        rather than deadlocking.
        """
        name = dep.metadata.name
        info = None
        try:
            info = ContainerMetadataRegistry().find_by_name(name)
        except Exception as e:  # noqa: BLE001 - a registry hiccup must not block the reap
            logger.warning("[reaper] registry read for %s failed before graph delete: %r", name, e)
        graph_id = getattr(info, "code_graph_id", "") or name
        tenant_id = getattr(info, "tenant_id", None) or _recover_tenant_id(
            dep.metadata.labels, dep.metadata.annotations
        )

        def _failed(reason: str) -> None:
            counts["graph_errors"] = int(counts.get("graph_errors", 0) or 0) + 1
            counts.setdefault("graph_names_failed", []).append(name)
            logger.error(
                "[reaper] could NOT delete the code graph of %s (tenant=%s): %s — it will be "
                "ORPHANED in coding-analysis once the Deployment is gone; manual cleanup needed",
                name,
                tenant_id,
                reason,
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            _failed("called on a running event loop; refusing to block it")
            return
        try:
            result = asyncio.run(
                get_graph_operations().delete_container_graph(graph_id, tenant_id=tenant_id)
            )
        except Exception as e:  # noqa: BLE001 - best-effort: never abort the reap
            _failed(repr(e))
            return
        if not result.get("success"):
            _failed(str(result.get("message")))
            return
        if result.get("already_absent"):
            counts["graphs_absent"] = int(counts.get("graphs_absent", 0) or 0) + 1
            logger.info("[reaper] no code graph to delete for %s (tenant=%s)", name, tenant_id)
            return
        counts["graphs_deleted"] = int(counts.get("graphs_deleted", 0) or 0) + 1
        logger.info(
            "[reaper] deleted code graph of %s: %s nodes, %s edges",
            name,
            result.get("deleted_nodes", 0),
            result.get("deleted_edges", 0),
        )

    @staticmethod
    def _forget_reaped_metadata(name: str, counts: dict[str, Any]) -> None:
        """Drop the reaped container's registry row once its Deployment is gone.

        ⭐ WHY HERE AND NOT IN ``_cleanup_resources``. The reap is the one place
        where "gone" is a *decision*. ``_cleanup_resources`` itself is SHARED
        with the provision-failure rollback in ``provision_from_github``
        (``except`` → ``self._cleanup_resources(deployment_name)``), and that
        path also runs for a FAILED re-provision driven by
        ``ContainerManager.ensure_container``. Removing the row there would turn
        a transient provisioning error into permanent unrecoverability: the
        registry hash is the SOLE surviving store of ``repo_url`` for a
        container whose Deployment is gone (the ``coding-job/repo-url``
        annotation dies with the Deployment). A reconcile pass must never delete
        what it cannot rebuild.

        The explicit-delete path needs nothing here — it already HDELs the row
        itself in ``ContainerManager.remove_container``.

        Failure semantics: LOUD but never fatal. The Deployment deletion has
        already succeeded and been counted by the caller, so a registry error is
        reported on its own counter (``registry_errors``, which escalates the
        sweep summary to WARNING via ``reaper.sweep_log_level``) and never
        aborts the sweep or the remaining deletions. The residual state on
        failure is exactly today's behaviour — a stale row that
        ``ensure_container`` may resurrect — which is no worse than before, and
        now it is visible instead of silent.

        ⭐ WHY ``remove_with_outcome`` AND NOT ``remove``. ``remove`` returns a
        bool, and its ``False`` conflates two states this method MUST keep
        apart: the row genuinely did not exist, and the HDEL never reached
        Redis. Reporting the second as the first is a positive falsehood about
        a row that still exists — and because the HDEL here is ONE-SHOT (the
        reaper decides off the cluster, so once the Deployment is deleted this
        name is never revisited, and ``save`` — the only other HDEL-of-stale
        path — has no production callers) a lost HDEL is a PERMANENT stale row,
        i.e. exactly the resurrection this method exists to prevent. Neither
        registry failure mode raises, either: ``remove`` swallows
        ``RedisError``/``OSError`` into ``False``, and ``_redis_client``
        swallows everything into ``None``, so an ``except`` clause here cannot
        be the detector. The tri-state is the detector; the ``except`` below is
        only a backstop for an unforeseen raise.

        Idempotent: the underlying HDEL is atomic and a genuinely absent row
        (``ABSENT``) is a no-op, not an error.
        """
        try:
            outcome = ContainerMetadataRegistry().remove_with_outcome(name)
            detail = outcome.value
        except Exception as e:  # noqa: BLE001 - never let registry state abort a sweep
            # Backstop only: neither documented failure mode of the registry
            # raises (see the docstring). Anything landing here is unforeseen,
            # and "unforeseen" cannot be treated as "the row is gone".
            outcome = RemoveOutcome.UNCERTAIN
            detail = repr(e)
        if outcome is RemoveOutcome.REMOVED:
            logger.info("[reaper] removed registry row for reaped container %s", name)
            return
        if outcome is RemoveOutcome.ABSENT:
            # Redis answered "no such field", so this is true rather than assumed.
            logger.info("[reaper] reaped container %s had no registry row (Redis confirmed)", name)
            return
        counts["registry_errors"] = int(counts.get("registry_errors", 0) or 0) + 1
        logger.error(
            "[reaper] reaped %s but could NOT confirm its registry row is gone (%s); "
            "a surviving row is permanent — this name is never revisited — and lets "
            "ensure_container resurrect the container",
            name,
            detail,
        )

    def reap_once(
        self,
        *,
        ttl_seconds: int,
        exclude_names: frozenset[str] | set[str] | None = None,
        dry_run: bool = True,
        legacy: bool = False,
        legacy_ttl: int | None = None,
    ) -> dict[str, Any]:
        """Run one reap sweep and return a counts dict for logging/metrics."""
        exclude = REAP_NEVER_NAMES | frozenset(exclude_names or ())
        counts: dict[str, Any] = {
            "scanned": 0,
            "reaped": 0,
            "skipped": 0,
            "errors": 0,
            "registry_errors": 0,
            "graphs_deleted": 0,
            "graphs_absent": 0,
            "graph_errors": 0,
            "graph_names_failed": [],
            "would_reap": 0,
            "names_reaped": [],
        }
        labelled = self._list_job_deployments()
        self._reap_deployments(
            labelled,
            ttl_seconds=ttl_seconds,
            exclude_names=exclude,
            dry_run=dry_run,
            counts=counts,
        )
        if legacy:
            eff_ttl = legacy_ttl if legacy_ttl is not None else ttl_seconds
            labelled_names = {d.metadata.name for d in labelled}
            unlabelled = [
                d
                for d in self._list_job_deployments(include_unlabelled=True)
                if d.metadata.name not in labelled_names
            ]
            self._reap_deployments(
                unlabelled,
                ttl_seconds=eff_ttl,
                exclude_names=exclude,
                dry_run=dry_run,
                counts=counts,
            )
        return counts

    # ------------------------------------------------------------------
    # Credential refresh: keep every live job's git-token Secret younger
    # than the App installation token's ~50 min lifetime.
    # ------------------------------------------------------------------
    async def refresh_expiring_git_tokens(
        self,
        *,
        max_age_seconds: int,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Re-mint ``<name>-git-token`` for live jobs whose credential is expiring.

        WHY THIS EXISTS. A GitHub App installation token lives ~50 min
        (coding-github mints with ``_TOKEN_TTL_SECONDS = 50 * 60``). The Secret is
        written ONCE at provisioning time, but the ``git-clone`` init container is
        free to run at any later moment: init containers re-run whenever a pod's
        sandbox is recreated (eviction, drain, node reboot), and ``_idempotent``
        re-runs the clone against the fresh emptyDir. Past the lifetime every one
        of those replays clones with a dead credential, which GitHub answers with
        exactly ``Invalid username or token`` — a credential error, so the
        container backs off and retries forever and the workspace never recovers.
        Measured 2026-11-01 on ``example-org/py2rs-c4-svc-rs``: four job pods
        failed this way.

        A job provisioned with a caller BYO PAT (``secret-config-id`` annotation)
        is SKIPPED: that credential is the caller's, is not ours to mint, and
        replacing it with an App token would silently narrow the job's access.

        Fail-closed only where it matters: a mint failure, a missing Secret or an
        unreadable annotation leaves the existing token in place and counts an
        error. It never deletes a Secret and never writes an empty token.
        """
        counts: dict[str, Any] = {
            "scanned": 0,
            "refreshed": 0,
            "fresh": 0,
            "skipped": 0,
            "errors": 0,
            "would_refresh": [],
        }
        now = time.time()
        for dep in self._list_job_deployments():
            name = getattr(getattr(dep, "metadata", None), "name", None)
            if not name:
                continue
            counts["scanned"] += 1
            annotations = getattr(dep.metadata, "annotations", None) or {}

            if annotations.get(SECRET_CONFIG_ANNOTATION):
                # Caller-supplied PAT: not ours to mint.
                counts["skipped"] += 1
                continue

            repo_url = annotations.get(REPO_URL_ANNOTATION)
            if not repo_url:
                # Nothing to mint against; the Deployment predates the annotation
                # or was created by hand. Leave it alone.
                counts["skipped"] += 1
                continue

            secret_name = f"{name}-git-token"
            try:
                secret = self.core_v1.read_namespaced_secret(
                    name=secret_name, namespace=self.namespace
                )
            except ApiException as e:
                if e.status == 404:
                    counts["skipped"] += 1
                    continue
                counts["errors"] += 1
                logger.warning("token refresh: cannot read Secret %s (%s)", secret_name, e.status)
                continue

            minted_at: int | None = None
            raw = (getattr(secret.metadata, "annotations", None) or {}).get(
                GIT_TOKEN_REFRESHED_ANNOTATION
            )
            if raw:
                try:
                    minted_at = int(raw)
                except TypeError, ValueError:
                    minted_at = None
            if minted_at is None:
                # No stamp: fall back to the Secret's age. Unreadable → refresh,
                # which is always safe.
                created = getattr(secret.metadata, "creation_timestamp", None)
                minted_at = int(created.timestamp()) if created is not None else 0

            if max_age_seconds > 0 and (now - minted_at) < max_age_seconds:
                counts["fresh"] += 1
                continue

            if dry_run:
                counts["would_refresh"].append(name)
                continue

            try:
                credential = await get_github_token_manager().get_app_clone_credential(
                    repo_url=repo_url,
                    tenant_id=annotations.get(TENANT_ANNOTATION),
                )
            except Exception:  # noqa: BLE001 — a mint failure must not kill the sweep
                counts["errors"] += 1
                logger.exception("token refresh: mint raised for %s", name)
                continue

            if not credential.ok or not credential.token:
                counts["errors"] += 1
                logger.warning(
                    "token refresh: no App token for %s (repo=%s reason=%s) — "
                    "existing credential left in place",
                    name,
                    repo_url,
                    credential.reason or credential.error or "-",
                )
                continue

            try:
                self.core_v1.patch_namespaced_secret(
                    name=secret_name,
                    namespace=self.namespace,
                    body={
                        "metadata": {
                            "annotations": {GIT_TOKEN_REFRESHED_ANNOTATION: str(int(time.time()))}
                        },
                        "data": {
                            "GIT_TOKEN": base64.b64encode(credential.token.encode("utf-8")).decode(
                                "ascii"
                            )
                        },
                    },
                )
            except ApiException as e:
                counts["errors"] += 1
                logger.warning("token refresh: cannot patch Secret %s (%s)", secret_name, e.status)
                continue

            counts["refreshed"] += 1
            logger.info("Refreshed git-token Secret: %s", secret_name)
        return counts

    # ------------------------------------------------------------------
    # Registry-ghost sweep: rows in ``coding_containers:container_registry``
    # whose Deployment no longer exists.
    # ------------------------------------------------------------------
    def _deployment_presence(self, name: str) -> Literal["present", "absent", "error"]:
        """Read ONE Deployment by name. Only a 404 is "absent"; anything else
        that is not a successful read is "error" — never "absent"."""
        try:
            self.apps_v1.read_namespaced_deployment(name=name, namespace=self.namespace)
        except ApiException as e:
            if e.status == 404:
                return "absent"
            logger.warning("[reaper] ghost check: reading Deployment %s failed: %s", name, e)
            return "error"
        except Exception as e:  # noqa: BLE001 - transport errors etc. are "unknown", not "absent"
            logger.warning("[reaper] ghost check: reading Deployment %s failed: %r", name, e)
            return "error"
        return "present"

    @staticmethod
    def _warn_ghost_graph_left(name: str) -> None:
        """A ghost's code graph (if any) is NOT deleted here, deliberately.

        Its Deployment is already gone, so coding-analysis's delete endpoint —
        which validates the container first — would 404, and a
        validation-bypassing delete is not something this service may do. Keep
        the orphan VISIBLE instead of silent.
        """
        logger.warning(
            "[reaper] registry ghost %s had no Deployment; its code graph (if any) in "
            "coding-analysis could not be deleted and is left for manual cleanup",
            name,
        )

    def reap_registry_ghosts(
        self,
        *,
        dry_run: bool = True,
        grace_seconds: float = 3600.0,
        exclude_names: frozenset[str] | set[str] | None = None,
    ) -> dict[str, Any]:
        """HDEL registry rows whose backing Deployment is CONFIRMED gone.

        The Deployment reaper only forgets the rows of names it reaped itself,
        so rows whose Deployment vanished any other way (reaped before
        ``_forget_reaped_metadata`` existed, a lost HDEL, an out-of-band delete,
        a failed provision) stay in the hash forever, still saying ``running``.

        A row is removed only when ALL of these hold:
          1. Redis answered ``HKEYS`` (``durable_names`` is not ``None``);
          2. the managed-Deployment LIST succeeded and does not contain it;
          3. a per-name READ of the Deployment returned 404 (a Deployment that
             lacks the managed labels is therefore still found and kept);
          4. it has been continuously confirmed absent for ``grace_seconds``
             (first sighting is held in memory, so a restart only restarts the
             clock). The grace keeps ``ContainerManager.ensure_container``'s
             re-provision-from-row recovery available for a recent loss — the
             registry row is the only copy of ``repo_url`` once the Deployment
             is gone, so it is kept for a while, not forever.
        ⛔ FAIL-CLOSED: any Redis or Kubernetes error removes NOTHING — a failed
        read is never evidence of absence. ``dry_run`` only logs.
        """
        counts: dict[str, Any] = {
            "ghosts_seen": 0,
            "ghosts_pending": 0,
            "ghosts_removed": 0,
            "ghosts_would_remove": 0,
            "ghost_errors": 0,
            "ghost_names_removed": [],
        }
        first_seen: dict[str, float] = self.__dict__.setdefault("_registry_ghost_first_seen", {})
        exclude = REAP_NEVER_NAMES | frozenset(exclude_names or ())
        registry = ContainerMetadataRegistry()

        names = registry.durable_names()
        if names is None:
            counts["ghost_errors"] += 1
            logger.warning("[reaper] ghost sweep skipped: registry Redis did not answer")
            return counts
        try:
            live = {
                d.metadata.name
                for d in self._list_job_deployments(include_unlabelled=True)
                if d.metadata is not None and d.metadata.name
            }
        except Exception as e:  # noqa: BLE001 - any list failure means "unknown": delete nothing
            counts["ghost_errors"] += 1
            logger.warning("[reaper] ghost sweep skipped: listing Deployments failed: %r", e)
            return counts

        candidates = sorted(n for n in names if n not in live and n not in exclude)
        for stale in set(first_seen) - set(candidates):
            first_seen.pop(stale, None)

        now = time.monotonic()
        for name in candidates:
            presence = self._deployment_presence(name)
            if presence == "present":
                first_seen.pop(name, None)
                continue
            if presence == "error":
                counts["ghost_errors"] += 1
                continue
            counts["ghosts_seen"] += 1
            since = first_seen.setdefault(name, now)
            if now - since < grace_seconds:
                counts["ghosts_pending"] += 1
                continue
            if dry_run:
                counts["ghosts_would_remove"] += 1
                logger.info("[reaper dry-run] would remove registry ghost %s (no Deployment)", name)
                self._warn_ghost_graph_left(name)
                continue
            try:
                outcome = registry.remove_with_outcome(name)
            except Exception as e:  # noqa: BLE001 - never abort the sweep on registry state
                logger.error("[reaper] removing registry ghost %s raised: %r", name, e)
                outcome = RemoveOutcome.UNCERTAIN
            if outcome is RemoveOutcome.REMOVED:
                first_seen.pop(name, None)
                counts["ghosts_removed"] += 1
                counts["ghost_names_removed"].append(name)
                logger.info(
                    "[reaper] removed registry ghost %s (Deployment confirmed absent)", name
                )
                self._warn_ghost_graph_left(name)
            elif outcome is RemoveOutcome.ABSENT:
                first_seen.pop(name, None)
            else:
                counts["ghost_errors"] += 1
                logger.error("[reaper] could NOT confirm registry ghost %s was removed", name)
        return counts


# Singleton instance
_k8s_provisioner: K8sContainerProvisioner | None = None


def get_k8s_container_provisioner() -> K8sContainerProvisioner:
    """Get or create the K8s container provisioner singleton."""
    global _k8s_provisioner
    if _k8s_provisioner is None:
        _k8s_provisioner = K8sContainerProvisioner()
    return _k8s_provisioner

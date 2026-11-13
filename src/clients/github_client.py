"""GitHub token management for coding-containers.

Owns the App-backed **clone credential** lookup: coding-github holds the
GitHub App installation, so it is the only service that can mint a short-lived
installation token (``ghs_...``) for a repo and authoritatively report that
repo's visibility. coding-containers needs both — the token to clone a PRIVATE
repo without a BYO PAT, and the visibility to know that a tokenless clone of a
PUBLIC repo is legitimate rather than a misconfiguration.

Also owns ``store_user_token``, which persists a BYO PAT. That one does **not**
go to coding-github: the store lives in the secrets service (see
``src/services/secret_resolver.py``). The module-level ``CodingGitHubClient``
singleton that used to back it is gone with it — the clone-credential lookup is
a direct ``httpx`` call against ``/api/internal/github/clone-token``, so nothing
else here needed the generated client.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple
from urllib.parse import unquote, urlparse

import httpx

from src.clients._tenant import optional_tenant_headers, service_token
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


# The clone-token call sits on the provisioning hot path; a short-lived App
# token mint is a couple of GitHub API round-trips.
_CLONE_TOKEN_TIMEOUT_SECONDS = 10.0

Visibility = Literal["public", "private"]

RefHeadState = Literal["present", "absent", "unknown", "empty_repository"]
_REF_HEAD_STATES: frozenset[str] = frozenset({"present", "absent", "unknown", "empty_repository"})

#: A full commit id (SHA-1 or SHA-256) as the GitHub API prints it.
_FULL_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class RefHead(NamedTuple):
    """``refs/heads/<branch>`` on GitHub, as coding-github read it.

    ``present`` always carries a full commit ``sha``; every other state carries
    none. ``unknown`` is a failed lookup and must refuse, never read as ``absent``.
    """

    state: RefHeadState
    sha: str | None = None


def _parse_ref_head(raw: Any) -> RefHead | None:
    """Strictly parse a coding-github ``{state, sha}`` object; ``None`` if absent.

    Anything malformed (unknown state, ``present`` without a full sha) is
    ``unknown`` — the caller refuses on it.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return RefHead("unknown")
    state = raw.get("state")
    sha = raw.get("sha")
    if state not in _REF_HEAD_STATES:
        return RefHead("unknown")
    if state == "present":
        if isinstance(sha, str) and _FULL_SHA_RE.match(sha):
            return RefHead("present", sha)
        return RefHead("unknown")
    return RefHead(state)


# Keep an error body short enough to sit on one log line without truncating the
# sentence that follows it.
_DETAIL_CHARS = 300


def _response_detail(response: httpx.Response, limit: int = _DETAIL_CHARS) -> str:
    """Best-effort one-line summary of an error body, for a log line.

    **Never raises.** A diagnostic that can itself fail is a diagnostic that goes
    missing exactly when it is needed — and this one runs inside an ``except``
    block, where an exception would replace a useful 403 report with a traceback
    about the reporting.

    Prefers the keys the fleet's error envelopes actually use (``error`` carries
    the machine-readable code, e.g. ``TENANT_ASSERTION_NOT_PERMITTED``; ``detail``
    the prose), and falls back to raw text so a proxy's HTML 403 is still visible
    rather than silently becoming an empty string.
    """
    try:
        payload = response.json()
    except ValueError, UnicodeDecodeError:
        payload = None
    if isinstance(payload, dict):
        named = [
            f"{key}={payload[key]!r}" for key in ("error", "detail", "message") if key in payload
        ]
        if named:
            return " ".join(named)[:limit]
    if payload is not None:
        return str(payload)[:limit]
    try:
        text = (response.text or "").strip()
    except Exception:  # noqa: BLE001 - a body that cannot even be decoded must not mask the 403
        return "<undecodable body>"
    return text[:limit] if text else "<empty body>"


APP_NOT_INSTALLED_ERROR = "github_app_not_installed"
PUSH_CREDENTIAL_ERROR = "github_app_push_credential_unavailable"
#: coding-github's clone-token ``reason`` when a PLATFORM-owned repo is asked for
#: by a tenant outside ``CODING_GITHUB_PLATFORM_TENANTS`` (coding-github #107).
#: It is a REFUSAL, not a missing credential: no caller here may answer it by
#: reaching for the system PAT, which would re-grant exactly what was refused.
PLATFORM_REPO_FORBIDDEN = "platform_repo_forbidden"


#: The only host a GitHub remote may name. Exact match after lowercasing:
#: ``www.github.com``, ``github.com.`` and every GitHub Enterprise host are
#: refused, because the credential chosen from the owner is a github.com one.
GITHUB_HOST = "github.com"

# scp-like ``user@host:path`` (no scheme). ``git`` is the only user GitHub accepts.
_SCP_REMOTE = re.compile(r"^(?P<user>[^@/:]+)@(?P<host>[^:/]+):(?P<path>.*)$")
# GitHub's own charsets: owners are alnum + hyphen (<=39), repos alnum + ``._-``.
_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
_REPO_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


def _has_control_chars(value: str) -> bool:
    return any(ord(c) <= 0x20 or ord(c) == 0x7F for c in value)


def canonical_github_repo(repo_url: str | None) -> tuple[str, str] | None:
    """Canonicalise a GitHub remote to ``(owner, repo)``; ``None`` if it is not one.

    Accepts exactly three shapes — ``https://github.com/o/r``,
    ``ssh://git@github.com[:22]/o/r`` and scp ``git@github.com:o/r`` — each with
    an optional ``.git`` and one trailing ``/``. The path is percent-decoded ONCE
    and must then be exactly two segments (an encoded ``/`` is refused), neither
    empty, ``.``, ``..`` nor containing ``..``, with no control characters,
    backslashes, residual ``%`` (double encoding), userinfo on https, query or
    fragment.

    Why this strict: git (via curl) resolves dot segments and percent escapes
    before it talks to GitHub, so ``https://github.com/nonagenticai/../acme/r``
    and ``.../nonagenticai/%2e%2e/acme/r`` REACH ``acme/r`` while a naive
    first-segment parse calls the owner ``nonagenticai``. Any string whose
    meaning to git could differ from its meaning here is refused, not repaired.
    """
    if not repo_url:
        return None
    raw = str(repo_url).strip()
    if not raw or _has_control_chars(raw) or "\\" in raw or "?" in raw or "#" in raw:
        return None

    scp = None if "://" in raw else _SCP_REMOTE.match(raw)
    if scp:
        if scp.group("user") != "git":
            return None
        host = scp.group("host")
        path = scp.group("path")
        if path.startswith("/"):
            return None
    else:
        parsed = urlparse(raw)
        scheme = parsed.scheme.lower()
        try:
            port = parsed.port
        except ValueError:
            return None
        if scheme == "https":
            if parsed.username is not None or parsed.password is not None or port is not None:
                return None
        elif scheme == "ssh":
            if parsed.username != "git" or parsed.password is not None or port not in (None, 22):
                return None
        else:
            return None
        if parsed.params:
            return None
        host = parsed.hostname or ""
        if not parsed.path.startswith("/"):
            return None
        path = parsed.path[1:]

    if host.lower() != GITHUB_HOST:
        return None

    decoded = unquote(path)
    if "%" in decoded or "\\" in decoded or _has_control_chars(decoded):
        return None
    # An encoded ``/`` (``nonagenticai%2Fr``) would become a segment boundary
    # here that the raw string does not have: what git sends and what we
    # parse would disagree about where the owner ends. Refused, not repaired.
    if decoded.count("/") != path.count("/"):
        return None
    if decoded.endswith("/"):
        decoded = decoded[:-1]
    segments = decoded.split("/")
    if len(segments) != 2:
        return None
    if any(seg in ("", ".", "..") or ".." in seg for seg in segments):
        return None
    owner, repo = segments
    if repo.lower().endswith(".git"):
        repo = repo[: -len(".git")]
    if repo in ("", ".", "..") or not _REPO_RE.match(repo) or not _OWNER_RE.match(owner):
        return None
    return owner, repo


def canonical_github_url(owner: str, repo: str) -> str:
    """The one URL form a push or clone is ever sent to."""
    return f"https://{GITHUB_HOST}/{owner}/{repo}"


def same_github_repo(a: tuple[str, str] | None, b: tuple[str, str] | None) -> bool:
    """GitHub owner and repo names are case-insensitive; ``None`` never matches."""
    if a is None or b is None:
        return False
    return a[0].lower() == b[0].lower() and a[1].lower() == b[1].lower()


def repo_owner_from_url(repo_url: str | None) -> str:
    """Return the canonical GitHub owner of ``repo_url``; ``""`` unless it canonicalises.

    Fail-closed on purpose: ``""`` is never a platform owner, so a remote that
    does not canonicalise (dot segments, ``%2e%2e``, a foreign host) can never
    select the platform PAT.
    """
    canonical = canonical_github_repo(repo_url)
    return canonical[0] if canonical else ""


def is_platform_owner(owner: str | None) -> bool:
    """True iff ``owner`` is one of the platform org(s) (``platform_github_owners``).

    Only a platform owner may be cloned or pushed with the system PAT. Every
    other owner is a customer, whose token must come from a BYO PAT or a
    public-App installation bound to the requesting tenant — never the PAT.
    An empty/unparseable owner is NOT a platform owner (fail closed).
    """
    if not owner:
        return False
    allowed = {
        o.strip().lower()
        for o in (get_settings().platform_github_owners or "").split(",")
        if o.strip()
    }
    return owner.strip().lower() in allowed


@dataclass(frozen=True)
class CloneCredential:
    """Outcome of the coding-github clone-token lookup.

    ``ok`` is the honest distinction the old fail-open-to-``None`` erased:

    * ``ok=True,  token="ghs_..."`` — the App is installed on the owner and
      minted a token; clone authenticated.
    * ``ok=True,  token=None, visibility="public"`` — a *legitimate* null: the
      App has no token for this repo but the repo is public, so no credential
      is needed. This is the case the service could never express before.
    * ``ok=True,  token=None, visibility="private"`` — legitimate null, but the
      repo is private: the caller has no way to clone and must fail loudly.
    * ``ok=False`` — the *call* failed (unset URL, network, 4xx/5xx). Nothing is
      known about the repo, least of all that it is safe to clone tokenless.
    """

    ok: bool
    token: str | None = None
    visibility: Visibility | None = None
    source: str = "none"
    error: str | None = None
    # coding-github's own explanation for a null token (``tenant_required``,
    # ``not_installed``, ``not_configured``, ``mint_failed``). Optional on the
    # wire; ``None`` means "not stated", never "fine".
    reason: str | None = None
    # ── Push-scope repository facts (coding-github >= the clone-token push-facts
    # change). Read from GitHub BY coding-github with the minted token: this pod
    # has no egress to api.github.com, and the workspace must never be the source.
    # ``None`` = not supplied (an older coding-github, a non-push scope, no token):
    # the push route treats every ``None`` as unknown and refuses.
    private: bool | None = None
    default_branch: str | None = None
    branch_head: RefHead | None = None
    default_branch_head: RefHead | None = None

    @property
    def is_public(self) -> bool:
        """True only when coding-github positively said 'public'."""
        return self.ok and self.visibility == "public"


class GitHubTokenManager:
    """GitHub token management - delegates to coding-github service."""

    def get_system_owner(self) -> str | None:
        """Return the GitHub system owner from GITHUB_OWNER env var."""
        owner = os.environ.get("GITHUB_OWNER") or os.environ.get("GITHUB_SYSTEM_OWNER")
        if not owner:
            logger.warning("No GITHUB_OWNER configured")
        return owner

    async def get_app_clone_credential(
        self,
        repo_url: str,
        tenant_id: str | None = None,
        scope: Literal["clone", "push"] = "clone",
        branch: str | None = None,
    ) -> CloneCredential:
        """Ask coding-github for an App-backed clone token + the repo visibility.

        Contract (coding-github)::

            POST {CODING_GITHUB_URL}/api/internal/github/clone-token
              headers: X-Internal-Token, Authorization (this service's licence)
              body:    {"repo_url": "https://github.com/owner/repo",
                        "tenant_id": "<provisioning tenant>"}   # tenant_id optional

              200 →    {"token": "ghs_..."|null, "expires_at": "<iso>"|null,
                        "visibility": "public"|"private", "source": "app"|"none"}
              403 →    EITHER the tenant middleware refused the request before the
                       route ran (``TENANT_ASSERTION_NOT_PERMITTED``) OR the
                       X-Internal-Token shared secret is wrong. Only the response
                       body distinguishes them — see the 403 branch below.

        Never raises: a failure is returned as ``ok=False`` so the caller decides
        the policy. But — unlike ``resolve_git_token``'s silent fail-open — every
        failure is logged at ERROR with the reason, because a silent ``None`` here
        is exactly what made the original no-PAT 500 unreadable.

        ``tenant_id`` travels in the JSON BODY, never as an ``X-Tenant-Id``
        header (see the comment on ``headers`` below for the measured reason).
        coding-github requires it for any repo owner outside the platform org:
        it only mints for a public-App installation bound to THAT tenant, and
        answers ``token: null, reason: "tenant_required"`` without it. It is a
        body field of a route guarded by the internal token, not an identity
        assertion, so the tenant middleware never sees it.

        ``scope="push"`` asks for a repo-scoped ``contents:write`` installation
        token (used by git_push for a customer repo). It is sent only when it is
        not the default, so the clone wire format is unchanged. For a non-platform
        owner coding-github mints it ONLY for an installation bound to
        ``tenant_id`` — the same tenant-bound rule as a clone.

        For ``scope="push"`` the answer also carries the repository facts the push
        route refuses on — ``private``, ``default_branch`` and, when ``branch`` is
        given, ``branch_head``/``default_branch_head`` — all read from GitHub by
        coding-github. ``branch`` is sent only for the push scope.
        """
        settings = get_settings()
        base_url = (settings.coding_github_url or "").rstrip("/")
        if not base_url:
            logger.error(
                "coding_github_url is unset — cannot look up an App clone token for %s. "
                "Set CODING_GITHUB_URL on the Deployment (see the deployment manifests); "
                "without it a project with no BYO PAT can never be provisioned.",
                repo_url,
            )
            return CloneCredential(ok=False, error="coding_github_url_unset")

        # Two credentials on one request, intentionally: the shared secret proves
        # "an internal caller", the licence proves *which* internal caller.
        # X-Internal-Token STAYS — it is what ``require_internal_token`` compares,
        # and removing it 403s every lookup.
        #
        # X-Tenant-Id is NEVER sent here, even when the provision HAS a tenant.
        # coding-github mounts AuthMiddleware + TenantMiddleware app-wide, and its
        # PUBLIC_PATHS covers only /, /health*, /metrics, /docs, /openapi.json and
        # /redoc — so /api/internal is behind the tenant middleware, which runs
        # BEFORE the route's Depends(require_internal_token). Asserting a tenant
        # other than the one our own licence names is refused there with
        # 403 TENANT_ASSERTION_NOT_PERMITTED unless this service's subject is in
        # coding-github's CODING_ONBEHALF_SUBJECTS — and it is not (coding-analysis
        # and coding-realtime list it; coding-github never did).
        #
        # Measured live 2026-08-15 22:37/22:49/23:00: three provisions of
        # enterprise_mcp_server 403'd here, POST /api/containers/ returned 412, the
        # container was never created, and every downstream chore step 404'd. Both
        # shared secrets were byte-identical the whole time — the secret gate was
        # never even reached.
        #
        # The tenant still reaches coding-github — in the request BODY, which the
        # route reads to pick the tenant-bound public-App installation for a
        # customer repo. This keeps the licence (identity), drops only the header
        # assertion (the "absent" row of ``resolve_tenant_assertion`` — accepted
        # regardless of the allowlist), and carries the tenant as data.
        headers = {
            "X-Internal-Token": settings.internal_service_token,
            **optional_tenant_headers(None, service_token=service_token()),
        }
        url = f"{base_url}/api/internal/github/clone-token"
        try:
            async with httpx.AsyncClient(timeout=_CLONE_TOKEN_TIMEOUT_SECONDS) as http:
                body: dict[str, str] = {"repo_url": repo_url}
                if tenant_id:
                    body["tenant_id"] = tenant_id
                if scope != "clone":
                    body["scope"] = scope
                if scope == "push" and branch:
                    body["branch"] = branch
                response = await http.post(url, json=body, headers=headers)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 403:
                # Report what the SERVER said, not what this client guesses. The
                # previous version of this line named the shared secret as THE
                # cause for every 403; on 2026-08-15 the cause was a refused
                # tenant assertion and both secrets were provably identical, and
                # that one hard-coded string sent a debugging session hunting a
                # secret mismatch that did not exist. An error message written by
                # the caller describes what its author expected, not what the
                # callee did.
                logger.error(
                    "coding-github rejected the clone-token call for %s (tenant=%s) with "
                    "403: %s — TWO independent gates return 403 on this path: the tenant "
                    "middleware (refuses BEFORE the route runs; look for "
                    "TENANT_ASSERTION_NOT_PERMITTED) and the X-Internal-Token shared "
                    "secret (CODING_CONTAINERS_INTERNAL_SERVICE_TOKEN here vs "
                    "CODING_GITHUB_INTERNAL_SERVICE_TOKEN there). The body above says "
                    "which; do not assume the secret.",
                    repo_url,
                    tenant_id or "<none>",
                    _response_detail(exc.response),
                )
            else:
                logger.error(
                    "coding-github clone-token call for %s failed with HTTP %s: %s",
                    repo_url,
                    status,
                    exc,
                )
            return CloneCredential(ok=False, error=f"http_{status}")
        except (httpx.HTTPError, ValueError) as exc:
            # ValueError covers a 200 with a non-JSON body (response.json()).
            logger.error(
                "coding-github clone-token call for %s failed: %s — repository "
                "visibility is UNKNOWN, so no tokenless clone can be attempted.",
                repo_url,
                exc,
            )
            return CloneCredential(ok=False, error=type(exc).__name__)

        token = payload.get("token") or None
        raw_visibility = payload.get("visibility")
        visibility: Visibility | None = (
            raw_visibility if raw_visibility in ("public", "private") else None
        )
        source = str(payload.get("source") or "none")
        reason = payload.get("reason") or None

        if visibility is None:
            logger.error(
                "coding-github clone-token response for %s carried no usable visibility "
                "(got %r); treating the lookup as failed.",
                repo_url,
                raw_visibility,
            )
            return CloneCredential(
                ok=False, token=token, source=source, error="bad_visibility", reason=reason
            )

        if token:
            logger.info(
                "coding-github minted an App clone token for %s (visibility=%s, source=%s).",
                repo_url,
                visibility,
                source,
            )
        else:
            # A legitimate null — NOT an error. Public repos need no credential;
            # private ones are about to fail loudly with an actionable message.
            logger.warning(
                "coding-github has no App token for %s (visibility=%s, source=%s, "
                "reason=%s, tenant=%s).",
                repo_url,
                visibility,
                source,
                reason or "-",
                tenant_id or "<none>",
            )

        raw_private = payload.get("private")
        raw_default = payload.get("default_branch")
        return CloneCredential(
            ok=True,
            token=token,
            visibility=visibility,
            source=source,
            reason=reason,
            # Strictly a bool / a non-empty str: anything else is unknown.
            private=raw_private if isinstance(raw_private, bool) else None,
            default_branch=raw_default if isinstance(raw_default, str) and raw_default else None,
            branch_head=_parse_ref_head(payload.get("branch_head")),
            default_branch_head=_parse_ref_head(payload.get("default_branch_head")),
        )

    async def store_user_token(
        self,
        user_id: str,
        token: str,
        container_id: str,
        tenant_id: str | None = None,
    ) -> str:
        """Store the user's GitHub PAT in the secrets service; return its ``secret_id``.

        The returned id is an opaque secrets-service reference and belongs on
        ``ContainerInfo.secret_config_id``, where ``resolve_git_token`` reads it
        back for the clone, the fork-push and the upstream PR. It is NOT a
        coding-github "token id" — that concept never existed outside this
        method's own docstring.

        What this replaces
        ------------------
        This used to POST the raw PAT to ``CodingGitHubClient.create_webhook`` →
        ``/api/repositories/webhooks/webhook``, which is coding-github's INBOUND
        GitHub webhook *delivery receiver* (``routers/github_webhooks.py``), not a
        token-storage route. It rejects anything without a valid
        ``X-Hub-Signature-256`` HMAC, and coding-github exposes no store-user-token
        endpoint at all — its own ``GitHubTokenManager.store_user_token``
        (``services/github_token_manager.py``) is process-local in memory and is
        never mounted on a router. So the call 4xx'd on every provision. The error
        was caught and ``None`` returned, which the three call sites wrote to
        ``ContainerInfo.user_token_id`` — a field nothing in this service ever
        reads back. Storing succeeded exactly never, and nothing surfaced it.

        The secrets service is the correct home and needed no new endpoint:
        ``POST /secrets`` already upserts on ``(subject_id, identifier)``, encrypts
        with Fernet at rest, and is the same store fr writes project tokens to.

        Raises:
            RuntimeError / httpx.HTTPError: propagated from
            :func:`~src.services.secret_resolver.store_git_token`. A failed store is
            NOT degraded to ``None`` here. The caller is holding a raw PAT that the
            user asked to be persisted; a soft failure produces a container whose
            private-repo clone or push will fail later with a message that points at
            git rather than at the store that never happened.
        """
        from src.services.secret_resolver import store_git_token

        logger.info(
            "Storing the GitHub token for user_id=%s container_id=%s in secrets-service.",
            user_id,
            container_id,
        )
        return await store_git_token(
            container_id=container_id,
            token=token,
            tenant_id=tenant_id,
        )

    async def resolve_provision_secret_config_id(
        self,
        *,
        access_token: str | None,
        existing_secret_config_id: str | None,
        user_id: str,
        container_id: str,
        tenant_id: str | None,
    ) -> str | None:
        """Return the ``secret_config_id`` to persist on this provision's ContainerInfo.

        One method for all three provisioners (docker, github-fork, kubernetes) so
        the precedence below is decided once. Three independent copies is how the
        three call sites drifted into three slightly different log messages around
        the same broken store.

        Precedence:

        1. ``existing_secret_config_id`` — the caller passed a ``secret_config_id``,
           so the token is already in the secrets service (fr's
           ``project:{id}:github_token``) and ``crud_write`` resolved
           ``access_token`` out of it on the way in. Storing it again would put the
           same secret in a second row under a second identifier, doubling the
           rotation surface for no gain. Reference the existing row.
        2. ``access_token`` with no existing id — an inline BYO PAT. Store it via
           :meth:`store_user_token` and return the new ``secret_id``.
        3. Neither — ``None``. Nothing to persist; the clone falls through to the
           App clone-token / system-token rungs.

        Raises:
            RuntimeError / httpx.HTTPError: from :meth:`store_user_token` in case 2.
                Not caught. A provision that cannot persist the user's token must
                not report success.
        """
        if existing_secret_config_id:
            logger.info(
                "Provision of %s references the caller-supplied secret_config_id; "
                "not re-storing the token.",
                container_id,
            )
            return existing_secret_config_id
        if not (access_token and access_token.strip()):
            return None
        return await self.store_user_token(
            user_id=user_id,
            token=access_token.strip(),
            container_id=container_id,
            tenant_id=tenant_id,
        )


_instance: GitHubTokenManager | None = None


def get_github_token_manager() -> GitHubTokenManager:
    global _instance
    if _instance is None:
        _instance = GitHubTokenManager()
    return _instance

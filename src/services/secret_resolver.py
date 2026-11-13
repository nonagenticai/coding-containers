"""Store and resolve the per-tenant GIT_TOKEN in the secrets service.

BYO (bring-your-own) token support: a project supplies its GitHub PAT through
the project request; fr stores it in the secrets service and passes only the
opaque ``secret_config_id`` (the project's ``github_token_secret_id``) down the
provisioning chain. coding-containers must turn that reference into the real
token to clone PRIVATE repos — the token is never logged and never travels in
plaintext through the orchestration layer.

``resolve_git_token`` is the READ half and mirrors coding-github's
``GitHubProvisioningService._get_environment_variables``
(coding/coding-github/src/services/github_provisioning.py): a tenant-scoped GET
against the secrets service, reading ``GIT_TOKEN`` from the merged
``env_vars`` + ``secrets`` blob. All failures are swallowed (return ``None``)
so PUBLIC-repo provisioning, which needs no token, is never blocked.

``store_git_token`` is the WRITE half, and it is the half that was missing.
When a caller provisions with an inline ``access_token`` rather than a
``secret_config_id``, that raw PAT used to be POSTed to coding-github's
*inbound GitHub webhook receiver* (``/api/repositories/webhooks/webhook``),
which has never been a token-storage route. The secrets service is the correct
home — it is the same store fr already writes to
(``fr/backend/src/integrations/secrets.ts storeProjectGithubToken``), it
encrypts at rest with Fernet, and it partitions by ``subject_id`` + ``tenant_id``.
The two halves are now symmetric: what ``store_git_token`` writes,
``resolve_git_token`` reads back.

The failure policies are deliberately OPPOSITE:

* a failed READ degrades to ``None`` — a public repo needs no token, so
  blocking provisioning on a secrets-service blip would be a regression;
* a failed WRITE **raises**. There is nothing to degrade to. The caller holds a
  raw PAT that the user expects to be persisted, and returning a plausible id
  for a store that did not happen is precisely the defect this replaces.
"""

from __future__ import annotations

import logging

import httpx

from src.clients._tenant import optional_tenant_headers
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 5.0

# The secrets service defaults ``ttl`` to 3600s (1h) and HARD-DELETES the row on
# expiry. A container is provisioned and then pushed from minutes-to-hours later,
# so a 1h token self-deletes mid-drive and the push fails with "could not read
# Password". fr hit exactly this and root-caused it on 2026-07-04
# (fr/backend/src/integrations/secrets.ts:96-100). 604800s (7d) is the schema
# maximum (``SecretData.ttl`` is ``ge=60, le=604800``); a re-provision refreshes it.
_TOKEN_TTL_SECONDS = 604800


def git_token_identifier(container_id: str) -> str:
    """Return the secrets-service ``identifier`` for a container's GIT_TOKEN.

    ``POST /secrets`` upserts on ``(subject_id, identifier)``, so the identifier
    must be deterministic for a re-provision of the same container to refresh the
    row rather than accumulate duplicates.

    The ``container:`` prefix is what keeps this namespace disjoint from fr's
    ``project:{project_id}:github_token`` rows. Both services write into the same
    ``subject_id`` (the tenant) and an upsert key collision would mean one service
    silently overwriting the other's token — so the prefix is load-bearing, not
    decoration. Container names are k8s-sanitised
    (``^[a-zA-Z][a-zA-Z0-9\\-_]{0,62}$``) and cannot contain ``:``, so no container
    id can be crafted to impersonate a ``project:`` identifier.
    """
    return f"container:{container_id}:github_token"


async def store_git_token(
    container_id: str,
    token: str,
    tenant_id: str | None,
) -> str:
    """Store ``token`` as the container's GIT_TOKEN and return its ``secret_id``.

    The returned id is the value to persist on ``ContainerInfo.secret_config_id``
    — it is exactly what ``resolve_git_token`` takes back. Persisting it anywhere
    that is never read (the old ``user_token_id``-only path) leaves the token
    unreachable and is the same dead end as not storing it at all.

    Args:
        container_id: Names the row via :func:`git_token_identifier`.
        token: The raw PAT. Never logged, here or anywhere downstream.
        tenant_id: Scopes the row. Sent as a BARE ``X-Tenant-Id`` value (not
            ``Bearer ...``) — matching fr, which is the reference writer.

    Returns:
        The opaque ``secret_id`` minted by the secrets service.

    Raises:
        RuntimeError: ``secrets_api_url`` is unset, or the response carried no
            ``secret_id``. Both mean the token is not retrievable.
        httpx.HTTPError: transport failure or a non-2xx status
            (``raise_for_status``). Notably a ``403 RBAC: access denied`` from the
            secrets waypoint, which is what this whole path silently ate before
            ``coding-containers`` was added to ``secrets-write-access``.
    """
    secrets_url = (get_settings().secrets_api_url or "").rstrip("/")
    if not secrets_url:
        raise RuntimeError(
            "Cannot store the git token: secrets_api_url is unset. Refusing to "
            "report a stored token that was never stored — the caller holds a raw "
            "PAT and would otherwise persist an id that resolves to nothing."
        )

    identifier = git_token_identifier(container_id)
    if not tenant_id:
        # Same policy as the read half: absent is absent, never a literal. A
        # tenantless write lands in whatever partition the secrets service assigns
        # a caller with no header, and a later tenant-scoped read will not find it.
        logger.warning(
            "Storing %s with no tenant; sending no X-Tenant-Id rather than a "
            "fabricated one. The row will not be readable by a tenant-scoped GET.",
            identifier,
        )

    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{secrets_url}/secrets",
            headers={
                "Content-Type": "application/json",
                **optional_tenant_headers(tenant_id),
            },
            # FLAT body. The old path called ``CodingGitHubClient.create_webhook``,
            # whose signature is ``(**kwargs) -> post(json=kwargs)``, so passing
            # ``json={...}`` double-wrapped it into ``{"json": {...}}`` on the wire.
            # Building the dict here makes that shape unrepresentable.
            json={
                "identifier": identifier,
                "secrets": {"GIT_TOKEN": token},
                "env_vars": {},
                "ttl": _TOKEN_TTL_SECONDS,
            },
        )
    response.raise_for_status()
    payload = response.json()
    secret_id = (payload or {}).get("secret_id") if isinstance(payload, dict) else None
    if not secret_id:
        raise RuntimeError(
            f"secrets-service accepted the store of {identifier} with "
            f"{response.status_code} but returned no secret_id "
            f"(body keys: {sorted(payload) if isinstance(payload, dict) else type(payload).__name__}). "
            "Without an id the token cannot be resolved back, so this is a failure."
        )
    logger.info(
        "Stored GIT_TOKEN for %s in secrets-service as secret_id=%s (ttl=%ss).",
        identifier,
        secret_id,
        _TOKEN_TTL_SECONDS,
    )
    return str(secret_id)


async def resolve_git_token(
    secret_config_id: str | None,
    tenant_id: str | None,
) -> str | None:
    """Return the GIT_TOKEN for ``secret_config_id`` from the secrets service.

    Returns ``None`` (never raises) when no id is given, the secrets URL is
    unset, the secret is missing, or the fetch fails — the caller then proceeds
    without a token (correct for public repos; private clones will fail loudly
    at the git step, same as before).
    """
    if not secret_config_id:
        return None

    secrets_url = (get_settings().secrets_api_url or "").rstrip("/")
    if not secrets_url:
        logger.warning(
            "secrets_api_url is unset; cannot resolve secret_config_id=%s "
            "(private repos will fail to clone).",
            secret_config_id,
        )
        return None

    # A missing tenant omits the header; it never substitutes a literal. The old
    # ``tenant_id or "anonymous"`` attributed the lookup to a tenant named
    # ``anonymous`` that no registry issues — a fabricated caller identity on a
    # request whose whole purpose is to read a TENANT-SCOPED secret. Omitting is
    # honest and lets the secrets service apply its own no-tenant policy; the
    # admin path (``user.is_admin()`` → ``tenant_id=None``) genuinely has no
    # tenant, so refusing outright would be a regression.
    if not tenant_id:
        logger.warning(
            "Resolving secret_config_id=%s with no tenant; sending no X-Tenant-Id "
            "rather than a fabricated one. A tenant-scoped secret will not resolve.",
            secret_config_id,
        )
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.get(
                f"{secrets_url}/secrets/{secret_config_id}",
                headers=optional_tenant_headers(tenant_id),
            )
        if response.status_code == 404:
            logger.warning(
                "Secret config %s not found in secrets-service; "
                "provisioning will proceed without token (private repos will fail).",
                secret_config_id,
            )
            return None
        response.raise_for_status()
        payload = response.json()
        # Keys for the git-clone step live under ``secrets`` by convention
        # (GIT_TOKEN); ``env_vars`` is merged first so ``secrets`` overrides.
        combined = {**(payload.get("env_vars") or {}), **(payload.get("secrets") or {})}
        token = combined.get("GIT_TOKEN")
        if token:
            logger.info(
                "Resolved secret_config_id=%s → GIT_TOKEN for clone auth.",
                secret_config_id,
            )
            return token
        logger.info(
            "Secret config %s had no GIT_TOKEN; nothing to inject.",
            secret_config_id,
        )
        return None
    except httpx.HTTPError as exc:
        logger.warning(
            "Secrets-service fetch failed for config_id=%s: %s — "
            "continuing without token (private repos will fail).",
            secret_config_id,
            exc,
        )
        return None

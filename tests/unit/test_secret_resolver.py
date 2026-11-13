"""Unit tests for BYO secret_config_id → GIT_TOKEN resolution."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.config.settings import get_settings
from src.services.secret_resolver import (
    git_token_identifier,
    resolve_git_token,
    store_git_token,
)

_BASE = get_settings().secrets_api_url.rstrip("/")


async def test_returns_none_without_secret_config_id() -> None:
    assert await resolve_git_token(None, "tenant-1") is None
    assert await resolve_git_token("", "tenant-1") is None


@respx.mock
async def test_resolves_git_token_and_sends_tenant_header() -> None:
    route = respx.get(f"{_BASE}/secrets/cfg-abc").mock(
        return_value=httpx.Response(200, json={"secrets": {"GIT_TOKEN": "ghp_xyz"}})
    )
    token = await resolve_git_token("cfg-abc", "tenant-42")
    assert token == "ghp_xyz"
    assert route.called
    assert route.calls.last.request.headers["X-Tenant-Id"] == "tenant-42"


@respx.mock
async def test_missing_tenant_omits_the_header_instead_of_inventing_one() -> None:
    """No tenant ⇒ no ``X-Tenant-Id`` at all — never the literal ``"anonymous"``.

    Reads a TENANT-SCOPED secret, so a fabricated tenant here is a caller identity
    the secrets service cannot distinguish from a real tenant named ``anonymous``.
    """
    route = respx.get(f"{_BASE}/secrets/cfg-no-tenant").mock(
        return_value=httpx.Response(200, json={"secrets": {"GIT_TOKEN": "ghp_x"}})
    )

    await resolve_git_token("cfg-no-tenant", None)

    assert route.called
    sent = route.calls.last.request.headers
    assert "X-Tenant-Id" not in sent, (
        f"expected no tenant header, got {sent.get('X-Tenant-Id')!r} — a fabricated identity"
    )


@respx.mock
async def test_secrets_override_env_vars() -> None:
    respx.get(f"{_BASE}/secrets/cfg-merge").mock(
        return_value=httpx.Response(
            200,
            json={"env_vars": {"GIT_TOKEN": "from_env"}, "secrets": {"GIT_TOKEN": "from_secrets"}},
        )
    )
    assert await resolve_git_token("cfg-merge", "t") == "from_secrets"


@respx.mock
async def test_404_returns_none() -> None:
    respx.get(f"{_BASE}/secrets/missing").mock(return_value=httpx.Response(404))
    assert await resolve_git_token("missing", "t") is None


@respx.mock
async def test_no_git_token_in_payload_returns_none() -> None:
    respx.get(f"{_BASE}/secrets/cfg-empty").mock(
        return_value=httpx.Response(200, json={"secrets": {"OTHER": "x"}})
    )
    assert await resolve_git_token("cfg-empty", "t") is None


@respx.mock
async def test_http_error_returns_none() -> None:
    respx.get(f"{_BASE}/secrets/cfg-boom").mock(side_effect=httpx.ConnectError("boom"))
    assert await resolve_git_token("cfg-boom", "t") is None


# ---------------------------------------------------------------------------
# WRITE half — store_git_token
#
# The read half degrades to None on every failure, which is correct for it: a
# public repo needs no token. The write half must do the OPPOSITE. Every test
# below that asserts a raise exists because the code it replaces returned a
# plausible id (or None) for a store that never happened, and three provisioners
# logged that as success.
# ---------------------------------------------------------------------------


def test_identifier_is_deterministic_and_disjoint_from_frs_namespace() -> None:
    """Same container ⇒ same identifier (so ``POST /secrets`` upserts, not duplicates).

    And the ``container:`` prefix must keep it out of fr's ``project:{id}:...``
    namespace: both services write into the same tenant ``subject_id``, so an
    upsert-key collision would mean one silently overwriting the other's token.
    """
    assert git_token_identifier("ctr-1") == git_token_identifier("ctr-1")
    assert git_token_identifier("ctr-1") != git_token_identifier("ctr-2")
    assert git_token_identifier("ctr-1").startswith("container:")
    # A container id cannot forge fr's namespace: k8s names admit no ':'.
    assert not git_token_identifier("ctr-1").startswith("project:")


@respx.mock
async def test_store_returns_the_real_secret_id_and_sends_the_flat_wire_body() -> None:
    """201 ⇒ the service's own ``secret_id``, and the body is the FLAT shape.

    The body assertion is the regression guard for the double-wrap: the old path
    called ``CodingGitHubClient.create_webhook(json={...})``, and that method is
    ``(**kwargs) -> post(json=kwargs)``, so the whole payload went out as
    ``{"json": {...}}``. Asserting the exact dict makes that shape a test failure
    rather than a silent 422.
    """
    route = respx.post(f"{_BASE}/secrets").mock(
        return_value=httpx.Response(201, json={"secret_id": "sec-real-123"})
    )

    secret_id = await store_git_token(
        container_id="ctr-99", token="ghp_the_real_pat", tenant_id="tenant-42"
    )

    assert secret_id == "sec-real-123", "must return the service's id, never a synthesised one"
    assert route.called
    body = json.loads(route.calls.last.request.content)
    assert "json" not in body, f"body is double-wrapped again: {body!r}"
    assert body == {
        "identifier": "container:ctr-99:github_token",
        "secrets": {"GIT_TOKEN": "ghp_the_real_pat"},
        "env_vars": {},
        "ttl": 604800,
    }


@respx.mock
async def test_store_sends_a_bare_tenant_header_not_a_bearer_value() -> None:
    """``X-Tenant-Id`` carries the BARE tenant, matching fr (the reference writer).

    fr's ``storeProjectGithubToken`` sends ``'X-Tenant-Id': tenantId``. A
    ``Bearer ...`` value here would land the row in a different partition than the
    one ``resolve_git_token`` reads with, and the read would 404.
    """
    route = respx.post(f"{_BASE}/secrets").mock(
        return_value=httpx.Response(201, json={"secret_id": "sec-1"})
    )

    await store_git_token(container_id="ctr-1", token="tok", tenant_id="tenant-7")

    sent = route.calls.last.request.headers["X-Tenant-Id"]
    assert sent == "tenant-7"
    assert not sent.lower().startswith("bearer")


@respx.mock
async def test_store_ttl_is_the_schema_max_not_the_one_hour_default() -> None:
    """``ttl`` must be 604800s. The service defaults to 3600s and HARD-DELETES on expiry.

    A container is pushed from minutes-to-hours after provisioning, so a 1h token
    self-deletes mid-drive and the push fails with "could not read Password". fr
    root-caused exactly this on 2026-07-04. 604800 is ``SecretData.ttl``'s
    ``le=604800`` maximum, so this is the longest the API can express.
    """
    route = respx.post(f"{_BASE}/secrets").mock(
        return_value=httpx.Response(201, json={"secret_id": "sec-ttl"})
    )

    await store_git_token(container_id="ctr-ttl", token="tok", tenant_id="t")

    assert json.loads(route.calls.last.request.content)["ttl"] == 604800


@respx.mock
async def test_store_omits_the_tenant_header_rather_than_inventing_one() -> None:
    """No tenant ⇒ no header. Same policy as the read half: absent is absent."""
    route = respx.post(f"{_BASE}/secrets").mock(
        return_value=httpx.Response(201, json={"secret_id": "sec-x"})
    )

    await store_git_token(container_id="ctr-nt", token="tok", tenant_id=None)

    sent = route.calls.last.request.headers
    assert "X-Tenant-Id" not in sent, f"fabricated a tenant: {sent.get('X-Tenant-Id')!r}"


@pytest.mark.parametrize("status", [400, 401, 403, 409, 422, 500, 502, 503])
@respx.mock
async def test_store_raises_on_any_error_status_and_never_returns_an_id(status: int) -> None:
    """A non-2xx must RAISE. 403 is the live one: ``RBAC: access denied`` from the
    secrets waypoint until coding-containers was added to ``secrets-write-access``.

    The predecessor caught this, returned a value, and let the provisioner log
    "User token stored". Whatever comes back here, it must not be a string a
    caller could persist as a ``secret_config_id``.
    """
    respx.post(f"{_BASE}/secrets").mock(
        return_value=httpx.Response(status, text="RBAC: access denied")
    )

    with pytest.raises(httpx.HTTPStatusError):
        await store_git_token(container_id="ctr-err", token="tok", tenant_id="t")


@respx.mock
async def test_store_raises_on_transport_failure_instead_of_degrading() -> None:
    """A timeout/connect error is not a store. The read half degrades here; this must not."""
    respx.post(f"{_BASE}/secrets").mock(side_effect=httpx.ConnectTimeout("timed out"))

    with pytest.raises(httpx.HTTPError):
        await store_git_token(container_id="ctr-to", token="tok", tenant_id="t")


@respx.mock
async def test_store_raises_when_a_2xx_carries_no_secret_id() -> None:
    """A 201 with no ``secret_id`` is not a usable store — nothing can resolve it back."""
    respx.post(f"{_BASE}/secrets").mock(return_value=httpx.Response(201, json={"status": "ok"}))

    with pytest.raises(RuntimeError, match="no secret_id"):
        await store_git_token(container_id="ctr-noid", token="tok", tenant_id="t")


async def test_store_raises_when_the_secrets_url_is_unset(monkeypatch) -> None:
    """No configured URL ⇒ raise, not a silent no-op that reports success."""
    settings = get_settings()
    monkeypatch.setattr(settings, "secrets_api_url", "", raising=False)

    with pytest.raises(RuntimeError, match="secrets_api_url is unset"):
        await store_git_token(container_id="ctr-nourl", token="tok", tenant_id="t")


@respx.mock
async def test_stored_token_round_trips_through_the_read_half() -> None:
    """What ``store_git_token`` writes, ``resolve_git_token`` reads back.

    The two halves are only useful as a pair. This asserts the contract that the
    old code broke by writing to ``user_token_id``, a field no reader consults:
    the id the write returns is the id the read accepts.
    """
    respx.post(f"{_BASE}/secrets").mock(
        return_value=httpx.Response(201, json={"secret_id": "sec-roundtrip"})
    )
    secret_id = await store_git_token(
        container_id="ctr-rt", token="ghp_round", tenant_id="tenant-rt"
    )

    respx.get(f"{_BASE}/secrets/{secret_id}").mock(
        return_value=httpx.Response(200, json={"secrets": {"GIT_TOKEN": "ghp_round"}})
    )
    assert await resolve_git_token(secret_id, "tenant-rt") == "ghp_round"

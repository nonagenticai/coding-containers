"""Unit tests for the App-backed clone-token lookup against coding-github.

Contract (owned by coding-github)::

    POST {CODING_GITHUB_URL}/api/internal/github/clone-token
      headers: X-Internal-Token, Authorization (this service's own licence)
      body:    {"repo_url": "https://github.com/owner/repo", "tenant_id": "<tenant>"?}
      200 →    {"token": "ghs_..."|null, "expires_at": ..., "visibility": "public"|"private",
                "source": "app"|"none", "reason": "tenant_required"|"not_installed"|...?}
      403 →    EITHER a refused tenant assertion (the middleware, before the route)
               OR a bad/missing X-Internal-Token. Only the body says which.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx

import src.config.settings as settings_mod
from src.clients.github_client import CloneCredential, GitHubTokenManager

_BASE = "http://coding-github.coding-github.svc.cluster.local:8692"
_URL = f"{_BASE}/api/internal/github/clone-token"
_REPO = "https://github.com/owner/repo"


@pytest.fixture(autouse=True)
def _configured_settings(monkeypatch):
    """Point the client at a fake coding-github and reset the settings singleton."""
    monkeypatch.setenv("CODING_CONTAINERS_ENVIRONMENT", "test")
    monkeypatch.setenv("CODING_GITHUB_URL", _BASE)
    monkeypatch.setenv("CODING_CONTAINERS_INTERNAL_SERVICE_TOKEN", "shhh")  # noqa: S105
    settings_mod._settings_instance = None
    yield
    settings_mod._settings_instance = None


@respx.mock
async def test_a_supplied_tenant_is_never_asserted_on_this_call(monkeypatch) -> None:
    """A tenant on the provision must NOT become an on-behalf assertion here.

    This assertion is inverted from what it used to be, deliberately. It used to
    require ``X-Tenant-Id == "tenant-42"``, which pinned the defect: coding-github
    runs the tenant middleware in front of ``/api/internal`` (PUBLIC_PATHS covers
    only /health, /docs, /metrics and friends), so asserting a tenant other than
    the one our licence names is refused with 403 unless this service's subject
    sits in its CODING_ONBEHALF_SUBJECTS — and it does not. Measured live
    2026-08-15: every provision of a repo with no stored PAT 403'd here and the
    container was never created.

    The endpoint is repo-scoped and never reads a tenant, so there is nothing to
    preserve by sending one.
    """
    monkeypatch.setenv("CODING_SERVICE_TOKEN", "licence-abc")
    route = respx.post(_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "token": "ghs_installtoken",
                "expires_at": "2026-07-14T12:00:00Z",
                "visibility": "private",
                "source": "app",
            },
        )
    )

    cred = await GitHubTokenManager().get_app_clone_credential(_REPO, tenant_id="tenant-42")

    assert cred == CloneCredential(
        ok=True, token="ghs_installtoken", visibility="private", source="app"
    )
    assert cred.is_public is False
    assert route.called
    request = route.calls.last.request
    assert "X-Tenant-Id" not in request.headers, (
        f"a tenant assertion went out ({request.headers.get('X-Tenant-Id')!r}); coding-github's "
        "tenant middleware refuses it with 403 before the route ever runs"
    )
    # IDENTITY IS KEPT. Dropping the assertion must not drop the licence with it:
    # the licence is what proves *which* internal caller this is, and a refactor
    # that removes both would look identical on the tenant assertion alone.
    assert request.headers.get("authorization") == "Bearer licence-abc"
    assert request.headers["X-Internal-Token"] == "shhh"
    # The tenant is NOT lost: it rides in the BODY, where coding-github's route
    # reads it to pick the installation bound to this tenant (customer repos).
    assert json.loads(request.read()) == {"repo_url": _REPO, "tenant_id": "tenant-42"}


@respx.mock
async def test_missing_tenant_omits_the_header_instead_of_inventing_one() -> None:
    """No tenant ⇒ no ``X-Tenant-Id`` at all — never the literal ``"anonymous"``.

    coding-github's clone-token route authorises on ``X-Internal-Token`` alone
    (``Depends(require_internal_token)``) and never reads the tenant, so omitting
    it costs nothing. Sending ``"anonymous"`` cost something: every tenantless
    provision claimed a caller identity that no tenant registry issues, and it is
    indistinguishable downstream from a real tenant of that name.
    """
    route = respx.post(_URL).mock(
        return_value=httpx.Response(
            200, json={"token": None, "visibility": "public", "source": "none"}
        )
    )

    cred = await GitHubTokenManager().get_app_clone_credential(_REPO, tenant_id=None)

    assert cred.ok is True  # omitting the header must not break the lookup
    assert route.called
    sent = route.calls.last.request.headers
    assert "X-Tenant-Id" not in sent, (
        f"expected no tenant header, got {sent.get('X-Tenant-Id')!r} — a fabricated identity"
    )
    # The real credential (the one the endpoint actually checks) is still sent.
    assert sent["X-Internal-Token"] == "shhh"


@respx.mock
async def test_public_repo_with_no_app_token_is_a_legitimate_null() -> None:
    respx.post(_URL).mock(
        return_value=httpx.Response(
            200,
            json={"token": None, "expires_at": None, "visibility": "public", "source": "none"},
        )
    )

    cred = await GitHubTokenManager().get_app_clone_credential(_REPO)

    # ok=True: the lookup SUCCEEDED and told us the repo needs no credential.
    assert cred.ok is True
    assert cred.token is None
    assert cred.visibility == "public"
    assert cred.is_public is True


@respx.mock
async def test_private_repo_with_no_app_token_is_a_legitimate_null_but_not_public() -> None:
    respx.post(_URL).mock(
        return_value=httpx.Response(
            200, json={"token": None, "visibility": "private", "source": "none"}
        )
    )

    cred = await GitHubTokenManager().get_app_clone_credential(_REPO)

    assert cred.ok is True
    assert cred.token is None
    assert cred.is_public is False


@respx.mock
async def test_403_is_an_error_not_a_silent_none(caplog) -> None:
    respx.post(_URL).mock(return_value=httpx.Response(403, json={"detail": "bad token"}))

    with caplog.at_level(logging.ERROR):
        cred = await GitHubTokenManager().get_app_clone_credential(_REPO)

    assert cred.ok is False
    assert cred.error == "http_403"
    assert cred.is_public is False
    # Loud, and it names the knob to turn.
    assert "X-Internal-Token" in caplog.text
    # ...and it repeats what the SERVER said, so the reader is not left with only
    # this client's guess about which of the two 403 gates fired.
    assert "bad token" in caplog.text


@respx.mock
async def test_a_refused_tenant_assertion_is_not_reported_as_a_bad_secret(caplog) -> None:
    """The 403 report must carry the server's reason, not a hard-coded cause.

    This is the exact body coding-github's middleware returned on 2026-08-15. The
    previous log line said only "the X-Internal-Token shared secret is missing or
    wrong (CODING_CONTAINERS_INTERNAL_SERVICE_TOKEN)" — while both secrets were
    byte-identical — and a debugging session spent hours on a secret that was
    provably correct. An error string written by the caller states what its author
    expected, not what the callee did.
    """
    respx.post(_URL).mock(
        return_value=httpx.Response(
            403,
            json={
                "error": "TENANT_ASSERTION_NOT_PERMITTED",
                "detail": (
                    "This credential may not assert a tenant other than the one its license names."
                ),
            },
        )
    )

    with caplog.at_level(logging.ERROR):
        cred = await GitHubTokenManager().get_app_clone_credential(_REPO, tenant_id="tenant-42")

    assert cred.ok is False
    assert cred.error == "http_403"
    assert "TENANT_ASSERTION_NOT_PERMITTED" in caplog.text, (
        "the server's own machine-readable reason must reach the operator; without it "
        "this 403 is indistinguishable from a shared-secret mismatch"
    )
    # The tenant that was NOT sent is still named, so the log says whose provision died.
    assert "tenant-42" in caplog.text


@respx.mock
async def test_a_non_json_403_body_still_reports_something(caplog) -> None:
    """An HTML 403 from a proxy must not degrade the report to an empty string."""
    respx.post(_URL).mock(
        return_value=httpx.Response(403, html="<html><body>Forbidden by proxy</body></html>")
    )

    with caplog.at_level(logging.ERROR):
        cred = await GitHubTokenManager().get_app_clone_credential(_REPO)

    assert cred.ok is False
    assert cred.error == "http_403"
    assert "Forbidden by proxy" in caplog.text


@respx.mock
async def test_transport_failure_is_an_error(caplog) -> None:
    respx.post(_URL).mock(side_effect=httpx.ConnectError("connection refused"))

    with caplog.at_level(logging.ERROR):
        cred = await GitHubTokenManager().get_app_clone_credential(_REPO)

    assert cred.ok is False
    assert cred.error == "ConnectError"
    assert cred.visibility is None
    assert cred.is_public is False
    assert "visibility is UNKNOWN" in caplog.text


@respx.mock
async def test_missing_visibility_in_response_is_an_error(caplog) -> None:
    respx.post(_URL).mock(return_value=httpx.Response(200, json={"token": None}))

    with caplog.at_level(logging.ERROR):
        cred = await GitHubTokenManager().get_app_clone_credential(_REPO)

    assert cred.ok is False
    assert cred.error == "bad_visibility"
    assert cred.is_public is False


async def test_unset_coding_github_url_logs_loudly_and_fails_the_lookup(
    monkeypatch, caplog
) -> None:
    monkeypatch.delenv("CODING_GITHUB_URL", raising=False)
    settings_mod._settings_instance = None

    with caplog.at_level(logging.ERROR):
        cred = await GitHubTokenManager().get_app_clone_credential(_REPO)

    assert cred.ok is False
    assert cred.error == "coding_github_url_unset"
    assert "CODING_GITHUB_URL" in caplog.text


# --- push-scope repository facts (this pod has no api.github.com egress) -----

_SHA = "c" * 40


@respx.mock
async def test_push_scope_sends_branch_and_parses_the_repository_facts() -> None:
    from src.clients.github_client import RefHead

    route = respx.post(_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "token": "ghs_push",
                "visibility": "private",
                "source": "app",
                "private": True,
                "default_branch": "conveyor-base",
                "branch_head": {"state": "absent", "sha": None},
                "default_branch_head": {"state": "present", "sha": _SHA},
            },
        )
    )
    cred = await GitHubTokenManager().get_app_clone_credential(
        _REPO, tenant_id="t", scope="push", branch="aa/fix-1"
    )
    sent = json.loads(route.calls.last.request.content)
    assert sent["scope"] == "push" and sent["branch"] == "aa/fix-1"
    assert cred.private is True
    assert cred.default_branch == "conveyor-base"
    assert cred.branch_head == RefHead("absent")
    assert cred.default_branch_head == RefHead("present", _SHA)


@respx.mock
async def test_clone_scope_never_sends_a_branch() -> None:
    route = respx.post(_URL).mock(
        return_value=httpx.Response(200, json={"token": "ghs_x", "visibility": "private"})
    )
    await GitHubTokenManager().get_app_clone_credential(_REPO, scope="clone", branch="aa/x")
    assert "branch" not in json.loads(route.calls.last.request.content)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({}, (None, None, None)),  # an older coding-github: every fact unknown
        ({"private": "true", "default_branch": ""}, (None, None, None)),
        ({"private": 1, "default_branch": 7}, (None, None, None)),
        ({"branch_head": {"state": "present", "sha": "short"}}, (None, None, "unknown")),
        ({"branch_head": {"state": "gone"}}, (None, None, "unknown")),
        ({"branch_head": "present"}, (None, None, "unknown")),
        ({"branch_head": {"state": "present", "sha": _SHA}}, (None, None, "present")),
    ],
)
async def test_facts_are_parsed_strictly(payload, expected) -> None:
    with respx.mock:
        respx.post(_URL).mock(
            return_value=httpx.Response(
                200, json={"token": "ghs_x", "visibility": "private", **payload}
            )
        )
        cred = await GitHubTokenManager().get_app_clone_credential(_REPO, scope="push", branch="b")
    head = cred.branch_head.state if cred.branch_head else None
    assert (cred.private, cred.default_branch, head) == expected

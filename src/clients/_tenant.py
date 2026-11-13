"""Tenant identity for outbound service-to-service calls — now a thin re-export.

This module used to carry its own implementation of ``tenant_headers`` /
``optional_tenant_headers`` / ``TenantMissingError``, and its docstring argued
that "the 'attach the tenant' step [should be] one function with one failure
mode, rather than N copies of a header dict that can each be forgotten
independently". coding-client 1.11.0 took that argument to its conclusion and
promoted the module into the SDK as ``coding_client.auth.outbound``, so all nine
services share the one implementation instead of this service being the only one
that had it.

What is re-exported here is therefore the SAME code, not a compatible copy: this
module is now an alias so the ~2 in-repo import sites keep working and so a
reader landing here is told where the logic lives. Keeping a second local
implementation is the failure this whole exercise exists to prevent — two copies
drift, and the one that drifts is the one nobody is testing.

The reasoning the old implementation carried, preserved because it is the reason
the SDK functions behave the way they do:

* **A missing tenant is an error, never a default.** Substituting
  ``DEFAULT_TENANT_ID``, ``"anonymous"``, or any other literal makes the code
  compile and the call succeed while attributing it to a tenant that does not own
  the resource. coding-client removed its ``"n8n-dev"`` fallback for this exact
  reason — that literal was the stale pre-uuid7 label of a *real* tenant, so
  substituting it wrote into an unaddressable silo. ``tenant_headers`` raises;
  ``optional_tenant_headers`` omits. Neither invents.

* **``TenantMissingError`` is deliberately not a subclass** of ``ValueError`` /
  ``RuntimeError`` / ``OSError`` / ``httpx.HTTPError``, so none of the broad
  ``except`` clauses wrapping these call sites — most of which degrade to a
  plausible-looking fallback result — can silently absorb it.

* **``optional_tenant_headers`` exists for calls the tenant does not authorise
  via the header.** coding-github's ``/api/internal/github/clone-token`` is
  guarded by the ``X-Internal-Token`` shared secret and must NOT receive
  ``X-Tenant-Id`` (its tenant middleware 403s the on-behalf assertion). The
  tenant is nevertheless REQUIRED there for customer repos (any owner outside
  the platform org): it travels as the ``tenant_id`` field of the JSON body,
  and without it coding-github returns ``token: null, reason:
  "tenant_required"`` — only platform-owned repos may omit it. Use
  ``tenant_headers`` whenever the tenant *scopes the resource* via the header —
  a guessed tenant there is a cross-tenant authorisation bug, not a lost log
  line.

New in 1.11.0, and the reason this file changed at all: both builders take
``service_token=``, which sends this service's OWN licence as
``Authorization: Bearer <token>`` **without touching ``X-Tenant-Id``**. That
separation is the whole point. Folding the credential into ``X-Tenant-Id`` would
collapse every customer into coding-containers' own tenant, because the callee
resolves the tenant from whatever it is given. Kept apart, the call reads "I am
coding-containers, acting for customer T", which the callee honours iff this
service's subject is in its ``CODING_ONBEHALF_SUBJECTS`` allowlist. Without it,
every sibling call 401s the moment the callee sets ``TRUST_TENANT_HEADER=false``.
"""

from __future__ import annotations

import os

from coding_client.auth.outbound import (
    TENANT_ID_HEADER,
    TenantMissingError,
    optional_tenant_headers,
    tenant_headers,
)

__all__ = [
    "TENANT_ID_HEADER",
    "TenantMissingError",
    "optional_tenant_headers",
    "service_token",
    "tenant_headers",
]


def service_token() -> str | None:
    """This service's own licence, read per call so a rotation needs no restart.

    Read from the environment at CALL time, never at import: the value is
    injected into the pod env, and binding it to a module-level constant would
    pin whatever was present when the first module happened to be imported —
    which in tests is "nothing" and in production survives a secret rotation.

    ``or None`` normalises a set-but-empty variable to absent. The SDK already
    tests the token for truthiness, so this is belt-and-braces rather than the
    guard that stops a bare ``Bearer `` going out — but an empty credential is a
    malformed one, not "no credential", and it would still burn an introspection
    round-trip on the fleet's hottest auth path.
    """
    return os.getenv("CODING_SERVICE_TOKEN") or None

"""Trust and index settings for a deployment's self-signed private Python index.

WHY THIS MODULE EXISTS. A deployment may serve its own packages from an index
over HTTPS with an internal CA that nothing in a workspace pod trusts. Two
things are needed to install from it, and they are not the same thing:

  * TRUST  -- `UV_INSECURE_HOST` / `PIP_TRUSTED_HOST`, or the resolver aborts
    with `invalid peer certificate: UnknownIssuer`.
  * DISCOVERY -- `UV_EXTRA_INDEX_URL` / `PIP_EXTRA_INDEX_URL`, or the resolver
    never asks that host in the first place and the package is simply "not
    found". A lock file pinning `source = registry https://<index>/...`
    does NOT supply this on its own.

MEASURED inside a live workspace pod (Python 3.14.7): DNS resolves, TCP connects,
and `/simple/` returns HTTP 200 -- but TLS verification fails. With the env
below, a cold `uv sync --frozen` of a 189-package lock completes in
**6 seconds**. Without it, the same sync fails in 11 s on `UnknownIssuer`.

⚙️ OFF BY DEFAULT. A checkout of this repository resolves against public PyPI.
A deployment that runs a private mirror opts in by setting
`CODING_PRIVATE_INDEX_HOST` to its host; setting it to "" is the same as unset.

⚠️ Two other copies of this knob predate this module and are NOT consolidated
here, deliberately -- they build a shell-prefix string rather than a mapping:
`src/services/package_upgrade/runner.py` (`_private_index_host`). Consolidating
them is worth doing and is a separate unit of work; adding a fourth divergent
copy was not.
"""

from __future__ import annotations

import os

#: Empty: no private index unless the deployment names one (see the module docstring).
DEFAULT_PRIVATE_INDEX_HOST = ""


def private_index_host() -> str:
    """The private index host, or "" when the mechanism is disabled.

    Read at CALL time, not import time, so a test or a deployment can change it
    without re-importing the module.
    """
    return os.environ.get("CODING_PRIVATE_INDEX_HOST", DEFAULT_PRIVATE_INDEX_HOST)


def private_index_url(host: str | None = None) -> str:
    """The PEP 503 simple-index URL for `host` (defaults to the configured one).

    "" when the mechanism is off -- never `https:///simple/`.
    """
    resolved = private_index_host() if host is None else host
    if not resolved:
        return ""
    return f"https://{resolved}/simple/"


def python_resolver_env() -> dict[str, str]:
    """Env that lets uv/pip both FIND and TRUST the private index.

    Empty when `CODING_PRIVATE_INDEX_HOST` is set to "" -- in that case nothing
    should be injected at all, rather than injecting an empty host, which would
    produce the nonsense URL `https:///simple/`.
    """
    host = private_index_host()
    if not host:
        return {}
    url = private_index_url(host)
    return {
        # Trust: uv uses the OS trust store, then treats this host as insecure.
        "UV_NATIVE_TLS": "1",
        "UV_INSECURE_HOST": host,
        "PIP_TRUSTED_HOST": host,
        # Discovery: without these the host is trusted but never queried.
        "UV_EXTRA_INDEX_URL": url,
        "PIP_EXTRA_INDEX_URL": url,
    }

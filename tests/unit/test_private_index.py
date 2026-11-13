"""The private-index env must supply BOTH trust and discovery, or neither.

Measured inside a live workspace pod: a cold `uv sync --frozen` of a 189-package
lock that pins a package from a self-signed private index fails in 11 s on
`invalid peer certificate: UnknownIssuer` without this env, and succeeds in 6 s
with it.

The mechanism is OFF unless the deployment names an index: a public checkout of
this repository resolves against public PyPI.
"""

from __future__ import annotations

import pytest

from src.core.private_index import (
    DEFAULT_PRIVATE_INDEX_HOST,
    private_index_host,
    private_index_url,
    python_resolver_env,
)

TRUST_KEYS = {"UV_NATIVE_TLS", "UV_INSECURE_HOST", "PIP_TRUSTED_HOST"}
DISCOVERY_KEYS = {"UV_EXTRA_INDEX_URL", "PIP_EXTRA_INDEX_URL"}


def test_default_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """A public checkout must not inject a private index — it has none."""
    monkeypatch.delenv("CODING_PRIVATE_INDEX_HOST", raising=False)
    assert DEFAULT_PRIVATE_INDEX_HOST == ""
    assert private_index_host() == ""
    assert python_resolver_env() == {}
    assert private_index_url() == ""


def test_env_carries_trust_and_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODING_PRIVATE_INDEX_HOST", "pkg.example.test")
    env = python_resolver_env()
    assert TRUST_KEYS <= env.keys(), "trust settings missing -> UnknownIssuer"
    assert DISCOVERY_KEYS <= env.keys(), "index URL missing -> package 'not found'"
    assert env["UV_EXTRA_INDEX_URL"] == "https://pkg.example.test/simple/"
    assert env["UV_INSECURE_HOST"] == "pkg.example.test"


def test_host_is_read_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import-time reads cannot be overridden by a deployment or a test."""
    monkeypatch.setenv("CODING_PRIVATE_INDEX_HOST", "other.example.test")
    env = python_resolver_env()
    assert env["UV_INSECURE_HOST"] == "other.example.test"
    assert env["UV_EXTRA_INDEX_URL"] == "https://other.example.test/simple/"
    assert private_index_url() == "https://other.example.test/simple/"


def test_disabled_injects_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not even an empty host -- `https:///simple/` would be worse than nothing."""
    monkeypatch.setenv("CODING_PRIVATE_INDEX_HOST", "")
    assert python_resolver_env() == {}

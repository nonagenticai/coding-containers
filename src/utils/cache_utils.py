"""Cache utilities for container-scoped caching."""

from __future__ import annotations

import hashlib


def generate_container_cache_key(
    container_name: str,
    operation: str,
    tenant_id: str = "",
    **kwargs,
) -> str:
    """Generate a cache key scoped to a tenant, container, and operation."""
    parts = [tenant_id, container_name, operation]
    if kwargs:
        extra = ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        parts.append(hashlib.md5(extra.encode(), usedforsecurity=False).hexdigest()[:8])
    return ":".join(filter(None, parts))


def get_container_cache_prefix(tenant_id: str, container_name: str) -> str:
    """Return prefix for pattern matching cache keys."""
    return f"{tenant_id}:{container_name}:"

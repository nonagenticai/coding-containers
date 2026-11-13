"""Container registry module for managing container metadata persistence.

Persists the ``container_name -> ContainerInfo`` map used across the
provisioning / graph-build paths. The store is a **Redis hash** shared by every
replica, which is what makes coding-containers safe to run as multiple
horizontal replicas: node-local JSON-file state used to diverge per replica (a
container provisioned on replica A 404'd when the Service round-robined its
exec/delete to replica B, which had a different local file). A Redis hash is a
single shared source of truth, and single-entry mutations (``upsert``/``remove``
-> ``HSET``/``HDEL``) are atomic, so concurrent writers on different replicas can
never clobber a sibling's just-stored entry.

Resilience:
    * If Redis is unavailable (unit tests, or a transient outage), the registry
      transparently falls back to an in-process dict so callers never crash. In
      production the labelled k8s Deployment remains the durable source of truth
      (``container_manager`` reconstructs a missing entry from the cluster), so a
      Redis blip degrades to slower k8s reads, not data loss.
    * ⚠️ That fallback is PER-POD and is not the durable store, so a write it
      accepts is not a write that persisted. A caller whose correctness depends
      on a delete having actually landed must use ``remove_with_outcome`` and
      handle ``RemoveOutcome.UNCERTAIN`` — ``remove``'s bool cannot express it.

Storage:
    Redis key ``coding_containers:container_registry`` (a hash);
    field = container name, value = ``json.dumps(ContainerInfo.model_dump())``.

Classes:
    ContainerMetadataRegistry: Redis-backed container metadata persistence.
"""

from __future__ import annotations

import builtins
import json
import logging
import threading
from enum import Enum
from typing import Any

from redis.exceptions import RedisError

from src.models.container_models import ContainerInfo

logger = logging.getLogger(__name__)


class RemoveOutcome(Enum):
    """What the DURABLE store said about a single-entry delete.

    ``remove`` collapses these into a bool and so cannot distinguish "there was
    no row" from "we could not reach the store"; the distinction matters to any
    caller for whom a surviving row has consequences. See
    ``ContainerMetadataRegistry.remove_with_outcome``.
    """

    REMOVED = "removed"
    """Redis answered and the field was deleted."""

    ABSENT = "absent"
    """Redis answered and there was no such field — safe to assert as true."""

    UNCERTAIN = "uncertain"
    """The HDEL never reached Redis; a durable row may survive."""


# Redis hash holding the whole registry (field = container name).
REGISTRY_HASH_KEY = "coding_containers:container_registry"

# Guards the in-process fallback dict below. Redis hash ops are themselves
# atomic, so this lock only serialises the degraded no-Redis path.
_REGISTRY_WRITE_LOCK = threading.RLock()

# In-process fallback used ONLY when Redis is unreachable (tests / outage). Not
# shared across replicas — production correctness relies on Redis + the k8s
# reconstruct backstop, never on this dict.
_MEMORY_FALLBACK: dict[str, dict[str, Any]] = {}

_REQUIRED_FIELDS = (
    "container_name",
    "api_endpoint",
    "code_graph_id",
    "codebase_path",
    "repo_url",
    "branch",
    "status",
)


def _redis_client():
    """Return a live Redis client, or ``None`` if Redis is unavailable.

    Never raises — a missing/broken Redis degrades to the in-process fallback
    (and, in production, to the k8s reconstruct path in container_manager)."""
    try:
        from src.core.redis_service import get_redis_service_sync

        svc = get_redis_service_sync()
        client = svc.client  # lazily connects + pings; raises on failure
        return client
    except (RedisError, OSError) as e:
        logger.warning("Registry Redis unavailable (%s); using in-process fallback", e)
        return None
    except Exception as e:  # noqa: BLE001 - degrade, never crash the caller
        logger.warning("Registry Redis error (%s); using in-process fallback", e)
        return None


def _parse(name: str, raw: Any) -> ContainerInfo | None:
    """Validate + build a ContainerInfo from a stored dict, or None if invalid."""
    try:
        if not name or not isinstance(name, str) or not isinstance(raw, dict):
            logger.warning("Invalid registry entry for %s", name)
            return None
        if not all(field in raw for field in _REQUIRED_FIELDS):
            logger.warning("Missing required fields in container info for %s", name)
            return None
        return ContainerInfo(**raw)
    except (KeyError, ValueError, TypeError) as e:
        logger.warning("Failed to load container info for %s: %s", name, e)
        return None


class ContainerMetadataRegistry:
    """Redis-backed registry for container metadata (multi-replica safe)."""

    def __init__(self) -> None:
        self.hash_key = REGISTRY_HASH_KEY

    # ------------------------------------------------------------------ reads
    def load(self) -> dict[str, ContainerInfo]:
        """Load the whole registry as ``{name: ContainerInfo}``."""
        client = _redis_client()
        if client is None:
            with _REGISTRY_WRITE_LOCK:
                raw_items: dict[str, Any] = dict(_MEMORY_FALLBACK)
        else:
            try:
                raw_items = {k: json.loads(v) for k, v in client.hgetall(self.hash_key).items()}
            except (RedisError, json.JSONDecodeError, OSError) as e:
                logger.error("Failed to load container registry from Redis: %s", e)
                return {}

        cleaned: dict[str, ContainerInfo] = {}
        for name, raw in raw_items.items():
            info = _parse(name, raw)
            if info is not None:
                cleaned[name] = info
        return cleaned

    def get_container(self, container_name: str) -> ContainerInfo | None:
        """Get a single container by exact name (fast single-field read)."""
        return self.find_by_name(container_name)

    def find_by_name(self, container_name: str) -> ContainerInfo | None:
        """Find a container by exact name."""
        client = _redis_client()
        if client is None:
            with _REGISTRY_WRITE_LOCK:
                raw = _MEMORY_FALLBACK.get(container_name)
            return _parse(container_name, raw) if raw is not None else None
        try:
            v = client.hget(self.hash_key, container_name)
        except (RedisError, OSError) as e:
            logger.error("Registry hget failed for %s: %s", container_name, e)
            return None
        if v is None:
            return None
        try:
            return _parse(container_name, json.loads(v))
        except json.JSONDecodeError:
            return None

    def durable_names(self) -> set[str] | None:
        """Field names in the DURABLE Redis hash, or ``None`` if Redis cannot answer.

        ``None`` is deliberately distinct from an empty set: the registry-ghost
        sweep deletes rows, and "I could not read Redis" must never look like
        "Redis has no rows" (nor may the per-pod fallback dict stand in for it).
        """
        client = _redis_client()
        if client is None:
            return None
        try:
            return {
                k.decode() if isinstance(k, bytes) else str(k) for k in client.hkeys(self.hash_key)
            }
        except (RedisError, OSError) as e:
            logger.error("Registry hkeys failed: %s", e)
            return None

    def list(self) -> builtins.list[ContainerInfo]:
        """List all containers in the registry."""
        return list(self.load().values())

    def find_by_name_pattern(self, identifier: str) -> ContainerInfo | None:
        """Find a container by exact name, then by partial/pattern match."""
        exact = self.find_by_name(identifier)
        if exact is not None:
            return exact
        registry = self.load()
        for name, container in registry.items():
            if identifier in name or name.endswith(f"_{identifier}") or identifier.startswith(name):
                logger.debug("Found container by name pattern match: %s", name)
                return container
        return None

    # ----------------------------------------------------------------- writes
    def save(self, registry: dict[str, ContainerInfo]) -> None:
        """Replace the whole registry with ``registry`` (reconcile, not clobber).

        HSETs every provided entry and HDELs any field no longer present, so a
        full-dict rebuild caller gets replace semantics. Prefer ``upsert``/
        ``remove`` for single-entry mutations — those are naturally atomic and
        never race a sibling replica."""
        payload = {k: json.dumps(v.model_dump()) for k, v in registry.items()}
        client = _redis_client()
        if client is None:
            with _REGISTRY_WRITE_LOCK:
                _MEMORY_FALLBACK.clear()
                _MEMORY_FALLBACK.update({k: json.loads(val) for k, val in payload.items()})
            return
        try:
            existing = set(client.hkeys(self.hash_key))
            stale = existing - set(payload)
            pipe = client.pipeline()
            if payload:
                pipe.hset(self.hash_key, mapping=payload)
            if stale:
                pipe.hdel(self.hash_key, *stale)
            pipe.execute()
            logger.debug("Container registry reconciled (%d entries)", len(payload))
        except (RedisError, OSError) as e:
            logger.error("Failed to save container registry to Redis: %s", e)
            raise

    def upsert(self, name: str, info: ContainerInfo) -> None:
        """Insert/replace a single container (atomic HSET; multi-replica safe)."""
        value = json.dumps(info.model_dump())
        client = _redis_client()
        if client is None:
            with _REGISTRY_WRITE_LOCK:
                _MEMORY_FALLBACK[name] = json.loads(value)
            logger.debug("Upserted container %s in in-process registry", name)
            return
        try:
            client.hset(self.hash_key, name, value)
        except (RedisError, OSError) as e:
            logger.error("Failed to upsert container %s: %s", name, e)
            raise
        logger.debug("Upserted container %s in registry", name)

    def remove(self, name: str) -> bool:
        """Delete a single container (atomic HDEL). Returns True if it existed.

        ⚠️ ``False`` is AMBIGUOUS and always has been: it means "no row was
        removed from the store this process could reach", which conflates *the
        durable row did not exist* with *we never reached the durable store*.
        Callers that only log the difference are fine with that. A caller whose
        CORRECTNESS depends on the row actually being gone must use
        ``remove_with_outcome`` instead and handle ``RemoveOutcome.UNCERTAIN``.

        This bool contract is deliberately unchanged — every pre-existing
        caller keeps exactly the semantics it was written against.
        """
        _outcome, existed = self._remove(name)
        return existed

    def remove_with_outcome(self, name: str) -> RemoveOutcome:
        """Delete a single container, reporting what the DURABLE store said.

        Use this instead of ``remove`` wherever "the row is gone" is a decision
        with consequences rather than a log line — the reaper is the motivating
        caller: it deletes a Deployment and then drops the row, and if that HDEL
        is silently lost the surviving row lets
        ``ContainerManager.ensure_container`` resurrect the container minutes
        later (measured 2026-09-13: reaped 16:37:18Z, back at 16:40:56Z).

        The three outcomes are distinguishable precisely because each is a
        different claim about REDIS, not about this process:

        * ``REMOVED`` — Redis answered and a field was deleted.
        * ``ABSENT``  — Redis answered and there was no such field. This is the
          only state in which "there was no row" may be asserted as true.
        * ``UNCERTAIN`` — the HDEL never reached Redis (unreachable, or it
          raised), so a durable row may well survive. ⚠️ This includes the
          case where the in-process fallback DID pop an entry: that dict is
          per-pod and is not the durable store, so popping it proves nothing
          about the Redis row, which becomes visible again the moment Redis
          returns. Anything treating a fallback pop as a durable delete is
          asserting a positive falsehood.
        """
        outcome, _existed = self._remove(name)
        return outcome

    def _remove(self, name: str) -> tuple[RemoveOutcome, bool]:
        """Shared HDEL. Returns ``(durable outcome, legacy "existed" bool)``.

        The two halves differ on purpose: the bool preserves ``remove``'s
        historical contract (True whenever *something* was popped, including
        from the per-pod fallback), while the outcome reports only what the
        durable store confirmed. See ``remove_with_outcome``.
        """
        client = _redis_client()
        if client is None:
            with _REGISTRY_WRITE_LOCK:
                existed = _MEMORY_FALLBACK.pop(name, None) is not None
            if existed:
                logger.info("Removed container %s from in-process registry", name)
            # Not an error for this module to report — a degraded registry is a
            # documented mode — but it is never a durable delete. Severity is
            # the caller's call; it is logged at WARNING here and escalated to
            # ERROR by callers for whom a surviving row is a real consequence.
            logger.warning(
                "Registry HDEL for %s did not reach Redis (unreachable); only the "
                "per-pod fallback was touched, so any durable row survives",
                name,
            )
            return RemoveOutcome.UNCERTAIN, existed
        try:
            removed = client.hdel(self.hash_key, name)
        except (RedisError, OSError) as e:
            logger.error("Failed to remove container %s: %s", name, e)
            return RemoveOutcome.UNCERTAIN, False
        if removed:
            logger.info("Removed container %s from registry", name)
            return RemoveOutcome.REMOVED, True
        return RemoveOutcome.ABSENT, False

    def update_container(self, container_name: str, updated_info: ContainerInfo) -> bool:
        """Update an existing container. Returns False if it was not present."""
        client = _redis_client()
        if client is None:
            with _REGISTRY_WRITE_LOCK:
                if container_name not in _MEMORY_FALLBACK:
                    logger.warning("Container %s not found in registry for update", container_name)
                    return False
                _MEMORY_FALLBACK[container_name] = json.loads(json.dumps(updated_info.model_dump()))
            logger.info("Updated container info for %s", container_name)
            return True
        try:
            if not client.hexists(self.hash_key, container_name):
                logger.warning("Container %s not found in registry for update", container_name)
                return False
            client.hset(self.hash_key, container_name, json.dumps(updated_info.model_dump()))
        except (RedisError, OSError) as e:
            logger.error("Failed to update container %s: %s", container_name, e)
            return False
        logger.info("Updated container info for %s", container_name)
        return True


class _RegistrySingleton:
    """Singleton holder for container registry instance."""

    instance: ContainerMetadataRegistry | None = None


def get_container_registry() -> ContainerMetadataRegistry:
    """Get singleton instance of container metadata registry."""
    if _RegistrySingleton.instance is None:
        _RegistrySingleton.instance = ContainerMetadataRegistry()
    return _RegistrySingleton.instance

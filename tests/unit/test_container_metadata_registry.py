"""Tests for core/container_metadata_registry.py (Redis-backed registry).

The registry stores the ``container_name -> ContainerInfo`` map in a Redis hash
so it is shared across horizontal replicas. These tests inject a minimal
dict-backed fake Redis (implementing only the hash ops the registry uses) via
monkeypatch, so the real Redis code path is exercised with no live server, plus
explicit coverage of the in-process fallback used when Redis is unavailable.
"""

from __future__ import annotations

import pytest

import src.core.container_metadata_registry as reg_mod
from src.core.container_metadata_registry import ContainerMetadataRegistry
from src.models.container_models import ContainerInfo, ContainerStatus

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _make_container_info(**overrides) -> ContainerInfo:
    defaults = dict(
        container_name="test-c",
        api_endpoint="http://gw:8687/api/containers/test-c",
        internal_endpoint="http://localhost:8700",
        public_endpoint="http://gw:8687/api/containers/test-c",
        code_graph_id="graph-1",
        codebase_path="/app/code",
        repo_url="https://github.com/org/repo",
        branch="main",
        status=ContainerStatus.RUNNING,
        description="Test container",
    )
    defaults.update(overrides)
    return ContainerInfo(**defaults)


class _FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._ops: list = []

    def hset(self, key, field=None, value=None, mapping=None):
        self._ops.append(("hset", key, field, value, mapping))
        return self

    def hdel(self, key, *fields):
        self._ops.append(("hdel", key, fields))
        return self

    def execute(self):
        for op in self._ops:
            if op[0] == "hset":
                self._redis.hset(op[1], op[2], op[3], mapping=op[4])
            elif op[0] == "hdel":
                self._redis.hdel(op[1], *op[2])
        self._ops.clear()


class FakeRedis:
    """Minimal dict-backed Redis hash (decode_responses=True semantics)."""

    def __init__(self) -> None:
        self.h: dict[str, dict[str, str]] = {}

    def hgetall(self, key):
        return dict(self.h.get(key, {}))

    def hget(self, key, field):
        return self.h.get(key, {}).get(field)

    def hset(self, key, field=None, value=None, mapping=None):
        d = self.h.setdefault(key, {})
        if mapping:
            d.update(mapping)
        if field is not None:
            d[field] = value
        return 1

    def hdel(self, key, *fields):
        d = self.h.get(key, {})
        removed = 0
        for f in fields:
            if f in d:
                del d[f]
                removed += 1
        return removed

    def hkeys(self, key):
        return list(self.h.get(key, {}).keys())

    def hexists(self, key, field):
        return field in self.h.get(key, {})

    def pipeline(self):
        return _FakePipeline(self)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_redis(monkeypatch):
    """A fresh fake Redis wired in as the registry's client."""
    client = FakeRedis()
    monkeypatch.setattr(reg_mod, "_redis_client", lambda: client)
    return client


@pytest.fixture
def registry(fake_redis):
    return ContainerMetadataRegistry()


@pytest.fixture
def populated_registry(registry):
    registry.save(
        {
            "c1": _make_container_info(container_name="c1", code_graph_id="g1"),
            "c2": _make_container_info(
                container_name="c2", code_graph_id="g2", status=ContainerStatus.STOPPED
            ),
        }
    )
    return registry


@pytest.fixture
def memory_registry(monkeypatch):
    """Registry with Redis unavailable -> in-process fallback path."""
    monkeypatch.setattr(reg_mod, "_redis_client", lambda: None)
    reg_mod._MEMORY_FALLBACK.clear()
    yield ContainerMetadataRegistry()
    reg_mod._MEMORY_FALLBACK.clear()


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_empty_when_fresh(self, registry):
        assert registry.load() == {}

    def test_load_valid_registry(self, populated_registry):
        result = populated_registry.load()
        assert set(result) == {"c1", "c2"}
        assert isinstance(result["c1"], ContainerInfo)
        assert result["c1"].code_graph_id == "g1"
        assert result["c1"].status == ContainerStatus.RUNNING
        assert result["c2"].status == ContainerStatus.STOPPED

    def test_load_skips_invalid_entries(self, registry, fake_redis):
        import json

        fake_redis.hset(
            registry.hash_key,
            "good",
            json.dumps(_make_container_info(container_name="good").model_dump()),
        )
        fake_redis.hset(
            registry.hash_key, "bad", json.dumps({"container_name": "bad"})
        )  # missing fields
        result = registry.load()
        assert "good" in result and "bad" not in result

    def test_load_skips_bad_status(self, registry, fake_redis):
        import json

        good = _make_container_info(container_name="good").model_dump()
        bad = dict(good, container_name="bad", status="INVALID_STATUS_VALUE")
        fake_redis.hset(registry.hash_key, "good", json.dumps(good))
        fake_redis.hset(registry.hash_key, "bad", json.dumps(bad))
        result = registry.load()
        assert "good" in result and "bad" not in result

    def test_load_returns_empty_when_redis_errors(self, registry, fake_redis):
        def boom(_key):
            raise reg_mod.RedisError("down")

        fake_redis.hgetall = boom
        assert registry.load() == {}


# ---------------------------------------------------------------------------
# save (reconcile / replace semantics)
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_persists(self, registry):
        registry.save({"c1": _make_container_info(container_name="c1")})
        assert set(registry.load()) == {"c1"}

    def test_save_replaces_and_prunes_stale(self, populated_registry):
        populated_registry.save({"c-new": _make_container_info(container_name="c-new")})
        result = populated_registry.load()
        assert "c1" not in result and "c2" not in result and "c-new" in result

    def test_save_empty_clears(self, populated_registry):
        populated_registry.save({})
        assert populated_registry.load() == {}


# ---------------------------------------------------------------------------
# get_container / list / find
# ---------------------------------------------------------------------------


class TestReads:
    def test_get_existing(self, populated_registry):
        assert populated_registry.get_container("c1").container_name == "c1"

    def test_get_nonexistent(self, populated_registry):
        assert populated_registry.get_container("nope") is None

    def test_list_all(self, populated_registry):
        names = {c.container_name for c in populated_registry.list()}
        assert names == {"c1", "c2"}

    def test_list_empty(self, registry):
        assert registry.list() == []

    def test_find_by_name(self, populated_registry):
        assert populated_registry.find_by_name("c2").container_name == "c2"
        assert populated_registry.find_by_name("nope") is None


# ---------------------------------------------------------------------------
# find_by_name_pattern
# ---------------------------------------------------------------------------


class TestFindByNamePattern:
    def test_exact_match(self, populated_registry):
        assert populated_registry.find_by_name_pattern("c1").container_name == "c1"

    def test_partial_match(self, registry):
        registry.upsert(
            "myapp-container-123", _make_container_info(container_name="myapp-container-123")
        )
        assert (
            registry.find_by_name_pattern("container-123").container_name == "myapp-container-123"
        )

    def test_suffix_match(self, registry):
        registry.upsert("prefix_suffix", _make_container_info(container_name="prefix_suffix"))
        assert registry.find_by_name_pattern("suffix").container_name == "prefix_suffix"

    def test_prefix_match(self, registry):
        registry.upsert("abc", _make_container_info(container_name="abc"))
        assert registry.find_by_name_pattern("abcdef").container_name == "abc"

    def test_no_match(self, populated_registry):
        assert populated_registry.find_by_name_pattern("zzz-not-here") is None


# ---------------------------------------------------------------------------
# atomic upsert / remove (multi-replica-safe by construction)
# ---------------------------------------------------------------------------


class TestAtomicUpsertRemove:
    def test_upsert_inserts_and_preserves_others(self, populated_registry):
        populated_registry.upsert(
            "c3", _make_container_info(container_name="c3", code_graph_id="g3")
        )
        assert set(populated_registry.load()) == {"c1", "c2", "c3"}

    def test_upsert_overwrites_same_key(self, registry):
        registry.upsert("c1", _make_container_info(container_name="c1", branch="main"))
        registry.upsert("c1", _make_container_info(container_name="c1", branch="dev"))
        loaded = registry.load()
        assert len(loaded) == 1 and loaded["c1"].branch == "dev"

    def test_remove_deletes_and_reports_presence(self, populated_registry):
        assert populated_registry.remove("c1") is True
        assert set(populated_registry.load()) == {"c2"}
        assert populated_registry.remove("c1") is False


# ---------------------------------------------------------------------------
# update_container
# ---------------------------------------------------------------------------


class TestUpdateContainer:
    def test_update_existing(self, populated_registry):
        updated = _make_container_info(
            container_name="c1", code_graph_id="g1-updated", description="updated"
        )
        assert populated_registry.update_container("c1", updated) is True
        loaded = populated_registry.load()
        assert loaded["c1"].code_graph_id == "g1-updated"

    def test_update_nonexistent(self, populated_registry):
        assert (
            populated_registry.update_container("nope", _make_container_info(container_name="nope"))
            is False
        )


# ---------------------------------------------------------------------------
# in-process fallback (Redis unavailable) — must never crash
# ---------------------------------------------------------------------------


class TestMemoryFallback:
    def test_upsert_get_remove(self, memory_registry):
        memory_registry.upsert("m1", _make_container_info(container_name="m1"))
        assert memory_registry.get_container("m1").container_name == "m1"
        assert set(memory_registry.load()) == {"m1"}
        assert memory_registry.remove("m1") is True
        assert memory_registry.load() == {}

    def test_save_and_update(self, memory_registry):
        memory_registry.save({"m1": _make_container_info(container_name="m1")})
        assert (
            memory_registry.update_container(
                "m1", _make_container_info(container_name="m1", branch="dev")
            )
            is True
        )
        assert memory_registry.load()["m1"].branch == "dev"
        assert (
            memory_registry.update_container("nope", _make_container_info(container_name="nope"))
            is False
        )


# ---------------------------------------------------------------------------
# singleton
# ---------------------------------------------------------------------------


class TestGetContainerRegistry:
    def test_singleton(self):
        from src.core.container_metadata_registry import _RegistrySingleton, get_container_registry

        _RegistrySingleton.instance = None
        r1 = get_container_registry()
        r2 = get_container_registry()
        assert r1 is r2
        _RegistrySingleton.instance = None

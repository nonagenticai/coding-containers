"""A single-entry registry mutation must not delete a sibling replica's container.

THE DEFECT THIS PINS (measured live 2026-08-17, board 1843503367372932614).

``verify-1843512668502099888`` was provisioned at 08:37:35Z, served an exec at
08:38:08Z (HTTP 200, the venv bootstrap printed VENV_OK), and by 08:39:27Z the
same endpoint answered ``404 Container not found`` in **1.32 ms** — a registry
miss, not a cluster query. Its k8s Deployment was Running the whole time and was
still Running three hours later. The verification_test card therefore blocked at
its two HARD gates with a 404 that reads like broken code.

The cause is a lost update. Every single-entry mutation went through the
whole-registry replace::

    registry_data = registry.load()      # snapshot S, taken at T
    del registry_data[name]              # or: registry_data[name] = info
    registry.save(registry_data)         # HDELs (existing - S) at T+delta

``save`` has deliberate replace semantics — it HDELs any field absent from the
payload, and ``test_save_replaces_and_prunes_stale`` pins that on purpose. The
bug is not ``save``; it is calling ``save`` with a STALE PARTIAL VIEW. Any
container created by another replica (or by another request on this one) between
the ``load`` and the ``save`` is absent from the payload, is therefore judged
"stale", and is deleted — even though nobody asked to delete it.

``ContainerMetadataRegistry`` already documents the fix in ``save``'s own
docstring: *"Prefer upsert/remove for single-entry mutations — those are
naturally atomic and never race a sibling replica."* These tests hold the
callers to that.

Each test simulates the race deterministically rather than with threads: the
sibling is written to the shared store AFTER the mutating caller has taken its
snapshot, which is exactly the interleaving that loses it.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import src.core.container_metadata_registry as reg_mod
from src.core.container_metadata_registry import ContainerMetadataRegistry
from src.models.container_models import ContainerStatus
from src.services.docker_services.container_lifecycle import ContainerLifecycleManager
from tests.unit.test_container_metadata_registry import FakeRedis, _make_container_info


@pytest.fixture
def fake_redis(monkeypatch):
    client = FakeRedis()
    monkeypatch.setattr(reg_mod, "_redis_client", lambda: client)
    return client


@pytest.fixture
def registry(fake_redis):
    return ContainerMetadataRegistry()


def _sibling_appears_after_snapshot(registry: ContainerMetadataRegistry, monkeypatch, sibling: str):
    """Make ``registry.load()`` return a pre-sibling snapshot, then add the sibling.

    This is the interleaving that loses the update: the caller's snapshot cannot
    contain a container that did not exist when it was taken.
    """
    snapshot = registry.load()

    def _stale_load():
        return dict(snapshot)

    monkeypatch.setattr(registry, "load", _stale_load)
    # The sibling replica's create lands in the SHARED store after the snapshot.
    registry.upsert(sibling, _make_container_info(container_name=sibling, code_graph_id="g-sib"))
    return snapshot


class TestSingleEntryMutationsAreSiblingSafe:
    def test_discard_entry_does_not_delete_a_sibling(self, registry, monkeypatch):
        """A failed-create cleanup must remove ONLY its own provisional row."""
        registry.upsert("victim", _make_container_info(container_name="victim"))
        _sibling_appears_after_snapshot(registry, monkeypatch, "verify-1843512668502099888")

        mgr = ContainerLifecycleManager(MagicMock(), registry, "proj")
        mgr._discard_registry_entry("victim")

        assert registry.get_container("victim") is None, "the targeted row must be gone"
        assert registry.get_container("verify-1843512668502099888") is not None, (
            "a sibling replica's container was deleted by an unrelated cleanup — "
            "this is the live 404 that blocked verification_test"
        )

    def test_write_entry_does_not_delete_a_sibling(self, registry, monkeypatch):
        """Creating a container must not evict one created moments earlier."""
        _sibling_appears_after_snapshot(registry, monkeypatch, "verify-1843512668502099888")

        mgr = ContainerLifecycleManager(MagicMock(), registry, "proj")
        mgr._write_registry_entry(_make_container_info(container_name="newly-created"))

        assert registry.get_container("newly-created") is not None, "the new row must be written"
        assert registry.get_container("verify-1843512668502099888") is not None, (
            "creating one container deleted another — the create path is the most "
            "frequent caller, so this is the widest blast radius of the race"
        )

    @pytest.mark.asyncio
    async def test_remove_container_does_not_delete_a_sibling(self, registry, monkeypatch):
        """Docker-unavailable teardown removes only the named row."""
        registry.upsert("victim", _make_container_info(container_name="victim"))
        _sibling_appears_after_snapshot(registry, monkeypatch, "verify-1843512668502099888")

        mgr = ContainerLifecycleManager(None, registry, "proj")  # no docker client
        ok = await mgr.remove_container(_make_container_info(container_name="victim"))

        assert ok is True
        assert registry.get_container("victim") is None
        assert registry.get_container("verify-1843512668502099888") is not None, (
            "teardown of one container deleted an unrelated sibling"
        )


class TestNegativeControlTheProbeCanFail:
    """The probe must be able to go red, or it proves nothing."""

    def test_stale_save_really_does_prune(self, registry, monkeypatch, fake_redis):
        """Direct evidence the interleaving is lossy — this is the mechanism itself.

        If this ever stops pruning, ``save``'s replace semantics changed and the
        tests above would pass for a reason unrelated to the callers being fixed.
        """
        snapshot = registry.load()
        registry.upsert("sibling", _make_container_info(container_name="sibling"))
        assert registry.get_container("sibling") is not None

        registry.save(dict(snapshot))  # a stale whole-registry replace

        assert registry.get_container("sibling") is None, (
            "save() no longer prunes; the sibling-safety tests above are now vacuous"
        )

    def test_registry_hash_round_trips(self, registry, fake_redis):
        """Guards against a fake that silently accepts everything."""
        info = _make_container_info(container_name="rt", status=ContainerStatus.RUNNING)
        registry.upsert("rt", info)
        raw = fake_redis.hget(registry.hash_key, "rt")
        assert raw is not None and json.loads(raw)["container_name"] == "rt"

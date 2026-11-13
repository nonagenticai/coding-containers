"""Tests for the Phase C-1 container reaper + admission-cap counting.

Covers both the provisioner-side reap primitives (``reap_once`` and helpers in
``K8sContainerProvisioner``) and the ``src.core.reaper`` scheduler module.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: obtain the REAL provisioner module (conftest installs a stub).
#
# IMPORTANT: REUSE the already-loaded real module if present rather than
# popping + reloading it. test_kubernetes_container_provisioner.py loads the
# real module first (alphabetical order) and binds its class + mock.patch
# targets to that module object; clobbering it here would leave that file's
# patches pointing at a different module than its bound class, silently
# breaking its tests. We only force-load when the current entry is the stub
# (detected by the absence of our Phase C-1 symbols).
# ---------------------------------------------------------------------------
_REAL_MOD_NAME = "src.core.kubernetes_container_provisioner"
_existing = sys.modules.get(_REAL_MOD_NAME)
if _existing is not None and hasattr(_existing, "JOB_LABEL_SELECTOR"):
    _mod = _existing
else:
    sys.modules.pop(_REAL_MOD_NAME, None)
    with patch("kubernetes.config.load_kube_config"):
        _mod = importlib.import_module(_REAL_MOD_NAME)

K8sContainerProvisioner = _mod.K8sContainerProvisioner
JOB_LABEL_SELECTOR = _mod.JOB_LABEL_SELECTOR

import src.core.reaper as reaper  # noqa: E402  (after the bootstrap above)


@pytest.fixture(autouse=True)
def _settings_test_env(monkeypatch):
    """Force ENVIRONMENT=test so real Settings construction never trips the
    production validator, and reset the singleton around each test."""
    import src.config.settings as settings_mod

    monkeypatch.setenv("CODING_CONTAINERS_ENVIRONMENT", "test")
    settings_mod._settings_instance = None
    yield
    settings_mod._settings_instance = None


@pytest.fixture(autouse=True)
def _graph_ops(monkeypatch):
    """Stub the coding-analysis graph client for EVERY test: a real reap now
    deletes the workspace's graph first, and no unit test may reach the network.
    Tests that care about the call inspect ``delete_container_graph``."""
    ops = MagicMock()
    ops.delete_container_graph = AsyncMock(
        return_value={"success": True, "deleted_nodes": 3, "deleted_edges": 2}
    )
    monkeypatch.setattr(_mod, "get_graph_operations", lambda: ops)
    return ops


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_provisioner(namespace: str = "test-ns") -> K8sContainerProvisioner:
    with patch("kubernetes.config.load_kube_config"):
        prov = K8sContainerProvisioner(namespace=namespace, in_cluster=False)
    prov.apps_v1 = MagicMock()
    prov.core_v1 = MagicMock()
    return prov


def _dep(
    name: str,
    *,
    age: float | None = None,
    created_at_label: int | None = None,
    labelled: bool = True,
    done: bool = False,
    ts_none: bool = False,
) -> SimpleNamespace:
    """Build a fake V1Deployment-shaped object."""
    now = datetime.now(UTC)
    if ts_none:
        ts = None
    elif age is not None:
        ts = now - timedelta(seconds=age)
    else:
        ts = now
    labels = {"app": "codegraph-repo", "repo": name, "managed-by": "code-graph-helper"}
    if labelled:
        labels["coding-job"] = "true"
    if created_at_label is not None:
        labels["created-at"] = str(created_at_label)
    annotations = {"coding-job/done": "true"} if done else {}
    meta = SimpleNamespace(name=name, labels=labels, annotations=annotations, creation_timestamp=ts)
    return SimpleNamespace(metadata=meta)


def _set_list(prov, *, labelled=None, all_managed=None) -> None:
    """Route list_namespaced_deployment by selector to the right fake list."""

    def _side(namespace, label_selector):
        items = labelled if "coding-job=true" in label_selector else all_managed
        return SimpleNamespace(items=list(items or []))

    prov.apps_v1.list_namespaced_deployment.side_effect = _side


# ---------------------------------------------------------------------------
# Provisioner reap primitives
# ---------------------------------------------------------------------------
class TestReapOnce:
    def test_ttl_boundary_keep_and_reap(self):
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[_dep("fresh", age=99), _dep("old", age=101)])
        counts = prov.reap_once(ttl_seconds=100, dry_run=False)
        assert counts["reaped"] == 1
        assert counts["names_reaped"] == ["old"]
        assert counts["skipped"] == 1
        prov._cleanup_resources.assert_called_once_with("old")

    def test_ttl_exact_boundary_is_reapable(self):
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[_dep("edge", age=100.5)])
        counts = prov.reap_once(ttl_seconds=100, dry_run=False)
        assert counts["reaped"] == 1

    def test_dry_run_logs_but_does_not_delete(self):
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[_dep("old", age=500)])
        counts = prov.reap_once(ttl_seconds=100, dry_run=True)
        assert counts["would_reap"] == 1
        assert counts["reaped"] == 0
        prov._cleanup_resources.assert_not_called()

    def test_allowlisted_singleton_never_reaped(self):
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        # Even mislabelled coding-job=true + ancient → allowlist blocks it.
        _set_list(prov, labelled=[_dep("enterprise-mcp-server-aa", age=10_000_000)])
        counts = prov.reap_once(ttl_seconds=100, dry_run=False)
        assert counts["reaped"] == 0
        assert counts["skipped"] == 1
        prov._cleanup_resources.assert_not_called()

    def test_extra_exclude_names_honoured(self):
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[_dep("keep-me", age=999)])
        counts = prov.reap_once(ttl_seconds=100, dry_run=False, exclude_names={"keep-me"})
        assert counts["reaped"] == 0
        prov._cleanup_resources.assert_not_called()

    def test_done_annotation_fast_path(self):
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        # Young, but explicitly marked done → reaped despite age << ttl.
        _set_list(prov, labelled=[_dep("young-done", age=1, done=True)])
        counts = prov.reap_once(ttl_seconds=100_000, dry_run=False)
        assert counts["reaped"] == 1
        prov._cleanup_resources.assert_called_once_with("young-done")

    def test_clock_skew_negative_age_not_reaped(self):
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[_dep("future", age=-500)])
        counts = prov.reap_once(ttl_seconds=100, dry_run=False)
        assert counts["reaped"] == 0
        prov._cleanup_resources.assert_not_called()

    def test_unknown_age_treated_as_fresh(self):
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[_dep("no-ts", ts_none=True)])
        counts = prov.reap_once(ttl_seconds=100, dry_run=False)
        assert counts["reaped"] == 0

    def test_created_at_label_fallback_used_when_no_timestamp(self):
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        old_epoch = int(datetime.now(UTC).timestamp()) - 500
        _set_list(
            prov,
            labelled=[_dep("by-label", ts_none=True, created_at_label=old_epoch)],
        )
        counts = prov.reap_once(ttl_seconds=100, dry_run=False)
        assert counts["reaped"] == 1
        prov._cleanup_resources.assert_called_once_with("by-label")

    def test_recent_label_refreshes_ttl_over_old_creation_timestamp(self):
        """Reprovision-in-place refreshes the lease.

        An in-place patch keeps the original (old) creation_timestamp but
        re-stamps the created-at label to now. The reaper must use the freshest
        of the two, so a just-reprovisioned Deployment is NOT reaped even though
        its creation_timestamp is well past the TTL.
        """
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        recent_epoch = int(datetime.now(UTC).timestamp())
        _set_list(
            prov,
            labelled=[_dep("reprovisioned", age=500, created_at_label=recent_epoch)],
        )
        counts = prov.reap_once(ttl_seconds=100, dry_run=False)
        assert counts["reaped"] == 0
        assert counts["skipped"] == 1
        prov._cleanup_resources.assert_not_called()

    def test_per_deletion_error_isolated(self):
        from kubernetes.client.rest import ApiException

        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock(side_effect=[ApiException(status=500), None])
        _set_list(prov, labelled=[_dep("bad", age=500), _dep("good", age=500)])
        counts = prov.reap_once(ttl_seconds=100, dry_run=False)
        assert counts["errors"] == 1
        assert counts["reaped"] == 1
        assert counts["names_reaped"] == ["good"]

    def test_legacy_mode_reaps_unlabelled_but_not_singleton(self):
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        labelled = [_dep("labelled-old", age=500)]
        all_managed = labelled + [
            _dep("legacy-unlabelled", age=1_000_000, labelled=False),
            _dep("enterprise-mcp-server-aa", age=1_000_000, labelled=False),
        ]
        _set_list(prov, labelled=labelled, all_managed=all_managed)
        counts = prov.reap_once(ttl_seconds=100, dry_run=False, legacy=True, legacy_ttl=10_000)
        reaped = set(counts["names_reaped"])
        assert reaped == {"labelled-old", "legacy-unlabelled"}
        assert "enterprise-mcp-server-aa" not in reaped

    def test_legacy_off_ignores_unlabelled(self):
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        labelled: list = []
        all_managed = [_dep("legacy", age=1_000_000, labelled=False)]
        _set_list(prov, labelled=labelled, all_managed=all_managed)
        counts = prov.reap_once(ttl_seconds=100, dry_run=False, legacy=False)
        assert counts["reaped"] == 0


class TestSelectorAndCount:
    def test_count_active_jobs(self):
        prov = _make_provisioner()
        _set_list(prov, labelled=[_dep("a"), _dep("b"), _dep("c")])
        assert prov.count_active_jobs() == 3

    def test_list_uses_opt_in_selector(self):
        prov = _make_provisioner()
        prov.apps_v1.list_namespaced_deployment.return_value = SimpleNamespace(items=[])
        prov.count_active_jobs()
        _, kwargs = prov.apps_v1.list_namespaced_deployment.call_args
        assert kwargs["label_selector"] == JOB_LABEL_SELECTOR
        assert "coding-job=true" in kwargs["label_selector"]


# ---------------------------------------------------------------------------
# Reaper scheduler module
# ---------------------------------------------------------------------------
class TestReaperModule:
    def test_parse_exclude_names(self):
        assert reaper.parse_exclude_names("a, b ,,c") == frozenset({"a", "b", "c"})
        assert reaper.parse_exclude_names("") == frozenset()

    def test_is_leader_true(self):
        assert reaper.is_leader() is True

    def test_sweep_log_level_warns_on_errors(self):
        import logging

        # A sweep that hit reap errors surfaces at WARNING (alertable);
        # a clean sweep stays at INFO.
        assert reaper.sweep_log_level({"reaped": 3, "errors": 1}) == logging.WARNING
        assert reaper.sweep_log_level({"reaped": 3, "errors": 0}) == logging.INFO
        assert reaper.sweep_log_level({"reaped": 0}) == logging.INFO
        # Malformed/absent errors value must not blow up the log path.
        assert reaper.sweep_log_level({"errors": "oops"}) == logging.INFO
        assert reaper.sweep_log_level({}) == logging.INFO

    async def test_run_one_sweep_invokes_reap_once_with_settings(self):
        fake_prov = MagicMock()
        fake_prov.reap_once.return_value = {"reaped": 2}
        fake_settings = SimpleNamespace(
            reaper_ttl_seconds=123,
            reap_exclude_names="x, y",
            reaper_dry_run=False,
            reap_legacy_unlabelled=True,
            reap_legacy_ttl_seconds=999,
        )
        with (
            patch.object(reaper, "get_settings", return_value=fake_settings),
            patch(
                "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
                return_value=fake_prov,
            ),
        ):
            result = await reaper.run_one_sweep()
        assert result == {"reaped": 2}
        fake_prov.reap_once.assert_called_once_with(
            ttl_seconds=123,
            exclude_names=frozenset({"x", "y"}),
            dry_run=False,
            legacy=True,
            legacy_ttl=999,
        )

    async def test_run_one_sweep_provisioner_unavailable_returns_none(self):
        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            side_effect=RuntimeError("no kube config"),
        ):
            assert await reaper.run_one_sweep() is None

    async def test_reaper_loop_runs_one_sweep_then_stops(self):
        stop = asyncio.Event()
        calls = []

        async def _fake_sweep():
            calls.append(1)
            stop.set()
            return {"reaped": 0}

        with patch.object(reaper, "run_one_sweep", _fake_sweep):
            await asyncio.wait_for(reaper._reaper_loop(stop), timeout=5)
        assert len(calls) == 1

    async def test_reaper_loop_survives_sweep_exception(self):
        stop = asyncio.Event()
        calls = []

        async def _boom():
            calls.append(1)
            stop.set()
            raise RuntimeError("sweep blew up")

        with patch.object(reaper, "run_one_sweep", _boom):
            await asyncio.wait_for(reaper._reaper_loop(stop), timeout=5)
        assert len(calls) == 1  # loop caught it and exited cleanly on stop

    async def test_start_and_stop_reaper(self):
        from fastapi import FastAPI

        app = FastAPI()
        with patch.object(reaper, "run_one_sweep", AsyncMock(return_value=None)):
            reaper.start_reaper(app)
            assert app.state.reaper_task is not None
            await asyncio.sleep(0)
            await reaper.stop_reaper(app)
        assert app.state.reaper_task.done()


# ---------------------------------------------------------------------------
# Reap -> registry row removal (the resurrection loop)
#
# Measured 2026-09-13: the reaper deleted Deployment
# ``it-flow1-bump-fastmcp-to-3-4-7`` at 16:37:18Z and
# ``ContainerManager.ensure_container`` re-provisioned it from the surviving
# Redis registry row at 16:40:56Z — 3m38s later, on a different replica. The
# reap decision has to land in BOTH stores or it is silently undone.
#
# ⛔ The removal lives in the reap loop, NOT inside ``_cleanup_resources``:
# that method is SHARED with the provision-failure rollback in
# ``provision_from_github``, which also fires for a FAILED re-provision driven
# by ``ensure_container``. The registry row is the only surviving copy of
# ``repo_url`` for a Deployment-gone container, so removing it there would make
# a transient provisioning error permanently unrecoverable.
# ---------------------------------------------------------------------------
class _FakeRegistryRedis:
    """Minimal dict-backed stand-in for the registry's Redis hash.

    Only ``hdel`` is exercised by the reap path. ``hdel_error``, when set, is
    raised instead — which is how a REAL ``redis`` client fails, and therefore
    the only fixture that can prove the registry's own
    ``except (RedisError, OSError)`` handler is what produces ``UNCERTAIN``.
    """

    def __init__(self) -> None:
        self.h: dict[str, dict[str, str]] = {}
        self.hdel_error: BaseException | None = None

    def hdel(self, key, *fields):
        if self.hdel_error is not None:
            raise self.hdel_error
        d = self.h.get(key, {})
        removed = 0
        for f in fields:
            if f in d:
                del d[f]
                removed += 1
        return removed


class TestReapRemovesRegistryRow:
    @pytest.fixture
    def fake_redis(self):
        """Registry backed by a reachable fake Redis — i.e. PRODUCTION's shape.

        ⚠️ Deliberately NOT the ``_MEMORY_FALLBACK`` path. That dict is per-pod
        and is not the durable store, so a reap driven through it cannot
        distinguish "the row is gone" from "the HDEL never landed" — which is
        the whole defect under test. Redis-up is the only fixture in which
        ``REMOVED``/``ABSENT`` mean anything.
        """
        import src.core.container_metadata_registry as reg_mod

        client = _FakeRegistryRedis()
        with patch.object(reg_mod, "_redis_client", return_value=client):
            yield client

    @pytest.fixture
    def rows(self, fake_redis):
        """The durable hash's field dict, for seeding and inspection."""
        import src.core.container_metadata_registry as reg_mod

        return fake_redis.h.setdefault(reg_mod.REGISTRY_HASH_KEY, {})

    @pytest.fixture
    def no_redis(self):
        """Redis UNREACHABLE: ``_redis_client`` returns None, as in production
        during an outage. Yields the per-pod fallback dict."""
        import src.core.container_metadata_registry as reg_mod

        with patch.object(reg_mod, "_redis_client", return_value=None):
            reg_mod._MEMORY_FALLBACK.clear()
            yield reg_mod._MEMORY_FALLBACK
            reg_mod._MEMORY_FALLBACK.clear()

    @staticmethod
    def _row(name: str) -> str:
        return json.dumps({"container_name": name, "repo_url": f"https://x/{name}"})

    def test_reaped_row_removed_and_skipped_row_kept(self, rows):
        rows["old"] = self._row("old")
        rows["fresh"] = self._row("fresh")
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[_dep("fresh", age=1), _dep("old", age=500)])

        counts = prov.reap_once(ttl_seconds=100, dry_run=False)

        assert counts["names_reaped"] == ["old"]
        assert counts["registry_errors"] == 0
        assert "old" not in rows, "reaped container's registry row must be gone"
        assert "fresh" in rows, "a skipped container's row must be untouched"

    def test_dry_run_removes_nothing(self, rows):
        rows["old"] = self._row("old")
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[_dep("old", age=500)])

        counts = prov.reap_once(ttl_seconds=100, dry_run=True)

        assert counts["would_reap"] == 1
        assert "old" in rows

    def test_absent_row_is_a_no_op_not_an_error(self, rows):
        """A row Redis CONFIRMS is absent is a no-op — the one case in which
        "had no registry row" may be logged as fact."""
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[_dep("no-row", age=500)])

        counts = prov.reap_once(ttl_seconds=100, dry_run=False)

        assert counts["reaped"] == 1
        assert counts["registry_errors"] == 0
        assert rows == {}

    # -- the failure-visibility half: both arms a REAL registry can exhibit --
    #
    # ⛔ Neither arm below raises out of ``remove``/``remove_with_outcome``:
    # ``remove`` swallows ``RedisError``/``OSError`` into a bool and
    # ``_redis_client`` swallows everything into ``None``. So an ``except``
    # clause at the reap site cannot detect either, and a fixture that makes
    # ``remove`` raise (e.g. ``side_effect=RuntimeError``) proves nothing about
    # production. These drive the real code paths instead.

    def test_hdel_raising_rediserror_is_counted_not_reported_as_absent(self, fake_redis, rows):
        """ARM 1 — Redis reachable, HDEL fails. The row SURVIVES."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        rows["old"] = self._row("old")
        fake_redis.hdel_error = RedisConnectionError("connection lost")
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[_dep("old", age=500)])

        counts = prov.reap_once(ttl_seconds=100, dry_run=False)

        assert counts["reaped"] == 1, "the Deployment delete happened and stays counted"
        assert counts["errors"] == 0
        assert counts["registry_errors"] == 1, "a lost HDEL must be visible, not silent"
        assert "old" in rows, "the row really did survive — this is the resurrection risk"
        assert reaper.sweep_log_level(counts) == logging.WARNING

    def test_unreachable_redis_is_counted_not_reported_as_absent(self, no_redis):
        """ARM 2 — ``_redis_client() is None``: the silent arm.

        ``remove`` pops the EMPTY per-pod fallback and returns False, which the
        old bool-only code logged as "had no registry row" with no error
        anywhere. There is no exception to catch here at all.
        """
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[_dep("old", age=500)])

        counts = prov.reap_once(ttl_seconds=100, dry_run=False)

        assert counts["reaped"] == 1
        assert counts["errors"] == 0
        assert counts["registry_errors"] == 1
        assert reaper.sweep_log_level(counts) == logging.WARNING

    def test_fallback_pop_is_not_a_durable_delete(self, no_redis):
        """⭐ A per-pod fallback pop must NOT be reported as success.

        ``remove`` returns True here (its historical contract, preserved), but
        the durable Redis row — which this process cannot even see — survives
        and reappears when Redis returns. So the reap site must still count
        this UNCERTAIN. This is the case the "check for a False" shape misses.
        """
        no_redis["old"] = {"container_name": "old", "repo_url": "https://x/old"}
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[_dep("old", age=500)])

        counts = prov.reap_once(ttl_seconds=100, dry_run=False)

        assert counts["reaped"] == 1
        assert counts["registry_errors"] == 1, "a fallback pop proves nothing about Redis"
        assert "old" not in no_redis, "the fallback pop itself still happens"

    def test_registry_failure_does_not_mask_the_deployment_deletion(self, fake_redis, rows):
        """Counting order: a registry failure can never un-count a deletion."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        rows["old"] = self._row("old")
        rows["older"] = self._row("older")
        fake_redis.hdel_error = RedisConnectionError("redis gone")
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[_dep("old", age=500), _dep("older", age=600)])

        counts = prov.reap_once(ttl_seconds=100, dry_run=False)

        # The deletions happened and stay counted; the registry failure is
        # reported on its own counter and never aborts the sweep.
        assert counts["reaped"] == 2
        assert counts["names_reaped"] == ["old", "older"]
        assert counts["errors"] == 0
        assert counts["registry_errors"] == 2
        assert prov._cleanup_resources.call_count == 2

    def test_unforeseen_registry_raise_is_counted_not_swallowed(self, rows):
        """The backstop ``except``: an UNFORESEEN raise is not "row is gone".

        ⚠️ This fixture is synthetic ON PURPOSE and proves only the backstop.
        It is NOT evidence about either production failure mode — those two are
        covered by the ARM 1 / ARM 2 tests above, which raise nothing.
        """
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[_dep("old", age=500)])

        with patch.object(_mod, "ContainerMetadataRegistry") as reg_cls:
            reg_cls.return_value.remove_with_outcome.side_effect = RuntimeError("boom")
            counts = prov.reap_once(ttl_seconds=100, dry_run=False)

        assert counts["reaped"] == 1
        assert counts["registry_errors"] == 1

    def test_failed_deployment_delete_keeps_the_row(self, rows):
        """A reap that FAILED must leave the recovery spec in place.

        ⚠️ The failure injected is an ``OSError`` on the Deployment delete, and
        that choice is load-bearing: ``_cleanup_resources`` catches ONLY
        ``ApiException``, so a transport-level ``OSError``/``RuntimeError`` is
        the reachable way for the reap loop's ``errors`` branch to fire. An
        ``ApiException`` would be swallowed inside ``_cleanup_resources`` and
        never reach it — see
        ``test_api_error_on_deployment_delete_reads_as_success`` for that gap.
        """
        rows["bad"] = self._row("bad")
        prov = _make_provisioner()
        prov.apps_v1.delete_namespaced_deployment.side_effect = OSError("connection reset")
        _set_list(prov, labelled=[_dep("bad", age=500)])

        counts = prov.reap_once(ttl_seconds=100, dry_run=False)

        assert counts["errors"] == 1
        assert counts["reaped"] == 0
        assert "bad" in rows

    def test_api_error_on_deployment_delete_reads_as_success(self, rows):
        """⚠️ PINS A KNOWN GAP, not a guarantee — read before "fixing" this.

        ``_cleanup_resources`` catches ``ApiException`` per-resource and only
        LOGS non-404s, returning normally. So a 500 on the Deployment delete is
        indistinguishable from success at the reap site: ``reaped`` increments
        and the registry row is forgotten while the Deployment is still alive.

        Harm is bounded, which is why this is pinned rather than changed here:
        the Deployment survives, so ``get_container`` reconstructs the spec
        (including ``repo_url``) from the live object's annotations, and the
        next sweep re-reaps the still-expired Deployment. Making the branch
        reachable would mean letting ``_cleanup_resources`` propagate — and it
        is SHARED with ``provision_from_github``'s rollback, where raising
        would replace the original provisioning error with a teardown one.
        """
        from kubernetes.client.rest import ApiException

        rows["bad"] = self._row("bad")
        prov = _make_provisioner()
        prov.apps_v1.delete_namespaced_deployment.side_effect = ApiException(status=500)
        _set_list(prov, labelled=[_dep("bad", age=500)])

        counts = prov.reap_once(ttl_seconds=100, dry_run=False)

        assert counts["errors"] == 0, "the 500 is swallowed — documented gap"
        assert counts["reaped"] == 1
        assert counts["registry_errors"] == 0
        assert "bad" not in rows, "row forgotten though the Deployment may live"

    def test_cleanup_resources_alone_never_touches_the_registry(self, rows):
        """⭐ Gate: an INVOLUNTARILY lost Deployment stays recoverable.

        ``_cleanup_resources`` is also the provision-failure rollback (and so
        runs on a failed ``ensure_container`` re-provision). It must leave the
        row — the sole store of ``repo_url`` — alone, or that container can
        never be recovered by ``ensure_container`` again.
        """
        rows["ext-lost"] = self._row("ext-lost")
        prov = _make_provisioner()

        prov._cleanup_resources("ext-lost")

        assert "ext-lost" in rows, "involuntary loss must NOT drop the recovery spec"
        assert json.loads(rows["ext-lost"])["repo_url"] == "https://x/ext-lost"
        prov.apps_v1.delete_namespaced_deployment.assert_called_once()

    def test_sweep_log_level_warns_on_registry_errors(self):
        import logging as _logging

        assert reaper.sweep_log_level({"errors": 0, "registry_errors": 1}) == _logging.WARNING
        assert reaper.sweep_log_level({"errors": 0, "registry_errors": 0}) == _logging.INFO

    def test_sweep_log_level_malformed_key_cannot_suppress_its_sibling(self):
        """A malformed value in ONE counter must not hide the OTHER's failures.

        Measured regression this pins: with both counters coerced inside a
        single ``try``, ``{"errors": 5, "registry_errors": None}`` returned
        INFO — 5 real reap failures silenced by an unrelated malformed key
        (origin/main, which summed nothing, returned WARNING for that input).
        Both directions are asserted: a fix that only handles the reported
        orientation is half a fix.
        """
        import logging as _logging

        assert reaper.sweep_log_level({"errors": 5, "registry_errors": None}) == (
            _logging.WARNING
        ), "malformed registry_errors must not suppress 5 real reap errors"
        assert reaper.sweep_log_level({"errors": None, "registry_errors": 5}) == (
            _logging.WARNING
        ), "malformed errors must not suppress 5 real registry errors"
        # Non-int-coercible strings take the same path (ValueError, not TypeError).
        assert reaper.sweep_log_level({"errors": 5, "registry_errors": "oops"}) == (
            _logging.WARNING
        ), "unparseable registry_errors must not suppress 5 real reap errors"
        assert reaper.sweep_log_level({"errors": "oops", "registry_errors": 5}) == (
            _logging.WARNING
        ), "unparseable errors must not suppress 5 real registry errors"


# ---------------------------------------------------------------------------
# Registry ghosts: rows whose Deployment is gone for a reason OTHER than this
# reaper (40 rows were found saying ``running`` with no Deployment). The sweep
# may remove a row only on CONFIRMED absence and must remove nothing on error.
# ---------------------------------------------------------------------------
class _FakeGhostRedis(_FakeRegistryRedis):
    """Adds ``hkeys`` (the ghost sweep's read) to the dict-backed fake."""

    def __init__(self) -> None:
        super().__init__()
        self.hkeys_error: BaseException | None = None

    def hkeys(self, key):
        if self.hkeys_error is not None:
            raise self.hkeys_error
        return list(self.h.get(key, {}))


def _api_error(status: int):
    from kubernetes.client.rest import ApiException

    return ApiException(status=status, reason="test")


class TestReapRegistryGhosts:
    @pytest.fixture
    def redis(self):
        import src.core.container_metadata_registry as reg_mod

        client = _FakeGhostRedis()
        client.h[reg_mod.REGISTRY_HASH_KEY] = {
            "ghost": json.dumps({"container_name": "ghost", "status": "running"}),
            "live": json.dumps({"container_name": "live", "status": "running"}),
        }
        with patch.object(reg_mod, "_redis_client", return_value=client):
            yield client

    @staticmethod
    def _rows(redis):
        import src.core.container_metadata_registry as reg_mod

        return redis.h[reg_mod.REGISTRY_HASH_KEY]

    @staticmethod
    def _prov(*, live=("live",), read=None):
        """Provisioner whose LIST returns ``live`` and whose per-name READ 404s
        for every name not in ``live`` unless ``read`` overrides it."""
        prov = _make_provisioner()
        _set_list(prov, all_managed=[_dep(n) for n in live])

        def _read(name, namespace):
            if read is not None:
                return read(name)
            if name in live:
                return _dep(name)
            raise _api_error(404)

        prov.apps_v1.read_namespaced_deployment.side_effect = _read
        return prov

    def test_ghost_removed_live_kept(self, redis):
        prov = self._prov()
        counts = prov.reap_registry_ghosts(dry_run=False, grace_seconds=0)
        assert counts["ghosts_removed"] == 1
        assert counts["ghost_names_removed"] == ["ghost"]
        assert counts["ghost_errors"] == 0
        assert "ghost" not in self._rows(redis)
        assert "live" in self._rows(redis), "a row WITH a Deployment must be kept"

    def test_dry_run_removes_nothing(self, redis):
        prov = self._prov()
        counts = prov.reap_registry_ghosts(dry_run=True, grace_seconds=0)
        assert counts["ghosts_would_remove"] == 1
        assert counts["ghosts_removed"] == 0
        assert set(self._rows(redis)) == {"ghost", "live"}

    def test_list_api_error_removes_nothing(self, redis):
        prov = self._prov()
        prov.apps_v1.list_namespaced_deployment.side_effect = _api_error(500)
        counts = prov.reap_registry_ghosts(dry_run=False, grace_seconds=0)
        assert counts["ghost_errors"] == 1
        assert counts["ghosts_removed"] == 0
        assert set(self._rows(redis)) == {"ghost", "live"}
        prov.apps_v1.read_namespaced_deployment.assert_not_called()

    @pytest.mark.parametrize(
        "exc", [_api_error(500), _api_error(403), _api_error(0), OSError("conn reset")]
    )
    def test_read_error_is_not_absence(self, redis, exc):
        """Only a 404 proves absence; a 5xx/403/transport error removes nothing."""

        def _read(name):
            raise exc

        prov = self._prov(read=_read)
        counts = prov.reap_registry_ghosts(dry_run=False, grace_seconds=0)
        assert counts["ghost_errors"] == 1
        assert counts["ghosts_removed"] == 0
        assert "ghost" in self._rows(redis)

    def test_redis_unreachable_removes_nothing(self):
        import src.core.container_metadata_registry as reg_mod

        with patch.object(reg_mod, "_redis_client", return_value=None):
            prov = self._prov()
            counts = prov.reap_registry_ghosts(dry_run=False, grace_seconds=0)
        assert counts["ghost_errors"] == 1
        assert counts["ghosts_removed"] == 0
        prov.apps_v1.list_namespaced_deployment.assert_not_called()

    def test_hkeys_error_removes_nothing(self, redis):
        from redis.exceptions import RedisError

        redis.hkeys_error = RedisError("down")
        prov = self._prov()
        counts = prov.reap_registry_ghosts(dry_run=False, grace_seconds=0)
        assert counts["ghost_errors"] == 1
        assert set(self._rows(redis)) == {"ghost", "live"}

    def test_unlabelled_deployment_found_by_read_is_kept(self, redis):
        """A Deployment missing the managed labels is absent from the LIST but
        the per-name READ finds it — the row must survive."""
        prov = self._prov(live=(), read=lambda name: _dep(name))
        counts = prov.reap_registry_ghosts(dry_run=False, grace_seconds=0)
        assert counts["ghosts_removed"] == 0
        assert set(self._rows(redis)) == {"ghost", "live"}

    def test_grace_holds_first_sighting(self, redis):
        prov = self._prov()
        counts = prov.reap_registry_ghosts(dry_run=False, grace_seconds=3600)
        assert counts["ghosts_pending"] == 1
        assert counts["ghosts_removed"] == 0
        assert "ghost" in self._rows(redis)

    def test_hdel_failure_is_counted(self, redis):
        from redis.exceptions import RedisError

        redis.hdel_error = RedisError("boom")
        prov = self._prov()
        counts = prov.reap_registry_ghosts(dry_run=False, grace_seconds=0)
        assert counts["ghosts_removed"] == 0
        assert counts["ghost_errors"] == 1

    def test_never_names_are_not_touched(self, redis):
        import src.core.container_metadata_registry as reg_mod

        self._rows(redis)["fr-test-verification"] = json.dumps({"container_name": "x"})
        prov = self._prov()
        prov.reap_registry_ghosts(dry_run=False, grace_seconds=0)
        assert "fr-test-verification" in redis.h[reg_mod.REGISTRY_HASH_KEY]

    def test_sweep_log_level_warns_on_ghost_errors(self):
        assert reaper.sweep_log_level({"ghost_errors": 1}) == logging.WARNING

    async def test_run_one_sweep_merges_ghost_counts(self):
        fake_prov = MagicMock()
        fake_prov.reap_once.return_value = {"reaped": 0}
        fake_prov.reap_registry_ghosts.return_value = {"ghosts_removed": 3}
        fake_settings = SimpleNamespace(
            reaper_ttl_seconds=1,
            reap_exclude_names="",
            reaper_dry_run=True,
            reap_legacy_unlabelled=False,
            reap_legacy_ttl_seconds=1,
            reaper_forget_registry_ghosts=True,
            reaper_registry_ghost_grace_seconds=77,
        )
        with (
            patch.object(reaper, "get_settings", return_value=fake_settings),
            patch(
                "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
                return_value=fake_prov,
            ),
        ):
            result = await reaper.run_one_sweep()
        assert result == {"reaped": 0, "ghosts_removed": 3}
        fake_prov.reap_registry_ghosts.assert_called_once_with(
            dry_run=True, grace_seconds=77, exclude_names=frozenset()
        )


# ---------------------------------------------------------------------------
# The reaper deletes the code graph at the source (983 orphaned graphs /
# ~1.15M rows measured in coding-analysis). coding-analysis validates the
# container before dropping its graph, so the delete MUST precede the
# Deployment delete, and it must never block the reap.
# ---------------------------------------------------------------------------
class _FakeGraphRedis(_FakeRegistryRedis):
    def hget(self, key, field):
        return self.h.get(key, {}).get(field)


class TestReapDeletesCodeGraph:
    @pytest.fixture
    def rows(self):
        import src.core.container_metadata_registry as reg_mod

        client = _FakeGraphRedis()
        with patch.object(reg_mod, "_redis_client", return_value=client):
            yield client.h.setdefault(reg_mod.REGISTRY_HASH_KEY, {})

    @staticmethod
    def _tenanted_dep(name: str, tenant: str | None) -> SimpleNamespace:
        dep = _dep(name, age=500)
        if tenant is not None:
            dep.metadata.annotations = {"coding-job/tenant-id": tenant}
        return dep

    def test_graph_deleted_with_registry_tenant_before_deployment(self, rows, _graph_ops):
        rows["old"] = json.dumps(
            {
                "container_name": "old",
                "api_endpoint": "http://x/old",
                "code_graph_id": "",
                "codebase_path": "/app/old/repo",
                "repo_url": "https://x/old",
                "branch": "main",
                "status": "running",
                "tenant_id": "tenant-a",
            }
        )
        order: list[str] = []
        _graph_ops.delete_container_graph.side_effect = lambda *a, **k: (
            order.append("graph"),
            {"success": True, "deleted_nodes": 7, "deleted_edges": 4},
        )[1]
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock(side_effect=lambda n: order.append("deployment"))
        # Deployment stamp names a DIFFERENT tenant: the registry row must win.
        _set_list(prov, labelled=[self._tenanted_dep("old", "tenant-dep"), _dep("fresh", age=1)])

        counts = prov.reap_once(ttl_seconds=100, dry_run=False)

        _graph_ops.delete_container_graph.assert_awaited_once_with("old", tenant_id="tenant-a")
        assert order == ["graph", "deployment"]
        assert counts["graphs_deleted"] == 1
        assert counts["graph_errors"] == 0
        assert counts["names_reaped"] == ["old"]
        assert "old" not in rows

    def test_tenant_falls_back_to_deployment_stamp_without_registry_row(self, rows, _graph_ops):
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[self._tenanted_dep("old", "tenant-dep")])
        prov.reap_once(ttl_seconds=100, dry_run=False)
        _graph_ops.delete_container_graph.assert_awaited_once_with("old", tenant_id="tenant-dep")

    @pytest.mark.parametrize(
        "failure",
        [
            {"return_value": {"success": False, "message": "HTTP 500"}},
            {"side_effect": RuntimeError("boom")},
        ],
        ids=["refused", "raised"],
    )
    def test_graph_failure_still_reaps(self, rows, _graph_ops, failure, caplog):
        _graph_ops.delete_container_graph = AsyncMock(**failure)
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[self._tenanted_dep("old", "t")])

        with caplog.at_level(logging.ERROR, logger=_mod.logger.name):
            counts = prov.reap_once(ttl_seconds=100, dry_run=False)

        prov._cleanup_resources.assert_called_once_with("old")
        assert counts["reaped"] == 1
        assert counts["graph_errors"] == 1
        assert counts["graph_names_failed"] == ["old"]
        assert "ORPHANED" in caplog.text
        assert reaper.sweep_log_level(counts) == logging.WARNING

    def test_already_absent_counted_not_error(self, rows, _graph_ops):
        _graph_ops.delete_container_graph.return_value = {"success": True, "already_absent": True}
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[self._tenanted_dep("old", "t")])
        counts = prov.reap_once(ttl_seconds=100, dry_run=False)
        assert counts["graphs_absent"] == 1
        assert counts["graph_errors"] == 0

    def test_dry_run_and_skipped_never_delete_graph(self, rows, _graph_ops):
        prov = _make_provisioner()
        prov._cleanup_resources = MagicMock()
        _set_list(prov, labelled=[_dep("old", age=500), _dep("fresh", age=1)])
        prov.reap_once(ttl_seconds=100, dry_run=True)
        prov.reap_once(ttl_seconds=100_000, dry_run=False)
        _graph_ops.delete_container_graph.assert_not_awaited()


class TestRegistryGhostGraphLeftVisible:
    def test_ghost_removal_warns_graph_left_and_never_calls_delete(self, _graph_ops, caplog):
        import src.core.container_metadata_registry as reg_mod

        client = _FakeGhostRedis()
        client.h[reg_mod.REGISTRY_HASH_KEY] = {
            "ghost": json.dumps({"container_name": "ghost", "status": "running"})
        }
        prov = TestReapRegistryGhosts._prov(live=())
        with (
            patch.object(reg_mod, "_redis_client", return_value=client),
            caplog.at_level(logging.WARNING, logger=_mod.logger.name),
        ):
            counts = prov.reap_registry_ghosts(dry_run=False, grace_seconds=0)

        assert counts["ghost_names_removed"] == ["ghost"]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "ghost" in r.getMessage() and "manual cleanup" in r.getMessage() for r in warnings
        )
        _graph_ops.delete_container_graph.assert_not_awaited()


# ---------------------------------------------------------------------------
# Credential refresh (App installation tokens live ~50 min)
# ---------------------------------------------------------------------------
def _dep_with_annotations(name: str, annotations: dict | None = None) -> SimpleNamespace:
    labels = {"app": "codegraph-repo", "repo": name, "managed-by": "code-graph-helper"}
    labels["coding-job"] = "true"
    meta = SimpleNamespace(
        name=name,
        labels=labels,
        annotations=dict(annotations or {}),
        creation_timestamp=datetime.now(UTC),
    )
    return SimpleNamespace(metadata=meta)


def _secret(*, minted_at: int | None, age: float = 10_000) -> SimpleNamespace:
    annotations = (
        {_mod.GIT_TOKEN_REFRESHED_ANNOTATION: str(minted_at)} if minted_at is not None else {}
    )
    meta = SimpleNamespace(
        name="job-git-token",
        annotations=annotations,
        creation_timestamp=datetime.now(UTC) - timedelta(seconds=age),
    )
    return SimpleNamespace(metadata=meta)


def _credential(token: str | None, ok: bool = True, reason: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(ok=ok, token=token, reason=reason, error=None)


class TestRefreshExpiringGitTokens:
    """A token minted at provisioning time expires in ~50 min; the git-clone init
    container can run long after that (sandbox recreation, eviction, node reboot)
    and would then fail with GitHub's ``Invalid username or token``. The sweep
    keeps the Secret fresh so that replay can succeed."""

    def _prov(self, deps, *, manager=None):
        prov = _make_provisioner()
        _set_list(prov, labelled=deps)
        prov.projects = MagicMock()
        prov.core_v1.read_namespaced_secret.return_value = _secret(minted_at=None)
        mgr = manager if manager is not None else MagicMock()
        if manager is None:
            mgr.get_app_clone_credential = AsyncMock(return_value=_credential("ghs_fresh"))
        return prov, mgr

    @pytest.mark.asyncio
    async def test_stale_token_is_reminted_and_patched(self):
        dep = _dep_with_annotations(
            "cv-1",
            {
                _mod.REPO_URL_ANNOTATION: "https://github.com/example-org/py2rs",
                _mod.TENANT_ANNOTATION: "t-1",
            },
        )
        prov, mgr = self._prov([dep])
        # Refresh stamp far in the past; creation_timestamp also old → stale.
        prov.core_v1.read_namespaced_secret.return_value = _secret(minted_at=1_000)
        with patch.object(_mod, "get_github_token_manager", lambda: mgr):
            counts = await prov.refresh_expiring_git_tokens(max_age_seconds=2400)

        assert counts["refreshed"] == 1
        mgr.get_app_clone_credential.assert_awaited_once()
        kwargs = mgr.get_app_clone_credential.await_args.kwargs
        assert kwargs["repo_url"] == "https://github.com/example-org/py2rs"
        assert kwargs["tenant_id"] == "t-1"
        body = prov.core_v1.patch_namespaced_secret.call_args.kwargs["body"]
        assert body["data"]["GIT_TOKEN"] == "Z2hzX2ZyZXNo"  # ghs_fresh
        assert _mod.GIT_TOKEN_REFRESHED_ANNOTATION in body["metadata"]["annotations"]

    @pytest.mark.asyncio
    async def test_fresh_token_is_left_alone(self):
        dep = _dep_with_annotations("cv-2", {_mod.REPO_URL_ANNOTATION: "https://github.com/o/r"})
        prov, mgr = self._prov([dep])
        now = int(datetime.now(UTC).timestamp())
        prov.core_v1.read_namespaced_secret.return_value = _secret(minted_at=now)
        with patch.object(_mod, "get_github_token_manager", lambda: mgr):
            counts = await prov.refresh_expiring_git_tokens(max_age_seconds=2400)

        assert counts["fresh"] == 1
        assert counts["refreshed"] == 0
        prov.core_v1.patch_namespaced_secret.assert_not_called()

    @pytest.mark.asyncio
    async def test_byo_pat_job_is_never_replaced_with_an_app_token(self):
        dep = _dep_with_annotations(
            "cv-3",
            {
                _mod.REPO_URL_ANNOTATION: "https://github.com/o/r",
                _mod.SECRET_CONFIG_ANNOTATION: "cfg-1",
            },
        )
        prov, mgr = self._prov([dep])
        prov.core_v1.read_namespaced_secret.return_value = _secret(minted_at=1_000)
        with patch.object(_mod, "get_github_token_manager", lambda: mgr):
            counts = await prov.refresh_expiring_git_tokens(max_age_seconds=2400)

        assert counts["skipped"] == 1
        mgr.get_app_clone_credential.assert_not_awaited()
        prov.core_v1.patch_namespaced_secret.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_minted_token_leaves_the_existing_credential_in_place(self):
        dep = _dep_with_annotations("cv-4", {_mod.REPO_URL_ANNOTATION: "https://github.com/o/r"})
        mgr = MagicMock()
        mgr.get_app_clone_credential = AsyncMock(
            return_value=_credential(None, ok=True, reason="tenant_required")
        )
        prov, _ = self._prov([dep], manager=mgr)
        prov.core_v1.read_namespaced_secret.return_value = _secret(minted_at=1_000)
        with patch.object(_mod, "get_github_token_manager", lambda: mgr):
            counts = await prov.refresh_expiring_git_tokens(max_age_seconds=2400)

        assert counts["errors"] == 1
        assert counts["refreshed"] == 0
        prov.core_v1.patch_namespaced_secret.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_secret_is_skipped_not_an_error(self):
        from kubernetes.client.rest import ApiException

        dep = _dep_with_annotations("cv-5", {_mod.REPO_URL_ANNOTATION: "https://github.com/o/r"})
        prov, mgr = self._prov([dep])
        prov.core_v1.read_namespaced_secret.side_effect = ApiException(status=404)
        with patch.object(_mod, "get_github_token_manager", lambda: mgr):
            counts = await prov.refresh_expiring_git_tokens(max_age_seconds=2400)

        assert counts["skipped"] == 1
        assert counts["errors"] == 0

    @pytest.mark.asyncio
    async def test_dry_run_reports_without_writing(self):
        dep = _dep_with_annotations("cv-6", {_mod.REPO_URL_ANNOTATION: "https://github.com/o/r"})
        prov, mgr = self._prov([dep])
        prov.core_v1.read_namespaced_secret.return_value = _secret(minted_at=1_000)
        with patch.object(_mod, "get_github_token_manager", lambda: mgr):
            counts = await prov.refresh_expiring_git_tokens(max_age_seconds=2400, dry_run=True)

        assert counts["would_refresh"] == ["cv-6"]
        assert counts["refreshed"] == 0
        prov.core_v1.patch_namespaced_secret.assert_not_called()

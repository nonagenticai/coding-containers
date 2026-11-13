"""Background reaper for stale per-job container Deployments (Phase C-1).

Each provisioned repo becomes a Kubernetes Deployment whose pod sleeps forever
holding an ``emptyDir`` workspace. Nothing deleted them, so they accumulated
indefinitely. This module runs a periodic sweep (gated behind
``CODING_CONTAINERS_REAPER_ENABLED``) that reaps Deployments past their TTL via
the provisioner's existing ``_cleanup_resources`` teardown.

Design notes:
- DECIDES off the CLUSTER (label-selected Deployments) — never off the Redis
  metadata registry — so it is immune to a registry wipe and a stale row can
  never cause a reap. ⚠️ It does, however, WRITE the registry after the fact:
  once a Deployment delete succeeds, ``_forget_reaped_metadata`` HDELs that
  container's row, because otherwise ``ContainerManager.ensure_container``
  re-provisions the container minutes later from the surviving row and silently
  undoes the reap (measured 2026-09-13: reaped 16:37:18Z, resurrected
  16:40:56Z). Read = cluster only; write = the reaped name only.
- A SEPARATE registry-ghost pass (``reap_registry_ghosts``, 2026-09-26) HDELs
  rows whose Deployment is gone for any OTHER reason — 40 rows were found still
  saying ``running`` with no Deployment behind them. Because such a row is the
  sole store of ``repo_url``, it is removed only after a per-name READ returns
  404 AND the absence has held for ``reaper_registry_ghost_grace_seconds``; any
  Redis/k8s error removes nothing. It honours ``reaper_dry_run``.
- All blocking kubernetes-client calls are off-loaded with ``asyncio.to_thread``
  so the FastAPI event loop is never stalled.
- Inert by default: with ``reaper_enabled=False`` nothing runs; with it on but
  ``reaper_dry_run=True`` (the default) a sweep only logs intended reaps.

Observability notes:
- Every completed sweep logs its counts dict (``scanned``/``reaped``/``would_reap``/
  ``skipped``/``errors``/``registry_errors``/``names_reaped``). That structured line is
  the reaper's audit trail on the cluster log pipeline (Loki). A sweep that hit reap
  ERRORS or REGISTRY errors is
  logged at WARNING (see ``sweep_log_level``) so it rises above the INFO summary
  and is alertable without a dedicated metric.
- A Prometheus counter/gauge was considered but is intentionally NOT added: no
  coding-* application service is scraped today (no ServiceMonitor targets these
  namespaces; the OpenTelemetry deps are declared but no MeterProvider is wired),
  so a ``/metrics`` endpoint would be inert. Wiring fleet-wide scraping is a
  separate observability decision; until then the WARNING-on-errors log is the
  actionable signal.
- The reaper has no "active in-use" guard: a Deployment older than
  ``reaper_ttl_seconds`` (24h) is reapable regardless of an in-flight exec. This
  is an accepted limitation — the 24h TTL far exceeds normal analysis-pod
  lifetimes and a live in-place reconcile re-stamps ``created-at`` (refreshing
  the lease), so a genuinely-active pod stays fresh. Revisit only if a >24h
  long-running job workload is introduced.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from src.config.settings import get_settings

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

_MIN_INTERVAL_SECONDS = 30
_STOP_TIMEOUT_SECONDS = 10


def parse_exclude_names(raw: str) -> frozenset[str]:
    """Parse the CSV ``reap_exclude_names`` env value into a name set."""
    return frozenset(part.strip() for part in (raw or "").split(",") if part.strip())


def _failure_count(result: dict[str, Any], key: str) -> int:
    """One sweep failure counter, read independently of its siblings.

    Returns 0 when ``key`` is absent or its value cannot be read as an int.
    Per-key on purpose: coercing both counters inside ONE ``try`` let a
    malformed value in either key abort the sum and drop the OTHER key's real
    failures to INFO. Scoping the tolerance to a single key makes that
    cross-suppression unrepresentable.
    """
    try:
        return int(result.get(key, 0))
    except TypeError, ValueError:
        return 0


def sweep_log_level(result: dict[str, Any]) -> int:
    """Level for the per-sweep summary line.

    WARNING when a completed sweep hit reap ``errors`` (individual deletions
    failed) or ``registry_errors`` (a Deployment was deleted but its registry
    row could NOT be confirmed gone — the HDEL failed or never reached Redis —
    so a surviving row lets ``ensure_container`` resurrect it) so it surfaces
    above the routine INFO summary and is alertable on the log pipeline; INFO
    otherwise.

    Each key is coerced SEPARATELY (see ``_failure_count``), so a malformed or
    absent value counts as 0 for that key ALONE and can never suppress the
    other key's escalation. ⚠️ That tolerance is still fail-open for the key
    that is malformed: an unreadable counter reads as "no failures" for itself.
    Production cannot reach this — ``reap_once`` always seeds both as ints —
    so it guards only a future non-reaper caller.

    ⚠️ ``registry_errors`` counts UNCERTAINTY, not confirmed survival: the
    reaper cannot read back a row it just failed to reach. Treat a nonzero
    value as "this name may resurrect", which is the actionable claim.
    """
    failures = (
        _failure_count(result, "errors")
        + _failure_count(result, "registry_errors")
        + _failure_count(result, "ghost_errors")
        + _failure_count(result, "graph_errors")
    )
    return logging.WARNING if failures > 0 else logging.INFO


def is_leader() -> bool:
    """Whether this replica should reap.

    coding-containers runs a single replica today, so this is always True.
    Even at >1 replica redundant reaps are harmless — deletion is idempotent
    (``_cleanup_resources`` swallows 404) and selector-scoped — so leader
    election would only save API calls. Left as a forward-compat no-op.
    """
    return True


async def run_one_sweep() -> dict[str, Any] | None:
    """Run a single reap sweep. Returns the counts dict, or None if the
    provisioner could not be obtained this tick (transient, non-fatal)."""
    from src.core.kubernetes_container_provisioner import get_k8s_container_provisioner

    settings = get_settings()
    try:
        prov = get_k8s_container_provisioner()
    except (RuntimeError, OSError) as exc:
        logger.warning("[reaper] provisioner unavailable this sweep: %s", exc)
        return None

    exclude = parse_exclude_names(settings.reap_exclude_names)
    result = await asyncio.to_thread(
        prov.reap_once,
        ttl_seconds=settings.reaper_ttl_seconds,
        exclude_names=exclude,
        dry_run=settings.reaper_dry_run,
        legacy=settings.reap_legacy_unlabelled,
        legacy_ttl=settings.reap_legacy_ttl_seconds,
    )
    if getattr(settings, "reaper_forget_registry_ghosts", False):
        ghosts = await asyncio.to_thread(
            prov.reap_registry_ghosts,
            dry_run=settings.reaper_dry_run,
            grace_seconds=settings.reaper_registry_ghost_grace_seconds,
            exclude_names=exclude,
        )
        result = {**result, **ghosts}

    # Credential refresh. ⛔ Deliberately NOT gated on ``reaper_dry_run``: that flag
    # governs DESTRUCTIVE reaps, and a refresh only re-mints and overwrites the
    # workspace's own token — it destroys nothing and is what keeps a pod that is
    # re-created hours after provisioning able to clone at all. Zero disables it.
    refresh_seconds = int(getattr(settings, "git_token_refresh_seconds", 0) or 0)
    if refresh_seconds > 0:
        refreshed = await prov.refresh_expiring_git_tokens(max_age_seconds=refresh_seconds)
        result = {**result, "token_refresh": refreshed}
    return result


async def _reaper_loop(stop: asyncio.Event) -> None:
    """Sweep on an interval until ``stop`` is set. Reads settings each tick so
    cadence / dry-run / TTL can change without a restart. A failed sweep is
    logged and the loop continues."""
    while not stop.is_set():
        if is_leader():
            try:
                result = await run_one_sweep()
                if result is not None:
                    logger.log(sweep_log_level(result), "[reaper] sweep result: %s", result)
            except Exception:
                logger.exception("[reaper] sweep failed")
        interval = max(_MIN_INTERVAL_SECONDS, int(get_settings().reaper_interval_seconds))
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            pass  # interval elapsed → next sweep


def start_reaper(app: FastAPI) -> None:
    """Start the background reaper task, stashing handles on ``app.state``."""
    stop = asyncio.Event()
    app.state.reaper_stop = stop
    app.state.reaper_task = asyncio.create_task(_reaper_loop(stop))


async def stop_reaper(app: FastAPI) -> None:
    """Signal the reaper to stop and await it (bounded)."""
    stop = getattr(app.state, "reaper_stop", None)
    task = getattr(app.state, "reaper_task", None)
    if stop is not None:
        stop.set()
    if task is not None:
        try:
            await asyncio.wait_for(task, timeout=_STOP_TIMEOUT_SECONDS)
        except TimeoutError, asyncio.CancelledError:
            task.cancel()

"""Workspace toolchains: which image and which resource profile a workspace gets.

A workspace is ``python`` (the default, and every workspace that existed before
this module) or ``python+rust`` (py2rust plan slice A1: the conveyor writes Rust
in it, so it needs cargo + a loopback Postgres next to the checkout).

⛔ ADDITIVE ONLY (plan gate G1). Every function here returns ``None`` / the
caller's existing value for ``python`` so the python pod spec is byte-identical
to the pre-toolchain one — ``tests/unit/test_toolchain_python_spec_snapshot.py``
pins that against a fixture generated from origin/main before this file existed.

Image map — ``PROVISIONER_IMAGES_JSON``: a JSON object ``{toolchain: image}``,
e.g. ``{"python+rust": "ghcr.io/nonagenticai/codegraph-runtime-rust:sha-…@sha256:…"}``.
``python`` is NOT read from it: the python image stays ``PROVISIONER_IMAGE``
(already digest-pinned in gitops), so introducing the map cannot move it.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

Toolchain = Literal["python", "python+rust"]

PYTHON: Toolchain = "python"
PYTHON_RUST: Toolchain = "python+rust"
TOOLCHAINS: tuple[Toolchain, ...] = (PYTHON, PYTHON_RUST)

#: Deployment annotation carrying a NON-default toolchain. Absent ⇒ python, so a
#: python Deployment carries no new annotation (G1) and every Deployment that
#: predates this field reads back as python.
TOOLCHAIN_ANNOTATION = "coding-job/toolchain"

#: Local/dev default for the python+rust image. In the cluster gitops pins it by
#: DIGEST via PROVISIONER_IMAGES_JSON (pods are IfNotPresent, so only a new
#: digest propagates — same contract as PROVISIONER_IMAGE).
DEFAULT_RUST_IMAGE = "ghcr.io/nonagenticai/codegraph-runtime-rust:latest"

#: Main-container resources for python+rust. cargo builds are memory- and
#: CPU-hungry (rustc + linker on a few hundred crates) and a Rust target dir is
#: GBs, so the python profile (2Gi / 500m, LimitRange-default storage) would OOM
#: or crawl. Bounded by the coding-jobs LimitRange max (cpu 4, memory 8Gi,
#: ephemeral-storage 40Gi). The CPU REQUEST stays small for the same reason the
#: python one does: the namespace quota is requests.cpu 10, and a request only
#: buys placement, the limit is the ceiling.
RUST_RESOURCES: dict[str, dict[str, str]] = {
    "requests": {"memory": "1Gi", "cpu": "100m", "ephemeral-storage": "4Gi"},
    "limits": {"memory": "6Gi", "cpu": "4", "ephemeral-storage": "30Gi"},
}


# ---------------------------------------------------------------------------
# Disk (py2rust gap #4-lite): workspace storage scales with the repositories in it
# ---------------------------------------------------------------------------
#
# A py2rust workspace holds the Rust target AND the N Python sources it is rewritten
# from (``POST {ws}/repos`` → ``/workspace/.repos/``), all in the ``repo-data``
# emptyDir. That volume had no sizeLimit and the container's ephemeral-storage was a
# fixed 30Gi (python+rust) or the namespace default (python), whatever N was — so a
# large N could only end in an EVICTED pod. A running pod cannot be resized, so the
# workspace is sized at CREATE from ``expected_repos``: ephemeral-storage request/limit
# and a ``repo-data`` sizeLimit. Nothing measures disk at add time.
#
# Bounded by the target namespace's LimitRange (read 2026-10-27):
# per container defaultRequest 2Gi, default limit 10Gi, max 40Gi ephemeral-storage.

#: LimitRange ``max`` ephemeral-storage per container in coding-jobs.
EPHEMERAL_MAX_GI = 40
#: Disk granted per expected secondary repository (limit) and reserved (request).
PER_REPO_LIMIT_GI = 2
PER_REPO_REQUEST_GI = 1
#: Room left OUTSIDE repo-data under the container limit: the ``tmp`` emptyDir
#: (detached exec output, service logs), the writable layer and container logs.
NON_REPO_HEADROOM_GI = 2
#: (request, limit) Gi with no secondary repository. python = the LimitRange
#: defaults it already gets implicitly; python+rust = RUST_RESOURCES.
_BASE_DISK_GI: dict[str, tuple[int, int]] = {"python": (2, 10), "python+rust": (4, 30)}
#: Most repositories a hint can usefully ask for (python: 10 + 2*15 = 40Gi).
MAX_EXPECTED_REPOS = (EPHEMERAL_MAX_GI - 10) // PER_REPO_LIMIT_GI
#: Deployment annotation carrying the repo-data size, so a registry-miss rebuild reports it.
DISK_BUDGET_ANNOTATION = "coding-job/disk-budget-gi"


@dataclass(frozen=True)
class DiskProfile:
    request_gi: int
    limit_gi: int
    repo_data_gi: int


def disk_profile(toolchain: str, expected_repos: int | None) -> DiskProfile | None:
    """Sized ephemeral storage for a workspace, or ``None`` to leave the spec as is.

    ⛔ ``None`` for a python workspace without a hint: that pod spec stays
    byte-identical (G1, ``test_toolchain_python_spec_snapshot.py``). Every
    python+rust workspace is sized (its N defaults to 0, keeping the 4Gi/30Gi it
    already had and adding the sizeLimit). The limit is capped at the LimitRange
    max, so a hint past ``MAX_EXPECTED_REPOS`` buys nothing more.
    """
    tc = PYTHON_RUST if toolchain == PYTHON_RUST else PYTHON
    if tc == PYTHON and expected_repos is None:
        return None
    n = max(0, int(expected_repos or 0))
    base_request, base_limit = _BASE_DISK_GI[tc]
    limit = min(EPHEMERAL_MAX_GI, base_limit + PER_REPO_LIMIT_GI * n)
    request = min(limit, base_request + PER_REPO_REQUEST_GI * n)
    return DiskProfile(request, limit, limit - NON_REPO_HEADROOM_GI)


# ---------------------------------------------------------------------------
# Memory / CPU (py2rust gap #4 full): a workspace that RUNS N source services
# ---------------------------------------------------------------------------
#
# To record a source system's behaviour the workspace runs its services (thin
# FastAPI apps, each a uvicorn process + its supervisor) next to their
# dependencies (Redis, Postgres), all in the ONE workspace pod's main container.
# Every process there shares that container's cgroup memory limit, so the limit
# is the only real bound and a request it cannot hold must be refused — never
# admitted to end in an OOM-killed pod (which takes the checkout with it).
#
# The model, from the target namespace's LimitRange (read 2026-10-27):
# per container max cpu 4, memory 8Gi.
#
#   limit = TOOLING_RESERVE (exec, tests, cargo, the agent's own tools)
#         + sum(dependency reserve)  (redis / postgres, when running)
#         + sum(service memory hint) (each service's peak RSS incl. supervisor)
#
# It runs once, at CREATE (``memory_profile``: sizes the pod from the hints, 409
# when it cannot fit 8Gi). Inside the pod nothing is measured: the container's
# memory limit is the bound (an OOM kill).

#: LimitRange ``max`` per container in coding-jobs.
MEMORY_MAX_MI = 8192
CPU_MAX_M = 4000
#: Memory kept for everything that is not a service or a dependency. python: the
#: exec'd tests and linters (half the 2Gi python default, which keeps an unsized
#: python workspace at exactly the old 4 services x 256Mi). python+rust: rustc +
#: the linker at up to 4 jobs; two thirds of the 6Gi python+rust default, which
#: keeps that default's 2Gi for services. A STARTING POINT, not a measurement:
#: re-size from `kubectl -n coding-jobs top pods` once Rust targets build there.
TOOLING_RESERVE_MI: dict[str, int] = {"python": 1024, "python+rust": 4096}
#: The container memory limit a workspace has WITHOUT a services hint (the
#: provisioner's python default and RUST_RESOURCES).
DEFAULT_LIMIT_MI: dict[str, int] = {"python": 2048, "python+rust": 6144}
_DEFAULT_REQUEST_MI: dict[str, int] = {"python": 256, "python+rust": 1024}
_DEFAULT_CPU_LIMIT_M: dict[str, int] = {"python": 500, "python+rust": 4000}
#: Per-service memory hint (MiB): the service's peak RSS including its children
#: and its supervisor (a small stdlib-only python process). The default is the
#: old per-service share (4 services in the 1Gi left of a 2Gi python pod).
SERVICE_MEMORY_DEFAULT_MI = 256
SERVICE_MEMORY_MIN_MI = 128
SERVICE_MEMORY_MAX_MI = 4096
#: What a running dependency holds. redis: ws-redis caps the dataset at
#: maxmemory 256mb (noeviction) + ~64Mi server overhead. postgres: ws-pg's
#: shared_buffers 128MB + backends + work_mem.
DEPENDENCY_MEMORY_MI: dict[str, int] = {"redis": 320, "postgres": 512}
#: Live services in a workspace NOT sized by a services hint: origin/main's limit.
MAX_SERVICES_PER_WORKSPACE = 4
#: Largest ``expected_services`` hint (the 8Gi max less the smallest reserve, at the
#: smallest per-service hint) and the live-service cap of a workspace sized by one.
MAX_EXPECTED_SERVICES = (MEMORY_MAX_MI - TOOLING_RESERVE_MI["python"]) // SERVICE_MEMORY_MIN_MI


def max_services_of(container: Any) -> int:
    """The live-service cap of a workspace: 4, or the hint bound when it was sized."""
    if getattr(container, "memory_limit_mi", None):
        return MAX_EXPECTED_SERVICES
    return MAX_SERVICES_PER_WORKSPACE


#: CPU limit added per expected service / dependency (idle services cost ~0; this
#: is for recording under load), capped at the LimitRange max.
_CPU_PER_SERVICE_M = 250
#: Deployment annotation with the sized memory limit (read back on registry miss).
MEMORY_LIMIT_ANNOTATION = "coding-job/memory-limit-mi"
_ROUND_MI = 256


@dataclass(frozen=True)
class MemoryProfile:
    request_mi: int
    limit_mi: int
    cpu_limit_m: int
    need_mi: int


class MemoryBudgetError(ValueError):
    """The requested services + dependencies do not fit one workspace pod."""

    def __init__(self, *, need_mi: int, max_mi: int, reserve_mi: int, toolchain: str) -> None:
        self.need_mi = need_mi
        self.max_mi = max_mi
        self.reserve_mi = reserve_mi
        self.toolchain = toolchain
        super().__init__(
            f"the requested services and dependencies need {need_mi} MiB of memory "
            f"(tooling reserve {reserve_mi} MiB for {toolchain} included), more than one "
            f"workspace pod may have ({max_mi} MiB, the coding-jobs LimitRange maximum): "
            "split the services over several workspaces or lower service_memory_mi"
        )


def memory_profile(
    toolchain: str,
    expected_services: int | None,
    service_memory_mi: int | None = None,
    dependencies: Sequence[str] | None = None,
) -> MemoryProfile | None:
    """Sized memory/CPU for a workspace that will run services, or ``None``.

    ``None`` when no hint was given (neither ``expected_services`` nor
    ``dependencies``): the pod keeps its toolchain's resources exactly (G1). The
    limit never drops below the toolchain default. Raises ``MemoryBudgetError``
    when the set cannot fit ``MEMORY_MAX_MI`` — refused, never silently shrunk.
    """
    if expected_services is None and not dependencies:
        return None
    tc = PYTHON_RUST if toolchain == PYTHON_RUST else PYTHON
    n = max(0, int(expected_services or 0))
    hint = int(service_memory_mi or SERVICE_MEMORY_DEFAULT_MI)
    deps = sorted(set(dependencies or ()))
    reserve = TOOLING_RESERVE_MI[tc]
    need = reserve + sum(DEPENDENCY_MEMORY_MI[d] for d in deps) + n * hint
    if need > MEMORY_MAX_MI:
        raise MemoryBudgetError(
            need_mi=need, max_mi=MEMORY_MAX_MI, reserve_mi=reserve, toolchain=tc
        )
    rounded = -(-need // _ROUND_MI) * _ROUND_MI
    limit = min(MEMORY_MAX_MI, max(DEFAULT_LIMIT_MI[tc], rounded))
    request = min(limit, _DEFAULT_REQUEST_MI[tc] + 64 * n)
    cpu = min(CPU_MAX_M, _DEFAULT_CPU_LIMIT_M[tc] + _CPU_PER_SERVICE_M * (n + len(deps)))
    return MemoryProfile(request_mi=request, limit_mi=limit, cpu_limit_m=cpu, need_mi=need)


class ToolchainImageUnavailableError(ValueError):
    """``PROVISIONER_IMAGES_JSON`` is set but unusable for a requested toolchain."""


def normalize_toolchain(value: Any) -> Toolchain:
    """Anything falsy → python; a known name → itself; else ``ValueError``."""
    if value is None or value == "":
        return PYTHON
    if value in TOOLCHAINS:
        return value
    raise ValueError(f"unknown toolchain {value!r}; expected one of {', '.join(TOOLCHAINS)}")


def toolchain_of(workspace: Any) -> Toolchain:
    """The toolchain of a ContainerInfo-like object; unknown / missing ⇒ python.

    Fails toward the NARROWER policy: anything that is not exactly
    ``python+rust`` is treated as python.
    """
    return PYTHON_RUST if getattr(workspace, "toolchain", None) == PYTHON_RUST else PYTHON


def toolchain_from_annotations(annotations: dict[str, str] | None) -> Toolchain:
    value = (annotations or {}).get(TOOLCHAIN_ANNOTATION)
    return PYTHON_RUST if value == PYTHON_RUST else PYTHON


def _images_map() -> dict[str, str]:
    raw = os.environ.get("PROVISIONER_IMAGES_JSON", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolchainImageUnavailableError(
            f"PROVISIONER_IMAGES_JSON is not valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in parsed.items()
    ):
        raise ToolchainImageUnavailableError(
            "PROVISIONER_IMAGES_JSON must be a JSON object of non-empty strings"
        )
    return parsed


def image_for(toolchain: Toolchain, python_image: str) -> str:
    """The image for ``toolchain``. ``python`` always returns ``python_image``."""
    if toolchain == PYTHON:
        return python_image
    return _images_map().get(toolchain) or DEFAULT_RUST_IMAGE


def resources_for(toolchain: Toolchain) -> dict[str, dict[str, str]] | None:
    """Main-container resources override, or ``None`` for python (keep as is)."""
    if toolchain == PYTHON_RUST:
        return {k: dict(v) for k, v in RUST_RESOURCES.items()}
    return None

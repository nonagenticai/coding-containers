"""The one place a workspace's backend is decided.

Two call paths used to decide this independently and disagree:

    src/routers/container_exec.py   Kubernetes if ".svc.cluster.local" in the
                                    endpoint, else agent-host, else Docker.
    src/core/container_manager.py   Docker, falling back to Kubernetes only
                                    when Docker is unavailable.

In cluster with a working Docker socket, the same workspace therefore ran on a
different backend depending on which entry point the caller came through — and
only the router path applied the command allowlist. This module reproduces the
ROUTER's precedence, because that is the one production traffic takes and the
one that matches how workspaces are actually provisioned.
"""

from __future__ import annotations

import logging
from typing import Any

from .agent_host import AgentHostRuntime
from .docker import DockerRuntime
from .kubernetes import KubernetesRuntime
from .protocol import WorkspaceRuntime
from .types import RuntimeKind

logger = logging.getLogger(__name__)

__all__ = ["resolve_runtime", "resolve_runtime_kind"]

#: An in-cluster Service address. A workspace whose endpoint carries this was
#: provisioned as a Kubernetes Deployment.
CLUSTER_DOMAIN = ".svc.cluster.local"


def resolve_runtime_kind(workspace: Any, *, core: Any = None) -> RuntimeKind:
    """Decide which backend owns this workspace.

    Order, and why:

    1. An explicit ``runtime_kind`` on the workspace, when the provisioner
       recorded one. Everything else is reconstruction.
    2. An in-cluster endpoint means Kubernetes. Checked before Docker
       availability precisely because ``container_manager`` checking Docker
       first is the bug: a Docker socket being present says nothing about where
       a given workspace actually lives.
    3. A known agent-host id.
    4. Docker.
    """
    declared = getattr(workspace, "runtime_kind", None)
    if declared:
        try:
            return RuntimeKind(declared)
        except ValueError:
            # Metadata carrying an unknown value is a glitch, not a reason to
            # refuse to run anything. Fall through and reconstruct from the
            # endpoint — which is how every workspace predating this field is
            # resolved anyway, so the fallback is a tested path, not a guess.
            logger.warning(
                "workspace %r has unrecognised runtime_kind %r; reconstructing",
                getattr(workspace, "container_name", workspace),
                declared,
            )

    raw_endpoint = (
        getattr(workspace, "internal_endpoint", "") or getattr(workspace, "api_endpoint", "") or ""
    )
    # Normalise ONCE, here. A malformed record can put a non-string in this
    # field, and treating that as "no information" is both safer and simpler
    # than guarding every later use: a bare `CLUSTER_DOMAIN in <non-str>` either
    # raises TypeError or, worse, answers quietly via someone else's
    # __contains__, and a truthy non-string would otherwise skip the
    # no-evidence tiebreak below while carrying no evidence at all.
    endpoint = raw_endpoint if isinstance(raw_endpoint, str) else ""
    if CLUSTER_DOMAIN in endpoint:
        return RuntimeKind.KUBERNETES

    name = getattr(workspace, "container_name", None) or (
        workspace if isinstance(workspace, str) else None
    )
    if core is not None and name and getattr(core, "uses_agent_host", False):
        if core.has_agent_host_id(name):
            return RuntimeKind.AGENT_HOST

    # Last resort: we know nothing about this workspace — no recorded kind, no
    # endpoint, no agent-host id. Only here, where there is genuinely no
    # evidence to go on, may availability act as a tiebreak.
    #
    # It matters because a registry miss lands here. In cluster, `execute_command`
    # used to reach Kubernetes for EVERY workspace (Docker being unavailable
    # there), so defaulting an unknown name to Docker would turn "we have no
    # metadata for this" into "cannot run", for workspaces that work today.
    # Note this is a tiebreak among unknowns, NOT the old precedence: a
    # workspace whose endpoint identifies it is already resolved above and is
    # unaffected by whether Docker happens to be up.
    if not endpoint and core is not None:
        available = getattr(core, "is_docker_available", None)
        if callable(available) and not available():
            return RuntimeKind.KUBERNETES

    return RuntimeKind.DOCKER


def resolve_runtime(
    workspace: Any,
    *,
    core: Any = None,
    provisioner: Any = None,
    docker_client: Any = None,
) -> WorkspaceRuntime:
    """Build the backend adapter for this workspace.

    Raises ``RuntimeError`` when the chosen backend has no usable client, rather
    than falling through to a different one. A silent fallback is how the two
    original selectors came to disagree: each was individually reasonable and
    together they made the backend a function of the caller.
    """
    kind = resolve_runtime_kind(workspace, core=core)

    if kind is RuntimeKind.KUBERNETES:
        if provisioner is None:
            from src.core.kubernetes_container_provisioner import (
                get_k8s_container_provisioner,
            )

            provisioner = get_k8s_container_provisioner()
        if provisioner is None or not provisioner.is_available():
            raise RuntimeError(
                "workspace resolves to the Kubernetes runtime but the provisioner is not available"
            )
        return KubernetesRuntime(provisioner)

    if kind is RuntimeKind.AGENT_HOST:
        if core is None:
            raise RuntimeError(
                "workspace resolves to the agent-host runtime but no manager core was supplied"
            )
        return AgentHostRuntime(core)

    client = docker_client if docker_client is not None else getattr(core, "client", None)
    # Availability VALIDATES the choice; it must never MAKE it. That distinction
    # is the whole bug this module removes: `container_manager` asked
    # "is Docker available?" first and let the answer decide where an in-cluster
    # workspace ran. Asked here, after the backend is already settled, it can
    # only say "the backend this workspace belongs to is down" — which is a
    # truthful error rather than a silent reroute.
    available = getattr(core, "is_docker_available", None)
    if client is None or (callable(available) and not available()):
        raise RuntimeError(
            "workspace resolves to the Docker runtime but the Docker client is not available"
        )
    return DockerRuntime(client)

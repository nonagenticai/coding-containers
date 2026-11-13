"""One ownership gate, shared by the routes that execute work in a named workspace.

WHY THIS IS A MODULE AND NOT A COPY IN EACH ROUTER
--------------------------------------------------
#133 established the invariant — resolve the workspace through the tenant-aware
``get_container(name, tenant_id=...)`` and fail CLOSED — across 12 sites, and it
took four audit rounds because each round found a DIFFERENT site. A gate that is
retyped per router is a gate that drifts per router: the next reviewer has to
diff three implementations to answer one question. There is exactly one
implementation here, so the question "does this route fail closed?" is answered
by whether it calls this function.

⛔ This does NOT call ``_recover_tenant_id``. ``ContainerManager.get_container``
already applies it on the registry-miss path (through
``reconstruct_container_info``) and already fails CLOSED when neither the
``coding-job/tenant-id`` annotation nor the ``tenant-id`` label yields an owner.
Recovering the owner a second time at this layer would be a second gate over the
same question, free to disagree with the first — the same reason #133
deliberately did not re-recover at the ``docker_service`` site, where
``reconstruct_container_info`` had already applied it.
"""

from __future__ import annotations

import logging
from typing import Any

from coding_client.models.responses import not_found
from src.core.container_manager import get_container_manager

logger = logging.getLogger(__name__)


def authorize_workspace(container_name: str, tenant_id: str, *, route: str) -> Any:
    """Refuse a workspace the caller does not own. Fails CLOSED.

    The routes that call this bound ``tenant_id`` via ``Depends(require_tenant)``
    and then never used it (#135): the caller was AUTHENTICATED, but the target
    workspace was selected by a caller-supplied NAME that was never compared
    against the caller's tenant. Nothing downstream could catch it either —
    ``exec_in_pod(deployment_name=...)`` and ``exec_in_container(docker_id=...)``
    take no tenant at all.

    404, never 403: a 403 confirms the name exists. ``get_container`` returns the
    same ``None`` for "owned by another tenant" and "no such workspace" precisely
    so this layer cannot leak the difference (BOLA), and this function preserves
    that by raising the identical error for both.

    Args:
        container_name: The caller-supplied workspace name.
        tenant_id: The caller's tenant, from ``require_tenant``.
        route: Route name, for the audit log only.

    Returns:
        The caller's own workspace (``ContainerInfo``), so a route can act on
        the record it was authorized for instead of looking it up again.

    Raises:
        HTTPException: 404 if the workspace is absent, foreign, or its owner
            cannot be recovered.
    """
    try:
        container = get_container_manager().get_container(container_name, tenant_id=tenant_id)
    except ValueError, RuntimeError:
        # ValueError is get_container's "no such workspace" (it raises rather
        # than returning None, for backward compatibility with callers that
        # catch it); RuntimeError is a backend fault. Neither may be reported as
        # anything other than absent.
        logger.warning(
            "Refusing %s on workspace %s to tenant %s: workspace not found. "
            "Provision it in this tenant, or call as its owner.",
            route,
            container_name,
            tenant_id,
        )
        raise not_found("Container", container_name)

    if container is None:
        # The gate fired. get_container collapses "owned by another tenant" and
        # "owner UNRECOVERABLE" into one None by design; it logs the
        # unrecoverable case itself, so this line is the cross-tenant signal.
        # WARNING, not INFO: a caller that worked yesterday now gets a 404, and
        # an operator has to be able to see why (#133 shipped the same log on its
        # fail-closed branches for the same reason).
        logger.warning(
            "Refusing %s on workspace %s to tenant %s: the workspace is not owned by this "
            "tenant, or its owner could not be recovered. This is a REFUSAL, not a missing "
            "workspace. Re-dispatch under the owning tenant, or re-provision the workspace "
            "so its owner is stamped.",
            route,
            container_name,
            tenant_id,
        )
        raise not_found("Container", container_name)
    return container

"""Canonical repository-workspace API — the surface published in OpenAPI.

A workspace is the stable domain resource: a checkout of a GitHub repository.
The Docker container or Kubernetes Deployment backing it is an implementation
detail, and the public vocabulary should never leak it.

**Mount new canonical route modules here, directly.** Do not add them to
:mod:`.legacy_compat`; that module's routers are also mounted at the hidden
``/api/containers`` alias, so anything routed through it is republished onto the
legacy surface. Keeping the two lists apart is what stops the compatibility
alias from growing every time the canonical API gains a route::

    from .fs import router as fs_router
    router.include_router(fs_router, tags=["Workspaces - Filesystem"])

Operation IDs are declared explicitly on each route decorator. An earlier
version of this module derived the workspace IDs at import time by looping over
``router.routes`` and running ``str.replace("container", "workspace")`` over
every ``operation_id`` and ``summary``. That produced correct IDs for the routes
that existed when it was written, but it made the published contract a function
of substring matching: a future route whose ID merely contained "container"
would be renamed behind its author's back, and ``coding-gateway``'s
``operation_metadata.py`` — which keys role metadata off these exact strings and
is enforced by a CI drift guard — depended on the loop's output rather than on
anything greppable. The IDs are now baked into the decorators verbatim, so the
published spec is unchanged and the coupling is visible.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..containers.git_fetch import router as git_fetch_router
from ..containers.secondary_repos import router as secondary_repos_router
from ..exec_async import router as exec_async_router
from ..workspace_services import router as workspace_services_router
from .legacy_compat import router as legacy_compat_router

router = APIRouter()
router.include_router(legacy_compat_router)
# Canonical-only (never on the /api/containers alias): see the docstring above.
router.include_router(git_fetch_router, tags=["Workspaces"])
router.include_router(exec_async_router, tags=["Workspaces - Exec"])
# P7 multi-repo probe (George 2026-10-12): secondary repos + loopback services.
router.include_router(secondary_repos_router, tags=["Workspaces - Repositories"])
router.include_router(workspace_services_router, tags=["Workspaces - Services"])

__all__ = ["router"]

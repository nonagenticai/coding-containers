"""The five pre-harmonization routers, re-exported onto the canonical surface.

Everything aggregated here is ALSO mounted at the hidden ``/api/containers``
alias in :mod:`src.main`. That is the whole point of this module: these handlers
predate the workspace vocabulary and must stay callable at both paths while
clients migrate.

It is also the reason this module is a closed set. Adding a route to any router
below publishes it on the legacy surface too, silently and forever, because the
legacy mount includes the same router objects. New canonical routes belong in a
new module mounted from :mod:`src.routers.workspaces` instead — see that
package's docstring.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..agents import router as agents_router
from ..container_dependencies import router as dependencies_router
from ..container_exec import router as exec_router
from ..containers import router as lifecycle_router
from ..package_upgrade import router as package_router

router = APIRouter()
router.include_router(lifecycle_router, tags=["Workspaces"])
router.include_router(dependencies_router, tags=["Workspaces - Dependencies"])
router.include_router(exec_router, tags=["Workspaces - Exec"])
router.include_router(package_router, tags=["Workspaces - Package Management"])
router.include_router(agents_router, tags=["Workspaces - Agents"])

__all__ = ["router"]

"""Router modules for the coding-containers microservice."""

from .agents import router as agents_router
from .container_dependencies import router as container_dependencies_router
from .container_exec import router as container_exec_router
from .containers import router as containers_router
from .health import router as health_router
from .package_upgrade import router as package_upgrade_router
from .validation import router as validation_router
from .workspaces import router as workspaces_router

__all__ = [
    "agents_router",
    "container_dependencies_router",
    "container_exec_router",
    "containers_router",
    "health_router",
    "package_upgrade_router",
    "validation_router",
    "workspaces_router",
]

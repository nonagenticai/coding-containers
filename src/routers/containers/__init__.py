"""Container management router package.

Splits the previous 2064-line ``containers.py`` god-router into per-resource
submodules. The public surface (``from src.routers.containers import router``
plus the helper functions and ``get_container_manager`` patch target) is kept
identical so existing imports and the ~115 ``mock.patch("src.routers.containers.…")``
call sites in tests continue to resolve unchanged.

Submodules
----------
- ``_common``:           shared helpers + late-bound ``get_manager``
- ``crud_read``:         list / get / status / GET-all 405 stub
- ``crud_write``:        create / delete-one / delete-all
- ``files_create``:      mkdir / touch
- ``files_io``:          write-file / list-or-read files
- ``files_manage``:      move / delete files and directories
- ``git_status_commit``: git status + commit + restore + branch
- ``git_push``:          git push-to-fork

Each submodule defines its own ``APIRouter`` with no prefix; this package
assembles the shared implementation mounted canonically at ``/api/workspaces``
and as the hidden compatibility alias ``/api/containers``.
"""

from __future__ import annotations

from fastapi import APIRouter

# Re-export get_container_manager at the package level so existing test patches
# of the form ``patch("src.routers.containers.get_container_manager")`` keep
# working. Submodules look up this attribute via ``src.routers.containers`` at
# call time (see ``_common.get_manager``).
from src.core.container_manager import get_container_manager

from . import (
    crud_read,
    crud_write,
    files_create,
    files_io,
    files_manage,
    git_diff,
    git_push,
    git_status_commit,
)
from ._common import (
    _canonical_workspace_path,
    _find_container_by_name,
    _normalize_workspace_path,
    _sanitize_error_message,
    _sanitize_shell_arg,
    _validate_git_branch_name,
    _validate_git_file_path,
)

router = APIRouter()
router.include_router(crud_read.router)
router.include_router(crud_write.router)
router.include_router(files_create.router)
router.include_router(files_io.router)
router.include_router(files_manage.router)
router.include_router(git_status_commit.router)
router.include_router(git_diff.router)
router.include_router(git_push.router)


__all__ = [
    "_canonical_workspace_path",
    "_find_container_by_name",
    "_normalize_workspace_path",
    "_sanitize_error_message",
    "_sanitize_shell_arg",
    "_validate_git_branch_name",
    "_validate_git_file_path",
    "get_container_manager",
    "router",
]

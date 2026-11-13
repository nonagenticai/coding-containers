"""Package upgrade dispatch — typed, per-manager command construction.

This subpackage owns the matrix of ``manager × mode → command``. The
public-facing route in ``coding-security`` calls into the typed
``POST /api/workspaces/{name}/package-upgrade`` endpoint defined in
``routers.package_upgrade``, which in turn delegates to ``runner.run_upgrade``.

The freeform ``POST /api/workspaces/{name}/exec`` whitelist
(``routers.container_exec``) is intentionally NOT extended — broadening it
for ``mvn``/``gradle``/``go``/etc. would expand the shell-injection surface
of the generic exec API for no reason. Typed endpoint = typed inputs =
no shell interpolation.
"""

from .runner import run_modify, run_upgrade
from .strategies import (
    AddSpec,
    MajorUpgradeSpec,
    ModifyAction,
    PackageSpec,
    UpgradeMode,
    UpgradeStrategy,
    get_strategy,
)

__all__ = [
    "AddSpec",
    "MajorUpgradeSpec",
    "ModifyAction",
    "PackageSpec",
    "UpgradeMode",
    "UpgradeStrategy",
    "get_strategy",
    "run_modify",
    "run_upgrade",
]

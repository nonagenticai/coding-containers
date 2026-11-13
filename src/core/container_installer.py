"""
Container Dependency Installer

Comprehensive installer that detects and installs dependencies for Python and JavaScript projects
in provisioned containers. Supports all major package managers and dependency configurations.

Enhanced with dynamic system dependency detection and installation.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import docker
import docker.errors
from docker.errors import DockerException
from docker.models.containers import Container

from src.core.private_index import python_resolver_env
from src.runtime import (
    CommandNotPermittedError,
    ExecutionProfile,
    RuntimeKind,
    resolve_runtime,
    resolve_runtime_kind,
)

logger = logging.getLogger(__name__)

#: Where a Kubernetes-provisioned workspace's repository actually lives. The
#: provisioner mounts the clone at this path, ``WorkspaceRuntime.run_once``
#: defaults its ``cwd`` to it, and the package-upgrade runner hardcodes it as
#: ``CONTAINER_REPO_ROOT``. There is no per-workspace variation to look up.
K8S_WORKSPACE_ROOT = "/workspace"

#: ``analyze_project`` (dependency-status) scan bounds on a Kubernetes workspace.
ANALYZE_PYTHON_NAMES = frozenset(
    {"pyproject.toml", "setup.py", "Pipfile", "poetry.lock", "pdm.lock"}
)
ANALYZE_JS_NAMES = frozenset(
    {"package.json", "yarn.lock", "pnpm-lock.yaml", "bun.lock", "bun.lockb"}
)
ANALYZE_MAX_DEPTH = 6
ANALYZE_MAX_FILES = 200
ANALYZE_OUTPUT_LIMIT = 256 * 1024

#: The default ``workspace_path`` on ``DependencyInstallRequest``, inherited
#: from the Docker deployment where repositories were cloned under ``/app/data``.
#:
#: On Kubernetes this path does not exist in ANY workspace, so honouring it
#: would turn "the caller sent the default body" into a guaranteed failure and
#: leave the endpoint as unreachable as it was before. It is therefore read as
#: "unspecified" on the Kubernetes branch and replaced with
#: :data:`K8S_WORKSPACE_ROOT`, which is recorded in the result's logs rather
#: than done silently. Any OTHER path the caller names is honoured as given —
#: the substitution can only ever replace a value that could not have worked.
DOCKER_ERA_DEFAULT_WORKSPACE = "/app/data/repo"

#: Wall-clock budget for one dependency-install command in a pod. Resolving and
#: building a real repository's dependency tree is minutes, not seconds; the
#: probe commands around it are cheap and get the shorter budget. Enforced
#: inside the pod by ``exec_in_pod``'s ``timeout`` wrapper, so a hung resolver
#: cannot pin the request open.
K8S_INSTALL_TIMEOUT_SECONDS = 900
K8S_PROBE_TIMEOUT_SECONDS = 60

#: Python resolvers inside a workspace pod reach the fleet's private index,
#: which is fronted by a self-signed CA they do not trust — the install fails
#: with ``invalid peer certificate: UnknownIssuer`` for every repository that
#: depends on a package published there. ``package_upgrade.runner`` already
#: carries this environment for exactly the same reason and reads the same
#: variable; set ``CODING_PRIVATE_INDEX_HOST=""`` to disable it.
#:
#: Passed as ``run_once(env=...)`` rather than baked into the argv. The runtime
#: adapter applies the command allowlist to the argv BEFORE prefixing
#: ``/usr/bin/env``, so the package manager — not ``env`` — is still what gets
#: checked.


def _python_resolver_env() -> dict[str, str]:
    """Trust AND discovery settings for the private index, or empty when disabled.

    Delegates to ``src.core.private_index`` so this and the workspace pod spec
    cannot drift apart. ⚠️ It used to return trust only (``UV_NATIVE_TLS`` /
    ``UV_INSECURE_HOST`` / ``PIP_TRUSTED_HOST``) with no index URL — which meant
    the host was trusted but never queried, so a package that lives ONLY on the
    configured index still resolved as "not found". Trust and discovery are two
    different failures and both have to be answered. Empty when the deployment
    configures no private index (the default).
    """
    return python_resolver_env()


class SystemDependencyError(Exception):
    """Exception raised when system dependencies are missing.

    Provides structured information about what's missing and how to fix it.

    Attributes:
        missing_packages: Instance attribute missing_packages.
        guidance: Instance attribute guidance."""

    def __init__(self, message: str, missing_packages: list[str], guidance: str):
        super().__init__(message)
        self.missing_packages = missing_packages
        self.guidance = guidance


class PackageManager(Enum):
    """PackageManager class."""

    # Python package managers
    POETRY = "poetry"
    UV = "uv"
    PIP = "pip"
    PIPENV = "pipenv"
    PDM = "pdm"
    CONDA = "conda"

    # JavaScript package managers
    NPM = "npm"
    YARN = "yarn"
    PNPM = "pnpm"
    BUN = "bun"


class ProjectType(Enum):
    """ProjectType class."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class DependencyFile:
    """Represents a detected dependency configuration file.

    Attributes:
        path (str): Class attribute path.
        file_type (str): Class attribute file_type.
        language (str): Class attribute language.
        package_manager (PackageManager): Class attribute package_manager.
        priority (int): Class attribute priority.
        content_preview (str): Class attribute content_preview.

    Example:
        >>> obj = DependencyFile()"""

    path: str
    file_type: str
    language: str
    package_manager: PackageManager
    priority: int
    content_preview: str = ""


@dataclass
class InstallationResult:
    """Result of dependency installation process.

    Attributes:
        success (bool): Class attribute success.
        installed_packages (List[str]): Class attribute installed_packages.
        package_managers_used (List[str]): Class attribute package_managers_used.
        errors (List[str]): Class attribute errors.
        installation_time (float): Class attribute installation_time.
        logs (List[str]): Class attribute logs.

    Example:
        >>> obj = InstallationResult()"""

    success: bool
    installed_packages: list[str]
    package_managers_used: list[str]
    errors: list[str]
    installation_time: float
    logs: list[str]


class ContainerInstaller:
    """Comprehensive dependency installer for containerized codebases.

    Handles Python and JavaScript projects with support for all major package managers.
    Enhanced with dynamic system dependency detection and installation.

    Attributes:
        target_container_name: Instance attribute target_container_name.
        project_type: Instance attribute project_type."""

    # Python dependency file patterns with priorities (lower = higher priority)
    PYTHON_PATTERNS = {
        # Primary configuration files (highest priority)
        "pyproject.toml": (PackageManager.POETRY, 1),  # Could be poetry, uv, pdm, etc.
        "poetry.lock": (PackageManager.POETRY, 1),
        "uv.lock": (PackageManager.UV, 1),
        "pdm.lock": (PackageManager.PDM, 1),
        "Pipfile": (PackageManager.PIPENV, 2),
        "Pipfile.lock": (PackageManager.PIPENV, 2),
        # Standard files (medium priority)
        "requirements.txt": (PackageManager.PIP, 3),
        "requirements-dev.txt": (PackageManager.PIP, 4),
        "requirements": (PackageManager.PIP, 4),
        "dev-requirements.txt": (PackageManager.PIP, 4),
        "test-requirements.txt": (PackageManager.PIP, 5),
        "setup.py": (PackageManager.PIP, 6),
        "setup.cfg": (PackageManager.PIP, 6),
        # Conda files
        "environment.yml": (PackageManager.CONDA, 3),
        "environment.yaml": (PackageManager.CONDA, 3),
        "conda-lock.yml": (PackageManager.CONDA, 2),
    }

    # JavaScript dependency file patterns with priorities
    JAVASCRIPT_PATTERNS = {
        "package.json": (PackageManager.NPM, 3),  # Default to npm, will be refined
        "yarn.lock": (PackageManager.YARN, 1),
        "pnpm-lock.yaml": (PackageManager.PNPM, 1),
        "bun.lock": (PackageManager.BUN, 1),  # text lockfile (bun >= 1.2 default)
        "bun.lockb": (PackageManager.BUN, 1),  # legacy binary lockfile
        ".yarnrc.yml": (PackageManager.YARN, 2),  # Yarn berry indicator
        ".npmrc": (PackageManager.NPM, 2),
        ".pnpmrc": (PackageManager.PNPM, 2),
    }

    def __init__(self, target_container_name: str | None = None):
        """Initialize the container installer."""
        self.target_container_name = target_container_name
        self.detected_files: list[DependencyFile] = []
        self.project_type = ProjectType.UNKNOWN

    def _get_manager_requirements(self, manager: PackageManager) -> list[str]:
        """Get the system requirements needed for a package manager.

        Returns minimal requirements - additional dependencies will be
        detected and installed automatically if needed.
        """
        requirements = {
            PackageManager.POETRY: ["curl"],  # For poetry installation
            PackageManager.UV: ["curl"],  # For uv installation
            PackageManager.PIP: ["python3-pip"],  # Basic pip
            PackageManager.PIPENV: ["python3-pip"],
            PackageManager.PDM: ["python3-pip"],
            PackageManager.CONDA: ["wget"],  # For miniconda installation
            PackageManager.NPM: ["nodejs", "npm"],
            PackageManager.YARN: [
                "nodejs",
                "npm",
            ],  # npm needed to install yarn
            PackageManager.PNPM: ["nodejs", "npm"],
            PackageManager.BUN: ["curl"],  # For bun installation
        }
        return requirements.get(manager, [])

    def _check_system_dependency_error(self, output: str) -> bool:
        """Check if an error is due to missing system dependencies."""
        error_patterns = [
            # Compiler errors
            "error: Microsoft Visual C++ 14.0 is required",
            "error: Microsoft Visual C++ 14.0 or greater is required",
            "fatal error:",
            "error: can't find Rust compiler",
            "error: cargo not found in PATH",
            "gcc: command not found",
            "g++: command not found",
            "cc: command not found",
            # Library errors
            "No package 'libxml-2.0' found",
            "libxml2 development packages are not installed",
            "libxslt development packages are not installed",
            "fatal error: Python.h: No such file or directory",
            "fatal error: libxml/xmlversion.h: No such file or directory",
            "fatal error: openssl/opensslv.h: No such file or directory",
            "fatal error: ffi.h: No such file or directory",
            # Config tool errors
            "Could not find",
            "Unable to find",
            "pkg-config: not found",
            "OSError: mysql_config not found",
            "pg_config executable not found",
            "xml2-config: not found",
            "xslt-config: not found",
            # Package specific errors
            "Please make sure the libxml2 and libxslt development packages are installed",
            "error: Setup script exited with error",
            "error: can't find file to patch",
            "fatal error: jpeglib.h: No such file or directory",
            "fatal error: png.h: No such file or directory",
            "fatal error: tiff.h: No such file or directory",
            "fatal error: webp/decode.h: No such file or directory",
        ]

        output_lower = output.lower()
        return any(pattern.lower() in output_lower for pattern in error_patterns)

    def _extract_system_dependencies(self, output: str) -> list[str]:
        """Extract system dependencies from error messages."""
        dependencies = []

        # Common dependency mappings based on error patterns
        dependency_patterns = {
            # Python development
            "Python.h: No such file": ["python3-dev"],
            "python.h": ["python3-dev"],
            # Compilers and build tools
            "vcvarsall.bat": ["build-essential", "gcc", "g++"],
            "Microsoft Visual C++": ["build-essential", "gcc", "g++"],
            "gcc: command not found": ["gcc"],
            "g++: command not found": ["g++"],
            "cc: command not found": ["gcc"],
            "make: command not found": ["make"],
            "cmake": ["cmake"],
            # XML libraries (for lxml)
            "xml2-config": ["libxml2-dev", "libxslt1-dev"],
            "libxml/xmlversion.h": ["libxml2-dev", "libxslt1-dev"],
            "libxml2 development packages": ["libxml2-dev", "libxslt1-dev"],
            "libxslt development packages": ["libxml2-dev", "libxslt1-dev"],
            "Please make sure the libxml2 and libxslt": ["libxml2-dev", "libxslt1-dev"],
            # Database libraries
            "pg_config": ["libpq-dev", "postgresql-client"],
            "mysql_config": ["default-libmysqlclient-dev"],
            "sqlite3.h": ["libsqlite3-dev"],
            # SSL/Crypto libraries
            "openssl/": ["libssl-dev"],
            "opensslv.h": ["libssl-dev"],
            "ffi.h": ["libffi-dev"],
            # Rust/Cargo
            "cargo not found": ["cargo", "rustc"],
            "can't find Rust compiler": ["cargo", "rustc"],
            # Image libraries (for Pillow)
            "jpeglib.h": ["libjpeg-dev"],
            "png.h": ["libpng-dev"],
            "tiff.h": ["libtiff-dev", "libtiff5-dev"],
            "webp/decode.h": ["libwebp-dev"],
            "freetype": ["libfreetype6-dev"],
            # Scientific computing
            "blas": ["libblas-dev", "liblapack-dev"],
            "lapack": ["liblapack-dev"],
            "gfortran": ["gfortran"],
            # Other common libraries
            "zlib.h": ["zlib1g-dev"],
            "bzlib.h": ["libbz2-dev"],
            "lzma.h": ["liblzma-dev"],
            "pcre.h": ["libpcre3-dev"],
            "curl/curl.h": ["libcurl4-openssl-dev"],
            "ldap.h": ["libldap2-dev", "libsasl2-dev"],
            "sasl/sasl.h": ["libsasl2-dev"],
            "event.h": ["libevent-dev"],
            "readline/readline.h": ["libreadline-dev"],
            # Package config
            "pkg-config: not found": ["pkg-config"],
        }

        # Check for each pattern in the output
        output_lower = output.lower()
        for pattern, deps in dependency_patterns.items():
            if pattern.lower() in output_lower:
                dependencies.extend(deps)

        # Remove duplicates while preserving order
        seen = set()
        unique_deps = []
        for dep in dependencies:
            if dep not in seen:
                seen.add(dep)
                unique_deps.append(dep)

        return unique_deps

    def _format_dependency_errors(self, output: str, manager: PackageManager) -> str:
        """Format dependency errors with helpful information."""
        deps = self._extract_system_dependencies(output)

        if not deps:
            return (
                "Unable to determine specific missing dependencies. Check the error output above."
            )

        formatted = "The following system packages are likely missing:\n"
        for dep in deps:
            formatted += f"  - {dep}\n"

        formatted += "\nTo fix this in the container, run:\n"
        formatted += f"  apt-get update && apt-get install -y {' '.join(deps)}\n"

        # Add package-specific guidance
        if manager == PackageManager.PIP:
            if "Python.h" in output:
                formatted += "\nFor Python packages with C extensions, you need python3-dev:\n"
                formatted += "  apt-get install -y python3-dev\n"

        return formatted

    async def _install_system_dependencies(
        self, container: Container, dependencies: list[str]
    ) -> bool:
        """Install system dependencies in the container.

        Args:
            container: Docker container object
            dependencies: List of system packages to install

        Returns:
            True if installation succeeded, False otherwise
        """
        if not dependencies:
            return True

        logger.info("Installing system dependencies: %s", ", ".join(dependencies))

        try:
            # Update package lists first
            update_cmd = "apt-get update"
            update_result = container.exec_run(f"bash -c '{update_cmd}'")

            if update_result.exit_code != 0:
                logger.warning(
                    "Failed to update package lists: %s",
                    update_result.output.decode() if update_result.output else "",
                )

            # Install dependencies
            install_cmd = f"apt-get install -y --no-install-recommends {' '.join(dependencies)}"
            install_result = container.exec_run(f"bash -c '{install_cmd}'")

            if install_result.exit_code == 0:
                logger.info("Successfully installed system dependencies")
                return True
            else:
                output = install_result.output.decode() if install_result.output else ""
                logger.error("Failed to install system dependencies: %s", output)

                # Check for specific errors and provide alternatives
                if "Unable to locate package" in output:
                    missing = self._extract_missing_packages(output)
                    logger.error("The following packages could not be found: %s", missing)

                    # Try alternative package names
                    alternatives = self._get_package_alternatives(missing)
                    if alternatives:
                        logger.info("Trying alternative package names: %s", alternatives)
                        alt_cmd = (
                            f"apt-get install -y --no-install-recommends {' '.join(alternatives)}"
                        )
                        alt_result = container.exec_run(f"bash -c '{alt_cmd}'")

                        if alt_result.exit_code == 0:
                            logger.info("Successfully installed using alternative package names")
                            return True

                return False

        except (DockerException, OSError, RuntimeError) as e:
            logger.error("Exception while installing system dependencies: %s", str(e))
            return False

    def _extract_missing_packages(self, error_output: str) -> list[str]:
        """Extract package names that couldn't be found."""
        missing = []
        lines = error_output.split("\n")
        for line in lines:
            if "Unable to locate package" in line:
                # Extract package name from error message
                parts = line.split("Unable to locate package")
                if len(parts) > 1:
                    pkg = parts[1].strip()
                    missing.append(pkg)
        return missing

    def _get_package_alternatives(self, missing_packages: list[str]) -> list[str]:
        """Get alternative package names for commonly misnamed packages."""
        alternatives = []

        # Common package name mappings
        package_mappings = {
            "python-dev": "python3-dev",
            "python-pip": "python3-pip",
            "libmysqlclient-dev": "default-libmysqlclient-dev",
            "postgresql-dev": "libpq-dev",
            "libjpeg-dev": "libjpeg62-turbo-dev",
            "libpng12-dev": "libpng-dev",
            "libfreetype6-dev": "libfreetype-dev",
            "node": "nodejs",
            "libxml2-devel": "libxml2-dev",
            "libxslt-devel": "libxslt1-dev",
            "openssl-devel": "libssl-dev",
            "libffi-devel": "libffi-dev",
        }

        for pkg in missing_packages:
            if pkg in package_mappings:
                alternatives.append(package_mappings[pkg])
            # Also try removing version numbers
            elif "-dev" in pkg:
                base = pkg.replace("-dev", "")
                # Try with lib prefix if not present
                if not base.startswith("lib"):
                    alternatives.append(f"lib{base}-dev")

        return alternatives

    async def _install_with_python_manager(
        self,
        container: Container,
        workspace_path: str,
        manager: PackageManager,
        files: list[DependencyFile],
    ) -> InstallationResult:
        """Install Python packages with a specific package manager with dynamic dependency resolution."""
        installed = []
        errors = []
        logs = []

        try:
            # Get installation command
            install_cmd = self._get_python_manager_install_cmd(manager, workspace_path)

            if install_cmd:
                logs.append(f"Installing dependencies with {manager.value}")

                # Execute installation command
                if manager == PackageManager.PIP:
                    result = container.exec_run(install_cmd, workdir=workspace_path)
                else:
                    result = container.exec_run(
                        f"bash -c 'cd {workspace_path} && {install_cmd}'",
                        workdir=workspace_path,
                    )

                output = result.output.decode() if result.output else ""
                logs.append(f"Installation output: {output[:1000]}")

                if result.exit_code == 0:
                    installed = self._parse_installed_packages(output, manager)
                    logs.append(f"Successfully installed {len(installed)} packages")
                else:
                    error_msg = f"Installation failed with exit code {result.exit_code}"
                    logs.append(error_msg)

                    # Check for common dependency errors and try to fix them
                    if self._check_system_dependency_error(output):
                        missing_deps = self._extract_system_dependencies(output)
                        logs.append(f"Detected missing system dependencies: {missing_deps}")

                        # Try to install missing system dependencies
                        if missing_deps:
                            logs.append("Attempting to install missing system dependencies...")
                            dep_install_success = await self._install_system_dependencies(
                                container, missing_deps
                            )

                            if dep_install_success:
                                logs.append(
                                    "System dependencies installed, retrying package installation..."
                                )
                                # Retry the installation
                                retry_result = container.exec_run(
                                    (
                                        install_cmd
                                        if manager == PackageManager.PIP
                                        else f"bash -c 'cd {workspace_path} && {install_cmd}'"
                                    ),
                                    workdir=workspace_path,
                                )
                                retry_output = (
                                    retry_result.output.decode() if retry_result.output else ""
                                )

                                if retry_result.exit_code == 0:
                                    installed = self._parse_installed_packages(
                                        retry_output, manager
                                    )
                                    logs.append(
                                        f"Retry successful! Installed {len(installed)} packages"
                                    )
                                    errors.clear()  # Clear errors since we succeeded
                                else:
                                    errors.append(
                                        f"Installation still failed after installing system dependencies: {retry_output}"
                                    )
                            else:
                                errors.append("Failed to install system dependencies")

                        # Add detailed error info if we still failed
                        if errors:
                            detailed_error = (
                                f"\n\nDependency Installation Failed:\n"
                                f"Missing system dependencies detected:\n"
                                f"{self._format_dependency_errors(output, manager)}"
                            )
                            errors.append(detailed_error)
                    else:
                        errors.append(f"{error_msg}: {output}")

        except (RuntimeError, OSError, ValueError) as e:
            error_msg = f"Exception during {manager.value} installation: {str(e)}"
            errors.append(error_msg)
            logs.append(error_msg)

        return InstallationResult(
            success=len(errors) == 0,
            installed_packages=installed,
            package_managers_used=[manager.value],
            errors=errors,
            installation_time=0.0,
            logs=logs,
        )

    def _get_python_manager_install_cmd(
        self,
        manager: PackageManager,
        workspace_path: str,
    ) -> str:
        """Get the installation command for a Python package manager."""
        commands = {
            PackageManager.POETRY: "poetry install --no-interaction",
            # Respect the committed uv.lock: a bare `uv sync` re-resolves the
            # lock to the LATEST versions allowed by pyproject constraints,
            # which silently upgrades away vulnerable (loosely-pinned) deps
            # before any CVE scan runs (masks fixable CVEs → empty sweep).
            # `--frozen` installs the lock as-committed without re-resolving.
            # Fall back to a bare sync when no lock exists (--frozen errors
            # without one). UV is always run via `bash -c`, so the shell
            # conditional is safe.
            PackageManager.UV: "if [ -f uv.lock ]; then uv sync --frozen; else uv sync; fi",
            PackageManager.PIP: "pip install -r requirements.txt",
            PackageManager.PIPENV: "pipenv install --system",
            PackageManager.PDM: "pdm install",
            PackageManager.CONDA: (
                "conda env update -f environment.yml || conda env update -f environment.yaml"
            ),
        }

        return commands.get(manager, "")

    def _parse_installed_packages(self, output: str, manager: PackageManager) -> list[str]:
        """Parse installed package names from installation output."""
        packages = []

        if manager == PackageManager.PIP:
            # Look for "Successfully installed" lines
            lines = output.split("\n")
            for line in lines:
                if "Successfully installed" in line:
                    # Extract package names after "Successfully installed"
                    parts = line.split("Successfully installed", 1)
                    if len(parts) > 1:
                        pkg_line = parts[1].strip()
                        # Split by spaces and extract package names (ignore versions)
                        pkg_parts = pkg_line.split()
                        for part in pkg_parts:
                            if "-" in part:
                                pkg_name = part.split("-")[0]
                                if pkg_name:
                                    packages.append(pkg_name)

        return packages

    def _select_kubernetes_runtime(self, container_name: str) -> tuple[Any, Any] | None:
        """Return ``(runtime, workspace)`` when this workspace runs on Kubernetes.

        ``None`` means "carry on with Docker", and it is returned for every
        outcome that is not a working Kubernetes backend — a workspace that
        resolves to Docker or the agent host, a provisioner that is absent or
        reports itself unavailable, a registry that cannot be reached.

        The two steps are separate on purpose. ``resolve_runtime_kind`` decides
        from the workspace's own metadata and needs no client, so asking it
        first means a Docker deployment never touches the Kubernetes machinery.
        ``resolve_runtime`` is only then asked to build the adapter, and it
        raises rather than silently rerouting when the backend it chose has no
        usable client — which here is a reason to leave the existing Docker path
        alone, not to fail the request.
        """
        try:
            from src.core.container_manager import get_container_manager
            from src.core.container_manager_core import (
                get_container_manager as get_core_manager,
            )

            core = get_core_manager()
            # A registry miss degrades to the bare name, exactly as
            # ``ContainerManager.execute_command`` does: having no metadata for
            # a workspace is not the same as the workspace not existing.
            workspace: Any = get_container_manager().get_container(container_name) or container_name
            if resolve_runtime_kind(workspace, core=core) is not RuntimeKind.KUBERNETES:
                return None
            return resolve_runtime(workspace, core=core), workspace
        except (RuntimeError, ValueError, KeyError, OSError, ImportError) as e:
            logger.debug(
                "No Kubernetes runtime for %s (%s); using the Docker installer",
                container_name,
                e,
            )
            return None

    async def _install_dependencies_via_runtime(
        self,
        runtime: Any,
        workspace: Any,
        container_name: str,
        workspace_path: str,
    ) -> InstallationResult:
        """Install dependencies through a :class:`~src.runtime.WorkspaceRuntime`.

        Every command is sent as an argv LIST under
        :attr:`~src.runtime.ExecutionProfile.CALLER_SUPPLIED`, which is what
        makes ``src.runtime.policy.validate_command`` the enforcement point
        rather than a formality. Two consequences worth stating:

        * The working directory travels as ``cwd``, never as a ``cd X && ...``
          shell string. A shell string would put ``sh`` at argv[0], and ``sh``
          is not on the allowlist — the command would either be refused or, if
          sent as server-authored, would skip the allowlist entirely. Passing
          ``cwd`` keeps the package manager itself at argv[0], so the list is
          actually checked against the thing being run.
        * Anything this method wants to run must already be allowlisted.
          ``uv``, ``npm``, ``bun``, ``pip`` and ``ls`` are (``bun`` since
          2026-09-17, when it was also added to the ``codegraph-runtime`` image
          — 5 org repos ship a ``bun.lock``); ``poetry`` and ``pnpm`` are not,
          and are not used here — they are also not in the image, so the two
          facts agree.

        The tool choices mirror the package-upgrade runner rather than the
        Docker branch above, because the pod runs as an unprivileged uid: a bare
        ``pip install`` resolves to the root-owned system ``site-packages`` and
        fails EACCES. ``uv`` installs into ``/workspace/.venv``, which is where
        the rest of the fleet already looks for the workspace interpreter.
        """
        start_time = time.time()
        logs: list[str] = []
        errors: list[str] = []
        installed_packages: list[str] = []
        package_managers_used: list[str] = []

        workdir = workspace_path
        if not workdir or workdir == DOCKER_ERA_DEFAULT_WORKSPACE:
            workdir = K8S_WORKSPACE_ROOT
            logs.append(
                f"workspace_path {workspace_path!r} does not exist on a Kubernetes "
                f"workspace; using {K8S_WORKSPACE_ROOT}"
            )

        async def run(
            argv: list[str],
            timeout: int = K8S_PROBE_TIMEOUT_SECONDS,
            env: dict[str, str] | None = None,
        ) -> Any:
            return await runtime.run_once(
                workspace,
                argv,
                cwd=workdir,
                env=env,
                timeout=timeout,
                execution_profile=ExecutionProfile.CALLER_SUPPLIED,
            )

        try:
            listing = await run(["ls", "-1", "-a"])
            if not listing.succeeded:
                message = (
                    f"Workspace path {workdir} not found in container "
                    f"{container_name}: {listing.stderr.strip() or listing.stdout.strip()}"
                )
                logs.append(message)
                return InstallationResult(
                    success=False,
                    installed_packages=[],
                    package_managers_used=[],
                    errors=[message],
                    installation_time=time.time() - start_time,
                    logs=logs,
                )

            entries = {line.strip() for line in listing.stdout.splitlines() if line.strip()}
            logs.append(f"Kubernetes workspace {workdir} lists {len(entries)} entries")

            await self._install_python_via_runtime(
                run, entries, logs, errors, installed_packages, package_managers_used
            )
            await self._install_javascript_via_runtime(
                run, entries, logs, errors, installed_packages, package_managers_used
            )

            if not package_managers_used and not errors:
                logs.append(f"No Python or JavaScript dependency manifest found in {workdir}")

        except CommandNotPermittedError as e:
            # The allowlist refused something this method built. That is a bug
            # here, not a caller error, and it means NOTHING was installed —
            # reported as a failure rather than an empty success.
            message = f"Dependency install command refused by the command policy: {e}"
            logger.error(message)
            errors.append(message)
        except (RuntimeError, OSError, ValueError, KeyError) as e:
            message = f"Error installing dependencies on the Kubernetes runtime: {e}"
            logger.error(message)
            errors.append(message)

        return InstallationResult(
            success=not errors,
            installed_packages=installed_packages,
            package_managers_used=sorted(set(package_managers_used)),
            errors=errors,
            installation_time=time.time() - start_time,
            logs=logs,
        )

    async def _install_python_via_runtime(
        self,
        run: Any,
        entries: set[str],
        logs: list[str],
        errors: list[str],
        installed_packages: list[str],
        package_managers_used: list[str],
    ) -> None:
        """Install Python dependencies in a pod, newest-manifest-wins.

        ``uv sync --frozen`` rather than a bare ``uv sync`` whenever a lock is
        committed: a bare sync re-resolves to the newest versions the pyproject
        constraints admit, which silently upgrades vulnerable pins away before
        any CVE scan sees them. Same reasoning, and same command, as
        ``_get_python_manager_install_cmd``.
        """
        requirements = next(
            (name for name in ("requirements.txt", "requirements-dev.txt") if name in entries),
            None,
        )
        if "uv.lock" in entries:
            steps = [["uv", "sync", "--frozen"]]
            manager = PackageManager.UV.value
        elif "pyproject.toml" in entries:
            steps = [["uv", "sync"]]
            manager = PackageManager.UV.value
        elif requirements:
            # No lock step to create the venv, so make one first — `uv pip
            # install` refuses to touch the root-owned system interpreter
            # without --system, which is the behaviour we want.
            steps = [
                ["uv", "venv", "--allow-existing"],
                ["uv", "pip", "install", "-r", requirements],
            ]
            manager = PackageManager.PIP.value
        else:
            return

        resolver_env = _python_resolver_env()
        before = await self._python_distributions(run)
        for argv in steps:
            logs.append(f"Running {' '.join(argv)}")
            result = await run(argv, K8S_INSTALL_TIMEOUT_SECONDS, resolver_env)
            logs.append(f"{argv[0]} output (first 1000 chars): {result.stdout[:1000]}")
            if not result.succeeded:
                errors.append(
                    f"{' '.join(argv)} failed with exit code {result.exit_code}: "
                    f"{(result.stderr or result.stdout)[:500]}"
                )
                return

        package_managers_used.append(manager)
        after = await self._python_distributions(run)
        # A set difference, matching what the Docker branch reports from its
        # `pip freeze` before/after diff: NEWLY installed distributions, not
        # everything present. `before` is empty when no venv existed yet, which
        # is the honest reading — nothing was installed before there was
        # anywhere to install it.
        installed_packages.extend(sorted(after - before))
        logs.append(f"Successfully installed {len(after - before)} new Python distributions")

    async def _python_distributions(self, run: Any) -> set[str]:
        """Distribution names installed in the workspace venv, or an empty set.

        ⚠️ ``--python .venv/bin/python`` is load-bearing, not tidiness. Measured
        in a live ``codegraph-runtime`` pod on 2026-09-13: a bare
        ``uv pip list --format json`` run before the venv exists does NOT fail —
        it silently falls back to the SYSTEM interpreter at ``/usr/local`` and
        lists its baked-in tools (92 distributions at the time: ruff, semgrep, …). The "after"
        listing then reads the freshly-created ``.venv`` with 1 package in it,
        so ``after - before`` came out EMPTY for an install that demonstrably
        worked (``uv sync --frozen`` printed ``+ idna==3.10``). Two different
        environments differenced against each other report zero and look calm.
        Pinned to the venv, the same pair measured ``{} -> {"idna"}``.

        An empty set when the listing fails: with no venv yet, the pinned form
        exits 2 with "No virtual environment ... found for path". That is the
        honest reading of one side of a difference, not an error to report.
        """
        result = await run(
            ["uv", "pip", "list", "--python", ".venv/bin/python", "--format", "json"],
            env=_python_resolver_env(),
        )
        if not result.succeeded:
            return set()
        try:
            return {entry["name"] for entry in json.loads(result.stdout) if entry.get("name")}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.debug("Could not parse `uv pip list` output: %s", e)
            return set()

    async def _install_javascript_via_runtime(
        self,
        run: Any,
        entries: set[str],
        logs: list[str],
        errors: list[str],
        installed_packages: list[str],
        package_managers_used: list[str],
    ) -> None:
        """Install JavaScript dependencies in a pod.

        ``bun install --frozen-lockfile`` when a bun lockfile is committed
        (text ``bun.lock``, or the legacy binary ``bun.lockb``): bun is baked
        into ``codegraph-runtime`` (Dockerfile.runtime, pinned 1.4.4) and on
        the runtime allowlist. Measured 2026-09-17: five org repos ship a text
        ``bun.lock`` and none ships ``bun.lockb``; falling back to npm for
        those resolved a DIFFERENT dependency tree from the one their own CI
        tests against. A bun lockfile wins over ``package-lock.json`` when
        both are present -- a repo that commits ``bun.lock`` chose bun.

        Otherwise ``npm ci`` when ``package-lock.json`` is committed — it
        installs the lock exactly and fails loudly when the lock and
        ``package.json`` disagree, where ``npm install`` would quietly rewrite
        the lock. yarn and pnpm are neither in the image nor on the command
        allowlist, so their lockfiles still fall back to npm.
        """
        if "package.json" not in entries:
            return
        if "bun.lock" in entries or "bun.lockb" in entries:
            argv = ["bun", "install", "--frozen-lockfile"]
            manager = PackageManager.BUN
        else:
            if "yarn.lock" in entries or "pnpm-lock.yaml" in entries:
                logs.append(
                    "yarn/pnpm lockfile present; the codegraph-runtime image "
                    "ships only npm and bun, so npm is used"
                )
            argv = ["npm", "ci"] if "package-lock.json" in entries else ["npm", "install"]
            manager = PackageManager.NPM

        logs.append(f"Running {' '.join(argv)}")
        result = await run(argv, K8S_INSTALL_TIMEOUT_SECONDS)
        logs.append(f"{argv[0]} output (first 1000 chars): {result.stdout[:1000]}")
        if not result.succeeded:
            errors.append(
                f"{' '.join(argv)} failed with exit code {result.exit_code}: "
                f"{(result.stderr or result.stdout)[:500]}"
            )
            return

        package_managers_used.append(manager.value)
        # bun writes an npm-compatible node_modules tree, so the same listing
        # reports what either manager installed.
        listing = await run(["npm", "ls", "--json", "--depth=0"])
        if listing.succeeded:
            try:
                dependencies = json.loads(listing.stdout).get("dependencies") or {}
                installed_packages.extend(dependencies.keys())
            except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as e:
                logger.debug("Could not parse `npm ls` output: %s", e)

    async def install_dependencies_in_container(
        self, container_name: str, workspace_path: str
    ) -> InstallationResult:
        """Main entry point for installing dependencies in a container.

        Args:
            container_name (str): The container_name parameter.
            workspace_path (str): The workspace_path parameter.

        Returns:
            InstallationResult: The installationresult result."""
        logger.info(f"Starting dependency installation in container: {container_name}")

        # Kubernetes workspaces never reach the Docker code below: it calls
        # ``docker.from_env()``, which has no socket in the cluster, so this
        # endpoint has been dead there (see src/runtime/policy.py's note on why
        # `uv` is allowlisted). Route them through the runtime abstraction the
        # rest of the service already uses instead.
        #
        # Resolution is deliberately a NO-OP unless Kubernetes is genuinely
        # reachable — the kind is decided first (cheap, needs no client), and a
        # provisioner that is absent or unavailable falls through to the Docker
        # path exactly as before.
        selected = self._select_kubernetes_runtime(container_name)
        if selected is not None:
            runtime, workspace = selected
            return await self._install_dependencies_via_runtime(
                runtime, workspace, container_name, workspace_path
            )

        start_time = time.time()
        errors = []
        logs = []
        installed_packages = []
        package_managers_used = []

        try:
            client = docker.from_env()
            container = client.containers.get(container_name)

            # Verify workspace exists
            check_cmd = f"test -d {workspace_path} && echo 'EXISTS' || echo 'NOT_FOUND'"
            result = container.exec_run(check_cmd, tty=True)
            if b"NOT_FOUND" in result.output:
                logs.append(f"Workspace path {workspace_path} not found in container")
                return InstallationResult(
                    success=False,
                    installed_packages=[],
                    package_managers_used=[],
                    errors=[f"Workspace path {workspace_path} not found in container"],
                    installation_time=time.time() - start_time,
                    logs=logs,
                )

            # Detect project type and dependencies
            project_type = await self._detect_project_type_in_container(container, workspace_path)
            logs.append(f"Detected project type: {project_type}")

            if project_type == "python":
                result = await self._install_python_dependencies_in_container(
                    container, workspace_path, logs
                )
                installed_packages.extend(result.get("packages", []))
                package_managers_used.extend(result.get("managers", []))
                errors.extend(result.get("errors", []))

            elif project_type == "javascript":
                result = await self._install_javascript_dependencies_in_container(
                    container, workspace_path, logs
                )
                installed_packages.extend(result.get("packages", []))
                package_managers_used.extend(result.get("managers", []))
                errors.extend(result.get("errors", []))

            else:
                logs.append(f"No dependencies detected for project type: {project_type}")

        except (DockerException, OSError, RuntimeError, ValueError, KeyError) as e:
            error_msg = f"Error installing dependencies: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

        total_time = time.time() - start_time
        success = len(errors) == 0

        return InstallationResult(
            success=success,
            installed_packages=installed_packages,
            package_managers_used=list(set(package_managers_used)),
            errors=errors,
            installation_time=total_time,
            logs=logs,
        )

    def _classify_dependency_file(self, file_path: str) -> dict[str, str] | None:
        """``{path, type, manager, file_name}`` for a dependency file, else ``None``."""
        file_name = os.path.basename(file_path)
        if file_name in ANALYZE_PYTHON_NAMES or (
            file_name.startswith("requirements") and file_name.endswith(".txt")
        ):
            kind, manager = "python", self._determine_python_manager(file_name)
        elif file_name in ANALYZE_JS_NAMES:
            kind, manager = "javascript", self._determine_js_manager(file_name)
        else:
            return None
        return {"path": file_path, "type": kind, "manager": manager.value, "file_name": file_name}

    async def _detect_dependency_files_via_runtime(
        self, runtime: Any, workspace: Any, workspace_path: str
    ) -> list[dict[str, str]]:
        """One server-authored ``find`` through the workspace runtime (Kubernetes).

        The Docker branch of :meth:`analyze_project` calls ``docker.from_env()``,
        which has no socket in the cluster: every Kubernetes workspace got an
        empty analysis there. Vendored trees (``.git``, ``node_modules``,
        ``.venv``) are pruned, so a lockfile inside a dependency is not reported
        as the project's own. A failed ``find`` raises: an empty answer would read
        as "no dependencies", which is a different fact.
        """
        root = workspace_path
        if not root or root == DOCKER_ERA_DEFAULT_WORKSPACE:
            root = K8S_WORKSPACE_ROOT
        names: list[str] = []
        for name in (*sorted(ANALYZE_PYTHON_NAMES), "requirements*.txt", *sorted(ANALYZE_JS_NAMES)):
            names += [*(["-o"] if names else []), "-name", name]
        argv = [
            "find", root, "-maxdepth", str(ANALYZE_MAX_DEPTH),
            "(", "-name", ".git", "-o", "-name", "node_modules", "-o", "-name", ".venv", ")",
            "-prune", "-o", "-type", "f", "(", *names, ")", "-print",
        ]  # fmt: skip
        result = await runtime.run_once(
            workspace,
            argv,
            cwd=root,
            timeout=K8S_PROBE_TIMEOUT_SECONDS,
            output_limit=ANALYZE_OUTPUT_LIMIT,
            execution_profile=ExecutionProfile.SERVER_AUTHORED,
        )
        if not result.succeeded:
            raise RuntimeError(
                f"dependency file scan failed in {root} (exit {result.exit_code}): "
                f"{(result.stderr or result.stdout).strip()[:300]}"
            )
        detected: list[dict[str, str]] = []
        for line in sorted(result.stdout.splitlines()):
            entry = self._classify_dependency_file(line.strip()) if line.strip() else None
            if entry is not None:
                detected.append(entry)
            if len(detected) >= ANALYZE_MAX_FILES:
                break
        return detected

    async def analyze_project(self, workspace_path: str) -> dict[str, Any]:
        """Analyze project to detect dependencies and determine installation plan.

        Returns ``detected_files`` as ``{path, type, manager, file_name}`` dicts and
        ``installation_plan`` as a list of strings (the shapes
        ``DependencyStatusResponse`` declares).
        """
        detected_files = []
        project_types = set()

        selected = (
            self._select_kubernetes_runtime(self.target_container_name)
            if self.target_container_name
            else None
        )
        if selected is not None:
            runtime, workspace = selected
            detected_files = await self._detect_dependency_files_via_runtime(
                runtime, workspace, workspace_path
            )
            project_types = {f["type"] for f in detected_files}
            return self._analysis_document(detected_files, project_types)

        try:
            client = docker.from_env()

            # Get the container
            if self.target_container_name:
                container = client.containers.get(self.target_container_name)

                # Check for Python dependency files
                python_patterns = [
                    "requirements*.txt",
                    "pyproject.toml",
                    "setup.py",
                    "Pipfile",
                    "poetry.lock",
                    "pdm.lock",
                ]
                for pattern in python_patterns:
                    find_cmd = (
                        f"find {workspace_path} -name '{pattern}' -type f 2>/dev/null | head -10"
                    )
                    result = container.exec_run(f"bash -c '{find_cmd}'")
                    if result.exit_code == 0 and result.output:
                        files = result.output.decode().strip().split("\n")
                        for file_path in files:
                            if file_path:
                                file_name = os.path.basename(file_path)
                                manager = self._determine_python_manager(file_name)
                                detected_files.append(
                                    {
                                        "path": file_path,
                                        "type": "python",
                                        "manager": manager.value,
                                        "file_name": file_name,
                                    }
                                )
                                project_types.add("python")

                # Check for JavaScript dependency files
                js_patterns = [
                    "package.json",
                    "yarn.lock",
                    "pnpm-lock.yaml",
                    "bun.lock",
                    "bun.lockb",
                ]
                for pattern in js_patterns:
                    find_cmd = (
                        f"find {workspace_path} -name '{pattern}' -type f 2>/dev/null | head -10"
                    )
                    result = container.exec_run(f"bash -c '{find_cmd}'")
                    if result.exit_code == 0 and result.output:
                        files = result.output.decode().strip().split("\n")
                        for file_path in files:
                            if file_path:
                                file_name = os.path.basename(file_path)
                                manager = self._determine_js_manager(file_name)
                                detected_files.append(
                                    {
                                        "path": file_path,
                                        "type": "javascript",
                                        "manager": manager.value,
                                        "file_name": file_name,
                                    }
                                )
                                project_types.add("javascript")
            else:
                # Fallback for when no container is specified
                detected_files = []
                project_types = set()

        except (DockerException, OSError, RuntimeError, ValueError) as e:
            logger.warning(f"Error analyzing project: {e}")
            # Return minimal info on error
            detected_files = []
            project_types = set()

        return self._analysis_document(detected_files, project_types)

    @staticmethod
    def _analysis_document(
        detected_files: list[dict[str, str]], project_types: set[str]
    ) -> dict[str, Any]:
        project_type = (
            "mixed"
            if len(project_types) > 1
            else (next(iter(project_types)) if project_types else "unknown")
        )

        return {
            "project_type": project_type,
            "detected_files": detected_files,
            "installation_plan": [
                f"Install {file['type']} dependencies using {file['manager']} from {file['file_name']}"
                for file in detected_files
            ],
            "total_files": len(detected_files),
        }

    def _determine_python_manager(self, file_name: str) -> PackageManager:
        """Determine the Python package manager based on file name."""
        if "requirements" in file_name:
            return PackageManager.PIP
        elif file_name == "pyproject.toml" or file_name == "poetry.lock":
            return PackageManager.POETRY
        elif file_name == "Pipfile" or file_name == "Pipfile.lock":
            return PackageManager.PIPENV
        elif file_name == "pdm.lock":
            return PackageManager.PDM
        elif file_name.endswith((".yml", ".yaml")) and "environment" in file_name:
            return PackageManager.CONDA
        else:
            return PackageManager.PIP

    def _determine_js_manager(self, file_name: str) -> PackageManager:
        """Determine the JavaScript package manager based on file name."""
        if file_name == "yarn.lock":
            return PackageManager.YARN
        elif file_name == "pnpm-lock.yaml":
            return PackageManager.PNPM
        elif file_name in ("bun.lock", "bun.lockb"):
            return PackageManager.BUN
        else:
            return PackageManager.NPM

    async def _detect_project_type_in_container(
        self, container: Container, workspace_path: str
    ) -> str:
        """Detect project type in container."""
        # Check for Python files
        python_check = container.exec_run(
            f"find {workspace_path} -name '*.py' -o -name 'requirements*.txt' -o -name 'pyproject.toml' -o -name 'setup.py' | head -1"
        )
        if python_check.output.strip():
            return "python"

        # Check for JavaScript files
        js_check = container.exec_run(
            f"find {workspace_path} -name 'package.json' -o -name '*.js' -o -name '*.ts' | head -1"
        )
        if js_check.output.strip():
            return "javascript"

        return "unknown"

    async def _install_python_dependencies_in_container(
        self, container: Container, workspace_path: str, logs: list[str]
    ) -> dict:
        """Install Python dependencies in container."""
        packages = []
        managers = []
        errors = []

        # Check for requirements.txt
        req_check = container.exec_run(
            f"test -f {workspace_path}/requirements.txt && echo 'EXISTS'"
        )
        if b"EXISTS" in req_check.output:
            logs.append("Found requirements.txt, installing with pip...")

            # First, get list of currently installed packages
            before_list = container.exec_run("pip freeze")
            before_packages = set()
            if before_list.exit_code == 0:
                before_packages = set(
                    line.split("==")[0]
                    for line in before_list.output.decode().strip().split("\n")
                    if line
                )

            # Install dependencies
            result = container.exec_run(
                f"cd {workspace_path} && pip install -r requirements.txt --no-cache-dir",
                tty=True,
                stream=True,
            )

            # Collect output
            output_lines = []
            exit_code = 0
            try:
                for chunk in result.output:
                    output_lines.append(chunk.decode("utf-8", errors="ignore"))
                exit_code = result.exit_code
            except (
                DockerException,
                OSError,
                UnicodeDecodeError,
                StopIteration,
                AttributeError,
            ) as e:
                logger.error(f"Error collecting output: {e}")
                # Handle non-streaming output
                if hasattr(result, "output") and result.output:
                    output_lines = [result.output.decode("utf-8", errors="ignore")]
                exit_code = result.exit_code if hasattr(result, "exit_code") else 1

            full_output = "".join(output_lines)
            logs.append(f"Installation output (first 1000 chars): {full_output[:1000]}")

            if exit_code == 0:
                managers.append("pip")

                # Get list of packages after installation
                after_list = container.exec_run("pip freeze")
                if after_list.exit_code == 0:
                    after_packages = set(
                        line.split("==")[0]
                        for line in after_list.output.decode().strip().split("\n")
                        if line
                    )
                    new_packages = after_packages - before_packages
                    packages.extend(list(new_packages))
                    logs.append(f"Successfully installed {len(new_packages)} new packages")

                # Also count packages that were already satisfied
                if "Requirement already satisfied" in full_output:
                    satisfied_count = full_output.count("Requirement already satisfied")
                    logs.append(f"Found {satisfied_count} packages already satisfied")

            else:
                errors.append(f"pip install failed with exit code {exit_code}: {full_output[:500]}")

        # Check for pyproject.toml
        pyproject_check = container.exec_run(
            f"test -f {workspace_path}/pyproject.toml && echo 'EXISTS'"
        )
        if b"EXISTS" in pyproject_check.output:
            # Try poetry first
            poetry_check = container.exec_run("which poetry")
            if poetry_check.exit_code == 0:
                logs.append("Found pyproject.toml, installing with poetry...")
                result = container.exec_run(f"cd {workspace_path} && poetry install", tty=True)
                if result.exit_code == 0:
                    managers.append("poetry")
                else:
                    errors.append(
                        f"poetry install failed: {result.output.decode('utf-8', errors='ignore')}"
                    )

        return {"packages": packages, "managers": managers, "errors": errors}

    async def _install_javascript_dependencies_in_container(
        self, container: Container, workspace_path: str, logs: list[str]
    ) -> dict:
        """Install JavaScript dependencies in container."""
        packages = []
        managers = []
        errors = []

        # Check for package.json
        pkg_check = container.exec_run(f"test -f {workspace_path}/package.json && echo 'EXISTS'")
        if b"EXISTS" in pkg_check.output:
            # Try npm
            npm_check = container.exec_run("which npm")
            if npm_check.exit_code == 0:
                logs.append("Found package.json, installing with npm...")
                result = container.exec_run(f"cd {workspace_path} && npm install", tty=True)
                if result.exit_code == 0:
                    managers.append("npm")
                    # Parse installed packages
                    npm_list = container.exec_run(
                        f"cd {workspace_path} && npm list --json --depth=0"
                    )
                    if npm_list.exit_code == 0:
                        try:
                            installed = json.loads(npm_list.output)
                            if "dependencies" in installed:
                                packages.extend(list(installed["dependencies"].keys()))
                        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                            logger.error(f"Failed to parse npm dependencies: {e}")
                else:
                    errors.append(
                        f"npm install failed: {result.output.decode('utf-8', errors='ignore')}"
                    )

        return {"packages": packages, "managers": managers, "errors": errors}

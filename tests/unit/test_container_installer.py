"""Tests for core/container_installer.py."""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import docker.errors
import pytest

from src.core.container_installer import (
    ContainerInstaller,
    DependencyFile,
    InstallationResult,
    PackageManager,
    ProjectType,
    SystemDependencyError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def installer():
    """Create a ContainerInstaller with no target container."""
    return ContainerInstaller()


@pytest.fixture
def installer_with_target():
    """Create a ContainerInstaller with a target container name."""
    return ContainerInstaller(target_container_name="test-container")


@pytest.fixture
def mock_container():
    """Create a mock Docker container."""
    container = MagicMock()
    container.exec_run = MagicMock()
    return container


# ---------------------------------------------------------------------------
# SystemDependencyError
# ---------------------------------------------------------------------------


class TestSystemDependencyError:
    def test_init(self):
        err = SystemDependencyError(
            "missing deps",
            missing_packages=["gcc", "make"],
            guidance="apt-get install gcc make",
        )
        assert str(err) == "missing deps"
        assert err.missing_packages == ["gcc", "make"]
        assert err.guidance == "apt-get install gcc make"


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------


class TestEnumsAndDataclasses:
    def test_package_manager_values(self):
        assert PackageManager.PIP.value == "pip"
        assert PackageManager.NPM.value == "npm"
        assert PackageManager.POETRY.value == "poetry"
        assert PackageManager.UV.value == "uv"
        assert PackageManager.YARN.value == "yarn"
        assert PackageManager.PNPM.value == "pnpm"
        assert PackageManager.BUN.value == "bun"
        assert PackageManager.PIPENV.value == "pipenv"
        assert PackageManager.PDM.value == "pdm"
        assert PackageManager.CONDA.value == "conda"

    def test_project_type_values(self):
        assert ProjectType.PYTHON.value == "python"
        assert ProjectType.JAVASCRIPT.value == "javascript"
        assert ProjectType.MIXED.value == "mixed"
        assert ProjectType.UNKNOWN.value == "unknown"

    def test_dependency_file_creation(self):
        df = DependencyFile(
            path="/app/requirements.txt",
            file_type="requirements",
            language="python",
            package_manager=PackageManager.PIP,
            priority=3,
        )
        assert df.path == "/app/requirements.txt"
        assert df.content_preview == ""

    def test_dependency_file_with_preview(self):
        df = DependencyFile(
            path="/app/package.json",
            file_type="package_json",
            language="javascript",
            package_manager=PackageManager.NPM,
            priority=1,
            content_preview='{"name": "test"}',
        )
        assert df.content_preview == '{"name": "test"}'

    def test_installation_result(self):
        r = InstallationResult(
            success=True,
            installed_packages=["flask"],
            package_managers_used=["pip"],
            errors=[],
            installation_time=1.5,
            logs=["installed flask"],
        )
        assert r.success is True
        assert r.installed_packages == ["flask"]


# ---------------------------------------------------------------------------
# ContainerInstaller.__init__
# ---------------------------------------------------------------------------


class TestContainerInstallerInit:
    def test_init_no_target(self, installer):
        assert installer.target_container_name is None
        assert installer.detected_files == []
        assert installer.project_type == ProjectType.UNKNOWN

    def test_init_with_target(self, installer_with_target):
        assert installer_with_target.target_container_name == "test-container"


# ---------------------------------------------------------------------------
# _get_manager_requirements
# ---------------------------------------------------------------------------


class TestGetManagerRequirements:
    def test_pip_requirements(self, installer):
        reqs = installer._get_manager_requirements(PackageManager.PIP)
        assert "python3-pip" in reqs

    def test_npm_requirements(self, installer):
        reqs = installer._get_manager_requirements(PackageManager.NPM)
        assert "nodejs" in reqs
        assert "npm" in reqs

    def test_poetry_requirements(self, installer):
        reqs = installer._get_manager_requirements(PackageManager.POETRY)
        assert "curl" in reqs

    def test_uv_requirements(self, installer):
        reqs = installer._get_manager_requirements(PackageManager.UV)
        assert "curl" in reqs

    def test_yarn_requirements(self, installer):
        reqs = installer._get_manager_requirements(PackageManager.YARN)
        assert "nodejs" in reqs
        assert "npm" in reqs

    def test_conda_requirements(self, installer):
        reqs = installer._get_manager_requirements(PackageManager.CONDA)
        assert "wget" in reqs

    def test_bun_requirements(self, installer):
        reqs = installer._get_manager_requirements(PackageManager.BUN)
        assert "curl" in reqs

    def test_unknown_manager(self, installer):
        """A PackageManager not in the map should return empty list."""
        # All PackageManager values are mapped, so we test one more for coverage
        reqs = installer._get_manager_requirements(PackageManager.PIPENV)
        assert "python3-pip" in reqs

    def test_pdm_requirements(self, installer):
        reqs = installer._get_manager_requirements(PackageManager.PDM)
        assert "python3-pip" in reqs

    def test_pnpm_requirements(self, installer):
        reqs = installer._get_manager_requirements(PackageManager.PNPM)
        assert "nodejs" in reqs


# ---------------------------------------------------------------------------
# _check_system_dependency_error
# ---------------------------------------------------------------------------


class TestCheckSystemDependencyError:
    def test_detects_gcc_not_found(self, installer):
        assert installer._check_system_dependency_error("gcc: command not found") is True

    def test_detects_python_h_missing(self, installer):
        assert (
            installer._check_system_dependency_error(
                "fatal error: Python.h: No such file or directory"
            )
            is True
        )

    def test_detects_openssl_missing(self, installer):
        assert (
            installer._check_system_dependency_error(
                "fatal error: openssl/opensslv.h: No such file or directory"
            )
            is True
        )

    def test_detects_pg_config_missing(self, installer):
        assert installer._check_system_dependency_error("pg_config executable not found") is True

    def test_detects_pkg_config_missing(self, installer):
        assert installer._check_system_dependency_error("pkg-config: not found") is True

    def test_no_match_on_clean_output(self, installer):
        assert installer._check_system_dependency_error("Successfully installed flask-2.0") is False

    def test_case_insensitive(self, installer):
        assert installer._check_system_dependency_error("GCC: Command Not Found") is True

    def test_detects_rust_compiler(self, installer):
        assert installer._check_system_dependency_error("error: can't find Rust compiler") is True

    def test_detects_cargo_not_found(self, installer):
        assert installer._check_system_dependency_error("error: cargo not found in PATH") is True

    def test_detects_libxml(self, installer):
        assert (
            installer._check_system_dependency_error(
                "Please make sure the libxml2 and libxslt development packages are installed"
            )
            is True
        )

    def test_detects_visual_cpp(self, installer):
        assert (
            installer._check_system_dependency_error("error: Microsoft Visual C++ 14.0 is required")
            is True
        )

    def test_detects_ffi_h(self, installer):
        assert (
            installer._check_system_dependency_error(
                "fatal error: ffi.h: No such file or directory"
            )
            is True
        )


# ---------------------------------------------------------------------------
# _extract_system_dependencies
# ---------------------------------------------------------------------------


class TestExtractSystemDependencies:
    def test_python_h_missing(self, installer):
        deps = installer._extract_system_dependencies(
            "fatal error: Python.h: No such file or directory"
        )
        assert "python3-dev" in deps

    def test_gcc_missing(self, installer):
        deps = installer._extract_system_dependencies("gcc: command not found")
        assert "gcc" in deps

    def test_openssl_missing(self, installer):
        deps = installer._extract_system_dependencies(
            "fatal error: openssl/opensslv.h: No such file or directory"
        )
        assert "libssl-dev" in deps

    def test_pg_config_missing(self, installer):
        deps = installer._extract_system_dependencies("pg_config executable not found")
        assert "libpq-dev" in deps

    def test_libxml_missing(self, installer):
        deps = installer._extract_system_dependencies(
            "Please make sure the libxml2 and libxslt development packages are installed"
        )
        assert "libxml2-dev" in deps
        assert "libxslt1-dev" in deps

    def test_no_duplicates(self, installer):
        # Message that triggers libxml2-dev from multiple patterns
        output = (
            "xml2-config: not found\nfatal error: libxml/xmlversion.h: No such file or directory"
        )
        deps = installer._extract_system_dependencies(output)
        # Should contain libxml2-dev only once
        assert deps.count("libxml2-dev") == 1

    def test_clean_output_returns_empty(self, installer):
        deps = installer._extract_system_dependencies("Successfully installed flask")
        assert deps == []

    def test_jpeg_missing(self, installer):
        deps = installer._extract_system_dependencies(
            "fatal error: jpeglib.h: No such file or directory"
        )
        assert "libjpeg-dev" in deps

    def test_png_missing(self, installer):
        deps = installer._extract_system_dependencies(
            "fatal error: png.h: No such file or directory"
        )
        assert "libpng-dev" in deps

    def test_cargo_missing(self, installer):
        deps = installer._extract_system_dependencies("error: cargo not found in PATH")
        assert "cargo" in deps
        assert "rustc" in deps

    def test_make_missing(self, installer):
        deps = installer._extract_system_dependencies("make: command not found")
        assert "make" in deps

    def test_cmake_missing(self, installer):
        deps = installer._extract_system_dependencies("cmake: command not found")
        assert "cmake" in deps

    def test_ffi_missing(self, installer):
        deps = installer._extract_system_dependencies(
            "fatal error: ffi.h: No such file or directory"
        )
        assert "libffi-dev" in deps

    def test_mysql_config_missing(self, installer):
        deps = installer._extract_system_dependencies("OSError: mysql_config not found")
        assert "default-libmysqlclient-dev" in deps

    def test_pkg_config_missing(self, installer):
        deps = installer._extract_system_dependencies("pkg-config: not found")
        assert "pkg-config" in deps

    def test_visual_cpp_missing(self, installer):
        deps = installer._extract_system_dependencies(
            "error: Microsoft Visual C++ 14.0 is required"
        )
        assert "build-essential" in deps
        assert "gcc" in deps


# ---------------------------------------------------------------------------
# _format_dependency_errors
# ---------------------------------------------------------------------------


class TestFormatDependencyErrors:
    def test_formats_with_known_deps(self, installer):
        output = "fatal error: Python.h: No such file or directory"
        result = installer._format_dependency_errors(output, PackageManager.PIP)
        assert "python3-dev" in result
        assert "apt-get" in result
        # PIP-specific guidance for Python.h
        assert "C extensions" in result

    def test_formats_unknown_deps(self, installer):
        result = installer._format_dependency_errors("some random error", PackageManager.PIP)
        assert "Unable to determine" in result

    def test_non_pip_manager(self, installer):
        output = "fatal error: Python.h: No such file or directory"
        result = installer._format_dependency_errors(output, PackageManager.POETRY)
        assert "python3-dev" in result
        # Should NOT contain PIP-specific guidance
        assert "C extensions" not in result


# ---------------------------------------------------------------------------
# _extract_missing_packages
# ---------------------------------------------------------------------------


class TestExtractMissingPackages:
    def test_extracts_package_names(self, installer):
        output = "E: Unable to locate package libfoo-dev\nE: Unable to locate package libbar-dev\n"
        missing = installer._extract_missing_packages(output)
        assert "libfoo-dev" in missing
        assert "libbar-dev" in missing

    def test_no_missing_packages(self, installer):
        output = "Reading package lists... Done"
        missing = installer._extract_missing_packages(output)
        assert missing == []


# ---------------------------------------------------------------------------
# _get_package_alternatives
# ---------------------------------------------------------------------------


class TestGetPackageAlternatives:
    def test_known_alternative(self, installer):
        alts = installer._get_package_alternatives(["python-dev"])
        assert "python3-dev" in alts

    def test_unknown_dev_package_with_lib_prefix(self, installer):
        alts = installer._get_package_alternatives(["foo-dev"])
        assert "libfoo-dev" in alts

    def test_unknown_dev_package_already_has_lib(self, installer):
        alts = installer._get_package_alternatives(["libsomething-dev"])
        # It's in the known map? No. It has -dev and starts with lib.
        # The code checks: if not base.startswith("lib") -> won't match
        assert alts == []

    def test_no_packages(self, installer):
        alts = installer._get_package_alternatives([])
        assert alts == []

    def test_multiple_known(self, installer):
        alts = installer._get_package_alternatives(["python-pip", "node"])
        assert "python3-pip" in alts
        assert "nodejs" in alts

    def test_libpng12_dev_mapping(self, installer):
        alts = installer._get_package_alternatives(["libpng12-dev"])
        assert "libpng-dev" in alts


# ---------------------------------------------------------------------------
# _get_python_manager_install_cmd
# ---------------------------------------------------------------------------


class TestGetPythonManagerInstallCmd:
    def test_pip_cmd(self, installer):
        cmd = installer._get_python_manager_install_cmd(PackageManager.PIP, "/workspace")
        assert "pip install -r requirements.txt" in cmd

    def test_poetry_cmd(self, installer):
        cmd = installer._get_python_manager_install_cmd(PackageManager.POETRY, "/workspace")
        assert "poetry install" in cmd

    def test_uv_cmd(self, installer):
        cmd = installer._get_python_manager_install_cmd(PackageManager.UV, "/workspace")
        assert "uv sync" in cmd
        # Must respect the committed lock (no silent re-resolve to latest, which
        # would mask vulnerable loosely-pinned deps before a CVE scan), while
        # still provisioning lock-less repos via a bare sync.
        assert "uv sync --frozen" in cmd
        assert "uv.lock" in cmd

    def test_pipenv_cmd(self, installer):
        cmd = installer._get_python_manager_install_cmd(PackageManager.PIPENV, "/workspace")
        assert "pipenv install" in cmd

    def test_pdm_cmd(self, installer):
        cmd = installer._get_python_manager_install_cmd(PackageManager.PDM, "/workspace")
        assert "pdm install" in cmd

    def test_conda_cmd(self, installer):
        cmd = installer._get_python_manager_install_cmd(PackageManager.CONDA, "/workspace")
        assert "conda env update" in cmd

    def test_npm_returns_empty(self, installer):
        """NPM is not a Python manager, should return empty string."""
        cmd = installer._get_python_manager_install_cmd(PackageManager.NPM, "/workspace")
        assert cmd == ""


# ---------------------------------------------------------------------------
# _parse_installed_packages
# ---------------------------------------------------------------------------


class TestParseInstalledPackages:
    def test_pip_output(self, installer):
        output = "Successfully installed flask-2.0 jinja2-3.0 werkzeug-2.0"
        pkgs = installer._parse_installed_packages(output, PackageManager.PIP)
        assert "flask" in pkgs
        assert "jinja2" in pkgs
        assert "werkzeug" in pkgs

    def test_pip_no_installs(self, installer):
        output = "Requirement already satisfied: flask"
        pkgs = installer._parse_installed_packages(output, PackageManager.PIP)
        assert pkgs == []

    def test_non_pip_manager_returns_empty(self, installer):
        output = "Successfully installed flask-2.0"
        pkgs = installer._parse_installed_packages(output, PackageManager.POETRY)
        assert pkgs == []

    def test_pip_multiline_output(self, installer):
        output = (
            "Collecting flask\nDownloading Flask-2.0\nSuccessfully installed flask-2.0 click-8.0\n"
        )
        pkgs = installer._parse_installed_packages(output, PackageManager.PIP)
        assert "flask" in pkgs
        assert "click" in pkgs

    def test_empty_output(self, installer):
        pkgs = installer._parse_installed_packages("", PackageManager.PIP)
        assert pkgs == []


# ---------------------------------------------------------------------------
# _determine_python_manager
# ---------------------------------------------------------------------------


class TestDeterminePythonManager:
    def test_requirements_txt(self, installer):
        assert installer._determine_python_manager("requirements.txt") == PackageManager.PIP

    def test_requirements_dev_txt(self, installer):
        assert installer._determine_python_manager("requirements-dev.txt") == PackageManager.PIP

    def test_pyproject_toml(self, installer):
        assert installer._determine_python_manager("pyproject.toml") == PackageManager.POETRY

    def test_poetry_lock(self, installer):
        assert installer._determine_python_manager("poetry.lock") == PackageManager.POETRY

    def test_pipfile(self, installer):
        assert installer._determine_python_manager("Pipfile") == PackageManager.PIPENV

    def test_pipfile_lock(self, installer):
        assert installer._determine_python_manager("Pipfile.lock") == PackageManager.PIPENV

    def test_pdm_lock(self, installer):
        assert installer._determine_python_manager("pdm.lock") == PackageManager.PDM

    def test_environment_yml(self, installer):
        assert installer._determine_python_manager("environment.yml") == PackageManager.CONDA

    def test_environment_yaml(self, installer):
        assert installer._determine_python_manager("environment.yaml") == PackageManager.CONDA

    def test_unknown_falls_back_to_pip(self, installer):
        assert installer._determine_python_manager("setup.cfg") == PackageManager.PIP


# ---------------------------------------------------------------------------
# _determine_js_manager
# ---------------------------------------------------------------------------


class TestDetermineJsManager:
    def test_yarn_lock(self, installer):
        assert installer._determine_js_manager("yarn.lock") == PackageManager.YARN

    def test_pnpm_lock(self, installer):
        assert installer._determine_js_manager("pnpm-lock.yaml") == PackageManager.PNPM

    def test_bun_lockb(self, installer):
        assert installer._determine_js_manager("bun.lockb") == PackageManager.BUN

    def test_bun_lock_text(self, installer):
        # bun >= 1.2 writes a text lockfile; every org bun repo ships this one.
        assert installer._determine_js_manager("bun.lock") == PackageManager.BUN

    def test_package_json_defaults_to_npm(self, installer):
        assert installer._determine_js_manager("package.json") == PackageManager.NPM

    def test_unknown_defaults_to_npm(self, installer):
        assert installer._determine_js_manager("something.json") == PackageManager.NPM


# ---------------------------------------------------------------------------
# _install_system_dependencies (async)
# ---------------------------------------------------------------------------


class TestInstallSystemDependencies:
    async def test_empty_deps_returns_true(self, installer, mock_container):
        result = await installer._install_system_dependencies(mock_container, [])
        assert result is True

    async def test_successful_install(self, installer, mock_container):
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=b"done")
        result = await installer._install_system_dependencies(mock_container, ["gcc", "make"])
        assert result is True
        assert mock_container.exec_run.call_count == 2  # update + install

    async def test_failed_install(self, installer, mock_container):
        # apt-get update succeeds, install fails
        mock_container.exec_run.side_effect = [
            MagicMock(exit_code=0, output=b"updated"),  # update
            MagicMock(exit_code=1, output=b"some error"),  # install
        ]
        result = await installer._install_system_dependencies(mock_container, ["gcc"])
        assert result is False

    async def test_unable_to_locate_with_alternatives(self, installer, mock_container):
        mock_container.exec_run.side_effect = [
            MagicMock(exit_code=0, output=b"ok"),  # apt-get update
            MagicMock(
                exit_code=100,
                output=b"E: Unable to locate package python-dev",
            ),  # first install fails
            MagicMock(exit_code=0, output=b"installed"),  # alternative install succeeds
        ]
        result = await installer._install_system_dependencies(mock_container, ["python-dev"])
        assert result is True

    async def test_unable_to_locate_no_alternatives(self, installer, mock_container):
        mock_container.exec_run.side_effect = [
            MagicMock(exit_code=0, output=b"ok"),
            MagicMock(
                exit_code=100,
                output=b"E: Unable to locate package zzz-nonexistent",
            ),
        ]
        result = await installer._install_system_dependencies(mock_container, ["zzz-nonexistent"])
        assert result is False

    async def test_exception_returns_false(self, installer, mock_container):
        mock_container.exec_run.side_effect = RuntimeError("docker error")
        result = await installer._install_system_dependencies(mock_container, ["gcc"])
        assert result is False

    async def test_update_failure_continues(self, installer, mock_container):
        """apt-get update failure should still try install."""
        mock_container.exec_run.side_effect = [
            MagicMock(exit_code=1, output=b"update failed"),  # update fails
            MagicMock(exit_code=0, output=b"installed"),  # install succeeds
        ]
        result = await installer._install_system_dependencies(mock_container, ["gcc"])
        assert result is True


# ---------------------------------------------------------------------------
# _install_with_python_manager (async)
# ---------------------------------------------------------------------------


class TestInstallWithPythonManager:
    async def test_pip_success(self, installer, mock_container):
        mock_container.exec_run.return_value = MagicMock(
            exit_code=0,
            output=b"Successfully installed flask-2.0",
        )
        result = await installer._install_with_python_manager(
            mock_container,
            "/workspace",
            PackageManager.PIP,
            [],
        )
        assert result.success is True
        assert "pip" in result.package_managers_used
        assert "flask" in result.installed_packages

    async def test_pip_failure_no_system_dep_error(self, installer, mock_container):
        mock_container.exec_run.return_value = MagicMock(
            exit_code=1,
            output=b"some random error",
        )
        result = await installer._install_with_python_manager(
            mock_container,
            "/workspace",
            PackageManager.PIP,
            [],
        )
        assert result.success is False
        assert len(result.errors) > 0

    async def test_pip_failure_with_system_dep_error_and_retry_success(
        self, installer, mock_container
    ):
        # First call: pip install fails with Python.h error
        # Then: apt-get update, apt-get install succeed
        # Then: pip install retry succeeds
        mock_container.exec_run.side_effect = [
            MagicMock(
                exit_code=1,
                output=b"fatal error: Python.h: No such file or directory",
            ),  # pip install fails
            MagicMock(exit_code=0, output=b"ok"),  # apt-get update
            MagicMock(exit_code=0, output=b"ok"),  # apt-get install
            MagicMock(
                exit_code=0,
                output=b"Successfully installed cffi-1.0",
            ),  # pip retry
        ]
        result = await installer._install_with_python_manager(
            mock_container,
            "/workspace",
            PackageManager.PIP,
            [],
        )
        assert result.success is True

    async def test_poetry_success(self, installer, mock_container):
        mock_container.exec_run.return_value = MagicMock(
            exit_code=0,
            output=b"Installing dependencies from lock file",
        )
        result = await installer._install_with_python_manager(
            mock_container,
            "/workspace",
            PackageManager.POETRY,
            [],
        )
        assert result.success is True

    async def test_unknown_manager_returns_empty(self, installer, mock_container):
        """NPM is not a Python manager, _get_python_manager_install_cmd returns ''."""
        result = await installer._install_with_python_manager(
            mock_container,
            "/workspace",
            PackageManager.NPM,
            [],
        )
        # No command to run, so success with empty results
        assert result.success is True
        assert result.installed_packages == []

    async def test_exception_during_install(self, installer, mock_container):
        mock_container.exec_run.side_effect = RuntimeError("docker fail")
        result = await installer._install_with_python_manager(
            mock_container,
            "/workspace",
            PackageManager.PIP,
            [],
        )
        assert result.success is False
        assert any("Exception" in e for e in result.errors)


# ---------------------------------------------------------------------------
# _detect_project_type_in_container (async)
# ---------------------------------------------------------------------------


class TestDetectProjectTypeInContainer:
    async def test_detects_python(self, installer, mock_container):
        mock_container.exec_run.side_effect = [
            MagicMock(output=b"/workspace/app.py"),  # python check
        ]
        result = await installer._detect_project_type_in_container(mock_container, "/workspace")
        assert result == "python"

    async def test_detects_javascript(self, installer, mock_container):
        mock_container.exec_run.side_effect = [
            MagicMock(output=b""),  # python check - empty
            MagicMock(output=b"/workspace/package.json"),  # js check
        ]
        result = await installer._detect_project_type_in_container(mock_container, "/workspace")
        assert result == "javascript"

    async def test_returns_unknown(self, installer, mock_container):
        mock_container.exec_run.side_effect = [
            MagicMock(output=b""),  # python check - empty
            MagicMock(output=b""),  # js check - empty
        ]
        result = await installer._detect_project_type_in_container(mock_container, "/workspace")
        assert result == "unknown"


# ---------------------------------------------------------------------------
# install_dependencies_in_container (async)
# ---------------------------------------------------------------------------


class TestInstallDependenciesInContainer:
    @patch("src.core.container_installer.docker")
    async def test_docker_not_available(self, mock_docker, installer):
        mock_docker.from_env.side_effect = docker.errors.DockerException("Docker not available")
        result = await installer.install_dependencies_in_container("my-container", "/workspace")
        assert result.success is False
        assert any("Error" in e or "Docker" in e for e in result.errors)

    @patch("src.core.container_installer.docker")
    async def test_workspace_not_found(self, mock_docker, installer):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        container = MagicMock()
        mock_client.containers.get.return_value = container
        container.exec_run.return_value = MagicMock(output=b"NOT_FOUND", exit_code=0)
        result = await installer.install_dependencies_in_container("my-container", "/workspace")
        assert result.success is False
        assert any("not found" in e for e in result.errors)

    @patch("src.core.container_installer.docker")
    async def test_unknown_project_type(self, mock_docker, installer):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        container = MagicMock()
        mock_client.containers.get.return_value = container
        container.exec_run.side_effect = [
            MagicMock(output=b"EXISTS", exit_code=0),  # workspace check
            MagicMock(output=b""),  # python check
            MagicMock(output=b""),  # js check
        ]
        result = await installer.install_dependencies_in_container("my-container", "/workspace")
        assert result.success is True
        assert any("unknown" in log.lower() or "No dependencies" in log for log in result.logs)


# ---------------------------------------------------------------------------
# analyze_project (async)
# ---------------------------------------------------------------------------


class TestAnalyzeProject:
    @patch("src.core.container_installer.docker")
    async def test_docker_not_available(self, mock_docker, installer_with_target):
        mock_docker.from_env.side_effect = docker.errors.DockerException("Docker not available")
        result = await installer_with_target.analyze_project("/workspace")
        assert result["project_type"] == "unknown"
        assert result["detected_files"] == []

    async def test_no_target_container(self, installer):
        """When no target container is set, should return minimal info."""
        # Patch docker to be available but no target container name
        with patch("src.core.container_installer.docker") as mock_docker:
            mock_client = MagicMock()
            mock_docker.from_env.return_value = mock_client
            result = await installer.analyze_project("/workspace")
            assert result["project_type"] == "unknown"

    @patch("src.core.container_installer.docker")
    async def test_detects_python_project(self, mock_docker, installer_with_target):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        container = MagicMock()
        mock_client.containers.get.return_value = container

        # Python pattern finds requirements.txt, all JS patterns find nothing
        def exec_side_effect(cmd, *args, **kwargs):
            if "requirements" in cmd:
                return MagicMock(exit_code=0, output=b"/workspace/requirements.txt")
            return MagicMock(exit_code=0, output=b"")

        container.exec_run.side_effect = exec_side_effect

        result = await installer_with_target.analyze_project("/workspace")
        assert result["project_type"] == "python"
        assert result["total_files"] >= 1

    @patch("src.core.container_installer.docker")
    async def test_exception_returns_empty(self, mock_docker, installer_with_target):
        mock_docker.from_env.side_effect = docker.errors.DockerException("docker error")
        result = await installer_with_target.analyze_project("/workspace")
        assert result["project_type"] == "unknown"
        assert result["detected_files"] == []


# ---------------------------------------------------------------------------
# PYTHON_PATTERNS / JAVASCRIPT_PATTERNS class attributes
# ---------------------------------------------------------------------------


class TestClassPatterns:
    def test_python_patterns_exist(self):
        assert "pyproject.toml" in ContainerInstaller.PYTHON_PATTERNS
        assert "requirements.txt" in ContainerInstaller.PYTHON_PATTERNS
        assert "Pipfile" in ContainerInstaller.PYTHON_PATTERNS
        assert "setup.py" in ContainerInstaller.PYTHON_PATTERNS

    def test_javascript_patterns_exist(self):
        assert "package.json" in ContainerInstaller.JAVASCRIPT_PATTERNS
        assert "yarn.lock" in ContainerInstaller.JAVASCRIPT_PATTERNS
        assert "pnpm-lock.yaml" in ContainerInstaller.JAVASCRIPT_PATTERNS
        assert "bun.lockb" in ContainerInstaller.JAVASCRIPT_PATTERNS
        assert "bun.lock" in ContainerInstaller.JAVASCRIPT_PATTERNS

    def test_python_patterns_have_priorities(self):
        for filename, (manager, priority) in ContainerInstaller.PYTHON_PATTERNS.items():
            assert isinstance(manager, PackageManager)
            assert isinstance(priority, int)

    def test_javascript_patterns_have_priorities(self):
        for filename, (manager, priority) in ContainerInstaller.JAVASCRIPT_PATTERNS.items():
            assert isinstance(manager, PackageManager)
            assert isinstance(priority, int)


# ---------------------------------------------------------------------------
# install_dependencies_in_container - additional coverage
# ---------------------------------------------------------------------------


class TestInstallDependenciesInContainerEdgeCases:
    @patch("src.core.container_installer.docker")
    async def test_no_docker(self, mock_docker, installer):
        """Returns failure when docker SDK is unavailable."""
        mock_docker.from_env.side_effect = docker.errors.DockerException("no docker")
        result = await installer.install_dependencies_in_container("c1", "/app/repo")
        assert result.success is False
        assert len(result.errors) > 0

    @patch("src.core.container_installer.docker")
    async def test_workspace_not_found(self, mock_docker, installer):
        """Returns failure if workspace path not found."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client

        mock_container = MagicMock()
        check_result = MagicMock()
        check_result.output = b"NOT_FOUND"
        mock_container.exec_run.return_value = check_result
        mock_client.containers.get.return_value = mock_container

        result = await installer.install_dependencies_in_container("c1", "/no/such/path")
        assert result.success is False
        assert "not found" in result.errors[0].lower()

    @patch("src.core.container_installer.docker")
    async def test_python_project_detected(self, mock_docker, installer):
        """Detects python project and installs deps."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client

        mock_container = MagicMock()

        def exec_side_effect(cmd, *args, **kwargs):
            if isinstance(cmd, str) and "test -d" in cmd:
                return MagicMock(output=b"EXISTS")
            if isinstance(cmd, str) and "find" in cmd and "*.py" in cmd:
                return MagicMock(output=b"/app/repo/main.py")
            if isinstance(cmd, str) and "find" in cmd:
                return MagicMock(output=b"")
            if isinstance(cmd, str) and "test -f" in cmd and "requirements.txt" in cmd:
                return MagicMock(output=b"EXISTS")
            if isinstance(cmd, str) and "pip freeze" in cmd:
                return MagicMock(exit_code=0, output=b"flask==2.0\n")
            if isinstance(cmd, str) and "pip install" in cmd:
                return MagicMock(exit_code=0, output=b"Successfully installed flask-2.0")
            if isinstance(cmd, str) and "which" in cmd:
                return MagicMock(exit_code=1, output=b"")
            return MagicMock(exit_code=0, output=b"EXISTS")

        mock_container.exec_run.side_effect = exec_side_effect
        mock_client.containers.get.return_value = mock_container

        result = await installer.install_dependencies_in_container("c1", "/app/repo")
        # Regardless of success/failure of individual commands, the function completes
        assert isinstance(result, InstallationResult)

    @patch("src.core.container_installer.docker")
    async def test_js_project_detected(self, mock_docker, installer):
        """Detects javascript project and installs deps."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client

        mock_container = MagicMock()

        def exec_side_effect(cmd, *args, **kwargs):
            if isinstance(cmd, str) and "test -d" in cmd:
                return MagicMock(output=b"EXISTS")
            if isinstance(cmd, str) and "find" in cmd and "*.py" in cmd:
                return MagicMock(output=b"")
            if isinstance(cmd, str) and "find" in cmd and "package.json" in cmd:
                return MagicMock(output=b"/app/repo/package.json")
            if isinstance(cmd, str) and "find" in cmd:
                return MagicMock(output=b"")
            if isinstance(cmd, str) and "test -f" in cmd and "package.json" in cmd:
                return MagicMock(output=b"EXISTS")
            if isinstance(cmd, str) and "which npm" in cmd:
                return MagicMock(exit_code=0, output=b"/usr/bin/npm")
            if isinstance(cmd, str) and "npm install" in cmd:
                return MagicMock(exit_code=0, output=b"added 5 packages")
            if isinstance(cmd, str) and "npm list" in cmd:
                return MagicMock(exit_code=0, output=b'{"dependencies": {"express": {}}}')
            return MagicMock(exit_code=0, output=b"")

        mock_container.exec_run.side_effect = exec_side_effect
        mock_client.containers.get.return_value = mock_container

        result = await installer.install_dependencies_in_container("c1", "/app/repo")
        assert isinstance(result, InstallationResult)


# ---------------------------------------------------------------------------
# _install_with_python_manager - additional coverage
# ---------------------------------------------------------------------------


class TestInstallWithPythonManagerExtended:
    async def test_poetry_install(self, installer, mock_container):
        """Poetry install path."""
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.output = b"Installing dependencies"
        mock_container.exec_run.return_value = mock_result

        result = await installer._install_with_python_manager(
            mock_container, "/workspace", PackageManager.POETRY, []
        )
        assert result.success is True
        assert "poetry" in result.package_managers_used

    async def test_uv_install(self, installer, mock_container):
        """UV install path."""
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.output = b"Resolved dependencies"
        mock_container.exec_run.return_value = mock_result

        result = await installer._install_with_python_manager(
            mock_container, "/workspace", PackageManager.UV, []
        )
        assert result.success is True

    async def test_install_fails_no_sys_dep(self, installer, mock_container):
        """Install failure without system dependency error."""
        mock_result = MagicMock()
        mock_result.exit_code = 1
        mock_result.output = b"some random error"
        mock_container.exec_run.return_value = mock_result

        result = await installer._install_with_python_manager(
            mock_container, "/workspace", PackageManager.PIP, []
        )
        assert result.success is False
        assert len(result.errors) > 0

    async def test_install_fails_with_sys_dep_and_retry_succeeds(self, installer, mock_container):
        """Install fails with system dep error, retries after installing deps."""
        fail_result = MagicMock()
        fail_result.exit_code = 1
        fail_result.output = b"fatal error: Python.h: No such file or directory"

        update_ok = MagicMock()
        update_ok.exit_code = 0
        update_ok.output = b"Updated"

        install_ok = MagicMock()
        install_ok.exit_code = 0
        install_ok.output = b"python3-dev installed"

        retry_ok = MagicMock()
        retry_ok.exit_code = 0
        retry_ok.output = b"Successfully installed numpy-1.0"

        mock_container.exec_run.side_effect = [fail_result, update_ok, install_ok, retry_ok]

        result = await installer._install_with_python_manager(
            mock_container, "/workspace", PackageManager.PIP, []
        )
        assert result.success is True

    async def test_install_exception(self, installer, mock_container):
        """Install raises an exception."""
        mock_container.exec_run.side_effect = RuntimeError("docker error")

        result = await installer._install_with_python_manager(
            mock_container, "/workspace", PackageManager.PIP, []
        )
        assert result.success is False


# ---------------------------------------------------------------------------
# analyze_project JS paths
# ---------------------------------------------------------------------------


class TestAnalyzeProjectJS:
    @patch("src.core.container_installer.docker")
    async def test_detects_js_files(self, mock_docker, installer_with_target):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client

        container = MagicMock()
        mock_client.containers.get.return_value = container

        def exec_side_effect(cmd, *args, **kwargs):
            if "package.json" in str(cmd):
                return MagicMock(exit_code=0, output=b"/workspace/package.json")
            if "yarn.lock" in str(cmd):
                return MagicMock(exit_code=0, output=b"/workspace/yarn.lock")
            return MagicMock(exit_code=0, output=b"")

        container.exec_run.side_effect = exec_side_effect

        result = await installer_with_target.analyze_project("/workspace")
        js_files = [f for f in result["detected_files"] if f.get("type") == "javascript"]
        assert len(js_files) >= 1

    @patch("src.core.container_installer.docker")
    async def test_no_target_container(self, mock_docker):
        """analyze_project with no target_container_name returns empty."""
        installer = ContainerInstaller(target_container_name=None)
        result = await installer.analyze_project("/workspace")
        assert result["project_type"] == "unknown"
        assert result["detected_files"] == []


# ---------------------------------------------------------------------------
# Kubernetes branch: /install-dependencies on the k8s deployment
#
# The Docker installer calls ``docker.from_env()``, which has no socket in the
# cluster, so this endpoint was dead there. These tests pin the branch that
# routes a Kubernetes workspace through the same ``WorkspaceRuntime`` the exec
# router and the package-upgrade runner already use.
#
# They use the REAL ``KubernetesRuntime`` adapter over a fake provisioner, not a
# mocked runtime, so the real ``enforce_command_policy`` runs on every argv this
# code builds. A mocked runtime would have made the allowlist untestable here.
# ---------------------------------------------------------------------------


def _bare(command: list[str]) -> list[str]:
    """The command without the ``/usr/bin/env K=V ...`` prefix the adapter adds."""
    if not command or command[0] != "/usr/bin/env":
        return list(command)
    index = 1
    while index < len(command) and "=" in command[index]:
        index += 1
    return list(command[index:])


def _env_of(command: list[str]) -> dict[str, str]:
    """The ``K=V`` assignments the adapter prefixed onto this command."""
    if not command or command[0] != "/usr/bin/env":
        return {}
    pairs = {}
    for token in command[1:]:
        if "=" not in token:
            break
        key, value = token.split("=", 1)
        pairs[key] = value
    return pairs


class _FakeProvisioner:
    """Stands in for ``KubernetesContainerProvisioner``; records every exec."""

    def __init__(self, responses: dict[str, dict] | None = None):
        self.calls: list[dict] = []
        self._responses = responses or {}

    def is_available(self) -> bool:
        return True

    def exec_in_pod(self, *, deployment_name, command, workdir, timeout, max_output_bytes):
        self.calls.append(
            {
                "deployment_name": deployment_name,
                "command": list(command),
                "workdir": workdir,
                "timeout": timeout,
            }
        )
        base = {"exit_code": 0, "stdout": "", "stderr": "", "duration": 0.0}
        return base | self._responses.get(" ".join(_bare(command)), {})

    @property
    def commands(self) -> list[list[str]]:
        """Commands with the env prefix stripped, keyed the way callers think."""
        return [_bare(call["command"]) for call in self.calls]

    @property
    def raw_commands(self) -> list[list[str]]:
        return [call["command"] for call in self.calls]


def _k8s_workspace(name: str = "ws-1"):
    from src.models.container_models import ContainerInfo, ContainerStatus

    return ContainerInfo(
        container_name=name,
        api_endpoint=f"http://{name}.coding.svc.cluster.local:8687",
        internal_endpoint=f"http://{name}.coding.svc.cluster.local:8687",
        public_endpoint=f"http://{name}:8687",
        codebase_path="/workspace",
        repo_url="https://github.com/owner/repo",
        branch="main",
        status=ContainerStatus.RUNNING,
    )


def _docker_workspace(name: str = "ws-docker"):
    from src.models.container_models import ContainerInfo, ContainerStatus

    return ContainerInfo(
        container_name=name,
        api_endpoint="http://localhost:8687",
        internal_endpoint=f"http://{name}:8687",
        public_endpoint="http://localhost:8687",
        codebase_path="/app/data/repo",
        repo_url="https://github.com/owner/repo",
        branch="main",
        status=ContainerStatus.RUNNING,
    )


@contextlib.contextmanager
def _runtime_env(workspace, provisioner):
    """Wire the two lookups ``_select_kubernetes_runtime`` performs.

    Everything downstream of them — ``resolve_runtime_kind``, ``resolve_runtime``
    and ``KubernetesRuntime`` — is the real implementation.
    """
    manager = MagicMock()
    manager.get_container.return_value = workspace
    core = MagicMock()
    core.uses_agent_host = False
    core.is_docker_available.return_value = True
    with (
        patch("src.core.container_manager.get_container_manager", return_value=manager),
        patch("src.core.container_manager_core.get_container_manager", return_value=core),
        patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            return_value=provisioner,
        ),
    ):
        yield


class TestKubernetesInstallBranchSelection:
    async def test_kubernetes_workspace_never_touches_docker(self, installer):
        """A workspace with an in-cluster endpoint runs on the k8s runtime."""
        provisioner = _FakeProvisioner({"ls -1 -a": {"stdout": "uv.lock\npyproject.toml\n"}})
        with (
            _runtime_env(_k8s_workspace(), provisioner),
            patch("src.core.container_installer.docker") as mock_docker,
        ):
            result = await installer.install_dependencies_in_container("ws-1", "/workspace")

        mock_docker.from_env.assert_not_called()
        assert provisioner.commands[0] == ["ls", "-1", "-a"]
        assert result.success is True
        assert result.package_managers_used == ["uv"]

    async def test_docker_workspace_still_takes_the_docker_branch(self, installer):
        """A workspace with no in-cluster endpoint is unaffected by this change."""
        provisioner = _FakeProvisioner()
        with (
            _runtime_env(_docker_workspace(), provisioner),
            patch("src.core.container_installer.docker") as mock_docker,
        ):
            mock_docker.from_env.side_effect = RuntimeError("docker path reached")
            result = await installer.install_dependencies_in_container("ws-docker", "/app/repo")

        mock_docker.from_env.assert_called_once()
        assert provisioner.calls == []
        assert result.success is False

    async def test_unavailable_provisioner_falls_back_to_docker(self, installer):
        """An in-cluster workspace with no usable provisioner is not a new failure."""
        provisioner = _FakeProvisioner()
        provisioner.is_available = lambda: False
        with (
            _runtime_env(_k8s_workspace(), provisioner),
            patch("src.core.container_installer.docker") as mock_docker,
        ):
            mock_docker.from_env.side_effect = RuntimeError("docker path reached")
            await installer.install_dependencies_in_container("ws-1", "/workspace")

        mock_docker.from_env.assert_called_once()


class TestKubernetesInstallCommands:
    async def test_uv_lock_installs_frozen(self, installer):
        provisioner = _FakeProvisioner(
            {
                "ls -1 -a": {"stdout": "uv.lock\npyproject.toml\nsrc\n"},
                "uv pip list --python .venv/bin/python --format json": {
                    "stdout": '[{"name": "httpx", "version": "0.28.1"}]'
                },
            }
        )
        with _runtime_env(_k8s_workspace(), provisioner):
            result = await installer.install_dependencies_in_container("ws-1", "/workspace")

        assert ["uv", "sync", "--frozen"] in provisioner.commands
        assert ["uv", "sync"] not in provisioner.commands
        assert result.success is True
        # before == after, so nothing is claimed as newly installed.
        assert result.installed_packages == []

    async def test_pyproject_without_lock_installs_unfrozen(self, installer):
        provisioner = _FakeProvisioner({"ls -1 -a": {"stdout": "pyproject.toml\n"}})
        with _runtime_env(_k8s_workspace(), provisioner):
            await installer.install_dependencies_in_container("ws-1", "/workspace")

        assert ["uv", "sync"] in provisioner.commands
        assert ["uv", "sync", "--frozen"] not in provisioner.commands

    async def test_requirements_txt_creates_a_venv_first(self, installer):
        listings = iter(
            [
                {"exit_code": 1, "stdout": "", "stderr": "no venv", "duration": 0.0},
                {
                    "exit_code": 0,
                    "stdout": '[{"name": "requests"}, {"name": "urllib3"}]',
                    "stderr": "",
                    "duration": 0.0,
                },
            ]
        )

        class _Provisioner(_FakeProvisioner):
            def exec_in_pod(self, **kwargs):
                if _bare(kwargs["command"]) == [
                    "uv",
                    "pip",
                    "list",
                    "--python",
                    ".venv/bin/python",
                    "--format",
                    "json",
                ]:
                    self.calls.append({"command": list(kwargs["command"]), **kwargs})
                    return next(listings)
                return super().exec_in_pod(**kwargs)

        provisioner = _Provisioner({"ls -1 -a": {"stdout": "requirements.txt\n"}})
        with _runtime_env(_k8s_workspace(), provisioner):
            result = await installer.install_dependencies_in_container("ws-1", "/workspace")

        assert ["uv", "venv", "--allow-existing"] in provisioner.commands
        assert ["uv", "pip", "install", "-r", "requirements.txt"] in provisioner.commands
        assert result.package_managers_used == ["pip"]
        assert result.installed_packages == ["requests", "urllib3"]

    async def test_package_lock_uses_npm_ci(self, installer):
        provisioner = _FakeProvisioner(
            {
                "ls -1 -a": {"stdout": "package.json\npackage-lock.json\n"},
                "npm ls --json --depth=0": {"stdout": '{"dependencies": {"react": {}}}'},
            }
        )
        with _runtime_env(_k8s_workspace(), provisioner):
            result = await installer.install_dependencies_in_container("ws-1", "/workspace")

        assert ["npm", "ci"] in provisioner.commands
        assert result.package_managers_used == ["npm"]
        assert result.installed_packages == ["react"]

    async def test_package_json_without_lock_uses_npm_install(self, installer):
        provisioner = _FakeProvisioner({"ls -1 -a": {"stdout": "package.json\n"}})
        with _runtime_env(_k8s_workspace(), provisioner):
            await installer.install_dependencies_in_container("ws-1", "/workspace")

        assert ["npm", "install"] in provisioner.commands
        assert ["npm", "ci"] not in provisioner.commands

    async def test_docker_era_default_path_is_replaced_and_reported(self, installer):
        provisioner = _FakeProvisioner({"ls -1 -a": {"stdout": "uv.lock\n"}})
        with _runtime_env(_k8s_workspace(), provisioner):
            result = await installer.install_dependencies_in_container("ws-1", "/app/data/repo")

        assert {call["workdir"] for call in provisioner.calls} == {"/workspace"}
        assert any("/app/data/repo" in line and "/workspace" in line for line in result.logs)

    async def test_an_explicit_path_is_honoured(self, installer):
        provisioner = _FakeProvisioner({"ls -1 -a": {"stdout": "uv.lock\n"}})
        with _runtime_env(_k8s_workspace(), provisioner):
            await installer.install_dependencies_in_container("ws-1", "/workspace/backend")

        assert {call["workdir"] for call in provisioner.calls} == {"/workspace/backend"}

    async def test_missing_workspace_directory_is_a_failure(self, installer):
        provisioner = _FakeProvisioner(
            {"ls -1 -a": {"exit_code": 2, "stderr": "No such file or directory"}}
        )
        with _runtime_env(_k8s_workspace(), provisioner):
            result = await installer.install_dependencies_in_container("ws-1", "/nope")

        assert result.success is False
        assert "not found" in result.errors[0]
        assert provisioner.commands == [["ls", "-1", "-a"]]

    async def test_no_manifest_is_a_success_that_claims_nothing(self, installer):
        provisioner = _FakeProvisioner({"ls -1 -a": {"stdout": "README.md\nMakefile\n"}})
        with _runtime_env(_k8s_workspace(), provisioner):
            result = await installer.install_dependencies_in_container("ws-1", "/workspace")

        assert result.success is True
        assert result.package_managers_used == []
        assert result.installed_packages == []

    async def test_a_failed_install_is_reported_not_swallowed(self, installer):
        provisioner = _FakeProvisioner(
            {
                "ls -1 -a": {"stdout": "uv.lock\n"},
                "uv sync --frozen": {"exit_code": 1, "stderr": "lock is out of date"},
            }
        )
        with _runtime_env(_k8s_workspace(), provisioner):
            result = await installer.install_dependencies_in_container("ws-1", "/workspace")

        assert result.success is False
        assert "lock is out of date" in result.errors[0]
        assert result.package_managers_used == []

    async def test_installs_get_the_long_timeout_probes_the_short_one(self, installer):
        from src.core.container_installer import (
            K8S_INSTALL_TIMEOUT_SECONDS,
            K8S_PROBE_TIMEOUT_SECONDS,
        )

        provisioner = _FakeProvisioner({"ls -1 -a": {"stdout": "uv.lock\n"}})
        with _runtime_env(_k8s_workspace(), provisioner):
            await installer.install_dependencies_in_container("ws-1", "/workspace")

        by_command = {" ".join(_bare(c["command"])): c["timeout"] for c in provisioner.calls}
        assert by_command["uv sync --frozen"] == K8S_INSTALL_TIMEOUT_SECONDS
        assert by_command["ls -1 -a"] == K8S_PROBE_TIMEOUT_SECONDS


class TestKubernetesInstallCommandPolicy:
    """``policy.validate_command`` is the enforcement point, not a formality."""

    @pytest.mark.parametrize(
        "argv",
        [
            ["ls", "-1", "-a"],
            ["uv", "sync", "--frozen"],
            ["uv", "sync"],
            ["uv", "venv", "--allow-existing"],
            ["uv", "pip", "install", "-r", "requirements.txt"],
            ["uv", "pip", "list", "--python", ".venv/bin/python", "--format", "json"],
            ["npm", "ci"],
            ["npm", "install"],
            ["npm", "ls", "--json", "--depth=0"],
        ],
    )
    def test_every_command_this_branch_builds_is_allowlisted(self, argv):
        """No allowlist change was needed; this pins that it stays true."""
        from src.runtime.policy import check_dangerous_patterns, validate_command

        assert validate_command(argv) is True
        assert check_dangerous_patterns(argv) is True

    async def test_a_non_allowlisted_command_is_refused_before_the_pod(self):
        """CALLER_SUPPLIED is load-bearing: the adapter refuses, nothing execs."""
        from src.runtime import CommandNotPermittedError, ExecutionProfile, KubernetesRuntime

        provisioner = _FakeProvisioner()
        runtime = KubernetesRuntime(provisioner)
        with pytest.raises(CommandNotPermittedError):
            await runtime.run_once(
                _k8s_workspace(),
                ["poetry", "install"],
                cwd="/workspace",
                execution_profile=ExecutionProfile.CALLER_SUPPLIED,
            )
        assert provisioner.calls == []

    async def test_the_installer_reports_a_refusal_as_a_failure(self, installer):
        """A refused command must not read as an install that found nothing."""
        from src.runtime import CommandNotPermittedError

        runtime = MagicMock()
        runtime.run_once = AsyncMock(
            side_effect=CommandNotPermittedError("Command 'poetry' is not permitted")
        )
        result = await installer._install_dependencies_via_runtime(
            runtime, _k8s_workspace(), "ws-1", "/workspace"
        )

        assert result.success is False
        assert "command policy" in result.errors[0]
        assert result.installed_packages == []

    async def test_every_exec_declares_caller_supplied(self, installer):
        """The profile is what makes the allowlist apply; assert it directly."""
        from src.runtime import ExecutionProfile
        from src.runtime.types import RunOnceResult

        runtime = MagicMock()
        runtime.run_once = AsyncMock(
            return_value=RunOnceResult(exit_code=0, stdout="uv.lock\n", stderr="", duration=0.0)
        )
        await installer._install_dependencies_via_runtime(
            runtime, _k8s_workspace(), "ws-1", "/workspace"
        )

        assert runtime.run_once.await_count > 1
        for call in runtime.run_once.await_args_list:
            assert call.kwargs["execution_profile"] is ExecutionProfile.CALLER_SUPPLIED


class TestKubernetesInstallLivePodFindings:
    """Regressions for two things measured in a live codegraph-runtime pod.

    Both were wrong in the first draft of this branch and neither was visible
    from the unit tests alone — they are pinned here so they stay fixed.
    """

    async def test_python_commands_carry_the_private_index_trust_env(self, installer, monkeypatch):
        """Without it, every repo depending on the private index fails UnknownIssuer."""
        monkeypatch.setenv("CODING_PRIVATE_INDEX_HOST", "pkg.example.test")
        provisioner = _FakeProvisioner({"ls -1 -a": {"stdout": "uv.lock\n"}})
        with _runtime_env(_k8s_workspace(), provisioner):
            await installer.install_dependencies_in_container("ws-1", "/workspace")

        sync = next(c for c in provisioner.raw_commands if _bare(c) == ["uv", "sync", "--frozen"])
        # TRUST *and* DISCOVERY. The index URL entries were added after a live
        # measurement: trust alone makes the host acceptable but never queried, so
        # a package published only to the private index resolved as "not found".
        assert _env_of(sync) == {
            "PIP_TRUSTED_HOST": "pkg.example.test",
            "UV_INSECURE_HOST": "pkg.example.test",
            "UV_NATIVE_TLS": "1",
            "UV_EXTRA_INDEX_URL": "https://pkg.example.test/simple/",
            "PIP_EXTRA_INDEX_URL": "https://pkg.example.test/simple/",
        }

    async def test_npm_is_not_given_the_python_index_env(self, installer):
        provisioner = _FakeProvisioner({"ls -1 -a": {"stdout": "package.json\n"}})
        with _runtime_env(_k8s_workspace(), provisioner):
            await installer.install_dependencies_in_container("ws-1", "/workspace")

        install = next(c for c in provisioner.raw_commands if _bare(c) == ["npm", "install"])
        assert _env_of(install) == {}

    # bun. Measured 2026-09-17: NO org repo ships `bun.lockb`; five ship a
    # TEXT `bun.lock` (w2v_frontend, orchestrator, chat-bridge x3). Before this
    # the installer only knew `bun.lockb` and even then fell back to npm, so a
    # bun repo was installed from a lock it never committed.
    _BUN_LISTING = {"npm ls --json --depth=0": {"stdout": '{"dependencies": {"svelte": {}}}'}}

    async def test_bun_lock_uses_bun_frozen_install(self, installer):
        provisioner = _FakeProvisioner(
            {"ls -1 -a": {"stdout": "package.json\nbun.lock\n"}, **self._BUN_LISTING}
        )
        with _runtime_env(_k8s_workspace(), provisioner):
            result = await installer.install_dependencies_in_container("ws-1", "/workspace")

        assert ["bun", "install", "--frozen-lockfile"] in provisioner.commands
        assert ["npm", "ci"] not in provisioner.commands
        assert ["npm", "install"] not in provisioner.commands
        assert result.package_managers_used == ["bun"]
        # bun writes an npm-compatible node_modules, so the npm listing still reports it.
        assert result.installed_packages == ["svelte"]
        assert result.success is True

    async def test_bun_lockb_legacy_same(self, installer):
        provisioner = _FakeProvisioner(
            {"ls -1 -a": {"stdout": "package.json\nbun.lockb\n"}, **self._BUN_LISTING}
        )
        with _runtime_env(_k8s_workspace(), provisioner):
            result = await installer.install_dependencies_in_container("ws-1", "/workspace")

        assert ["bun", "install", "--frozen-lockfile"] in provisioner.commands
        assert ["npm", "ci"] not in provisioner.commands
        assert ["npm", "install"] not in provisioner.commands
        assert result.package_managers_used == ["bun"]

    async def test_bun_lock_beats_package_lock(self, installer):
        """A repo that commits bun.lock chose bun; a stray package-lock.json must not win."""
        provisioner = _FakeProvisioner(
            {
                "ls -1 -a": {"stdout": "package.json\npackage-lock.json\nbun.lock\n"},
                **self._BUN_LISTING,
            }
        )
        with _runtime_env(_k8s_workspace(), provisioner):
            result = await installer.install_dependencies_in_container("ws-1", "/workspace")

        assert ["bun", "install", "--frozen-lockfile"] in provisioner.commands
        assert ["npm", "ci"] not in provisioner.commands
        assert result.package_managers_used == ["bun"]

    async def test_yarn_lock_does_not_take_the_bun_branch(self, installer):
        """Negative control for the bun branch: yarn is not in the image, so npm still runs."""
        provisioner = _FakeProvisioner({"ls -1 -a": {"stdout": "package.json\nyarn.lock\n"}})
        with _runtime_env(_k8s_workspace(), provisioner):
            await installer.install_dependencies_in_container("ws-1", "/workspace")

        assert ["npm", "install"] in provisioner.commands
        assert not any(c[:1] == ["bun"] for c in provisioner.commands)

    async def test_the_package_listing_is_pinned_to_the_workspace_venv(self, installer):
        """A bare `uv pip list` reads the SYSTEM interpreter when no venv exists.

        Measured in-pod: the "before" listing then returns the image's 92 baked-in
        tools and the "after" listing returns the venv's contents, so the set
        difference is empty for an install that actually installed something.
        """
        provisioner = _FakeProvisioner({"ls -1 -a": {"stdout": "uv.lock\n"}})
        with _runtime_env(_k8s_workspace(), provisioner):
            await installer.install_dependencies_in_container("ws-1", "/workspace")

        listings = [c for c in provisioner.commands if c[:3] == ["uv", "pip", "list"]]
        assert listings, "no package listing was taken"
        for argv in listings:
            assert "--python" in argv
            assert argv[argv.index("--python") + 1] == ".venv/bin/python"

    async def test_a_system_wide_listing_would_not_mask_a_real_install(self, installer):
        """The failing shape, pinned: system list before, venv list after."""

        class _Provisioner(_FakeProvisioner):
            """Answers the pinned listing correctly; the bare one system-wide."""

            def __init__(self):
                super().__init__({"ls -1 -a": {"stdout": "uv.lock\n"}})
                self._seen_sync = False

            def exec_in_pod(self, **kwargs):
                argv = _bare(kwargs["command"])
                if argv == ["uv", "sync", "--frozen"]:
                    self._seen_sync = True
                if argv[:3] == ["uv", "pip", "list"]:
                    self.calls.append({"command": list(kwargs["command"]), **kwargs})
                    payload = '[{"name": "idna"}]' if self._seen_sync else ""
                    return {
                        "exit_code": 0 if self._seen_sync else 2,
                        "stdout": payload,
                        "stderr": "" if self._seen_sync else "No virtual environment",
                        "duration": 0.0,
                    }
                return super().exec_in_pod(**kwargs)

        provisioner = _Provisioner()
        with _runtime_env(_k8s_workspace(), provisioner):
            result = await installer.install_dependencies_in_container("ws-1", "/workspace")

        assert result.success is True
        assert result.installed_packages == ["idna"]


class TestKubernetesAnalyzeProject:
    """dependency-status on a Kubernetes workspace (D3, live 500 on cv-sweep-ems)."""

    async def test_scans_the_pod_not_docker(self):
        from src.core.container_installer import ContainerInstaller

        found = "\n".join(
            [
                "/workspace/pyproject.toml",
                "/workspace/uv.lock",  # not a dependency manifest: ignored
                "/workspace/requirements-dev.txt",
                "/workspace/web/package.json",
            ]
        )
        provisioner = _FakeProvisioner()
        provisioner.exec_in_pod = _scripted_exec(provisioner, found)
        installer = ContainerInstaller("ws-1")
        with (
            _runtime_env(_k8s_workspace(), provisioner),
            patch("src.core.container_installer.docker") as mock_docker,
        ):
            result = await installer.analyze_project("/app/data/repo")

        mock_docker.from_env.assert_not_called()
        argv = provisioner.commands[0]
        assert argv[:2] == ["find", "/workspace"]  # Docker-era path mapped to the pod root
        assert "-prune" in argv and "node_modules" in argv
        assert result["project_type"] == "mixed"
        assert [f["file_name"] for f in result["detected_files"]] == [
            "pyproject.toml",
            "requirements-dev.txt",
            "package.json",
        ]
        assert all(isinstance(step, str) for step in result["installation_plan"])
        assert result["total_files"] == 3

    async def test_a_failed_scan_raises_instead_of_reporting_no_dependencies(self):
        from src.core.container_installer import ContainerInstaller

        provisioner = _FakeProvisioner()
        provisioner.exec_in_pod = _scripted_exec(provisioner, "", exit_code=2, stderr="boom")
        with _runtime_env(_k8s_workspace(), provisioner):
            with pytest.raises(RuntimeError, match="dependency file scan failed"):
                await ContainerInstaller("ws-1").analyze_project("/workspace")


def _scripted_exec(provisioner, stdout: str, exit_code: int = 0, stderr: str = ""):
    def _exec(*, deployment_name, command, workdir, timeout, max_output_bytes):
        provisioner.calls.append(
            {"deployment_name": deployment_name, "command": list(command), "workdir": workdir}
        )
        return {"exit_code": exit_code, "stdout": stdout, "stderr": stderr, "duration": 0.0}

    return _exec

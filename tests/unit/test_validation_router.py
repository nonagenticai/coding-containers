"""Tests for routers/validation.py."""

import json
import os
import shutil
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.runtime import ExecutionProfile, RunOnceResult, RuntimeKind


@pytest.fixture(autouse=True)
def _owns_workspace(grant_workspace_ownership):
    """See ``grant_workspace_ownership``. These tests predate #135's ownership
    gate; the gate itself is tested in ``test_agent_exec_tenant_scope.py``."""


def _result(stdout="", exit_code=0, stderr=""):
    return RunOnceResult(exit_code=exit_code, stdout=stdout, stderr=stderr, duration=0.01)


def _fake_workspace(
    resolver=None,
    exec_result=None,
    resolver_error=None,
    resolver_result=None,
    exec_error=None,
    missing_roots=(),
):
    """Stand-in for ``run_workspace_command``: answers the resolver exec
    (``python3 -I -c PROG [--root R] items...``) from ``resolver`` and the
    snippet exec (``python -c code``) with ``exec_result``. A ``resolver``
    value is a status, or a ``(status, symbol)`` pair."""

    async def fake(name, tenant_id, argv, *, cwd=None, timeout=30):
        from src.routers.validation import _ROOT_PROBE

        if argv[:4] == ["python3", "-I", "-c", _ROOT_PROBE]:
            return RuntimeKind.KUBERNETES, _result(exit_code=3 if argv[4] in missing_roots else 0)
        if argv[0] == "python3":
            if resolver_error is not None:
                raise resolver_error
            if resolver_result is not None:
                return RuntimeKind.KUBERNETES, resolver_result
            items = argv[4:]
            if items[:1] == ["--root"]:
                items = items[2:]
            results = []
            for it in items:
                status, symbol = (
                    resolver[it] if isinstance(resolver[it], tuple) else (resolver[it], None)
                )
                results.append({"item": it, "status": status, "symbol": symbol})
            payload = {"interpreter": "venv", "results": results}
            return RuntimeKind.KUBERNETES, _result(json.dumps(payload))
        if exec_error is not None:
            raise exec_error
        return RuntimeKind.KUBERNETES, exec_result or _result()

    return AsyncMock(side_effect=fake)


async def _post(app, client, body, fake=None):
    from coding_client.auth.tenant import require_tenant

    app.dependency_overrides[require_tenant] = lambda: "test-tenant"
    try:
        if fake is None:
            return await client.post("/api/validation/multi-line-code-snippet", json=body)
        with patch("src.routers.validation.run_workspace_command", fake):
            return await client.post("/api/validation/multi-line-code-snippet", json=body)
    finally:
        app.dependency_overrides.clear()


# ---- model/helper tests ----


class TestMultiLineCodeSnippetRequest:
    def test_valid_request(self):
        from src.routers.validation import MultiLineCodeSnippetRequest

        req = MultiLineCodeSnippetRequest(code_snippet="x = 1\nprint(x)")
        assert req.code_snippet == "x = 1\nprint(x)"
        assert req.include_execution_validation is False
        assert req.execution_timeout == 10

    def test_empty_snippet_rejected(self):
        from src.routers.validation import MultiLineCodeSnippetRequest

        with pytest.raises(Exception):
            MultiLineCodeSnippetRequest(code_snippet="")

    def test_whitespace_only_rejected(self):
        from src.routers.validation import MultiLineCodeSnippetRequest

        with pytest.raises(Exception):
            MultiLineCodeSnippetRequest(code_snippet="   ")

    def test_too_short_rejected(self):
        from src.routers.validation import MultiLineCodeSnippetRequest

        with pytest.raises(Exception):
            MultiLineCodeSnippetRequest(code_snippet="x")

    def test_optional_fields(self):
        from src.routers.validation import MultiLineCodeSnippetRequest

        req = MultiLineCodeSnippetRequest(
            code_snippet="x = 1\nprint(x)",
            container_name="test-ctr",
            codebase_path="/app",
            include_execution_validation=True,
            execution_timeout=30,
        )
        assert req.container_name == "test-ctr"
        assert req.codebase_path == "/app"
        assert req.include_execution_validation is True
        assert req.execution_timeout == 30


class TestStaticAnalysis:
    def test_valid_code(self):
        from src.routers.validation import _static_analysis

        result = _static_analysis("x = 1\ny = 2\nprint(x + y)")
        assert result.valid is True
        assert result.overall_score == 1.0
        assert result.total_checks == 1
        assert result.passed_checks == 1
        assert result.issues == []
        assert result.import_resolution == "not_computed"
        assert result.not_computed_reason == "no workspace"

    def test_syntax_error(self):
        from src.routers.validation import _static_analysis

        result = _static_analysis("def foo(:\n    pass")
        assert result.valid is False
        assert result.overall_score == 0.0
        assert result.total_checks == 1
        assert result.passed_checks == 0
        assert len(result.issues) == 1
        assert result.issues[0]["type"] == "syntax"
        assert result.issues[0]["severity"] == "error"

    def test_multiline_valid(self):
        from src.routers.validation import _static_analysis

        code = """
class Foo:
    def bar(self):
        return 42

def main():
    f = Foo()
    print(f.bar())
"""
        result = _static_analysis(code)
        assert result.valid is True

    def test_imports_collected_with_relative_unverifiable_and_guarded(self):
        from src.routers.validation import _static_analysis

        code = (
            "import os, json as j\n"
            "from .sibling import thing\n"
            "try:\n"
            "    import ujson\n"
            "except ImportError:\n"
            "    ujson = None\n"
            "def f():\n"
            "    from a.b import C\n"
        )
        result = _static_analysis(code)
        got = [(i.module, i.name, i.line, i.status, i.guarded) for i in result.imports]
        assert got == [
            ("os", None, 1, "not_checked", False),
            ("json", None, 1, "not_checked", False),
            (".sibling", "thing", 2, "unverifiable", False),
            ("ujson", None, 4, "not_checked", True),
            ("a.b", "C", 8, "not_checked", False),
        ]
        # Unresolved-until-checked never invalidates on its own.
        assert result.valid is True


@pytest.mark.asyncio
class TestExecuteInContainer:
    async def test_execution_uses_runtime_run_once_not_docker_service(self):
        """The workspace is a Kubernetes pod: execution must go through
        resolve_runtime -> run_once, never src.services.docker_service."""
        from src.routers.validation import _execute_in_container

        runtime = MagicMock()
        runtime.kind = RuntimeKind.KUBERNETES
        runtime.run_once = AsyncMock(return_value=_result("1\n"))
        manager = MagicMock()
        manager.get_container.return_value = MagicMock(codebase_path=None)

        with (
            patch("src.routers.container_exec.get_container_manager", return_value=manager),
            patch("src.routers.container_exec.resolve_runtime", return_value=runtime),
            patch("src.core.container_manager_core.get_container_manager"),
            patch("src.services.docker_service.get_docker_service") as docker_svc,
        ):
            result = await _execute_in_container("print(1)", "test-ctr", 10, "test-tenant")

        assert result.execution_successful is True
        assert result.stdout == "1\n"
        docker_svc.assert_not_called()
        manager.get_container.assert_called_once_with("test-ctr", tenant_id="test-tenant")
        args, kwargs = runtime.run_once.await_args
        assert args[1] == ["python", "-c", "print(1)"]
        assert kwargs["cwd"] == "/workspace"
        assert kwargs["timeout"] == 10
        assert kwargs["execution_profile"] is ExecutionProfile.CALLER_SUPPLIED

    async def test_runtime_unavailable_is_reported_not_docker(self):
        from src.routers.validation import _execute_in_container

        manager = MagicMock()
        manager.get_container.return_value = MagicMock(codebase_path=None)
        with (
            patch("src.routers.container_exec.get_container_manager", return_value=manager),
            patch(
                "src.routers.container_exec.resolve_runtime",
                side_effect=RuntimeError("provisioner is not available"),
            ),
            patch("src.core.container_manager_core.get_container_manager"),
        ):
            result = await _execute_in_container("print(1)", "test-ctr", 10, "test-tenant")

        assert result.execution_successful is False
        assert "503" in result.stderr
        assert "provisioner is not available" in result.stderr
        assert "Docker not available" not in result.stderr

    async def test_execution_failure(self):
        from src.routers.validation import _execute_in_container

        fake = _fake_workspace(exec_result=_result(exit_code=1, stderr="NameError: x"))
        with patch("src.routers.validation.run_workspace_command", fake):
            result = await _execute_in_container("print(x)", "test-ctr", 10, "test-tenant")

        assert result.execution_successful is False
        assert len(result.runtime_errors) == 1
        assert "NameError" in result.runtime_errors[0]

    async def test_uses_default_container_name(self):
        from src.routers.validation import _execute_in_container

        fake = _fake_workspace()
        with patch("src.routers.validation.run_workspace_command", fake):
            await _execute_in_container("x=1", None, 10, "test-tenant")

        assert fake.await_args.args[0] == "default"
        assert fake.await_args.args[2] == ["python", "-c", "x=1"]


# ---- endpoint tests ----


@pytest.mark.asyncio
class TestValidateMultiLineEndpoint:
    async def test_valid_code_static_only(self, app, client):
        response = await _post(app, client, {"code_snippet": "x = 1\nprint(x)"})

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["static_analysis"]["valid"] is True
        assert data["execution_analysis"] is None
        assert data["overall_confidence"] == 1.0
        assert "Static: PASS" in data["validation_summary"]
        assert data["suggestions"] == []

    async def test_syntax_error_code(self, app, client):
        response = await _post(app, client, {"code_snippet": "def foo(:\n    pass"})

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert data["static_analysis"]["valid"] is False
        assert data["overall_confidence"] == 0.0
        assert "Static: FAIL" in data["validation_summary"]
        assert any("syntax" in s.lower() for s in data["suggestions"])

    async def test_with_execution_validation_success(self, app, client):
        fake = _fake_workspace(exec_result=_result("42\n"))
        response = await _post(
            app,
            client,
            {"code_snippet": "print(42)", "include_execution_validation": True},
            fake,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["execution_analysis"] is not None
        assert data["execution_analysis"]["execution_successful"] is True
        assert "Execution: PASS" in data["validation_summary"]

    async def test_with_execution_validation_runtime_error(self, app, client):
        fake = _fake_workspace(exec_result=_result(exit_code=1, stderr="ZeroDivisionError"))
        response = await _post(
            app,
            client,
            {"code_snippet": "print(1/0)", "include_execution_validation": True},
            fake,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert data["execution_analysis"]["execution_successful"] is False
        assert "Execution: FAIL" in data["validation_summary"]
        # Confidence is average of static (1.0) and execution (0.0) = 0.5
        assert data["overall_confidence"] == 0.5
        assert any("runtime" in s.lower() for s in data["suggestions"])

    async def test_package_info_present(self, app, client):
        response = await _post(app, client, {"code_snippet": "import os\nprint(os.getcwd())"})

        assert response.status_code == 200
        data = response.json()
        assert "package_info" in data
        assert "packages" in data["package_info"]

    async def test_internal_error_handling(self, app, client):
        """Test that unexpected exceptions return 500."""
        with patch(
            "src.routers.validation._static_analysis",
            side_effect=RuntimeError("unexpected"),
        ):
            response = await _post(app, client, {"code_snippet": "x = 1\nprint(x)"})

        assert response.status_code == 500


@pytest.mark.asyncio
class TestImportResolution:
    """Import resolution against the named workspace, runtime mocked."""

    async def test_ghost_package_is_invalid(self, app, client):
        fake = _fake_workspace({"totally_ghost_pkg_xyz": "unresolved"})
        response = await _post(
            app,
            client,
            {"code_snippet": "import totally_ghost_pkg_xyz\n", "container_name": "ws"},
            fake,
        )

        data = response.json()
        static = data["static_analysis"]
        assert data["valid"] is False
        assert static["valid"] is False
        assert static["import_resolution"] == "workspace"
        assert static["unresolved_imports"] == ["totally_ghost_pkg_xyz"]
        assert static["imports"][0]["status"] == "unresolved"
        assert any(i["type"] == "import" for i in static["issues"])
        assert "Imports: FAIL" in data["validation_summary"]
        assert any("totally_ghost_pkg_xyz" in s for s in data["suggestions"])

    async def test_stdlib_import_is_valid(self, app, client):
        fake = _fake_workspace({"json": "stdlib"})
        response = await _post(
            app,
            client,
            {"code_snippet": "import json\nprint(json.dumps(1))\n", "container_name": "ws"},
            fake,
        )

        data = response.json()
        assert data["valid"] is True
        assert data["static_analysis"]["imports"][0]["status"] == "stdlib"
        assert data["static_analysis"]["unresolved_imports"] == []
        assert data["static_analysis"]["total_checks"] == 2
        assert data["static_analysis"]["passed_checks"] == 2
        assert "Imports: PASS" in data["validation_summary"]

    async def test_repo_module_present_is_valid(self, app, client):
        fake = _fake_workspace({"src.mymod:Foo": "repo"})
        response = await _post(
            app,
            client,
            {"code_snippet": "from src.mymod import Foo\nFoo()\n", "container_name": "ws"},
            fake,
        )

        data = response.json()
        assert data["valid"] is True
        imp = data["static_analysis"]["imports"][0]
        assert (imp["module"], imp["name"], imp["status"]) == ("src.mymod", "Foo", "repo")

    async def test_ghost_symbol_from_repo_module_is_invalid(self, app, client):
        fake = _fake_workspace({"src.mymod:Ghost": "unresolved_symbol"})
        response = await _post(
            app,
            client,
            {"code_snippet": "from src.mymod import Ghost\n", "container_name": "ws"},
            fake,
        )

        data = response.json()
        assert data["valid"] is False
        assert data["static_analysis"]["unresolved_imports"] == ["src.mymod.Ghost"]

    async def test_declared_not_installed_is_valid_but_listed(self, app, client):
        fake = _fake_workspace({"httpx": "declared_not_installed"})
        response = await _post(
            app,
            client,
            {"code_snippet": "import httpx\n", "container_name": "ws"},
            fake,
        )

        data = response.json()
        assert data["valid"] is True
        assert data["static_analysis"]["imports"][0]["status"] == "declared_not_installed"

    async def test_exec_failure_is_not_computed_never_a_silent_pass(self, app, client):
        fake = _fake_workspace(resolver_error=HTTPException(status_code=503, detail="down"))
        response = await _post(
            app,
            client,
            {"code_snippet": "import totally_ghost_pkg_xyz\n", "container_name": "ws"},
            fake,
        )

        assert response.status_code == 200
        data = response.json()
        static = data["static_analysis"]
        assert static["import_resolution"] == "not_computed"
        assert "503" in static["not_computed_reason"]
        assert static["imports"][0]["status"] == "not_checked"
        # valid reflects syntax only — but the caller is TOLD imports were unchecked.
        assert data["valid"] is True
        assert any("Imports were NOT checked" in s for s in data["suggestions"])
        assert "Imports: NOT CHECKED" in data["validation_summary"]

    async def test_resolver_nonzero_exit_is_not_computed(self, app, client):
        fake = _fake_workspace(resolver_result=_result(exit_code=1, stderr="probe failed"))
        response = await _post(
            app,
            client,
            {"code_snippet": "import json\n", "container_name": "ws"},
            fake,
        )

        static = response.json()["static_analysis"]
        assert static["import_resolution"] == "not_computed"
        assert "probe failed" in static["not_computed_reason"]

    async def test_relative_only_snippet_is_not_computed_and_not_executed(self, app, client):
        """Nothing was checked, so nothing may be reported as checked."""
        fake = _fake_workspace({})
        response = await _post(
            app,
            client,
            {"code_snippet": "from .sibling import thing\n", "container_name": "ws"},
            fake,
        )

        data = response.json()
        static = data["static_analysis"]
        assert data["valid"] is True
        assert data["verdict"] == "not_computed"
        assert static["imports"][0]["status"] == "unverifiable"
        assert static["import_resolution"] == "not_computed"
        assert static["not_computed_reason"] == "only relative imports"
        fake.assert_not_awaited()

    async def test_guarded_unresolved_import_does_not_invalidate(self, app, client):
        fake = _fake_workspace({"ujson": "unresolved"})
        code = "try:\n    import ujson\nexcept ImportError:\n    ujson = None\n"
        response = await _post(app, client, {"code_snippet": code, "container_name": "ws"}, fake)

        data = response.json()
        assert data["valid"] is True
        imp = data["static_analysis"]["imports"][0]
        assert imp["status"] == "unresolved" and imp["guarded"] is True

    async def test_no_workspace_is_not_computed_and_never_execs(self, app, client):
        fake = _fake_workspace({})
        response = await _post(app, client, {"code_snippet": "import ghost_pkg\n"}, fake)

        static = response.json()["static_analysis"]
        assert static["import_resolution"] == "not_computed"
        assert static["not_computed_reason"] == "no workspace"
        fake.assert_not_awaited()

    async def test_resolver_argv_is_fixed_program_plus_names(self, app, client):
        """User text reaches the workspace only as argv ITEMS after a constant
        program — never interpolated into source, never through a shell."""
        from src.routers.validation import _RESOLVER_PROGRAM

        fake = _fake_workspace({"zzz_marker_mod": "unresolved", "json": "stdlib"})
        await _post(
            app,
            client,
            {"code_snippet": "import zzz_marker_mod\nimport json\n", "container_name": "ws"},
            fake,
        )

        name, tenant, argv = fake.await_args.args
        assert (name, tenant) == ("ws", "test-tenant")
        # -I: the repository (the cwd) never shadows the program's own imports.
        assert argv[:4] == ["python3", "-I", "-c", _RESOLVER_PROGRAM]
        assert argv[4:] == ["json", "zzz_marker_mod"]
        assert fake.await_args.kwargs["cwd"] is None
        assert "zzz_marker_mod" not in _RESOLVER_PROGRAM


@pytest.mark.asyncio
class TestVerdictAndRoots:
    """``verdict``, ``codebase_path`` and refusals, runtime mocked."""

    async def test_verdict_valid_invalid_not_computed(self, app, client):
        ok = await _post(
            app,
            client,
            {"code_snippet": "import json\n", "container_name": "ws"},
            _fake_workspace({"json": "stdlib"}),
        )
        bad = await _post(
            app,
            client,
            {"code_snippet": "import ghost_pkg\n", "container_name": "ws"},
            _fake_workspace({"ghost_pkg": "unresolved"}),
        )
        unchecked = await _post(app, client, {"code_snippet": "import json\n"})
        no_imports = await _post(app, client, {"code_snippet": "x = 1\n"})

        assert ok.json()["verdict"] == "valid"
        assert bad.json()["verdict"] == "invalid"
        # No workspace: the import was never checked, so the verdict cannot be valid.
        assert unchecked.json()["valid"] is True
        assert unchecked.json()["verdict"] == "not_computed"
        assert no_imports.json()["verdict"] == "valid"

    async def test_unverifiable_symbol_counts_module_and_stays_valid(self, app, client):
        fake = _fake_workspace({"sys:path": ("stdlib", "unverifiable")})
        response = await _post(
            app, client, {"code_snippet": "from sys import path\n", "container_name": "ws"}, fake
        )

        data = response.json()
        imp = data["static_analysis"]["imports"][0]
        assert (imp["status"], imp["symbol_status"]) == ("stdlib", "unverifiable")
        assert data["valid"] is True
        assert data["verdict"] == "valid"

    async def test_unverifiable_module_is_not_computed_verdict(self, app, client):
        fake = _fake_workspace({"maybe_pkg": "unverifiable"})
        response = await _post(
            app, client, {"code_snippet": "import maybe_pkg\n", "container_name": "ws"}, fake
        )

        data = response.json()
        assert data["valid"] is True
        assert data["verdict"] == "not_computed"

    async def test_bad_symbol_value_is_not_computed(self, app, client):
        fake = _fake_workspace({"json": ("stdlib", "bogus")})
        response = await _post(
            app, client, {"code_snippet": "import json\n", "container_name": "ws"}, fake
        )

        static = response.json()["static_analysis"]
        assert static["import_resolution"] == "not_computed"
        assert "incomplete" in static["not_computed_reason"]

    async def test_codebase_path_is_the_resolution_and_execution_root(self, app, client):
        fake = _fake_workspace({"json": "stdlib"})
        await _post(
            app,
            client,
            {
                "code_snippet": "import json\n",
                "container_name": "ws",
                "codebase_path": " /srv/app ",
                "include_execution_validation": True,
            },
            fake,
        )

        resolver_call, exec_call = fake.await_args_list
        assert resolver_call.args[2][4:] == ["--root", "/srv/app", "json"]
        assert resolver_call.kwargs["cwd"] == "/srv/app"
        assert exec_call.args[2][0] == "python"
        assert exec_call.kwargs["cwd"] == "/srv/app"

    @pytest.mark.parametrize(("missing", "status"), [(True, "not_computed"), (False, "computed")])
    async def test_nonexistent_codebase_path_execution_is_not_computed(
        self, app, client, missing, status
    ):
        """The runtime ``cd``s into the root first; a failed cd is a non-zero exit
        that says nothing about the code. Control: an existing root keeps the
        failure as a computed one."""
        fake = _fake_workspace(
            {"json": "stdlib"},
            exec_result=_result(exit_code=2, stderr="sh: cd: can't cd to /srv/missing"),
            missing_roots=("/srv/missing",) if missing else (),
        )
        response = await _post(
            app,
            client,
            {
                "code_snippet": "import json\n",
                "container_name": "ws",
                "codebase_path": "/srv/missing",
                "include_execution_validation": True,
            },
            fake,
        )

        data = response.json()
        execution = data["execution_analysis"]
        assert execution["execution_status"] == status
        probe = fake.await_args_list[-1]
        assert probe.args[2][4:] == ["/srv/missing"]
        assert probe.kwargs["cwd"] == "/"
        if missing:
            assert "does not exist" in execution["not_computed_reason"]
            assert data["verdict"] == "not_computed"
        else:
            assert data["verdict"] == "invalid"

    @pytest.mark.parametrize("path", ["relative/dir", "/", "./x"])
    async def test_invalid_codebase_path_is_rejected(self, app, client, path):
        fake = _fake_workspace({})
        response = await _post(
            app,
            client,
            {"code_snippet": "import json\n", "container_name": "ws", "codebase_path": path},
            fake,
        )

        assert response.status_code == 422
        fake.assert_not_awaited()

    async def test_refused_execution_is_not_computed_never_invalid(self, app, client):
        """A policy refusal is a statement about the policy, not the code."""
        fake = _fake_workspace(
            {"concurrent.futures:ThreadPoolExecutor": ("stdlib", "found")},
            exec_error=HTTPException(
                status_code=400, detail="Command contains a dangerous pattern"
            ),
        )
        code = (
            "from concurrent.futures import ThreadPoolExecutor\n"
            "executor = ThreadPoolExecutor()\n"
            "executor.shutdown(wait=True)\n"
        )
        response = await _post(
            app,
            client,
            {"code_snippet": code, "container_name": "ws", "include_execution_validation": True},
            fake,
        )

        data = response.json()
        execution = data["execution_analysis"]
        assert execution["execution_status"] == "not_computed"
        assert execution["not_computed_reason"].startswith("refused by command policy")
        assert data["valid"] is True
        assert data["verdict"] == "not_computed"
        assert data["overall_confidence"] == 1.0
        assert "Execution: NOT RUN" in data["validation_summary"]
        assert any("Execution was NOT run" in s for s in data["suggestions"])

    async def test_refused_resolver_is_not_computed(self, app, client):
        fake = _fake_workspace(resolver_error=HTTPException(status_code=400, detail="refused"))
        response = await _post(
            app, client, {"code_snippet": "import json\n", "container_name": "ws"}, fake
        )

        static = response.json()["static_analysis"]
        assert static["import_resolution"] == "not_computed"
        assert static["not_computed_reason"].startswith("refused by command policy")


@pytest.mark.asyncio
async def test_policy_refusal_through_the_real_exec_path_is_not_computed():
    """End to end through ``run_workspace_command``: the runtime applies the real
    command policy and refuses; the snippet is reported as not run."""
    from src.routers.validation import _execute_in_container
    from src.runtime.policy import enforce_command_policy

    async def run_once(info, argv, *, execution_profile, **_):
        # The policy matches program tokens, so snippet text can no longer trip
        # it; feed the real policy an argv it refuses to drive the path.
        enforce_command_policy(["shutdown", "-h", "now"], execution_profile)
        return _result()

    runtime = MagicMock()
    runtime.kind = RuntimeKind.KUBERNETES
    runtime.run_once = AsyncMock(side_effect=run_once)
    manager = MagicMock()
    manager.get_container.return_value = MagicMock(codebase_path=None)
    with (
        patch("src.routers.container_exec.get_container_manager", return_value=manager),
        patch("src.routers.container_exec.resolve_runtime", return_value=runtime),
        patch("src.core.container_manager_core.get_container_manager"),
    ):
        result = await _execute_in_container(
            "import os\nos.system('true')\nx.shutdown(wait=True)\n", "ws", 10, "test-tenant"
        )

    runtime.run_once.assert_awaited_once()
    assert result.execution_status == "not_computed"
    assert result.not_computed_reason.startswith("refused by command policy")


# ---- the resolver program itself, run for real ----


def test_resolver_program_passes_the_runtime_command_policy():
    """The constant program must pass the real runtime command policy, or EVERY
    resolution would be refused."""
    from src.routers.validation import _RESOLVER_PROGRAM
    from src.runtime.policy import check_dangerous_patterns, validate_command

    argv = ["python3", "-I", "-c", _RESOLVER_PROGRAM, "--root", "/workspace", "json"]
    assert validate_command(argv)
    assert check_dangerous_patterns(argv)


def _resolver_payload(root, *items, cwd=None, flags=()):
    from src.routers.validation import _RESOLVER_PROGRAM

    proc = subprocess.run(
        [sys.executable, "-I", "-c", _RESOLVER_PROGRAM, *flags, "--root", str(root), *items],
        cwd=cwd or root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _statuses(payload):
    return {r["item"]: (r["status"], r["symbol"]) for r in payload["results"]}


def _run_resolver(root, *items, cwd=None):
    return _statuses(_resolver_payload(root, *items, cwd=cwd))


def _other_python():
    """A real interpreter of a DIFFERENT minor version than the test's, or None."""
    names = [f"python3.{m}" for m in range(8, 20)] + ["python3"]
    cands = [shutil.which(n) for n in names] + ["/usr/bin/python3"]
    for cand in cands:
        if not cand or not os.access(cand, os.X_OK):
            continue
        try:
            out = subprocess.run(
                [cand, "-I", "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout.strip()
        except OSError, subprocess.SubprocessError:
            continue
        if out and out != f"{sys.version_info[0]}.{sys.version_info[1]}":
            return os.path.realpath(cand), out
    return None


def _ext_suffix(python):
    return subprocess.run(
        [python, "-I", "-c", "import importlib.machinery as m; print(m.EXTENSION_SUFFIXES[0])"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()


def _site_packages(root):
    """A workspace venv, so the test interpreter's own packages never leak in."""
    sp = root / ".venv" / "lib" / "python3.99" / "site-packages"
    sp.mkdir(parents=True)
    return sp


def _write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class TestResolverProgram:
    def test_migrations_dir_declared_but_not_installed(self, tmp_path):
        _site_packages(tmp_path)
        _write(tmp_path / "alembic" / "env.py", "x = 1\n")
        _write(
            tmp_path / "pyproject.toml", '[project]\nname="a"\ndependencies = ["alembic>=1.13"]\n'
        )

        got = _run_resolver(tmp_path, "alembic:op", "alembic.config:Config")

        assert got == {
            "alembic:op": ("declared_not_installed", None),
            "alembic.config:Config": ("declared_not_installed", None),
        }

    def test_migrations_dir_undeclared_is_unresolved(self, tmp_path):
        _site_packages(tmp_path)
        _write(tmp_path / "alembic" / "env.py", "x = 1\n")

        got = _run_resolver(tmp_path, "alembic:op", "alembic.config:Config")

        assert got == {
            "alembic:op": ("unresolved", None),
            "alembic.config:Config": ("unresolved", None),
        }

    def test_installed_package_beats_repo_namespace_dir(self, tmp_path):
        sp = _site_packages(tmp_path)
        _write(tmp_path / "alembic" / "env.py", "x = 1\n")
        _write(sp / "alembic" / "__init__.py")
        _write(sp / "alembic" / "op.py", "def create_table(): ...\n")
        _write(sp / "alembic" / "config.py", "class Config: ...\n")

        got = _run_resolver(
            tmp_path, "alembic:op", "alembic.config:Config", "alembic.config:Ghost", "alembic.env"
        )

        assert got == {
            "alembic:op": ("installed", "found"),
            "alembic.config:Config": ("installed", "found"),
            "alembic.config:Ghost": ("unresolved_symbol", "missing"),
            # The repo directory is shadowed by the regular package, as at import time.
            "alembic.env": ("unresolved", None),
        }

    def test_repo_package_module_level_bindings(self, tmp_path):
        _site_packages(tmp_path)
        _write(
            tmp_path / "pkg" / "__init__.py",
            "import os.path\n"
            "from json import dumps as jd\n"
            "for loop_name in range(1):\n    pass\n"
            "with open(__file__) as with_name:\n    pass\n"
            "a_name, (b_name, c_name) = 1, (2, 3)\n"
            "ann_name: int = 1\n"
            "if (walrus_name := 1):\n    pass\n"
            "try:\n    try_name = 1\nexcept ImportError:\n    except_name = 2\n"
            "while False:\n    while_name = 1\n"
            "match 1:\n    case [cap_name, *star_name]:\n        pass\n"
            "    case {'k': v, **rest_name}:\n        pass\n"
            "def setter():\n    global global_name\n    global_name = 1\n"
            "class Klass:\n    inner_name = 1\n"
            "only_annotated: int\n",
        )
        _write(tmp_path / "pkg" / "sub.py", "def f(): ...\n")
        found = [
            "os",
            "jd",
            "loop_name",
            "with_name",
            "a_name",
            "c_name",
            "ann_name",
            "walrus_name",
            "try_name",
            "except_name",
            "while_name",
            "cap_name",
            "star_name",
            "rest_name",
            "global_name",
            "Klass",
            "setter",
            "sub",
        ]
        missing = ["ghost", "inner_name", "only_annotated", "dumps"]

        got = _run_resolver(tmp_path, *[f"pkg:{n}" for n in found + missing], "pkg.sub", "pkg.nope")

        for n in found:
            assert got[f"pkg:{n}"] == ("repo", "found"), n
        for n in missing:
            assert got[f"pkg:{n}"] == ("unresolved_symbol", "missing"), n
        assert got["pkg.sub"] == ("repo", None)
        assert got["pkg.nope"] == ("unresolved", None)

    def test_opaque_modules_are_unverifiable(self, tmp_path):
        _site_packages(tmp_path)
        _write(tmp_path / "star_mod.py", "from os import *\n")
        _write(tmp_path / "lazy_mod.py", "def __getattr__(name):\n    return name\n")
        _write(tmp_path / "broken_mod.py", "def (:\n")

        got = _run_resolver(tmp_path, "star_mod:anything", "lazy_mod:anything", "broken_mod:x")

        assert got == {
            "star_mod:anything": ("repo", "unverifiable"),
            "lazy_mod:anything": ("repo", "unverifiable"),
            "broken_mod:x": ("repo", "unverifiable"),
        }

    def test_stdlib_and_installed_symbols_and_submodules(self, tmp_path):
        import importlib.machinery

        sp = _site_packages(tmp_path)
        _write(sp / "requests" / "__init__.py", "class Session: ...\n")
        _write(sp / "numpy" / "__init__.py", "array = list\n")
        _write(sp / ("fastext" + importlib.machinery.EXTENSION_SUFFIXES[0]))

        got = _run_resolver(
            tmp_path,
            "json:dumps",
            "json:totally_ghost",
            "os",
            "os.path",
            "os.ghostmod",
            "sys:path",
            "xml.etree.ElementTree:Element",
            "requests:Session",
            "requests:Ghost",
            "numpy",
            "numpy.ghost",
            "fastext:anything",
        )

        assert got == {
            "json:dumps": ("stdlib", "found"),
            "json:totally_ghost": ("unresolved_symbol", "missing"),
            "os": ("stdlib", None),
            "os.path": ("stdlib", None),
            "os.ghostmod": ("unresolved", None),
            "sys:path": ("stdlib", "unverifiable"),
            "xml.etree.ElementTree:Element": ("stdlib", "found"),
            "requests:Session": ("installed", "found"),
            "requests:Ghost": ("unresolved_symbol", "missing"),
            "numpy": ("installed", None),
            "numpy.ghost": ("unresolved", None),
            "fastext:anything": ("installed", "unverifiable"),
        }

    def test_repo_json_py_does_not_shadow_the_program(self, tmp_path):
        """Run with -I from the repo root: the program's own ``import json`` must
        not load the repo's json.py. (For the CALLER's ``import json`` the repo
        file does win — that is real precedence from the repo root.)"""
        _site_packages(tmp_path)
        _write(tmp_path / "json.py", "raise SystemExit('shadowed')\n")

        got = _run_resolver(tmp_path, "os", "json")

        assert got == {"os": ("stdlib", None), "json": ("repo", None)}

    def test_requirements_dist_names_and_includes(self, tmp_path):
        _site_packages(tmp_path)
        _write(
            tmp_path / "requirements.txt",
            "# comment\n--index-url https://example.invalid/simple\n-r base.txt\n"
            "git+https://example.invalid/x.git#egg=beautifulsoup4\n"
            "python-dotenv==1.0  # inline\nPillow\n",
        )
        _write(tmp_path / "base.txt", "PyYAML>=6\nscikit-learn\n")

        got = _run_resolver(tmp_path, "yaml", "bs4", "dotenv", "PIL", "sklearn", "ghostpkg")

        assert {k: v[0] for k, v in got.items()} == {
            "yaml": "declared_not_installed",
            "bs4": "declared_not_installed",
            "dotenv": "declared_not_installed",
            "PIL": "declared_not_installed",
            "sklearn": "declared_not_installed",
            "ghostpkg": "unresolved",
        }

    def test_unreadable_manifest_makes_absent_modules_unverifiable(self, tmp_path):
        """setup.py declares its deps in a non-literal: the declared set is
        unknown, so an absent module is unverifiable, never unresolved."""
        _site_packages(tmp_path)
        _write(tmp_path / "setup.py", "REQS = []\nsetup(install_requires=REQS)\n")

        assert _run_resolver(tmp_path, "somepkg") == {"somepkg": ("unverifiable", None)}

    def test_toml_fallback_parses_dependencies(self):
        from src.routers.validation import _RESOLVER_PROGRAM

        ns: dict = {"__name__": "resolver_under_test"}
        exec(compile(_RESOLVER_PROGRAM, "<resolver>", "exec"), ns)  # noqa: S102
        text = (
            '[project]\nname = "x"\ndependencies = [\n  "requests[socks]>=2",\n'
            "  'httpx',\n]\n[project.optional-dependencies]\ndev = [\"pytest\"]\n"
            '[tool.poetry.dependencies]\npython = "^3.11"\nrich = "*"\n'
        )

        got = ns["toml_fallback"](text)

        assert {"requests[socks]>=2", "httpx", "pytest", "rich"} <= set(got)
        assert "x" not in got

    # -- interpreter: the venv's own Python, or an honest fallback ---------------

    def _foreign_venv(self, tmp_path):
        """``<root>/.venv`` whose interpreter is a real Python of ANOTHER minor
        version, holding a compiled extension tagged for that version only."""
        other = _other_python()
        if other is None:
            pytest.skip("only one Python minor version is installed on this machine")
        python, version = other
        sp = tmp_path / ".venv" / "lib" / f"python{version}" / "site-packages"
        _write(sp / f"fakeext{_ext_suffix(python)}", "not really a binary\n")
        (tmp_path / ".venv" / "bin").mkdir(parents=True)
        (tmp_path / ".venv" / "bin" / "python").symlink_to(python)
        return python, version

    def test_reexecs_under_the_venv_interpreter_of_another_version(self, tmp_path):
        python, version = self._foreign_venv(tmp_path)

        payload = _resolver_payload(tmp_path, "fakeext", "ghost_mod_zz")

        assert payload["interpreter"] == "venv"
        assert payload["python_version"] == version
        assert os.path.realpath(payload["executable"]) == python
        assert _statuses(payload) == {
            "fakeext": ("installed", None),
            "ghost_mod_zz": ("unresolved", None),
        }

    def test_no_reexec_guard_stays_put_and_never_false_fails(self, tmp_path):
        """The guard the re-exec passes to itself: with it, the program never
        execs again, and a foreign-ABI extension it cannot see is unverifiable."""
        self._foreign_venv(tmp_path)

        payload = _resolver_payload(tmp_path, "fakeext", "json", flags=["--no-reexec"])

        assert payload["interpreter"] == "system-fallback"
        assert payload["python_version"] == f"{sys.version_info[0]}.{sys.version_info[1]}"
        assert _statuses(payload) == {
            "fakeext": ("unverifiable", None),
            "json": ("stdlib", None),
        }

    def test_unrunnable_venv_interpreter_falls_back_to_unverifiable(self, tmp_path):
        """execv fails (no shebang, not a binary): continue here, never fail."""
        _site_packages(tmp_path)
        _write(tmp_path / ".venv" / "bin" / "python", "not an interpreter\n")
        (tmp_path / ".venv" / "bin" / "python").chmod(0o755)

        payload = _resolver_payload(tmp_path, "ghost_mod_zz", "json:ghost_name_zz")

        assert payload["interpreter"] == "system-fallback"
        assert _statuses(payload) == {
            "ghost_mod_zz": ("unverifiable", None),
            "json:ghost_name_zz": ("unverifiable", None),
        }

    def test_venv_interpreter_that_is_the_running_one_is_not_reexecuted(self, tmp_path):
        _site_packages(tmp_path)
        (tmp_path / ".venv" / "bin").mkdir(parents=True)
        (tmp_path / ".venv" / "bin" / "python3").symlink_to(sys.executable)

        payload = _resolver_payload(tmp_path, "ghost_mod_zz")

        assert payload["interpreter"] == "venv"
        assert _statuses(payload) == {"ghost_mod_zz": ("unresolved", None)}

    def test_no_venv_interpreter_reports_system(self, tmp_path):
        _site_packages(tmp_path)

        payload = _resolver_payload(tmp_path, "ghost_mod_zz")

        assert payload["interpreter"] == "system"
        assert payload["site_packages"] == "venv"
        assert _statuses(payload) == {"ghost_mod_zz": ("unresolved", None)}

    def test_foreign_abi_extension_without_venv_interpreter_is_unverifiable(self, tmp_path):
        """A venv with site-packages but no bin/python: an extension built for
        another Python is invisible to this finder, so it must read unverifiable,
        never unresolved; a true ghost next to it stays unresolved."""
        sp = _site_packages(tmp_path)
        _write(sp / "fakeext.cpython-312-x86_64-linux-gnu.so", "")
        _write(sp / "pkgx" / "__init__.py", "")
        _write(sp / "pkgx" / "_speed.cpython-312-darwin.so", "")

        payload = _resolver_payload(
            tmp_path, "fakeext", "pkgx._speed", "pkgx:_speed", "ghost_mod_zz"
        )

        assert payload["interpreter"] == "system"
        got = _statuses(payload)
        assert got["fakeext"] == ("unverifiable", None)
        assert got["pkgx._speed"] == ("unverifiable", None)
        assert got["pkgx:_speed"] == ("unverifiable", None)
        assert got["ghost_mod_zz"] == ("unresolved", None)

    def test_no_venv_at_all_reports_system_and_ghost_is_unresolved(self, tmp_path):
        payload = _resolver_payload(tmp_path, "ghost_mod_zz")

        assert payload["interpreter"] == "system"
        assert _statuses(payload) == {"ghost_mod_zz": ("unresolved", None)}

    # -- search order --------------------------------------------------------------

    def test_src_dir_comes_after_the_stdlib(self, tmp_path):
        """src/ (editable / src layout) sits after the stdlib on sys.path, so a
        repo src/email package cannot shadow the stdlib's; the root still does."""
        _site_packages(tmp_path)
        _write(tmp_path / "src" / "email" / "__init__.py", "x = 1\n")
        _write(tmp_path / "src" / "calendar" / "__init__.py", "x = 1\n")
        _write(tmp_path / "src" / "myapp" / "__init__.py", "VALUE = 1\n")
        _write(tmp_path / "logging" / "__init__.py", "y = 1\n")

        got = _run_resolver(
            tmp_path,
            "email.mime.text:MIMEText",
            "calendar:monthrange",
            "myapp:VALUE",
            "logging:y",
        )

        assert got == {
            "email.mime.text:MIMEText": ("stdlib", "found"),
            "calendar:monthrange": ("stdlib", "found"),
            "myapp:VALUE": ("repo", "found"),
            "logging:y": ("repo", "found"),
        }


class TestCollectImportsGuards:
    def test_type_checking_and_suppress_blocks_are_guarded(self):
        from src.routers.validation import _static_analysis

        code = (
            "import contextlib, typing\n"
            "from contextlib import suppress\n"
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    import ghost_a\n"
            "else:\n"
            "    import ghost_b\n"
            "if typing.TYPE_CHECKING:\n"
            "    from ghost_c import X\n"
            "with contextlib.suppress(ImportError):\n"
            "    import ghost_d\n"
            "with suppress(ValueError, ModuleNotFoundError):\n"
            "    import ghost_e\n"
            "with suppress(ValueError):\n"
            "    import ghost_f\n"
            "with open('x') as fh:\n"
            "    import ghost_g\n"
        )
        result = _static_analysis(code)
        guarded = {i.module: i.guarded for i in result.imports}
        assert guarded == {
            "contextlib": False,
            "typing": False,
            "ghost_a": True,
            "ghost_b": False,
            "ghost_c": True,
            "ghost_d": True,
            "ghost_e": True,
            "ghost_f": False,
            "ghost_g": False,
        }

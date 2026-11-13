"""py2rust A1: the optional workspace ``toolchain`` (python | python+rust).

Covers the field, the image map, the resource profile, the Deployment stamp and
its read-back, the toolchain-gated exec policy, and the create-path reuse rules.
The python byte-identity half of G1 lives in test_toolchain_python_spec_snapshot.py.
"""

from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from kubernetes.client.rest import ApiException
from pydantic import ValidationError

from src.core import toolchain as tc
from src.models.container_models import ContainerInfo, ContainerStatus, GitHubRepoRequest
from src.runtime import ExecutionProfile
from src.runtime.policy import (
    ANY_TOOLCHAIN,
    RUST_TOOLCHAIN_COMMANDS,
    CommandNotPermittedError,
    enforce_command_policy,
    validate_command,
)
from tests.unit.test_docker_service import _build_service, _make_info

_REAL_MOD_NAME = "src.core.kubernetes_container_provisioner"
_existing = sys.modules.get(_REAL_MOD_NAME)
if _existing is not None and hasattr(_existing, "K8sContainerProvisioner"):
    kcp = _existing
else:
    sys.modules.pop(_REAL_MOD_NAME, None)
    with patch("kubernetes.config.load_kube_config"):
        kcp = importlib.import_module(_REAL_MOD_NAME)

RUST_IMAGE = "ghcr.io/nonagenticai/codegraph-runtime-rust:sha-abc@sha256:" + "1" * 64
PY_IMAGE = "ghcr.io/nonagenticai/codegraph-runtime:sha-def@sha256:" + "2" * 64


def _prov(monkeypatch, images_json: str | None = None):
    monkeypatch.setenv("PROVISIONER_IMAGE", PY_IMAGE)
    if images_json is None:
        monkeypatch.delenv("PROVISIONER_IMAGES_JSON", raising=False)
    else:
        monkeypatch.setenv("PROVISIONER_IMAGES_JSON", images_json)
    with patch("kubernetes.config.load_kube_config"):
        prov = kcp.K8sContainerProvisioner(namespace="coding-jobs", in_cluster=False)
    prov.apps_v1 = MagicMock()
    prov.core_v1 = MagicMock()
    return prov


def _create(prov, **kw):
    prov._create_deployment(
        name="ws",
        repo_url="https://github.com/example-org/target-rs",
        branch="main",
        github_token="ghs_x",
        tenant_id="tenant-a",
        **kw,
    )
    return prov.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]


def _main(body):
    return body.spec.template.spec.containers[0]


# --------------------------------------------------------------------------- model


class TestField:
    def test_create_request_defaults_to_python(self):
        req = GitHubRepoRequest(repo_url="https://github.com/o/r")
        assert req.toolchain == "python"

    def test_create_request_accepts_python_rust(self):
        req = GitHubRepoRequest(repo_url="https://github.com/o/r", toolchain="python+rust")
        assert req.toolchain == "python+rust"

    @pytest.mark.parametrize("bad", ["rust", "Python", "python+go", ""])
    def test_create_request_rejects_unknown_toolchains(self, bad):
        with pytest.raises(ValidationError):
            GitHubRepoRequest(repo_url="https://github.com/o/r", toolchain=bad)

    def test_container_info_defaults_to_python_for_old_registry_rows(self):
        info = _make_info("old")
        assert info.toolchain == "python"
        # A row persisted before the field existed round-trips as python.
        dumped = info.model_dump()
        dumped.pop("toolchain")
        assert ContainerInfo(**dumped).toolchain == "python"

    def test_openapi_field_is_optional_with_python_default(self, app):
        schema = app.openapi()["components"]["schemas"]["GitHubRepoRequest"]
        assert "toolchain" not in schema.get("required", [])
        assert schema["properties"]["toolchain"]["default"] == "python"
        assert schema["properties"]["toolchain"]["enum"] == ["python", "python+rust"]


# --------------------------------------------------------------------------- mapping


class TestMapping:
    def test_normalize(self):
        assert tc.normalize_toolchain(None) == "python"
        assert tc.normalize_toolchain("") == "python"
        assert tc.normalize_toolchain("python+rust") == "python+rust"
        with pytest.raises(ValueError):
            tc.normalize_toolchain("rust")

    def test_toolchain_of_fails_narrow(self):
        assert tc.toolchain_of(SimpleNamespace(toolchain="python+rust")) == "python+rust"
        assert tc.toolchain_of(SimpleNamespace(toolchain="weird")) == "python"
        assert tc.toolchain_of(SimpleNamespace()) == "python"
        assert tc.toolchain_of(None) == "python"

    def test_python_image_never_comes_from_the_map(self, monkeypatch):
        monkeypatch.setenv("PROVISIONER_IMAGES_JSON", json.dumps({"python": "evil:1"}))
        assert tc.image_for("python", PY_IMAGE) == PY_IMAGE

    def test_rust_image_from_the_map(self, monkeypatch):
        monkeypatch.setenv("PROVISIONER_IMAGES_JSON", json.dumps({"python+rust": RUST_IMAGE}))
        assert tc.image_for("python+rust", PY_IMAGE) == RUST_IMAGE

    def test_rust_image_default_when_unmapped(self, monkeypatch):
        monkeypatch.delenv("PROVISIONER_IMAGES_JSON", raising=False)
        assert tc.image_for("python+rust", PY_IMAGE) == tc.DEFAULT_RUST_IMAGE

    @pytest.mark.parametrize("raw", ["{not json", "[1,2]", '{"python+rust": ""}'])
    def test_malformed_map_fails_loudly_for_rust(self, monkeypatch, raw):
        monkeypatch.setenv("PROVISIONER_IMAGES_JSON", raw)
        with pytest.raises(tc.ToolchainImageUnavailableError):
            tc.image_for("python+rust", PY_IMAGE)
        # ...and never affects python.
        assert tc.image_for("python", PY_IMAGE) == PY_IMAGE

    def test_resources(self):
        assert tc.resources_for("python") is None
        rust = tc.resources_for("python+rust")
        assert rust == tc.RUST_RESOURCES
        # Within the coding-jobs LimitRange max (cpu 4, memory 8Gi, storage 40Gi).
        assert rust["limits"]["cpu"] == "4"
        assert rust["limits"]["memory"] == "6Gi"
        rust["limits"]["cpu"] = "99"
        assert tc.RUST_RESOURCES["limits"]["cpu"] == "4", "must hand out a copy"


# --------------------------------------------------------------------------- provisioner


class TestProvisioner:
    def test_rust_deployment_gets_image_resources_and_annotation(self, monkeypatch):
        prov = _prov(monkeypatch, json.dumps({"python+rust": RUST_IMAGE}))
        body = _create(prov, toolchain="python+rust")
        main = _main(body)
        assert main.image == RUST_IMAGE
        assert main.resources.limits == tc.RUST_RESOURCES["limits"]
        assert main.resources.requests == tc.RUST_RESOURCES["requests"]
        assert body.metadata.annotations[tc.TOOLCHAIN_ANNOTATION] == "python+rust"
        # The clone init container is not toolchain-dependent.
        init = body.spec.template.spec.init_containers[0]
        assert init.image == "alpine/git:latest"

    def test_python_deployment_has_no_toolchain_annotation(self, monkeypatch):
        prov = _prov(monkeypatch, json.dumps({"python+rust": RUST_IMAGE}))
        body = _create(prov, toolchain="python")
        assert tc.TOOLCHAIN_ANNOTATION not in body.metadata.annotations
        assert _main(body).image == PY_IMAGE

    def test_reprovision_as_python_nulls_a_stale_toolchain_annotation(self, monkeypatch):
        prov = _prov(monkeypatch)
        prov.apps_v1.create_namespaced_deployment.side_effect = ApiException(status=409)
        incumbent = MagicMock()
        incumbent.metadata.deletion_timestamp = None
        incumbent.metadata.labels = {"tenant-id": "tenant-a"}
        incumbent.metadata.annotations = {
            kcp.TENANT_ANNOTATION: "tenant-a",
            tc.TOOLCHAIN_ANNOTATION: "python+rust",
        }
        prov.apps_v1.read_namespaced_deployment.return_value = incumbent
        _create(prov, toolchain="python")
        patched = prov.apps_v1.patch_namespaced_deployment.call_args.kwargs["body"]
        assert patched.metadata.annotations[tc.TOOLCHAIN_ANNOTATION] is None

    def test_reconstruct_reads_the_toolchain_back(self, monkeypatch):
        prov = _prov(monkeypatch)
        dep = MagicMock()
        dep.metadata.labels = {"managed-by": "code-graph-helper", "tenant-id": "tenant-a"}
        dep.metadata.annotations = {
            kcp.REPO_URL_ANNOTATION: "https://github.com/o/r",
            tc.TOOLCHAIN_ANNOTATION: "python+rust",
        }
        dep.status.ready_replicas = 1
        dep.spec.replicas = 1
        prov.apps_v1.read_namespaced_deployment.return_value = dep
        assert prov.reconstruct_container_info("ws").toolchain == "python+rust"
        dep.metadata.annotations = {kcp.REPO_URL_ANNOTATION: "https://github.com/o/r"}
        assert prov.reconstruct_container_info("ws").toolchain == "python"

    async def test_provision_container_threads_toolchain_into_the_request(self, monkeypatch):
        prov = _prov(monkeypatch)
        prov.provision_from_github = AsyncMock(return_value="ok")
        await prov.provision_container(
            container_name="ws", repo_url="https://github.com/o/r", toolchain="python+rust"
        )
        assert prov.provision_from_github.await_args.args[0].toolchain == "python+rust"
        await prov.provision_container(container_name="ws", repo_url="https://github.com/o/r")
        assert prov.provision_from_github.await_args.args[0].toolchain == "python"


# --------------------------------------------------------------------------- policy


RUST_ONLY = sorted(RUST_TOOLCHAIN_COMMANDS)

#: Every argv the validator showed past the old "loopback-only psql" check, plus
#: the plain forms. Raw psql is refused outright in every toolchain now; SQL goes
#: through `ws-pg psql` (see scripts/ws-pg and tests/unit/test_ws_pg_script.py).
PSQL_BYPASSES = [
    ["psql", "-c", "select 1"],
    ["psql", "-h", "127.0.0.1", "-c", "select 1"],
    ["psql", "-Xh", "evil"],
    ["psql", "-Xhevil"],
    ["psql", "--ho=evil"],
    ["psql", "postgresql://u@127.0.0.1/x?hostaddr=1.2.3.4"],
    ["psql", "-d", "service=evil"],
    ["psql", "-c", "\\connect postgresql://evil/x"],
    ["psql", "-c", "\\! id"],
    ["psql", "-h", "db.example.test"],
    ["/usr/lib/postgresql/18/bin/psql", "-c", "select 1"],
]


class TestPolicy:
    @pytest.mark.parametrize("cmd", [*RUST_ONLY, "psql"])
    def test_rust_commands_refused_in_python_workspaces(self, cmd):
        assert validate_command([cmd, "--version"]) is False
        assert validate_command([cmd, "--version"], toolchain="python") is False

    @pytest.mark.parametrize("cmd", RUST_ONLY)
    def test_rust_commands_allowed_in_python_rust_workspaces(self, cmd):
        assert validate_command([cmd, "--version"], toolchain="python+rust") is True
        assert validate_command([f"/usr/local/cargo/bin/{cmd}"], toolchain="python+rust")

    def test_cargo_stays_allowed_everywhere(self):
        # Already on the base list before toolchains existed — G1 keeps it.
        assert validate_command(["cargo", "build"]) is True
        assert validate_command(["cargo", "nextest", "run"], toolchain="python+rust") is True

    def test_base_list_is_unchanged_for_python(self):
        assert validate_command(["uv", "sync"]) is True
        assert validate_command(["sh", "-c", "x"]) is False
        assert validate_command(["sh", "-c", "x"], toolchain="python+rust") is False

    def test_psql_is_not_a_toolchain_command(self):
        assert "psql" not in RUST_TOOLCHAIN_COMMANDS
        assert "ws-pg" in RUST_TOOLCHAIN_COMMANDS

    @pytest.mark.parametrize("argv", PSQL_BYPASSES)
    @pytest.mark.parametrize("toolchain", ["python", "python+rust", ANY_TOOLCHAIN])
    def test_raw_psql_is_refused_in_every_toolchain(self, argv, toolchain):
        assert validate_command(argv, toolchain=toolchain) is False
        with pytest.raises(CommandNotPermittedError):
            enforce_command_policy(argv, ExecutionProfile.CALLER_SUPPLIED, toolchain=toolchain)

    def test_ws_pg_psql_is_admitted_only_with_rust(self):
        argv = ["ws-pg", "psql", "ws", "-c", "select 1"]
        assert validate_command(argv, toolchain="python+rust") is True
        assert validate_command(argv, toolchain="python") is False

    def test_any_toolchain_admits_the_union(self):
        assert validate_command(["rustc", "-V"], toolchain=ANY_TOOLCHAIN) is True
        assert validate_command(["sh"], toolchain=ANY_TOOLCHAIN) is False
        assert validate_command(["psql", "-h", "evil"], toolchain=ANY_TOOLCHAIN) is False

    def test_enforce_uses_the_toolchain(self):
        with pytest.raises(CommandNotPermittedError):
            enforce_command_policy(["rustc", "-V"], ExecutionProfile.CALLER_SUPPLIED)
        enforce_command_policy(
            ["rustc", "-V"], ExecutionProfile.CALLER_SUPPLIED, toolchain="python+rust"
        )
        # Server-authored argv stays exempt regardless.
        enforce_command_policy(["sh", "-c", "x"], ExecutionProfile.SERVER_AUTHORED)

    async def test_kubernetes_runtime_enforces_the_workspace_toolchain(self):
        from src.runtime.kubernetes import KubernetesRuntime

        provisioner = MagicMock()
        provisioner.exec_in_pod.return_value = {"exit_code": 0, "stdout": "ok", "stderr": ""}
        runtime = KubernetesRuntime(provisioner)
        py_ws = SimpleNamespace(container_name="ws", toolchain="python")
        rs_ws = SimpleNamespace(container_name="ws", toolchain="python+rust")
        with pytest.raises(CommandNotPermittedError):
            await runtime.run_once(
                py_ws, ["ws-pg", "start"], execution_profile=ExecutionProfile.CALLER_SUPPLIED
            )
        provisioner.exec_in_pod.assert_not_called()
        await runtime.run_once(
            rs_ws, ["ws-pg", "start"], execution_profile=ExecutionProfile.CALLER_SUPPLIED
        )
        provisioner.exec_in_pod.assert_called_once()


# --------------------------------------------------------------------------- async exec


class TestAsyncExecRoute:
    @pytest.fixture
    def setup(self, app):
        from coding_client.auth.tenant import require_tenant
        from src.routers import exec_async

        app.dependency_overrides[require_tenant] = lambda: "tenant-a"
        store = MagicMock()
        store.create_job = AsyncMock(
            side_effect=lambda **kw: ({"job_id": kw["job_id"], **kw}, True)
        )
        runner = SimpleNamespace(store=store, owner="pod:1:x", lease_seconds=60, submit=MagicMock())
        manager = MagicMock()
        toolchains = {"py-ws": "python", "rs-ws": "python+rust"}
        manager.get_container.side_effect = lambda name, tenant_id=None: (
            SimpleNamespace(container_name=name, toolchain=toolchains[name])
            if name in toolchains
            else None
        )
        with (
            patch.object(exec_async, "get_exec_job_runner", return_value=runner),
            patch.object(exec_async, "get_container_manager", return_value=manager),
        ):
            yield runner
        app.dependency_overrides.pop(require_tenant, None)

    async def test_rust_command_in_python_workspace_is_400(self, client, setup):
        resp = await client.post(
            "/api/workspaces/py-ws/exec/async", json={"command": ["cargo-nextest", "run"]}
        )
        assert resp.status_code == 400
        setup.store.create_job.assert_not_awaited()

    async def test_rust_command_in_rust_workspace_is_accepted(self, client, setup):
        resp = await client.post(
            "/api/workspaces/rs-ws/exec/async", json={"command": ["cargo-nextest", "run"]}
        )
        assert resp.status_code == 202
        setup.submit.assert_called_once()


# --------------------------------------------------------------------------- detached launch


class TestDetachedLaunch:
    async def test_launch_uses_the_toolchain_located_for_that_pod(self):
        from src.services.exec_jobs.detached import DetachedHandle, K8sDetachedBackend

        prov = MagicMock()
        prov.resolve_running_pod.return_value = "pod-1"
        prov.exec_in_pod.return_value = {"exit_code": 0, "stdout": "launched", "stderr": ""}
        manager = MagicMock()
        manager.get_container.return_value = SimpleNamespace(
            container_name="ws", toolchain="python+rust"
        )
        ex = K8sDetachedBackend(provisioner=prov, container_manager=manager)
        with patch("src.runtime.resolve_runtime_kind") as kind:
            from src.runtime import RuntimeKind

            kind.return_value = RuntimeKind.KUBERNETES
            assert await ex.locate("ws", "tenant-a") == "pod-1"
        await ex.launch(
            DetachedHandle("ws", "pod-1", "/tmp/j"), ["ws-pg", "start"], cwd="/", timeout=5, cap=10
        )
        # Not remembered for an unknown pod (adoption) → python → refused.
        with pytest.raises(CommandNotPermittedError):
            await ex.launch(
                DetachedHandle("ws", "pod-2", "/tmp/j"),
                ["ws-pg", "start"],
                cwd="/",
                timeout=5,
                cap=10,
            )


# --------------------------------------------------------------------------- create path


def _k8s(fresh):
    k8s = MagicMock()
    k8s.is_available.return_value = True
    k8s.provision_container = AsyncMock(return_value=fresh)
    return k8s


class TestCreatePath:
    async def test_toolchain_is_threaded_to_k8s_provision(self):
        svc = _build_service(docker_available=True, registry_data={})
        svc.find_container_by_name = AsyncMock(return_value=None)
        svc.find_existing_container = AsyncMock(return_value=None)
        k8s = _k8s(_make_info("rs-1", toolchain="python+rust"))
        req = GitHubRepoRequest(
            repo_url="https://github.com/o/r", container_name="rs-1", toolchain="python+rust"
        )
        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            return_value=k8s,
        ):
            await svc.create_container(req, tenant_id=None)
        assert k8s.provision_container.await_args.kwargs["toolchain"] == "python+rust"

    async def test_python_rust_refused_on_docker_fallback(self):
        svc = _build_service(docker_available=True, registry_data={})
        svc.find_container_by_name = AsyncMock(return_value=None)
        svc.find_existing_container = AsyncMock(return_value=None)
        svc.lifecycle_manager.create_container = AsyncMock(side_effect=AssertionError("no"))
        req = GitHubRepoRequest(repo_url="https://github.com/o/r", toolchain="python+rust")
        with (
            patch(
                "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
                return_value=None,
            ),
            pytest.raises(HTTPException) as exc,
        ):
            await svc.create_container(req, tenant_id=None)
        assert exc.value.status_code == 400
        assert "python+rust" in exc.value.detail

    async def test_named_python_rust_request_will_not_reuse_a_python_workspace(self):
        existing = _make_info("ws-1")
        svc = _build_service(registry_data={"ws-1": existing})
        svc.find_container_by_name = AsyncMock(return_value=existing)
        svc.delete_container = AsyncMock(side_effect=AssertionError("must not tear down"))
        req = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo", container_name="ws-1", toolchain="python+rust"
        )
        with pytest.raises(HTTPException) as exc:
            await svc.create_container(req, tenant_id=None)
        assert exc.value.status_code == 409
        assert "toolchain refused" in exc.value.detail

    async def test_named_python_request_still_reuses_a_rust_workspace(self):
        existing = _make_info("ws-1", toolchain="python+rust", status=ContainerStatus.RUNNING)
        svc = _build_service(registry_data={"ws-1": existing})
        svc.find_container_by_name = AsyncMock(return_value=existing)
        req = GitHubRepoRequest(repo_url="https://github.com/owner/repo", container_name="ws-1")
        assert (await svc.create_container(req, tenant_id=None)) is existing

    async def test_repo_dedupe_skips_a_python_workspace_for_a_rust_request(self):
        existing = _make_info("py-1")
        svc = _build_service(docker_available=True, registry_data={"py-1": existing})
        svc.find_existing_container = AsyncMock(return_value=existing)
        fresh = _make_info("rs-new", toolchain="python+rust")
        k8s = _k8s(fresh)
        req = GitHubRepoRequest(repo_url="https://github.com/owner/repo", toolchain="python+rust")
        with patch(
            "src.core.kubernetes_container_provisioner.get_k8s_container_provisioner",
            return_value=k8s,
        ):
            result = await svc.create_container(req, tenant_id=None)
        assert result is fresh
        k8s.provision_container.assert_awaited_once()

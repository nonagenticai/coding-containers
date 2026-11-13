"""py2rust gap #4 full: a workspace sized to RUN N source services + their dependencies.

* ``memory_profile``: no hint ⇒ ``None`` (G1: the pod keeps its toolchain resources);
  with a hint, limit = tooling reserve + dependencies + N x hint, never below the
  toolchain default, refused (``MemoryBudgetError``) above the coding-jobs LimitRange
  max (8Gi per container, gitops origin/main) — never silently shrunk.
* The Deployment carries the sized memory/CPU + an annotation read back on registry miss.
* The create route maps ``MemoryBudgetError`` to 409 WORKSPACE_MEMORY_BUDGET_EXCEEDED
  (route test with a stub service at the end).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.core import toolchain as tc
from src.models.container_models import GitHubRepoRequest
from tests.unit.test_toolchain import _create, _main, _prov


def test_the_service_ceiling_is_derived_from_the_limitrange():
    # 8Gi max less the python tooling reserve, at the smallest per-service hint.
    assert tc.MEMORY_MAX_MI == 8192 and tc.CPU_MAX_M == 4000
    assert tc.MAX_EXPECTED_SERVICES == (8192 - 1024) // 128 == 56
    # An unsized workspace keeps origin/main's 4; a sized one gets the hint bound.
    assert tc.MAX_SERVICES_PER_WORKSPACE == 4
    assert tc.max_services_of(SimpleNamespace(memory_limit_mi=None)) == 4
    assert tc.max_services_of(SimpleNamespace(memory_limit_mi=6656)) == 56
    # An unsized python workspace keeps the old 4 services at the default hint.
    budget = tc.DEFAULT_LIMIT_MI["python"] - tc.TOOLING_RESERVE_MI["python"]
    assert budget // tc.SERVICE_MEMORY_DEFAULT_MI == 4


@pytest.mark.parametrize(
    ("toolchain", "n", "hint", "deps", "expected"),
    [
        ("python", None, None, None, None),
        ("python+rust", None, None, None, None),
        # 1024 + 4x256 = 2048 = the python default: not raised, only stamped.
        ("python", 4, None, None, (512, 2048, 1500)),
        # 1024 + 320 + 20x256 = 6464 -> rounded up to 6656.
        ("python", 20, None, ["redis"], (1536, 6656, 4000)),
        # rust: never below its 6Gi default.
        ("python+rust", 2, None, ["redis"], (1152, 6144, 4000)),
        # 4096 + 320 + 512 + 6x512 = 8000 -> 8192 (exactly the max fits).
        ("python+rust", 6, 512, ["redis", "postgres"], (1408, 8192, 4000)),
        # dependencies alone are a hint too.
        ("python", None, None, ["redis"], (256, 2048, 750)),
    ],
)
def test_memory_profile(toolchain, n, hint, deps, expected):
    profile = tc.memory_profile(toolchain, n, hint, deps)
    if expected is None:
        assert profile is None
        return
    assert (profile.request_mi, profile.limit_mi, profile.cpu_limit_m) == expected
    assert profile.limit_mi <= tc.MEMORY_MAX_MI and profile.request_mi <= profile.limit_mi
    assert profile.limit_mi >= profile.need_mi


def test_a_set_that_cannot_fit_one_pod_is_refused_naming_the_budget():
    # 4096 + 320 + 512 + 7x512 = 8512 > 8192: refused, not shrunk to fit.
    with pytest.raises(tc.MemoryBudgetError) as exc:
        tc.memory_profile("python+rust", 7, 512, ["redis", "postgres"])
    e = exc.value
    assert (e.need_mi, e.max_mi, e.reserve_mi) == (8512, 8192, 4096)
    assert "8512 MiB" in str(e) and "8192 MiB" in str(e)
    # Negative control: one service fewer fits exactly.
    assert tc.memory_profile("python+rust", 6, 512, ["redis", "postgres"]).limit_mi == 8192


def test_python_without_a_hint_keeps_its_resources(monkeypatch):
    body = _create(_prov(monkeypatch))
    res = _main(body).resources
    assert res.limits["memory"] == "2Gi" and res.limits["cpu"] == "500m"
    assert tc.MEMORY_LIMIT_ANNOTATION not in body.metadata.annotations


def test_a_sized_workspace_carries_the_profile(monkeypatch):
    profile = tc.memory_profile("python", 20, None, ["redis"])
    body = _create(_prov(monkeypatch), memory=profile)
    res = _main(body).resources
    assert res.limits["memory"] == "6656Mi" and res.requests["memory"] == "1536Mi"
    assert res.limits["cpu"] == "4000m"
    assert body.metadata.annotations[tc.MEMORY_LIMIT_ANNOTATION] == "6656"
    # The shared rust profile is never mutated by sizing one workspace.
    rust = _create(
        _prov(monkeypatch),
        toolchain="python+rust",
        memory=tc.memory_profile("python+rust", 6, 512, ["redis", "postgres"]),
    )
    assert _main(rust).resources.limits["memory"] == "8192Mi"
    assert tc.RUST_RESOURCES["limits"]["memory"] == "6Gi"


def test_request_field_bounds():
    base = {"repo_url": "https://github.com/o/r"}
    req = GitHubRepoRequest(**base)
    assert (req.expected_services, req.service_memory_mi, req.expected_dependencies) == (
        None,
        None,
        None,
    )
    for bad in (
        {"expected_services": -1},
        {"expected_services": tc.MAX_EXPECTED_SERVICES + 1},
        {"service_memory_mi": 127},
        {"service_memory_mi": 4097},
        {"expected_dependencies": ["mysql"]},
        # ws-pg only exists in the python+rust image.
        {"expected_dependencies": ["postgres"]},
    ):
        with pytest.raises(ValidationError):
            GitHubRepoRequest(**base, **bad)
    ok = GitHubRepoRequest(**base, toolchain="python+rust", expected_dependencies=["postgres"])
    assert ok.expected_dependencies == ["postgres"]


def test_openapi_exposes_the_memory_fields(app):
    app.openapi_schema = None
    spec = app.openapi()
    props = spec["components"]["schemas"]["GitHubRepoRequest"]["properties"]
    assert props["expected_services"]["x-n8n-param-name"] == "expectedServices"
    assert props["service_memory_mi"]["x-n8n-param-name"] == "serviceMemoryMi"
    assert props["expected_dependencies"]["x-n8n-param-name"] == "expectedDependencies"
    assert "memory_limit_mi" in spec["components"]["schemas"]["ContainerInfo"]["properties"]


async def test_create_route_maps_a_set_that_cannot_fit_to_409(app, client):
    """The create route turns MemoryBudgetError into 409 WORKSPACE_MEMORY_BUDGET_EXCEEDED."""
    from unittest.mock import patch

    from coding_client.auth.tenant import get_current_user_with_tenant

    seen: list[GitHubRepoRequest] = []

    class _Service:
        async def create_container(self, request, tenant_id=None):
            seen.append(request)
            tc.memory_profile(
                request.toolchain,
                request.expected_services,
                request.service_memory_mi,
                request.expected_dependencies,
            )
            raise AssertionError("an over-budget set must not be provisioned")

    app.dependency_overrides[get_current_user_with_tenant] = lambda: SimpleNamespace(
        effective_tenant_id="tenant-a"
    )
    try:
        with patch("src.services.docker_service.get_docker_service", return_value=_Service()):
            resp = await client.post(
                "/api/workspaces/",
                json={
                    "repo_url": "https://github.com/o/r",
                    "expected_services": 56,
                    "service_memory_mi": 4096,
                },
            )
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 409, resp.text
    assert "WORKSPACE_MEMORY_BUDGET_EXCEEDED" in resp.text
    assert seen and seen[0].expected_services == 56

"""py2rust gap #4-lite: workspace disk scales with the repositories it will hold.

* ``disk_profile``: python without a hint is UNSIZED (G1: its pod spec stays
  byte-identical, pinned by test_toolchain_python_spec_snapshot.py); python+rust is
  always sized; ``expected_repos`` adds 2Gi each, capped at the coding-jobs
  LimitRange max (40Gi ephemeral-storage per container, gitops origin/main).
* The Deployment gets the ephemeral request/limit, a ``repo-data`` sizeLimit and an
  annotation read back on a registry miss.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core import toolchain as tc
from src.models.container_models import GitHubRepoRequest
from tests.unit.test_toolchain import _create, _main, _prov


def _volume(body, name):
    return next(v for v in body.spec.template.spec.volumes if v.name == name)


def _env(body):
    return {e.name: e.value for e in _main(body).env}


@pytest.mark.parametrize(
    ("toolchain", "n", "expected"),
    [
        ("python", None, None),
        ("python", 0, (2, 10, 8)),
        ("python", 3, (5, 16, 14)),
        ("python", 15, (17, 40, 38)),
        ("python+rust", None, (4, 30, 28)),
        ("python+rust", 0, (4, 30, 28)),
        ("python+rust", 3, (7, 36, 34)),
        ("python+rust", 15, (19, 40, 38)),  # capped at the LimitRange max
    ],
)
def test_disk_profile(toolchain, n, expected):
    profile = tc.disk_profile(toolchain, n)
    if expected is None:
        assert profile is None
        return
    assert (profile.request_gi, profile.limit_gi, profile.repo_data_gi) == expected
    assert profile.limit_gi <= tc.EPHEMERAL_MAX_GI
    assert profile.request_gi <= profile.limit_gi


def test_python_without_a_hint_adds_nothing_to_the_spec(monkeypatch):
    body = _create(_prov(monkeypatch))
    assert _volume(body, "repo-data").empty_dir.size_limit is None
    assert "ephemeral-storage" not in (_main(body).resources.limits or {})
    assert tc.DISK_BUDGET_ANNOTATION not in body.metadata.annotations


def test_python_with_a_hint_is_sized(monkeypatch):
    body = _create(_prov(monkeypatch), disk=tc.disk_profile("python", 3))
    main = _main(body)
    assert main.resources.requests["ephemeral-storage"] == "5Gi"
    assert main.resources.limits["ephemeral-storage"] == "16Gi"
    # memory/cpu are untouched by the disk sizing
    assert main.resources.limits["memory"] == "2Gi"
    assert _volume(body, "repo-data").empty_dir.size_limit == "14Gi"
    assert _volume(body, "tmp").empty_dir.size_limit is None
    assert body.metadata.annotations[tc.DISK_BUDGET_ANNOTATION] == "14"


def test_rust_default_keeps_its_30gi_and_gains_a_size_limit(monkeypatch):
    body = _create(
        _prov(monkeypatch), toolchain="python+rust", disk=tc.disk_profile("python+rust", None)
    )
    main = _main(body)
    assert main.resources.limits["ephemeral-storage"] == "30Gi"
    assert main.resources.requests["ephemeral-storage"] == "4Gi"
    assert main.resources.limits["memory"] == tc.RUST_RESOURCES["limits"]["memory"]
    assert _volume(body, "repo-data").empty_dir.size_limit == "28Gi"
    # The shared profile is never mutated by sizing one workspace.
    assert tc.RUST_RESOURCES["limits"]["ephemeral-storage"] == "30Gi"


def test_request_field_bounds():
    base = {"repo_url": "https://github.com/o/r"}
    assert GitHubRepoRequest(**base).expected_repos is None
    assert GitHubRepoRequest(**base, expected_repos=0).expected_repos == 0
    assert GitHubRepoRequest(**base, expected_repos=tc.MAX_EXPECTED_REPOS).expected_repos == 15
    for bad in (-1, tc.MAX_EXPECTED_REPOS + 1):
        with pytest.raises(ValidationError):
            GitHubRepoRequest(**base, expected_repos=bad)


def test_openapi_create_workspace_exposes_expected_repos(app):
    app.openapi_schema = None
    spec = app.openapi()
    props = spec["components"]["schemas"]["GitHubRepoRequest"]["properties"]
    assert props["expected_repos"]["x-n8n-param-name"] == "expectedRepos"
    info = spec["components"]["schemas"]["ContainerInfo"]["properties"]
    assert "disk_budget_gi" in info

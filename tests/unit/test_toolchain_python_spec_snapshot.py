"""G1 snapshot: a ``toolchain`` omitted / ``"python"`` workspace is byte-identical.

The fixture ``data/k8s_workspace_deployment_python.json`` was generated from
origin/main 3d03a9b — BEFORE the ``toolchain`` field existed — by serialising the
Deployment body ``_create_deployment`` hands to the Kubernetes API, with the
clock and environment pinned. Adding the ``python+rust`` toolchain must not move
a single byte of it: an existing python workspace (every workspace the conveyor
provisions today) gets exactly the pod it got before.

Regenerate ONLY for a deliberate change to the python pod spec:
``WRITE_TOOLCHAIN_SNAPSHOT=1 pytest tests/unit/test_toolchain_python_spec_snapshot.py``
— and say why in the PR, because this file is the gate that proves additivity.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from kubernetes import client

_REAL_MOD_NAME = "src.core.kubernetes_container_provisioner"
_existing = sys.modules.get(_REAL_MOD_NAME)
if _existing is not None and hasattr(_existing, "K8sContainerProvisioner"):
    _mod = _existing
else:
    sys.modules.pop(_REAL_MOD_NAME, None)
    with patch("kubernetes.config.load_kube_config"):
        _mod = importlib.import_module(_REAL_MOD_NAME)

FIXTURE = Path(__file__).parent / "data" / "k8s_workspace_deployment_python.json"

#: The environment that shapes the pod spec, pinned so the snapshot is not a
#: function of the developer's shell.
_PINNED_ENV = {
    "PROVISIONER_IMAGE": ("ghcr.io/nonagenticai/codegraph-runtime:sha-snapshot@sha256:" + "0" * 64),
    "PROVISIONER_NODE_ARCH": "amd64",
}


def _deployment_body(monkeypatch: pytest.MonkeyPatch, **extra) -> dict:
    for key, value in _PINNED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("PROVISIONER_IMAGES_JSON", raising=False)
    with patch("kubernetes.config.load_kube_config"):
        prov = _mod.K8sContainerProvisioner(namespace="coding-jobs", in_cluster=False)
    prov.apps_v1 = MagicMock()
    prov.core_v1 = MagicMock()
    with patch.object(_mod.time, "time", return_value=1767225600.0):
        prov._create_deployment(
            name="snapshot-ws",
            repo_url="https://github.com/example-org/py2rs-fixture-svc",
            branch="main",
            github_token="ghs_snapshot",
            tenant_id="0195f48b-d000-7000-8000-000000000001",
            secret_config_id="sec-snapshot",
            **extra,
        )
    body = prov.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]
    return client.ApiClient().sanitize_for_serialization(body)


def _render(body: dict) -> str:
    return json.dumps(body, indent=2, sort_keys=True) + "\n"


def test_python_workspace_spec_matches_the_pre_toolchain_snapshot(monkeypatch):
    rendered = _render(_deployment_body(monkeypatch))
    if os.environ.get("WRITE_TOOLCHAIN_SNAPSHOT") == "1":
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(rendered)
    assert rendered == FIXTURE.read_text(), (
        "the python workspace Deployment changed — G1 requires toolchain=python to stay "
        "byte-identical to the pre-toolchain spec"
    )


def test_explicit_python_toolchain_is_the_same_bytes_as_omitted(monkeypatch):
    rendered = _render(_deployment_body(monkeypatch, toolchain="python"))
    assert rendered == FIXTURE.read_text()

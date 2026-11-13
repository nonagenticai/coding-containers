"""K8sDetachedBackend against a fake provisioner: pod pinning, pod-gone typing,
the allowlist on launch, and the launch marker."""

from __future__ import annotations

from typing import Any

import pytest
from kubernetes.client.rest import ApiException

from src.runtime import CommandNotPermittedError
from src.services.exec_jobs import detached as dx

HANDLE = dx.DetachedHandle("ws-1", "ws-1-abc", dx.job_dir("0199-job"))


class FakeProv:
    def __init__(self, result: dict[str, Any], gone: bool = False) -> None:
        self.result = result
        self.gone = gone
        self.calls: list[dict[str, Any]] = []

    def exec_in_pod(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.result

    def pod_is_gone(self, pod: str) -> bool:
        return self.gone


async def test_every_exec_is_pinned_to_the_recorded_pod() -> None:
    prov = FakeProv({"exit_code": 0, "stdout": "running\n", "stderr": ""})
    status = await dx.K8sDetachedBackend(prov).poll(HANDLE)
    assert status.state is dx.PollState.RUNNING
    assert prov.calls[0]["pod_name"] == "ws-1-abc"
    assert prov.calls[0]["deployment_name"] == "ws-1"


async def test_exec_failure_on_a_gone_pod_is_typed() -> None:
    prov = FakeProv({"exit_code": -1, "stdout": "", "stderr": "pods not found"}, gone=True)
    with pytest.raises(dx.WorkspaceGoneError):
        await dx.K8sDetachedBackend(prov).poll(HANDLE)


async def test_exec_failure_on_a_live_pod_is_transient_not_gone() -> None:
    prov = FakeProv({"exit_code": -1, "stdout": "", "stderr": "stream reset"}, gone=False)
    with pytest.raises(RuntimeError) as info:
        await dx.K8sDetachedBackend(prov).poll(HANDLE)
    assert not isinstance(info.value, dx.WorkspaceGoneError)


async def test_launch_applies_the_caller_allowlist() -> None:
    prov = FakeProv({"exit_code": 0, "stdout": "launched\n", "stderr": ""})
    with pytest.raises(CommandNotPermittedError):
        await dx.K8sDetachedBackend(prov).launch(
            HANDLE, ["curl", "http://x"], cwd="/workspace", timeout=10, cap=100
        )
    assert prov.calls == []


async def test_launch_without_the_marker_is_a_failure() -> None:
    prov = FakeProv({"exit_code": 0, "stdout": "", "stderr": "mkdir: denied"})
    with pytest.raises(RuntimeError, match="detached launch failed"):
        await dx.K8sDetachedBackend(prov).launch(
            HANDLE, ["python", "-m", "pytest"], cwd="/workspace", timeout=10, cap=100
        )


def test_pod_is_gone_reads_404_and_deletion() -> None:
    from unittest.mock import MagicMock, patch

    with patch("kubernetes.config.load_kube_config"):
        from src.core.kubernetes_container_provisioner import K8sContainerProvisioner

        prov = K8sContainerProvisioner(namespace="ns", in_cluster=False)
    prov.core_v1 = MagicMock()
    prov.core_v1.read_namespaced_pod.side_effect = ApiException(status=404)
    assert prov.pod_is_gone("p") is True
    prov.core_v1.read_namespaced_pod.side_effect = ApiException(status=500)
    assert prov.pod_is_gone("p") is False  # an API error is not proof of absence
    pod = MagicMock()
    pod.metadata.deletion_timestamp = "2026-10-12T00:00:00Z"
    prov.core_v1.read_namespaced_pod.side_effect = None
    prov.core_v1.read_namespaced_pod.return_value = pod
    assert prov.pod_is_gone("p") is True
    pod.metadata.deletion_timestamp = None
    pod.status.phase = "Running"
    assert prov.pod_is_gone("p") is False

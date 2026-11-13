"""Workspace background services (P7 multi-repo probe): validation, pod program, routes.

Three layers:

* the server-side validation (port / env / argv / path / id) — pure functions;
* the REAL pod program run as a real local process: a provider is started detached,
  becomes ready, is probed, survives the launching process, and is stopped with its
  whole process group; failures come back with their log tail;
* the routes end-to-end, with ``execute_command`` running the generated argv as a
  local subprocess (so the route, the spec and the program are exercised together),
  plus tenant isolation and boundary refusals with a recording fake.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from coding_client.auth.tenant import require_tenant
from src.routers import workspace_services as ws_router
from src.services.workspace_services import pod_program
from src.services.workspace_services import service as svc

TENANT = "tenant-a"
WS = "ws-1"
SID = "svc-0123456789abcdef"
_SECRET = "s3cr3t-value-never-logged"  # noqa: S105  # pragma: allowlist secret


def _free_port() -> int:
    """An ephemeral port nothing listens on (never 8000: the OS picks >= 1024 here)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert port != 8000
    return port


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A reaped-later zombie still answers kill(0); ps tells them apart.
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(out.stdout.strip()) and not out.stdout.strip().startswith("Z")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("port", [8000, 80, 1023, 0, -1, 65536, True])
def test_bad_ports_are_refused(port):
    with pytest.raises(svc.ServiceSpecError):
        svc.validate_port(port)


@pytest.mark.parametrize("port", [1024, 18080, 65535])
def test_good_ports_pass(port):
    assert svc.validate_port(port) == port


@pytest.mark.parametrize(
    "env",
    [
        {"PATH": "/x"},
        {"LD_PRELOAD": "/x.so"},
        {"A-B": "1"},
        {"1A": "1"},
        {"": "1"},
        {"OK": "a\x00b"},
        {"OK": "x" * 4097},
        {f"K{i}": "v" for i in range(33)},
    ],
)
def test_bad_env_is_refused(env):
    with pytest.raises(svc.ServiceSpecError):
        svc.validate_env(env)


def test_good_env_passes():
    env = {"PORT": "18080", "DATABASE_URL": _SECRET, "log_level": "debug"}
    assert svc.validate_env(env) == env


@pytest.mark.parametrize(
    "argv",
    [[], ["bash", "-c", "x"], ["sh", "-c", "x"], ["curl", "x"], ["rm", "-rf", "/"], ["shutdown"]],
)
def test_argv_follows_the_exec_policy(argv):
    with pytest.raises(svc.ServiceSpecError):
        svc.validate_argv(argv)


def test_allowlisted_argv_passes():
    argv = ["uv", "run", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "18080"]
    assert svc.validate_argv(argv) == argv


def test_argv_uses_the_workspace_toolchain():
    # python+rust services may run ws-pg (and only ws-pg's SQL path, never raw psql);
    # a python workspace keeps the narrower policy (the default).
    assert svc.validate_argv(["ws-pg", "start"], toolchain="python+rust") == ["ws-pg", "start"]
    with pytest.raises(svc.ServiceSpecError):
        svc.validate_argv(["ws-pg", "start"])
    with pytest.raises(svc.ServiceSpecError):
        svc.validate_argv(["psql", "-h", "evil"], toolchain="python+rust")
    spec = svc.build_launch_spec(
        service_id="svc-0123456789abcdef",
        argv=["cargo-nextest", "run"],
        workdir="/workspace",
        port=18080,
        env=None,
        readiness_kind="tcp",
        readiness_path=None,
        ready_timeout=5,
        max_lifetime=60,
        toolchain="python+rust",
    )
    assert spec["argv"] == ["cargo-nextest", "run"]


@pytest.mark.parametrize("path", ["", "health", "/a b", "/a\r\nX: y", "http://evil/"])
def test_bad_http_paths_are_refused(path):
    with pytest.raises(svc.ServiceSpecError):
        svc.validate_http_path(path)


@pytest.mark.parametrize("sid", ["", "svc-", "svc-../../etc", "svc-0123456789ABCDEF", "x" * 20])
def test_bad_service_ids_are_refused(sid):
    with pytest.raises(svc.ServiceSpecError):
        svc.validate_service_id(sid)


def test_spec_travels_as_one_base64_word_and_hides_env_values():
    spec = svc.build_launch_spec(
        service_id=SID,
        argv=["python3", "-m", "http.server", "18080"],
        workdir="/workspace",
        port=18080,
        env={"API_KEY": _SECRET},
        readiness_kind="http",
        readiness_path="/health",
        ready_timeout=10,
        max_lifetime=60,
    )
    argv = svc.pod_program_argv("launch", spec)
    assert argv[:3] == ["python3", "-I", "-c"]
    assert argv[3] == svc.pod_program_source()
    assert argv[4] == "launch"
    assert json.loads(base64.b64decode(argv[5]))["env"] == {"API_KEY": _SECRET}
    assert not any(_SECRET in word for word in argv)


def test_oversized_spec_is_refused():
    with pytest.raises(svc.ServiceSpecError):
        svc.pod_program_argv("launch", {"blob": "x" * (svc.MAX_SPEC_BYTES + 1)})


def test_the_program_uses_the_standard_library_only():
    """It runs in the pod with ``-I``: nothing from ``src`` is importable there."""
    source = svc.pod_program_source()
    assert "from src" not in source and "import src" not in source


# ---------------------------------------------------------------------------
# /proc/net/tcp parsing (the loopback-only rule)
# ---------------------------------------------------------------------------

_TCP = (
    "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid\n"
    "   0: 0100007F:4E20 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000\n"
    "   1: 00000000:4E21 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000\n"
    "   2: 0100007F:4E22 0100007F:9999 01 00000000:00000000 00:00000000 00000000  1000\n"
)
_TCP6 = (
    "  sl  local_address                         remote_address                        st\n"
    "   0: 00000000000000000000000001000000:4E23 00000000000000000000000000000000:0000 0A\n"
    "   1: 00000000000000000000000000000000:4E24 00000000000000000000000000000000:0000 0A\n"
    "   2: 0000000000000000FFFF00000100007F:4E25 00000000000000000000000000000000:0000 0A\n"
)


@pytest.mark.parametrize(
    ("port", "expected"),
    [
        (0x4E20, [True]),  # 127.0.0.1
        (0x4E21, [False]),  # 0.0.0.0
        (0x4E22, []),  # not LISTEN
        (0x4E23, [True]),  # ::1
        (0x4E24, [False]),  # ::
        (0x4E25, [True]),  # ::ffff:127.0.0.1
    ],
)
def test_listen_addresses_classifies_loopback(port, expected):
    found = pod_program.listen_addresses(port, {"tcp": _TCP, "tcp6": _TCP6})
    assert [lb for _, lb in found] == expected


# ---------------------------------------------------------------------------
# The real pod program, run as a real local process
# ---------------------------------------------------------------------------


@pytest.fixture
def program(tmp_path):
    root = tmp_path / "svc"
    work = tmp_path / "repo"
    work.mkdir()
    (work / "health").write_text("ok\n")
    started: list[str] = []

    def run(action: str, spec: dict, timeout: float = 60) -> dict:
        argv = svc.pod_program_argv(action, spec)
        argv[0] = sys.executable
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        lines = proc.stdout.strip().splitlines()
        assert len(lines) == 1, (proc.stdout, proc.stderr)
        return json.loads(lines[0])

    def launch(argv: list[str], port: int, **kw) -> dict:
        sid = kw.pop("service_id", f"svc-{os.urandom(8).hex()}")
        spec = svc.build_launch_spec(
            service_id=sid,
            argv=argv,
            workdir=str(work),
            port=port,
            env=kw.pop("env", None),
            readiness_kind=kw.pop("kind", "http"),
            readiness_path=kw.pop("path", "/health"),
            ready_timeout=kw.pop("ready_timeout", 20),
            max_lifetime=kw.pop("max_lifetime", 120),
            root=str(root),
        )
        spec.update(kw)
        started.append(sid)
        return run("launch", spec)

    def ref(sid: str, **kw) -> dict:
        return {"root": str(root), "service_id": sid, "tail_bytes": 4096, "stop_grace": 2, **kw}

    yield SimpleNamespace(run=run, launch=launch, ref=ref, root=root, work=work)
    for sid in started:  # never leave a process behind
        run("stop", ref(sid))


def _http_server(port: int) -> list[str]:
    return ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1"]


def test_provider_starts_detached_is_probed_and_stopped(program):
    port = _free_port()
    out = program.launch(_http_server(port), port, service_id=SID, env={"API_KEY": _SECRET})
    assert out["result"] == "ready", out
    assert out["http_status"] == 200
    state = out["service"]
    assert state["state"] == "running" and state["port"] == port
    assert out["bind"] in ("loopback", "unverified")  # unverified only off Linux
    # The launching process has exited; the provider is still serving.
    assert _pid_alive(state["pid"]) and _pid_alive(state["supervisor_pid"])
    assert _SECRET not in json.dumps(out)

    probe = program.run(
        "probe",
        program.ref(
            SID,
            method="GET",
            path="/health",
            headers={"X-Probe": "1"},
            body=None,
            timeout=5,
            body_cap=1024,
        ),
    )
    assert probe["result"] == "ok" and probe["status"] == 200
    assert probe["body"] == "ok\n" and probe["body_truncated"] is False

    listed = program.run("list", {"root": str(program.root)})
    assert [s["service_id"] for s in listed["services"]] == [SID]

    stopped = program.run("stop", program.ref(SID))
    assert stopped["service"]["state"] == "stopped"
    assert "GET /health" in stopped["log_tail"]
    deadline = time.monotonic() + 5
    while _pid_alive(state["pid"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _pid_alive(state["pid"])
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1", port)) != 0
    # Idempotent: stopping again reports the same terminal state.
    assert program.run("stop", program.ref(SID))["service"]["state"] == "stopped"


def test_probe_body_is_capped(program):
    port = _free_port()
    (program.work / "big").write_text("x" * 5000)
    assert program.launch(_http_server(port), port, service_id=SID)["result"] == "ready"
    probe = program.run(
        "probe",
        program.ref(SID, method="GET", path="/big", headers={}, body=None, timeout=5, body_cap=100),
    )
    assert probe["body"] == "x" * 100 and probe["body_truncated"] is True


def test_a_provider_that_exits_is_reported_with_its_log(program):
    port = _free_port()
    argv = ["python3", "-c", "import sys; print('boom: missing config'); sys.exit(3)"]
    out = program.launch(argv, port)
    assert out["result"] == "failed" and out["reason"] == "exited_before_ready"
    assert out["service"]["state"] == "exited" and out["service"]["exit_code"] == 3
    assert "boom: missing config" in out["log_tail"]


def test_a_provider_that_never_gets_ready_is_stopped(program):
    port = _free_port()
    out = program.launch(
        ["python3", "-c", "import time; time.sleep(60)"], port, kind="tcp", ready_timeout=1
    )
    assert out["result"] == "failed" and out["reason"] == "ready_timeout"
    assert out["service"]["state"] == "stopped"
    assert not _pid_alive(out["service"]["pid"])


def test_a_port_already_in_use_is_refused(program):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        out = program.launch(_http_server(port), port)
    assert out == {"result": "port_in_use", "port": port}


def test_the_lifetime_ceiling_stops_a_forgotten_service(program):
    port = _free_port()
    out = program.launch(_http_server(port), port, service_id=SID, max_lifetime=2)
    assert out["result"] == "ready", out
    deadline = time.monotonic() + 10
    state = None
    while time.monotonic() < deadline:
        state = program.run("status", program.ref(SID))["service"]["state"]
        if state != "running":
            break
        time.sleep(0.2)
    assert state == "expired"
    assert not _pid_alive(out["service"]["pid"])


def test_service_output_is_capped_in_the_pod(program):
    port = _free_port()
    argv = [
        "python3",
        "-c",
        "import sys, time\nsys.stdout.write('y' * 50000); sys.stdout.flush()\ntime.sleep(60)",
    ]
    out = program.launch(argv, port, service_id=SID, kind="tcp", ready_timeout=1, log_cap=1000)
    assert out["result"] == "failed"
    assert out["service"]["log_bytes"] == 1000
    assert out["service"]["log_dropped"] == 49000


def test_the_service_count_is_bounded(program):
    ports = [_free_port() for _ in range(svc.MAX_SERVICES_PER_WORKSPACE + 1)]
    for port in ports[:-1]:
        assert program.launch(_http_server(port), port)["result"] == "ready"
    out = program.launch(_http_server(ports[-1]), ports[-1])
    assert out["result"] == "too_many_services"


@pytest.mark.skipif(not Path("/proc/net/tcp").exists(), reason="needs Linux /proc/net")
def test_a_provider_bound_beyond_loopback_is_refused(program):
    port = _free_port()
    argv = ["python3", "-m", "http.server", str(port), "--bind", "0.0.0.0"]  # noqa: S104
    out = program.launch(argv, port, path="/")
    assert out["result"] == "failed" and out["reason"] == "not_loopback"
    assert not _pid_alive(out["service"]["pid"])


def test_status_of_an_unknown_service_is_not_found(program):
    assert program.run("status", program.ref(SID)) == {"result": "not_found"}


def test_a_traversal_id_never_reaches_the_filesystem(program):
    out = program.run("status", program.ref("svc-../../etc"))
    assert out["result"] == "error"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@pytest.fixture
def as_tenant(app):
    app.dependency_overrides[require_tenant] = lambda: TENANT
    yield
    app.dependency_overrides.pop(require_tenant, None)


@pytest.fixture
def local_pod(tmp_path):
    """A workspace owned by TENANT whose 'pod' is this machine: execs really run."""
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "health").write_text("ok\n")
    calls: list[tuple] = []
    manager = MagicMock()

    def _get(name, tenant_id=None):
        if (name, tenant_id) == (WS, TENANT):
            return SimpleNamespace(container_name=name, codebase_path=str(root))
        raise ValueError(f"Container '{name}' not found")

    async def _exec(container_name, command, tenant_id=None, timeout=300, env=None):
        calls.append((container_name, command, tenant_id, timeout, env))
        argv = list(command)
        argv[0] = sys.executable
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    manager.get_container.side_effect = _get
    manager.execute_command.side_effect = _exec
    with (
        patch.object(ws_router, "get_container_manager", return_value=manager),
        patch.object(ws_router, "SERVICE_ROOT", str(tmp_path / "svc")),
    ):
        yield SimpleNamespace(manager=manager, calls=calls, root=root)


URL = f"/api/workspaces/{WS}/services"


async def test_route_start_probe_status_stop_end_to_end(client, as_tenant, local_pod, caplog):
    port = _free_port()
    caplog.set_level(logging.DEBUG)
    resp = await client.post(
        URL,
        json={
            "command": _http_server(port),
            "workdir": "/workspace",
            "port": port,
            "env": {"API_KEY": _SECRET},
            "readiness": {"kind": "http", "path": "/health"},
            "ready_timeout": 20,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    sid = body["service_id"]
    try:
        assert body["ready"] is True and body["result"] == "ready"
        assert body["url"] == f"http://127.0.0.1:{port}"
        assert body["service"]["state"] == "running"
        # The env value went to the pod (base64) and nowhere else.
        assert _SECRET not in resp.text and _SECRET not in caplog.text
        _, argv, tenant, _, env = local_pod.calls[0]
        assert argv[1:3] == ["-I", "-c"] and tenant == TENANT and env is None
        assert _SECRET not in argv[3]

        probe = await client.post(f"{URL}/{sid}/probe", json={"path": "/health"})
        assert probe.status_code == 200, probe.text
        assert probe.json()["ok"] is True and probe.json()["status"] == 200
        assert probe.json()["body"] == "ok\n"

        status = await client.get(f"{URL}/{sid}")
        assert status.status_code == 200 and status.json()["service"]["state"] == "running"
        listed = await client.get(URL)
        assert [s["service_id"] for s in listed.json()["services"]] == [sid]
    finally:
        stopped = await client.delete(f"{URL}/{sid}")
    assert stopped.status_code == 200, stopped.text
    assert stopped.json()["service"]["state"] == "stopped"

    gone = await client.post(f"{URL}/{sid}/probe", json={"path": "/health"})
    assert gone.status_code == 409


async def test_route_reports_a_failed_start_as_evidence(client, as_tenant, local_pod):
    resp = await client.post(
        URL,
        json={
            "command": ["python3", "-c", "raise SystemExit('provider: no DATABASE_URL')"],
            "port": _free_port(),
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ready"] is False and body["reason"] == "exited_before_ready"
    assert "provider: no DATABASE_URL" in body["log_tail"]


async def test_route_port_in_use_is_409(client, as_tenant, local_pod):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        resp = await client.post(URL, json={"command": _http_server(port), "port": port})
    assert resp.status_code == 409, resp.text


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ({"command": ["python3", "-m", "http.server", "8000"], "port": 8000}, "reserved"),
        ({"command": ["bash", "-c", "nc -l 9000"], "port": 19000}, "not allowed"),
        ({"command": ["python3", "x"], "port": 19000, "env": {"PATH": "/x"}}, "PATH"),
        ({"command": ["python3", "x"], "port": 19000, "workdir": "/etc"}, "workdir"),
        ({"command": ["python3", "x"], "port": 19000, "workdir": "/workspace/../etc"}, "workdir"),
        (
            {"command": ["python3", "x"], "port": 19000, "readiness": {"path": "/a b"}},
            "path",
        ),
    ],
)
async def test_route_refuses_bad_requests_before_any_exec(
    client, as_tenant, local_pod, body, fragment
):
    resp = await client.post(URL, json=body)
    assert resp.status_code == 400, resp.text
    assert fragment in resp.text
    assert local_pod.calls == []


@pytest.mark.parametrize(("toolchain", "expected"), [("python+rust", 200), ("python", 400)])
async def test_route_validates_with_the_workspaces_toolchain(
    client, as_tenant, local_pod, toolchain, expected
):
    original = local_pod.manager.get_container.side_effect

    def _get(name, tenant_id=None):
        info = original(name, tenant_id=tenant_id)
        info.toolchain = toolchain
        return info

    local_pod.manager.get_container.side_effect = _get
    resp = await client.post(
        URL, json={"command": ["ws-pg", "start"], "port": _free_port(), "ready_timeout": 1}
    )
    assert resp.status_code == expected, resp.text
    # python+rust: admitted and handed to the pod; python: refused before any exec.
    assert bool(local_pod.calls) is (expected == 200)


async def test_another_tenants_workspace_is_404_and_never_execs(app, client, local_pod):
    app.dependency_overrides[require_tenant] = lambda: "tenant-b"
    try:
        start = await client.post(URL, json={"command": ["python3", "x"], "port": 19000})
        status = await client.get(f"{URL}/{SID}")
        stop = await client.delete(f"{URL}/{SID}")
        probe = await client.post(f"{URL}/{SID}/probe", json={})
        listed = await client.get(URL)
    finally:
        app.dependency_overrides.pop(require_tenant, None)
    assert [r.status_code for r in (start, status, stop, probe, listed)] == [404] * 5
    assert local_pod.calls == []


@pytest.mark.parametrize(
    "body",
    [
        {"method": "TRACE"},
        {"path": "http://evil.example/"},
        {"headers": {"Host": "evil"}},
        {"headers": {"X-A": "a\r\nX-B: b"}},
        {"headers": {"Bad Name": "1"}},
        {"headers": {"X-Ä": "1"}},
        {"headers": {"X-١": "1"}},
        {"headers": {"X-A": "€"}},
    ],
)
async def test_route_probe_refuses_bad_requests(client, as_tenant, local_pod, body):
    resp = await client.post(f"{URL}/{SID}/probe", json=body)
    assert resp.status_code == 400, resp.text
    assert local_pod.calls == []


async def test_route_unknown_service_is_404(client, as_tenant, local_pod):
    assert (await client.get(f"{URL}/{SID}")).status_code == 404
    assert (await client.delete(f"{URL}/{SID}")).status_code == 404


def test_the_slowest_start_and_probe_answer_inside_300_seconds():
    """n8n's axios default is 300 s and the gateway's exec budget 330 s: the backend's
    own verdict (ready / failed + logs) must arrive before either gives up."""
    start = ws_router.StartServiceRequest.model_fields["ready_timeout"].metadata
    probe = ws_router.ProbeServiceRequest.model_fields["timeout"].metadata
    start_le = next(m.le for m in start if getattr(m, "le", None) is not None)
    probe_le = next(m.le for m in probe if getattr(m, "le", None) is not None)
    assert start_le + ws_router._LAUNCH_EXEC_SLACK_S < 300
    assert probe_le + ws_router._SHORT_EXEC_S < 300

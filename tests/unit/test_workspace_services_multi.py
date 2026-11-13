"""py2rust gap #4 full: N source services RUN in one workspace (one POST each).

To record a source system's behaviour the workspace runs its services (e.g. source A,
a FastAPI app that calls source B over HTTP). Two layers, as in
``test_workspace_services.py``:

* the REAL pod program as real local processes: A and B started, A calls B, the cap;
* the routes end-to-end with ``execute_command`` running the generated argv locally.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from coding_client.auth.tenant import require_tenant
from src.routers import workspace_services as ws_router
from src.services.workspace_services import service as svc

TENANT = "tenant-a"
WS = "ws-1"
FIXTURES = Path(__file__).parent / "data" / "py2rust_services"
#: The venv's interpreter first on PATH, so ``python3 -m uvicorn`` finds FastAPI.
_PATH = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert port != 8000
    return port


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(out.stdout.strip()) and not out.stdout.strip().startswith("Z")


def _uvicorn(port: int) -> list[str]:
    return ["python3", "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port)]


def _http_server(port: int) -> list[str]:
    return ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1"]


# ---------------------------------------------------------------------------
# The real pod program
# ---------------------------------------------------------------------------


@pytest.fixture
def pod(tmp_path):
    root = tmp_path / "svc"
    work = tmp_path / "repo"
    work.mkdir()
    (work / "health").write_text("ok\n")
    started: list[str] = []

    def run(action: str, spec: dict, timeout: float = 90, env: dict | None = None) -> dict:
        argv = svc.pod_program_argv(action, spec)
        argv[0] = sys.executable
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "PATH": _PATH, **(env or {})},
        )
        lines = proc.stdout.strip().splitlines()
        assert len(lines) == 1, (proc.stdout, proc.stderr)
        return json.loads(lines[0])

    def item(argv, port, workdir=None, **kw) -> dict:
        sid = f"svc-{os.urandom(8).hex()}"
        started.append(sid)
        spec = svc.build_launch_spec(
            service_id=sid,
            argv=argv,
            workdir=str(workdir or work),
            port=port,
            env=kw.pop("env", None),
            readiness_kind="http",
            readiness_path=kw.pop("path", "/health"),
            ready_timeout=30,
            max_lifetime=300,
            root=str(root),
            name=kw.pop("name", None),
        )
        spec.update(stop_grace=2)
        spec.update(kw)
        return spec

    def probe(sid: str, path: str) -> dict:
        return run(
            "probe",
            {
                "root": str(root),
                "service_id": sid,
                "tail_bytes": 0,
                "method": "GET",
                "path": path,
                "headers": {},
                "body": None,
                "timeout": 10,
                "body_cap": 65536,
            },
        )

    def stop(sid: str) -> dict:
        return run("stop", {"root": str(root), "service_id": sid, "tail_bytes": 0, "stop_grace": 2})

    yield SimpleNamespace(run=run, item=item, probe=probe, stop=stop, root=root)
    for sid in started:
        if (root / sid).exists():
            stop(sid)


def _a_and_b(pod, **b_env) -> tuple[dict, dict, int, int]:
    pa, pb = _free_port(), _free_port()
    b = pod.item(_uvicorn(pb), pb, workdir=FIXTURES / "svc_b", name="svc-b", env=b_env or None)
    a = pod.item(
        _uvicorn(pa),
        pa,
        workdir=FIXTURES / "svc_a",
        name="svc-a",
        env={"B_URL": f"http://127.0.0.1:{pb}"},
    )
    return a, b, pa, pb


def test_source_a_calls_source_b(pod):
    a, b, _, pb = _a_and_b(pod)
    outs = [pod.run("launch", b), pod.run("launch", a)]
    assert [o["result"] for o in outs] == ["ready", "ready"], outs
    assert [o["service"]["name"] for o in outs] == ["svc-b", "svc-a"]
    # A -> B over loopback HTTP; B's state persists across calls.
    first = json.loads(pod.probe(a["service_id"], "/chain/k1")["body"])
    second = json.loads(pod.probe(a["service_id"], "/chain/k1")["body"])
    assert first["service"] == "a" and first["b"] == {
        "service": "b",
        "key": "k1",
        "count": 1,
    }
    assert second["b"]["count"] == 2
    # B is reachable on its own port too, and the list shows both.
    assert json.loads(pod.probe(b["service_id"], "/items/k1")["body"])["count"] == 3
    listed = pod.run("list", {"root": str(pod.root)})
    assert {s["service_id"] for s in listed["services"]} == {a["service_id"], b["service_id"]}


def test_the_service_count_cap_refuses_and_a_free_slot_admits(pod):
    port = _free_port()
    out = pod.run("launch", pod.item(_http_server(port), port, path="/", max_services=0))
    assert out == {"result": "too_many_services", "running": 0, "max_services": 0}
    # Negative control: the same service with a free slot is started.
    out = pod.run("launch", pod.item(_http_server(port), port, path="/", max_services=1))
    assert out["result"] == "ready", out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@pytest.fixture
def as_tenant(app):
    app.dependency_overrides[require_tenant] = lambda: TENANT
    yield
    app.dependency_overrides.pop(require_tenant, None)


@pytest.fixture
def local_pod(tmp_path, monkeypatch):
    """A workspace owned by TENANT whose 'pod' is this machine: execs really run."""
    root = tmp_path / "workspace"
    (root / ".repos").mkdir(parents=True)
    for name in ("svc_a", "svc_b"):
        (root / ".repos" / f"acme__{name}").symlink_to(FIXTURES / name)
    calls: list[tuple] = []
    monkeypatch.setenv("PATH", _PATH)

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

    manager = SimpleNamespace(get_container=_get, execute_command=_exec)
    with (
        patch.object(ws_router, "get_container_manager", return_value=manager),
        patch.object(ws_router, "SERVICE_ROOT", str(tmp_path / "svc")),
    ):
        yield SimpleNamespace(manager=manager, calls=calls, root=root)
    for doc in _list_now(tmp_path / "svc"):
        subprocess.run(
            [
                sys.executable,
                *svc.pod_program_argv(
                    "stop",
                    {
                        "root": str(tmp_path / "svc"),
                        "service_id": doc,
                        "tail_bytes": 0,
                        "stop_grace": 2,
                    },
                )[1:],
            ],
            capture_output=True,
            check=False,
        )


def _list_now(root: Path) -> list[str]:
    return [p.name for p in root.iterdir()] if root.exists() else []


BASE = f"/api/workspaces/{WS}"


async def test_route_two_services_from_repos_checkouts(client, as_tenant, local_pod):
    """A + B from their own .repos/<name> checkouts: two POST /services, B first."""
    pa, pb = _free_port(), _free_port()
    common = {"readiness": {"kind": "http", "path": "/health"}}
    b = {
        "name": "svc-b",
        "command": _uvicorn(pb),
        "workdir": "/workspace/.repos/acme__svc_b",
        "port": pb,
        **common,
    }
    a = {
        "name": "svc-a",
        "command": _uvicorn(pa),
        "workdir": "/workspace/.repos/acme__svc_a",
        "port": pa,
        "env": {"B_URL": f"http://127.0.0.1:{pb}"},
        **common,
    }
    rb = await client.post(f"{BASE}/services", json=b)
    ra = await client.post(f"{BASE}/services", json=a)
    assert rb.status_code == 200 and rb.json()["ready"] is True, rb.text
    assert ra.status_code == 200 and ra.json()["ready"] is True, ra.text
    a_id = ra.json()["service_id"]
    p = await client.post(f"{BASE}/services/{a_id}/probe", json={"path": "/chain/x"})
    assert p.status_code == 200 and json.loads(p.json()["body"])["b"]["count"] == 1
    listed = (await client.get(f"{BASE}/services")).json()
    assert {s["name"] for s in listed["services"]} == {"svc-a", "svc-b"}


async def test_route_service_env_admits_app_names_and_refuses_pod_wiring(
    client, as_tenant, local_pod
):
    """POST {ws}/services with PYTHONUNBUFFERED=1 was a 400 under the namespace list
    (200 on main). These six reach the real service; PATH is still refused."""
    names = {
        "PYTHONUNBUFFERED": "1",
        "PGHOST": "127.0.0.1",
        "GIT_AUTHOR_NAME": "x",
        "CARGO_TRACKING_URL": "x",
        "MAKER_ID": "x",
        "JAVASCRIPT_URL": "x",
    }
    port = _free_port()
    code = (
        "import http.server, json, os\n"
        f"names = {sorted(names)!r}\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        body = json.dumps({n: os.environ.get(n) for n in names}).encode()\n"
        "        self.send_response(200); self.end_headers(); self.wfile.write(body)\n"
        f"http.server.HTTPServer(('127.0.0.1', {port}), H).serve_forever()\n"
    )
    body = {"command": ["python3", "-c", code], "port": port, "env": names}
    r = await client.post(f"{BASE}/services", json=body)
    assert r.status_code == 200 and r.json()["ready"] is True, r.text
    sid = r.json()["service_id"]
    p = await client.post(f"{BASE}/services/{sid}/probe", json={"path": "/"})
    assert json.loads(p.json()["body"]) == names
    port2 = _free_port()
    r = await client.post(
        f"{BASE}/services", json={**body, "port": port2, "env": {**names, "PATH": "/x"}}
    )
    assert r.status_code == 400 and "PATH" in r.text, r.text


async def test_routes_never_touch_another_tenants_workspace(app, client, local_pod):
    app.dependency_overrides[require_tenant] = lambda: "tenant-b"
    try:
        port = _free_port()
        r = await client.post(
            f"{BASE}/services", json={"command": _http_server(port), "port": port}
        )
    finally:
        app.dependency_overrides.pop(require_tenant, None)
    assert r.status_code == 404
    assert local_pod.calls == []

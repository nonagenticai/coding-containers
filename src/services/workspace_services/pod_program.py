"""The program that runs INSIDE a workspace pod to supervise a background service.

⛔ This file is not imported by the service at run time. Its SOURCE is read and
handed to ``python3 -I -c <source> <action> <base64 spec>`` in the workspace pod
(see :mod:`src.services.workspace_services.service`), so it must:

* use the standard library only (the pod runs ``python:3.14-slim`` + git/node);
* never import anything from ``src`` (it is not on the pod's ``sys.path``);
* print exactly ONE JSON document on stdout per invocation, and nothing else.

It is still a real module so that ruff/ty/bandit see it and the unit tests can run
the very same code as a real local process.

Process model (why it is shaped like this):

    exec (kubectl) ─ timeout ─ python3 -c PROGRAM launch      ← returns when ready
                                  └─ fork → setsid → SUPERVISOR  (survives the exec)
                                                 └─ fork → setpgid → SERVICE argv

* The SUPERVISOR leaves the exec's process group and session (``setsid``), so
  neither the end of the exec stream nor GNU ``timeout``'s group kill reaches it.
  Its stdio is ``/dev/null``: an inherited exec pipe would hold the exec open.
* The SERVICE runs in its OWN process group, so a stop is one ``killpg`` that
  also takes every child the service spawned (uvicorn workers, ``npm run``).
* The supervisor pumps the service's stdout+stderr into ``output.log`` up to a
  byte cap, then counts and drops: a chatty service cannot fill the emptyDir.
* It enforces ``max_lifetime`` itself — the safety ceiling — and writes the final
  state (``exited`` / ``stopped`` / ``expired``) before it exits. The pod is the
  outer bound: deleting the workspace kills every process in it.

State lives in ``<root>/<service_id>/state.json`` (atomic rename). The files are
writable by the workspace (same uid); nothing here trusts them for anything but
the tenant's OWN pod, so a forged file can only mislead its own author.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import re
import select
import signal
import socket
import sys
import time

STATE_FILE = "state.json"
LOG_FILE = "output.log"
TERMINAL = ("exited", "stopped", "expired", "failed", "lost")
LIVE = ("starting", "running")
_ID_RE = re.compile(r"^svc-[0-9a-f]{16}$")
_MAXFD = 1024


def _out(doc: dict) -> None:
    sys.stdout.write(json.dumps(doc, sort_keys=True) + "\n")
    sys.stdout.flush()


def _write_state(d: str, doc: dict) -> None:
    tmp = os.path.join(d, ".state.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    os.replace(tmp, os.path.join(d, STATE_FILE))


def _read_state(d: str) -> dict | None:
    try:
        with open(os.path.join(d, STATE_FILE), encoding="utf-8") as f:
            doc = json.load(f)
    except OSError, ValueError:
        return None
    return doc if isinstance(doc, dict) else None


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _alive(pid: object) -> bool:
    p = _int(pid)
    if p is None or p <= 1:
        return False
    try:
        os.kill(p, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _log_tail(d: str, limit: int) -> str:
    path = os.path.join(d, LOG_FILE)
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > limit:
                f.seek(size - limit)
            data = f.read(limit)
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def _port_open(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def listen_addresses(port: int, tables: dict[str, str]) -> list[tuple[str, bool]]:
    """``(address, is_loopback)`` for each LISTEN socket on ``port`` in ``/proc/net/tcp*`` text.

    ``tables`` maps ``"tcp"``/``"tcp6"`` to the file contents. IPv4 addresses are
    little-endian hex (``0100007F`` = 127.0.0.1); IPv6 are four little-endian words
    (``...01000000`` = ::1, ``0000000000000000FFFF00000100007F`` = ::ffff:127.0.0.1).
    """
    found: list[tuple[str, bool]] = []
    for family, text in tables.items():
        for line in text.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 4 or parts[3] != "0A":  # 0A = TCP_LISTEN
                continue
            addr, _, port_hex = parts[1].partition(":")
            try:
                if int(port_hex, 16) != port:
                    continue
            except ValueError:
                continue
            addr = addr.upper()
            if family == "tcp":
                loopback = len(addr) == 8 and addr.endswith("7F")
            else:
                loopback = addr == "00000000000000000000000001000000" or (
                    addr.startswith("0000000000000000FFFF0000") and addr.endswith("7F")
                )
            found.append((addr, loopback))
    return found


def _bind_check(port: int) -> str:
    """``loopback`` | ``exposed`` | ``unverified`` (no ``/proc/net``: not Linux).

    The pod's network namespace is the workspace's own, but the coding-jobs
    NetworkPolicy admits ingress from the rest of that namespace — i.e. from OTHER
    tenants' workspace pods. A provider bound to 0.0.0.0 would be reachable from
    them, so a service must listen on loopback only.
    """
    tables: dict[str, str] = {}
    for family in ("tcp", "tcp6"):
        try:
            with open(f"/proc/net/{family}", encoding="ascii") as f:
                tables[family] = f.read()
        except OSError:
            continue
    if not tables:
        return "unverified"
    found = listen_addresses(port, tables)
    if not found:
        return "unverified"
    return "loopback" if all(lb for _, lb in found) else "exposed"


def _ready(spec: dict) -> tuple[bool, int | None]:
    port = spec["port"]
    readiness = spec.get("readiness") or {}
    if readiness.get("kind") == "http":
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", readiness.get("path") or "/")
            resp = conn.getresponse()
            resp.read(4096)
            conn.close()
        except OSError, http.client.HTTPException:
            return False, None
        return resp.status < 500, resp.status
    return _port_open(port), None


class _Child:
    """The service process: its pid (== its process group id) and exit status."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.status: int | None = None

    def reap(self, block: bool = False) -> bool:
        if self.status is not None:
            return True
        try:
            pid, status = os.waitpid(self.pid, 0 if block else os.WNOHANG)
        except ChildProcessError:
            self.status = 0
            return True
        if pid == self.pid:
            self.status = status
            return True
        return False

    def group_alive(self) -> bool:
        try:
            os.killpg(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def kill_group(self, grace: float) -> None:
        """SIGTERM the whole group, wait ``grace``, then SIGKILL whatever is left."""
        try:
            os.killpg(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            self.reap()
            return
        end = time.monotonic() + grace
        while time.monotonic() < end:
            self.reap()
            if self.status is not None and not self.group_alive():
                return
            time.sleep(0.1)
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.reap(block=True)


def _exec_service(spec: dict, write_fd: int) -> None:
    """In the service child: own process group, stdio → pipe, exec the argv."""
    try:
        os.setpgid(0, 0)
        null = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null, 0)
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.closerange(3, _MAXFD)
        os.chdir(spec["workdir"])
        env = dict(os.environ)
        env.update(spec.get("env") or {})
        os.execvpe(spec["argv"][0], spec["argv"], env)  # noqa: S606  # nosec B606
    except BaseException as exc:  # noqa: BLE001 — must never return into the supervisor
        try:
            os.write(2, f"service failed to start: {type(exc).__name__}: {exc}\n".encode())
        except OSError:
            pass
    os._exit(127)


def _supervise(d: str, spec: dict) -> None:
    """The detached supervisor: run the service, pump its log, enforce the lifetime."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        _exec_service(spec, write_fd)
    os.close(write_fd)
    try:
        os.setpgid(pid, pid)  # closes the race with the child's own setpgid
    except OSError:
        pass
    child = _Child(pid)
    stop: dict[str, int | None] = {"sig": None}

    def _on_signal(signum: int, _frame: object) -> None:
        stop["sig"] = signum

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    started = time.time()
    state = {
        "service_id": spec["service_id"],
        "state": "running",
        "supervisor_pid": os.getpid(),
        "pid": pid,
        "port": spec["port"],
        "workdir": spec["workdir"],
        "program": os.path.basename(spec["argv"][0]),
        "name": spec.get("name"),
        "started_at": started,
        "max_lifetime": spec["max_lifetime"],
    }
    _write_state(d, state)

    cap = int(spec.get("log_cap") or 0)
    written = 0
    dropped = 0
    deadline = time.monotonic() + float(spec["max_lifetime"])
    reason = "exited"
    eof = False
    with open(os.path.join(d, LOG_FILE), "ab", buffering=0) as log:

        def pump(chunk: bytes) -> None:
            nonlocal written, dropped
            room = max(cap - written, 0)
            if room:
                log.write(chunk[:room])
                written += min(room, len(chunk))
            dropped += max(len(chunk) - room, 0)

        while True:
            if not eof:
                ready, _, _ = select.select([read_fd], [], [], 0.5)
                if ready:
                    chunk = os.read(read_fd, 65536)
                    if chunk:
                        pump(chunk)
                    else:
                        eof = True
            else:
                time.sleep(0.2)
            if child.reap():
                break
            if stop["sig"] is not None:
                reason = "stopped"
                break
            if time.monotonic() >= deadline:
                reason = "expired"
                break
        # Whatever ended the loop, nothing the service spawned outlives it.
        child.kill_group(float(spec.get("stop_grace") or 5))
        os.set_blocking(read_fd, False)
        try:
            while True:
                chunk = os.read(read_fd, 65536)
                if not chunk:
                    break
                pump(chunk)
        except BlockingIOError, OSError:
            pass
    exit_code = os.waitstatus_to_exitcode(child.status) if child.status is not None else None
    state.update(
        state=reason,
        exit_code=exit_code,
        ended_at=time.time(),
        log_bytes=written,
        log_dropped=dropped,
    )
    _write_state(d, state)


def _detach_and_supervise(d: str, spec: dict) -> None:
    """In the forked child: new session, no inherited stdio/fds, then supervise."""
    try:
        os.setsid()
        null = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(null, fd)
        os.closerange(3, _MAXFD)
        _supervise(d, spec)
    finally:
        os._exit(0)


def _summary(d: str, spec: dict, st: dict | None) -> dict:
    st = dict(st or {})
    if st.get("state") in LIVE and not _alive(st.get("supervisor_pid")):
        st["state"] = "lost"
    return {"service": st, "log_tail": _log_tail(d, int(spec.get("tail_bytes") or 8192))}


def _service_dir(spec: dict) -> str:
    sid = spec.get("service_id") or ""
    if not _ID_RE.match(sid):
        raise ValueError("invalid service id")
    return os.path.join(spec["root"], sid)


def _stop_dir(d: str, grace: float) -> dict | None:
    """Ask the supervisor to stop; force-kill if it does not answer in time."""
    st = _read_state(d)
    if st is None or st.get("state") in TERMINAL:
        return st
    sup = _int(st.get("supervisor_pid"))
    if sup and _alive(sup):
        try:
            os.kill(sup, signal.SIGTERM)
        except ProcessLookupError:
            pass
    end = time.monotonic() + grace + 3
    while time.monotonic() < end:
        st = _read_state(d) or st
        if st.get("state") in TERMINAL:
            return st
        time.sleep(0.1)
    # The supervisor did not finish: kill the service group and the supervisor.
    pgid = _int(st.get("pid"))
    own_pgid = os.getpgid(0)
    if pgid and pgid > 1 and pgid != own_pgid:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError, PermissionError:
            pass
    if sup and sup > 1 and sup != os.getpid():
        try:
            os.kill(sup, signal.SIGKILL)
        except ProcessLookupError, PermissionError:
            pass
    st = dict(st, state="stopped", forced=True, ended_at=time.time())
    _write_state(d, st)
    return st


def _all_states(root: str) -> list[tuple[str, dict]]:
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return []
    out = []
    for name in names:
        if _ID_RE.match(name):
            st = _read_state(os.path.join(root, name))
            if st is not None:
                out.append((name, st))
    return out


def launch(spec: dict) -> dict:
    root = spec["root"]
    os.makedirs(root, mode=0o700, exist_ok=True)
    live = [
        st
        for _, st in _all_states(root)
        if st.get("state") in LIVE and _alive(st.get("supervisor_pid"))
    ]
    if len(live) >= int(spec["max_services"]):
        return {
            "result": "too_many_services",
            "running": len(live),
            "max_services": int(spec["max_services"]),
        }
    port = int(spec["port"])
    if any(_int(st.get("port")) == port for st in live) or _port_open(port):
        return {"result": "port_in_use", "port": port}
    if not os.path.isdir(spec["workdir"]):
        return {"result": "workdir_missing", "workdir": spec["workdir"]}
    d = _service_dir(spec)
    os.mkdir(d, 0o700)
    _write_state(d, {"service_id": spec["service_id"], "state": "starting", "port": port})
    sys.stdout.flush()
    pid = os.fork()
    if pid == 0:
        _detach_and_supervise(d, spec)
    deadline = time.monotonic() + float(spec["ready_timeout"])
    while True:
        st = _read_state(d) or {}
        if st.get("state") in TERMINAL:
            return {"result": "failed", "reason": "exited_before_ready", **_summary(d, spec, st)}
        if st.get("state") == "running":
            ok, status = _ready(spec)
            if ok:
                bind = _bind_check(port)
                if bind == "exposed":
                    st = _stop_dir(d, float(spec.get("stop_grace") or 5))
                    return {"result": "failed", "reason": "not_loopback", **_summary(d, spec, st)}
                return {
                    "result": "ready",
                    "http_status": status,
                    "bind": bind,
                    **_summary(d, spec, st),
                }
        if time.monotonic() >= deadline:
            st = _stop_dir(d, float(spec.get("stop_grace") or 5))
            return {"result": "failed", "reason": "ready_timeout", **_summary(d, spec, st)}
        time.sleep(0.25)


def status(spec: dict) -> dict:
    d = _service_dir(spec)
    st = _read_state(d)
    if st is None:
        return {"result": "not_found"}
    return {"result": "ok", **_summary(d, spec, st)}


def list_services(spec: dict) -> dict:
    services = []
    for name, st in _all_states(spec["root"]):
        st = dict(st)
        if st.get("state") in LIVE and not _alive(st.get("supervisor_pid")):
            st["state"] = "lost"
        st["service_id"] = name
        services.append(st)
    return {"result": "ok", "services": services}


def stop(spec: dict) -> dict:
    d = _service_dir(spec)
    if _read_state(d) is None:
        return {"result": "not_found"}
    st = _stop_dir(d, float(spec.get("stop_grace") or 5))
    return {"result": "ok", **_summary(d, spec, st)}


def probe(spec: dict) -> dict:
    d = _service_dir(spec)
    st = _read_state(d)
    if st is None:
        return {"result": "not_found"}
    summary = _summary(d, spec, st)
    if summary["service"].get("state") != "running":
        return {"result": "not_running", **summary}
    port = _int(st.get("port"))
    if port is None or not 0 < port < 65536:
        return {"result": "not_running", **summary}
    cap = int(spec["body_cap"])
    body = spec.get("body")
    payload = base64.b64decode(body) if body else None
    started = time.monotonic()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=float(spec["timeout"]))
        conn.request(spec["method"], spec["path"], body=payload, headers=spec.get("headers") or {})
        resp = conn.getresponse()
        data = resp.read(cap + 1)
        headers = [[k, v[:1024]] for k, v in resp.getheaders()[:64]]
        conn.close()
    except (OSError, http.client.HTTPException) as exc:
        return {
            "result": "request_failed",
            "error": f"{type(exc).__name__}: {exc}"[:1024],
            "duration": time.monotonic() - started,
        }
    return {
        "result": "ok",
        "status": resp.status,
        "reason": (resp.reason or "")[:256],
        "headers": headers,
        "body": data[:cap].decode("utf-8", errors="replace"),
        "body_truncated": len(data) > cap,
        "duration": time.monotonic() - started,
    }


ACTIONS = {
    "launch": launch,
    "status": status,
    "list": list_services,
    "stop": stop,
    "probe": probe,
}


def main(argv: list[str]) -> int:
    try:
        action = ACTIONS[argv[1]]
        spec = json.loads(base64.b64decode(argv[2]))
        _out(action(spec))
    except Exception as exc:  # noqa: BLE001 — one JSON answer, always
        _out({"result": "error", "error": f"{type(exc).__name__}: {exc}"[:1024]})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

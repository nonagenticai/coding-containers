#!/usr/bin/env python3
"""Drift checks for the vendored bug-hunt agent (vendor/hunt-agent/).

coding-analysis execs ``python3 -I -P /opt/hunt-agent/hunt.py`` inside every
workspace pod and FAILS the hunt when the agent's reported ``agent_sha`` or
tree-sitter/grammar versions differ from its own. This script keeps the copy
baked into codegraph-runtime equal to the source, in three independent ways:

    check          vendored files are self-consistent: MANIFEST.sha256 is the
                   recomputed listing of agent/*.py, sha256(MANIFEST) equals
                   PIN.json agent_sha, the file set is exact, and the pinned
                   packages are exactly tree-sitter + every grammar in GRAMMARS.
    source DIR     DIR is a coding-analysis checkout (at PIN source_ref): its
                   src/hunt_agent equals agent/ byte for byte, and its uv.lock
                   pins equal PIN.json python_packages.
    requirements   print ``name==version`` lines (the Dockerfile installs these).
    image-smoke    run the agent EXACTLY as coding-analysis does (python3 -I -P)
                   on a throwaway repo with one file per grammar, and assert the
                   reported agent_sha / versions equal PIN and that every file
                   parsed. Run inside the built image.

The manifest is recomputed HERE (stdlib, never by importing hunt.py), so a
broken agent cannot vouch for itself. Stdlib only: it also runs as
``python3 - image-smoke < this-file`` inside the image, where only the pin is
passed in (``HUNT_PIN_JSON``).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess  # nosec B404 - fixed argv, no shell
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

MANIFEST_NAME = "MANIFEST.sha256"
PIN_NAME = "PIN.json"
AGENT_SUBDIR = "agent"
PENDING_REF = "PENDING_MERGE"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_VERSION = re.compile(r"^[0-9]+(\.[0-9]+)*$")
_PIN_KEYS = ("source_repo", "source_path", "source_ref", "agent_sha", "python_packages")

# One file per grammar in hunt.py GRAMMARS; a.py carries a syntax error so the
# smoke also proves detection runs, not just parsing.
SMOKE_FILES: dict[str, str] = {
    "a.py": "def broken(:\n    pass\n",
    "b.js": "function f(a) { return a + 1; }\n",
    "c.ts": "export function f(a: number): number { return a + 1; }\n",
    "d.go": "package main\n\nfunc f(a int) int { return a + 1 }\n",
    "e.java": "class E { int f(int a) { return a + 1; } }\n",
    "f.rs": "fn f(a: i32) -> i32 { a + 1 }\n",
    "g.cpp": "int f(int a) { return a + 1; }\n",
    "h.cs": "class H { int F(int a) { return a + 1; } }\n",
}


def default_vendor_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "vendor" / "hunt-agent"


def manifest_text(agent_dir: Path) -> str:
    """``sha256sum``-format listing of the top-level ``*.py`` files, by name."""
    files = sorted((p for p in agent_dir.glob("*.py") if p.is_file()), key=lambda p: p.name)
    return "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in files)


def load_pin(vendor_dir: Path) -> dict[str, Any]:
    return json.loads((vendor_dir / PIN_NAME).read_text(encoding="utf-8"))


def pin_errors(pin: dict[str, Any]) -> list[str]:
    errors = [f"PIN.json: missing key {key!r}" for key in _PIN_KEYS if key not in pin]
    if errors:
        return errors
    if not _HEX64.match(str(pin["agent_sha"])):
        errors.append(f"PIN.json: agent_sha is not 64 lower-case hex: {pin['agent_sha']!r}")
    packages = pin["python_packages"]
    if not isinstance(packages, dict) or "tree-sitter" not in packages:
        errors.append("PIN.json: python_packages must be a mapping that includes tree-sitter")
        return errors
    for name, version in packages.items():
        if not isinstance(version, str) or not _VERSION.match(version):
            errors.append(f"PIN.json: {name} must be an exact version, got {version!r}")
    return errors


def grammar_distributions(hunt_py: Path) -> set[str]:
    """Distribution names in hunt.py's ``GRAMMARS`` literal (3rd tuple element)."""
    tree = ast.parse(hunt_py.read_bytes(), filename=str(hunt_py))
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            first = node.targets[0]
            target, value = (first.id if isinstance(first, ast.Name) else None), node.value
        if target == "GRAMMARS" and value is not None:
            literal = ast.literal_eval(value)
            return {entry[2] for entry in literal.values()}
    raise ValueError(f"no GRAMMARS literal in {hunt_py}")


def check_vendor(vendor_dir: Path) -> list[str]:
    """Self-consistency of the vendored copy. Empty list = consistent."""
    agent_dir = vendor_dir / AGENT_SUBDIR
    pin = load_pin(vendor_dir)
    errors = pin_errors(pin)
    if errors:
        return errors

    present = {p.name for p in agent_dir.iterdir()}
    py_files = {p.name for p in agent_dir.glob("*.py") if p.is_file()}
    unexpected = sorted(present - py_files - {MANIFEST_NAME})
    if unexpected:
        errors.append(f"agent/ holds files outside the manifest's scope: {unexpected}")
    if "hunt.py" not in py_files:
        return [*errors, "agent/hunt.py is missing"]
    manifest = agent_dir / MANIFEST_NAME
    committed = manifest.read_bytes() if manifest.is_file() else b""
    computed = manifest_text(agent_dir).encode("utf-8")
    if committed != computed:
        errors.append(
            "agent/MANIFEST.sha256 does not match the vendored *.py files "
            f"(a vendored file was edited):\n--- committed\n{committed.decode()}"
            f"+++ computed\n{computed.decode()}"
        )
    actual_sha = hashlib.sha256(committed).hexdigest()
    if actual_sha != pin["agent_sha"]:
        errors.append(f"sha256(MANIFEST.sha256)={actual_sha} != PIN agent_sha={pin['agent_sha']}")

    wanted = {"tree-sitter", *grammar_distributions(agent_dir / "hunt.py")}
    if set(pin["python_packages"]) != wanted:
        errors.append(
            f"PIN python_packages {sorted(pin['python_packages'])} != tree-sitter + "
            f"hunt.py GRAMMARS distributions {sorted(wanted)}"
        )
    return errors


def lock_versions(uv_lock: Path, names: set[str]) -> dict[str, str | None]:
    data = tomllib.loads(uv_lock.read_text(encoding="utf-8"))
    found: dict[str, str | None] = dict.fromkeys(names)
    for package in data.get("package", []):
        if package.get("name") in found:
            found[package["name"]] = package.get("version")
    return found


def check_source(vendor_dir: Path, source_root: Path) -> list[str]:
    """Byte-for-byte equality with a coding-analysis checkout, plus uv.lock pins."""
    pin = load_pin(vendor_dir)
    agent_dir = vendor_dir / AGENT_SUBDIR
    source_dir = source_root / pin["source_path"]
    errors: list[str] = []

    def files(root: Path) -> set[str]:
        return {p.name for p in root.iterdir() if p.is_file()}

    vendored, upstream = files(agent_dir), files(source_dir)
    if vendored != upstream:
        errors.append(
            f"file set differs: only vendored {sorted(vendored - upstream)}, "
            f"only upstream {sorted(upstream - vendored)}"
        )
    for name in sorted(vendored & upstream):
        if (agent_dir / name).read_bytes() != (source_dir / name).read_bytes():
            errors.append(f"{name}: vendored bytes differ from {pin['source_repo']}")

    locked = lock_versions(source_root / "uv.lock", set(pin["python_packages"]))
    for name, version in sorted(pin["python_packages"].items()):
        if locked[name] != version:
            errors.append(f"{name}: PIN {version} != coding-analysis uv.lock {locked[name]}")
    return errors


def requirements(pin: dict[str, Any]) -> str:
    return "".join(
        f"{name}=={version}\n" for name, version in sorted(pin["python_packages"].items())
    )


def image_smoke(pin: dict[str, Any], agent_dir: Path, python: str) -> list[str]:
    """Run the agent as coding-analysis does and grade its own report against PIN."""
    git = shutil.which("git")
    if git is None:
        return ["git is not on PATH; the agent cannot run without it"]
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="hunt-smoke-") as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        for name, body in SMOKE_FILES.items():
            (repo / name).write_text(body, encoding="utf-8")
        ident = ["-c", "user.name=smoke", "-c", "user.email=smoke@invalid"]
        for argv in (["init", "-q"], ["add", "-A"], [*ident, "commit", "-q", "-m", "smoke"]):
            subprocess.run([git, "-C", str(repo), *argv], check=True)  # nosec B603
        out = Path(tmp) / "hunt-agent-smoke.json"
        argv = [
            python,
            "-I",
            "-P",
            str(agent_dir / "hunt.py"),
            "--repo",
            str(repo),
            "--out",
            str(out),
        ]
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)  # nosec B603
        if proc.returncode != 0:
            return [f"agent exited {proc.returncode}: {proc.stdout.strip()} {proc.stderr.strip()}"]
        summary = json.loads(proc.stdout)
        document = json.loads(out.read_text(encoding="ascii"))

    packages = pin["python_packages"]
    checks = [
        ("summary agent_sha", summary.get("agent_sha"), pin["agent_sha"]),
        ("result agent_sha", document.get("agent_sha"), pin["agent_sha"]),
        ("tree_sitter_version", document.get("tree_sitter_version"), packages["tree-sitter"]),
        ("grammar_versions", document.get("grammar_versions"), packages),
        ("tracked_source_files", document.get("tracked_source_files"), len(SMOKE_FILES)),
        ("scanned_files", document.get("scanned_files"), len(SMOKE_FILES)),
        ("not_scanned", document.get("not_scanned"), []),
    ]
    for label, actual, expected in checks:
        if actual != expected:
            errors.append(f"{label}: agent reported {actual!r}, expected {expected!r}")
    if not any(
        f.get("file_path") == "a.py" and f.get("error_type") == "syntax_error"
        for f in document.get("findings", [])
    ):
        errors.append("the seeded syntax error in a.py was not reported")
    return errors


def _report(errors: list[str], ok: str) -> int:
    for error in errors:
        sys.stderr.write(f"FAIL: {error}\n")
    if errors:
        return 1
    sys.stdout.write(f"OK: {ok}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drift checks for vendor/hunt-agent.")
    parser.add_argument("--vendor-dir", type=Path, default=None)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("check")
    source = sub.add_parser("source")
    source.add_argument("repo", type=Path, help="coding-analysis checkout at PIN source_ref")
    sub.add_parser("requirements")
    smoke = sub.add_parser("image-smoke")
    smoke.add_argument("--agent-dir", type=Path, default=Path("/opt/hunt-agent"))
    smoke.add_argument(
        "--python", default="python3", help="interpreter, as coding-analysis names it"
    )
    args = parser.parse_args(argv)
    command = args.command or "check"

    if command == "image-smoke" and os.environ.get("HUNT_PIN_JSON"):
        pin = json.loads(os.environ["HUNT_PIN_JSON"])
    else:
        pin = load_pin(args.vendor_dir or default_vendor_dir())
    vendor_dir = args.vendor_dir or (None if command == "image-smoke" else default_vendor_dir())

    if command == "check":
        assert vendor_dir is not None
        if pin.get("source_ref") == PENDING_REF:
            sys.stderr.write(
                "NOTICE: PIN source_ref is PENDING_MERGE; set it to the merged "
                "coding-analysis commit so the source check has a ref to grade.\n"
            )
        return _report(check_vendor(vendor_dir), f"vendored agent_sha {pin['agent_sha']}")
    if command == "source":
        assert vendor_dir is not None
        return _report(check_source(vendor_dir, args.repo), f"identical to {args.repo}")
    if command == "requirements":
        errors = pin_errors(pin)
        if errors:
            return _report(errors, "")
        sys.stdout.write(requirements(pin))
        return 0
    errors = pin_errors(pin) or image_smoke(pin, args.agent_dir, args.python)
    return _report(errors, f"agent_sha {pin['agent_sha']} + tree-sitter pins verified in place")


if __name__ == "__main__":
    sys.exit(main())

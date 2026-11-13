"""The vendored bug-hunt agent (vendor/hunt-agent/) cannot drift silently.

coding-analysis fails every hunt whose reported agent_sha it does not expect,
so a vendored file edited here (by hand, by a formatter, by a merge) must red
CI rather than ship. These tests run scripts/check-hunt-agent-vendor.py's
``check`` on the REAL vendored tree and on mutated copies of it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = REPO_ROOT / "vendor" / "hunt-agent"
SCRIPT = REPO_ROOT / "scripts" / "check-hunt-agent-vendor.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_hunt_agent_vendor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


@pytest.fixture
def vendor_copy(tmp_path: Path) -> Path:
    target = tmp_path / "hunt-agent"
    shutil.copytree(VENDOR_DIR, target, ignore=shutil.ignore_patterns("__pycache__"))
    return target


def test_real_vendored_tree_is_consistent() -> None:
    assert checker.check_vendor(VENDOR_DIR) == []


def test_pin_agent_sha_is_sha256_of_the_committed_manifest() -> None:
    pin = json.loads((VENDOR_DIR / "PIN.json").read_text(encoding="utf-8"))
    manifest = (VENDOR_DIR / "agent" / "MANIFEST.sha256").read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == pin["agent_sha"]


def test_copy_is_consistent_before_mutation(vendor_copy: Path) -> None:
    # Control: the mutation tests below are only meaningful if the unmutated
    # copy passes, i.e. the copy itself is not what reds them.
    assert checker.check_vendor(vendor_copy) == []


def test_mutated_agent_file_is_detected(vendor_copy: Path) -> None:
    hunt = vendor_copy / "agent" / "hunt.py"
    hunt.write_bytes(
        hunt.read_bytes().replace(b"MAX_FUNCTION_LINES = 100", b"MAX_FUNCTION_LINES = 101")
    )
    errors = checker.check_vendor(vendor_copy)
    assert any("MANIFEST.sha256 does not match" in e for e in errors), errors


def test_mutated_agent_with_refreshed_manifest_is_detected_by_the_pin(vendor_copy: Path) -> None:
    # A careful drift (edit + manifest refresh) still cannot match PIN agent_sha.
    agent = vendor_copy / "agent"
    (agent / "__init__.py").write_bytes((agent / "__init__.py").read_bytes() + b"\n")
    (agent / "MANIFEST.sha256").write_text(checker.manifest_text(agent), encoding="utf-8")
    errors = checker.check_vendor(vendor_copy)
    assert any("!= PIN agent_sha" in e for e in errors), errors


def test_extra_file_in_agent_dir_is_detected(vendor_copy: Path) -> None:
    (vendor_copy / "agent" / "notes.txt").write_text("x", encoding="utf-8")
    errors = checker.check_vendor(vendor_copy)
    assert any("outside the manifest's scope" in e for e in errors), errors


def test_pins_must_cover_exactly_the_agents_grammars(vendor_copy: Path) -> None:
    pin_path = vendor_copy / "PIN.json"
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    del pin["python_packages"]["tree-sitter-go"]
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    errors = checker.check_vendor(vendor_copy)
    assert any("GRAMMARS distributions" in e for e in errors), errors


def test_pin_versions_must_be_exact(vendor_copy: Path) -> None:
    pin_path = vendor_copy / "PIN.json"
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    pin["python_packages"]["tree-sitter"] = ">=0.25"
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    errors = checker.check_vendor(vendor_copy)
    assert any("exact version" in e for e in errors), errors


def test_source_check_flags_a_byte_difference(vendor_copy: Path, tmp_path: Path) -> None:
    # Build a fake coding-analysis checkout from the vendored tree, then drift it.
    pin = json.loads((vendor_copy / "PIN.json").read_text(encoding="utf-8"))
    upstream = tmp_path / "coding-analysis"
    shutil.copytree(vendor_copy / "agent", upstream / pin["source_path"])
    lock = "".join(
        f'[[package]]\nname = "{name}"\nversion = "{version}"\n\n'
        for name, version in pin["python_packages"].items()
    )
    (upstream / "uv.lock").write_text(lock, encoding="utf-8")
    assert checker.check_source(vendor_copy, upstream) == []

    hunt = upstream / pin["source_path"] / "hunt.py"
    hunt.write_bytes(hunt.read_bytes() + b"# drift\n")
    errors = checker.check_source(vendor_copy, upstream)
    assert errors == [f"hunt.py: vendored bytes differ from {pin['source_repo']}"]


def test_requirements_are_the_exact_pins() -> None:
    pin = json.loads((VENDOR_DIR / "PIN.json").read_text(encoding="utf-8"))
    lines = checker.requirements(pin).splitlines()
    assert lines == [f"{n}=={v}" for n, v in sorted(pin["python_packages"].items())]

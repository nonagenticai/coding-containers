"""The ``"n8n-dev"`` tenant literal must not come back into ``src/``.

WHAT WAS WRONG
--------------
``src/main.py:35`` carried ``DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID",
"n8n-dev")`` as a module-level constant with **zero readers**.

Zero readers is not the same as harmless. ``"n8n-dev"`` is not junk and it is not
a real tenant_id either, which is exactly what made it dangerous: it **was** a
real tenant, as a text tenant_id, until the 2026-04-29 Keycloak migration moved
tenant identity to uuid7 and demoted the label to a ``tenant_name`` attribute.
So the next person to reach for the nearest-looking constant would have got a
tenant nobody chose rather than a failure.

This repo already owns the argument against it. ``src/clients/_tenant.py`` says
in its own docstring that substituting ``DEFAULT_TENANT_ID``, ``"anonymous"`` or
any other literal "makes the code compile and the call succeed while attributing
it to a tenant that does not own the resource" — and the constant sat in
``main.py`` the whole time.

WHY THE SUITE DID NOT CATCH IT
------------------------------
Nothing read the constant, so no behavioural test could have covered it. That is
the point of a source-shape rule: it catches what has no runtime path *yet*.

MUTANTS — each was run against this file
----------------------------------------
    M1  restore the constant/field with the literal   -> FAILS
    M2  same literal under another name               -> FAILS (keyed on the
        literal, not the field name)
    M3  literal in a comment or docstring only        -> passes, deliberately
    M4  scan root at an empty directory               -> the NON-VACUITY test
        fails, not the rules
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

# Resolve the scan root through an IMPORTED module, never a path relative to
# this file. A relative path is the classic way a guard ends up scanning an
# empty directory and passing vacuously (M4).
_MODULE_FILE = importlib.import_module("config.settings").__file__
assert _MODULE_FILE is not None, "config.settings has no __file__"
SRC = pathlib.Path(_MODULE_FILE).parent.parent

BANNED_TENANT_LITERAL = "n8n-dev"


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """``id()`` of every docstring Constant — module, class and function.

    Docstrings are prose, not values the code can hand out, so they are exempt.
    Excluded by IDENTITY rather than by "is the first statement", because
    ``ast.walk`` yields nodes flattened and a positional test would also exempt
    a real assignment that happens to be first in its block.
    """
    out: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and ast.get_docstring(node, clean=False) is not None
        ):
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                out.add(id(first.value))
    return out


@pytest.fixture(scope="module")
def parsed() -> list[tuple[pathlib.Path, ast.Module]]:
    """Every ``src/`` module, parsed once."""
    out = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        out.append((path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))))
    return out


def test_the_scan_is_not_vacuous(parsed):
    """Prove the walk found this service's real source before trusting a pass.

    Without this, deleting or mistyping ``SRC`` turns every rule below into a
    green that measured nothing.
    """
    files = {p for p, _ in parsed}
    assert len(files) >= 50, f"only {len(files)} source files found under {SRC}"
    names = {p.name for p in files}
    for required in ("settings.py", "main.py", "_tenant.py"):
        assert required in names, f"{required} not in the scanned set under {SRC}"

    # Positive control that the AST walk actually parses string constants. The
    # value is MEASURED (24 occurrences of "coding-containers" in src/), not
    # guessed: an earlier draft of this control in a sibling repo named a string
    # that the fix itself had just removed, so the control could never hold.
    found_known = any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "coding-containers" in node.value
        for _, tree in parsed
        for node in ast.walk(tree)
    )
    assert found_known, "the AST walk found no known string constant — it is not seeing code"


def test_the_n8n_dev_tenant_literal_is_absent_from_src(parsed):
    """No ``src/`` module may carry ``"n8n-dev"`` as a string constant.

    Keyed on the LITERAL, not on the field name, so reintroducing it as
    ``fallback_tenant`` or ``tenant_default`` is still caught. Comments and
    docstrings are invisible to the AST walk by design — prose explaining the
    removal must stay legal, and this docstring is the standing false-positive
    control for that.
    """
    offenders = []
    for path, tree in parsed:
        exempt = _docstring_nodes(tree)
        offenders += [
            f"{path.relative_to(SRC)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and BANNED_TENANT_LITERAL in node.value
            and id(node) not in exempt
        ]
    assert not offenders, (
        f"the {BANNED_TENANT_LITERAL!r} tenant literal is back at: {offenders}. "
        "It is the stale pre-uuid7 label of a real tenant, so it fails SILENTLY "
        "rather than loudly. Read the tenant from "
        "coding_client.auth.tenant.DEFAULT_TENANT_ID, which is None when unset."
    )

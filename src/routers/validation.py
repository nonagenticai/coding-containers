"""Validation endpoints that need a workspace.

The gateway proxies /api/validation/multi-line-code-snippet here because the
two checks that catch hallucinated code both run INSIDE the caller's workspace
(a Kubernetes pod in production), through the same runtime path as
``POST /api/containers/{name}/exec``.

WHAT IS CHECKED
---------------
* Syntax — ``ast.parse`` here, always.
* Import resolution (only with a ``container_name``) — one exec of the constant
  ``_RESOLVER_PROGRAM`` (``python3 -I -c PROG [--root R] item...``) in the
  workspace. Every ABSOLUTE import is located with Python's own path finder
  (``importlib.machinery.PathFinder``) over, in ``sys.path`` order: the
  repository root (``codebase_path`` when given, else the workspace root), the
  stdlib, the root's ``src/``, then the workspace interpreter's site-packages
  (``<root>/.venv`` or ``/workspace/.venv``, else the running interpreter's).
  With a venv interpreter present the program re-execs itself under it.
  Nothing is imported or executed: specs are located, never loaded, so a
  regular package beats a namespace directory exactly as it would at import
  time, and every dotted segment is checked (``os.ghostmod`` is unresolved).
  A module that is not found anywhere but is declared by the repository
  (pyproject, setup.cfg/setup.py, requirements*.txt, normalised dist→import
  names) is ``declared_not_installed`` — listed, never invalidating.
* Imported symbols (``from M import N``) — N must be a submodule of M or bound
  at M's module level, read from M's SOURCE with ``ast``. Repo, stdlib and
  installed modules alike. A module whose source cannot be read (extension,
  builtin, sourceless), or that has ``from x import *``, a module-level
  ``__getattr__`` or a ``sys.modules[...]`` assignment is ``unverifiable`` for
  the symbol (``symbol_status``) — the module itself still counts.
* Optional execution — ``python -c <snippet>`` in the workspace.

WHAT IS NOT CHECKED
-------------------
* Relative imports (the snippet's package is unknown) — ``unverifiable``.
* Attribute access, call signatures, types — only names bound by import.
* Execution that the runtime REFUSES (command policy, e.g. a snippet that
  mentions ``shutdown``) or cannot reach is ``execution_status:
  "not_computed"``, never a failure of the code.

Every response carries ``verdict``: ``invalid`` (syntax error, unresolved
import/symbol, or an execution that ran and failed), ``not_computed``
(syntax fine but some import or a requested execution was not checked) or
``valid``. Without a ``container_name`` imports are ``not_computed``.
"""

from __future__ import annotations

import ast
import json
import logging
import time
from types import SimpleNamespace
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from coding_client.auth.tenant import require_tenant
from coding_client.models.responses import internal_error
from src.routers.container_exec import run_workspace_command
from src.routers.tenant_guard import authorize_workspace
from src.runtime import RunOnceResult

logger = logging.getLogger(__name__)
router = APIRouter()

ROUTE = "multi-line-code-snippet"
_RESOLVE_TIMEOUT = 30
_REFUSED = "refused by command policy"

ImportStatus = Literal[
    "repo",
    "stdlib",
    "installed",
    "declared_not_installed",
    "unresolved",
    "unresolved_symbol",
    "unverifiable",
    "not_checked",
]
SymbolStatus = Literal["found", "missing", "unverifiable"]
_RESOLVER_STATUSES = frozenset(
    {
        "repo",
        "stdlib",
        "installed",
        "declared_not_installed",
        "unresolved",
        "unresolved_symbol",
        "unverifiable",
    }
)
_SYMBOL_STATUSES = frozenset({"found", "missing", "unverifiable", None})
_FAILING_STATUSES = frozenset({"unresolved", "unresolved_symbol"})
_UNCHECKED_STATUSES = frozenset({"not_checked", "unverifiable"})
_IMPORT_GUARD_EXCEPTIONS = frozenset(
    {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
)


# ---------------------------------------------------------------------------
# Models (self-contained, matching the monolith's API contract)
# ---------------------------------------------------------------------------


class MultiLineCodeSnippetRequest(BaseModel):
    """Request for multi-line code snippet validation."""

    code_snippet: str = Field(
        ..., min_length=1, description="Multi-line Python code snippet to validate"
    )
    container_name: str | None = Field(
        None, description="Workspace name for import resolution and execution"
    )
    codebase_path: str | None = Field(
        None,
        description="Absolute repository root inside the workspace; resolution and "
        "execution run there instead of the workspace's registered root",
    )
    include_execution_validation: bool = Field(
        False, description="Whether to include runtime execution"
    )
    execution_timeout: int = Field(
        default=10, gt=0, le=60, description="Execution timeout in seconds"
    )

    @field_validator("code_snippet")
    @classmethod
    def validate_code_snippet(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("Code snippet cannot be empty")
        if len(v.strip()) < 3:
            raise ValueError("Code snippet too short to be meaningful")
        return v

    @field_validator("codebase_path")
    @classmethod
    def validate_codebase_path(cls, v: str | None) -> str | None:
        """Same rule as the file routes' ``_repository_root``: an absolute path
        other than ``/``. Anything else is refused rather than silently replaced
        by ``/workspace`` — that would resolve against a root the caller did not
        name."""
        if v is None or not v.strip():
            return None
        from src.routers.containers.files_manage import _repository_root

        normalized = v.strip()
        if (
            "\x00" in normalized
            or _repository_root(SimpleNamespace(codebase_path=normalized)) != normalized
        ):
            raise ValueError("codebase_path must be an absolute path other than '/'")
        return normalized


class ExecutionAnalysis(BaseModel):
    execution_successful: bool
    #: ``not_computed`` when the workspace refused or could not run the snippet —
    #: a statement about the workspace, never about the code.
    execution_status: Literal["computed", "not_computed"] = "computed"
    not_computed_reason: str | None = None
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    runtime_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ImportCheck(BaseModel):
    """One imported name and where (if anywhere) it resolves."""

    module: str
    name: str | None = None
    line: int
    status: ImportStatus = "not_checked"
    #: For ``from M import N``: whether N was found in M (submodule or binding),
    #: absent, or could not be read (extension/builtin module, star import...).
    symbol_status: SymbolStatus | None = None
    #: Inside ``try: ... except ImportError`` — an optional import, listed but
    #: never invalidating.
    guarded: bool = False


class StaticAnalysis(BaseModel):
    valid: bool
    issues: list[dict[str, Any]] = Field(default_factory=list)
    overall_score: float
    total_checks: int = 0
    passed_checks: int = 0
    imports: list[ImportCheck] = Field(default_factory=list)
    unresolved_imports: list[str] = Field(default_factory=list)
    import_resolution: Literal["workspace", "not_computed"] = "not_computed"
    not_computed_reason: str | None = "no workspace"


class MultiLineValidationResponse(BaseModel):
    valid: bool
    verdict: Literal["valid", "invalid", "not_computed"]
    static_analysis: StaticAnalysis
    execution_analysis: ExecutionAnalysis | None = None
    overall_confidence: float
    suggestions: list[str] = Field(default_factory=list)
    validation_summary: str = ""
    package_info: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Import resolution program
# ---------------------------------------------------------------------------

# Runs inside the workspace as ``python3 -I -c _RESOLVER_PROGRAM [--root R]
# <item> ...``. ``-I`` keeps the repository (the cwd) off the program's own
# ``sys.path``, so a repo-level ``json.py`` cannot shadow the program's imports.
# It is a CONSTANT: the caller's module and symbol names arrive only as argv
# items (``module`` or ``module:symbol``, built from AST identifiers) and are
# never interpolated into source. ``--root`` (a validated absolute path) comes
# before the items. Prints one JSON object; any failure exits non-zero and the
# router reports ``not_computed`` rather than a pass.
#
# It LOCATES modules and never imports them: ``PathFinder.find_spec`` per
# segment with an explicit search path (a dotted ``importlib.util.find_spec``
# would import every parent). It must also pass the runtime's
# command policy (``check_dangerous_patterns``) — a unit test pins that.
#
# Interpreter: the allowlist only admits ``python3``/``python`` as argv[0], yet
# the venv's compiled extensions only match its own interpreter's ABI tags. So
# when ``<venv>/bin/python[3]`` exists and is not the running interpreter, the
# body re-execs itself under it (``os.execv``, same program and args plus a
# leading ``--no-reexec`` so it can never loop). If that is impossible it stays
# put, reports ``"interpreter": "system-fallback"`` and turns every would-be
# ``unresolved``/``unresolved_symbol`` into ``unverifiable`` — never a false fail.
_RESOLVER_BODY = r'''
import ast, configparser, glob, json, os, re, site, sys, sysconfig, types
from importlib.machinery import FrozenImporter, PathFinder

ALIASES = {
    "pyyaml": "yaml", "pillow": "pil", "beautifulsoup4": "bs4",
    "scikit_learn": "sklearn", "python_dateutil": "dateutil",
    "opencv_python": "cv2", "opencv_python_headless": "cv2",
    "opencv_contrib_python": "cv2", "opencv_contrib_python_headless": "cv2",
    "protobuf": "google", "attrs": "attr", "pyjwt": "jwt",
    "psycopg_binary": "psycopg", "psycopg2_binary": "psycopg2",
    "typing_extensions": "typing_extensions", "python_dotenv": "dotenv",
    "pymupdf": "fitz",
}
NAME_RE = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
QUOTED_RE = re.compile(r"\"([^\"]*)\"|'([^']*)'")


def norm(s):
    return s.lower().replace("-", "_").replace(".", "_")


def import_names(dist):
    n = norm(dist)
    out = {n}
    if n in ALIASES:
        out.add(norm(ALIASES[n]))
    for prefix in ("python_", "py"):
        if n.startswith(prefix) and len(n) > len(prefix):
            out.add(n[len(prefix):])
    return out


def real_dirs(paths):
    out = []
    for p in paths:
        if p and os.path.isdir(p):
            rp = os.path.realpath(p)
            if rp not in out:
                out.append(rp)
    return out


def under(path, bases):
    path = os.path.realpath(path)
    return any(path == b or path.startswith(b.rstrip(os.sep) + os.sep) for b in bases)


# -- module-level bindings, read from source -------------------------------

def bindings(path):
    """Module-level bindings of a source file, or None when it cannot be read.

    ``star`` (``from x import *``), ``getattr`` (module ``__getattr__``) and
    ``sysmodules`` (writes to ``sys.modules``) mark names that may exist
    without being bound here, so their absence proves nothing."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        tree = ast.parse(text)
    except (OSError, SyntaxError, ValueError):
        return None
    names = set()
    state = {"star": False}
    alias_t = getattr(ast, "TypeAlias", None)
    match_t = getattr(ast, "Match", None)
    scope_t = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

    def targets(t):
        for x in ast.walk(t):
            if isinstance(x, ast.Name):
                names.add(x.id)

    def walrus(node):
        stack = [node]
        while stack:
            n = stack.pop()
            if isinstance(n, scope_t):
                continue
            if isinstance(n, ast.NamedExpr):
                targets(n.target)
            stack.extend(ast.iter_child_nodes(n))

    def pattern(p):
        for x in ast.walk(p):
            for attr in ("name", "rest"):
                v = getattr(x, attr, None)
                if isinstance(v, str):
                    names.add(v)

    def stmt(n):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
            for d in n.decorator_list:
                walrus(d)
            return
        if isinstance(n, ast.Import):
            for a in n.names:
                names.add(a.asname or a.name.split(".")[0])
            return
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name == "*":
                    state["star"] = True
                else:
                    names.add(a.asname or a.name)
            return
        if isinstance(n, ast.Assign):
            for t in n.targets:
                targets(t)
        elif isinstance(n, ast.AnnAssign):
            if n.value is not None:
                targets(n.target)
        elif isinstance(n, ast.AugAssign):
            targets(n.target)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            targets(n.target)
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            for item in n.items:
                if item.optional_vars is not None:
                    targets(item.optional_vars)
        elif alias_t is not None and isinstance(n, alias_t):
            targets(n.name)
        elif match_t is not None and isinstance(n, match_t):
            for case in n.cases:
                pattern(case.pattern)
        nested = ("body", "orelse", "finalbody", "handlers", "cases")
        for field, value in ast.iter_fields(n):
            if field in nested:
                continue
            for v in value if isinstance(value, list) else [value]:
                if isinstance(v, ast.AST):
                    walrus(v)
        for field in ("body", "orelse", "finalbody"):
            for child in getattr(n, field, None) or []:
                stmt(child)
        for h in getattr(n, "handlers", None) or []:
            for child in h.body:
                stmt(child)
        for c in getattr(n, "cases", None) or []:
            for child in c.body:
                stmt(child)

    for n in tree.body:
        stmt(n)
    for x in ast.walk(tree):
        if isinstance(x, ast.Global):
            names.update(x.names)
    return {
        "names": names,
        "star": state["star"],
        "getattr": "__getattr__" in names,
        "sysmodules": re.search(r"sys\.modules\s*(\[|\.update|\.setdefault)", text) is not None,
    }


# -- declared dependencies ---------------------------------------------------

def toml_fallback(text):
    """Dependency strings from pyproject text when tomllib is unavailable."""
    out = []
    section = ""
    in_array = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        bare = QUOTED_RE.sub("", line)
        head = re.match(r"^\[\[?([^\]\"']+)\]\]?$", line)
        if head and not in_array:
            section = head.group(1).strip()
            continue
        if section.startswith("tool.poetry") and section.endswith("dependencies"):
            key = re.match(r"^[\"']?([A-Za-z0-9][A-Za-z0-9._-]*)[\"']?\s*=", line)
            if key:
                out.append(key.group(1))
            continue
        group = section in ("project.optional-dependencies", "dependency-groups")
        if section == "project" and re.match(r"^dependencies\s*=\s*\[", line):
            in_array = True
        if in_array or group:
            out.extend(a or b for a, b in QUOTED_RE.findall(line))
        if in_array and "]" in bare.split("=", 1)[-1]:
            in_array = False
    return out


def requirement_specs(path, depth):
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    for raw in lines:
        line = raw.strip()
        egg = re.search(r"[#&]egg=([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if egg:
            out.append(egg.group(1))
            continue
        line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
        if not line or line.startswith("#"):
            continue
        inc = re.match(r"^(?:-r|--requirement)[\s=]+(\S+)", line)
        if inc:
            if depth == 0:
                out.extend(requirement_specs(os.path.join(os.path.dirname(path), inc.group(1)), 1))
            continue
        if line.startswith("-") or re.match(r"^[A-Za-z][\w+.-]*://", line):
            continue
        out.append(line)
    return out


def declared(root):
    """(import-name set, ok). ok=False when a manifest exists but could not be read."""
    specs = []
    ok = True
    pyproject = os.path.join(root, "pyproject.toml")
    if os.path.isfile(pyproject):
        try:
            with open(pyproject, "rb") as fh:
                raw = fh.read()
            data = None
            try:
                import tomllib
                data = tomllib.loads(raw.decode("utf-8", "replace"))
            except Exception:
                data = None
            if data is None:
                specs.extend(toml_fallback(raw.decode("utf-8", "replace")))
            else:
                proj = data.get("project") or {}
                specs.extend(proj.get("dependencies") or [])
                for group in (proj.get("optional-dependencies") or {}).values():
                    specs.extend(group or [])
                for group in (data.get("dependency-groups") or {}).values():
                    specs.extend(s for s in group or [] if isinstance(s, str))
                poetry = (data.get("tool") or {}).get("poetry") or {}
                specs.extend(poetry.get("dependencies") or {})
                specs.extend(poetry.get("dev-dependencies") or {})
                for group in (poetry.get("group") or {}).values():
                    specs.extend((group or {}).get("dependencies") or {})
        except Exception:
            ok = False
    setup_cfg = os.path.join(root, "setup.cfg")
    if os.path.isfile(setup_cfg):
        try:
            cp = configparser.ConfigParser(interpolation=None)
            cp.read(setup_cfg, encoding="utf-8")
            if cp.has_section("options"):
                specs.extend(cp.get("options", "install_requires", fallback="").splitlines())
            if cp.has_section("options.extras_require"):
                for _, value in cp.items("options.extras_require"):
                    specs.extend(value.splitlines())
        except Exception:
            ok = False
    setup_py = os.path.join(root, "setup.py")
    if os.path.isfile(setup_py):
        try:
            with open(setup_py, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            if "install_requires" in text:
                m = re.search(
                    r"install_requires\s*=\s*\[((?:[^\]\"']|\"[^\"]*\"|'[^']*')*)\]", text
                )
                if m:
                    specs.extend(a or b for a, b in QUOTED_RE.findall(m.group(1)))
                else:
                    ok = False
        except OSError:
            ok = False
    try:
        listing = sorted(os.listdir(root))
    except OSError:
        listing = []
    for fn in listing:
        if fn.startswith("requirements") and fn.endswith(".txt"):
            try:
                specs.extend(requirement_specs(os.path.join(root, fn), 0))
            except OSError:
                ok = False
    out = set()
    for spec in specs:
        m = NAME_RE.match(spec) if isinstance(spec, str) else None
        if m:
            out |= import_names(m.group(1))
    return out, ok


# -- search path ---------------------------------------------------------------

def find_venv(root):
    """(venv dir, its site-packages dirs) of the first workspace venv, else (None, [])."""
    for venv in (os.path.join(root, ".venv"), "/workspace/.venv"):
        if os.path.isdir(venv):
            sites = real_dirs(
                sorted(glob.glob(os.path.join(venv, "lib*", "python*", "site-packages")))
                + [os.path.join(venv, "Lib", "site-packages")]
            )
            if sites:
                return venv, sites
    return None, []


def venv_python(venv):
    """(path, runnable) of the venv's interpreter, or (None, False) when it has none."""
    if venv is None:
        return None, False
    for name in ("python3", "python"):
        py = os.path.join(venv, "bin", name)
        if os.path.lexists(py):
            return py, os.path.isfile(py) and os.access(py, os.X_OK)
    return None, False


def same_file(a, b):
    try:
        return os.path.realpath(a) == os.path.realpath(b) or os.path.samefile(a, b)
    except OSError:
        return False


def search_dirs(root):
    """Lookup order as at runtime: the repo root, the stdlib, then ``src/``
    (editable / src-layout paths sit after the stdlib), then site-packages."""
    repo = real_dirs([root])
    src = real_dirs([os.path.join(root, "src")])
    paths = sysconfig.get_paths()
    std_bases = [paths.get("stdlib"), paths.get("platstdlib")]
    std = real_dirs(std_bases + [os.path.join(b, "lib-dynload") for b in std_bases if b])
    _, sites = find_venv(root)
    site_source = "venv" if sites else "system"
    if not sites:
        cands = []
        try:
            cands.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            cands.append(site.getusersitepackages())
        except Exception:
            pass
        sites = real_dirs(cands)
    extra = []
    editable = {}
    for sd in sites:
        for pth in sorted(glob.glob(os.path.join(sd, "*.pth"))):
            try:
                with open(pth, encoding="utf-8", errors="replace") as fh:
                    lines = fh.read().splitlines()
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if line and not line.startswith(("#", "import ", "import\t")):
                    extra.append(line if os.path.isabs(line) else os.path.join(sd, line))
        for finder in sorted(glob.glob(os.path.join(sd, "__editable__*finder.py"))):
            try:
                with open(finder, encoding="utf-8", errors="replace") as fh:
                    tree = ast.parse(fh.read())
            except (OSError, SyntaxError, ValueError):
                continue
            for n in tree.body:
                tgt = n.targets[0] if isinstance(n, ast.Assign) else getattr(n, "target", None)
                value = getattr(n, "value", None)
                if isinstance(tgt, ast.Name) and tgt.id == "MAPPING" and value is not None:
                    try:
                        mapping = ast.literal_eval(value)
                    except ValueError:
                        continue
                    if isinstance(mapping, dict):
                        editable.update(
                            (k, v) for k, v in mapping.items()
                            if isinstance(k, str) and isinstance(v, str)
                        )
    return repo, src, std, sites, real_dirs(extra), editable, site_source


# -- resolution ----------------------------------------------------------------

MISSING = object()


def find(name, locations):
    """PathFinder lookup that never imports. A nested namespace package needs its
    parent in sys.modules, so a stub carrying only __path__ stands in for it."""
    parent = name.rpartition(".")[0]
    saved = MISSING
    if parent:
        saved = sys.modules.get(parent, MISSING)
        stub = types.ModuleType(parent)
        stub.__path__ = list(locations)
        sys.modules[parent] = stub
    try:
        spec = PathFinder.find_spec(name, list(locations))
        locs = None
        if spec is not None and spec.submodule_search_locations is not None:
            locs = list(spec.submodule_search_locations)
        return spec, locs
    except (ImportError, ValueError, KeyError, OSError):
        return None, None
    finally:
        if parent:
            if saved is MISSING:
                sys.modules.pop(parent, None)
            else:
                sys.modules[parent] = saved


def source_of(spec):
    origin = getattr(spec, "origin", None) if spec is not None else None
    if origin and getattr(spec, "has_location", False) and origin.endswith(".py"):
        return origin
    return None


def is_namespace(spec):
    return spec is not None and getattr(spec, "origin", None) in (None, "namespace")


class Resolver(object):
    def __init__(self, root, fallback=False):
        self.root = os.path.realpath(root)
        (repo, src, self.std, self.sites, self.extra,
         self.editable, self.site_source) = search_dirs(self.root)
        self.repo = repo + [d for d in src if d not in repo]
        self.path = repo + self.std + src + self.sites + self.extra
        # Running under another interpreter than the venv's: its compiled
        # extensions carry a foreign ABI tag the path finder will not match,
        # so "not found" is no evidence of absence.
        self.fallback = fallback
        self.std_names = set(getattr(sys, "stdlib_module_names", ()))
        self.decl, self.decl_ok = declared(self.root)

    def where(self, spec, locs):
        origin = getattr(spec, "origin", None)
        if origin in ("built-in", "frozen"):
            return "stdlib"
        if origin and getattr(spec, "has_location", False):
            paths = [origin]
        else:
            paths = list(locs or [])
        classes = set()
        for p in paths:
            if under(p, self.sites):
                classes.add("installed")
            elif under(p, self.std):
                classes.add("stdlib")
            elif under(p, self.repo):
                classes.add("repo")
            else:
                classes.add("installed")
        for c in ("installed", "stdlib", "repo"):
            if c in classes:
                return c
        return "installed"

    def repo_namespace_only(self, spec, locs):
        return (
            is_namespace(spec)
            and bool(locs)
            and all(under(p, self.repo) and not under(p, self.sites) for p in locs)
        )

    def top(self, name):
        """(class, spec, locs, readable). class None = not found."""
        if name in sys.builtin_module_names:
            return "stdlib", None, None, False
        if name in self.std_names and FrozenImporter.find_spec(name) is not None:
            spec, locs = find(name, self.std)
            return "stdlib", spec, locs, spec is not None
        spec, locs = find(name, self.path)
        if spec is None and name in self.editable:
            spec, locs = find(name, [os.path.dirname(self.editable[name])])
            if spec is not None:
                return "installed", spec, locs, True
        if spec is None:
            return None, None, None, False
        return self.where(spec, locs), spec, locs, True

    def info(self, spec):
        src = source_of(spec)
        return bindings(src) if src else None

    def segment(self, fullname, spec, locs, readable):
        """A dotted MODULE path segment: found (spec, locs) / bound / missing /
        unverifiable. ``import a.b`` needs ``a.b`` to be a module: a submodule,
        or a name the parent both binds and registers in sys.modules (os.path)."""
        if not readable:
            return "unverifiable", None, None
        if locs is not None:
            sub, sub_locs = find(fullname, locs)
            if sub is not None:
                return "found", sub, sub_locs
        if is_namespace(spec):
            return "missing", None, None
        info = self.info(spec)
        if info is None:
            return "unverifiable", None, None
        if info["sysmodules"] and fullname.rpartition(".")[2] in info["names"]:
            return "bound", None, None
        return "missing", None, None

    def symbol(self, fullname, spec, locs, readable):
        """``from M import N``: N is a submodule of M or bound in M's source."""
        if not readable:
            return "unverifiable"
        if locs is not None and find(fullname, locs)[0] is not None:
            return "found"
        if is_namespace(spec):
            return "missing"
        info = self.info(spec)
        if info is None:
            return "unverifiable"
        if fullname.rpartition(".")[2] in info["names"]:
            return "found"
        if info["star"] or info["getattr"] or info["sysmodules"]:
            return "unverifiable"
        return "missing"

    def absent(self, top):
        if norm(top) in self.decl:
            return "declared_not_installed"
        if not self.decl_ok:
            return "unverifiable"
        return "unresolved"

    def resolve(self, item):
        status, symbol = self.locate(item)
        if status in ("unresolved", "unresolved_symbol") and (
            self.fallback or self.foreign_extension(item)
        ):
            return "unverifiable", None
        return status, symbol

    def foreign_extension(self, item):
        # A compiled extension for another ABI (e.g. a venv without its own
        # interpreter, built for another Python) is invisible to this path
        # finder. If a file with the module's dotted path and an extension
        # suffix exists in site-packages, "not found" is no evidence of absence.
        mod, _, sym = item.partition(":")
        names = [mod] + ([mod + "." + sym] if sym else [])
        for name in names:
            parts = name.split(".")
            for site in self.sites:
                base = os.path.join(site, *parts)
                for pattern in (base + ".*.so", base + ".*.pyd", base + ".so", base + ".pyd"):
                    if glob.glob(pattern):
                        return True
        return False

    def locate(self, item):
        mod, _, sym = item.partition(":")
        parts = mod.split(".")
        cls, spec, locs, readable = self.top(parts[0])
        if cls is None:
            return self.absent(parts[0]), None
        # A repo directory that is only a namespace portion (an alembic/
        # migrations folder) is not evidence the package exists: if the chain
        # fails below it, the import is judged as if the top were absent.
        ns_only = self.repo_namespace_only(spec, locs)
        name = parts[0]
        for seg in parts[1:]:
            name = name + "." + seg
            state, sub, sub_locs = self.segment(name, spec, locs, readable)
            if state == "missing":
                return (self.absent(parts[0]) if ns_only else "unresolved"), None
            if state != "found":
                # Bound-and-registered (os.path) or unreadable: the module path
                # stands, nothing below it can be followed.
                return cls, ("unverifiable" if sym else None)
            spec, locs = sub, sub_locs
            cls = self.where(spec, locs)
            ns_only = ns_only and self.repo_namespace_only(spec, locs)
        if not sym:
            return cls, None
        if sym == "*":
            return cls, "found"
        state = self.symbol(mod + "." + sym, spec, locs, readable)
        if state == "missing":
            if ns_only:
                return self.absent(parts[0]), None
            if is_namespace(spec):
                return "unresolved", None
            return "unresolved_symbol", "missing"
        return cls, state


def bootstrap(body):
    return "S = " + repr(body) + "\nexec(S)\n"


def main(argv):
    args = list(argv)
    reexec = True
    if argv[:1] == ["--no-reexec"]:
        reexec, argv = False, argv[1:]
    root = os.getcwd()
    if argv[:1] == ["--root"]:
        root, argv = argv[1], argv[2:]
    # The venv's site-packages hold extensions built for the venv's Python
    # (pydantic_core, yaml._yaml), which only that interpreter's path finder
    # recognises: run under it. --no-reexec makes a second hop impossible.
    py, runnable = venv_python(find_venv(os.path.realpath(root))[0])
    interpreter = "system"
    if py is not None:
        if same_file(py, sys.executable):
            interpreter = "venv"
        else:
            if reexec and runnable:
                try:
                    os.execv(py, [py, "-I", "-c", bootstrap(S), "--no-reexec"] + args)
                except OSError as exc:
                    sys.stderr.write("resolver: cannot run %s: %r\n" % (py, exc))
            interpreter = "system-fallback"
    r = Resolver(root, fallback=interpreter == "system-fallback")
    results = []
    for item in argv:
        try:
            status, symbol = r.resolve(item)
        except Exception as exc:
            status, symbol = "unverifiable", None
            sys.stderr.write("resolver: %s: %r\n" % (item, exc))
        results.append({"item": item, "status": status, "symbol": symbol})
    print(json.dumps({
        "interpreter": interpreter,
        "executable": sys.executable,
        "python_version": "%d.%d" % sys.version_info[:2],
        "site_packages": r.site_source,
        "root": r.root,
        "results": results,
    }))


if __name__ == "__main__":
    main(sys.argv[1:])
'''
# What is sent: a two-line bootstrap that binds the body to ``S`` and runs it,
# so the body can re-exec itself under the venv interpreter (see ``main``).
_RESOLVER_PROGRAM = "S = " + repr(_RESOLVER_BODY) + "\nexec(S)\n"


class WorkspaceExecError(RuntimeError):
    """The workspace could not run the command (backend down, pod gone)."""


class WorkspaceCommandRefusedError(WorkspaceExecError):
    """The runtime's command policy refused the argv (400). Says nothing about
    the code: the policy substring-scans argv, so ``executor.shutdown()`` in a
    snippet is enough to be refused."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _handler_guards_import(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    elts = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(e, ast.Name) and e.id in _IMPORT_GUARD_EXCEPTIONS for e in elts)


def _is_type_checking(test: ast.expr) -> bool:
    """``if TYPE_CHECKING:`` / ``if typing.TYPE_CHECKING:`` (never run)."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _suppresses_import_error(item: ast.withitem) -> bool:
    """``with suppress(ImportError, ...)`` / ``with contextlib.suppress(...)``."""
    call = item.context_expr
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
    return name == "suppress" and any(
        isinstance(a, (ast.Name, ast.Attribute))
        and (a.id if isinstance(a, ast.Name) else a.attr) in _IMPORT_GUARD_EXCEPTIONS
        for a in call.args
    )


def _collect_imports(tree: ast.AST) -> list[ImportCheck]:
    """Every import in the snippet, in source order. Relative imports cannot be
    resolved without knowing the snippet's package, so they are ``unverifiable``."""
    found: list[ImportCheck] = []

    def visit(node: ast.AST, guarded: bool) -> None:
        if isinstance(node, (ast.Try, ast.TryStar)):
            body_guarded = guarded or any(_handler_guards_import(h) for h in node.handlers)
            for stmt in node.body:
                visit(stmt, body_guarded)
            for part in (node.handlers, node.orelse, node.finalbody):
                for stmt in part:
                    visit(stmt, guarded)
            return
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            for stmt in node.body:
                visit(stmt, True)
            for stmt in node.orelse:
                visit(stmt, guarded)
            return
        if isinstance(node, (ast.With, ast.AsyncWith)) and any(
            _suppresses_import_error(i) for i in node.items
        ):
            for stmt in node.body:
                visit(stmt, True)
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append(ImportCheck(module=alias.name, line=node.lineno, guarded=guarded))
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            status: ImportStatus = "unverifiable" if node.level else "not_checked"
            for alias in node.names:
                found.append(
                    ImportCheck(
                        module=module,
                        name=alias.name,
                        line=node.lineno,
                        status=status,
                        guarded=guarded,
                    )
                )
        for child in ast.iter_child_nodes(node):
            visit(child, guarded)

    visit(tree, False)
    return found


def _resolver_item(check: ImportCheck) -> str:
    return f"{check.module}:{check.name}" if check.name else check.module


def _import_label(check: ImportCheck) -> str:
    return f"{check.module}.{check.name}" if check.name else check.module


def _finalise(static: StaticAnalysis) -> StaticAnalysis:
    """Recompute validity, issues and score from the syntax result + imports."""
    syntax_issues = [i for i in static.issues if i.get("type") == "syntax"]
    failing = [i for i in static.imports if i.status in _FAILING_STATUSES and not i.guarded]
    checked = [
        i for i in static.imports if i.status in _RESOLVER_STATUSES and i.status != "unverifiable"
    ]

    static.unresolved_imports = [_import_label(i) for i in failing]
    static.issues = syntax_issues + [
        {
            "type": "import",
            "line": i.line,
            "message": (
                f"'{i.name}' is not defined in module '{i.module}'"
                if i.status == "unresolved_symbol"
                else f"Module '{i.module}' not found in the workspace repo, "
                "stdlib, installed or declared packages"
            ),
            "severity": "error",
            "valid": False,
        }
        for i in failing
    ]
    static.total_checks = 1 + len(checked)
    static.passed_checks = (0 if syntax_issues else 1) + len(checked) - len(failing)
    static.overall_score = static.passed_checks / static.total_checks
    static.valid = not syntax_issues and not failing
    return static


def _static_analysis(code: str) -> StaticAnalysis:
    """Syntax check via ast.parse, plus the snippet's imports (not yet resolved)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return StaticAnalysis(
            valid=False,
            issues=[
                {
                    "type": "syntax",
                    "line": e.lineno,
                    "message": str(e.msg),
                    "severity": "error",
                    "valid": False,
                }
            ],
            overall_score=0.0,
            total_checks=1,
            passed_checks=0,
        )
    return _finalise(StaticAnalysis(valid=True, overall_score=1.0, imports=_collect_imports(tree)))


def _http_reason(exc: HTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, dict):
        detail = detail.get("message") or detail.get("detail") or detail
    return f"{exc.status_code}: {detail}"


async def _run_in_workspace(
    name: str, tenant_id: str, argv: list[str], timeout: int, cwd: str | None = None
) -> RunOnceResult:
    """Run ``argv`` once in the workspace (``cwd`` or its repository root).

    Same runtime path as ``POST /api/containers/{name}/exec``
    (``run_workspace_command``). This route reports failures in its body, so a
    policy refusal (400) becomes ``WorkspaceCommandRefusedError`` and every
    other backend fault ``WorkspaceExecError``. Callers MUST have run
    ``authorize_workspace`` first.
    """
    try:
        _, result = await run_workspace_command(name, tenant_id, argv, cwd=cwd, timeout=timeout)
    except HTTPException as exc:
        if exc.status_code == 400:
            raise WorkspaceCommandRefusedError(_http_reason(exc)) from exc
        raise WorkspaceExecError(_http_reason(exc)) from exc
    return result


def _resolver_argv(items: list[str], root: str | None) -> list[str]:
    root_args = ["--root", root] if root else []
    return ["python3", "-I", "-c", _RESOLVER_PROGRAM, *root_args, *items]


async def _resolve_imports(
    static: StaticAnalysis, container_name: str, tenant_id: str, root: str | None = None
) -> StaticAnalysis:
    """Classify every absolute import in ONE workspace exec.

    Never a silent pass: if the exec or its output fails, the result stays
    ``not_computed`` with the reason, and ``valid`` reflects syntax only.
    """
    # ⭐ AUTHORIZATION FIRST — before anything is sent to the workspace. See
    # ``_execute_in_container`` for why it must sit outside any ``except``.
    authorize_workspace(container_name, tenant_id, route=ROUTE)

    pending = [i for i in static.imports if i.status == "not_checked"]
    if not pending:
        if static.imports:
            # Only relative imports: nothing was checked, so nothing may pass as checked.
            static.import_resolution = "not_computed"
            static.not_computed_reason = "only relative imports"
        else:
            static.import_resolution = "workspace"
            static.not_computed_reason = None
        return _finalise(static)

    items = sorted({_resolver_item(i) for i in pending})
    reason: str | None = None
    results: dict[str, dict[str, Any]] = {}
    try:
        result = await _run_in_workspace(
            container_name, tenant_id, _resolver_argv(items, root), _RESOLVE_TIMEOUT, cwd=root
        )
        if result.exit_code != 0:
            reason = f"resolver exited {result.exit_code}: {result.stderr.strip()[-300:]}"
        elif result.truncated:
            reason = "resolver output truncated"
        else:
            payload = json.loads(result.stdout)
            results = {r["item"]: r for r in payload["results"]}
            if any(
                results.get(it, {}).get("status") not in _RESOLVER_STATUSES
                or results[it].get("symbol") not in _SYMBOL_STATUSES
                for it in items
            ):
                reason = "resolver returned an incomplete result"
    except WorkspaceCommandRefusedError as exc:
        reason = f"{_REFUSED}: {exc}"
    except WorkspaceExecError as exc:
        reason = f"workspace exec failed: {exc}"
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        reason = f"unparseable resolver output: {exc}"

    if reason is not None:
        logger.warning("Import resolution not computed for %s: %s", container_name, reason)
        static.import_resolution = "not_computed"
        static.not_computed_reason = reason
        return _finalise(static)

    for check in pending:
        entry = results[_resolver_item(check)]
        # Membership in both status sets was checked above, so the casts are sound.
        check.status = cast(ImportStatus, entry["status"])
        check.symbol_status = cast("SymbolStatus | None", entry.get("symbol"))
    static.import_resolution = "workspace"
    static.not_computed_reason = None
    return _finalise(static)


# Exits 0 when argv[1] is a directory, 3 when it is not. Constant source; the
# path is an argv item.
_ROOT_PROBE = "import os, sys; sys.exit(0 if os.path.isdir(sys.argv[1]) else 3)"


async def _root_missing(name: str, tenant_id: str, root: str) -> bool:
    """True only when the workspace CONFIRMS ``root`` is not a directory.

    Asked after a failed run, from ``/`` (which always exists), because every
    backend enters ``cwd`` before the command and reports a failed ``cd`` as
    an ordinary non-zero exit. Any doubt (probe failed or refused) is False:
    the original result then stands as reported.
    """
    try:
        probe = await _run_in_workspace(
            name, tenant_id, ["python3", "-I", "-c", _ROOT_PROBE, root], _RESOLVE_TIMEOUT, "/"
        )
    except WorkspaceExecError:
        return False
    return probe.exit_code == 3


async def _execute_in_container(
    code: str,
    container_name: str | None,
    timeout: int,
    tenant_id: str,
    *,
    root: str | None = None,
) -> ExecutionAnalysis:
    """Execute code inside the caller's workspace via the runtime layer.

    ``tenant_id`` is REQUIRED and positional-by-convention rather than defaulted:
    this helper runs caller-supplied Python inside a caller-named workspace, and a
    default would silently restore the #135 gap for any future call site that
    forgot to pass one. There is no "no tenant" case — the only route that reaches
    here binds ``Depends(require_tenant)``.

    A run the workspace refused or could not perform is ``execution_status:
    "not_computed"`` — it says nothing about the code, so it must not fail it.
    """
    # ⭐ AUTHORIZATION, and it is deliberately the FIRST statement — ahead of the
    # try below, whose `except` would otherwise convert a refusal into a 200 with
    # `execution_status: not_computed`. That is not a stylistic preference: this
    # helper's contract is to swallow execution faults and report them in the
    # body, so a gate placed inside it would be reported as an unrun snippet
    # rather than a refused caller, and the caller would keep its foreign
    # workspace name. `not_found` raises HTTPException, which is in neither this
    # helper's `except` tuple nor the route's, so it propagates as a real 404.
    #
    # `name` is resolved BEFORE the check so the workspace that is authorized is
    # the exact one that is dispatched to — including the `or "default"` fallback,
    # which is a real workspace name and must be owned like any other.
    name = container_name or "default"
    authorize_workspace(name, tenant_id, route=ROUTE)

    start = time.monotonic()
    try:
        result = await _run_in_workspace(name, tenant_id, ["python", "-c", code], timeout, root)
    except WorkspaceExecError as exc:
        why = _REFUSED if isinstance(exc, WorkspaceCommandRefusedError) else "workspace exec failed"
        logger.warning("Execution in workspace not computed (%s): %s", why, exc)
        return ExecutionAnalysis(
            execution_successful=False,
            execution_status="not_computed",
            not_computed_reason=f"{why}: {exc}",
            stderr=str(exc),
            runtime_errors=[str(exc)],
            execution_time=time.monotonic() - start,
        )

    if result.exit_code != 0 and root is not None and await _root_missing(name, tenant_id, root):
        # The runtime could not enter ``root`` (``cd`` failed), so the snippet
        # never ran: a statement about the path, not about the code.
        reason = f"codebase_path {root} does not exist in the workspace"
        logger.warning("Execution in workspace not computed: %s", reason)
        return ExecutionAnalysis(
            execution_successful=False,
            execution_status="not_computed",
            not_computed_reason=reason,
            stderr=result.stderr,
            runtime_errors=[reason],
            execution_time=time.monotonic() - start,
        )

    errors: list[str] = []
    if result.exit_code != 0:
        errors.append(result.stderr.strip() if result.stderr else f"Exit code {result.exit_code}")
    warnings = ["output truncated"] if result.truncated else []
    if result.timed_out:
        warnings.append(f"timed out after {timeout}s")
    return ExecutionAnalysis(
        execution_successful=result.exit_code == 0,
        stdout=result.stdout,
        stderr=result.stderr,
        execution_time=result.duration or (time.monotonic() - start),
        runtime_errors=errors,
        warnings=warnings,
    )


def _imports_unchecked(static: StaticAnalysis) -> bool:
    """True when some non-optional import was not (fully) checked."""
    return any(i.status in _UNCHECKED_STATUSES for i in static.imports if not i.guarded)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/multi-line-code-snippet",
    response_model=MultiLineValidationResponse,
    operation_id="validate_multi_line_code_snippet",
    summary="Validate Multi-Line Code Snippet",
)
async def validate_multi_line_code_snippet(
    request: MultiLineCodeSnippetRequest,
    tenant_id: str = Depends(require_tenant),
):
    """Validate a multi-line code snippet: syntax, import resolution against the
    named workspace, and optional execution."""
    try:
        static = _static_analysis(request.code_snippet)
        syntax_ok = not any(i.get("type") == "syntax" for i in static.issues)

        resolution_attempted = False
        if request.container_name:
            if syntax_ok:
                resolution_attempted = True
                static = await _resolve_imports(
                    static, request.container_name, tenant_id, request.codebase_path
                )
            else:
                static.not_computed_reason = "syntax error"

        execution: ExecutionAnalysis | None = None
        if request.include_execution_validation:
            execution = await _execute_in_container(
                request.code_snippet,
                request.container_name,
                request.execution_timeout,
                tenant_id,
                root=request.codebase_path,
            )
        executed = execution is not None and execution.execution_status == "computed"

        # Overall result. An execution that never ran cannot fail the code.
        overall_valid = static.valid and (
            not executed or execution is None or execution.execution_successful
        )
        confidence = static.overall_score
        if executed and execution is not None:
            confidence = (confidence + (1.0 if execution.execution_successful else 0.0)) / 2

        if not overall_valid:
            verdict = "invalid"
        elif _imports_unchecked(static) or (execution is not None and not executed):
            verdict = "not_computed"
        else:
            verdict = "valid"

        suggestions: list[str] = []
        if not syntax_ok:
            suggestions.append("Fix syntax errors before further validation")
        if static.unresolved_imports:
            suggestions.append(
                "Fix unresolved imports (not in the workspace repo, stdlib, installed or "
                "declared packages): " + ", ".join(static.unresolved_imports)
            )
        if resolution_attempted and static.import_resolution == "not_computed":
            suggestions.append(f"Imports were NOT checked: {static.not_computed_reason}")
        if execution is not None and not executed:
            suggestions.append(f"Execution was NOT run: {execution.not_computed_reason}")
        if executed and execution is not None and not execution.execution_successful:
            suggestions.append("Fix runtime errors found during execution")

        summary_parts = ["Static: PASS" if static.valid else "Static: FAIL"]
        if resolution_attempted:
            if static.import_resolution == "workspace":
                summary_parts.append(
                    "Imports: FAIL" if static.unresolved_imports else "Imports: PASS"
                )
            else:
                summary_parts.append("Imports: NOT CHECKED")
        if execution is not None:
            if not executed:
                summary_parts.append("Execution: NOT RUN")
            else:
                summary_parts.append(
                    "Execution: PASS" if execution.execution_successful else "Execution: FAIL"
                )
        validation_summary = " | ".join(summary_parts)

        return MultiLineValidationResponse(
            valid=overall_valid,
            verdict=verdict,
            static_analysis=static,
            execution_analysis=execution,
            overall_confidence=confidence,
            suggestions=suggestions,
            validation_summary=validation_summary,
            package_info={
                "packages": [],
                "modules_index": {},
                "total_packages": 0,
                "standard_library_modules": 0,
                "total_importable_modules": 0,
            },
        )
    except (RuntimeError, OSError, ValueError, TypeError) as e:
        logger.error("Multi-line validation failed: %s", e, exc_info=True)
        raise internal_error(f"Validation failed: {e}") from e

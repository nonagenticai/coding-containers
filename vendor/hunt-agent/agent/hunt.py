"""Bug-hunt agent: scans a repository's committed HEAD INSIDE the tenant workspace.

coding-analysis never copies the tenant's code out of the workspace. It execs
this script there (``python3 -I -P /opt/hunt-agent/hunt.py --repo <root> --out
<file>``) through coding-containers, and only the findings JSON comes back.

SELF-CONTAINED BY CONTRACT. The workspace image has no coding-analysis package,
so this file imports the stdlib and ``tree_sitter`` (+ grammar packages) only,
and it is one file because ``-P`` keeps the script's own directory off
``sys.path``. coding-analysis itself imports the scope and detection logic from
HERE, so the service and the agent cannot drift apart.

WHAT IS SCANNED: the committed tree at ``HEAD``, never the working tree. Files
are enumerated with ``git ls-tree -r`` at the resolved commit and read with
``git cat-file``, so ``commit_sha`` always names exactly the bytes parsed. An
uncommitted edit is NOT scanned; ``worktree_dirty`` (``git status
--porcelain``, untracked files included) records that the workspace differed
from HEAD when the hunt ran, so a dirty hunt is visibly a HEAD scan.

COVERAGE. ``tracked_source_files`` counts the tracked blobs under the hunt root
that pass the source-file predicate and are not under an excluded directory.
``scanned_files`` counts the subset actually read AND parsed. Both use the SAME
scope (``EXCLUDED_DIRS`` via :func:`is_in_excluded_dir`). The per-language size
cap is NOT a scope rule: an oversized tracked file stays in the denominator and
shows as ``scanned < tracked``, as does a non-UTF-8 file or a file whose
grammar is unavailable. Non-regular tree entries (symlinks, mode 120000, and
submodule gitlinks, mode 160000) are out of scope for BOTH counts and declared
as ``excluded_entry_kinds``: they are not this repository's source bytes. So is
a tracked path that is not valid UTF-8 (kind ``non_utf8_path``): it cannot be
named in a finding or stored by the caller, so it is excluded by the same single
scope test rather than failing the hunt later. ``not_scanned`` lists each
in-scope file that was not parsed, and why.

RECHECK. ``--recheck --commit <sha> --path <rel>`` re-runs the SAME scope test
and the SAME detection (:func:`scan_entry`) on ONE file as it is in that commit's
tree, read with ``git cat-file`` (no checkout, no working-tree read). It answers
"does the detector still flag this at the hunted commit?". Every way the file
could not be examined (unknown commit, path absent or out of scope, unreadable)
is a distinct ``status`` in the result, never an empty finding list.

FINGERPRINT. Every finding carries ``fingerprint`` (:func:`finding_fingerprint`),
a content anchor that survives unrelated edits elsewhere in the file; see there.

AGENT IDENTITY. ``agent_sha`` is the sha256 of :func:`manifest_text` over this
directory's ``*.py`` files, which is byte-identical to the committed
``MANIFEST.sha256``. The caller compares it with the sha it expects and fails
the hunt on any mismatch; ``tree_sitter_version`` / ``grammar_versions`` are
reported for the same reason.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import subprocess  # nosec B404 - fixed argv, no shell
import sys
import tempfile
import time
from pathlib import Path, PurePath, PurePosixPath
from typing import Any

#: Bumped by hand on a contract change; ``agent_sha`` is the real identity.
AGENT_VERSION = "1"
RESULT_SCHEMA = "hunt_agent_result/v1"
RECHECK_SCHEMA = "hunt_agent_recheck/v1"

# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

#: Directory names out of the hunt's scope, for the tracked count AND the scan.
#: Emitted on every result as ``excluded_dirs``. Also the deep analyzer's walk
#: policy (``multi_language_entities`` re-exports it): there is no second list.
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        "target",
        "vendor",
        ".next",
        ".nuxt",
        "out",
        "coverage",
        ".idea",
        ".vscode",
    }
)


#: Tree entry kinds out of the hunt's scope for BOTH counts, by git mode.
#: Emitted on every result as ``excluded_entry_kinds``.
EXCLUDED_ENTRY_MODES: dict[str, str] = {"120000": "symlink", "160000": "submodule"}

#: Entry kind for a tracked path that is not valid UTF-8 (out of BOTH counts).
NON_UTF8_PATH_KIND = "non_utf8_path"

#: Every excluded entry kind, as emitted in ``excluded_entry_kinds``.
EXCLUDED_ENTRY_KINDS: tuple[str, ...] = tuple(
    sorted({*EXCLUDED_ENTRY_MODES.values(), NON_UTF8_PATH_KIND})
)


def is_utf8_path(rel: str) -> bool:
    """False for a path decoded with ``surrogateescape`` from non-UTF-8 bytes."""
    try:
        rel.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def is_in_excluded_dir(rel_path: PurePath, excluded_dirs: frozenset[str]) -> bool:
    """True when a ROOT-RELATIVE path lies under a directory named in ``excluded_dirs``.

    Only directory components are tested (never the file name), and only below
    the root: a hunt root that itself lives under e.g. ``/srv/build/`` is not
    thereby excluded wholesale.
    """
    return any(part in excluded_dirs for part in rel_path.parts[:-1])


#: tree-sitter grammar -> (python module, language factory, distribution name).
#: TypeScript's module exposes ``language_typescript`` rather than ``language``.
GRAMMARS: dict[str, tuple[str, str, str]] = {
    "python": ("tree_sitter_python", "language", "tree-sitter-python"),
    "javascript": ("tree_sitter_javascript", "language", "tree-sitter-javascript"),
    "typescript": ("tree_sitter_typescript", "language_typescript", "tree-sitter-typescript"),
    "go": ("tree_sitter_go", "language", "tree-sitter-go"),
    "java": ("tree_sitter_java", "language", "tree-sitter-java"),
    "rust": ("tree_sitter_rust", "language", "tree-sitter-rust"),
    "cpp": ("tree_sitter_cpp", "language", "tree-sitter-cpp"),
    "c_sharp": ("tree_sitter_c_sharp", "language", "tree-sitter-c-sharp"),
}

#: File extension (lower-case) -> (grammar, max file size in MiB). Kept equal to
#: coding-analysis's language registry by a parity test
#: (``tests/unit/test_hunt_agent.py``); the registry cannot be imported here.
SOURCE_EXTENSIONS: dict[str, tuple[str, int]] = {
    ".py": ("python", 5),
    ".pyw": ("python", 5),
    ".js": ("javascript", 5),
    ".jsx": ("javascript", 5),
    ".mjs": ("javascript", 5),
    ".cjs": ("javascript", 5),
    ".ts": ("typescript", 5),
    ".tsx": ("typescript", 5),
    ".d.ts": ("typescript", 5),
    ".go": ("go", 5),
    ".java": ("java", 5),
    ".rs": ("rust", 10),
    ".c": ("cpp", 10),
    ".h": ("cpp", 10),
    ".cc": ("cpp", 10),
    ".cpp": ("cpp", 10),
    ".cxx": ("cpp", 10),
    ".c++": ("cpp", 10),
    ".hh": ("cpp", 10),
    ".hpp": ("cpp", 10),
    ".hxx": ("cpp", 10),
    ".h++": ("cpp", 10),
    ".cs": ("c_sharp", 5),
}


def source_language(path: PurePath) -> tuple[str, int] | None:
    """``(grammar, max_mb)`` for a supported source file, else ``None``."""
    return SOURCE_EXTENSIONS.get(path.suffix.lower())


# ---------------------------------------------------------------------------
# Detection (the ONE implementation; the service has no other)
# ---------------------------------------------------------------------------

#: Tree-sitter node types that denote a function/method/constructor across the
#: supported grammars. The names are distinct enough between languages that a
#: single union set classifies reliably.
FUNCTION_NODE_TYPES = frozenset(
    {
        "function_definition",  # python, cpp
        "function_declaration",  # javascript, typescript, go
        "generator_function_declaration",  # javascript, typescript
        "function_expression",  # javascript, typescript
        "arrow_function",  # javascript, typescript
        "method_definition",  # javascript, typescript
        "method_declaration",  # go, java, c#
        "constructor_declaration",  # java, c#
        "local_function_statement",  # c#
        "function_item",  # rust
    }
)

#: A function longer than this many lines is flagged as a maintainability risk.
MAX_FUNCTION_LINES = 100

SYNTAX_ERROR = "syntax_error"
LOGIC_ERROR = "logic_error"


def first_error_line(node: Any) -> int | None:
    """Return the 1-based line of the first ERROR/MISSING node under *node*."""
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "ERROR" or current.is_missing:
            return current.start_point[0] + 1
        stack.extend(reversed(current.children))
    return None


def find_issues(root_node: Any, display_path: str) -> list[dict[str, Any]]:
    """Syntax errors and oversized functions in one parsed file.

    Each issue is ``{error_type, message, file_path, line_number,
    function_name, confidence}``; ``error_type`` is ``syntax_error`` or
    ``logic_error`` (the service's ``ErrorType`` values).
    """
    issues: list[dict[str, Any]] = []
    # Tree-sitter still produces a (partial) tree for broken input, with ERROR
    # / missing nodes marking the break.
    if root_node.has_error:
        issues.append(
            {
                "error_type": SYNTAX_ERROR,
                "message": f"Syntax error detected in {display_path}",
                "file_path": display_path,
                "line_number": first_error_line(root_node),
                "function_name": None,
                "confidence": 0.9,
            }
        )
    # Oversized functions: a language-agnostic maintainability smell derived
    # purely from node spans.
    stack = [root_node]
    while stack:
        node = stack.pop()
        if node.type in FUNCTION_NODE_TYPES:
            span = node.end_point[0] - node.start_point[0] + 1
            if span > MAX_FUNCTION_LINES:
                issues.append(
                    {
                        "error_type": LOGIC_ERROR,
                        "message": (
                            f"Function spanning {span} lines exceeds the "
                            f"{MAX_FUNCTION_LINES}-line maintainability threshold"
                        ),
                        "file_path": display_path,
                        "line_number": node.start_point[0] + 1,
                        "function_name": None,
                        "confidence": 0.4,
                    }
                )
        stack.extend(reversed(node.children))
    return issues


_PARSERS: dict[str, Any] = {}


def get_parser(grammar: str) -> Any | None:
    """A cached ``tree_sitter.Parser`` for *grammar*, or ``None`` if unavailable."""
    if grammar in _PARSERS:
        return _PARSERS[grammar]
    parser = None
    entry = GRAMMARS.get(grammar)
    if entry is not None:
        module_name, factory, _dist = entry
        try:
            from tree_sitter import Language, Parser

            module = importlib.import_module(module_name)
            parser = Parser(Language(getattr(module, factory)()))
        except (ImportError, AttributeError, ValueError, TypeError):
            parser = None
    _PARSERS[grammar] = parser
    return parser


def parse_bytes(grammar: str, content: bytes) -> Any | None:
    """Parse *content* with *grammar*; ``None`` when the grammar or parse fails."""
    parser = get_parser(grammar)
    if parser is None:
        return None
    try:
        return parser.parse(content)
    except (ValueError, RuntimeError, TypeError):
        return None


def tree_sitter_versions() -> dict[str, str | None]:
    """Installed distribution versions of tree-sitter and every grammar used.

    ``None`` for a distribution that is not installed. The caller compares this
    dict, computed in ITS environment, with the one the agent reports.
    """
    versions: dict[str, str | None] = {}
    for dist in ("tree-sitter", *sorted(entry[2] for entry in GRAMMARS.values())):
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = None
    return versions


#: Version tag hashed into every fingerprint; bump it if the recipe changes.
FINGERPRINT_VERSION = "bugfp/v1"

_WS_RE = re.compile(r"\s+")


def anchor_text(content: bytes, line_number: int | None) -> str:
    """The whitespace-normalised text of 1-based *line_number* in *content*.

    Leading/trailing whitespace is dropped and internal runs collapse to one
    space, so re-indenting the flagged line does not move its anchor. ``""`` when
    there is no such line (a finding without a line, or past the end).
    """
    if not isinstance(line_number, int) or line_number < 1:
        return ""
    lines = content.split(b"\n")
    if line_number > len(lines):
        return ""
    text = lines[line_number - 1].decode("utf-8", "replace")
    return _WS_RE.sub(" ", text).strip()


def finding_fingerprint(error_type: str, rel_path: str, anchor: str, ordinal: int) -> str:
    """Deterministic identity of one finding, stable against unrelated edits.

    ``sha256("bugfp/v1" NUL error_type NUL rel_path NUL anchor NUL ordinal)``.

    WHY THESE INPUTS, AND WHY NOT THE LINE NUMBER: a recheck must recognise the
    SAME finding after edits that have nothing to do with it. A line number moves
    whenever any line is inserted above it, so hashing it would turn every
    unrelated edit above the finding into "not reproduced". The anchor is the
    flagged line's own text (the syntax error's first line, the oversized
    function's header line), whitespace-normalised by :func:`anchor_text`, so:

    - an edit ABOVE or BELOW it, or re-indenting it, keeps the fingerprint;
    - an edit TO the flagged line changes it (it is a different line now);
    - the function's span is deliberately NOT hashed: an oversized function that
      grows or shrinks but stays oversized is still the same finding.

    ``ordinal`` (0-based, in file order) separates findings of the same type
    whose anchors are byte-identical in one file (two ``def wrapper(*args):``
    headers); it changes only if a same-type, same-anchor finding appears or
    disappears ABOVE it, which is a real change to that set of findings.
    ``rel_path`` is the hunt-root-relative path, so a file rename is a new
    finding by design.
    """
    material = "\0".join((FINGERPRINT_VERSION, error_type, rel_path, anchor, str(ordinal)))
    return hashlib.sha256(material.encode("utf-8", "surrogateescape")).hexdigest()


def fingerprinted_issues(root_node: Any, rel_path: str, content: bytes) -> list[dict[str, Any]]:
    """:func:`find_issues` with a ``fingerprint`` on every issue (the ONE detection path)."""
    issues = find_issues(root_node, rel_path)
    seen: dict[tuple[str, str], int] = {}
    for issue in issues:
        anchor = anchor_text(content, issue.get("line_number"))
        key = (str(issue["error_type"]), anchor)
        ordinal = seen.get(key, 0)
        seen[key] = ordinal + 1
        issue["fingerprint"] = finding_fingerprint(key[0], rel_path, anchor, ordinal)
    return issues


# ---------------------------------------------------------------------------
# Agent identity
# ---------------------------------------------------------------------------

AGENT_DIR = Path(__file__).resolve().parent
MANIFEST_NAME = "MANIFEST.sha256"


def agent_files(agent_dir: Path = AGENT_DIR) -> list[Path]:
    """The files that make up the agent: every top-level ``*.py``, sorted by name."""
    return sorted((p for p in agent_dir.glob("*.py") if p.is_file()), key=lambda p: p.name)


def manifest_text(agent_dir: Path = AGENT_DIR) -> str:
    """``sha256sum``-format listing of :func:`agent_files` (``<hex>  <name>\\n``)."""
    return "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in agent_files(agent_dir)
    )


def compute_agent_sha(agent_dir: Path = AGENT_DIR) -> str:
    """sha256 of :func:`manifest_text`: equals ``sha256(MANIFEST.sha256)`` when in sync."""
    return hashlib.sha256(manifest_text(agent_dir).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Git (hardened, read-only)
# ---------------------------------------------------------------------------

GIT_TIMEOUT_S = 120.0
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class AgentError(RuntimeError):
    """A measurement the result depends on could not be made."""


def _git_env() -> dict[str, str]:
    """No inherited ``GIT_*``: ``GIT_DIR``/``GIT_WORK_TREE``/``GIT_INDEX_FILE``
    would point the measurement at another repository, ``GIT_CONFIG_*`` could
    inject config past the hardening below."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1", "GIT_OPTIONAL_LOCKS": "0"})
    return env


def _git_base(root: Path) -> list[str]:
    # fsmonitor / hooks are the config keys that make read-only commands run
    # repo-supplied programs; both are pinned off. safe.directory matches the
    # live quality path (quality_result_store.resolve_commit_sha).
    return [
        "git",
        "--no-optional-locks",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        f"safe.directory={root}",
        "-C",
        str(root),
    ]


def _git_run(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(  # nosec B603 - fixed argv, no shell
            [*_git_base(root), *args],
            capture_output=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
            env=_git_env(),
            cwd="/",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AgentError(f"git {args[0]} could not run: {exc}") from exc


def _git_failed(args: tuple[str, ...], proc: subprocess.CompletedProcess[bytes]) -> AgentError:
    detail = proc.stderr.decode("utf-8", "replace").strip()[:300]
    return AgentError(f"git {args[0]} exited {proc.returncode}: {detail}")


def _git(root: Path, *args: str) -> bytes:
    proc = _git_run(root, *args)
    if proc.returncode != 0:
        raise _git_failed(args, proc)
    return proc.stdout


def resolve_head(root: Path) -> str:
    """Full sha of HEAD's commit."""
    sha = _git(root, "rev-parse", "--verify", "--end-of-options", "HEAD^{commit}")
    text = sha.decode("ascii", "replace").strip()
    if not _SHA_RE.match(text):
        raise AgentError(f"HEAD did not resolve to a full sha: {text[:80]!r}")
    return text


def commit_exists(root: Path, commit: str) -> bool:
    """True when the full sha *commit* names a commit object in the repository.

    ``rev-parse --verify --quiet`` exits 1 for an unknown object and 128 for a
    broken repository; only the former is "not found", the latter is an error.
    """
    args = ("rev-parse", "--verify", "--quiet", "--end-of-options", f"{commit}^{{commit}}")
    proc = _git_run(root, *args)
    if proc.returncode == 1:
        return False
    if proc.returncode != 0:
        raise _git_failed(args, proc)
    return proc.stdout.decode("ascii", "replace").strip() == commit


def worktree_is_dirty(root: Path) -> bool:
    """True when the working tree differs from HEAD (untracked files count)."""
    return bool(_git(root, "status", "--porcelain", "--ignore-submodules=all").strip())


def list_head_blobs(root: Path, commit: str) -> list[tuple[str, str, str, int]]:
    """``(mode, object_id, root-relative path, size)`` for every entry in *commit*
    under *root*. ``ls-tree`` without ``--full-tree`` lists the subtree of the
    ``-C`` directory with paths relative to it, i.e. relative to the hunt root."""
    entries: list[tuple[str, str, str, int]] = []
    for record in _git(root, "ls-tree", "-r", "-l", "-z", commit).split(b"\0"):
        if not record:
            continue
        meta, _, raw_path = record.partition(b"\t")
        fields = meta.split()
        if len(fields) != 4:
            continue
        mode, oid = fields[0].decode(), fields[2].decode()
        # A gitlink has no size ("-"); it is filtered by mode in hunt().
        size = int(fields[3]) if fields[3].isdigit() else 0
        entries.append((mode, oid, raw_path.decode("utf-8", "surrogateescape"), size))
    return entries


class BlobReader:
    """One ``git cat-file --batch`` process; one object written, one read, in turn."""

    def __init__(self, root: Path) -> None:
        self._proc = subprocess.Popen(  # nosec B603 - fixed argv, no shell
            [*_git_base(root), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_git_env(),
            cwd="/",
        )

    def read(self, oid: str) -> bytes:
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(oid.encode("ascii") + b"\n")
        self._proc.stdin.flush()
        header = self._proc.stdout.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            raise AgentError(f"git cat-file returned an unexpected header for {oid}")
        size = int(header[2])
        content = self._proc.stdout.read(size)
        self._proc.stdout.read(1)  # trailing LF
        if len(content) != size:
            raise AgentError(f"git cat-file returned {len(content)} of {size} bytes for {oid}")
        return content

    def close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        try:
            self._proc.wait(timeout=GIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._proc.kill()


# ---------------------------------------------------------------------------
# The hunt
# ---------------------------------------------------------------------------


#: ``(mode, object_id, rel_path, size, grammar, max_mb)`` of one in-scope blob.
ScopedEntry = tuple[str, str, str, int, str, int]

#: ``not_scanned`` reasons; also the recheck ``status`` for the same outcome.
NOT_SCANNED_REASONS = ("oversized", "not_utf8", "grammar_unavailable")


def scope_entry(
    mode: str, oid: str, rel: str, size: int, excluded_dirs: frozenset[str]
) -> ScopedEntry | None:
    """The ONE scope test: the entry as a :data:`ScopedEntry`, or ``None`` if out of scope."""
    rel_path = PurePosixPath(rel)
    language = source_language(rel_path)
    if (
        mode in EXCLUDED_ENTRY_MODES
        or not is_utf8_path(rel)
        or language is None
        or is_in_excluded_dir(rel_path, excluded_dirs)
    ):
        return None
    return (mode, oid, rel, size, language[0], language[1])


def scan_entry(reader: BlobReader, entry: ScopedEntry) -> tuple[str | None, list[dict[str, Any]]]:
    """Read and scan one in-scope blob: ``(not_scanned_reason, findings)``.

    The ONE detection path, shared by :func:`hunt` and :func:`recheck`.
    """
    _mode, oid, rel, size, grammar, max_mb = entry
    if size > max_mb * 1024 * 1024:
        return "oversized", []
    content = reader.read(oid)
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return "not_utf8", []
    tree = parse_bytes(grammar, content)
    if tree is None:
        return "grammar_unavailable", []
    return None, fingerprinted_issues(tree.root_node, rel, content)


def _identity() -> dict[str, Any]:
    tree_sitter = tree_sitter_versions()
    return {
        "agent_version": AGENT_VERSION,
        "agent_sha": compute_agent_sha(),
        "tree_sitter_version": tree_sitter.get("tree-sitter"),
        "grammar_versions": tree_sitter,
    }


def hunt(root: Path, excluded_dirs: frozenset[str] = EXCLUDED_DIRS) -> dict[str, Any]:
    """Scan HEAD of the repository at *root*; return the result document."""
    started = time.monotonic()
    commit = resolve_head(root)
    dirty = worktree_is_dirty(root)

    tracked: list[ScopedEntry] = []
    for mode, oid, rel, size in list_head_blobs(root, commit):
        entry = scope_entry(mode, oid, rel, size, excluded_dirs)
        if entry is not None:  # ONE scope test, applied before either count exists
            tracked.append(entry)

    findings: list[dict[str, Any]] = []
    not_scanned: list[dict[str, str]] = []
    scanned = 0
    reader = BlobReader(root)
    try:
        for entry in tracked:
            reason, issues = scan_entry(reader, entry)
            if reason is not None:
                not_scanned.append({"path": entry[2], "reason": reason})
                continue
            scanned += 1
            findings.extend(issues)
    finally:
        reader.close()

    return {
        "schema": RESULT_SCHEMA,
        **_identity(),
        "repo_root": str(root),
        "commit_sha": commit,
        "worktree_dirty": dirty,
        "tracked_source_files": len(tracked),
        "scanned_files": scanned,
        "excluded_dirs": sorted(excluded_dirs),
        "excluded_entry_kinds": list(EXCLUDED_ENTRY_KINDS),
        "not_scanned": not_scanned,
        "findings": findings,
        "duration_s": round(time.monotonic() - started, 3),
    }


#: Recheck ``status`` values. ``scanned`` is the only one with findings; every
#: other value says WHY the file could not be examined at that commit.
RECHECK_SCANNED = "scanned"
RECHECK_STATUSES = (
    RECHECK_SCANNED,
    "invalid_commit",
    "commit_not_found",
    "path_not_found",
    "path_out_of_scope",
    *NOT_SCANNED_REASONS,
)


def recheck(
    root: Path, commit: str, path: str, excluded_dirs: frozenset[str] = EXCLUDED_DIRS
) -> dict[str, Any]:
    """Re-run detection on ONE file as it is in *commit*'s tree (never the working tree).

    *path* is hunt-root-relative, exactly as a finding names it. Same scope test
    (:func:`scope_entry`) and same detection (:func:`scan_entry`) as :func:`hunt`.
    """
    started = time.monotonic()
    status = RECHECK_SCANNED
    findings: list[dict[str, Any]] = []
    if not _SHA_RE.match(commit):
        status = "invalid_commit"
    elif not commit_exists(root, commit):
        status = "commit_not_found"
    else:
        entry: ScopedEntry | None = None
        present = False
        for mode, oid, rel, size in list_head_blobs(root, commit):
            if rel == path:
                present = True
                entry = scope_entry(mode, oid, rel, size, excluded_dirs)
                break
        if not present:
            status = "path_not_found"
        elif entry is None:
            status = "path_out_of_scope"
        else:
            reader = BlobReader(root)
            try:
                reason, findings = scan_entry(reader, entry)
            finally:
                reader.close()
            if reason is not None:
                status = reason
    return {
        "schema": RECHECK_SCHEMA,
        **_identity(),
        "repo_root": str(root),
        "commit_sha": commit,
        "path": path,
        "status": status,
        "findings": findings,
        "excluded_dirs": sorted(excluded_dirs),
        "duration_s": round(time.monotonic() - started, 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

#: The only files ``--read-result`` / ``--remove-result`` will touch.
_RESULT_NAME_RE = re.compile(r"^hunt-agent-[A-Za-z0-9_-]{1,128}\.json$")


def _check_result_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute() or not _RESULT_NAME_RE.match(path.name):
        raise AgentError(f"result path must be absolute and named hunt-agent-<id>.json: {raw!r}")
    return path


def write_result(document: dict[str, Any], out: Path) -> tuple[int, str]:
    """Write *document* atomically as ASCII JSON; return ``(bytes, sha256)``."""
    data = json.dumps(document, ensure_ascii=True, sort_keys=True).encode("ascii")
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), prefix=".hunt-agent-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return len(data), hashlib.sha256(data).hexdigest()


def _emit(summary: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(summary, ensure_ascii=True, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hunt.py", description=__doc__.splitlines()[0])
    parser.add_argument("--repo", help="repository root to scan (HEAD is scanned)")
    parser.add_argument(
        "--recheck", action="store_true", help="re-scan ONE file (--path) at --commit"
    )
    parser.add_argument("--commit", help="--recheck: full commit sha to read the file at")
    parser.add_argument("--path", help="--recheck: hunt-root-relative file path")
    parser.add_argument("--out", help="absolute path of the result file (hunt-agent-<id>.json)")
    parser.add_argument("--read-result", help="print a byte range of a result file")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--length", type=int, default=0)
    parser.add_argument("--remove-result", help="delete a result file")
    parser.add_argument("--print-agent-sha", action="store_true")
    parser.add_argument("--print-manifest", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.print_agent_sha:
            sys.stdout.write(compute_agent_sha() + "\n")
            return 0
        if args.print_manifest:
            sys.stdout.write(manifest_text())
            return 0
        if args.read_result:
            path = _check_result_path(args.read_result)
            if args.offset < 0 or args.length <= 0:
                raise AgentError("--offset must be >= 0 and --length > 0")
            with path.open("rb") as handle:
                handle.seek(args.offset)
                sys.stdout.write(handle.read(args.length).decode("ascii"))
            return 0
        if args.remove_result:
            _check_result_path(args.remove_result).unlink(missing_ok=True)
            return 0
        if not args.repo or not args.out:
            parser.error("--repo and --out are required")
        if args.recheck and (args.commit is None or args.path is None):
            parser.error("--recheck requires --commit and --path")
        root = Path(args.repo).resolve()
        out = _check_result_path(args.out)
        if out.resolve().is_relative_to(root):
            raise AgentError("the result file must not be inside the repository")
        document = recheck(root, args.commit, args.path) if args.recheck else hunt(root)
        size, digest = write_result(document, out)
    except (AgentError, OSError, ValueError) as exc:
        _emit({"ok": False, "error": str(exc)[:500], "agent_sha": compute_agent_sha()})
        return 3

    if args.recheck:
        _emit(
            {
                "ok": True,
                "agent_sha": document["agent_sha"],
                "result_path": str(out),
                "result_bytes": size,
                "result_sha256": digest,
                "commit_sha": document["commit_sha"],
                "path": document["path"],
                "status": document["status"],
                "findings": len(document["findings"]),
            }
        )
        return 0
    _emit(
        {
            "ok": True,
            "agent_sha": document["agent_sha"],
            "result_path": str(out),
            "result_bytes": size,
            "result_sha256": digest,
            "commit_sha": document["commit_sha"],
            "worktree_dirty": document["worktree_dirty"],
            "scanned_files": document["scanned_files"],
            "tracked_source_files": document["tracked_source_files"],
            "findings": len(document["findings"]),
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

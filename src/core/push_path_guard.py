"""Which paths a workspace push may carry to a user's GitHub repository.

A user's repository receives card deliverables ONLY — never process data (run
logs, traces, ``.aa/`` state, probe directories, agent plans). ``git/push``
(``push_workspace_to_fork``) pushes whatever the workspace committed, so the
route asks :func:`refused_paths` about every path the pushed commits touch and
refuses the WHOLE push (no partial push) if any answer is non-empty.

This module is the single definition of the rule on this service. It MIRRORS
the coding-github guard (``coding-github/src/services/path_guard.py``): the same
three error codes and the same ``PROCESS_DATA_DENYLIST`` / ``PROCESS_DATA_ROOT_NAMES``,
matched the same way. Change both together.

Matching semantics (repository-relative, ``/``-separated paths as git prints them):

* The path is LOWERCASED, then matched with ``fnmatchcase`` against every
  ``PROCESS_DATA_DENYLIST`` pattern (all lowercase). ``*`` crosses ``/``, so
  ``*.log`` is any depth and ``*_probe/*`` is a ``_probe`` directory at any depth;
  ``.aa``/``.claude`` are denied at ANY depth, ``plans``/``traces``/``run-state``
  only at the root (``src/billing/plans/`` is ordinary domain code). ``.AA/x`` and
  ``X.LOG`` are refused like ``.aa/x`` and ``x.log``.
* A bare root entry named exactly like a denied root directory
  (``PROCESS_DATA_ROOT_NAMES``: ``.aa``, ``.claude``, ``plans``, ``traces``,
  ``run-state``, compared lowercased) is refused too.
* Absolute paths, empty paths, ``.``/``..``/empty segments, backslashes, NUL, and
  paths git printed C-quoted (a leading ``"``: a name it would not print verbatim)
  are refused as ``path_invalid`` — a name the guard cannot read literally is
  never approved.

A denylist is hygiene against a harness writing its run artefacts into the tree,
not a security boundary against a hostile workspace: renaming ``run.log`` to
``run.txt`` passes it. ``target_files`` (when the caller supplies it) is the
allowlist that closes that gap; it is compared exactly (git paths are
case-sensitive).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase

#: Error code: a pushed path is not a plain repository-relative path.
PATH_INVALID = "path_invalid"

#: Error code: a pushed path is process data.
PATH_DENIED_PROCESS_DATA = "path_denied_process_data"

#: Error code: ``target_files`` was supplied and a pushed path is outside it.
PATH_NOT_IN_TARGET_FILES = "path_not_in_target_files"

#: ⛔ The process-data denylist — verbatim the coding-github list. Lowercase
#: ``fnmatchcase`` patterns matched against the LOWERCASED path.
PROCESS_DATA_DENYLIST: tuple[str, ...] = (
    ".aa/*",
    "*/.aa/*",
    ".claude/*",
    "*/.claude/*",
    "plans/*",
    "traces/*",
    "run-state/*",
    "*_probe/*",
    "*.log",
    "*.trace.json",
)

#: Bare root entries named exactly like a denied root directory (no slash).
PROCESS_DATA_ROOT_NAMES: frozenset[str] = frozenset(
    {".aa", ".claude", "plans", "traces", "run-state"}
)


@dataclass(frozen=True)
class PathRefusal:
    """One refused path and the code that refused it."""

    path: str
    error_code: str
    rule: str


def _unsafe(path: str) -> str | None:
    """The reason ``path`` is not a plain repository-relative path, or ``None``."""
    if not path:
        return "empty path"
    if path.startswith('"'):
        return "git-quoted path"
    if path.startswith("/"):
        return "absolute path"
    if "\\" in path or "\x00" in path:
        return "backslash or NUL in path"
    segments = path.split("/")
    if any(seg in ("", ".", "..") for seg in segments):
        return "'..', '.' or empty segment"
    return None


def process_data_rule(path: str) -> str | None:
    """The denylist entry ``path`` matches (``path`` must already be safe), or ``None``."""
    folded = path.lower()
    if folded in PROCESS_DATA_ROOT_NAMES:
        return folded
    for pattern in PROCESS_DATA_DENYLIST:
        if fnmatchcase(folded, pattern):
            return pattern
    return None


def _in_target_files(path: str, target_files: Iterable[str]) -> bool:
    """``path`` equals a target, or sits under a target that ends with ``/``."""
    for target in target_files:
        if target.endswith("/"):
            if path.startswith(target):
                return True
        elif path == target:
            return True
    return False


def refused_paths(
    paths: Iterable[str], target_files: Iterable[str] | None = None
) -> list[PathRefusal]:
    """Every path the push must not carry. Empty list ⇔ the push may proceed.

    ``target_files`` is applied only when supplied (``None`` = no allowlist). An
    EMPTY list is an allowlist that admits nothing. A path the denylist refuses is
    reported under ``path_denied_process_data`` even when it is also a target —
    naming process data as a target does not make it a deliverable.
    """
    targets = None if target_files is None else list(target_files)
    refusals: list[PathRefusal] = []
    for path in dict.fromkeys(paths):  # de-duplicate, keep order
        reason = _unsafe(path)
        if reason is not None:
            refusals.append(PathRefusal(path, PATH_INVALID, reason))
            continue
        rule = process_data_rule(path)
        if rule is not None:
            refusals.append(PathRefusal(path, PATH_DENIED_PROCESS_DATA, rule))
        elif targets is not None and not _in_target_files(path, targets):
            refusals.append(PathRefusal(path, PATH_NOT_IN_TARGET_FILES, "target_files"))
    return refusals

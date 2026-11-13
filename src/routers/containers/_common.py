"""Shared helpers and re-exports for the containers router package.

Per-resource submodules import from this module (or from the package root)
rather than directly from src.core / src.models so that test patches targeting
``src.routers.containers.<name>`` resolve uniformly.
"""

from __future__ import annotations

import logging
import os
import posixpath
import re
import shlex
from typing import Any
from urllib.parse import unquote, urlparse

from fastapi import HTTPException

from src.core.log_safety import redact_secrets
from src.models.container_models import ContainerInfo

logger = logging.getLogger(__name__)

# scp-like SSH remote syntax, e.g. ``git@github.com:owner/repo.git`` — has no
# URL scheme, so urlparse cannot extract the host from it.
_SCP_LIKE_REMOTE = re.compile(r"^[A-Za-z0-9._-]+@(?P<host>[A-Za-z0-9.-]+):")
_WORKSPACE_ROOT = "/workspace"

# ``core.fsmonitor`` names a COMMAND that git runs whenever it reads the index
# (``status``, ``ls-files``, a worktree ``diff``). The workspace repo's config is
# attacker-controllable (committed via an include, or planted by an exec), so every
# git read the service runs overrides it. Command-line ``-c`` config is read LAST,
# so it wins over every file scope. Shared with the push path.
GIT_FSMONITOR_OFF = "core.fsmonitor=false"


# ``filter.<driver>.clean``/``process``/``smudge`` also name COMMANDS, selected per path
# by a ``filter=<driver>`` attribute (``.gitattributes`` or ``.git/info/attributes``).
# git runs clean/process while it merely READS (``status``, a worktree ``diff``), and
# neither ``--attr-source`` nor ``core.attributesFile`` stops it. Only overriding each
# filter key by NAME does, so the read first enumerates the EFFECTIVE keys AND values
# (every scope, include.path/includeIf included; ``git config`` itself runs no filter)
# and blanks each with ``-c <key>=``. A key containing ``=`` cannot be expressed as a
# ``-c`` override, so the read is refused (exit 97) rather than run unneutralised.
#
# One exception: a ``filter.lfs.*`` key whose effective (last) value is EXACTLY the
# canonical ``git lfs install`` value is left alone. Blanking it breaks every git-lfs
# repo (``filter.lfs.required=true`` + an empty clean filter = "clean filter failed"
# on any stat-dirty LFS file). Exact match only: any other value, including one that
# runs git-lfs by absolute path, is blanked like every other filter.
_FILTER_KEYS_RE = r"^filter\..*\.(clean|process|smudge)$"
FILTER_KEY_REFUSED_EXIT = 97
CANONICAL_LFS_FILTERS = {
    "filter.lfs.clean": "git-lfs clean -- %f",
    "filter.lfs.smudge": "git-lfs smudge -- %f",
    "filter.lfs.process": "git-lfs filter-process",
}

# Submodules are separate repos with their OWN config, which the filter enumeration
# above never sees. ``status`` and a worktree ``diff`` recurse into each submodule's
# worktree and run its ``filter.*.clean``; ``--ignore-submodules=dirty`` stops that
# (the config route, ``diff.ignoreSubmodules``, does not: ``submodule.<n>.ignore``
# overrides it). ``diff.submodule=diff`` in the superproject makes ``diff`` run a
# nested diff INSIDE a submodule whose pointer moved, which honours the submodule's
# ``diff.<driver>.command`` despite ``--no-ext-diff``, for commit ranges too:
# ``--submodule=short`` pins the one-line form. Pointer changes stay visible.
_SUBCOMMAND_FLAGS = {
    "status": ["--ignore-submodules=dirty"],
    "diff": ["--ignore-submodules=dirty", "--submodule=short"],
}

# ⛔ CANONICAL copy — ncn site 5 (movies-70) mirrors WORKSPACE_GIT_ENV,
# WORKSPACE_GIT_CONFIG and WORKSPACE_GIT_TRANSPORT_CONFIG byte-for-byte; change both
# together.
#
# Exported for every git a read OR a mutation runs (the enumeration included). A
# partial clone (``extensions.partialClone`` + a promisor remote) lazily FETCHES a
# missing object, running the remote's transport: ``core.sshCommand``, an ``ext::``
# URL, a local remote's ``remote.<n>.uploadpack``, all workspace config.
# GIT_NO_LAZY_FETCH (git >= 2.45) turns that into an error instead, for ``restore``
# too. GIT_TERMINAL_PROMPT=0 so no git can block on a credential prompt.
#
# HISTORY ISOLATION (coding-containers#156). Code that ran in the workspace (tests,
# dependency sync, a worker's candidate) can plant ``refs/replace/*``,
# ``info/grafts`` or a forged ``objects/info/commit-graph``. A later READ then sees
# another commit, tree or blob than the one ``commit``/``push`` actually write, so a
# "verify, then act" check passes on content that is not what ships.
# ``GIT_NO_REPLACE_OBJECTS=1`` + ``GIT_GRAFT_FILE=/dev/null`` + ``-c
# core.commitGraph=false`` make every git read the REAL object graph. The workspace is
# also REFUSED outright when it holds any of them (``workspace_poison_check``): the env is the
# control, the refusal is the tripwire, and either one alone still stops the attack.
#
# CONFIG ISOLATION. ``GIT_CONFIG_NOSYSTEM=1`` + ``GIT_CONFIG_GLOBAL=/dev/null`` (an
# always-empty file the workspace cannot write): only the repository's own config and
# our ``-c`` overrides apply, exactly as on the push (git_push.py). What the system
# config used to supply is passed as ``-c`` instead (``WORKSPACE_GIT_CONFIG``):
# ``safe.directory`` (a Docker image may have set it there; the k8s pod sets it via
# ``GIT_CONFIG_PARAMETERS``, which is command scope and still applies). Measured: neither
# runtime image writes a system or global gitconfig (no identity, no git-lfs), and the
# commit route passes the author with ``-c``, so nothing else is lost.
#
# ONE list, shared with git_push.py (``_in_workdir``, ``_hardened_git``,
# ``_push_command``). Read at CALL time (never frozen into a module constant) so a
# test can drop one entry and prove the attack goes red without it.
#
# ATTRIBUTE ISOLATION. ``GIT_ATTR_NOSYSTEM=1`` drops ``$(prefix)/etc/gitattributes``, and
# ``-c core.attributesFile=/dev/null`` (``WORKSPACE_GIT_CONFIG``) the global/XDG one,
# which defaults to ``$XDG_CONFIG_HOME/git/attributes`` or ``~/.config/git/attributes``
# EVEN with ``GIT_CONFIG_GLOBAL=/dev/null`` -- a file in the workspace user's own home.
# The repository's ``.gitattributes`` / ``info/attributes`` still apply, on the read
# and on the commit alike; what they can select is neutralised by name: filter
# drivers are blanked (``_filter_neutraliser``), diff drivers and textconv are off
# (every diff read passes ``--no-ext-diff --no-textconv``), ``-diff`` is overridden by
# ``--text``, and no route runs a merge.
#
# TRACE ISOLATION (coding-containers#157 audit, reproduced with git 2.50). A global
# config (``~/.gitconfig`` / ``~/.config/git/config`` -- HOME is workspace-owned)
# setting ``trace2.envVars=GIT_PUSH_TOKEN`` + ``trace2.eventTarget=<file>`` makes every
# git that reads it write that variable's VALUE to a file the workspace reads back;
# the push exec's config guard reads global scope BY DESIGN, so it cannot be fixed by
# config isolation alone. ``GIT_TRACE2*=0`` overrides the ``trace2.{normal,event,perf}
# Target`` config keys (env wins; measured: no file written). Only the three trace2
# targets: they are the only tracing a CONFIG key can switch on -- the legacy
# ``GIT_TRACE*`` family is env-only, and the exec env is ours, not the workspace's.
# The push exec exports them before its first git (``git_push._PUSH_PRELUDE_KEYS``),
# and its config guard refuses ``trace2.*`` in any scope as the second layer.
WORKSPACE_GIT_ENV: tuple[str, ...] = (
    "GIT_NO_LAZY_FETCH=1",
    "GIT_TERMINAL_PROMPT=0",
    "GIT_NO_REPLACE_OBJECTS=1",
    "GIT_GRAFT_FILE=/dev/null",
    "GIT_CONFIG_NOSYSTEM=1",
    "GIT_CONFIG_GLOBAL=/dev/null",
    "GIT_ATTR_NOSYSTEM=1",
    "GIT_TRACE2=0",
    "GIT_TRACE2_EVENT=0",
    "GIT_TRACE2_PERF=0",
    # TRANSPORT ISOLATION: set-but-empty = NO transport is allowed, and git documents
    # the variable as "overriding any existing configuration" -- including a
    # ``protocol.<name>.allow=always`` in a SUBMODULE's own config, which no ``-c``
    # default (``protocol.allow=never``) beats (measured). No workspace git route
    # needs a transport; the push re-exports it as ``https`` for its one exec.
    "GIT_ALLOW_PROTOCOL=",
)

# ``-c`` overrides every workspace git runs with (command-line config is read LAST,
# so each wins over the repository's own ``.git/config``):
#
# * ``safe.directory=/workspace``: replaces the system-config entry NOSYSTEM drops.
# * ``core.hooksPath=/dev/null``: ``commit``/``switch``/``update-ref`` run workspace
#   hooks, and so can a READ: ``status`` refreshes and rewrites the index, which runs
#   ``post-index-change``.
# * ``core.fsmonitor=false``: see ``GIT_FSMONITOR_OFF``.
# * ``core.commitGraph=false``: a forged commit-graph rewrites parent links like a
#   graft, and git does not checksum it on read.
# * ``status.showUntrackedFiles=normal``: a workspace ``no`` would hide every new file
#   from ``status`` while ``add -A`` still commits it, so the status route's "changed
#   files" check (s.done.commit.fix-unit) would pass on a commit carrying more.
# * ``diff.relative=false``: ``true`` limits ``diff`` to the cwd subtree.
# * ``core.attributesFile=/dev/null``: see ATTRIBUTE ISOLATION above.
# * ``core.untrackedCache=false``: the untracked cache lives in the workspace-writable
#   index and lets ``status``/``ls-files --others`` skip directories it records as
#   unchanged; a forged cache hides new files from the read while ``add -A`` still
#   finds them. Off, every directory is scanned (kept identical to the ncn mirror).
#
# Repo-local config that still applies, deliberately: ``core.excludesFile`` /
# ``info/exclude`` / ``.gitignore`` hide untracked files from ``status`` AND from
# ``add -A`` alike (the read and the write agree), ``core.autocrlf``/``eol`` and
# canonical git-lfs filters convert on ``hash-object`` exactly as on ``add``.
WORKSPACE_GIT_CONFIG: tuple[str, ...] = (
    f"safe.directory={_WORKSPACE_ROOT}",
    "core.hooksPath=/dev/null",
    GIT_FSMONITOR_OFF,
    "core.commitGraph=false",
    "status.showUntrackedFiles=normal",
    "diff.relative=false",
    "core.attributesFile=/dev/null",
    "core.untrackedCache=false",
)


def workspace_git_env() -> str:
    """``export <WORKSPACE_GIT_ENV>; `` — the env prefix every workspace git runs under."""
    return "export " + " ".join(WORKSPACE_GIT_ENV) + "; " if WORKSPACE_GIT_ENV else ""


# SUBMODULE + TRANSPORT ISOLATION (coding-containers#157 audit). A workspace can make
# a git command RECURSE into a submodule -- a separate repository whose own
# ``.git/modules/<n>/config`` no config guard here reads -- and run a transport from
# there: ``push.recurseSubmodules=on-demand`` made the token-holding push run ``git
# push`` inside a submodule whose ``remote.origin.url=ext::<cmd>`` (with
# ``protocol.ext.allow=always``) executed ``<cmd>`` with ``$GIT_PUSH_TOKEN`` in its
# environment (reproduced). ``submodule.recurse`` does the same for ``switch`` /
# ``restore`` / ``checkout`` / ``fetch``. Command-line config reaches the submodule's
# git too (``GIT_CONFIG_PARAMETERS``), so these pins apply inside it:
#
# * recursion off: ``submodule.recurse=false``, ``push.recurseSubmodules=no``,
#   ``fetch.recurseSubmodules=false`` (the push also passes ``--no-recurse-submodules``);
# * transports off: ``protocol.allow=never`` is only the DEFAULT for a protocol with no
#   ``protocol.<name>.allow`` of its own, so ``ext`` and ``file`` (the two that run a
#   local command / a local repo's config) are pinned by name as well. The class-level
#   control is ``GIT_ALLOW_PROTOCOL`` (``WORKSPACE_GIT_ENV``), which also covers any
#   ``<helper>::`` transport name.
#
# Kept OUT of ``WORKSPACE_GIT_CONFIG`` because the push's config guard lists config
# with that list and refuses exactly these keys (``git_push._DANGEROUS_CONFIG_RE``):
# the guard runs ``workspace_git_config_args(transport=False)``, every other git gets
# both. Read at CALL time so a test can drop an entry.
WORKSPACE_GIT_TRANSPORT_CONFIG: tuple[str, ...] = (
    "submodule.recurse=false",
    "push.recurseSubmodules=no",
    "fetch.recurseSubmodules=false",
    "protocol.allow=never",
    "protocol.ext.allow=never",
    "protocol.file.allow=never",
)


def workspace_git_config_args(*, transport: bool = True) -> list[str]:
    """``WORKSPACE_GIT_CONFIG`` (+ ``WORKSPACE_GIT_TRANSPORT_CONFIG`` unless
    ``transport=False``) as argv: ``["-c", k=v, "-c", k=v, ...]``."""
    pairs = (*WORKSPACE_GIT_CONFIG, *(WORKSPACE_GIT_TRANSPORT_CONFIG if transport else ()))
    return [a for pair in pairs for a in ("-c", pair)]


def isolated_git() -> str:
    """``git -c <WORKSPACE_GIT_CONFIG>...`` as a shell-quoted command prefix."""
    return " ".join(shlex.quote(a) for a in ["git", *workspace_git_config_args()])


# Exit code and stderr marker of a refused (poisoned) workspace, and the HTTP error
# the git routes turn it into (``get_git_manager``).
WORKSPACE_POISONED_EXIT = 96
WORKSPACE_POISONED = "workspace_git_poisoned"

# Set False only by tests (the mutation matrix): builds the commands without the
# refusal so the env layer can be shown to stop the attack on its own.
POISON_CHECK_ENABLED = True


def workspace_poison_check() -> str:
    """Shell fragment refusing (exit ``WORKSPACE_POISONED_EXIT``) a workspace that holds
    a ``refs/replace/*`` ref, ``info/grafts`` or ``objects/info/alternates``.

    Runs in the enclosing (sub)shell BEFORE any git read, so a refused script never
    prints a partial answer. The env already neutralises replace refs and grafts; the
    refusal exists because a workspace that holds them was tampered with, and because
    ``alternates`` has no env switch at all: it grafts another object store into this
    one, and every read, commit and push then trusts objects from a directory the
    workspace names.

    ``info/grafts`` and ``objects/`` live in the COMMON dir (shared by linked
    worktrees), so the paths are resolved from ``rev-parse --git-common-dir``. When that
    fails the directory is not a repository (or git cannot run): the check is skipped
    and the real git command fails the same way, as it always did.

    ``shallow`` is NOT refused: every workspace is a ``clone --depth 1`` (the
    provisioners), so ``shallow`` is expected. It only removes PARENT edges; it cannot
    change what a commit, tree or blob id resolves to, and every verify read here names
    its commits explicitly (diff endpoints, restore source, status against HEAD). The
    push listing's own residual note covers the one history walk (git_push.py).
    """
    if not POISON_CHECK_ENABLED:
        return workspace_submodule_check()
    g = isolated_git()
    fmt = shlex.quote("--format=%(refname)")
    return workspace_submodule_check() + (
        f"_poisoned() {{ echo \"{WORKSPACE_POISONED}: $1; this workspace's git history "
        'cannot be trusted (planted by code that ran in it). Refusing to read it." >&2; '
        f"exit {WORKSPACE_POISONED_EXIT}; }}; "
        f"if _pc=$({g} rev-parse --git-common-dir 2>/dev/null); then "
        f"_pr=$({g} for-each-ref {fmt} refs/replace/) "
        "|| _poisoned 'refs/replace/ could not be listed'; "
        '[ -z "$_pr" ] || _poisoned "replace refs present ($(echo $_pr))"; '
        'if [ -e "$_pc/info/grafts" ] || [ -L "$_pc/info/grafts" ]; then '
        "_poisoned 'info/grafts present'; fi; "
        'if [ -e "$_pc/objects/info/alternates" ] || [ -L "$_pc/objects/info/alternates" ]; '
        "then _poisoned 'objects/info/alternates present'; fi; "
        "fi; "
    )


# SUBMODULES ARE UNSUPPORTED (coordinator ruling movies-64, audit on #157). A submodule
# is a SEPARATE repository whose config, hooks, filters and transports no guard here
# reads, and a gitlink's content is another repository's commit that no diff, blob
# check or tree binding of this route can see. So a workspace whose HEAD tree, index,
# working tree, a diff/commit endpoint tree or a pushed range holds a gitlink (mode
# 160000) or a root ``.gitmodules`` is REFUSED (409 ``workspace_submodules_unsupported``),
# fail closed, at every site: the prefix runs in every workspace git exec
# (``workspace_poison_check``), the endpoint trees in ``workspace_object_check``, and the
# pushed range in the push listing. The ``-c`` pins (``WORKSPACE_GIT_TRANSPORT_CONFIG``)
# stay as defence in depth.
WORKSPACE_SUBMODULES_EXIT = 92
WORKSPACE_SUBMODULES = "workspace_submodules_unsupported"

# Set False only by tests (the mutation matrix), like ``POISON_CHECK_ENABLED``.
SUBMODULE_CHECK_ENABLED = True

# stdin: ``ls-tree -r`` / ``ls-files -s`` lines (``<mode> ...\t<path>``) or ``--raw``
# lines (``:<mode1> <mode2> ...\t<path>``); exit 0 when one is a gitlink or the root
# ``.gitmodules``. (git C-quotes a path holding a TAB, so ``$2`` is the whole path.)
NOSUB_AWK = (
    "awk -F'\t' '{ m = $1; sub(/^:/, \"\", m) } "
    'm ~ /^160000 / || m ~ /^[0-7]+ 160000 / || $2 == ".gitmodules" { f = 1 } '
    "END { exit !f }'"
)


def workspace_submodule_check() -> str:
    """Shell fragment: defines ``_nosub_tree <tree>`` (used by ``workspace_object_check``
    and the push listing) and refuses (exit ``WORKSPACE_SUBMODULES_EXIT``) a workspace
    whose HEAD tree, index or working tree holds a gitlink or a root ``.gitmodules``.
    Skipped where ``rev-parse --git-dir`` fails (not a repository: the real git command
    fails the same way). ``_nosub_tree`` is a no-op when the check is disabled."""
    if not SUBMODULE_CHECK_ENABLED:
        return "_nosub_tree() { :; }; "
    g = isolated_git()
    return (
        f'_nosub() {{ echo "{WORKSPACE_SUBMODULES}: $1; submodules are not supported '
        "(a submodule is another repository whose config and content this route cannot "
        f'verify). Refusing." >&2; exit {WORKSPACE_SUBMODULES_EXIT}; }}; '
        f'_nosub_tree() {{ _nsl=$({g} ls-tree -r --full-tree "$1") '
        '|| _nosub "tree $1 could not be listed"; '
        f"printf '%s\n' \"$_nsl\" | {NOSUB_AWK} "
        '&& _nosub "tree $1 holds a gitlink or .gitmodules"; }; '
        f"if {g} rev-parse --git-dir >/dev/null 2>&1; then "
        f"_nsh=$({g} rev-parse --verify --quiet 'HEAD^{{tree}}') && _nosub_tree \"$_nsh\"; "
        f'_nsi=$({g} ls-files -s) || _nosub "the index could not be listed"; '
        f"printf '%s\n' \"$_nsi\" | {NOSUB_AWK} "
        '&& _nosub "the index holds a gitlink or .gitmodules"; '
        f"_nst=$({g} rev-parse --show-toplevel 2>/dev/null) && "
        '{ [ -e "$_nst/.gitmodules" ] || [ -L "$_nst/.gitmodules" ]; } '
        '&& _nosub "the working tree holds .gitmodules"; '
        "fi; "
    )


def workspace_submodules_refusal(result: Any) -> HTTPException | None:
    """409 ``workspace_submodules_unsupported`` for an exec result the check refused
    (exit code AND stderr marker, as for the poison refusal)."""
    if not isinstance(result, dict) or result.get("exit_code") != WORKSPACE_SUBMODULES_EXIT:
        return None
    stderr = str(result.get("stderr") or "")
    if WORKSPACE_SUBMODULES not in stderr:
        return None
    found = [
        line.split(":", 1)[1].split(";", 1)[0].strip()
        for line in stderr.splitlines()
        if line.startswith(f"{WORKSPACE_SUBMODULES}:")
    ]
    return HTTPException(
        status_code=409,
        detail={
            "error_code": WORKSPACE_SUBMODULES,
            "message": (
                f"{WORKSPACE_SUBMODULES}: {', '.join(found) or 'the workspace uses submodules'}"
                ". Submodules are not supported: nothing was read, changed or pushed."
            ),
            "found": found,
        },
    )


# Exit code and stderr marker of a refused object-integrity check, and the HTTP error
# the git routes turn it into (``get_git_manager``).
WORKSPACE_OBJECT_CORRUPT_EXIT = 93
WORKSPACE_OBJECT_CORRUPT = "workspace_object_corrupt"

# Set False only by tests (the mutation matrix), like ``POISON_CHECK_ENABLED``.
OBJECT_CHECK_ENABLED = True


def workspace_object_check(base_commit: str, trees: list[str], pairs: list[tuple[str, str]]) -> str:
    """Shell fragment (needs ``_g`` from ``workspace_git_function``) that refuses (exit
    ``WORKSPACE_OBJECT_CORRUPT_EXIT``) unless every object a decision reads hashes to
    its own id.

    WHY. git does NOT verify a loose object's hash when it reads it (``cat-file``,
    ``diff``, ``ls-tree``). Code that ran in the workspace can overwrite
    ``.git/objects/xx/yyyy`` IN PLACE with other content (measured: ``cat-file blob E``
    printed the planted bytes; ``add -A`` of a file hashing to ``E`` keeps the planted
    object, since it only freshens an existing one), so a diff, a status or a blob
    listing reports content other than what the ids commit. The push itself is
    protected by the receiver re-hashing every object (measured: ``remote rejected ...
    missing necessary objects``), but a decision made on the forged read is not.

    WHAT. ``base_commit`` is read ONCE into a temp file, its bytes hashed (``hash-object
    --literally --no-filters -t commit --stdin``) and its tree id taken from THOSE bytes
    into ``$_vbt`` (so a caller diffs ``"$_vbt"``, never ``<commit>^{tree}``, which
    would re-read the object). Then every id in ``trees`` (shell words, e.g. ``"$_a"``)
    and every object ``diff-tree -r -t --no-renames A B`` lists for each pair (both
    sides, trees and blobs; gitlinks are another repository's) is hashed the same way.
    That set is exactly what a diff between the pair reads: git descends only into
    subtrees whose ids differ, and each such subtree is listed by a parent that was
    itself verified, so a forged subtree cannot hide its own children.

    RESIDUAL (stated, not hidden): a same-uid process that FLIPS an object file back
    and forth can still race a read that follows the check in the same exec. Nothing on
    this exec surface can prevent that; the receiver's re-hash still refuses anything
    forged that reaches a push.
    """
    nosub = "".join(f"_nosub_tree {t}; " for t in ['"$_vbt"', *trees])
    if not OBJECT_CHECK_ENABLED:
        return (
            "_vbt=$(_g rev-parse --verify --quiet " + base_commit + "^{tree}) || exit 1; " + nosub
        )
    awk = (
        "awk '/^:/ { m1 = substr($1, 2); m2 = $2; "
        'if (m1 != "000000" && m1 != "160000") print (m1 == "040000" ? "tree" : "blob"), $3; '
        'if (m2 != "000000" && m2 != "160000") print (m2 == "040000" ? "tree" : "blob"), $4 }\''
    )
    checks = [f"_vo tree {t};" for t in trees]
    checks += [f"_vp {a} {b};" for a, b in pairs]
    return (
        f'_corrupt() {{ echo "{WORKSPACE_OBJECT_CORRUPT}: $1; the workspace object store '
        'was altered (planted by code that ran in it). Refusing to read it." >&2; '
        f"exit {WORKSPACE_OBJECT_CORRUPT_EXIT}; }}; "
        # A MISSING object is git failing (exit 1, the caller's 500/404 path), not a forgery.
        '_vo() { _g cat-file -e "$2" 2>/dev/null || { echo "missing object $2" >&2; exit 1; }; '
        '_vh=$(_g cat-file "$1" "$2" | _g hash-object --literally --no-filters '
        '-t "$1" --stdin) && [ "$_vh" = "$2" ] || _corrupt "$1 $2 does not hash to its id"; }; '
        '_vp() { _vd=$(_g diff-tree -r -t --no-renames "$1" "$2") '
        '|| _corrupt "diff-tree $1 $2 failed"; '
        f"printf '%s\\n' \"$_vd\" | {awk} "
        # The loop runs in a pipeline subshell: its exit status (93 forged, 1 missing)
        # is re-raised in the script itself.
        '| { while read -r _vk _vi; do _vo "$_vk" "$_vi"; done; } || exit $?; }; '
        "_vgd=$(_g rev-parse --git-dir) || exit 1; "
        '_vf=$(mktemp "$_vgd/verify-commit.XXXXXX") || exit 1; '
        f'_g cat-file commit {base_commit} > "$_vf" || {{ rm -f "$_vf"; exit 1; }}; '
        '_vh=$(_g hash-object --literally --no-filters -t commit --stdin < "$_vf"); '
        f'_vbt=$(sed -n \'1s/^tree //p\' "$_vf"); rm -f "$_vf"; '
        f'[ "$_vh" = {base_commit} ] || _corrupt "commit {base_commit} does not hash to its id"; '
        '_vo tree "$_vbt"; ' + " ".join(checks) + " " + nosub
    )


# Global options that take their value as the NEXT argv token. ``_subcommand_index``
# skips them so ``git -c k=v --literal-pathspecs status`` still finds ``status``.
_GLOBAL_OPTS_WITH_VALUE = frozenset(
    {"-c", "-C", "--git-dir", "--work-tree", "--namespace", "--config-env", "--attr-source"}
)


def _subcommand_index(args: list[str]) -> int | None:
    """Index of the git subcommand in ``args``: the first token that is not a global
    option (nor the value of one). ``None`` when every token is an option."""
    i = 0
    while i < len(args):
        a = args[i]
        if a in _GLOBAL_OPTS_WITH_VALUE:
            i += 2
        elif a.startswith("-"):
            i += 1
        else:
            return i
    return None


def _with_subcommand_flags(args: list[str]) -> list[str]:
    i = _subcommand_index(args)
    if i is None or args[i] not in _SUBCOMMAND_FLAGS:
        return args
    return [*args[: i + 1], *_SUBCOMMAND_FLAGS[args[i]], *args[i + 1 :]]


def _filter_neutraliser(emit: str) -> str:
    """Shell fragment that enumerates the EFFECTIVE ``filter.*.{clean,process,smudge}``
    keys and runs ``emit`` once per key to blank (the key is in ``$_k``), except a
    ``filter.lfs.*`` key whose last value is exactly ``CANONICAL_LFS_FILTERS``. Exits
    ``FILTER_KEY_REFUSED_EXIT`` (the enclosing subshell) when the read must be refused.
    Must run under ``set -f`` in a subshell: it sets IFS and ``_``-prefixed variables.
    """
    enum = " ".join(
        shlex.quote(a) for a in [*workspace_git_config_args(), "config", "-z", "--get-regexp"]
    )
    lfs_checks = " ".join(
        f'[ -z "$_{var}" ] || [ "$_{var}" = {shlex.quote("x" + CANONICAL_LFS_FILTERS[key])} ]'
        f" || {{ _k={shlex.quote(key)}; {emit} }};"
        for var, key in (
            ("lc", "filter.lfs.clean"),
            ("ls", "filter.lfs.smudge"),
            ("lp", "filter.lfs.process"),
        )
    )
    # `git config -z --get-regexp` prints "<key>\n<value>\0" per entry ("<key>\0" for
    # a value-less key). tr maps NUL -> \002 (record) and '\n' -> \003 (the FIRST one
    # in a record ends the key: git config files cannot put a newline in a name).
    # Any \001-\003 already present, or a failed enumeration (exit 1 is "no match",
    # anything else injects \001), becomes \001 = refuse. For lfs keys only the LAST
    # value counts (git's own rule), so they are collected and decided after the loop;
    # the "x" prefix tells "seen with an empty value" from "not seen".
    return (
        f"_r=$({{ git {enum} {shlex.quote(_FILTER_KEYS_RE)} || [ $? -eq 1 ] || printf '\\001'; }}"
        " | tr '\\001\\002\\003\\n\\000' '\\001\\001\\001\\003\\002') || exit "
        f"{FILTER_KEY_REFUSED_EXIT}; "
        "_a=$(printf '\\001'); _s=$(printf '\\002'); _n=$(printf '\\003'); "
        "_refuse() { echo 'refusing git read: filter config key contains =, a control "
        f"byte, or git config failed' >&2; exit {FILTER_KEY_REFUSED_EXIT}; }}; "
        'case "$_r" in *"$_a"*) _refuse;; esac; '
        "_lc=; _ls=; _lp=; "
        'IFS="$_s"; '
        "for _e in $_r; do "
        '_k=${_e%%"$_n"*}; '
        'case "$_e" in *"$_n"*) _v=${_e#*"$_n"};; *) _v="$_n";; esac; '
        'case "$_k" in ""|*=*) _refuse;; esac; '
        'case "$_k" in '
        'filter.lfs.clean) _lc="x$_v";; '
        'filter.lfs.smudge) _ls="x$_v";; '
        'filter.lfs.process) _lp="x$_v";; '
        f"*) {emit.rstrip(';')};; "
        "esac; done; "
        "unset IFS; "
        f"{lfs_checks}"
    )


# What the neutraliser does per key: append a ``-c <key>=`` argv pair, or collect the key.
_EMIT_ARG = 'set -- "$@" -c "$_k=";'
_EMIT_KEY = '_GK="$_GK$_k$_s";'


def _workspace_git(args: list[str]) -> str:
    """``cd /workspace && ( <env> <poison check> ... exec git <WORKSPACE_GIT_CONFIG>
    <-c filter blanks> <args> )``: see ``workspace_git_read``."""
    quoted = " ".join(shlex.quote(a) for a in [*workspace_git_config_args(), *args])
    script = (
        f"( {workspace_git_env()}set -f; {workspace_poison_check()}set --; "
        f'{_filter_neutraliser(_EMIT_ARG)} exec git "$@" {quoted} )'
    )
    return f"cd {shlex.quote(_WORKSPACE_ROOT)} && {script}"


def workspace_git_read(args: list[str]) -> str:
    """Build a hardened read-only git command in ``/workspace``, every arg shell-quoted.

    Runs ``git <WORKSPACE_GIT_CONFIG> -c filter.<d>.<kind>= ... <args>`` with one
    ``-c <key>=`` per effective ``filter.*.{clean,process,smudge}`` key (one extra
    ``git config`` call), except a ``filter.lfs.*`` key whose last value equals
    ``CANONICAL_LFS_FILTERS`` exactly. Fails closed with exit
    ``FILTER_KEY_REFUSED_EXIT`` when a key name contains ``=``, when the config output
    holds a \\001-\\003 control byte, or when the enumeration itself errors.

    ``status``/``diff`` also get ``_SUBCOMMAND_FLAGS`` right after the subcommand (the
    first non-option token, so leading ``-c <val>`` pairs are skipped), and
    ``WORKSPACE_GIT_ENV`` is exported (no lazy fetch, no terminal prompt, no replace
    refs / grafts, no system or global config). A workspace holding replace refs,
    grafts or alternates is refused first (exit ``WORKSPACE_POISONED_EXIT``).

    Diff callers must also pass ``--no-ext-diff --no-textconv``:
    ``diff.<driver>.command``/``textconv`` selected through ``.gitattributes`` are
    commands too.
    """
    return _workspace_git(_with_subcommand_flags(args))


def workspace_git_mutate(args: list[str]) -> str:
    """``workspace_git_read``'s hardening for a git command that WRITES the
    worktree/index: the same env, ``-c`` list and poison refusal, filters blanked, so a
    mutation never lazily fetches either. The promisor remote is workspace config: a
    planted ``ext::`` URL or a local-path remote with ``remote.<n>.uploadpack`` runs a
    command on fetch, and per-protocol ``-c`` overrides do not close the latter. The
    provisioners only ``clone --depth 1`` (never a partial clone), so a missing blob is
    an error, not something to fetch. No ``_SUBCOMMAND_FLAGS`` are added."""
    return _workspace_git(args)


def workspace_git_function(name: str = "_g") -> str:
    """Shell definitions of ``<name>``, a drop-in for ``git`` inside a multi-git script.

    The filter enumeration runs ONCE, up front, and a refusal exits the whole script
    with ``FILTER_KEY_REFUSED_EXIT`` (fail closed: a refused ``_g`` inside a pipeline
    would otherwise read as empty output). The keys to blank are kept \\002-separated
    in ``_GK`` (the enumeration refuses any key holding \\001-\\003 or ``=``), and
    ``<name> ARGS`` runs ``git <WORKSPACE_GIT_CONFIG> -c <key>= ... ARGS`` in a
    subshell. ``WORKSPACE_GIT_ENV`` is exported for the rest of the script, and a
    poisoned workspace exits the whole script with ``WORKSPACE_POISONED_EXIT`` before
    anything runs. Callers add no
    ``_SUBCOMMAND_FLAGS``: pass what the subcommand needs explicitly.
    """
    return (
        f"{workspace_git_env()}"
        f"{workspace_poison_check()}"
        # A function, not an inline $(...): bash 3.2 (macOS /bin/sh) mis-parses a
        # `case` inside a command substitution.
        f'_gk() ( set -f; _GK=; {_filter_neutraliser(_EMIT_KEY)} printf %s "$_GK" ); '
        f"_GK=$(_gk) || exit {FILTER_KEY_REFUSED_EXIT}; "
        f"{name}() ( set -f; _c=$#; IFS=$(printf '\\002'); "
        'for _k in $_GK; do set -- "$@" -c "$_k="; done; unset IFS; '
        'while [ "$_c" -gt 0 ]; do set -- "$@" "$1"; shift; _c=$((_c - 1)); done; '
        f'exec {isolated_git()} "$@" ); '
    )


def get_manager() -> Any:
    """Late-bound lookup of ``get_container_manager`` via the package root.

    Submodules call this rather than importing ``get_container_manager``
    directly so that ``mock.patch("src.routers.containers.get_container_manager")``
    works uniformly across every per-resource submodule.
    """
    from src.routers import containers as _pkg

    return _pkg.get_container_manager()


def workspace_poisoned_refusal(result: Any) -> HTTPException | None:
    """409 ``workspace_git_poisoned`` for an exec result the poison check refused, else None.

    Both the exit code AND the stderr marker are required, so an unrelated command
    exiting 96 is not misreported. A workspace can forge the pair only to refuse its own
    request (fail closed).
    """
    if not isinstance(result, dict) or result.get("exit_code") != WORKSPACE_POISONED_EXIT:
        return None
    stderr = str(result.get("stderr") or "")
    if WORKSPACE_POISONED not in stderr:
        return None
    found = [
        line.split(":", 1)[1].split(";", 1)[0].strip()
        for line in stderr.splitlines()
        if line.startswith(f"{WORKSPACE_POISONED}:")
    ]
    return HTTPException(
        status_code=409,
        detail={
            "error_code": WORKSPACE_POISONED,
            "message": (
                f"{WORKSPACE_POISONED}: the workspace repository contains "
                f"{', '.join(found) or 'replace refs, grafts or alternates'}, which make git "
                "report a history or content other than what a commit or push writes. It was "
                "planted by code that ran in the workspace, so nothing was read, changed or "
                "pushed. Re-provision the workspace."
            ),
            "found": found,
        },
    )


def workspace_object_corrupt_refusal(result: Any) -> HTTPException | None:
    """409 ``workspace_object_corrupt`` for an exec result the object check refused."""
    if not isinstance(result, dict) or result.get("exit_code") != WORKSPACE_OBJECT_CORRUPT_EXIT:
        return None
    stderr = str(result.get("stderr") or "")
    if WORKSPACE_OBJECT_CORRUPT not in stderr:
        return None
    found = [
        line.split(":", 1)[1].split(";", 1)[0].strip()
        for line in stderr.splitlines()
        if line.startswith(f"{WORKSPACE_OBJECT_CORRUPT}:")
    ]
    return HTTPException(
        status_code=409,
        detail={
            "error_code": WORKSPACE_OBJECT_CORRUPT,
            "message": (
                f"{WORKSPACE_OBJECT_CORRUPT}: an object in the workspace repository does not "
                f"hash to its id ({', '.join(found) or 'unknown object'}); it was altered by "
                "code that ran in the workspace, so git would report content other than what "
                "the ids commit. Nothing was read, changed or pushed. Re-provision the "
                "workspace."
            ),
            "found": found,
        },
    )


class _PoisonGuardedManager:
    """Container manager proxy for the git routes: an ``execute_command`` result the
    poison check (or the object-integrity check) refused raises its 409 at once, so no route can
    read a refused command's (empty) output as an answer. Everything else is delegated
    unchanged."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def execute_command(self, *args: Any, **kwargs: Any) -> Any:
        result = await self._inner.execute_command(*args, **kwargs)
        refusal = (
            workspace_poisoned_refusal(result)
            or workspace_object_corrupt_refusal(result)
            or workspace_submodules_refusal(result)
        )
        if refusal is not None:
            detail = refusal.detail
            logger.error("%s", detail.get("message") if isinstance(detail, dict) else detail)
            raise refusal
        return result


def get_git_manager() -> Any:
    """``get_manager()`` wrapped so a poisoned workspace is a 409, never a git answer."""
    return _PoisonGuardedManager(get_manager())


def _sanitize_shell_arg(value: str) -> str:
    """Sanitize a value for safe shell usage using shlex.quote."""
    if not value:
        raise ValueError("Empty value not allowed")
    return shlex.quote(value)


def _normalize_workspace_path(path: str, *, workspace_root: str = _WORKSPACE_ROOT) -> str:
    """Resolve a path strictly below a container's repository workspace.

    ``/workspace`` is the stable API alias even when a Docker-provisioned repo
    is mounted elsewhere. Relative paths resolve from the actual root. The root
    itself is deliberately excluded because callers use this helper for
    destructive move/delete operations. Runtime commands additionally resolve
    parent directories to prevent an intermediate symlink escape.
    """
    stripped = path.strip()
    if not stripped:
        raise ValueError("path must not be empty")
    if any(ord(character) < 32 or ord(character) == 127 for character in stripped):
        raise ValueError("path must not contain control characters")

    root = posixpath.normpath(workspace_root.strip())
    if not posixpath.isabs(root) or root == "/":
        raise ValueError("container repository root is invalid")

    if posixpath.isabs(stripped):
        api_path = posixpath.normpath(stripped)
        if api_path == _WORKSPACE_ROOT or api_path.startswith(f"{_WORKSPACE_ROOT}/"):
            relative_path = posixpath.relpath(api_path, _WORKSPACE_ROOT)
            candidate = posixpath.join(root, relative_path)
        else:
            candidate = api_path
    else:
        candidate = posixpath.join(root, stripped)

    normalized = posixpath.normpath(candidate)
    if normalized == root:
        raise ValueError("the workspace root cannot be moved or deleted")
    if not normalized.startswith(f"{root}/"):
        raise ValueError("path must be inside the repository workspace")
    return normalized


def _normalize_workspace_read_path(path: str, *, workspace_root: str = _WORKSPACE_ROOT) -> str:
    """Map a READ path to an absolute path inside the repository, or raise ValueError.

    Unlike :func:`_normalize_workspace_path` (built for move/delete) the root itself
    is allowed, and traversal is REFUSED rather than normalised away: a caller that
    sends ``src/../../etc`` is told no, not quietly served ``/etc``'s sibling.
    Accepted: a repo-relative path, or an absolute path under the ``/workspace``
    alias. Refused: ``..`` segments (also after one round of percent-decoding, since
    the query string has already been decoded once and ``%252e%252e`` arrives as
    ``%2e%2e``), backslashes, control characters, and any other absolute path.

    This is only the lexical half. Symlinks are resolved IN the container by the
    caller (``realpath`` plus a containment ``case``), because only the container
    knows where a link points.
    """
    stripped = path.strip()
    if not stripped:
        raise ValueError("path must not be empty")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in stripped):
        raise ValueError("path must not contain control characters")
    if "\\" in stripped:
        raise ValueError("backslashes are not permitted")
    for form in (stripped, unquote(stripped)):
        if ".." in form.replace("\\", "/").split("/"):
            raise ValueError("path traversal ('..') is not permitted")

    root = posixpath.normpath(workspace_root.strip())
    if not posixpath.isabs(root) or root == "/":
        raise ValueError("container repository root is invalid")

    if posixpath.isabs(stripped):
        api_path = posixpath.normpath(stripped)
        if api_path == _WORKSPACE_ROOT:
            return root
        if not api_path.startswith(f"{_WORKSPACE_ROOT}/"):
            raise ValueError(
                f"absolute paths must be under {_WORKSPACE_ROOT}; use a repo-relative path"
            )
        return posixpath.join(root, posixpath.relpath(api_path, _WORKSPACE_ROOT))
    normalized = posixpath.normpath(posixpath.join(root, stripped))
    if normalized != root and not normalized.startswith(f"{root}/"):
        raise ValueError("path must be inside the repository workspace")
    return normalized


#: Where ``POST {ws}/repos`` checks out secondary (source) repositories, relative
#: to the repository root. Read-only by contract (py2rust gap #14).
SECONDARY_REPOS_DIR = ".repos"


def _workspace_relative(path: str, *, workspace_root: str = _WORKSPACE_ROOT) -> str | None:
    """``path`` relative to the repository root (lexically), or None if outside it.

    Accepts a repo-relative path, the ``/workspace`` alias or the real root. Purely
    lexical: callers use it to CLASSIFY a path, never to confine one (confinement is
    the normalisers above plus the in-container realpath check).
    """
    stripped = (path or "").strip()
    if not stripped:
        return None
    root = posixpath.normpath(workspace_root.strip() or _WORKSPACE_ROOT)
    if posixpath.isabs(stripped):
        normalized = posixpath.normpath(stripped)
        for base in (_WORKSPACE_ROOT, root):
            if normalized == base:
                return ""
            if normalized.startswith(f"{base}/"):
                return posixpath.relpath(normalized, base)
        return None
    rel = posixpath.normpath(stripped)
    return None if rel == ".." or rel.startswith("../") else ("" if rel == "." else rel)


def secondary_repo_of_path(path: str, *, workspace_root: str = _WORKSPACE_ROOT) -> str | None:
    """``<owner>/<repo>`` when ``path`` lies inside a secondary checkout, else None.

    ``.repos/<owner>__<repo>/…`` → ``<owner>/<repo>`` (GitHub owners never contain
    ``_``, so the first ``__`` splits unambiguously). ``.repos`` itself, and a name
    that is not ``<owner>__<repo>`` (a ``.clone.*`` scratch dir), attribute to None.
    """
    rel = _workspace_relative(path, workspace_root=workspace_root)
    if not rel or not rel.startswith(f"{SECONDARY_REPOS_DIR}/"):
        return None
    name = rel.split("/", 2)[1]
    owner, sep, repo = name.partition("__")
    return f"{owner}/{repo}" if sep and owner and repo else None


def refuse_write_into_secondary_repos(
    path: str, *, workspace_root: str = _WORKSPACE_ROOT, container_name: str = ""
) -> None:
    """409 ``SECONDARY_REPO_READ_ONLY`` for a write/move/delete at or under ``.repos/``.

    Secondary repositories are read-only sources (py2rust gap #14) at the API: this
    refusal answers BEFORE the workspace is touched. ``.repos`` itself is covered
    too: moving or deleting it would drop every source. Lexical only (the checkout
    stays writable on disk: the conveyor runs ``uv sync``/``uv run`` inside it).
    """
    from coding_client.exceptions import BaseAPIException

    rel = _workspace_relative(path, workspace_root=workspace_root)
    if rel is None or not (rel == SECONDARY_REPOS_DIR or rel.startswith(f"{SECONDARY_REPOS_DIR}/")):
        return
    repo = secondary_repo_of_path(path, workspace_root=workspace_root)
    raise BaseAPIException(
        status_code=409,
        message=(
            f"'{path.strip()}' is inside {_WORKSPACE_ROOT}/{SECONDARY_REPOS_DIR}/, where "
            "secondary repositories are checked out READ-ONLY"
            + (f" ({repo})" if repo else "")
            + ". Write into the workspace's own repository instead; to change a source, "
            "re-add it with POST {ws}/repos."
        ),
        error_code="SECONDARY_REPO_READ_ONLY",
        details={
            "error_code": "SECONDARY_REPO_READ_ONLY",
            "container_name": container_name,
            "repo": repo,
        },
    )


def _canonical_workspace_path(path: str, *, workspace_root: str) -> str:
    """Convert an internal repository path to its stable ``/workspace`` API path."""
    relative_path = posixpath.relpath(path, workspace_root)
    return posixpath.join(_WORKSPACE_ROOT, relative_path)


#: Longest branch name accepted. Matches coding-github's clone-token ``branch``
#: field (``max_length=255``): a longer name there is a 422, which the push route
#: would otherwise surface as an opaque "repository facts unavailable" refusal.
MAX_GIT_BRANCH_NAME_LENGTH = 255


def _validate_git_branch_name(branch: str) -> bool:
    """Validate git branch name follows safe patterns."""
    if not branch or len(branch) > MAX_GIT_BRANCH_NAME_LENGTH:
        return False
    if "\n" in branch or "\r" in branch:
        return False
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9/_-]*", branch):
        return False
    return ".." not in branch and not branch.startswith("/") and not branch.endswith("/")


def _allowed_git_push_hosts() -> set[str]:
    """Hosts a container's stored GitHub token may be pushed to.

    Defaults to ``github.com`` only. GitHub Enterprise deployments can widen it
    via ``GIT_PUSH_ALLOWED_HOSTS`` (comma-separated). The default is closed: the
    push credential helper prints the tenant's token for *any* remote git
    contacts, so an un-allowlisted ``remote_url`` would exfiltrate it.
    """
    raw = os.getenv("GIT_PUSH_ALLOWED_HOSTS", "github.com")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def _git_remote_host(remote_url: str) -> str | None:
    """Extract the lowercased host of a git remote URL, or None if unparseable.

    Handles both ``https://host/…`` URLs and scp-like ``user@host:path`` SSH
    syntax (which has no scheme for urlparse to key off).
    """
    if not remote_url:
        return None
    url = remote_url.strip()
    scp = _SCP_LIKE_REMOTE.match(url)
    if scp:
        return scp.group("host").lower()
    host = urlparse(url).hostname
    return host.lower() if host else None


def _validate_git_push_remote(remote_url: str) -> bool:
    """True only when ``remote_url``'s host is on the push allowlist.

    Fail-closed: an empty/unparseable URL, or one pointing at any host outside
    ``_allowed_git_push_hosts()``, is rejected so the container's GitHub token is
    never sent to an untrusted remote.
    """
    host = _git_remote_host(remote_url)
    return host is not None and host in _allowed_git_push_hosts()


def _validate_git_file_path(path: str) -> bool:
    """Validate file path for git operations."""
    if not path:
        return False
    dangerous = [";", "|", "&", "$", "`", "(", ")", "<", ">", "\n", "\r", "\x00"]
    return not any(c in path for c in dangerous)


def _sanitize_error_message(error: str, token: str | None = None) -> str:
    """Remove sensitive data from error messages before returning to client."""
    if not error:
        return error
    sanitized = error
    if token:
        sanitized = sanitized.replace(token, "[REDACTED]")
    # The known-shape patterns live in ONE place, src.core.log_safety, so the
    # API-response sanitiser and the logger sanitiser cannot drift apart. They
    # already had: this function knew ghs_/ghu_ and the logger knew nothing.
    return redact_secrets(sanitized)


def _find_container_by_name(container_manager: Any, container_name: str) -> ContainerInfo | None:
    """Find a container by container_name.

    Args:
        container_manager: The container manager instance
        container_name: The container name to find

    Returns:
        ContainerInfo if found, None otherwise
    """
    try:
        logger.debug("Looking up container with name: %s", container_name)
        try:
            container = container_manager.get_container(container_name)
            logger.debug("Found container by direct name lookup: %s", container.container_name)
            return container
        except ValueError:
            logger.warning("Container '%s' not found", container_name)
            return None
    except (ValueError, OSError, RuntimeError) as e:
        logger.error("Error in container lookup for '%s': %s", container_name, e)
        return None

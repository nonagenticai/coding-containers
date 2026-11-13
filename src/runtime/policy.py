"""Command policy for caller-supplied argv.

Moved here from ``src.routers.container_exec`` so it can be enforced in TWO
places rather than one. The API boundary still checks first, and still answers
400 before a workspace is touched — a fast, honest rejection is the right
behaviour for a bad request. The runtime adapters now check again for anything
marked :attr:`~src.runtime.types.ExecutionProfile.CALLER_SUPPLIED`.

The second check exists because the first one is a property of a CALL PATH, not
of the workspace. ``ContainerManager.execute_command`` reaches the same
backends and applies no allowlist at all; today every one of its callers builds
its own argv, but that is a fact about today's callers rather than about the
function. Enforcing at the runtime layer means the guarantee holds for whoever
calls it next.

It should never fire in practice. If it does, something reached a backend
without passing the boundary, and that is worth an error rather than a shrug.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping

logger = logging.getLogger(__name__)

__all__ = [
    "ANY_TOOLCHAIN",
    "RUST_TOOLCHAIN_COMMANDS",
    "CommandNotPermittedError",
    "check_dangerous_patterns",
    "enforce_command_policy",
    "env_assignment_refusal",
    "env_rule_summary",
    "exec_env_violation",
    "validate_command",
]


#: Admitted ONLY in a ``python+rust`` workspace (py2rust A1), on top of the base
#: allowlist below. ``cargo`` itself is NOT here: it has been on the base list for
#: every workspace since long before toolchains existed, and removing it from
#: python workspaces would be a behaviour change (G1 is additive only). In a
#: python workspace these binaries do not exist anyway; gating them is about
#: keeping the python allowlist exactly what it was.
RUST_TOOLCHAIN_COMMANDS = frozenset(
    {
        "rustc",
        "rustup",
        "rustfmt",
        "cargo-clippy",
        "cargo-fmt",
        "cargo-nextest",
        "cargo-audit",
        "cargo-deny",
        "sqlx",
        "cargo-sqlx",
        # The ONLY way to run SQL: `ws-pg psql <db> -c <sql>` pins host, port,
        # user and flags itself (see scripts/ws-pg). Raw `psql` is deliberately
        # NOT admitted: its argv grammar (bundled short options, abbreviated
        # long options, conninfo/URL/service hosts, `\connect`, `\!`) cannot be
        # constrained to loopback by argv inspection.
        "ws-pg",
    }
)

#: Boundary sentinel: "admitted by SOME toolchain". The API boundary runs before
#: the workspace (and therefore its toolchain) is looked up, so it uses this to
#: keep the fast 400 for commands no workspace may run; the runtime adapters then
#: re-check with the workspace's real toolchain.
ANY_TOOLCHAIN = "*"


#: ``env`` flags that take no argument and cannot change WHICH program runs.
_ENV_PLAIN_FLAGS = frozenset({"-i", "-", "--ignore-environment", "-0", "--null", "-v", "--debug"})
# ---------------------------------------------------------------------------
# Per-command environment: ONE rule for every door a caller sets one
# ---------------------------------------------------------------------------
#
# ``env NAME=VALUE cmd`` in argv (#183), the exec request's ``env`` field (sync and
# async) and a background service's ``env`` all use :func:`env_assignment_refusal`.
# Not a sandbox — the POD is the security boundary and the caller's argv already
# runs arbitrary code in it. The set is exactly origin/main's ``env K=V`` set: each
# redirects which code the wrapped program executes (``PATH``, ``LD_PRELOAD``,
# ``BASH_ENV``…).

#: Refused exactly (compared upper-cased).
_ENV_REFUSED_EXACT: frozenset[str] = frozenset(
    {
        "PATH",
        "IFS",
        "ENV",
        "BASH_ENV",
        "PS4",
        "PROMPT_COMMAND",
    }
)
#: Refused: every name that STARTS with one of these (compared upper-cased).
_ENV_REFUSED_PREFIXES: tuple[str, ...] = (
    "LD_",
    "DYLD_",
    "BASH_FUNC_",
)


def env_rule_summary() -> str:
    """The rule in words, generated from the tables above (API field descriptions)."""
    return (
        "Refused with 400 (names compared case-insensitively): "
        f"{', '.join(sorted(_ENV_REFUSED_EXACT))} and every name starting with "
        f"{', '.join(_ENV_REFUSED_PREFIXES)}. Every other name is admitted."
    )


#: Shape and size limits of the exec request's ``env`` field.
EXEC_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
EXEC_ENV_MAX_VARS = 64
EXEC_ENV_MAX_NAME_CHARS = 128
EXEC_ENV_MAX_VALUE_BYTES = 8 * 1024
EXEC_ENV_MAX_TOTAL_BYTES = 64 * 1024


def env_assignment_refusal(name: str) -> str | None:
    """Why ``name`` may not be set for a caller's command, or None if it may.

    The ONE rule for ``env NAME=VALUE`` in argv, the exec ``env`` field (sync and
    async) and a service's ``env``.
    """
    upper = name.upper()
    if upper in _ENV_REFUSED_EXACT or upper.startswith(_ENV_REFUSED_PREFIXES):
        return f"{name} may not be set: it rewires the workspace pod's own setup"
    return None


def exec_env_violation(env: Mapping[str, str] | None, argv: list[str] | None = None) -> str | None:
    """Why the exec request's ``env`` field is refused, or None.

    Shape (name grammar, counts, sizes, NUL) and the short rule above
    (:func:`env_assignment_refusal`). With ``argv``, also refuses a command whose
    first word contains ``=``: the backends apply
    ``env`` as ``/usr/bin/env K=V <argv>``, where such a word would be read as one
    more ASSIGNMENT and leave ``env`` with no command — which PRINTS the
    environment (and with it the workspace's secrets).
    """
    if not env:
        return None
    if len(env) > EXEC_ENV_MAX_VARS:
        return f"env has {len(env)} variables; at most {EXEC_ENV_MAX_VARS} are allowed"
    total = 0
    for name, value in env.items():
        if not isinstance(name, str) or not isinstance(value, str):
            return "env names and values must be strings"
        if len(name) > EXEC_ENV_MAX_NAME_CHARS or not EXEC_ENV_NAME_RE.match(name):
            return f"env name {name[:64]!r} must match ^[A-Z_][A-Z0-9_]*$ (max {EXEC_ENV_MAX_NAME_CHARS} chars)"
        size = len(value.encode("utf-8"))
        if size > EXEC_ENV_MAX_VALUE_BYTES:
            return f"env value of {name} is {size} bytes; at most {EXEC_ENV_MAX_VALUE_BYTES} are allowed"
        if "\x00" in value:
            return f"env value of {name} contains a NUL byte"
        total += len(name) + 1 + size
        refusal = env_assignment_refusal(name)
        if refusal:
            return refusal
    if total > EXEC_ENV_MAX_TOTAL_BYTES:
        return f"env totals {total} bytes; at most {EXEC_ENV_MAX_TOTAL_BYTES} are allowed"
    if argv and "=" in argv[0]:
        return "a command whose first word contains '=' cannot be combined with env"
    return None


def _env_wrapped_command(args: list[str]) -> list[str] | None:
    """Return the argv ``env args...`` would execute, or None to refuse.

    Mirrors GNU coreutils ``env`` parsing closely enough to be conservative:
    options first (until ``--`` or the first non-option), then ``NAME=VALUE``
    assignments (any token whose ``=`` is not at position 0), then the command.
    Fails CLOSED on anything it does not model:

    * ``-S``/``--split-string`` re-splits a string into a NEW argv that argv
      inspection cannot see, and ``-C``/``--chdir``, ``-P`` (BSD alt path),
      signal options and any unknown or bundled flag are refused outright.
    * No command at all: bare ``env`` PRINTS the environment, which carries the
      workspace's secrets (DB passwords were leaked exactly this way).
    """
    i = 0
    n = len(args)
    while i < n:
        tok = args[i]
        if tok == "--":
            i += 1
            break
        if tok in _ENV_PLAIN_FLAGS:
            i += 1
            continue
        if tok in ("-u", "--unset"):
            if i + 1 >= n:
                logger.warning("env %s without a variable name", tok)
                return None
            i += 2
            continue
        if tok.startswith("--unset=") or (tok.startswith("-u") and not tok.startswith("--")):
            i += 1
            continue
        if tok.startswith("-"):
            logger.warning("env option %r is not permitted", tok)
            return None
        break
    while i < n and args[i].find("=") > 0:
        name = args[i].split("=", 1)[0]
        if env_assignment_refusal(name):
            logger.warning("env may not set %s", name.upper())
            return None
        i += 1
    rest = args[i:]
    if not rest:
        logger.warning("env without a command (would print the environment)")
        return None
    return rest


def validate_command(command: list[str], toolchain: str = "python") -> bool:
    """Validate command for security.

    Uses a whitelist approach to allow only safe, commonly-used commands.
    Commands are matched by their base name (first argument).

    Args:
        command: List of command arguments
        toolchain: The workspace's toolchain (``python``, ``python+rust``) or
            :data:`ANY_TOOLCHAIN` at the API boundary.

    Returns:
        True if command is allowed, False otherwise

    Examples:
        >>> validate_command(["ls", "-la"])
        True
        >>> validate_command(["python", "--version"])
        True
        >>> validate_command(["rm", "-rf", "/"])
        False
    """
    if not command or len(command) == 0:
        logger.warning("Empty command provided")
        return False

    # Whitelist of allowed commands
    allowed_commands = {
        # File operations
        "ls",
        "cat",
        "pwd",
        "echo",
        "find",
        "head",
        "tail",
        "wc",
        "grep",
        # Process inspection
        "ps",
        "top",
        "htop",
        "df",
        "du",
        # Programming languages
        "python",
        "python3",
        "node",
        "ruby",
        "java",
        "go",
        "rust",
        "npm",
        "pip",
        "pip3",
        # `uv` is how a Python workspace becomes RUNNABLE, and without it the
        # conveyor could provision a container it could never use. Measured
        # 2026-08-14: the AA `verification_test` card runs
        # `/workspace/.venv/bin/python -c "import <pkg>"` and the repo's own
        # suite via that same venv, and both exited 127 -- the venv did not
        # exist. It does not exist because the k8s provisioner (see
        # core/kubernetes_container_provisioner.py) is a THIN VOLUME-HOLDER: an
        # init container clones the repo and the main container sleeps. Nothing
        # installs dependencies at provision time, and the only server-side
        # installer (core/container_installer.py) is docker-only, so
        # /install-dependencies is dead on the k8s deployment. The venv that
        # DID exist in the upgrade container was made by the package-upgrade
        # runner, which execs uv through container_manager.execute_command --
        # the INTERNAL path, which never passes this whitelist. So the caller
        # had no way to say `uv sync --frozen` even though it demonstrably
        # works when run directly.
        # Blast radius: uv installs from the repo's own committed manifest into
        # the repo's own throwaway per-repo container. That is strictly
        # narrower than `pip` and `apt-get`, both already allowed above/below,
        # and uv is on PATH in codegraph-runtime by design (the provisioner
        # installs it so `uv add` can run for cross-major upgrades).
        # Pinned by tests/unit/test_container_exec_router.py.
        "uv",
        # bun: measured 2026-09-17, 5 org repos commit a text `bun.lock` (w2v_frontend,
        # orchestrator, chat-bridge x3) and none `bun.lockb`; baked into
        # codegraph-runtime at 1.4.4 (latest release). Same blast radius
        # as npm above: installs the repo's own committed lock into its own
        # throwaway container. Pinned by tests/unit/test_container_exec_router.py.
        "bun",
        # ws-redis (py2rust gap #4 full): the ONLY way to run the workspace's
        # loopback Valkey — it pins bind/port/persistence/maxmemory itself
        # (scripts/ws-redis). Raw valkey-server/valkey-cli are NOT admitted:
        # their argv can bind beyond loopback or load a module. Baked into every
        # toolchain's image, so it is on the base list.
        "ws-redis",
        "yarn",
        "bundler",
        "mvn",
        "cargo",
        # Version and info
        "version",
        "--version",
        "-version",
        "which",
        "whoami",
        # Git operations (read-only)
        "git",
        "git-log",
        "git-status",
        "git-diff",
        "git-show",
        # Package managers
        "apt",
        "apt-get",
        "yum",
        "brew",
        # Text processing
        "sed",
        "awk",
        "tr",
        "cut",
        "sort",
        "uniq",
        "diff",
        # System info
        "uname",
        "lsb_release",
        "hostnamectl",
        # Basic utilities
        "env",
        "date",
        "uptime",
        "free",
        # AI coding agents
        "claude",
        "codex",
        # Code-analysis tools (run by coding-analysis / coding-testing via exec
        # into the per-repo codegraph-runtime container; installed in that image).
        # Python
        "ruff",
        "ty",
        "bandit",
        "radon",
        "vulture",
        # JS / TS
        "eslint",
        "prettier",
        "tsc",
        "jshint",
        # Multi-language
        "semgrep",
    }

    base_command = command[0]

    # Extract command name (handle paths like /usr/bin/python)
    if "/" in base_command:
        base_command = base_command.split("/")[-1]

    # `env` is a launcher, not a leaf (#183): `env <prog> ...` used to be
    # admitted on `env`'s basename alone, so ANY program got through. Validate
    # the program it wraps, recursively, under the SAME toolchain.
    if base_command.lower() == "env":
        wrapped = _env_wrapped_command(command[1:])
        if wrapped is None:
            return False
        return validate_command(wrapped, toolchain=toolchain)

    is_allowed = base_command.lower() in allowed_commands

    # python+rust extras (py2rust A1). The python path is untouched: for
    # toolchain "python" this block never admits anything.
    if (
        not is_allowed
        and toolchain in ("python+rust", ANY_TOOLCHAIN)
        and base_command.lower() in RUST_TOOLCHAIN_COMMANDS
    ):
        is_allowed = True

    if not is_allowed:
        logger.warning(
            "Command '%s' not in whitelist. Allowed: %s",
            base_command,
            ", ".join(sorted(allowed_commands)),
        )

    return is_allowed


_POWER_PROGRAMS = frozenset({"reboot", "shutdown", "poweroff", "halt"})
# ``mkfs`` and its ``mkfs.<fs>`` front-ends; a file such as ``mkfs.txt`` or
# ``mkfs-notes`` is not one of them.
_MKFS_TYPES = frozenset(
    {"ext2", "ext3", "ext4", "xfs", "btrfs", "vfat", "fat", "msdos", "ntfs", "exfat"}
    | {"minix", "bfs", "cramfs", "f2fs"}
)


def _program(token: str) -> str:
    return token.rsplit("/", 1)[-1].lower()


def _is_mkfs(program: str) -> bool:
    name, _, fs_type = program.partition(".")
    return name == "mkfs" and (not fs_type or fs_type in _MKFS_TYPES)


def check_dangerous_patterns(command: list[str]) -> bool:
    """Check for dangerous command patterns.

    Matches PROGRAM TOKENS, never substrings. This used to regex-scan
    ``" ".join(argv)``, so a path or a search term that merely contained a
    word was refused: ``find . -name shutdown_handler.py`` and
    ``grep -n executor.shutdown src/`` both answered 400, and coding-gateway's
    file-reference reported 503 for any real file named like that. A token is
    dangerous when its basename IS the program (``/sbin/shutdown``,
    ``mkfs.ext4``) -- in any position, so ``find -exec shutdown`` is still
    refused.

    Not a sandbox: ``python``/``node`` are allowlisted and can run arbitrary
    code, so this list stops an obvious mistake, not an adversary.

    Args:
        command: List of command arguments

    Returns:
        True if command is safe, False if dangerous
    """
    programs = [_program(token) for token in command]
    for index, program in enumerate(programs):
        rest = [token.lower() for token in command[index + 1 :]]
        if program in _POWER_PROGRAMS or _is_mkfs(program):
            reason = program
        elif program == "rm" and any(
            t.startswith("-") and not t.startswith("--") and "r" in t and "f" in t for t in rest
        ):
            reason = "rm -rf"
        elif program == "dd" and any(t.startswith("if=") for t in rest):
            reason = "dd if="
        elif program == "kill" and any(t in {"-9", "-kill", "-sigkill"} for t in rest):
            reason = "kill -9"
        else:
            continue
        logger.warning("Dangerous pattern detected in command: %s", reason)
        return False

    return True


class CommandNotPermittedError(RuntimeError):
    """A caller-supplied command failed the allowlist at the runtime layer.

    Distinct from a transport failure so the router can map it to 400 — the
    request is wrong, not the backend.
    """


def enforce_command_policy(
    argv, execution_profile, toolchain: str = "python", env: Mapping[str, str] | None = None
) -> None:
    """Apply the allowlist to caller-supplied argv. No-op for server-authored.

    One helper rather than three copies, so a backend cannot end up with its own
    idea of what is permitted — the divergence this whole runtime package exists
    to remove.

    Server-authored argv is exempt deliberately, not by oversight. This service
    builds `sh -c`, `find`, `git` plumbing and `cat` invocations that the
    allowlist does not admit (`sh` is not on it), and gating them on a list
    designed for untrusted input would break the service against itself.
    """
    from .types import ExecutionProfile

    if execution_profile is not ExecutionProfile.CALLER_SUPPLIED:
        return
    if not validate_command(list(argv), toolchain=toolchain):
        raise CommandNotPermittedError(
            f"Command {argv[0]!r} is not permitted" if argv else "Empty command"
        )
    if not check_dangerous_patterns(list(argv)):
        raise CommandNotPermittedError("Command contains a dangerous pattern")
    # The caller's per-command environment, under the SAME rule as the boundary
    # (and as ``env NAME=VALUE`` in argv). Server-authored env (a clone token in
    # GIT_PUSH_TOKEN) never reaches this line: it returned above.
    violation = exec_env_violation(dict(env) if env else None, list(argv))
    if violation:
        raise CommandNotPermittedError(violation)

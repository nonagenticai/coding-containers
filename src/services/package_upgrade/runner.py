"""Run a package upgrade inside a container.

Orchestration:
  1. Resolve container + tenant (raises if missing).
  2. Validate ``project_path`` against traversal / shell metachars.
  3. Read pre-upgrade manifest files (best-effort, missing files = empty).
  4. If the strategy isn't a native rewriter AND mode == cross-major, edit the
     relevant manifests in-process and write them back into the container.
  5. Run each command from the strategy under the project directory.
  6. Read post-upgrade manifest files.

The runner deliberately uses ``container_manager.execute_command`` for the
package-manager invocation rather than constructing one giant shell string,
because the manager already chooses between k8s / agent-host / docker exec
backends. Command tokens are joined with ``shlex.quote`` only when we MUST
use shell features (cd + && + command); for plain reads/writes we use argv
form via ``sh -c`` with quoting.
"""

from __future__ import annotations

import base64
import fnmatch
import logging
import os
import shlex
import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from coding_client.models.package_manager import LanguagePackageManager
from src.core.container_manager import get_container_manager
from src.core.log_safety import redact_secrets

from .strategies import (
    AddSpec,
    MajorUpgradeSpec,
    ModifyAction,
    PackageSpec,
    UpgradeMode,
    get_strategy,
    rewrite_requirements_pins,
)

logger = logging.getLogger(__name__)

# Inside every coding-containers-managed container the cloned repository
# lives under this path.
CONTAINER_REPO_ROOT = "/workspace"

# Characters we refuse in project_path. ``..`` is matched separately.
_FORBIDDEN_PATH_CHARS = set("`$;|&\n\r\x00<>()\"'\\")


# Python package managers may resolve dependencies from a deployment's private
# package index, which is typically fronted by a self-signed CA. uv / pip inside
# a freshly cloned container do NOT trust that CA, so ``uv lock`` and pip
# installs fail with ``invalid peer certificate: UnknownIssuer`` and the whole
# upgrade aborts (exit 1) — blocking a PR for every repo that depends on a
# private-index package. When the deployment names an index via
# ``CODING_PRIVATE_INDEX_HOST``, forward it as a trusted/insecure host so the
# resolver can fetch. Unset (the default) means no prefix at all.
def _private_index_host() -> str:
    """The configured private index host, or "" when the mechanism is off.

    Read at CALL time, not import time, so a test or a deployment can change it
    without re-importing this module (same contract as ``src.core.private_index``).
    """
    return os.environ.get("CODING_PRIVATE_INDEX_HOST", "")


_PYTHON_MANAGERS = frozenset(
    {
        LanguagePackageManager.UV,
        LanguagePackageManager.PIP,
        LanguagePackageManager.POETRY,
        LanguagePackageManager.PIPENV,
    }
)


def _resolver_env_prefix(manager: LanguagePackageManager) -> str:
    """Shell ``VAR=val`` assignments so Python resolvers trust the private
    index's self-signed cert. Returns a trailing-space-terminated prefix for
    Python managers, or ``""`` for non-Python managers / when disabled.

    ``UV_NATIVE_TLS`` makes uv use the OS trust store; ``UV_INSECURE_HOST`` /
    ``PIP_TRUSTED_HOST`` allow the private index host even when its CA is not
    trusted (the standard private-index CI posture).
    """
    host = _private_index_host()
    if manager not in _PYTHON_MANAGERS or not host:
        return ""
    quoted = shlex.quote(host)
    return f"UV_NATIVE_TLS=1 UV_INSECURE_HOST={quoted} PIP_TRUSTED_HOST={quoted} "


@dataclass
class UpgradeRunResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    manifest_files_before: dict[str, str] = field(default_factory=dict)
    manifest_files_after: dict[str, str] = field(default_factory=dict)
    commands_executed: list[list[str]] = field(default_factory=list)


class PathTraversalError(ValueError):
    """Raised when project_path tries to escape /workspace."""


class ManagerNotSupportedError(ValueError):
    """Raised when no strategy is registered for the requested manager."""


def _validate_project_path(project_path: str) -> str:
    """Return the absolute container path. Reject traversal / metachars."""
    if not project_path:
        return CONTAINER_REPO_ROOT

    if any(c in _FORBIDDEN_PATH_CHARS for c in project_path):
        raise PathTraversalError(f"project_path contains forbidden characters: {project_path!r}")

    # Build the candidate path. ``PurePosixPath`` normalises without touching
    # the filesystem and rejects absolute paths gracefully.
    if project_path.startswith("/"):
        raise PathTraversalError(
            f"project_path must be relative to the repo root, got {project_path!r}"
        )

    candidate = (PurePosixPath(CONTAINER_REPO_ROOT) / project_path).as_posix()
    normalised = os.path.normpath(candidate)
    if not (normalised == CONTAINER_REPO_ROOT or normalised.startswith(CONTAINER_REPO_ROOT + "/")):
        raise PathTraversalError(
            f"project_path resolves outside repo root: {project_path!r} -> {normalised}"
        )
    return normalised


def _shell_quote_argv(argv: list[str]) -> str:
    return " ".join(shlex.quote(tok) for tok in argv)


async def _exec_in_container(container_name: str, command: str, tenant_id: str) -> dict:
    """Thin wrapper around ``container_manager.execute_command``.

    Returns the raw dict with ``exit_code``/``stdout``/``stderr``.
    """
    container_manager = get_container_manager()
    return await container_manager.execute_command(container_name, command, tenant_id=tenant_id)


async def _read_manifests(
    container_name: str,
    tenant_id: str,
    workdir: str,
    patterns: tuple[str, ...],
) -> dict[str, str]:
    """Read every manifest matching ``patterns`` under ``workdir``.

    Returns a dict keyed by path RELATIVE to ``workdir``. Missing files are
    silently omitted (a fresh project may not have ``uv.lock`` yet).
    """
    if not patterns:
        return {}

    # Run a single ``ls`` per pattern under workdir; collect filenames.
    # We deliberately keep this shell-side so glob expansion is consistent
    # with how each package manager would see the directory.
    by_path: dict[str, str] = {}
    for pat in patterns:
        if any(ch in pat for ch in "*?["):
            # Glob — list matching files at depth 1 only (intentional: we don't
            # want to slurp deeply-nested .csproj files in a giant monorepo).
            cmd = f"cd {shlex.quote(workdir)} && ls -1 {pat} 2>/dev/null || true"
        else:
            cmd = f"cd {shlex.quote(workdir)} && ls -1 {shlex.quote(pat)} 2>/dev/null || true"
        ls_res = await _exec_in_container(container_name, cmd, tenant_id)
        files = [line.strip() for line in (ls_res.get("stdout") or "").splitlines() if line.strip()]
        for f in files:
            if f in by_path:
                continue
            # Belt-and-braces: skip anything that looks like a directory entry
            # ending in a slash, or anything containing forbidden characters.
            if not f or "/" in f:
                # We deliberately read flat — nested manifests would need a
                # different ``project_path``.
                continue
            cat_cmd = f"cd {shlex.quote(workdir)} && cat {shlex.quote(f)}"
            cat_res = await _exec_in_container(container_name, cat_cmd, tenant_id)
            if cat_res.get("exit_code", 1) != 0:
                continue
            by_path[f] = cat_res.get("stdout") or ""
    return by_path


async def _write_manifest(
    container_name: str,
    tenant_id: str,
    workdir: str,
    filename: str,
    content: str,
) -> None:
    """Write ``content`` to ``workdir/filename`` inside the container.

    Uses base64 to avoid every conceivable shell-escaping disaster with
    TOML / JSON / YAML content. Mirrors the technique in
    ``routers.containers.write_file_content``.
    """
    if any(c in filename for c in _FORBIDDEN_PATH_CHARS) or "/" in filename or ".." in filename:
        raise PathTraversalError(f"refused to write manifest with suspicious name: {filename!r}")
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    cmd = (
        f"cd {shlex.quote(workdir)} && "
        f"echo {shlex.quote(b64)} | base64 -d > {shlex.quote(filename)}"
    )
    res = await _exec_in_container(container_name, cmd, tenant_id)
    if res.get("exit_code", 1) != 0:
        raise RuntimeError(
            f"manifest write failed for {filename}: {(res.get('stderr') or '').strip()[:200]}"
        )


def _filter_manifests_for_edit(
    manifests: dict[str, str], patterns: tuple[str, ...]
) -> dict[str, str]:
    """Restrict ``manifests`` to those matching the strategy's declared files.

    A strategy that lists ``("Gemfile", "Gemfile.lock")`` shouldn't be asked
    to edit ``Gemfile.lock`` (a derived lockfile). The convention here is
    that ``edit_manifest`` itself returns ``content`` unchanged for any file
    it doesn't recognise.
    """
    out: dict[str, str] = {}
    for path, content in manifests.items():
        for pat in patterns:
            if fnmatch.fnmatch(path, pat):
                out[path] = content
                break
    return out


async def run_upgrade(
    *,
    container_name: str,
    tenant_id: str,
    manager: LanguagePackageManager,
    mode: UpgradeMode,
    packages: list[PackageSpec],
    major_specs: list[MajorUpgradeSpec],
    project_path: str = ".",
    dry_run: bool = False,
    timeout_seconds: int = 300,
) -> UpgradeRunResult:
    """Run a package upgrade. See module docstring for the orchestration steps."""
    workdir = _validate_project_path(project_path)

    try:
        strategy = get_strategy(manager)
    except KeyError as exc:
        raise ManagerNotSupportedError(f"no strategy registered for manager: {manager}") from exc

    patterns = strategy.manifest_files

    manifests_before = await _read_manifests(container_name, tenant_id, workdir, patterns)

    if mode is UpgradeMode.CROSS_MAJOR and not strategy.rewrites_manifest_natively:
        editable = _filter_manifests_for_edit(manifests_before, patterns)
        for path, content in editable.items():
            new_content = strategy.edit_manifest(path, content, major_specs)
            if new_content != content:
                await _write_manifest(container_name, tenant_id, workdir, path, new_content)

    if mode is UpgradeMode.IN_RANGE:
        commands = strategy.in_range_commands(packages, dry_run)
    else:
        commands = strategy.cross_major_commands(major_specs, dry_run)

    all_stdout: list[str] = []
    all_stderr: list[str] = []
    exit_code = 0
    started = time.monotonic()

    for argv in commands:
        # ``timeout`` ensures a hung resolver doesn't pin the request open
        # forever, regardless of which exec backend the container manager
        # picks (k8s/docker exec_run has no native timeout).
        timeout_token = ["timeout", str(timeout_seconds)]
        env_prefix = _resolver_env_prefix(manager)
        shell_cmd = (
            f"cd {shlex.quote(workdir)} && {env_prefix}{_shell_quote_argv(timeout_token + argv)}"
        )
        # ⚠️ Denylist layer here, NOT summarize_command — a deliberate exception
        # to the rule used at the exec sites, so the reason is stated. `argv` is
        # a package-manager invocation and the PACKAGE LIST IS THE DIAGNOSTIC
        # PAYLOAD: projecting it away would render `uv pip install
        # requests==2.31.0` as `uv pip install ?` and gut the only log line that
        # says what this service did. The realistic carrier is narrow and known
        # — a pip/npm spec may embed a URL carrying `user:token` userinfo —
        # — and that is exactly the shape `redact_secrets`' userinfo rule
        # removes. A novel-shaped secret in a package spec would survive; that
        # residual is accepted here and nowhere on the exec path.
        logger.info(
            "package_upgrade: container=%s manager=%s mode=%s cmd=%s",
            container_name,
            manager.value,
            mode.value,
            redact_secrets(repr(argv)),
        )
        res = await _exec_in_container(container_name, shell_cmd, tenant_id)
        all_stdout.append(res.get("stdout") or "")
        all_stderr.append(res.get("stderr") or "")
        exit_code = int(res.get("exit_code") or 0)
        if exit_code == 127:
            # Manager binary not in pod (`timeout: failed to run command 'X':
            # No such file or directory`). The image needs that toolchain
            # baked in — surface the fix path explicitly so callers don't
            # waste a debug round-trip parsing the timeout wrapper output.
            all_stderr.append(
                f"\n[coding-containers] manager binary not found in pod: "
                f"{argv[0]!r}. Rebuild Dockerfile.runtime to include it, "
                f"or override PROVISIONER_IMAGE on the coding-containers "
                f"Deployment to point at an image that ships {argv[0]!r}.\n"
            )
        if exit_code != 0:
            # Stop on first failure — running subsequent commands against a
            # half-broken lockfile produces misleading errors.
            break

    duration = time.monotonic() - started

    manifests_after = await _read_manifests(container_name, tenant_id, workdir, patterns)

    # --- keep a sibling requirements.txt in sync with the resolved lock -------
    # Managers like uv rewrite pyproject.toml + uv.lock but never a pip-style
    # ``requirements.txt`` that the project may also ship. Left alone, an upgraded
    # package stays stale-pinned there — so anything reading requirements.txt
    # (the dep-CVE scanner, pip consumers, Docker builds) never sees the bump.
    # Bump only the upgraded packages' pins to the now-resolved lock version;
    # the file is never reformatted. Best-effort: a sync hiccup must not fail an
    # otherwise-successful upgrade.
    if exit_code == 0 and not dry_run and strategy.syncs_requirements_after_upgrade:
        try:
            upgraded_names = [p.name for p in packages] + [s.name for s in major_specs]
            constraints = (
                strategy.requirements_sync_constraints(upgraded_names, manifests_after)
                if upgraded_names
                else {}
            )
            if constraints:
                req = await _read_manifests(
                    container_name, tenant_id, workdir, ("requirements.txt",)
                )
                original = req.get("requirements.txt")
                if original is not None:
                    synced = rewrite_requirements_pins(original, constraints)
                    if synced != original:
                        await _write_manifest(
                            container_name, tenant_id, workdir, "requirements.txt", synced
                        )
                        manifests_before.setdefault("requirements.txt", original)
                        manifests_after["requirements.txt"] = synced
                        logger.info(
                            "package_upgrade: synced requirements.txt pins to lock for %s",
                            sorted(constraints),
                        )
        except Exception:  # noqa: BLE001 — never fail a good upgrade on sync
            logger.warning(
                "package_upgrade: requirements.txt sync skipped after upgrade",
                exc_info=True,
            )

    return UpgradeRunResult(
        exit_code=exit_code,
        stdout="\n".join(s for s in all_stdout if s),
        stderr="\n".join(s for s in all_stderr if s),
        duration_seconds=duration,
        manifest_files_before=manifests_before,
        manifest_files_after=manifests_after,
        commands_executed=commands,
    )


async def run_modify(
    *,
    container_name: str,
    tenant_id: str,
    manager: LanguagePackageManager,
    action: ModifyAction,
    add_specs: list[AddSpec],
    remove_names: list[str],
    project_path: str = ".",
    dry_run: bool = False,
    timeout_seconds: int = 600,
) -> UpgradeRunResult:
    """Run ADD / REMOVE / REPLACE for a package set inside a container.

    Same orchestration shape as :func:`run_upgrade` but with the verbs
    flipped: manifest read → optional edit for non-native strategies →
    command exec → manifest read. REPLACE runs remove-commands first then
    add-commands; if the remove block exits non-zero we stop before
    touching add so callers can refuse the commit.
    """
    workdir = _validate_project_path(project_path)

    try:
        strategy = get_strategy(manager)
    except KeyError as exc:
        raise ManagerNotSupportedError(f"no strategy registered for manager: {manager}") from exc

    patterns = strategy.manifest_files
    manifests_before = await _read_manifests(container_name, tenant_id, workdir, patterns)

    # --- manifest pre-edits for non-native strategies ---------------------
    # Order matters for REPLACE: remove then add. For ADD-only / REMOVE-only,
    # only the relevant block runs.
    if action in (ModifyAction.REMOVE, ModifyAction.REPLACE) and remove_names:
        if not strategy.rewrites_manifest_natively_on_remove:
            for path, content in _filter_manifests_for_edit(manifests_before, patterns).items():
                new_content = strategy.edit_manifest_for_remove(path, content, remove_names)
                if new_content != content:
                    await _write_manifest(container_name, tenant_id, workdir, path, new_content)
                    manifests_before[path] = new_content
                    # ^ shadow update so the subsequent add-edit sees the
                    # post-remove state when this is REPLACE.

    if action in (ModifyAction.ADD, ModifyAction.REPLACE) and add_specs:
        if not strategy.rewrites_manifest_natively_on_add:
            for path, content in _filter_manifests_for_edit(manifests_before, patterns).items():
                new_content = strategy.edit_manifest_for_add(path, content, add_specs)
                if new_content != content:
                    await _write_manifest(container_name, tenant_id, workdir, path, new_content)

    # --- command sequence -------------------------------------------------
    commands: list[list[str]] = []
    if action is ModifyAction.REMOVE:
        commands = strategy.remove_commands(remove_names, dry_run)
    elif action is ModifyAction.ADD:
        commands = strategy.add_commands(add_specs, dry_run)
    elif action is ModifyAction.REPLACE:
        # Remove first, then add. The runner's first-failure-stops semantics
        # mean a half-broken state (remove succeeded, add failed) surfaces
        # as exit_code != 0 — the coding-security orchestrator refuses to
        # commit in that case so no half-migration ships.
        commands = strategy.remove_commands(remove_names, dry_run) + strategy.add_commands(
            add_specs, dry_run
        )

    all_stdout: list[str] = []
    all_stderr: list[str] = []
    exit_code = 0
    started = time.monotonic()

    for argv in commands:
        timeout_token = ["timeout", str(timeout_seconds)]
        env_prefix = _resolver_env_prefix(manager)
        shell_cmd = (
            f"cd {shlex.quote(workdir)} && {env_prefix}{_shell_quote_argv(timeout_token + argv)}"
        )
        # Same reasoning as the package_upgrade render above.
        logger.info(
            "package_modify: container=%s manager=%s action=%s cmd=%s",
            container_name,
            manager.value,
            action.value,
            redact_secrets(repr(argv)),
        )
        res = await _exec_in_container(container_name, shell_cmd, tenant_id)
        all_stdout.append(res.get("stdout") or "")
        all_stderr.append(res.get("stderr") or "")
        exit_code = int(res.get("exit_code") or 0)
        if exit_code == 127:
            # Manager binary not in pod (see run_upgrade for the same handling).
            all_stderr.append(
                f"\n[coding-containers] manager binary not found in pod: "
                f"{argv[0]!r}. Rebuild Dockerfile.runtime to include it, "
                f"or override PROVISIONER_IMAGE on the coding-containers "
                f"Deployment to point at an image that ships {argv[0]!r}.\n"
            )
        if exit_code != 0:
            break

    duration = time.monotonic() - started
    manifests_after = await _read_manifests(container_name, tenant_id, workdir, patterns)

    return UpgradeRunResult(
        exit_code=exit_code,
        stdout="\n".join(s for s in all_stdout if s),
        stderr="\n".join(s for s in all_stderr if s),
        duration_seconds=duration,
        manifest_files_before=manifests_before,
        manifest_files_after=manifests_after,
        commands_executed=commands,
    )

"""Per-package-manager upgrade strategies.

Each strategy knows three things:
  - which manifest files the manager touches (for diffing pre/post)
  - how to build the in-range upgrade command(s) (bump within current constraint)
  - how to build the cross-major upgrade command(s) (rewrite constraint + bump)

Strategies whose ``rewrites_manifest_natively`` is False additionally implement
``edit_manifest`` — the runner reads the manifest, calls ``edit_manifest`` to
produce the new bytes, writes them back into the container, then runs the
in-range command. This covers managers without a constraint-aware ``add`` verb
(pip with ``requirements.txt``, conda, maven, gradle, bundler).

The strategy classes are deliberately dumb and stateless — they only return
command tokens. The runner owns I/O.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from coding_client.models.package_manager import LanguagePackageManager

# Package names that this module is willing to accept. Anything outside this
# pattern would never be a valid name on any of the public registries we
# target and is rejected before reaching command construction. Belt and
# braces — names are also passed as separate argv tokens, not via shell.
# ``\Z`` (not ``$``) anchors to absolute end of string — ``$`` would match
# just before a trailing newline, accidentally letting ``"foo\n"`` through.
# ``:`` admitted for Maven / Gradle coordinates (``groupId:artifactId``).
# ``@`` admitted as a LEADING char for npm scoped packages (``@types/node``,
# ``@scope/pkg``) — the body already allowed ``@``/``/``, only the first char was
# alnum-only, so a scoped name 422'd with "invalid package name" (found live
# 2026-07-04 upgrading ``@types/node`` in an n8n node module).
# These are still safe argv tokens — ``@``/``:`` have no shell-metachar meaning.
_PKG_NAME_RE = re.compile(r"\A[@A-Za-z0-9][A-Za-z0-9._@/+:-]{0,127}\Z")
# Version literals: digits, dots, alpha (for pre-releases), dashes, plus.
_VERSION_RE = re.compile(r"\A[0-9A-Za-z.+-]{1,64}\Z")


class UpgradeMode(StrEnum):
    IN_RANGE = "in-range"
    CROSS_MAJOR = "cross-major"


class ModifyAction(StrEnum):
    """Verbs supported by ``run_modify`` and the ``/package-modify`` endpoint."""

    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"


class PackageSpec(BaseModel):
    """Identifier for an in-range upgrade — just a package name."""

    name: str = Field(..., min_length=1, max_length=128)
    is_dev: bool = False

    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _PKG_NAME_RE.match(v):
            raise ValueError(f"invalid package name: {v!r}")
        return v


class AddSpec(BaseModel):
    """Identifier for a brand-new dependency.

    Distinct from :class:`MajorUpgradeSpec` because adding a dep that the
    manifest never knew about doesn't have a "previous constraint" concept —
    only a desired one.
    """

    name: str = Field(..., min_length=1, max_length=128)
    version: str | None = Field(default=None, max_length=64)
    constraint: str | None = Field(default=None, max_length=128)
    is_dev: bool = False
    extras: list[str] = Field(default_factory=list, max_length=16)

    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _PKG_NAME_RE.match(v):
            raise ValueError(f"invalid package name: {v!r}")
        return v

    @field_validator("version")
    @classmethod
    def _validate_version(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not _VERSION_RE.match(v):
            raise ValueError(f"invalid version: {v!r}")
        return v

    @field_validator("constraint")
    @classmethod
    def _validate_constraint(cls, v: str | None) -> str | None:
        if v is None:
            return v
        forbidden = set("`$\n\r\x00")
        if any(c in forbidden for c in v):
            raise ValueError(f"invalid characters in constraint: {v!r}")
        return v

    @field_validator("extras")
    @classmethod
    def _validate_extras(cls, v: list[str]) -> list[str]:
        for extra in v:
            if not _PKG_NAME_RE.match(extra):
                raise ValueError(f"invalid extra: {extra!r}")
        return v


class MajorUpgradeSpec(BaseModel):
    """Identifier for a cross-major upgrade — name + optional version/constraint."""

    name: str = Field(..., min_length=1, max_length=128)
    target_version: str | None = Field(default=None, max_length=64)
    new_constraint: str | None = Field(default=None, max_length=128)
    is_dev: bool = False

    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _PKG_NAME_RE.match(v):
            raise ValueError(f"invalid package name: {v!r}")
        return v

    @field_validator("target_version")
    @classmethod
    def _validate_version(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not _VERSION_RE.match(v):
            raise ValueError(f"invalid target_version: {v!r}")
        return v

    @field_validator("new_constraint")
    @classmethod
    def _validate_constraint(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # Constraint strings legitimately need ``< > ( ) [ ] , ! ^ ~ = *``
        # depending on ecosystem (PEP 440, SemVer, Maven). We only forbid
        # characters that survive ``shlex.quote`` to attack the shell:
        # NUL / newlines (can break argv terminators) and the four classic
        # command-substitution sigils. Single-quote wrapping by ``shlex.quote``
        # neutralises ``< > ( ) | & ;`` etc.
        forbidden = set("`$\n\r\x00")
        if any(c in forbidden for c in v):
            raise ValueError(f"invalid characters in new_constraint: {v!r}")
        return v


def derive_constraint(target_version: str) -> str:
    """Derive a SemVer-style upper-bound constraint when one isn't supplied.

    ``3.0.0`` → ``>=3.0.0,<4.0.0``. Pre-release tags ignored for the bump.
    """
    head = target_version.split("-", 1)[0].split("+", 1)[0]
    parts = head.split(".")
    try:
        major = int(parts[0])
    except ValueError, IndexError:
        # Non-numeric leading component — caller should pass new_constraint.
        return f">={target_version}"
    return f">={target_version},<{major + 1}.0.0"


class UpgradeStrategy(ABC):
    """Base class — subclasses override ``manifest_files`` + ``in_range_commands``
    + ``cross_major_commands``. Non-native-rewrite strategies also override
    ``rewrites_manifest_natively`` and ``edit_manifest``.

    For ADD / REMOVE / REPLACE flows (driven by ``/package-modify``), strategies
    additionally implement ``add_commands`` + ``remove_commands``. For managers
    without a constraint-aware ``add`` / ``remove`` verb, the corresponding
    ``rewrites_manifest_natively_on_add`` / ``..._on_remove`` returns False and
    ``edit_manifest_for_add`` / ``edit_manifest_for_remove`` is called by the
    runner to edit the manifest in-process before the manager runs.
    """

    manager: LanguagePackageManager

    @property
    @abstractmethod
    def manifest_files(self) -> tuple[str, ...]:
        """Files to diff before/after the upgrade run."""

    @abstractmethod
    def in_range_commands(self, packages: list[PackageSpec], dry_run: bool) -> list[list[str]]:
        """Commands to run for an in-range bump (lockfile-only)."""

    @abstractmethod
    def cross_major_commands(self, specs: list[MajorUpgradeSpec], dry_run: bool) -> list[list[str]]:
        """Commands to run for a cross-major bump (constraint + lockfile)."""

    @abstractmethod
    def add_commands(self, specs: list[AddSpec], dry_run: bool) -> list[list[str]]:
        """Commands to add brand-new dependencies."""

    @abstractmethod
    def remove_commands(self, names: list[str], dry_run: bool) -> list[list[str]]:
        """Commands to remove existing dependencies."""

    @property
    def rewrites_manifest_natively(self) -> bool:
        """True if the cross-major command also rewrites the manifest in place."""
        return True

    @property
    def syncs_requirements_after_upgrade(self) -> bool:
        """True if, after the upgrade commands run, the runner should also keep
        a sibling ``requirements.txt`` in sync with the resolved lock. Most
        managers don't ship a stray requirements.txt, so this defaults False.
        """
        return False

    def requirements_sync_constraints(
        self, upgraded_names: list[str], manifests_after: dict[str, str]
    ) -> dict[str, str]:
        """Map each upgraded package name → the ``requirements.txt`` constraint
        suffix (e.g. ``"==3.4.0"``) to apply, derived from the post-upgrade
        lock in ``manifests_after``. Default: no sync. Only consulted when
        ``syncs_requirements_after_upgrade`` is True.
        """
        return {}

    @property
    def rewrites_manifest_natively_on_add(self) -> bool:
        """True if ``add_commands`` writes the manifest itself (most managers
        with an ``add`` verb do). False = runner must call
        ``edit_manifest_for_add`` to rewrite the manifest first.
        """
        return True

    @property
    def rewrites_manifest_natively_on_remove(self) -> bool:
        """True if ``remove_commands`` writes the manifest itself. False =
        runner must call ``edit_manifest_for_remove`` to delete the manifest
        entry first.
        """
        return True

    def edit_manifest(self, file_path: str, content: str, specs: list[MajorUpgradeSpec]) -> str:
        """Return new manifest content. Only called when not rewrites_manifest_natively."""
        raise NotImplementedError(
            f"{type(self).__name__} declares native rewrite — edit_manifest is unused"
        )

    def edit_manifest_for_add(self, file_path: str, content: str, specs: list[AddSpec]) -> str:
        """Mutate the manifest to record new dependencies.

        Only called when ``rewrites_manifest_natively_on_add`` is False.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares native add — edit_manifest_for_add is unused"
        )

    def edit_manifest_for_remove(self, file_path: str, content: str, names: list[str]) -> str:
        """Mutate the manifest to drop dependencies.

        Only called when ``rewrites_manifest_natively_on_remove`` is False.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares native remove — edit_manifest_for_remove is unused"
        )


# ============================================================================
# PYTHON
# ============================================================================


class UvStrategy(UpgradeStrategy):
    manager = LanguagePackageManager.UV

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("pyproject.toml", "uv.lock")

    def in_range_commands(self, packages: list[PackageSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["uv", "lock"]
        if dry_run:
            cmd.append("--dry-run")
        for p in packages:
            cmd.extend(["--upgrade-package", p.name])
        return [cmd]

    def cross_major_commands(self, specs: list[MajorUpgradeSpec], dry_run: bool) -> list[list[str]]:
        # ``uv add`` rewrites pyproject.toml's [project.dependencies] AND
        # updates uv.lock — single canonical path. ``--dry-run`` is supported.
        cmd: list[str] = ["uv", "add"]
        if dry_run:
            cmd.append("--dry-run")
        for s in specs:
            if s.is_dev:
                cmd.append("--dev")
            spec_token = _python_spec_token(s)
            cmd.append(spec_token)
        return [cmd]

    def add_commands(self, specs: list[AddSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["uv", "add"]
        if dry_run:
            cmd.append("--dry-run")
        for s in specs:
            if s.is_dev:
                cmd.append("--dev")
            cmd.append(_python_add_token(s))
        return [cmd]

    def remove_commands(self, names: list[str], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["uv", "remove", *names]
        if dry_run:
            cmd.append("--dry-run")
        return [cmd]

    @property
    def syncs_requirements_after_upgrade(self) -> bool:
        # uv projects commonly also ship a pip-style ``requirements.txt`` (for
        # Docker images / pip consumers). ``uv`` rewrites pyproject.toml + uv.lock
        # but never that file, so an upgraded package stays stale-pinned there —
        # and any tool reading requirements.txt (e.g. the dep-CVE scanner) won't
        # see the bump. We bump the upgraded packages' pins to the resolved lock
        # version so all manifests agree. Surgical: only upgraded, already-pinned
        # lines change; the file is never reformatted/regenerated.
        return True

    def requirements_sync_constraints(
        self, upgraded_names: list[str], manifests_after: dict[str, str]
    ) -> dict[str, str]:
        lock = manifests_after.get("uv.lock", "")
        if not lock:
            return {}
        resolved = parse_uv_lock_versions(lock, set(upgraded_names))
        return {name: f"=={version}" for name, version in resolved.items()}


def _python_spec_token(s: MajorUpgradeSpec) -> str:
    if s.new_constraint:
        return f"{s.name}{s.new_constraint}"
    if s.target_version:
        return f"{s.name}=={s.target_version}"
    return f"{s.name}@latest"


# Matches a requirements.txt dependency line: ``name[extras] constraint``.
# Shared by the pip cross-major edit and the uv-strategy requirements sync.
_REQ_LINE_RE = re.compile(r"^\s*([A-Za-z0-9._@/+-]+)\s*(\[[^\]]*\])?\s*(.*)$")
# Pulls ``name``/``version`` pairs out of a uv.lock ``[[package]]`` block.
_UV_LOCK_PKG_RE = re.compile(r'name\s*=\s*"([^"]+)"\s*\nversion\s*=\s*"([^"]+)"')


def rewrite_requirements_pins(
    content: str,
    name_to_constraint: dict[str, str],
    *,
    append_missing: bool = False,
) -> str:
    """Rewrite the pins of named packages inside a ``requirements.txt`` body.

    ``name_to_constraint`` maps a package name (any case) to the constraint
    suffix to apply, e.g. ``"==3.4.0"`` or ``">=3.4.0,<4.0.0"``. Lines whose
    package is not in the map are left byte-identical (comments / blanks too).
    Matching is case-insensitive; the original line's name + ``[extras]`` are
    preserved. When ``append_missing`` is True, packages in the map that are
    absent from the file are appended (using the name exactly as supplied).
    """
    lower = {name.lower(): (name, constraint) for name, constraint in name_to_constraint.items()}
    new_lines: list[str] = []
    seen: set[str] = set()
    for line in content.splitlines(keepends=False):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        m = _REQ_LINE_RE.match(line)
        if not m:
            new_lines.append(line)
            continue
        pkg = m.group(1).lower()
        if pkg not in lower:
            new_lines.append(line)
            continue
        seen.add(pkg)
        extras = m.group(2) or ""
        _, constraint = lower[pkg]
        new_lines.append(f"{m.group(1)}{extras}{constraint}")
    if append_missing:
        for pkg, (orig_name, constraint) in lower.items():
            if pkg in seen:
                continue
            new_lines.append(f"{orig_name}{constraint}")
    return "\n".join(new_lines) + ("\n" if content.endswith("\n") else "")


def parse_uv_lock_versions(content: str, names: set[str]) -> dict[str, str]:
    """Return ``{lowercased name: resolved version}`` for ``names`` from a
    ``uv.lock`` body. uv.lock is TOML; each dependency is a ``[[package]]``
    block with ``name = "x"`` immediately followed by ``version = "y"``. Names
    not present in the lock are simply absent from the result.
    """
    wanted = {n.lower() for n in names}
    out: dict[str, str] = {}
    for m in _UV_LOCK_PKG_RE.finditer(content):
        nm = m.group(1).lower()
        if nm in wanted and nm not in out:
            out[nm] = m.group(2)
    return out


def _python_add_token(s: AddSpec) -> str:
    """Render an :class:`AddSpec` as a pip/uv/poetry-style spec token.

    Includes ``[extras]`` and the constraint/version. Empty constraint falls
    back to ``@latest`` so the manager pins to the newest compatible release.
    """
    name = s.name
    if s.extras:
        name = f"{name}[{','.join(s.extras)}]"
    if s.constraint:
        return f"{name}{s.constraint}"
    if s.version:
        return f"{name}=={s.version}"
    return f"{name}@latest"


def _js_add_token(s: AddSpec) -> str:
    """Render an :class:`AddSpec` for the npm/yarn/pnpm/bun family.

    JS ecosystems don't have ``[extras]``; the name is used as-is.
    """
    if s.constraint:
        return f"{s.name}@{s.constraint}"
    if s.version:
        return f"{s.name}@^{s.version}"
    return f"{s.name}@latest"


def _ensure_venv(dry_run: bool) -> list[list[str]]:
    """Create ``/workspace/.venv`` if absent, so ``uv pip`` has a target.

    ``uv pip install`` discovers ``.venv`` from the working directory and
    refuses to touch the system interpreter without ``--system`` — which is the
    behaviour we want, since the system interpreter is root-owned and
    unwritable. A requirements.txt repo has no lock step to create the venv (the
    uv-managed path gets one from ``uv sync``), so it is created here.

    ``--allow-existing`` makes this idempotent: a repo that already has a venv
    from a previous run keeps it, rather than having its installed packages
    discarded between an upgrade and the test gate that verifies it.

    Skipped entirely on a dry run — a dry run must not mutate the workspace, and
    creating a directory is a mutation even when the install that follows is not.
    """
    if dry_run:
        return []
    return [["uv", "venv", "--allow-existing"]]


class PipStrategy(UpgradeStrategy):
    """A requirements.txt project, driven by ``uv pip`` into ``/workspace/.venv``.

    The MANAGER is pip because that is what the cloned repository uses — this
    strategy is selected by detecting a customer repo's manifests
    (``container_installer.PYTHON_PATTERNS``), not by our own tooling choices.
    Plenty of real repositories ship a requirements.txt and no lock, and we do
    not get to dictate that.

    The TOOL is uv, because we do get to choose that: it is already baked into
    the runtime image, it reads requirements.txt natively, and it is what the
    fleet standardises on.

    Installing into the project venv rather than the interpreter matters twice
    over. It is the only writable target now that workspaces run as an
    unprivileged uid — a bare ``pip install`` resolves to
    ``/usr/local/lib/python3.14/site-packages`` and fails EACCES. And
    ``/workspace/.venv`` is where the rest of the fleet already looks:
    coding-repo-jobs runs its test gate as ``/workspace/.venv/bin/python -m
    pytest`` (orchestrator.py:48) and the exec whitelist's verification path
    assumes the same (container_exec.py:244). ``pip install --user`` would also
    have dodged the permission problem, but into ``$HOME/.local``, which a venv
    does not put on its path — so an upgrade would have "succeeded" into a
    directory the very next test run could not see.

    No constraint-rewriting ``add`` verb — we edit requirements.txt ourselves
    for cross-major.
    """

    manager = LanguagePackageManager.PIP

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("requirements.txt",)

    @property
    def rewrites_manifest_natively(self) -> bool:
        return False

    def in_range_commands(self, packages: list[PackageSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["uv", "pip", "install", "--upgrade"]
        if dry_run:
            cmd.append("--dry-run")
        cmd.extend(p.name for p in packages)
        return [*_ensure_venv(dry_run), cmd]

    def cross_major_commands(self, specs: list[MajorUpgradeSpec], dry_run: bool) -> list[list[str]]:
        # After edit_manifest rewrites requirements.txt, re-install with -r.
        cmd: list[str] = ["uv", "pip", "install", "--upgrade", "-r", "requirements.txt"]
        if dry_run:
            cmd.append("--dry-run")
        return [*_ensure_venv(dry_run), cmd]

    def edit_manifest(self, file_path: str, content: str, specs: list[MajorUpgradeSpec]) -> str:
        if not file_path.endswith("requirements.txt"):
            return content
        name_to_constraint = {
            s.name: s.new_constraint or (f"=={s.target_version}" if s.target_version else "")
            for s in specs
        }
        return rewrite_requirements_pins(content, name_to_constraint, append_missing=True)

    @property
    def rewrites_manifest_natively_on_add(self) -> bool:
        return False

    @property
    def rewrites_manifest_natively_on_remove(self) -> bool:
        return False

    def add_commands(self, specs: list[AddSpec], dry_run: bool) -> list[list[str]]:
        # After ``edit_manifest_for_add`` writes the new entry, re-install
        # from the file so the resolver records the new pin.
        cmd: list[str] = ["uv", "pip", "install", "--upgrade", "-r", "requirements.txt"]
        if dry_run:
            cmd.append("--dry-run")
        return [*_ensure_venv(dry_run), cmd]

    def remove_commands(self, names: list[str], dry_run: bool) -> list[list[str]]:
        # uninstall removes the package from the active environment; the
        # manifest edit is what makes the removal persistent.
        cmd: list[str] = ["uv", "pip", "uninstall", *names]
        if dry_run:
            # pip uninstall has no --dry-run; turn the call into a no-op echo
            # so the runner still observes a 0-exit "command".
            return [["true"]]
        return [cmd]

    def edit_manifest_for_add(self, file_path: str, content: str, specs: list[AddSpec]) -> str:
        if not file_path.endswith("requirements.txt"):
            return content
        existing = {s.name.lower() for s in specs}
        # First strip any existing lines for these packages so we never end
        # up with two entries.
        kept: list[str] = []
        for line in content.splitlines(keepends=False):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                kept.append(line)
                continue
            m = re.match(r"^\s*([A-Za-z0-9._@/+-]+)\s*(\[[^\]]*\])?", line)
            if m and m.group(1).lower() in existing:
                continue
            kept.append(line)
        # Append the new specs.
        for s in specs:
            kept.append(_pip_requirements_line(s))
        return "\n".join(kept) + ("\n" if content.endswith("\n") else "")

    def edit_manifest_for_remove(self, file_path: str, content: str, names: list[str]) -> str:
        if not file_path.endswith("requirements.txt"):
            return content
        targets = {n.lower() for n in names}
        kept: list[str] = []
        for line in content.splitlines(keepends=False):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                kept.append(line)
                continue
            m = re.match(r"^\s*([A-Za-z0-9._@/+-]+)", line)
            if m and m.group(1).lower() in targets:
                continue
            kept.append(line)
        return "\n".join(kept) + ("\n" if content.endswith("\n") else "")


def _pip_requirements_line(s: AddSpec) -> str:
    name = s.name
    if s.extras:
        name = f"{name}[{','.join(s.extras)}]"
    if s.constraint:
        return f"{name}{s.constraint}"
    if s.version:
        return f"{name}=={s.version}"
    return name


class PoetryStrategy(UpgradeStrategy):
    manager = LanguagePackageManager.POETRY

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("pyproject.toml", "poetry.lock")

    def in_range_commands(self, packages: list[PackageSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["poetry", "update"]
        if dry_run:
            cmd.append("--dry-run")
        cmd.extend(p.name for p in packages)
        return [cmd]

    def cross_major_commands(self, specs: list[MajorUpgradeSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["poetry", "add"]
        if dry_run:
            cmd.append("--dry-run")
        for s in specs:
            if s.is_dev:
                cmd.append("--group=dev")
            cmd.append(_python_spec_token(s))
        return [cmd]

    def add_commands(self, specs: list[AddSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["poetry", "add"]
        if dry_run:
            cmd.append("--dry-run")
        for s in specs:
            if s.is_dev:
                cmd.append("--group=dev")
            cmd.append(_python_add_token(s))
        return [cmd]

    def remove_commands(self, names: list[str], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["poetry", "remove", *names]
        if dry_run:
            cmd.append("--dry-run")
        return [cmd]


class PipenvStrategy(UpgradeStrategy):
    manager = LanguagePackageManager.PIPENV

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("Pipfile", "Pipfile.lock")

    def in_range_commands(self, packages: list[PackageSpec], dry_run: bool) -> list[list[str]]:
        # pipenv has no --dry-run; caller is responsible for handling absence.
        return [["pipenv", "update"] + [p.name for p in packages]]

    def cross_major_commands(self, specs: list[MajorUpgradeSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["pipenv", "install"]
        for s in specs:
            if s.is_dev:
                cmd.append("--dev")
            cmd.append(_python_spec_token(s))
        return [cmd]

    def add_commands(self, specs: list[AddSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["pipenv", "install"]
        for s in specs:
            if s.is_dev:
                cmd.append("--dev")
            cmd.append(_python_add_token(s))
        return [cmd]

    def remove_commands(self, names: list[str], dry_run: bool) -> list[list[str]]:
        return [["pipenv", "uninstall", *names]]


class CondaStrategy(UpgradeStrategy):
    manager = LanguagePackageManager.CONDA

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("environment.yml", "environment.yaml")

    @property
    def rewrites_manifest_natively(self) -> bool:
        return False

    def in_range_commands(self, packages: list[PackageSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["conda", "update", "--yes"]
        if dry_run:
            cmd.append("--dry-run")
        cmd.extend(p.name for p in packages)
        return [cmd]

    def cross_major_commands(self, specs: list[MajorUpgradeSpec], dry_run: bool) -> list[list[str]]:
        return [["conda", "env", "update", "-f", "environment.yml", "--prune"]]

    def edit_manifest(self, file_path: str, content: str, specs: list[MajorUpgradeSpec]) -> str:
        if not (file_path.endswith("environment.yml") or file_path.endswith("environment.yaml")):
            return content
        # Conservative regex edit on entries of the form ``- pkg=ver``.
        by_name = {s.name.lower(): s for s in specs}
        out_lines: list[str] = []
        for line in content.splitlines(keepends=False):
            m = re.match(r"^(\s*-\s*)([A-Za-z0-9._@/+-]+)\s*([=<>!~][^#\s]*)?(.*)$", line)
            if not m:
                out_lines.append(line)
                continue
            name = m.group(2).lower()
            spec = by_name.get(name)
            if spec is None:
                out_lines.append(line)
                continue
            constraint = spec.new_constraint or (
                f"={spec.target_version}" if spec.target_version else ""
            )
            out_lines.append(f"{m.group(1)}{m.group(2)}{constraint}{m.group(4) or ''}")
        return "\n".join(out_lines) + ("\n" if content.endswith("\n") else "")

    @property
    def rewrites_manifest_natively_on_add(self) -> bool:
        return False

    @property
    def rewrites_manifest_natively_on_remove(self) -> bool:
        return False

    def add_commands(self, specs: list[AddSpec], dry_run: bool) -> list[list[str]]:
        # ``conda env update`` reconciles environment.yml after edit.
        return [["conda", "env", "update", "-f", "environment.yml"]]

    def remove_commands(self, names: list[str], dry_run: bool) -> list[list[str]]:
        return [["conda", "env", "update", "-f", "environment.yml", "--prune"]]

    def edit_manifest_for_add(self, file_path: str, content: str, specs: list[AddSpec]) -> str:
        if not (file_path.endswith("environment.yml") or file_path.endswith("environment.yaml")):
            return content
        existing = {s.name.lower() for s in specs}
        # Strip existing entries for these packages, then append.
        kept: list[str] = []
        for line in content.splitlines(keepends=False):
            m = re.match(r"^(\s*-\s*)([A-Za-z0-9._@/+-]+)", line)
            if m and m.group(2).lower() in existing:
                continue
            kept.append(line)
        # Find the ``dependencies:`` block and insert below it; if absent,
        # append a new block.
        out = list(kept)
        dep_idx = next(
            (i for i, line in enumerate(out) if re.match(r"^\s*dependencies\s*:\s*$", line)),
            None,
        )
        new_entries = [_conda_entry(s) for s in specs]
        if dep_idx is None:
            out.extend(["dependencies:", *new_entries])
        else:
            out[dep_idx + 1 : dep_idx + 1] = new_entries
        return "\n".join(out) + ("\n" if content.endswith("\n") else "")

    def edit_manifest_for_remove(self, file_path: str, content: str, names: list[str]) -> str:
        if not (file_path.endswith("environment.yml") or file_path.endswith("environment.yaml")):
            return content
        targets = {n.lower() for n in names}
        kept: list[str] = []
        for line in content.splitlines(keepends=False):
            m = re.match(r"^\s*-\s*([A-Za-z0-9._@/+-]+)", line)
            if m and m.group(1).lower() in targets:
                continue
            kept.append(line)
        return "\n".join(kept) + ("\n" if content.endswith("\n") else "")


def _conda_entry(s: AddSpec) -> str:
    head = s.name
    if s.constraint:
        return f"  - {head}{s.constraint}"
    if s.version:
        return f"  - {head}={s.version}"
    return f"  - {head}"


# ============================================================================
# JAVASCRIPT / TYPESCRIPT
# ============================================================================


def _js_spec_token(s: MajorUpgradeSpec) -> str:
    if s.new_constraint:
        # npm/yarn/pnpm interpret a leading ``>=`` etc. correctly when passed
        # as part of the spec token via ``add`` — but we prefer the explicit
        # ``@`` form for clarity. If the user provided a full constraint
        # string, we trust it.
        return f"{s.name}@{s.new_constraint}"
    if s.target_version:
        return f"{s.name}@^{s.target_version}"
    return f"{s.name}@latest"


class NpmStrategy(UpgradeStrategy):
    manager = LanguagePackageManager.NPM

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("package.json", "package-lock.json")

    # `--ignore-scripts` on every install/update: an automated version-bump PR must
    # not COMPILE native addons (e.g. isolated-vm needs a C++ toolchain the sandbox
    # image deliberately doesn't ship) and must not RUN arbitrary dependency
    # postinstall scripts during an unattended upgrade (supply-chain hardening). npm
    # still resolves + rewrites package-lock.json, so the manifest change (the PR
    # deliverable) is intact.
    def in_range_commands(self, packages: list[PackageSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["npm", "update", "--ignore-scripts"]
        if dry_run:
            cmd.append("--dry-run")
        cmd.extend(p.name for p in packages)
        return [cmd]

    def cross_major_commands(self, specs: list[MajorUpgradeSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["npm", "install", "--ignore-scripts"]
        if dry_run:
            cmd.append("--dry-run")
        # npm needs --save-dev as a single flag for all dev specs in a call.
        if any(s.is_dev for s in specs):
            cmd.append("--save-dev")
        cmd.extend(_js_spec_token(s) for s in specs)
        return [cmd]

    def add_commands(self, specs: list[AddSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["npm", "install", "--ignore-scripts"]
        if dry_run:
            cmd.append("--dry-run")
        if any(s.is_dev for s in specs):
            cmd.append("--save-dev")
        cmd.extend(_js_add_token(s) for s in specs)
        return [cmd]

    def remove_commands(self, names: list[str], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["npm", "uninstall", *names]
        if dry_run:
            cmd.append("--dry-run")
        return [cmd]


class YarnStrategy(UpgradeStrategy):
    manager = LanguagePackageManager.YARN

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("package.json", "yarn.lock")

    def in_range_commands(self, packages: list[PackageSpec], dry_run: bool) -> list[list[str]]:
        # yarn has no --dry-run for ``upgrade``; we still pass the flag for
        # Yarn Berry which accepts it on ``up``. Older yarn 1.x ignores it.
        return [["yarn", "upgrade"] + [p.name for p in packages]]

    def cross_major_commands(self, specs: list[MajorUpgradeSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["yarn", "add"]
        if any(s.is_dev for s in specs):
            cmd.append("--dev")
        cmd.extend(_js_spec_token(s) for s in specs)
        return [cmd]

    def add_commands(self, specs: list[AddSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["yarn", "add"]
        if any(s.is_dev for s in specs):
            cmd.append("--dev")
        cmd.extend(_js_add_token(s) for s in specs)
        return [cmd]

    def remove_commands(self, names: list[str], dry_run: bool) -> list[list[str]]:
        return [["yarn", "remove", *names]]


class PnpmStrategy(UpgradeStrategy):
    manager = LanguagePackageManager.PNPM

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("package.json", "pnpm-lock.yaml")

    def in_range_commands(self, packages: list[PackageSpec], dry_run: bool) -> list[list[str]]:
        return [["pnpm", "update"] + [p.name for p in packages]]

    def cross_major_commands(self, specs: list[MajorUpgradeSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["pnpm", "add"]
        if any(s.is_dev for s in specs):
            cmd.append("--save-dev")
        cmd.extend(_js_spec_token(s) for s in specs)
        return [cmd]

    def add_commands(self, specs: list[AddSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["pnpm", "add"]
        if any(s.is_dev for s in specs):
            cmd.append("--save-dev")
        cmd.extend(_js_add_token(s) for s in specs)
        return [cmd]

    def remove_commands(self, names: list[str], dry_run: bool) -> list[list[str]]:
        return [["pnpm", "remove", *names]]


class BunStrategy(UpgradeStrategy):
    manager = LanguagePackageManager.BUN

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("package.json", "bun.lockb")

    def in_range_commands(self, packages: list[PackageSpec], dry_run: bool) -> list[list[str]]:
        return [["bun", "update"] + [p.name for p in packages]]

    def cross_major_commands(self, specs: list[MajorUpgradeSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["bun", "add"]
        if any(s.is_dev for s in specs):
            cmd.append("--dev")
        cmd.extend(_js_spec_token(s) for s in specs)
        return [cmd]

    def add_commands(self, specs: list[AddSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["bun", "add"]
        if any(s.is_dev for s in specs):
            cmd.append("--dev")
        cmd.extend(_js_add_token(s) for s in specs)
        return [cmd]

    def remove_commands(self, names: list[str], dry_run: bool) -> list[list[str]]:
        return [["bun", "remove", *names]]


# ============================================================================
# RUST / JVM / RUBY / GO / PHP / .NET
# ============================================================================


class CargoStrategy(UpgradeStrategy):
    manager = LanguagePackageManager.CARGO

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("Cargo.toml", "Cargo.lock")

    def in_range_commands(self, packages: list[PackageSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["cargo", "update"]
        if dry_run:
            cmd.append("--dry-run")
        for p in packages:
            cmd.extend(["-p", p.name])
        return [cmd]

    def cross_major_commands(self, specs: list[MajorUpgradeSpec], dry_run: bool) -> list[list[str]]:
        # cargo add rewrites Cargo.toml AND updates Cargo.lock since 1.62.
        commands: list[list[str]] = []
        for s in specs:
            cmd: list[str] = ["cargo", "add", s.name]
            if s.target_version:
                cmd.append(f"--vers={s.target_version}")
            if s.is_dev:
                cmd.append("--dev")
            if dry_run:
                cmd.append("--dry-run")
            commands.append(cmd)
        return commands

    def add_commands(self, specs: list[AddSpec], dry_run: bool) -> list[list[str]]:
        commands: list[list[str]] = []
        for s in specs:
            cmd: list[str] = ["cargo", "add", s.name]
            version = s.version or s.constraint
            if version:
                cmd.append(f"--vers={version}")
            if s.is_dev:
                cmd.append("--dev")
            if s.extras:
                cmd.extend(["--features", ",".join(s.extras)])
            if dry_run:
                cmd.append("--dry-run")
            commands.append(cmd)
        return commands

    def remove_commands(self, names: list[str], dry_run: bool) -> list[list[str]]:
        return [["cargo", "remove", *names]]


class MavenStrategy(UpgradeStrategy):
    manager = LanguagePackageManager.MAVEN

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("pom.xml",)

    @property
    def rewrites_manifest_natively(self) -> bool:
        # versions:set-property mutates pom.xml; treat as native.
        return True

    def in_range_commands(self, packages: list[PackageSpec], dry_run: bool) -> list[list[str]]:
        # Maven has no real in-range upgrade; we use versions:use-latest-versions
        # constrained to the includes list. Dry-run uses ``-DgenerateBackupPoms=true``.
        includes = ",".join(p.name for p in packages)
        return [["mvn", "versions:use-latest-versions", f"-Dincludes={includes}"]]

    def cross_major_commands(self, specs: list[MajorUpgradeSpec], dry_run: bool) -> list[list[str]]:
        commands: list[list[str]] = []
        for s in specs:
            if s.target_version is None:
                # Without an explicit target, just allow major bumps.
                commands.append(
                    [
                        "mvn",
                        "versions:use-latest-versions",
                        f"-Dincludes={s.name}",
                        "-DallowMajorUpdates=true",
                    ]
                )
            else:
                # Property-driven version is the safest: requires the user's
                # pom.xml to template ``${pkg.version}`` style.
                commands.append(
                    [
                        "mvn",
                        "versions:set-property",
                        f"-Dproperty={s.name}.version",
                        f"-DnewVersion={s.target_version}",
                    ]
                )
        return commands

    @property
    def rewrites_manifest_natively_on_add(self) -> bool:
        return False

    @property
    def rewrites_manifest_natively_on_remove(self) -> bool:
        return False

    def add_commands(self, specs: list[AddSpec], dry_run: bool) -> list[list[str]]:
        # ``mvn install`` re-resolves after the XML edit.
        return [["mvn", "install", "-N"]]

    def remove_commands(self, names: list[str], dry_run: bool) -> list[list[str]]:
        return [["mvn", "install", "-N"]]

    def edit_manifest_for_add(self, file_path: str, content: str, specs: list[AddSpec]) -> str:
        if not file_path.endswith("pom.xml"):
            return content
        # Maven coordinates: ``groupId:artifactId``. Require ":" in name.
        # We append a <dependency> block per spec inside <dependencies>.
        block_parts: list[str] = []
        for s in specs:
            if ":" not in s.name:
                continue
            group, artifact = s.name.split(":", 1)
            version = s.version or s.constraint or "LATEST"
            block_parts.append(
                "    <dependency>\n"
                f"      <groupId>{group}</groupId>\n"
                f"      <artifactId>{artifact}</artifactId>\n"
                f"      <version>{version}</version>\n"
                + ("      <scope>test</scope>\n" if s.is_dev else "")
                + "    </dependency>"
            )
        if not block_parts:
            return content
        block = "\n".join(block_parts)
        if "</dependencies>" in content:
            return content.replace("</dependencies>", f"{block}\n  </dependencies>", 1)
        # If there's no <dependencies> block, insert one before </project>.
        if "</project>" in content:
            return content.replace(
                "</project>",
                f"  <dependencies>\n{block}\n  </dependencies>\n</project>",
                1,
            )
        return content + f"\n<dependencies>\n{block}\n</dependencies>\n"

    def edit_manifest_for_remove(self, file_path: str, content: str, names: list[str]) -> str:
        if not file_path.endswith("pom.xml"):
            return content
        new_content = content
        for name in names:
            if ":" not in name:
                continue
            group, artifact = name.split(":", 1)
            pattern = re.compile(
                r"\s*<dependency>\s*"
                r"<groupId>" + re.escape(group) + r"</groupId>\s*"
                r"<artifactId>" + re.escape(artifact) + r"</artifactId>"
                r".*?</dependency>",
                re.DOTALL,
            )
            new_content = pattern.sub("", new_content)
        return new_content


class GradleStrategy(UpgradeStrategy):
    manager = LanguagePackageManager.GRADLE

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("build.gradle", "build.gradle.kts")

    @property
    def rewrites_manifest_natively(self) -> bool:
        return False

    def in_range_commands(self, packages: list[PackageSpec], dry_run: bool) -> list[list[str]]:
        # No native bump verb; ``--refresh-dependencies`` re-resolves caches
        # but won't cross constraints. Treat in-range as a no-op resolve.
        return [["gradle", "--refresh-dependencies", "dependencies"]]

    def cross_major_commands(self, specs: list[MajorUpgradeSpec], dry_run: bool) -> list[list[str]]:
        return [["gradle", "--refresh-dependencies", "dependencies"]]

    def edit_manifest(self, file_path: str, content: str, specs: list[MajorUpgradeSpec]) -> str:
        if not (file_path.endswith("build.gradle") or file_path.endswith("build.gradle.kts")):
            return content
        new_content = content
        for s in specs:
            if s.target_version is None and s.new_constraint is None:
                continue
            target = s.target_version or (s.new_constraint or "")
            # Match ``"group:artifact:VERSION"`` — only the version is rewritten.
            pattern = re.compile(r'(["\'])([A-Za-z0-9._-]+:' + re.escape(s.name) + r'):[^"\']+\1')
            new_content = pattern.sub(rf"\1\2:{target}\1", new_content)
        return new_content

    @property
    def rewrites_manifest_natively_on_add(self) -> bool:
        return False

    @property
    def rewrites_manifest_natively_on_remove(self) -> bool:
        return False

    def add_commands(self, specs: list[AddSpec], dry_run: bool) -> list[list[str]]:
        return [["gradle", "--refresh-dependencies", "dependencies"]]

    def remove_commands(self, names: list[str], dry_run: bool) -> list[list[str]]:
        return [["gradle", "--refresh-dependencies", "dependencies"]]

    def edit_manifest_for_add(self, file_path: str, content: str, specs: list[AddSpec]) -> str:
        if not (file_path.endswith("build.gradle") or file_path.endswith("build.gradle.kts")):
            return content
        # Names with no ":" are not valid Gradle coordinates — skip them.
        usable = [s for s in specs if ":" in s.name]
        if not usable:
            return content
        new_lines: list[str] = []
        for s in usable:
            version = s.version or s.constraint or ""
            scope = "testImplementation" if s.is_dev else "implementation"
            coord = f"{s.name}:{version}" if version else s.name
            quote = "'" if not file_path.endswith(".kts") else '"'
            new_lines.append(f"    {scope} {quote}{coord}{quote}")
        # Insert before the first closing brace of a ``dependencies {`` block.
        m = re.search(r"(dependencies\s*\{[^}]*?)(\n\s*\})", content, flags=re.DOTALL)
        if m:
            return content.replace(
                m.group(0), m.group(1) + "\n" + "\n".join(new_lines) + m.group(2), 1
            )
        return content + "\ndependencies {\n" + "\n".join(new_lines) + "\n}\n"

    def edit_manifest_for_remove(self, file_path: str, content: str, names: list[str]) -> str:
        if not (file_path.endswith("build.gradle") or file_path.endswith("build.gradle.kts")):
            return content
        new_lines: list[str] = []
        targets = {n for n in names if ":" in n}
        for line in content.splitlines(keepends=False):
            if any(target in line for target in targets):
                continue
            new_lines.append(line)
        return "\n".join(new_lines) + ("\n" if content.endswith("\n") else "")


class GemStrategy(UpgradeStrategy):
    """Bundler-driven Ruby gems. Both GEM and BUNDLER point here."""

    manager = LanguagePackageManager.BUNDLER

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("Gemfile", "Gemfile.lock")

    @property
    def rewrites_manifest_natively(self) -> bool:
        return False

    def in_range_commands(self, packages: list[PackageSpec], dry_run: bool) -> list[list[str]]:
        return [["bundle", "update"] + [p.name for p in packages]]

    def cross_major_commands(self, specs: list[MajorUpgradeSpec], dry_run: bool) -> list[list[str]]:
        # After edit_manifest rewrites Gemfile, bundle update applies the new
        # constraints to Gemfile.lock.
        return [["bundle", "update"] + [s.name for s in specs]]

    def edit_manifest(self, file_path: str, content: str, specs: list[MajorUpgradeSpec]) -> str:
        if not file_path.endswith("Gemfile"):
            return content
        new_content = content
        for s in specs:
            target = s.new_constraint or (f"~> {s.target_version}" if s.target_version else None)
            if target is None:
                continue
            # Match ``gem "name", "..."`` or ``gem 'name', '...'`` and rewrite
            # the version literal. Lines without a version literal are left.
            pattern = re.compile(
                r'(gem\s+(["\']))' + re.escape(s.name) + r'(\2\s*,\s*)(["\'])[^"\']+(\4)'
            )
            new_content = pattern.sub(rf"\g<1>{s.name}\g<3>\g<4>{target}\g<5>", new_content)
        return new_content

    @property
    def rewrites_manifest_natively_on_add(self) -> bool:
        return False

    @property
    def rewrites_manifest_natively_on_remove(self) -> bool:
        return False

    def add_commands(self, specs: list[AddSpec], dry_run: bool) -> list[list[str]]:
        return [["bundle", "install"]]

    def remove_commands(self, names: list[str], dry_run: bool) -> list[list[str]]:
        return [["bundle", "install"]]

    def edit_manifest_for_add(self, file_path: str, content: str, specs: list[AddSpec]) -> str:
        if not file_path.endswith("Gemfile"):
            return content
        existing = {s.name for s in specs}
        # Drop any prior gem line for these names so add is idempotent.
        kept_lines: list[str] = []
        for line in content.splitlines(keepends=False):
            m = re.match(r'\s*gem\s+["\']([A-Za-z0-9._@/+-]+)["\']', line)
            if m and m.group(1) in existing:
                continue
            kept_lines.append(line)
        new_lines: list[str] = []
        for s in specs:
            ver = s.constraint or (f"~> {s.version}" if s.version else None)
            if ver:
                new_lines.append(f'gem "{s.name}", "{ver}"')
            else:
                new_lines.append(f'gem "{s.name}"')
        return "\n".join(kept_lines + new_lines) + ("\n" if content.endswith("\n") else "")

    def edit_manifest_for_remove(self, file_path: str, content: str, names: list[str]) -> str:
        if not file_path.endswith("Gemfile"):
            return content
        targets = set(names)
        kept_lines: list[str] = []
        for line in content.splitlines(keepends=False):
            m = re.match(r'\s*gem\s+["\']([A-Za-z0-9._@/+-]+)["\']', line)
            if m and m.group(1) in targets:
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines) + ("\n" if content.endswith("\n") else "")


class GoModStrategy(UpgradeStrategy):
    manager = LanguagePackageManager.GO_MOD

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("go.mod", "go.sum")

    def in_range_commands(self, packages: list[PackageSpec], dry_run: bool) -> list[list[str]]:
        # Go modules have no ``in-range`` concept — go.mod pins minimum
        # versions and the toolchain picks the actual one. ``go get`` always
        # crosses majors when explicitly requested.
        return [["go", "get"] + [f"{p.name}@latest" for p in packages]]

    def cross_major_commands(self, specs: list[MajorUpgradeSpec], dry_run: bool) -> list[list[str]]:
        return [
            [
                "go",
                "get",
                f"{s.name}@{s.target_version}" if s.target_version else f"{s.name}@latest",
            ]
            for s in specs
        ]

    def add_commands(self, specs: list[AddSpec], dry_run: bool) -> list[list[str]]:
        return [
            [
                "go",
                "get",
                f"{s.name}@{s.version}" if s.version else f"{s.name}@latest",
            ]
            for s in specs
        ]

    def remove_commands(self, names: list[str], dry_run: bool) -> list[list[str]]:
        commands: list[list[str]] = []
        for n in names:
            commands.append(["go", "mod", "edit", f"-droprequire={n}"])
        commands.append(["go", "mod", "tidy"])
        return commands


class ComposerStrategy(UpgradeStrategy):
    manager = LanguagePackageManager.COMPOSER

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("composer.json", "composer.lock")

    def in_range_commands(self, packages: list[PackageSpec], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["composer", "update"]
        if dry_run:
            cmd.append("--dry-run")
        cmd.extend(p.name for p in packages)
        return [cmd]

    def cross_major_commands(self, specs: list[MajorUpgradeSpec], dry_run: bool) -> list[list[str]]:
        commands: list[list[str]] = []
        for s in specs:
            cmd: list[str] = ["composer", "require"]
            if s.is_dev:
                cmd.append("--dev")
            constraint = s.new_constraint or (f"^{s.target_version}" if s.target_version else "*")
            cmd.append(f"{s.name}:{constraint}")
            if dry_run:
                cmd.append("--dry-run")
            commands.append(cmd)
        return commands

    def add_commands(self, specs: list[AddSpec], dry_run: bool) -> list[list[str]]:
        commands: list[list[str]] = []
        for s in specs:
            cmd: list[str] = ["composer", "require"]
            if s.is_dev:
                cmd.append("--dev")
            constraint = s.constraint or (f"^{s.version}" if s.version else "*")
            cmd.append(f"{s.name}:{constraint}")
            if dry_run:
                cmd.append("--dry-run")
            commands.append(cmd)
        return commands

    def remove_commands(self, names: list[str], dry_run: bool) -> list[list[str]]:
        cmd: list[str] = ["composer", "remove", *names]
        if dry_run:
            cmd.append("--dry-run")
        return [cmd]


class DotnetStrategy(UpgradeStrategy):
    """Both NUGET and DOTNET resolve to this strategy."""

    manager = LanguagePackageManager.DOTNET

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("*.csproj", "*.fsproj", "*.vbproj", "packages.config")

    def in_range_commands(self, packages: list[PackageSpec], dry_run: bool) -> list[list[str]]:
        # dotnet add idempotently re-pins to latest.
        return [["dotnet", "add", "package", p.name] for p in packages]

    def cross_major_commands(self, specs: list[MajorUpgradeSpec], dry_run: bool) -> list[list[str]]:
        commands: list[list[str]] = []
        for s in specs:
            cmd: list[str] = ["dotnet", "add", "package", s.name]
            if s.target_version:
                cmd.extend(["--version", s.target_version])
            commands.append(cmd)
        return commands

    def add_commands(self, specs: list[AddSpec], dry_run: bool) -> list[list[str]]:
        commands: list[list[str]] = []
        for s in specs:
            cmd: list[str] = ["dotnet", "add", "package", s.name]
            if s.version:
                cmd.extend(["--version", s.version])
            commands.append(cmd)
        return commands

    def remove_commands(self, names: list[str], dry_run: bool) -> list[list[str]]:
        return [["dotnet", "remove", "package", n] for n in names]


_REGISTRY: dict[LanguagePackageManager, UpgradeStrategy] = {
    LanguagePackageManager.UV: UvStrategy(),
    LanguagePackageManager.PIP: PipStrategy(),
    LanguagePackageManager.POETRY: PoetryStrategy(),
    LanguagePackageManager.PIPENV: PipenvStrategy(),
    LanguagePackageManager.CONDA: CondaStrategy(),
    LanguagePackageManager.NPM: NpmStrategy(),
    LanguagePackageManager.YARN: YarnStrategy(),
    LanguagePackageManager.PNPM: PnpmStrategy(),
    LanguagePackageManager.BUN: BunStrategy(),
    LanguagePackageManager.CARGO: CargoStrategy(),
    LanguagePackageManager.MAVEN: MavenStrategy(),
    LanguagePackageManager.GRADLE: GradleStrategy(),
    LanguagePackageManager.GEM: GemStrategy(),
    LanguagePackageManager.BUNDLER: GemStrategy(),
    LanguagePackageManager.GO_MOD: GoModStrategy(),
    LanguagePackageManager.COMPOSER: ComposerStrategy(),
    LanguagePackageManager.NUGET: DotnetStrategy(),
    LanguagePackageManager.DOTNET: DotnetStrategy(),
}


def get_strategy(manager: LanguagePackageManager) -> UpgradeStrategy:
    """Resolve the strategy for ``manager``. Raises KeyError if unmapped."""
    return _REGISTRY[manager]

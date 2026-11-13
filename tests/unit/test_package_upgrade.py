"""Tests for the typed /package-upgrade endpoint + strategy dispatch."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from coding_client.models.package_manager import LanguagePackageManager
from src.services.package_upgrade.runner import (
    PathTraversalError,
    _resolver_env_prefix,
    _validate_project_path,
)
from src.services.package_upgrade.strategies import (
    AddSpec,
    MajorUpgradeSpec,
    PackageSpec,
    derive_constraint,
    get_strategy,
)

# ---------------------------------------------------------------------------
# Path-traversal validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "good_path,resolved",
    [
        (".", "/workspace"),
        ("", "/workspace"),
        ("packages/api", "/workspace/packages/api"),
        ("./sub/dir", "/workspace/sub/dir"),
    ],
)
def test_validate_project_path_accepts_relative(good_path, resolved):
    assert _validate_project_path(good_path) == resolved


@pytest.mark.parametrize(
    "bad_path",
    [
        "../etc",
        "packages/../../etc",
        "/etc/passwd",
        "foo;rm -rf /",
        "foo`cat`",
        "foo$bar",
        "foo|cat",
        "foo&ls",
        "foo<>",
        "foo\nbar",
        'foo"bar',
    ],
)
def test_validate_project_path_rejects_traversal_and_metachars(bad_path):
    with pytest.raises(PathTraversalError):
        _validate_project_path(bad_path)


# ---------------------------------------------------------------------------
# Strategy command construction — argv only, no shell tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "manager,packages,expected_head",
    [
        (LanguagePackageManager.UV, ["httpx"], ["uv", "lock", "--upgrade-package", "httpx"]),
        (LanguagePackageManager.POETRY, ["httpx"], ["poetry", "update", "httpx"]),
        (LanguagePackageManager.PIP, ["httpx"], ["uv", "venv", "--allow-existing"]),
        (LanguagePackageManager.NPM, ["react"], ["npm", "update", "--ignore-scripts"]),
        (LanguagePackageManager.YARN, ["react"], ["yarn", "upgrade", "react"]),
        (LanguagePackageManager.PNPM, ["react"], ["pnpm", "update", "react"]),
        (LanguagePackageManager.CARGO, ["serde"], ["cargo", "update", "-p", "serde"]),
        (
            LanguagePackageManager.GO_MOD,
            ["github.com/gin-gonic/gin"],
            ["go", "get", "github.com/gin-gonic/gin@latest"],
        ),
        (
            LanguagePackageManager.COMPOSER,
            ["guzzlehttp/guzzle"],
            ["composer", "update", "guzzlehttp/guzzle"],
        ),
    ],
)
def test_in_range_command_construction(manager, packages, expected_head):
    strategy = get_strategy(manager)
    specs = [PackageSpec(name=n) for n in packages]
    commands = strategy.in_range_commands(specs, dry_run=False)
    # Compare the prefix — strategies are free to add trailing flags.
    assert commands[0][: len(expected_head)] == expected_head


@pytest.mark.parametrize(
    "manager,name,version,expected_token",
    [
        (LanguagePackageManager.UV, "fastmcp", "3.0.0", "fastmcp==3.0.0"),
        (LanguagePackageManager.POETRY, "fastmcp", "3.0.0", "fastmcp==3.0.0"),
        (LanguagePackageManager.NPM, "react", "19.0.0", "react@^19.0.0"),
        (LanguagePackageManager.YARN, "react", "19.0.0", "react@^19.0.0"),
        (LanguagePackageManager.PNPM, "react", "19.0.0", "react@^19.0.0"),
        (LanguagePackageManager.COMPOSER, "guzzlehttp/guzzle", "8.0.0", "guzzlehttp/guzzle:^8.0.0"),
    ],
)
def test_cross_major_command_uses_correct_spec_token(manager, name, version, expected_token):
    strategy = get_strategy(manager)
    specs = [MajorUpgradeSpec(name=name, target_version=version)]
    commands = strategy.cross_major_commands(specs, dry_run=False)
    flat = " ".join(commands[0])
    assert expected_token in flat


def test_uv_cross_major_command_includes_dry_run_flag():
    strategy = get_strategy(LanguagePackageManager.UV)
    specs = [MajorUpgradeSpec(name="httpx", target_version="1.0.0")]
    cmd = strategy.cross_major_commands(specs, dry_run=True)[0]
    assert cmd[:3] == ["uv", "add", "--dry-run"]


def test_npm_dev_dependency_emits_save_dev_flag():
    strategy = get_strategy(LanguagePackageManager.NPM)
    specs = [MajorUpgradeSpec(name="vitest", target_version="2.0.0", is_dev=True)]
    cmd = strategy.cross_major_commands(specs, dry_run=False)[0]
    assert "--save-dev" in cmd


# ---------------------------------------------------------------------------
# Manifest editing for non-native-rewrite strategies
# ---------------------------------------------------------------------------


def test_pip_requirements_txt_rewrite():
    strategy = get_strategy(LanguagePackageManager.PIP)
    before = "httpx>=0.25,<1.0\nrequests==2.31.0\n"
    specs = [MajorUpgradeSpec(name="httpx", new_constraint=">=1.0,<2.0")]
    after = strategy.edit_manifest("requirements.txt", before, specs)
    assert "httpx>=1.0,<2.0" in after
    assert "requests==2.31.0" in after  # untouched


def test_pip_requirements_txt_appends_missing_package():
    strategy = get_strategy(LanguagePackageManager.PIP)
    before = "requests==2.31.0\n"
    specs = [MajorUpgradeSpec(name="httpx", target_version="1.0.0")]
    after = strategy.edit_manifest("requirements.txt", before, specs)
    assert "httpx==1.0.0" in after


def test_gem_bundler_rewrite():
    strategy = get_strategy(LanguagePackageManager.BUNDLER)
    before = 'source "https://rubygems.org"\ngem "rails", "~> 6.1"\ngem "puma"\n'
    specs = [MajorUpgradeSpec(name="rails", target_version="7.0")]
    after = strategy.edit_manifest("Gemfile", before, specs)
    assert "~> 7.0" in after
    assert 'gem "puma"' in after  # version-less entries untouched


def test_gradle_rewrite_in_build_gradle():
    strategy = get_strategy(LanguagePackageManager.GRADLE)
    before = (
        "dependencies {\n"
        "    implementation 'com.example:httpx:0.25.0'\n"
        "    implementation 'com.other:lib:1.0.0'\n"
        "}\n"
    )
    specs = [MajorUpgradeSpec(name="httpx", target_version="1.0.0")]
    after = strategy.edit_manifest("build.gradle", before, specs)
    assert "com.example:httpx:1.0.0" in after
    assert "com.other:lib:1.0.0" in after  # untouched


# ---------------------------------------------------------------------------
# Constraint derivation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "version,expected",
    [
        ("3.0.0", ">=3.0.0,<4.0.0"),
        ("19.2.1", ">=19.2.1,<20.0.0"),
        ("0.5.0", ">=0.5.0,<1.0.0"),
    ],
)
def test_derive_constraint_semver(version, expected):
    assert derive_constraint(version) == expected


# ---------------------------------------------------------------------------
# Spec validation — package name + version must reject shell metachars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_name",
    ["foo;rm", "foo`bar", "foo$bar", "foo|bar", "foo bar", "foo\nbar", ""],
)
def test_package_spec_rejects_dangerous_names(bad_name):
    with pytest.raises(Exception):
        PackageSpec(name=bad_name)


@pytest.mark.parametrize(
    "good_name",
    ["@types/node", "@scope/pkg", "@nonagentic/license-validator", "typescript", "httpx"],
)
def test_package_spec_accepts_npm_scoped_names(good_name):
    # npm scoped packages start with '@' — the validator must accept a leading '@'
    # (regression: '@types/node' 422'd with "invalid package name", found live 2026-07-04).
    assert PackageSpec(name=good_name).name == good_name
    assert MajorUpgradeSpec(name=good_name, target_version="1.0.0").name == good_name


@pytest.mark.parametrize(
    "bad_version",
    ["1.0; rm -rf", "1.0`cat`", "1.0$x", "1.0\n"],
)
def test_major_upgrade_spec_rejects_dangerous_versions(bad_version):
    with pytest.raises(Exception):
        MajorUpgradeSpec(name="foo", target_version=bad_version)


# ---------------------------------------------------------------------------
# End-to-end via the FastAPI route — exec is fully mocked.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_endpoint_in_range_uv_happy_path(client, monkeypatch):
    """POST /api/containers/{name}/package-upgrade with manager=uv, mode=in-range.

    Mocks container_manager + execute_command so the test doesn't need Docker.
    Verifies the right argv was constructed and the response is shaped right.
    """
    # Configured BEFORE the request: the prefix is built at command time.
    monkeypatch.setenv("CODING_PRIVATE_INDEX_HOST", "pkg.example.test")

    captured: list[str] = []

    async def fake_execute(self, container_name, command, tenant_id=None):
        captured.append(command)
        # pyproject.toml + uv.lock listed by ``ls -1``; then their contents,
        # then the upgrade command's stdout, then the post-upgrade reads.
        if command.startswith("cd /workspace && ls -1 pyproject.toml"):
            return {"exit_code": 0, "stdout": "pyproject.toml\n", "stderr": "", "success": True}
        if command.startswith("cd /workspace && ls -1 uv.lock"):
            return {"exit_code": 0, "stdout": "uv.lock\n", "stderr": "", "success": True}
        if "cat pyproject.toml" in command:
            return {
                "exit_code": 0,
                "stdout": '[project]\ndependencies = ["httpx>=0.25,<1.0"]\n',
                "stderr": "",
                "success": True,
            }
        if "cat uv.lock" in command:
            return {"exit_code": 0, "stdout": "# lockfile\n", "stderr": "", "success": True}
        # The upgrade command itself.
        return {"exit_code": 0, "stdout": "Resolved 12 packages.", "stderr": "", "success": True}

    def fake_get_container(name, tenant_id=None):
        from unittest.mock import MagicMock

        info = MagicMock()
        info.container_name = name
        return info

    with (
        patch(
            "src.core.container_manager.ContainerManager.execute_command",
            new=fake_execute,
        ),
        patch(
            "src.core.container_manager.ContainerManager.get_container",
            side_effect=fake_get_container,
        ),
    ):
        resp = await client.post(
            "/api/containers/my-repo/package-upgrade",
            json={
                "manager": "uv",
                "mode": "in-range",
                "packages": [{"name": "httpx"}],
                "project_path": ".",
                "dry_run": False,
                "timeout_seconds": 60,
            },
            headers={"X-Tenant-Id": "n8n-dev"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["manager"] == "uv"
    assert body["mode"] == "in-range"
    assert body["exit_code"] == 0
    # The actual ``uv lock --upgrade-package httpx`` must have hit
    # execute_command — find it in captured.
    matched = [
        c
        for c in captured
        if "uv" in c and "lock" in c and "--upgrade-package" in c and "httpx" in c
    ]
    assert matched, f"expected uv lock command, got: {captured}"
    # The uv lock command must carry the private-index TLS-trust prefix, else
    # `uv lock` 500s on the self-signed index cert (UnknownIssuer).
    assert any(
        "UV_INSECURE_HOST=pkg.example.test" in c and "UV_NATIVE_TLS=1" in c for c in matched
    ), f"uv command missing private-index TLS-trust env: {matched}"


def test_resolver_env_prefix_python_managers_trust_private_index(
    monkeypatch,
) -> None:
    """Python managers get UV/PIP trusted-host env when an index is configured."""
    monkeypatch.setenv("CODING_PRIVATE_INDEX_HOST", "pkg.example.test")
    for mgr in (
        LanguagePackageManager.UV,
        LanguagePackageManager.PIP,
        LanguagePackageManager.POETRY,
        LanguagePackageManager.PIPENV,
    ):
        prefix = _resolver_env_prefix(mgr)
        assert "UV_NATIVE_TLS=1" in prefix
        assert "UV_INSECURE_HOST=pkg.example.test" in prefix
        assert "PIP_TRUSTED_HOST=pkg.example.test" in prefix
        assert prefix.endswith(" ")  # safe concatenation before the argv


def test_resolver_env_prefix_is_empty_when_no_index_is_configured(monkeypatch) -> None:
    """The default must not inject anything: a public deployment has no private index."""
    monkeypatch.delenv("CODING_PRIVATE_INDEX_HOST", raising=False)
    for mgr in (
        LanguagePackageManager.UV,
        LanguagePackageManager.PIP,
        LanguagePackageManager.POETRY,
        LanguagePackageManager.PIPENV,
    ):
        assert _resolver_env_prefix(mgr) == ""


def test_resolver_env_prefix_non_python_managers_empty() -> None:
    """npm/cargo/etc. don't touch the private PyPI index — no prefix."""
    for mgr in (LanguagePackageManager.NPM, LanguagePackageManager.CARGO):
        assert _resolver_env_prefix(mgr) == ""


@pytest.mark.asyncio
async def test_endpoint_rejects_mode_in_range_with_no_packages(client):
    resp = await client.post(
        "/api/containers/my-repo/package-upgrade",
        json={
            "manager": "uv",
            "mode": "in-range",
            "packages": [],
            "major_specs": [],
        },
        headers={"X-Tenant-Id": "n8n-dev"},
    )
    assert resp.status_code in (400, 422)


@pytest.mark.asyncio
async def test_endpoint_rejects_path_traversal(client):
    with patch(
        "src.core.container_manager.ContainerManager.get_container",
        return_value=__import__("unittest").mock.MagicMock(),
    ):
        resp = await client.post(
            "/api/containers/my-repo/package-upgrade",
            json={
                "manager": "uv",
                "mode": "in-range",
                "packages": [{"name": "httpx"}],
                "project_path": "../../etc",
            },
            headers={"X-Tenant-Id": "n8n-dev"},
        )
    assert resp.status_code == 400


# ============================================================================
# Strategy add / remove coverage (per-manager command construction)
# ============================================================================


@pytest.mark.parametrize(
    "manager,expected_head",
    [
        (LanguagePackageManager.UV, ["uv", "add"]),
        (LanguagePackageManager.POETRY, ["poetry", "add"]),
        (LanguagePackageManager.PIPENV, ["pipenv", "install"]),
        (LanguagePackageManager.NPM, ["npm", "install"]),
        (LanguagePackageManager.YARN, ["yarn", "add"]),
        (LanguagePackageManager.PNPM, ["pnpm", "add"]),
        (LanguagePackageManager.BUN, ["bun", "add"]),
        (LanguagePackageManager.CARGO, ["cargo", "add"]),
        (LanguagePackageManager.GO_MOD, ["go", "get"]),
        (LanguagePackageManager.COMPOSER, ["composer", "require"]),
        (LanguagePackageManager.DOTNET, ["dotnet", "add", "package"]),
        (LanguagePackageManager.NUGET, ["dotnet", "add", "package"]),
    ],
)
def test_native_add_command_starts_with_expected_verb(manager, expected_head):
    strategy = get_strategy(manager)
    cmd = strategy.add_commands([AddSpec(name="httpx", version="1.0.0")], dry_run=False)[0]
    assert cmd[: len(expected_head)] == expected_head


@pytest.mark.parametrize(
    "manager,expected_head",
    [
        (LanguagePackageManager.UV, ["uv", "remove"]),
        (LanguagePackageManager.POETRY, ["poetry", "remove"]),
        (LanguagePackageManager.NPM, ["npm", "uninstall"]),
        (LanguagePackageManager.YARN, ["yarn", "remove"]),
        (LanguagePackageManager.PNPM, ["pnpm", "remove"]),
        (LanguagePackageManager.BUN, ["bun", "remove"]),
        (LanguagePackageManager.CARGO, ["cargo", "remove"]),
        (LanguagePackageManager.COMPOSER, ["composer", "remove"]),
    ],
)
def test_native_remove_command(manager, expected_head):
    strategy = get_strategy(manager)
    cmd = strategy.remove_commands(["httpx"], dry_run=False)[0]
    assert cmd[: len(expected_head)] == expected_head


def test_go_mod_remove_uses_droprequire_then_tidy():
    strategy = get_strategy(LanguagePackageManager.GO_MOD)
    cmds = strategy.remove_commands(["github.com/example/foo"], dry_run=False)
    assert cmds[0] == ["go", "mod", "edit", "-droprequire=github.com/example/foo"]
    assert cmds[-1] == ["go", "mod", "tidy"]


def test_uv_add_includes_extras_and_version():
    strategy = get_strategy(LanguagePackageManager.UV)
    cmd = strategy.add_commands(
        [AddSpec(name="psycopg", version="3.2.0", extras=["binary", "pool"])],
        dry_run=False,
    )[0]
    assert "psycopg[binary,pool]==3.2.0" in cmd


def test_cargo_add_translates_extras_to_features():
    strategy = get_strategy(LanguagePackageManager.CARGO)
    cmd = strategy.add_commands(
        [AddSpec(name="serde", version="1.0.0", extras=["derive"])],
        dry_run=False,
    )[0]
    assert "--features" in cmd
    assert "derive" in cmd


def test_pip_edit_manifest_for_add_appends_line():
    strategy = get_strategy(LanguagePackageManager.PIP)
    before = "requests==2.31.0\n"
    specs = [AddSpec(name="httpx", version="1.0.0")]
    after = strategy.edit_manifest_for_add("requirements.txt", before, specs)
    assert "httpx==1.0.0" in after
    assert "requests==2.31.0" in after


def test_pip_edit_manifest_for_remove_deletes_line():
    strategy = get_strategy(LanguagePackageManager.PIP)
    before = "httpx>=0.25\nrequests==2.31.0\n"
    after = strategy.edit_manifest_for_remove("requirements.txt", before, ["httpx"])
    assert "httpx" not in after
    assert "requests==2.31.0" in after


def test_bundler_edit_manifest_for_add_inserts_gem_line():
    strategy = get_strategy(LanguagePackageManager.BUNDLER)
    before = 'source "https://rubygems.org"\ngem "rails", "7.0"\n'
    specs = [AddSpec(name="puma", version="6.0")]
    after = strategy.edit_manifest_for_add("Gemfile", before, specs)
    assert 'gem "puma"' in after


def test_maven_edit_manifest_for_add_inserts_dependency_block():
    strategy = get_strategy(LanguagePackageManager.MAVEN)
    before = (
        "<project>\n"
        "  <dependencies>\n"
        "    <dependency><groupId>x</groupId><artifactId>y</artifactId>"
        "<version>1</version></dependency>\n"
        "  </dependencies>\n"
        "</project>\n"
    )
    specs = [AddSpec(name="com.acme:foo", version="2.0.0")]
    after = strategy.edit_manifest_for_add("pom.xml", before, specs)
    assert "<groupId>com.acme</groupId>" in after
    assert "<artifactId>foo</artifactId>" in after
    assert "<version>2.0.0</version>" in after


def test_conda_edit_manifest_for_add_inserts_under_dependencies():
    strategy = get_strategy(LanguagePackageManager.CONDA)
    before = "name: env\ndependencies:\n  - python=3.12\n"
    specs = [AddSpec(name="numpy", version="2.0")]
    after = strategy.edit_manifest_for_add("environment.yml", before, specs)
    assert "numpy=2.0" in after
    assert "python=3.12" in after


@pytest.mark.asyncio
async def test_run_modify_replace_atomicity_remove_then_add_fail():
    """REPLACE = remove + add. If remove succeeds (exit 0) but add fails
    (exit 1), the runner stops on the first failure and returns exit≠0.
    The orchestrator's commit gate then refuses to commit the half-applied
    state — verified separately in coding-security tests.
    """
    from src.services.package_upgrade.runner import run_modify
    from src.services.package_upgrade.strategies import ModifyAction

    call_count = {"n": 0}

    async def fake_execute(self, container_name, command, tenant_id=None):
        if "ls -1" in command or "cat " in command:
            return {"exit_code": 0, "stdout": "pyproject.toml\n", "stderr": "", "success": True}
        # The upgrade commands themselves — alternate exit 0 then exit 1.
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"exit_code": 0, "stdout": "Removed asyncpg", "stderr": "", "success": True}
        return {"exit_code": 1, "stdout": "", "stderr": "psycopg not found", "success": False}

    with patch(
        "src.core.container_manager.ContainerManager.execute_command",
        new=fake_execute,
    ):
        result = await run_modify(
            container_name="my-repo",
            tenant_id="n8n-dev",
            manager=LanguagePackageManager.UV,
            action=ModifyAction.REPLACE,
            add_specs=[AddSpec(name="psycopg", version="3.2.0")],
            remove_names=["asyncpg"],
        )

    assert result.exit_code == 1
    # Stopped after the failing 2nd command — only 2 modify commands ran.
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_run_upgrade_exit_127_appends_actionable_hint():
    """Regression: when a manager binary is missing from the pod (exit 127),
    the runner appends a clear next-step hint naming PROVISIONER_IMAGE +
    Dockerfile.runtime so callers don't have to parse `timeout: failed to
    run command 'uv': No such file or directory` and guess the fix path.
    """
    from src.services.package_upgrade.runner import run_upgrade
    from src.services.package_upgrade.strategies import (
        MajorUpgradeSpec,
        PackageSpec,
        UpgradeMode,
    )

    async def fake_execute(self, container_name, command, tenant_id=None):
        if "ls -1" in command or "cat " in command:
            return {"exit_code": 0, "stdout": "pyproject.toml\n", "stderr": "", "success": True}
        # Simulate `timeout: failed to run command 'uv': No such file or directory`.
        return {
            "exit_code": 127,
            "stdout": "",
            "stderr": "timeout: failed to run command 'uv': No such file or directory\n",
            "success": False,
        }

    with patch(
        "src.core.container_manager.ContainerManager.execute_command",
        new=fake_execute,
    ):
        result = await run_upgrade(
            container_name="my-repo",
            tenant_id="n8n-dev",
            manager=LanguagePackageManager.UV,
            mode=UpgradeMode.CROSS_MAJOR,
            packages=[PackageSpec(name="fastmcp")],
            major_specs=[MajorUpgradeSpec(name="fastmcp", target_version="3.0.0")],
        )

    assert result.exit_code == 127
    assert "manager binary not found in pod" in result.stderr
    assert "PROVISIONER_IMAGE" in result.stderr
    assert "Dockerfile.runtime" in result.stderr
    # Original stderr from `timeout` is preserved (not replaced) so existing
    # log scrapers continue to work.
    assert "No such file or directory" in result.stderr


# ---------------------------------------------------------------------------
# requirements.txt sync after a uv upgrade (manifest-mismatch fix)
# ---------------------------------------------------------------------------


def test_rewrite_requirements_pins_bumps_only_named_and_preserves_extras():
    from src.services.package_upgrade.strategies import rewrite_requirements_pins

    before = "# Authentication\npython-jose[cryptography]==3.3.0  # vuln\nrequests==2.31.0\n"
    after = rewrite_requirements_pins(before, {"python-jose": "==3.4.0"})
    assert "python-jose[cryptography]==3.4.0" in after  # extras + new pin preserved
    assert "requests==2.31.0" in after  # untouched
    assert "# Authentication" in after  # comments untouched
    # No append by default — a package absent from the file is NOT added.
    assert "anyio" not in rewrite_requirements_pins(before, {"anyio": "==4.0.0"})


def test_rewrite_requirements_pins_append_missing_matches_pip_edit_manifest():
    """The extracted helper must reproduce PipStrategy.edit_manifest behaviour."""
    from src.services.package_upgrade.strategies import rewrite_requirements_pins

    before = "requests==2.31.0\n"
    strat = get_strategy(LanguagePackageManager.PIP)
    via_strategy = strat.edit_manifest(
        "requirements.txt", before, [MajorUpgradeSpec(name="httpx", target_version="1.0.0")]
    )
    via_helper = rewrite_requirements_pins(before, {"httpx": "==1.0.0"}, append_missing=True)
    assert via_strategy == via_helper
    assert "httpx==1.0.0" in via_helper


def test_parse_uv_lock_versions_extracts_requested_packages():
    from src.services.package_upgrade.strategies import parse_uv_lock_versions

    lock = (
        '[[package]]\nname = "python-jose"\nversion = "3.4.0"\n\n'
        '[[package]]\nname = "anyio"\nversion = "4.0.0"\n'
    )
    got = parse_uv_lock_versions(lock, {"python-jose", "missing"})
    assert got == {"python-jose": "3.4.0"}  # only requested + present names


def test_uv_strategy_advertises_requirements_sync_others_do_not():
    uv = get_strategy(LanguagePackageManager.UV)
    pip = get_strategy(LanguagePackageManager.PIP)
    assert uv.syncs_requirements_after_upgrade is True
    assert pip.syncs_requirements_after_upgrade is False
    constraints = uv.requirements_sync_constraints(
        ["python-jose"],
        {"uv.lock": '[[package]]\nname = "python-jose"\nversion = "3.4.0"\n'},
    )
    assert constraints == {"python-jose": "==3.4.0"}


def _make_fake_execute(files: dict[str, str], written: dict[str, str]):
    """Build a ContainerManager.execute_command fake backed by ``files``.

    Routes ``ls``/``cat`` reads to ``files`` and captures base64 writes into
    ``written``; any other command (the upgrade verb) returns a clean 0-exit.
    """
    import base64 as _b64

    async def fake_execute(self, container_name, command, tenant_id=None):
        if "base64 -d >" in command:
            fname = command.rsplit(">", 1)[1].strip().strip("'")
            b64 = command.split("echo ", 1)[1].split("| base64", 1)[0].strip().strip("'")
            written[fname] = _b64.b64decode(b64).decode()
            return {"exit_code": 0, "stdout": "", "stderr": "", "success": True}
        if "ls -1" in command:
            for name in files:
                if f"ls -1 {name} " in command or f"ls -1 {name}\n" in command:
                    return {"exit_code": 0, "stdout": name + "\n", "stderr": "", "success": True}
            return {"exit_code": 0, "stdout": "", "stderr": "", "success": True}
        if " cat " in command:
            for name, content in files.items():
                if command.rstrip().endswith(f"cat {name}"):
                    return {"exit_code": 0, "stdout": content, "stderr": "", "success": True}
            return {"exit_code": 1, "stdout": "", "stderr": "", "success": False}
        return {"exit_code": 0, "stdout": "", "stderr": "", "success": True}

    return fake_execute


@pytest.mark.asyncio
async def test_run_upgrade_uv_syncs_requirements_txt_to_resolved_lock():
    """The manifest-mismatch fix: a uv in-range upgrade bumps the upgraded
    package's stale pin in a sibling requirements.txt to the resolved lock
    version, leaving unrelated lines untouched, and surfaces it in the diff.
    """
    from src.services.package_upgrade.runner import run_upgrade
    from src.services.package_upgrade.strategies import PackageSpec, UpgradeMode

    written: dict[str, str] = {}
    files = {
        "pyproject.toml": '[project]\ndependencies = ["python-jose[cryptography]>=3.4.0"]\n',
        "uv.lock": '[[package]]\nname = "python-jose"\nversion = "3.4.0"\n',
        "requirements.txt": "python-jose[cryptography]==3.3.0  # CVE\nrequests==2.31.0\n",
    }
    with patch(
        "src.core.container_manager.ContainerManager.execute_command",
        new=_make_fake_execute(files, written),
    ):
        result = await run_upgrade(
            container_name="my-repo",
            tenant_id="n8n-dev",
            manager=LanguagePackageManager.UV,
            mode=UpgradeMode.IN_RANGE,
            packages=[PackageSpec(name="python-jose")],
            major_specs=[],
        )

    assert result.exit_code == 0
    assert "python-jose[cryptography]==3.4.0" in written["requirements.txt"]
    assert "requests==2.31.0" in written["requirements.txt"]  # untouched
    # The synced file shows up in the diff so the downstream PR carries it.
    assert "requirements.txt" in result.manifest_files_after
    assert "==3.4.0" in result.manifest_files_after["requirements.txt"]
    assert (
        result.manifest_files_before.get("requirements.txt", "")
        .strip()
        .startswith("python-jose[cryptography]==3.3.0")
    )


@pytest.mark.asyncio
async def test_run_upgrade_uv_no_requirements_txt_is_noop():
    """No sibling requirements.txt → nothing written, no crash, clean result."""
    from src.services.package_upgrade.runner import run_upgrade
    from src.services.package_upgrade.strategies import PackageSpec, UpgradeMode

    written: dict[str, str] = {}
    files = {
        "pyproject.toml": '[project]\ndependencies = ["python-jose>=3.4.0"]\n',
        "uv.lock": '[[package]]\nname = "python-jose"\nversion = "3.4.0"\n',
    }
    with patch(
        "src.core.container_manager.ContainerManager.execute_command",
        new=_make_fake_execute(files, written),
    ):
        result = await run_upgrade(
            container_name="my-repo",
            tenant_id="n8n-dev",
            manager=LanguagePackageManager.UV,
            mode=UpgradeMode.IN_RANGE,
            packages=[PackageSpec(name="python-jose")],
            major_specs=[],
        )

    assert result.exit_code == 0
    assert "requirements.txt" not in written
    assert "requirements.txt" not in result.manifest_files_after


class TestRequirementsProjectsUseUvIntoTheProjectVenv:
    """The manager is pip; the tool is uv.

    This strategy is selected by detecting a CUSTOMER repo's manifests
    (container_installer.PYTHON_PATTERNS), so we do not get to stop
    requirements.txt projects existing. We DO get to choose what drives them,
    and uv is already baked into the runtime image.

    Two reasons the target is /workspace/.venv rather than the interpreter:
    workspaces run as an unprivileged uid and cannot write
    /usr/local/lib/python3.14/site-packages; and the rest of the fleet already
    looks in that venv — coding-repo-jobs runs its test gate as
    /workspace/.venv/bin/python (orchestrator.py:48). `pip install --user` would
    have dodged the permission error into $HOME/.local, which a venv does not
    put on its path — so the upgrade would have "succeeded" somewhere the very
    next test run could not see.
    """

    CALLS = [
        lambda s: s.in_range_commands([PackageSpec(name="httpx")], dry_run=False),
        lambda s: s.cross_major_commands([], dry_run=False),
        lambda s: s.add_commands([AddSpec(name="httpx", constraint=">=1.0")], dry_run=False),
    ]

    @pytest.mark.parametrize("call", CALLS)
    def test_no_bare_pip_survives(self, call):
        strategy = get_strategy(LanguagePackageManager.PIP)
        for command in call(strategy):
            assert command[0] == "uv", command

    @pytest.mark.parametrize("call", CALLS)
    def test_the_venv_exists_before_anything_installs_into_it(self, call):
        strategy = get_strategy(LanguagePackageManager.PIP)
        commands = call(strategy)
        assert commands[0] == ["uv", "venv", "--allow-existing"]
        assert any(c[:3] == ["uv", "pip", "install"] for c in commands[1:])

    def test_a_dry_run_creates_nothing(self):
        """A dry run must not mutate the workspace, and mkdir is a mutation."""
        strategy = get_strategy(LanguagePackageManager.PIP)
        commands = strategy.in_range_commands([PackageSpec(name="httpx")], dry_run=True)
        assert not any(c[:2] == ["uv", "venv"] for c in commands)
        assert all("--dry-run" in c for c in commands if c[:3] == ["uv", "pip", "install"])

    def test_uninstall_also_targets_the_venv(self):
        strategy = get_strategy(LanguagePackageManager.PIP)
        command = strategy.remove_commands(["httpx"], dry_run=False)[0]
        assert command[:3] == ["uv", "pip", "uninstall"]

    def test_it_stays_an_argv_list_with_no_shell_tokens(self):
        strategy = get_strategy(LanguagePackageManager.PIP)
        for command in strategy.in_range_commands([PackageSpec(name="httpx")], dry_run=False):
            assert all(isinstance(token, str) for token in command)
            assert not any(tok in command for tok in ("&&", "|", ";", ">"))

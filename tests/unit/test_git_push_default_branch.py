"""The push route refuses a public repository and any push to the default branch.

coding-github #108 refuses default-branch writes through its own API, but its
``clone-token`` endpoint hands this route a ``contents:write`` installation token
and GitHub cannot scope a token to a branch. So ``POST /git/push`` with
``branch=<default>`` was a write to the default branch that bypassed #108.

⛔ This pod has NO egress to api.github.com (only coding-github does; gitops
``services/coding-containers/configmap.yaml``). The route used to read
``GET /repos/{o}/{r}`` itself, so in prod EVERY push failed 503
``repo_visibility_unknown`` (ConnectError, 2026-10-12, nonagenticai/ems). The
facts now come from coding-github's push-scope clone-token answer (``private``,
``default_branch``, ``branch_head``/``default_branch_head``), read there with the
token it minted — never from the workspace, and never from GitHub directly.

Every test here runs under ``_no_github_egress``: any outbound HTTP call fails
the test, and a call to api.github.com is counted and asserted to be zero.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from src.clients.github_client import CloneCredential, RefHead

_OWNER = "acme"
_REPO = "widget"
_URL = f"https://github.com/{_OWNER}/{_REPO}"
_APP_TOKEN = "ghs_app_push_token_test"  # noqa: S105  # pragma: allowlist secret
_BYO_TOKEN = "ghp_byo_push_token_test"  # noqa: S105  # pragma: allowlist secret
_TIP = "a" * 40
_BASE = "b" * 40


@pytest.fixture(autouse=True)
def _no_github_egress():
    """Fail on ANY outbound HTTP from the route; count api.github.com separately.

    ``assert_all_mocked=True``: an unrouted request raises instead of leaving the
    process. The api.github.com route answers 200 (so a regression would otherwise
    *pass*) and is asserted uncalled afterwards.
    """
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as mock:
        github = mock.route(host="api.github.com").mock(
            return_value=httpx.Response(200, json={"default_branch": "main", "private": True})
        )
        yield
        assert github.call_count == 0, [str(c.request.url) for c in github.calls]


def _facts(
    *,
    ok: bool = True,
    token: str | None = _APP_TOKEN,
    private: bool | None = True,
    default_branch: str | None = "main",
    branch_head: RefHead | None = RefHead("present", _BASE),
    default_branch_head: RefHead | None = None,
    reason: str | None = None,
) -> CloneCredential:
    return CloneCredential(
        ok=ok,
        token=token,
        visibility="private" if ok else None,
        source="app" if token else "none",
        reason=reason,
        private=private,
        default_branch=default_branch,
        branch_head=branch_head,
        default_branch_head=default_branch_head,
    )


def _exec_recorder():
    calls: list[tuple[str, dict | None]] = []

    async def _run(container_name, command, tenant_id=None, timeout=300, env=None):
        calls.append((command, env))
        if "rev-list --count" in command:
            return {"exit_code": 0, "stdout": "1\n"}
        if " push " in command and "GIT_CONFIG_NOSYSTEM" in command:
            return {"exit_code": 0, "stdout": "", "stderr": ""}
        if "config --get-regexp" in command:
            return {"exit_code": 1, "stdout": "", "stderr": ""}
        if " log --no-renames " in command:  # the pushed-path listing: tip, no paths
            return {"exit_code": 0, "stdout": f"tip {_TIP}\n", "stderr": ""}
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    return _run, calls


async def _push(
    app,
    client,
    branch: str,
    facts: CloneCredential,
    *,
    byo: str | None = None,
    clone_branch: object = None,
    by_branch: dict[str, CloneCredential] | None = None,
):
    from coding_client.auth.tenant import get_current_user_with_tenant
    from src.models.container_models import ContainerStatus

    user = MagicMock()
    user.is_admin.return_value = True
    user.effective_tenant_id = "tenant-7"
    app.dependency_overrides[get_current_user_with_tenant] = lambda: user
    container = MagicMock()
    container.container_name = "ctr"
    container.status = ContainerStatus("running")
    container.tenant_id = "tenant-7"
    container.repo_url = _URL
    container.github_access_token = byo
    container.secret_config_id = None
    container.branch = clone_branch
    run, calls = _exec_recorder()
    manager = MagicMock()
    manager.get_container.return_value = container
    manager.execute_command = AsyncMock(side_effect=run)
    tm = MagicMock()
    if by_branch is None:
        tm.get_app_clone_credential = AsyncMock(return_value=facts)
    else:

        async def _answer(url, *, tenant_id=None, scope="clone", branch=None):
            return by_branch.get(branch, facts)

        tm.get_app_clone_credential = AsyncMock(side_effect=_answer)
    try:
        with (
            patch("src.routers.containers.get_container_manager", return_value=manager),
            patch("src.routers.containers.git_push.get_github_token_manager", return_value=tm),
        ):
            resp = await client.post("/api/containers/ctr/git/push", json={"branch": branch})
    finally:
        app.dependency_overrides.clear()
    return resp, [c for c, _ in calls], calls, tm


def _pushed(calls: list[str]) -> bool:
    return any(" push " in c and "GIT_CONFIG_NOSYSTEM" in c for c in calls)


def _touched_origin(calls: list[str]) -> bool:
    return any("git remote" in c for c in calls)


def _message(resp) -> str:
    return resp.json()["error"]["message"]


# --- the facts come from coding-github, asked with the branch ----------------


async def test_feature_branch_push_succeeds_with_no_github_call(app, client):
    """The live failure: with no api.github.com egress the push must still land."""
    resp, calls, raw, tm = await _push(app, client, "aa/fix-123", _facts())
    assert resp.status_code == 200, resp.text
    assert _pushed(calls)
    assert any(f"{_TIP}:refs/heads/aa/fix-123" in c for c in calls)
    tm.get_app_clone_credential.assert_awaited_once_with(
        _URL, tenant_id="tenant-7", scope="push", branch="aa/fix-123"
    )
    # The pushed-path range starts at the branch head coding-github reported.
    assert any(f"--not {_BASE}" in c for c in calls), calls


async def test_new_branch_is_ranged_from_the_default_branch_head(app, client):
    facts = _facts(branch_head=RefHead("absent"), default_branch_head=RefHead("present", "d" * 40))
    resp, calls, *_ = await _push(app, client, "aa/new", facts)
    assert resp.status_code == 200, resp.text
    assert any(f"--not {'d' * 40}" in c for c in calls), calls


# --- default branch -----------------------------------------------------------


async def test_push_to_default_branch_is_refused_before_the_credential_is_used(app, client):
    resp, calls, raw, _ = await _push(app, client, "main", _facts())
    assert resp.status_code == 403, resp.text
    assert _message(resp).startswith("default_branch_push_refused: main ")
    assert not _pushed(calls)
    assert not _touched_origin(calls)
    assert all(env is None for _, env in raw)  # the token never reached the workspace


async def test_a_non_main_default_branch_is_the_one_refused(app, client):
    """The refusal keys on the RESOLVED default, not on a hardcoded ``main``."""
    refused, refused_calls, *_ = await _push(
        app, client, "develop", _facts(default_branch="develop")
    )
    assert refused.status_code == 403, refused.text
    assert not _pushed(refused_calls)

    allowed, allowed_calls, *_ = await _push(app, client, "main", _facts(default_branch="develop"))
    assert allowed.status_code == 200, allowed.text
    assert _pushed(allowed_calls)


async def test_case_variant_of_the_default_branch_is_refused(app, client):
    resp, calls, *_ = await _push(app, client, "Main", _facts())
    assert resp.status_code == 403, resp.text
    assert not _pushed(calls)


async def test_unresolvable_default_branch_refuses_the_push(app, client):
    """coding-github named no default branch (or is too old to send one)."""
    resp, calls, *_ = await _push(app, client, "aa/fix-123", _facts(default_branch=None))
    assert resp.status_code == 502, resp.text
    assert _message(resp).startswith("default_branch_unresolved:")
    assert not _pushed(calls)
    assert not _touched_origin(calls)


# --- visibility: a PUBLIC repository is never a push target -------------------


async def test_public_refusal_is_caused_by_the_private_field_alone(app, client):
    """Mutation control: the same request with ONLY ``private`` flipped pushes."""
    results = {}
    for private in (True, False):
        resp, calls, *_ = await _push(app, client, "aa/fix-123", _facts(private=private))
        results[private] = (resp.status_code, _pushed(calls))
        if not private:
            assert _message(resp).startswith("public_repo_refused: ")
            assert all("rev-list --count" in c for c in calls), calls
    assert results[True] == (200, True), results
    assert results[False] == (403, False), results


@pytest.mark.parametrize(
    "facts",
    [
        _facts(private=None),  # an older coding-github: visibility but no `private`
        _facts(ok=False, token=None),  # the clone-token call itself failed
    ],
    ids=["old-coding-github", "call-failed"],
)
async def test_unknown_visibility_fails_closed_with_503(app, client, facts):
    resp, calls, *_ = await _push(app, client, "aa/fix-123", facts)
    assert resp.status_code in (412, 503), resp.text
    assert not _pushed(calls)
    assert not _touched_origin(calls)


async def test_old_coding_github_answer_is_repo_visibility_unknown(app, client):
    """An answer with a token but none of the new fields (pre-change coding-github)."""
    old = CloneCredential(ok=True, token=_APP_TOKEN, visibility="private", source="app")
    resp, calls, *_ = await _push(app, client, "aa/fix-123", old)
    assert resp.status_code == 503, resp.text
    assert _message(resp).startswith("repo_visibility_unknown:")
    assert not _pushed(calls)


# --- push base ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("branch_head", "default_head"),
    [
        (None, None),
        (RefHead("unknown"), None),
        (RefHead("absent"), None),
        (RefHead("absent"), RefHead("unknown")),
        (RefHead("absent"), RefHead("absent")),
    ],
    ids=["no-head", "unknown", "absent-no-default", "absent-default-unknown", "both-absent"],
)
async def test_unknown_base_refuses(app, client, branch_head, default_head):
    facts = _facts(branch_head=branch_head, default_branch_head=default_head)
    resp, calls, *_ = await _push(app, client, "aa/fix-123", facts)
    assert resp.status_code == 502, resp.text
    assert _message(resp).startswith("push_base_unresolved:")
    assert "(lookup failed)" in _message(resp)
    assert not _pushed(calls)


@pytest.mark.parametrize(
    ("branch_head", "default_head"),
    [
        (RefHead("empty_repository"), None),
        (RefHead("absent"), RefHead("empty_repository")),
    ],
)
async def test_empty_repository_is_reported_as_empty(app, client, branch_head, default_head):
    facts = _facts(branch_head=branch_head, default_branch_head=default_head)
    resp, calls, *_ = await _push(app, client, "aa/fix-123", facts)
    assert resp.status_code == 502, resp.text
    assert "is an empty repository" in _message(resp)
    assert not _pushed(calls)


# --- BYO token: the facts still come from coding-github -----------------------


async def test_byo_push_takes_facts_from_coding_github_and_pushes_the_byo_token(app, client):
    resp, calls, raw, tm = await _push(app, client, "aa/fix-123", _facts(), byo=_BYO_TOKEN)
    assert resp.status_code == 200, resp.text
    tm.get_app_clone_credential.assert_awaited_once_with(
        _URL, tenant_id="tenant-7", scope="push", branch="aa/fix-123"
    )
    push_envs = [env for c, env in raw if " push " in c and "GIT_CONFIG_NOSYSTEM" in c]
    assert push_envs and all(_BYO_TOKEN in (env or {}).values() for env in push_envs)
    assert all(_APP_TOKEN not in str(env) for _, env in raw)  # facts-only token discarded


async def test_byo_push_without_coding_github_facts_is_refused(app, client):
    facts = _facts(token=None, private=None, default_branch=None, reason="not_installed")
    resp, calls, *_ = await _push(app, client, "aa/fix-123", facts, byo=_BYO_TOKEN)
    assert resp.status_code == 503, resp.text
    assert _message(resp).startswith("repo_visibility_unknown:")
    assert not _pushed(calls)


async def test_byo_push_to_default_branch_is_refused(app, client):
    resp, calls, *_ = await _push(app, client, "main", _facts(), byo=_BYO_TOKEN)
    assert resp.status_code == 403, resp.text
    assert not _pushed(calls)


async def test_unknown_branch_head_is_never_read_as_absent(app, client):
    """``unknown`` must refuse even when a usable default-branch head is present.

    Reading it as ``absent`` would range the pushed paths from the DEFAULT branch
    while the remote branch may already hold other commits — a wrong base, not a
    refusal. This is the one case where unknown and absent diverge.
    """
    facts = _facts(branch_head=RefHead("unknown"), default_branch_head=RefHead("present", "d" * 40))
    resp, calls, *_ = await _push(app, client, "aa/fix-123", facts)
    assert resp.status_code == 502, resp.text
    assert _message(resp).startswith("push_base_unresolved:")
    assert "(lookup failed)" in _message(resp)
    assert not _pushed(calls)
    assert not _touched_origin(calls)


async def test_overlong_branch_is_refused_before_coding_github_is_asked(app, client):
    """coding-github caps ``branch`` at 255; refuse here with a clear 400 instead."""
    resp, calls, _, tm = await _push(app, client, "a" * 256, _facts())
    assert resp.status_code == 400, resp.text
    tm.get_app_clone_credential.assert_not_called()
    assert not _pushed(calls)


# --- R288: a new branch on a workspace cloned from a non-default branch ----------
#
# Run 25 (2026-10-16): the workspace was cloned from ``conveyor-proof`` (6140950),
# the default ``main`` (7cb9e64) is not in its history, so ranging the new branch
# from the default head refused every push ``push_base_unknown``. The base is now
# the GitHub head of the branch the SERVICE recorded as the clone branch.

_CLONE_HEAD = "c" * 40
_DEFAULT_HEAD = "d" * 40


def _new_branch_facts() -> CloneCredential:
    return _facts(
        branch_head=RefHead("absent"), default_branch_head=RefHead("present", _DEFAULT_HEAD)
    )


async def test_new_branch_is_ranged_from_the_clone_branch_head(app, client):
    clone = _facts(branch_head=RefHead("present", _CLONE_HEAD))
    resp, calls, _, tm = await _push(
        app,
        client,
        "conveyor/123",
        _new_branch_facts(),
        clone_branch="conveyor-proof",
        by_branch={"conveyor-proof": clone},
    )
    assert resp.status_code == 200, resp.text
    assert any(f"--not {_CLONE_HEAD}" in c for c in calls), calls
    assert not any(f"--not {_DEFAULT_HEAD}" in c for c in calls), calls
    tm.get_app_clone_credential.assert_any_await(
        _URL, tenant_id="tenant-7", scope="push", branch="conveyor-proof"
    )
    assert tm.get_app_clone_credential.await_count == 2


@pytest.mark.parametrize(
    "clone_answer",
    [
        _facts(branch_head=RefHead("absent")),
        _facts(branch_head=RefHead("unknown")),
        _facts(branch_head=None),
        _facts(ok=False, token=None, branch_head=None, reason="lookup_failed"),
    ],
    ids=["clone-branch-absent", "clone-head-unknown", "no-head", "call-failed"],
)
async def test_unusable_clone_branch_head_keeps_the_default_base(app, client, clone_answer):
    """Exactly today's behaviour: the default head, which still fails closed when
    the workspace lacks it (the listing exits ``_BASE_UNKNOWN_EXIT``)."""
    resp, calls, *_ = await _push(
        app,
        client,
        "conveyor/123",
        _new_branch_facts(),
        clone_branch="conveyor-proof",
        by_branch={"conveyor-proof": clone_answer},
    )
    assert resp.status_code == 200, resp.text
    assert any(f"--not {_DEFAULT_HEAD}" in c for c in calls), calls


@pytest.mark.parametrize(
    "clone_branch",
    [None, "", "main", "Main", "conveyor/123", "bad;branch", "../x", object()],
    ids=["none", "empty", "default", "default-case", "push-branch", "unsafe", "dotdot", "non-str"],
)
async def test_unrecorded_or_irrelevant_clone_branch_is_not_asked(app, client, clone_branch):
    resp, calls, _, tm = await _push(
        app, client, "conveyor/123", _new_branch_facts(), clone_branch=clone_branch
    )
    assert resp.status_code == 200, resp.text
    assert any(f"--not {_DEFAULT_HEAD}" in c for c in calls), calls
    tm.get_app_clone_credential.assert_awaited_once()


async def test_unknown_clone_base_still_fails_closed(app, client):
    """A clone-branch head the workspace lacks refuses ``push_base_unknown``, unpushed."""
    clone = _facts(branch_head=RefHead("present", _CLONE_HEAD))
    run, raw = _exec_recorder()

    async def _lacks_base(container_name, command, tenant_id=None, timeout=300, env=None):
        if " cat-file -e " in command and " log --no-renames " in command:
            raw.append((command, env))
            return {"exit_code": 95, "stdout": "", "stderr": ""}
        return await run(container_name, command, tenant_id, timeout, env)

    with patch(f"{__name__}._exec_recorder", return_value=(_lacks_base, raw)):
        resp, calls, *_ = await _push(
            app,
            client,
            "conveyor/123",
            _new_branch_facts(),
            clone_branch="conveyor-proof",
            by_branch={"conveyor-proof": clone},
        )
    assert resp.status_code == 409, resp.text
    assert "push_base_unknown" in _message(resp)
    assert _CLONE_HEAD in _message(resp)
    assert not _pushed(calls)


async def test_existing_remote_branch_ignores_the_clone_branch(app, client):
    resp, calls, _, tm = await _push(
        app, client, "conveyor/123", _facts(), clone_branch="conveyor-proof"
    )
    assert resp.status_code == 200, resp.text
    assert any(f"--not {_BASE}" in c for c in calls), calls
    tm.get_app_clone_credential.assert_awaited_once_with(
        _URL, tenant_id="tenant-7", scope="push", branch="conveyor/123"
    )


async def test_forged_workspace_refs_do_not_choose_the_base(app, client):
    """The base is never read from the workspace: no command before the listing
    reads a remote-tracking ref or ``origin/HEAD`` to pick it, and the listing's
    ``--not`` is exactly the GitHub-reported clone head."""
    clone = _facts(branch_head=RefHead("present", _CLONE_HEAD))
    resp, calls, *_ = await _push(
        app,
        client,
        "conveyor/123",
        _new_branch_facts(),
        clone_branch="conveyor-proof",
        by_branch={"conveyor-proof": clone},
    )
    assert resp.status_code == 200, resp.text
    listing = next(c for c in calls if " log --no-renames " in c)
    assert f"--not {_CLONE_HEAD} --" in listing
    assert "--remotes" not in listing and "origin/HEAD" not in listing
    assert "refs/remotes" not in listing

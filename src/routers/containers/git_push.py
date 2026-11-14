"""Container git push-to-fork route."""

from __future__ import annotations

import logging
import os
import re
import shlex
from typing import Any, Literal, NamedTuple
from uuid import uuid7

import docker.errors
from fastapi import APIRouter, Body, Depends, HTTPException, Path

from coding_client.auth.models import User
from coding_client.auth.tenant import get_current_user_with_tenant
from coding_client.models.responses import (
    bad_request,
    internal_error,
    not_found,
    unauthorized,
)
from src.clients.github_client import (
    APP_NOT_INSTALLED_ERROR,
    GITHUB_HOST,
    PLATFORM_REPO_FORBIDDEN,
    PUSH_CREDENTIAL_ERROR,
    CloneCredential,
    RefHead,
    canonical_github_repo,
    canonical_github_url,
    get_github_token_manager,
    is_platform_owner,
    same_github_repo,
)
from src.core.git_clone_credentials import credential_helper_script
from src.core.kubernetes_container_provisioner import (
    PLATFORM_CREDENTIAL_UNAVAILABLE,
    PLATFORM_TENANT_MISSING,
    GitCredentialUnavailableError,
    platform_token_or_refuse,
    require_tenant_for_platform_repo,
)
from src.core.push_path_guard import (
    PATH_DENIED_PROCESS_DATA,
    PATH_INVALID,
    PATH_NOT_IN_TARGET_FILES,
    PathRefusal,
    refused_paths,
)
from src.models.container_api_models import (
    GitPushRequest,
    GitPushResponse,
)
from src.models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error422Response,
    Error500Response,
)

from . import _common
from ._common import (
    GIT_FSMONITOR_OFF,
    WORKSPACE_OBJECT_CORRUPT,
    WORKSPACE_OBJECT_CORRUPT_EXIT,
    _sanitize_error_message,
    _sanitize_shell_arg,
    _validate_git_branch_name,
    _validate_git_push_remote,
    get_git_manager,
    workspace_git_config_args,
    workspace_git_env,
    workspace_poison_check,
)

logger = logging.getLogger(__name__)
router = APIRouter()


#: The workspace checkout every command below runs in. A module constant so the
#: behavioural tests can run the REAL generated commands against a scratch repo.
_WORKDIR = "/workspace"

#: Environment variable that carries the push credential into the ONE `git push`
#: exec and nowhere else. See ``_push_command`` for the residual exposure.
_TOKEN_ENV = "GIT_PUSH_TOKEN"

#: Exit status the push exec uses when its own pre-push resolution check finds
#: the explicit URL rewritten. Chosen outside git's (0/1/128/129) range.
_REWRITE_EXIT = 97

#: Exit status the push exec uses when its own config guard (run in the SAME
#: exec, immediately before the push) finds a refused key.
_CONFIG_EXIT = 96

#: Exit status of the pushed-path listing when the base commit GitHub reported is
#: not in the workspace's object store (``push_base_unknown``).
_BASE_UNKNOWN_EXIT = 95

#: A full commit id (SHA-1 or SHA-256), as ``rev-parse`` and the GitHub API print it.
_COMMIT_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")

#: First line of the listing's output: the commit the push will send.
_TIP_PREFIX = "tip "

#: Workspace git config that can redirect a push, hand the credential to a
#: program, or MITM the TLS session that carries it. Any match in ANY scope the
#: container can write (system, global, local, worktree, and whatever their
#: ACTIVE includes pull in) refuses the push. Refusal, not removal: deleting a
#: tenant's config is a write to state we do not own, and "remove then push"
#: still leaves the window in which it is re-added — which the host/path-bound
#: helper and the in-exec resolution check below exist to close.
#:
#: * ``url.*.insteadof`` / ``pushinsteadof`` — rewrites the push URL.
#: * ``credential.*helper`` — git runs EVERY configured helper with ``store``
#:   after a successful push, i.e. hands it the token.
#: * ``http.*`` proxy / TLS trust / resolve — lets a same-host MITM read it.
#: * ``remote.<url>.*`` — a remote whose NAME is a URL is what ``git push <url>``
#:   looks up, so ``remote.https://github.com/o/r.pushurl`` would redirect it.
_REDIRECT_CONFIG_RE = (
    r"^url\..*\.(insteadof|pushinsteadof)$"
    r"|^credential\.(.*\.)?helper$"
    r"|^http\.(.*\.)?(proxy|sslverify|sslcainfo|sslcapath|sslcert|sslkey|sslbackend"
    r"|curloptresolve|proxysslcainfo|proxysslcert|proxysslkey)$"
    r"|^remote\..*[:/].*\.[a-z]+$"
)

#: ANY include directive, in any scope, refuses the push — whatever it points at.
#:
#: ⛔ Following includes is not enough, because a CONDITIONAL include is only
#: followed while its condition holds, and the route itself changes the
#: condition. Reproduced (auditor, #138): ``includeIf "hasconfig:remote.*.url:
#: https://github.com/**".path = evil.gitconfig`` is INACTIVE on a workspace whose
#: origin is not yet the canonical URL, so the guard listed nothing; the route
#: then re-pointed origin at ``https://github.com/<owner>/<repo>``, the include
#: became ACTIVE, and the push ran under config the guard never saw. ``gitdir:``
#: and ``onbranch:`` conditions can flip the same way. So the guard runs after
#: the re-point AND refuses the directive itself: a workspace has no legitimate
#: need for an include at push time, and "is it active right now" is a question
#: whose answer the next command can change. (``git config --get-regexp``
#: lowercases section names, hence ``includeif``.)
_INCLUDE_CONFIG_RE = r"^include\.|^includeif\."

#: Config that makes a git command FETCH, or names the program a fetch runs.
#: A partial clone (``extensions.partialClone`` + a ``promisor`` remote) lazily
#: fetches every missing object, and ``git push`` does: pack-objects faulted a
#: missing blob in through a planted ``remote.<n>.url=ext::<command>`` and ran the
#: command with ``$GIT_PUSH_TOKEN`` in its environment (reproduced, audit on this
#: branch). ``protocol.*.allow`` is what re-enables ``ext::``; a local-path
#: remote's ``uploadpack``/``receivepack`` is a command on its own.
#: ``GIT_NO_LAZY_FETCH=1`` in every exec is the control; this refusal is the second
#: layer, and it names the key instead of failing the push with "unpacker error".
_FETCH_CONFIG_RE = (
    r"^extensions\.partialclone$"
    r"|^remote\..*\.(promisor|partialclonefilter|uploadpack|receivepack)$"
    r"|^protocol\.(.*\.)?allow$"
    r"|^uploadpack\.packobjectshook$"
)

#: Config that makes a git command RECURSE into a submodule, i.e. run git in a
#: repository whose own config (``.git/modules/<n>/config``) this guard never reads.
#: ``push.recurseSubmodules=on-demand`` made the push run ``git push`` inside a
#: submodule whose ``ext::`` origin executed a command with ``$GIT_PUSH_TOKEN`` in
#: its environment (reproduced, audit on #157). ``submodule.<n>.update`` names a
#: command (``!cmd``). The CONTROL is not this list: it is the push's
#: ``--no-recurse-submodules`` + ``WORKSPACE_GIT_TRANSPORT_CONFIG`` +
#: ``GIT_ALLOW_PROTOCOL=https``, which hold whatever the submodule's config says.
#: Deliberately NOT refused: ``submodule.<n>.url``/``active`` -- ``git submodule
#: init`` writes them for any repository with submodules, and under the pins nothing
#: the push runs reads them (a helper-scheme ``url`` value is still refused, below).
_SUBMODULE_CONFIG_RE = (
    r"^(push|fetch)\.recursesubmodules$"
    r"|^submodule\.recurse$"
    r"|^submodule\..*\.(update|fetchrecursesubmodules)$"
)

#: ``trace2.*`` in any scope: ``trace2.envVars`` names environment variables whose
#: VALUES git writes to the ``trace2.*Target`` file, so ``trace2.envVars=GIT_PUSH_TOKEN``
#: in the workspace-owned global config wrote the token to a readable file from every
#: git the push exec ran before its config isolation (reproduced, audit on #157). The
#: CONTROL is ``GIT_TRACE2*=0`` (``_PUSH_PRELUDE_KEYS``); this refusal is the second
#: layer. (git reads trace2 settings from system and global config only.)
_TRACE_CONFIG_RE = r"^trace2\."

_DANGEROUS_CONFIG_RE = (
    f"{_REDIRECT_CONFIG_RE}|{_INCLUDE_CONFIG_RE}|{_FETCH_CONFIG_RE}|{_SUBMODULE_CONFIG_RE}"
    f"|{_TRACE_CONFIG_RE}"
)

#: A remote URL of the form ``<helper>::<address>`` makes git run
#: ``git-remote-<helper>``; ``ext::<command>`` runs the command itself. Matched on
#: the VALUE of every remote URL key, whatever the remote is called. (Same
#: character class git uses to find the helper name: ``transport.c``,
#: ``is_urlschemechar``.)
_REMOTE_URL_KEY_RE = r"^(remote|submodule)\..*\.(url|pushurl)$"
_HELPER_URL_RE = r"^[A-Za-z0-9+.-]*::"


def _credential_helper_script(owner: str, repo: str) -> str:
    """The ``!``-helper git runs; prints the token ONLY for ``https://github.com/<owner>/<repo>``.

    git feeds a helper ``key=value`` lines on stdin. With
    ``credential.useHttpPath=true`` those include ``path`` — without it the
    helper sees no path and answers nothing, so dropping that flag fails CLOSED.
    Only the ``get`` action is answered; ``store``/``erase`` (which git sends
    after a push, WITH the password) are drained and ignored. The token is read
    from ``$GIT_PUSH_TOKEN`` at call time, so it never appears in this text, in
    any config file, or in git's argv.

    ``owner``/``repo`` come from ``canonical_github_repo`` (``[A-Za-z0-9._-]``),
    so they are safe as literals here; the whole script is still shell-quoted
    as one ``-c`` argument by the caller.
    """
    return credential_helper_script(GITHUB_HOST, f"{owner}/{repo}", _TOKEN_ENV)


def _push_git_config_args(owner: str, repo: str) -> list[str]:
    """``-c`` overrides for the push; command-line config is read LAST, so it wins.

    * ``credential.helper=`` first — an empty value clears every helper git has
      collected so far (system, global, local, url-scoped), then ours is added,
      scoped to ``https://github.com`` and path-bound via ``useHttpPath``.
    * ``core.hooksPath=/dev/null`` — a workspace ``pre-push`` hook would run
      with ``$GIT_PUSH_TOKEN`` in its environment.
    * ``core.askPass=`` (plus ``GIT_ASKPASS=`` in the env) — git falls back to
      an askpass program when no helper answers, and that program inherits the
      environment too.
    * The ``http.<exact url>.*`` pins tie the most specific key git can match
      for this URL and come last, so a workspace proxy / resolve list /
      ``sslVerify=false`` added after the config guard ran cannot win.
    """
    url = f"https://{GITHUB_HOST}/{owner}/{repo}"
    pairs = [
        "credential.helper=",
        f"credential.https://{GITHUB_HOST}.helper={_credential_helper_script(owner, repo)}",
        "credential.useHttpPath=true",
        "core.hooksPath=/dev/null",
        GIT_FSMONITOR_OFF,
        "core.askPass=",
        f"http.{url}.proxy=",
        f"http.{url}.sslVerify=true",
        f"http.{url}.curloptResolve=",
        f"safe.directory={_WORKDIR}",
        # After the shared ``protocol.allow=never``: the push's own scheme, and only it.
        "protocol.https.allow=always",
    ]
    args: list[str] = []
    for pair in pairs:
        args += ["-c", pair]
    return args


def _config_guard_command() -> str:
    """List refused config across every scope (no scope flag ⇒ all, active includes followed).

    Must run AFTER origin is re-pointed (see ``_INCLUDE_CONFIG_RE``); the push
    exec repeats it as its first step.
    """
    # The shared ``-c`` list too (hooks off, fsmonitor off, ...): none of its keys
    # matches the refused patterns, and every git this route runs carries it. NOT the
    # transport pins: command-line config is listed too, and those keys ARE refused.
    git = f"{_hardened_git(transport=False)} config --get-regexp"
    # Exit 1 = neither listing matched (git's "no match"); 0 = a match was printed;
    # anything else = a listing failed. Same contract as one ``--get-regexp``.
    return (
        f"cd {shlex.quote(_WORKDIR)} && ( "
        f"{git} {shlex.quote(_DANGEROUS_CONFIG_RE)}; k=$?; "
        f"{git} {shlex.quote(_REMOTE_URL_KEY_RE)} {shlex.quote(_HELPER_URL_RE)}; u=$?; "
        'if [ "$k" -gt 1 ] || [ "$u" -gt 1 ]; then exit 2; fi; '
        'if [ "$k" -eq 1 ] && [ "$u" -eq 1 ]; then exit 1; fi; exit 0 )'
    )


def _hardened_git(*, transport: bool = True) -> str:
    """``git`` plus the overrides every non-push git on this route runs with: the
    push's own ``safe.directory``, then the shared ``WORKSPACE_GIT_CONFIG``
    (``_common.py``: ``core.hooksPath=/dev/null`` since ``update-ref`` and ``remote
    remove`` run a workspace ``reference-transaction`` hook, ``core.fsmonitor=false``,
    ``core.commitGraph=false``, ...). Use inside ``_in_workdir``, which exports the env."""
    return " ".join(
        shlex.quote(a)
        for a in [
            "git",
            "-c",
            f"safe.directory={_WORKDIR}",
            *workspace_git_config_args(transport=transport),
        ]
    )


def _in_workdir(script: str) -> str:
    """``cd`` to the workspace and run ``script`` in a subshell with the shared
    ``WORKSPACE_GIT_ENV`` exported (no lazy fetch: a partial clone's promisor remote is
    workspace config, and a lazy fetch runs its transport; no terminal prompt; no
    replace refs, grafts, system or global config), after the shared poison check: a
    workspace holding replace refs, grafts or alternates exits
    ``WORKSPACE_POISONED_EXIT`` before ``script`` runs (the route's first exec, so a
    poisoned workspace is refused before anything is listed or pushed)."""
    return (
        f"cd {shlex.quote(_WORKDIR)} && ( {workspace_git_env()}{workspace_poison_check()}{script} )"
    )


#: ``WORKSPACE_GIT_ENV`` entries exported before the FIRST git of the push exec (it
#: holds ``$GIT_PUSH_TOKEN``): no lazy fetch, no transport, no trace2 output (a
#: workspace ``trace2.envVars=GIT_PUSH_TOKEN`` in global config, which the guard below
#: reads by design, would otherwise write the token to a file; see TRACE ISOLATION in
#: ``_common.py``). Not the config-scope exports: the guard must still see every scope.
_PUSH_PRELUDE_KEYS = (
    "GIT_NO_LAZY_FETCH=",
    "GIT_ALLOW_PROTOCOL=",
    "GIT_TRACE2=",
    "GIT_TRACE2_EVENT=",
    "GIT_TRACE2_PERF=",
)


def _push_exec_prelude() -> str:
    """``export GIT_NO_LAZY_FETCH=1 GIT_ALLOW_PROTOCOL= GIT_TRACE2=0 ...; `` taken from the shared
    ``WORKSPACE_GIT_ENV`` at call time (one list: a test dropping an entry there drops
    it here too)."""
    env = [e for e in _common.WORKSPACE_GIT_ENV if e.startswith(_PUSH_PRELUDE_KEYS)]
    return "export " + " ".join(env) + "; " if env else ""


#: The push never recurses into a submodule (see ``_SUBMODULE_CONFIG_RE``); the flag
#: beats any ``push.recurseSubmodules`` config, and the shared ``-c`` pins it too.
_PUSH_NO_RECURSE_FLAG: str = "--no-recurse-submodules"
#: ``GIT_ALLOW_PROTOCOL`` for the push exec: overrides every ``protocol.*.allow``, in
#: every repository the push could touch, so no ``ext::``/``file``/``<helper>::``
#: transport runs with the token in its environment.
_PUSH_ALLOWED_PROTOCOLS: str = "https"


def _push_command(owner: str, repo: str, branch: str, tip: str, *, force: bool) -> str:
    """One exec: refuse a poisoned workspace, re-run the config guard, isolate config,
    re-verify URL resolution, push to the EXPLICIT URL.

    The refspec is ``<tip>:refs/heads/<branch>`` — the commit id the path listing
    resolved and checked, never the branch NAME, so a workspace that moves the
    branch after the check cannot push anything the check did not list.

    The guard runs FIRST, before the scope-narrowing exports, so it still sees
    every scope, and after the route's origin re-point, so a conditional include
    keyed on the origin URL is judged in the state the push will run in. A match
    exits ``_CONFIG_EXIT`` with the matched keys on stdout.

    ``GIT_CONFIG_NOSYSTEM=1`` + ``GIT_CONFIG_GLOBAL=/dev/null`` (an always-empty
    file the workspace cannot write) mean only the repository's own config and
    our ``-c`` overrides apply. The resolution check runs in the SAME exec as the
    push, so the window between it and the push is one fork, not a round trip.
    It is a tripwire, not the control: the control is the helper, which answers
    only ``https://github.com/<owner>/<repo>`` whatever git ends up contacting.

    ``GIT_NO_LAZY_FETCH=1``: pack-objects in a partial clone lazily fetches missing
    objects through the workspace's promisor remote, i.e. runs whatever transport that
    remote names, with the token in its environment. ``_FETCH_CONFIG_RE`` refuses the
    config; this env keeps a missing object an error even if the guard is blind.

    Residual, stated rather than hidden: ``$GIT_PUSH_TOKEN`` lives in the
    environment of this exec's processes for the duration of the push, and a
    workspace process running as the SAME uid can read ``/proc/<pid>/environ``
    while it runs. Nothing on this exec surface can prevent that (no stdin, and
    the workspace runs as the exec uid); it is bounded to the push, whereas the
    token file it replaces sat in ``/tmp`` for the whole push AND survived a
    failed cleanup. On Kubernetes the token also transits ``/usr/bin/env``'s argv
    for the instant before it ``exec``s ``sh`` (see ``with_env``).
    """
    url = f"https://{GITHUB_HOST}/{owner}/{repo}"
    q_url = shlex.quote(url)
    if not _COMMIT_SHA_RE.match(tip):
        raise ValueError("push tip must be a full commit id")
    refspec = shlex.quote(f"{tip}:refs/heads/{branch}")
    safe_dir = " ".join(
        shlex.quote(a) for a in ["-c", f"safe.directory={_WORKDIR}", *workspace_git_config_args()]
    )
    # The shared list first, the push's own pins last (command-line config: last wins).
    git_c = " ".join(
        shlex.quote(a) for a in [*workspace_git_config_args(), *_push_git_config_args(owner, repo)]
    )
    force_flag = " --force" if force else ""
    # Read at call time (tests drop them to show each layer is load-bearing).
    no_recurse = f" {_PUSH_NO_RECURSE_FLAG}" if _PUSH_NO_RECURSE_FLAG else ""
    allow = f" GIT_ALLOW_PROTOCOL={_PUSH_ALLOWED_PROTOCOLS}" if _PUSH_ALLOWED_PROTOCOLS else ""
    return (
        f"cd {shlex.quote(_WORKDIR)} || exit 1; "
        # FIRST, before any git: the poison/submodule check below lists
        # ``HEAD^{tree}``, and in a partial clone a missing object is LAZILY FETCHED
        # through the workspace's promisor remote -- running its transport
        # (``core.sshCommand``, ``core.gitProxy``, ``ext::``) with ``$GIT_PUSH_TOKEN``
        # in its environment (reproduced, validator on #157). No git in this exec may
        # run without these two; the transport is re-opened (``https`` only) for the
        # push itself further down.
        f"{_push_exec_prelude()}"
        # Its OWN poison refusal (the listing's ran several execs earlier): alternates
        # planted in between would let pack-objects read objects from a directory the
        # workspace names. The remote re-hashes what it receives, so this is
        # consistency, not the only fence.
        f"{workspace_poison_check()}"
        f"guard_keys=$({_config_guard_command()}); guard_rc=$?; "
        'if [ "$guard_rc" -ne 1 ] || [ -n "$guard_keys" ]; then '
        f'printf "%s\\n" "$guard_keys"; exit {_CONFIG_EXIT}; fi; '
        f"{workspace_git_env()}export GIT_ASKPASS= SSH_ASKPASS={allow} && "
        f"fetch_url=$(git {safe_dir} ls-remote --get-url {q_url}) && "
        f'test "$fetch_url" = {q_url} && '
        f"push_url=$(git {safe_dir} remote get-url --push origin) && "
        f'test "$push_url" = {q_url} || '
        f"{{ echo 'push_url_rewritten: the push URL does not resolve to itself' >&2; "
        f"exit {_REWRITE_EXIT}; }}; "
        f"git {git_c} push{no_recurse}{force_flag} {q_url} {refspec}"
    )


class _RepoInfo(NamedTuple):
    """What the push route reads from ``GET /repos/{o}/{r}``: one call, two answers.

    ``default_branch`` is ``None`` when GitHub answered but named no usable
    default branch; ``private`` is ``None`` when it answered without a boolean
    ``private`` field. Either ``None`` is an unknown, and the route refuses.
    """

    default_branch: str | None
    private: bool | None


def _credential_error_message(exc: GitCredentialUnavailableError) -> str:
    """The ``message`` a ``GitCredentialUnavailableError`` carries.

    ``BaseAPIException`` always builds ``detail`` as a dict holding ``message``;
    ``HTTPException.detail`` is typed ``str`` upstream, so narrow it here rather
    than subscripting blind.
    """
    detail = exc.detail
    if isinstance(detail, dict):
        return str(detail.get("message", ""))
    return str(detail)


async def _remote_repo_info(
    owner: str, repo: str, facts: CloneCredential | None
) -> _RepoInfo | None:
    """The repository's default branch and visibility, as coding-github read them.

    ``facts`` is coding-github's push-scope clone-token answer. This pod has NO
    egress to api.github.com (only coding-github does), so it never asks GitHub
    itself: coding-github reads ``GET /repos/{o}/{r}`` with the push token it
    minted and returns ``private`` + ``default_branch``. Nothing the workspace
    container runs (a replaced ``git``, a rewritten ``refs/remotes/origin/HEAD``, a
    faked ``ls-remote``) can change the answer.

    ``None`` when there is no usable answer (no call, a failed call). A field the
    answer lacks — e.g. an older coding-github that does not send it — is ``None``
    inside ``_RepoInfo``. Every unknown makes the caller refuse (fail closed).
    """
    if facts is None or not facts.ok:
        logger.error(
            "repository facts for %s/%s unavailable from coding-github (%s)",
            owner,
            repo,
            "no answer" if facts is None else (facts.reason or facts.error or "not ok"),
        )
        return None
    return _RepoInfo(default_branch=facts.default_branch, private=facts.private)


def _is_default_branch(branch: str, default_branch: str) -> bool:
    """True when pushing ``refs/heads/<branch>`` would write the default branch.

    Compared case-insensitively on purpose: refusing ``Main`` when the default is
    ``main`` costs nothing, and ref case handling differs between filesystems.
    The refspec is built by this route from a name ``_validate_git_branch_name``
    restricts to ``[A-Za-z0-9/_-]`` — no ``:``, ``+``, ``*`` or ``refs/`` prefix
    can reach it, and the command never carries ``--all``/``--mirror`` — so the
    destination is exactly ``refs/heads/<branch>`` and this comparison is the
    whole question.
    """
    return branch.casefold() == default_branch.casefold()


class _EmptyRepositoryError(Exception):
    """GitHub reports the repository has no commits at all (no branches).

    Raised by ``_push_base_sha`` so the route can say "empty repository" instead
    of "could not be read": both refuse (there is no base to range the pushed
    paths from), but only one of them is fixed by retrying.
    """


def _recorded_clone_branch(container: Any) -> str | None:
    """The branch THIS SERVICE cloned the workspace from, as it recorded it.

    ``ContainerInfo.branch`` is written at provision time from the provision
    request (registry row, and the ``coding-job/branch`` Deployment annotation the
    registry-miss path reads back). Nothing the workspace container runs can write
    either. ``None`` when absent, not a string, or not a safe branch name.
    """
    branch = getattr(container, "branch", None)
    if not isinstance(branch, str) or not _validate_git_branch_name(branch):
        return None
    return branch


async def _clone_branch_head(
    container: Any,
    *,
    clean_url: str,
    tenant_id: str | None,
    push_branch: str,
    default_branch: str,
    facts: CloneCredential | None,
) -> RefHead | None:
    """GitHub's head of the clone branch, asked only when it can change the base.

    Asked only when the push branch is ``absent`` on the remote and the workspace
    was cloned from a recorded branch other than the default (and other than the
    push branch): a workspace cloned from a project branch (e.g. ``conveyor-proof``)
    does not hold the default-branch head when the default has diverged, so that
    base is refused as unknown (run 25, 2026-10-16). The head comes from a second
    push-scope coding-github answer (``branch=<clone branch>`` → ``branch_head``,
    read from GitHub on the PUSH target); its token is discarded. Anything but a
    ``present`` head is ``None`` and the caller keeps today's default-branch base.
    """
    head = facts.branch_head if facts is not None and facts.ok else None
    if head is None or head.state != "absent":
        return None
    clone_branch = _recorded_clone_branch(container)
    if clone_branch is None or clone_branch.casefold() in (
        default_branch.casefold(),
        push_branch.casefold(),
    ):
        return None
    answer = await _app_credential(clean_url, tenant_id, "push", clone_branch)
    clone_head = answer.branch_head if answer.ok and answer.token else None
    if (
        clone_head is None
        or clone_head.state != "present"
        or not clone_head.sha
        or not _COMMIT_SHA_RE.match(clone_head.sha)
    ):
        logger.warning(
            "clone-branch head of %s (%s) unavailable (%s); ranging from the default branch",
            clean_url,
            clone_branch,
            "no answer" if clone_head is None else clone_head.state,
        )
        return None
    return clone_head


async def _push_base_sha(
    owner: str, repo: str, branch: str, default_branch: str, facts: CloneCredential | None
) -> str | None:
    """The commit the remote already has, which the pushed-path range starts from.

    The remote ``<branch>`` head if the branch exists on GitHub, else the
    ``<default_branch>`` head — both as coding-github read them
    (``GET /repos/{o}/{r}/git/ref/heads/{b}``, exact match) and returned as
    ``branch_head`` / ``default_branch_head`` on the push-scope answer. Never from
    the workspace, whose remote-tracking refs it can forge. (A new branch on a
    workspace cloned from a non-default branch is ranged from that branch's head
    instead, by the route, via ``_clone_branch_head`` before this is asked.)

    ``None`` (caller refuses, fail closed) if a head is missing or ``unknown``.
    Raises ``_EmptyRepositoryError`` when GitHub says the repository has no commits.
    """
    head = facts.branch_head if facts is not None and facts.ok else None
    if head is None or head.state == "unknown":
        return None
    if head.state == "empty_repository":
        raise _EmptyRepositoryError(f"{owner}/{repo}")
    if head.state == "present":
        return head.sha
    # absent: a new branch, ranged from the default branch head.
    base = facts.default_branch_head if facts is not None else None
    if base is None or base.state in ("unknown", "absent"):
        logger.error(
            "default-branch head of %s/%s (%s) unavailable: %s",
            owner,
            repo,
            default_branch,
            "not supplied" if base is None else base.state,
        )
        return None
    if base.state == "empty_repository":
        raise _EmptyRepositoryError(f"{owner}/{repo}")
    return base.sha


#: A gitlink (submodule pointer) the push sends must be LISTED, or a ``target_files``
#: allowlist is blind to it: ``diff.ignoreSubmodules=all`` in the workspace config, or
#: ``submodule.<n>.ignore=all`` in ``.gitmodules`` (which the config pin does NOT
#: override), drop it from ``log --name-only`` (measured). The flag beats both; the
#: ``-c`` pin is the second layer for the config route. A listed gitlink then fails a
#: ``target_blobs`` check (its type is ``commit``, never ``blob``). Read at call time.
_LOG_IGNORE_SUBMODULES_FLAG: str = "--ignore-submodules=none"
_LOG_IGNORE_SUBMODULES_CONFIG: str = "diff.ignoreSubmodules=none"


def _log_submodule_args() -> str:
    c = _LOG_IGNORE_SUBMODULES_CONFIG
    return f" -c {shlex.quote(c)}" if c else ""


def _log_submodule_flag() -> str:
    return f" {_LOG_IGNORE_SUBMODULES_FLAG}" if _LOG_IGNORE_SUBMODULES_FLAG else ""


def _pushed_range_submodule_check(git: str, base_sha: str) -> str:
    """Refuse (``WORKSPACE_SUBMODULES_EXIT``, via ``_nosub`` from the prefix) a pushed
    range in which ANY commit adds, changes or removes a gitlink or the root
    ``.gitmodules`` (``--raw`` shows both modes; ``-m`` each merge parent), even one
    the tip no longer holds: every commit in the range travels in the pack."""
    if not _common.SUBMODULE_CHECK_ENABLED:
        return ""
    return (
        f"_nsr=$({git} log --raw --no-abbrev --no-renames --format= -m "
        f'--ignore-submodules=none "$tip" --not {base_sha} --) || exit 1; '
        f"printf '%s\\n' \"$_nsr\" | {_common.NOSUB_AWK} "
        '&& _nosub "the pushed commits add, change or remove a gitlink or .gitmodules"; '
    )


def _pushed_paths_command(branch: str, base_sha: str) -> str:
    """Resolve the tip once, then list every path touched by a commit the push sends.

    Output: ``tip <sha>`` on the first line (the commit the refspec will push —
    ``_push_command`` takes this id, not the branch name), then one path per line.
    Exit ``_BASE_UNKNOWN_EXIT`` when ``base_sha`` is not in the local object store.

    Range: ``<tip> --not <base_sha>``, where ``base_sha`` came from GitHub via coding-github
    (``_push_base_sha``), never from workspace refs — ``--remotes`` read refs the
    route itself deletes (``remote remove origin``) and a workspace can forge
    (``update-ref refs/remotes/x/y <tip>``). A base the store lacks refuses rather
    than listing the whole history. EVERY commit is listed, not the net diff: a
    file added in one commit and deleted in the next still travels in the pack.
    ``-m`` lists merges against each parent (over-inclusive, fails toward refusal).

    Hardening against a workspace that hides paths from ``log`` (the env and the
    ``-c`` list are the shared ``WORKSPACE_GIT_ENV``/``WORKSPACE_GIT_CONFIG``, and
    ``_in_workdir`` refuses a workspace holding replace refs / grafts / alternates):
    ``GIT_NO_REPLACE_OBJECTS=1`` (``refs/replace/*``), ``GIT_GRAFT_FILE=/dev/null``
    (``info/grafts``), ``core.commitGraph=false`` (a forged
    ``objects/info/commit-graph`` rewrites parent links exactly like a graft, and
    git does not checksum it on read), ``log.showRoot=true`` (a root commit's files),
    ``--no-renames``, ``diff.relative=false``, ``core.quotePath=false`` (non-ASCII
    names verbatim; anything git still quotes is refused as unreadable).
    Residual: ``.git/shallow`` also cuts history, but pack-objects honours the same
    boundary, so a hidden commit is not sent either.
    """
    if not _COMMIT_SHA_RE.match(base_sha):
        raise ValueError("push base must be a full commit id")
    ref = shlex.quote(f"refs/heads/{branch}^{{commit}}")
    git = f"{_hardened_git()} -c core.quotePath=false -c log.showRoot=true{_log_submodule_args()}"
    return _in_workdir(
        f"tip=$({git} rev-parse --verify --quiet {ref}) || exit 1; "
        f"{git} cat-file -e {base_sha}^{{commit}} 2>/dev/null || exit {_BASE_UNKNOWN_EXIT}; "
        f"{_pushed_range_submodule_check(git, base_sha)}"
        f'printf "{_TIP_PREFIX}%s\\n" "$tip" && '
        f"{git} log --no-renames --no-ext-diff --no-textconv --name-only{_log_submodule_flag()} "
        f'--format= -m "$tip" --not {base_sha} --'
    )


def _parse_listing(stdout: str) -> tuple[str, list[str]] | None:
    """``(tip, paths)`` from the listing's output, or ``None`` if it is malformed."""
    lines = stdout.splitlines()
    if not lines or not lines[0].startswith(_TIP_PREFIX):
        return None
    tip = lines[0][len(_TIP_PREFIX) :]
    if not _COMMIT_SHA_RE.match(tip):
        return None
    return tip, [p for p in lines[1:] if p.strip()]


#: Error code: ``target_blobs`` was supplied and a pushed path has no entry in it.
PUSH_CONTENT_UNVERIFIED = "push_content_unverified"
#: Error code: a pushed path's blob at the tip differs from its ``target_blobs`` entry.
PUSH_CONTENT_MISMATCH = "push_content_mismatch"

#: Budget for one tip-blob listing exec (one ``sh -c`` argument; Linux caps one argv
#: string at 131072 bytes and a backend may re-quote it).
_TIP_BLOBS_ARG_BUDGET = 64 * 1024


def _tip_blobs_command(tip: str, paths: list[str]) -> str:
    """``ls-tree -z --full-tree <tip> -- <paths>`` under the shared hardening (env,
    ``-c`` list, poison refusal): ``<mode> <type> <id>\\t<path>`` NUL-terminated, one
    entry per path that exists at ``tip``."""
    if not _COMMIT_SHA_RE.match(tip):
        raise ValueError("push tip must be a full commit id")
    quoted = " ".join(shlex.quote(p) for p in paths)
    return _in_workdir(
        f"{_hardened_git()} -c core.quotePath=false --literal-pathspecs "
        f"ls-tree -z --full-tree {tip} -- {quoted}"
    )


def _tip_blob_chunks(paths: list[str]) -> list[list[str]]:
    chunks: list[list[str]] = []
    chunk: list[str] = []
    size = 0
    for p in paths:
        weight = 4 * len(p.encode("utf-8", "surrogateescape")) + 8
        if chunk and size + weight > _TIP_BLOBS_ARG_BUDGET:
            chunks.append(chunk)
            chunk, size = [], 0
        chunk.append(p)
        size += weight
    if chunk:
        chunks.append(chunk)
    return chunks


async def _check_pushed_content(
    manager: Any,
    container_name: str,
    tenant_id: str | None,
    tip: str,
    pushed_paths: list[str],
    target_blobs: dict[str, str | None],
) -> None:
    """Refuse (403) unless every pushed path's blob at ``tip`` equals ``target_blobs``.

    A pushed path with no ``target_blobs`` entry is ``push_content_unverified``; one
    whose tip blob differs (or that is absent / not a blob when a blob was expected,
    or present when ``null`` was) is ``push_content_mismatch``. The tip entries are
    read with the shared hardening, so a replace ref cannot swap the blob the check
    sees for the one the push sends. A listing that fails refuses too (409).
    """
    paths = sorted(set(pushed_paths))
    unverified = [p for p in paths if p not in target_blobs]
    if unverified:
        shown = unverified[:20]
        logger.warning(
            "%s for container %s: %d pushed path(s) without a reviewed blob, e.g. %s",
            PUSH_CONTENT_UNVERIFIED,
            container_name,
            len(unverified),
            ", ".join(shown),
        )
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": PUSH_CONTENT_UNVERIFIED,
                "message": (
                    f"{PUSH_CONTENT_UNVERIFIED}: the pushed commits touch {len(unverified)} "
                    f"path(s) with no reviewed content in target_blobs ({', '.join(shown)}). "
                    "Nothing was pushed."
                ),
                "paths": unverified,
            },
        )
    actual: dict[str, tuple[str, str]] = {}
    for part in _tip_blob_chunks(paths):
        result = await manager.execute_command(
            container_name, _tip_blobs_command(tip, part), tenant_id=tenant_id
        )
        if result.get("exit_code", 1) != 0 or result.get("truncated"):
            logger.error(
                "tip blob listing failed for container %s (exit %s); push refused",
                container_name,
                result.get("exit_code"),
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": PUSH_CONTENT_MISMATCH,
                    "message": (
                        f"{PUSH_CONTENT_MISMATCH}: the content of the pushed files at "
                        f"{tip} could not be read, so it cannot be matched to the "
                        "reviewed content and the push was refused."
                    ),
                },
            )
        for record in (result.get("stdout") or "").split("\0"):
            meta, tab, path = record.partition("\t")
            fields = meta.split(" ")
            if tab and len(fields) == 3:
                actual[path] = (fields[1], fields[2])
    mismatched: list[dict[str, str | None]] = []
    for p in paths:
        want = target_blobs[p]
        kind, got = actual.get(p, ("", ""))
        got_blob = got if kind == "blob" else None
        if kind not in ("", "blob") or got_blob != want:
            mismatched.append(
                {"path": p, "expected": want, "actual": got or None, "type": kind or None}
            )
    if mismatched:
        logger.error(
            "%s for container %s at %s: %s",
            PUSH_CONTENT_MISMATCH,
            container_name,
            tip,
            mismatched[:20],
        )
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": PUSH_CONTENT_MISMATCH,
                "message": (
                    f"{PUSH_CONTENT_MISMATCH}: {len(mismatched)} pushed file(s) differ at "
                    f"{tip} from the content that was reviewed "
                    f"({', '.join(str(m['path']) for m in mismatched[:20])}). Nothing was pushed."
                ),
                "paths": mismatched,
            },
        )


#: Error code: ``approved_tree`` was supplied and the pushed range is not exactly one
#: commit whose only parent is the base the remote has.
PUSH_NOT_SINGLE_COMMIT = "push_not_single_commit"
#: Error code: the pushed commit's tree is not the ``approved_tree``.
PUSH_TREE_MISMATCH = "push_tree_mismatch"


def _tip_commit_command(tip: str) -> str:
    """``cat-file commit <tip>`` under the shared hardening, with its OWN poison
    refusal: the raw commit object (``tree``/``parent`` header lines), which neither a
    graft nor ``shallow`` can rewrite; replace refs are off (env) and refused.

    The bytes are read ONCE into a temp file, hashed, and only printed when they hash to
    ``tip`` (else exit ``WORKSPACE_OBJECT_CORRUPT_EXIT``): git does not verify a loose
    object on read, and the header parsed is exactly the header verified."""
    if not _COMMIT_SHA_RE.match(tip):
        raise ValueError("push tip must be a full commit id")
    g = _hardened_git()
    verify = (
        f'_h=$({g} hash-object --literally --no-filters -t commit --stdin < "$_f"); '
        f'if [ "$_h" != {tip} ]; then rm -f "$_f"; '
        f'echo "{WORKSPACE_OBJECT_CORRUPT}: commit {tip} does not hash to its id; the '
        'workspace object store was altered. Refusing to read it." >&2; '
        f"exit {WORKSPACE_OBJECT_CORRUPT_EXIT}; fi; "
        if _common.OBJECT_CHECK_ENABLED
        else ""
    )
    return _in_workdir(
        f"_gd=$({g} rev-parse --git-dir) || exit 1; "
        '_f=$(mktemp "$_gd/verify-tip.XXXXXX") || exit 1; '
        f'{g} cat-file commit {tip} > "$_f" || {{ rm -f "$_f"; exit 1; }}; '
        f'{verify}cat "$_f"; rm -f "$_f"'
    )


def _commit_header(raw: str) -> tuple[str | None, list[str]] | None:
    """``(tree, parents)`` from a raw commit object's header, or ``None`` if malformed."""
    tree: str | None = None
    parents: list[str] = []
    for line in raw.split("\n"):
        if not line:
            break
        key, _, value = line.partition(" ")
        if key == "tree":
            if tree is not None or not _COMMIT_SHA_RE.match(value):
                return None
            tree = value
        elif key == "parent":
            if not _COMMIT_SHA_RE.match(value):
                return None
            parents.append(value)
    return (tree, parents) if tree is not None else None


async def _check_approved_range(
    manager: Any,
    container_name: str,
    tenant_id: str | None,
    tip: str,
    base_sha: str,
    approved_tree: str,
) -> None:
    """Refuse (403) unless the push sends EXACTLY ONE commit, ``base_sha`` -> ``tip``,
    whose tree is ``approved_tree``.

    WHY ONE COMMIT (coding-containers#156 validator). A tip-only content check passes
    ``c1`` (adds evil to an allowed file) + ``c2`` (reverts it): the tip's tree is the
    reviewed one, yet ``c1`` travels in the pack and shows in the pull request. The
    approved tree is ONE reviewed snapshot; any intermediate commit carries a tree
    nobody reviewed, so verifying intermediates would still need an approval that does
    not exist. The commit route (``approved_tree``) makes exactly this one commit, so
    the rule costs the conveyor nothing. Pushes without ``approved_tree`` (no content
    check requested) keep their multi-commit behaviour.

    ``tip``'s only parent being ``base_sha`` (the commit GitHub reports the remote
    has) makes the range ``tip --not base_sha`` exactly ``{tip}`` by construction, and
    tree-id equality binds every byte of it. Read from the raw commit object in its OWN
    hardened, poison-checked exec. An unreadable object refuses too (409).
    """
    result = await manager.execute_command(
        container_name, _tip_commit_command(tip), tenant_id=tenant_id
    )
    header = _commit_header(result.get("stdout") or "") if result.get("exit_code", 1) == 0 else None
    if header is None:
        logger.error(
            "tip commit read failed for container %s (exit %s); push refused",
            container_name,
            result.get("exit_code"),
        )
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": PUSH_TREE_MISMATCH,
                "message": (
                    f"{PUSH_TREE_MISMATCH}: the pushed commit {tip} could not be read, so it "
                    "cannot be matched to the approved tree and the push was refused."
                ),
            },
        )
    tree, parents = header
    if parents != [base_sha]:
        logger.error(
            "%s for container %s: tip %s has parents %s, base is %s",
            PUSH_NOT_SINGLE_COMMIT,
            container_name,
            tip,
            parents,
            base_sha,
        )
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": PUSH_NOT_SINGLE_COMMIT,
                "message": (
                    f"{PUSH_NOT_SINGLE_COMMIT}: with approved_tree the push must send exactly "
                    f"one commit on top of {base_sha} (the commit the remote has); {tip} has "
                    f"parent(s) {', '.join(parents) or '<none>'}. Commit the approved tree "
                    "with git/commit approved_tree + approved_base. Nothing was pushed."
                ),
                "tip": tip,
                "parents": parents,
                "base_sha": base_sha,
            },
        )
    if tree != approved_tree:
        logger.error(
            "%s for container %s: tip %s has tree %s, approved %s",
            PUSH_TREE_MISMATCH,
            container_name,
            tip,
            tree,
            approved_tree,
        )
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": PUSH_TREE_MISMATCH,
                "message": (
                    f"{PUSH_TREE_MISMATCH}: the pushed commit {tip} has tree {tree}, not the "
                    f"approved tree {approved_tree}. Nothing was pushed."
                ),
                "tip": tip,
                "tree": tree,
                "approved_tree": approved_tree,
            },
        )


def _path_refusal(container_name: str, refusals: list[PathRefusal]) -> HTTPException:
    """403 with ``path_denied_process_data`` if any path is process data, else
    ``path_not_in_target_files``; every refused path is listed in ``paths``."""
    codes = {r.error_code for r in refusals}
    code = next(
        c for c in (PATH_DENIED_PROCESS_DATA, PATH_INVALID, PATH_NOT_IN_TARGET_FILES) if c in codes
    )
    shown = [r.path for r in refusals[:20]]
    logger.warning(
        "%s for container %s: %d refused path(s), e.g. %s",
        code,
        container_name,
        len(refusals),
        ", ".join(shown),
    )
    return HTTPException(
        status_code=403,
        detail={
            "error_code": code,
            "message": (
                f"{code}: the push was refused because the pushed commits touch "
                f"{len(refusals)} path(s) a user repository must not receive "
                f"({', '.join(shown)}). Nothing was pushed."
            ),
            "paths": [
                {"path": r.path, "error_code": r.error_code, "rule": r.rule} for r in refusals
            ],
        },
    )


def _guard_keys(result: dict) -> list[str]:
    """The refused keys a guard printed (first token of each ``key value`` line)."""
    return sorted(
        {line.split()[0] for line in (result.get("stdout") or "").splitlines() if line.strip()}
    )


def _config_refusal(container_name: str, keys: list[str]) -> HTTPException:
    logger.error(
        "push_config_refused for container %s: workspace git config sets %s",
        container_name,
        ", ".join(keys) or "<unreadable>",
    )
    return HTTPException(
        status_code=409,
        detail={
            "error_code": "push_config_refused",
            "message": (
                "push_config_refused: the workspace git configuration contains settings "
                "that could redirect the push or hand its credential to another program "
                f"({', '.join(keys) or 'git config could not be read'}). Remove them and retry."
            ),
            "keys": keys,
        },
    )


async def _app_credential(
    clean_url: str, tenant_id: str | None, scope: Literal["clone", "push"], branch: str | None
) -> CloneCredential:
    """coding-github's answer; ``branch`` travels only with the push scope.

    The clone call is unchanged (no ``branch`` kwarg), so the fetch route and its
    callers see exactly the call they saw before.
    """
    manager = get_github_token_manager()
    if scope == "push" and branch:
        return await manager.get_app_clone_credential(
            clean_url, tenant_id=tenant_id, scope=scope, branch=branch
        )
    return await manager.get_app_clone_credential(clean_url, tenant_id=tenant_id, scope=scope)


#: Error code when coding-github issues no tenant-bound FETCH credential for a
#: customer repository (the fetch twin of ``PUSH_CREDENTIAL_ERROR``).
FETCH_CREDENTIAL_ERROR = "github_app_fetch_credential_unavailable"


async def resolve_repo_credential(
    container: Any,
    *,
    tenant_id: str | None,
    owner: str,
    clean_url: str,
    container_name: str,
    scope: Literal["clone", "push"],
) -> str | None:
    """The GitHub credential for ONE git transfer (see ``_resolve_repo_credential``)."""
    token, _ = await _resolve_repo_credential(
        container,
        tenant_id=tenant_id,
        owner=owner,
        clean_url=clean_url,
        container_name=container_name,
        scope=scope,
    )
    return token


async def _resolve_repo_credential(
    container: Any,
    *,
    tenant_id: str | None,
    owner: str,
    clean_url: str,
    container_name: str,
    scope: Literal["clone", "push"],
    branch: str | None = None,
) -> tuple[str | None, CloneCredential | None]:
    """The GitHub credential for ONE git transfer to/from ``clean_url``, resolved NOW.

    Returns ``(token, credential)``: ``credential`` is coding-github's answer when
    the token came from it (``None`` for a BYO token), so the push route can read
    the repository facts that answer carries (``branch`` is forwarded for those).

    Shared by ``git/push`` (``scope="push"``) and ``git/fetch`` (``scope="clone"``)
    so the two routes cannot drift apart on who may get which token. Re-resolved on
    every call; nothing is cached or written into the workspace.

    Ladder (first hit wins, every refusal is an ``HTTPException``):

    1. BYO: ``container.github_access_token``, else the per-tenant ``GIT_TOKEN``
       re-resolved from secrets-service via ``secret_config_id``.
    2. Customer owner: coding-github's tenant-bound App token (tenant in the body);
       none ⇒ 412 with coding-github's reason. The system PAT is NEVER used.
    3. Platform owner: a tenantless call is refused (``platform_tenant_missing``);
       coding-github's ``platform_repo_forbidden`` (a non-platform tenant naming a
       repository outside its allowlist) is a 403 and STOPS — no PAT fallback; any
       other null token is a 412 (``platform_credential_unavailable``).

    Returns ``None`` only when nothing above produced a token and nothing refused;
    the caller turns that into a 401.
    """
    is_push = scope == "push"
    verb = "push" if is_push else "fetch"
    github_token: str | None = None
    if hasattr(container, "github_access_token"):
        github_token = container.github_access_token

    # BYO flow: the container is provisioned with a secret_config_id (not a
    # plaintext token), so github_access_token is intentionally None. Re-resolve
    # the per-tenant GIT_TOKEN from secrets-service — the same source the clone
    # used — keyed by the persisted secret_config_id + the caller's tenant.
    if not github_token and getattr(container, "secret_config_id", None):
        from src.services.secret_resolver import resolve_git_token

        github_token = await resolve_git_token(
            container.secret_config_id,
            tenant_id or getattr(container, "tenant_id", None),
        )

    if github_token:
        return github_token, None

    if not is_platform_owner(owner):
        # Customer repo: the system PAT is NEVER a credential here (tenant-bound
        # invariant). Ask coding-github for a repo-scoped token from the
        # public-App installation bound to THIS tenant (tenant in the body, never
        # X-Tenant-Id); with no such token, fail closed with coding-github's
        # stated reason.
        req_tenant = tenant_id or getattr(container, "tenant_id", None)
        credential = await _app_credential(clean_url, req_tenant, scope, branch)
        if credential.ok and credential.token:
            return credential.token, credential
        owner_label = owner or "<unknown owner>"
        reason = credential.reason or credential.error or "unknown"
        if reason == "tenant_required":
            error_code = "tenant_required"
            direction = "a push to" if is_push else "a fetch from"
            message = (
                f"tenant_required for {owner_label}: {direction} a customer "
                "repository needs the provisioning tenant, and none is known "
                "for this container. Re-provision it from the fr workspace with the "
                f"nonagentic GitHub App installed on {owner_label}, or provision it "
                "with a GitHub token as access_token."
            )
        elif reason == "not_installed":
            error_code = APP_NOT_INSTALLED_ERROR
            message = (
                f"{APP_NOT_INSTALLED_ERROR} for {owner_label}: the GitHub App "
                "is not installed for this tenant on that owner. Install the "
                f"nonagentic GitHub App on {owner_label} from the fr workspace, or "
                "provision this workspace with a GitHub token as access_token."
            )
        else:
            error_code = PUSH_CREDENTIAL_ERROR if is_push else FETCH_CREDENTIAL_ERROR
            message = (
                f"{error_code} for {owner_label}: no BYO token is "
                f"attached and coding-github issued no tenant-bound {verb} "
                f"credential (reason={reason}). Install the nonagentic GitHub App "
                f"on {owner_label} from the fr workspace, or provision this "
                "workspace with a GitHub token as access_token."
            )
        logger.error(
            "%s for %s: no BYO token for container %s and no tenant-bound %s "
            "token (tenant=%s, reason=%s)",
            error_code,
            owner_label,
            container_name,
            verb,
            req_tenant or "<none>",
            reason,
        )
        raise HTTPException(
            status_code=412,
            detail={
                "error_code": error_code,
                "message": message,
                "owner": owner,
                "reason": reason,
            },
        )

    # Platform owner. coding-github is asked FIRST: it alone knows whether this
    # tenant may touch a platform repository at all (``platform_repo_forbidden``,
    # coding-github #107), and its App token is short-lived and single-repo where
    # the PAT is neither. A refusal STOPS here — answering it with the PAT would
    # grant the tenant exactly what coding-github just refused.
    #
    # A platform transfer is only ever made under a STATED tenant. An admin caller
    # (tenant_id=None) on a legacy tenantless row has none; refuse HERE, before
    # coding-github is asked and before any exec — never forward tenant=None and
    # let the far side decide.
    try:
        req_tenant = require_tenant_for_platform_repo(
            tenant_id or getattr(container, "tenant_id", None),
            owner=owner,
            repo_url=clean_url,
            context=f"git_{verb}",
        )
    except GitCredentialUnavailableError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "error_code": PLATFORM_TENANT_MISSING,
                "message": _credential_error_message(exc),
                "owner": owner,
                "reason": PLATFORM_TENANT_MISSING,
            },
        ) from exc
    credential = await _app_credential(clean_url, req_tenant, scope, branch)
    if credential.reason == PLATFORM_REPO_FORBIDDEN:
        logger.error(
            "%s for %s: coding-github refused a %s credential for container "
            "%s (tenant=%s); the system PAT is not a fallback",
            PLATFORM_REPO_FORBIDDEN,
            clean_url,
            verb,
            container_name,
            req_tenant or "<none>",
        )
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": PLATFORM_REPO_FORBIDDEN,
                "message": (
                    f"{PLATFORM_REPO_FORBIDDEN} for {owner}: platform "
                    "repositories are internal-tenant only."
                ),
                "owner": owner,
                "reason": PLATFORM_REPO_FORBIDDEN,
            },
        )
    # Fail closed on EVERY null token: no environment PAT is ever a credential for
    # a platform repository (org-wide, non-expiring — strictly more than
    # coding-github was willing to issue).
    try:
        token = platform_token_or_refuse(
            credential,
            repo_url=clean_url,
            owner=owner,
            tenant_id=req_tenant,
        )
        return token, credential
    except GitCredentialUnavailableError as exc:
        raise HTTPException(
            status_code=412,
            detail={
                "error_code": PLATFORM_CREDENTIAL_UNAVAILABLE,
                "message": _credential_error_message(exc),
                "owner": owner,
                "reason": credential.reason or credential.error or "none",
            },
        ) from exc


@router.post(
    "/{container_name}/git/push",
    response_model=GitPushResponse,
    operation_id="push_workspace_to_fork",
    summary="Push to GitHub Fork",
    responses={
        200: {"description": "Commits pushed successfully"},
        400: {"model": Error400Response, "description": "Invalid request or no commits"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Container not found"},
        422: {"model": Error422Response, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Error pushing commits"},
    },
)
async def push_to_fork(
    container_name: str = Path(..., description="Container name"),
    request: GitPushRequest = Body(...),
    user: User = Depends(get_current_user_with_tenant),
) -> GitPushResponse:
    """Push commits from container to GitHub fork."""
    github_token: str | None = None
    try:
        container_manager = get_git_manager()

        tenant_id = None if user.is_admin() else user.effective_tenant_id
        try:
            container = container_manager.get_container(container_name, tenant_id=tenant_id)
        except ValueError:
            container = None
        if not container:
            raise not_found("Container", container_name)

        if container.status.value.lower() != "running":
            raise bad_request(
                f"Container '{container_name}' is not running (current status: {container.status.value})"
            )

        # Validated BEFORE the first command that interpolates it.
        if not _validate_git_branch_name(request.branch):
            raise bad_request("Invalid branch name: contains forbidden characters or patterns")

        unpushed_result = await container_manager.execute_command(
            container_name,
            _in_workdir(
                f"{_hardened_git()} rev-list --count {shlex.quote(f'origin/{request.branch}')}..HEAD "
                f"2>/dev/null || {_hardened_git()} rev-list --count HEAD"
            ),
            tenant_id=tenant_id,
        )
        commits_to_push = int(unpushed_result.get("stdout", "0").strip() or 0)

        if commits_to_push == 0:
            raise bad_request("No commits to push - branch is up to date")

        remote_url: str = request.remote_url or ""
        if not remote_url:
            if hasattr(container, "repo_url") and container.repo_url:
                remote_url = container.repo_url
            else:
                raise bad_request(
                    "No remote URL provided and container has no configured repository"
                )

        # Fail-closed host allowlist. The credential helper installed below is
        # host-blind — it prints the tenant's GitHub token to whatever remote git
        # contacts — so an arbitrary `remote_url` would exfiltrate it. Reject any
        # remote outside the allowlist BEFORE the token is resolved or written.
        if not _validate_git_push_remote(remote_url):
            raise bad_request(
                "remote_url host is not an allowed GitHub push target; "
                "refusing to send credentials to an untrusted remote"
            )

        # Canonicalise BEFORE any credential is chosen. The owner that picks the
        # credential and the repository git actually pushes to must be the SAME
        # parse: the old code classified the raw string's first path segment
        # (``https://github.com/nonagenticai/../acme/r`` -> "nonagenticai", so the
        # platform PAT) and then handed the raw string to git, which resolves the
        # dot segments and pushes to acme/r with that PAT.
        push_target = canonical_github_repo(remote_url)
        if push_target is None:
            raise bad_request(
                "remote_url is not a canonical GitHub repository URL "
                "(https://github.com/<owner>/<repo>, git@github.com:<owner>/<repo>)"
            )

        # A workspace pushes only to the repository it was provisioned from. Any
        # other target is refused before a credential is resolved — this is what
        # stops a tenant naming someone else's repo (a customer's, or a
        # nonagenticai one the platform PAT can write) as remote_url.
        origin = canonical_github_repo(getattr(container, "repo_url", None))
        if not same_github_repo(push_target, origin):
            push_label = f"{push_target[0]}/{push_target[1]}"
            origin_label = f"{origin[0]}/{origin[1]}" if origin else "<none>"
            logger.warning(
                "push_target_mismatch for container %s: requested %s, provisioned from %s",
                container_name,
                push_label,
                origin_label,
            )
            raise HTTPException(
                status_code=403,
                detail={
                    "error_code": "push_target_mismatch",
                    "message": (
                        f"push_target_mismatch: {push_label} is not this workspace's "
                        f"origin repository ({origin_label}); a workspace may only "
                        "push to the repository it was provisioned from."
                    ),
                    "requested": push_label,
                    "origin": origin_label,
                },
            )
        push_owner, push_repo = push_target
        clean_url = canonical_github_url(push_owner, push_repo)

        github_token, facts = await _resolve_repo_credential(
            container,
            tenant_id=tenant_id,
            owner=push_owner,
            clean_url=clean_url,
            container_name=container_name,
            scope="push",
            branch=request.branch,
        )

        if not github_token:
            raise unauthorized("GitHub authentication not configured for this container")

        if facts is None:
            # BYO token: no coding-github answer came with it, but the refusals
            # below still need GitHub's facts and this pod has no GitHub egress.
            # Ask coding-github for the push-scope answer purely for its facts (its
            # token is discarded, never used); no token ⇒ no facts ⇒ refused below.
            facts = await _app_credential(
                clean_url,
                tenant_id or getattr(container, "tenant_id", None),
                "push",
                request.branch,
            )
            if not facts.token:
                facts = None

        # coding-github's push-scope answer (it read ``GET /repos/{o}/{r}`` with the
        # token it minted) answers both repository-level refusals below. Both happen
        # HERE: after the credential is resolved but before it reaches the container
        # and before origin is re-pointed. This pod never calls api.github.com.
        repo_info = await _remote_repo_info(push_owner, push_repo, facts)

        # A PUBLIC repository is never a conveyor target (operator ruling; design:
        # n8n_custom_nodes AA_IMPLEMENTATION/TENANCY_AND_IMPROVEMENT.md §8). An
        # unreadable or missing visibility refuses too (fail closed): "could not
        # tell" must never be treated as "private".
        if repo_info is None or repo_info.private is None:
            logger.error(
                "repo_visibility_unknown for container %s: could not read the visibility "
                "of %s; push refused",
                container_name,
                clean_url,
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": "repo_visibility_unknown",
                    "message": (
                        f"repo_visibility_unknown: the visibility of {clean_url} could "
                        "not be established (coding-github, which reads it from GitHub "
                        "for this service, supplied no answer), so the push was refused "
                        "(a public repository is never a push target). Retry; if it "
                        "persists, check coding-github's clone-token log and that its "
                        "App can read the repository."
                    ),
                    "fork_url": clean_url,
                },
            )
        if not repo_info.private:
            logger.warning(
                "public_repo_refused for container %s: %s is public", container_name, clean_url
            )
            raise HTTPException(
                status_code=403,
                detail={
                    "error_code": "public_repo_refused",
                    "message": (
                        f"public_repo_refused: {clean_url} is a public repository; "
                        "a workspace never pushes to a public repository."
                    ),
                    "fork_url": clean_url,
                },
            )

        # Refuse any push whose destination is the repository's default branch —
        # the same invariant coding-github #108 enforces on its own API. This route
        # holds a contents:write credential, and GitHub cannot scope a token to a
        # branch, so the refusal has to happen HERE, before the credential reaches
        # the container. The default branch comes from coding-github (read from
        # GitHub there), never from the workspace; an unresolvable one refuses (fail
        # closed) — including an older coding-github that does not send it.
        default_branch = repo_info.default_branch
        if default_branch is None:
            logger.error(
                "default_branch_unresolved for container %s: could not read the default "
                "branch of %s; push refused",
                container_name,
                clean_url,
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "error_code": "default_branch_unresolved",
                    "message": (
                        f"default_branch_unresolved: the default branch of {clean_url} "
                        "could not be established (coding-github supplied none), so "
                        "the push was refused."
                    ),
                },
            )
        if _is_default_branch(request.branch, default_branch):
            logger.warning(
                "default_branch_push_refused for container %s: %s is the default branch of %s",
                container_name,
                request.branch,
                clean_url,
            )
            raise HTTPException(
                status_code=403,
                detail={
                    "error_code": "default_branch_push_refused",
                    "message": (
                        f"default_branch_push_refused: {request.branch} is the default "
                        f"branch of {clean_url}; a workspace pushes to a feature branch "
                        "and changes reach the default branch only through a pull request."
                    ),
                    "branch": request.branch,
                    "default_branch": default_branch,
                },
            )

        # A user repository receives card deliverables only. Refuse the WHOLE push
        # if any pushed commit touches process data (or, when the caller named
        # target_files, anything outside them) — before origin is re-pointed and
        # before the credential reaches the container. The range starts at a
        # commit GitHub reports (remote branch head, else the head of the branch
        # this service recorded as the clone branch, else default-branch head),
        # never at a workspace ref; the listing resolves the tip ONCE and the push
        # sends exactly that commit. A lookup or listing that fails refuses too:
        # an unread range is never approved.
        try:
            clone_head = await _clone_branch_head(
                container,
                clean_url=clean_url,
                tenant_id=tenant_id or getattr(container, "tenant_id", None),
                push_branch=request.branch,
                default_branch=default_branch,
                facts=facts,
            )
            base_sha = (
                clone_head.sha
                if clone_head is not None
                else await _push_base_sha(
                    push_owner, push_repo, request.branch, default_branch, facts
                )
            )
            repo_empty = False
        except _EmptyRepositoryError:
            base_sha, repo_empty = None, True
        if base_sha is None:
            reason = "empty_repository" if repo_empty else "lookup_failed"
            logger.error(
                "push_base_unresolved for container %s: no base commit for %s on %s (%s); "
                "push refused",
                container_name,
                request.branch,
                clean_url,
                reason,
            )
            if repo_empty:
                message = (
                    f"push_base_unresolved: {clean_url} is an empty repository (no "
                    "branches, no commits), so there is no base commit to check the "
                    "pushed paths against and the push was refused. Create an initial "
                    f"commit on its default branch ({default_branch}) on GitHub, e.g. "
                    "a README, and retry."
                )
            else:
                message = (
                    f"push_base_unresolved: the head of {request.branch} (or of the "
                    f"default branch {default_branch}) on {clean_url} could not be read "
                    "via coding-github (lookup failed), so the pushed paths cannot be checked "
                    "and the push was refused. Retry; if it persists, check that the "
                    "credential can read the repository."
                )
            raise HTTPException(
                status_code=502,
                detail={
                    "error_code": "push_base_unresolved",
                    "message": message,
                    "reason": reason,
                    "fork_url": clean_url,
                },
            )
        listing = await container_manager.execute_command(
            container_name,
            _pushed_paths_command(request.branch, base_sha),
            tenant_id=tenant_id,
        )
        if listing.get("exit_code", 1) == _BASE_UNKNOWN_EXIT:
            logger.warning(
                "push_base_unknown for container %s: base %s is not in the workspace",
                container_name,
                base_sha,
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": "push_base_unknown",
                    "message": (
                        f"push_base_unknown: the fork {clean_url} is at base commit "
                        f"{base_sha}, which this workspace does not have, so the commits "
                        "the push would send cannot be listed and the push was refused. "
                        f"If the fork is ahead (someone pushed to {request.branch} or its "
                        f"default branch), fetch {base_sha} from {clean_url} and rebase "
                        "your commits onto it. If the fork is private, that fetch needs a "
                        "credential that can read it (an anonymous fetch reports "
                        "'repository not found'). Then retry the push."
                    ),
                    "base_sha": base_sha,
                    "fork_url": clean_url,
                },
            )
        parsed = (
            _parse_listing(listing.get("stdout") or "")
            if listing.get("exit_code", 1) == 0
            else None
        )
        if parsed is None:
            logger.error(
                "pushed-path listing failed for container %s (exit %s); push refused",
                container_name,
                listing.get("exit_code"),
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": PATH_DENIED_PROCESS_DATA,
                    "message": (
                        f"{PATH_DENIED_PROCESS_DATA}: the paths touched by the pushed "
                        "commits could not be listed, so the push was refused."
                    ),
                },
            )
        push_tip, pushed_paths = parsed
        refusals = refused_paths(pushed_paths, request.target_files)
        if refusals:
            raise _path_refusal(container_name, refusals)

        # CONTENT, not only paths. PRIMARY: the push sends exactly one commit on top of
        # the remote's base, and its tree IS the reviewed snapshot tree (one object id
        # binds every byte; no intermediate commit rides along).
        if request.approved_tree is not None:
            await _check_approved_range(
                container_manager,
                container_name,
                tenant_id,
                push_tip,
                base_sha,
                request.approved_tree,
            )
        # SECONDARY: every pushed path has a reviewed blob and the tip holds it (the
        # diff's ``blobs``, read from the same snapshot tree).
        if request.target_blobs is not None:
            await _check_pushed_content(
                container_manager,
                container_name,
                tenant_id,
                push_tip,
                pushed_paths,
                request.target_blobs,
            )

        # origin is re-pointed at the canonical URL so the workspace's own view
        # (and ``git remote get-url --push origin`` in the push exec) agrees with
        # where the push goes. It is NOT what the push uses: that is the explicit
        # URL, because a remote NAME is resolved through config the tenant owns.
        await container_manager.execute_command(
            container_name,
            _in_workdir(f"{_hardened_git()} remote remove origin 2>/dev/null || true"),
            tenant_id=tenant_id,
        )

        add_remote_result = await container_manager.execute_command(
            container_name,
            _in_workdir(f"{_hardened_git()} remote add origin {_sanitize_shell_arg(clean_url)}"),
            tenant_id=tenant_id,
        )
        if add_remote_result.get("exit_code", 1) != 0:
            error_msg = _sanitize_error_message(
                add_remote_result.get("stderr", "Unknown error"), github_token
            )
            raise internal_error(f"Failed to configure git remote: {error_msg}")

        # Refuse BEFORE the credential is handed to anything: any config in any
        # scope that could redirect the push, collect the token, or MITM it, and
        # any include directive at all. This runs AFTER the origin re-point above,
        # never before it: a conditional include keyed on the origin URL
        # (``includeIf "hasconfig:remote.*.url:https://github.com/**"``) is
        # inactive until origin is canonical, so a guard run earlier judged a
        # config the push would not run under (auditor, #138). The push exec
        # repeats this check as its first step, with no round trip in between.
        # ``git config --get-regexp`` exits 1 for "no match"; 0 is a match and
        # anything else is an error — both refuse (fail closed).
        guard = await container_manager.execute_command(
            container_name, _config_guard_command(), tenant_id=tenant_id
        )
        guard_rc = guard.get("exit_code", 1)
        if guard_rc != 1 or (guard.get("stdout") or "").strip():
            raise _config_refusal(container_name, _guard_keys(guard))

        # The token reaches git ONLY as an environment variable of this one exec:
        # no file, no config entry, nothing left behind to clean up. See
        # ``_push_command`` for the residual (same-uid /proc/<pid>/environ).
        #
        # Containers are shallow (`git clone --depth 1`) clones, so `git push`
        # cannot always negotiate a minimal delta against the fork and may
        # transfer a near-full snapshot; give it a generous, env-tunable timeout.
        push_timeout = int(os.getenv("GIT_PUSH_TIMEOUT", "900"))
        push_result = await container_manager.execute_command(
            container_name,
            _push_command(push_owner, push_repo, request.branch, push_tip, force=request.force),
            tenant_id=tenant_id,
            timeout=push_timeout,
            env={_TOKEN_ENV: github_token},
        )

        if push_result.get("exit_code", 1) == _CONFIG_EXIT:
            raise _config_refusal(container_name, _guard_keys(push_result))
        if push_result.get("exit_code", 1) == _REWRITE_EXIT:
            logger.error(
                "push_url_rewritten for container %s: %s did not resolve to itself "
                "inside the push exec; refused",
                container_name,
                clean_url,
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": "push_url_rewritten",
                    "message": (
                        f"push_url_rewritten: {clean_url} is rewritten by the workspace "
                        "git configuration; the push was refused."
                    ),
                },
            )
        if push_result.get("exit_code", 1) != 0:
            error_msg = _sanitize_error_message(
                push_result.get("stderr", "Unknown error"), github_token
            )
            if "authentication" in error_msg.lower() or "permission" in error_msg.lower():
                raise unauthorized(f"GitHub authentication failed: {error_msg}")
            raise internal_error(f"Failed to push commits: {error_msg}")

        # A push to a URL updates no remote-tracking ref, so record what was
        # pushed as origin/<branch> (what the unpushed-commit probe above reads)
        # and set the upstream the old `push -u origin` used to. Best effort: the
        # push itself already succeeded.
        tracking = await container_manager.execute_command(
            container_name,
            _in_workdir(
                f"{_hardened_git()} update-ref {shlex.quote(f'refs/remotes/origin/{request.branch}')} "
                f"{push_tip} && "
                f"{_hardened_git()} branch --set-upstream-to={shlex.quote(f'origin/{request.branch}')} "
                f"{shlex.quote(request.branch)}"
            ),
            tenant_id=tenant_id,
        )
        if tracking.get("exit_code", 1) != 0:
            logger.warning(
                "Pushed %s but could not update origin/%s in container %s",
                clean_url,
                request.branch,
                container_name,
            )

        logger.info(
            "Successfully pushed %d commits to %s (branch: %s)",
            commits_to_push,
            clean_url,
            request.branch,
        )

        return GitPushResponse(
            message="Commits pushed successfully to fork",
            container_name=container_name,
            branch=request.branch,
            remote_url=clean_url,
            commits_pushed=commits_to_push,
            request_id=str(uuid7()),
        )

    except HTTPException:
        raise
    except (docker.errors.DockerException, ValueError, RuntimeError, OSError) as e:
        logger.error(
            "Failed to push commits in container '%s': %s",
            container_name,
            _sanitize_error_message(str(e), github_token),
        )
        raise internal_error(
            f"Failed to push commits: {_sanitize_error_message(str(e), github_token)}"
        ) from e

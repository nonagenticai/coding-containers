"""Clone without the credential ever reaching the URL, and prove nothing was left behind.

⛔ THE DEFECT THIS CLOSES (auditor P0 on #138). Every clone path built
``https://x-access-token:$GIT_TOKEN@github.com/<owner>/<repo>`` and handed it to
``git clone``. git records the clone URL VERBATIM as ``remote.origin.url``, so the
token sat at rest in ``/workspace/.git/config`` on the volume the workspace (uid
1000, running customer code) shares — for the life of the workspace. Only the
benchmark ``history_cutoff`` path removed it, and only as a side effect of
dropping ``origin``.

The fix is structural, not a cleanup: the URL is the canonical, tokenless
``https://<host>/<owner>/<repo>``, and the token reaches git through a one-shot
``!``-credential helper passed with ``git -c`` (process-scoped; ``-c`` is NEVER
written into the new repository's config). The helper reads the token from the
environment at call time, answers only ``get`` for exactly that host and path,
and drains ``store``/``erase`` — so it is the same host/path-bound shape as the
push route's (``routers/containers/git_push.py``), which now delegates here.

Then, as a tripwire rather than the control, :func:`scrub_and_verify` re-points
``origin`` at the canonical URL (which also repairs a volume cloned by an older
build) and FAILS the clone if any remote URL still carries userinfo or any file
under ``.git`` other than the fetched objects contains the token (config,
FETCH_HEAD, packed-refs, logs/). On failure the clone fails loudly; see
:func:`scrub_and_verify` for when the destination is also wiped.
"""

from __future__ import annotations

import re
import shlex

_HOST_RE = re.compile(r"^[A-Za-z0-9.\-]+$")
_SEGMENT = r"[A-Za-z0-9._\-]+"
_REPO_PATH_RE = re.compile(rf"^{_SEGMENT}/{_SEGMENT}$")
_ENV_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_REMOTE_URL_KEYS = shlex.quote(r"^remote\..*url$")

#: Env var the clone containers (k8s init container, Docker alpine/git) read.
CLONE_TOKEN_ENV = "GIT_TOKEN"


def repo_path_from_url_path(path: str) -> str:
    """``/owner/repo`` or ``/owner/repo.git`` → ``owner/repo``; ValueError otherwise."""
    target = (path or "").strip("/")
    if target.endswith(".git"):
        target = target[: -len(".git")]
    if not _REPO_PATH_RE.match(target):
        raise ValueError(f"Invalid repository path for a credential helper: {path!r}")
    return target


def credential_helper_script(host: str, repo_path: str, token_env: str) -> str:
    """The ``!``-helper git runs; prints the token ONLY for ``https://<host>/<repo_path>``.

    git feeds a helper ``key=value`` lines on stdin. With
    ``credential.useHttpPath=true`` those include ``path`` — without it the
    helper sees no path and answers nothing, so dropping that flag fails CLOSED.
    Only the ``get`` action is answered; ``store``/``erase`` (which git sends
    after a successful transfer, WITH the password) are drained and ignored. The
    token is read from ``$<token_env>`` at call time, so it never appears in this
    text, in any config file, or in git's argv.

    ``host``/``repo_path`` are charset-validated here, so they are safe as
    unquoted literals; the whole script is still shell-quoted as one ``-c``
    argument by the caller.
    """
    if not _HOST_RE.match(host or ""):
        raise ValueError(f"Invalid credential host: {host!r}")
    if not _REPO_PATH_RE.match(repo_path or ""):
        raise ValueError(f"Invalid credential repository path: {repo_path!r}")
    if not _ENV_RE.match(token_env or ""):
        raise ValueError(f"Invalid credential env var name: {token_env!r}")
    return (
        "!f() { "
        'test "$1" = get || { cat >/dev/null; return 0; }; '
        "s=; h=; p=; "
        'while IFS= read -r l; do case "$l" in '
        'protocol=*) s="${l#protocol=}";; '
        'host=*) h="${l#host=}";; '
        'path=*) p="${l#path=}";; '
        "esac; done; "
        f'if test "$s" = https && test "$h" = {host} '
        f'&& {{ test "$p" = {repo_path} || test "$p" = {repo_path}.git; }} '
        f'&& test -n "${token_env}"; then '
        f'printf "username=x-access-token\\npassword=%s\\n" "${token_env}"; '
        "fi; }; f"
    )


def clone_credential_config_args(
    host: str, repo_path: str, token_env: str = CLONE_TOKEN_ENV
) -> list[str]:
    """``git -c`` overrides for a clone/fetch that authenticates via the env var.

    ``credential.helper=`` first clears every helper git collected from the
    image's system/global config (a ``store`` helper there would be handed the
    token after the clone), then the host/path-bound one is added.
    ``core.askPass=`` stops the askpass fallback, which would inherit the env.
    """
    pairs = [
        "credential.helper=",
        f"credential.https://{host}.helper={credential_helper_script(host, repo_path, token_env)}",
        "credential.useHttpPath=true",
        "core.askPass=",
    ]
    args: list[str] = []
    for pair in pairs:
        args += ["-c", pair]
    return args


def clone_credential_shell_args(host: str, repo_path: str, token_env: str = CLONE_TOKEN_ENV) -> str:
    """:func:`clone_credential_config_args`, shell-quoted for a ``sh -c`` string."""
    return " ".join(
        shlex.quote(a) for a in clone_credential_config_args(host, repo_path, token_env)
    )


def scrub_and_verify(
    qdest: str, canonical_url: str, *, token_env: str | None, wipe_on_failure: bool
) -> str:
    """Shell fragment: re-point ``origin`` tokenless, then FAIL if a credential remains.

    ``qdest`` must already be shell-quoted. ``token_env`` is the env var the
    token was supplied in (``None`` on a tokenless public clone, where only the
    URL check runs). ``.git/objects`` is excluded from the content scan: it holds
    what the remote sent, not anything local git wrote with the token, and
    scanning a multi-GB pack under the init container's CPU ceiling is not free.

    On any failure the shell exits 1. ``wipe_on_failure`` additionally empties
    the destination first — right for the Docker paths, whose destination lives
    on a volume that outlives the clone container and was freshly created by
    this call. The k8s init container passes ``False``: its fragment also runs
    over a VALID pre-existing checkout (the idempotency guard skips the clone),
    which may hold uncommitted work and must never be deleted, and a failing
    init container already keeps the workspace container from ever starting
    against that emptyDir — so the failure is loud (Init:CrashLoopBackOff) and
    nothing the tenant runs can read what remains.
    """
    if token_env is not None and not _ENV_RE.match(token_env):
        raise ValueError(f"Invalid credential env var name: {token_env!r}")
    git = f"git -c safe.directory={qdest} -C {qdest}"
    qurl = shlex.quote(canonical_url)
    scrub = (
        f"if {git} config --get remote.origin.url >/dev/null 2>&1; then "
        f"{git} remote set-url origin {qurl}; fi"
    )
    # Braced so a failed scrub cannot be masked by the `;` inside the check.
    url_check = (
        f"{{ urls=$({git} config --get-regexp {_REMOTE_URL_KEYS} 2>/dev/null); "
        'case "$urls" in *@*) false;; esac; }'
    )
    token_check = ""
    if token_env is not None:
        token_check = (
            f' && {{ [ -z "${token_env}" ] || [ -z "$(find {qdest}/.git '
            f"-path {qdest}/.git/objects -prune -o -type f "
            f'-exec grep -lF -e "${token_env}" {{}} \\;)" ]; }}'
        )
    wipe = f"find {qdest} -mindepth 1 -delete 2>/dev/null || true; " if wipe_on_failure else ""
    return (
        f"{{ {scrub} && {url_check}{token_check}; }} || "
        f"{{ echo 'git-clone: a credential remains under .git; refusing the checkout' >&2; "
        f"{wipe}exit 1; }}"
    )

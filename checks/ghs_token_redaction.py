r"""⛔ CONVEYOR FIXTURE — DO NOT FIX THE DEFECT THIS DETECTS BY HAND.

This module is a DETECTOR for a deliberately-unfixed defect. The fix is absent on
purpose: the AA conveyor is driven at it end-to-end, and a hand-written fix
discharges the card while proving nothing about the machine.

  ✅ WELCOME, and does not consume the target: wiring this into CI, hardening it,
     widening its cases, correcting its prose.
  ⛔ NOT welcome: writing the fix in `src/`. That removes the target.

MEASURED 2026-09-15: a sibling session hand-fixed this repo's SIBLING fixture
(`enterprise_mcp_server/checks/http_status_binding.py`) ten minutes after it was
merged, explicitly discharging the conveyor's own card. The call was reasonable
from where they stood — an obvious bug with a detector sitting next to it — which
is exactly why the rule has to be readable HERE, at the point of temptation.

⚠️ This defect is PRE-EXISTING, not seeded. It was found by reading
`origin/main`, not planted. If the conveyor does not consume this fixture
promptly, fixing it by hand is the RIGHT call — it is a credential-redaction gap,
and no experiment is worth leaving one open indefinitely. Say plainly, in that
case, that the machine did not do it.

⚠️ If this detector already exits 0, the fixture has been consumed: the conveyor
delivered (or someone fixed it). Then this module is just a regression guard, and
normal rules apply.

---

`_sanitize_error_message` must redact every GitHub token class this service
actually handles.

THE DEFECT THIS EXISTS FOR. `src/routers/containers/_common.py` states its own
job in its docstring — "Remove sensitive data from error messages before
returning to client" — and then redacts only three token prefixes:

    ghp_[a-zA-Z0-9]{36}        classic personal access token
    gho_[a-zA-Z0-9]{36}        OAuth token
    github_pat_[a-zA-Z0-9_]{22,}   fine-grained PAT

It does NOT redact `ghs_` — GitHub App **installation** tokens — even though this
very service mints and handles them:

    src/clients/github_client.py:82          `ok=True, token="ghs_..."`
    src/core/kubernetes_container_provisioner.py:143,676

`ghs_` is also the token class the AA conveyor's own identity
(`code-graph-helper-app[bot]`) authenticates with. An installation token that
appears in a git error message is therefore returned to the caller in clear.
`ghu_` (user-to-server) has the same gap.

A THIRD leak, same function, same purpose: the embedded-credential URL rule
hardcodes `github\.com`, while `_allowed_git_push_hosts()` in the same module
explicitly supports GitHub Enterprise hosts via `GIT_PUSH_ALLOWED_HOSTS`. A
credential in an Enterprise remote URL is passed through untouched.

This is a BEHAVIOURAL check: it calls the real function rather than inspecting
the regexes, so any correct fix passes — widening the character class, adding
prefixes, or rewriting the redaction entirely — and a fix that merely looks right
does not.

    uv run python -m checks.ghs_token_redaction

Exit 0 = every secret redacted, nothing over-redacted. Exit 1 = a secret leaks,
the defect is live. Exit 2 = the check could not run and must NOT be read as a
pass.
"""

from __future__ import annotations

import sys

_B36 = "NOTAREALTOKEN" + "0" * 23  # 36 alphanumerics; deliberately low-entropy
# so the repo secret scanner does not read this fixture as a real credential.
_CRED_USER = "user"
_CRED_PW = "notarealpassword"  # split from the URL for the same reason: an
# inline credential-in-URL literal would read as Basic Auth credentials.

# (label, error text, the exact secret substring that must NOT survive)
# The ghp_/gho_/github_pat_ rows already pass today. They are POSITIVE CONTROLS
# held in the same run: if they ever fail, this check is broken or the fix
# regressed, and "everything leaks" is distinguishable from "the new case leaks".
MUST_REDACT = (
    ("ghp_ classic PAT (control)", f"fatal: auth failed for ghp_{_B36}", f"ghp_{_B36}"),
    ("gho_ oauth (control)", f"fatal: auth failed for gho_{_B36}", f"gho_{_B36}"),
    (
        "github_pat_ fine-grained (control)",
        f"fatal: bad credentials github_pat_{_B36}",
        f"github_pat_{_B36}",
    ),
    ("ghs_ APP INSTALLATION", f"fatal: auth failed for ghs_{_B36}", f"ghs_{_B36}"),
    ("ghu_ user-to-server", f"fatal: auth failed for ghu_{_B36}", f"ghu_{_B36}"),
    (
        "credential in github.com URL (control)",
        f"remote: https://{_CRED_USER}:{_CRED_PW}@github.com/org/repo.git rejected",
        _CRED_PW,
    ),
    (
        "credential in ENTERPRISE URL",
        f"remote: https://{_CRED_USER}:{_CRED_PW}@github.enterprise.corp/org/repo.git rejected",
        _CRED_PW,
    ),
)

# Text that MUST survive unchanged. If a fix over-corrects into a blanket scrub,
# that is a regression and this check reports it as one rather than going green.
MUST_PRESERVE = (
    ("ordinary error", "fatal: repository 'origin' does not exist"),
    ("url without credentials", "remote: https://github.com/org/repo.git not found"),
    ("prefix-like but not a token", "see the ghs_ prefix documentation for details"),
)


def main() -> int:
    try:
        from src.routers.containers._common import _sanitize_error_message
    except Exception as exc:  # pragma: no cover - reported, never swallowed
        print(f"FAIL: could not import _sanitize_error_message: {exc!r}")
        print("This check measured nothing; do not read it as a pass.")
        return 2

    leaked: list[tuple[str, str]] = []
    destroyed: list[tuple[str, str]] = []

    for label, error, secret in MUST_REDACT:
        try:
            out = _sanitize_error_message(error)
        except Exception as exc:
            print(f"FAIL: _sanitize_error_message({error!r}) raised {exc!r}")
            return 2
        if secret in out:
            leaked.append((label, out))

    for label, error in MUST_PRESERVE:
        try:
            out = _sanitize_error_message(error)
        except Exception as exc:
            print(f"FAIL: _sanitize_error_message({error!r}) raised {exc!r}")
            return 2
        if out != error:
            destroyed.append((label, out))

    checked = len(MUST_REDACT) + len(MUST_PRESERVE)
    print(f"messages checked: {checked}")

    if leaked or destroyed:
        if leaked:
            print(f"\nFAIL: {len(leaked)} secret(s) SURVIVED redaction:")
            for label, out in leaked:
                print(f"  [{label}] -> {out}")
            print(
                "\nsrc/routers/containers/_common.py :: _sanitize_error_message\n"
                "redacts ghp_/gho_/github_pat_ and github.com credential URLs only.\n"
                "ghs_ (App installation) and ghu_ (user-to-server) tokens, and\n"
                "credentials in GitHub Enterprise remote URLs, are returned in clear."
            )
        if destroyed:
            print(f"\nFAIL: {len(destroyed)} message(s) that must survive were ALTERED:")
            for label, out in destroyed:
                print(f"  [{label}] -> {out}")
            print("\nThe redaction has been widened too far; this is a regression.")
        return 1

    print(f"\nok: all {checked} message(s) handled correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

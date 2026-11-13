"""⛔ CONVEYOR FIXTURE — DO NOT FIX THE DEFECT THIS DETECTS BY HAND.

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
is exactly why the rule has to be readable HERE, at the point of temptation,
rather than in a session message that dies with the session holding it.

⚠️ If this detector already exits 0, the fixture has been consumed: the conveyor
delivered (or someone fixed it). Then this module is just a regression guard, and
normal rules apply.

STATUS: consumed 2026-09-15 by the conveyor itself — card 1864601471895471901 →
PR #123 → merged `8c7ec950`. Kept as a regression guard.

---

`_validate_git_branch_name` must reject a branch name with a newline in it.

THE DEFECT THIS EXISTS FOR. `src/routers/containers/_common.py` validates branch
names with

    re.match(r"^[a-zA-Z0-9][a-zA-Z0-9/_-]*$", branch)

In Python, `$` matches at the end of the string **or immediately before a
trailing newline**. So `"main\n"` satisfies the pattern; the three remaining
clauses (`".." not in branch`, no leading `/`, no trailing `/`) all pass for it
as well, and the function returns True. The value then flows on to the git-push
path (`src/routers/containers/git_push.py`), which is precisely where a name
carrying a newline should never arrive.

`\\Z` is the anchor that means "end of string" with no newline exemption.

This is a BEHAVIOURAL check: it calls the real function rather than inspecting
the regex, so any correct fix passes -- switching the anchor, or rejecting
newlines explicitly -- and a fix that merely looks right does not.

    uv run python -m checks.branch_name_anchor

Exit 0 = rejected as it should be. Exit 1 = accepted, the defect is live.
Exit 2 = the check could not run and must not be read as a pass.
"""

from __future__ import annotations

import sys

# Names that MUST be accepted. If the fix over-corrects and starts rejecting
# ordinary branches, that is a regression and this check reports it as one
# rather than quietly going green.
MUST_ACCEPT = ("main", "feature/new-thing", "release_1-0", "aa/fix-20260915")

# Names that MUST be rejected. Each carries a newline somewhere.
MUST_REJECT = ("main\n", "main\nrm -rf /", "\nmain", "feature/x\n")


def main() -> int:
    try:
        from src.routers.containers._common import _validate_git_branch_name
    except Exception as exc:  # pragma: no cover - reported, never swallowed
        print(f"FAIL: could not import _validate_git_branch_name: {exc!r}")
        print("This check measured nothing; do not read it as a pass.")
        return 2

    wrong_accepts: list[str] = []
    wrong_rejects: list[str] = []

    for name in MUST_REJECT:
        try:
            if _validate_git_branch_name(name) is not False:
                wrong_accepts.append(name)
        except Exception as exc:
            print(f"FAIL: _validate_git_branch_name({name!r}) raised {exc!r}")
            return 2

    for name in MUST_ACCEPT:
        try:
            if _validate_git_branch_name(name) is not True:
                wrong_rejects.append(name)
        except Exception as exc:
            print(f"FAIL: _validate_git_branch_name({name!r}) raised {exc!r}")
            return 2

    checked = len(MUST_ACCEPT) + len(MUST_REJECT)
    print(f"branch names checked: {checked}")

    if wrong_accepts or wrong_rejects:
        if wrong_accepts:
            print(f"\nFAIL: {len(wrong_accepts)} name(s) with a newline were ACCEPTED:")
            for name in wrong_accepts:
                print(f"  {name!r}")
            print(
                "\nsrc/routers/containers/_common.py :: _validate_git_branch_name\n"
                "anchors its pattern with `$`, which matches before a trailing\n"
                "newline. These values reach the git-push path."
            )
        if wrong_rejects:
            print(f"\nFAIL: {len(wrong_rejects)} ordinary name(s) were REJECTED:")
            for name in wrong_rejects:
                print(f"  {name!r}")
            print("\nThe validator has been tightened too far; this is a regression.")
        return 1

    print(f"\nok: all {checked} name(s) classified correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

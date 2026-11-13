"""Standalone conveyor oracles.

Modules here are NOT pytest tests and are deliberately not collected: each one
is a self-contained check that exits 0 when a specific named defect is absent
and non-zero while it is present. They exist so an automated worker can be given
a machine-executable definition of done that does not depend on reading a test
report, and so `main` stays green while a defect is still open.

Run one as:  uv run python -m checks.<name>

⛔ SOME MODULES HERE ARE CONVEYOR FIXTURES, AND FIXING WHAT THEY DETECT DESTROYS
AN INSTRUMENT. A module whose whole content is a detector for a defect that is
STILL PRESENT in `src/` is deliberately unfixed: the AA conveyor is driven at it
end to end, and a hand-written fix discharges the card while proving nothing
about the machine. Each such module says so in its own docstring, and names the
card driving at it.

  ✅ Welcome, and consumes nothing: wiring one into CI, hardening it, widening
     its cases, fixing its prose.
  ⛔ Not welcome: writing the fix in `src/`.

A detector that already exits 0 has been consumed — the defect is gone and the
module is now an ordinary regression guard, with no special handling.

MEASURED 2026-09-15: a sibling session hand-fixed this repo's sibling fixture in
`enterprise_mcp_server` ten minutes after it merged, explicitly discharging the
conveyor's own card. Reasonable from where they stood — which is why the rule
lives here rather than in a session message.
"""

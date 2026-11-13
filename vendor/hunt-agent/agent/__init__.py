"""Self-contained bug-hunt agent (see ``hunt.py``).

This directory is VENDORED, file for file, into the workspace image at
``/opt/hunt-agent/`` by coding-containers. It is the single source of truth for
the hunt's detection logic and scope; the service imports it from here.

Rules for every file in this directory:

* stdlib + ``tree_sitter`` (and its grammar packages) only. Nothing from
  coding-analysis may be imported: the workspace image has no coding-analysis.
* ``MANIFEST.sha256`` must match the files (``scripts/hunt-agent-sha.py
  --check``); a unit test enforces it, so an edit without a manifest refresh
  reds CI.
"""

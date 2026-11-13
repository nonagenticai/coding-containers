#!/usr/bin/env bash
# Smoke for the codegraph-runtime image (provisioned-pod image).
#
# Runs in CI against the JUST-BUILT local image ($1), BEFORE it is published —
# see .github/workflows/runtime-image.yaml and ci-workflows/image-only.yml. A
# failure here therefore cannot move :latest.
#
# The interpreter assertion is the whole point: it is the check that was
# missing when the base silently sat two versions behind.
set -euo pipefail

IMAGE="${1:?usage: runtime-smoke.sh <image-ref>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== python =="
ACTUAL="$(docker run --rm "$IMAGE" python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "python: ${ACTUAL}"
if [ "$ACTUAL" != "3.14" ]; then
  echo "::error::runtime image python is ${ACTUAL}, expected 3.14"
  exit 1
fi

echo "== baked tools =="
docker run --rm "$IMAGE" uv --version
docker run --rm "$IMAGE" git --version
# ws-redis (py2rust gap #4 full): start, the server's own config (loopback, no
# persistence, the memory cap), set / get, stop.
docker run --rm "$IMAGE" sh -ec '
  test "$(ws-redis start)" = "redis://127.0.0.1:6379/0"
  cfg() { valkey-cli -p 6379 --raw config get "$1" | sed -n 2p; }
  test "$(cfg bind)" = 127.0.0.1
  test -z "$(cfg save)"
  test "$(cfg appendonly)" = no
  test "$(cfg maxmemory)" = 268435456
  valkey-cli -p 6379 set k v >/dev/null && test "$(valkey-cli -p 6379 get k)" = v
  ws-redis stop | grep -qx stopped'
docker run --rm "$IMAGE" sh -lc 'semgrep --version && node --version && npm --version'
# As USER 1000, the identity the grid legs exec as — the build-time smoke in
# Dockerfile.runtime runs as root and cannot prove this.
docker run --rm "$IMAGE" python -m pytest --version

# The astral quality tools as coding-analysis execs them (uid 1000): a finding
# must exit 1 and a missing path must exit 2 — never 0.
echo "== quality tools =="
docker run --rm -w /tmp "$IMAGE" bash -c '
  set -u
  ruff --version && ty --version
  printf "import os\ndef f(x: int) -> str:\n    return x\n" > bad.py
  rc=0; ruff check --output-format=concise --extend-select I,PL bad.py || rc=$?
  [ "$rc" = 1 ] || { echo "::error::ruff check exit $rc, expected 1"; exit 1; }
  rc=0; ty check --output-format gitlab bad.py >/dev/null || rc=$?
  [ "$rc" = 1 ] || { echo "::error::ty exit $rc, expected 1"; exit 1; }
  rc=0; ty check --output-format gitlab missing.py >/dev/null 2>&1 || rc=$?
  [ "$rc" = 2 ] || { echo "::error::ty missing-path exit $rc, expected 2"; exit 1; }
  for t in mypy pylint black isort flake8 autoflake; do
    if command -v "$t" >/dev/null; then echo "::error::$t is still baked"; exit 1; fi
  done'

# The bug hunt as coding-analysis runs it: `python3 -I -P
# /opt/hunt-agent/hunt.py` as uid 1000, on a throwaway repo with one file per
# grammar. Fails unless the agent reports PIN's agent_sha and tree-sitter /
# grammar versions and parses every file. The check script is fed on stdin (it
# is not in the image); only the pin is passed in.
echo "== hunt agent =="
HUNT_PIN_JSON="$(cat "$ROOT/vendor/hunt-agent/PIN.json")"
docker run --rm -i -e HUNT_PIN_JSON="$HUNT_PIN_JSON" "$IMAGE" \
  python3 - image-smoke --agent-dir /opt/hunt-agent < "$ROOT/scripts/check-hunt-agent-vendor.py"

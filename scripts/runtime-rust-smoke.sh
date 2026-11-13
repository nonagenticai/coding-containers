#!/usr/bin/env bash
# Smoke for the codegraph-runtime-rust image: rust toolchain, a loopback
# Postgres (ws-pg), and a real `cargo add` + `cargo build` against crates.io.
#
# Runs in CI against the JUST-BUILT local image ($1), BEFORE it is published —
# see .github/workflows/runtime-image.yaml and ci-workflows/image-only.yml.
#
# CARGO_BUILD_JOBS=2 and the tmpfs mounts keep the build inside the runner's
# memory/disk budget; the container runs as uid 1000, so HOME must be writable.
set -euo pipefail

IMAGE="${1:?usage: runtime-rust-smoke.sh <image-ref>}"

docker run --rm -e CARGO_BUILD_JOBS=2 \
  --tmpfs /tmp:size=2g,mode=1777,exec \
  --tmpfs /home/coder:uid=1000,gid=1000,size=1g,mode=0755,exec \
  "$IMAGE" sh -ec '
    test "$(id -u)" = 1000
    cargo --version | grep -q "^cargo 1.99.0 "
    rustc --version | grep -q "^rustc 1.99.0 "
    cargo clippy --version; cargo fmt --version
    cargo nextest --version; cargo audit --version; cargo deny --version; sqlx --version
    python3 --version; uv --version
    U="$(ws-pg start)"
    test "$U" = "postgresql://coder@127.0.0.1:5432/ws"
    psql "$U" -tAc "select version()" | grep -q "PostgreSQL 18"
    psql "$(ws-pg createdb smoke_db)" -tAc "select current_database()" | grep -qx smoke_db
    ws-pg psql smoke_db -c "select current_database()" | grep -qx smoke_db
    psql --version | grep -q " 18\.6"
    if ws-pg psql smoke_db -c "\\! id" 2>/dev/null; then echo "ws-pg psql ran a meta-command" >&2; exit 1; fi
    cd /tmp && cargo new -q smoke && cd smoke
    cargo add -q serde --features derive
    cargo build -q && ./target/debug/smoke
    R="$(ws-redis start)"
    test "$R" = "redis://127.0.0.1:6379/0"
    valkey-cli -p 6379 set k v >/dev/null && test "$(valkey-cli -p 6379 get k)" = v
    ws-redis stop >/dev/null
    ws-pg stop'

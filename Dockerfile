ARG BASE_REGISTRY=docker.io

# --- Stage 1: Builder ---
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./

# The coding-client dependency is a pinned git URL (see [tool.uv.sources]);
# uv shells out to `git`, which the slim base does not ship.
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN uv sync --frozen --no-dev --no-install-project

# --- Stage 2: Production ---
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create appuser BEFORE the copies so each COPY can set ownership inline via
# --chown. A trailing `chown -R appuser:appuser /app` instead measured
# `#19 DONE 175.4s` on this repo's own cache-cold BuildKit log -- the inode
# traversal, plus a second non-dedupable layer uploaded whole. /app itself is
# chowned non-recursively so the directory keeps the ownership the old
# recursive chown gave it.
RUN useradd -m -u 1000 appuser && mkdir -p /app && chown appuser:appuser /app
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app/uv.lock /app/uv.lock
COPY --from=builder --chown=appuser:appuser /app/pyproject.toml /app/pyproject.toml
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser src/ ./src/
# Migrations for the coding_containers DB (async exec job store); the Deployment's
# `migrate` initContainer runs `alembic upgrade head` from /app, as coding-analysis.
COPY --chown=appuser:appuser alembic/ ./alembic/
COPY --chown=appuser:appuser alembic.ini ./

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# The useradd + chown above always intended a non-root runtime, but without a
# USER directive the service ran as uid 0 in production purely by omission
# (`kubectl -n coding-containers exec deploy/coding-containers -- id -u` -> 0,
# while every other coding-* service returned 1000).
#
# Nothing here needs root: in Kubernetes mode (KUBERNETES_IN_CLUSTER=true, the
# production config) all container work is done through the k8s API and
# `exec_in_pod`, never locally. The local docker/subprocess provisioning path is
# already dead in this image -- it has no docker socket mounted and no `git`
# binary installed. Runtime writes are confined to /app (chowned above), /tmp
# (1777) and $HOME.
#
# HOME is set explicitly rather than left to the runtime: as root it resolved to
# /root (0700, root-owned), and the handful of os.path.expanduser("~") call sites
# must land somewhere writable by uid 1000 rather than silently failing.
ENV HOME=/home/appuser
USER appuser

EXPOSE 8691

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8691/health || exit 1

# --timeout-graceful-shutdown: on SIGTERM uvicorn waits for in-flight requests
# for at most this long, cancels the rest, and only THEN runs the lifespan
# shutdown (async exec ``interrupted`` callbacks). Without it a 300 s sync exec
# held the process past the kubelet's SIGKILL and the lifespan never ran.
# Budget (src/services/exec_jobs/runner.py, pinned by tests/unit/test_shutdown_budget.py):
#   0.1 + 10 (this flag) + 22 (exec runner) + 5 (pool close) + 10 (reaper stop)
#   = 47.1 s < terminationGracePeriodSeconds 60 (set on the Deployment in gitops).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8691", "--timeout-graceful-shutdown", "10"]

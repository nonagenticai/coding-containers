"""FastAPI application entry point for coding-containers."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import docker.errors
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.exceptions import RedisError

from coding_client.api.exception_handlers import register_exception_handlers
from coding_client.api.logging_setup import configure_logging
from coding_client.api.openapi_config import configure_openapi
from coding_client.api.request_logging import HealthCheckFilter, RequestLoggingMiddleware
from src.routers import (
    agents_router,
    container_dependencies_router,
    container_exec_router,
    containers_router,
    health_router,
    package_upgrade_router,
    validation_router,
    workspaces_router,
)

configure_logging()

logger = logging.getLogger(__name__)

SERVICE_NAME = os.getenv("SERVICE_NAME", "coding-containers")
VERSION = os.getenv("VERSION", "1.0.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
ENABLE_DOCS = os.getenv("ENABLE_DOCS", "false").lower() == "true"
# NB no module-level ``DEFAULT_TENANT_ID`` here. It was
# ``os.getenv("DEFAULT_TENANT_ID", "n8n-dev")`` with ZERO readers — but a dead
# constant carrying that literal is a booby trap, not inert: ``"n8n-dev"`` is the
# stale pre-uuid7 label of a tenant that really exists, so the next person to
# reach for it would have silently got a tenant nobody chose instead of a
# failure. ``clients/_tenant.py`` in this very repo already argues the rule:
# a missing tenant is an error, never a default. Read it from
# ``coding_client.auth.tenant.DEFAULT_TENANT_ID``, which is ``None`` when unset.
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan events."""
    logger.info(f"Starting {SERVICE_NAME} v{VERSION} ({ENVIRONMENT})")

    # Initialize Docker service
    try:
        from src.services.docker_service import get_docker_service

        docker_svc = get_docker_service()
        if docker_svc.is_docker_available():
            logger.info("Docker service initialized successfully")
        else:
            logger.warning("Docker service initialized but Docker is not available")
    except (docker.errors.DockerException, ImportError, OSError) as e:
        logger.error(f"Failed to initialize Docker service: {e}")

    # Initialize Redis connection
    redis_svc = None
    try:
        from src.core.redis_service import RedisService

        redis_svc = RedisService()
        if redis_svc.ping():
            logger.info(f"Redis connected at {redis_svc.host}:{redis_svc.port}")
        else:
            logger.warning("Redis service created but ping failed")
    except (RedisError, ImportError, OSError) as e:
        logger.warning(f"Redis not available at startup (non-fatal): {e}")

    # Start the container-Deployment reaper (Phase C-1). Inert unless
    # CODING_CONTAINERS_REAPER_ENABLED is set; even then dry-run by default.
    reaper_started = False
    try:
        from src.config.settings import get_settings
        from src.core.reaper import start_reaper

        settings = get_settings()
        if settings.reaper_enabled:
            start_reaper(app)
            reaper_started = True
            logger.info(
                "Reaper started (dry_run=%s, ttl=%ss, interval=%ss)",
                settings.reaper_dry_run,
                settings.reaper_ttl_seconds,
                settings.reaper_interval_seconds,
            )
        else:
            logger.info("Reaper disabled (CODING_CONTAINERS_REAPER_ENABLED is not set)")
    except (ImportError, OSError, RuntimeError) as e:
        logger.warning(f"Reaper not started (non-fatal): {e}")

    # Async exec job runner + orphan reaper (coding_containers DB). Inert when no
    # job-store DSN is configured: the async route then answers 503. The first
    # reaper sweep runs immediately, so jobs orphaned by the previous pod
    # generation are failed and called back at startup, not one interval later.
    exec_runner = None
    try:
        from src.config.settings import get_settings as _get_settings

        _s = _get_settings()
        if _s.exec_job_dsn:
            from src.services.exec_jobs.detached import K8sDetachedBackend
            from src.services.exec_jobs.runner import (
                ExecJobRunner,
                make_owner_id,
                set_exec_job_runner,
            )
            from src.services.exec_jobs.store import ExecJobStore

            exec_runner = ExecJobRunner(
                ExecJobStore(),
                owner=make_owner_id(),
                lease_seconds=_s.exec_job_lease_seconds,
                max_concurrent=_s.exec_async_max_concurrent,
                # Kubernetes workspaces run async commands detached in the
                # workspace pod, so a rollout of THIS service no longer kills
                # them; other runtimes keep the blocking path (locate -> None).
                detached=K8sDetachedBackend(),
            )
            set_exec_job_runner(exec_runner)
            exec_runner.start_reaper(_s.exec_job_reaper_interval_seconds)
            logger.info("Async exec runner started (owner=%s)", exec_runner.owner)
        else:
            logger.info("Async exec disabled (no PGBOUNCER_URL / DATABASE_URL)")
    except (ImportError, OSError, RuntimeError) as e:
        logger.warning(f"Async exec runner not started (non-fatal): {e}")

    yield

    # Shutdown
    logger.info(f"Shutting down {SERVICE_NAME}")
    if exec_runner is not None:
        try:
            from src.db.session import close_pool
            from src.services.exec_jobs.runner import set_exec_job_runner

            await exec_runner.shutdown()
            set_exec_job_runner(None)
            await close_pool()
        except Exception as e:  # shutdown must continue to the other resources
            logger.warning(f"Async exec shutdown error (non-fatal): {e}")
    if reaper_started:
        try:
            from src.core.reaper import stop_reaper

            await stop_reaper(app)
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning(f"Reaper shutdown error (non-fatal): {e}")
    if redis_svc is not None:
        try:
            redis_svc.close()
        except RedisError, OSError:
            pass


app = FastAPI(
    title=SERVICE_NAME,
    description=(
        "Repository workspace service backed by isolated Docker or Kubernetes runtimes.\n\n"
        "**Auth:** Istio-forwarded headers (`x-user-id`, `x-tenant-id`, `x-user-roles`) "
        "with trusted-network bypass for RFC1918/loopback IPs. "
        "License token introspection as fallback. No direct Keycloak interaction.\n\n"
        "**Endpoint groups:**\n"
        "- `/api/workspaces` -- repository workspace lifecycle, files, git, exec, agents, dependencies\n"
        "- `/api/containers` -- legacy compatibility alias (not published in OpenAPI)\n"
        "- `/api/health` -- per-container health, readiness, liveness, statistics\n"
        "- `/api/validation` -- Python code snippet validation with optional container execution\n"
    ),
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware (after CORS, before auth)
logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())
app.add_middleware(RequestLoggingMiddleware)

# Auth middleware (gated behind ENABLE_AUTH env var)
if ENABLE_AUTH:
    try:
        from coding_client.auth.middleware import AuthMiddleware

        app.add_middleware(AuthMiddleware)
        logger.info("AuthMiddleware enabled")
    except ImportError:
        logger.warning("coding_client.auth.middleware not available, auth disabled")

# Canonical public API. A workspace is the checked-out repository resource;
# Docker containers and Kubernetes Deployments are runtime implementations.
app.include_router(workspaces_router, prefix="/api/workspaces")

# Legacy compatibility aliases. Keep them callable so existing automation does
# not break, but publish only the canonical workspace surface in OpenAPI.
app.include_router(
    containers_router,
    prefix="/api/containers",
    tags=["Containers (legacy)"],
    include_in_schema=False,
)

# Container dependency operations (sub-paths under /api/containers)
app.include_router(
    container_dependencies_router,
    prefix="/api/containers",
    tags=["Containers - Dependencies (legacy)"],
    include_in_schema=False,
)

# Container exec operations (sub-paths under /api/containers)
app.include_router(
    container_exec_router,
    prefix="/api/containers",
    tags=["Containers - Exec (legacy)"],
    include_in_schema=False,
)

# Typed package-upgrade dispatcher (sub-paths under /api/containers)
app.include_router(
    package_upgrade_router,
    prefix="/api/containers",
    tags=["Containers - Package Upgrade (legacy)"],
    include_in_schema=False,
)

# Agent execution in containers (sub-paths under /api/containers)
app.include_router(
    agents_router,
    prefix="/api/containers",
    tags=["Containers - Agents (legacy)"],
    include_in_schema=False,
)

# Validation endpoints requiring Docker access (proxied from gateway)
app.include_router(validation_router, prefix="/api/validation", tags=["Validation"])

# Health endpoints for container health monitoring (proxied from gateway)
app.include_router(health_router, prefix="/api/health", tags=["Health"])


@app.get("/health", operation_id="health_check")
async def health():
    return {"status": "ok", "service": SERVICE_NAME, "version": VERSION}


@app.get("/", operation_id="get_root")
async def root():
    return {"service": SERVICE_NAME, "version": VERSION, "docs": "/docs", "health": "/health"}


# KAG API standard: centralized exception handlers from coding-client
register_exception_handlers(app, debug=os.getenv("DEBUG", "false").lower() == "true")

# KAG API standard: OpenAPI security config
configure_openapi(
    app,
    protected_prefixes={"/api/workspaces", "/api/containers", "/api/validation"},
    public_endpoints={"/", "/health", "/docs", "/redoc", "/openapi.json"},
)

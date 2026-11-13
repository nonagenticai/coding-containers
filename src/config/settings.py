"""Settings module for coding-containers.

Subclasses ``coding_client.config.BaseMicroserviceSettings`` for the shared
fail-loud-in-production enforcement (R-14), while keeping the
containers-specific fields, nested models, and database-manager stubs local.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import SettingsConfigDict

from coding_client.config import BaseMicroserviceSettings


class WorkerSettings(BaseModel):
    """Worker-related configuration."""

    internal_worker_port: int = Field(default=8700, description="Internal worker port")


class RedisSettings(BaseModel):
    """Redis connection configuration."""

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: str | None = Field(default=None, description="Redis password")


class _DatabaseManagerStub:
    """Stub database manager that raises NotImplementedError.

    This exists so that code paths calling ``settings.get_database_manager()``
    get a clear error instead of an obscure AttributeError.  Services that
    actually need a database manager should wire up a real implementation via
    dependency injection rather than relying on this stub.
    """

    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError(
            "DatabaseManager is not available in coding-containers. "
            "Inject a real DatabaseManager or use a service that provides one."
        )


class Settings(BaseMicroserviceSettings):
    """Coding-containers service settings.

    Values are read from environment variables.  The ``env_prefix`` is
    ``CODING_CONTAINERS_`` so, for example, ``docker_base_port`` can be
    set via ``CODING_CONTAINERS_DOCKER_BASE_PORT``.

    Redis and worker settings can also be overridden via flat env vars
    (e.g. ``REDIS_HOST``, ``REDIS_PORT``) thanks to the nested model
    ``env_prefix`` configuration.
    """

    model_config = SettingsConfigDict(
        env_prefix="CODING_CONTAINERS_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    docker_base_port: int = Field(default=8700, description="Base port for Docker containers")
    main_api_endpoint: str = Field(
        default="",
        description=(
            "Main API endpoint URL (e.g. coding-gateway). No default — must be "
            "supplied via env (CODING_CONTAINERS_MAIN_API_ENDPOINT) in production."
        ),
    )
    temp_dir: str = Field(default="/tmp", description="Temporary directory path")
    # Per-stream byte budget for the PUBLIC /exec route. Deliberately far below
    # the provisioner's EXEC_HARD_OUTPUT_BYTES ceiling, which internal
    # server-authored commands (repository tree walks, single-file reads) keep
    # using: this bounds what an arbitrary caller-supplied command can return,
    # and 1 MiB of text is already more than any caller reads.
    #
    # Raise it via CODING_CONTAINERS_EXEC_MAX_OUTPUT_BYTES rather than in code —
    # the right value is a property of the deployment, not of the service.
    exec_max_output_bytes: int = Field(
        default=1024 * 1024,
        ge=1024,
        description="Max bytes returned per stream from the public /exec route",
    )
    environment: str = Field(
        default="production",
        description="Deployment environment (production|staging|test|dev)",
    )
    # Secrets API base URL — used to resolve a request's ``secret_config_id``
    # into the per-tenant GIT_TOKEN so PRIVATE repos can be cloned (BYO token).
    # Mirrors coding-github's ``CODING_GITHUB_SECRETS_API_URL``; defaulted to the
    # in-cluster secrets service so private clones work without extra deploy env,
    # overridable via CODING_CONTAINERS_SECRETS_API_URL.
    secrets_api_url: str = Field(
        default="http://secrets.secrets.svc.cluster.local:8543",
        description="Secrets service base URL for resolving secret_config_id → GIT_TOKEN",
    )

    # ------------------------------------------------------------------
    # coding-analysis, addressed DIRECTLY (east-west), not through the gateway.
    #
    # The eager code-graph build used to POST to
    # ``{main_api_endpoint}/api/analysis/analyze`` — i.e. through coding-gateway.
    # coding-gateway is a NORTH-SOUTH edge: its Istio AuthorizationPolicies admit
    # only istio-system/the ingressgateway plus the UI namespaces (fr, n8n, kag,
    # mcp, workers, kag-frontend). ``coding-containers`` is not among them, so
    # every eager build was reset at the mesh (`[Errno 104] Connection reset by
    # peer`) and ``graph_initialized`` never became true for any k8s-provisioned
    # container. Verified in-cluster 2026-08-02 from the real workload identity:
    # coding-containers → coding-gateway = ConnectionReset, while
    # coding-containers → coding-analysis = 200.
    #
    # ``coding-analysis``'s own ``allow-coding-gateway-to-microservices`` policy
    # already admits the ``coding-containers`` namespace, so the direct hop is the
    # one the mesh was configured for. Defaulted in-cluster (like
    # ``secrets_api_url``) so the fix needs no gitops change to take effect.
    # ------------------------------------------------------------------
    coding_analysis_url: str = Field(
        default="http://coding-analysis.coding-analysis.svc.cluster.local:8688",
        validation_alias=AliasChoices(
            "CODING_ANALYSIS_URL",
            "CODING_CONTAINERS_CODING_ANALYSIS_URL",
        ),
        description="coding-analysis base URL for the eager code-graph build (east-west, direct)",
    )

    # ------------------------------------------------------------------
    # coding-github clone-token endpoint (GitHub App-backed clone credential).
    #
    # A project created without a BYO PAT has no GIT_TOKEN anywhere: the
    # secret_config_id resolves to nothing and no system token is set in the
    # cluster. coding-github owns the GitHub App installation, so it is the only
    # service that can (a) mint a short-lived installation token for a repo whose
    # owner has the App installed and (b) authoritatively report the repo's
    # visibility. coding-containers asks it, then either clones with the App
    # token, clones a PUBLIC repo with no credential at all, or fails loudly.
    #
    # ``coding_github_url`` is env-only (no in-code default: R-14 / the
    # config-externalization policy) and REQUIRED in production — a forgotten
    # gitops env var must crash the boot, not surface as a 500 on the first
    # provision. Read from the UNPREFIXED ``CODING_GITHUB_URL`` (the fleet-wide
    # name for this service address, matching how the other coding-* services
    # address each other) with the prefixed form accepted as an alias.
    # ------------------------------------------------------------------
    coding_github_url: str = Field(
        default="",
        validation_alias=AliasChoices(
            "CODING_GITHUB_URL",
            "CODING_CONTAINERS_CODING_GITHUB_URL",
        ),
        description=(
            "coding-github base URL, used for the App-backed clone-token lookup. "
            "No default — must be supplied via env (CODING_GITHUB_URL) in production."
        ),
    )
    # The platform GitHub org(s) — comma-separated, case-insensitive. These
    # owners get coding-github's App token (or a typed 412); any other owner is a
    # customer whose credential must be a BYO PAT or a public-App token bound to
    # the requesting tenant. No owner is ever cloned or pushed with an env PAT.
    platform_github_owners: str = Field(
        default="nonagenticai",
        description=(
            "Comma-separated platform GitHub owners (credential from coding-github's App) "
            "(CODING_CONTAINERS_PLATFORM_GITHUB_OWNERS)."
        ),
    )
    internal_service_token: str = Field(
        default="",
        description=(
            "Shared secret sent as X-Internal-Token on coding-github's internal "
            "clone-token endpoint (CODING_CONTAINERS_INTERNAL_SERVICE_TOKEN)."
        ),
    )

    # ------------------------------------------------------------------
    # Phase C-1 — container lifecycle reaper + admission cap.
    # ALL inert by default: deploying this code changes nothing until the
    # corresponding CODING_CONTAINERS_* env vars are set. Even with the reaper
    # enabled, reaper_dry_run defaults to True so the first sweep only logs.
    # ------------------------------------------------------------------
    reaper_enabled: bool = Field(
        default=False, description="Run the background job-Deployment reaper"
    )
    reaper_interval_seconds: int = Field(default=300, description="Seconds between reaper sweeps")
    reaper_ttl_seconds: int = Field(
        default=86400,
        description="Age (s) after which a labelled job Deployment becomes reapable",
    )
    reaper_dry_run: bool = Field(
        default=True, description="Log intended reaps without deleting anything"
    )
    reaper_forget_registry_ghosts: bool = Field(
        default=True,
        description=(
            "Also HDEL registry rows whose Deployment is confirmed gone "
            "(honours reaper_dry_run; fail-closed on any k8s/Redis error)"
        ),
    )
    reaper_registry_ghost_grace_seconds: int = Field(
        default=3600,
        description="Seconds a row must stay confirmed Deployment-less before it is removed",
    )
    git_token_refresh_seconds: int = Field(
        default=2400,
        description=(
            "Re-mint a job's git-token Secret once its credential is older than this. "
            "GitHub App installation tokens expire ~50 min after mint, and the git-clone "
            "init container can run at any later moment (sandbox recreation, eviction, "
            "node reboot), which would otherwise fail with 'Invalid username or token'. "
            "Runs on the reaper loop; 0 disables the refresh."
        ),
    )
    reap_exclude_names: str = Field(
        default="", description="Extra Deployment names never to reap (comma-separated)"
    )
    reap_legacy_unlabelled: bool = Field(
        default=False,
        description="Also reap pre-label legacy Deployments (uses reap_legacy_ttl_seconds)",
    )
    reap_legacy_ttl_seconds: int = Field(
        default=604800, description="TTL (s) for legacy unlabelled Deployments"
    )
    max_inflight_jobs: int = Field(
        default=0,
        description="Max concurrent job Deployments before admission rejects (0=unlimited)",
    )
    admission_reject_mode: str = Field(
        default="reject", description="Over-cap behaviour: 'reject' (429) or 'wait'"
    )
    admission_wait_timeout_seconds: int = Field(
        default=0, description="Max seconds to wait under cap when admission_reject_mode='wait'"
    )

    # ------------------------------------------------------------------
    # Async workspace exec (execInWorkspaceAsync) — durable job store + callback.
    #
    # The job store is this service's OWN Postgres database (``coding_containers``,
    # one service / one DB / one user), reached through PgBouncer in transaction
    # mode exactly like coding-analysis. Unprefixed names: they are the fleet-wide
    # spelling and what the SOPS secret carries. With neither set the async route
    # answers 503 and no reaper runs; the sync /exec route is unaffected.
    # ------------------------------------------------------------------
    pgbouncer_url: str = Field(
        default="",
        validation_alias=AliasChoices("PGBOUNCER_URL", "CODING_CONTAINERS_PGBOUNCER_URL"),
        description="Pooled DSN of the coding_containers DB (preferred at runtime)",
    )
    database_url: str = Field(
        default="",
        validation_alias=AliasChoices("DATABASE_URL", "CODING_CONTAINERS_DATABASE_URL"),
        description="Direct DSN of the coding_containers DB (fallback; alembic reads it)",
    )
    # Comma-separated URL prefixes a job's ``callback_url`` must sit under (e.g. an
    # n8n Wait-node resume base ``http://host:5678/webhook-waiting/``). EMPTY =
    # every callback_url is refused (fail closed): the control pod POSTs it from
    # inside the cluster, so an open value is an SSRF primitive. Same semantics as
    # coding-analysis BUG_HUNT_CALLBACK_URL_PREFIXES.
    exec_callback_url_prefixes: str = Field(
        default="",
        validation_alias=AliasChoices(
            "EXEC_CALLBACK_URL_PREFIXES", "CODING_CONTAINERS_EXEC_CALLBACK_URL_PREFIXES"
        ),
        description="Allowed callback_url prefixes for async exec jobs (empty = callbacks off)",
    )
    # A running job holds a lease it renews every third of this; a lease that
    # lapses (the owning pod died: rollout, OOM, node loss) is reaped to
    # failed/orphaned and its callback sent, so no n8n cell stays parked.
    exec_job_lease_seconds: int = Field(default=60, ge=15, le=600)
    exec_job_reaper_interval_seconds: int = Field(default=20, ge=1, le=600)
    # Per-replica concurrency. Each running job holds one thread of the default
    # executor for its whole duration (the k8s exec stream is blocking), so an
    # unbounded count would starve every other ``to_thread`` caller in the pod.
    # Jobs beyond this wait as ``queued`` (lease still renewed).
    exec_async_max_concurrent: int = Field(default=8, ge=1, le=64)

    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)

    _SERVICE_NAME: ClassVar[str] = "coding-containers"
    _REQUIRED_IN_PRODUCTION: ClassVar[tuple[str, ...]] = (
        "main_api_endpoint",
        # Without it, every no-PAT provision silently loses the App-token ladder
        # rung and dies at the git-token Secret. Fail at boot, not at provision.
        "coding_github_url",
    )

    # ------------------------------------------------------------------
    # Methods that mirror the coding_client.config.Settings interface
    # ------------------------------------------------------------------

    def get_database_manager(self) -> Any:
        """Return a database manager instance.

        Raises ``NotImplementedError`` -- coding-containers does not ship
        its own database manager.  Callers should inject one or use a
        service that provides one.
        """
        return _DatabaseManagerStub()

    @property
    def exec_callback_prefixes(self) -> tuple[str, ...]:
        """The allowed async-exec callback URL prefixes; empty means callbacks are off."""
        return tuple(p.strip() for p in self.exec_callback_url_prefixes.split(",") if p.strip())

    @property
    def exec_job_dsn(self) -> str:
        """The job-store DSN (PgBouncer preferred), or "" when no store is configured."""
        return self.pgbouncer_url or self.database_url

    def get_code_graph_db_path(self, graph_id: str) -> str:
        """Return the expected database file path for *graph_id*.

        This is a best-effort stub that mirrors the convention used by the
        monolith.  Override ``CODING_CONTAINERS_TEMP_DIR`` or
        ``CODE_GRAPH_DB_DIR`` to change the base directory.
        """
        base_dir = os.environ.get("CODE_GRAPH_DB_DIR", os.path.join(self.temp_dir, "code_graphs"))
        return str(Path(base_dir) / f"{graph_id}.db")


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Return the global ``Settings`` singleton, creating it on first call."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def reload_settings() -> Settings:
    """Force-recreate the ``Settings`` singleton from the environment."""
    global _settings_instance
    _settings_instance = Settings()
    return _settings_instance

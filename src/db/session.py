"""Single async psycopg connection pool for the ``coding_containers`` database.

Same shape as coding-analysis ``db/session.py``: raw psycopg, one
``AsyncConnectionPool`` per process, opened lazily.

PgBouncer transaction-mode safety:
- ``prepare_threshold = None`` disables server-side prepared statements, which
  break under transaction pooling.
- Every statement runs in its own short transaction; nothing session-level is
  ever ``SET`` (it would leak to the next PgBouncer client).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from src.config.settings import get_settings

_pool: AsyncConnectionPool | None = None
_pool_lock = asyncio.Lock()


def normalize_dsn(url: str) -> str:
    """Return a libpq DSN psycopg accepts (strip any SQLAlchemy ``+driver``)."""
    if url.startswith("postgresql+"):
        _scheme, rest = url.split("://", 1)
        return "postgresql://" + rest
    return url


def job_store_configured() -> bool:
    """True when a job-store DSN is configured (else the async exec route is off)."""
    return bool(get_settings().exec_job_dsn)


async def _configure_connection(conn: AsyncConnection) -> None:
    """Disable prepared statements so the pool is PgBouncer-txn-mode safe."""
    conn.prepare_threshold = None


async def get_pool() -> AsyncConnectionPool:
    """Return the process-wide pool, opening it lazily on first use."""
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is None:
            dsn = get_settings().exec_job_dsn
            if not dsn:
                raise RuntimeError(
                    "No job-store database configured: set PGBOUNCER_URL or DATABASE_URL "
                    "(coding-containers owns the 'coding_containers' PostgreSQL database)."
                )
            pool = AsyncConnectionPool(
                conninfo=normalize_dsn(dsn),
                open=False,
                configure=_configure_connection,
                # Validate before handing out: after a PgBouncer roll a pooled socket
                # points at a pod that no longer exists (coding-analysis measured a
                # 3h outage from exactly this on 2026-09-01).
                check=AsyncConnectionPool.check_connection,
                min_size=1,
                max_size=5,
            )
            await pool.open(wait=True)
            _pool = pool
    return _pool


@asynccontextmanager
async def get_connection() -> AsyncGenerator[AsyncConnection]:
    """Yield a pooled connection; the block is one transaction (commit on exit)."""
    pool = await get_pool()
    async with pool.connection() as conn:
        yield conn


#: Bound on ``close_pool`` (psycopg_pool's own default, stated so the SIGTERM
#: budget in ``src.services.exec_jobs.runner`` can add it up).
POOL_CLOSE_TIMEOUT_S = 5.0


async def close_pool() -> None:
    """Close the process-wide pool (call on application shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close(timeout=POOL_CLOSE_TIMEOUT_S)
        _pool = None

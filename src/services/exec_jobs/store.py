"""PostgreSQL persistence for async workspace exec jobs (table ``exec_jobs``).

Every state change is ONE conditional statement, so concurrent replicas cannot
both win a transition:

* a terminal transition (``finish`` / ``reap_expired`` / ``fail_owned_active``)
  matches only ``status IN ('queued', 'running')`` — under READ COMMITTED the
  loser re-evaluates the WHERE after the winner commits and matches nothing;
* the same statement moves ``callback_state`` to ``sending`` under
  ``callback_owner`` = the winner, so exactly one replica POSTs the callback;
* ``FOR UPDATE SKIP LOCKED`` keeps two reapers from blocking on each other;
* a retryable callback is parked as ``retrying`` with ``callback_next_attempt_at``
  and re-claimed (back to ``sending``, one winner) by any replica's reaper once
  due — durable across pod restarts, retried for as long as it is retryable;
* at most one job HOLDING each ``(tenant_id, callback_url)`` (partial unique
  index ``exec_jobs_active_callback_uidx``): a job holds its key while it is
  queued/running AND while its callback is still ``sending``/``retrying`` — i.e.
  until the resume URL has had its final answer. The terminal transition moves
  ``callback_state`` to ``sending`` in the SAME statement, so the key is never
  free between "job ended" and "callback settled"; a client retry of the async
  POST in that window gets the existing job, never a second run.

All times are the database's ``now()`` — never a replica's clock.
"""

from __future__ import annotations

from typing import Any, LiteralString

from psycopg.rows import dict_row

from src.db.session import get_connection

ACTIVE = ("queued", "running")
#: Callback states in which a terminal job still holds its ``(tenant_id, callback_url)``.
CALLBACK_PENDING = ("sending", "retrying")

# The idempotency-key predicate. MUST equal the WHERE of the partial unique index
# exec_jobs_active_callback_uidx (migration 0002) — ON CONFLICT infers the index
# from it — and is also the read-back filter, so a row that has just released
# the key is never returned as its holder.
_HOLDS_KEY = """
    callback_url IS NOT NULL
    AND (status IN ('queued', 'running') OR callback_state IN ('sending', 'retrying'))
"""

_COLUMNS = """
    id::text AS job_id, tenant_id, container_name, command_summary, workdir,
    timeout_seconds, status, exit_code, stdout, stderr, stdout_truncated,
    stderr_truncated, bytes_omitted, timed_out, duration_seconds, error, owner,
    lease_expires_at, callback_url, callback_state, callback_owner,
    callback_lease_expires_at, callback_attempts, callback_next_attempt_at, created_at, started_at,
    completed_at, updated_at, detached_pod, detached_dir, detached_at
"""

# The callback hand-off written by every terminal transition: the winner of the
# transition becomes the only sender.
_CALLBACK_HANDOFF = """
    callback_state = CASE WHEN callback_url IS NULL THEN 'none' ELSE 'sending' END,
    callback_owner = CASE WHEN callback_url IS NULL THEN NULL ELSE %(owner)s END,
    callback_lease_expires_at = CASE WHEN callback_url IS NULL THEN NULL
        ELSE now() + (%(callback_lease)s * interval '1 second') END
"""


class ExecJobStore:
    """Raw-psycopg store over ``exec_jobs`` (one short transaction per call)."""

    async def _one(self, sql: LiteralString, params: dict[str, Any]) -> dict[str, Any] | None:
        async with get_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            return await cur.fetchone()

    async def _all(self, sql: LiteralString, params: dict[str, Any]) -> list[dict[str, Any]]:
        async with get_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            return list(await cur.fetchall())

    async def create_job(
        self,
        *,
        job_id: str,
        tenant_id: str,
        container_name: str,
        command_summary: str,
        workdir: str,
        timeout_seconds: int,
        callback_url: str | None,
        owner: str,
        lease_seconds: int,
    ) -> tuple[dict[str, Any], bool]:
        """Insert a ``queued`` job already leased to ``owner``; ``(row, created)``.

        Leased from birth: a pod that dies between this insert and the runner
        starting must still be reaped, or the caller's cell is parked forever.

        Idempotent on ``(tenant_id, callback_url)`` while a job holds it (see
        ``_HOLDS_KEY``): a client retrying the POST (after a 503 or a lost 202)
        gets the holder back with ``created = False`` instead of a second run that
        would race the first for the same resume URL. The caller decides what a
        holder whose callback is ``sending`` means (outcome unknown: retry later).
        Enforced by the partial unique index ``exec_jobs_active_callback_uidx``.

        The INSERT and the read-back are separate statements, so the holder can
        release the key in between (callback delivered/refused). The read-back
        filters on the same predicate, so it then finds nothing and the INSERT is
        retried; it never returns a row that no longer holds the key.
        """
        params = {
            "id": job_id,
            "tenant_id": tenant_id,
            "container_name": container_name,
            "command_summary": command_summary,
            "workdir": workdir,
            "timeout": timeout_seconds,
            "callback_url": callback_url,
            "owner": owner,
            "lease": lease_seconds,
        }
        # Each pass that finds no holder means one holder released the key in
        # between; several passes cover a URL that is released and re-taken.
        for _ in range(5):
            row = await self._one(
                f"""
                INSERT INTO exec_jobs (id, tenant_id, container_name, command_summary, workdir,
                                       timeout_seconds, status, callback_url, owner,
                                       lease_expires_at)
                VALUES (%(id)s, %(tenant_id)s, %(container_name)s, %(command_summary)s,
                        %(workdir)s, %(timeout)s, 'queued', %(callback_url)s, %(owner)s,
                        now() + (%(lease)s * interval '1 second'))
                ON CONFLICT (tenant_id, callback_url) WHERE {_HOLDS_KEY} DO NOTHING
                RETURNING {_COLUMNS}
                """,
                params,
            )
            if row is not None:
                return row, True
            existing = await self._one(
                f"""
                SELECT {_COLUMNS} FROM exec_jobs
                 WHERE tenant_id = %(tenant_id)s AND callback_url = %(callback_url)s
                   AND {_HOLDS_KEY}
                """,
                params,
            )
            if existing is not None:
                return existing, False
        raise RuntimeError("exec job insert kept conflicting with a releasing holder")

    async def mark_running(self, *, job_id: str, owner: str, lease_seconds: int) -> bool:
        """``queued`` -> ``running`` for the owner; False if the job was reaped meanwhile."""
        row = await self._one(
            """
            UPDATE exec_jobs
               SET status = 'running', started_at = now(), updated_at = now(),
                   lease_expires_at = now() + (%(lease)s * interval '1 second')
             WHERE id = %(id)s AND owner = %(owner)s AND status = 'queued'
            RETURNING id
            """,
            {"id": job_id, "owner": owner, "lease": lease_seconds},
        )
        return row is not None

    async def renew_lease(self, *, job_id: str, owner: str, lease_seconds: int) -> bool:
        """Extend the owner's lease; False means the job is no longer ours to finish."""
        row = await self._one(
            """
            UPDATE exec_jobs
               SET lease_expires_at = now() + (%(lease)s * interval '1 second'),
                   updated_at = now()
             WHERE id = %(id)s AND owner = %(owner)s AND status IN ('queued', 'running')
            RETURNING id
            """,
            {"id": job_id, "owner": owner, "lease": lease_seconds},
        )
        return row is not None

    async def finish(
        self,
        *,
        job_id: str,
        owner: str,
        status: str,
        callback_lease_seconds: int,
        exit_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
        stdout_truncated: bool = False,
        stderr_truncated: bool = False,
        bytes_omitted: int = 0,
        timed_out: bool = False,
        duration_seconds: float | None = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        """Terminal transition by the owner. Returns the row only to the winner.

        ``None`` means another replica (the reaper) already finished this job —
        the caller must then NOT send a callback; the winner does.
        """
        if status not in ("completed", "failed"):
            raise ValueError(f"not a terminal status: {status!r}")
        return await self._one(
            f"""
            UPDATE exec_jobs
               SET status = %(status)s, exit_code = %(exit_code)s, stdout = %(stdout)s,
                   stderr = %(stderr)s, stdout_truncated = %(stdout_truncated)s,
                   stderr_truncated = %(stderr_truncated)s,
                   bytes_omitted = %(bytes_omitted)s, timed_out = %(timed_out)s,
                   duration_seconds = %(duration)s, error = %(error)s,
                   completed_at = now(), updated_at = now(), lease_expires_at = NULL,
                   {_CALLBACK_HANDOFF}
             WHERE id = %(id)s AND owner = %(owner)s AND status IN ('queued', 'running')
            RETURNING {_COLUMNS}
            """,
            {
                "id": job_id,
                "owner": owner,
                "status": status,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "bytes_omitted": bytes_omitted,
                "timed_out": timed_out,
                "duration": duration_seconds,
                "error": error,
                "callback_lease": callback_lease_seconds,
            },
        )

    async def fail_owned_active(
        self, *, owner: str, error: str, callback_lease_seconds: int
    ) -> list[dict[str, Any]]:
        """Fail every still-active job this process owns (graceful shutdown).

        A DETACHED job is not failed: its command keeps running in the workspace
        pod. ``release_owned_detached`` hands it to the next reaper instead.
        """
        return await self._all(
            f"""
            UPDATE exec_jobs
               SET status = 'failed', error = %(error)s, completed_at = now(),
                   updated_at = now(), lease_expires_at = NULL,
                   {_CALLBACK_HANDOFF}
             WHERE owner = %(owner)s AND status IN ('queued', 'running')
               AND detached_pod IS NULL
            RETURNING {_COLUMNS}
            """,
            {"owner": owner, "error": error, "callback_lease": callback_lease_seconds},
        )

    async def reap_expired(
        self, *, reaper: str, error: str, callback_lease_seconds: int, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Fail active jobs whose lease lapsed (owner dead). Rows go to the reaper.

        Never re-run: a workspace command may have side effects, so an orphan is
        reported, not retried. A DETACHED job is not an orphan (its command
        outlives the replica): ``adopt_detached`` re-attaches it instead.
        """
        return await self._all(
            f"""
            WITH victims AS (
                SELECT id FROM exec_jobs
                 WHERE status IN ('queued', 'running') AND lease_expires_at < now()
                   AND detached_pod IS NULL
                 ORDER BY lease_expires_at
                 LIMIT %(limit)s
                 FOR UPDATE SKIP LOCKED
            )
            UPDATE exec_jobs j
               SET status = 'failed', error = %(error)s, completed_at = now(),
                   updated_at = now(), lease_expires_at = NULL,
                   {_CALLBACK_HANDOFF}
              FROM victims v
             WHERE j.id = v.id AND j.status IN ('queued', 'running')
            RETURNING {_COLUMNS.replace("id::text", "j.id::text")}
            """,
            {
                "owner": reaper,
                "error": error,
                "callback_lease": callback_lease_seconds,
                "limit": limit,
            },
        )

    async def record_detached(
        self, *, job_id: str, owner: str, pod: str, job_dir: str
    ) -> dict[str, Any] | None:
        """Record the detached handle on the owner's running job, BEFORE launching.

        ``None`` means the job is no longer ours to launch (reaped or finished).
        """
        return await self._one(
            f"""
            UPDATE exec_jobs
               SET detached_pod = %(pod)s, detached_dir = %(dir)s, detached_at = now(),
                   updated_at = now()
             WHERE id = %(id)s AND owner = %(owner)s AND status = 'running'
            RETURNING {_COLUMNS}
            """,
            {"id": job_id, "owner": owner, "pod": pod, "dir": job_dir},
        )

    async def adopt_detached(
        self, *, owner: str, lease_seconds: int, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Take over running detached jobs whose owner's lease lapsed (one winner each).

        The command is still running (or has finished) in the workspace pod; the
        adopter re-attaches to its status files and completes the row. The
        ``owner`` change is what makes the dead owner's late ``finish`` lose.
        Each row also carries ``previous_owner`` (the owner whose lease lapsed).
        """
        return await self._all(
            f"""
            WITH lapsed AS (
                SELECT id, owner AS previous_owner FROM exec_jobs
                 WHERE status = 'running' AND detached_pod IS NOT NULL
                   AND lease_expires_at < now()
                 ORDER BY lease_expires_at
                 LIMIT %(limit)s
                 FOR UPDATE SKIP LOCKED
            )
            UPDATE exec_jobs j
               SET owner = %(owner)s, updated_at = now(),
                   lease_expires_at = now() + (%(lease)s * interval '1 second')
              FROM lapsed l
             WHERE j.id = l.id AND j.status = 'running' AND j.lease_expires_at < now()
            RETURNING {_COLUMNS.replace("id::text", "j.id::text")}, l.previous_owner
            """,
            {"owner": owner, "lease": lease_seconds, "limit": limit},
        )

    async def release_owned_detached(self, *, owner: str) -> int:
        """Graceful shutdown: lapse the lease of this process's detached jobs NOW,
        so a surviving replica adopts them on its next sweep instead of one full
        lease later. The rows stay ``running``; nothing is failed or called back."""
        rows = await self._all(
            """
            UPDATE exec_jobs
               SET lease_expires_at = now() - interval '1 second', updated_at = now()
             WHERE owner = %(owner)s AND status = 'running' AND detached_pod IS NOT NULL
            RETURNING id
            """,
            {"owner": owner},
        )
        return len(rows)

    async def claim_due_callbacks(
        self, *, owner: str, callback_lease_seconds: int, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Claim callbacks to (re)send: one winner per row across replicas.

        * ``sending`` with a lapsed lease — the sender died mid-send;
        * ``retrying`` whose ``callback_next_attempt_at`` is due — the last round
          ended retryable (n8n not parked yet, n8n/tailnet down, 5xx).
        """
        return await self._all(
            f"""
            WITH due AS (
                SELECT id FROM exec_jobs
                 WHERE (callback_state = 'sending' AND callback_lease_expires_at < now())
                    OR (callback_state = 'retrying' AND callback_next_attempt_at <= now())
                 ORDER BY coalesce(callback_lease_expires_at, callback_next_attempt_at)
                 LIMIT %(limit)s
                 FOR UPDATE SKIP LOCKED
            )
            UPDATE exec_jobs j
               SET callback_state = 'sending', callback_owner = %(owner)s,
                   callback_next_attempt_at = NULL, updated_at = now(),
                   callback_lease_expires_at =
                       now() + (%(callback_lease)s * interval '1 second')
              FROM due d
             WHERE j.id = d.id
               AND ((j.callback_state = 'sending' AND j.callback_lease_expires_at < now())
                    OR (j.callback_state = 'retrying' AND j.callback_next_attempt_at <= now()))
            RETURNING {_COLUMNS.replace("id::text", "j.id::text")}
            """,
            {"owner": owner, "callback_lease": callback_lease_seconds, "limit": limit},
        )

    async def record_callback(
        self,
        *,
        job_id: str,
        owner: str,
        outcome: str,
        attempts: int = 1,
        retry_in_seconds: int | None = None,
    ) -> bool:
        """Persist a round's outcome; only the current callback owner may.

        ``outcome == 'retrying'`` releases the claim and schedules the next round
        ``retry_in_seconds`` from the database's ``now()``.
        """
        if outcome == "retrying" and retry_in_seconds is None:
            raise ValueError("a retrying outcome needs retry_in_seconds")
        row = await self._one(
            """
            UPDATE exec_jobs
               SET callback_state = %(outcome)s, callback_lease_expires_at = NULL,
                   callback_owner = CASE WHEN %(outcome)s = 'retrying' THEN NULL
                                         ELSE callback_owner END,
                   callback_next_attempt_at = CASE WHEN %(outcome)s = 'retrying'
                       THEN now() + (%(retry_in)s * interval '1 second') ELSE NULL END,
                   callback_attempts = callback_attempts + %(attempts)s, updated_at = now()
             WHERE id = %(id)s AND callback_state = 'sending' AND callback_owner = %(owner)s
            RETURNING id
            """,
            {
                "id": job_id,
                "owner": owner,
                "outcome": outcome,
                "attempts": attempts,
                "retry_in": retry_in_seconds or 0,
            },
        )
        return row is not None

    async def get_job(
        self, *, job_id: str, tenant_id: str, container_name: str
    ) -> dict[str, Any] | None:
        """Tenant- and workspace-scoped read: a foreign job is indistinguishable from none."""
        return await self._one(
            f"""
            SELECT {_COLUMNS} FROM exec_jobs
             WHERE id = %(id)s AND tenant_id = %(tenant_id)s
               AND container_name = %(container_name)s
            """,
            {"id": job_id, "tenant_id": tenant_id, "container_name": container_name},
        )

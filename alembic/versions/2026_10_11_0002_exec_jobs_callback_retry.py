"""exec_jobs: durable callback retries + callback_url idempotency.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-11

A separate revision rather than an edit of 0001: 0001 never shipped, but it has
been applied to local/dev databases, and alembic would silently skip an edited
0001 there (its revision is already recorded), leaving those schemas without
these columns.

* ``callback_state = 'retrying'`` + ``callback_next_attempt_at``: a round that
  ends retryable (n8n 409 "running already" because the Wait node has not parked
  yet, n8n/tailnet down, 5xx) is parked here and re-claimed by any replica's
  reaper once due — retried for as long as it stays retryable. ``undeliverable``
  (the old give-up state) is gone; existing rows are revived as due retries.
* ``exec_jobs_active_callback_uidx``: at most one job per
  ``(tenant_id, callback_url)`` that is queued/running OR whose callback is still
  ``sending``/``retrying`` (the key is held until the resume URL has its final
  answer), so a client retry of the async POST returns the existing job instead
  of starting a second one for the same resume URL.
* A 0001 database (no such index) may already hold several such jobs for one
  key. Before the index is built they are resolved deterministically: the
  oldest (``created_at``, then ``id``) stays the holder; every newer one is
  failed (if still active) with error ``duplicate_callback_url`` and its
  callback set to ``refused`` — it must not compete with the holder for the one
  parked cell, so it is never sent. Its owner's later ``finish`` / lease renewal
  matches nothing (they require an active status), so the run is abandoned.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# A job holds its (tenant_id, callback_url) idempotency key while it runs AND
# until its callback has a final answer. Must equal store._HOLDS_KEY.
HOLDS_KEY = (
    "callback_url IS NOT NULL "
    "AND (status IN ('queued', 'running') OR callback_state IN ('sending', 'retrying'))"
)


def upgrade() -> None:
    op.execute(
        "ALTER TABLE exec_jobs ADD COLUMN IF NOT EXISTS callback_next_attempt_at timestamptz"
    )
    op.execute("ALTER TABLE exec_jobs DROP CONSTRAINT IF EXISTS exec_jobs_callback_state_check")
    op.execute(
        """
        UPDATE exec_jobs
           SET callback_state = 'retrying', callback_owner = NULL,
               callback_lease_expires_at = NULL, callback_next_attempt_at = now()
         WHERE callback_state = 'undeliverable'
        """
    )
    op.execute(
        """
        ALTER TABLE exec_jobs ADD CONSTRAINT exec_jobs_callback_state_check
            CHECK (callback_state IN
                   ('none', 'sending', 'retrying', 'delivered', 'refused', 'disallowed'))
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS exec_jobs_callback_retry_idx "
        "ON exec_jobs (callback_next_attempt_at) WHERE callback_state = 'retrying'"
    )
    op.execute(
        f"""
        WITH ranked AS (
            SELECT id, row_number() OVER (PARTITION BY tenant_id, callback_url
                                          ORDER BY created_at, id) AS rn
              FROM exec_jobs
             WHERE {HOLDS_KEY}
        )
        UPDATE exec_jobs j
           SET status = CASE WHEN j.status IN ('queued', 'running') THEN 'failed'
                             ELSE j.status END,
               error = CASE WHEN j.status IN ('queued', 'running')
                            THEN 'duplicate_callback_url' ELSE j.error END,
               completed_at = coalesce(j.completed_at, now()),
               lease_expires_at = NULL,
               callback_state = 'refused', callback_owner = NULL,
               callback_lease_expires_at = NULL, callback_next_attempt_at = NULL,
               updated_at = now()
          FROM ranked r
         WHERE j.id = r.id AND r.rn > 1
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS exec_jobs_active_callback_uidx "
        f"ON exec_jobs (tenant_id, callback_url) WHERE {HOLDS_KEY}"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS exec_jobs_active_callback_uidx")
    op.execute("DROP INDEX IF EXISTS exec_jobs_callback_retry_idx")
    op.execute("ALTER TABLE exec_jobs DROP CONSTRAINT IF EXISTS exec_jobs_callback_state_check")
    op.execute(
        """
        UPDATE exec_jobs SET callback_state = 'undeliverable', callback_next_attempt_at = NULL
         WHERE callback_state = 'retrying'
        """
    )
    op.execute(
        """
        ALTER TABLE exec_jobs ADD CONSTRAINT exec_jobs_callback_state_check
            CHECK (callback_state IN
                   ('none', 'sending', 'delivered', 'refused', 'undeliverable', 'disallowed'))
        """
    )
    op.execute("ALTER TABLE exec_jobs DROP COLUMN IF EXISTS callback_next_attempt_at")

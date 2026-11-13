"""exec_jobs: durable async workspace-exec jobs with an owner lease.

Revision ID: 0001
Revises:
Create Date: 2026-10-11

Lease model copied from coding-analysis ``bug_hunts``: the replica running a job
renews ``lease_expires_at``; a job whose lease lapses (its pod died) is reaped to
``failed`` with error ``orphaned`` and its callback is sent. Unlike a bug hunt it
is never re-run: a workspace command can have side effects.

``callback_state`` makes the callback single-sender across replicas: the
transaction that moves a job to a terminal status also moves it to ``sending``
under ``callback_owner``, so only that winner POSTs. A sender that dies mid-send
leaves ``sending`` with an expired ``callback_lease_expires_at``, which the
reaper re-claims (at-least-once only in that crash window).

The argv is deliberately NOT stored, only ``command_summary``
(``log_safety.summarize_command``): a caller-supplied argv can carry a credential.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS exec_jobs (
            id                        uuid PRIMARY KEY,
            tenant_id                 text NOT NULL,
            container_name            text NOT NULL,
            command_summary           text NOT NULL,
            workdir                   text NOT NULL,
            timeout_seconds           integer NOT NULL CHECK (timeout_seconds > 0),
            status                    text NOT NULL
                CHECK (status IN ('queued', 'running', 'completed', 'failed')),
            exit_code                 integer,
            stdout                    text NOT NULL DEFAULT '',
            stderr                    text NOT NULL DEFAULT '',
            stdout_truncated          boolean NOT NULL DEFAULT false,
            stderr_truncated          boolean NOT NULL DEFAULT false,
            bytes_omitted             bigint NOT NULL DEFAULT 0,
            timed_out                 boolean NOT NULL DEFAULT false,
            duration_seconds          double precision,
            error                     text,
            owner                     text NOT NULL,
            lease_expires_at          timestamptz,
            callback_url              text,
            callback_state            text NOT NULL DEFAULT 'none'
                CHECK (callback_state IN
                       ('none', 'sending', 'delivered', 'refused', 'undeliverable', 'disallowed')),
            callback_owner            text,
            callback_lease_expires_at timestamptz,
            callback_attempts         integer NOT NULL DEFAULT 0,
            created_at                timestamptz NOT NULL DEFAULT now(),
            started_at                timestamptz,
            completed_at              timestamptz,
            updated_at                timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS exec_jobs_active_lease_idx "
        "ON exec_jobs (lease_expires_at) WHERE status IN ('queued', 'running')"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS exec_jobs_callback_sending_idx "
        "ON exec_jobs (callback_lease_expires_at) WHERE callback_state = 'sending'"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS exec_jobs_tenant_container_idx "
        "ON exec_jobs (tenant_id, container_name, created_at)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS exec_jobs")

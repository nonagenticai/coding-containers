"""exec_jobs: detached handle, so a job survives the replica that started it.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-12

A Kubernetes workspace command now runs detached in the workspace pod
(``src/services/exec_jobs/detached.py``). The row records where:

* ``detached_pod`` — the workspace pod it was launched in (a replacement pod
  has a fresh ``/tmp`` and none of the job's status files);
* ``detached_dir`` — the job directory in that pod;
* ``detached_at`` — when the handle was recorded (BEFORE launching, so a
  missing directory is only a verdict once the handle is old enough).

A running row with a handle whose lease lapsed is ADOPTED by a surviving replica
(it re-attaches and completes the row) instead of being failed as ``orphaned``.
``exec_jobs_detached_lease_idx`` serves that adoption sweep.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE exec_jobs ADD COLUMN IF NOT EXISTS detached_pod text")
    op.execute("ALTER TABLE exec_jobs ADD COLUMN IF NOT EXISTS detached_dir text")
    op.execute("ALTER TABLE exec_jobs ADD COLUMN IF NOT EXISTS detached_at timestamptz")
    op.execute(
        "CREATE INDEX IF NOT EXISTS exec_jobs_detached_lease_idx ON exec_jobs (lease_expires_at) "
        "WHERE status = 'running' AND detached_pod IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS exec_jobs_detached_lease_idx")
    op.execute("ALTER TABLE exec_jobs DROP COLUMN IF EXISTS detached_at")
    op.execute("ALTER TABLE exec_jobs DROP COLUMN IF EXISTS detached_dir")
    op.execute("ALTER TABLE exec_jobs DROP COLUMN IF EXISTS detached_pod")

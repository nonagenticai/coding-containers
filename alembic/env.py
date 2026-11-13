"""Alembic environment for coding-containers' ``coding_containers`` database.

Raw-SQL migrations (no ORM metadata): the job store talks to Postgres through
psycopg directly, like coding-analysis' report stores.
"""

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None

# DATABASE_URL overrides alembic.ini. In a deployment the migrate initContainer
# points it at the connection pooler (direct database egress is blocked from
# pods); NullPool + one short DDL transaction is transaction-pooling safe.
_db_url = os.getenv("DATABASE_URL")
if _db_url:
    # Fleet standard is psycopg v3; SQLAlchemy would pick psycopg2 for a bare
    # postgresql:// URL, and psycopg2 is not installed.
    if _db_url.startswith("postgresql://"):
        _db_url = _db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    # Escape % for configparser interpolation (e.g. %40 in passwords).
    config.set_main_option("sqlalchemy.url", _db_url.replace("%", "%%"))


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

# File role: Alembic environment configuration that binds migrations to app settings/metadata.
# Connects to: sqlalchemy, app.core.config, app.db.base.
# Key symbols/vars: config, target_metadata, run_migrations_offline, run_migrations_online.
from __future__ import annotations

import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# ------------------------------------------------------------------
# Make sure "app" is importable when running alembic from /backend
# ------------------------------------------------------------------
sys.path.append(".")

# ------------------------------------------------------------------
# Alembic Config
# ------------------------------------------------------------------
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ------------------------------------------------------------------
# Import app settings + Base + models
# ------------------------------------------------------------------
from app.core.config import settings
from app.db.base import Base

# IMPORTANT: import models so Alembic can detect tables
from app.db.models.user import User  # noqa
from app.db.models.trip import Trip  # noqa
from app.db.models.driving_event import DrivingEvent  # noqa

# Metadata for autogenerate
target_metadata = Base.metadata

# Force Alembic to use our real DB URL (prevents "driver://" bug)
config.set_main_option("sqlalchemy.url", settings.database_url)

# ------------------------------------------------------------------
# Offline migrations
# ------------------------------------------------------------------
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()

# ------------------------------------------------------------------
# Online migrations
# ------------------------------------------------------------------
def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()

# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

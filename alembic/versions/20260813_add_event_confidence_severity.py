"""add per-event confidence/severity/duration to driving_events

Phase 10 (hackathon): separates Event / Confidence / Severity concepts and
carries event duration for the replay timeline.

Revision ID: 20260813_add_event_confidence_severity
Revises: 20260723_add_sensor_sample_altitude
Create Date: 2026-08-13
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260813_add_event_confidence_severity"
down_revision: Union[str, Sequence[str], None] = "20260723_add_sensor_sample_altitude"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Dialect-safe and idempotent: SQLite does not support `ADD COLUMN IF NOT
    # EXISTS` (Postgres-only), and the app's init_db self-heal may already have
    # added these columns. Check the live schema before adding each one.
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("driving_events")}
    for column_name in ("confidence", "severity", "duration_s"):
        if column_name not in columns:
            op.add_column("driving_events", sa.Column(column_name, sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("driving_events", "duration_s")
    op.drop_column("driving_events", "severity")
    op.drop_column("driving_events", "confidence")

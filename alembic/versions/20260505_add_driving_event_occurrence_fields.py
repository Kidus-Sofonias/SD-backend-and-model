"""add driving event occurrence fields

Revision ID: 20260505_add_driving_event_occurrence_fields
Revises: 20260328_add_trip_review_fields
Create Date: 2026-05-05 13:10:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260505_add_driving_event_occurrence_fields"
down_revision = "20260328_add_trip_review_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("driving_events", sa.Column("occurred_at", sa.DateTime(), nullable=True))
    op.add_column("driving_events", sa.Column("lat", sa.Float(), nullable=True))
    op.add_column("driving_events", sa.Column("lon", sa.Float(), nullable=True))
    op.create_index("ix_driving_events_occurred_at", "driving_events", ["occurred_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_driving_events_occurred_at", table_name="driving_events")
    op.drop_column("driving_events", "lon")
    op.drop_column("driving_events", "lat")
    op.drop_column("driving_events", "occurred_at")

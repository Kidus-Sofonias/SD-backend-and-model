"""add trip scoring fields

Revision ID: 20260321_add_trip_scoring_fields
Revises: 5e2ece5421a1
Create Date: 2026-03-21
"""

revision = "20260321_add_trip_scoring_fields"
down_revision = "5e2ece5421a1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # This revision previously duplicated the schema changes from 5e2ece5421a1.
    # Keep it as a no-op follow-up so existing databases can advance to a single head.
    pass


def downgrade() -> None:
    pass

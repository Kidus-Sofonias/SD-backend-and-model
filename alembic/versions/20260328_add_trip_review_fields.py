"""add trip review and risk fields

Revision ID: 20260328_add_trip_review_fields
Revises: 20260321_add_trip_scoring_fields
Create Date: 2026-03-28
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260328_add_trip_review_fields"
down_revision: Union[str, Sequence[str], None] = "20260321_add_trip_scoring_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("trips", sa.Column("risk_probability", sa.Float(), nullable=True))
    op.add_column("trips", sa.Column("risk_level", sa.String(), nullable=True))
    op.add_column("trips", sa.Column("reviewed_label", sa.Integer(), nullable=True))
    op.add_column("trips", sa.Column("reviewed_label_source", sa.String(), nullable=True))
    op.add_column("trips", sa.Column("review_notes", sa.Text(), nullable=True))
    op.add_column("trips", sa.Column("reviewed_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("trips", "reviewed_at")
    op.drop_column("trips", "review_notes")
    op.drop_column("trips", "reviewed_label_source")
    op.drop_column("trips", "reviewed_label")
    op.drop_column("trips", "risk_level")
    op.drop_column("trips", "risk_probability")

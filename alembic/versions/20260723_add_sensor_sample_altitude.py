"""add altitude_m to sensor_samples

Revision ID: 20260723_add_sensor_sample_altitude
Revises: 20260505_add_driving_event_occurrence_fields
Create Date: 2026-07-23 12:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260723_add_sensor_sample_altitude"
down_revision: Union[str, Sequence[str], None] = "20260505_add_driving_event_occurrence_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("sensor_samples", sa.Column("altitude_m", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("sensor_samples", "altitude_m")

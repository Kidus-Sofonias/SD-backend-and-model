"""add vehicle_profiles table and trips.vehicle_profile_id

Phase 3 (hackathon): vehicle-aware driver onboarding. One profile per user;
trips optionally record which vehicle they were driven in (admin replay context).

Revision ID: 20260813_add_vehicle_profile
Revises: 20260813_add_event_confidence_severity
Create Date: 2026-08-13
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260813_add_vehicle_profile"
down_revision: Union[str, Sequence[str], None] = "20260813_add_event_confidence_severity"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Dialect-safe and idempotent (SQLite has no `ADD COLUMN IF NOT EXISTS` and
    # the app's init_db self-heal may have created things first).
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "vehicle_profiles" not in tables:
        op.create_table(
            "vehicle_profiles",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("user_id", sa.String(), nullable=False),
            sa.Column("category", sa.String(length=32), nullable=False),
            sa.Column("make_model", sa.String(length=120), nullable=True),
            sa.Column("size_class", sa.String(length=32), nullable=True),
            sa.Column("drive_type", sa.String(length=16), nullable=True),
            sa.Column("mass_kg", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            op.f("ix_vehicle_profiles_user_id"),
            "vehicle_profiles",
            ["user_id"],
            unique=True,
        )

    trip_columns = {c["name"] for c in inspector.get_columns("trips")}
    if "vehicle_profile_id" not in trip_columns:
        op.add_column("trips", sa.Column("vehicle_profile_id", sa.String(), nullable=True))
        op.create_index(
            op.f("ix_trips_vehicle_profile_id"),
            "trips",
            ["vehicle_profile_id"],
            unique=False,
        )


def downgrade() -> None:
    op.drop_index(op.f("ix_trips_vehicle_profile_id"), table_name="trips")
    op.drop_column("trips", "vehicle_profile_id")
    op.drop_index(op.f("ix_vehicle_profiles_user_id"), table_name="vehicle_profiles")
    op.drop_table("vehicle_profiles")

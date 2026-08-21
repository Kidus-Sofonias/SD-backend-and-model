"""add organization and partner API key boundary

Revision ID: 20260821_add_partner_api
Revises: 20260813_add_vehicle_profile
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260821_add_partner_api"
down_revision: Union[str, Sequence[str], None] = "20260813_add_vehicle_profile"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if "organizations" not in tables:
        op.create_table(
            "organizations",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("name", sa.String(length=255), nullable=False),
            sa.Column("slug", sa.String(length=120), nullable=False),
            sa.Column("active", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("slug"),
        )
        op.create_index("ix_organizations_slug", "organizations", ["slug"], unique=True)
    if "partner_api_keys" not in tables:
        op.create_table(
            "partner_api_keys",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("organization_id", sa.String(length=36), nullable=False),
            sa.Column("name", sa.String(length=120), nullable=False),
            sa.Column("key_prefix", sa.String(length=24), nullable=False),
            sa.Column("key_hash", sa.String(length=64), nullable=False),
            sa.Column("active", sa.Boolean(), nullable=False),
            sa.Column("last_used_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("key_hash"),
        )
        op.create_index("ix_partner_api_keys_organization_id", "partner_api_keys", ["organization_id"])
        op.create_index("ix_partner_api_keys_key_prefix", "partner_api_keys", ["key_prefix"])
    user_columns = {column["name"] for column in sa.inspect(bind).get_columns("users")}
    if "organization_id" not in user_columns:
        op.add_column("users", sa.Column("organization_id", sa.String(length=36), nullable=True))
        op.create_index("ix_users_organization_id", "users", ["organization_id"])
    if "external_driver_id" not in user_columns:
        op.add_column("users", sa.Column("external_driver_id", sa.String(length=255), nullable=True))
        op.create_index("ix_users_external_driver_id", "users", ["external_driver_id"])
    user_indexes = {index["name"] for index in sa.inspect(bind).get_indexes("users")}
    if "ix_users_org_external_driver" not in user_indexes:
        op.create_index(
            "ix_users_org_external_driver",
            "users",
            ["organization_id", "external_driver_id"],
            unique=True,
        )
    trip_columns = {column["name"] for column in sa.inspect(bind).get_columns("trips")}
    if "source_trip_id" not in trip_columns:
        op.add_column("trips", sa.Column("source_trip_id", sa.String(length=255), nullable=True))
        op.create_index("ix_trips_user_source_trip_id", "trips", ["user_id", "source_trip_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_trips_user_source_trip_id", table_name="trips")
    op.drop_column("trips", "source_trip_id")
    op.drop_index("ix_users_org_external_driver", table_name="users")
    op.drop_index("ix_users_external_driver_id", table_name="users")
    op.drop_column("users", "external_driver_id")
    op.drop_index("ix_users_organization_id", table_name="users")
    op.drop_column("users", "organization_id")
    op.drop_index("ix_partner_api_keys_key_prefix", table_name="partner_api_keys")
    op.drop_index("ix_partner_api_keys_organization_id", table_name="partner_api_keys")
    op.drop_table("partner_api_keys")
    op.drop_index("ix_organizations_slug", table_name="organizations")
    op.drop_table("organizations")
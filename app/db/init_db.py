# File role: Database bootstrapping/session module used by repositories and route dependency injection.
# Connects to: app.db.session, app.db.base, app.db.models.user.
# Key symbols/vars: init_db.
import uuid

from sqlalchemy import inspect, select, text

from app.core.config import settings
from app.core.security import hash_password, verify_password
from app.db.session import SessionLocal, commit_with_retry, engine
from app.db.base import Base

from app.db.models.user import User
from app.db.models.trip import Trip
from app.db.models.driving_event import DrivingEvent
from app.db.models.sensor_sample import SensorSample
from app.db.models.organization import Organization, PartnerApiKey  # noqa: F401


def _ensure_user_role_column() -> None:
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("users")}
    if "role" in columns:
        return

    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR(32) NOT NULL DEFAULT 'driver'"))


def _ensure_partner_columns() -> None:
    inspector = inspect(engine)
    user_columns = {column["name"] for column in inspector.get_columns("users")}
    with engine.begin() as connection:
        if "organization_id" not in user_columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN organization_id VARCHAR(36)"))
        if "external_driver_id" not in user_columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN external_driver_id VARCHAR(255)"))
    trip_columns = {column["name"] for column in inspector.get_columns("trips")}
    with engine.begin() as connection:
        if "source_trip_id" not in trip_columns:
            connection.execute(text("ALTER TABLE trips ADD COLUMN source_trip_id VARCHAR(255)"))

    # Keep startup upgrades compatible with databases created before Alembic
    # was introduced for the partner integration.
    indexes = {index["name"] for index in inspect(engine).get_indexes("users")}
    if "ix_users_org_external_driver" not in indexes:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX ix_users_org_external_driver "
                    "ON users (organization_id, external_driver_id)"
                )
            )


def _seed_default_admin() -> None:
    admin_password = settings.admin_password
    if not admin_password:
        # No admin password configured — skip seeding.
        # This is fine in development where secrets are auto-generated.
        # In production, the @model_validator will catch missing secrets.
        print("Skipping admin seed: ADMIN_PASSWORD not configured. Set it in .env to create the admin user.")
        return

    db = SessionLocal()
    try:
        admin_email = settings.admin_email.lower().strip()
        if not admin_email:
            return

        admin = db.execute(select(User).where(User.email == admin_email)).scalar_one_or_none()

        if admin is None:
            db.add(
                User(
                    id=str(uuid.uuid4()),
                    email=admin_email,
                    password_hash=hash_password(admin_password),
                    role="admin",
                )
            )
            commit_with_retry(db)
            return

        changed = False
        if admin.role != "admin":
            admin.role = "admin"
            changed = True
        if not verify_password(admin_password, admin.password_hash):
            admin.password_hash = hash_password(admin_password)
            changed = True
        if changed:
            db.add(admin)
            commit_with_retry(db)
    finally:
        db.close()


def _ensure_driving_event_columns() -> None:
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("driving_events")}

    with engine.begin() as connection:
        if "occurred_at" not in columns:
            connection.execute(text("ALTER TABLE driving_events ADD COLUMN occurred_at TIMESTAMP"))
        if "lat" not in columns:
            connection.execute(text("ALTER TABLE driving_events ADD COLUMN lat FLOAT"))
        if "lon" not in columns:
            connection.execute(text("ALTER TABLE driving_events ADD COLUMN lon FLOAT"))
        # Phase 10 (hackathon): per-event confidence/severity/duration.
        if "confidence" not in columns:
            connection.execute(text("ALTER TABLE driving_events ADD COLUMN confidence FLOAT"))
        if "severity" not in columns:
            connection.execute(text("ALTER TABLE driving_events ADD COLUMN severity FLOAT"))
        if "duration_s" not in columns:
            connection.execute(text("ALTER TABLE driving_events ADD COLUMN duration_s FLOAT"))


def _ensure_trip_vehicle_column() -> None:
    """Phase 3 (hackathon): trips.vehicle_profile_id for vehicle-aware trips."""
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("trips")}
    if "vehicle_profile_id" in columns:
        return

    with engine.begin() as connection:
        connection.execute(
            text("ALTER TABLE trips ADD COLUMN vehicle_profile_id VARCHAR REFERENCES vehicle_profiles(id)")
        )


def ensure_sensor_sample_columns() -> None:
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("sensor_samples")}

    with engine.begin() as connection:
        if "altitude_m" not in columns:
            connection.execute(text("ALTER TABLE sensor_samples ADD COLUMN altitude_m FLOAT"))
        # Phase 9: phone-handling windows (driver picked up/adjusted the phone)
        # must be distinguishable from driving so they never generate phantom
        # events. Column added non-destructively for existing installs.
        if "phone_handling" not in columns:
            connection.execute(text("ALTER TABLE sensor_samples ADD COLUMN phone_handling BOOLEAN"))


def _resync_table_id_sequence(table_name: str) -> None:
    """Resync a table's primary-key sequence after explicit-ID imports.

    migrate_to_supabase.py re-inserts rows with explicit IDs. Explicit-ID
    inserts do NOT advance the Postgres sequence, so the next auto-generated id
    can collide with an existing row (UniqueViolation on the primary key). This
    resets the sequence to MAX(id)+1 so future auto-increment ids never collide.
    No-op on SQLite.
    """
    if not settings.database_url.startswith("postgresql"):
        return

    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return

    with engine.begin() as connection:
        seq = connection.execute(
            text("SELECT pg_get_serial_sequence(:tbl, 'id')"),
            {"tbl": table_name},
        ).scalar()
        if not seq:
            return
        connection.execute(
            text(
                "SELECT setval(:seq, "
                f"COALESCE((SELECT MAX(id) FROM {table_name}), 0) + 1, false)"
            ),
            {"seq": seq},
        )


def ensure_vehicle_profile_columns() -> None:
    """Phase 3b (hackathon): onboarding fields that tune detection further.
    Non-destructive; existing profiles keep working with nulls."""
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("vehicle_profiles")}

    additions = {
        "transmission": "VARCHAR(16)",
        "phone_placement": "VARCHAR(24)",
        "load_level": "VARCHAR(16)",
        "road_context": "VARCHAR(16)",
    }
    with engine.begin() as connection:
        for name, sql_type in additions.items():
            if name not in columns:
                connection.execute(
                    text(f"ALTER TABLE vehicle_profiles ADD COLUMN {name} {sql_type}")
                )


def ensure_sensor_sample_id_sequence() -> None:
    _resync_table_id_sequence("sensor_samples")


def ensure_driving_event_id_sequence() -> None:
    _resync_table_id_sequence("driving_events")


_ensure_sensor_sample_columns = ensure_sensor_sample_columns


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_user_role_column()
    _ensure_partner_columns()
    _ensure_driving_event_columns()
    _ensure_sensor_sample_columns()
    _ensure_trip_vehicle_column()
    ensure_vehicle_profile_columns()
    ensure_sensor_sample_id_sequence()
    ensure_driving_event_id_sequence()
    _seed_default_admin()

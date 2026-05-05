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


def _ensure_user_role_column() -> None:
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("users")}
    if "role" in columns:
        return

    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR(32) NOT NULL DEFAULT 'driver'"))


def _seed_default_admin() -> None:
    db = SessionLocal()
    try:
        admin_email = settings.admin_email.lower()
        admin = db.execute(select(User).where(User.email == admin_email)).scalar_one_or_none()
        desired_password = settings.admin_password

        if admin is None:
            db.add(
                User(
                    id=str(uuid.uuid4()),
                    email=admin_email,
                    password_hash=hash_password(desired_password),
                    role="admin",
                )
            )
            commit_with_retry(db)
            return

        changed = False
        if admin.role != "admin":
            admin.role = "admin"
            changed = True
        if not verify_password(desired_password, admin.password_hash):
            admin.password_hash = hash_password(desired_password)
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


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_user_role_column()
    _ensure_driving_event_columns()
    _seed_default_admin()

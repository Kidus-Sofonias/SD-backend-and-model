# File role: Data-access repository encapsulating SQLAlchemy queries used by service and route layers.
# Connects to: sqlalchemy, app.db.models.user.
# Key symbols/vars: UserRecord, SqlUserRepository.
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy.orm import Session
from sqlalchemy import func, select

from app.core.errors import AppError, NotFoundError
from app.db.models.trip import Trip
from app.db.models.user import User
from app.db.session import commit_with_retry


@dataclass
class UserRecord:
    id: str
    email: str
    password_hash: str
    role: str = "driver"

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


@dataclass
class DriverRecord:
    id: str
    email: str
    role: str
    trip_count: int = 0
    latest_trip_at: datetime | None = None


class SqlUserRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_email(self, email: str) -> Optional[UserRecord]:
        stmt = select(User).where(User.email == email.lower())
        user = self.db.execute(stmt).scalar_one_or_none()
        if not user:
            return None
        return UserRecord(id=user.id, email=user.email, password_hash=user.password_hash, role=user.role)

    def get_by_id(self, user_id: str) -> Optional[UserRecord]:
        stmt = select(User).where(User.id == user_id)
        user = self.db.execute(stmt).scalar_one_or_none()
        if not user:
            return None
        return UserRecord(id=user.id, email=user.email, password_hash=user.password_hash, role=user.role)

    def create(self, email: str, password_hash: str, role: str = "driver") -> UserRecord:
        user = User(
            id=str(uuid.uuid4()),
            email=email.lower(),
            password_hash=password_hash,
            role=role,
        )
        self.db.add(user)
        commit_with_retry(self.db)
        self.db.refresh(user)
        return UserRecord(id=user.id, email=user.email, password_hash=user.password_hash, role=user.role)

    def list_drivers(self) -> list[DriverRecord]:
        stmt = (
            select(
                User.id,
                User.email,
                User.role,
                func.count(Trip.id).label("trip_count"),
                func.max(func.coalesce(Trip.ended_at, Trip.started_at)).label("latest_trip_at"),
            )
            .outerjoin(Trip, Trip.user_id == User.id)
            .where(User.role == "driver")
            .group_by(User.id, User.email, User.role)
            .order_by(User.email.asc())
        )
        rows = self.db.execute(stmt).all()
        return [
            DriverRecord(
                id=row.id,
                email=row.email,
                role=row.role,
                trip_count=int(row.trip_count or 0),
                latest_trip_at=row.latest_trip_at,
            )
            for row in rows
        ]

    def get_driver_by_id(self, user_id: str) -> DriverRecord | None:
        stmt = (
            select(
                User.id,
                User.email,
                User.role,
                func.count(Trip.id).label("trip_count"),
                func.max(func.coalesce(Trip.ended_at, Trip.started_at)).label("latest_trip_at"),
            )
            .outerjoin(Trip, Trip.user_id == User.id)
            .where(User.id == user_id, User.role == "driver")
            .group_by(User.id, User.email, User.role)
        )
        row = self.db.execute(stmt).one_or_none()
        if row is None:
            return None
        return DriverRecord(
            id=row.id,
            email=row.email,
            role=row.role,
            trip_count=int(row.trip_count or 0),
            latest_trip_at=row.latest_trip_at,
        )

    def update_driver_credentials(
        self,
        user_id: str,
        *,
        email: str | None = None,
        password_hash: str | None = None,
    ) -> UserRecord:
        user = self.db.execute(select(User).where(User.id == user_id, User.role == "driver")).scalar_one_or_none()
        if user is None:
            raise NotFoundError(message_key="admin.driver_not_found")

        next_email = email.lower() if email else None
        if next_email and next_email != user.email:
            existing = self.db.execute(select(User).where(User.email == next_email, User.id != user_id)).scalar_one_or_none()
            if existing:
                raise AppError(message_key="auth.email_already_exists", status_code=409)
            user.email = next_email

        if password_hash:
            user.password_hash = password_hash

        self.db.add(user)
        commit_with_retry(self.db)
        self.db.refresh(user)
        return UserRecord(id=user.id, email=user.email, password_hash=user.password_hash, role=user.role)

    def delete_driver(self, user_id: str) -> None:
        user = self.db.execute(select(User).where(User.id == user_id, User.role == "driver")).scalar_one_or_none()
        if user is None:
            raise NotFoundError(message_key="admin.driver_not_found")
        self.db.delete(user)
        commit_with_retry(self.db)

# File role: Business-logic service that coordinates admin-only driver management use cases.
# Connects to: app.core.errors, app.core.security, app.repositories.user_repository.
# Key symbols/vars: AdminService.
from __future__ import annotations

import json

from sqlalchemy import delete
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import AppError, ForbiddenError, NotFoundError
from app.core.security import hash_password
from app.db.models.driving_event import DrivingEvent
from app.db.models.sensor_sample import SensorSample
from app.db.models.trip import Trip
from app.db.session import commit_with_retry
from app.repositories.user_repository import DriverRecord, SqlUserRepository, UserRecord


class AdminService:
    def __init__(self, db: Session, users: SqlUserRepository) -> None:
        self.db = db
        self.users = users

    def _require_admin(self, actor: UserRecord) -> None:
        if not actor.is_admin:
            raise ForbiddenError(message_key="auth.forbidden")

    def list_drivers(self, actor: UserRecord) -> list[DriverRecord]:
        self._require_admin(actor)
        return self.users.list_drivers()

    def _is_not_enough_samples(self, raw_breakdown: str | None) -> bool:
        if not raw_breakdown:
            return False
        try:
            breakdown = json.loads(raw_breakdown)
        except Exception:
            return False
        if not isinstance(breakdown, dict):
            return False
        if breakdown.get("error") == "not_enough_samples":
            return True
        nested = breakdown.get("rule_breakdown")
        return isinstance(nested, dict) and nested.get("error") == "not_enough_samples"

    def _cleanup_failed_insufficient_trips(self, driver_id: str) -> None:
        candidates = self.db.execute(
            select(Trip.id, Trip.score_breakdown).where(
                Trip.user_id == driver_id,
                Trip.score.is_(None),
                Trip.score_breakdown.is_not(None),
            )
        ).all()
        trip_ids = [row.id for row in candidates if self._is_not_enough_samples(row.score_breakdown)]
        if not trip_ids:
            return

        self.db.execute(
            delete(DrivingEvent).where(
                DrivingEvent.user_id == driver_id,
                DrivingEvent.trip_id.in_(trip_ids),
            )
        )
        self.db.execute(
            delete(SensorSample).where(
                SensorSample.user_id == driver_id,
                SensorSample.trip_id.in_(trip_ids),
            )
        )
        self.db.execute(
            delete(Trip).where(
                Trip.user_id == driver_id,
                Trip.id.in_(trip_ids),
            )
        )
        commit_with_retry(self.db)

    def get_driver_trips(self, actor: UserRecord, driver_id: str) -> list[Trip]:
        self._require_admin(actor)
        driver = self.users.get_driver_by_id(driver_id)
        if driver is None:
            raise NotFoundError(message_key="admin.driver_not_found")

        self._cleanup_failed_insufficient_trips(driver_id)

        stmt = select(Trip).where(Trip.user_id == driver_id).order_by(Trip.started_at.desc())
        return self.db.execute(stmt).scalars().all()

    def get_driver_trip_route(self, actor: UserRecord, driver_id: str, trip_id: str) -> dict:
        self._require_admin(actor)
        driver = self.users.get_driver_by_id(driver_id)
        if driver is None:
            raise NotFoundError(message_key="admin.driver_not_found")

        trip = self.db.execute(
            select(Trip).where(
                Trip.id == trip_id,
                Trip.user_id == driver_id,
            )
        ).scalar_one_or_none()
        if trip is None:
            raise NotFoundError(message_key="trip.not_found")

        samples = self.db.execute(
            select(SensorSample)
            .where(
                SensorSample.user_id == driver_id,
                SensorSample.trip_id == trip_id,
                SensorSample.lat.is_not(None),
                SensorSample.lon.is_not(None),
            )
            .order_by(SensorSample.ts.asc())
        ).scalars().all()

        points = [
            {
                "ts": sample.ts,
                "lat": float(sample.lat),
                "lon": float(sample.lon),
                "speed_mps": sample.speed_mps,
                "accuracy_m": sample.accuracy_m,
            }
            for sample in samples
        ]

        return {
            "trip_id": trip.id,
            "driver_user_id": driver_id,
            "point_count": len(points),
            "points": points,
        }

    def update_driver_credentials(
        self,
        actor: UserRecord,
        driver_id: str,
        *,
        email: str | None,
        password: str | None,
    ) -> UserRecord:
        self._require_admin(actor)
        if not email and not password:
            raise AppError(message_key="admin.no_updates_supplied", status_code=422)

        password_hash = hash_password(password) if password else None
        return self.users.update_driver_credentials(
            driver_id,
            email=email,
            password_hash=password_hash,
        )

    def delete_driver(self, actor: UserRecord, driver_id: str) -> None:
        self._require_admin(actor)
        self.users.delete_driver(driver_id)

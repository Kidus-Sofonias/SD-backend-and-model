# File role: Business-logic service that coordinates admin-only driver management use cases.
# Connects to: app.core.errors, app.core.security, app.repositories.user_repository.
# Key symbols/vars: AdminService.
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

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
from app.services.route_snap_service import RouteSnapService


class AdminService:
    def __init__(self, db: Session, users: SqlUserRepository) -> None:
        self.db = db
        self.users = users

    def _require_admin(self, actor: UserRecord) -> None:
        if not actor.is_admin:
            raise ForbiddenError(message_key="auth.forbidden")

    def _as_utc_timestamp(self, ts):
        if ts is None:
            return ts
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    def list_drivers(self, actor: UserRecord) -> list[DriverRecord]:
        self._require_admin(actor)
        return self.users.list_drivers()

    def _trip_anchor_timestamp(self, trip: Trip) -> datetime:
        candidate = trip.processed_at or trip.ended_at or trip.started_at
        if candidate.tzinfo is None:
            return candidate.replace(tzinfo=timezone.utc)
        return candidate.astimezone(timezone.utc)

    def _week_start(self, dt: datetime) -> datetime:
        midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight - timedelta(days=midnight.weekday())

    def _month_start(self, dt: datetime) -> datetime:
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def _add_months(self, dt: datetime, months: int) -> datetime:
        year = dt.year + ((dt.month - 1 + months) // 12)
        month = ((dt.month - 1 + months) % 12) + 1
        return dt.replace(year=year, month=month, day=1)

    def _empty_snapshot(self, label: str) -> dict:
        return {
            "label": label,
            "average_score": None,
            "trip_count": 0,
            "high_risk_trip_count": 0,
        }

    def _snapshot_from_point(self, point: dict) -> dict:
        return {
            "label": point["label"],
            "average_score": point["average_score"],
            "trip_count": point["trip_count"],
            "high_risk_trip_count": point["high_risk_trip_count"],
        }

    def _direction_for_delta(self, delta_score: float | None) -> str:
        if delta_score is None or abs(delta_score) < 0.05:
            return "flat"
        return "up" if delta_score > 0 else "down"

    def _build_trend_window(
        self,
        *,
        trips: list[Trip],
        periods: int,
        period_start_fn,
        next_period_fn,
        label_fn,
    ) -> dict:
        now = datetime.now(timezone.utc)
        current_start = period_start_fn(now)
        starts = [current_start]
        while len(starts) < periods:
            starts.insert(0, next_period_fn(starts[0], -1))

        buckets: dict[datetime, list[Trip]] = {start: [] for start in starts}
        for trip in trips:
            start = period_start_fn(self._trip_anchor_timestamp(trip))
            if start in buckets:
                buckets[start].append(trip)

        points: list[dict] = []
        for start in starts:
            trip_bucket = buckets[start]
            period_end = next_period_fn(start, 1) - timedelta(microseconds=1)
            avg_score = None
            if trip_bucket:
                avg_score = round(sum(int(trip.score or 0) for trip in trip_bucket) / len(trip_bucket), 1)
            points.append(
                {
                    "period_start": start,
                    "period_end": period_end,
                    "label": label_fn(start),
                    "average_score": avg_score,
                    "trip_count": len(trip_bucket),
                    "high_risk_trip_count": sum(1 for trip in trip_bucket if trip.risk_level == "high"),
                }
            )

        current_point = points[-1]
        previous_point = points[-2] if len(points) > 1 else None
        current_snapshot = self._snapshot_from_point(current_point)
        previous_snapshot = self._snapshot_from_point(previous_point) if previous_point else self._empty_snapshot("Previous")
        delta_score = None
        if current_snapshot["average_score"] is not None and previous_snapshot["average_score"] is not None:
            delta_score = round(float(current_snapshot["average_score"]) - float(previous_snapshot["average_score"]), 1)

        return {
            "current": current_snapshot,
            "previous": previous_snapshot,
            "delta_score": delta_score,
            "direction": self._direction_for_delta(delta_score),
            "points": points,
        }

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

    def get_driver_insights(self, actor: UserRecord, driver_id: str) -> dict:
        self._require_admin(actor)
        driver = self.users.get_driver_by_id(driver_id)
        if driver is None:
            raise NotFoundError(message_key="admin.driver_not_found")

        self._cleanup_failed_insufficient_trips(driver_id)

        trips = self.db.execute(
            select(Trip)
            .where(
                Trip.user_id == driver_id,
                Trip.score.is_not(None),
            )
            .order_by(Trip.started_at.asc())
        ).scalars().all()

        overall_average_score = None
        if trips:
            overall_average_score = round(sum(int(trip.score or 0) for trip in trips) / len(trips), 1)

        weekly = self._build_trend_window(
            trips=trips,
            periods=8,
            period_start_fn=self._week_start,
            next_period_fn=lambda start, step: start + timedelta(weeks=step),
            label_fn=lambda start: start.strftime("%b %d"),
        )
        monthly = self._build_trend_window(
            trips=trips,
            periods=6,
            period_start_fn=self._month_start,
            next_period_fn=self._add_months,
            label_fn=lambda start: start.strftime("%b %Y"),
        )

        return {
            "driver_id": driver.id,
            "driver_email": driver.email,
            "overall_average_score": overall_average_score,
            "scored_trip_count": len(trips),
            "high_risk_trip_count": sum(1 for trip in trips if trip.risk_level == "high"),
            "weekly": weekly,
            "monthly": monthly,
        }

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
                "ts": self._as_utc_timestamp(sample.ts),
                "lat": float(sample.lat),
                "lon": float(sample.lon),
                "speed_mps": sample.speed_mps,
                "accuracy_m": sample.accuracy_m,
            }
            for sample in samples
        ]
        snap_result = RouteSnapService().snap(points)

        return {
            "trip_id": trip.id,
            "driver_user_id": driver_id,
            "point_count": len(points),
            "points": points,
            "snapped_points": snap_result.snapped_points,
            "snapped_source": snap_result.source,
            "snapped_status": snap_result.status,
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

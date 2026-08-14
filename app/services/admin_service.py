# File role: Business-logic service that coordinates admin-only driver management use cases.
# Connects to: app.core.errors, app.core.security, app.repositories.user_repository.
# Key symbols/vars: AdminService.
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import AppError, ForbiddenError, NotFoundError
from app.core.security import hash_password
from app.core.utils import as_utc_timestamp
from app.db.models.sensor_sample import SensorSample
from app.db.models.trip import Trip
from app.db.models.vehicle_profile import VehicleProfile
from app.realtime.live_detector import live_alert_detector
from app.repositories.sensor_sample_repository import SensorSampleRepository
from app.repositories.user_repository import DriverRecord, SqlUserRepository, UserRecord
from app.services.live_monitor_service import (
    _accel_magnitude,
    _lateral_accel,
    _longitudinal_accel,
    _provisional_live_score,
    _vertical_accel,
    vehicle_tuned_cfg,
)
from app.services.route_snap_service import RouteSnapService

# A trip is "live" if its latest sample is under this age; "stale" while it
# still has some recent data; otherwise "disconnected" (Phase 7).
LIVE_SAMPLE_AGE_S = 60
STALE_SAMPLE_AGE_S = 300


def _connection_status(last_sample_age_s: float | None) -> str:
    if last_sample_age_s is None:
        return "disconnected"
    if last_sample_age_s <= LIVE_SAMPLE_AGE_S:
        return "live"
    if last_sample_age_s <= STALE_SAMPLE_AGE_S:
        return "stale"
    return "disconnected"


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

    def list_live_trips(self, actor: UserRecord) -> list[dict]:
        """Fleet-wide live monitoring snapshot (Phase 7).

        Every active trip across all drivers with latest telemetry, live event
        counters, provisional score and connection status so admins can spot
        drivers who need attention at a glance.
        """
        self._require_admin(actor)
        now = datetime.now(timezone.utc)

        trips = self.db.execute(
            select(Trip)
            .where(Trip.status == "active")
            .order_by(Trip.started_at.desc())
        ).scalars().all()

        sample_repo = SensorSampleRepository(self.db)

        # Batch-load vehicle profiles once so each trip's provisional score is
        # computed with its driver's vehicle-tuned config (no N+1 lookups).
        profile_ids = [trip.vehicle_profile_id for trip in trips if trip.vehicle_profile_id]
        profiles: dict[str, VehicleProfile] = {}
        if profile_ids:
            rows = self.db.execute(
                select(VehicleProfile).where(VehicleProfile.id.in_(profile_ids))
            ).scalars().all()
            profiles = {p.id: p for p in rows}

        results: list[dict] = []
        for trip in trips:
            samples = sample_repo.list_latest_by_trip(
                user_id=trip.user_id,
                trip_id=trip.id,
                limit=2,
            )
            latest = samples[0] if samples else None
            prev = samples[1] if len(samples) > 1 else None
            sample_count = sample_repo.count_by_trip(user_id=trip.user_id, trip_id=trip.id)

            last_sample_age_s = None
            if latest and latest.ts is not None:
                ts = latest.ts
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                last_sample_age_s = round(max(0.0, (now - ts).total_seconds()), 1)

            started_at = trip.started_at
            if started_at is not None and started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
            elapsed_s = max(0.0, (now - started_at).total_seconds()) if started_at else 0.0

            counts = live_alert_detector.event_counts(trip.id)
            cfg = vehicle_tuned_cfg(self.db, trip, profiles=profiles)
            driver = self.users.get_driver_by_id(trip.user_id)

            profile = profiles.get(trip.vehicle_profile_id) if trip.vehicle_profile_id else None
            results.append(
                {
                    "trip_id": trip.id,
                    "driver_user_id": trip.user_id,
                    "driver_email": driver.email if driver else None,
                    "vehicle_category": profile.category if profile else None,
                    "started_at": started_at.isoformat() if started_at else None,
                    "elapsed_s": round(elapsed_s, 1),
                    "latest": {
                        "ts": latest.ts.isoformat() if latest and latest.ts is not None else None,
                        "speed_mps": latest.speed_mps if latest else None,
                        "lat": latest.lat if latest else None,
                        "lon": latest.lon if latest else None,
                        "accuracy_m": latest.accuracy_m if latest else None,
                        "accel_mag_mps2": _accel_magnitude(latest),
                        "longitudinal_accel_mps2": _longitudinal_accel(prev, latest),
                        "lateral_accel_mps2": _lateral_accel(latest),
                        "vertical_accel_mps2": _vertical_accel(latest),
                    },
                    "samples_uploaded": sample_count,
                    "event_counts": counts,
                    "event_total": int(sum(counts.values())),
                    "live_score": _provisional_live_score(counts, elapsed_s, cfg),
                    "connection_status": _connection_status(last_sample_age_s),
                    "last_sample_age_s": last_sample_age_s,
                }
            )
        return results

    def get_live_trip_telemetry(self, actor: UserRecord, trip_id: str) -> dict:
        """Live telemetry for ANY trip (admin only, Phase 7 fleet -> detail).

        The driver-scoped endpoint serves only the caller's own trip; admins
        opening a live trip from the fleet dashboard need the same payload for
        whichever driver's trip they are inspecting.
        """
        self._require_admin(actor)
        trip = self.db.execute(
            select(Trip).where(Trip.id == trip_id)
        ).scalar_one_or_none()
        if trip is None:
            raise NotFoundError(message_key="trip.not_found")
        from app.services.live_monitor_service import LiveMonitorService

        return LiveMonitorService(self.db).get_trip_telemetry(
            user_id=trip.user_id,
            trip_id=trip.id,
        )

    def list_all_trips(self, actor: UserRecord, limit: int = 200, offset: int = 0) -> list[Trip]:
        """List trips across all drivers (admin only), most recent first, paginated."""
        self._require_admin(actor)
        stmt = select(Trip).order_by(Trip.started_at.desc()).limit(limit).offset(offset)
        return self.db.execute(stmt).scalars().all()

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

    def get_driver_trips(self, actor: UserRecord, driver_id: str) -> list[Trip]:
        self._require_admin(actor)
        driver = self.users.get_driver_by_id(driver_id)
        if driver is None:
            raise NotFoundError(message_key="admin.driver_not_found")

        stmt = select(Trip).where(Trip.user_id == driver_id).order_by(Trip.started_at.desc())
        return self.db.execute(stmt).scalars().all()

    def get_driver_insights(self, actor: UserRecord, driver_id: str) -> dict:
        self._require_admin(actor)
        driver = self.users.get_driver_by_id(driver_id)
        if driver is None:
            raise NotFoundError(message_key="admin.driver_not_found")

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
                "ts": as_utc_timestamp(sample.ts),
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

    def get_trip_samples(self, actor: UserRecord, trip_id: str, limit: int = 3000) -> dict:
        """Raw sensor timeline for ANY trip (admin only, replay/forensics).

        Returns the per-sample timeline (speed, GPS, IMU axes) so the admin
        replay UI can scrub through the trip with the 3D vehicle, sensor
        traces and events synchronized to the same clock.
        """
        self._require_admin(actor)
        trip = self.db.execute(
            select(Trip).where(Trip.id == trip_id)
        ).scalar_one_or_none()
        if trip is None:
            raise NotFoundError(message_key="trip.not_found")

        rows = SensorSampleRepository(self.db).list_by_trip(
            user_id=trip.user_id,
            trip_id=trip.id,
            limit=limit,
        )
        samples = [
            {
                "ts": as_utc_timestamp(sample.ts),
                "speed_mps": sample.speed_mps,
                "lat": sample.lat,
                "lon": sample.lon,
                "accuracy_m": sample.accuracy_m,
                "ax": sample.ax,
                "ay": sample.ay,
                "az": sample.az,
                "gz": sample.gz,
            }
            for sample in rows
        ]
        return {"trip_id": trip.id, "count": len(samples), "samples": samples}

    def get_trip_route(self, actor: UserRecord, trip_id: str) -> dict:
        """Get route for ANY trip (admin only, no driver_id needed)."""
        self._require_admin(actor)

        trip = self.db.execute(
            select(Trip).where(Trip.id == trip_id)
        ).scalar_one_or_none()
        if trip is None:
            raise NotFoundError(message_key="trip.not_found")

        samples = self.db.execute(
            select(SensorSample)
            .where(
                SensorSample.trip_id == trip_id,
                SensorSample.lat.is_not(None),
                SensorSample.lon.is_not(None),
            )
            .order_by(SensorSample.ts.asc())
        ).scalars().all()

        points = [
            {
                "ts": as_utc_timestamp(sample.ts),
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
            "driver_user_id": trip.user_id,
            "point_count": len(points),
            "points": points,
            "snapped_points": snap_result.snapped_points,
            "snapped_source": snap_result.source,
            "snapped_status": snap_result.status,
        }

    def delete_driver(self, actor: UserRecord, driver_id: str) -> None:
        self._require_admin(actor)
        self.users.delete_driver(driver_id)

# File role: Live trip telemetry for Phase 6 driver monitoring (glance +
# details modes). Aggregates the latest stored sensor samples, the live
# detector's in-memory event counters/alerts, and trip state into one payload
# the mobile app polls alongside the WebSocket alert stream.
# Connects to:
# - app.repositories.trip_repository
# - app.repositories.sensor_sample_repository
# - app.realtime.live_detector
# Key symbols/vars: LiveMonitorService.
from __future__ import annotations

import math
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.ml.config import FeatureConfigV2
from app.realtime.live_detector import live_alert_detector
from app.repositories.sensor_sample_repository import SensorSampleRepository
from app.repositories.trip_repository import SqlTripRepository

LATEST_SAMPLE_WINDOW = 3


def _provisional_live_score(counts: dict, elapsed_s: float) -> dict:
    """Estimate the trip's safety score live from detected event counts.

    Uses the same v2 penalty weights and event-density normalization as
    finalize, but omits the smoothness terms (p95 jerk, speed variance) that
    are only known at finalize - so this is a conservative, provisional
    reading that tightens toward the real score as the trip progresses.
    """
    cfg = FeatureConfigV2()

    per_event = {
        "emergency_brake": cfg.w_emergency_brake,
        "hard_brake": cfg.w_brake,
        "hard_accel": cfg.w_accel,
        "aggressive_turn": cfg.w_turn,
        "overspeed": cfg.w_overspeed,
        "severe_overspeed": cfg.w_severe_overspeed,
        "unstable_motion": cfg.w_unstable_motion,
    }
    penalties = {key: weight * int(counts.get(key, 0)) for key, weight in per_event.items()}
    chargeable = sum(penalties.values())

    hours = max(elapsed_s / 3600.0, 1.0 / 60.0)
    events_per_hour = chargeable / hours
    density = cfg.w_density * min(
        max(events_per_hour, 0.0) / cfg.density_normalize_high,
        1.0,
    )

    raw_score = 100 - chargeable - density
    score = int(max(0, min(100, round(raw_score))))
    if score >= 85:
        risk_level = "low"
    elif score >= 65:
        risk_level = "medium"
    else:
        risk_level = "high"

    return {
        "score": score,
        "risk_level": risk_level,
        "penalties": penalties,
        "density_penalty": round(density, 2),
        "provisional": True,
        "scoring_version": "v2-live",
    }


def _accel_magnitude(row) -> float | None:
    """IMU magnitude (m/s^2) from the latest sample, if any axis is present."""
    if row is None:
        return None
    values = [row.ax, row.ay, row.az]
    if all(v is None for v in values):
        return None
    return math.sqrt(sum((float(v) if v is not None else 0.0) ** 2 for v in values))


def _longitudinal_accel(prev, latest) -> float | None:
    """dv/dt (m/s^2) between the two most recent samples, if both are valid."""
    if prev is None or latest is None:
        return None
    if prev.speed_mps is None or latest.speed_mps is None:
        return None
    if latest.ts is None or prev.ts is None:
        return None
    dt = (latest.ts - prev.ts).total_seconds()
    if not dt or dt <= 0 or dt > 15:
        return None
    return (latest.speed_mps - prev.speed_mps) / dt


class LiveMonitorService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.trip_repo = SqlTripRepository(db)
        self.sample_repo = SensorSampleRepository(db)

    def get_trip_telemetry(self, *, user_id: str, trip_id: str) -> dict:
        trip = self.trip_repo.get_by_id(trip_id=trip_id, user_id=user_id)
        if not trip:
            raise NotFoundError(message_key="trip.not_found")

        samples = self.sample_repo.list_latest_by_trip(
            user_id=user_id,
            trip_id=trip_id,
            limit=LATEST_SAMPLE_WINDOW,
        )
        # list_latest_by_trip returns newest first.
        latest = samples[0] if samples else None
        prev = samples[1] if len(samples) > 1 else None

        sample_count = self.sample_repo.count_by_trip(user_id=user_id, trip_id=trip_id)

        now = datetime.now(timezone.utc)
        started_at = trip.started_at
        if started_at is not None and started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        elapsed_s = max(0.0, (now - started_at).total_seconds()) if started_at else 0.0

        counts = live_alert_detector.event_counts(trip_id)
        recent_alerts = live_alert_detector.recent_alerts(trip_id)

        return {
            "trip_id": trip.id,
            "status": trip.status,
            "started_at": started_at.isoformat() if started_at else None,
            "elapsed_s": round(elapsed_s, 1),
            "samples_uploaded": sample_count,
            "latest": {
                "ts": latest.ts.isoformat() if latest and latest.ts is not None else None,
                "speed_mps": latest.speed_mps if latest else None,
                "lat": latest.lat if latest else None,
                "lon": latest.lon if latest else None,
                "accuracy_m": latest.accuracy_m if latest else None,
                "accel_mag_mps2": _accel_magnitude(latest),
                "longitudinal_accel_mps2": _longitudinal_accel(prev, latest),
            },
            "live_score": _provisional_live_score(counts, elapsed_s),
            "event_counts": counts,
            "event_total": int(sum(counts.values())),
            "recent_alerts": recent_alerts,
        }

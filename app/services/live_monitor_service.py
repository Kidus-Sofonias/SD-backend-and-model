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
from app.db.models.vehicle_profile import VehicleProfile
from app.ml.config import FeatureConfigV2
from app.ml.vehicle_profiles import config_for_profile
from app.realtime.live_detector import live_alert_detector
from app.repositories.sensor_sample_repository import SensorSampleRepository
from app.repositories.trip_repository import SqlTripRepository

LATEST_SAMPLE_WINDOW = 3


def vehicle_tuned_cfg(
    db: Session,
    trip,
    *,
    profiles: dict[str, VehicleProfile] | None = None,
) -> FeatureConfigV2:
    """Vehicle-tuned detection config for a trip's live scoring/alerting.

    Falls back to the universal default when the trip has no vehicle profile
    (or it cannot be loaded). Pass a preloaded ``{id: profile}`` map to avoid
    N+1 lookups in fleet-wide admin views.
    """
    profile = None
    profile_id = getattr(trip, "vehicle_profile_id", None)
    if profile_id:
        if profiles is not None:
            profile = profiles.get(profile_id)
        else:
            profile = db.get(VehicleProfile, profile_id)
    return config_for_profile(FeatureConfigV2(), profile)


def _provisional_live_score(
    counts: dict,
    elapsed_s: float,
    cfg: FeatureConfigV2 | None = None,
) -> dict:
    """Estimate the trip's safety score live from detected event counts.

    Mirrors the v3 (Phase 10) finalize model: per-event penalties use the same
    weights (live events default to severity 1.0 / high confidence because
    per-event impacts are computed at finalize), and the exposure term uses the
    reduced time-based density floor. Smoothness terms (p95 jerk, speed
    variance) are only known at finalize - so this is a    conservative,
    provisional reading that tightens toward the real score as the trip
    progresses. ``cfg`` may be a vehicle-tuned config (same weights, so the
    vehicle effect enters through the event counts produced by the tuned
    detection thresholds).
    """
    cfg = cfg or FeatureConfigV2()

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

    hours = max(elapsed_s / 3600.0, cfg.density_min_duration_s / 3600.0)
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
        "scoring_version": "v3-live",
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


def _lateral_accel(latest) -> float | None:
    """Signed lateral acceleration proxy (m/s^2): speed * yaw-rate (gz).

    Mirrors the offline pipeline's lateral proxy (``speed_s * gz_s``) so the
    3D vehicle leans the same way the event detector reads the turn. Signed so
    the visualization knows which direction the body rolls. Returns None when
    either input is missing.
    """
    if latest is None:
        return None
    if latest.speed_mps is None or latest.gz is None:
        return None
    return float(latest.speed_mps) * float(latest.gz)


def _vertical_accel(latest) -> float | None:
    """Dynamic vertical acceleration (m/s^2), gravity removed.

    The phone's z-axis includes ~9.81 m/s^2 of gravity when roughly level, so
    the bounce/suspension signal is the deviation from g (az - 9.81). Positive
    = upward jolt (bump crest), negative = downward dip. Only valid while the
    phone stays reasonably level; the 3D view treats it as a soft suspension
    input, not a precise measurement.
    """
    if latest is None or latest.az is None:
        return None
    return float(latest.az) - 9.81


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
        cfg = vehicle_tuned_cfg(self.db, trip)

        vehicle_category = None
        profile_id = getattr(trip, "vehicle_profile_id", None)
        if profile_id:
            profile = self.db.get(VehicleProfile, profile_id)
            vehicle_category = profile.category if profile else None

        # While the phone is being handled (picked up/adjusted), its IMU
        # readings are the hand's motion, not the vehicle's — surface no
        # accel so the 3D view settles instead of reacting to phantom input.
        phone_handling = bool(latest is not None and latest.phone_handling)
        longitudinal = None if phone_handling else _longitudinal_accel(prev, latest)
        lateral = None if phone_handling else _lateral_accel(latest)
        vertical = None if phone_handling else _vertical_accel(latest)

        return {
            "trip_id": trip.id,
            "status": trip.status,
            "started_at": started_at.isoformat() if started_at else None,
            "elapsed_s": round(elapsed_s, 1),
            "samples_uploaded": sample_count,
            "vehicle_category": vehicle_category,
            "phone_handling": phone_handling,
            "latest": {
                "ts": latest.ts.isoformat() if latest and latest.ts is not None else None,
                "speed_mps": latest.speed_mps if latest else None,
                "lat": latest.lat if latest else None,
                "lon": latest.lon if latest else None,
                "accuracy_m": latest.accuracy_m if latest else None,
                "accel_mag_mps2": None if phone_handling else _accel_magnitude(latest),
                "longitudinal_accel_mps2": longitudinal,
                "lateral_accel_mps2": lateral,
                "vertical_accel_mps2": vertical,
            },
            "live_score": _provisional_live_score(counts, elapsed_s, cfg),
            "event_counts": counts,
            "event_total": int(sum(counts.values())),
            "recent_alerts": recent_alerts,
        }

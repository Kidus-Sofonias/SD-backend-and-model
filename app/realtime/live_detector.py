# File role: Incremental live-event detection for real-time driver alerts.
# Keeps a per-trip rolling window of recent raw samples, re-runs the same v2
# detection pipeline used at finalize over that window on each upload, and
# publishes only *new* events over the AlertHub. Persisted events are never
# written here - finalize remains the single authority for stored events.
# Key symbols/vars: LiveAlertDetector, live_alert_detector.
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.driving_event import DrivingEvent
from app.db.models.trip import Trip
from app.db.models.vehicle_profile import VehicleProfile
from app.ml.config import FeatureConfigV2
from app.ml.event_generation import generate_trip_events
from app.ml.features import compute_per_sample_features
from app.ml.preprocessing import preprocess_samples
from app.ml.vehicle_profiles import config_for_profile
from app.realtime.hub import FLEET_GLOBAL_KEY, alert_hub

logger = logging.getLogger(__name__)

# Rolling window: keep the most recent N samples so event detection stays cheap
# (a few ms) while covering the longest sustained category (overspeed >= 10 s).
WINDOW_SIZE = 240
# Cap the recent-alert replay buffer per trip.
RECENT_ALERTS_CAP = 50


class LiveAlertDetector:
    def __init__(self, cfg: Optional[FeatureConfigV2] = None) -> None:
        self.cfg = cfg or FeatureConfigV2()
        self._lock = threading.Lock()
        # trip_id -> list of pipeline-ready sample dicts (most recent last)
        self._windows: Dict[str, List[dict]] = {}
        # trip_id -> set of already-alerted keys (event_type, occurred_at)
        self._alerted: Dict[str, set] = {}
        # trip_id -> ordered recent alerts (newest last) for reconnect replay
        self._recent: Dict[str, List[dict]] = {}
        # trip_id -> set of trips whose DB-seed has been applied
        self._seeded: set = set()
        # trip_id -> vehicle-tuned FeatureConfigV2 (Phase 3/5: live alerts use
        # the driver's vehicle thresholds). Falls back to self.cfg.
        self._trip_cfgs: Dict[str, FeatureConfigV2] = {}
        # trip_id -> trips whose vehicle-config DB lookup has been attempted
        # (so trips without a profile are not re-queried on every upload).
        self._config_seeded: set = set()

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _row_to_pipeline(row: dict) -> dict:
        """Convert an uploaded sample dict into the pipeline payload shape
        (timestamp ISO, speed in km/h, raw IMU axes, lat/lon)."""
        raw_ts = row.get("ts") or row.get("timestamp")
        ts = raw_ts
        if isinstance(raw_ts, datetime):
            if raw_ts.tzinfo is None:
                ts = raw_ts.replace(tzinfo=timezone.utc)
            ts = ts.isoformat()

        speed_mps = row.get("speed_mps", row.get("speed"))
        return {
            "timestamp": ts,
            # pipeline expects km/h (FeatureConfigV2.input_speed_unit == "kmh")
            "speed": (float(speed_mps) * 3.6) if speed_mps is not None else None,
            "lat": row.get("lat"),
            "lon": row.get("lon"),
            "ax": row.get("ax"),
            "ay": row.get("ay"),
            "az": row.get("az"),
            "gx": row.get("gx"),
            "gy": row.get("gy"),
            "gz": row.get("gz"),
        }

    @staticmethod
    def _normalize_ts(value: object) -> str:
        """Canonicalize a timestamp (ISO string or datetime) to a UTC ISO
        string so DB-seeded keys (naive datetimes read back from SQLite) match
        detection keys (Z-suffixed ISO strings)."""
        if isinstance(value, datetime):
            dt = value
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        if isinstance(value, str) and value:
            normalized = value.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(normalized)
            except ValueError:
                return value
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        return str(value) if value is not None else ""

    @staticmethod
    def _event_key(event_type: str, occurred_at: object) -> tuple:
        return (event_type, LiveAlertDetector._normalize_ts(occurred_at))

    # -- lifecycle -----------------------------------------------------------

    def set_trip_config(self, trip_id: str, cfg: FeatureConfigV2) -> None:
        """Bind a vehicle-tuned detection config to a trip (set at trip start
        so live alert thresholds match the driver's vehicle)."""
        with self._lock:
            self._trip_cfgs[trip_id] = cfg

    def _bind_config_from_db(self, db: Session, trip_id: str) -> None:
        """Best-effort lazy re-bind of the trip's vehicle-tuned config.

        ``set_trip_config`` is called at trip start; after a server restart the
        in-memory binding is gone, so this restores it from the persisted trip
        row (trip.vehicle_profile_id) on the next upload. Never raises into
        the upload path and only queries once per trip.
        """
        if trip_id in self._config_seeded:
            return
        try:
            trip = db.get(Trip, trip_id)
            profile = None
            if trip is not None and trip.vehicle_profile_id:
                profile = db.get(VehicleProfile, trip.vehicle_profile_id)
            if profile is not None:
                with self._lock:
                    self._trip_cfgs[trip_id] = config_for_profile(FeatureConfigV2(), profile)
        except Exception:
            logger.exception("Failed to bind vehicle config from DB (trip %s)", trip_id)
        finally:
            with self._lock:
                self._config_seeded.add(trip_id)

    def _seed_from_db(self, db: Session, user_id: str, trip_id: str) -> None:
        """Pre-populate the alerted set with events already persisted for this
        trip so a server restart never replays old alerts."""
        if trip_id in self._seeded:
            return
        try:
            rows = db.execute(
                select(DrivingEvent).where(
                    DrivingEvent.trip_id == trip_id,
                    DrivingEvent.user_id == user_id,
                )
            ).scalars().all()
            with self._lock:
                alerted = self._alerted.setdefault(trip_id, set())
                for ev in rows:
                    alerted.add(self._event_key(ev.event_type, ev.occurred_at))
                self._seeded.add(trip_id)
        except Exception:
            logger.exception("Failed to seed live detector from DB (trip %s)", trip_id)
            # Still mark seeded so we don't retry every upload on a flaky DB.
            with self._lock:
                self._seeded.add(trip_id)

    def process_upload(
        self,
        *,
        db: Session,
        user_id: str,
        trip_id: str,
        rows: List[dict],
    ) -> List[dict]:
        """Append a newly-uploaded sample batch to the trip's window, detect
        events, and publish any new ones over the alert hub. Returns the list
        of published alert payloads (empty if nothing new)."""
        if not rows:
            return []

        self._seed_from_db(db, user_id, trip_id)
        self._bind_config_from_db(db, trip_id)

        payloads = [self._row_to_pipeline(row) for row in rows]

        with self._lock:
            window = self._windows.setdefault(trip_id, [])
            window.extend(payloads)
            if len(window) > WINDOW_SIZE:
                del window[: len(window) - WINDOW_SIZE]
            alerted = self._alerted.setdefault(trip_id, set())

            # Detect on the window (best-effort; never raise into the upload path).
            new_events: List[dict] = []
            try:
                new_events = self._detect_events(trip_id, window, alerted)
            except Exception:
                logger.exception("Live event detection failed (trip %s)", trip_id)

            if not new_events:
                return []

            recent = self._recent.setdefault(trip_id, [])
            alerts: List[dict] = []
            for event in new_events:
                alert = {
                    "type": "event_alert",
                    "trip_id": trip_id,
                    "event": event,
                    "sent_at": datetime.now(timezone.utc).isoformat(),
                }
                alerts.append(alert)
                recent.append(alert)
            if len(recent) > RECENT_ALERTS_CAP:
                del recent[: len(recent) - RECENT_ALERTS_CAP]

        # Publish outside the lock (hub has its own locking). Each alert goes
        # to the driver AND to the fleet-wide key so admin sessions can watch
        # the whole fleet in real time (Phase 7).
        for alert in alerts:
            try:
                alert_hub.publish(user_id, alert)
            except Exception:
                logger.exception("Failed to publish live alert (user %s)", user_id)
            try:
                alert_hub.publish(FLEET_GLOBAL_KEY, alert)
            except Exception:
                logger.exception("Failed to publish fleet alert")
        return alerts

    def _detect_events(self, trip_id: str, window: List[dict], alerted: set) -> List[dict]:
        """Run the detection pipeline over the window with the trip's
        vehicle-tuned config (universal default when the trip has no profile)
        and return events not already alerted (also marks them as alerted)."""
        cfg = self._trip_cfgs.get(trip_id, self.cfg)
        df = preprocess_samples(
            window,
            cfg.max_gap_s,
            cfg.ema_alpha,
            cfg.input_speed_unit,
        )
        if df.empty or len(df) < 2:
            return []

        per = compute_per_sample_features(df, nominal_dt_s=cfg.nominal_dt_s)
        # generate_trip_events only uses trip_features as a truthiness guard.
        events = generate_trip_events(
            per,
            {"confidence": 1.0},
            harsh_brake_dv=cfg.harsh_brake_dv,
            harsh_accel_dv=cfg.harsh_accel_dv,
            emergency_brake_dv=cfg.emergency_brake_dv,
            emergency_brake_min_speed_mps=cfg.emergency_brake_min_speed_mps,
            aggressive_turn_threshold=cfg.aggressive_turn_threshold,
            turn_min_duration_s=cfg.turn_min_duration_s,
            min_event_duration_s=cfg.min_event_duration_s,
            merge_gap_s=cfg.merge_gap_s,
            unstable_motion_jerk_threshold=cfg.unstable_motion_jerk_threshold,
            overspeed_threshold_mps=cfg.overspeed_threshold_mps,
            overspeed_min_duration_s=cfg.overspeed_min_duration_s,
            severe_overspeed_threshold_mps=cfg.severe_overspeed_threshold_mps,
            severe_overspeed_min_duration_s=cfg.severe_overspeed_min_duration_s,
            # Keep live alerts consistent with finalize (Phase 10).
            event_cooldown_s=cfg.event_cooldown_s,
            dv_min_speed_delta_mps=cfg.dv_min_speed_delta_mps,
            dv_single_sample_peak_mps2=cfg.dv_single_sample_peak_mps2,
            nominal_dt_s=cfg.nominal_dt_s,
            unstable_cooldown_s=cfg.unstable_cooldown_s,
            turn_min_speed_mps=cfg.turn_min_speed_mps,
            brake_severity_ref_mps2=cfg.brake_severity_ref_mps2,
            accel_severity_ref_mps2=cfg.accel_severity_ref_mps2,
            turn_severity_ref_mps2=cfg.turn_severity_ref_mps2,
            unstable_severity_ref_mps3=cfg.unstable_severity_ref_mps3,
            overspeed_severity_ref_mps=cfg.overspeed_severity_ref_mps,
            severe_overspeed_severity_ref_mps=cfg.severe_overspeed_severity_ref_mps,
        )

        new_events: List[dict] = []
        for event in events:
            key = self._event_key(event["event_type"], event.get("occurred_at"))
            if key in alerted:
                continue
            alerted.add(key)
            new_events.append(event)
        return new_events

    # -- read API ------------------------------------------------------------

    def recent_alerts(self, trip_id: str) -> List[dict]:
        with self._lock:
            return list(self._recent.get(trip_id, []))

    def event_counts(self, trip_id: str) -> Dict[str, int]:
        """Aggregate live-detected event counters per event type for a trip.

        Counts every event the detector has flagged this process lifetime (the
        in-memory alerted set), including events seeded from persisted rows.
        Used by the live telemetry endpoint for glance/details modes.
        """
        with self._lock:
            # Copy the set so we never iterate a shared set that a concurrent
            # upload may be mutating (process_upload adds keys under the lock).
            alerted = set(self._alerted.get(trip_id, ()))
        counts: Dict[str, int] = {}
        for event_type, _occurred_at in alerted:
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts

    def clear_trip(self, trip_id: str) -> None:
        with self._lock:
            self._windows.pop(trip_id, None)
            self._alerted.pop(trip_id, None)
            self._recent.pop(trip_id, None)
            self._trip_cfgs.pop(trip_id, None)
            self._config_seeded.discard(trip_id)
            self._seeded.discard(trip_id)


live_alert_detector = LiveAlertDetector()

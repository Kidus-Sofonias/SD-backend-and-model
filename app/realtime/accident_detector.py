# File role: Accident detection (Phase 8).
# Analyzes uploaded sensor samples for impact signatures - extreme acceleration
# spikes combined with corroborating signals (speed collapse, loss of movement,
# repeated impacts) - and publishes high-confidence accident alerts to the
# driver and the fleet-wide admin channel.
#
# False-positive minimization: a single high-g sample (pothole, speed bump,
# phone drop) never alerts on its own. Confidence must clear a threshold that
# requires the impact to be corroborated by at least one kinematic consequence.
# Key symbols/vars: AccidentDetector, accident_detector.
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from app.realtime.hub import FLEET_GLOBAL_KEY, alert_hub

logger = logging.getLogger(__name__)

# Rolling window per trip (keeps detection cheap and bounded).
WINDOW_SIZE = 200
# Candidate impact: acceleration magnitude >= ~2.5 g.
IMPACT_LOW_ACCEL_MPS2 = 25.0
# Strong impact: >= ~4 g.
IMPACT_HIGH_ACCEL_MPS2 = 40.0
# Speed collapse corroboration: drop of >= 12 m/s (~43 km/h) within 3 s.
COLLAPSE_DV_MPS = 12.0
COLLAPSE_WINDOW_S = 3.0
# Loss-of-movement corroboration.
NO_MOVEMENT_SPEED_MPS = 1.5
NO_MOVEMENT_LOOKBACK_S = 15.0
# How long after an impact we look for corroborating signals.
CONFIRM_WINDOW_S = 15.0
# Confidence floor to notify.
CONFIDENCE_THRESHOLD = 0.70
# Don't re-alert the same trip within this window.
TRIP_ALERT_COOLDOWN_S = 90.0
# Replay buffer caps.
RECENT_ACCIDENTS_CAP = 20


def _a_mag(row: dict) -> float:
    values = [row.get("ax"), row.get("ay"), row.get("az")]
    if all(v is None for v in values):
        return 0.0
    return (sum((float(v) if v is not None else 0.0) ** 2 for v in values)) ** 0.5


def _ts_to_seconds(value) -> Optional[float]:
    """Convert an ISO/datetime timestamp to epoch seconds (UTC)."""
    if value is None:
        return None
    try:
        if isinstance(value, datetime):
            dt = value
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        if isinstance(value, str):
            normalized = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
    except (ValueError, TypeError):
        return None
    return None


class AccidentDetector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # trip_id -> list of raw sample dicts (newest last)
        self._windows: Dict[str, List[dict]] = {}
        # trip_id -> set of impact timestamps already evaluated
        self._evaluated_impacts: Dict[str, set] = {}
        # trip_id -> last alert time (cooldown)
        self._last_alert_at: Dict[str, float] = {}
        # fleet-wide recent accident alerts (newest last)
        self._recent: List[dict] = []

    def process_upload(
        self,
        *,
        user_id: str,
        trip_id: str,
        rows: List[dict],
    ) -> List[dict]:
        """Scan a newly-uploaded batch for accident signatures and publish any
        high-confidence accident alerts. Best-effort - never raises into the
        upload path. Returns the alerts published."""
        if not rows:
            return []

        with self._lock:
            window = self._windows.setdefault(trip_id, [])
            for row in rows:
                window.append(
                    {
                        "ts": row.get("ts", row.get("timestamp")),
                        "speed_mps": row.get("speed_mps", row.get("speed")),
                        "lat": row.get("lat"),
                        "lon": row.get("lon"),
                        "ax": row.get("ax"),
                        "ay": row.get("ay"),
                        "az": row.get("az"),
                    }
                )
            if len(window) > WINDOW_SIZE:
                del window[: len(window) - WINDOW_SIZE]

            try:
                alert = self._evaluate(window, trip_id)
            except Exception:
                logger.exception("Accident detection failed (trip %s)", trip_id)
                alert = None

        if alert is None:
            return []

        # Publish outside the lock.
        for key in (user_id, FLEET_GLOBAL_KEY):
            try:
                alert_hub.publish(key, alert)
            except Exception:
                logger.exception("Failed to publish accident alert (key %s)", key)
        return [alert]

    def _evaluate(self, window: List[dict], trip_id: str) -> Optional[dict]:
        evaluated = self._evaluated_impacts.setdefault(trip_id, set())

        # Locate the most recent candidate impact we have not yet evaluated.
        candidate: Optional[dict] = None
        for row in reversed(window):
            if _a_mag(row) >= IMPACT_LOW_ACCEL_MPS2:
                impact_ts = _ts_to_seconds(row.get("ts"))
                if impact_ts is not None and impact_ts not in evaluated:
                    candidate = row
                    break
        if candidate is None:
            return None

        impact_ts = _ts_to_seconds(candidate.get("ts")) or 0.0
        evaluated.add(impact_ts)

        # Corroborating signals inside the confirm window after the impact.
        confirm_end = impact_ts + CONFIRM_WINDOW_S
        after = [row for row in window if (_ts_to_seconds(row.get("ts")) or 0.0) >= impact_ts and (_ts_to_seconds(row.get("ts")) or 0.0) <= confirm_end]

        strong_impact = any(_a_mag(row) >= IMPACT_HIGH_ACCEL_MPS2 for row in after)
        impacts_in_window = sum(1 for row in after if _a_mag(row) >= IMPACT_LOW_ACCEL_MPS2)
        repeated_impacts = impacts_in_window >= 2

        speed_collapse = False
        speeds = [(row.get("speed_mps"), _ts_to_seconds(row.get("ts")) or 0.0) for row in after]
        for i in range(1, len(speeds)):
            prev_speed, prev_ts = speeds[i - 1]
            curr_speed, curr_ts = speeds[i]
            if prev_speed is None or curr_speed is None:
                continue
            dt = curr_ts - prev_ts
            if 0 < dt <= COLLAPSE_WINDOW_S and (float(prev_speed) - float(curr_speed)) >= COLLAPSE_DV_MPS:
                speed_collapse = True
                break

        # Loss of movement: speed dropped below the threshold and stayed there
        # for the lookback window (checked against the oldest sample after impact).
        no_movement = False
        if speeds:
            moving_early = any(float(s) > NO_MOVEMENT_SPEED_MPS for s, _ in speeds[: max(1, len(speeds) // 3)])
            last_speed = speeds[-1][0]
            last_ts = speeds[-1][1]
            if moving_early and last_speed is not None and float(last_speed) <= NO_MOVEMENT_SPEED_MPS:
                if (last_ts - impact_ts) >= NO_MOVEMENT_LOOKBACK_S:
                    no_movement = True

        confidence = 0.40
        if strong_impact:
            confidence += 0.25
        if speed_collapse:
            confidence += 0.30
        if no_movement:
            confidence += 0.20
        if repeated_impacts:
            confidence += 0.20
        confidence = min(1.0, confidence)

        if confidence < CONFIDENCE_THRESHOLD:
            return None

        # Cooldown per trip so a single crash cluster doesn't spam admins.
        now = datetime.now(timezone.utc).timestamp()
        last_alert = self._last_alert_at.get(trip_id)
        if last_alert is not None and (now - last_alert) < TRIP_ALERT_COOLDOWN_S:
            return None
        self._last_alert_at[trip_id] = now

        alert = {
            "type": "accident_alert",
            "trip_id": trip_id,
            "occurred_at": datetime.fromtimestamp(impact_ts, tz=timezone.utc).isoformat(),
            "lat": candidate.get("lat"),
            "lon": candidate.get("lon"),
            "confidence": round(confidence, 2),
            "max_accel_mps2": round(max(_a_mag(row) for row in after), 1),
            "speed_at_impact_mps": candidate.get("speed_mps"),
            "signals": {
                "strong_impact": strong_impact,
                "speed_collapse": speed_collapse,
                "no_movement": no_movement,
                "repeated_impacts": repeated_impacts,
            },
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
        self._recent.append(alert)
        if len(self._recent) > RECENT_ACCIDENTS_CAP:
            del self._recent[: len(self._recent) - RECENT_ACCIDENTS_CAP]
        return alert

    def recent_accidents(self) -> List[dict]:
        with self._lock:
            return list(self._recent)

    def clear_trip(self, trip_id: str) -> None:
        with self._lock:
            self._windows.pop(trip_id, None)
            self._evaluated_impacts.pop(trip_id, None)
            self._last_alert_at.pop(trip_id, None)


accident_detector = AccidentDetector()

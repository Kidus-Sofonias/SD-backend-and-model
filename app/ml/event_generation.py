# File role: Rule-based driving event generation from trip-level and per-sample ML pipeline outputs.
# Produces persisted event instances suitable for route maps and user explanations.
# Connects to:
# - app.ml.pipeline outputs
# - pandas/numpy for per-sample event extraction
# Key symbols/vars:
# - generate_trip_events
# - build_human_reasons

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .braking import classify_brake_segment
from .event_utils import (
    apply_cooldown,
    event_segments,
    event_severity_and_confidence,
    filter_net_speed_delta,
)

UNSTABLE_MOTION_JERK_THRESHOLD = 2.5


def _isoformat_timestamp(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().isoformat().replace("+00:00", "Z")
    return str(value)


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def _build_event_payload(
    per: pd.DataFrame,
    *,
    index: int,
    event_type: str,
    value: float,
    severity: float,
    confidence: float,
    duration_s: float,
) -> dict:
    row = per.iloc[index]
    return {
        "event_type": event_type,
        "value": float(value),
        "severity": round(float(severity), 4),
        "confidence": round(float(confidence), 4),
        "duration_s": round(float(duration_s), 3),
        "occurred_at": _isoformat_timestamp(row.get("timestamp")),
        "lat": _float_or_none(row.get("lat")),
        "lon": _float_or_none(row.get("lon")),
    }


def _peak_index(values: np.ndarray, start: int, end: int, *, mode: str) -> int:
    window = values[start : end + 1]
    if mode == "min":
        return start + int(np.argmin(window))
    if mode == "max":
        return start + int(np.argmax(window))
    return start + int(np.argmax(np.abs(window)))


def generate_trip_events(
    per: pd.DataFrame,
    trip_features: dict,
    *,
    harsh_brake_dv: float,
    harsh_accel_dv: float,
    emergency_brake_dv: float,
    emergency_brake_min_speed_mps: float,
    aggressive_turn_threshold: float,
    turn_min_duration_s: float,
    min_event_duration_s: float,
    merge_gap_s: float,
    unstable_motion_jerk_threshold: float,
    overspeed_threshold_mps: float,
    overspeed_min_duration_s: float,
    severe_overspeed_threshold_mps: float,
    severe_overspeed_min_duration_s: float,
    event_cooldown_s: float = 0.0,
    dv_min_speed_delta_mps: float = 0.0,
    dv_single_sample_peak_mps2: float = 0.0,
    nominal_dt_s: float = 0.5,
    unstable_cooldown_s: float = 0.0,
    turn_min_speed_mps: float = 0.0,
    brake_severity_ref_mps2: float = 6.5,
    accel_severity_ref_mps2: float = 6.5,
    turn_severity_ref_mps2: float = 8.8,
    unstable_severity_ref_mps3: float = 6.0,
    overspeed_severity_ref_mps: float = 33.3,
    severe_overspeed_severity_ref_mps: float = 38.9,
) -> list[dict]:
    """
    Build persisted driving-event instances with their own timestamps and coordinates.

    Phase 3: detection uses the raw-signal columns produced by
    compute_per_sample_features (dv from raw speed, lateral_accel, jerk_mag_raw)
    and adds the overspeed categories.
    """
    if per.empty or not trip_features:
        return []

    t = per["t"].to_numpy()
    dv = per["dv"].to_numpy()
    speed = per["speed_s"].to_numpy()
    speed_raw = per["speed"].to_numpy(dtype=float)
    lateral = per["lateral_accel"].to_numpy()
    jerk_raw_norm = per["jerk_mag_raw_norm"].to_numpy()

    events: list[dict] = []

    # Phase 10: FILTER first (windowed net speed change + extreme-peak escape),
    # THEN cooldown (see features.py; kept identical so live detection matches
    # finalize). Rough road gets its own longer cooldown.
    harsh_brake_segments = filter_net_speed_delta(
        event_segments(dv < harsh_brake_dv, t, min_event_duration_s, merge_gap_s),
        speed_raw,
        direction=-1, min_delta_mps=dv_min_speed_delta_mps,
        extreme_peak_mps2=dv_single_sample_peak_mps2, nominal_dt_s=nominal_dt_s,
    )
    harsh_brake_segments = apply_cooldown(harsh_brake_segments, t, event_cooldown_s)
    for start, end in harsh_brake_segments:
        peak_idx = _peak_index(dv, start, end, mode="min")
        event_type = classify_brake_segment(
            dv,
            speed,
            start,
            end,
            emergency_brake_dv=emergency_brake_dv,
            emergency_brake_min_speed_mps=emergency_brake_min_speed_mps,
        )
        peak = abs(float(dv[peak_idx]))
        duration_s = max(0.0, float(t[end] - t[start]))
        severity, confidence = event_severity_and_confidence(
            event_type=event_type,
            value=peak,
            duration_s=duration_s,
            activation=abs(harsh_brake_dv),
            reference=brake_severity_ref_mps2,
        )
        events.append(
            _build_event_payload(
                per,
                index=peak_idx,
                event_type=event_type,
                value=peak,
                severity=severity,
                confidence=confidence,
                duration_s=duration_s,
            )
        )

    harsh_accel_segments = filter_net_speed_delta(
        event_segments(dv > harsh_accel_dv, t, min_event_duration_s, merge_gap_s),
        speed_raw,
        direction=1, min_delta_mps=dv_min_speed_delta_mps,
        extreme_peak_mps2=dv_single_sample_peak_mps2, nominal_dt_s=nominal_dt_s,
    )
    harsh_accel_segments = apply_cooldown(harsh_accel_segments, t, event_cooldown_s)
    for start, end in harsh_accel_segments:
        peak_idx = _peak_index(dv, start, end, mode="max")
        peak = float(dv[peak_idx])
        duration_s = max(0.0, float(t[end] - t[start]))
        severity, confidence = event_severity_and_confidence(
            event_type="hard_accel",
            value=peak,
            duration_s=duration_s,
            activation=abs(harsh_accel_dv),
            reference=accel_severity_ref_mps2,
        )
        events.append(
            _build_event_payload(
                per,
                index=peak_idx,
                event_type="hard_accel",
                value=peak,
                severity=severity,
                confidence=confidence,
                duration_s=duration_s,
            )
        )

    aggressive_turn_segments = event_segments(
        (lateral > aggressive_turn_threshold) & (speed_raw >= turn_min_speed_mps),
        t,
        turn_min_duration_s,
        merge_gap_s,
        event_cooldown_s,
    )
    for start, end in aggressive_turn_segments:
        peak_idx = _peak_index(lateral, start, end, mode="max")
        peak = float(lateral[peak_idx])
        duration_s = max(0.0, float(t[end] - t[start]))
        severity, confidence = event_severity_and_confidence(
            event_type="aggressive_turn",
            value=peak,
            duration_s=duration_s,
            activation=aggressive_turn_threshold,
            reference=turn_severity_ref_mps2,
        )
        events.append(
            _build_event_payload(
                per,
                index=peak_idx,
                event_type="aggressive_turn",
                value=peak,
                severity=severity,
                confidence=confidence,
                duration_s=duration_s,
            )
        )

    unstable_motion_segments = event_segments(
        jerk_raw_norm >= unstable_motion_jerk_threshold,
        t,
        min_event_duration_s,
        merge_gap_s,
        unstable_cooldown_s,
    )
    for start, end in unstable_motion_segments:
        peak_idx = _peak_index(jerk_raw_norm, start, end, mode="max")
        peak = float(jerk_raw_norm[peak_idx])
        duration_s = max(0.0, float(t[end] - t[start]))
        severity, confidence = event_severity_and_confidence(
            event_type="unstable_motion",
            value=peak,
            duration_s=duration_s,
            activation=unstable_motion_jerk_threshold,
            reference=unstable_severity_ref_mps3,
        )
        events.append(
            _build_event_payload(
                per,
                index=peak_idx,
                event_type="unstable_motion",
                value=peak,
                severity=severity,
                confidence=confidence,
                duration_s=duration_s,
            )
        )

    overspeed_segments = event_segments(
        speed_raw >= overspeed_threshold_mps,
        t,
        overspeed_min_duration_s,
        merge_gap_s,
        event_cooldown_s,
    )
    for start, end in overspeed_segments:
        peak_idx = _peak_index(speed_raw, start, end, mode="max")
        peak = float(speed_raw[peak_idx])
        duration_s = max(0.0, float(t[end] - t[start]))
        severity, confidence = event_severity_and_confidence(
            event_type="overspeed",
            value=peak,
            duration_s=duration_s,
            activation=overspeed_threshold_mps,
            reference=overspeed_severity_ref_mps,
        )
        events.append(
            _build_event_payload(
                per,
                index=peak_idx,
                event_type="overspeed",
                value=peak,
                severity=severity,
                confidence=confidence,
                duration_s=duration_s,
            )
        )

    severe_overspeed_segments = event_segments(
        speed_raw >= severe_overspeed_threshold_mps,
        t,
        severe_overspeed_min_duration_s,
        merge_gap_s,
        event_cooldown_s,
    )
    for start, end in severe_overspeed_segments:
        peak_idx = _peak_index(speed_raw, start, end, mode="max")
        peak = float(speed_raw[peak_idx])
        duration_s = max(0.0, float(t[end] - t[start]))
        severity, confidence = event_severity_and_confidence(
            event_type="severe_overspeed",
            value=peak,
            duration_s=duration_s,
            activation=severe_overspeed_threshold_mps,
            reference=severe_overspeed_severity_ref_mps,
        )
        events.append(
            _build_event_payload(
                per,
                index=peak_idx,
                event_type="severe_overspeed",
                value=peak,
                severity=severity,
                confidence=confidence,
                duration_s=duration_s,
            )
        )

    events.sort(key=lambda item: (item.get("occurred_at") or "", item["event_type"]))
    return events


def build_human_reasons(
    trip_features: dict,
    ml_prediction: int | None,
    ml_risk_probability: float | None,
) -> list[str]:
    """
    Build short user-facing reasons explaining why a trip was scored as risky/safe.
    """
    reasons: list[str] = []

    if not trip_features:
        return ["Not enough usable trip data"]

    emergency_brake_count = int(trip_features.get("emergency_brake_count", 0))
    chargeable_hard_brake_count = int(trip_features.get("chargeable_hard_brake_count", 0))

    if emergency_brake_count > 0:
        reasons.append(f"Emergency braking detected ({emergency_brake_count})")

    if chargeable_hard_brake_count > 0:
        reasons.append(f"Hard braking detected ({chargeable_hard_brake_count})")

    if int(trip_features.get("harsh_accel_count", 0)) > 0:
        reasons.append(f"Harsh acceleration detected ({trip_features['harsh_accel_count']})")

    if int(trip_features.get("aggressive_turn_count", 0)) > 0:
        reasons.append(f"Aggressive turning detected ({trip_features['aggressive_turn_count']})")

    if int(trip_features.get("overspeed_count", 0)) > 0:
        reasons.append(f"Overspeeding detected ({trip_features['overspeed_count']})")

    if int(trip_features.get("severe_overspeed_count", 0)) > 0:
        reasons.append(f"Severe overspeeding detected ({trip_features['severe_overspeed_count']})")

    if int(trip_features.get("unstable_motion_count", 0)) > 0:
        reasons.append(f"Rough road / unstable motion detected ({trip_features['unstable_motion_count']})")

    if float(trip_features.get("p95_jerk", 0.0)) >= UNSTABLE_MOTION_JERK_THRESHOLD:
        reasons.append("Trip motion was not smooth")

    if float(trip_features.get("speed_variance", 0.0)) >= 20:
        reasons.append("Speed changed sharply during the trip")

    if ml_prediction == 1 and ml_risk_probability is not None:
        reasons.append(f"ML risk confidence: {ml_risk_probability:.2f}")

    if not reasons:
        reasons.append("Trip looked smooth overall")

    return reasons

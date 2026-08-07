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
from .event_utils import event_segments

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
) -> dict:
    row = per.iloc[index]
    return {
        "event_type": event_type,
        "value": float(value),
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
    jerk_raw = per["jerk_mag_raw"].to_numpy()

    events: list[dict] = []

    harsh_brake_segments = event_segments(dv < harsh_brake_dv, t, min_event_duration_s, merge_gap_s)
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
        events.append(
            _build_event_payload(
                per,
                index=peak_idx,
                event_type=event_type,
                value=abs(float(dv[peak_idx])),
            )
        )

    harsh_accel_segments = event_segments(dv > harsh_accel_dv, t, min_event_duration_s, merge_gap_s)
    for start, end in harsh_accel_segments:
        peak_idx = _peak_index(dv, start, end, mode="max")
        events.append(
            _build_event_payload(
                per,
                index=peak_idx,
                event_type="hard_accel",
                value=float(dv[peak_idx]),
            )
        )

    aggressive_turn_segments = event_segments(lateral > aggressive_turn_threshold, t, turn_min_duration_s, merge_gap_s)
    for start, end in aggressive_turn_segments:
        peak_idx = _peak_index(lateral, start, end, mode="max")
        events.append(
            _build_event_payload(
                per,
                index=peak_idx,
                event_type="aggressive_turn",
                value=float(lateral[peak_idx]),
            )
        )

    unstable_motion_segments = event_segments(
        jerk_raw >= unstable_motion_jerk_threshold,
        t,
        min_event_duration_s,
        merge_gap_s,
    )
    for start, end in unstable_motion_segments:
        peak_idx = _peak_index(jerk_raw, start, end, mode="max")
        events.append(
            _build_event_payload(
                per,
                index=peak_idx,
                event_type="unstable_motion",
                value=float(jerk_raw[peak_idx]),
            )
        )

    overspeed_segments = event_segments(
        speed_raw >= overspeed_threshold_mps,
        t,
        overspeed_min_duration_s,
        merge_gap_s,
    )
    for start, end in overspeed_segments:
        peak_idx = _peak_index(speed_raw, start, end, mode="max")
        events.append(
            _build_event_payload(
                per,
                index=peak_idx,
                event_type="overspeed",
                value=float(speed_raw[peak_idx]),
            )
        )

    severe_overspeed_segments = event_segments(
        speed_raw >= severe_overspeed_threshold_mps,
        t,
        severe_overspeed_min_duration_s,
        merge_gap_s,
    )
    for start, end in severe_overspeed_segments:
        peak_idx = _peak_index(speed_raw, start, end, mode="max")
        events.append(
            _build_event_payload(
                per,
                index=peak_idx,
                event_type="severe_overspeed",
                value=float(speed_raw[peak_idx]),
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

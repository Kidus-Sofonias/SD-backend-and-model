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

UNSTABLE_MOTION_JERK_THRESHOLD = 0.12
SPEED_VARIATION_DV_THRESHOLD = 2.25


def _event_segments(
    mask: np.ndarray,
    timestamps: np.ndarray,
    min_duration_s: float,
    merge_gap_s: float,
) -> list[tuple[int, int]]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []

    groups: list[tuple[int, int]] = []
    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            groups.append((start, prev))
            start = i
            prev = i
    groups.append((start, prev))

    merged: list[list[int]] = []
    for s, e in groups:
        if not merged:
            merged.append([s, e])
            continue

        _, prev_e = merged[-1]
        gap = timestamps[s] - timestamps[prev_e]
        if gap <= merge_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    segments: list[tuple[int, int]] = []
    for s, e in merged:
        duration = timestamps[e] - timestamps[s]
        if duration >= min_duration_s:
            segments.append((s, e))

    return segments


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
    min_event_duration_s: float,
    merge_gap_s: float,
) -> list[dict]:
    """
    Build persisted driving-event instances with their own timestamps and coordinates.
    """
    if per.empty or not trip_features:
        return []

    t = per["t"].to_numpy()
    dv = per["dv"].to_numpy()
    speed = per["speed_s"].to_numpy()
    turn = per["turn_intensity"].to_numpy()
    jerk = per["jerk_mag"].to_numpy()

    events: list[dict] = []

    harsh_brake_segments = _event_segments(dv < harsh_brake_dv, t, min_event_duration_s, merge_gap_s)
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

    harsh_accel_segments = _event_segments(dv > harsh_accel_dv, t, min_event_duration_s, merge_gap_s)
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

    aggressive_turn_segments = _event_segments(turn > aggressive_turn_threshold, t, min_event_duration_s, merge_gap_s)
    for start, end in aggressive_turn_segments:
        peak_idx = _peak_index(turn, start, end, mode="max")
        events.append(
            _build_event_payload(
                per,
                index=peak_idx,
                event_type="aggressive_turn",
                value=float(turn[peak_idx]),
            )
        )

    unstable_motion_segments = _event_segments(jerk >= UNSTABLE_MOTION_JERK_THRESHOLD, t, min_event_duration_s, merge_gap_s)
    for start, end in unstable_motion_segments:
        peak_idx = _peak_index(jerk, start, end, mode="max")
        events.append(
            _build_event_payload(
                per,
                index=peak_idx,
                event_type="unstable_motion",
                value=float(jerk[peak_idx]),
            )
        )

    speed_variation_segments = _event_segments(np.abs(dv) >= SPEED_VARIATION_DV_THRESHOLD, t, min_event_duration_s, merge_gap_s)
    for start, end in speed_variation_segments:
        peak_idx = _peak_index(dv, start, end, mode="abs")
        events.append(
            _build_event_payload(
                per,
                index=peak_idx,
                event_type="speed_variation",
                value=abs(float(dv[peak_idx])),
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
    chargeable_hard_brake_count = int(
        trip_features.get(
            "chargeable_hard_brake_count",
            max(0, int(trip_features.get("harsh_brake_count", 0)) - emergency_brake_count),
        )
    )

    if emergency_brake_count > 0:
        reasons.append(
            f"Emergency braking detected ({emergency_brake_count}); treated as an emergency safety response"
        )

    if chargeable_hard_brake_count > 0:
        reasons.append(f"Hard braking detected ({chargeable_hard_brake_count})")

    if int(trip_features.get("harsh_accel_count", 0)) > 0:
        reasons.append(f"Harsh acceleration detected ({trip_features['harsh_accel_count']})")

    if int(trip_features.get("aggressive_turn_count", 0)) > 0:
        reasons.append(f"Aggressive turning detected ({trip_features['aggressive_turn_count']})")

    if float(trip_features.get("p95_jerk", 0.0)) >= UNSTABLE_MOTION_JERK_THRESHOLD:
        reasons.append("Trip motion was not smooth")

    if float(trip_features.get("speed_variance", 0.0)) >= 20:
        reasons.append("Speed changed sharply during the trip")

    if ml_prediction == 1 and ml_risk_probability is not None:
        reasons.append(f"ML risk confidence: {ml_risk_probability:.2f}")

    if not reasons:
        reasons.append("Trip looked smooth overall")

    return reasons

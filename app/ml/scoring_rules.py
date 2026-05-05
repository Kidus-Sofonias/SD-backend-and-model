# File role: Rule-based baseline scoring module for trip behavior assessment.
# Converts aggregated trip features into a simple interpretable score and penalty breakdown.
# Connects to: app.ml.pipeline and later backend fallback scoring.
# Key symbols/vars:
# - score_trip_rules_v1

from __future__ import annotations

import numpy as np


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def score_trip_rules_v1(
    trip_features: dict,
    w_brake: float,
    w_accel: float,
    w_turn: float,
    w_jerk: float,
    w_speed_var: float,
) -> dict:
    """
    Produce a baseline trip score in the range 0..100.
    Lower-quality driving behavior increases penalties and reduces score.
    """
    jerk_penalty = w_jerk * _normalize(trip_features["p95_jerk"], 0.5, 6.0)
    speed_var_penalty = w_speed_var * _normalize(trip_features["speed_variance"], 0.0, 25.0)

    emergency_brake_count = int(trip_features.get("emergency_brake_count", 0))
    chargeable_brake_count = int(
        trip_features.get(
            "chargeable_hard_brake_count",
            max(0, int(trip_features["harsh_brake_count"]) - emergency_brake_count),
        )
    )
    brake_penalty = w_brake * chargeable_brake_count
    accel_penalty = w_accel * trip_features["harsh_accel_count"]
    turn_penalty = w_turn * trip_features["aggressive_turn_count"]

    total_penalty = brake_penalty + accel_penalty + turn_penalty + jerk_penalty + speed_var_penalty
    score = int(np.clip(round(100 - total_penalty), 0, 100))

    return {
        "score": score,
        "penalties": {
            "harsh_brake": brake_penalty,
            "emergency_brake": 0.0,
            "harsh_accel": accel_penalty,
            "aggressive_turn": turn_penalty,
            "jerk": jerk_penalty,
            "speed_variance": speed_var_penalty,
        },
        "trip_features": trip_features,
        "confidence": trip_features["confidence"],
    }

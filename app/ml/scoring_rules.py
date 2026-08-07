# File role: Rule-based scoring module for trip behavior assessment.
# Converts aggregated trip features into an interpretable 0..100 safety score.
# Connects to: app.ml.pipeline and later backend fallback scoring.
# Key symbols/vars:
# - score_trip_rules_v2

from __future__ import annotations

import numpy as np


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    value = np.nan_to_num(float(value), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def score_trip_rules_v2(
    trip_features: dict,
    w_emergency_brake: float,
    w_brake: float,
    w_accel: float,
    w_turn: float,
    w_overspeed: float,
    w_severe_overspeed: float,
    w_unstable_motion: float,
    w_jerk: float,
    w_speed_var: float,
    w_density: float,
    jerk_normalize_low: float,
    jerk_normalize_high: float,
    speed_var_normalize_high: float,
    density_normalize_low: float,
    density_normalize_high: float,
) -> dict:
    """Produce a trip safety score in the range 0..100 (Phase 3 / v2).

    Model:
        score = 100 - sum(per-event penalties) - smoothness penalties
                - event-density penalty

    Design notes (Phase 3):
    - Emergency braking is CHARGEABLE (w_emergency_brake): it is a risk
      indicator (following distance, approach speed), not an excuse. The old
      scorer exempted it, so trips full of emergency stops scored ~80.
    - Penalties are fixed per event type, and a density term
      (w_density * normalize(events_per_hour)) keeps scores consistent across
      short and long trips: one hard brake in 10 minutes costs more than one in
      two hours.
    - Smoothness penalties (jerk percentile, speed variance) stay bounded via
      normalization, so noisy phone data can only contribute a limited share.
    """
    confidence = float(trip_features.get("confidence", 0.0))

    penalties = {
        "emergency_brake": w_emergency_brake * int(trip_features.get("emergency_brake_count", 0)),
        "harsh_brake": w_brake * int(trip_features.get("chargeable_hard_brake_count", 0)),
        "harsh_accel": w_accel * int(trip_features.get("harsh_accel_count", 0)),
        "aggressive_turn": w_turn * int(trip_features.get("aggressive_turn_count", 0)),
        "overspeed": w_overspeed * int(trip_features.get("overspeed_count", 0)),
        "severe_overspeed": w_severe_overspeed * int(trip_features.get("severe_overspeed_count", 0)),
        "unstable_motion": w_unstable_motion * int(trip_features.get("unstable_motion_count", 0)),
        "jerk": w_jerk * _normalize(
            trip_features.get("p95_jerk", 0.0),
            jerk_normalize_low,
            jerk_normalize_high,
        ),
        "speed_variance": w_speed_var * _normalize(
            trip_features.get("speed_variance", 0.0),
            0.0,
            speed_var_normalize_high,
        ),
        "density": w_density * _normalize(
            trip_features.get("events_per_hour", 0.0),
            density_normalize_low,
            density_normalize_high,
        ),
    }

    total_penalty = float(sum(penalties.values()))
    score = int(np.clip(round(100 - total_penalty), 0, 100))

    return {
        "score": score,
        "penalties": penalties,
        "trip_features": trip_features,
        "confidence": confidence,
        "scoring_version": "v2",
    }

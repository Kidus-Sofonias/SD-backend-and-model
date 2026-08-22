# File role: Rule-based scoring module for trip behavior assessment.
# Converts aggregated trip features into an interpretable 0..100 safety score.
# Connects to: app.ml.pipeline and later backend fallback scoring.
# Key symbols/vars:
# - score_trip_rules_v3

from __future__ import annotations

import numpy as np


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    value = np.nan_to_num(float(value), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def score_trip_rules_v3(
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
    density_distance_normalize_high: float = 4.0,
    density_min_duration_s: float = 120.0,
) -> dict:
    """Produce a trip safety score in the range 0..100 (Phase 10 / v3).

    Model:
        score = 100 - sum(per-event impacts) - smoothness penalties
                - exposure penalty

    Phase 10 (hackathon) changes vs v2:
    - Per-event penalty = weight x IMPACT, where impact = severity x
      confidence-factor (from ``trip_features['event_impacts']``). A noisy
      3.3 m/s^2 spike with low confidence costs far less than a sustained
      6.4 m/s^2 emergency brake. Falls back to count x weight for legacy
      feature dicts without impacts.
    - Exposure is normalised by DISTANCE driven (events per km) when GPS moved,
      with a time-based fallback that uses a reduced duration floor so short
      trips are not catastrophically penalised by per-hour rates.
    - Emergency braking remains CHARGEABLE (a following-distance / approach
      speed risk indicator).
    - Smoothness penalties (jerk percentile, speed variance) stay bounded via
      normalization.
    """
    confidence = float(trip_features.get("confidence", 0.0))
    impacts = trip_features.get("event_impacts") or {}

    def _cat_penalty(category: str, weight: float, count_key: str) -> float:
        impact = float(impacts.get(category, 0.0) or 0.0)
        if impact > 0:
            return weight * impact
        # Legacy fallback: feature dicts without impacts use count x weight.
        return weight * int(trip_features.get(count_key, 0))

    distance_km = float(trip_features.get("distance_km", 0.0) or 0.0)
    events_per_km = float(trip_features.get("events_per_km", 0.0) or 0.0)
    if distance_km >= 0.05 and events_per_km > 0:
        density_value = events_per_km
        density_high = density_distance_normalize_high
    else:
        density_value = float(trip_features.get("events_per_hour", 0.0) or 0.0)
        density_high = density_normalize_high

    penalties = {
        "emergency_brake": _cat_penalty("emergency_brake", w_emergency_brake, "emergency_brake_count"),
        "harsh_brake": _cat_penalty("hard_brake", w_brake, "chargeable_hard_brake_count"),
        "harsh_accel": _cat_penalty("hard_accel", w_accel, "harsh_accel_count"),
        "aggressive_turn": _cat_penalty("aggressive_turn", w_turn, "aggressive_turn_count"),
        "overspeed": _cat_penalty("overspeed", w_overspeed, "overspeed_count"),
        "severe_overspeed": _cat_penalty("severe_overspeed", w_severe_overspeed, "severe_overspeed_count"),
        "unstable_motion": _cat_penalty("unstable_motion", w_unstable_motion, "unstable_motion_count"),
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
            density_value,
            density_normalize_low,
            density_high,
        ),
    }

    total_penalty = float(sum(penalties.values()))
    score = int(np.clip(round(100 - total_penalty), 0, 100))

    return {
        "score": score,
        "penalties": penalties,
        "trip_features": trip_features,
        "confidence": confidence,
        "scoring_version": "v3",
    }

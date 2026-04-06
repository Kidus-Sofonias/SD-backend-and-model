# File role: Rule-based driving event generation from trip-level and per-sample ML pipeline outputs.
# Produces event summaries suitable for persistence in DrivingEvent and for frontend explanations.
# Connects to:
# - app.ml.config
# - app.ml.features / app.ml.pipeline outputs
# Key symbols/vars:
# - generate_trip_events
# - build_human_reasons

from __future__ import annotations


def generate_trip_events(trip_features: dict) -> list[dict]:
    """
    Convert trip-level features into persisted driving events.

    Output format:
    [
        {"event_type": "...", "value": float},
        ...
    ]
    """
    if not trip_features:
        return []

    events: list[dict] = []

    harsh_brake_count = int(trip_features.get("harsh_brake_count", 0))
    harsh_accel_count = int(trip_features.get("harsh_accel_count", 0))
    aggressive_turn_count = int(trip_features.get("aggressive_turn_count", 0))
    p95_jerk = float(trip_features.get("p95_jerk", 0.0))
    speed_variance = float(trip_features.get("speed_variance", 0.0))

    if harsh_brake_count > 0:
        events.append({"event_type": "hard_brake", "value": float(harsh_brake_count)})

    if harsh_accel_count > 0:
        events.append({"event_type": "hard_accel", "value": float(harsh_accel_count)})

    if aggressive_turn_count > 0:
        events.append({"event_type": "aggressive_turn", "value": float(aggressive_turn_count)})

    # derived stability event
    if p95_jerk >= 0.12:
        events.append({"event_type": "unstable_motion", "value": p95_jerk})

    if speed_variance >= 20:
        events.append({"event_type": "speed_variation", "value": speed_variance})

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

    if int(trip_features.get("harsh_brake_count", 0)) > 0:
        reasons.append(f"Hard braking detected ({trip_features['harsh_brake_count']})")

    if int(trip_features.get("harsh_accel_count", 0)) > 0:
        reasons.append(f"Harsh acceleration detected ({trip_features['harsh_accel_count']})")

    if int(trip_features.get("aggressive_turn_count", 0)) > 0:
        reasons.append(f"Aggressive turning detected ({trip_features['aggressive_turn_count']})")

    if float(trip_features.get("p95_jerk", 0.0)) >= 0.12:
        reasons.append("Trip motion was not smooth")

    if float(trip_features.get("speed_variance", 0.0)) >= 20:
        reasons.append("Speed changed sharply during the trip")

    if ml_prediction == 1 and ml_risk_probability is not None:
        reasons.append(f"ML risk confidence: {ml_risk_probability:.2f}")

    if not reasons:
        reasons.append("Trip looked smooth overall")

    return reasons
# File role: Shared ML feature contract module.
# Defines the stable feature version and ordered feature columns used by:
# - dataset building
# - model training
# - model inference in backend
# Connects to: app.ml.training/inference-related modules via imports.
# Key symbols/vars:
# - FEATURE_VERSION
# - FEATURE_COLUMNS_FV1

FEATURE_VERSION = "fv1"
MODEL_VERSION_RULES_V1 = "rules_v1"

FEATURE_COLUMNS_FV1 = [
    "duration_s",
    "n_samples",
    "mean_speed_mps",
    "max_speed_mps",
    "speed_variance",
    "p95_jerk",
    "max_jerk",
    "harsh_brake_count",
    "harsh_accel_count",
    "aggressive_turn_count",
    "confidence",
    # Sequence-based features: distributional + temporal shape that the rule
    # scorer does not use (rules only look at jerk percentiles and event counts).
    "jerk_entropy",
    "mean_event_duration_s",
    "max_event_duration_s",
    "max_consecutive_event_run_s",
    # Phase 8b (hackathon): vehicle-aware context. The log-mass of the trip's
    # vehicle lets the model condition its risk reading on the vehicle class:
    # the same event counts mean different things for a sedan (1400 kg) and a
    # 36 t tractor-trailer, because detection thresholds are tuned per vehicle
    # (config_for_profile). Log scale keeps the 1400..36000 kg range tractable
    # for both scaled-LR and tree models.
    "log_vehicle_mass_kg",
]
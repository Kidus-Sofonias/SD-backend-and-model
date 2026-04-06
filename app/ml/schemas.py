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
]
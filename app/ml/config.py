# File role: ML/scoring pipeline component used during trip finalization to derive features, score, and confidence.
# Connects to: nearby package modules via local imports.
# Key symbols/vars: FeatureConfigV1.
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfigV1:
    # data handling
    max_gap_s: float = 3.0
    ema_alpha: float = 0.25

    # IMPORTANT: your app currently sends speed in km/h
    input_speed_unit: str = "kmh"

    # thresholds in m/s^2
    harsh_brake_dv: float = -3.0
    harsh_accel_dv: float = 3.0

    min_event_duration_s: float = 0.30
    merge_gap_s: float = 0.20

    # turning threshold (gyro magnitude proxy)
    aggressive_turn_threshold: float = 2.5

    # baseline score penalties
    w_brake: float = 8.0
    w_accel: float = 6.0
    w_turn: float = 6.0
    w_jerk: float = 10.0
    w_speed_var: float = 6.0
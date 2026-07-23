# File role: ML/scoring pipeline component used during trip finalization to derive features, score, and confidence.
# Connects to: nearby package modules via local imports.
# Key symbols/vars: FeatureConfigV2 (tuned for better cross-phone accuracy).
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfigV2:
    """
    V2: Tuned for broader phone compatibility, higher sensitivity, and more samples.
    
    Key changes from V1:
    - More sensitive to brake/accel events (lower thresholds)
    - More aggressive turn detection (lower gyro threshold)
    - Tighter gap handling to reject sparse/sporadic data
    - Slightly higher penalties for risky behaviors
    - Lower minimum samples to score short trips
    """
    
    # data handling
    max_gap_s: float = 2.5
    ema_alpha: float = 0.30
    min_samples_for_scoring: int = 2

    # IMPORTANT: your app currently sends speed in km/h
    input_speed_unit: str = "kmh"

    # thresholds in m/s^2
    harsh_brake_dv: float = -2.5
    harsh_accel_dv: float = 2.5
    emergency_brake_dv: float = -5.0
    emergency_brake_min_speed_mps: float = 6.0

    min_event_duration_s: float = 0.25
    merge_gap_s: float = 0.15

    # turning threshold (gyro magnitude proxy)
    aggressive_turn_threshold: float = 2.0

    # baseline score penalties
    w_brake: float = 10.0
    w_accel: float = 7.0
    w_turn: float = 7.0
    w_jerk: float = 12.0
    w_speed_var: float = 7.0

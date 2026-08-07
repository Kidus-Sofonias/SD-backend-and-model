# File role: ML/scoring pipeline component used during trip finalization to derive features, score, and confidence.
# Connects to: nearby package modules via local imports.
# Key symbols/vars: FeatureConfigV2 (tuned for better cross-phone accuracy).
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfigV2:
    """
    V3 (Phase 3): Redesigned event detection and scoring based on internationally
    accepted telematics norms.

    Detection redesign (vs V2):
    - Events are detected on RAW speed/IMU signals (EMA smoothing is kept only for
      trip-level model features), fixing the EMA alpha=0.3 attenuation that hid
      real braking/acceleration events.
    - Thresholds follow common telematics practice:
        * harsh braking/acceleration: >= 0.33 g (3.2 m/s^2) sustained
        * emergency braking:         >= 0.66 g (6.5 m/s^2) at speed
        * aggressive cornering:      >= 0.31 g lateral acceleration (v * yaw-rate)
        * rough-road / unstable:     jerk >= 2.5 m/s^3
        * overspeed:                 >= 100 km/h sustained >= 10 s
        * severe overspeed:          >= 130 km/h sustained >= 5 s
    - Emergency braking is now CHARGEABLE (it is a risk indicator, e.g. following
      too close), so risky trips no longer score near-clean.

    Scoring redesign:
    - Score = 100 - per-event penalties - smoothness penalties - event-density penalty.
    - Per-event penalties are fixed per type; a density term penalizes event
      frequency per hour so scores are consistent across trip durations.
    - Risk bands: >= 85 low, 65-84 medium, < 65 high.
    """
    
    # data handling
    max_gap_s: float = 10.0
    ema_alpha: float = 0.30
    min_samples_for_scoring: int = 10

    # Speed is stored in m/s in the database (speed_mps column); the pipeline
    # receives km/h and converts to m/s internally.
    input_speed_unit: str = "kmh"

    # --- Event detection thresholds ---
    # longitudinal acceleration/deceleration (m/s^2), measured on raw speed
    harsh_brake_dv: float = -3.2
    harsh_accel_dv: float = 3.2
    emergency_brake_dv: float = -6.5
    emergency_brake_min_speed_mps: float = 8.0

    # lateral acceleration proxy (m/s^2) = speed_mps * |yaw-rate|, ~0.45 g.
    # Turns must be SUSTAINED (turn_min_duration_s) — single-sample gyro noise
    # spikes at speed must not count as aggressive cornering. Note the EMA'd
    # yaw-rate asymptotes below the raw value, so the effective detection floor
    # is slightly above 0.45 g.
    aggressive_turn_threshold: float = 4.4
    turn_min_duration_s: float = 1.0

    # rough-road / unstable-motion jerk floor (m/s^3)
    unstable_motion_jerk_threshold: float = 2.5

    # overspeed (m/s): >= 100 km/h sustained, and severe >= 130 km/h sustained
    overspeed_threshold_mps: float = 27.8
    overspeed_min_duration_s: float = 10.0
    severe_overspeed_threshold_mps: float = 36.1
    severe_overspeed_min_duration_s: float = 5.0

    # segment filtering (interpreted relative to median sampling interval)
    min_event_duration_s: float = 0.25
    merge_gap_s: float = 0.15

    # --- Scoring weights (points per event / per normalized unit) ---
    w_emergency_brake: float = 8.0
    w_brake: float = 6.0
    w_accel: float = 5.0
    w_turn: float = 5.0
    w_overspeed: float = 4.0
    w_severe_overspeed: float = 8.0
    w_unstable_motion: float = 2.0
    w_jerk: float = 10.0
    w_speed_var: float = 6.0
    w_density: float = 10.0

    # density normalization bounds (chargeable events per hour)
    density_normalize_low: float = 0.0
    density_normalize_high: float = 24.0

    # jerk / variance normalization bounds
    jerk_normalize_low: float = 1.0
    jerk_normalize_high: float = 8.0
    speed_var_normalize_high: float = 30.0

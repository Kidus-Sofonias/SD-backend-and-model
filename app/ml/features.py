# File role: Feature engineering module for per-sample and per-trip driving features.
# Computes derived motion signals and aggregates them into trip-level features.
# Connects to: app.ml.pipeline and app.ml.scoring_rules.
# Key symbols/vars:
# - compute_per_sample_features
# - aggregate_trip_features

from __future__ import annotations

import numpy as np
import pandas as pd

from .braking import classify_brake_segment
from .event_utils import event_segments


def _shannon_entropy(values: np.ndarray, bins: int = 16, max_value: float = 6.0) -> float:
    """Shannon entropy (bits) of a jerk-magnitude distribution on an absolute scale.

    Uses fixed bin edges spanning [0, max_value] (max_value matches the rule
    scorer's p95-jerk upper normalization bound) so entropy is comparable across
    trips instead of being normalized to each trip's own range. Values above
    max_value are clipped into the top bin.

    A smooth trip where jerk is consistently ~0 concentrates in the first bin
    (entropy ~ 0); an erratic trip spreads across bins (higher entropy). The
    rule scorer only uses jerk percentiles, so this distributional shape is new
    information for the model.
    """
    if len(values) == 0:
        return 0.0
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, max_value)
    edges = np.linspace(0.0, max_value, bins + 1)
    hist, _ = np.histogram(clipped, bins=edges)
    probs = hist[hist > 0] / hist.sum()
    return float(-float(np.sum(probs * np.log2(probs))))


def _segment_durations(
    segments: list[tuple[int, int]],
    t: np.ndarray,
) -> list[float]:
    """Duration in seconds of each sustained event segment."""
    return [float(t[end] - t[start]) for start, end in segments]


def compute_per_sample_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived per-sample features used later for event detection and trip aggregation.

    Phase 3: detection signals are computed on RAW (unsmoothed) inputs where the
    old code used the EMA-smoothed stream. EMA (alpha=0.3) attenuated real braking
    peaks by ~50-70%, hiding genuine events; raw signals keep them visible. The
    smoothed streams remain the basis for trip-level model features (p95_jerk, ...)
    so the existing model inputs stay consistent.
    """
    out = df.copy()

    # Acceleration magnitude (smoothed inputs -> model features)
    out["a_mag"] = np.sqrt(out["ax_s"] ** 2 + out["ay_s"] ** 2 + out["az_s"] ** 2)

    # Gyroscope magnitude
    out["g_mag"] = np.sqrt(out["gx_s"] ** 2 + out["gy_s"] ** 2 + out["gz_s"] ** 2)

    # Jerk = change in acceleration magnitude over time (smoothed inputs)
    out["jerk"] = 0.0
    if len(out) > 1:
        out.loc[1:, "jerk"] = (
            (out["a_mag"].iloc[1:].to_numpy() - out["a_mag"].iloc[:-1].to_numpy())
            / np.maximum(out["dt"].iloc[1:].to_numpy(), 1e-6)
        )
    out["jerk_mag"] = np.abs(out["jerk"])

    # Raw jerk magnitude (unsmoothed) for rough-road / unstable-motion detection.
    # The old unstable_motion floor (0.12 m/s^3 on smoothed jerk) fired 20+ times
    # per trip on ordinary road noise; raw jerk with a 2.5 m/s^3 floor is a real
    # rough-road signal.
    jerk_raw = np.zeros(len(out))
    if len(out) > 1:
        a_raw = np.sqrt(
            out["ax"].to_numpy(dtype=float) ** 2
            + out["ay"].to_numpy(dtype=float) ** 2
            + out["az"].to_numpy(dtype=float) ** 2
        )
        jerk_raw[1:] = (a_raw[1:] - a_raw[:-1]) / np.maximum(out["dt"].iloc[1:].to_numpy(), 1e-6)
    out["jerk_mag_raw"] = np.abs(jerk_raw)

    # Longitudinal acceleration from RAW speed (primary braking/accel detection
    # signal; GPS speed quantization noise ~0.3 m/s^2 is far below thresholds).
    #
    # Zero-speed guard: some devices report 0 m/s when the GPS speed fix is
    # unavailable. A genuine stop keeps speed at 0 for several consecutive
    # samples; an isolated 0 (or a 0->moving recovery in < 3 samples) is a
    # measurement artifact that would otherwise look like a ~1.4 g deceleration.
    # Such samples are treated as invalid (NaN) so they never produce events.
    speed_raw = out["speed"].to_numpy(dtype=float).astype(float)
    zero_mask = speed_raw == 0.0
    # Label each zero sample with the length of the contiguous zero-run it belongs
    # to. A run of >= 3 consecutive zeros is a genuine stop; shorter runs are
    # measurement artifacts (missing GPS speed reported as 0) and are treated as
    # invalid so they can never generate events.
    zero_run_len = np.zeros(len(speed_raw), dtype=int)
    zero_indices = np.flatnonzero(zero_mask)
    if len(zero_indices):
        boundaries = np.flatnonzero(np.diff(zero_indices) > 1) + 1
        for group in np.split(zero_indices, boundaries):
            if len(group):
                zero_run_len[group] = len(group)
    invalid_speed = zero_mask & (zero_run_len < 3)
    speed_clean = np.where(invalid_speed, np.nan, speed_raw)

    out["dv"] = 0.0
    if len(out) > 1:
        dv = (speed_clean[1:] - speed_clean[:-1]) / np.maximum(out["dt"].iloc[1:].to_numpy(), 1e-6)
        # Clip to physically plausible bounds (~1.2 g): larger values are data
        # artifacts, not vehicle motion.
        dv = np.clip(dv, -11.8, 11.8)
        out.loc[1:, "dv"] = dv

    # Lateral acceleration proxy = speed * |yaw-rate| (~v * omega_z). This
    # replaces the old raw |gz| >= 2.0 rad/s test, which required unrealistically
    # violent rotations and essentially never fired on real trips. With speed in
    # the formula, a normal turn at speed produces meaningful lateral g.
    out["lateral_accel"] = np.abs(out["speed_s"].to_numpy(dtype=float)) * np.abs(out["gz_s"].to_numpy(dtype=float))

    return out


def aggregate_trip_features(
    per: pd.DataFrame,
    harsh_brake_dv: float,
    harsh_accel_dv: float,
    emergency_brake_dv: float,
    emergency_brake_min_speed_mps: float,
    aggressive_turn_threshold: float,
    turn_min_duration_s: float,
    min_event_duration_s: float,
    merge_gap_s: float,
    unstable_motion_jerk_threshold: float,
    overspeed_threshold_mps: float,
    overspeed_min_duration_s: float,
    severe_overspeed_threshold_mps: float,
    severe_overspeed_min_duration_s: float,
) -> dict:
    """
    Aggregate per-sample features into one training/inference row per trip.
    """
    if per.empty:
        return {}

    t = per["t"].to_numpy()
    speed = per["speed_s"].to_numpy()
    speed_raw = per["speed"].to_numpy(dtype=float)
    dt = per["dt"].to_numpy()
    dv = per["dv"].to_numpy()

    harsh_brake_mask = dv < harsh_brake_dv
    harsh_accel_mask = dv > harsh_accel_dv
    aggressive_turn_mask = per["lateral_accel"].to_numpy() > aggressive_turn_threshold
    unstable_motion_mask = per["jerk_mag_raw"].to_numpy() >= unstable_motion_jerk_threshold
    # Overspeed uses RAW speed (the EMA-smoothed stream lags behind the limit
    # and would undercount sustained high-speed stretches).
    overspeed_mask = speed_raw >= overspeed_threshold_mps
    severe_overspeed_mask = speed_raw >= severe_overspeed_threshold_mps

    harsh_brake_segments = event_segments(harsh_brake_mask, t, min_event_duration_s, merge_gap_s)
    harsh_accel_segments = event_segments(harsh_accel_mask, t, min_event_duration_s, merge_gap_s)
    # Turns need to be sustained (turn_min_duration_s) so single-sample gyro
    # noise spikes do not count as aggressive cornering.
    aggressive_turn_segments = event_segments(aggressive_turn_mask, t, turn_min_duration_s, merge_gap_s)
    unstable_motion_segments = event_segments(unstable_motion_mask, t, min_event_duration_s, merge_gap_s)
    overspeed_segments = event_segments(overspeed_mask, t, overspeed_min_duration_s, merge_gap_s)
    severe_overspeed_segments = event_segments(severe_overspeed_mask, t, severe_overspeed_min_duration_s, merge_gap_s)

    harsh_brake_count = len(harsh_brake_segments)
    harsh_accel_count = len(harsh_accel_segments)
    aggressive_turn_count = len(aggressive_turn_segments)
    unstable_motion_count = len(unstable_motion_segments)
    overspeed_count = len(overspeed_segments)
    severe_overspeed_count = len(severe_overspeed_segments)

    # --- Sequence-based features (signal the rule scorer does not use) ---
    event_durations = (
        _segment_durations(harsh_brake_segments, t)
        + _segment_durations(harsh_accel_segments, t)
        + _segment_durations(aggressive_turn_segments, t)
    )
    mean_event_duration_s = float(np.mean(event_durations)) if event_durations else 0.0
    max_event_duration_s = float(np.max(event_durations)) if event_durations else 0.0

    # Longest contiguous run where ANY longitudinal/lateral event condition held.
    any_event_segments = event_segments(
        harsh_brake_mask | harsh_accel_mask | aggressive_turn_mask,
        t,
        min_event_duration_s,
        merge_gap_s,
    )
    max_consecutive_event_run_s = (
        float(np.max([t[end] - t[start] for start, end in any_event_segments]))
        if any_event_segments
        else 0.0
    )

    jerk_entropy = _shannon_entropy(per["jerk_mag"].to_numpy())

    emergency_brake_count = sum(
        1
        for start, end in harsh_brake_segments
        if classify_brake_segment(
            dv,
            speed,
            start,
            end,
            emergency_brake_dv=emergency_brake_dv,
            emergency_brake_min_speed_mps=emergency_brake_min_speed_mps,
        )
        == "emergency_brake"
    )
    chargeable_hard_brake_count = max(0, harsh_brake_count - emergency_brake_count)

    duration_s = float(t[-1] - t[0]) if len(t) >= 2 else 0.0
    positive_dt = dt[dt > 0]
    max_gap_s = float(np.max(positive_dt)) if len(positive_dt) else 0.0
    median_dt_s = float(np.median(positive_dt)) if len(positive_dt) else 0.0

    confidence = 1.0
    if len(per) < 30:
        confidence -= 0.45
    if duration_s < 20:
        confidence -= 0.2
    if max_gap_s > 2.0:
        confidence -= 0.2
    if max_gap_s > 5.0:
        confidence -= 0.25
    confidence = float(max(0.0, min(1.0, confidence)))

    # Exposure normalization for scoring: events per hour (floor duration at 1 min
    # so very short trips do not produce absurd per-hour rates).
    total_chargeable_events = int(
        emergency_brake_count
        + chargeable_hard_brake_count
        + harsh_accel_count
        + aggressive_turn_count
        + overspeed_count
        + severe_overspeed_count
        + unstable_motion_count
    )
    hours = max(duration_s, 60.0) / 3600.0
    events_per_hour = float(total_chargeable_events / hours)

    return {
        "duration_s": duration_s,
        "n_samples": int(len(per)),
        "max_gap_s": max_gap_s,
        "median_dt_s": median_dt_s,
        "mean_speed_mps": float(np.nanmean(speed)) if len(speed) else 0.0,
        "max_speed_mps": float(np.nanmax(speed)) if len(speed) else 0.0,
        "speed_variance": float(np.nanvar(speed)) if len(speed) else 0.0,
        "p95_jerk": float(np.nanpercentile(per["jerk_mag"], 95)) if len(per) else 0.0,
        "max_jerk": float(np.nanmax(per["jerk_mag"])) if len(per) else 0.0,
        "harsh_brake_count": harsh_brake_count,
        "emergency_brake_count": int(min(harsh_brake_count, emergency_brake_count)),
        "chargeable_hard_brake_count": int(chargeable_hard_brake_count),
        "harsh_accel_count": harsh_accel_count,
        "aggressive_turn_count": aggressive_turn_count,
        "unstable_motion_count": unstable_motion_count,
        "overspeed_count": overspeed_count,
        "severe_overspeed_count": severe_overspeed_count,
        "total_chargeable_events": total_chargeable_events,
        "events_per_hour": events_per_hour,
        "confidence": confidence,
        "jerk_entropy": jerk_entropy,
        "mean_event_duration_s": mean_event_duration_s,
        "max_event_duration_s": max_event_duration_s,
        "max_consecutive_event_run_s": max_consecutive_event_run_s,
    }

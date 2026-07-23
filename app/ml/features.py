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
from .event_utils import count_events, event_segments


def compute_per_sample_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived per-sample features used later for event detection and trip aggregation.
    """
    out = df.copy()

    # Acceleration magnitude
    out["a_mag"] = np.sqrt(out["ax_s"] ** 2 + out["ay_s"] ** 2 + out["az_s"] ** 2)

    # Gyroscope magnitude
    out["g_mag"] = np.sqrt(out["gx_s"] ** 2 + out["gy_s"] ** 2 + out["gz_s"] ** 2)

    # Jerk = change in acceleration magnitude over time
    out["jerk"] = 0.0
    if len(out) > 1:
        out.loc[1:, "jerk"] = (
            (out["a_mag"].iloc[1:].to_numpy() - out["a_mag"].iloc[:-1].to_numpy())
            / np.maximum(out["dt"].iloc[1:].to_numpy(), 1e-6)
        )
    out["jerk_mag"] = np.abs(out["jerk"])

    # dv = change in speed over time
    out["dv"] = 0.0
    if len(out) > 1:
        out.loc[1:, "dv"] = (
            (out["speed_s"].iloc[1:].to_numpy() - out["speed_s"].iloc[:-1].to_numpy())
            / np.maximum(out["dt"].iloc[1:].to_numpy(), 1e-6)
        )

    # Turning proxy: absolute smoothed z-gyro
    out["turn_intensity"] = np.abs(out["gz_s"])

    return out



def aggregate_trip_features(
    per: pd.DataFrame,
    harsh_brake_dv: float,
    harsh_accel_dv: float,
    emergency_brake_dv: float,
    emergency_brake_min_speed_mps: float,
    aggressive_turn_threshold: float,
    min_event_duration_s: float,
    merge_gap_s: float,
) -> dict:
    """
    Aggregate per-sample features into one training/inference row per trip.
    """
    if per.empty:
        return {}

    t = per["t"].to_numpy()
    speed = per["speed_s"].to_numpy()
    dt = per["dt"].to_numpy()

    harsh_brake_mask = per["dv"].to_numpy() < harsh_brake_dv
    harsh_accel_mask = per["dv"].to_numpy() > harsh_accel_dv
    aggressive_turn_mask = per["turn_intensity"].to_numpy() > aggressive_turn_threshold
    harsh_brake_segments = event_segments(
        harsh_brake_mask,
        t,
        min_event_duration_s,
        merge_gap_s,
    )
    harsh_brake_count = len(harsh_brake_segments)
    harsh_accel_count = count_events(harsh_accel_mask, t, min_event_duration_s, merge_gap_s)
    aggressive_turn_count = count_events(aggressive_turn_mask, t, min_event_duration_s, merge_gap_s)

    emergency_brake_count = sum(
        1
        for start, end in harsh_brake_segments
        if classify_brake_segment(
            per["dv"].to_numpy(),
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

    return {
        "duration_s": duration_s,
        "n_samples": int(len(per)),
        "max_gap_s": max_gap_s,
        "median_dt_s": median_dt_s,
        "mean_speed_mps": float(np.mean(speed)),
        "max_speed_mps": float(np.max(speed)),
        "speed_variance": float(np.var(speed)),
        "p95_jerk": float(np.percentile(per["jerk_mag"], 95)),
        "max_jerk": float(np.max(per["jerk_mag"])),
        "harsh_brake_count": harsh_brake_count,
        "emergency_brake_count": int(min(harsh_brake_count, emergency_brake_count)),
        "chargeable_hard_brake_count": int(chargeable_hard_brake_count),
        "harsh_accel_count": harsh_accel_count,
        "aggressive_turn_count": aggressive_turn_count,
        "confidence": confidence,
    }

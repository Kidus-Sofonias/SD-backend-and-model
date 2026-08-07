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
    harsh_accel_segments = event_segments(harsh_accel_mask, t, min_event_duration_s, merge_gap_s)
    aggressive_turn_segments = event_segments(aggressive_turn_mask, t, min_event_duration_s, merge_gap_s)

    harsh_brake_count = len(harsh_brake_segments)
    harsh_accel_count = len(harsh_accel_segments)
    aggressive_turn_count = len(aggressive_turn_segments)

    # --- Sequence-based features (signal the rule scorer does not use) ---
    event_durations = (
        _segment_durations(harsh_brake_segments, t)
        + _segment_durations(harsh_accel_segments, t)
        + _segment_durations(aggressive_turn_segments, t)
    )
    mean_event_duration_s = float(np.mean(event_durations)) if event_durations else 0.0
    max_event_duration_s = float(np.max(event_durations)) if event_durations else 0.0

    # Longest contiguous run where ANY event condition held — the worst sustained incident.
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

    # NaN-safe aggregation: preprocessing guarantees finite speed/IMU values, but
    # keeping the aggregations NaN-proof protects scoring/model inference from any
    # future data path that slips through non-finite values (CRIT-2).
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
        "confidence": confidence,
        "jerk_entropy": jerk_entropy,
        "mean_event_duration_s": mean_event_duration_s,
        "max_event_duration_s": max_event_duration_s,
        "max_consecutive_event_run_s": max_consecutive_event_run_s,
    }

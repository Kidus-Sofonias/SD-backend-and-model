# File role: Raw sensor sample preprocessing module.
# Cleans and normalizes raw trip samples before feature extraction:
# - sorts timestamps
# - computes dt
# - drops invalid timing rows
# - converts speed to m/s
# - applies EMA smoothing
# Connects to: app.ml.pipeline and app.ml.features.
# Key symbols/vars:
# - ema
# - preprocess_samples

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def ema(series: np.ndarray, alpha: float) -> np.ndarray:
    """
    Exponential moving average smoothing.

    Why it exists:
    Sensor streams are noisy. EMA reduces spikes before jerk/event calculations.
    """
    if len(series) == 0:
        return series.astype(float)

    out = np.empty(len(series), dtype=float)
    out[0] = float(series[0])

    for i in range(1, len(series)):
        out[i] = alpha * float(series[i]) + (1 - alpha) * out[i - 1]

    return out


def _to_epoch_seconds(timestamps: pd.Series) -> pd.Series:
    """
    Convert timezone-aware pandas timestamps into Unix seconds explicitly.

    Using timedeltas keeps this resilient to datetime storage/resolution quirks
    that can otherwise leak millisecond/microsecond scaling into downstream dt.
    """
    epoch = pd.Timestamp("1970-01-01T00:00:00Z")
    return (timestamps - epoch).dt.total_seconds()


def preprocess_samples(
    samples: list[dict],
    max_gap_s: float,
    ema_alpha: float,
    input_speed_unit: str,
) -> pd.DataFrame:
    """
    Convert raw sample dictionaries into a clean DataFrame ready for feature extraction.

    Expected input sample keys:
    - timestamp
    - speed
    - ax, ay, az
    - gx, gy, gz
    - lat, lon, altitude_m (optional)
    """
    df = pd.DataFrame(samples)
    if df.empty:
        return df

    required = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "speed"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Phone-handling windows are NOT vehicle motion: the driver picked up or
    # adjusted the phone, so its IMU readings describe the hand, not the car.
    # Dropping them here protects BOTH the live detector and the offline
    # scoring pipeline from phantom braking/turn/bump events. The timestamp
    # gap they create is handled by the existing max_gap_s logic.
    if "phone_handling" in df.columns:
        handled = df["phone_handling"].fillna(False).astype(bool)
        if handled.any():
            df = df[~handled].copy()
            if df.empty:
                return df

    # --- Numeric sanitization (CRIT-2: null GPS speed crashed finalization) ---
    # Coerce IMU columns to float, treating missing values as 0 (phones legitimately
    # lack some sensors; a missing IMU reading is "no motion signal", not an error).
    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Speed is the primary event signal: rows without a valid GPS speed cannot
    # contribute dv/event features and would poison aggregates with NaN. Drop them
    # here (before dt computation) so the rest of the pipeline only sees finite
    # speed values. Route maps are built from raw samples, so dropping a sample
    # from the scoring pipeline never affects the displayed route.
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    before_speed_drop = len(df)
    df = df[np.isfinite(df["speed"].to_numpy(dtype=float))].copy()
    if len(df) != before_speed_drop:
        logger.warning(
            "Dropped %d sample(s) with missing/invalid GPS speed during preprocessing",
            before_speed_drop - len(df),
        )
    if df.empty:
        # All rows were dropped (e.g. every sample lacked a GPS speed). Return
        # early so downstream timestamp parsing never sees an empty frame.
        return df

    # Parse timestamps one-by-one to avoid pandas choking on mixed formats
    # (e.g. timestamps with/without microseconds or timezone info).
    parsed: list[pd.Timestamp | pd.NaT] = []
    for raw in df["timestamp"]:
        try:
            parsed.append(pd.Timestamp(raw))
        except (ValueError, TypeError):
            parsed.append(pd.NaT)
    df["timestamp"] = parsed
    if df["timestamp"].dt.tz is None:
        # Assume UTC for naive timestamps
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        # Convert any timezone to UTC for consistency
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        return df

    # Numeric time in seconds
    df["t"] = _to_epoch_seconds(df["timestamp"])

    # Time delta between consecutive samples
    df["dt"] = df["t"].diff()

    # Keep first row; remove rows with bad dt afterward
    df = df[(df["dt"].isna()) | ((df["dt"] > 0) & (df["dt"] <= max_gap_s))].copy()
    df = df.reset_index(drop=True)

    if df.empty:
        return df

    # Convert speed to standard unit for physics calculations
    # Current project note: incoming value is treated as km/h
    if input_speed_unit == "kmh":
        df["speed"] = df["speed"].astype(float) / 3.6
    else:
        df["speed"] = df["speed"].astype(float)

    # Smooth the key motion fields
    for col in ["ax", "ay", "az", "gx", "gy", "gz", "speed"]:
        df[f"{col}_s"] = ema(df[col].to_numpy(), ema_alpha)

    return df

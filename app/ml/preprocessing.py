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

import numpy as np
import pandas as pd


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
    - lat, lon (optional for current pipeline)
    """
    df = pd.DataFrame(samples)
    if df.empty:
        return df

    required = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "speed"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Parse and sort timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
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

    # Cast numeric sensor columns
    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        df[col] = df[col].astype(float)

    # Smooth the key motion fields
    for col in ["ax", "ay", "az", "gx", "gy", "gz", "speed"]:
        df[f"{col}_s"] = ema(df[col].to_numpy(), ema_alpha)

    return df

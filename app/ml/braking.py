from __future__ import annotations

import numpy as np


def classify_brake_segment(
    dv: np.ndarray,
    speed: np.ndarray,
    start: int,
    end: int,
    *,
    emergency_brake_dv: float,
    emergency_brake_min_speed_mps: float,
) -> str:
    min_dv = float(np.min(dv[start : end + 1]))
    peak_speed = float(np.max(speed[start : end + 1]))
    if min_dv <= emergency_brake_dv and peak_speed >= emergency_brake_min_speed_mps:
        return "emergency_brake"
    return "hard_brake"

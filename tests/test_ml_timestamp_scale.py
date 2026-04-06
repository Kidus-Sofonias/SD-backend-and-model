from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.config import FeatureConfigV1
from app.ml.pipeline import run_trip_pipeline
from app.ml.preprocessing import preprocess_samples


def _sample(timestamp: str, speed: float, ax: float = 0.0) -> dict:
    return {
        "timestamp": timestamp,
        "speed": speed,
        "ax": ax,
        "ay": 0.0,
        "az": 9.81,
        "gx": 0.0,
        "gy": 0.0,
        "gz": 0.0,
    }


def test_preprocess_samples_preserves_half_second_dt() -> None:
    samples = [
        _sample("2026-01-30T08:01:11.302824Z", 36.0),
        _sample("2026-01-30T08:01:11.802824Z", 36.0),
        _sample("2026-01-30T08:01:12.302824Z", 36.0),
    ]

    df = preprocess_samples(samples, max_gap_s=3.0, ema_alpha=0.25, input_speed_unit="kmh")

    assert math.isclose(float(df.loc[1, "dt"]), 0.5, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(df.loc[2, "dt"]), 0.5, rel_tol=0.0, abs_tol=1e-9)


def test_run_trip_pipeline_uses_seconds_not_milliseconds() -> None:
    cfg = FeatureConfigV1()
    start = datetime(2026, 1, 30, 8, 1, 11, 302824, tzinfo=timezone.utc)
    samples = [_sample((start + timedelta(seconds=0.5 * i)).isoformat(), 36.0) for i in range(240)]

    result = run_trip_pipeline(samples, cfg)

    assert math.isclose(result["trip_features"]["duration_s"], 119.5, rel_tol=0.0, abs_tol=1e-9)

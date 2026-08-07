from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.config import FeatureConfigV2
from app.ml.features import aggregate_trip_features, compute_per_sample_features
from app.ml.pipeline import run_trip_pipeline
from app.ml.schemas import FEATURE_COLUMNS_FV1

NEW_SEQUENCE_FEATURES = [
    "jerk_entropy",
    "mean_event_duration_s",
    "max_event_duration_s",
    "max_consecutive_event_run_s",
]


def _sample(timestamp: str, speed: float, ax: float = 0.0, ay: float = 0.0, az: float = 9.81, gz: float = 0.0) -> dict:
    return {
        "timestamp": timestamp,
        "speed": speed,
        "ax": ax,
        "ay": ay,
        "az": az,
        "gx": 0.0,
        "gy": 0.0,
        "gz": gz,
    }


def _trip_samples(
    speeds: list[float],
    *,
    dt_s: float = 0.5,
    start: datetime | None = None,
    ax_noise: float = 0.1,
    gz_noise: float = 0.1,
) -> list[dict]:
    start = start or datetime(2026, 1, 30, 8, 0, 0, tzinfo=timezone.utc)
    rng = random.Random(42)
    samples = []
    for i, speed_kph in enumerate(speeds):
        samples.append(
            _sample(
                (start + timedelta(seconds=dt_s * i)).isoformat(),
                speed_kph,
                ax=rng.uniform(-ax_noise, ax_noise),
                ay=rng.uniform(-ax_noise, ax_noise),
                gz=rng.uniform(-gz_noise, gz_noise),
            )
        )
    return samples


def _preprocess(speeds: list[float], cfg: FeatureConfigV2, *, ax_noise: float = 0.1):
    from app.ml.preprocessing import preprocess_samples

    return preprocess_samples(_trip_samples(speeds, ax_noise=ax_noise), cfg.max_gap_s, cfg.ema_alpha, cfg.input_speed_unit)


def _features(speeds: list[float], *, ax_noise: float = 0.1) -> dict:
    cfg = FeatureConfigV2()
    df = compute_per_sample_features(_preprocess(speeds, cfg, ax_noise=ax_noise))
    return aggregate_trip_features(
        df,
        aggressive_turn_threshold=cfg.aggressive_turn_threshold,
        turn_min_duration_s=cfg.turn_min_duration_s,
        harsh_brake_dv=cfg.harsh_brake_dv,
        harsh_accel_dv=cfg.harsh_accel_dv,
        emergency_brake_dv=cfg.emergency_brake_dv,
        emergency_brake_min_speed_mps=cfg.emergency_brake_min_speed_mps,
        min_event_duration_s=cfg.min_event_duration_s,
        merge_gap_s=cfg.merge_gap_s,
        unstable_motion_jerk_threshold=cfg.unstable_motion_jerk_threshold,
        overspeed_threshold_mps=cfg.overspeed_threshold_mps,
        overspeed_min_duration_s=cfg.overspeed_min_duration_s,
        severe_overspeed_threshold_mps=cfg.severe_overspeed_threshold_mps,
        severe_overspeed_min_duration_s=cfg.severe_overspeed_min_duration_s,
    )


def test_sequence_features_are_in_feature_columns_contract() -> None:
    for feature in NEW_SEQUENCE_FEATURES:
        assert feature in FEATURE_COLUMNS_FV1


def test_smooth_trip_has_low_jerk_entropy_and_zero_event_durations() -> None:
    speeds = [36.0] * 240  # constant cruising, no IMU noise
    features = _features(speeds, ax_noise=0.0)

    assert features["jerk_entropy"] < 0.01
    assert features["mean_event_duration_s"] == 0.0
    assert features["max_event_duration_s"] == 0.0
    assert features["max_consecutive_event_run_s"] == 0.0


def test_erratic_trip_has_higher_jerk_entropy_than_smooth_trip() -> None:
    smooth = _features([36.0] * 240, ax_noise=0.0)
    erratic = _features([36.0] * 240, ax_noise=3.0)

    assert erratic["jerk_entropy"] > smooth["jerk_entropy"]


def test_sustained_brake_produces_positive_event_durations_and_run() -> None:
    # Cruising at 60 km/h, then a hard deceleration over several samples,
    # then resume. Each step drops ~20 km/h per 0.5s sample (~11 m/s^2).
    speeds = [60.0] * 40 + [60.0, 40.0, 20.0, 0.0, 0.0, 0.0, 0.0] + [40.0, 50.0, 60.0] + [60.0] * 60
    features = _features(speeds)

    assert features["harsh_brake_count"] >= 1
    assert features["max_event_duration_s"] > 0.0
    assert features["mean_event_duration_s"] > 0.0
    assert features["max_consecutive_event_run_s"] > 0.0
    assert features["max_consecutive_event_run_s"] >= features["max_event_duration_s"]


def test_pipeline_trip_features_contain_all_feature_columns() -> None:
    cfg = FeatureConfigV2()
    samples = _trip_samples([36.0] * 120)
    result = run_trip_pipeline(samples, cfg)

    for column in FEATURE_COLUMNS_FV1:
        assert column in result["trip_features"]

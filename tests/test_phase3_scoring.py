"""Phase 3 regression tests for the redesigned event detection and scoring:

- overspeed / severe-overspeed detection (new categories)
- aggressive turn via lateral acceleration proxy (v * yaw-rate)
- emergency brakes are now chargeable (risky trips score meaningfully lower)
- event-density normalization keeps scores consistent across trip durations
- persisted events use the new categories; scoring_version is v2
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.config import FeatureConfigV2
from app.ml.pipeline import run_trip_pipeline


def _sample(timestamp: str, speed: float, *, ax: float = 0.0, gz: float = 0.0, gx: float = 0.0, gy: float = 0.0) -> dict:
    return {
        "timestamp": timestamp,
        "speed": speed,
        "ax": ax,
        "ay": 0.0,
        "az": 9.81,
        "gx": gx,
        "gy": gy,
        "gz": gz,
    }


def _trip(
    speeds: list[float],
    *,
    dt_s: float = 0.5,
    start: datetime | None = None,
    gz: float = 0.0,
) -> list[dict]:
    start = start or datetime(2026, 1, 30, 8, 0, 0, tzinfo=timezone.utc)
    return [
        _sample((start + timedelta(seconds=dt_s * i)).isoformat(), v, gz=gz)
        for i, v in enumerate(speeds)
    ]


def _run(speeds: list[float], *, dt_s: float = 0.5, gz: float = 0.0):
    return run_trip_pipeline(_trip(speeds, dt_s=dt_s, gz=gz), FeatureConfigV2())


# ---------------------------------------------------------------------------
# New event categories
# ---------------------------------------------------------------------------

def test_overspeed_detected_when_sustained() -> None:
    # 120 km/h (33.3 m/s) sustained for 12 s at 1 Hz -> overspeed event.
    speeds = [50.0] * 5 + [120.0] * 12 + [50.0] * 5
    result = _run(speeds, dt_s=1.0)

    assert result["trip_features"]["overspeed_count"] >= 1
    assert any(e["event_type"] == "overspeed" for e in result["event_instances"])


def test_severe_overspeed_detected() -> None:
    # 140 km/h (38.9 m/s) sustained for 12 s at 1 Hz -> severe overspeed event,
    # which also clears the 10 s overspeed floor.
    speeds = [50.0] * 5 + [140.0] * 12 + [50.0] * 5
    result = _run(speeds, dt_s=1.0)

    assert result["trip_features"]["severe_overspeed_count"] >= 1
    assert result["trip_features"]["overspeed_count"] >= 1
    assert any(e["event_type"] == "severe_overspeed" for e in result["event_instances"])


def test_brief_speed_above_limit_is_not_overspeed() -> None:
    # A 2-second burst above the limit should NOT count (min duration 10 s at 1 Hz).
    speeds = [50.0] * 5 + [120.0, 120.0] + [50.0] * 5
    result = _run(speeds, dt_s=1.0)

    assert result["trip_features"]["overspeed_count"] == 0


def test_aggressive_turn_detected_via_lateral_acceleration() -> None:
    # 54 km/h (15 m/s) with a sustained yaw rate of 0.32 rad/s -> lateral accel
    # ~4.75 m/s^2 > 4.4 threshold (EMA asymptote considered).
    gz = 0.32
    speeds = [54.0] * 40
    result = _run(speeds, gz=gz)

    assert result["trip_features"]["aggressive_turn_count"] >= 1
    assert any(e["event_type"] == "aggressive_turn" for e in result["event_instances"])


def test_sustained_gyro_noise_does_not_produce_turn_events() -> None:
    # Ordinary road/phone gyro noise (0.16 rad/s) at 54 km/h gives lateral accel
    # ~2.4 m/s^2, well below the 4.4 threshold: no turn flood.
    result = _run([54.0] * 40, gz=0.16)
    assert result["trip_features"]["aggressive_turn_count"] == 0


def test_smooth_cruise_has_no_events() -> None:
    result = _run([54.0] * 120)

    features = result["trip_features"]
    for key in [
        "harsh_brake_count",
        "emergency_brake_count",
        "harsh_accel_count",
        "aggressive_turn_count",
        "overspeed_count",
        "severe_overspeed_count",
        "unstable_motion_count",
    ]:
        assert features[key] == 0, f"{key} should be 0 on a smooth cruise"
    assert result["score"] >= 95


# ---------------------------------------------------------------------------
# Scoring behavior
# ---------------------------------------------------------------------------

def test_emergency_brakes_are_chargeable() -> None:
    # Repeated hard stops (60 -> 20 km/h steps at 0.5 s = ~11 m/s^2 raw dv):
    # these are emergency-grade stops and must now cost score.
    base = [60.0] * 10
    stop = [20.0, 20.0, 20.0]
    speeds = base + stop + [60.0] * 10 + stop + [60.0] * 10 + stop + [60.0] * 10
    result = _run(speeds)

    features = result["trip_features"]
    penalties = result["breakdown"]["penalties"]

    assert features["emergency_brake_count"] >= 1
    assert penalties["emergency_brake"] > 0
    assert result["score"] < 90


def test_risky_trip_scores_lower_than_safe_trip() -> None:
    safe_speeds = [54.0] * 240
    risky_speeds = [60.0] * 20 + [20.0, 20.0, 20.0] + [60.0] * 20 + [20.0, 20.0, 20.0] + [60.0] * 20

    safe = _run(safe_speeds)
    risky = _run(risky_speeds)

    assert risky["score"] < safe["score"]
    # risk_level is derived in the service layer; a clean cruise must be >= 85.
    assert safe["score"] >= 85


def test_events_per_hour_normalizes_by_duration() -> None:
    # The same two hard brakes spread over 10 minutes vs 1 minute must produce
    # different event densities, and the density penalty must differ accordingly.
    brake_ramp = [20.0, 20.0]  # one hard stop
    short = [60.0] * 4 + brake_ramp + [60.0] * 4 + brake_ramp + [60.0] * 4
    long = [60.0] * 300 + brake_ramp + [60.0] * 300 + brake_ramp + [60.0] * 300

    short_result = _run(short, dt_s=1.0)
    long_result = _run(long, dt_s=1.0)

    assert short_result["trip_features"]["events_per_hour"] > long_result["trip_features"]["events_per_hour"]
    assert (
        short_result["breakdown"]["penalties"]["density"]
        > long_result["breakdown"]["penalties"]["density"]
    )


def test_scoring_version_is_v2() -> None:
    result = _run([54.0] * 120)
    assert result["breakdown"]["scoring_version"] == "v2"

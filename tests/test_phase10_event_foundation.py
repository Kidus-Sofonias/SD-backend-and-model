"""Phase 10 (hackathon) — event foundation & v3 scoring tests.

Verifies the fixes for the audit's headline problem (hundreds of noise
"events" and scores of 0 on short trips):
- cooldown + net-speed-delta collapse GPS-noise clusters into real maneuvers
- rate-normalised jerk bounds rough-road unstable_motion
- per-event severity/confidence are stored and drive the score
- short trips (5/8/10 min) keep meaningful scores
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.config import FeatureConfigV2
from app.ml.event_utils import event_segments, event_severity_and_confidence
from app.ml.pipeline import run_trip_pipeline
from app.ml.scoring_rules import score_trip_rules_v3

cfg = FeatureConfigV2()


def _noisy_trip(
    duration_s: float,
    hz: float,
    base_speed_mps: float,
    speed_wobble: float,
    imu_noise: float,
    *,
    vertical_bumps: float = 0.0,
    events: list[tuple[float, float]] | None = None,
    seed: int = 7,
) -> list[dict]:
    """GPS speed + IMU noise matching the mobile demo simulator.

    The app's ``wobble(center, amplitude)`` adds UNIFORM per-sample noise of
    +-amplitude to the speed (0.8 m/s normal / 1.5 m/s risky), and the same
    uniform profile to the IMU axes. We replicate that profile here (the
    argument is the amplitude, not a gaussian sigma) so the tests exercise the
    same signal the demo trips actually send.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * hz)
    t0 = 1_723_000_000.0
    lat, lon = 9.0, 38.7
    samples: list[dict] = []
    for i in range(n):
        t = t0 + i / hz
        # Independent per-sample jitter AROUND the cruise speed, matching the
        # app's wobble(center, amplitude) (NOT a random walk -- a walk would
        # drift and inflate speed_variance without representing real GPS).
        speed = max(0.5, base_speed_mps + float(rng.uniform(-speed_wobble, speed_wobble)))
        for ev_t, ev_dv in (events or []):
            if abs(ev_t - i / hz) <= 0.5:
                speed = max(0.5, speed + ev_dv)
        # Advance GPS roughly proportional to speed.
        lat += speed / 111_000.0 * (1.0 / hz)
        lon += speed / (111_000.0 * np.cos(np.deg2rad(9.0))) * (1.0 / hz)
        az = 9.8 + float(rng.uniform(-(imu_noise * 0.4 + vertical_bumps), imu_noise * 0.4 + vertical_bumps))
        samples.append(
            {
                "timestamp": datetime.fromtimestamp(t, tz=timezone.utc).isoformat(),
                "speed": speed * 3.6,  # km/h as the app sends
                "lat": lat,
                "lon": lon,
                "ax": float(rng.uniform(-imu_noise, imu_noise)),
                "ay": float(rng.uniform(-imu_noise, imu_noise)),
                "az": az,
                "gx": float(rng.uniform(-0.05, 0.05)),
                "gy": float(rng.uniform(-0.05, 0.05)),
                "gz": float(rng.uniform(-0.08, 0.08)),
            }
        )
    return samples


def _events_by_type(result: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in result["event_instances"]:
        counts[e["event_type"]] = counts.get(e["event_type"], 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Noise robustness (the audit's headline problem, now fixed)
# ---------------------------------------------------------------------------

def test_clean_cruise_still_has_zero_events() -> None:
    result = run_trip_pipeline(_noisy_trip(600, 2.0, 16.7, 0.15, 0.08), cfg)
    assert result["event_instances"] == []
    assert result["score"] == 100


def test_demo_normal_noise_no_longer_floods_events() -> None:
    # Real demo "normal" profile: uniform speed wobble +-0.8 m/s. Before
    # Phase 10 this produced hundreds of events and a score of 0.
    result = run_trip_pipeline(_noisy_trip(600, 2.0, 11.0, 0.8, 0.14), cfg)
    assert len(result["event_instances"]) <= 8, _events_by_type(result)
    assert result["score"] >= 80


def test_demo_risky_noise_is_bounded_and_score_meaningful() -> None:
    # Real demo "risky" profile: uniform speed wobble +-1.5 m/s. Before Phase
    # 10 this produced ~400+ events and a score of 0.
    result = run_trip_pipeline(_noisy_trip(600, 2.0, 15.0, 1.5, 0.30), cfg)
    assert len(result["event_instances"]) <= 25, _events_by_type(result)
    assert result["score"] >= 60


def test_extreme_noise_still_bounded_not_zero() -> None:
    # 2x the risky demo wobble: ~54% of samples cross the per-sample dv
    # threshold, so events cannot be zero. The guarantee is that each
    # category is cooldown-bounded (at most 1 per 5 s window = 120 per
    # category over 600 s) instead of the ~400-800 event flood the
    # pre-Phase-10 pipeline produced from this pathological input. 142 =
    # 71 brake-family + 71 accel, both comfortably under their ceilings.
    result = run_trip_pipeline(_noisy_trip(600, 2.0, 15.0, 3.0, 0.60), cfg)
    assert len(result["event_instances"]) <= 200, _events_by_type(result)
    assert result["score"] is not None


def test_genuine_maneuvers_still_detected() -> None:
    # Two genuine hard stops in 5 minutes must still be detected and scored
    # reasonably (not clamped to 0, not invisible).
    samples = _noisy_trip(
        300, 2.0, 14.0, 0.25, 0.12,
        events=[(90, -5.5), (200, -5.0)],
    )
    result = run_trip_pipeline(samples, cfg)
    types = _events_by_type(result)
    assert sum(types.get(k, 0) for k in ("hard_brake", "emergency_brake")) >= 1
    assert len(result["event_instances"]) <= 6, types
    assert 50 <= result["score"] <= 95


def test_rough_road_unstable_motion_is_bounded() -> None:
    # Rough road is a continuous CONDITION: with a dedicated long cooldown it
    # yields a handful of unstable_motion events, not ~260 in 10 minutes.
    result = run_trip_pipeline(_noisy_trip(600, 2.0, 13.0, 0.6, 0.30, vertical_bumps=1.5), cfg)
    types = _events_by_type(result)
    assert types.get("unstable_motion", 0) <= 40, types
    assert len(result["event_instances"]) <= 50, types


# ---------------------------------------------------------------------------
# Short-trip scoring (the brief: no catastrophic scores on 5/8/10-min trips)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("minutes", [5, 8, 10])
def test_short_trip_with_light_noise_keeps_meaningful_score(minutes: int) -> None:
    result = run_trip_pipeline(
        _noisy_trip(minutes * 60, 2.0, 13.9, 0.4, 0.15, seed=minutes),
        cfg,
    )
    assert result["score"] is not None
    assert result["score"] >= 70, result["score"]


@pytest.mark.parametrize("minutes", [5, 8, 10])
def test_short_trip_with_one_genuine_brake_scores_sanely(minutes: int) -> None:
    result = run_trip_pipeline(
        _noisy_trip(
            minutes * 60, 2.0, 14.0, 0.25, 0.12,
            events=[(minutes * 30, -5.5)],
        ),
        cfg,
    )
    assert result["score"] is not None
    assert 40 <= result["score"] <= 95


# ---------------------------------------------------------------------------
# Per-event severity / confidence
# ---------------------------------------------------------------------------

def test_events_carry_severity_confidence_and_duration() -> None:
    result = run_trip_pipeline(
        _noisy_trip(300, 2.0, 14.0, 0.25, 0.12, events=[(90, -5.5)]),
        cfg,
    )
    events = [e for e in result["event_instances"] if e["event_type"] in ("hard_brake", "emergency_brake")]
    assert events, result["event_instances"]
    for e in events:
        assert 0.0 <= e["severity"] <= 1.0
        assert 0.0 <= e["confidence"] <= 1.0
        assert e["duration_s"] >= 0.0


def test_event_severity_and_confidence_monotonic() -> None:
    low_sev, low_conf = event_severity_and_confidence(
        event_type="hard_brake", value=3.2, duration_s=0.0, activation=3.2, reference=6.5
    )
    high_sev, high_conf = event_severity_and_confidence(
        event_type="hard_brake", value=6.5, duration_s=4.0, activation=3.2, reference=6.5
    )
    assert low_sev == 0.0
    assert high_sev == 1.0
    assert high_conf > low_conf
    assert 0.0 <= low_conf <= 1.0
    assert 0.0 <= high_conf <= 1.0


def test_event_segments_cooldown_drops_close_clusters() -> None:
    mask = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0])
    t = np.arange(9.0, dtype=float)

    both = event_segments(mask, t, min_duration_s=0.0, merge_gap_s=0.1)
    assert both == [(1, 1), (7, 7)]

    # Segments start 6 s apart: a 5 s cooldown keeps both (gap >= cooldown);
    # a 7 s cooldown drops the second (gap < cooldown).
    cooled = event_segments(mask, t, min_duration_s=0.0, merge_gap_s=0.1, cooldown_s=5.0)
    assert cooled == [(1, 1), (7, 7)]
    far = event_segments(mask, t, min_duration_s=0.0, merge_gap_s=0.1, cooldown_s=7.0)
    assert far == [(1, 1)]


# ---------------------------------------------------------------------------
# v3 scoring uses impacts, not raw counts
# ---------------------------------------------------------------------------

def _score_with_impact(impact: float) -> dict:
    feats = {
        "confidence": 1.0,
        "event_impacts": {"emergency_brake": impact},
        "emergency_brake_count": 1,
        "chargeable_hard_brake_count": 0,
        "harsh_accel_count": 0,
        "aggressive_turn_count": 0,
        "overspeed_count": 0,
        "severe_overspeed_count": 0,
        "unstable_motion_count": 0,
        "p95_jerk": 0.0,
        "speed_variance": 0.0,
        "events_per_hour": 0.0,
        "events_per_km": 0.0,
        "distance_km": 0.0,
        "duration_s": 600.0,
    }
    return score_trip_rules_v3(
        feats,
        cfg.w_emergency_brake, cfg.w_brake, cfg.w_accel, cfg.w_turn,
        cfg.w_overspeed, cfg.w_severe_overspeed, cfg.w_unstable_motion,
        cfg.w_jerk, cfg.w_speed_var, cfg.w_density,
        cfg.jerk_normalize_low, cfg.jerk_normalize_high, cfg.speed_var_normalize_high,
        cfg.density_normalize_low, cfg.density_normalize_high,
        cfg.density_distance_normalize_high, cfg.density_min_duration_s,
    )


def test_v3_score_scales_penalty_with_impact() -> None:
    full = _score_with_impact(1.0)
    half = _score_with_impact(0.5)
    assert full["penalties"]["emergency_brake"] == pytest.approx(cfg.w_emergency_brake)
    assert half["penalties"]["emergency_brake"] == pytest.approx(cfg.w_emergency_brake * 0.5)
    assert half["score"] > full["score"]
    assert full["scoring_version"] == "v3"

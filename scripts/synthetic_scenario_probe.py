"""Synthetic scenario battery — hackathon verification deliverable.

Runs 15+ reproducible scenarios against the production scoring pipeline and
records event counts + scores, including vehicle-aware comparisons (the SAME
sensor stream interpreted as a sedan vs a 36 t tractor-trailer).

Usage (from backend/):
    python scripts/synthetic_scenario_probe.py
    python scripts/synthetic_scenario_probe.py --duration 600 --hz 2

Scenario groups:
  A  Clean cruise / quiet sensors .......... expect ~0 events, score ~100
  B  Demo-simulator-like noise ............. Phase 2: bounded events, sane scores
  C  Genuine maneuvers + light noise ....... exactly the forced maneuvers
  D  Rough road / bumps .................... bounded unstable_motion
  E  Short trips (5/8/10 min) .............. no absurdly low scores from noise
  F  Vehicle-aware ......................... same signal, sedan vs truck configs
  G  GPS noise ............................. bounded jitter, no event storm

Phase 0 baseline (pre-fix): normal noise ~205 events/score 0, risky ~404/0,
rough road ~272/0. All groups below should stay far below that.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, ".")

from app.ml.config import FeatureConfigV2  # noqa: E402
from app.ml.pipeline import run_trip_pipeline  # noqa: E402
from app.ml.vehicle_profiles import VEHICLE_CATEGORIES, config_for_profile  # noqa: E402


@dataclass(frozen=True)
class FakeProfile:
    """Minimal stand-in for a VehicleProfile row (category is all we need)."""

    category: str
    size_class: str | None = None
    mass_kg: float | None = None


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def make_trip(
    duration_s: float,
    hz: float,
    base_speed_mps: float,
    speed_wobble: float,   # std of per-sample GPS speed noise (m/s)
    imu_noise: float,      # std of lateral/longitudinal IMU noise (m/s^2)
    gz_noise: float = 0.08,
    vertical_bumps: float = 0.0,  # extra az std (rough road)
    events: list | None = None,   # (t_s, dv_mps2) forced maneuvers
) -> list[dict]:
    n = int(duration_s * hz)
    t0 = 1_723_000_000.0
    # Match the real mobile demo simulator (sensorCapture.ts wobble()):
    # uniform jitter around the cruise value, NOT a random walk. The demo
    # uses speed_wobble amplitudes of 0.8 (normal) / 1.5 (risky) m/s.
    # Forced maneuvers permanently shift the cruise speed (braking to a lower
    # speed and staying there) so no fake rebound acceleration is generated.
    offset = 0.0
    fired: set[int] = set()
    samples: list[dict] = []
    events = events or []
    for i in range(n):
        t = t0 + i / hz
        for k, (ev_t, ev_dv) in enumerate(events):
            if k not in fired and abs(ev_t - i / hz) <= 0.5:
                fired.add(k)
                offset = max(-base_speed_mps + 0.5, offset + ev_dv)
        speed = max(0.5, base_speed_mps + offset + float(np.random.uniform(-speed_wobble, speed_wobble)))
        az = 9.8 + float(np.random.normal(0.0, imu_noise * 0.4 + vertical_bumps))
        samples.append(
            {
                "timestamp": iso(t),
                "speed": speed * 3.6,  # km/h, as the mobile app sends
                "lat": 9.0,
                "lon": 38.7,
                "ax": float(np.random.normal(0.0, imu_noise)),
                "ay": float(np.random.normal(0.0, imu_noise)),
                "az": az,
                "gx": float(np.random.normal(0.0, 0.05)),
                "gy": float(np.random.normal(0.0, 0.05)),
                "gz": float(np.random.normal(0.0, gz_noise)),
            }
        )
    return samples


def run(label: str, samples: list[dict], cfg: FeatureConfigV2) -> None:
    res = run_trip_pipeline(samples, cfg)
    feats = res["trip_features"]
    counts = Counter(e["event_type"] for e in res["event_instances"])
    print(
        f"{label:44s} n={len(samples):5d} dur={feats.get('duration_s', 0):6.1f}s "
        f"score={res['score']} conf={res['confidence']:.2f} "
        f"events={len(res['event_instances'])} {dict(counts)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--hz", type=float, default=2.0)
    args = parser.parse_args()

    cfg = FeatureConfigV2()
    d, hz = args.duration, args.hz

    print("=== A: clean cruise (no events expected) ===")
    run("A1: cruise 60km/h, quiet", make_trip(d, hz, 16.7, 0.15, 0.08), cfg)
    run("A2: cruise 50km/h, quiet", make_trip(min(d, 300), hz, 13.9, 0.15, 0.08), cfg)

    print("\n=== B: demo-simulator-like noise (mobile profiles) ===")
    # Amplitudes mirror the real app: speed wobble 0.8/1.5 m/s, accel noise
    # ~0.3-0.6 m/s^2. Noise alone must NOT create an event storm.
    run("B1: 'normal' noise", make_trip(d, hz, 11.0, 0.8, 0.30, 0.10), cfg)
    run("B2: 'risky' noise", make_trip(d, hz, 15.0, 1.5, 0.60, 0.25), cfg)

    print("\n=== C: genuine maneuvers + light noise ===")
    run(
        "C1: 2 genuine hard brakes",
        make_trip(min(d, 300), hz, 14.0, 0.25, 0.12, events=[(90, -5.5), (200, -5.0)]),
        cfg,
    )
    run(
        "C2: hard brake + hard accel",
        make_trip(min(d, 480), hz, 14.0, 0.25, 0.12, events=[(150, -4.5), (300, 4.0)]),
        cfg,
    )
    run("C3: rough road (az jitter)", make_trip(d, hz, 13.0, 0.6, 0.30, vertical_bumps=1.5), cfg)
    run("C4: normal braking (gentle)", make_trip(min(d, 240), hz, 14.0, 0.2, 0.1, events=[(60, -2.0)]), cfg)
    run("C5: normal acceleration (gentle)", make_trip(min(d, 240), hz, 12.0, 0.2, 0.1, events=[(60, 2.0)]), cfg)
    run(
        "C6: sharp cornering (lateral)",
        make_trip(min(d, 300), hz, 14.0, 0.25, 0.12, events=[(100, -0.5)]),  # lat via ay noise spike below
        cfg,
    )

    print("\n=== D: bumps / rough road ===")
    run("D1: single speed bump", make_trip(min(d, 120), hz, 10.0, 0.2, 0.1, vertical_bumps=3.0), cfg)
    run("D2: rough road sustained", make_trip(min(d, 300), hz, 13.0, 0.6, 0.30, vertical_bumps=1.5), cfg)

    print("\n=== E: short trips (5/8/10 min) ===")
    for minutes, wobble, noise in [(5, 0.8, 0.30), (8, 0.8, 0.30), (10, 0.8, 0.30)]:
        run(
            f"E{minutes}m: short trip, normal noise",
            make_trip(minutes * 60, hz, 11.0, wobble, noise, 0.10),
            cfg,
        )
    run("E5m risky: 5m short, risky noise", make_trip(300, hz, 15.0, 1.5, 0.60, 0.25), cfg)

    print("\n=== F: vehicle-aware (same signal, different vehicle class) ===")
    risky_signal = make_trip(min(d, 480), hz, 15.0, 1.5, 0.60, 0.25)
    for key in ["sedan", "suv", "pickup", "van", "bus", "heavy_truck", "tractor_trailer"]:
        profile = FakeProfile(category=key)
        tuned = config_for_profile(FeatureConfigV2(), profile)
        label = f"F:{key:16s} risky noise"
        run(label, risky_signal, tuned)
    run("F: universal cfg (no profile)", risky_signal, FeatureConfigV2())

    print("\n=== G: GPS noise ===")
    run("G1: heavy GPS speed jitter", make_trip(min(d, 300), hz, 16.7, 3.0, 0.15), cfg)
    run("G2: light GPS jitter", make_trip(min(d, 300), hz, 16.7, 0.5, 0.15), cfg)


if __name__ == "__main__":
    main()

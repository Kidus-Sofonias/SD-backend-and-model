"""Synthetic scenario probe — Phase 0 audit deliverable.

Reproduces the hackathon brief's complaint ("50-70 events / scores 3-5 on
5-10 minute trips") against the *production* scoring pipeline so fixes can be
measured before/after (Phase 2+).

Usage (from backend/):
    python scripts/synthetic_scenario_probe.py

Scenarios:
  A  Clean cruise, quiet sensors ............ expect ~0 events, score ~100
  B  Demo-simulator-like noise levels ....... Phase 2: 0 events (normal) / ~15
                                            bounded events (risky) with sane scores
  C  Genuine maneuvers + light noise ....... exactly the forced maneuvers
  D  Rough road (vertical jitter) .......... bounded unstable_motion (~1/20 s)

Phase 0 baseline for the B/C/D scenarios (pre-fix pipeline):
  normal noise ~205 events/score 0, risky ~404/0, rough road ~272/0.

Run `python scripts/synthetic_scenario_probe.py --duration 600 --hz 2` to tweak.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, ".")

from app.ml.config import FeatureConfigV2  # noqa: E402
from app.ml.pipeline import run_trip_pipeline  # noqa: E402


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


if __name__ == "__main__":
    main()

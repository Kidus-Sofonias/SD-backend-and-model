# File role: Synthetic trip/sample generator for ML pipeline testing.
# Creates completed trips and realistic-ish sensor samples directly in the database,
# and writes a synthetic label registry for dataset building.
# Intended for:
# - dataset building smoke tests
# - training pipeline testing
# - class-balance bootstrapping
# Not intended for final real-world model validation.
# Connects to:
# - app.db.session
# - app.db.models.user
# - app.db.models.trip
# - app.db.models.sensor_sample
# - artifacts/datasets/synthetic_trip_labels.json
# Key symbols/vars:
# - SYNTHETIC_LABELS_PATH
# - generate_safe_profile
# - generate_risky_profile
# - create_trip_with_samples
# - main

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import func, select

from app.db.session import SessionLocal
from app.db.models.user import User
from app.db.models.driving_event import DrivingEvent
from app.db.models.trip import Trip
from app.db.models.sensor_sample import SensorSample

DEFAULT_TOTAL_TRIPS = 50
DEFAULT_SAMPLES_PER_TRIP = 400   # 400 * 0.3s = 120 seconds (more samples, finer granularity)
DEFAULT_DT_SECONDS = 0.3         # Faster sampling rate for richer training data

SYNTHETIC_LABELS_PATH = Path("artifacts/datasets/synthetic_trip_labels.json")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def resolve_user_id(db, user_id: str | None) -> str:
    if user_id:
        user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
        if not user:
            raise ValueError(f"User '{user_id}' not found")
        return user.id

    first_user = db.execute(select(User).order_by(User.id.asc()).limit(1)).scalar_one_or_none()
    if not first_user:
        raise ValueError("No users found in database. Create a user first.")
    return first_user.id


# ---------------------------------------------------------------------------
# Multiple Ethiopian base locations for geographic diversity
# ---------------------------------------------------------------------------
LOCATIONS: list[tuple[float, float, float]] = [
    # (lat, lon, base_altitude_m)
    (9.0300, 38.7400, 2355.0),   # Addis Ababa centre
    (8.9800, 38.8000, 2320.0),   # Addis Bole area
    (9.0200, 38.6900, 2380.0),   # Addis summit area
    (9.0500, 38.7600, 2340.0),   # Addis north
    (8.5400, 39.2700, 1720.0),   # Adama (Nazret)
    (9.6000, 41.8600, 2430.0),   # Dire Dawa area
    (11.5700, 37.3900, 2130.0),  # Bahir Dar
    (7.0500, 38.5000, 1750.0),   # Hawassa
]


def pick_location() -> tuple[float, float, float, float, float]:
    """Return (lat, lon, altitude_m, lat_drift, lon_drift) for a random route."""
    loc = random.choice(LOCATIONS)
    # Direction of travel (random bearing)
    bearing_rad = random.uniform(0, 2 * math.pi)
    drift_lat = math.cos(bearing_rad) * 0.0003
    drift_lon = math.sin(bearing_rad) * 0.0003 / math.cos(loc[0] * math.pi / 180)
    return (*loc, drift_lat, drift_lon)



def altitude_wobble(base_alt: float, progress: float, roughness: float = 5.0) -> float:
    """Simulate gentle road elevation changes."""
    wave1 = math.sin(progress * math.pi * 0.7) * roughness
    wave2 = math.cos(progress * math.pi * 2.3) * roughness * 0.4
    wave3 = math.sin(progress * math.pi * 5.1) * roughness * 0.15
    return base_alt + wave1 + wave2 + wave3 + random.uniform(-1.0, 1.0)



def generate_safe_profile(samples_per_trip: int, dt_s: float) -> list[dict]:
    rows = []
    lat, lon, base_alt, drift_lat, drift_lon = pick_location()

    speed_kmh = random.uniform(24, 38)

    for i in range(samples_per_trip):
        t = i * dt_s
        progress = i / samples_per_trip

        # Smooth, gradual speed changes
        speed_kmh += random.uniform(-0.2, 0.2) + 0.12 * math.sin(t / 12.0)
        speed_kmh = clamp(speed_kmh, 18, 50)

        # Gentle acceleration (m/s^2)
        ax = random.uniform(-0.12, 0.12) + 0.02 * math.sin(t / 4.0)
        ay = random.uniform(-0.10, 0.10) + 0.02 * math.cos(t / 6.0)
        az = 9.81 + random.uniform(-0.06, 0.06)

        # Low gyro activity
        gx = random.uniform(-0.04, 0.04)
        gy = random.uniform(-0.04, 0.04)
        gz = random.uniform(-0.10, 0.10) + 0.02 * math.sin(t / 7.0)

        # Steady GPS with smooth progress
        lat += drift_lat * speed_kmh * 0.001 + random.uniform(-0.000008, 0.000008)
        lon += drift_lon * speed_kmh * 0.001 + random.uniform(-0.000008, 0.000008)
        alt = altitude_wobble(base_alt, progress, roughness=3.0)

        rows.append({
            "speed": speed_kmh,
            "lat": lat,
            "lon": lon,
            "accuracy_m": random.uniform(2.5, 6.0),
            "altitude_m": alt,
            "ax": ax,
            "ay": ay,
            "az": az,
            "gx": gx,
            "gy": gy,
            "gz": gz,
        })

    return rows


def generate_moderate_profile(samples_per_trip: int, dt_s: float) -> list[dict]:
    """Moderate driving with occasional mild events."""
    rows = []
    lat, lon, base_alt, drift_lat, drift_lon = pick_location()
    speed_kmh = random.uniform(28, 45)
    n_events = random.randint(3, 6)
    event_centers = random.sample(range(int(samples_per_trip * 0.15), int(samples_per_trip * 0.85)), n_events)

    for i in range(samples_per_trip):
        t = i * dt_s
        progress = i / samples_per_trip
        delta = random.uniform(-0.4, 0.4)

        for c in event_centers:
            dist = abs(i - c)
            if dist <= 6:
                intensity = 1.0 - dist / 6.0
                if random.random() < 0.5:
                    delta -= intensity * random.uniform(1.2, 3.0)  # brake
                else:
                    delta += intensity * random.uniform(1.2, 3.0)  # accel

        speed_kmh += delta
        speed_kmh = clamp(speed_kmh, 12, 75)

        ax = random.uniform(-0.4, 0.4) + 0.10 * math.sin(t / 3.0)
        ay = random.uniform(-0.3, 0.3) + 0.08 * math.cos(t / 4.0)
        az = 9.81 + random.uniform(-0.12, 0.12)

        gx = random.uniform(-0.15, 0.15)
        gy = random.uniform(-0.15, 0.15)
        gz = random.uniform(-0.25, 0.25)

        lat += drift_lat * speed_kmh * 0.001 + random.uniform(-0.00001, 0.00001)
        lon += drift_lon * speed_kmh * 0.001 + random.uniform(-0.00001, 0.00001)
        alt = altitude_wobble(base_alt, progress, roughness=4.0)

        rows.append({
            "speed": speed_kmh,
            "lat": lat,
            "lon": lon,
            "accuracy_m": random.uniform(3.0, 9.0),
            "altitude_m": alt,
            "ax": ax,
            "ay": ay,
            "az": az,
            "gx": gx,
            "gy": gy,
            "gz": gz,
        })

    return rows


def generate_risky_profile(samples_per_trip: int, dt_s: float) -> list[dict]:
    rows = []
    lat, lon, base_alt, drift_lat, drift_lon = pick_location()

    speed_kmh = random.uniform(35, 55)

    hard_brake_centers = random.sample(range(int(samples_per_trip * 0.1), int(samples_per_trip * 0.9)), 6)
    hard_accel_centers = random.sample(range(int(samples_per_trip * 0.1), int(samples_per_trip * 0.9)), 6)
    turn_centers = random.sample(range(int(samples_per_trip * 0.1), int(samples_per_trip * 0.9)), 8)

    for i in range(samples_per_trip):
        t = i * dt_s
        progress = i / samples_per_trip

        delta = random.uniform(-1.0, 1.0)

        for c in hard_accel_centers:
            if abs(i - c) <= 5:
                delta += random.uniform(2.8, 6.5)

        for c in hard_brake_centers:
            if abs(i - c) <= 5:
                delta -= random.uniform(3.5, 7.0)

        speed_kmh += delta
        speed_kmh = clamp(speed_kmh, 3, 100)

        ax = random.uniform(-1.0, 1.0) + 0.20 * math.sin(t / 1.8)
        ay = random.uniform(-0.8, 0.8) + 0.16 * math.cos(t / 2.5)
        az = 9.81 + random.uniform(-0.30, 0.30)

        gx = random.uniform(-0.35, 0.35)
        gy = random.uniform(-0.35, 0.35)
        gz = random.uniform(-0.50, 0.50)

        for c in turn_centers:
            if abs(i - c) <= 6:
                gz += random.choice([-1, 1]) * random.uniform(1.5, 3.5)
                ay += random.choice([-1, 1]) * random.uniform(0.8, 2.0)
                ax += random.uniform(-0.3, 0.3)

        lat += drift_lat * speed_kmh * 0.001 + random.uniform(-0.00002, 0.00002)
        lon += drift_lon * speed_kmh * 0.001 + random.uniform(-0.00002, 0.00002)
        alt = altitude_wobble(base_alt, progress, roughness=7.0)

        rows.append({
            "speed": speed_kmh,
            "lat": lat,
            "lon": lon,
            "accuracy_m": random.uniform(4.0, 14.0),
            "altitude_m": alt,
            "ax": ax,
            "ay": ay,
            "az": az,
            "gx": gx,
            "gy": gy,
            "gz": gz,
        })

    return rows


def create_trip_with_samples(
    db,
    user_id: str,
    rows: list[dict],
    started_at: datetime,
    dt_s: float,
) -> str:
    trip = Trip(
        user_id=user_id,
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=dt_s * len(rows)),
        status="completed",
    )
    db.add(trip)
    db.flush()

    for idx, row in enumerate(rows):
        ts = started_at + timedelta(seconds=idx * dt_s)

        sample = SensorSample(
            user_id=user_id,
            trip_id=trip.id,
            ts=ts,
            speed_mps=row["speed"],
            lat=row["lat"],
            lon=row["lon"],
            accuracy_m=row["accuracy_m"],
            altitude_m=row.get("altitude_m"),
            ax=row["ax"],
            ay=row["ay"],
            az=row["az"],
            gx=row["gx"],
            gy=row["gy"],
            gz=row["gz"],
        )
        db.add(sample)

    return trip.id


def load_existing_synthetic_labels() -> dict[str, int]:
    if not SYNTHETIC_LABELS_PATH.exists():
        return {}
    try:
        return json.loads(SYNTHETIC_LABELS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_synthetic_labels(labels: dict[str, int]) -> None:
    SYNTHETIC_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_LABELS_PATH.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def count_completed_trips(db) -> int:
    return int(
        db.execute(
            select(func.count()).select_from(Trip).where(Trip.status == "completed")
        ).scalar_one()
        or 0
    )


def generate_synthetic_trips(
    *,
    count: int,
    user_id: str | None = None,
    samples_per_trip: int = DEFAULT_SAMPLES_PER_TRIP,
    dt: float = DEFAULT_DT_SECONDS,
    seed: int = 42,
) -> dict[str, object]:
    if count <= 0:
        return {
            "created_count": 0,
            "safe_count": 0,
            "moderate_count": 0,
            "risky_count": 0,
            "samples_per_trip": samples_per_trip,
            "dt": dt,
            "seed": seed,
            "user_id": user_id,
            "created_trip_ids": [],
            "synthetic_labels_path": str(SYNTHETIC_LABELS_PATH),
        }

    random.seed(seed)

    db = SessionLocal()
    try:
        resolved_user_id = resolve_user_id(db, user_id)

        safe_count = count // 3
        moderate_count = count // 3
        risky_count = count - safe_count - moderate_count

        now = datetime.now(timezone.utc)
        created_trip_ids: list[tuple[str, str]] = []
        synthetic_labels = load_existing_synthetic_labels()

        trip_index = 0

        for _ in range(safe_count):
            started_at = now - timedelta(days=trip_index + 1, minutes=random.randint(0, 120))
            rows = generate_safe_profile(samples_per_trip, dt)
            trip_id = create_trip_with_samples(db, resolved_user_id, rows, started_at, dt)
            created_trip_ids.append((trip_id, "safe"))
            synthetic_labels[trip_id] = 0
            trip_index += 1

        for _ in range(moderate_count):
            started_at = now - timedelta(days=trip_index + 1, minutes=random.randint(0, 120))
            rows = generate_moderate_profile(samples_per_trip, dt)
            trip_id = create_trip_with_samples(db, resolved_user_id, rows, started_at, dt)
            created_trip_ids.append((trip_id, "moderate"))
            synthetic_labels[trip_id] = 0
            trip_index += 1

        for _ in range(risky_count):
            started_at = now - timedelta(days=trip_index + 1, minutes=random.randint(0, 120))
            rows = generate_risky_profile(samples_per_trip, dt)
            trip_id = create_trip_with_samples(db, resolved_user_id, rows, started_at, dt)
            created_trip_ids.append((trip_id, "risky"))
            synthetic_labels[trip_id] = 1
            trip_index += 1

        db.commit()
        save_synthetic_labels(synthetic_labels)

        return {
            "created_count": len(created_trip_ids),
            "safe_count": safe_count,
            "moderate_count": moderate_count,
            "risky_count": risky_count,
            "samples_per_trip": samples_per_trip,
            "dt": dt,
            "seed": seed,
            "user_id": resolved_user_id,
            "created_trip_ids": created_trip_ids,
            "synthetic_labels_path": str(SYNTHETIC_LABELS_PATH),
        }
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic safe/risky trips")
    parser.add_argument("--count", type=int, default=DEFAULT_TOTAL_TRIPS, help="Total trips to generate")
    parser.add_argument("--user-id", type=str, default=None, help="Existing user ID to assign trips to")
    parser.add_argument("--samples-per-trip", type=int, default=DEFAULT_SAMPLES_PER_TRIP, help="Samples per trip")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_SECONDS, help="Seconds between samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    result = generate_synthetic_trips(
        count=args.count,
        user_id=args.user_id,
        samples_per_trip=args.samples_per_trip,
        dt=args.dt,
        seed=args.seed,
    )

    print(f"Generated {result['created_count']} synthetic trips for user {result['user_id']}")
    print(f"Safe trips:     {result['safe_count']}")
    print(f"Moderate trips: {result['moderate_count']}")
    print(f"Risky trips:    {result['risky_count']}")
    print(f"Synthetic label registry updated at: {result['synthetic_labels_path']}")
    print("Example trip IDs:")
    for trip_id, label in list(result["created_trip_ids"])[:10]:
        print(f"  {trip_id} -> {label}")


if __name__ == "__main__":
    main()
#python -m scripts.generate_synthetic_trips --count 50 --user-id YOUR_USER_ID

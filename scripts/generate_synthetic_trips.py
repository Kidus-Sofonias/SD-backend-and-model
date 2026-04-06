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
DEFAULT_SAMPLES_PER_TRIP = 240   # 240 * 0.5s = 120 seconds
DEFAULT_DT_SECONDS = 0.5

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


def base_location() -> tuple[float, float]:
    return 9.03, 38.74


def generate_safe_profile(samples_per_trip: int, dt_s: float) -> list[dict]:
    rows = []
    lat, lon = base_location()
    speed_kmh = random.uniform(28, 42)

    for i in range(samples_per_trip):
        t = i * dt_s

        speed_kmh += random.uniform(-0.25, 0.25) + 0.15 * math.sin(t / 10.0)
        speed_kmh = clamp(speed_kmh, 22, 55)

        ax = random.uniform(-0.15, 0.15) + 0.03 * math.sin(t / 3.5)
        ay = random.uniform(-0.12, 0.12) + 0.02 * math.cos(t / 5.0)
        az = 9.81 + random.uniform(-0.08, 0.08)

        gx = random.uniform(-0.05, 0.05)
        gy = random.uniform(-0.05, 0.05)
        gz = random.uniform(-0.12, 0.12) + 0.03 * math.sin(t / 6.0)

        lat += random.uniform(-0.00001, 0.00001) + speed_kmh * 0.0000004
        lon += random.uniform(-0.00001, 0.00001) + speed_kmh * 0.0000002

        rows.append({
            "speed": speed_kmh,   # current pipeline expects this as km/h input
            "lat": lat,
            "lon": lon,
            "accuracy_m": random.uniform(3.0, 8.0),
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
    lat, lon = base_location()
    speed_kmh = random.uniform(35, 50)

    hard_brake_centers = random.sample(range(30, samples_per_trip - 30), 5)
    hard_accel_centers = random.sample(range(30, samples_per_trip - 30), 5)
    turn_centers = random.sample(range(30, samples_per_trip - 30), 6)

    for i in range(samples_per_trip):
        t = i * dt_s

        delta = random.uniform(-0.8, 0.8)

        for c in hard_accel_centers:
            if abs(i - c) <= 4:
                delta += random.uniform(2.5, 5.5)

        for c in hard_brake_centers:
            if abs(i - c) <= 4:
                delta -= random.uniform(3.0, 6.0)

        speed_kmh += delta
        speed_kmh = clamp(speed_kmh, 5, 95)

        ax = random.uniform(-0.8, 0.8) + 0.18 * math.sin(t / 2.0)
        ay = random.uniform(-0.7, 0.7) + 0.14 * math.cos(t / 2.8)
        az = 9.81 + random.uniform(-0.25, 0.25)

        gx = random.uniform(-0.3, 0.3)
        gy = random.uniform(-0.3, 0.3)
        gz = random.uniform(-0.45, 0.45)

        for c in turn_centers:
            if abs(i - c) <= 5:
                gz += random.choice([-1, 1]) * random.uniform(1.2, 2.8)
                ay += random.choice([-1, 1]) * random.uniform(0.7, 1.5)

        lat += random.uniform(-0.00002, 0.00002) + speed_kmh * 0.00000045
        lon += random.uniform(-0.00002, 0.00002) + speed_kmh * 0.00000025

        rows.append({
            "speed": speed_kmh,
            "lat": lat,
            "lon": lon,
            "accuracy_m": random.uniform(4.0, 12.0),
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
            speed_mps=row["speed"],   # stored as current pipeline expects: km/h input
            lat=row["lat"],
            lon=row["lon"],
            accuracy_m=row["accuracy_m"],
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

        safe_count = count // 2
        risky_count = count - safe_count

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
    print(f"Safe trips:  {result['safe_count']}")
    print(f"Risky trips: {result['risky_count']}")
    print(f"Synthetic label registry updated at: {result['synthetic_labels_path']}")
    print("Example trip IDs:")
    for trip_id, label in list(result["created_trip_ids"])[:10]:
        print(f"  {trip_id} -> {label}")


if __name__ == "__main__":
    main()
#python -m scripts.generate_synthetic_trips --count 50 --user-id YOUR_USER_ID

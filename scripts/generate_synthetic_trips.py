# File role: Synthetic trip/sample generator for ML pipeline testing.
# Creates completed trips and realistic sensor samples directly in the database,
# and writes a synthetic label registry for dataset building.
#
# v2 — realistic driving patterns:
#  - Variable trip durations (2–35 min) instead of a fixed 10 min
#  - Stop-and-go traffic simulation with idle periods at intersections
#  - Speed transitions with smooth acceleration/deceleration ramps
#  - Realistic time-of-day clustering (morning/midday/evening commute)
#  - Multiple trips per day, rest days between driving days
#  - GPS tracks proportional to actual speed
#
# Intended for:
# - dataset building smoke tests
# - training pipeline testing
# - class-balance bootstrapping
# Not intended for final real-world model validation.
# Connects to:
# - app.db.session
# - app.db.models.user / trip / sensor_sample / driving_event
# - artifacts/datasets/synthetic_trip_labels.json
# Key symbols/vars:
# - SYNTHETIC_LABELS_PATH
# - generate_safe_profile / generate_moderate_profile / generate_risky_profile
# - create_trip_with_samples
# - generate_synthetic_trips
# - generate_realistic_trip_times  / pick_trip_duration

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import func, select

from app.core.security import hash_password
from app.db.session import SessionLocal
from app.db.models.user import User
from app.db.models.driving_event import DrivingEvent
from app.db.models.trip import Trip
from app.db.models.sensor_sample import SensorSample
from app.db.models.vehicle_profile import VehicleProfile

DEFAULT_TOTAL_TRIPS = 50
DEFAULT_DT_SECONDS = 0.3  # Sampling rate (~3 Hz)

# ---------------------------------------------------------------------------
# Realistic trip duration ranges (in seconds, before dividing by dt)
# ---------------------------------------------------------------------------
SHORT_TRIP_SAMPLES  = (400, 1400)    #  2 –  7 min
MEDIUM_TRIP_SAMPLES = (1400, 4000)   #  7 – 20 min
LONG_TRIP_SAMPLES   = (4000, 7000)   # 20 – 35 min

SYNTHETIC_LABELS_PATH = Path("artifacts/datasets/synthetic_trip_labels.json")
STRONG_LABELS_PATH = Path("artifacts/datasets/synthetic_ground_truth_labels.json")


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
# Ethiopian base locations for geographic diversity
# ---------------------------------------------------------------------------
LOCATIONS: list[tuple[float, float, float]] = [
    (9.0300, 38.7400, 2355.0),   # Addis Ababa centre
    (8.9800, 38.8000, 2320.0),   # Addis Bole area
    (9.0200, 38.6900, 2380.0),   # Addis summit area
    (9.0500, 38.7600, 2340.0),   # Addis north
    (8.5400, 39.2700, 1720.0),   # Adama (Nazret)
    (9.6000, 41.8600, 2430.0),   # Dire Dawa area
    (11.5700, 37.3900, 2130.0),  # Bahir Dar
    (7.0500, 38.5000, 1750.0),   # Hawassa
]


def pick_location(base_speed_mps: float, dt_s: float) -> tuple[float, float, float, float, float]:
    """Return (lat, lon, altitude_m, lat_drift, lon_drift) for a random route.

    Drift is calculated so that at the given speed, each sample moves the
    GPS position by the correct real-world distance.
    1 degree of latitude ≈ 111 km = 111000 m.
    """
    loc = random.choice(LOCATIONS)
    lat_rad = loc[0] * math.pi / 180
    # Random bearing in radians
    bearing_rad = random.uniform(0, 2 * math.pi)
    displacement_deg = base_speed_mps * dt_s / 111000.0
    drift_lat = math.cos(bearing_rad) * displacement_deg
    drift_lon = math.sin(bearing_rad) * displacement_deg / math.cos(lat_rad)
    return (*loc, drift_lat, drift_lon)


def altitude_wobble(base_alt: float, progress: float, roughness: float = 5.0) -> float:
    """Simulate gentle road elevation changes."""
    wave1 = math.sin(progress * math.pi * 0.7) * roughness
    wave2 = math.cos(progress * math.pi * 2.3) * roughness * 0.4
    wave3 = math.sin(progress * math.pi * 5.1) * roughness * 0.15
    return base_alt + wave1 + wave2 + wave3 + random.uniform(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Realistic trip time generation
# ---------------------------------------------------------------------------

def generate_realistic_trip_times(
    count: int,
    *,
    now: datetime | None = None,
    max_days_back: int = 30,
    daily_min_trips: int = 1,
    daily_max_trips: int = 3,
    seed: int = 42,
) -> list[datetime]:
    """Generate start times clustered around realistic driving hours.

    Time clusters with weights:
      - Morning commute  (07:00–09:00)  35 %
      - Midday errands   (11:00–14:00)  20 %
      - Evening commute  (16:00–19:00)  30 %
      - Evening leisure  (19:00–22:00)  15 %

    Some days have no trips (weekend-style rest days).
    Some days have 1–3 trips simulating errand patterns.
    """
    rng = random.Random(seed)

    if now is None:
        now = datetime.now(timezone.utc)

    # How many driving days do we need to spread `count` trips?
    trips_per_day_min = daily_min_trips
    trips_per_day_max = min(daily_max_trips, count)
    avg_trips_per_day = (trips_per_day_min + trips_per_day_max) / 2.0
    needed_days = max(1, math.ceil(count / avg_trips_per_day))

    # Pick active days from the available pool, including rest gaps
    available_days = max(needed_days, max_days_back)
    rest_days = max(1, available_days - needed_days)
    active_day_count = min(needed_days, available_days)

    # Build a pattern of active/rest days
    pattern: list[int] = []
    remaining_active = active_day_count
    remaining_total = available_days
    while remaining_active > 0 and remaining_total > 0:
        pattern.append(1)  # active day
        remaining_active -= 1
        remaining_total -= 1
        # Maybe add a rest day
        if remaining_total > remaining_active > 0 and rng.random() < 0.25:
            pattern.append(0)  # rest day
            remaining_total -= 1

    # Pad the rest with rest days if needed
    while remaining_total > 0:
        pattern.append(0)
        remaining_total -= 1

    # Time clusters with relative weights
    clusters: list[tuple[int, int, float]] = [
        (7,  9,  0.35),   # morning commute
        (11, 14, 0.20),   # midday
        (16, 19, 0.30),   # evening commute
        (19, 22, 0.15),   # evening leisure
    ]

    times: list[datetime] = []
    day_offset = 0
    trips_remaining = count

    for is_active in pattern:
        if trips_remaining <= 0:
            break
        if not is_active:
            day_offset += 1
            continue

        # How many trips on this active day?
        max_today = min(trips_per_day_max, trips_remaining)
        day_trips = rng.randint(trips_per_day_min, max_today)
        trips_remaining -= day_trips

        # Pick distinct time clusters for each trip on this day
        day_hours: list[float] = []
        for _ in range(day_trips):
            cluster = rng.choices(clusters, weights=[c[2] for c in clusters])[0]
            hour_f = rng.uniform(cluster[0], cluster[1])
            # Add a little jitter so two trips on the same day don't overlap
            for existing in day_hours:
                if abs(hour_f - existing) < 0.5:
                    hour_f = (hour_f + 1.0) % 24
            day_hours.append(hour_f)

        day_hours.sort()

        for hour_f in day_hours:
            h = int(hour_f)
            m = int((hour_f - h) * 60)
            s = rng.randint(0, 59)
            dt = now - timedelta(days=day_offset)
            try:
                dt = dt.replace(hour=h, minute=m, second=s, microsecond=0)
            except ValueError:
                dt = dt.replace(hour=h, minute=m, second=0, microsecond=0)
            times.append(dt)

        day_offset += 1

    times.sort()
    return times


def pick_trip_duration(
    *,
    samples_per_trip: int | None = None,
    rng: random.Random | None = None,
) -> int:
    """Return a realistic number of samples for one trip.

    If samples_per_trip is given, it is used directly (backward compat).
    Otherwise picks from short/medium/long ranges with weighted probabilities:
      - Short  (2–7 min)   30 %
      - Medium (7–20 min)  50 %
      - Long   (20–35 min) 20 %
    """
    if samples_per_trip is not None:
        return samples_per_trip

    rng = rng or random
    roll = rng.random()
    if roll < 0.30:
        lo, hi = SHORT_TRIP_SAMPLES
    elif roll < 0.80:
        lo, hi = MEDIUM_TRIP_SAMPLES
    else:
        lo, hi = LONG_TRIP_SAMPLES
    return rng.randint(lo, hi)


# ---------------------------------------------------------------------------
# Stop-and-go traffic helpers
# ---------------------------------------------------------------------------

def _build_stop_pattern(
    n_samples: int,
    *,
    avg_interval: int = 500,
    stop_duration: tuple[int, int] = (8, 35),
    rng: random.Random | None = None,
) -> set[int]:
    """Return set of indices where a traffic stop occurs (idle at intersection).

    Each stop lasts for *stop_duration* samples (roughly 2.4–10.5 sec at 0.3s dt).
    """
    rng = rng or random
    stops: set[int] = set()
    pos = rng.randint(200, 400)  # first stop after a warm-up period
    while pos < n_samples - 200:
        dur = rng.randint(*stop_duration)
        for i in range(pos, min(pos + dur, n_samples - 1)):
            stops.add(i)
        pos += rng.randint(avg_interval - 100, avg_interval + 100)
    return stops


def _build_accel_decel_events(
    n_samples: int,
    *,
    stop_indices: set[int],
    event_count: int,
    severity_range: tuple[float, float],
    rng: random.Random | None = None,
) -> list[int]:
    """Place aggressive accel/decel events away from stop zones."""
    rng = rng or random
    margin = int(n_samples * 0.08)
    pool = [
        i for i in range(margin, n_samples - margin)
        if not any(abs(i - s) < 40 for s in stop_indices)
    ]
    if not pool:
        return []
    return rng.sample(pool, min(event_count, len(pool)))


# ---------------------------------------------------------------------------
# Enhanced driving profiles
# ---------------------------------------------------------------------------

def _cruise_speed(
    base: float,
    t: float,
    variation_amp: float = 0.15,
    slow_wave: float = 12.0,
    fast_wave: float = 4.5,
) -> float:
    """Add smooth sinusoidal variation around a base cruising speed."""
    return base + variation_amp * (
        0.6 * math.sin(t / slow_wave) + 0.4 * math.cos(t / fast_wave)
    )


def _gps_step(
    lat: float, lon: float,
    drift_lat: float, drift_lon: float,
    speed_ratio: float,
    gps_noise: float = 0.000008,
) -> tuple[float, float]:
    """Advance GPS coordinates proportional to current speed ratio."""
    lat += drift_lat * speed_ratio + random.uniform(-gps_noise, gps_noise)
    lon += drift_lon * speed_ratio + random.uniform(-gps_noise, gps_noise)
    return lat, lon


# =====================================================================
# SAFE PROFILE — Smooth city driving, few events
# =====================================================================

def generate_safe_profile(n_samples: int, dt_s: float) -> list[dict]:
    """Smooth, cautious driving with periodic stops at intersections.

    Speed range: 10–40 km/h, gentle acceleration, rare harsh events.
    """
    base_speed_kmh = random.uniform(18, 30)
    base_speed_mps = base_speed_kmh / 3.6
    lat, lon, base_alt, drift_lat, drift_lon = pick_location(base_speed_mps, dt_s)

    # Traffic patterns
    stop_indices = _build_stop_pattern(n_samples, avg_interval=550, stop_duration=(8, 25))

    rows: list[dict] = []
    speed_kmh = 0.0  # start from stop

    for i in range(n_samples):
        t = i * dt_s
        progress = i / n_samples

        # Check if we're in a stop zone
        if i in stop_indices:
            # Inside a stop — decelerate quickly to 0 then idle
            speed_kmh *= 0.80
            if speed_kmh < 1.5:
                speed_kmh = 0.0
        else:
            # Accelerate smoothly from 0 back to cruise
            if speed_kmh < 3.0:
                speed_kmh += random.uniform(0.6, 1.8)  # gentle acceleration
            else:
                # Normal cruising with slight variation
                target = _cruise_speed(base_speed_kmh, t, variation_amp=0.08)
                speed_kmh += (target - speed_kmh) * 0.04 + random.uniform(-0.12, 0.12)

        speed_kmh = clamp(speed_kmh, 0, 40)

        # IMU — gentle
        ax = random.uniform(-0.10, 0.10) + 0.02 * math.sin(t / 4.0)
        ay = random.uniform(-0.08, 0.08) + 0.02 * math.cos(t / 6.0)
        az = 9.81 + random.uniform(-0.06, 0.06)

        gx = random.uniform(-0.04, 0.04)
        gy = random.uniform(-0.04, 0.04)
        gz = random.uniform(-0.10, 0.10) + 0.02 * math.sin(t / 7.0)

        # GPS
        current_speed_mps = speed_kmh / 3.6
        speed_ratio = current_speed_mps / max(base_speed_mps, 0.01)
        lat, lon = _gps_step(lat, lon, drift_lat, drift_lon, speed_ratio, gps_noise=0.000006)
        alt = altitude_wobble(base_alt, progress, roughness=3.0)

        rows.append({
            "speed": speed_kmh,  # Store in km/h (pipeline expects this unit)
            "lat": lat,
            "lon": lon,
            "accuracy_m": random.uniform(2.5, 6.0),
            "altitude_m": alt,
            "ax": ax, "ay": ay, "az": az,
            "gx": gx, "gy": gy, "gz": gz,
        })

    return rows


# =====================================================================
# MODERATE PROFILE — Typical city driving with occasional events
# =====================================================================

def generate_moderate_profile(n_samples: int, dt_s: float) -> list[dict]:
    """Moderate driving: mix of smooth cruising and mild events.

    Speed range: 10–50 km/h, some braking/acceleration events,
    periodic stops, occasional sharper turns.
    """
    base_speed_kmh = random.uniform(22, 35)
    base_speed_mps = base_speed_kmh / 3.6
    lat, lon, base_alt, drift_lat, drift_lon = pick_location(base_speed_mps, dt_s)

    # Stops + events
    stop_indices = _build_stop_pattern(n_samples, avg_interval=450, stop_duration=(10, 30))
    event_density = n_samples / 400.0
    n_events = random.randint(max(2, int(3 * event_density * 0.5)), max(5, int(6 * event_density * 0.6)))
    event_centers = _build_accel_decel_events(
        n_samples, stop_indices=stop_indices,
        event_count=n_events, severity_range=(1.2, 3.0),
    )

    rows: list[dict] = []
    speed_kmh = 0.0

    for i in range(n_samples):
        t = i * dt_s
        progress = i / n_samples

        # Stop zones
        if i in stop_indices:
            speed_kmh *= 0.80
            if speed_kmh < 1.5:
                speed_kmh = 0.0
        else:
            if speed_kmh < 3.0:
                speed_kmh += random.uniform(0.8, 2.5)
            else:
                delta = random.uniform(-0.3, 0.3)
                # Moderate events
                for c in event_centers:
                    dist = abs(i - c)
                    if dist <= 6:
                        intensity = 1.0 - dist / 6.0
                        if random.random() < 0.5:
                            delta -= intensity * random.uniform(1.2, 3.0)
                        else:
                            delta += intensity * random.uniform(1.2, 3.0)
                target = _cruise_speed(base_speed_kmh, t, variation_amp=0.15)
                speed_kmh += (target - speed_kmh) * 0.04 + delta

        speed_kmh = clamp(speed_kmh, 0, 50)

        # IMU — moderate
        ax = random.uniform(-0.3, 0.3) + 0.08 * math.sin(t / 3.0)
        ay = random.uniform(-0.25, 0.25) + 0.06 * math.cos(t / 4.0)
        az = 9.81 + random.uniform(-0.12, 0.12)

        gx = random.uniform(-0.12, 0.12)
        gy = random.uniform(-0.12, 0.12)
        gz = random.uniform(-0.20, 0.20) + 0.10 * math.sin(t / 5.0)

        # GPS
        current_speed_mps = speed_kmh / 3.6
        speed_ratio = current_speed_mps / max(base_speed_mps, 0.01)
        lat, lon = _gps_step(lat, lon, drift_lat, drift_lon, speed_ratio, gps_noise=0.00001)
        alt = altitude_wobble(base_alt, progress, roughness=4.0)

        rows.append({
            "speed": speed_kmh,  # Store in km/h (pipeline expects this unit)
            "lat": lat,
            "lon": lon,
            "accuracy_m": random.uniform(3.0, 9.0),
            "altitude_m": alt,
            "ax": ax, "ay": ay, "az": az,
            "gx": gx, "gy": gy, "gz": gz,
        })

    return rows


# =====================================================================
# RISKY PROFILE — Aggressive driving, hard events, high speeds
# =====================================================================

def generate_risky_profile(n_samples: int, dt_s: float) -> list[dict]:
    """Aggressive driving with hard braking, rapid acceleration, sharp turns.

    Speed range: 0–60 km/h, frequent harsh events, less idle time.
    """
    base_speed_kmh = random.uniform(28, 42)
    base_speed_mps = base_speed_kmh / 3.6
    lat, lon, base_alt, drift_lat, drift_lon = pick_location(base_speed_mps, dt_s)

    # Fewer stops (aggressive drivers run yellows!)
    stop_indices = _build_stop_pattern(
        n_samples, avg_interval=700, stop_duration=(5, 15)
    )

    # Many events
    event_density = n_samples / 400.0
    event_range = range(int(n_samples * 0.08), int(n_samples * 0.92))
    eligible = [
        i for i in event_range
        if not any(abs(i - s) < 30 for s in stop_indices)
    ]
    n_brakes = min(max(5, int(6 * event_density)), len(eligible) // 3)
    n_accels = min(max(5, int(6 * event_density)), len(eligible) // 3)
    n_turns  = min(max(6, int(8 * event_density)), len(eligible) // 3)

    hard_brake_centers = random.sample(eligible, n_brakes) if eligible else []
    hard_accel_centers = random.sample(
        [i for i in eligible if i not in hard_brake_centers], n_accels
    ) if len(eligible) > n_brakes else []
    turn_centers = random.sample(
        [i for i in eligible if i not in hard_brake_centers and i not in hard_accel_centers],
        n_turns,
    ) if len(eligible) > n_brakes + n_accels else []

    rows: list[dict] = []
    speed_kmh = 5.0  # rolling start

    for i in range(n_samples):
        t = i * dt_s
        progress = i / n_samples

        # Stop zones
        if i in stop_indices:
            speed_kmh *= 0.75
            if speed_kmh < 2.0:
                speed_kmh = 0.0
        else:
            if speed_kmh < 5.0:
                speed_kmh += random.uniform(1.5, 4.0)  # quick launch
            else:
                delta = random.uniform(-0.8, 0.8)

                # Hard accelerations
                for c in hard_accel_centers:
                    if abs(i - c) <= 5:
                        delta += random.uniform(2.8, 6.5)

                # Hard brakes
                for c in hard_brake_centers:
                    if abs(i - c) <= 5:
                        delta -= random.uniform(3.5, 7.0)

                target = _cruise_speed(base_speed_kmh, t, variation_amp=0.25)
                speed_kmh += (target - speed_kmh) * 0.06 + delta

        speed_kmh = clamp(speed_kmh, 0, 60)

        # IMU — aggressive
        ax = random.uniform(-0.8, 0.8) + 0.20 * math.sin(t / 1.8)
        ay = random.uniform(-0.6, 0.6) + 0.16 * math.cos(t / 2.5)
        az = 9.81 + random.uniform(-0.30, 0.30)

        gx = random.uniform(-0.30, 0.30)
        gy = random.uniform(-0.30, 0.30)
        gz = random.uniform(-0.40, 0.40)

        # Sharp turns add severe gyro + lateral accel
        for c in turn_centers:
            if abs(i - c) <= 6:
                turn_mag = random.uniform(1.5, 3.5) * (-1 if random.random() < 0.5 else 1)
                gz += turn_mag
                ay += turn_mag * random.uniform(0.4, 0.7)
                ax += random.uniform(-0.3, 0.3)

        # GPS
        current_speed_mps = speed_kmh / 3.6
        speed_ratio = current_speed_mps / max(base_speed_mps, 0.01)
        lat, lon = _gps_step(lat, lon, drift_lat, drift_lon, speed_ratio, gps_noise=0.00002)
        alt = altitude_wobble(base_alt, progress, roughness=7.0)

        rows.append({
            "speed": speed_kmh,  # Store in km/h (pipeline expects this unit)
            "lat": lat,
            "lon": lon,
            "accuracy_m": random.uniform(4.0, 14.0),
            "altitude_m": alt,
            "ax": ax, "ay": ay, "az": az,
            "gx": gx, "gy": gy, "gz": gz,
        })

    return rows


# =====================================================================
# Trip builder
# =====================================================================

def create_trip_with_samples(
    db,
    user_id: str,
    rows: list[dict],
    started_at: datetime,
    dt_s: float,
    vehicle_profile_id: str | None = None,
) -> str:
    trip = Trip(
        user_id=user_id,
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=dt_s * len(rows)),
        status="completed",
        vehicle_profile_id=vehicle_profile_id,
    )
    db.add(trip)
    db.flush()

    for idx, row in enumerate(rows):
        ts = started_at + timedelta(seconds=idx * dt_s)

        sample = SensorSample(
            user_id=user_id,
            trip_id=trip.id,
            ts=ts,
            speed_mps=row["speed"] / 3.6,
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


def load_existing_strong_labels() -> dict[str, int]:
    if not STRONG_LABELS_PATH.exists():
        return {}
    try:
        return json.loads(STRONG_LABELS_PATH.read_text(encoding="utf-8"))
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


# =====================================================================
# Vehicle-aware fleet (Phase 8b)
# =====================================================================

SYNTH_VEHICLE_EMAIL = "synth.{category}@drivepulse.test"
SYNTH_VEHICLE_PASSWORD = "synth1234"


def ensure_synth_vehicle_driver(db, category: str) -> tuple[str, str]:
    """Create (or reuse) a synthetic driver with a VehicleProfile for category.

    Returns (user_id, vehicle_profile_id). One driver per category keeps the
    trips' vehicle context stable and lets build_training_dataset score each
    trip through its own tuned config (config_for_profile).
    """
    email = SYNTH_VEHICLE_EMAIL.format(category=category)
    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user is None:
        user = User(
            id=str(uuid.uuid4()),
            email=email,
            password_hash=hash_password(SYNTH_VEHICLE_PASSWORD),
            role="driver",
        )
        db.add(user)
        db.flush()
    profile = db.execute(
        select(VehicleProfile).where(VehicleProfile.user_id == user.id)
    ).scalar_one_or_none()
    if profile is None:
        profile = VehicleProfile(user_id=user.id, category=category)
        db.add(profile)
        db.flush()
    return user.id, profile.id


# =====================================================================
# Main generator entry point
# =====================================================================

def generate_synthetic_trips(
    *,
    count: int,
    user_id: str | None = None,
    samples_per_trip: int | None = None,
    dt: float = DEFAULT_DT_SECONDS,
    seed: int = 42,
    strong_labels: bool = True,
    vehicle_categories: list[str] | None = None,
) -> dict[str, object]:
    """
    Generate synthetic trips with realistic variable durations and start times.

    If *samples_per_trip* is provided, all trips get the same length (backward
    compat).  Otherwise each trip gets a variable duration (2–35 min).

    strong_labels=True: Only generate safe_profile (label 0) and risky_profile
    (label 1) — skip moderate trips.  Saves ground-truth labels separately so
    the dataset builder can use them as strong labels instead of weak rule labels.

    vehicle_categories (Phase 8b): when provided, creates a synthetic fleet of
    drivers — one per category, each with a VehicleProfile — and distributes
    the trips round-robin across them. Every trip then carries a
    vehicle_profile_id so the training dataset is built through the
    vehicle-aware detection signal (config_for_profile per trip).
    """
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
            "vehicle_categories": vehicle_categories,
            "created_trip_ids": [],
            "synthetic_labels_path": str(SYNTHETIC_LABELS_PATH),
        }

    random.seed(seed)

    db = SessionLocal()
    try:
        # Phase 8b: resolve the trip owners first. With vehicle_categories the
        # trips are spread across a synthetic fleet (one driver per category);
        # otherwise they all go to the single resolved user (backward compat).
        if vehicle_categories:
            fleet: list[tuple[str, str]] = [
                ensure_synth_vehicle_driver(db, category)
                for category in vehicle_categories
            ]
            resolved_user_id = fleet[0][0]
        else:
            fleet = []
            resolved_user_id = resolve_user_id(db, user_id)

        def _owner_for(index: int) -> tuple[str, str | None]:
            """(user_id, vehicle_profile_id) for the trip at generation index."""
            if not fleet:
                return resolved_user_id, None
            return fleet[index % len(fleet)]

        if strong_labels:
            # Only safe + risky extremes — no moderate ambiguity
            safe_count = count // 2
            risky_count = count - safe_count
            moderate_count = 0
        else:
            safe_count = count // 3
            moderate_count = count // 3
            risky_count = count - safe_count - moderate_count

        now = datetime.now(timezone.utc)

        # ----- realistic start times across all trips -----
        trip_times = generate_realistic_trip_times(
            count, now=now, max_days_back=28, seed=seed,
        )

        created_trip_ids: list[tuple[str, str]] = []
        synthetic_labels = load_existing_synthetic_labels()
        # Merge with existing ground truth labels instead of overwriting
        strong_ground_truth: dict[str, int] = load_existing_strong_labels()  # trip_id -> label (0=safe, 1=risky)

        trip_index = 0

        # Safe trips
        for _ in range(safe_count):
            owner_user_id, owner_profile_id = _owner_for(trip_index)
            n_samp = pick_trip_duration(samples_per_trip=samples_per_trip)
            started_at = trip_times[trip_index] if trip_index < len(trip_times) else (
                now - timedelta(days=trip_index + 1, minutes=random.randint(0, 120))
            )
            rows = generate_safe_profile(n_samp, dt)
            trip_id = create_trip_with_samples(
                db, owner_user_id, rows, started_at, dt,
                vehicle_profile_id=owner_profile_id,
            )
            created_trip_ids.append((trip_id, "safe"))
            synthetic_labels[trip_id] = 0
            strong_ground_truth[trip_id] = 0
            trip_index += 1

        # Moderate trips (only when strong_labels=False)
        for _ in range(moderate_count):
            owner_user_id, owner_profile_id = _owner_for(trip_index)
            n_samp = pick_trip_duration(samples_per_trip=samples_per_trip)
            started_at = trip_times[trip_index] if trip_index < len(trip_times) else (
                now - timedelta(days=trip_index + 1, minutes=random.randint(0, 120))
            )
            rows = generate_moderate_profile(n_samp, dt)
            trip_id = create_trip_with_samples(
                db, owner_user_id, rows, started_at, dt,
                vehicle_profile_id=owner_profile_id,
            )
            created_trip_ids.append((trip_id, "moderate"))
            synthetic_labels[trip_id] = 0
            trip_index += 1

        # Risky trips
        for _ in range(risky_count):
            owner_user_id, owner_profile_id = _owner_for(trip_index)
            n_samp = pick_trip_duration(samples_per_trip=samples_per_trip)
            started_at = trip_times[trip_index] if trip_index < len(trip_times) else (
                now - timedelta(days=trip_index + 1, minutes=random.randint(0, 120))
            )
            rows = generate_risky_profile(n_samp, dt)
            trip_id = create_trip_with_samples(
                db, owner_user_id, rows, started_at, dt,
                vehicle_profile_id=owner_profile_id,
            )
            created_trip_ids.append((trip_id, "risky"))
            synthetic_labels[trip_id] = 1
            strong_ground_truth[trip_id] = 1
            trip_index += 1

        db.commit()
        save_synthetic_labels(synthetic_labels)

        # Save ground-truth labels for strong-label training
        if strong_labels and strong_ground_truth:
            STRONG_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
            STRONG_LABELS_PATH.write_text(
                json.dumps(strong_ground_truth, indent=2), encoding="utf-8"
            )
            print(f"Saved {len(strong_ground_truth)} ground-truth labels to {STRONG_LABELS_PATH}")

        return {
            "created_count": len(created_trip_ids),
            "safe_count": safe_count,
            "moderate_count": moderate_count,
            "risky_count": risky_count,
            "samples_per_trip": samples_per_trip,
            "dt": dt,
            "seed": seed,
            "user_id": resolved_user_id,
            "vehicle_categories": vehicle_categories,
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
    parser.add_argument("--samples-per-trip", type=int, default=None, help="Fixed samples per trip (omit for variable 2–35 min)")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_SECONDS, help="Seconds between samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--strong-labels", action="store_true", default=True,
                        help="Generate only safe+risky extremes (no moderate) with ground-truth labels")
    parser.add_argument("--no-strong-labels", dest="strong_labels", action="store_false")
    parser.add_argument("--vehicle-categories", type=str, default=None,
                        help="Comma-separated vehicle categories for a synthetic fleet "
                             "(e.g. sedan,suv,pickup,van,bus,heavy_truck,tractor_trailer). "
                             "Each trip gets a vehicle profile so the dataset is built through "
                             "the vehicle-aware detection signal.")
    args = parser.parse_args()

    vehicle_categories = None
    if args.vehicle_categories:
        vehicle_categories = [c.strip() for c in args.vehicle_categories.split(",") if c.strip()]

    result = generate_synthetic_trips(
        count=args.count,
        user_id=args.user_id,
        samples_per_trip=args.samples_per_trip,
        dt=args.dt,
        seed=args.seed,
        strong_labels=args.strong_labels,
        vehicle_categories=vehicle_categories,
    )

    print(f"Generated {result['created_count']} synthetic trips for user {result['user_id']}")
    print(f"Safe trips:     {result['safe_count']}")
    print(f"Moderate trips: {result['moderate_count']}")
    print(f"Risky trips:    {result['risky_count']}")
    if result.get("samples_per_trip") is None:
        print(f"Trip durations: variable (2–35 min)")
    else:
        print(f"Samples/trip:   {result['samples_per_trip']}")
    print(f"Synthetic label registry updated at: {result['synthetic_labels_path']}")
    if result.get("vehicle_categories"):
        print(f"Vehicle fleet:   {', '.join(result['vehicle_categories'])}")
    print("Example trip IDs:")
    for trip_id, label in list(result["created_trip_ids"])[:10]:
        print(f"  {trip_id} -> {label}")


if __name__ == "__main__":
    main()
#python -m scripts.generate_synthetic_trips --count 50 --user-id YOUR_USER_ID

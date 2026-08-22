"""Phase 8b (hackathon) — vehicle-aware ML training signal tests.

Covers:
- run_trip_pipeline emits log_vehicle_mass_kg for every trip (default = sedan)
- The vehicle feature reflects the profile's resolved mass
- build_training_dataset scores each trip with its own tuned config and
  records vehicle context columns
- generate_synthetic_trips fleet mode creates profiles + attaches them to trips
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models.sensor_sample import SensorSample
from app.db.models.trip import Trip
from app.db.models.user import User
from app.db.models.vehicle_profile import VehicleProfile
from app.ml.config import FeatureConfigV2
from app.ml.pipeline import run_trip_pipeline
from app.ml.schemas import FEATURE_COLUMNS_FV1
from app.ml.vehicle_profiles import config_for_profile, resolve_mass_kg, vehicle_feature_row


def _session_factory(tmp_path: Path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'ml-vehicle-aware.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


def _profile(category: str, size_class: str | None = None, mass_kg: float | None = None) -> VehicleProfile:
    return VehicleProfile(
        id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        category=category,
        size_class=size_class,
        mass_kg=mass_kg,
    )


def _trip_samples(speed_kph_start: float = 60.0, decel: float = 6.0, n: int = 40) -> list[dict]:
    base_ts = datetime(2026, 8, 13, 10, 0, 0, tzinfo=timezone.utc)
    samples = []
    speed = speed_kph_start
    for i in range(n):
        if 12 <= i < 24:
            speed -= decel
        ts = (base_ts + __import__("datetime").timedelta(seconds=i * 0.5)).isoformat().replace("+00:00", "Z")
        samples.append(
            {
                "timestamp": ts,
                "speed": max(speed, 0.0),
                "lat": 9.03 + i * 0.0001,
                "lon": 38.74 + i * 0.0001,
                "accuracy_m": 6.0,
                "ax": 0.2,
                "ay": 0.3,
                "az": 9.8,
                "gx": 0.01,
                "gy": 0.01,
                "gz": 0.01,
            }
        )
    return samples


# ---------------------------------------------------------------------------
# Pipeline emits the vehicle feature
# ---------------------------------------------------------------------------

def test_pipeline_emits_log_vehicle_mass_with_default() -> None:
    result = run_trip_pipeline(_trip_samples(), FeatureConfigV2())
    assert "log_vehicle_mass_kg" in result["trip_features"]
    assert result["trip_features"]["log_vehicle_mass_kg"] == round(__import__("math").log10(1400.0), 4)


def test_pipeline_emits_vehicle_mass_from_profile() -> None:
    truck = _profile("heavy_truck")
    result = run_trip_pipeline(_trip_samples(), FeatureConfigV2(), vehicle_profile=truck)
    assert result["trip_features"]["log_vehicle_mass_kg"] == round(__import__("math").log10(18000.0), 4)


def test_vehicle_feature_is_in_model_contract() -> None:
    assert "log_vehicle_mass_kg" in FEATURE_COLUMNS_FV1
    # Pipeline output must satisfy the contract for every scored trip.
    result = run_trip_pipeline(_trip_samples(), FeatureConfigV2())
    for column in FEATURE_COLUMNS_FV1:
        assert column in result["trip_features"]


def test_vehicle_feature_row_defaults_and_tuning() -> None:
    assert vehicle_feature_row(None)["log_vehicle_mass_kg"] == round(__import__("math").log10(1400.0), 4)
    compact_sedan = _profile("sedan", size_class="compact")
    assert vehicle_feature_row(compact_sedan)["log_vehicle_mass_kg"] == round(
        __import__("math").log10(1400.0 * 0.85), 4
    )


# ---------------------------------------------------------------------------
# Dataset builder uses per-trip tuned config + records vehicle columns
# ---------------------------------------------------------------------------

def test_dataset_builder_vehicle_aware(tmp_path: Path, monkeypatch) -> None:
    session_factory = _session_factory(tmp_path)
    db = session_factory()

    user = User(id=str(uuid.uuid4()), email="ml-va@example.com", password_hash="x")
    db.add(user)
    db.flush()
    sedan = VehicleProfile(user_id=user.id, category="sedan")
    truck = VehicleProfile(user_id=str(uuid.uuid4()), category="heavy_truck")
    db.add_all([sedan, truck])
    db.flush()

    # Same driving profile, different vehicles. With the truck's tuned config
    # (lower threshold), the same deceleration is a hard/emergency brake.
    now = datetime(2026, 8, 13, 11, 0, 0, tzinfo=timezone.utc)
    for idx, (owner, profile) in enumerate([(user.id, sedan), (user.id, truck)]):
        trip = Trip(
            id=str(uuid.uuid4()),
            user_id=owner,
            vehicle_profile_id=profile.id,
            started_at=now,
            ended_at=now,
            status="completed",
        )
        db.add(trip)
        db.flush()
        from datetime import timedelta

        for j, row in enumerate(_trip_samples(decel=4.0)):
            db.add(
                SensorSample(
                    user_id=owner,
                    trip_id=trip.id,
                    ts=now + timedelta(seconds=j * 0.5),
                    speed_mps=row["speed"] / 3.6,
                    lat=row["lat"],
                    lon=row["lon"],
                    accuracy_m=row["accuracy_m"],
                    altitude_m=100.0,
                    ax=row["ax"],
                    ay=row["ay"],
                    az=row["az"],
                    gx=row["gx"],
                    gy=row["gy"],
                    gz=row["gz"],
                )
            )
        db.flush()
        # synthetic strong label: both trips are "risky" (label 1)
        trip.reviewed_label = 1
        trip.reviewed_label_source = "synthetic_ground_truth"
    db.commit()
    db.close()

    # Point the dataset builder at this DB + temp artifact paths.
    import scripts.build_training_dataset as btd

    monkeypatch.setattr(btd, "SessionLocal", session_factory)
    out = tmp_path / "trip_features_va.csv"
    report = tmp_path / "dataset_summary_va.json"
    labels = tmp_path / "strong_labels.json"
    labels.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr(btd, "OUTPUT_PATH", out)
    monkeypatch.setattr(btd, "REPORT_PATH", report)
    monkeypatch.setattr(btd, "STRONG_LABELS_PATH", labels)
    monkeypatch.setattr(btd, "SYNTHETIC_LABELS_PATH", tmp_path / "syn.json")
    monkeypatch.setattr(btd, "REVIEWED_LABELS_PATH", tmp_path / "rev.json")

    summary = btd.main()
    assert summary is not None
    assert summary["row_count"] == 2

    df = pd.read_csv(out)
    assert set(df["vehicle_category"]) == {"sedan", "heavy_truck"}
    truck_row = df[df["vehicle_category"] == "heavy_truck"].iloc[0]
    sedan_row = df[df["vehicle_category"] == "sedan"].iloc[0]
    assert truck_row["vehicle_mass_kg"] == 18000.0
    assert sedan_row["vehicle_mass_kg"] == 1400.0
    assert truck_row["log_vehicle_mass_kg"] == round(__import__("math").log10(18000.0), 4)
    # Truck's lower brake threshold -> it detects the same deceleration as
    # harsher, so its event counts must be >= the sedan's.
    assert truck_row["emergency_brake_count"] >= sedan_row["emergency_brake_count"]
    assert truck_row["harsh_brake_count"] >= sedan_row["harsh_brake_count"]


# ---------------------------------------------------------------------------
# Synthetic generator fleet mode
# ---------------------------------------------------------------------------

def test_generator_fleet_attaches_vehicle_profiles(tmp_path: Path, monkeypatch) -> None:
    session_factory = _session_factory(tmp_path)
    import scripts.generate_synthetic_trips as gen

    monkeypatch.setattr(gen, "SessionLocal", session_factory)
    monkeypatch.setattr(gen, "SYNTHETIC_LABELS_PATH", tmp_path / "syn_labels.json")
    monkeypatch.setattr(gen, "STRONG_LABELS_PATH", tmp_path / "strong_labels.json")

    result = gen.generate_synthetic_trips(
        count=8,
        samples_per_trip=240,
        dt=0.5,
        seed=7,
        strong_labels=True,
        vehicle_categories=["sedan", "heavy_truck"],
    )
    assert result["created_count"] == 8

    db = session_factory()
    trips = db.execute(select(Trip)).scalars().all()
    assert len(trips) == 8
    profiles = db.execute(select(VehicleProfile)).scalars().all()
    assert {p.category for p in profiles} == {"sedan", "heavy_truck"}
    attached = {t.vehicle_profile_id for t in trips}
    assert attached == {p.id for p in profiles}
    # Round-robin: both categories present among trips.
    assert {t.vehicle_profile_id for t in trips} == {p.id for p in profiles}
    db.close()

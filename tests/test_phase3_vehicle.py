"""Phase 3 (hackathon) — vehicle-aware driver onboarding tests.

Covers:
- Vehicle profile API (GET 404 before set, PUT upsert, derived thresholds)
- Threshold scaling across vehicle classes (heavy vs sedan)
- config_for_profile falls back to the universal default
- Trip start records the vehicle; finalize tunes detection and records context
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.deps import get_current_user
from app.db.base import Base
from app.db.models.user import User
from app.db.session import get_db
from app.main import app
from app.ml.config import FeatureConfigV2
from app.ml.vehicle_profiles import (
    config_for_profile,
    longitudinal_scale,
    profile_thresholds,
    resolve_mass_kg,
    unstable_jerk_scale,
)
from app.repositories.user_repository import UserRecord


def _make_session_factory(tmp_path: Path):
    db_path = tmp_path / "phase3-vehicle.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


def _client_with_overrides(session_factory, user: UserRecord) -> TestClient:
    def _override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    def _override_get_current_user():
        return user

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_user] = _override_get_current_user
    return TestClient(app)


# ---------------------------------------------------------------------------
# Threshold scaling
# ---------------------------------------------------------------------------

def test_mass_resolution_respects_category_and_size_class() -> None:
    assert resolve_mass_kg("sedan") == 1400.0
    assert resolve_mass_kg("heavy_truck") == 18000.0
    assert resolve_mass_kg("sedan", size_class="compact") == 1400.0 * 0.85
    assert resolve_mass_kg("sedan", size_class="large") == 1400.0 * 1.2
    assert resolve_mass_kg("sedan", mass_kg=900) == 900.0


def test_heavy_vehicles_get_more_sensitive_thresholds() -> None:
    sedan = profile_thresholds(type("P", (), {"category": "sedan", "size_class": None, "mass_kg": None})())
    truck = profile_thresholds(type("P", (), {"category": "heavy_truck", "size_class": None, "mass_kg": None})())

    # Heavier -> smaller MAGNITUDE (more sensitive) brake/accel/turn
    # thresholds: a truck braking at -1.7 m/s^2 is already harsh, a sedan
    # needs -3.2 m/s^2.
    assert abs(truck["harsh_brake_dv"]) < abs(sedan["harsh_brake_dv"])
    assert abs(truck["harsh_accel_dv"]) < abs(sedan["harsh_accel_dv"])
    assert abs(truck["emergency_brake_dv"]) < abs(sedan["emergency_brake_dv"])
    assert abs(truck["aggressive_turn_threshold"]) < abs(sedan["aggressive_turn_threshold"])
    # ...but a HIGHER rough-road jerk floor (more suspension travel tolerated).
    assert truck["unstable_motion_jerk_threshold"] > sedan["unstable_motion_jerk_threshold"]


def test_longitudinal_scale_is_clamped() -> None:
    assert 0.5 <= longitudinal_scale(18000.0) <= 1.2
    assert 0.5 <= longitudinal_scale(400.0) <= 1.2
    assert unstable_jerk_scale(36000.0) <= 1.6
    assert unstable_jerk_scale(400.0) >= 0.85


def test_config_for_profile_defaults_when_missing() -> None:
    base = FeatureConfigV2()
    assert config_for_profile(base, None) is base
    assert config_for_profile(base, type("P", (), {"category": None})()) is base


def test_config_for_profile_tunes_thresholds() -> None:
    base = FeatureConfigV2()
    truck = type("P", (), {"category": "heavy_truck", "size_class": None, "mass_kg": None})()
    tuned = config_for_profile(base, truck)
    assert tuned is not base
    assert abs(tuned.harsh_brake_dv) < abs(base.harsh_brake_dv)
    assert abs(tuned.aggressive_turn_threshold) < abs(base.aggressive_turn_threshold)
    assert tuned.unstable_motion_jerk_threshold > base.unstable_motion_jerk_threshold


def test_config_for_profile_does_not_mutate_input() -> None:
    # The shared service config must never be contaminated by a vehicle tune:
    # a truck trip followed by a sedan trip (or a default fallback) must each
    # see pristine thresholds. dataclasses.replace on the frozen config is
    # expected to produce a new object and leave the input untouched.
    base = FeatureConfigV2()
    before = {
        "harsh_brake_dv": base.harsh_brake_dv,
        "harsh_accel_dv": base.harsh_accel_dv,
        "emergency_brake_dv": base.emergency_brake_dv,
        "aggressive_turn_threshold": base.aggressive_turn_threshold,
        "unstable_motion_jerk_threshold": base.unstable_motion_jerk_threshold,
    }
    truck = type("P", (), {"category": "heavy_truck", "size_class": None, "mass_kg": None})()
    config_for_profile(base, truck)
    config_for_profile(base, None)
    assert base.harsh_brake_dv == before["harsh_brake_dv"]
    assert base.harsh_accel_dv == before["harsh_accel_dv"]
    assert base.emergency_brake_dv == before["emergency_brake_dv"]
    assert base.aggressive_turn_threshold == before["aggressive_turn_threshold"]
    assert base.unstable_motion_jerk_threshold == before["unstable_motion_jerk_threshold"]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def _new_user(session_factory, email: str) -> UserRecord:
    user = UserRecord(id=str(uuid.uuid4()), email=email, password_hash="hashed")
    with session_factory() as db:
        db.add(User(id=user.id, email=user.email, password_hash=user.password_hash))
        db.commit()
    return user


def test_vehicle_profile_404_before_set(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = _new_user(session_factory, "v1@example.com")
    client = _client_with_overrides(session_factory, user)
    try:
        res = client.get("/api/v1/users/me/vehicle")
        assert res.status_code == 404
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_vehicle_profile_upsert_and_get(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = _new_user(session_factory, "v2@example.com")
    client = _client_with_overrides(session_factory, user)
    try:
        res = client.put(
            "/api/v1/users/me/vehicle",
            json={
                "category": "heavy_truck",
                "make_model": "Daf XF",
                "size_class": "large",
                "drive_type": "4wd",
            },
        )
        assert res.status_code == 200, res.text
        payload = res.json()
        assert payload["category"] == "heavy_truck"
        assert payload["make_model"] == "Daf XF"
        assert payload["mass_kg"] == 18000.0 * 1.2  # heavy_truck large
        assert "thresholds" in payload
        assert abs(payload["thresholds"]["harsh_brake_dv"]) < abs(FeatureConfigV2().harsh_brake_dv)

        again = client.get("/api/v1/users/me/vehicle")
        assert again.status_code == 200
        assert again.json()["category"] == "heavy_truck"

        # Upsert replaces, does not duplicate.
        res2 = client.put(
            "/api/v1/users/me/vehicle",
            json={"category": "sedan", "size_class": "compact"},
        )
        assert res2.status_code == 200
        assert res2.json()["category"] == "sedan"
        assert res2.json()["mass_kg"] == 1400.0 * 0.85
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_vehicle_profile_rejects_unknown_category(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = _new_user(session_factory, "v3@example.com")
    client = _client_with_overrides(session_factory, user)
    try:
        res = client.put("/api/v1/users/me/vehicle", json={"category": "spaceship"})
        assert res.status_code == 422
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_vehicle_options_endpoint(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = _new_user(session_factory, "v4@example.com")
    client = _client_with_overrides(session_factory, user)
    try:
        res = client.get("/api/v1/users/me/vehicle/options")
        assert res.status_code == 200
        keys = {item["key"] for item in res.json()}
        assert {"sedan", "suv", "pickup", "van", "bus", "heavy_truck", "tractor_trailer"} <= keys
    finally:
        client.close()
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Trip recording + vehicle-aware finalize
# ---------------------------------------------------------------------------

def test_start_trip_and_finalize_use_vehicle_profile(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = _new_user(session_factory, "v5@example.com")
    client = _client_with_overrides(session_factory, user)
    try:
        vehicle = client.put(
            "/api/v1/users/me/vehicle",
            json={"category": "heavy_truck"},
        ).json()

        started = client.post(
            "/api/v1/trips/start",
            json={"vehicle_profile_id": vehicle["id"]},
        )
        assert started.status_code == 200, started.text
        trip_id = started.json()["id"]

        # Upload a deceleration burst (>= min_samples_for_scoring=10 rows).
        base_ts = datetime(2026, 8, 13, 8, 0, 0, tzinfo=timezone.utc)
        rows = []
        speed = 22.0
        for i in range(12):
            ts = (base_ts + timedelta(seconds=i)).isoformat().replace("+00:00", "Z")
            rows.append(
                {
                    "timestamp": ts,
                    "speed": speed,
                    "lat": 9.0 + i * 0.0001,
                    "lon": 38.7 + i * 0.0001,
                    "accuracy_m": 6.0,
                    "ax": 0.2,
                    "ay": 0.3,
                    "az": 9.8,
                    "gx": 0.01,
                    "gy": 0.01,
                    "gz": 0.01,
                }
            )
            speed -= 2.5  # ~2.5 m/s per second = -2.5 m/s^2
        upload = client.post(f"/api/v1/trips/{trip_id}/samples", json={"samples": rows})
        assert upload.status_code == 200, upload.text

        finalized = client.post(f"/api/v1/trips/{trip_id}/finalize")
        assert finalized.status_code == 200, finalized.text
        payload = finalized.json()
        assert payload["breakdown"]["vehicle_category"] == "heavy_truck"
        assert payload["score"] is not None
        assert 0 <= payload["score"] <= 100

        detail = client.get(f"/api/v1/trips/{trip_id}")
        assert detail.status_code == 200

        # vehicle_profile_id actually persisted on the trip row.
        from app.db.models.trip import Trip

        with session_factory() as db:
            persisted = db.get(Trip, trip_id)
            assert persisted is not None
            assert persisted.vehicle_profile_id == vehicle["id"]
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_finalize_without_vehicle_profile_keeps_default(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = _new_user(session_factory, "v6@example.com")
    client = _client_with_overrides(session_factory, user)
    try:
        started = client.post("/api/v1/trips/start")
        trip_id = started.json()["id"]
        base_ts = datetime(2026, 8, 13, 9, 0, 0, tzinfo=timezone.utc)
        rows = []
        for i in range(12):
            ts = (base_ts + timedelta(seconds=i)).isoformat().replace("+00:00", "Z")
            rows.append(
                {
                    "timestamp": ts,
                    "speed": 14.0,
                    "lat": 9.0,
                    "lon": 38.7,
                    "accuracy_m": 6.0,
                    "ax": 0.1,
                    "ay": 0.1,
                    "az": 9.8,
                    "gx": 0.01,
                    "gy": 0.01,
                    "gz": 0.01,
                }
            )
        client.post(f"/api/v1/trips/{trip_id}/samples", json={"samples": rows})
        finalized = client.post(f"/api/v1/trips/{trip_id}/finalize")
        assert finalized.status_code == 200
        assert finalized.json()["breakdown"]["vehicle_category"] is None
    finally:
        client.close()
        app.dependency_overrides.clear()

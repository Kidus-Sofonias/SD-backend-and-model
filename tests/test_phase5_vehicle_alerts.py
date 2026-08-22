"""Phase 5 (hackathon) — vehicle-aware live alerts.

Covers:
- LiveAlertDetector per-trip vehicle-tuned config (same marginal brake is an
  alert in a truck but not a sedan)
- Trip start binds the driver's vehicle config to the live detector (and
  falls back to the saved profile when no id is passed)
- start rejects another user's vehicle profile
- Provisional live score honours a provided (tuned) config
- vehicle_tuned_cfg helper (single + preloaded profile map)
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
from app.db.models.trip import Trip
from app.db.models.user import User
from app.db.models.vehicle_profile import VehicleProfile
from app.db.session import get_db
from app.main import app
from app.ml.config import FeatureConfigV2
from app.ml.vehicle_profiles import config_for_profile
from app.realtime.live_detector import LiveAlertDetector, live_alert_detector
from app.repositories.user_repository import UserRecord
from app.services.live_monitor_service import _provisional_live_score, vehicle_tuned_cfg


def _make_session_factory(tmp_path: Path):
    db_path = tmp_path / "phase5-vehicle-alerts.db"
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


def _new_user(session_factory, email: str) -> UserRecord:
    user = UserRecord(id=str(uuid.uuid4()), email=email, password_hash="hashed")
    with session_factory() as db:
        db.add(User(id=user.id, email=user.email, password_hash=user.password_hash))
        db.commit()
    return user


def _marginal_brake_rows() -> list[dict]:
    """A steady -2.0 m/s^2 deceleration (20 -> 10 m/s over 6 s @ 1 Hz).

    Below the sedan's -3.2 m/s^2 hard-brake threshold but above a heavy
    truck's tuned ~-1.7 m/s^2 threshold, so it cleanly discriminates
    vehicle-aware detection.
    """
    base_ts = datetime(2026, 8, 13, 10, 0, 0, tzinfo=timezone.utc)
    rows: list[dict] = []
    for i in range(6):
        ts = (base_ts + timedelta(seconds=i)).isoformat().replace("+00:00", "Z")
        rows.append(
            {
                "timestamp": ts,
                "speed": 20.0 - i * 2.0,  # m/s
                "lat": 9.0 + i * 0.0001,
                "lon": 38.7 + i * 0.0001,
                "ax": 0.1,
                "ay": 0.1,
                "az": 9.8,
                "gx": 0.01,
                "gy": 0.01,
                "gz": 0.01,
            }
        )
    return rows


def _truck_profile():
    return type("P", (), {"category": "heavy_truck", "size_class": None, "mass_kg": None})()


# ---------------------------------------------------------------------------
# Detector per-trip config
# ---------------------------------------------------------------------------

def test_live_detector_per_trip_config_changes_detection(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = _new_user(session_factory, "veh-live@example.com")

    detector = LiveAlertDetector()
    sedan_cfg = FeatureConfigV2()
    truck_cfg = config_for_profile(FeatureConfigV2(), _truck_profile())
    # Sanity: the tuning really lowers the truck's hard-brake floor.
    assert abs(truck_cfg.harsh_brake_dv) < abs(sedan_cfg.harsh_brake_dv)

    rows = _marginal_brake_rows()
    with session_factory() as db:
        # Default (no profile bound) behaves like a sedan: marginal brake is NOT harsh.
        alerts = detector.process_upload(db=db, user_id=user.id, trip_id="trip-sedan", rows=rows)
        assert alerts == []

        # A heavy truck's tuned floor makes the SAME samples a hard brake.
        detector.set_trip_config("trip-truck", truck_cfg)
        alerts = detector.process_upload(db=db, user_id=user.id, trip_id="trip-truck", rows=rows)
        assert alerts, "marginal brake should alert for a truck"
        assert {a["event"]["event_type"] for a in alerts} == {"hard_brake"}

    # clear_trip releases the bound config (falls back to the default again).
    detector.clear_trip("trip-truck")
    with session_factory() as db:
        again = detector.process_upload(db=db, user_id=user.id, trip_id="trip-truck", rows=rows)
        assert again == []


def test_live_detector_lazy_binds_config_after_restart(tmp_path: Path) -> None:
    """A fresh detector (server restart) restores the vehicle-tuned config
    from the persisted trip row on the next upload instead of falling back to
    the universal default."""
    session_factory = _make_session_factory(tmp_path)
    user = _new_user(session_factory, "veh-restart@example.com")
    trip_id = "trip-restart-1"

    with session_factory() as db:
        profile = VehicleProfile(user_id=user.id, category="heavy_truck")
        db.add(profile)
        db.flush()
        db.add(
            Trip(
                id=trip_id,
                user_id=user.id,
                status="active",
                vehicle_profile_id=profile.id,
                started_at=datetime.now(timezone.utc),
            )
        )
        db.commit()

    detector = LiveAlertDetector()  # empty in-memory state, like after a restart
    with session_factory() as db:
        alerts = detector.process_upload(db=db, user_id=user.id, trip_id=trip_id, rows=_marginal_brake_rows())
    assert alerts, "marginal brake should alert for the truck after lazy re-bind"
    assert {a["event"]["event_type"] for a in alerts} == {"hard_brake"}
    bound = detector._trip_cfgs.get(trip_id)
    assert bound is not None
    assert abs(bound.harsh_brake_dv) < abs(FeatureConfigV2().harsh_brake_dv)


# ---------------------------------------------------------------------------
# Trip start wiring (route level, shared singleton)
# ---------------------------------------------------------------------------

def test_start_trip_binds_vehicle_config_to_live_detector(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = _new_user(session_factory, "veh-start@example.com")
    client = _client_with_overrides(session_factory, user)
    trip_id: str | None = None
    try:
        vehicle = client.put(
            "/api/v1/users/me/vehicle",
            json={"category": "heavy_truck"},
        ).json()

        started = client.post("/api/v1/trips/start", json={"vehicle_profile_id": vehicle["id"]})
        assert started.status_code == 200, started.text
        trip_id = started.json()["id"]

        # The detector now holds a truck-tuned config for this trip.
        bound = live_alert_detector._trip_cfgs.get(trip_id)
        assert bound is not None
        assert abs(bound.harsh_brake_dv) < abs(FeatureConfigV2().harsh_brake_dv)

        # Marginal brake upload -> live alert path flags a hard brake for the truck.
        upload = client.post(
            f"/api/v1/trips/{trip_id}/samples",
            json={"samples": _marginal_brake_rows()},
        )
        assert upload.status_code == 200, upload.text
        counts = live_alert_detector.event_counts(trip_id)
        assert counts.get("hard_brake", 0) >= 1

        # Ending the trip releases the in-memory config.
        ended = client.post(f"/api/v1/trips/{trip_id}/end")
        assert ended.status_code == 200
        assert trip_id not in live_alert_detector._trip_cfgs
    finally:
        client.close()
        if trip_id:
            live_alert_detector.clear_trip(trip_id)
        app.dependency_overrides.clear()


def test_start_trip_falls_back_to_saved_profile(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = _new_user(session_factory, "veh-fallback@example.com")
    client = _client_with_overrides(session_factory, user)
    trip_id: str | None = None
    try:
        client.put("/api/v1/users/me/vehicle", json={"category": "sedan", "size_class": "compact"})

        # No vehicle_profile_id passed: the saved profile should be used.
        started = client.post("/api/v1/trips/start")
        assert started.status_code == 200, started.text
        trip_id = started.json()["id"]

        with session_factory() as db:
            persisted = db.get(Trip, trip_id)
            assert persisted is not None
            assert persisted.vehicle_profile_id is not None

        bound = live_alert_detector._trip_cfgs.get(trip_id)
        assert bound is not None
        expected = config_for_profile(
            FeatureConfigV2(),
            type("P", (), {"category": "sedan", "size_class": "compact", "mass_kg": None})(),
        )
        assert bound.harsh_brake_dv == expected.harsh_brake_dv
    finally:
        client.close()
        if trip_id:
            live_alert_detector.clear_trip(trip_id)
        app.dependency_overrides.clear()


def test_start_trip_rejects_foreign_vehicle_profile(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    owner = _new_user(session_factory, "veh-owner@example.com")
    other = _new_user(session_factory, "veh-other@example.com")

    owner_client = _client_with_overrides(session_factory, owner)
    profile_id: str | None = None
    try:
        profile_id = owner_client.put(
            "/api/v1/users/me/vehicle",
            json={"category": "suv"},
        ).json()["id"]
    finally:
        owner_client.close()
        app.dependency_overrides.clear()

    client = _client_with_overrides(session_factory, other)
    try:
        started = client.post("/api/v1/trips/start", json={"vehicle_profile_id": profile_id})
        assert started.status_code == 404
    finally:
        client.close()
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Live score + cfg helper
# ---------------------------------------------------------------------------

def test_provisional_live_score_uses_provided_cfg() -> None:
    counts = {"hard_brake": 1}
    default = _provisional_live_score(counts, elapsed_s=3600.0)
    custom = _provisional_live_score(counts, elapsed_s=3600.0, cfg=FeatureConfigV2(w_brake=20.0))
    assert default["penalties"]["hard_brake"] == 6.0
    assert custom["penalties"]["hard_brake"] == 20.0
    assert custom["score"] < default["score"]


def test_vehicle_tuned_cfg_helper(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = _new_user(session_factory, "veh-cfg@example.com")

    with session_factory() as db:
        profile = VehicleProfile(user_id=user.id, category="heavy_truck", size_class=None, mass_kg=None)
        db.add(profile)
        db.commit()
        profile_id = profile.id

        tuned_trip = Trip(
            id=str(uuid.uuid4()),
            user_id=user.id,
            status="active",
            vehicle_profile_id=profile_id,
            started_at=datetime.now(timezone.utc),
        )
        plain_trip = Trip(
            id=str(uuid.uuid4()),
            user_id=user.id,
            status="active",
            vehicle_profile_id=None,
            started_at=datetime.now(timezone.utc),
        )
        db.add_all([tuned_trip, plain_trip])
        db.commit()

        tuned = vehicle_tuned_cfg(db, tuned_trip)
        assert abs(tuned.harsh_brake_dv) < abs(FeatureConfigV2().harsh_brake_dv)

        # Preloaded map path avoids a second lookup.
        tuned_from_map = vehicle_tuned_cfg(db, tuned_trip, profiles={profile_id: profile})
        assert tuned_from_map.harsh_brake_dv == tuned.harsh_brake_dv

        # No profile -> universal default.
        plain = vehicle_tuned_cfg(db, plain_trip)
        assert plain.harsh_brake_dv == FeatureConfigV2().harsh_brake_dv

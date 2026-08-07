# Phase 6 prep — live trip telemetry endpoint.
# - Returns latest speed / acceleration / location / sample count / event counts
# - Owner-scoped (404 for other users' trips and unknown trips)
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
from app.db.session import get_db
from app.main import app
from app.realtime.live_detector import LiveAlertDetector, live_alert_detector
from app.repositories.user_repository import UserRecord
from app.services.live_monitor_service import LiveMonitorService


def _make_session_factory(tmp_path: Path):
    db_path = tmp_path / "phase6-test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


def _samples(n: int = 6, *, start_speed: float = 20.0, speed_step: float = 0.0) -> list[dict]:
    base_ts = datetime(2026, 4, 2, 9, 0, 0, tzinfo=timezone.utc)
    samples: list[dict] = []
    for i in range(n):
        ts = (base_ts + timedelta(seconds=i)).isoformat().replace("+00:00", "Z")
        samples.append(
            {
                "timestamp": ts,
                "speed": start_speed + i * speed_step,  # m/s
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
    return samples


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


def test_live_telemetry_returns_speed_accel_location_and_counts(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = UserRecord(id=str(uuid.uuid4()), email="telemetry@example.com", password_hash="hashed")

    with session_factory() as db:
        db.add(User(id=user.id, email=user.email, password_hash=user.password_hash))
        db.commit()

    client = _client_with_overrides(session_factory, user)
    try:
        trip_id = client.post("/api/v1/trips/start").json()["id"]

        # Upload a braking burst so the live detector flags an event.
        rows = _samples(6, start_speed=25.0, speed_step=-4.0)
        upload = client.post(f"/api/v1/trips/{trip_id}/samples", json={"samples": rows})
        assert upload.status_code == 200

        res = client.get(f"/api/v1/trips/{trip_id}/telemetry")
        assert res.status_code == 200
        payload = res.json()

        assert payload["trip_id"] == trip_id
        assert payload["status"] == "active"
        assert payload["elapsed_s"] >= 0
        assert payload["samples_uploaded"] == len(rows)
        assert payload["event_total"] >= 1
        assert any(key in payload["event_counts"] for key in ("hard_brake", "emergency_brake"))

        # Provisional live score derived from detected event counts.
        live_score = payload["live_score"]
        assert live_score["provisional"] is True
        assert 0 <= live_score["score"] <= 100
        assert live_score["risk_level"] in ("low", "medium", "high")
        assert live_score["scoring_version"] == "v2-live"

        latest = payload["latest"]
        assert latest["speed_mps"] == rows[-1]["speed"]
        assert latest["lat"] == rows[-1]["lat"]
        assert latest["lon"] == rows[-1]["lon"]
        assert latest["accuracy_m"] == 6.0
        # IMU magnitude sqrt(0.2^2 + 0.3^2 + 9.8^2)
        assert latest["accel_mag_mps2"] is not None
        assert latest["accel_mag_mps2"] > 9.8
        # Braking burst -> negative longitudinal acceleration.
        assert latest["longitudinal_accel_mps2"] < 0

        assert len(payload["recent_alerts"]) >= 1
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_live_telemetry_owner_scoped_and_unknown_trip_404(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    owner = UserRecord(id=str(uuid.uuid4()), email="owner@example.com", password_hash="hashed")
    other = UserRecord(id=str(uuid.uuid4()), email="other@example.com", password_hash="hashed")

    with session_factory() as db:
        db.add(User(id=owner.id, email=owner.email, password_hash=owner.password_hash))
        db.add(User(id=other.id, email=other.email, password_hash=other.password_hash))
        trip = Trip(id="trip-telemetry-1", user_id=owner.id, status="active", started_at=datetime.now(timezone.utc))
        db.add(trip)
        db.commit()

    other_client = _client_with_overrides(session_factory, other)
    try:
        res = other_client.get("/api/v1/trips/trip-telemetry-1/telemetry")
        assert res.status_code == 404

        missing = other_client.get("/api/v1/trips/does-not-exist/telemetry")
        assert missing.status_code == 404
    finally:
        other_client.close()
        app.dependency_overrides.clear()


def test_live_monitor_service_empty_trip_returns_empty_latest(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = UserRecord(id=str(uuid.uuid4()), email="empty@example.com", password_hash="hashed")

    with session_factory() as db:
        db.add(User(id=user.id, email=user.email, password_hash=user.password_hash))
        db.commit()

    client = _client_with_overrides(session_factory, user)
    try:
        trip_id = client.post("/api/v1/trips/start").json()["id"]

        res = client.get(f"/api/v1/trips/{trip_id}/telemetry")
        assert res.status_code == 200
        payload = res.json()

        assert payload["samples_uploaded"] == 0
        assert payload["latest"]["speed_mps"] is None
        assert payload["latest"]["lat"] is None
        assert payload["latest"]["accel_mag_mps2"] is None
        assert payload["event_total"] == 0
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_live_alert_detector_event_counts_aggregates() -> None:
    detector = LiveAlertDetector()
    with detector._lock:
        detector._alerted.setdefault("trip-x", set()).add(("hard_brake", "ts-1"))
        detector._alerted.setdefault("trip-x", set()).add(("hard_brake", "ts-2"))
        detector._alerted.setdefault("trip-x", set()).add(("overspeed", "ts-3"))

    assert detector.event_counts("trip-x") == {"hard_brake": 2, "overspeed": 1}
    assert detector.event_counts("other-trip") == {}

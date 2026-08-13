# Phase 7 — admin live monitoring.
# - GET /admin/trips/live returns active trips with telemetry + connection status
# - Non-admins are rejected (403)
# - Driver alerts are fanned out to the fleet-wide hub channel for admin sessions
from __future__ import annotations

import asyncio
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
from app.realtime.hub import FLEET_GLOBAL_KEY, alert_hub
from app.repositories.user_repository import UserRecord


def _make_session_factory(tmp_path: Path):
    db_path = tmp_path / "phase7-admin.db"
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


def _samples(n: int = 4) -> list[dict]:
    base_ts = datetime(2026, 8, 8, 10, 0, 0, tzinfo=timezone.utc)
    return [
        {
            "timestamp": (base_ts + timedelta(seconds=i)).isoformat().replace("+00:00", "Z"),
            "speed": 22.0,
            "lat": 9.02 + i * 0.0001,
            "lon": 38.74 + i * 0.0001,
            "accuracy_m": 6.0,
            "ax": 0.2,
            "ay": 0.3,
            "az": 9.8,
            "gx": 0.01,
            "gy": 0.01,
            "gz": 0.01,
        }
        for i in range(n)
    ]


def test_admin_live_trips_returns_fleet_snapshot(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    driver = UserRecord(id=str(uuid.uuid4()), email="fleet-driver@example.com", password_hash="hashed")
    admin = UserRecord(id=str(uuid.uuid4()), email="fleet-admin@example.com", password_hash="hashed", role="admin")

    with session_factory() as db:
        db.add(User(id=driver.id, email=driver.email, password_hash=driver.password_hash, role=driver.role))
        db.add(User(id=admin.id, email=admin.email, password_hash=admin.password_hash, role=admin.role))
        db.commit()

    driver_client = _client_with_overrides(session_factory, driver)
    try:
        trip_id = driver_client.post("/api/v1/trips/start").json()["id"]
        upload = driver_client.post(f"/api/v1/trips/{trip_id}/samples", json={"samples": _samples()})
        assert upload.status_code == 200
    finally:
        driver_client.close()
        app.dependency_overrides.clear()

    admin_client = _client_with_overrides(session_factory, admin)
    try:
        res = admin_client.get("/api/v1/admin/trips/live")
        assert res.status_code == 200
        payload = res.json()
        assert len(payload) == 1
        trip = payload[0]

        assert trip["trip_id"] == trip_id
        assert trip["driver_email"] == driver.email
        assert trip["elapsed_s"] >= 0
        assert trip["samples_uploaded"] == 4
        assert trip["latest"]["speed_mps"] == 22.0
        assert trip["latest"]["lat"] is not None
        assert trip["latest"]["accel_mag_mps2"] is not None
        assert trip["live_score"]["provisional"] is True
        assert 0 <= trip["live_score"]["score"] <= 100
        assert trip["connection_status"] in ("live", "stale", "disconnected")
    finally:
        admin_client.close()
        app.dependency_overrides.clear()


def test_admin_trip_telemetry_returns_any_drivers_live_trip(tmp_path: Path) -> None:
    """Phase 7 live-update: admins can poll telemetry for ANY trip, not just
    their own, so opening a live fleet trip in the detail screen stays live."""
    session_factory = _make_session_factory(tmp_path)
    driver = UserRecord(id=str(uuid.uuid4()), email="telemetry-driver@example.com", password_hash="hashed")
    admin = UserRecord(id=str(uuid.uuid4()), email="telemetry-admin@example.com", password_hash="hashed", role="admin")

    with session_factory() as db:
        db.add(User(id=driver.id, email=driver.email, password_hash=driver.password_hash, role=driver.role))
        db.add(User(id=admin.id, email=admin.email, password_hash=admin.password_hash, role=admin.role))
        db.commit()

    driver_client = _client_with_overrides(session_factory, driver)
    try:
        trip_id = driver_client.post("/api/v1/trips/start").json()["id"]
        upload = driver_client.post(f"/api/v1/trips/{trip_id}/samples", json={"samples": _samples()})
        assert upload.status_code == 200
    finally:
        driver_client.close()
        app.dependency_overrides.clear()

    admin_client = _client_with_overrides(session_factory, admin)
    try:
        res = admin_client.get(f"/api/v1/admin/trips/{trip_id}/telemetry")
        assert res.status_code == 200
        payload = res.json()
        assert payload["trip_id"] == trip_id
        assert payload["status"] == "active"
        assert payload["samples_uploaded"] == 4
        assert payload["latest"]["speed_mps"] == 22.0
        assert payload["live_score"]["provisional"] is True
        assert payload["event_total"] >= 0
    finally:
        admin_client.close()
        app.dependency_overrides.clear()


def test_admin_trip_telemetry_rejects_non_admin_and_missing_trip(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    driver = UserRecord(id=str(uuid.uuid4()), email="telemetry-driver2@example.com", password_hash="hashed")
    admin = UserRecord(id=str(uuid.uuid4()), email="telemetry-admin2@example.com", password_hash="hashed", role="admin")

    with session_factory() as db:
        db.add(User(id=driver.id, email=driver.email, password_hash=driver.password_hash, role=driver.role))
        db.add(User(id=admin.id, email=admin.email, password_hash=admin.password_hash, role=admin.role))
        db.commit()

    # Non-admin cannot read another driver's live telemetry.
    driver_client = _client_with_overrides(session_factory, driver)
    try:
        trip_id = driver_client.post("/api/v1/trips/start").json()["id"]
        res = driver_client.get(f"/api/v1/admin/trips/{trip_id}/telemetry")
        assert res.status_code == 403
    finally:
        driver_client.close()
        app.dependency_overrides.clear()

    # Admin gets 404 for a missing trip.
    admin_client = _client_with_overrides(session_factory, admin)
    try:
        res = admin_client.get(f"/api/v1/admin/trips/{uuid.uuid4()}/telemetry")
        assert res.status_code == 404
    finally:
        admin_client.close()
        app.dependency_overrides.clear()


def test_admin_live_trips_rejects_non_admin(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    driver = UserRecord(id=str(uuid.uuid4()), email="plain-driver@example.com", password_hash="hashed")

    with session_factory() as db:
        db.add(User(id=driver.id, email=driver.email, password_hash=driver.password_hash, role=driver.role))
        db.commit()

    client = _client_with_overrides(session_factory, driver)
    try:
        res = client.get("/api/v1/admin/trips/live")
        assert res.status_code == 403
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_fleet_hub_channel_delivers_to_admin_subscriber() -> None:
    async def scenario():
        admin_queue = alert_hub.subscribe(FLEET_GLOBAL_KEY)

        # Mirrors the detector: the driver alert is ALSO published to the fleet
        # channel so a single admin subscription sees every driver's events.
        published = alert_hub.publish(FLEET_GLOBAL_KEY, {"type": "event_alert", "trip_id": "t1", "sent_at": "now"})
        assert published >= 1

        await asyncio.sleep(0)
        admin_message = admin_queue.get_nowait()
        assert admin_message["trip_id"] == "t1"

        alert_hub.unsubscribe(FLEET_GLOBAL_KEY, admin_queue)

    asyncio.run(scenario())


def test_live_detector_publishes_to_fleet_channel(tmp_path: Path, monkeypatch) -> None:
    from app.realtime.live_detector import LiveAlertDetector

    session_factory = _make_session_factory(tmp_path)
    user = UserRecord(id=str(uuid.uuid4()), email="detector@example.com", password_hash="hashed")
    with session_factory() as db:
        db.add(User(id=user.id, email=user.email, password_hash=user.password_hash))
        db.commit()

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "app.realtime.live_detector.alert_hub.publish",
        lambda user_id, payload: calls.append((user_id, payload)) or 1,
    )

    detector = LiveAlertDetector()
    # A braking burst that triggers a hard-brake event.
    base_ts = datetime(2026, 8, 8, 10, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(8):
        rows.append(
            {
                "ts": (base_ts + timedelta(seconds=i)).isoformat().replace("+00:00", "Z"),
                "speed_mps": 28.0 if i < 3 else (28.0 - 4.5 * (i - 2)),
                "lat": 9.02,
                "lon": 38.74,
                "ax": 0.1,
                "ay": 0.1,
                "az": 9.8,
                "gx": 0.0,
                "gy": 0.0,
                "gz": 0.0,
            }
        )

    with session_factory() as db:
        alerts = detector.process_upload(db=db, user_id=user.id, trip_id="trip-fleet-1", rows=rows)

    assert alerts, "expected at least one detected event"
    targets = [user_id for user_id, _ in calls]
    assert user.id in targets
    assert FLEET_GLOBAL_KEY in targets

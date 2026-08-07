# Phase 8 — accident detection.
# - High-confidence alerts only when an impact is corroborated (speed collapse,
#   repeated impacts, etc.) -> minimizes pothole/phone-drop false positives
# - Per-trip cooldown prevents alert spam for one crash cluster
# - Admin-only recent-accidents endpoint (403 for non-admins)
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
from app.realtime.accident_detector import AccidentDetector, accident_detector
from app.repositories.user_repository import UserRecord


def _row(ts: datetime, *, speed: float, az: float = 9.8, ax: float = 0.0) -> dict:
    return {
        "ts": ts.isoformat().replace("+00:00", "Z"),
        "speed_mps": speed,
        "lat": 9.02,
        "lon": 38.74,
        "ax": ax,
        "ay": 0.0,
        "az": az,
        "gx": 0.0,
        "gy": 0.0,
        "gz": 0.0,
    }


def _impact_and_collapse_burst(start: datetime) -> list[dict]:
    """Cruise at 25 m/s -> strong impact spike -> speed collapses to 10 m/s."""
    rows = [_row(start + timedelta(seconds=i), speed=25.0) for i in range(3)]
    rows.append(_row(start + timedelta(seconds=3), speed=25.0, az=45.0))  # ~4.6 g impact
    rows.append(_row(start + timedelta(seconds=4), speed=10.0))  # -15 m/s in 1 s
    rows.append(_row(start + timedelta(seconds=5), speed=8.0))
    return rows


def _single_spike_burst(start: datetime) -> list[dict]:
    """One moderate spike (~3 g) with no speed change - pothole/speed bump."""
    rows = [_row(start + timedelta(seconds=i), speed=25.0) for i in range(3)]
    rows.append(_row(start + timedelta(seconds=3), speed=25.0, az=30.0))  # candidate, not strong
    rows.append(_row(start + timedelta(seconds=4), speed=25.0))
    return rows


def test_detector_alerts_on_impact_plus_speed_collapse() -> None:
    detector = AccidentDetector()
    start = datetime(2026, 8, 8, 12, 0, 0, tzinfo=timezone.utc)
    alerts = detector.process_upload(
        user_id="user-x",
        trip_id="trip-accident-1",
        rows=_impact_and_collapse_burst(start),
    )
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["type"] == "accident_alert"
    assert alert["trip_id"] == "trip-accident-1"
    assert alert["confidence"] >= 0.7
    assert alert["signals"]["strong_impact"] is True
    assert alert["signals"]["speed_collapse"] is True
    assert alert["max_accel_mps2"] >= 40.0
    assert alert["lat"] is not None


def test_detector_no_false_positive_on_single_spike() -> None:
    detector = AccidentDetector()
    start = datetime(2026, 8, 8, 12, 0, 0, tzinfo=timezone.utc)
    alerts = detector.process_upload(user_id="user-x", trip_id="trip-pothole", rows=_single_spike_burst(start))
    assert alerts == []


def test_detector_respects_trip_cooldown() -> None:
    detector = AccidentDetector()
    start = datetime(2026, 8, 8, 12, 0, 0, tzinfo=timezone.utc)
    first = detector.process_upload(user_id="user-x", trip_id="trip-crash", rows=_impact_and_collapse_burst(start))
    assert len(first) == 1

    # A second cluster a few seconds later must be suppressed by the cooldown.
    second = detector.process_upload(
        user_id="user-x",
        trip_id="trip-crash",
        rows=_impact_and_collapse_burst(start + timedelta(seconds=10)),
    )
    assert second == []


def _make_session_factory(tmp_path: Path):
    db_path = tmp_path / "phase8-accident.db"
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


def test_admin_recent_accidents_endpoint(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    admin = UserRecord(id=str(uuid.uuid4()), email="accident-admin@example.com", password_hash="hashed", role="admin")

    with session_factory() as db:
        db.add(User(id=admin.id, email=admin.email, password_hash=admin.password_hash, role=admin.role))
        db.commit()

    with accident_detector._lock:
        accident_detector._recent.append(
            {"type": "accident_alert", "trip_id": "trip-acc-1", "confidence": 0.9, "sent_at": "now"}
        )

    client = _client_with_overrides(session_factory, admin)
    try:
        res = client.get("/api/v1/admin/accidents/recent")
        assert res.status_code == 200
        payload = res.json()
        assert payload["accidents"]
        assert payload["accidents"][0]["trip_id"] == "trip-acc-1"
    finally:
        client.close()
        app.dependency_overrides.clear()
        with accident_detector._lock:
            accident_detector._recent.clear()


def test_admin_recent_accidents_rejects_non_admin(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    driver = UserRecord(id=str(uuid.uuid4()), email="plain-driver@example.com", password_hash="hashed")

    with session_factory() as db:
        db.add(User(id=driver.id, email=driver.email, password_hash=driver.password_hash, role=driver.role))
        db.commit()

    client = _client_with_overrides(session_factory, driver)
    try:
        res = client.get("/api/v1/admin/accidents/recent")
        assert res.status_code == 403
    finally:
        client.close()
        app.dependency_overrides.clear()

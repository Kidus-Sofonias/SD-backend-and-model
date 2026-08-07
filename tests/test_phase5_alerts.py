# Phase 5 — Real-time driver alerts.
# - AlertHub pub/sub (thread-safe publish from sync code)
# - LiveAlertDetector incremental detection + dedupe + DB seeding
# - WebSocket stream: auth rejection, connected handshake, event delivery
from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.deps import get_current_user
from app.core.jwt import create_access_token
from app.db.base import Base
from app.db.models.trip import Trip
from app.db.models.user import User
from app.db.session import get_db
from app.main import app
from app.realtime.hub import AlertHub
from app.realtime.live_detector import LiveAlertDetector
from app.repositories.user_repository import UserRecord


def _make_session_factory(tmp_path: Path):
    db_path = tmp_path / "phase5-test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


def _samples(n: int = 12, *, start_speed: float = 20.0, speed_step: float = 0.0) -> list[dict]:
    """Build n samples at 1 Hz starting from start_speed (m/s). Negative
    speed_step produces deceleration (braking)."""
    base_ts = datetime(2026, 4, 1, 8, 0, 0, tzinfo=timezone.utc)
    samples: list[dict] = []
    for i in range(n):
        ts = (base_ts + timedelta(seconds=i)).isoformat().replace("+00:00", "Z")
        samples.append(
            {
                "timestamp": ts,
                "speed": start_speed + i * speed_step,  # m/s
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


# ---------------------------------------------------------------------------
# AlertHub
# ---------------------------------------------------------------------------

def test_alert_hub_publishes_to_subscribed_user() -> None:
    async def scenario() -> None:
        hub = AlertHub()
        queue_a = hub.subscribe("user-a")
        hub.subscribe("user-b")

        # publish() is designed to be callable from a worker thread; from inside
        # the loop it schedules via call_soon_threadsafe, so the message lands
        # on the next loop iteration.
        sent = hub.publish("user-a", {"type": "event_alert", "event_type": "hard_brake"})
        assert sent == 1

        await asyncio.sleep(0)
        message = queue_a.get_nowait()
        assert message["event_type"] == "hard_brake"

        # Unsubscribing stops delivery
        hub.unsubscribe("user-a", queue_a)
        assert hub.publish("user-a", {"type": "x"}) == 0

    asyncio.run(scenario())


def test_alert_hub_publish_no_subscriber_is_noop() -> None:
    async def scenario() -> None:
        hub = AlertHub()
        assert hub.publish("nobody", {"type": "x"}) == 0

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# LiveAlertDetector
# ---------------------------------------------------------------------------

def test_live_detector_detects_brake_and_dedupes(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = UserRecord(id=str(uuid.uuid4()), email="live@example.com", password_hash="hashed")

    with session_factory() as db:
        db.add(User(id=user.id, email=user.email, password_hash=user.password_hash))
        db.add(Trip(id="trip-live-1", user_id=user.id, status="active", started_at=datetime.now(timezone.utc)))
        db.commit()

    detector = LiveAlertDetector()
    # 25 -> 5 m/s over 6 samples at 1 Hz = a steady -4 m/s^2 deceleration,
    # which clears the -3.2 m/s^2 hard-brake threshold.
    rows = _samples(6, start_speed=25.0, speed_step=-4.0)
    with session_factory() as db:
        alerts = detector.process_upload(db=db, user_id=user.id, trip_id="trip-live-1", rows=rows)
    assert alerts, "expected at least one brake alert"
    types = {a["event"]["event_type"] for a in alerts}
    assert "hard_brake" in types or "emergency_brake" in types

    # Re-processing the same window must not re-alert (dedupe by timestamp).
    with session_factory() as db:
        again = detector.process_upload(db=db, user_id=user.id, trip_id="trip-live-1", rows=rows)
    assert again == []

    # Recent replay buffer is populated.
    assert detector.recent_alerts("trip-live-1")


def test_live_detector_seeds_from_persisted_events(tmp_path: Path) -> None:
    """Events already persisted (e.g. after a server restart) are not re-alerted."""
    from app.db.models.driving_event import DrivingEvent

    session_factory = _make_session_factory(tmp_path)
    user = UserRecord(id=str(uuid.uuid4()), email="seed@example.com", password_hash="hashed")
    trip_id = "trip-seed-1"
    occurred = datetime.now(timezone.utc)

    with session_factory() as db:
        db.add(User(id=user.id, email=user.email, password_hash=user.password_hash))
        db.add(Trip(id=trip_id, user_id=user.id, status="active", started_at=datetime.now(timezone.utc)))
        db.add(DrivingEvent(user_id=user.id, trip_id=trip_id, event_type="hard_brake", value=4.0, occurred_at=occurred))
        db.commit()

    detector = LiveAlertDetector()
    # Build samples whose brake peaks at roughly the same occurred_at as the
    # persisted event is not trivial to align; instead verify the DB-seed marks
    # the exact key and the same event key is suppressed.
    with session_factory() as db:
        detector._seed_from_db(db, user_id=user.id, trip_id=trip_id)
    key = detector._event_key("hard_brake", occurred)
    assert key in detector._alerted[trip_id]

    # New event keys not in the DB are still alertable.
    assert detector._event_key("hard_brake", datetime.now(timezone.utc)) not in detector._alerted[trip_id]


# ---------------------------------------------------------------------------
# WebSocket stream (auth + delivery)
# ---------------------------------------------------------------------------

def test_ws_alerts_rejects_invalid_token(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = UserRecord(id=str(uuid.uuid4()), email="ws-reject@example.com", password_hash="hashed")

    with session_factory() as db:
        db.add(User(id=user.id, email=user.email, password_hash=user.password_hash))
        db.commit()

    client = _client_with_overrides(session_factory, user)
    try:
        with client.websocket_connect("/api/v1/ws/alerts?token=not-a-jwt") as ws:
            ws.receive_json()  # pragma: no cover - should not reach here
            raise AssertionError("connection should have been rejected")
    except Exception as exc:
        # starlette raises WebSocketDisconnect(code=4401) whose str() is empty;
        # assert on the code when present, else on the exception type.
        code = getattr(exc, "code", None)
        assert code == 4401 or "WebSocketDisconnect" in type(exc).__name__


def test_ws_alerts_delivers_events_for_active_trip(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = UserRecord(id=str(uuid.uuid4()), email="ws-live@example.com", password_hash="hashed")
    token = create_access_token(user.id)

    with session_factory() as db:
        db.add(User(id=user.id, email=user.email, password_hash=user.password_hash))
        db.commit()

    client = _client_with_overrides(session_factory, user)
    try:
        trip_id = client.post("/api/v1/trips/start").json()["id"]

        with client.websocket_connect(f"/api/v1/ws/alerts?token={token}") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "connected"

            # Upload a braking burst -> the detector should push an alert.
            rows = _samples(6, start_speed=25.0, speed_step=-4.0)
            upload = client.post(f"/api/v1/trips/{trip_id}/samples", json={"samples": rows})
            assert upload.status_code == 200

            alert = ws.receive_json()
            assert alert["type"] == "event_alert"
            assert alert["trip_id"] == trip_id
            assert alert["event"]["event_type"] in {"hard_brake", "emergency_brake"}

        # Recent alerts endpoint is owner-scoped.
        recent = client.get(f"/api/v1/trips/{trip_id}/alerts/recent")
        assert recent.status_code == 200
        assert recent.json()["trip_id"] == trip_id
        assert len(recent.json()["alerts"]) >= 1
    finally:
        client.close()
        app.dependency_overrides.clear()

# Phase 6 prep — offline-resilient upload guard (remote-area queue enabler).
# - Uploads to a completed-but-UNFINALIZED trip succeed so an offline driver can
#   flush queued samples right before finalize.
# - Uploads to a sealed (scored/processed) trip are still rejected (H-7 intent).
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
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
from app.repositories.user_repository import UserRecord


def _make_session_factory(tmp_path: Path):
    db_path = tmp_path / "phase6-queue.db"
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


def _sample(ts: str) -> dict:
    return {
        "timestamp": ts,
        "speed": 20.0,
        "lat": 9.0,
        "lon": 38.7,
        "accuracy_m": 6.0,
        "ax": 0.1,
        "ay": 0.1,
        "az": 9.8,
        "gx": 0.0,
        "gy": 0.0,
        "gz": 0.0,
    }


def test_upload_allowed_for_completed_but_unfinalized_trip(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = UserRecord(id=str(uuid.uuid4()), email="flush@example.com", password_hash="hashed")

    with session_factory() as db:
        db.add(User(id=user.id, email=user.email, password_hash=user.password_hash))
        trip = Trip(
            id="trip-flush-1",
            user_id=user.id,
            status="completed",
            started_at=datetime(2026, 8, 8, 9, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 8, 8, 9, 30, tzinfo=timezone.utc),
            score=None,
            processed_at=None,
        )
        db.add(trip)
        db.commit()

    client = _client_with_overrides(session_factory, user)
    try:
        res = client.post("/api/v1/trips/trip-flush-1/samples", json={"samples": [_sample("2026-08-08T09:10:00Z")]})
        assert res.status_code == 200, res.text
        assert res.json()["inserted"] == 1
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_upload_rejected_for_sealed_scored_trip(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = UserRecord(id=str(uuid.uuid4()), email="sealed@example.com", password_hash="hashed")

    with session_factory() as db:
        db.add(User(id=user.id, email=user.email, password_hash=user.password_hash))
        trip = Trip(
            id="trip-sealed-1",
            user_id=user.id,
            status="completed",
            started_at=datetime(2026, 8, 8, 9, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 8, 8, 9, 30, tzinfo=timezone.utc),
            score=85,
            processed_at=datetime(2026, 8, 8, 9, 31, tzinfo=timezone.utc),
        )
        db.add(trip)
        db.commit()

    client = _client_with_overrides(session_factory, user)
    try:
        res = client.post("/api/v1/trips/trip-sealed-1/samples", json={"samples": [_sample("2026-08-08T09:20:00Z")]})
        assert res.status_code == 409, res.text
    finally:
        client.close()
        app.dependency_overrides.clear()

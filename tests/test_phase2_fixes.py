"""Phase 2 regression tests for the critical/high fixes:

- CRIT-1: single-sample events are counted (no more zero event counts)
- CRIT-2: missing GPS speed no longer crashes finalization
- CRIT-4: speed_variation no longer double-counts brake/accel events
- H-4:    event creation enforces trip ownership
- H-7:    sensor uploads rejected for non-active trips; rate limiter works
- H-5:    admin trip listing is paginated
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.errors import AppError, NotFoundError
from app.core.rate_limit import SlidingWindowRateLimiter
from app.db.base import Base
from app.db.models.sensor_sample import SensorSample
from app.db.models.trip import Trip
from app.db.models.user import User
from app.ml.config import FeatureConfigV2
from app.ml.pipeline import run_trip_pipeline
from app.repositories.driving_event_repository import DrivingEventRepository
from app.repositories.sensor_sample_repository import SensorSampleRepository
from app.repositories.trip_repository import SqlTripRepository
from app.repositories.user_repository import SqlUserRepository, UserRecord
from app.services.admin_service import AdminService
from app.services.driving_event_service import DrivingEventService
from app.services.sensor_sample_service import SensorSampleService
from app.services.trip_processing_service import TripProcessingService


def _make_session(tmp_path: Path) -> Session:
    db_path = tmp_path / "phase2_test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)
    return session_factory()


def _add_user(db: Session, user_id: str | None = None, email: str | None = None) -> str:
    user_id = user_id or str(uuid.uuid4())
    db.add(User(id=user_id, email=email or f"{user_id}@example.com", password_hash="hashed"))
    return user_id


def _add_trip(db: Session, user_id: str, trip_id: str | None = None, status: str = "completed") -> str:
    trip_id = trip_id or str(uuid.uuid4())
    now = datetime(2026, 8, 1, 8, 0, 0, tzinfo=timezone.utc)
    db.add(
        Trip(
            id=trip_id,
            user_id=user_id,
            started_at=now,
            ended_at=now + timedelta(minutes=5) if status == "completed" else None,
            status=status,
        )
    )
    return trip_id


def _sample_payload(timestamp: str, speed: float | None, *, ax: float = 0.0, gz: float = 0.0) -> dict:
    return {
        "timestamp": timestamp,
        "speed": speed,
        "ax": ax,
        "ay": 0.0,
        "az": 9.81,
        "gx": 0.0,
        "gy": 0.0,
        "gz": gz,
    }


# ---------------------------------------------------------------------------
# CRIT-1 — single-sample events are counted at low (GPS) sample rates
# ---------------------------------------------------------------------------

def test_single_sample_hard_brake_is_counted() -> None:
    # 1 Hz samples: 60 km/h cruise, ONE sample at 20 km/h (a genuine hard brake
    # spanning a single GPS fix), then back to 60. Previously this produced zero
    # events because the 0.25 s duration floor discarded 1-sample segments.
    start = datetime(2026, 1, 30, 8, 0, 0, tzinfo=timezone.utc)
    speeds = [60.0] * 20 + [20.0] + [60.0] * 20
    samples = [
        _sample_payload((start + timedelta(seconds=i)).isoformat(), v)
        for i, v in enumerate(speeds)
    ]

    result = run_trip_pipeline(samples, FeatureConfigV2())
    features = result["trip_features"]

    assert features["harsh_brake_count"] >= 1
    assert result["score"] is not None
    assert any(ev["event_type"] in {"hard_brake", "emergency_brake"} for ev in result["event_instances"])


def test_two_sample_emergency_brake_is_counted_as_brake() -> None:
    start = datetime(2026, 1, 30, 8, 0, 0, tzinfo=timezone.utc)
    speeds = [60.0] * 20 + [20.0, 20.0] + [60.0] * 20
    samples = [
        _sample_payload((start + timedelta(seconds=i)).isoformat(), v)
        for i, v in enumerate(speeds)
    ]

    result = run_trip_pipeline(samples, FeatureConfigV2())
    assert result["trip_features"]["harsh_brake_count"] >= 1


# ---------------------------------------------------------------------------
# CRIT-2 — missing GPS speed does not crash finalization
# ---------------------------------------------------------------------------

def test_finalize_trip_survives_missing_gps_speed(tmp_path: Path) -> None:
    db = _make_session(tmp_path)
    user_id = _add_user(db)
    trip_id = _add_trip(db, user_id, status="completed")
    start = datetime(2026, 8, 1, 8, 0, 0, tzinfo=timezone.utc)

    for i in range(30):
        # 4 samples have no GPS speed (e.g. first fixes / tunnels)
        speed_mps = None if i % 8 in {3, 4, 12, 20} else 12.0
        db.add(
            SensorSample(
                user_id=user_id,
                trip_id=trip_id,
                ts=start + timedelta(seconds=i),
                speed_mps=speed_mps,
                lat=9.0,
                lon=38.7,
                ax=0.0,
                ay=0.0,
                az=9.81,
                gx=0.0,
                gy=0.0,
                gz=0.0,
            )
        )
    db.commit()

    result = TripProcessingService(db).finalize_trip(user_id=user_id, trip_id=trip_id, delete_raw=False)

    assert result["score"] is not None
    assert result["breakdown"].get("error") != "not_enough_samples"
    trip = db.execute(select(Trip).where(Trip.id == trip_id)).scalar_one()
    assert trip.processed_at is not None


def test_finalize_trip_handles_all_samples_missing_speed_without_crash(tmp_path: Path) -> None:
    db = _make_session(tmp_path)
    user_id = _add_user(db)
    trip_id = _add_trip(db, user_id, status="completed")
    start = datetime(2026, 8, 1, 8, 0, 0, tzinfo=timezone.utc)

    for i in range(5):
        db.add(
            SensorSample(
                user_id=user_id,
                trip_id=trip_id,
                ts=start + timedelta(seconds=i),
                speed_mps=None,
                lat=9.0,
                lon=38.7,
                ax=0.0,
                ay=0.0,
                az=9.81,
            )
        )
    db.commit()

    result = TripProcessingService(db).finalize_trip(user_id=user_id, trip_id=trip_id, delete_raw=False)

    # No crash; the trip is preserved as unscored rather than deleted.
    assert result["score"] is None
    assert db.execute(select(Trip).where(Trip.id == trip_id)).scalar_one_or_none() is not None


# ---------------------------------------------------------------------------
# CRIT-4 — speed_variation no longer double-counts brake/accel events
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset_name", ["risky_trip_240_samples_1.json", "risky_trip_240_samples_2.json", "risky_trip_240_samples_3.json"])
def test_speed_variation_no_longer_duplicates_brake_accel_events(dataset_name: str) -> None:
    # The speed_variation category used to emit a duplicate event for every hard
    # brake / hard acceleration (its |dv| >= 2.25 window overlapped the 2.5
    # brake/accel thresholds). Phase 2 removed it from persisted events.
    dataset_path = BACKEND_ROOT / "artifacts" / "datasets" / "risky_batch" / dataset_name
    if not dataset_path.exists():
        pytest.skip("risky_batch dataset not present")
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples = payload["samples"]
    for s in samples:
        s["speed"] = s.get("speed", s.get("speed_kph"))

    cfg = FeatureConfigV2()
    result = run_trip_pipeline(samples, cfg)
    assert result["trip_features"]

    events = result["event_instances"]
    assert all(e["event_type"] != "speed_variation" for e in events)
    assert any(e["event_type"] in {"hard_brake", "emergency_brake", "hard_accel"} for e in events)


# ---------------------------------------------------------------------------
# H-4 — event creation enforces trip ownership
# ---------------------------------------------------------------------------

def test_add_event_rejects_unowned_trip(tmp_path: Path) -> None:
    db = _make_session(tmp_path)
    owner_id = _add_user(db)
    other_id = _add_user(db)
    trip_id = _add_trip(db, owner_id, status="active")
    db.commit()

    service = DrivingEventService(
        repo=DrivingEventRepository(db),
        trip_repo=SqlTripRepository(db),
    )

    with pytest.raises(NotFoundError):
        service.add_event(user_id=other_id, trip_id=trip_id, event_type="hard_brake", value=3.0)

    # Owner can still create events.
    event = service.add_event(user_id=owner_id, trip_id=trip_id, event_type="hard_brake", value=3.0)
    assert event is not None


# ---------------------------------------------------------------------------
# H-7 — uploads only for active trips + rate limiter
# ---------------------------------------------------------------------------

def test_upload_samples_rejects_sealed_trip_and_allows_unfinalized(tmp_path: Path) -> None:
    # H-7 (Phase 6 refinement): completed-but-UNFINALIZED trips accept uploads
    # (offline flush before finalize); scored/sealed trips reject them.
    db = _make_session(tmp_path)
    user_id = _add_user(db)
    completed_id = _add_trip(db, user_id, status="completed")
    active_id = _add_trip(db, user_id, status="active")

    now = datetime(2026, 8, 1, 8, 0, 0, tzinfo=timezone.utc)
    sealed_id = str(uuid.uuid4())
    db.add(
        Trip(
            id=sealed_id,
            user_id=user_id,
            started_at=now,
            ended_at=now + timedelta(minutes=5),
            status="completed",
            score=80,
            processed_at=now,
        )
    )
    db.commit()

    service = SensorSampleService(
        repo=SensorSampleRepository(db),
        trip_repo=SqlTripRepository(db),
    )
    sample = {
        "ts": datetime(2026, 8, 1, 8, 5, 0, tzinfo=timezone.utc),
        "speed_mps": 10.0,
        "ax": 0.0,
        "ay": 0.0,
        "az": 9.81,
    }

    with pytest.raises(AppError):
        service.add_samples(user_id=user_id, trip_id=sealed_id, samples=[sample])

    # Completed-but-unfinalized and active trips both accept uploads.
    inserted = service.add_samples(user_id=user_id, trip_id=completed_id, samples=[sample])
    assert inserted == 1
    inserted = service.add_samples(user_id=user_id, trip_id=active_id, samples=[sample])
    assert inserted == 1


def test_rate_limiter_blocks_after_limit() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60.0)

    assert limiter.allow("key") is True
    assert limiter.allow("key") is True
    assert limiter.allow("key") is True
    assert limiter.allow("key") is False
    # Other keys are independent.
    assert limiter.allow("other") is True


def test_rate_limiter_recovers_after_window_elapses(monkeypatch) -> None:
    import time

    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60.0)
    assert limiter.allow("k") is True
    assert limiter.allow("k") is True
    assert limiter.allow("k") is False

    # Simulate the window sliding forward (capture the original first to avoid
    # recursive patching).
    original_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original_monotonic() + 61.0)
    assert limiter.allow("k") is True


# ---------------------------------------------------------------------------
# H-5 — admin trip listing pagination
# ---------------------------------------------------------------------------

def test_admin_list_all_trips_respects_pagination(tmp_path: Path) -> None:
    db = _make_session(tmp_path)
    admin = UserRecord(id="admin-1", email="admin@example.com", password_hash="h", role="admin")
    user_id = _add_user(db)
    now = datetime(2026, 8, 1, 8, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        db.add(
            Trip(
                id=str(uuid.uuid4()),
                user_id=user_id,
                started_at=now + timedelta(hours=i),
                ended_at=now + timedelta(hours=i, minutes=5),
                status="completed",
            )
        )
    db.commit()

    service = AdminService(db, SqlUserRepository(db))
    all_trips = service.list_all_trips(actor=admin)
    page = service.list_all_trips(actor=admin, limit=2, offset=1)

    assert len(all_trips) == 5
    assert len(page) == 2
    assert [t.id for t in page] == [t.id for t in all_trips[1:3]]

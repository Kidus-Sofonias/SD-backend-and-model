from __future__ import annotations

import json
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
from app.db.base import Base
from app.db.models.driving_event import DrivingEvent
from app.db.models.trip import Trip
from app.db.models.user import User
from app.db.session import get_db
from app.main import app
from app.ml.inference import ModelScorer
from app.repositories.user_repository import UserRecord


def _make_session_factory(tmp_path: Path):
    db_path = tmp_path / "api-test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


def _load_samples() -> list[dict]:
    dataset_path = BACKEND_ROOT / "artifacts" / "datasets" / "risky_batch" / "risky_trip_240_samples_1.json"
    return json.loads(dataset_path.read_text(encoding="utf-8"))["samples"]


def _load_too_few_samples() -> list[dict]:
    base_ts = datetime(2026, 3, 24, 11, 6, 0, tzinfo=timezone.utc)
    samples: list[dict] = []
    for i in range(2):
        ts = (base_ts + timedelta(seconds=i)).isoformat().replace("+00:00", "Z")
        samples.append(
            {
                "timestamp": ts,
                "speed": 12.0 + i,
                "lat": 1.0,
                "lon": 1.0,
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


def test_trip_api_critical_path_persists_score_and_events(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = UserRecord(id=str(uuid.uuid4()), email="api@example.com", password_hash="hashed")

    with session_factory() as db:
        db.add(User(id=user.id, email=user.email, password_hash=user.password_hash))
        db.commit()

    client = _client_with_overrides(session_factory, user)
    samples = _load_samples()

    try:
        start_res = client.post("/api/v1/trips/start")
        assert start_res.status_code == 200
        trip_id = start_res.json()["id"]

        upload_res = client.post(f"/api/v1/trips/{trip_id}/samples", json={"samples": samples})
        assert upload_res.status_code == 200
        assert upload_res.json()["inserted"] == len(samples)

        end_res = client.post(f"/api/v1/trips/{trip_id}/end")
        assert end_res.status_code == 200
        assert end_res.json()["status"] == "completed"

        finalize_res = client.post(f"/api/v1/trips/{trip_id}/finalize")
        assert finalize_res.status_code == 200
        payload = finalize_res.json()

        assert payload["score"] is not None
        assert payload["risk_level"] in {"low", "medium", "high"}
        assert payload["risk_probability"] is not None
        assert payload["confidence"] is not None
        assert payload["confidence_band"] in {"high", "medium", "low"}
        assert payload["confidence_display"] in {
            "show_normally",
            "show_with_caution",
            "insufficient_data",
        }
        assert isinstance(payload["events"], list)
        assert isinstance(payload["reasons"], list)

        review_res = client.get(f"/api/v1/trips/{trip_id}/review")
        assert review_res.status_code == 403

        dashboard_res = client.get("/api/v1/trips/review-dashboard")
        assert dashboard_res.status_code == 403

        set_label_res = client.post(
            f"/api/v1/trips/{trip_id}/review-label",
            json={"reviewed_label": 1, "reviewed_label_source": "human_review", "review_notes": "driver should not set this"},
        )
        assert set_label_res.status_code == 403

        with session_factory() as db:
            trip = db.execute(select(Trip).where(Trip.id == trip_id)).scalar_one()
            events = db.execute(select(DrivingEvent).where(DrivingEvent.trip_id == trip_id)).scalars().all()

            assert trip.score == payload["score"]
            assert trip.processed_at is not None
            assert len(events) == payload["events_generated"]
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_trip_api_reports_uploaded_sample_count(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = UserRecord(id=str(uuid.uuid4()), email="samplecount@example.com", password_hash="hashed")

    with session_factory() as db:
        db.add(User(id=user.id, email=user.email, password_hash=user.password_hash))
        db.commit()

    client = _client_with_overrides(session_factory, user)
    samples = _load_samples()[:12]

    try:
        trip_id = client.post("/api/v1/trips/start").json()["id"]
        upload_res = client.post(f"/api/v1/trips/{trip_id}/samples", json={"samples": samples})
        assert upload_res.status_code == 200
        assert upload_res.json()["inserted"] == len(samples)

        count_res = client.get(f"/api/v1/trips/{trip_id}/samples/count")
        assert count_res.status_code == 200
        assert count_res.json() == {"trip_id": trip_id, "count": len(samples)}
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_trip_api_finalize_uses_rules_fallback_when_model_fails(tmp_path: Path, monkeypatch) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = UserRecord(id=str(uuid.uuid4()), email="fallback@example.com", password_hash="hashed")

    with session_factory() as db:
        db.add(User(id=user.id, email=user.email, password_hash=user.password_hash))
        db.commit()

    client = _client_with_overrides(session_factory, user)
    samples = _load_samples()

    def _boom(self, trip_features: dict) -> dict:
        raise RuntimeError("model failed")

    monkeypatch.setattr(ModelScorer, "predict", _boom)

    try:
        trip_id = client.post("/api/v1/trips/start").json()["id"]
        client.post(f"/api/v1/trips/{trip_id}/samples", json={"samples": samples})
        client.post(f"/api/v1/trips/{trip_id}/end")

        finalize_res = client.post(f"/api/v1/trips/{trip_id}/finalize")
        assert finalize_res.status_code == 200
        payload = finalize_res.json()

        assert payload["decision_source"] == "rules_fallback"
        assert payload["breakdown"]["rule_score"] == payload["score"]
        assert "ml_error" in payload["breakdown"]["rule_breakdown"]
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_trip_api_deletes_trip_when_samples_are_insufficient(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = UserRecord(id=str(uuid.uuid4()), email="insufficient@example.com", password_hash="hashed")

    with session_factory() as db:
        db.add(User(id=user.id, email=user.email, password_hash=user.password_hash))
        db.commit()

    client = _client_with_overrides(session_factory, user)
    tiny_samples = _load_too_few_samples()

    try:
        trip_id = client.post("/api/v1/trips/start").json()["id"]
        upload_res = client.post(f"/api/v1/trips/{trip_id}/samples", json={"samples": tiny_samples})
        assert upload_res.status_code == 200
        client.post(f"/api/v1/trips/{trip_id}/end")

        finalize_res = client.post(f"/api/v1/trips/{trip_id}/finalize")
        assert finalize_res.status_code == 200
        payload = finalize_res.json()
        assert payload["score"] is None
        assert payload["breakdown"]["error"] == "not_enough_samples"
        assert payload["breakdown"]["trip_deleted"] is True

        trips_res = client.get("/api/v1/trips")
        assert trips_res.status_code == 200
        assert all(trip["id"] != trip_id for trip in trips_res.json())

        with session_factory() as db:
            deleted_trip = db.execute(select(Trip).where(Trip.id == trip_id)).scalar_one_or_none()
            assert deleted_trip is None
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_admin_review_dashboard_can_see_other_drivers(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    driver = UserRecord(id=str(uuid.uuid4()), email="driver@example.com", password_hash="hashed")
    admin = UserRecord(id=str(uuid.uuid4()), email="admin@sdb.com", password_hash="hashed", role="admin")

    with session_factory() as db:
        db.add(User(id=driver.id, email=driver.email, password_hash=driver.password_hash, role=driver.role))
        db.add(User(id=admin.id, email=admin.email, password_hash=admin.password_hash, role=admin.role))
        db.commit()

    driver_client = _client_with_overrides(session_factory, driver)
    samples = _load_samples()

    try:
        trip_id = driver_client.post("/api/v1/trips/start").json()["id"]
        driver_client.post(f"/api/v1/trips/{trip_id}/samples", json={"samples": samples})
        driver_client.post(f"/api/v1/trips/{trip_id}/end")
        driver_client.post(f"/api/v1/trips/{trip_id}/finalize")
    finally:
        driver_client.close()
        app.dependency_overrides.clear()

    admin_client = _client_with_overrides(session_factory, admin)
    try:
        dashboard_res = admin_client.get("/api/v1/trips/review-dashboard")
        assert dashboard_res.status_code == 200
        payload = dashboard_res.json()

        assert payload
        assert payload[0]["trip_id"] == trip_id
        assert payload[0]["driver_email"] == driver.email

        review_res = admin_client.get(f"/api/v1/trips/{trip_id}/review")
        assert review_res.status_code == 200
        review_payload = review_res.json()
        assert review_payload["driver_email"] == driver.email
        assert review_payload["driver_user_id"] == driver.id

        set_label_res = admin_client.post(
            f"/api/v1/trips/{trip_id}/review-label",
            json={"reviewed_label": 1, "reviewed_label_source": "human_review", "review_notes": "confirmed by admin"},
        )
        assert set_label_res.status_code == 200
        set_label_payload = set_label_res.json()
        assert set_label_payload["reviewed_label"] == 1
        assert set_label_payload["review_notes"] == "confirmed by admin"
    finally:
        admin_client.close()
        app.dependency_overrides.clear()


def test_admin_can_list_update_and_delete_drivers(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    driver = UserRecord(id=str(uuid.uuid4()), email="driver@example.com", password_hash="hashed")
    admin = UserRecord(id=str(uuid.uuid4()), email="admin@sdb.com", password_hash="hashed", role="admin")

    with session_factory() as db:
        db.add(User(id=driver.id, email=driver.email, password_hash=driver.password_hash, role=driver.role))
        db.add(User(id=admin.id, email=admin.email, password_hash=admin.password_hash, role=admin.role))
        db.add(
            Trip(
                id=str(uuid.uuid4()),
                user_id=driver.id,
                started_at=datetime.utcnow(),
                ended_at=datetime.utcnow(),
                status="completed",
            )
        )
        db.commit()

    client = _client_with_overrides(session_factory, admin)
    try:
        list_res = client.get("/api/v1/admin/drivers")
        assert list_res.status_code == 200
        payload = list_res.json()
        assert payload
        assert payload[0]["id"] == driver.id
        assert payload[0]["trip_count"] == 1

        trips_res = client.get(f"/api/v1/admin/drivers/{driver.id}/trips")
        assert trips_res.status_code == 200
        assert len(trips_res.json()) == 1

        update_res = client.patch(
            f"/api/v1/admin/drivers/{driver.id}",
            json={"email": "updated@example.com", "password": "new-password-123"},
        )
        assert update_res.status_code == 200
        assert update_res.json()["email"] == "updated@example.com"

        with session_factory() as db:
            updated_driver = db.execute(select(User).where(User.id == driver.id)).scalar_one()
            assert updated_driver.email == "updated@example.com"
            assert updated_driver.password_hash != "hashed"

        delete_res = client.delete(f"/api/v1/admin/drivers/{driver.id}")
        assert delete_res.status_code == 204

        with session_factory() as db:
            deleted_driver = db.execute(select(User).where(User.id == driver.id)).scalar_one_or_none()
            deleted_trips = db.execute(select(Trip).where(Trip.user_id == driver.id)).scalars().all()
            assert deleted_driver is None
            assert deleted_trips == []
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_admin_can_fetch_driver_trip_route_points(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    driver = UserRecord(id=str(uuid.uuid4()), email="driver-route@example.com", password_hash="hashed")
    admin = UserRecord(id=str(uuid.uuid4()), email="admin-route@sdb.com", password_hash="hashed", role="admin")

    with session_factory() as db:
        db.add(User(id=driver.id, email=driver.email, password_hash=driver.password_hash, role=driver.role))
        db.add(User(id=admin.id, email=admin.email, password_hash=admin.password_hash, role=admin.role))
        db.commit()

    driver_client = _client_with_overrides(session_factory, driver)
    samples = _load_samples()[:8]

    try:
        trip_id = driver_client.post("/api/v1/trips/start").json()["id"]
        upload_res = driver_client.post(f"/api/v1/trips/{trip_id}/samples", json={"samples": samples})
        assert upload_res.status_code == 200
        driver_client.post(f"/api/v1/trips/{trip_id}/end")
    finally:
        driver_client.close()
        app.dependency_overrides.clear()

    admin_client = _client_with_overrides(session_factory, admin)
    try:
        route_res = admin_client.get(f"/api/v1/admin/drivers/{driver.id}/trips/{trip_id}/route")
        assert route_res.status_code == 200
        payload = route_res.json()

        assert payload["trip_id"] == trip_id
        assert payload["driver_user_id"] == driver.id
        assert payload["point_count"] == len(samples)
        assert len(payload["points"]) == len(samples)
        assert payload["points"][0]["lat"] == samples[0]["lat"]
        assert payload["points"][0]["lon"] == samples[0]["lon"]
        assert payload["points"][0]["ts"] == samples[0]["timestamp"]
    finally:
        admin_client.close()
        app.dependency_overrides.clear()


def test_non_admin_cannot_access_driver_management(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    driver = UserRecord(id=str(uuid.uuid4()), email="driver@example.com", password_hash="hashed")
    other_driver = UserRecord(id=str(uuid.uuid4()), email="other@example.com", password_hash="hashed")

    with session_factory() as db:
        db.add(User(id=driver.id, email=driver.email, password_hash=driver.password_hash, role=driver.role))
        db.add(User(id=other_driver.id, email=other_driver.email, password_hash=other_driver.password_hash, role=other_driver.role))
        db.commit()

    client = _client_with_overrides(session_factory, driver)
    try:
        list_res = client.get("/api/v1/admin/drivers")
        assert list_res.status_code == 403

        route_res = client.get(f"/api/v1/admin/drivers/{other_driver.id}/trips/{uuid.uuid4()}/route")
        assert route_res.status_code == 403

        update_res = client.patch(
            f"/api/v1/admin/drivers/{other_driver.id}",
            json={"email": "nope@example.com"},
        )
        assert update_res.status_code == 403
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_admin_driver_trips_omits_insufficient_sample_failures(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    driver = UserRecord(id=str(uuid.uuid4()), email="driver-insufficient@example.com", password_hash="hashed")
    admin = UserRecord(id=str(uuid.uuid4()), email="admin-insufficient@sdb.com", password_hash="hashed", role="admin")

    with session_factory() as db:
        db.add(User(id=driver.id, email=driver.email, password_hash=driver.password_hash, role=driver.role))
        db.add(User(id=admin.id, email=admin.email, password_hash=admin.password_hash, role=admin.role))
        db.commit()

    driver_client = _client_with_overrides(session_factory, driver)
    tiny_samples = _load_too_few_samples()
    try:
        trip_id = driver_client.post("/api/v1/trips/start").json()["id"]
        driver_client.post(f"/api/v1/trips/{trip_id}/samples", json={"samples": tiny_samples})
        driver_client.post(f"/api/v1/trips/{trip_id}/end")
        finalize_res = driver_client.post(f"/api/v1/trips/{trip_id}/finalize")
        assert finalize_res.status_code == 200
        assert finalize_res.json()["breakdown"]["error"] == "not_enough_samples"
    finally:
        driver_client.close()
        app.dependency_overrides.clear()

    admin_client = _client_with_overrides(session_factory, admin)
    try:
        trips_res = admin_client.get(f"/api/v1/admin/drivers/{driver.id}/trips")
        assert trips_res.status_code == 200
        assert trips_res.json() == []
    finally:
        admin_client.close()
        app.dependency_overrides.clear()

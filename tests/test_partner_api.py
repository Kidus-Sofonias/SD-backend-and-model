from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.security import generate_api_key, hash_api_key
from app.db.base import Base
from app.db.models.organization import Organization, PartnerApiKey
from app.db.models.trip import Trip
from app.db.models.user import User
from app.db.session import get_db
from app.main import app


def test_partner_key_is_hashed_and_isolated_by_organization(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'partner.db'}")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, class_=Session)

    first_org = Organization(id=str(uuid.uuid4()), name="First", slug="first")
    second_org = Organization(id=str(uuid.uuid4()), name="Second", slug="second")
    first_driver = User(
        id=str(uuid.uuid4()),
        email="first@example.com",
        password_hash="unused",
        organization_id=first_org.id,
        external_driver_id="customer-driver-1",
    )
    second_driver = User(
        id=str(uuid.uuid4()),
        email="second@example.com",
        password_hash="unused",
        organization_id=second_org.id,
        external_driver_id="customer-driver-1",
    )
    second_driver_id = second_driver.id
    raw_key = generate_api_key()
    key = PartnerApiKey(
        organization_id=first_org.id,
        name="test",
        key_prefix=raw_key[:16],
        key_hash=hash_api_key(raw_key),
    )
    with session_factory() as db:
        db.add_all([first_org, second_org, first_driver, second_driver, key])
        db.add(
            Trip(
                id=str(uuid.uuid4()),
                user_id=second_driver_id,
                started_at=datetime.now(timezone.utc),
                status="completed",
                score=20,
            )
        )
        db.commit()

    def override_get_db():
        with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    try:
        response = client.get("/api/v1/partner/drivers", headers={"X-API-Key": raw_key})
        assert response.status_code == 200
        assert [item["external_driver_id"] for item in response.json()] == ["customer-driver-1"]

        response = client.get(
            f"/api/v1/partner/drivers/{second_driver_id}/trips",
            headers={"X-API-Key": raw_key},
        )
        assert response.status_code == 200
        assert response.json() == []

        with session_factory() as db:
            stored = db.execute(select(PartnerApiKey)).scalar_one()
            assert stored.key_hash == hash_api_key(raw_key)
            assert stored.key_hash != raw_key
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_partner_trip_ingestion_is_idempotent(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'partner-ingest.db'}")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, class_=Session)
    organization = Organization(id=str(uuid.uuid4()), name="Fleet", slug="fleet")
    raw_key = generate_api_key()
    key = PartnerApiKey(
        organization_id=organization.id,
        name="test",
        key_prefix=raw_key[:16],
        key_hash=hash_api_key(raw_key),
    )
    with session_factory() as db:
        db.add_all([organization, key])
        db.commit()

    def override_get_db():
        with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    body = {
        "trips": [
            {
                "source_trip_id": "company-trip-42",
                "external_driver_id": "driver-7",
                "started_at": "2026-08-22T10:00:00Z",
                "ended_at": "2026-08-22T10:30:00Z",
                "status": "completed",
                "score": 88,
                "risk_level": "low",
                "model_version": "lr_v1",
            }
        ]
    }
    try:
        headers = {"X-API-Key": raw_key}
        first = client.post("/api/v1/partner/ingest/trips", json=body, headers=headers)
        second = client.post("/api/v1/partner/ingest/trips", json=body, headers=headers)
        assert first.json() == {"received": 1, "created": 1, "updated": 0}
        assert second.json() == {"received": 1, "created": 0, "updated": 1}

        with session_factory() as db:
            assert db.query(Trip).count() == 1
            assert db.execute(select(User).where(User.external_driver_id == "driver-7")).scalar_one()
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_admin_can_filter_and_sort_organizations(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'organizations.db'}")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, class_=Session)
    admin = User(id=str(uuid.uuid4()), email="admin@example.com", password_hash="unused", role="admin")
    admin_id = admin.id
    admin_email = admin.email
    active_org = Organization(id=str(uuid.uuid4()), name="Zeta Freight", slug="zeta-freight", active=True)
    inactive_org = Organization(id=str(uuid.uuid4()), name="Alpha Transit", slug="alpha-transit", active=False)
    driver = User(
        id=str(uuid.uuid4()),
        email="zeta-driver@example.com",
        password_hash="unused",
        role="driver",
        organization_id=active_org.id,
        external_driver_id="zeta-1",
    )
    with session_factory() as db:
        db.add_all([admin, active_org, inactive_org, driver])
        db.add(Trip(id=str(uuid.uuid4()), user_id=driver.id, started_at=datetime.now(timezone.utc), status="completed"))
        db.commit()

    def override_get_db():
        with session_factory() as db:
            yield db

    from app.api.deps import get_current_user
    from app.repositories.user_repository import UserRecord

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: UserRecord(
        id=admin_id, email=admin_email, password_hash="unused", role="admin"
    )
    client = TestClient(app)
    try:
        response = client.get("/api/v1/admin/organizations?active=true&sort=name")
        assert response.status_code == 200
        assert [item["slug"] for item in response.json()] == ["zeta-freight"]
        assert response.json()[0]["driver_count"] == 1
        assert response.json()[0]["trip_count"] == 1

        response = client.get("/api/v1/admin/organizations?search=alpha")
        assert response.status_code == 200
        assert [item["slug"] for item in response.json()] == ["alpha-transit"]
    finally:
        client.close()
        app.dependency_overrides.clear()
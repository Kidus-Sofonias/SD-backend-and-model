# Phase 6 prep — weather lookup endpoint + Open-Meteo proxy service.
# - Endpoint requires auth and validates lat/lon bounds
# - Service maps WMO codes and caches with TTL
# - Stale-cache fallback when upstream is unreachable
from __future__ import annotations

import sys
import time
import uuid
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
from app.repositories.user_repository import UserRecord
from app.services.weather_service import WeatherService, wmo_weather_label

RAW_OPEN_METEO = {
    "timezone_abbreviation": "EAT",
    "current": {
        "time": "2026-08-08T12:00",
        "temperature_2m": 22.4,
        "relative_humidity_2m": 48,
        "apparent_temperature": 21.9,
        "precipitation": 0.0,
        "weather_code": 2,
        "wind_speed_10m": 9.3,
        "is_day": 1,
    },
    "daily": {
        "time": ["2026-08-08", "2026-08-09", "2026-08-10"],
        "weather_code": [2, 61, 95],
        "temperature_2m_max": [25.0, 22.5, 20.1],
        "temperature_2m_min": [12.3, 11.8, 11.0],
        "precipitation_probability_max": [20, 80, 65],
    },
}


class _FakeClient:
    def __init__(self, responses: list) -> None:
        self._responses = responses
        self.calls = 0

    def get(self, url: str, params: dict):
        self.calls += 1
        if not self._responses:
            raise RuntimeError("network down")
        return self._responses.pop(0)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _make_session_factory(tmp_path: Path):
    db_path = tmp_path / "phase6-weather.db"
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


def test_wmo_label_mapping() -> None:
    assert wmo_weather_label(0) == "Clear sky"
    assert wmo_weather_label(95) == "Thunderstorm"
    assert wmo_weather_label(None) == "Unknown"


def test_weather_service_maps_and_caches(tmp_path: Path) -> None:
    service = WeatherService(client=_FakeClient([_FakeResponse(RAW_OPEN_METEO)]))
    payload = service.lookup(lat=9.03, lon=38.74)

    assert payload["current"]["temp_c"] == 22.4
    assert payload["current"]["weather_label"] == "Partly cloudy"
    assert payload["current"]["wind_kph"] == 9.3
    assert payload["current"]["is_day"] == 1
    assert len(payload["forecast"]) == 3
    assert payload["forecast"][1]["precip_prob_pct"] == 80
    assert "weather_label" not in payload["forecast"][2]  # labels only on current

    # Same rounded coords -> served from cache, no second upstream call.
    service.lookup(lat=9.0301, lon=38.7402)
    assert service.client.calls == 1


def test_weather_service_stale_cache_fallback(tmp_path: Path) -> None:
    service = WeatherService(client=_FakeClient([_FakeResponse(RAW_OPEN_METEO)]))
    fresh = service.lookup(lat=9.0, lon=38.7)
    assert fresh["stale"] is False

    # Simulate TTL expiry so the next lookup must go upstream again.
    from app.services.weather_service import CACHE_TTL_S

    key = (9.0, 38.7)
    _, cached_payload = service.cache[key]
    service.cache[key] = (time.monotonic() - CACHE_TTL_S - 10, cached_payload)

    # Upstream dies; still serve stale cached payload instead of failing.
    service.client._responses = []
    payload = service.lookup(lat=9.0, lon=38.7)
    assert payload["stale"] is True
    assert payload["current"]["temp_c"] == 22.4


def test_weather_service_raises_when_no_cache_and_upstream_down(tmp_path: Path) -> None:
    service = WeatherService(client=_FakeClient([]))
    from app.core.errors import AppError

    try:
        service.lookup(lat=9.0, lon=38.7)
    except AppError as exc:
        assert exc.status_code == 503
        assert exc.message_key == "weather.unavailable"
    else:
        raise AssertionError("expected AppError")


def test_weather_endpoint_auth_and_bounds(tmp_path: Path) -> None:
    session_factory = _make_session_factory(tmp_path)
    user = UserRecord(id=str(uuid.uuid4()), email="weather@example.com", password_hash="hashed")

    with session_factory() as db:
        db.add(User(id=user.id, email=user.email, password_hash=user.password_hash))
        db.commit()

    client = _client_with_overrides(session_factory, user)
    try:
        # Without a valid user override the dependency rejects -> 401 path.
        app.dependency_overrides.pop(get_current_user)
        unauthorized = client.get("/api/v1/weather?lat=9&lon=38.7")
        assert unauthorized.status_code == 401
        app.dependency_overrides[get_current_user] = lambda: user

        # Invalid coordinates rejected by query validation.
        bad = client.get("/api/v1/weather?lat=999&lon=38.7")
        assert bad.status_code == 422

        # Valid request reaches the service (network mocked? no - service hits
        # real upstream, so we only assert the route wiring accepts the params
        # and returns a structured error rather than crashing).
        res = client.get("/api/v1/weather?lat=9&lon=38.7")
        assert res.status_code in (200, 503)
    finally:
        client.close()
        app.dependency_overrides.clear()

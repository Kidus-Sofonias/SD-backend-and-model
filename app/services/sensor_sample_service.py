# File role: Business-logic service that coordinates repositories/schemas and enforces use-case rules.
# Connects to: app.core.errors, app.repositories.sensor_sample_repository, app.repositories.trip_repository.
# Key symbols/vars: SensorSampleService.
from __future__ import annotations

from datetime import datetime
from typing import Optional

from app.core.errors import NotFoundError
from app.repositories.sensor_sample_repository import SensorSampleRepository
from app.repositories.trip_repository import SqlTripRepository


class SensorSampleService:
    def __init__(self, repo: SensorSampleRepository, trip_repo: SqlTripRepository) -> None:
        self.repo = repo
        self.trip_repo = trip_repo  # ✅ trip_repo already has db

    def add_samples(self, *, user_id: str, trip_id: str, samples: list[dict]) -> int:
        trip = self.trip_repo.get_by_id(trip_id, user_id=user_id)  # ✅ no db param
        if not trip:
            raise NotFoundError("Trip not found")

        return self.repo.create_many(user_id=user_id, trip_id=trip_id, rows=samples)

    def list_samples(
        self,
        *,
        user_id: str,
        trip_id: str,
        limit: int = 1000,
        after_ts: Optional[datetime] = None,
    ):
        trip = self.trip_repo.get_by_id(trip_id, user_id=user_id)
        if not trip:
            raise NotFoundError("Trip not found")

        return self.repo.list_by_trip(user_id=user_id, trip_id=trip_id, limit=limit, after_ts=after_ts)

    def count_samples(self, *, user_id: str, trip_id: str) -> int:
        trip = self.trip_repo.get_by_id(trip_id, user_id=user_id)
        if not trip:
            raise NotFoundError("Trip not found")

        return self.repo.count_by_trip(user_id=user_id, trip_id=trip_id)

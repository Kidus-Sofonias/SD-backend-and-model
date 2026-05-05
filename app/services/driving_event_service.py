# File role: Business-logic service that coordinates repositories/schemas and enforces use-case rules.
# Connects to: app.core.errors, app.repositories.driving_event_repository, app.repositories.trip_repository.
# Key symbols/vars: DrivingEventService.
from __future__ import annotations

from datetime import datetime
from typing import Optional

from app.core.errors import NotFoundError
from app.repositories.driving_event_repository import DrivingEventRepository
from app.repositories.trip_repository import SqlTripRepository


class DrivingEventService:
    def __init__(self, repo: DrivingEventRepository, trip_repo: SqlTripRepository):
        self.repo = repo
        self.trip_repo = trip_repo

    def add_event(
        self,
        user_id: str,
        trip_id: str,
        event_type: str,
        value: float,
        occurred_at: datetime | None = None,
        lat: float | None = None,
        lon: float | None = None,
    ):
        # validate trip belongs to user (prevents cheating)
        _trip = self.trip_repo.get_by_id(trip_id=trip_id, user_id=user_id)
        # create event
        return self.repo.create(
            user_id=user_id,
            trip_id=trip_id,
            event_type=event_type,
            value=value,
            occurred_at=occurred_at,
            lat=lat,
            lon=lon,
        )

    def list_for_trip(self, *, user_id: str, trip_id: str):
        trip = self.trip_repo.get_by_id(trip_id=trip_id, user_id=user_id)
        if not trip:
            raise NotFoundError("Trip not found")
        return self.repo.list_for_trip(user_id=user_id, trip_id=trip_id)

    def history(
        self,
        *,
        user_id: str,
        trip_id: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
    ):
        return self.repo.history(
            user_id=user_id,
            trip_id=trip_id,
            start=start,
            end=end,
            limit=limit,
            offset=offset,
        )

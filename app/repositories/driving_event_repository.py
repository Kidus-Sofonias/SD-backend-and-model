# File role: Data-access repository encapsulating SQLAlchemy queries used by service and route layers.
# Connects to: sqlalchemy, app.db.models.driving_event.
# Key symbols/vars: DrivingEventRepository.
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models.driving_event import DrivingEvent
from app.db.session import commit_with_retry


class DrivingEventRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self, user_id: str, trip_id: str, event_type: str, value: float
    ) -> DrivingEvent:
        event = DrivingEvent(
            user_id=user_id,
            trip_id=trip_id,
            event_type=event_type,
            value=value,
        )
        self.db.add(event)
        commit_with_retry(self.db)
        self.db.refresh(event)
        return event

    def list_for_trip(self, *, user_id: str, trip_id: str) -> list[DrivingEvent]:
        stmt = (
            select(DrivingEvent)
            .where(DrivingEvent.trip_id == trip_id, DrivingEvent.user_id == user_id)
            .order_by(DrivingEvent.created_at.asc())
        )
        return list(self.db.execute(stmt).scalars().all())

    def history(
        self,
        *,
        user_id: str,
        trip_id: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[DrivingEvent], int]:
        stmt = select(DrivingEvent).where(DrivingEvent.user_id == user_id)
        count_stmt = (
            select(func.count())
            .select_from(DrivingEvent)
            .where(DrivingEvent.user_id == user_id)
        )
        if trip_id is not None:
            stmt = stmt.where(DrivingEvent.trip_id == trip_id)
            count_stmt = count_stmt.where(DrivingEvent.trip_id == trip_id)
        if start is not None:
            stmt = stmt.where(DrivingEvent.created_at >= start)
            count_stmt = count_stmt.where(DrivingEvent.created_at >= start)
        if end is not None:
            stmt = stmt.where(DrivingEvent.created_at <= end)
            count_stmt = count_stmt.where(DrivingEvent.created_at <= end)

        stmt = stmt.order_by(DrivingEvent.created_at.desc()).limit(limit).offset(offset)

        total = int(self.db.execute(count_stmt).scalar_one())
        rows = list(self.db.execute(stmt).scalars().all())
        return rows, total

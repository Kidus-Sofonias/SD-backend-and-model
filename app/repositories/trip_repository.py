# File role: Data-access repository encapsulating SQLAlchemy queries used by service and route layers.
# Connects to: sqlalchemy, app.db.models.trip.
# Key symbols/vars: SqlTripRepository.
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.trip import Trip
from app.db.session import commit_with_retry


class SqlTripRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, trip_id: str, user_id: str) -> Trip | None:
        stmt = select(Trip).where(Trip.id == trip_id, Trip.user_id == user_id)
        return self.db.execute(stmt).scalar_one_or_none()

    def create_trip(self, user_id: str) -> Trip:
        t = Trip(
            user_id=user_id,
            started_at=datetime.now(timezone.utc),
            ended_at=None,
            status="active",
        )
        self.db.add(t)
        commit_with_retry(self.db)
        self.db.refresh(t)
        return t

    def get_active_trip(self, user_id: str) -> Trip | None:
        stmt = (
            select(Trip)
            .where(Trip.user_id == user_id, Trip.status == "active")
            .order_by(Trip.started_at.desc())
        )
        return self.db.execute(stmt).scalars().first()

    def end_trip(self, trip_id: str, user_id: str) -> Trip | None:
        trip = self.get_by_id(trip_id, user_id)
        if not trip:
            return None
        trip.ended_at = datetime.now(timezone.utc)
        trip.status = "completed"
        self.db.add(trip)
        commit_with_retry(self.db)
        self.db.refresh(trip)
        return trip

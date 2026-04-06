# File role: SQLAlchemy ORM model defining a persisted entity and relationships consumed by repositories/services.
# Connects to: sqlalchemy, app.db.base.
# Key symbols/vars: DrivingEvent.
from __future__ import annotations

from datetime import datetime
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class DrivingEvent(Base):
    __tablename__ = "driving_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False, index=True)
    trip_id: Mapped[str] = mapped_column(String, ForeignKey("trips.id"), nullable=False, index=True)

    event_type: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    trip = relationship("Trip", back_populates="events")

# File role: SQLAlchemy Trip model.
# Stores trip lifecycle data plus ML/rule scoring outputs.
# Connects to:
# - app.db.base
# - app.db.models.driving_event
# - app.db.models.sensor_sample
# Key symbols/vars:
# - Trip

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Trip(Base):
    __tablename__ = "trips"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        index=True,
        default=lambda: str(uuid.uuid4()),
    )

    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False, index=True)

    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    status: Mapped[str] = mapped_column(String, nullable=False, default="active")

    # persisted scoring / processing outputs
    score: Mapped[int | None] = mapped_column(nullable=True)
    score_breakdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    feature_version: Mapped[str | None] = mapped_column(String, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_level: Mapped[str | None] = mapped_column(String, nullable=True)
    reviewed_label: Mapped[int | None] = mapped_column(nullable=True)
    reviewed_label_source: Mapped[str | None] = mapped_column(String, nullable=True)
    review_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    raw_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    events = relationship("DrivingEvent", back_populates="trip", cascade="all, delete-orphan")
    samples = relationship("SensorSample", back_populates="trip", cascade="all, delete-orphan")


# Import related models so SQLAlchemy can resolve string relationships when
# this module is imported directly.
from app.db.models import driving_event as _driving_event  # noqa: F401,E402
from app.db.models import sensor_sample as _sensor_sample  # noqa: F401,E402

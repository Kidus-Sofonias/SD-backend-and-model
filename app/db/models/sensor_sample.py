# File role: SQLAlchemy ORM model defining a persisted entity and relationships consumed by repositories/services.
# Connects to: sqlalchemy, app.db.base.
# Key symbols/vars: SensorSample.
from __future__ import annotations

from datetime import datetime
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class SensorSample(Base):
    __tablename__ = "sensor_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False, index=True)
    trip_id: Mapped[str] = mapped_column(String, ForeignKey("trips.id"), nullable=False, index=True)

    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)

    speed_mps: Mapped[float | None] = mapped_column(Float, nullable=True)
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    accuracy_m: Mapped[float | None] = mapped_column(Float, nullable=True)

    ax: Mapped[float | None] = mapped_column(Float, nullable=True)
    ay: Mapped[float | None] = mapped_column(Float, nullable=True)
    az: Mapped[float | None] = mapped_column(Float, nullable=True)

    gx: Mapped[float | None] = mapped_column(Float, nullable=True)
    gy: Mapped[float | None] = mapped_column(Float, nullable=True)
    gz: Mapped[float | None] = mapped_column(Float, nullable=True)

    trip = relationship("Trip", back_populates="samples")

    __table_args__ = (
        Index("ix_sensor_samples_trip_ts", "trip_id", "ts"),
        Index("ix_sensor_samples_user_trip", "user_id", "trip_id"),
    )

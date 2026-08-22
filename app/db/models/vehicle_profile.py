# File role: SQLAlchemy ORM model for a driver's vehicle profile (Phase 3).
# Captures the vehicle information that tunes sensor interpretation, event
# thresholds, and (later) the 3D vehicle simulation.
# Connects to: sqlalchemy, app.db.base, app.db.models.user.
# Key symbols/vars: VehicleProfile.
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class VehicleProfile(Base):
    __tablename__ = "vehicle_profiles"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    # One profile per user.
    user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("users.id"),
        nullable=False,
        unique=True,
        index=True,
    )

    # sedan | suv | pickup | van | bus | heavy_truck | tractor_trailer | other
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    # Free-text make/model, e.g. "Toyota Corolla" (optional).
    make_model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    # compact | midsize | large (optional, refines mass estimate).
    size_class: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # fwd | rwd | awd | 4wd (optional; informs acceleration expectations).
    drive_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # auto | manual (optional; engine braking changes braking-signature reading).
    transmission: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Where the phone lives while driving: mount_dashboard | cupholder | pocket | lap.
    # Calibrates phone-handling detection and vertical-noise tolerance.
    phone_placement: Mapped[str | None] = mapped_column(String(24), nullable=True)
    # Typical load: light | normal | heavy (optional; refines effective mass).
    load_level: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Typical road context: city | highway | mixed (optional; speed expectations).
    road_context: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Curated mass in kg (category default, refined by size_class/load or explicit).
    mass_kg: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    user = relationship("User", back_populates="vehicle_profile")

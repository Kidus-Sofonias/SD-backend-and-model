# File role: SQLAlchemy ORM model defining a persisted entity and relationships consumed by repositories/services.
# Connects to: sqlalchemy, app.db.base.
# Key symbols/vars: User.
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey, Index, String

from app.db.base import Base


class User(Base):
    __tablename__ = "users"
    __table_args__ = (Index("ix_users_org_external_driver", "organization_id", "external_driver_id", unique=True),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="driver")
    organization_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("organizations.id"), nullable=True, index=True
    )
    external_driver_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)

    trips = relationship("Trip", cascade="all, delete-orphan")
    organization = relationship("Organization", back_populates="users")
    vehicle_profile = relationship(
        "VehicleProfile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )

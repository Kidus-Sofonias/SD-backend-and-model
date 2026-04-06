# File role: SQLAlchemy ORM model defining a persisted entity and relationships consumed by repositories/services.
# Connects to: sqlalchemy, app.db.base.
# Key symbols/vars: User.
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="driver")

    trips = relationship("Trip", cascade="all, delete-orphan")

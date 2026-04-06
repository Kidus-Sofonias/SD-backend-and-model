# File role: Pydantic schema contract for admin-only driver management APIs.
# Connects to: app.schemas.trip.
# Key symbols/vars: DriverSummaryOut, AdminUpdateDriverIn.
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class DriverSummaryOut(BaseModel):
    id: str
    email: EmailStr
    role: str = "driver"
    trip_count: int = 0
    latest_trip_at: datetime | None = None


class AdminUpdateDriverIn(BaseModel):
    email: EmailStr | None = None
    password: str | None = Field(default=None, min_length=8, max_length=72)


class TripRoutePointOut(BaseModel):
    ts: datetime
    lat: float
    lon: float
    speed_mps: float | None = None
    accuracy_m: float | None = None


class TripRouteOut(BaseModel):
    trip_id: str
    driver_user_id: str
    point_count: int = 0
    points: list[TripRoutePointOut] = Field(default_factory=list)

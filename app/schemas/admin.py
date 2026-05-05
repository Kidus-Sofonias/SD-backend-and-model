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
    snapped_points: list[TripRoutePointOut] = Field(default_factory=list)
    snapped_source: str | None = None
    snapped_status: str = "unavailable"


class DriverTrendPointOut(BaseModel):
    period_start: datetime
    period_end: datetime
    label: str
    average_score: float | None = None
    trip_count: int = 0
    high_risk_trip_count: int = 0


class DriverTrendSnapshotOut(BaseModel):
    label: str
    average_score: float | None = None
    trip_count: int = 0
    high_risk_trip_count: int = 0


class DriverTrendWindowOut(BaseModel):
    current: DriverTrendSnapshotOut
    previous: DriverTrendSnapshotOut
    delta_score: float | None = None
    direction: str = "flat"
    points: list[DriverTrendPointOut] = Field(default_factory=list)


class DriverInsightsOut(BaseModel):
    driver_id: str
    driver_email: EmailStr
    overall_average_score: float | None = None
    scored_trip_count: int = 0
    high_risk_trip_count: int = 0
    weekly: DriverTrendWindowOut
    monthly: DriverTrendWindowOut

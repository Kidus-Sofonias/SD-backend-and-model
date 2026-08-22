from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PartnerKeyCreateIn(BaseModel):
    organization_name: str = Field(min_length=2, max_length=255)
    organization_slug: str = Field(min_length=2, max_length=120, pattern=r"^[a-z0-9-]+$")
    key_name: str = Field(default="primary", min_length=1, max_length=120)


class PartnerKeyCreateOut(BaseModel):
    organization_id: str
    api_key_id: str
    api_key: str
    warning: str = "Store this key now. It cannot be retrieved later."


class PartnerKeySummaryOut(BaseModel):
    id: str
    organization_id: str
    name: str
    key_prefix: str
    active: bool
    created_at: datetime
    last_used_at: datetime | None = None


class OrganizationSummaryOut(BaseModel):
    id: str
    name: str
    slug: str
    active: bool
    created_at: datetime
    driver_count: int = 0
    trip_count: int = 0
    active_key_count: int = 0
    latest_trip_at: datetime | None = None


class PartnerDriverOut(BaseModel):
    external_driver_id: str | None = None
    driver_id: str
    trip_count: int = 0
    latest_trip_at: datetime | None = None


class PartnerTripOut(BaseModel):
    id: str
    source_trip_id: str | None = None
    external_driver_id: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    status: str
    score: int | None = None
    risk_probability: float | None = None
    risk_level: str | None = None
    confidence: float | None = None
    feature_version: str | None = None
    model_version: str | None = None
    processed_at: datetime | None = None


class PartnerDriverStatsOut(BaseModel):
    external_driver_id: str | None = None
    trip_count: int = 0
    scored_trip_count: int = 0
    average_score: float | None = None
    high_risk_trip_count: int = 0


class PartnerTripIn(BaseModel):
    source_trip_id: str = Field(min_length=1, max_length=255)
    external_driver_id: str = Field(min_length=1, max_length=255)
    started_at: datetime
    ended_at: datetime | None = None
    status: str = Field(default="completed", max_length=32)
    score: int | None = Field(default=None, ge=0, le=100)
    risk_probability: float | None = Field(default=None, ge=0, le=1)
    risk_level: str | None = Field(default=None, max_length=32)
    confidence: float | None = Field(default=None, ge=0, le=1)
    feature_version: str | None = Field(default=None, max_length=64)
    model_version: str | None = Field(default=None, max_length=64)
    processed_at: datetime | None = None


class PartnerTripBatchIn(BaseModel):
    trips: list[PartnerTripIn] = Field(min_length=1, max_length=500)


class PartnerIngestOut(BaseModel):
    received: int
    created: int
    updated: int
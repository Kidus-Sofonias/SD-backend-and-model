# File role: Pydantic schema contract for request validation and response serialization across API layers.
# Connects to: app.schemas.events.
# Key symbols/vars: TripStartRequest, TripEndRequest, TripOut, TripDetailOut.
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.events import DrivingEventOut


class TripStartRequest(BaseModel):
    # keep it optional for now (frontend can send empty body)
    pass


class TripEndRequest(BaseModel):
    # same
    pass


class TripOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    started_at: datetime
    ended_at: Optional[datetime]
    status: str
    score: int | None = None
    risk_level: str | None = None
    risk_probability: float | None = None
    confidence: float | None = None
    confidence_band: str | None = None
    confidence_display: str | None = None
    feature_version: str | None = None
    model_version: str | None = None
    processed_at: datetime | None = None


class TripDetailOut(TripOut):
    decision_source: str | None = None
    raw_deleted: bool | None = None
    already_processed: bool | None = None
    reasons: list[str] = Field(default_factory=list)
    events: list[DrivingEventOut] = Field(default_factory=list)
    breakdown: dict[str, Any] = Field(default_factory=dict)
    trip_features: dict[str, Any] = Field(default_factory=dict)
    events_generated: int | None = None


class TripSummaryOut(BaseModel):
    trip_id: str
    status: str
    started_at: datetime
    ended_at: Optional[datetime]

    total_events: int
    counts: dict[str, int]

    # simple score (100 - penalties)
    score: int


class FinalizeTripOut(BaseModel):
    trip_id: str
    score: int | None
    risk_level: str | None = None
    risk_probability: float | None = None
    confidence: float | None = None
    confidence_band: str | None = None
    confidence_display: str | None = None
    model_version: str | None = None
    feature_version: str | None = None
    decision_source: str | None = None
    processing_timestamp: datetime | None = None
    raw_deleted: bool | None = None
    already_processed: bool | None = None
    reasons: list[str] = Field(default_factory=list)
    events: list[DrivingEventOut] = Field(default_factory=list)
    breakdown: dict[str, Any] = Field(default_factory=dict)
    trip_features: dict[str, Any] = Field(default_factory=dict)
    events_generated: int | None = None


class TripReviewOut(BaseModel):
    trip_id: str
    driver_user_id: str | None = None
    driver_email: str | None = None
    score: int | None = None
    risk_level: str | None = None
    risk_probability: float | None = None
    confidence: float | None = None
    confidence_band: str | None = None
    confidence_display: str | None = None
    feature_version: str | None = None
    model_version: str | None = None
    processed_at: datetime | None = None
    trip_features: dict[str, Any] = Field(default_factory=dict)
    rule_score: int | None = None
    ml_prediction: int | None = None
    predicted_label: int | None = None
    reasons: list[str] = Field(default_factory=list)
    events: list[DrivingEventOut] = Field(default_factory=list)
    reviewed_label: int | None = None
    reviewed_label_source: str | None = None
    review_disagrees_with_prediction: bool | None = None
    review_notes: str | None = None
    reviewed_at: datetime | None = None


class TripReviewLabelIn(BaseModel):
    reviewed_label: int | None = None
    reviewed_label_source: str = "human_review"
    review_notes: str | None = None


class ReprocessTripsOut(BaseModel):
    matched: int
    reprocessed: int
    failed: int
    trip_ids: list[str] = Field(default_factory=list)


class TripReviewDashboardItemOut(BaseModel):
    trip_id: str
    driver_user_id: str | None = None
    driver_email: str | None = None
    score: int | None = None
    risk_level: str | None = None
    risk_probability: float | None = None
    confidence: float | None = None
    confidence_band: str | None = None
    confidence_display: str | None = None
    rule_score: int | None = None
    predicted_label: int | None = None
    reasons: list[str] = Field(default_factory=list)
    generated_events: list[DrivingEventOut] = Field(default_factory=list)
    trip_events: list[DrivingEventOut] = Field(default_factory=list)
    generated_event_count: int = 0
    trip_event_count: int = 0
    review_label: int | None = None
    review_label_source: str | None = None
    review_disagrees_with_prediction: bool | None = None
    model_version: str | None = None
    feature_version: str | None = None
    processed_at: datetime | None = None
    reviewed_at: datetime | None = None

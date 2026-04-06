# File role: Pydantic schema contract for request validation and response serialization across API layers.
# Connects to: nearby package modules via local imports.
# Key symbols/vars: DrivingEventCreate, DrivingEventOut, DrivingEventListResponse, DrivingEventHistoryResponse.
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class DrivingEventCreate(BaseModel):
    event_type: str = Field(..., min_length=1, max_length=50)
    value: float


class DrivingEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    trip_id: str
    event_type: str
    value: float
    created_at: datetime

class DrivingEventListResponse(BaseModel):
    events: list[DrivingEventOut] = Field(default_factory=list)

class DrivingEventHistoryResponse(BaseModel):
    events: list[DrivingEventOut] = Field(default_factory=list)
    limit: int
    offset: int
    total: int


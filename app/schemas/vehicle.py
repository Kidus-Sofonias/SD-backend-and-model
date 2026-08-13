# File role: Pydantic schema contract for the vehicle profile API (Phase 3).
# Connects to: app.ml.vehicle_profiles for derived thresholds.
# Key symbols/vars: VehicleProfileIn, VehicleProfileOut.
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ml.vehicle_profiles import VEHICLE_CATEGORIES


class VehicleProfileIn(BaseModel):
    category: str = Field(..., description="Vehicle category key (sedan, suv, pickup, van, bus, heavy_truck, tractor_trailer, other)")
    make_model: str | None = Field(default=None, max_length=120)
    size_class: str | None = Field(default=None, description="compact | midsize | large")
    drive_type: str | None = Field(default=None, description="fwd | rwd | awd | 4wd")
    mass_kg: float | None = Field(default=None, gt=200, le=100000)

    @field_validator("category")
    @classmethod
    def _validate_category(cls, value: str) -> str:
        if value not in VEHICLE_CATEGORIES:
            raise ValueError(f"Unknown vehicle category: {value}")
        return value

    @field_validator("size_class")
    @classmethod
    def _validate_size_class(cls, value: str | None) -> str | None:
        if value not in (None, "compact", "midsize", "large"):
            raise ValueError(f"Unknown size class: {value}")
        return value

    @field_validator("drive_type")
    @classmethod
    def _validate_drive_type(cls, value: str | None) -> str | None:
        if value not in (None, "fwd", "rwd", "awd", "4wd"):
            raise ValueError(f"Unknown drive type: {value}")
        return value


class VehicleProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    category: str
    make_model: str | None = None
    size_class: str | None = None
    drive_type: str | None = None
    mass_kg: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)

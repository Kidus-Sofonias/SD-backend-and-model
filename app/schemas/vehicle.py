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
    transmission: str | None = Field(default=None, description="auto | manual")
    phone_placement: str | None = Field(default=None, description="mount_dashboard | cupholder | pocket | lap")
    load_level: str | None = Field(default=None, description="light | normal | heavy")
    road_context: str | None = Field(default=None, description="city | highway | mixed")
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

    @field_validator("transmission")
    @classmethod
    def _validate_transmission(cls, value: str | None) -> str | None:
        if value not in (None, "auto", "manual"):
            raise ValueError(f"Unknown transmission: {value}")
        return value

    @field_validator("phone_placement")
    @classmethod
    def _validate_phone_placement(cls, value: str | None) -> str | None:
        if value not in (None, "mount_dashboard", "cupholder", "pocket", "lap"):
            raise ValueError(f"Unknown phone placement: {value}")
        return value

    @field_validator("load_level")
    @classmethod
    def _validate_load_level(cls, value: str | None) -> str | None:
        if value not in (None, "light", "normal", "heavy"):
            raise ValueError(f"Unknown load level: {value}")
        return value

    @field_validator("road_context")
    @classmethod
    def _validate_road_context(cls, value: str | None) -> str | None:
        if value not in (None, "city", "highway", "mixed"):
            raise ValueError(f"Unknown road context: {value}")
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
    transmission: str | None = None
    phone_placement: str | None = None
    load_level: str | None = None
    road_context: str | None = None
    mass_kg: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)

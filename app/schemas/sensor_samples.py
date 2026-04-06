# File role: Pydantic schema contract for request validation and response serialization across API layers.
# Connects to: nearby package modules via local imports.
# Key symbols/vars: SensorSampleIn, SensorSamplesBatchIn, SensorSampleOut.
from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

class SensorSampleIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Accept both legacy API field names and the payload shape used by the ML pipeline.
    ts: datetime = Field(validation_alias=AliasChoices("ts", "timestamp"))

    speed_mps: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("speed_mps", "speed"),
    )
    lat: Optional[float] = None
    lon: Optional[float] = None
    accuracy_m: Optional[float] = None

    ax: Optional[float] = None
    ay: Optional[float] = None
    az: Optional[float] = None

    gx: Optional[float] = None
    gy: Optional[float] = None
    gz: Optional[float] = None


class SensorSamplesBatchIn(BaseModel):
    samples: List[SensorSampleIn] = Field(..., min_length=1, max_length=5000)


class SensorSampleOut(SensorSampleIn):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: str
    trip_id: str


class SensorSampleCountOut(BaseModel):
    trip_id: str
    count: int

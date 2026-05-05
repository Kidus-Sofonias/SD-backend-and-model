# File role: Pydantic schema contract for request validation and response serialization across API layers.
# Connects to: nearby package modules via local imports.
# Key symbols/vars: DrivingEventCreate.
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

class DrivingEventCreate(BaseModel):
    event_type: str
    value: Optional[float] = None
    occurred_at: Optional[datetime] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

# File role: Pydantic schema contract for request validation and response serialization across API layers.
# Connects to: nearby package modules via local imports.
# Key symbols/vars: DrivingEventCreate.
from pydantic import BaseModel
from typing import Optional

class DrivingEventCreate(BaseModel):
    event_type: str
    value: Optional[float] = None
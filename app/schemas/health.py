# File role: Pydantic schema contract for request validation and response serialization across API layers.
# Connects to: nearby package modules via local imports.
# Key symbols/vars: HealthData.
from pydantic import BaseModel

class HealthData(BaseModel):
    service:str
    env:str
    version:str
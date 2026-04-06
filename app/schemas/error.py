# File role: Pydantic schema contract for request validation and response serialization across API layers.
# Connects to: nearby package modules via local imports.
# Key symbols/vars: ErrorDetail, ErrorResponse.
from typing import Any, Optional
from pydantic import BaseModel, Field

class ErrorDetail(BaseModel):
    # A localization key Flutter can map to Amharic/English
    message_key: str = Field(..., examples=["error.validation"])
    # Optional machine-friendly detail for debugging
    details: Optional[Any] = None

class ErrorResponse(BaseModel):
    ok: bool = False
    error: ErrorDetail
    request_id: str = Field(..., description = "Correlation ID for debugging")
# File role: Pydantic schema contract for request validation and response serialization across API layers.
# Connects to: nearby package modules via local imports.
# Key symbols/vars: APIResponse.
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    message_key: str
    data: dict[str, Any] = Field(default_factory=dict)
    errors: Optional[list[dict[str, Any]]] = None
    request_id: Optional[str] = None

    @classmethod
    def ok(cls, message_key: str, data: Optional[dict[str, Any]] = None) -> "APIResponse":
        return cls(
            message_key=message_key,
            data=data or {},
            errors=None,
        )

    @classmethod
    def fail(
        cls,
        message_key: str,
        *,
        status_code: int,
        errors: Optional[list[dict[str, Any]]] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> "APIResponse":
        # status_code is handled by exception handlers / route responses,
        # but we keep it here for consistency if you want it later.
        return cls(
            message_key=message_key,
            data=data or {},
            errors=errors or [],
        )

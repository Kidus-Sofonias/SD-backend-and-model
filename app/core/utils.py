# File role: Shared utilities used across services and routes.
# Connects to: nearby package modules via local imports.
# Key symbols/vars: as_utc_timestamp.

from __future__ import annotations

from datetime import datetime, timezone


def as_utc_timestamp(ts: datetime | None) -> datetime | None:
    """Normalize a datetime to UTC, handling naive timestamps."""
    if ts is None:
        return ts
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)

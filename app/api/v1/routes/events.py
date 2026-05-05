# File role: HTTP route layer that maps requests to services/repositories and returns schema-shaped responses.
# Connects to: fastapi, app.api.deps, app.db.session.
# Key symbols/vars: router, _service, add_event, list_trip_events.
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.repositories.driving_event_repository import DrivingEventRepository
from app.repositories.trip_repository import SqlTripRepository
from app.schemas.common import APIResponse
from app.schemas.events import (
    DrivingEventCreate,
    DrivingEventHistoryResponse,
    DrivingEventListResponse,
    DrivingEventOut,
)
from app.services.driving_event_service import DrivingEventService

router = APIRouter(prefix="/trips", tags=["events"])


def _service(db: Session) -> DrivingEventService:
    return DrivingEventService(
        repo=DrivingEventRepository(db),
        trip_repo=SqlTripRepository(db),
    )


@router.post("/{trip_id}/events", response_model=APIResponse)
def add_event(
    trip_id: str,
    payload: DrivingEventCreate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    service = _service(db)

    event = service.add_event(
        user_id=user.id,
        trip_id=trip_id,
        event_type=payload.event_type,
        value=payload.value,
        occurred_at=payload.occurred_at,
        lat=payload.lat,
        lon=payload.lon,
    )

    return APIResponse.ok(
        "events.created",
        data={"event": DrivingEventOut.model_validate(event).model_dump()},
    )


@router.get("/{trip_id}/events", response_model=APIResponse)
def list_trip_events(
    trip_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    service = _service(db)
    events = service.list_for_trip(user_id=user.id, trip_id=trip_id)
    out = DrivingEventListResponse(
        events=[DrivingEventOut.model_validate(e) for e in events]
    )
    return APIResponse.ok("events.list", data=out.model_dump())


@router.get("/events/history", response_model=APIResponse)
def events_history(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
    trip_id: Optional[str] = Query(default=None),
    start: Optional[datetime] = Query(default=None),
    end: Optional[datetime] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    service = _service(db)
    rows, total = service.history(
        user_id=user.id,
        trip_id=trip_id,
        start=start,
        end=end,
        limit=limit,
        offset=offset,
    )

    out = DrivingEventHistoryResponse(
        events=[DrivingEventOut.model_validate(e) for e in rows],
        limit=limit,
        offset=offset,
        total=total,
    )
    return APIResponse.ok("events.history", data=out.model_dump())

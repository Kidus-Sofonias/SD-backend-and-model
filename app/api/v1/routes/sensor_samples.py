# File role: HTTP route layer that maps requests to services/repositories and returns schema-shaped responses.
# Connects to: fastapi, app.api.deps, app.schemas.sensor_samples.
# Key symbols/vars: router, upload_samples, list_samples.
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.schemas.sensor_samples import SensorSampleCountOut, SensorSamplesBatchIn, SensorSampleOut
from app.repositories.sensor_sample_repository import SensorSampleRepository
from app.repositories.trip_repository import SqlTripRepository
from app.services.sensor_sample_service import SensorSampleService

router = APIRouter()


@router.post("/{trip_id}/samples")
def upload_samples(
    trip_id: str,
    payload: SensorSamplesBatchIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    service = SensorSampleService(
        repo=SensorSampleRepository(db),
        trip_repo=SqlTripRepository(db),
    )

    rows = [s.model_dump() for s in payload.samples]
    inserted = service.add_samples(user_id=user.id, trip_id=trip_id, samples=rows)
    return {"inserted": inserted}


@router.get("/{trip_id}/samples", response_model=list[SensorSampleOut])
def list_samples(
    trip_id: str,
    limit: int = Query(500, ge=1, le=5000),
    after_ts: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    service = SensorSampleService(
        repo=SensorSampleRepository(db),     
        trip_repo=SqlTripRepository(db),
    )

    return service.list_samples(user_id=user.id, trip_id=trip_id, limit=limit, after_ts=after_ts)


@router.get("/{trip_id}/samples/count", response_model=SensorSampleCountOut)
def count_samples(
    trip_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    service = SensorSampleService(
        repo=SensorSampleRepository(db),
        trip_repo=SqlTripRepository(db),
    )

    return {
        "trip_id": trip_id,
        "count": service.count_samples(user_id=user.id, trip_id=trip_id),
    }

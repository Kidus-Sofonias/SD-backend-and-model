# File role: HTTP route layer that maps requests to services/repositories and returns schema-shaped responses.
# Connects to: fastapi, app.api.deps, app.schemas.sensor_samples.
# Key symbols/vars: router, upload_samples, list_samples.
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.db.init_db import ensure_sensor_sample_columns
from app.schemas.sensor_samples import SensorSampleCountOut, SensorSamplesBatchIn, SensorSampleOut
from app.repositories.sensor_sample_repository import SensorSampleRepository
from app.repositories.trip_repository import SqlTripRepository
from app.services.sensor_sample_service import SensorSampleService

logger = logging.getLogger("app.routes.sensor_samples")

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

    # Defensively ensure the sensor_samples table has all required columns.
    # This catches cases where a migration was not applied on the production database.
    try:
        inserted = service.add_samples(user_id=user.id, trip_id=trip_id, samples=rows)
    except ProgrammingError as exc:
        logger.warning(
            "Sample insert failed with ProgrammingError — ensuring sensor_samples schema and retrying. Error: %s",
            exc,
        )
        # The _ensure call runs on the raw engine, outside the current transaction.
        # Rollback the failed transaction before trying again.
        db.rollback()
        ensure_sensor_sample_columns()
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

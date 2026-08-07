# File role: HTTP route layer that maps requests to services/repositories and returns schema-shaped responses.
# Connects to: fastapi, app.api.deps, app.schemas.sensor_samples.
# Key symbols/vars: router, upload_samples, list_samples.
from __future__ import annotations

import logging
import traceback
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.core.errors import AppError
from app.core.rate_limit import UPLOAD_RATE_LIMITER
from app.db.init_db import ensure_sensor_sample_columns, ensure_sensor_sample_id_sequence
from app.schemas.sensor_samples import SensorSampleCountOut, SensorSamplesBatchIn, SensorSampleOut
from app.repositories.sensor_sample_repository import SensorSampleRepository
from app.repositories.trip_repository import SqlTripRepository
from app.services.sensor_sample_service import SensorSampleService

logger = logging.getLogger("app.routes.sensor_samples")

router = APIRouter()


@router.post("/{trip_id}/samples")
def upload_samples(
    request: Request,
    trip_id: str,
    payload: SensorSamplesBatchIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    service = SensorSampleService(
        repo=SensorSampleRepository(db),
        trip_repo=SqlTripRepository(db),
    )

    if not UPLOAD_RATE_LIMITER.allow(f"upload:{user.id}"):
        raise HTTPException(status_code=429, detail="Too many requests")

    rows = [s.model_dump() for s in payload.samples]

    try:
        inserted = service.add_samples(user_id=user.id, trip_id=trip_id, samples=rows)
        return {"inserted": inserted}
    except AppError:
        raise
    except Exception as exc:
        is_sqla = isinstance(exc, SQLAlchemyError)
        logger.error(
            "Sample insert failed: type=%s is_db=%s message=%s\n%s",
            type(exc).__name__,
            is_sqla,
            str(exc),
            traceback.format_exc(),
            extra={"trip_id": trip_id, "user_id": user.id, "sample_count": len(rows)},
        )
        db.rollback()

        # For DB-level errors, try to fix missing columns and resync the id
        # sequence (explicit-ID imports desync the Postgres sequence, which
        # causes UniqueViolation on the primary key for every subsequent insert),
        # then retry once.
        if is_sqla:
            try:
                ensure_sensor_sample_columns()
                ensure_sensor_sample_id_sequence()
                inserted = service.add_samples(user_id=user.id, trip_id=trip_id, samples=rows)
                return {"inserted": inserted}
            except Exception as retry_exc:
                logger.error(
                    "Sample insert retry failed: type=%s message=%s\n%s",
                    type(retry_exc).__name__,
                    str(retry_exc),
                    traceback.format_exc(),
                )

        # Return a structured JSON response so the error detail reaches the
        # frontend (the generic http_exception_handler strips it).
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": {
                    "message_key": "error.sample_upload",
                    "details": {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:500],
                    },
                },
                "request_id": getattr(request.state, "request_id", None),
            },
        )


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

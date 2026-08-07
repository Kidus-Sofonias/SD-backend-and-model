# File role: Trip API routes.
# Exposes trip lifecycle endpoints, trip queries, summary, and trip finalization.
# Connects to:
# - app.api.deps
# - app.repositories.trip_repository
# - app.services.trip_processing_service
# - app.schemas.trip
# Key symbols/vars:
# - router
# - active_trip
# - start_trip
# - end_trip
# - list_trips
# - get_trip_details
# - trip_summary
# - finalize_trip

from __future__ import annotations

import logging
import traceback

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.errors import AppError
from app.core.utils import as_utc_timestamp
from app.db.init_db import ensure_driving_event_id_sequence, ensure_sensor_sample_id_sequence
from app.db.models.trip import Trip
from app.db.session import get_db
from app.repositories.sensor_sample_repository import SensorSampleRepository
from app.repositories.trip_repository import SqlTripRepository
from app.schemas.admin import TripRouteOut
from app.schemas.trip import (
    FinalizeTripOut,
    ReprocessTripsOut,
    TripDetailOut,
    TripOut,
    TripReviewDashboardItemOut,
    TripReviewLabelIn,
    TripReviewOut,
)
from app.services.trip_processing_service import TripProcessingService
from app.services.route_snap_service import RouteSnapService

router = APIRouter(prefix="/trips", tags=["trips"])

logger = logging.getLogger("app.routes.trips")


def _run_finalize_with_recovery(
    request: Request,
    service: TripProcessingService,
    *,
    user_id: str,
    trip_id: str,
    delete_raw: bool,
    force_reprocess: bool,
):
    """Run trip finalization, self-healing the Postgres id sequences on
    SQLAlchemy errors (explicit-ID migrations desync sequences, causing
    UniqueViolation on generated DrivingEvent/sample rows), then retry once.
    Returns a structured JSON error on failure so the client sees the real
    error instead of a generic error.internal_server.
    """
    try:
        return service.finalize_trip(
            user_id=user_id,
            trip_id=trip_id,
            delete_raw=delete_raw,
            force_reprocess=force_reprocess,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AppError:
        raise
    except Exception as exc:
        is_sqla = isinstance(exc, SQLAlchemyError)
        logger.error(
            "Trip finalize failed: type=%s is_db=%s message=%s\n%s",
            type(exc).__name__,
            is_sqla,
            str(exc),
            traceback.format_exc(),
            extra={"trip_id": trip_id, "user_id": user_id},
        )
        service.db.rollback()

        reported_exc = exc
        if is_sqla:
            try:
                ensure_sensor_sample_id_sequence()
                ensure_driving_event_id_sequence()
                return service.finalize_trip(
                    user_id=user_id,
                    trip_id=trip_id,
                    delete_raw=delete_raw,
                    force_reprocess=force_reprocess,
                )
            except Exception as retry_exc:
                logger.error(
                    "Trip finalize retry failed: type=%s message=%s\n%s",
                    type(retry_exc).__name__,
                    str(retry_exc),
                    traceback.format_exc(),
                )
                reported_exc = retry_exc

        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": {
                    "message_key": "error.trip_finalize",
                    "details": {
                        "error_type": type(reported_exc).__name__,
                        "error_message": str(reported_exc)[:500],
                    },
                },
                "request_id": getattr(request.state, "request_id", None),
            },
        )


@router.get("/active", response_model=TripOut | None)
def active_trip(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    repo = SqlTripRepository(db)
    return repo.get_active_trip(user_id=user.id)


@router.post("/start", response_model=TripOut)
def start_trip(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    repo = SqlTripRepository(db)

    active = repo.get_active_trip(user_id=user.id)
    if active:
        raise HTTPException(status_code=400, detail="You already have an active trip")

    return repo.create_trip(user_id=user.id)


@router.post("/{trip_id}/end", response_model=TripOut)
def end_trip(
    trip_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    repo = SqlTripRepository(db)
    return repo.end_trip(trip_id=trip_id, user_id=user.id)


@router.get("", response_model=list[TripOut])
def list_trips(
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    # H-5 fix: bounded pagination (backward compatible — defaults return the
    # most recent 200 trips).
    stmt = (
        select(Trip)
        .where(Trip.user_id == user.id)
        .order_by(Trip.started_at.desc())
        .limit(limit)
        .offset(offset)
    )
    trips = db.execute(stmt).scalars().all()
    return trips


@router.get("/review-dashboard", response_model=list[TripReviewDashboardItemOut])
def review_dashboard(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    service = TripProcessingService(db)
    return service.list_review_dashboard(actor=user, limit=limit)


@router.get("/{trip_id}", response_model=TripDetailOut)
def get_trip_details(
    trip_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    service = TripProcessingService(db)
    try:
        return service.get_trip_detail(actor=user, trip_id=trip_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{trip_id}/route", response_model=TripRouteOut)
def get_trip_route(
    trip_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    repo = SqlTripRepository(db)
    trip = repo.get_by_id(trip_id=trip_id, user_id=user.id)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")

    sample_repo = SensorSampleRepository(db)
    samples = sample_repo.list_route_points_by_trip(user_id=user.id, trip_id=trip_id)
    points = [
        {
            "ts": as_utc_timestamp(sample.ts),
            "lat": float(sample.lat),
            "lon": float(sample.lon),
            "speed_mps": sample.speed_mps,
            "accuracy_m": sample.accuracy_m,
        }
        for sample in samples
    ]
    snap_result = RouteSnapService().snap(points)

    return TripRouteOut(
        trip_id=trip.id,
        driver_user_id=user.id,
        point_count=len(points),
        points=points,
        snapped_points=snap_result.snapped_points,
        snapped_source=snap_result.source,
        snapped_status=snap_result.status,
    )


@router.get("/{trip_id}/summary", response_model=TripDetailOut)
def trip_summary(
    trip_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    service = TripProcessingService(db)
    return service.get_trip_detail(actor=user, trip_id=trip_id)


@router.post("/{trip_id}/finalize", response_model=FinalizeTripOut)
def finalize_trip(
    request: Request,
    trip_id: str,
    delete_raw: bool = False,
    force_reprocess: bool = False,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    service = TripProcessingService(db)

    return _run_finalize_with_recovery(
        request,
        service,
        user_id=user.id,
        trip_id=trip_id,
        delete_raw=delete_raw,
        force_reprocess=force_reprocess,
    )


@router.post("/{trip_id}/reprocess", response_model=FinalizeTripOut)
def reprocess_trip(
    request: Request,
    trip_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    service = TripProcessingService(db)

    return _run_finalize_with_recovery(
        request,
        service,
        user_id=user.id,
        trip_id=trip_id,
        delete_raw=False,
        force_reprocess=True,
    )


@router.post("/reprocess", response_model=ReprocessTripsOut)
def reprocess_trips(
    trip_id: str | None = Query(default=None),
    model_version: str | None = Query(default=None),
    feature_version: str | None = Query(default=None),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    service = TripProcessingService(db)
    return service.reprocess_trips(
        user_id=user.id,
        trip_id=trip_id,
        model_version=model_version,
        feature_version=feature_version,
    )


@router.get("/{trip_id}/review", response_model=TripReviewOut)
def trip_review(
    trip_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    service = TripProcessingService(db)
    try:
        return service.get_trip_review(actor=user, trip_id=trip_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{trip_id}/review-label", response_model=TripReviewOut)
def set_trip_review_label(
    trip_id: str,
    payload: TripReviewLabelIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    service = TripProcessingService(db)
    try:
        return service.set_trip_review_label(
            actor=user,
            trip_id=trip_id,
            reviewed_label=payload.reviewed_label,
            reviewed_label_source=payload.reviewed_label_source,
            review_notes=payload.review_notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

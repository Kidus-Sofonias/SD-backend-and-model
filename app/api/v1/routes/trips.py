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

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
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
    TripSummaryOut,
)
from app.services.trip_processing_service import TripProcessingService

router = APIRouter(prefix="/trips", tags=["trips"])


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
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    stmt = select(Trip).where(Trip.user_id == user.id).order_by(Trip.started_at.desc())
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
            "ts": sample.ts,
            "lat": float(sample.lat),
            "lon": float(sample.lon),
            "speed_mps": sample.speed_mps,
            "accuracy_m": sample.accuracy_m,
        }
        for sample in samples
    ]

    return TripRouteOut(
        trip_id=trip.id,
        driver_user_id=user.id,
        point_count=len(points),
        points=points,
    )


@router.get("/{trip_id}/summary", response_model=TripSummaryOut)
def trip_summary(
    trip_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    repo = SqlTripRepository(db)
    trip = repo.get_by_id(trip_id=trip_id, user_id=user.id)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")

    counts: dict[str, int] = {}
    for ev in trip.events:
        counts[ev.event_type] = counts.get(ev.event_type, 0) + 1

    total = len(trip.events)

    penalties = 0
    penalties += counts.get("speeding", 0) * 5
    penalties += counts.get("hard_brake", 0) * 3
    penalties += counts.get("phone_use", 0) * 8

    score = max(0, 100 - penalties)

    return TripSummaryOut(
        trip_id=trip.id,
        status=trip.status,
        started_at=trip.started_at,
        ended_at=trip.ended_at,
        total_events=total,
        counts=counts,
        score=score,
    )


@router.post("/{trip_id}/finalize", response_model=FinalizeTripOut)
def finalize_trip(
    trip_id: str,
    delete_raw: bool = False,
    force_reprocess: bool = False,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    service = TripProcessingService(db)

    try:
        return service.finalize_trip(
            user_id=user.id,
            trip_id=trip_id,
            delete_raw=delete_raw,
            force_reprocess=force_reprocess,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{trip_id}/reprocess", response_model=FinalizeTripOut)
def reprocess_trip(
    trip_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    service = TripProcessingService(db)

    try:
        return service.finalize_trip(
            user_id=user.id,
            trip_id=trip_id,
            delete_raw=False,
            force_reprocess=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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

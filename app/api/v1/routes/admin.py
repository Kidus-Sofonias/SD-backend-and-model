# File role: HTTP route layer for admin-only driver management endpoints.
# Connects to: fastapi, app.api.deps, app.services.admin_service.
# Key symbols/vars: router.
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_users_repo
from app.db.session import get_db
from app.realtime.accident_detector import accident_detector
from app.repositories.user_repository import SqlUserRepository
from app.schemas.admin import AdminUpdateDriverIn, DriverInsightsOut, DriverSummaryOut, TripRouteOut
from app.schemas.trip import TripOut
from app.services.admin_service import AdminService

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/accidents/recent")
def recent_accidents(
    db: Session = Depends(get_db),
    users: SqlUserRepository = Depends(get_users_repo),
    user=Depends(get_current_user),
):
    """Recent high-confidence accident alerts (admin only, Phase 8).

    In-memory replay buffer for admin sessions that reconnect or poll; the
    real-time stream arrives over the fleet WebSocket channel.
    """
    AdminService(db, users)._require_admin(user)
    return {"accidents": accident_detector.recent_accidents()}


@router.get("/trips/{trip_id}/telemetry")
def get_admin_trip_telemetry(
    trip_id: str,
    db: Session = Depends(get_db),
    users: SqlUserRepository = Depends(get_users_repo),
    user=Depends(get_current_user),
):
    """Live telemetry for ANY trip (admin only, Phase 7 fleet -> detail).

    Mirrors the driver-scoped telemetry endpoint so admins opening a live
    trip from the fleet dashboard get continuously updating speed, samples,
    event counts and provisional score.
    """
    service = AdminService(db, users)
    return service.get_live_trip_telemetry(actor=user, trip_id=trip_id)


@router.get("/trips/{trip_id}/samples")
def get_admin_trip_samples(
    trip_id: str,
    limit: int = Query(default=3000, ge=1, le=10000),
    db: Session = Depends(get_db),
    users: SqlUserRepository = Depends(get_users_repo),
    user=Depends(get_current_user),
):
    """Raw sensor timeline for ANY trip (admin only, replay/forensics).

    Feeds the synchronized admin replay: 3D vehicle + speed/accel traces +
    event positions scrubbed on one timeline.
    """
    service = AdminService(db, users)
    return service.get_trip_samples(actor=user, trip_id=trip_id, limit=limit)


@router.get("/trips/live")
def list_live_trips(
    db: Session = Depends(get_db),
    users: SqlUserRepository = Depends(get_users_repo),
    user=Depends(get_current_user),
):
    """Fleet-wide live monitoring snapshot (Phase 7 admin dashboard).

    Active trips across all drivers with latest speed/location, live event
    counters, provisional score and connection status.
    """
    service = AdminService(db, users)
    return service.list_live_trips(actor=user)


@router.get("/trips", response_model=list[TripOut])
def list_all_trips(
    # Admin default is the max page size so the current admin UI (which fetches
    # all trips without paging) keeps working; proper pagination UI lands in
    # Phase 7.
    limit: int = Query(default=1000, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    users: SqlUserRepository = Depends(get_users_repo),
    user=Depends(get_current_user),
):
    service = AdminService(db, users)
    return service.list_all_trips(actor=user, limit=limit, offset=offset)


@router.get("/drivers", response_model=list[DriverSummaryOut])
def list_drivers(
    db: Session = Depends(get_db),
    users: SqlUserRepository = Depends(get_users_repo),
    user=Depends(get_current_user),
):
    service = AdminService(db, users)
    return service.list_drivers(actor=user)


@router.get("/drivers/{driver_id}/trips", response_model=list[TripOut])
def list_driver_trips(
    driver_id: str,
    db: Session = Depends(get_db),
    users: SqlUserRepository = Depends(get_users_repo),
    user=Depends(get_current_user),
):
    service = AdminService(db, users)
    return service.get_driver_trips(actor=user, driver_id=driver_id)


@router.get("/drivers/{driver_id}/insights", response_model=DriverInsightsOut)
def get_driver_insights(
    driver_id: str,
    db: Session = Depends(get_db),
    users: SqlUserRepository = Depends(get_users_repo),
    user=Depends(get_current_user),
):
    service = AdminService(db, users)
    return service.get_driver_insights(actor=user, driver_id=driver_id)


@router.get("/trips/{trip_id}/route", response_model=TripRouteOut)
def get_admin_trip_route(
    trip_id: str,
    db: Session = Depends(get_db),
    users: SqlUserRepository = Depends(get_users_repo),
    user=Depends(get_current_user),
):
    service = AdminService(db, users)
    return service.get_trip_route(actor=user, trip_id=trip_id)


@router.get("/drivers/{driver_id}/trips/{trip_id}/route", response_model=TripRouteOut)
def get_driver_trip_route(
    driver_id: str,
    trip_id: str,
    db: Session = Depends(get_db),
    users: SqlUserRepository = Depends(get_users_repo),
    user=Depends(get_current_user),
):
    service = AdminService(db, users)
    return service.get_driver_trip_route(actor=user, driver_id=driver_id, trip_id=trip_id)


@router.patch("/drivers/{driver_id}", response_model=DriverSummaryOut)
def update_driver(
    driver_id: str,
    payload: AdminUpdateDriverIn,
    db: Session = Depends(get_db),
    users: SqlUserRepository = Depends(get_users_repo),
    user=Depends(get_current_user),
):
    service = AdminService(db, users)
    updated = service.update_driver_credentials(
        actor=user,
        driver_id=driver_id,
        email=payload.email,
        password=payload.password,
    )
    refreshed = users.get_driver_by_id(updated.id)
    if refreshed is None:
        raise RuntimeError("Updated driver could not be reloaded")
    return refreshed


@router.delete("/drivers/{driver_id}", status_code=204)
def delete_driver(
    driver_id: str,
    db: Session = Depends(get_db),
    users: SqlUserRepository = Depends(get_users_repo),
    user=Depends(get_current_user),
):
    service = AdminService(db, users)
    service.delete_driver(actor=user, driver_id=driver_id)

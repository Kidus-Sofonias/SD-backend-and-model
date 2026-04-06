# File role: HTTP route layer for admin-only driver management endpoints.
# Connects to: fastapi, app.api.deps, app.services.admin_service.
# Key symbols/vars: router.
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_users_repo
from app.db.session import get_db
from app.repositories.user_repository import SqlUserRepository
from app.schemas.admin import AdminUpdateDriverIn, DriverSummaryOut, TripRouteOut
from app.schemas.trip import TripOut
from app.services.admin_service import AdminService

router = APIRouter(prefix="/admin", tags=["admin"])


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

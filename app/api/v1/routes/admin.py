# File role: HTTP route layer for admin-only driver management endpoints.
# Connects to: fastapi, app.api.deps, app.services.admin_service.
# Key symbols/vars: router.
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_users_repo
from app.db.session import get_db
from app.realtime.accident_detector import accident_detector
from app.repositories.user_repository import SqlUserRepository
from app.schemas.admin import AdminUpdateDriverIn, DriverInsightsOut, DriverSummaryOut, TripRouteOut
from app.schemas.trip import TripOut
from app.services.admin_service import AdminService
from app.core.security import generate_api_key, hash_api_key
from app.db.models.organization import Organization, PartnerApiKey
from app.db.models.trip import Trip
from app.db.models.user import User
from app.schemas.partner import (
    OrganizationSummaryOut,
    PartnerKeyCreateIn,
    PartnerKeyCreateOut,
    PartnerKeySummaryOut,
)
from app.db.session import commit_with_retry

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/organizations", response_model=list[OrganizationSummaryOut])
def list_organizations(
    search: str | None = Query(default=None, max_length=120),
    active: bool | None = Query(default=None),
    sort: str = Query(default="created", pattern="^(name|drivers|trips|created)$"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    if not user.is_admin:
        from app.core.errors import ForbiddenError
        raise ForbiddenError(message_key="auth.forbidden")

    driver_count = func.count(func.distinct(User.id))
    trip_count = func.count(func.distinct(Trip.id))
    active_key_count = func.count(func.distinct(PartnerApiKey.id)).filter(PartnerApiKey.active.is_(True))
    latest_trip_at = func.max(Trip.ended_at)
    stmt = (
        select(
            Organization,
            driver_count.label("driver_count"),
            trip_count.label("trip_count"),
            active_key_count.label("active_key_count"),
            latest_trip_at.label("latest_trip_at"),
        )
        .outerjoin(User, User.organization_id == Organization.id)
        .outerjoin(Trip, Trip.user_id == User.id)
        .outerjoin(PartnerApiKey, PartnerApiKey.organization_id == Organization.id)
        .where(Organization.active == active if active is not None else True)
        .group_by(Organization.id)
    )
    if search and search.strip():
        term = f"%{search.strip().lower()}%"
        stmt = stmt.where(
            (func.lower(Organization.name).like(term))
            | (func.lower(Organization.slug).like(term))
        )
    sort_column = {
        "name": func.lower(Organization.name),
        "drivers": driver_count,
        "trips": trip_count,
        "created": Organization.created_at,
    }[sort]
    stmt = stmt.order_by(sort_column.asc(), Organization.id.asc()).limit(limit).offset(offset)
    return [
        OrganizationSummaryOut(
            id=organization.id,
            name=organization.name,
            slug=organization.slug,
            active=organization.active,
            created_at=organization.created_at,
            driver_count=int(driver_total or 0),
            trip_count=int(trip_total or 0),
            active_key_count=int(key_total or 0),
            latest_trip_at=latest_trip,
        )
        for organization, driver_total, trip_total, key_total, latest_trip in db.execute(stmt).all()
    ]


@router.post("/partner-keys", response_model=PartnerKeyCreateOut)
def create_partner_key(
    payload: PartnerKeyCreateIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    if not user.is_admin:
        from app.core.errors import ForbiddenError
        raise ForbiddenError(message_key="auth.forbidden")
    organization = db.execute(
        select(Organization).where(Organization.slug == payload.organization_slug)
    ).scalar_one_or_none()
    if organization is None:
        organization = Organization(name=payload.organization_name, slug=payload.organization_slug)
        db.add(organization)
        db.flush()
    api_key = generate_api_key()
    record = PartnerApiKey(
        organization_id=organization.id,
        name=payload.key_name,
        key_prefix=api_key[:16],
        key_hash=hash_api_key(api_key),
    )
    db.add(record)
    commit_with_retry(db)
    db.refresh(record)
    return PartnerKeyCreateOut(
        organization_id=organization.id,
        api_key_id=record.id,
        api_key=api_key,
    )


@router.delete("/partner-keys/{api_key_id}", status_code=204)
def revoke_partner_key(
    api_key_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    if not user.is_admin:
        from app.core.errors import ForbiddenError
        raise ForbiddenError(message_key="auth.forbidden")
    key = db.get(PartnerApiKey, api_key_id)
    if key is not None:
        key.active = False
        commit_with_retry(db)


@router.get("/partner-keys", response_model=list[PartnerKeySummaryOut])
def list_partner_keys(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    if not user.is_admin:
        from app.core.errors import ForbiddenError
        raise ForbiddenError(message_key="auth.forbidden")
    return db.execute(
        select(PartnerApiKey).order_by(PartnerApiKey.created_at.desc())
    ).scalars().all()


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

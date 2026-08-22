from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.api.partner_auth import PartnerContext, get_partner_context
from app.db.models.trip import Trip
from app.db.models.user import User
from app.db.session import get_db
from app.core.security import hash_password
from app.schemas.partner import (
    PartnerDriverOut,
    PartnerDriverStatsOut,
    PartnerIngestOut,
    PartnerTripBatchIn,
    PartnerTripOut,
)

router = APIRouter(prefix="/partner", tags=["partner"])


@router.post("/ingest/trips", response_model=PartnerIngestOut)
def ingest_partner_trips(
    payload: PartnerTripBatchIn,
    db: Session = Depends(get_db),
    partner: PartnerContext = Depends(get_partner_context),
):
    """Idempotently ingest compact trip results from a company's system.

    Driver credentials and raw telemetry stay with the company. A local
    shadow user is created only to preserve the existing trip ownership model.
    """
    driver_ids = {trip.external_driver_id for trip in payload.trips}
    users = {
        user.external_driver_id: user
        for user in db.execute(
            select(User).where(
                User.organization_id == partner.organization_id,
                User.external_driver_id.in_(driver_ids),
            )
        ).scalars().all()
    }
    created = 0
    updated = 0
    for item in payload.trips:
        user = users.get(item.external_driver_id)
        if user is None:
            user = User(
                id=str(uuid.uuid4()),
                email=f"partner-{uuid.uuid4()}@internal.invalid",
                password_hash=hash_password(uuid.uuid4().hex),
                role="driver",
                organization_id=partner.organization_id,
                external_driver_id=item.external_driver_id,
            )
            db.add(user)
            db.flush()
            users[item.external_driver_id] = user

        trip = db.execute(
            select(Trip).where(Trip.user_id == user.id, Trip.source_trip_id == item.source_trip_id)
        ).scalar_one_or_none()
        if trip is None:
            trip = Trip(user_id=user.id, source_trip_id=item.source_trip_id)
            db.add(trip)
            created += 1
        else:
            updated += 1
        trip.started_at = item.started_at
        trip.ended_at = item.ended_at
        trip.status = item.status
        trip.score = item.score
        trip.risk_probability = item.risk_probability
        trip.risk_level = item.risk_level
        trip.confidence = item.confidence
        trip.feature_version = item.feature_version
        trip.model_version = item.model_version
        trip.processed_at = item.processed_at
    db.commit()
    return PartnerIngestOut(received=len(payload.trips), created=created, updated=updated)


@router.get("/drivers", response_model=list[PartnerDriverOut])
def list_partner_drivers(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    partner: PartnerContext = Depends(get_partner_context),
):
    stmt = (
        select(
            User.external_driver_id,
            User.id,
            func.count(Trip.id).label("trip_count"),
            func.max(func.coalesce(Trip.ended_at, Trip.started_at)).label("latest_trip_at"),
        )
        .outerjoin(Trip, Trip.user_id == User.id)
        .where(User.organization_id == partner.organization_id, User.role == "driver")
        .group_by(User.id, User.external_driver_id)
        .order_by(User.id)
        .limit(limit)
        .offset(offset)
    )
    return [
        PartnerDriverOut(
            external_driver_id=row.external_driver_id,
            driver_id=row.id,
            trip_count=int(row.trip_count or 0),
            latest_trip_at=row.latest_trip_at,
        )
        for row in db.execute(stmt).all()
    ]


@router.get("/drivers/{driver_id}/trips", response_model=list[PartnerTripOut])
def list_partner_driver_trips(
    driver_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    partner: PartnerContext = Depends(get_partner_context),
):
    stmt = (
        select(Trip, User.external_driver_id)
        .join(User, User.id == Trip.user_id)
        .where(
            User.organization_id == partner.organization_id,
            User.role == "driver",
            (User.id == driver_id) | (User.external_driver_id == driver_id),
        )
        .order_by(Trip.started_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return [
        PartnerTripOut(
            id=trip.id,
            source_trip_id=trip.source_trip_id,
            external_driver_id=external_id,
            started_at=trip.started_at,
            ended_at=trip.ended_at,
            status=trip.status,
            score=trip.score,
            risk_probability=trip.risk_probability,
            risk_level=trip.risk_level,
            confidence=trip.confidence,
            feature_version=trip.feature_version,
            model_version=trip.model_version,
            processed_at=trip.processed_at,
        )
        for trip, external_id in db.execute(stmt).all()
    ]


@router.get("/drivers/{driver_id}/stats", response_model=PartnerDriverStatsOut)
def partner_driver_stats(
    driver_id: str,
    db: Session = Depends(get_db),
    partner: PartnerContext = Depends(get_partner_context),
):
    user = db.execute(
        select(User).where(
            User.organization_id == partner.organization_id,
            User.role == "driver",
            (User.id == driver_id) | (User.external_driver_id == driver_id),
        )
    ).scalar_one_or_none()
    if user is None:
        return PartnerDriverStatsOut(external_driver_id=driver_id)
    row = db.execute(
        select(
            func.count(Trip.id),
            func.count(Trip.score),
            func.avg(Trip.score),
            func.sum(case((Trip.risk_level == "high", 1), else_=0)),
        ).where(Trip.user_id == user.id)
    ).one()
    return PartnerDriverStatsOut(
        external_driver_id=user.external_driver_id,
        trip_count=int(row[0] or 0),
        scored_trip_count=int(row[1] or 0),
        average_score=float(row[2]) if row[2] is not None else None,
        high_risk_trip_count=int(row[3] or 0),
    )
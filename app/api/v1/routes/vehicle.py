# File role: Vehicle profile API routes (Phase 3, hackathon).
# Lets a driver set/update the vehicle used for sensor interpretation, and
# lets the client discover whether onboarding is still needed.
# Connects to: app.api.deps, app.db.models.vehicle_profile, app.ml.vehicle_profiles.
# Key symbols/vars: router, get_vehicle_profile, upsert_vehicle_profile.
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models.vehicle_profile import VehicleProfile
from app.db.session import commit_with_retry, get_db
from app.ml.vehicle_profiles import profile_thresholds
from app.schemas.vehicle import VehicleProfileIn, VehicleProfileOut

router = APIRouter(prefix="/users/me/vehicle", tags=["vehicle"])


def _to_out(profile: VehicleProfile) -> VehicleProfileOut:
    out = VehicleProfileOut.model_validate(profile)
    out.thresholds = profile_thresholds(profile)
    return out


@router.get("", response_model=VehicleProfileOut)
def get_vehicle_profile(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    profile = db.execute(
        select(VehicleProfile).where(VehicleProfile.user_id == user.id)
    ).scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="No vehicle profile set")
    return _to_out(profile)


@router.put("", response_model=VehicleProfileOut)
def upsert_vehicle_profile(
    payload: VehicleProfileIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    profile = db.execute(
        select(VehicleProfile).where(VehicleProfile.user_id == user.id)
    ).scalar_one_or_none()

    if profile is None:
        profile = VehicleProfile(user_id=user.id, category=payload.category)
        db.add(profile)

    profile.category = payload.category
    profile.make_model = payload.make_model
    profile.size_class = payload.size_class
    profile.drive_type = payload.drive_type
    profile.transmission = payload.transmission
    profile.phone_placement = payload.phone_placement
    profile.load_level = payload.load_level
    profile.road_context = payload.road_context
    # Persist the curated mass (category default refined by size class and
    # typical load) so the client and 3D simulation read one source of truth.
    from app.ml.vehicle_profiles import resolve_mass_kg

    profile.mass_kg = resolve_mass_kg(
        payload.category,
        size_class=payload.size_class,
        load_level=payload.load_level,
        mass_kg=payload.mass_kg,
    )

    db.add(profile)
    commit_with_retry(db)
    db.refresh(profile)
    return _to_out(profile)


@router.get("/options", response_model=list[dict[str, str]])
def vehicle_category_options():
    from app.ml.vehicle_profiles import category_options

    return category_options()

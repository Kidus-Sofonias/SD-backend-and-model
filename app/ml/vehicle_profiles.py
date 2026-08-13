"""Vehicle-aware parameter layer (Phase 3, hackathon).

Maps a driver's vehicle profile to physical expectations that tune how sensor
signals are interpreted instead of assuming one universal car:

    Vehicle Profile -> physical parameters -> sensor interpretation
                     -> event thresholds -> 3D simulation (Phase 4)

Scaling rationale (documented, not arbitrary):
- Capability scaling: peak achievable longitudinal accel/decel and lateral
  accel fall with mass. A truck physically cannot sustain 3.2 m/s^2 of
  acceleration, so the SAME reading is more severe in a truck than a sedan.
  Longitudinal thresholds therefore scale with (m_ref / m)^0.25, clamped to
  [0.5, 1.2] — heavier vehicles get lower (more sensitive) thresholds.
- Turning: lateral capability scales slightly slower, (m_ref / m)^0.2, clamped
  to [0.6, 1.1] (a loaded bus cornering at 3 m/s^2 is aggressive; a sports
  sedan at 4.4 m/s^2 is expected).
- Suspension response: heavier vehicles (trucks, buses) have more suspension
  travel and bounce more on ordinary roads, so the rough-road / unstable-motion
  jerk floor scales UP with mass as (m / m_ref)^0.15, clamped to [0.85, 1.6].
- Reference mass = 1400 kg (mid-size sedan) — the current universal default.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from app.ml.config import FeatureConfigV2

REFERENCE_MASS_KG = 1400.0


@dataclass(frozen=True)
class VehicleCategoryParams:
    """Physical parameters for a vehicle category."""

    key: str
    label: str
    mass_kg: float


# Category -> reference physical parameters. mass_kg is the curated default for
# the category; size_class can refine it (compact -15%, large +20%).
VEHICLE_CATEGORIES: dict[str, VehicleCategoryParams] = {
    "sedan": VehicleCategoryParams(key="sedan", label="Sedan", mass_kg=1400.0),
    "suv": VehicleCategoryParams(key="suv", label="SUV", mass_kg=2100.0),
    "pickup": VehicleCategoryParams(key="pickup", label="Pickup", mass_kg=2300.0),
    "van": VehicleCategoryParams(key="van", label="Van", mass_kg=2600.0),
    "bus": VehicleCategoryParams(key="bus", label="Bus", mass_kg=11000.0),
    "heavy_truck": VehicleCategoryParams(key="heavy_truck", label="Heavy truck", mass_kg=18000.0),
    "tractor_trailer": VehicleCategoryParams(key="tractor_trailer", label="Tractor-trailer", mass_kg=36000.0),
    "other": VehicleCategoryParams(key="other", label="Other", mass_kg=1600.0),
}

SIZE_CLASS_MULTIPLIERS: dict[str, float] = {
    "compact": 0.85,
    "midsize": 1.0,
    "large": 1.2,
}


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def resolve_mass_kg(category: str, *, size_class: str | None = None, mass_kg: float | None = None) -> float:
    """Curated vehicle mass in kg: explicit override > category default
    refined by size class."""
    if mass_kg is not None and mass_kg > 0:
        return float(mass_kg)
    params = VEHICLE_CATEGORIES.get(category, VEHICLE_CATEGORIES["other"])
    multiplier = SIZE_CLASS_MULTIPLIERS.get(size_class or "", 1.0)
    return float(params.mass_kg * multiplier)


def longitudinal_scale(mass_kg: float) -> float:
    """Multiplier for longitudinal (brake/accel) thresholds, [0.5, 1.2]."""
    return _clamp((REFERENCE_MASS_KG / max(float(mass_kg), 1.0)) ** 0.25, 0.5, 1.2)


def turn_scale(mass_kg: float) -> float:
    """Multiplier for the aggressive-turn lateral threshold, [0.6, 1.1]."""
    return _clamp((REFERENCE_MASS_KG / max(float(mass_kg), 1.0)) ** 0.2, 0.6, 1.1)


def unstable_jerk_scale(mass_kg: float) -> float:
    """Multiplier for the rough-road jerk floor, [0.85, 1.6]."""
    return _clamp((max(float(mass_kg), 1.0) / REFERENCE_MASS_KG) ** 0.15, 0.85, 1.6)


def profile_thresholds(profile: Any) -> dict[str, float]:
    """Derived detection thresholds for a VehicleProfile (or category string)."""
    category = profile.category if hasattr(profile, "category") else str(profile)
    mass_kg = (
        resolve_mass_kg(category, size_class=profile.size_class, mass_kg=profile.mass_kg)
        if hasattr(profile, "size_class")
        else resolve_mass_kg(category)
    )
    base = FeatureConfigV2()
    scale = longitudinal_scale(mass_kg)
    return {
        "mass_kg": mass_kg,
        "harsh_brake_dv": round(base.harsh_brake_dv * scale, 3),
        "harsh_accel_dv": round(base.harsh_accel_dv * scale, 3),
        "emergency_brake_dv": round(base.emergency_brake_dv * scale, 3),
        "aggressive_turn_threshold": round(base.aggressive_turn_threshold * turn_scale(mass_kg), 3),
        "unstable_motion_jerk_threshold": round(
            base.unstable_motion_jerk_threshold * unstable_jerk_scale(mass_kg), 3
        ),
    }


def config_for_profile(cfg: FeatureConfigV2, profile: Any) -> FeatureConfigV2:
    """FeatureConfigV2 tuned for the vehicle's physical expectations.

    Returns the SAME config object when the profile is missing so the
    universal-car default applies unchanged.
    """
    if profile is None or not getattr(profile, "category", None):
        return cfg
    thresholds = profile_thresholds(profile)
    return dataclasses.replace(
        cfg,
        harsh_brake_dv=thresholds["harsh_brake_dv"],
        harsh_accel_dv=thresholds["harsh_accel_dv"],
        emergency_brake_dv=thresholds["emergency_brake_dv"],
        aggressive_turn_threshold=thresholds["aggressive_turn_threshold"],
        unstable_motion_jerk_threshold=thresholds["unstable_motion_jerk_threshold"],
    )


def category_options() -> list[dict[str, str]]:
    """Public category options (for the mobile onboarding picker)."""
    return [
        {"key": params.key, "label": params.label, "mass_kg": str(params.mass_kg)}
        for params in VEHICLE_CATEGORIES.values()
    ]

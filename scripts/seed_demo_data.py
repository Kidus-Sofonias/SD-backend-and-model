#!/usr/bin/env python3
"""
Seed the database with rich demo data for events and demonstrations.

Creates fake driver users, generates trips with realistic sensor data
(hard-brakes, aggressive turns, smooth cruising) across Ethiopian routes,
finalizes every trip through the ML pipeline, and applies review labels
to showcase the full admin workflow.

USAGE
-----
# Use the existing configured database (Supabase / SQLite):
    python -m scripts.seed_demo_data

# Use a local SQLite file (no network needed, great for events):
    python -m scripts.seed_demo_data --local-db demo.db

# Full customisation:
    python -m scripts.seed_demo_data --drivers 5 --trips-per-driver 4 --local-db demo.db
"""

from __future__ import annotations

import argparse
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import func

from app.core.security import hash_password
from app.db.base import Base
from app.db.models.user import User
from app.db.models.trip import Trip
from app.db.models.driving_event import DrivingEvent
from app.db.models.sensor_sample import SensorSample
from app.ml.config import FeatureConfigV2
from app.repositories.user_repository import UserRecord
from app.services.trip_processing_service import TripProcessingService

# ---------------------------------------------------------------------------
# Reuse the excellent sensor-data generators from the synthetic trip script
# ---------------------------------------------------------------------------
from scripts.generate_synthetic_trips import (
    generate_safe_profile,
    generate_moderate_profile,
    generate_risky_profile,
    create_trip_with_samples,
    generate_realistic_trip_times,
    pick_trip_duration,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEMO_DRIVERS: list[dict[str, str]] = [
    {"name": "Abebe Kebede",   "email": "abebe@gmail.com",   "profile": "safe"},
    {"name": "Bruktawit Alemu", "email": "bruktawit@gmail.com", "profile": "moderate"},
    {"name": "Chala Tadesse",   "email": "chala@gmail.com",    "profile": "risky"},
    {"name": "Desta Wolde",     "email": "desta@gmail.com",    "profile": "safe"},
    {"name": "Ephrem Girma",    "email": "ephrem@gmail.com",   "profile": "moderate"},
]

DEFAULT_PASSWORD = "demo1234"
# Variable trip durations: each trip gets 400–7000 samples (2–35 min)
# Fixed samples_per_trip no longer used; call pick_trip_duration() per trip.
DT_SECONDS = 0.3


def _pick_trip_types(profile: str, count: int) -> list[str]:
    """Return a list of trip-type labels for a driver profile."""
    pool: dict[str, list[str]] = {
        "safe":     ["safe"] * 3 + ["moderate"] * 1 + ["risky"] * 1,
        "moderate": ["moderate"] * 3 + ["safe"] * 1 + ["risky"] * 1,
        "risky":    ["risky"] * 3 + ["moderate"] * 1 + ["safe"] * 1,
    }
    choices = pool.get(profile, ["moderate"])
    random.shuffle(choices)
    return (choices * (count // len(choices) + 1))[:count]


def _make_session_maker(local_db: str | None) -> sessionmaker:
    """Create a sessionmaker bound to the configured or local database."""
    if local_db:
        db_path = Path(local_db)
        db_url = f"sqlite:///{db_path.resolve()}"
        engine = create_engine(db_url, connect_args={"check_same_thread": False})
        # Create all tables
        Base.metadata.create_all(bind=engine)
        # Ensure altitude_m column exists for the sensor_samples table
        try:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE sensor_samples ADD COLUMN altitude_m FLOAT"))
        except Exception:
            pass  # Column already exists
        return sessionmaker(bind=engine, class_=Session)

    # Use the app's configured database
    from app.db.session import SessionLocal
    return SessionLocal


# ============================================================================
# CORE SEEDING LOGIC
# ============================================================================

def seed_demo_data(
    *,
    driver_count: int,
    trips_per_driver: int | list[int],
    password: str,
    admin_password: str | None,
    local_db: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Seed demo data and return a summary dictionary.

    Parameters
    ----------
    local_db : str or None
        If set, use this SQLite file path instead of the configured database.
    """
    random.seed(seed)

    # Suppress ML model-loading warnings (we're in demo / dev mode)
    import logging
    logging.getLogger("app.ml.inference").setLevel(logging.ERROR)

    SessionLocal = _make_session_maker(local_db)

    # ------------------------------------------------------------------
    # Use ONE session for the entire lifecycle to avoid connection issues.
    # We commit before calling finalize_trip (which may rollback internally).
    # ------------------------------------------------------------------
    db = SessionLocal()
    finalized: list[dict[str, Any]] = []
    created_users: list[User] = []
    trip_infos: list[dict[str, Any]] = []
    review_labels: list[dict[str, Any]] = []
    admin_created = False
    admin_email = "admin@sdb.com"

    try:
        # ==============================================================
        # 1.  Ensure an admin user exists
        # ==============================================================
        admin = db.execute(
            select(User).where(User.role == "admin")
        ).scalar_one_or_none()

        if admin is None:
            pwd_hash = hash_password(admin_password or password)
            admin = User(
                id=str(uuid.uuid4()),
                email=admin_email,
                password_hash=pwd_hash,
                role="admin",
            )
            db.add(admin)
            db.commit()
            db.refresh(admin)
            admin_created = True
        else:
            admin_email = admin.email

        admin_record = UserRecord(
            id=admin.id,
            email=admin.email,
            password_hash=admin.password_hash,
            role=admin.role,
        )

        # ==============================================================
        # 2.  Create demo driver users
        # ==============================================================
        driver_configs = DEMO_DRIVERS[:driver_count]

        for cfg in driver_configs:
            existing = db.execute(
                select(User).where(User.email == cfg["email"])
            ).scalar_one_or_none()
            if existing:
                created_users.append(existing)
                continue
            user = User(
                id=str(uuid.uuid4()),
                email=cfg["email"],
                password_hash=hash_password(password),
                role="driver",
            )
            db.add(user)
            db.flush()
            db.refresh(user)
            created_users.append(user)

        db.commit()

        # Re-fetch users (ORM objects may be expired after commit)
        user_map: dict[str, User] = {}
        for u in created_users:
            refreshed = db.execute(
                select(User).where(User.id == u.id)
            ).scalar_one()
            user_map[refreshed.email] = refreshed

        # ==============================================================
        # 3.  Generate trips + sensor samples for each driver
        # ==============================================================
        now = datetime.now(timezone.utc)
        for idx, cfg in enumerate(driver_configs):
            user = user_map[cfg["email"]]
            driver_trip_count = trips_per_driver[idx] if isinstance(trips_per_driver, list) else trips_per_driver
            trip_types = _pick_trip_types(cfg["profile"], driver_trip_count)

            # Realistic start times for this driver (each driver has their own pattern)
            driver_seed = seed + idx * 100
            trip_times = generate_realistic_trip_times(
                driver_trip_count,
                now=now,
                max_days_back=28,
                daily_min_trips=1,
                daily_max_trips=3,
                seed=driver_seed,
            )

            for tidx, trip_type in enumerate(trip_types):
                started_at = trip_times[tidx] if tidx < len(trip_times) else (
                    now - timedelta(days=tidx + 1, minutes=random.randint(0, 120))
                )

                n_samp = pick_trip_duration()

                if trip_type == "safe":
                    rows = generate_safe_profile(n_samp, DT_SECONDS)
                elif trip_type == "moderate":
                    rows = generate_moderate_profile(n_samp, DT_SECONDS)
                else:
                    rows = generate_risky_profile(n_samp, DT_SECONDS)

                trip_id = create_trip_with_samples(
                    db, user.id, rows, started_at, DT_SECONDS
                )
                trip_infos.append({
                    "user_id": user.id,
                    "trip_id": trip_id,
                    "trip_type": trip_type,
                    "started_at": started_at.isoformat(),
                })

        db.commit()
        print(f"  [OK] Created {len(trip_infos)} trips with sensor data")

        # ==============================================================
        # 4.  Finalize every trip via the ML pipeline
        # ==============================================================
        for info in trip_infos:
            # Commit any pending work so finalize_trip's internal rollback
            # doesn't discard anything.
            db.commit()

            service = TripProcessingService(db, cfg=FeatureConfigV2())
            try:
                result = service.finalize_trip(
                    user_id=info["user_id"],
                    trip_id=info["trip_id"],
                    delete_raw=False,
                    force_reprocess=False,
                )
                finalized.append({**info, "result": result})
                print(f"  [OK] Finalized trip {info['trip_id'][:8]}... "
                      f"score {result.get('score', 'N/A')}")
            except Exception as exc:
                print(f"  [FAIL] Finalize trip {info['trip_id'][:8]}...: {exc}")
                finalized.append({**info, "result": None})

        success_count = sum(1 for f in finalized if f["result"] is not None)
        print(f"  [DONE] Finalized {success_count}/{len(finalized)} trips")

        # ==============================================================
        # 5.  Apply DEMO review labels (admin reviews ~70% of trips)
        #
        # These showcase the admin review workflow. The label is derived from
        # the trip score, so it carries NO independent ground-truth signal:
        # it is stored as ``demo_review`` so dataset builders and the review
        # analysis treat it as a demo tier, never as real human review.
        # Real human reviews come from the admin UI (source "human_review").
        # ==============================================================
        reviewable = [f for f in finalized if f["result"] is not None]
        random.shuffle(reviewable)
        if reviewable:
            to_review_count = max(1, int(len(reviewable) * 0.7))
            to_review = reviewable[:to_review_count]
        else:
            to_review = []

        for item in to_review:
            db.commit()  # ensure clean state before service writes

            score = item["result"].get("score") or 50
            # Smart label: score >= 80 → safe (0), else risky (1)
            label = 0 if score >= 80 else 1
            notes = random.choice([
                "Smooth driving, no issues",
                "Minor speeding but overall okay",
                "Aggressive driving detected",
                "Hard braking event noted",
                "Good driver, safe habits",
                "Needs improvement on turns",
                None,
                None,
            ])

            service = TripProcessingService(db, cfg=FeatureConfigV2())
            try:
                review_result = service.set_trip_review_label(
                    actor=admin_record,
                    trip_id=item["trip_id"],
                    reviewed_label=label,
                    reviewed_label_source="demo_review",
                    review_notes=notes,
                )
                review_labels.append({
                    "trip_id": item["trip_id"],
                    "label": label,
                    "notes": notes,
                })
            except Exception as exc:
                print(f"  [FAIL] Review trip {item['trip_id'][:8]}...: {exc}")

        print(f"  [DONE] Reviewed {len(review_labels)} trips")

        # ==============================================================
        # 6.  Gather summary statistics
        # ==============================================================
        all_trips = db.execute(
            select(Trip).where(Trip.status == "completed")
        ).scalars().all()

        scores = [t.score for t in all_trips if t.score is not None]
        reviewed = [t for t in all_trips if t.reviewed_label is not None]

        total_events = (
            db.execute(
                select(func.count(DrivingEvent.id)).where(
                    DrivingEvent.trip_id.in_([t.id for t in all_trips])
                )
            ).scalar()
            or 0
        ) if all_trips else 0

        return {
            "admin_created": admin_created,
            "admin_email": admin_email,
            "drivers_created": len(created_users),
            "drivers": [
                {"email": cfg["email"], "profile": cfg["profile"], "name": cfg["name"]}
                for cfg in driver_configs
            ],
            "default_password": password,
            "database_type": "SQLite (local)" if local_db else "configured database",
            "database_path": str(Path(local_db).resolve()) if local_db else "N/A",
            "trips_created": len(trip_infos),
            "trips_finalized": success_count,
            "trips_reviewed": len(review_labels),
            "scores": {
                "min": min(scores) if scores else None,
                "max": max(scores) if scores else None,
                "avg": round(sum(scores) / len(scores), 1) if scores else None,
                "distribution": {
                    "high (>=80)": sum(1 for s in scores if s >= 80),
                    "medium (55-79)": sum(1 for s in scores if 55 <= s < 80),
                    "low (<55)": sum(1 for s in scores if s < 55),
                },
            },
            "reviewed_count": len(reviewed),
            "total_driving_events": total_events,
        }

    finally:
        db.close()


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed demo data for events — fake drivers, trips, finalization & reviews"
    )
    parser.add_argument(
        "--drivers",
        type=int,
        default=3,
        help="Number of demo drivers (1-5, default 3)",
    )
    parser.add_argument(
        "--trips-per-driver",
        type=int,
        default=4,
        help="Trips per driver (default 4) — used when --trips-per-driver-list is not set",
    )
    parser.add_argument(
        "--trips-per-driver-list",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated trip counts per driver (e.g. '10,8,12,8,10' for 5 drivers). Overrides --trips-per-driver.",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=DEFAULT_PASSWORD,
        help=f"Password for all demo accounts (default {DEFAULT_PASSWORD})",
    )
    parser.add_argument(
        "--admin-password",
        type=str,
        default=None,
        help="Admin password (used if admin doesn't exist yet)",
    )
    parser.add_argument(
        "--local-db",
        type=str,
        default=None,
        metavar="FILE",
        help="Use a local SQLite file (e.g. demo.db) instead of the configured database. "
             "Great for events with no network.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default 42)",
    )
    args = parser.parse_args()

    drivers = min(args.drivers, 5)
    total_trips = drivers * args.trips_per_driver

    print("=" * 60)
    print("  DrivePulse  –  Demo Data Seeder")
    print("=" * 60)
    print()
    print(f"  Drivers:       {drivers}")
    if args.trips_per_driver_list:
        parsed_counts = [int(x.strip()) for x in args.trips_per_driver_list.split(",")]
        print(f"  Trips/driver:  {', '.join(str(c) for c in parsed_counts[:drivers])}")
        print(f"  Total trips:   {sum(parsed_counts[:drivers])}")
    else:
        print(f"  Trips/driver:  {args.trips_per_driver}")
        print(f"  Total trips:   {total_trips}")

    if args.local_db:
        db_path = Path(args.local_db)
        if db_path.exists():
            db_path.unlink()
        print(f"  Database:      {db_path.resolve()}  (local SQLite — offline)")
    else:
        from app.core.config import settings
        print(f"  Database:      {settings.database_url}")
    print()

    # Parse per-driver trip counts if provided
    trips_per_driver = args.trips_per_driver
    if args.trips_per_driver_list:
        parsed = [int(x.strip()) for x in args.trips_per_driver_list.split(",")]
        if len(parsed) != drivers:
            print(f"  [WARN] --trips-per-driver-list has {len(parsed)} values but --drivers={drivers}. Using --trips-per-driver ({args.trips_per_driver}) instead.")
        else:
            trips_per_driver = parsed

    summary = seed_demo_data(
        driver_count=drivers,
        trips_per_driver=trips_per_driver,
        password=args.password,
        admin_password=args.admin_password,
        local_db=args.local_db,
        seed=args.seed,
    )

    print()
    print("-" * 60)
    print("  [SUCCESS]  SEEDING COMPLETE")
    print("-" * 60)
    print()
    print(f"  Database:      {summary['database_type']}")
    if summary["database_path"] != "N/A":
        print(f"                 File: {summary['database_path']}")
    print(f"  Admin:         {summary['admin_email']}")
    if summary["admin_created"]:
        print(f"                 Password: {args.admin_password or args.password}")
    else:
        print(f"                 (already existed — use existing password)")
    for d in summary["drivers"]:
        print(f"  Driver:        {d['email']}  ({d['name']} — {d['profile']})")
    print(f"                 Password: {args.password}")
    print()
    print(f"  Trips created:   {summary['trips_created']}")
    print(f"  Trips finalized: {summary['trips_finalized']}")
    print(f"  Trips reviewed:  {summary['trips_reviewed']}")
    print()
    s = summary["scores"]
    print(f"  Score range:     {s['min']} – {s['max']}  (avg {s['avg']})")
    print(f"  Score dist:      {s['distribution']}")
    print(f"  Driving events:  {summary['total_driving_events']}")
    print()

    if args.local_db:
        print(f"  *** Demo data ready! Copy '{args.local_db}' to the backend root and")
        print(f"     set DATABASE_URL=sqlite:///{args.local_db} in your .env to use it.")
    else:
        print(f"  *** Demo data is live in your configured database!")
    print()


if __name__ == "__main__":
    main()

# File role: Backfill/reprocessing script for trips whose stored ML outputs need recomputation.
# Reloads samples for existing trips, reruns the ML pipeline with current code, and updates stored trip outputs.
# Connects to:
# - app.db.session
# - app.db.models.trip
# - app.services.trip_processing_service
# Key symbols/vars:
# - main

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import select

from app.db.session import SessionLocal
from app.db.models.trip import Trip
from app.services.trip_processing_service import TripProcessingService


def main():
    parser = argparse.ArgumentParser(description="Reprocess trips with current ML pipeline")
    parser.add_argument("--trip-id", type=str, default=None, help="Reprocess only one trip")
    parser.add_argument("--user-id", type=str, default=None, help="Restrict to one user")
    parser.add_argument("--only-processed", action="store_true", help="Only reprocess trips that already have processed_at")
    parser.add_argument("--model-version", type=str, default=None, help="Reprocess trips with this stored model version")
    parser.add_argument("--feature-version", type=str, default=None, help="Reprocess trips with this stored feature version")
    args = parser.parse_args()

    db = SessionLocal()
    service = TripProcessingService(db)

    try:
        stmt = select(Trip)

        if args.trip_id:
            stmt = stmt.where(Trip.id == args.trip_id)

        if args.user_id:
            stmt = stmt.where(Trip.user_id == args.user_id)

        if args.only_processed:
            stmt = stmt.where(Trip.processed_at.is_not(None))
        if args.model_version:
            stmt = stmt.where(Trip.model_version == args.model_version)
        if args.feature_version:
            stmt = stmt.where(Trip.feature_version == args.feature_version)

        trips = db.execute(stmt.order_by(Trip.started_at.desc())).scalars().all()

        print(f"Found {len(trips)} trips to reprocess")

        updated = 0
        failed = 0

        for trip in trips:
            try:
                result = service.finalize_trip(
                    user_id=trip.user_id,
                    trip_id=trip.id,
                    delete_raw=False,
                    force_reprocess=True,
                )

                updated += 1
                print(
                    f"Reprocessed trip {trip.id}: "
                    f"score={result['score']}, "
                    f"model={result['model_version']}, "
                    f"feature_version={result['feature_version']}"
                )
            except Exception as exc:
                failed += 1
                db.rollback()
                print(f"Failed trip {trip.id}: {exc}")

        print(f"Done. Updated={updated}, Failed={failed}")

    finally:
        db.close()


if __name__ == "__main__":
    main()

#How to use it

# To reprocess the one stale trip:

# python -m scripts.reprocess_finalized_trips --trip-id 4533f9d4-2a9c-4ee3-8699-b32cea43b66b

# To reprocess all previously finalized trips:

# python -m scripts.reprocess_finalized_trips --only-processed

# To reprocess only one user’s trips:

# python -m scripts.reprocess_finalized_trips --user-id 082e95b8-c206-49f4-a968-72021bfd

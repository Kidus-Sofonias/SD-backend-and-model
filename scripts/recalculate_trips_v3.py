# File role: Phase 10 — recalculate historical trips with the v3 scoring
# methodology (noise-robust detection + per-event impact weighting).
#
# Safety behavior (mirrors recalculate_trips_phase4.py):
# - Trips whose raw samples were deleted (raw_deleted=True) are skipped.
# - Trips already processed under scoring_version "v3" are skipped by default
#   (pass --include-v3 to force). Resumable and idempotent.
# - --dry-run computes new scores WITHOUT writing.
#
# Usage:
#   python -m scripts.recalculate_trips_v3 --dry-run
#   python -m scripts.recalculate_trips_v3
#   python -m scripts.recalculate_trips_v3 --trip-id <id>
#
# Report: artifacts/reports/phase10_v3_recalc_report.json

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import select

from app.db.models.sensor_sample import SensorSample
from app.db.models.trip import Trip
from app.db.session import SessionLocal
from app.ml.config import FeatureConfigV2
from app.ml.pipeline import run_trip_pipeline
from app.services.trip_processing_service import TripProcessingService

REPORT_PATH = BACKEND_ROOT / "artifacts" / "reports" / "phase10_v3_recalc_report.json"


def _is_scoring_version_v3(breakdown_raw: str | None) -> bool:
    if not breakdown_raw:
        return False
    try:
        breakdown = json.loads(breakdown_raw)
    except Exception:
        return False
    if not isinstance(breakdown, dict):
        return False
    if breakdown.get("scoring_version") == "v3":
        return True
    nested = breakdown.get("rule_breakdown")
    return isinstance(nested, dict) and nested.get("scoring_version") == "v3"


def _samples_to_payload(rows: list[SensorSample]) -> list[dict]:
    payload: list[dict] = []
    for row in rows:
        ts = row.ts
        if ts is not None and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        speed_kph = row.speed_mps * 3.6 if row.speed_mps is not None else None
        payload.append(
            {
                "timestamp": ts.isoformat() if ts is not None else None,
                "speed": speed_kph,
                "lat": row.lat,
                "lon": row.lon,
                "ax": row.ax,
                "ay": row.ay,
                "az": row.az,
                "gx": row.gx,
                "gy": row.gy,
                "gz": row.gz,
            }
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalculate historical trips with v3 scoring (Phase 10)")
    parser.add_argument("--trip-id", type=str, default=None)
    parser.add_argument("--user-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Preview new scores without writing")
    parser.add_argument("--include-v3", action="store_true", help="Re-run trips already scored under v3")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    db = SessionLocal()
    cfg = FeatureConfigV2()
    service = TripProcessingService(db)

    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "totals": {},
        "trip_deltas": [],
        "failures": [],
        "skipped": {"raw_deleted": [], "already_v3": [], "active": []},
        "preserved_insufficient": [],
    }

    try:
        stmt = select(Trip).where(Trip.status == "completed")
        if args.trip_id:
            stmt = stmt.where(Trip.id == args.trip_id)
        if args.user_id:
            stmt = stmt.where(Trip.user_id == args.user_id)
        trips = db.execute(stmt.order_by(Trip.started_at.asc())).scalars().all()
        if args.limit:
            trips = trips[: args.limit]

        print(f"Found {len(trips)} completed trips")

        processed = 0
        failed = 0
        preserved = 0
        skipped_raw_deleted = 0
        skipped_v3 = 0
        score_deltas: list[int] = []
        score_before: list[int | None] = []
        score_after: list[int | None] = []
        event_deltas: list[int] = []

        for trip in trips:
            if trip.raw_deleted:
                skipped_raw_deleted += 1
                report["skipped"]["raw_deleted"].append(trip.id)
                print(f"  SKIP {trip.id}: raw samples deleted, cannot recompute")
                continue

            if not args.include_v3 and _is_scoring_version_v3(trip.score_breakdown):
                skipped_v3 += 1
                report["skipped"]["already_v3"].append(trip.id)
                continue

            old_score = trip.score
            old_events = 0
            if trip.score_breakdown:
                try:
                    old_events = len(json.loads(trip.score_breakdown).get("generated_events", []))
                except Exception:
                    pass

            if args.dry_run:
                samples = db.execute(
                    select(SensorSample)
                    .where(SensorSample.trip_id == trip.id)
                    .order_by(SensorSample.ts.asc())
                ).scalars().all()
                result = run_trip_pipeline(_samples_to_payload(samples), cfg)
                new_score = result["score"]
                new_events = len(result["event_instances"])
                preserved_flag = result["breakdown"].get("error") == "not_enough_samples"
            else:
                try:
                    result = service.finalize_trip(
                        user_id=trip.user_id,
                        trip_id=trip.id,
                        delete_raw=False,
                        force_reprocess=True,
                    )
                    new_score = result["score"]
                    new_events = result.get("events_generated", len(result.get("events", [])))
                    preserved_flag = result["breakdown"].get("error") == "not_enough_samples"
                    processed += 1
                except Exception as exc:
                    failed += 1
                    db.rollback()
                    report["failures"].append({"trip_id": trip.id, "error": str(exc)[:500]})
                    print(f"  FAIL {trip.id}: {exc}")
                    continue

            if preserved_flag:
                preserved += 1
                report["preserved_insufficient"].append(trip.id)
                print(f"  PRESERVED {trip.id}: not enough samples (unscored, kept)")
                continue

            score_deltas.append(int(new_score or 0) - int(old_score or 0))
            score_before.append(old_score)
            score_after.append(new_score)
            event_deltas.append(int(new_events) - int(old_events))
            report["trip_deltas"].append(
                {
                    "trip_id": trip.id,
                    "user_id": trip.user_id,
                    "old_score": old_score,
                    "new_score": new_score,
                    "score_delta": int(new_score or 0) - int(old_score or 0),
                    "old_events": old_events,
                    "new_events": new_events,
                }
            )
            print(
                f"  {'PREVIEW' if args.dry_run else 'OK'} {trip.id}: "
                f"score {old_score} -> {new_score} (events {old_events} -> {new_events})"
            )

        totals: dict[str, Any] = {
            "trips_matched": len(trips),
            "processed": processed if not args.dry_run else 0,
            "dry_run_previewed": len(score_after) if args.dry_run else 0,
            "preserved_insufficient": preserved,
            "failed": failed,
            "skipped_raw_deleted": skipped_raw_deleted,
            "skipped_already_v3": skipped_v3,
        }
        if score_after:
            valid_before = [s for s in score_before if s is not None]
            valid_after = [s for s in score_after if s is not None]
            totals["mean_score_before"] = round(sum(valid_before) / len(valid_before), 1) if valid_before else None
            totals["mean_score_after"] = round(sum(valid_after) / len(valid_after), 1) if valid_after else None
            totals["mean_score_delta"] = round(sum(score_deltas) / len(score_deltas), 1)
            totals["mean_event_delta"] = round(sum(event_deltas) / len(event_deltas), 1)
            totals["score_increased_count"] = sum(1 for d in score_deltas if d > 0)
            totals["score_decreased_count"] = sum(1 for d in score_deltas if d < 0)

        report["totals"] = totals
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print("\n=== Summary ===")
        for key, value in totals.items():
            print(f"  {key}: {value}")
        print(f"Report written to {REPORT_PATH}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

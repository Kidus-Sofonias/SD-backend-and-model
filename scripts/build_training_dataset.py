# File role: Offline dataset-building script for ML training.
# Reads completed trips and their sensor samples from the database, runs the shared ML pipeline,
# generates labels using reviewed labels first, weak labels as support, and synthetic labels last,
# then saves a clean training dataset CSV.
# Connects to:
# - app.db.session
# - app.db.models.trip
# - app.db.models.sensor_sample
# - app.ml.config
# - app.ml.pipeline
# - app.ml.schemas
# - artifacts/datasets/synthetic_trip_labels.json
# Key symbols/vars:
# - OUTPUT_PATH
# - SYNTHETIC_LABELS_PATH
# - REVIEWED_LABELS_PATH
# - make_weak_label
# - choose_label
# - main

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select

from app.db.session import SessionLocal
from app.db.models.trip import Trip
from app.db.models.sensor_sample import SensorSample
from app.db.models.vehicle_profile import VehicleProfile
from app.ml.config import FeatureConfigV2
from app.ml.labels import review_label_tier
from app.ml.pipeline import run_trip_pipeline
from app.ml.schemas import FEATURE_VERSION
from app.ml.vehicle_profiles import config_for_profile, resolve_mass_kg
from scripts.reporting_utils import build_dataset_summary


OUTPUT_PATH = Path("artifacts/datasets/trip_features_fv1.csv")
SYNTHETIC_LABELS_PATH = ROOT / "artifacts" / "datasets" / "synthetic_trip_labels.json"
STRONG_LABELS_PATH = ROOT / "artifacts" / "datasets" / "synthetic_ground_truth_labels.json"
REVIEWED_LABELS_PATH = ROOT / "artifacts" / "datasets" / "reviewed_trip_labels.json"
REPORT_PATH = Path("artifacts/reports/dataset_summary_fv1.json")


def load_synthetic_labels() -> dict[str, int]:
    if not SYNTHETIC_LABELS_PATH.exists():
        return {}
    try:
        data = json.loads(SYNTHETIC_LABELS_PATH.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in data.items()}
    except Exception as exc:
        print(f"Warning: failed to load synthetic label registry: {exc}")
        return {}


def load_strong_labels() -> dict[str, int]:
    """Load ground-truth labels from the synthetic generator (strong labels)."""
    if not STRONG_LABELS_PATH.exists():
        return {}
    try:
        data = json.loads(STRONG_LABELS_PATH.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in data.items()}
    except Exception as exc:
        print(f"Warning: failed to load strong label registry: {exc}")
        return {}


def load_reviewed_labels() -> dict[str, int]:
    if not REVIEWED_LABELS_PATH.exists():
        return {}
    try:
        data = json.loads(REVIEWED_LABELS_PATH.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in data.items()}
    except Exception as exc:
        print(f"Warning: failed to load reviewed label registry: {exc}")
        return {}


def make_weak_label(rule_score: int | None) -> int | None:
    if rule_score is None:
        return None
    if rule_score >= 88:
        return 0
    if rule_score <= 84:
        return 1
    return None


def choose_label(
    trip: Trip,
    rule_score: int | None,
    reviewed_labels: dict[str, int],
    synthetic_labels: dict[str, int],
    strong_labels: dict[str, int],
) -> tuple[int | None, str, str]:
    """
    Label priority (highest to lowest):
      1. Human-reviewed label (trip.reviewed_label or reviewed_labels registry)
      2. Synthetic ground-truth label (strong_labels — from generator, known ground truth)
      3. Demo review label (seed script — score-derived showcase labels, NOT human ground truth)
      4. Weak rule-based label (from rule_score — useful when ground truth unknown)
      5. Synthetic bootstrap label (from old synthetic_registry — lowest confidence)
    """
    if trip.reviewed_label is not None:
        reviewed_source = (trip.reviewed_label_source or "reviewed_real").strip() or "reviewed_real"
        return int(trip.reviewed_label), reviewed_source, review_label_tier(trip.reviewed_label_source)

    if trip.id in reviewed_labels:
        return reviewed_labels[trip.id], "reviewed_registry", "reviewed_real"

    # Strong ground-truth from generator — these are known-safe or known-risky
    if trip.id in strong_labels:
        return strong_labels[trip.id], "synthetic_ground_truth", "synthetic_ground_truth"

    weak = make_weak_label(rule_score)
    if weak is not None:
        return weak, "rule_weak_label", "weak_label"

    if trip.id in synthetic_labels:
        return synthetic_labels[trip.id], "synthetic_registry", "synthetic_bootstrap"

    return None, "unlabeled", "unlabeled"


def _class_counts(rows: list[dict]) -> dict[int, int]:
    counts: dict[int, int] = {0: 0, 1: 0}
    for row in rows:
        counts[int(row["label_binary"])] = counts.get(int(row["label_binary"]), 0) + 1
    return counts


def select_rows_for_training(rows: list[dict]) -> tuple[list[dict], dict[str, dict[int, int] | int]]:
    priority_order = [
        "reviewed_real",
        "reviewed_synthetic",
        "synthetic_ground_truth",  # Ground-truth from generator — stronger than weak rules
        # Demo reviews are score-derived showcase labels from the seed script;
        # they rank below generator ground truth but above weak rule labels.
        "reviewed_demo",
        "weak_label",
        "synthetic_bootstrap",
    ]

    rows_by_tier: dict[str, list[dict]] = {tier: [] for tier in priority_order}
    for row in rows:
        tier = str(row["label_tier"])
        if tier in rows_by_tier:
            rows_by_tier[tier].append(row)

    selected: list[dict] = []
    selected_trip_ids: set[str] = set()
    added_by_tier: dict[str, int] = {tier: 0 for tier in priority_order}

    for tier in priority_order:
        tier_rows = sorted(rows_by_tier[tier], key=lambda item: str(item["trip_id"]))
        if not tier_rows:
            continue

        chosen = [
            row for row in tier_rows
            if str(row["trip_id"]) not in selected_trip_ids
        ]
        if not chosen:
            continue

        selected.extend(chosen)
        selected_trip_ids.update(str(row["trip_id"]) for row in chosen)
        added_by_tier[tier] += len(chosen)

    summary = {
        "selected_total": len(selected),
        "selected_class_counts": _class_counts(selected),
        "selected_by_tier": added_by_tier,
    }
    return selected, summary


def main() -> dict[str, Any] | None:
    print("Starting build_training_dataset")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    synthetic_labels = load_synthetic_labels()
    strong_labels = load_strong_labels()
    reviewed_labels = load_reviewed_labels()
    print(f"Loaded {len(synthetic_labels)} synthetic labels from {SYNTHETIC_LABELS_PATH}")
    print(f"Loaded {len(strong_labels)} ground-truth labels from {STRONG_LABELS_PATH}")
    print(f"Loaded {len(reviewed_labels)} reviewed labels from {REVIEWED_LABELS_PATH}")

    db = SessionLocal()
    base_cfg = FeatureConfigV2()

    try:
        trips = db.execute(
            select(Trip).where(Trip.status == "completed")
        ).scalars().all()

        print(f"Found {len(trips)} completed trips")

        candidate_rows = []
        skipped_no_features = 0
        skipped_unlabeled = 0

        # Phase 8b: load the vehicle profiles referenced by the trips up front
        # so each trip is scored through its own vehicle-tuned detection config
        # (config_for_profile). Trips without a profile use the universal default.
        profile_ids = {t.vehicle_profile_id for t in trips if t.vehicle_profile_id}
        profiles: dict[str, VehicleProfile] = {}
        if profile_ids:
            for profile in db.execute(
                select(VehicleProfile).where(VehicleProfile.id.in_(profile_ids))
            ).scalars().all():
                profiles[profile.id] = profile

        for trip in trips:
            sample_rows = db.execute(
                select(SensorSample)
                .where(SensorSample.trip_id == trip.id)
                .order_by(SensorSample.ts.asc())
            ).scalars().all()

            samples = []
            for row in sample_rows:
                # speed_mps stores m/s; pipeline expects km/h — convert here
                speed_kph = row.speed_mps * 3.6 if row.speed_mps is not None else None
                samples.append({
                    "timestamp": row.ts.isoformat() if row.ts else None,
                    "speed": speed_kph,
                    "lat": row.lat,
                    "lon": row.lon,
                    "altitude_m": row.altitude_m,
                    "ax": row.ax,
                    "ay": row.ay,
                    "az": row.az,
                    "gx": row.gx,
                    "gy": row.gy,
                    "gz": row.gz,
                })

            # Phase 8b: the vehicle profile tunes detection thresholds for THIS
            # trip (sedan vs truck), so the feature values the model trains on
            # reflect the same vehicle-aware signal the runtime uses.
            profile = profiles.get(trip.vehicle_profile_id) if trip.vehicle_profile_id else None
            cfg = config_for_profile(base_cfg, profile)
            result = run_trip_pipeline(samples, cfg, vehicle_profile=profile)
            trip_features = result["trip_features"]
            rule_score = result["score"]

            if not trip_features:
                skipped_no_features += 1
                continue

            label_binary, label_source, label_tier = choose_label(
                trip,
                rule_score,
                reviewed_labels,
                synthetic_labels,
                strong_labels,
            )

            if label_binary is None:
                skipped_unlabeled += 1
                continue

            row = {
                "trip_id": trip.id,
                "feature_version": FEATURE_VERSION,
                "model_version": trip.model_version,
                "processed_at": trip.processed_at.isoformat() if trip.processed_at else None,
                # Phase 8b: vehicle context columns for analysis/stratification.
                "vehicle_category": profile.category if profile else None,
                "vehicle_mass_kg": resolve_mass_kg(
                    profile.category,
                    size_class=profile.size_class,
                    mass_kg=profile.mass_kg,
                ) if profile else None,
                **trip_features,
                "rule_score": rule_score,
                "label_binary": int(label_binary),
                "label_source": label_source,
                "label_tier": label_tier,
            }
            candidate_rows.append(row)

        rows, selection_summary = select_rows_for_training(candidate_rows)

        if not rows:
            print("No labeled dataset rows generated.")
            summary = build_dataset_summary(
                rows,
                selection_summary=selection_summary,
                skipped_no_features=skipped_no_features,
                skipped_unlabeled=skipped_unlabeled,
            )
            REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary

        fieldnames = list(rows[0].keys())
        with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        summary = build_dataset_summary(
            rows,
            selection_summary=selection_summary,
            skipped_no_features=skipped_no_features,
            skipped_unlabeled=skipped_unlabeled,
        )
        REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"Dataset created with {len(rows)} labeled rows at {OUTPUT_PATH}")
        print(f"Risk class distribution: {summary['risk_class_counts']}")
        print(f"Label sources: {summary['label_source_counts']}")
        print(f"Label tiers: {summary['label_tier_counts']}")
        print(f"Model versions: {summary['model_version_counts']}")
        print(f"Feature versions: {summary['feature_version_counts']}")
        print(f"Selection summary: {selection_summary}")
        print(f"Skipped (no features): {skipped_no_features}")
        print(f"Skipped (unlabeled): {skipped_unlabeled}")
        print(f"Summary report: {REPORT_PATH}")
        return summary

    finally:
        db.close()


if __name__ == "__main__":
    main()

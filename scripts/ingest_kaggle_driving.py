"""Ingest the Kaggle Large-Scale Driver Behavior Sensor Dataset.

Expected CSV columns (from kaggle.com/datasets/shakilofficial0/large-scale-driver-behavior-sensor-dataset):
    AccX, AccY, AccZ   — accelerometer (m/s²)
    GyroX, GyroY, GyroZ — gyroscope (rad/s)
    Class               — SLOW | NORMAL | AGGRESSIVE
    Timestamp           — seconds since session start

This script:
1. Reads the CSV
2. Segments continuous sensor rows into trip-length chunks (5-15 min)
3. Maps labels: AGGRESSIVE → 1 (risky), NORMAL/SLOW → 0 (safe)
4. Creates Trip + SensorSample records in the DB
5. Runs the feature pipeline on each trip
6. Saves labels into the reviewed_trip_labels registry

Usage:
    python -m scripts.ingest_kaggle_driving --csv path/to/dataset.csv [--max-trips 500] [--db sqlite:///./demo_v3.db]
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models.trip import Trip
from app.db.models.sensor_sample import SensorSample
from app.db.models.user import User
from app.ml.config import FeatureConfigV2
from app.ml.pipeline import run_trip_pipeline
from app.ml.schemas import FEATURE_COLUMNS_FV1

# Import all models so Base.metadata knows about them
import app.db.models.user  # noqa: F401
import app.db.models.trip  # noqa: F401
import app.db.models.sensor_sample  # noqa: F401
import app.db.models.driving_event  # noqa: F401
import app.db.models.vehicle_profile  # noqa: F401

LABELS_PATH = ROOT / "artifacts" / "datasets" / "reviewed_trip_labels.json"

# Trip segmentation params
MIN_TRIP_SECONDS = 5 * 60   # 5 minutes
MAX_TRIP_SECONDS = 15 * 60  # 15 minutes
TARGET_TRIP_SECONDS = 8 * 60  # 8 minutes target
MIN_SAMPLES_PER_TRIP = 100   # at ~50 Hz this is ~2 seconds, but we want more

# Label mapping
LABEL_MAP = {
    "AGGRESSIVE": 1,
    "NORMAL": 0,
    "SLOW": 0,
}

# Synthetic user for imported trips
IMPORT_USER_EMAIL = "kaggle-import@drivepulse.local"
IMPORT_USER_PASSWORD = "kaggle-import-placeholder"


def ensure_import_user(session: Session) -> str:
    """Create or find the import user, return user_id."""
    user = session.execute(
        text("SELECT id FROM users WHERE email = :email"),
        {"email": IMPORT_USER_EMAIL},
    ).fetchone()
    if user:
        return user[0]

    user_id = str(uuid.uuid4())
    session.execute(
        text(
            "INSERT INTO users (id, email, password_hash, full_name, is_admin, created_at) "
            "VALUES (:id, :email, :pw, :name, 0, :now)"
        ),
        {
            "id": user_id,
            "email": IMPORT_USER_EMAIL,
            "pw": IMPORT_USER_PASSWORD,
            "name": "Kaggle Import Driver",
            "now": datetime.utcnow(),
        },
    )
    session.commit()
    print(f"Created import user {user_id}")
    return user_id


def segment_trips(df: pd.DataFrame, max_trips: int) -> list[dict]:
    """Segment the continuous sensor stream into trip-length chunks.

    Returns list of dicts: {rows: DataFrame, label: int, label_source: str}
    """
    trips: list[dict] = []

    # Sort by timestamp
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Find natural boundaries: gaps > 30s or label changes
    df["time_diff"] = df["Timestamp"].diff().fillna(0)
    df["label_change"] = df["Class"] != df["Class"].shift(1)
    df["boundary"] = (df["time_diff"] > 30) | df["label_change"]

    # Assign segment IDs
    df["segment_id"] = df["boundary"].cumsum()

    # Merge short segments (< MIN_TRIP_SECONDS) with adjacent same-label segments
    segments: list[dict] = []
    for seg_id, group in df.groupby("segment_id"):
        duration = group["Timestamp"].iloc[-1] - group["Timestamp"].iloc[0]
        label = group["Class"].iloc[0]
        segments.append({
            "rows": group,
            "duration_s": duration,
            "label": label,
        })

    # Merge adjacent same-label short segments
    merged: list[dict] = []
    for seg in segments:
        if merged and merged[-1]["label"] == seg["label"] and merged[-1]["duration_s"] < MIN_TRIP_SECONDS:
            # Merge
            merged[-1]["rows"] = pd.concat([merged[-1]["rows"], seg["rows"]])
            merged[-1]["duration_s"] = merged[-1]["rows"]["Timestamp"].iloc[-1] - merged[-1]["rows"]["Timestamp"].iloc[0]
        else:
            merged.append(seg)

    # Split long segments into MAX_TRIP_SECONDS chunks
    for seg in merged:
        if seg["duration_s"] < MIN_TRIP_SECONDS / 2:
            continue  # Too short even after merging

        rows = seg["rows"]
        ts_start = rows["Timestamp"].iloc[0]
        ts_end = rows["Timestamp"].iloc[-1]

        # Split into chunks
        current_start = ts_start
        while current_start < ts_end:
            current_end = min(current_start + TARGET_TRIP_SECONDS, ts_end)
            chunk = rows[(rows["Timestamp"] >= current_start) & (rows["Timestamp"] < current_end)]

            if len(chunk) >= MIN_SAMPLES_PER_TRIP:
                trip_duration = chunk["Timestamp"].iloc[-1] - chunk["Timestamp"].iloc[0]
                if trip_duration >= MIN_TRIP_SECONDS / 2:
                    label_str = chunk["Class"].iloc[0]
                    label_int = LABEL_MAP.get(label_str, 0)
                    trips.append({
                        "rows": chunk,
                        "label": label_int,
                        "label_source": f"kaggle_{label_str.lower()}",
                    })

            current_start = current_end

            if len(trips) >= max_trips:
                break
        if len(trips) >= max_trips:
            break

    return trips


def insert_trips(
    session: Session,
    user_id: str,
    trip_chunks: list[dict],
    base_time: datetime,
) -> list[str]:
    """Insert trips + sensor samples into the DB. Return trip IDs."""
    trip_ids: list[str] = []
    cfg = FeatureConfigV2()

    for i, chunk in enumerate(trip_chunks):
        trip_id = str(uuid.uuid4())
        rows = chunk["rows"]
        ts_start_s = float(rows["Timestamp"].iloc[0])
        ts_end_s = float(rows["Timestamp"].iloc[-1])
        started_at = base_time + timedelta(seconds=ts_start_s)
        ended_at = base_time + timedelta(seconds=ts_end_s)

        # Create trip
        trip = Trip(
            id=trip_id,
            user_id=user_id,
            started_at=started_at,
            ended_at=ended_at,
            status="completed",
            reviewed_label=chunk["label"],
            reviewed_label_source=chunk["label_source"],
            reviewed_at=datetime.utcnow(),
        )
        session.add(trip)

        # Create sensor samples
        for _, row in rows.iterrows():
            ts_s = float(row["Timestamp"])
            sample_ts = base_time + timedelta(seconds=ts_s)

            # Compute speed estimate from accelerometer integration (rough)
            # The Kaggle dataset doesn't have GPS speed, so we estimate from ax
            sample = SensorSample(
                user_id=user_id,
                trip_id=trip_id,
                ts=sample_ts,
                ax=float(row.get("AccX", 0)),
                ay=float(row.get("AccY", 0)),
                az=float(row.get("AccZ", 0)),
                gx=float(row.get("GyroX", 0)),
                gy=float(row.get("GyroY", 0)),
                gz=float(row.get("GyroZ", 0)),
                speed_mps=None,  # No GPS speed in this dataset
                lat=None,
                lon=None,
                accuracy_m=None,
            )
            session.add(sample)

        session.flush()  # Get the trip ID committed

        # Run feature pipeline
        try:
            # Re-fetch samples for the pipeline
            from sqlalchemy import select as sa_select
            samples = session.execute(
                sa_select(SensorSample).where(SensorSample.trip_id == trip_id).order_by(SensorSample.ts)
            ).scalars().all()

            trip_features = run_trip_pipeline(samples, cfg)

            # Persist features on the trip
            trip.feature_version = "fv1"
            trip.processed_at = datetime.utcnow()
            trip.score = trip_features.get("rule_score")

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(trip_chunks)} trips...")
                session.commit()

        except Exception as exc:
            print(f"  Warning: pipeline failed for trip {trip_id}: {exc}")
            # Still keep the trip with its label — just skip feature computation

        trip_ids.append(trip_id)

    session.commit()
    return trip_ids


def save_labels(trip_ids: list[str], trip_chunks: list[dict]) -> None:
    """Append labels to the reviewed_trip_labels.json registry."""
    existing: dict[str, int] = {}
    if LABELS_PATH.exists():
        try:
            existing = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    for trip_id, chunk in zip(trip_ids, trip_chunks):
        existing[trip_id] = chunk["label"]

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Saved {len(trip_ids)} labels to {LABELS_PATH} (total: {len(existing)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Kaggle driving behavior dataset")
    parser.add_argument("--csv", required=True, help="Path to the downloaded CSV file")
    parser.add_argument("--max-trips", type=int, default=500, help="Max trips to ingest (default: 500)")
    parser.add_argument("--db", default="sqlite:///./demo_v3.db", help="Database URL")
    parser.add_argument("--dry-run", action="store_true", help="Only segment, don't insert")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        sys.exit(1)

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    # Normalize column names (handle various formats)
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ("accx", "accelerometerx", "acc_x", "acceleration along x-axis, m/s²"):
            col_map[col] = "AccX"
        elif cl in ("accy", "accelerometery", "acc_y", "acceleration along y-axis, m/s²"):
            col_map[col] = "AccY"
        elif cl in ("accz", "accelerometerz", "acc_z", "acceleration along z-axis, m/s²"):
            col_map[col] = "AccZ"
        elif cl in ("gyrox", "gyroscopex", "gyro_x", "gyroscope x-axis (rad/s)"):
            col_map[col] = "GyroX"
        elif cl in ("gyroy", "gyroscopey", "gyro_y", "gyroscope y-axis (rad/s)"):
            col_map[col] = "GyroY"
        elif cl in ("gyroz", "gyroscopez", "gyro_z", "gyroscope z-axis (rad/s)"):
            col_map[col] = "GyroZ"
        elif cl in ("class", "label", "behavior", "driving behavior label ( slow , normal , aggressive ), categorical"):
            col_map[col] = "Class"
        elif cl in ("timestamp", "time", "time in seconds since the start of the driving session"):
            col_map[col] = "Timestamp"

    if col_map:
        df = df.rename(columns=col_map)
        print(f"  Normalized columns: {list(df.columns)}")

    # Validate required columns
    required = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "Class", "Timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: missing columns {missing}")
        print(f"  Available: {list(df.columns)}")
        print("  Please check the CSV format and adjust the column mapping in this script.")
        sys.exit(1)

    # Clean data
    df = df.dropna(subset=["Timestamp"])
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df["Class"] = df["Class"].str.strip().str.upper()

    print(f"  Label distribution: {df['Class'].value_counts().to_dict()}")

    # Segment into trips
    print(f"Segmenting into trips (target: {args.max_trips})...")
    trip_chunks = segment_trips(df, args.max_trips)
    print(f"  Generated {len(trip_chunks)} trip segments")

    # Show distribution
    labels = [c["label"] for c in trip_chunks]
    print(f"  Label distribution: safe={labels.count(0)}, risky={labels.count(1)}")

    if args.dry_run:
        print("Dry run — not inserting into DB.")
        for i, chunk in enumerate(trip_chunks[:5]):
            rows = chunk["rows"]
            dur = rows["Timestamp"].iloc[-1] - rows["Timestamp"].iloc[0]
            print(f"  Trip {i}: {len(rows)} samples, {dur:.0f}s, label={'risky' if chunk['label'] else 'safe'}")
        return

    # Insert into DB
    print(f"Connecting to {args.db}...")
    engine = create_engine(args.db)
    Base.metadata.create_all(bind=engine)

    with Session(engine) as session:
        user_id = ensure_import_user(session)
        print(f"Import user: {user_id}")

        base_time = datetime(2026, 1, 1)  # Synthetic base time for imports
        print(f"Inserting {len(trip_chunks)} trips...")
        trip_ids = insert_trips(session, user_id, trip_chunks, base_time)
        print(f"Inserted {len(trip_ids)} trips")

        # Save labels
        save_labels(trip_ids, trip_chunks)

    print("Done! Run 'python -m scripts.build_training_dataset' to rebuild the training set.")


if __name__ == "__main__":
    main()

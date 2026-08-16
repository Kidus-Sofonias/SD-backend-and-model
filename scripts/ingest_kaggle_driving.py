"""Ingest the Kaggle Large-Scale Driver Behavior Sensor Dataset.

Dataset structure (verified against the shipped CSV):
    - 7,000,000 rows / 8 columns: AccX..GyroZ, Class (SLOW|NORMAL|AGGRESSIVE), Timestamp
    - Timestamp is milliseconds since session start. There are 3,500,000 unique
      timestamps and every timestamp appears EXACTLY twice — the file contains
      two independent recordings (stream A = first occurrence, stream B = second).
    - Each stream spans timestamps 0 .. ~3,500,000 ms (~58 min of driving).
    - Class is assigned by timestamp RANGE: NORMAL [0, 1,925,000), SLOW
      [1,925,000, 2,625,000), AGGRESSIVE [2,625,000, 3,500,000).
    - Sensor magnitudes genuinely separate per label (AGGRESSIVE gyro p95 ~32
      rad/s vs NORMAL ~9.7 vs SLOW ~1.1) but are NOT physical phone units, so
      they are calibrated to realistic units before the pipeline sees them.

This script:
1. Splits the file into the two streams (first/second occurrence per timestamp)
2. Calibrates IMU magnitudes to physical units (accel ~m/s^2, gyro ~rad/s)
3. Synthesizes a GPS speed trace from the calibrated accelerometer (the
   dataset has no speed column, and the pipeline requires one)
4. Downsamples to 10 Hz (realistic for our scoring pipeline)
5. Segments each stream into trip-length chunks
6. Maps labels: AGGRESSIVE -> 1 (risky), NORMAL/SLOW -> 0 (safe)
7. Creates Trip + SensorSample records, runs the feature pipeline
8. Stores labels with source "kaggle_external" (a distinct training-data tier,
   NOT a human admin review)

Usage:
    python -m scripts.ingest_kaggle_driving --csv ../artifacts/datasets/dataset_7M.csv [--trip-seconds 90] [--db sqlite:///./demo_v3.db]
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models.trip import Trip
from app.db.models.sensor_sample import SensorSample
from app.ml.config import FeatureConfigV2
from app.ml.pipeline import run_trip_pipeline

# Import all models so Base.metadata knows about them
import app.db.models.user  # noqa: F401
import app.db.models.trip  # noqa: F401
import app.db.models.sensor_sample  # noqa: F401
import app.db.models.driving_event  # noqa: F401
import app.db.models.vehicle_profile  # noqa: F401

LABELS_PATH = ROOT / "artifacts" / "datasets" / "reviewed_trip_labels.json"

# --- Segmentation / sampling ---
DOWNSAMPLE_DT_S = 0.1          # 10 Hz output rate
MAX_TRIP_SECONDS = 15 * 60
MIN_SAMPLES_PER_TRIP = 50

# --- Label mapping ---
LABEL_MAP = {"AGGRESSIVE": 1, "NORMAL": 0, "SLOW": 0}

# --- Physical calibration targets ---
# The synthetic magnitudes are internally consistent but not in physical units
# (AGGRESSIVE |ax| p95 ~0.33, |gz| p95 ~32). We rescale so the AGGRESSIVE
# p99 lands at a realistic harsh-maneuver magnitude:
#   - accel p99  -> ~5.0 m/s^2  (above the 3.2 harsh threshold)
#   - gyro  p99  -> ~0.9 rad/s  (realistic aggressive yaw rate)
# The transform is a single fixed global scaling (same for every trip), so it
# never leaks per-trip label information; it only maps the dataset's magnitude
# scale into physical units our thresholds understand.
ACCEL_P99_TARGET_MPS2 = 5.0
GYRO_P99_TARGET_RADS = 0.9

# Speed synthesis cruise (label-agnostic; only the sensor signal differentiates
# trips, so the model cannot read the label off the speed profile).
CRUISE_SPEED_MPS = 15.0
MAX_SPEED_MPS = 40.0

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

    # Schema-aware insert: newer DBs use full_name/is_admin/created_at, older
    # seeded SQLite files use a bare role column. Introspect once and pick.
    cols = [row[1] for row in session.execute(text("PRAGMA table_info(users)")).fetchall()]
    user_id = str(uuid.uuid4())
    if "full_name" in cols and "is_admin" in cols and "created_at" in cols:
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
                "now": datetime.now(timezone.utc),
            },
        )
    else:
        session.execute(
            text(
                "INSERT INTO users (id, email, password_hash, role) "
                "VALUES (:id, :email, :pw, 'driver')"
            ),
            {
                "id": user_id,
                "email": IMPORT_USER_EMAIL,
                "pw": IMPORT_USER_PASSWORD,
            },
        )
    session.commit()
    print(f"Created import user {user_id}")
    return user_id


def split_streams(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the file into stream A (first occurrence of each timestamp) and
    stream B (second occurrence). Rows are streamed in chunks to bound memory."""
    seen: set[int] = set()
    a_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []

    for chunk in pd.read_csv(
        csv_path,
        usecols=["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "Class", "Timestamp"],
        chunksize=500_000,
    ):
        ts = chunk["Timestamp"].to_numpy()
        accel = chunk[["AccX", "AccY", "AccZ"]].to_numpy(dtype=float)
        gyro = chunk[["GyroX", "GyroY", "GyroZ"]].to_numpy(dtype=float)
        cls = chunk["Class"].str.strip().str.upper().to_numpy()

        for i in range(len(ts)):
            t = int(ts[i])
            if t in seen:
                b_rows.append(np.concatenate([[t], accel[i], gyro[i], [1.0 if cls[i] == "AGGRESSIVE" else 0.0]]))
            else:
                seen.add(t)
                a_rows.append(np.concatenate([[t], accel[i], gyro[i], [1.0 if cls[i] == "AGGRESSIVE" else 0.0]]))

    cols = ["t_ms", "ax", "ay", "az", "gx", "gy", "gz", "label"]
    a = pd.DataFrame(np.asarray(a_rows, dtype=float), columns=cols) if a_rows else pd.DataFrame(columns=cols)
    b = pd.DataFrame(np.asarray(b_rows, dtype=float), columns=cols) if b_rows else pd.DataFrame(columns=cols)
    print(f"Stream A: {len(a)} rows | Stream B: {len(b)} rows")
    return a, b


def calibrate_and_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Calibrate magnitudes to physical units and synthesize a speed trace.

    The scaling factors are computed from the data's OWN distribution (p99 of
    the whole stream), so the transform is global and label-agnostic.
    """
    ax_p99 = float(np.percentile(np.abs(df["ax"].to_numpy()), 99))
    gz_p99 = float(np.percentile(np.abs(df["gz"].to_numpy()), 99))
    accel_scale = ACCEL_P99_TARGET_MPS2 / max(ax_p99, 1e-6)
    gyro_scale = GYRO_P99_TARGET_RADS / max(gz_p99, 1e-6)
    print(f"  Calibration: accel scale {accel_scale:.3f} (ax p99 {ax_p99:.3f}), gyro scale {gyro_scale:.4f} (gz p99 {gz_p99:.3f})")

    for col in ["ax", "ay", "az"]:
        df[col] = df[col] * accel_scale
    for col in ["gx", "gy", "gz"]:
        df[col] = df[col] * gyro_scale

    # Sort chronologically and downsample to ~10 Hz (every 100 ms).
    df = df.sort_values("t_ms").reset_index(drop=True)
    df = df[df["t_ms"] % 100 == 0].reset_index(drop=True)

    # Synthesize speed by integrating forward acceleration, with a gentle
    # pull back toward the cruise speed so it does not drift to the clamps.
    # dv = ax * dt, so the speed profile directly encodes the accel signal.
    t_s = df["t_ms"].to_numpy() / 1000.0
    ax = df["ax"].to_numpy()
    dt = np.diff(t_s, prepend=t_s[0])
    v = np.empty(len(df), dtype=float)
    v[0] = CRUISE_SPEED_MPS
    for i in range(1, len(df)):
        v[i] = v[i - 1] + ax[i] * dt[i] + (CRUISE_SPEED_MPS - v[i - 1]) * 0.004
        v[i] = float(np.clip(v[i], 0.0, MAX_SPEED_MPS))
    df["speed_mps"] = v
    return df


def segment_trips(df: pd.DataFrame, trip_seconds: int) -> list[dict]:
    """Split a stream into trip-length chunks (by elapsed ms)."""
    trips: list[dict] = []
    t0 = df["t_ms"].iloc[0]
    t1 = df["t_ms"].iloc[-1]
    span_ms = t1 - t0

    start_ms = t0
    while start_ms < t1:
        end_ms = start_ms + trip_seconds * 1000
        chunk = df[(df["t_ms"] >= start_ms) & (df["t_ms"] < end_ms)]
        if len(chunk) >= MIN_SAMPLES_PER_TRIP:
            trips.append({"rows": chunk, "label": int(chunk["label"].iloc[0])})
        start_ms = end_ms
        if span_ms <= 0:
            break
    return trips


def insert_trips(session: Session, user_id: str, trip_chunks: list[dict], base_time: datetime) -> list[str]:
    """Insert trips + sensor samples, run the feature pipeline, return trip IDs."""
    trip_ids: list[str] = []
    cfg = FeatureConfigV2()

    for i, chunk in enumerate(trip_chunks):
        trip_id = str(uuid.uuid4())
        rows = chunk["rows"]
        ts_start_ms = float(rows["t_ms"].iloc[0])
        ts_end_ms = float(rows["t_ms"].iloc[-1])
        started_at = base_time + timedelta(milliseconds=ts_start_ms)
        ended_at = base_time + timedelta(milliseconds=ts_end_ms)

        trip = Trip(
            id=trip_id,
            user_id=user_id,
            started_at=started_at,
            ended_at=ended_at,
            status="completed",
            reviewed_label=chunk["label"],
            reviewed_label_source="kaggle_external",
            reviewed_at=datetime.now(timezone.utc),
        )
        session.add(trip)

        for _, row in rows.iterrows():
            sample_ts = base_time + timedelta(milliseconds=float(row["t_ms"]))
            sample = SensorSample(
                user_id=user_id,
                trip_id=trip_id,
                ts=sample_ts,
                ax=float(row["ax"]),
                ay=float(row["ay"]),
                az=float(row["az"]),
                gx=float(row["gx"]),
                gy=float(row["gy"]),
                gz=float(row["gz"]),
                speed_mps=float(row["speed_mps"]),
                lat=None,
                lon=None,
                accuracy_m=None,
            )
            session.add(sample)

        session.flush()

        try:
            samples = session.execute(
                select(SensorSample).where(SensorSample.trip_id == trip_id).order_by(SensorSample.ts)
            ).scalars().all()
            payload = [
                {
                    "timestamp": s.ts.replace(tzinfo=timezone.utc).isoformat() if s.ts.tzinfo is None else s.ts.isoformat(),
                    "speed": s.speed_mps * 3.6 if s.speed_mps is not None else None,
                    "lat": s.lat,
                    "lon": s.lon,
                    "ax": s.ax,
                    "ay": s.ay,
                    "az": s.az,
                    "gx": s.gx,
                    "gy": s.gy,
                    "gz": s.gz,
                }
                for s in samples
            ]
            result = run_trip_pipeline(payload, cfg)
            trip_features = result["trip_features"]
            trip.feature_version = "fv1"
            trip.processed_at = datetime.now(timezone.utc)
            trip.score = result["score"]
        except Exception as exc:
            print(f"  Warning: pipeline failed for trip {trip_id}: {exc}")

        trip_ids.append(trip_id)
        if (i + 1) % 25 == 0:
            session.commit()
            print(f"  Processed {i + 1}/{len(trip_chunks)} trips...")

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
    print(f"Saved {len(trip_ids)} labels to {LABELS_PATH} (total registry: {len(existing)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Kaggle driving behavior dataset")
    parser.add_argument("--csv", required=True, help="Path to the downloaded CSV file")
    parser.add_argument("--trip-seconds", type=int, default=90, help="Target trip length in seconds (default: 90)")
    parser.add_argument("--db", default="sqlite:///./demo_v3.db", help="Database URL")
    parser.add_argument("--dry-run", action="store_true", help="Only split/calibrate/segment, don't insert")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        sys.exit(1)

    print(f"Splitting {csv_path} into the two embedded streams...")
    stream_a, stream_b = split_streams(csv_path)

    all_chunks: list[tuple[pd.DataFrame, int]] = []
    for name, stream in (("A", stream_a), ("B", stream_b)):
        if stream.empty:
            print(f"  Stream {name} empty — skipping")
            continue
        print(f"Calibrating + synthesizing speed for stream {name}...")
        stream = calibrate_and_speed(stream)
        chunks = segment_trips(stream, args.trip_seconds)
        print(f"  Stream {name}: {len(stream)} samples @10Hz, {len(chunks)} trips "
              f"(safe={sum(1 for c in chunks if c['label'] == 0)}, risky={sum(1 for c in chunks if c['label'] == 1)})")
        all_chunks.extend((c["rows"], c["label"]) for c in chunks)

    trip_chunks = [{"rows": rows, "label": label} for rows, label in all_chunks]
    print(f"\nTotal trips: {len(trip_chunks)} | safe={sum(1 for c in trip_chunks if c['label'] == 0)}, "
          f"risky={sum(1 for c in trip_chunks if c['label'] == 1)}")

    if args.dry_run:
        print("\nDry run — not inserting into DB.")
        for i, chunk in enumerate(trip_chunks[:8]):
            rows = chunk["rows"]
            dur_s = (rows["t_ms"].iloc[-1] - rows["t_ms"].iloc[0]) / 1000.0
            print(f"  Trip {i}: {len(rows)} samples, {dur_s:.0f}s, label={'risky' if chunk['label'] else 'safe'}, "
                  f"mean speed {rows['speed_mps'].mean():.1f} m/s")
        return

    print(f"\nConnecting to {args.db}...")
    engine = create_engine(args.db)
    Base.metadata.create_all(bind=engine)

    with Session(engine) as session:
        user_id = ensure_import_user(session)
        print(f"Import user: {user_id}")
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        print(f"Inserting {len(trip_chunks)} trips...")
        trip_ids = insert_trips(session, user_id, trip_chunks, base_time)
        print(f"Inserted {len(trip_ids)} trips")
        save_labels(trip_ids, trip_chunks)

    print("\nDone! Run 'python -m scripts.build_training_dataset' to rebuild the training set.")


if __name__ == "__main__":
    main()

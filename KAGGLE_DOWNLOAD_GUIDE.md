# Kaggle Driving Dataset — Download & Ingest Guide

The dataset was **already downloaded and ingested** (Aug 16, 2026). This guide
documents what happened and how to re-run it.

## The dataset

**Kaggle — Large-Scale Driver Behavior Sensor Dataset**
https://www.kaggle.com/datasets/shakilofficial0/large-scale-driver-behavior-sensor-dataset

- **File:** `artifacts/datasets/dataset_7M.csv` (947 MB, 7,000,000 rows)
- **Columns:** `AccX, AccY, AccZ, GyroX, GyroY, GyroZ, Class (SLOW|NORMAL|AGGRESSIVE), Timestamp`
- **License:** free Kaggle download (account required)

### Actual structure (decoded during ingestion)

- **Two interleaved recordings** — 3,500,000 unique timestamps, each appearing
  exactly twice (stream A = first occurrence, stream B = second).
- Each stream spans ~58 minutes of driving (timestamps are a synthetic ms
  index, labels assigned by timestamp range).
- **No GPS speed** — the ingestion synthesizes speed from the calibrated
  accelerometer.
- **Magnitudes are not physical units** — calibrated with a fixed global scale
  before the pipeline sees them.

## Already done (committed)

1. CSV moved from repo root → `artifacts/datasets/dataset_7M.csv`
2. `scripts/ingest_kaggle_driving.py` rewritten for the real structure:
   split streams → calibrate → synthesize speed → 10 Hz downsample → segment
   into 45 s trips → **156 trips** inserted → feature pipeline run
3. New `reviewed_external` label tier (`app/ml/labels.py`) so benchmark labels
   never count as human reviews
4. `build_training_dataset` → **185 labeled rows** (was 14)
5. `train_model_v2` → promoted `lr_20260816T031505Z` (risky-F1 **0.952**,
   Brier 0.021) — see `docs/HACKATHON_PHASE8D_KAGGLE_RETRAIN.md`

## Re-running from scratch

```bash
cd backend

# Fresh local DB with current schema + demo seed
python -m scripts.reset_and_seed --local-db kaggle_train.db --drivers 5 --trips-per-driver 6

# Dry run (just segment + calibrate, no DB writes)
python -m scripts.ingest_kaggle_driving --csv ../artifacts/datasets/dataset_7M.csv --trip-seconds 45 --dry-run

# Ingest into the fresh DB
python -m scripts.ingest_kaggle_driving --csv ../artifacts/datasets/dataset_7M.csv --trip-seconds 45 --db "sqlite:///./kaggle_train.db"

# Rebuild dataset + retrain + benchmark (point at the same DB)
DATABASE_URL="sqlite:///./kaggle_train.db" python -m scripts.build_training_dataset
python -m scripts.train_model_v2
python -m scripts.benchmark_models
```

### Why 156 trips and not 500

The file contains only ~116 minutes of unique driving (2 × 58 min). 156 trips
at 45 s is the honest ceiling from this dataset — 500 trips would require
7-second slices, which would be noise. The path to 500+ is the **admin review
loop**: every admin review of a completed trip feeds the training set as a
safe/risky human label, and the auto-retrain loop retrains at
`AUTO_RETRAIN_MIN_REVIEWED = 30` real reviews.

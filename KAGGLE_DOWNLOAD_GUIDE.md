# How to Download & Ingest the Kaggle Driving Dataset

## Step 1: Download

1. Go to **https://www.kaggle.com/datasets/shakilofficial0/large-scale-driver-behavior-sensor-dataset**
2. Click **Download** (free account required)
3. You'll get a ZIP — unzip it to get a CSV file (likely `driver_behavior.csv` or similar)

## Step 2: Place the file

Put the CSV anywhere accessible, e.g.:
```
backend/driver_behavior.csv
```

## Step 3: Dry run (optional — verify it works)

```bash
cd backend
python -m scripts.ingest_kaggle_driving --csv driver_behavior.csv --dry-run
```

This will segment the data into trips and show you what would be inserted without touching the DB.

## Step 4: Ingest

```bash
cd backend
python -m scripts.ingest_kaggle_driving --csv driver_behavior.csv --max-trips 500
```

This will:
- Read the CSV (7M rows)
- Segment into 500 trips (5-15 minutes each)
- Map labels: AGGRESSIVE → risky (1), NORMAL/SLOW → safe (0)
- Insert trips + sensor samples into the DB
- Run the feature pipeline on each trip
- Save labels to `artifacts/datasets/reviewed_trip_labels.json`

## Step 5: Rebuild training dataset

```bash
cd backend
python -m scripts.build_training_dataset
```

## Step 6: Retrain

```bash
cd backend
python -m scripts.train_model_v2
python -m scripts.benchmark_models
```

## What the dataset contains

| Column | Description |
|---|---|
| AccX, AccY, AccZ | Accelerometer (m/s²) |
| GyroX, GyroY, GyroZ | Gyroscope (rad/s) |
| Class | SLOW, NORMAL, or AGGRESSIVE |
| Timestamp | Seconds since session start |

## Label mapping

- **AGGRESSIVE** → risky (1) — harsh braking, sudden turns, speeding
- **NORMAL** → safe (0) — smooth driving
- **SLOW** → safe (0) — cautious driving (still safe, just slower)

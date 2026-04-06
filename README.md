# Safe Driving Backend

FastAPI backend and ML pipeline for ingesting trip sensor samples, finalizing trips, generating driving events, and scoring risky driving behavior.

## What is included

- Auth endpoints for register, login, and current-user lookup
- Trip lifecycle endpoints for start, end, list, summary, finalize, review, and reprocess
- Sensor sample upload and retrieval
- Driving event creation and history
- Offline ML dataset generation, training, evaluation, comparison, and trip reprocessing

## Recommended environment

Use the dedicated conda environment defined in `environment.yml`. Do not install these packages into the system Python.

## Setup

```powershell
conda env create -f environment.yml
conda run -n safe-driving-backend python --version
```

If the environment already exists:

```powershell
conda env update -f environment.yml --prune
```

## Run the API

From the `backend` directory:

```powershell
conda run -n safe-driving-backend uvicorn app.main:app --reload
```

Health check:

```powershell
curl http://127.0.0.1:8000/api/v1/health
```

## Tests

```powershell
conda run -n safe-driving-backend python -m pytest -q
```

## ML workflow

Build dataset:

```powershell
conda run -n safe-driving-backend python scripts/build_training_dataset.py
```

Top up the training pool to 500 completed trips, then rebuild, retrain, compare, test, and promote:

```powershell
conda run -n safe-driving-backend python scripts/refresh_model_cycle.py --target-trip-count 500
```

Train model:

```powershell
conda run -n safe-driving-backend python scripts/train_model.py
```

Evaluate saved model:

```powershell
conda run -n safe-driving-backend python scripts/evaluate_model.py
```

Reprocess finalized trips against the latest model:

```powershell
conda run -n safe-driving-backend python scripts/reprocess_finalized_trips.py
```

Compare the current production model against a candidate:

```powershell
conda run -n safe-driving-backend python scripts/compare_models.py --current lr_v1 --candidate gb_v1
```

Export reviewed-label mistake analysis for the current model:

```powershell
conda run -n safe-driving-backend python scripts/reviewed_model_analysis.py
```

Tune risky-trip probability thresholds on reviewed trips:

```powershell
conda run -n safe-driving-backend python scripts/tune_risk_thresholds.py --thresholds 0.3 0.4 0.5 0.6 0.7
```

## Data/versioning workflow

Each finalized trip now stores:

- `feature_version`
- `model_version`
- `processed_at`
- `risk_probability`
- `risk_level`
- `confidence`

Training datasets also preserve `label_source` so reviewed, weak, and synthetic labels stay distinguishable.
Phase 8 also writes a dataset summary report with label-source, risk-class, model-version, and feature-version counts to `artifacts/reports/dataset_summary_fv1.json`.

## Notes on speed values

The current payloads and synthetic datasets send `speed` values in km/h. The database column is historically named `speed_mps`, but the ML preprocessing configuration currently treats incoming values as km/h and converts them to m/s during feature extraction.

## Additional docs

- Sample ingestion/finalization flow: `SAMPLE_COLLECTION_PROCESS.md`

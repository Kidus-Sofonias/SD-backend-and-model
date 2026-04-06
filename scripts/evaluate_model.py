# File role: Offline model evaluation/reporting script.
# Loads the saved model and dataset, evaluates classification performance,
# and writes a detailed report including confusion matrix and optional ROC-AUC.
# Connects to:
# - app.ml.schemas
# - artifacts/models
# - artifacts/reports
# Key symbols/vars:
# - DATASET_PATH
# - MODELS_DIR
# - REPORTS_DIR
# - main

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from app.ml.schemas import FEATURE_COLUMNS_FV1, FEATURE_VERSION


DATASET_PATH = Path("artifacts/datasets/trip_features_fv1.csv")
MODELS_DIR = Path("artifacts/models")
REPORTS_DIR = Path("artifacts/reports")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        print(f"Dataset file {DATASET_PATH} not found. Run build_training_dataset.py first.")
        return

    df = pd.read_csv(DATASET_PATH).dropna(subset=["label_binary"])
    if len(df) < 2:
        print(f"Not enough data for evaluation: {len(df)} samples. Need at least 2.")
        return

    X = df[FEATURE_COLUMNS_FV1]
    y = df["label_binary"].astype(int)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2 if len(df) >= 20 else 0.5,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2 if len(df) >= 20 else 0.5,
            random_state=42,
            stratify=None,
        )

    metadata_files = sorted(MODELS_DIR.glob(f"metadata_{FEATURE_VERSION}_*.json"))
    if not metadata_files:
        raise FileNotFoundError("No model metadata found")

    latest_metadata = metadata_files[-1]
    metadata = json.loads(latest_metadata.read_text(encoding="utf-8"))
    model_version = metadata["model_version"]

    model_path = MODELS_DIR / f"model_{FEATURE_VERSION}_{model_version}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No model file found at {model_path}")

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    result = {
        "model_version": model_version,
        "feature_version": FEATURE_VERSION,
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
    }

    if hasattr(model, "predict_proba") and y_test.nunique() > 1:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        result["roc_auc"] = None if math.isnan(roc_auc) else float(roc_auc)
    else:
        result["roc_auc"] = None

    out_path = REPORTS_DIR / f"eval_{FEATURE_VERSION}_{model_version}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
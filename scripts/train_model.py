# File role: Offline training script for supervised ML models.
# Loads the generated dataset, trains candidate classifiers, compares metrics,
# and saves timestamped candidate artifacts plus metadata.

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from app.ml.schemas import FEATURE_COLUMNS_FV1, FEATURE_VERSION


DATASET_PATH = Path("artifacts/datasets/trip_features_fv1.csv")
MODELS_DIR = Path("artifacts/models")
REPORTS_DIR = Path("artifacts/reports")


def _build_version(model_key: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{model_key}_{stamp}"


def evaluate(y_true, y_pred, y_prob=None) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "risky_trip_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if y_prob is not None:
        metrics["brier_score"] = float(((y_prob - y_true) ** 2).mean())
    return metrics


def train_candidates(df: pd.DataFrame) -> dict[str, Any]:
    X = df[FEATURE_COLUMNS_FV1]
    y = df["label_binary"].astype(int)

    test_size = 0.2 if len(df) >= 20 else 0.5

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=None,
        )

    models = {
        "lr": LogisticRegression(max_iter=1000),
        "gb": GradientBoostingClassifier(random_state=42),
    }

    trained_candidates: list[dict[str, Any]] = []
    for model_key, model in models.items():
        version = _build_version(model_key)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        trained_candidates.append(
            {
                "model_key": model_key,
                "model_version": version,
                "model": model,
                "metrics": evaluate(y_test, y_pred, y_prob=y_prob),
            }
        )

    best = max(
        trained_candidates,
        key=lambda item: (
            item["metrics"]["risky_trip_f1"],
            -item["metrics"]["false_positive_rate"],
            -item["metrics"].get("brier_score", 1.0),
        ),
    )

    return {
        "candidates": trained_candidates,
        "best": best,
        "test_size": test_size,
        "row_count": int(len(df)),
        "class_distribution": {str(k): int(v) for k, v in df["label_binary"].value_counts().to_dict().items()},
        "label_tier_distribution": {
            str(k): int(v) for k, v in df["label_tier"].value_counts().to_dict().items()
        } if "label_tier" in df.columns else {},
    }


def _persist_candidate(candidate: dict[str, Any], training_summary: dict[str, Any]) -> None:
    version = str(candidate["model_version"])
    model = candidate["model"]
    metrics = candidate["metrics"]

    model_path = MODELS_DIR / f"model_{FEATURE_VERSION}_{version}.joblib"
    joblib.dump(model, model_path)

    feature_path = MODELS_DIR / f"feature_columns_{FEATURE_VERSION}.json"
    feature_path.write_text(json.dumps(FEATURE_COLUMNS_FV1, indent=2), encoding="utf-8")

    metadata = {
        "model_version": version,
        "model_key": candidate["model_key"],
        "feature_version": FEATURE_VERSION,
        "target": "label_binary",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "row_count": training_summary["row_count"],
        "class_distribution": training_summary["class_distribution"],
        "label_tier_distribution": training_summary["label_tier_distribution"],
        "metrics": metrics,
        "test_size": training_summary["test_size"],
    }

    metadata_path = MODELS_DIR / f"metadata_{FEATURE_VERSION}_{version}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_training() -> dict[str, Any] | None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        print(f"Dataset file {DATASET_PATH} not found. Run build_training_dataset.py first.")
        return None

    df = pd.read_csv(DATASET_PATH).dropna(subset=["label_binary"])

    if df.empty:
        print("No labeled rows found in dataset.")
        return None

    class_distribution = df["label_binary"].value_counts().to_dict()
    print(f"Class distribution: {class_distribution}")

    class_count = df["label_binary"].nunique()
    if class_count < 2:
        print(
            f"Not enough class diversity for training: {class_count} classes found. "
            "Need at least 2 (e.g., safe and risky trips)."
        )
        return None

    training_summary = train_candidates(df)
    candidates = training_summary["candidates"]

    for candidate in candidates:
        _persist_candidate(candidate, training_summary)

    summary = {
        candidate["model_version"]: candidate["metrics"]
        for candidate in candidates
    }
    best = training_summary["best"]

    report = {
        "feature_version": FEATURE_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_model_version": best["model_version"],
        "best_metrics": best["metrics"],
        "candidates": summary,
        "row_count": training_summary["row_count"],
        "class_distribution": training_summary["class_distribution"],
        "label_tier_distribution": training_summary["label_tier_distribution"],
    }
    report_path = REPORTS_DIR / f"train_{FEATURE_VERSION}_{best['model_version']}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Best model: {best['model_version']}")
    print(json.dumps(summary, indent=2))
    return report


def main():
    run_training()


if __name__ == "__main__":
    main()

# File role: Model competence benchmark.
# Runs the production candidates (LogisticRegression, GradientBoosting) against a
# wider field (RandomForest, RBF-SVM, k-NN, Gaussian Naive Bayes, DecisionTree)
# on the *same* stratified 5-fold splits and the same out-of-fold threshold
# tuning, then prints a comparison table and writes a JSON report.
# Reuses the CV/metric helpers from train_model_v2.py so every model is measured
# identically to the promoted one.
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.ml.schemas import FEATURE_COLUMNS_FV1, FEATURE_VERSION

from scripts.train_model_v2 import (
    DATASET_PATH,
    REPORTS_DIR,
    RISKY_F1_TARGET,
    _cross_validate,
    _metrics,  # noqa: F401 (re-exported for parity checks)
    _tune_threshold,  # noqa: F401
)

MODEL_KEY_ORDER = ["lr", "gb", "rf", "svm", "knn", "nb", "dt"]

# Display metadata for the report.
DESCRIPTIONS = {
    "lr": "LogisticRegression (scaled, balanced) [PRODUCTION]",
    "gb": "GradientBoosting",
    "rf": "RandomForest",
    "svm": "RBF Support Vector Machine (scaled)",
    "knn": "k-Nearest Neighbors (scaled)",
    "nb": "Gaussian Naive Bayes",
    "dt": "DecisionTree",
}


def _build_benchmark_models() -> dict[str, Any]:
    return {
        "lr": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "gb": GradientBoostingClassifier(
            random_state=42,
            n_estimators=120,
            learning_rate=0.08,
            max_depth=3,
            subsample=0.85,
            min_samples_leaf=2,
        ),
        "rf": RandomForestClassifier(
            random_state=42,
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
        ),
        "svm": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=42),
        ),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "nb": GaussianNB(),
        "dt": DecisionTreeClassifier(random_state=42, max_depth=5, class_weight="balanced"),
    }


def main() -> None:
    if not DATASET_PATH.exists():
        print(f"Dataset file {DATASET_PATH} not found. Run build_training_dataset.py first.")
        return None

    df = pd.read_csv(DATASET_PATH).dropna(subset=["label_binary"])
    X = df[FEATURE_COLUMNS_FV1]
    y = df["label_binary"].astype(int)

    print(f"Dataset rows: {len(df)}  class dist: {y.value_counts().to_dict()}")
    print(f"{'model':<6} {'riskyF1':>8} {'acc':>6} {'prec':>6} {'rec':>6} {'fpr':>6} {'fnr':>6} {'brier':>8} {'rocAUC':>8} {'prAUC':>8} {'thr':>5}")
    print("-" * 88)

    results: dict[str, Any] = {}
    for model_key in MODEL_KEY_ORDER:
        model = _build_benchmark_models()[model_key]
        cv = _cross_validate(model, X, y)
        oof = cv["oof"]
        tuning = cv["oof_threshold_tuning"]
        results[model_key] = {
            "description": DESCRIPTIONS[model_key],
            "cv": {
                "n_folds": cv["n_folds"],
                "oof_metrics": oof,
                "means": {k: v for k, v in cv.items() if k not in ("oof", "n_folds", "oof_threshold_tuning")},
                "oof_threshold_tuning": tuning,
            },
        }
        print(
            f"{model_key:<6} {oof.get('risky_trip_f1', 0):>8.3f} {oof.get('accuracy', 0):>6.3f} "
            f"{oof.get('precision', 0):>6.3f} {oof.get('recall', 0):>6.3f} "
            f"{oof.get('false_positive_rate', 0):>6.3f} {oof.get('false_negative_rate', 0):>6.3f} "
            f"{oof.get('brier_score', 1):>8.4f} {oof.get('roc_auc', 0) or 0:>8.3f} "
            f"{oof.get('pr_auc', 0) or 0:>8.3f} {tuning.get('threshold', 0.5):>5.2f}"
        )

    report = {
        "feature_version": FEATURE_VERSION,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "method": "stratified 5-fold CV, shared splits, OOF threshold tuning (max risky_f1 with fpr<=0.35)",
        "row_count": int(len(df)),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
        "risky_f1_target": RISKY_F1_TARGET,
        "results": results,
        "head_to_head": {
            "production_lr_beats_all_others_on_risky_f1": all(
                results["lr"]["cv"]["oof_metrics"]["risky_trip_f1"]
                >= results[k]["cv"]["oof_metrics"]["risky_trip_f1"]
                for k in MODEL_KEY_ORDER
                if k != "lr"
            ),
            "production_lr_lowest_brier": all(
                results["lr"]["cv"]["oof_metrics"]["brier_score"]
                <= results[k]["cv"]["oof_metrics"]["brier_score"]
                for k in MODEL_KEY_ORDER
                if k != "lr"
            ),
            "all_models_reach_80pct_target": all(
                results[k]["cv"]["oof_metrics"]["risky_trip_f1"] >= RISKY_F1_TARGET
                for k in MODEL_KEY_ORDER
            ),
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "benchmark_models.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    print("Head-to-head:", json.dumps(report["head_to_head"], indent=2))
    return report


if __name__ == "__main__":
    main()

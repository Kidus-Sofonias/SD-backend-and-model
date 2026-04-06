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

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from app.ml.model_registry import get_production_model_version
from app.ml.schemas import FEATURE_COLUMNS_FV1, FEATURE_VERSION


DATASET_PATH = Path("artifacts/datasets/trip_features_fv1.csv")
MODELS_DIR = Path("artifacts/models")
REPORTS_DIR = Path("artifacts/reports")


def _load_model(version: str):
    model_path = MODELS_DIR / f"model_{FEATURE_VERSION}_{version}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def _metrics(y_true, y_pred, y_prob=None) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
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


def _subset_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict[str, Any] | None:
    if X.empty:
        return None
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return _metrics(y, y_pred, y_prob=y_prob)


def _promotion_decision(current: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    current_brier = float(current.get("brier_score", 1.0))
    candidate_brier = float(candidate.get("brier_score", 1.0))

    checks = {
        "better_risky_trip_f1": bool(candidate["risky_trip_f1"] > current["risky_trip_f1"]),
        "false_positive_rate_not_worse": bool(candidate["false_positive_rate"] <= current["false_positive_rate"]),
        "acceptable_calibration": bool(candidate_brier <= max(current_brier + 0.02, 0.20)),
    }
    checks["promote"] = all(checks.values())
    return checks


def run_compare(current_version: str | None, candidate_version: str) -> dict[str, Any]:
    current_version = current_version or get_production_model_version()
    if not current_version:
        raise FileNotFoundError("No current production model found. Promote or specify --current first.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH).dropna(subset=["label_binary"])
    test_size = 0.2 if len(df) >= 20 else 0.5

    try:
        _, test_df = train_test_split(
            df,
            test_size=0.2 if len(df) >= 20 else 0.5,
            random_state=42,
            stratify=df["label_binary"].astype(int),
        )
    except ValueError:
        _, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=None,
        )

    X_test = test_df[FEATURE_COLUMNS_FV1]
    y_test = test_df["label_binary"].astype(int)

    current_model = _load_model(current_version)
    candidate_model = _load_model(candidate_version)

    current_pred = current_model.predict(X_test)
    candidate_pred = candidate_model.predict(X_test)
    current_prob = current_model.predict_proba(X_test)[:, 1] if hasattr(current_model, "predict_proba") else None
    candidate_prob = candidate_model.predict_proba(X_test)[:, 1] if hasattr(candidate_model, "predict_proba") else None

    reviewed_real_subset = (
        test_df[test_df["label_tier"] == "reviewed_real"]
        if "label_tier" in test_df.columns else test_df.iloc[0:0]
    )
    reviewed_subset = (
        reviewed_real_subset
        if not reviewed_real_subset.empty
        else test_df[test_df["label_tier"].isin(["reviewed_real", "reviewed_synthetic"])]
        if "label_tier" in test_df.columns else test_df.iloc[0:0]
    )
    risky_reviewed_subset = reviewed_subset[reviewed_subset["label_binary"].astype(int) == 1] if not reviewed_subset.empty else reviewed_subset

    result = {
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "feature_version": FEATURE_VERSION,
        "current_model_version": current_version,
        "candidate_model_version": candidate_version,
        "test_row_count": int(len(test_df)),
        "current": _metrics(y_test, current_pred, y_prob=current_prob),
        "candidate": _metrics(y_test, candidate_pred, y_prob=candidate_prob),
        "reviewed_real_subset": {
            "row_count": int(len(reviewed_real_subset)),
            "current": _subset_metrics(
                current_model,
                reviewed_real_subset[FEATURE_COLUMNS_FV1],
                reviewed_real_subset["label_binary"].astype(int),
            ),
            "candidate": _subset_metrics(
                candidate_model,
                reviewed_real_subset[FEATURE_COLUMNS_FV1],
                reviewed_real_subset["label_binary"].astype(int),
            ),
        },
        "reviewed_subset": {
            "row_count": int(len(reviewed_subset)),
            "current": _subset_metrics(
                current_model,
                reviewed_subset[FEATURE_COLUMNS_FV1],
                reviewed_subset["label_binary"].astype(int),
            ),
            "candidate": _subset_metrics(
                candidate_model,
                reviewed_subset[FEATURE_COLUMNS_FV1],
                reviewed_subset["label_binary"].astype(int),
            ),
        },
        "reviewed_risky_subset": {
            "row_count": int(len(risky_reviewed_subset)),
            "current_recall": float(
                recall_score(
                    risky_reviewed_subset["label_binary"].astype(int),
                    current_model.predict(risky_reviewed_subset[FEATURE_COLUMNS_FV1]),
                    zero_division=0,
                )
            ) if not risky_reviewed_subset.empty else None,
            "candidate_recall": float(
                recall_score(
                    risky_reviewed_subset["label_binary"].astype(int),
                    candidate_model.predict(risky_reviewed_subset[FEATURE_COLUMNS_FV1]),
                    zero_division=0,
                )
            ) if not risky_reviewed_subset.empty else None,
        },
    }
    result["promotion_check"] = _promotion_decision(result["current"], result["candidate"])

    out_path = REPORTS_DIR / f"compare_{FEATURE_VERSION}_{current_version}_vs_{candidate_version}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two trained model versions on the same dataset split")
    parser.add_argument("--current", required=False, help="Current production model version")
    parser.add_argument("--candidate", required=True, help="Candidate model version")
    args = parser.parse_args()

    result = run_compare(current_version=args.current, candidate_version=args.candidate)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pandas as pd

from app.ml.model_registry import get_production_model_version, model_path_for
from app.ml.schemas import FEATURE_COLUMNS_FV1, FEATURE_VERSION


DATASET_PATH = Path("artifacts/datasets/trip_features_fv1.csv")
REPORTS_DIR = Path("artifacts/reports")


def _safe_shift(baseline: float, recent: float) -> float:
    denominator = abs(baseline) if abs(baseline) > 1e-6 else 1.0
    return float((recent - baseline) / denominator)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH).dropna(subset=["label_binary"])
    if df.empty:
        raise RuntimeError("Dataset is empty; nothing to analyze.")

    df = df.sort_values(by="processed_at", kind="stable", na_position="last").reset_index(drop=True)
    split_index = max(1, int(len(df) * 0.7))
    baseline = df.iloc[:split_index]
    recent = df.iloc[split_index:]
    if recent.empty:
        recent = df.tail(max(1, min(20, len(df))))
        baseline = df.head(max(1, len(df) - len(recent)))

    feature_shifts = {}
    for column in FEATURE_COLUMNS_FV1:
        baseline_mean = float(baseline[column].mean())
        recent_mean = float(recent[column].mean())
        feature_shifts[column] = {
            "baseline_mean": baseline_mean,
            "recent_mean": recent_mean,
            "relative_mean_shift": _safe_shift(baseline_mean, recent_mean),
        }

    label_distribution = {
        "baseline": {str(k): int(v) for k, v in baseline["label_binary"].value_counts().to_dict().items()},
        "recent": {str(k): int(v) for k, v in recent["label_binary"].value_counts().to_dict().items()},
    }

    confidence = {
        "baseline_mean": float(baseline["confidence"].mean()),
        "recent_mean": float(recent["confidence"].mean()),
        "delta": float(recent["confidence"].mean() - baseline["confidence"].mean()),
    }

    disagreement = None
    production_version = get_production_model_version()
    if production_version:
        import joblib

        model = joblib.load(model_path_for(production_version))
        reviewed_real = df[df["label_tier"] == "reviewed_real"] if "label_tier" in df.columns else df.iloc[0:0]
        reviewed = (
            reviewed_real
            if not reviewed_real.empty
            else df[df["label_tier"].isin(["reviewed_real", "reviewed_synthetic"])]
            if "label_tier" in df.columns else df.iloc[0:0]
        )
        if not reviewed.empty:
            reviewed_pred = model.predict(reviewed[FEATURE_COLUMNS_FV1])
            reviewed_true = reviewed["label_binary"].astype(int)
            disagreement = {
                "reviewed_real_row_count": int(len(reviewed_real)),
                "reviewed_row_count": int(len(reviewed)),
                "disagreement_rate": float((reviewed_pred != reviewed_true).mean()),
            }

    report = {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "feature_version": FEATURE_VERSION,
        "production_model_version": production_version,
        "row_count": int(len(df)),
        "feature_distribution_shifts": feature_shifts,
        "label_distribution_shift": label_distribution,
        "confidence_shift": confidence,
        "reviewed_label_disagreement": disagreement,
    }

    out_path = REPORTS_DIR / f"drift_{FEATURE_VERSION}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

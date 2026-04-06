"""Phase 8 reviewed-trip analysis helpers and export entrypoint.

This module scores reviewed trips against the current or requested model so we
can inspect disagreements, threshold behavior, and confidence-bucket error
rates using the same reviewed-label source of truth across scripts.
"""

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
from sqlalchemy import select

from app.db.models.trip import Trip
from app.db.session import SessionLocal
from app.ml.model_registry import get_production_model_version, model_path_for
from app.ml.schemas import FEATURE_COLUMNS_FV1, FEATURE_VERSION
from scripts.reporting_utils import (
    build_confidence_bucket_report,
    build_model_mistake_log,
    build_threshold_report,
    confidence_band,
)


REPORTS_DIR = Path("artifacts/reports")
DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def _load_model(version: str):
    model_path = model_path_for(version)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def _load_breakdown(trip: Trip) -> dict[str, Any]:
    if not trip.score_breakdown:
        return {}
    try:
        loaded = json.loads(trip.score_breakdown)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _review_label_tier(reviewed_label_source: str | None) -> str:
    source = (reviewed_label_source or "reviewed_real").strip().lower()
    if "synthetic" in source:
        return "reviewed_synthetic"
    return "reviewed_real"


def load_reviewed_trip_rows(*, include_synthetic: bool = False) -> list[dict[str, Any]]:
    """Load finalized reviewed trips that still retain enough features for rescoring."""
    db = SessionLocal()
    try:
        trips = db.execute(
            select(Trip)
            .where(Trip.status == "completed", Trip.reviewed_label.is_not(None))
            .order_by(Trip.processed_at.desc(), Trip.started_at.desc())
        ).scalars().all()

        rows: list[dict[str, Any]] = []
        for trip in trips:
            label_tier = _review_label_tier(trip.reviewed_label_source)
            if label_tier == "reviewed_synthetic" and not include_synthetic:
                continue

            breakdown = _load_breakdown(trip)
            trip_features = breakdown.get("trip_features")
            if not isinstance(trip_features, dict):
                continue
            if any(column not in trip_features for column in FEATURE_COLUMNS_FV1):
                continue

            rows.append(
                {
                    "trip_id": trip.id,
                    "reviewed_label": int(trip.reviewed_label),
                    "reviewed_label_source": trip.reviewed_label_source or label_tier,
                    "label_tier": label_tier,
                    "confidence": trip.confidence,
                    "confidence_band": confidence_band(trip.confidence),
                    "reasons": breakdown.get("reasons", []),
                    "stored_model_version": trip.model_version,
                    "stored_feature_version": trip.feature_version,
                    "processed_at": trip.processed_at.isoformat() if trip.processed_at else None,
                    **{column: trip_features[column] for column in FEATURE_COLUMNS_FV1},
                }
            )
        return rows
    finally:
        db.close()


def score_reviewed_trip_rows(
    rows_or_df: list[dict[str, Any]] | pd.DataFrame,
    *,
    model_version: str | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame(rows_or_df) if not isinstance(rows_or_df, pd.DataFrame) else rows_or_df.copy()
    if df.empty:
        return df

    chosen_version = model_version or get_production_model_version()
    if not chosen_version:
        raise FileNotFoundError("No production model found. Train/promote a model or pass --model-version.")

    model = _load_model(chosen_version)
    feature_frame = df[FEATURE_COLUMNS_FV1]
    prediction = model.predict(feature_frame)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_frame)[:, 1]
    else:
        probabilities = prediction

    scored = df.copy()
    scored["prediction"] = [int(value) for value in prediction]
    scored["predicted_risk_probability"] = [float(value) for value in probabilities]
    scored["model_version"] = chosen_version
    scored["feature_version"] = FEATURE_VERSION
    return scored


def build_reviewed_model_analysis(
    rows_or_df: list[dict[str, Any]] | pd.DataFrame,
    *,
    threshold: float = 0.5,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    df = pd.DataFrame(rows_or_df) if not isinstance(rows_or_df, pd.DataFrame) else rows_or_df.copy()
    threshold_values = thresholds or DEFAULT_THRESHOLDS
    mistake_log = build_model_mistake_log(df, threshold=threshold)
    threshold_report = build_threshold_report(df, threshold_values)
    confidence_report = build_confidence_bucket_report(df, threshold=threshold)

    return {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(df)),
        "reviewed_real_row_count": int((df.get("label_tier") == "reviewed_real").sum()) if "label_tier" in df else 0,
        "reviewed_synthetic_row_count": int((df.get("label_tier") == "reviewed_synthetic").sum()) if "label_tier" in df else 0,
        "model_version": None if df.empty else df["model_version"].iloc[0],
        "feature_version": None if df.empty else df["feature_version"].iloc[0],
        "mistake_count": int(len(mistake_log)),
        "mistakes": mistake_log,
        "threshold_report": threshold_report,
        "confidence_bucket_report": confidence_report,
    }


def export_reviewed_model_analysis(
    *,
    model_version: str | None = None,
    include_synthetic: bool = False,
    threshold: float = 0.5,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_reviewed_trip_rows(include_synthetic=include_synthetic)
    scored = score_reviewed_trip_rows(rows, model_version=model_version)
    report = build_reviewed_model_analysis(scored, threshold=threshold, thresholds=thresholds)

    version = str(report["model_version"] or "unknown")
    report_path = REPORTS_DIR / f"reviewed_model_analysis_{FEATURE_VERSION}_{version}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    scored_export = scored.copy()
    if not scored_export.empty and "reasons" in scored_export.columns:
        scored_export["reasons"] = scored_export["reasons"].apply(json.dumps)
    scored_path = REPORTS_DIR / f"reviewed_model_scores_{FEATURE_VERSION}_{version}.csv"
    scored_export.to_csv(scored_path, index=False)

    mistakes_path = REPORTS_DIR / f"model_mistakes_{FEATURE_VERSION}_{version}.json"
    mistakes_path.write_text(json.dumps(report["mistakes"], indent=2), encoding="utf-8")

    report["report_path"] = str(report_path)
    report["scored_rows_path"] = str(scored_path)
    report["mistakes_path"] = str(mistakes_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export reviewed-trip mistakes and analysis for the current model")
    parser.add_argument("--model-version", required=False, help="Model version to analyze")
    parser.add_argument("--include-synthetic", action="store_true", help="Include reviewed synthetic labels")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold used for disagreement/error reporting")
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=None,
        help="Optional threshold sweep values. Example: --thresholds 0.3 0.4 0.5 0.6 0.7",
    )
    args = parser.parse_args()

    result = export_reviewed_model_analysis(
        model_version=args.model_version,
        include_synthetic=args.include_synthetic,
        threshold=args.threshold,
        thresholds=args.thresholds,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

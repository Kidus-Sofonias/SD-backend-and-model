"""Phase 8 reporting helpers for dataset, threshold, and mistake analysis.

These helpers stay intentionally small and data-frame friendly so the offline
training/reporting scripts can share one source of truth for:
- dataset composition summaries
- reviewed-label mistake logs
- threshold sweep reports
- confidence bucket error reporting
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


LOW_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.8


def _frame(rows_or_df: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(rows_or_df, pd.DataFrame):
        return rows_or_df.copy()
    return pd.DataFrame(rows_or_df)


def _count_series(series: pd.Series) -> dict[str, int]:
    cleaned = series.fillna("unknown").astype(str)
    return {str(key): int(value) for key, value in cleaned.value_counts().to_dict().items()}


def _risk_class_counts(series: pd.Series) -> dict[str, int]:
    counts = {0: 0, 1: 0}
    for value in series.dropna().astype(int):
        counts[value] = counts.get(value, 0) + 1
    return {
        "safe": int(counts.get(0, 0)),
        "risky": int(counts.get(1, 0)),
    }


def build_dataset_summary(
    rows_or_df: list[dict[str, Any]] | pd.DataFrame,
    *,
    selection_summary: dict[str, Any] | None = None,
    skipped_no_features: int = 0,
    skipped_unlabeled: int = 0,
) -> dict[str, Any]:
    df = _frame(rows_or_df)
    summary: dict[str, Any] = {
        "row_count": int(len(df)),
        "skipped_no_features": int(skipped_no_features),
        "skipped_unlabeled": int(skipped_unlabeled),
        "risk_class_counts": {"safe": 0, "risky": 0},
        "label_source_counts": {},
        "label_tier_counts": {},
        "model_version_counts": {},
        "feature_version_counts": {},
    }
    if selection_summary is not None:
        summary["selection_summary"] = selection_summary

    if df.empty:
        return summary

    if "label_binary" in df.columns:
        summary["risk_class_counts"] = _risk_class_counts(df["label_binary"])
    if "label_source" in df.columns:
        summary["label_source_counts"] = _count_series(df["label_source"])
    if "label_tier" in df.columns:
        summary["label_tier_counts"] = _count_series(df["label_tier"])
    if "model_version" in df.columns:
        summary["model_version_counts"] = _count_series(df["model_version"])
    if "feature_version" in df.columns:
        summary["feature_version_counts"] = _count_series(df["feature_version"])

    return summary


def prediction_from_probability(probability: float | None, *, threshold: float = 0.5) -> int | None:
    if probability is None or pd.isna(probability):
        return None
    return int(float(probability) >= float(threshold))


def confidence_band(confidence: float | None) -> str:
    if confidence is None or pd.isna(confidence):
        return "unknown"
    value = float(confidence)
    if value >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if value >= LOW_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def _normalize_reasons(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(loaded, list):
            return [str(item) for item in loaded]
        return [str(loaded)]
    return [str(value)]


def build_model_mistake_log(
    rows_or_df: list[dict[str, Any]] | pd.DataFrame,
    *,
    reviewed_label_col: str = "reviewed_label",
    probability_col: str = "predicted_risk_probability",
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    df = _frame(rows_or_df)
    if df.empty:
        return []

    mistakes: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        reviewed_label = row.get(reviewed_label_col)
        prediction = prediction_from_probability(row.get(probability_col), threshold=threshold)
        if reviewed_label is None or prediction is None:
            continue
        reviewed_value = int(reviewed_label)
        if reviewed_value == prediction:
            continue
        mistakes.append(
            {
                "trip_id": row.get("trip_id"),
                "reviewed_label": reviewed_value,
                "prediction": prediction,
                "probability": float(row.get(probability_col)),
                "confidence": float(row["confidence"]) if row.get("confidence") is not None else None,
                "reasons": _normalize_reasons(row.get("reasons")),
                "model_version": row.get("model_version"),
                "feature_version": row.get("feature_version"),
            }
        )

    mistakes.sort(key=lambda item: (str(item.get("trip_id")), -abs(float(item["probability"]) - threshold)))
    return mistakes


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    true_values = y_true.astype(int).tolist()
    pred_values = y_pred.astype(int).tolist()

    tn = fp = fn = tp = 0
    for true_value, pred_value in zip(true_values, pred_values, strict=False):
        if true_value == 0 and pred_value == 0:
            tn += 1
        elif true_value == 0 and pred_value == 1:
            fp += 1
        elif true_value == 1 and pred_value == 0:
            fn += 1
        else:
            tp += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(false_positive_rate),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def build_threshold_report(
    rows_or_df: list[dict[str, Any]] | pd.DataFrame,
    thresholds: list[float],
    *,
    reviewed_label_col: str = "reviewed_label",
    probability_col: str = "predicted_risk_probability",
) -> dict[str, Any]:
    df = _frame(rows_or_df)
    if df.empty:
        return {
            "row_count": 0,
            "thresholds": [],
        }

    working = df.dropna(subset=[reviewed_label_col, probability_col]).copy()
    if working.empty:
        return {
            "row_count": 0,
            "thresholds": [],
        }

    y_true = working[reviewed_label_col].astype(int)
    results: list[dict[str, Any]] = []
    for threshold in thresholds:
        y_pred = working[probability_col].apply(lambda value: prediction_from_probability(value, threshold=threshold))
        metrics = _binary_metrics(y_true, y_pred.astype(int))
        metrics["threshold"] = float(threshold)
        metrics["row_count"] = int(len(working))
        results.append(metrics)

    return {
        "row_count": int(len(working)),
        "thresholds": results,
    }


def build_confidence_bucket_report(
    rows_or_df: list[dict[str, Any]] | pd.DataFrame,
    *,
    reviewed_label_col: str = "reviewed_label",
    probability_col: str = "predicted_risk_probability",
    confidence_col: str = "confidence",
    threshold: float = 0.5,
) -> dict[str, Any]:
    df = _frame(rows_or_df)
    if df.empty:
        return {"row_count": 0, "buckets": {}}

    working = df.dropna(subset=[reviewed_label_col, probability_col]).copy()
    if working.empty:
        return {"row_count": 0, "buckets": {}}

    working["confidence_band"] = working[confidence_col].apply(confidence_band)
    working["prediction"] = working[probability_col].apply(
        lambda value: prediction_from_probability(value, threshold=threshold)
    )
    working["is_error"] = working["prediction"].astype(int) != working[reviewed_label_col].astype(int)

    bucket_report: dict[str, Any] = {}
    for band in ["low", "medium", "high", "unknown"]:
        band_df = working[working["confidence_band"] == band]
        if band_df.empty:
            bucket_report[band] = {
                "row_count": 0,
                "error_count": 0,
                "error_rate": None,
                "confusion_matrix": [[0, 0], [0, 0]],
            }
            continue
        metrics = _binary_metrics(
            band_df[reviewed_label_col].astype(int),
            band_df["prediction"].astype(int),
        )
        bucket_report[band] = {
            "row_count": int(len(band_df)),
            "error_count": int(band_df["is_error"].sum()),
            "error_rate": float(band_df["is_error"].mean()),
            "confusion_matrix": metrics["confusion_matrix"],
        }

    return {
        "row_count": int(len(working)),
        "buckets": bucket_report,
    }

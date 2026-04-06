from __future__ import annotations

import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db.models.trip import Trip
from scripts.build_training_dataset import choose_label
from scripts.reporting_utils import (
    build_confidence_bucket_report,
    build_model_mistake_log,
    build_threshold_report,
)


def _trip(**overrides) -> Trip:
    payload = {
        "id": str(uuid.uuid4()),
        "user_id": str(uuid.uuid4()),
        "started_at": datetime.fromisoformat("2026-03-24T11:06:00+00:00"),
        "ended_at": datetime.fromisoformat("2026-03-24T11:16:00+00:00"),
        "status": "completed",
        "reviewed_label": None,
        "reviewed_label_source": None,
    }
    payload.update(overrides)
    return Trip(**payload)


def test_choose_label_prioritizes_reviewed_real_before_other_sources() -> None:
    trip = _trip(reviewed_label=0, reviewed_label_source="human_review")

    label, source, tier = choose_label(
        trip=trip,
        rule_score=40,
        reviewed_labels={trip.id: 1},
        synthetic_labels={trip.id: 1},
    )

    assert label == 0
    assert source == "human_review"
    assert tier == "reviewed_real"


def test_choose_label_uses_reviewed_registry_before_weak_or_synthetic() -> None:
    trip = _trip()

    label, source, tier = choose_label(
        trip=trip,
        rule_score=40,
        reviewed_labels={trip.id: 0},
        synthetic_labels={trip.id: 1},
    )

    assert label == 0
    assert source == "reviewed_registry"
    assert tier == "reviewed_real"


def test_build_model_mistake_log_returns_review_disagreements() -> None:
    df = pd.DataFrame(
        [
            {
                "trip_id": "trip-a",
                "reviewed_label": 1,
                "predicted_risk_probability": 0.20,
                "confidence": 0.91,
                "reasons": ["hard_brake", "speed_variation"],
                "model_version": "lr_phase8",
                "feature_version": "fv1",
            },
            {
                "trip_id": "trip-b",
                "reviewed_label": 0,
                "predicted_risk_probability": 0.10,
                "confidence": 0.73,
                "reasons": ["steady_trip"],
                "model_version": "lr_phase8",
                "feature_version": "fv1",
            },
        ]
    )

    mistakes = build_model_mistake_log(df, threshold=0.5)

    assert len(mistakes) == 1
    assert mistakes[0]["trip_id"] == "trip-a"
    assert mistakes[0]["reviewed_label"] == 1
    assert mistakes[0]["prediction"] == 0
    assert mistakes[0]["probability"] == 0.20


def test_build_threshold_report_sweeps_multiple_thresholds() -> None:
    df = pd.DataFrame(
        [
            {"trip_id": "a", "reviewed_label": 1, "predicted_risk_probability": 0.90},
            {"trip_id": "b", "reviewed_label": 1, "predicted_risk_probability": 0.55},
            {"trip_id": "c", "reviewed_label": 0, "predicted_risk_probability": 0.45},
            {"trip_id": "d", "reviewed_label": 0, "predicted_risk_probability": 0.10},
        ]
    )

    report = build_threshold_report(df, [0.4, 0.6])

    assert report["row_count"] == 4
    assert [item["threshold"] for item in report["thresholds"]] == [0.4, 0.6]
    assert report["thresholds"][0]["confusion_matrix"] == [[1, 1], [0, 2]]
    assert report["thresholds"][0]["recall"] == 1.0
    assert report["thresholds"][1]["confusion_matrix"] == [[2, 0], [1, 1]]
    assert report["thresholds"][1]["precision"] == 1.0


def test_build_confidence_bucket_report_groups_error_rates_by_band() -> None:
    df = pd.DataFrame(
        [
            {
                "trip_id": "low-band",
                "reviewed_label": 1,
                "predicted_risk_probability": 0.10,
                "confidence": 0.30,
            },
            {
                "trip_id": "medium-band",
                "reviewed_label": 0,
                "predicted_risk_probability": 0.80,
                "confidence": 0.65,
            },
            {
                "trip_id": "high-band",
                "reviewed_label": 1,
                "predicted_risk_probability": 0.95,
                "confidence": 0.92,
            },
        ]
    )

    report = build_confidence_bucket_report(df, threshold=0.5)

    assert report["row_count"] == 3
    assert report["buckets"]["low"]["row_count"] == 1
    assert report["buckets"]["low"]["error_rate"] == 1.0
    assert report["buckets"]["medium"]["row_count"] == 1
    assert report["buckets"]["medium"]["error_rate"] == 1.0
    assert report["buckets"]["high"]["row_count"] == 1
    assert report["buckets"]["high"]["error_rate"] == 0.0

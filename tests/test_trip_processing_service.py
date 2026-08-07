from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db.base import Base
from app.db.models.driving_event import DrivingEvent
from app.db.models.sensor_sample import SensorSample
from app.db.models.trip import Trip
from app.db.models.user import User
from app.ml.auto_retrain import milestone_for_completed_trips, should_request_auto_retrain
from app.repositories.user_repository import UserRecord
from app.services.trip_processing_service import (
    ML_SCORE_BLEND_WEIGHT,
    ML_WEIGHT_MAX,
    ML_WEIGHT_MIN,
    TripProcessingService,
    adaptive_ml_blend_weight,
)


def _make_session(tmp_path: Path) -> Session:
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)
    return session_factory()


def _seed_trip_with_samples(db: Session, dataset_name: str) -> tuple[str, str]:
    user_id = str(uuid.uuid4())
    trip_id = str(uuid.uuid4())

    db.add(User(id=user_id, email=f"{user_id}@example.com", password_hash="hashed"))
    db.add(
        Trip(
            id=trip_id,
            user_id=user_id,
            started_at=datetime.fromisoformat("2026-03-24T11:06:00+00:00"),
            ended_at=None,
            status="completed",
        )
    )

    dataset_path = BACKEND_ROOT / "artifacts" / "datasets" / "risky_batch" / dataset_name
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))

    for sample in payload["samples"]:
        # JSON test data has speed in km/h (legacy format); DB stores m/s
        speed_mps = sample["speed"] / 3.6 if sample.get("speed") is not None else None
        db.add(
            SensorSample(
                user_id=user_id,
                trip_id=trip_id,
                ts=datetime.fromisoformat(sample["timestamp"].replace("Z", "+00:00")),
                speed_mps=speed_mps,
                lat=sample.get("lat"),
                lon=sample.get("lon"),
                ax=sample.get("ax"),
                ay=sample.get("ay"),
                az=sample.get("az"),
                gx=sample.get("gx"),
                gy=sample.get("gy"),
                gz=sample.get("gz"),
            )
        )

    db.commit()
    return user_id, trip_id


def _seed_trip_with_too_few_samples(db: Session, sample_count: int = 1) -> tuple[str, str]:
    user_id = str(uuid.uuid4())
    trip_id = str(uuid.uuid4())
    base_ts = datetime(2026, 3, 24, 11, 6, 0, tzinfo=timezone.utc)

    db.add(User(id=user_id, email=f"{user_id}@example.com", password_hash="hashed"))
    db.add(
        Trip(
            id=trip_id,
            user_id=user_id,
            started_at=base_ts,
            ended_at=base_ts + timedelta(minutes=1),
            status="completed",
        )
    )

    for i in range(sample_count):
        ts = base_ts + timedelta(seconds=i)
        db.add(
            SensorSample(
                user_id=user_id,
                trip_id=trip_id,
                ts=ts,
                speed_mps=10.0 + i,
                lat=1.0,
                lon=1.0,
                ax=0.1,
                ay=0.1,
                az=9.8,
                gx=0.01,
                gy=0.01,
                gz=0.01,
            )
        )

    db.commit()
    return user_id, trip_id


def test_finalize_trip_persists_score_breakdown_and_generated_events(tmp_path: Path) -> None:
    db = _make_session(tmp_path)
    user_id, trip_id = _seed_trip_with_samples(db, "risky_trip_240_samples_1.json")

    result = TripProcessingService(db).finalize_trip(user_id=user_id, trip_id=trip_id, delete_raw=False)

    assert result["trip_id"] == trip_id
    assert result["score"] is not None
    assert result["feature_version"] == "fv1"
    assert result["model_version"] is not None
    assert result["risk_level"] in {"low", "medium", "high"}
    assert result["risk_probability"] is not None
    assert result["decision_source"] in {"ml_with_rules", "rules_fallback"}
    assert "rule_breakdown" in result["breakdown"]

    trip = db.execute(select(Trip).where(Trip.id == trip_id)).scalar_one()
    assert trip.processed_at is not None
    assert trip.score == result["score"]
    assert trip.risk_level == result["risk_level"]
    assert trip.risk_probability == result["risk_probability"]
    assert trip.raw_deleted is False

    generated_events = db.execute(select(DrivingEvent).where(DrivingEvent.trip_id == trip_id)).scalars().all()
    assert len(generated_events) == result["events_generated"]
    assert any(event.occurred_at is not None for event in generated_events)
    assert any(event.lat is not None and event.lon is not None for event in generated_events)


def test_finalize_trip_with_delete_raw_removes_sensor_rows(tmp_path: Path) -> None:
    db = _make_session(tmp_path)
    user_id, trip_id = _seed_trip_with_samples(db, "risky_trip_240_samples_2.json")

    result = TripProcessingService(db).finalize_trip(user_id=user_id, trip_id=trip_id, delete_raw=True)

    assert result["raw_deleted"] is True

    remaining_samples = db.execute(select(SensorSample).where(SensorSample.trip_id == trip_id)).scalars().all()
    assert remaining_samples == []


def test_finalize_trip_returns_cached_result_after_processing(tmp_path: Path) -> None:
    db = _make_session(tmp_path)
    user_id, trip_id = _seed_trip_with_samples(db, "risky_trip_240_samples_3.json")

    service = TripProcessingService(db)
    first = service.finalize_trip(user_id=user_id, trip_id=trip_id, delete_raw=False)
    second = service.finalize_trip(user_id=user_id, trip_id=trip_id, delete_raw=False)

    assert first["score"] == second["score"]
    assert second["already_processed"] is True


def test_finalize_trip_falls_back_to_rules_when_model_prediction_fails(tmp_path: Path, monkeypatch) -> None:
    db = _make_session(tmp_path)
    user_id, trip_id = _seed_trip_with_samples(db, "risky_trip_240_samples_1.json")

    service = TripProcessingService(db)

    def _boom(_trip_features: dict) -> dict:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(service.model_scorer, "predict", _boom)

    result = service.finalize_trip(user_id=user_id, trip_id=trip_id, delete_raw=False)

    assert result["decision_source"] == "rules_fallback"
    assert result["breakdown"]["rule_score"] == result["score"]
    assert "ml_error" in result["breakdown"]["rule_breakdown"]


def test_finalize_trip_deletes_trip_when_samples_are_insufficient(tmp_path: Path) -> None:
    db = _make_session(tmp_path)
    user_id, trip_id = _seed_trip_with_too_few_samples(db, sample_count=1)

    result = TripProcessingService(db).finalize_trip(user_id=user_id, trip_id=trip_id, delete_raw=False)

    assert result["score"] is None
    assert result["breakdown"]["error"] == "not_enough_samples"
    assert result["breakdown"]["trip_deleted"] is True

    trip = db.execute(select(Trip).where(Trip.id == trip_id)).scalar_one_or_none()
    samples = db.execute(select(SensorSample).where(SensorSample.trip_id == trip_id)).scalars().all()
    events = db.execute(select(DrivingEvent).where(DrivingEvent.trip_id == trip_id)).scalars().all()

    assert trip is None
    assert samples == []
    assert events == []


def test_finalize_trip_scores_trip_at_minimum_sample_threshold(tmp_path: Path) -> None:
    db = _make_session(tmp_path)
    user_id, trip_id = _seed_trip_with_too_few_samples(db, sample_count=12)

    result = TripProcessingService(db).finalize_trip(user_id=user_id, trip_id=trip_id, delete_raw=False)

    assert result["score"] is not None
    assert result["breakdown"].get("error") != "not_enough_samples"

    trip = db.execute(select(Trip).where(Trip.id == trip_id)).scalar_one_or_none()
    assert trip is not None
    assert trip.processed_at is not None


def test_finalize_trip_schedules_auto_retrain_after_successful_finalize(tmp_path: Path, monkeypatch) -> None:
    db = _make_session(tmp_path)
    user_id, trip_id = _seed_trip_with_samples(db, "risky_trip_240_samples_1.json")
    service = TripProcessingService(db)
    scheduled_counts: list[int] = []

    monkeypatch.setattr(service, "_count_completed_trips", lambda: 100)
    monkeypatch.setattr(
        "app.services.trip_processing_service.maybe_schedule_auto_retrain",
        lambda *, completed_trip_count: scheduled_counts.append(completed_trip_count) or True,
    )

    service.finalize_trip(user_id=user_id, trip_id=trip_id, delete_raw=False)

    assert scheduled_counts == [100]


def test_compute_final_score_blends_rule_score_with_ml_probability(tmp_path: Path) -> None:
    db = _make_session(tmp_path)
    service = TripProcessingService(db)

    score = service._compute_final_score(rule_score=88, ml_prediction=1, ml_risk_probability=0.82, confidence=0.92)
    risk_probability = service._risk_probability_from_score(score=score, ml_risk_probability=0.82)

    assert score is not None
    assert 40 < score < 88
    assert risk_probability is not None
    assert 0.35 < risk_probability < 0.82


def test_compute_final_score_pulls_extreme_scores_toward_neutral_when_confidence_is_low(tmp_path: Path) -> None:
    db = _make_session(tmp_path)
    service = TripProcessingService(db)

    high_score = service._compute_final_score(rule_score=100, ml_prediction=None, ml_risk_probability=None, confidence=0.35)
    low_score = service._compute_final_score(rule_score=20, ml_prediction=None, ml_risk_probability=None, confidence=0.35)

    assert high_score is not None
    assert low_score is not None
    assert 65 <= high_score < 100
    assert 20 < low_score <= 45


def test_compute_final_score_uses_adaptive_blend_weight_when_provided(tmp_path: Path) -> None:
    db = _make_session(tmp_path)
    service = TripProcessingService(db)

    # ml probability 0.5 -> ml score 50; rule score 80
    score_low_weight = service._compute_final_score(
        rule_score=80,
        ml_prediction=1,
        ml_risk_probability=0.5,
        confidence=0.9,
        ml_blend_weight=0.2,
    )
    score_high_weight = service._compute_final_score(
        rule_score=80,
        ml_prediction=1,
        ml_risk_probability=0.5,
        confidence=0.9,
        ml_blend_weight=0.5,
    )

    assert score_low_weight is not None
    assert score_high_weight is not None
    # 0.8*80 + 0.2*50 = 74 ; 0.5*80 + 0.5*50 = 65
    assert score_low_weight > score_high_weight
    assert score_low_weight == 74
    assert score_high_weight == 65


def test_adaptive_weight_increases_with_confidence(tmp_path: Path) -> None:
    calibration = {
        "brier_score": 0.05,
        "risky_trip_f1": 0.9,
        "false_positive_rate": 0.1,
    }

    low_conf = adaptive_ml_blend_weight(confidence=0.5, calibration_metrics=calibration)
    high_conf = adaptive_ml_blend_weight(confidence=0.9, calibration_metrics=calibration)

    assert high_conf > low_conf
    assert ML_WEIGHT_MIN <= low_conf <= ML_WEIGHT_MAX
    assert ML_WEIGHT_MIN <= high_conf <= ML_WEIGHT_MAX


def test_adaptive_weight_increases_with_model_calibration(tmp_path: Path) -> None:
    poor_calibration = {
        "brier_score": 0.24,
        "risky_trip_f1": 0.56,
        "false_positive_rate": 0.34,
    }
    good_calibration = {
        "brier_score": 0.01,
        "risky_trip_f1": 0.98,
        "false_positive_rate": 0.0,
    }

    poor_weight = adaptive_ml_blend_weight(confidence=0.9, calibration_metrics=poor_calibration)
    good_weight = adaptive_ml_blend_weight(confidence=0.9, calibration_metrics=good_calibration)

    assert good_weight > poor_weight
    # Well-calibrated model should exceed the old constant; poorly calibrated should not.
    assert good_weight > ML_SCORE_BLEND_WEIGHT
    assert poor_weight < ML_SCORE_BLEND_WEIGHT


def test_adaptive_weight_bounded_and_unknown_calibration_neutral(tmp_path: Path) -> None:
    # Gate-worst calibration with zero confidence must floor at the minimum.
    gate_worst = {
        "brier_score": 0.25,
        "risky_trip_f1": 0.55,
        "false_positive_rate": 0.35,
    }
    worst = adaptive_ml_blend_weight(confidence=0.0, calibration_metrics=gate_worst)
    # Perfect calibration at max confidence: 0.35 * 1.0 * (0.6 + 0.8*1.0) = 0.49
    perfect = {
        "brier_score": 0.0,
        "risky_trip_f1": 1.0,
        "false_positive_rate": 0.0,
    }
    best = adaptive_ml_blend_weight(confidence=1.0, calibration_metrics=perfect)

    assert worst == ML_WEIGHT_MIN
    assert best == pytest.approx(0.49)  # 0.35 * 1.0 * (0.6 + 0.8*1.0)
    assert ML_WEIGHT_MIN <= best <= ML_WEIGHT_MAX

    # Unknown calibration defaults to a moderate (not extreme) weight.
    unknown = adaptive_ml_blend_weight(confidence=0.9, calibration_metrics=None)
    assert ML_WEIGHT_MIN <= unknown <= ML_WEIGHT_MAX
    assert ML_SCORE_BLEND_WEIGHT <= unknown


def test_adaptive_weight_saturates_at_medium_confidence(tmp_path: Path) -> None:
    calibration = {"brier_score": 0.05, "risky_trip_f1": 0.9, "false_positive_rate": 0.1}

    at_medium = adaptive_ml_blend_weight(confidence=0.8, calibration_metrics=calibration)
    beyond_medium = adaptive_ml_blend_weight(confidence=1.0, calibration_metrics=calibration)

    # Confidence factor saturates at 1.0 once confidence >= 0.8.
    assert at_medium == beyond_medium


def test_review_surfaces_ml_blend_weight(tmp_path: Path) -> None:
    db = _make_session(tmp_path)
    user_id, trip_id = _seed_trip_with_samples(db, "risky_trip_240_samples_1.json")

    service = TripProcessingService(db)
    result = service.finalize_trip(user_id=user_id, trip_id=trip_id, delete_raw=False)

    admin = UserRecord(id="admin-1", email="admin@example.com", password_hash="hashed", role="admin")

    review = service.get_trip_review(actor=admin, trip_id=trip_id)
    assert "ml_blend_weight" in review
    assert review["ml_blend_weight"] == result.get("ml_blend_weight")

    items = service.list_review_dashboard(actor=admin, limit=10)
    dashboard_item = next((item for item in items if item["trip_id"] == trip_id), None)
    assert dashboard_item is not None
    assert "ml_blend_weight" in dashboard_item
    assert dashboard_item["ml_blend_weight"] == result.get("ml_blend_weight")


def test_auto_retrain_milestone_only_matches_exact_interval() -> None:
    assert milestone_for_completed_trips(completed_trip_count=99, trip_interval=100) is None
    assert milestone_for_completed_trips(completed_trip_count=100, trip_interval=100) == 100
    assert milestone_for_completed_trips(completed_trip_count=200, trip_interval=100) == 200


def test_auto_retrain_request_guard_skips_duplicate_milestones() -> None:
    assert should_request_auto_retrain(
        completed_trip_count=100,
        trip_interval=100,
        last_requested_milestone=0,
    ) is True
    assert should_request_auto_retrain(
        completed_trip_count=100,
        trip_interval=100,
        last_requested_milestone=100,
    ) is False
    assert should_request_auto_retrain(
        completed_trip_count=200,
        trip_interval=100,
        last_requested_milestone=100,
    ) is True

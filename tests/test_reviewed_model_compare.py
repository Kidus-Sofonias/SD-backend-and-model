"""Phase 8b — reviewed-trip model compare wiring tests.

Covers:
- load_reviewed_trip_rows backfills log_vehicle_mass_kg for pre-8b trips
  (15-feature stored features) instead of dropping them
- score_reviewed_trip_rows selects each model's own trained columns, so the
  old 15-feature model can score rows that now carry 16 columns
- compare_reviewed_models produces a head-to-head report with threshold sweeps
"""

from __future__ import annotations

import json
import math
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models.trip import Trip
from app.ml.schemas import FEATURE_COLUMNS_FV1, FEATURE_VERSION


def _session_factory(tmp_path: Path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'reviewed-compare.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


def _feature_row(mass_log: float | None = None) -> dict:
    row = {column: 0.5 for column in FEATURE_COLUMNS_FV1}
    # None (default) simulates a PRE-8b trip: the vehicle column is absent.
    if mass_log is not None:
        row["log_vehicle_mass_kg"] = mass_log
    else:
        row.pop("log_vehicle_mass_kg")
    return row


def _seed_trip(db, *, trip_id: str, label: int, features: dict, source: str = "human_review") -> None:
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    db.add(
        Trip(
            id=trip_id,
            user_id=str(uuid.uuid4()),
            started_at=now,
            ended_at=now + timedelta(minutes=10),
            status="completed",
            score=70,
            score_breakdown=json.dumps({"trip_features": features}),
            feature_version=FEATURE_VERSION,
            model_version="old-v1",
            confidence=0.9,
            processed_at=now,
            reviewed_label=label,
            reviewed_label_source=source,
            reviewed_at=now,
        )
    )
    db.flush()


def test_load_backfills_log_vehicle_mass_for_pre8b_trips(tmp_path: Path, monkeypatch) -> None:
    session_factory = _session_factory(tmp_path)
    db = session_factory()
    # Pre-8b trip: stored features WITHOUT log_vehicle_mass_kg (15-column contract).
    _seed_trip(db, trip_id="pre8b", label=1, features=_feature_row())
    # Post-8b trip: has the vehicle feature.
    _seed_trip(db, trip_id="post8b", label=0, features=_feature_row(mass_log=round(math.log10(18000.0), 4)))
    db.commit()
    db.close()

    import scripts.reviewed_model_analysis as rma

    monkeypatch.setattr(rma, "SessionLocal", session_factory)
    rows = rma.load_reviewed_trip_rows()
    assert len(rows) == 2
    by_id = {row["trip_id"]: row for row in rows}
    # Pre-8b trip backfilled with the universal sedan default.
    assert by_id["pre8b"]["log_vehicle_mass_kg"] == round(math.log10(1400.0), 4)
    # Post-8b trip keeps its truck mass.
    assert by_id["post8b"]["log_vehicle_mass_kg"] == round(math.log10(18000.0), 4)
    assert by_id["pre8b"]["label_tier"] == "reviewed_real"


def test_score_selects_model_own_columns(tmp_path: Path, monkeypatch) -> None:
    # A 15-feature model (pre-8b) must score rows that carry 16 columns.
    old_columns = [c for c in FEATURE_COLUMNS_FV1 if c != "log_vehicle_mass_kg"]
    rng = __import__("numpy").random.RandomState(42)
    X = rng.rand(12, len(old_columns))
    y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    model = LogisticRegression(max_iter=500)
    model.fit(pd.DataFrame(X, columns=old_columns), y)

    model_path = tmp_path / "old_model.joblib"
    joblib.dump(model, model_path)

    import scripts.reviewed_model_analysis as rma

    monkeypatch.setattr(rma, "model_path_for", lambda version: model_path)
    rows = [
        {**{c: 0.5 for c in old_columns}, "log_vehicle_mass_kg": round(math.log10(1400.0), 4),
         "reviewed_label": 1, "label_tier": "reviewed_real",
         "trip_id": "t1", "reasons": [], "confidence": 0.9, "confidence_band": "high"}
    ]
    scored = rma.score_reviewed_trip_rows(rows, model_version="old")
    assert len(scored) == 1
    assert scored.iloc[0]["model_version"] == "old"
    assert "prediction" in scored.columns


def test_compare_reviewed_models_writes_report(tmp_path: Path, monkeypatch) -> None:
    session_factory = _session_factory(tmp_path)
    db = session_factory()
    for i in range(6):
        _seed_trip(
            db,
            trip_id=f"t{i}",
            label=1 if i >= 3 else 0,
            features=_feature_row(mass_log=round(math.log10(1400.0), 4)),
        )
    db.commit()
    db.close()

    import scripts.reviewed_model_analysis as rma

    monkeypatch.setattr(rma, "SessionLocal", session_factory)
    # Two trivial-but-distinct models so per-model column selection is exercised.
    rng = __import__("numpy").random.RandomState(0)
    X = rng.rand(12, len(FEATURE_COLUMNS_FV1))
    y = [0] * 6 + [1] * 6
    m_old = LogisticRegression(max_iter=500)
    m_old.fit(pd.DataFrame(X, columns=FEATURE_COLUMNS_FV1), y)
    m_new = LogisticRegression(max_iter=500)
    m_new.fit(pd.DataFrame(X, columns=FEATURE_COLUMNS_FV1), y)

    paths = {"old": tmp_path / "old.joblib", "new": tmp_path / "new.joblib"}
    joblib.dump(m_old, paths["old"])
    joblib.dump(m_new, paths["new"])
    monkeypatch.setattr(rma, "model_path_for", lambda version: paths[version])

    monkeypatch.setattr(rma, "REPORTS_DIR", tmp_path)

    result = rma.compare_reviewed_models(
        current_version="old",
        candidate_version="new",
        thresholds=[0.3, 0.5, 0.7],
    )
    assert result["current_model_version"] == "old"
    assert result["candidate_model_version"] == "new"
    assert result["row_count"] == 6
    assert result["reviewed_real_row_count"] == 6
    assert len(result["current"]["threshold_report"]["thresholds"]) == 3
    written = tmp_path / f"reviewed_model_compare_{FEATURE_VERSION}_old_vs_new.json"
    assert written.exists()

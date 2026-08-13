"""Review-gated auto-retrain and label-tier tests (hackathon feature).

Covers:
- review_milestone_for_count: batch-safe review milestones
- choose_label: demo_review never masquerades as human ground truth
- TripProcessingService._count_reviewed_real_trips: source filtering
- maybe_schedule_auto_retrain: schedules on the review trigger
"""

from __future__ import annotations

import sys
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db.base import Base
from app.db.models.trip import Trip
from app.db.models.user import User
from app.ml import auto_retrain
from app.ml.auto_retrain import review_milestone_for_count
from app.services.trip_processing_service import TripProcessingService
from scripts.build_training_dataset import choose_label


def _make_session(tmp_path: Path) -> Session:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'review.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)
    return factory()


# ---------------------------------------------------------------------------
# review_milestone_for_count
# ---------------------------------------------------------------------------


def test_review_milestone_below_threshold_is_none() -> None:
    assert (
        review_milestone_for_count(
            reviewed_trip_count=29, min_reviewed=30, trip_interval=100
        )
        is None
    )


def test_review_milestone_fires_at_threshold() -> None:
    assert (
        review_milestone_for_count(
            reviewed_trip_count=30, min_reviewed=30, trip_interval=100
        )
        == 30
    )


def test_review_milestone_is_batch_safe() -> None:
    # An admin reviewing 15 trips in one sitting skips 30 -> 45; the bucket
    # still fires exactly once (caller's last-requested guard dedupes).
    assert (
        review_milestone_for_count(
            reviewed_trip_count=45, min_reviewed=30, trip_interval=100
        )
        == 30
    )
    assert (
        review_milestone_for_count(
            reviewed_trip_count=129, min_reviewed=30, trip_interval=100
        )
        == 30
    )


def test_review_milestone_advances_per_interval() -> None:
    assert (
        review_milestone_for_count(
            reviewed_trip_count=130, min_reviewed=30, trip_interval=100
        )
        == 130
    )
    assert (
        review_milestone_for_count(
            reviewed_trip_count=231, min_reviewed=30, trip_interval=100
        )
        == 230
    )


def test_review_milestone_disabled_when_min_is_zero() -> None:
    assert (
        review_milestone_for_count(
            reviewed_trip_count=500, min_reviewed=0, trip_interval=100
        )
        is None
    )


def test_review_milestone_none_count_is_none() -> None:
    assert (
        review_milestone_for_count(
            reviewed_trip_count=None, min_reviewed=30, trip_interval=100
        )
        is None
    )


# ---------------------------------------------------------------------------
# choose_label tiers
# ---------------------------------------------------------------------------


def _trip(reviewed_label=None, source=None, trip_id="t-1"):
    return types.SimpleNamespace(
        id=trip_id,
        reviewed_label=reviewed_label,
        reviewed_label_source=source,
    )


def test_choose_label_real_human_review_is_top_tier() -> None:
    label, source, tier = choose_label(
        _trip(reviewed_label=1, source="human_review"),
        rule_score=90,
        reviewed_labels={},
        synthetic_labels={},
        strong_labels={},
    )
    assert (label, source, tier) == (1, "human_review", "reviewed_real")


def test_choose_label_demo_review_is_demo_tier() -> None:
    label, source, tier = choose_label(
        _trip(reviewed_label=0, source="demo_review"),
        rule_score=95,
        reviewed_labels={},
        synthetic_labels={},
        strong_labels={},
    )
    assert (label, source, tier) == (0, "demo_review", "reviewed_demo")


def test_choose_label_synthetic_review_is_synthetic_tier() -> None:
    label, source, tier = choose_label(
        _trip(reviewed_label=1, source="reviewed_synthetic"),
        rule_score=90,
        reviewed_labels={},
        synthetic_labels={},
        strong_labels={},
    )
    assert (label, source, tier) == (1, "reviewed_synthetic", "reviewed_synthetic")


def test_choose_label_falls_back_to_ground_truth_without_review() -> None:
    label, source, tier = choose_label(
        _trip(),
        rule_score=90,
        reviewed_labels={},
        synthetic_labels={},
        strong_labels={"t-1": 1},
    )
    assert (label, source, tier) == (1, "synthetic_ground_truth", "synthetic_ground_truth")


# ---------------------------------------------------------------------------
# _count_reviewed_real_trips
# ---------------------------------------------------------------------------


def test_count_reviewed_real_excludes_demo_and_synthetic(tmp_path: Path) -> None:
    db = _make_session(tmp_path)
    user_id = str(uuid.uuid4())
    db.add(User(id=user_id, email=f"{user_id}@example.com", password_hash="hashed"))

    def add_trip(suffix: str, label: int, source: str) -> None:
        db.add(
            Trip(
                id=f"{user_id}-{suffix}",
                user_id=user_id,
                started_at=datetime.now(timezone.utc),
                status="completed",
                reviewed_label=label,
                reviewed_label_source=source,
            )
        )

    add_trip("a", 1, "human_review")
    add_trip("b", 0, "human_review")
    add_trip("c", 1, "demo_review")
    add_trip("d", 0, "reviewed_synthetic")
    add_trip("e", None, None)
    db.commit()

    service = TripProcessingService(db)
    assert service._count_reviewed_real_trips() == 2


# ---------------------------------------------------------------------------
# maybe_schedule_auto_retrain review trigger
# ---------------------------------------------------------------------------


class _FakeThread:
    def __init__(self, *args, **kwargs) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return False


def _install_state_mocks(monkeypatch, initial_state: dict) -> dict:
    state = {
        "status": "idle",
        "last_requested_milestone": 0,
        "last_succeeded_milestone": 0,
        "pending_milestone": 0,
    }
    state.update(initial_state)
    monkeypatch.setattr(auto_retrain, "_load_state", lambda: dict(state))
    monkeypatch.setattr(auto_retrain, "_save_state", lambda s: state.update(s))
    monkeypatch.setattr(auto_retrain, "_active_thread", None)
    return state


def test_schedules_on_review_milestone(monkeypatch) -> None:
    monkeypatch.setattr(auto_retrain.settings, "auto_retrain_enabled", True)
    monkeypatch.setattr(auto_retrain.settings, "auto_retrain_min_reviewed", 30)
    monkeypatch.setattr(auto_retrain.settings, "auto_retrain_trip_interval", 100)
    state = _install_state_mocks(monkeypatch, {})
    monkeypatch.setattr(
        auto_retrain.threading,
        "Thread",
        lambda *a, **k: _FakeThread(),  # noqa: ARG005
    )

    # Completed count far from a milestone; review count just crossed 30.
    scheduled = auto_retrain.maybe_schedule_auto_retrain(
        completed_trip_count=10, reviewed_trip_count=32
    )
    assert scheduled is True
    assert state["last_requested_milestone"] == 30

    # Same bucket again -> no duplicate scheduling.
    assert (
        auto_retrain.maybe_schedule_auto_retrain(
            completed_trip_count=11, reviewed_trip_count=40
        )
        is False
    )


def test_does_not_schedule_before_min_reviewed(monkeypatch) -> None:
    monkeypatch.setattr(auto_retrain.settings, "auto_retrain_enabled", True)
    monkeypatch.setattr(auto_retrain.settings, "auto_retrain_min_reviewed", 30)
    monkeypatch.setattr(auto_retrain.settings, "auto_retrain_trip_interval", 100)
    _install_state_mocks(monkeypatch, {})

    assert (
        auto_retrain.maybe_schedule_auto_retrain(
            completed_trip_count=10, reviewed_trip_count=29
        )
        is False
    )


def test_completed_retrain_cannot_starve_review_trigger(monkeypatch) -> None:
    """Regression: a completed-trip retrain at 100 must not suppress a later
    review-gated retrain at 30 (the two triggers keep independent watermarks)."""
    monkeypatch.setattr(auto_retrain.settings, "auto_retrain_enabled", True)
    monkeypatch.setattr(auto_retrain.settings, "auto_retrain_min_reviewed", 30)
    monkeypatch.setattr(auto_retrain.settings, "auto_retrain_trip_interval", 100)
    state = _install_state_mocks(monkeypatch, {})
    monkeypatch.setattr(
        auto_retrain.threading,
        "Thread",
        lambda *a, **k: _FakeThread(),  # noqa: ARG005
    )

    # Completed-trip milestone fires first (watermark -> 100).
    assert (
        auto_retrain.maybe_schedule_auto_retrain(
            completed_trip_count=100, reviewed_trip_count=10
        )
        is True
    )
    assert state["last_requested_milestone"] == 100

    # Reviews then cross 30 (bucket 0). The review trigger must still fire,
    # and the shared completed watermark must NOT regress.
    assert (
        auto_retrain.maybe_schedule_auto_retrain(
            completed_trip_count=105, reviewed_trip_count=32
        )
        is True
    )
    assert state["last_review_requested_milestone"] == 30
    assert state["last_requested_milestone"] == 100

    # Same review bucket does not re-fire.
    assert (
        auto_retrain.maybe_schedule_auto_retrain(
            completed_trip_count=110, reviewed_trip_count=45
        )
        is False
    )

    # Completed-trip trigger still fires again at its next milestone.
    assert (
        auto_retrain.maybe_schedule_auto_retrain(
            completed_trip_count=200, reviewed_trip_count=45
        )
        is True
    )
    assert state["last_requested_milestone"] == 200

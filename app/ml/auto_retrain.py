from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings


logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = BACKEND_ROOT / "artifacts" / "reports"
STATE_PATH = REPORTS_DIR / "auto_retrain_state.json"

# RLock is required because _queue_pending_run_if_needed() is called from
# _run_refresh_cycle() while the lock is already held.
_state_lock = threading.RLock()
_active_thread: threading.Thread | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state() -> dict[str, Any]:
    return {
        "status": "idle",
        "last_requested_milestone": 0,
        # Hackathon: review-gated retrains keep their own watermark so a
        # completed-trip retrain can never starve the review trigger.
        "last_review_requested_milestone": 0,
        "last_succeeded_milestone": 0,
        "pending_milestone": 0,
    }


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return _default_state()
    try:
        loaded = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _default_state()
    return {**_default_state(), **loaded}


def _save_state(state: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {**_default_state(), **state}
    temp_path = STATE_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(STATE_PATH)


def milestone_for_completed_trips(*, completed_trip_count: int, trip_interval: int) -> int | None:
    if completed_trip_count <= 0 or trip_interval <= 0:
        return None
    if completed_trip_count % trip_interval != 0:
        return None
    return completed_trip_count


def review_milestone_for_count(
    *,
    reviewed_trip_count: int | None,
    min_reviewed: int,
    trip_interval: int,
) -> int | None:
    """Milestone for retraining on REAL human-reviewed trips (hackathon).

    Fires once per review bucket: the first time the human-reviewed count
    reaches `min_reviewed`, and then once per `trip_interval` reviewed trips
    after that (e.g. with min_reviewed=30, interval=100: buckets at 30, 130,
    230, ...). Bucketing makes the trigger batch-safe - an admin who reviews
    15 trips in one sitting cannot skip a milestone. The caller's
    last-requested guard ensures at most one retrain per bucket.

    A value of 0 for min_reviewed disables the review trigger entirely.
    """
    if min_reviewed <= 0 or trip_interval <= 0 or not reviewed_trip_count:
        return None
    if reviewed_trip_count < min_reviewed:
        return None
    bucket = (reviewed_trip_count - min_reviewed) // trip_interval
    return min_reviewed + bucket * trip_interval


def should_request_auto_retrain(
    *,
    completed_trip_count: int,
    trip_interval: int,
    last_requested_milestone: int,
) -> bool:
    milestone = milestone_for_completed_trips(
        completed_trip_count=completed_trip_count,
        trip_interval=trip_interval,
    )
    if milestone is None:
        return False
    return milestone > max(0, last_requested_milestone)


def _run_refresh_cycle_func(skip_tests: bool = True) -> dict[str, Any]:
    """Run the full refresh cycle as a direct function call instead of subprocess."""
    # Lazy import to avoid circular imports and speed up the common path
    from scripts.refresh_model_cycle import main as refresh_main
    return refresh_main(skip_tests=skip_tests)


def _queue_pending_run_if_needed() -> None:
    with _state_lock:
        state = _load_state()
        pending_milestone = int(state.get("pending_milestone") or 0)
        last_requested_milestone = int(state.get("last_requested_milestone") or 0)
        if pending_milestone <= last_requested_milestone:
            return

        state["pending_milestone"] = 0
        state["status"] = "queued"
        state["last_requested_milestone"] = pending_milestone
        state["queued_at"] = _utc_now_iso()
        state["active_milestone"] = pending_milestone
        _save_state(state)

        global _active_thread
        _active_thread = threading.Thread(
            target=_run_refresh_cycle,
            args=(pending_milestone,),
            name=f"auto-retrain-{pending_milestone}",
            daemon=True,
        )
        _active_thread.start()


def _run_refresh_cycle(milestone: int) -> None:
    with _state_lock:
        state = _load_state()
        state["status"] = "running"
        state["active_milestone"] = milestone
        state["started_at"] = _utc_now_iso()
        _save_state(state)

    try:
        skip_tests = settings.auto_retrain_skip_tests
        cycle_report = _run_refresh_cycle_func(skip_tests=skip_tests)
        succeeded = True
        error_message = None
    except Exception as exc:
        logger.exception("Auto retrain cycle failed")
        succeeded = False
        cycle_report = None
        error_message = str(exc)

    with _state_lock:
        state = _load_state()
        state["status"] = "succeeded" if succeeded else "failed"
        state["finished_at"] = _utc_now_iso()
        state["active_milestone"] = milestone
        state["last_error"] = error_message
        if succeeded:
            state["last_succeeded_milestone"] = milestone
        _save_state(state)
        _queue_pending_run_if_needed()


def maybe_schedule_auto_retrain(
    *,
    completed_trip_count: int,
    reviewed_trip_count: int | None = None,
) -> bool:
    """Schedule the refresh cycle on either trigger (hackathon).

    - Completed-trip trigger: every ``AUTO_RETRAIN_TRIP_INTERVAL`` completed
      trips (unchanged behavior).
    - Review trigger: once per review bucket once ``AUTO_RETRAIN_MIN_REVIEWED``
      REAL human-reviewed trips exist (30, 130, 230, ...).

    Both triggers drive the same worker but keep INDEPENDENT watermarks, so a
    completed-trip retrain can never suppress the review-gated retrain and
    vice-versa. The shared ``last_requested_milestone`` only ever grows.
    """
    if not settings.auto_retrain_enabled:
        return False

    trip_interval = int(settings.auto_retrain_trip_interval)
    completed_milestone = milestone_for_completed_trips(
        completed_trip_count=completed_trip_count,
        trip_interval=trip_interval,
    )
    review_milestone = review_milestone_for_count(
        reviewed_trip_count=reviewed_trip_count,
        min_reviewed=int(settings.auto_retrain_min_reviewed),
        trip_interval=trip_interval,
    )

    with _state_lock:
        state = _load_state()
        last_requested = int(state.get("last_requested_milestone") or 0)
        last_review_requested = int(state.get("last_review_requested_milestone") or 0)

        completed_new = completed_milestone is not None and completed_milestone > last_requested
        review_new = review_milestone is not None and review_milestone > last_review_requested
        if not completed_new and not review_new:
            return False

        # Highest new milestone drives the shared worker; both watermarks are
        # advanced independently so neither trigger can be starved or repeat.
        milestone = max(
            completed_milestone if completed_new else 0,
            review_milestone if review_new else 0,
        )

        global _active_thread
        if _active_thread is not None and _active_thread.is_alive():
            # A cycle is running: remember the milestone and advance the review
            # watermark now (batch-safe); the queued run starts when the
            # current cycle finishes (see _queue_pending_run_if_needed).
            state["pending_milestone"] = max(int(state.get("pending_milestone") or 0), milestone)
            if review_new:
                state["last_review_requested_milestone"] = max(last_review_requested, review_milestone)
            state["last_seen_completed_trip_count"] = completed_trip_count
            _save_state(state)
            return False

        state["status"] = "queued"
        state["last_requested_milestone"] = max(last_requested, milestone)
        if review_new:
            state["last_review_requested_milestone"] = max(last_review_requested, review_milestone)
        state["last_seen_completed_trip_count"] = completed_trip_count
        state["active_milestone"] = milestone
        state["queued_at"] = _utc_now_iso()
        _save_state(state)

        _active_thread = threading.Thread(
            target=_run_refresh_cycle,
            args=(milestone,),
            name=f"auto-retrain-{milestone}",
            daemon=True,
        )
        _active_thread.start()
        return True

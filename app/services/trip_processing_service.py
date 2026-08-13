# File role: Trip finalization/orchestration service for feature extraction, rule scoring, and ML inference.
# Loads one trip's samples, runs the shared ML pipeline, applies model inference if available,
# saves results back to the Trip record, and optionally deletes raw samples.
# Connects to:
# - app.repositories.trip_repository
# - app.repositories.sensor_sample_repository
# - app.db.models.trip
# - app.ml.config
# - app.ml.pipeline
# - app.ml.inference
# Key symbols/vars:
# - TripProcessingService
# - finalize_trip

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone

from sqlalchemy import delete
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import ForbiddenError
from app.db.models.driving_event import DrivingEvent
from app.db.models.sensor_sample import SensorSample
from app.db.models.trip import Trip
from app.db.models.user import User
from app.db.session import commit_with_retry
from app.ml.auto_retrain import maybe_schedule_auto_retrain
from app.ml.config import FeatureConfigV2
from app.ml.event_generation import build_human_reasons
from app.ml.inference import ModelScorer
from app.ml.pipeline import run_trip_pipeline
from app.ml.schemas import MODEL_VERSION_RULES_V1
from app.repositories.trip_repository import SqlTripRepository
from app.repositories.user_repository import UserRecord

logger = logging.getLogger(__name__)
ML_CONFIDENCE_THRESHOLD = 0.5
MEDIUM_CONFIDENCE_THRESHOLD = 0.8
LOW_CONFIDENCE_REASON = "Low confidence trip data, used rules fallback"
LOW_CONFIDENCE_SCORE_REASON = "Low confidence reduced score certainty"
# Phase 9: base ML blend raised 0.35 -> 0.40 and ceiling 0.50 -> 0.65 after
# the retrained model (v2 training methodology) validated at OOF risky-F1 1.0,
# Brier 0.0009. The adaptive weight still scales with data confidence and model
# calibration, so weak data or a poorly calibrated model keeps leaning on rules.
ML_SCORE_BLEND_WEIGHT = 0.40
ML_WEIGHT_MIN = 0.15
ML_WEIGHT_MAX = 0.65
ML_PREDICTION_BLEND_WEIGHT = 0.2
NEUTRAL_SCORE = 60

# Promotion-gate reference values used to normalize calibration metrics.
# Mirrors scripts/promote_model.py._bootstrap_threshold_check.
CALIB_BRIER_MAX = 0.25
CALIB_RISKY_F1_MIN = 0.55
CALIB_FPR_MAX = 0.35


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _calibration_score(metrics: dict | None) -> float:
    """Map model calibration metrics to a 0..1 quality score.

    Each metric is normalized against the promotion-gate reference values:
    brier_score (lower better, gate 0.25), risky_trip_f1 (higher better, gate
    0.55), false_positive_rate (lower better, gate 0.35). Returns the average
    of the available metrics; defaults to 0.75 when no metrics are present so
    an unpromoted/unmeasured model is treated as reasonably (not perfectly)
    calibrated.
    """
    if not metrics:
        return 0.75

    norms: list[float] = []
    brier = metrics.get("brier_score")
    if brier is not None:
        norms.append(1.0 - _clamp(float(brier) / CALIB_BRIER_MAX, 0.0, 1.0))
    f1 = metrics.get("risky_trip_f1")
    if f1 is not None:
        norms.append(_clamp((float(f1) - CALIB_RISKY_F1_MIN) / (1.0 - CALIB_RISKY_F1_MIN), 0.0, 1.0))
    fpr = metrics.get("false_positive_rate")
    if fpr is not None:
        norms.append(1.0 - _clamp(float(fpr) / CALIB_FPR_MAX, 0.0, 1.0))

    if not norms:
        return 0.75
    return sum(norms) / len(norms)


def adaptive_ml_blend_weight(
    confidence: float | None,
    calibration_metrics: dict | None,
) -> float:
    """Compute the ML blend weight for a trip, scaling with data confidence and
    the promoted model's calibration instead of the constant 0.40.

    - Confidence factor ramps from 0.5 at the ML threshold (0.5) to 1.0 at the
      medium-confidence threshold (0.8) and above, so low-quality data leans
      more heavily on the deterministic rules.
    - Calibration factor maps the model's metrics to a 0.6..1.4 multiplier
      (poor calibration suppresses ML influence; great calibration boosts it).
    - The result is clamped to [ML_WEIGHT_MIN, ML_WEIGHT_MAX] so the rules
      always retain a meaningful share of the final score.

    Note: confidence also feeds the neutral-score pull inside
    _compute_final_score, so low-confidence trips are dampened twice (blend
    weight and neutral pull). This is intentional belt-and-suspenders: weak
    data leans on the rules both at blend time and at score shaping time.

    Returns a moderate weight (not the base 0.40) when confidence is unknown.
    """
    if confidence is None:
        conf_factor = 0.5
    else:
        ramp = (float(confidence) - ML_CONFIDENCE_THRESHOLD) / (MEDIUM_CONFIDENCE_THRESHOLD - ML_CONFIDENCE_THRESHOLD)
        conf_factor = 0.5 + 0.5 * _clamp(ramp, 0.0, 1.0)

    calib_score = _calibration_score(calibration_metrics)
    calib_factor = 0.6 + 0.8 * calib_score  # 0.6 .. 1.4

    weight = ML_SCORE_BLEND_WEIGHT * conf_factor * calib_factor
    return _clamp(weight, ML_WEIGHT_MIN, ML_WEIGHT_MAX)


class TripProcessingService:
    # Note: "speed_variation" is intentionally retained in this set (Phase 2/3)
    # so reprocessing deletes stale pre-Phase-2 rows even though the generator no
    # longer produces them.
    GENERATED_EVENT_TYPES = {
        "hard_brake",
        "emergency_brake",
        "hard_accel",
        "aggressive_turn",
        "unstable_motion",
        "overspeed",
        "severe_overspeed",
        "speed_variation",
    }

    def __init__(self, db: Session, cfg: FeatureConfigV2 | None = None) -> None:
        self.db = db
        self.trip_repo = SqlTripRepository(db)
        self.cfg = cfg or FeatureConfigV2()
        self.model_scorer = ModelScorer()

    def _load_trip(self, user_id: str, trip_id: str) -> Trip:
        trip = self.trip_repo.get_by_id(trip_id=trip_id, user_id=user_id)
        if not trip:
            raise ValueError("Trip not found")
        return trip

    def _load_trip_for_actor(self, actor: UserRecord, trip_id: str) -> Trip:
        if actor.is_admin:
            trip = self.db.execute(select(Trip).where(Trip.id == trip_id)).scalar_one_or_none()
        else:
            trip = self.trip_repo.get_by_id(trip_id=trip_id, user_id=actor.id)
        if not trip:
            raise ValueError("Trip not found")
        return trip

    def _driver_email_for_trip(self, trip: Trip) -> str | None:
        return self.db.execute(select(User.email).where(User.id == trip.user_id)).scalar_one_or_none()

    def _require_admin(self, actor: UserRecord) -> None:
        if not actor.is_admin:
            raise ForbiddenError(message_key="auth.forbidden")

    def _load_samples(self, user_id: str, trip_id: str) -> list[SensorSample]:
        stmt = (
            select(SensorSample)
            .where(SensorSample.user_id == user_id, SensorSample.trip_id == trip_id)
            .order_by(SensorSample.ts.asc())
        )
        return self.db.execute(stmt).scalars().all()

    def _count_completed_trips(self) -> int:
        return int(
            self.db.execute(
                select(func.count()).select_from(Trip).where(Trip.status == "completed")
            ).scalar_one()
            or 0
        )

    def _count_reviewed_real_trips(self) -> int:
        """Count trips carrying a REAL human review label (admin review screen).

        Classified via the shared review-label tier helper (case-safe on both
        SQLite and PostgreSQL) so the review-gated retrain trigger only counts
        genuine human ground truth, never demo or synthetic sources.
        """
        from app.ml.labels import is_real_review_source

        sources = self.db.execute(
            select(Trip.reviewed_label_source).where(Trip.reviewed_label.is_not(None))
        ).scalars().all()
        return sum(1 for source in sources if is_real_review_source(source))

    def _samples_to_payload(self, rows: list[SensorSample]) -> list[dict]:
        payload: list[dict] = []

        for row in rows:
            # Ensure the timestamp always includes timezone info so that
            # preprocess_samples can parse it reliably with pd.to_datetime.
            # SQLite's DateTime column strips tzinfo on read-back, and
            # pandas chokes on a Series with mixed formats (with/without
            # microseconds). Adding a timezone makes all entries uniform.
            ts = row.ts
            if ts is not None and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            # speed_mps stores m/s; pipeline expects km/h — convert here
            speed_kph = row.speed_mps * 3.6 if row.speed_mps is not None else None
            payload.append(
                {
                    "timestamp": ts.isoformat() if ts is not None else None,
                    "speed": speed_kph,
                    "lat": row.lat,
                    "lon": row.lon,
                    "ax": row.ax,
                    "ay": row.ay,
                    "az": row.az,
                    "gx": row.gx,
                    "gy": row.gy,
                    "gz": row.gz,
                }
            )

        return payload

    def _compute_final_score(
        self,
        rule_score: int | None,
        ml_prediction: int | None,
        ml_risk_probability: float | None,
        confidence: float | None,
        ml_blend_weight: float | None = None,
    ) -> int | None:
        # CRIT-2: never let a non-finite input produce a NaN blended score
        # (previously crashed with "cannot convert float NaN to integer").
        if rule_score is not None and not math.isfinite(float(rule_score)):
            rule_score = None
        if ml_risk_probability is not None and not math.isfinite(float(ml_risk_probability)):
            ml_risk_probability = None
        if confidence is not None and not math.isfinite(float(confidence)):
            confidence = None
        if ml_blend_weight is not None and not math.isfinite(float(ml_blend_weight)):
            ml_blend_weight = None

        blend_weight = ML_SCORE_BLEND_WEIGHT if ml_blend_weight is None else ml_blend_weight
        weighted_scores: list[tuple[float, float]] = []

        if rule_score is not None:
            rule_weight = 1.0 if ml_risk_probability is None and ml_prediction is None else (1.0 - blend_weight)
            weighted_scores.append((float(rule_score), rule_weight))

        if ml_risk_probability is not None:
            calibrated_probability = max(0.05, min(0.95, float(ml_risk_probability)))
            weighted_scores.append((100.0 * (1.0 - calibrated_probability), blend_weight))
        elif ml_prediction is not None:
            proxy_score = 78.0 if ml_prediction == 0 else 34.0
            weight = 1.0 if rule_score is None else ML_PREDICTION_BLEND_WEIGHT
            weighted_scores.append((proxy_score, weight))

        if not weighted_scores:
            return None

        total_weight = sum(weight for _, weight in weighted_scores)
        if total_weight <= 0:
            return None
        blended_score = sum(score * weight for score, weight in weighted_scores) / total_weight
        bounded_score = float(max(0.0, min(100.0, blended_score)))
        if not math.isfinite(bounded_score):
            return None

        if confidence is None:
            return int(round(bounded_score))

        confidence_scale = float(max(0.0, min(1.0, confidence / MEDIUM_CONFIDENCE_THRESHOLD)))
        confidence_adjusted_score = NEUTRAL_SCORE + ((bounded_score - NEUTRAL_SCORE) * confidence_scale)
        return int(round(max(0.0, min(100.0, confidence_adjusted_score))))

    def _risk_probability_from_score(
        self,
        score: int | None,
        ml_risk_probability: float | None,
    ) -> float | None:
        if score is None:
            if ml_risk_probability is None:
                return None
            return float(max(0.0, min(1.0, ml_risk_probability)))

        score_risk_probability = float(max(0.0, min(1.0, 1.0 - (score / 100.0))))
        if ml_risk_probability is None:
            return score_risk_probability

        return float(
            max(
                0.0,
                min(
                    1.0,
                    round(0.55 * float(ml_risk_probability) + 0.45 * score_risk_probability, 4),
                ),
            )
        )

    def _risk_level_from_score(self, score: int | None) -> str | None:
        # Phase 3 bands calibrated for the v2 scoring model: >=85 low,
        # 65-84 medium, <65 high.
        if score is None:
            return None
        if score >= 85:
            return "low"
        if score >= 65:
            return "medium"
        return "high"

    def _decision_source(self, ml_used: bool) -> str:
        return "ml_with_rules" if ml_used else "rules_fallback"

    def _confidence_band(self, confidence: float | None) -> str | None:
        if confidence is None:
            return None
        if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
            return "high"
        if confidence >= ML_CONFIDENCE_THRESHOLD:
            return "medium"
        return "low"

    def _confidence_display(self, confidence: float | None) -> str | None:
        band = self._confidence_band(confidence)
        if band == "high":
            return "show_normally"
        if band == "medium":
            return "show_with_caution"
        if band == "low":
            return "insufficient_data"
        return None

    def _prediction_label(
        self,
        *,
        risk_probability: float | None,
        breakdown: dict,
    ) -> int | None:
        if risk_probability is not None:
            return int(float(risk_probability) >= ML_CONFIDENCE_THRESHOLD)
        ml_prediction = breakdown.get("ml_prediction")
        if ml_prediction is None:
            return None
        return int(ml_prediction)

    def _review_disagreement(
        self,
        *,
        reviewed_label: int | None,
        predicted_label: int | None,
    ) -> bool | None:
        if reviewed_label is None or predicted_label is None:
            return None
        return int(reviewed_label) != int(predicted_label)

    def _event_payloads(self, trip: Trip) -> list[dict]:
        return [
            {
                "id": ev.id,
                "trip_id": ev.trip_id,
                "event_type": ev.event_type,
                "value": float(ev.value),
                "confidence": ev.confidence,
                "severity": ev.severity,
                "duration_s": ev.duration_s,
                "occurred_at": ev.occurred_at,
                "lat": ev.lat,
                "lon": ev.lon,
                "created_at": ev.created_at,
            }
            for ev in sorted(
                trip.events,
                key=lambda item: ((item.occurred_at or item.created_at), item.id),
            )
        ]

    def _generated_event_payloads(self, trip: Trip) -> list[dict]:
        return [
            payload
            for payload in self._event_payloads(trip)
            if payload["event_type"] in self.GENERATED_EVENT_TYPES
        ]

    def _load_breakdown(self, trip: Trip) -> dict:
        if not getattr(trip, "score_breakdown", None):
            return {}
        try:
            return json.loads(trip.score_breakdown)
        except Exception:
            return {}

    def _is_not_enough_samples(self, breakdown: dict | None) -> bool:
        if not isinstance(breakdown, dict):
            return False
        if breakdown.get("error") == "not_enough_samples":
            return True
        nested = breakdown.get("rule_breakdown")
        return isinstance(nested, dict) and nested.get("error") == "not_enough_samples"

    def _build_not_enough_samples_response(
        self,
        *,
        trip_id: str,
        feature_version: str | None,
        confidence: float | None,
        rule_breakdown: dict,
        processed_at: datetime | None = None,
    ) -> dict:
        breakdown = {
            "error": "not_enough_samples",
            "rule_breakdown": rule_breakdown,
            "trip_preserved": True,
        }
        return {
            "trip_id": trip_id,
            "score": None,
            "risk_level": None,
            "risk_probability": None,
            "confidence": confidence,
            "confidence_band": self._confidence_band(confidence),
            "confidence_display": self._confidence_display(confidence),
            "model_version": MODEL_VERSION_RULES_V1,
            "feature_version": feature_version,
            "decision_source": "rules_fallback",
            "processing_timestamp": processed_at,
            "raw_deleted": False,
            "already_processed": False,
            "reasons": [],
            "events": [],
            "breakdown": breakdown,
            "trip_features": {},
            "events_generated": 0,
        }

    def _build_response(self, trip: Trip, breakdown: dict, already_processed: bool = False) -> dict:
        confidence = getattr(trip, "confidence", None)
        return {
            "trip_id": trip.id,
            "score": getattr(trip, "score", None),
            "risk_level": getattr(trip, "risk_level", None),
            "risk_probability": getattr(trip, "risk_probability", None),
            "ml_blend_weight": breakdown.get("ml_blend_weight"),
            "confidence": confidence,
            "confidence_band": self._confidence_band(confidence),
            "confidence_display": self._confidence_display(confidence),
            "model_version": getattr(trip, "model_version", None),
            "feature_version": getattr(trip, "feature_version", None),
            "decision_source": breakdown.get("decision_source"),
            "processing_timestamp": getattr(trip, "processed_at", None),
            "raw_deleted": getattr(trip, "raw_deleted", None),
            "already_processed": already_processed,
            "reasons": breakdown.get("reasons", []),
            "events": self._event_payloads(trip),
            "breakdown": breakdown,
            "trip_features": breakdown.get("trip_features", {}),
            "events_generated": len(breakdown.get("generated_events", [])),
        }

    def get_trip_review(self, actor: UserRecord, trip_id: str) -> dict:
        self._require_admin(actor)
        trip = self._load_trip_for_actor(actor=actor, trip_id=trip_id)
        breakdown = self._load_breakdown(trip)
        confidence = trip.confidence
        predicted_label = self._prediction_label(risk_probability=trip.risk_probability, breakdown=breakdown)
        driver_email = self._driver_email_for_trip(trip)

        return {
            "trip_id": trip.id,
            "driver_user_id": trip.user_id,
            "driver_email": driver_email,
            "score": trip.score,
            "risk_level": trip.risk_level,
            "risk_probability": trip.risk_probability,
            "ml_blend_weight": breakdown.get("ml_blend_weight"),
            "confidence": confidence,
            "confidence_band": self._confidence_band(confidence),
            "confidence_display": self._confidence_display(confidence),
            "feature_version": trip.feature_version,
            "model_version": trip.model_version,
            "processed_at": trip.processed_at,
            "trip_features": breakdown.get("trip_features", {}),
            "rule_score": breakdown.get("rule_score"),
            "ml_prediction": breakdown.get("ml_prediction"),
            "predicted_label": predicted_label,
            "reasons": breakdown.get("reasons", []),
            "events": self._event_payloads(trip),
            "reviewed_label": trip.reviewed_label,
            "reviewed_label_source": trip.reviewed_label_source,
            "review_disagrees_with_prediction": self._review_disagreement(
                reviewed_label=trip.reviewed_label,
                predicted_label=predicted_label,
            ),
            "review_notes": trip.review_notes,
            "reviewed_at": trip.reviewed_at,
        }

    def get_trip_detail(self, actor: UserRecord, trip_id: str) -> dict:
        trip = self._load_trip_for_actor(actor=actor, trip_id=trip_id)
        breakdown = self._load_breakdown(trip)
        return {
            "id": trip.id,
            "user_id": trip.user_id,
            "started_at": trip.started_at,
            "ended_at": trip.ended_at,
            "status": trip.status,
            **self._build_response(trip, breakdown, already_processed=False),
        }

    def list_review_dashboard(self, actor: UserRecord, limit: int = 50) -> list[dict]:
        self._require_admin(actor)
        stmt = (
            select(Trip, User.email)
            .join(User, User.id == Trip.user_id)
            .where(
                Trip.status == "completed",
                Trip.score.is_not(None),
            )
        )
        stmt = stmt.order_by(Trip.processed_at.desc(), Trip.started_at.desc()).limit(limit)
        rows = self.db.execute(stmt).all()

        items: list[dict] = []
        for trip, driver_email in rows:
            breakdown = self._load_breakdown(trip)
            confidence = trip.confidence
            predicted_label = self._prediction_label(risk_probability=trip.risk_probability, breakdown=breakdown)
            trip_events = self._event_payloads(trip)
            generated_events = self._generated_event_payloads(trip)
            items.append(
                {
                    "trip_id": trip.id,
                    "driver_user_id": trip.user_id,
                    "driver_email": driver_email,
                    "score": trip.score,
                    "risk_level": trip.risk_level,
                    "risk_probability": trip.risk_probability,
                    "ml_blend_weight": breakdown.get("ml_blend_weight"),
                    "confidence": confidence,
                    "confidence_band": self._confidence_band(confidence),
                    "confidence_display": self._confidence_display(confidence),
                    "rule_score": breakdown.get("rule_score"),
                    "predicted_label": predicted_label,
                    "reasons": breakdown.get("reasons", []),
                    "generated_events": generated_events,
                    "trip_events": trip_events,
                    "generated_event_count": len(generated_events),
                    "trip_event_count": len(trip_events),
                    "review_label": trip.reviewed_label,
                    "review_label_source": trip.reviewed_label_source,
                    "review_disagrees_with_prediction": self._review_disagreement(
                        reviewed_label=trip.reviewed_label,
                        predicted_label=predicted_label,
                    ),
                    "model_version": trip.model_version,
                    "feature_version": trip.feature_version,
                    "processed_at": trip.processed_at,
                    "reviewed_at": trip.reviewed_at,
                }
            )

        return items

    def set_trip_review_label(
        self,
        actor: UserRecord,
        trip_id: str,
        reviewed_label: int | None,
        reviewed_label_source: str,
        review_notes: str | None,
    ) -> dict:
        self._require_admin(actor)
        trip = self._load_trip_for_actor(actor=actor, trip_id=trip_id)
        trip.reviewed_label = reviewed_label
        trip.reviewed_label_source = reviewed_label_source
        trip.review_notes = review_notes
        trip.reviewed_at = datetime.now(timezone.utc)

        self.db.add(trip)
        commit_with_retry(self.db)
        self.db.refresh(trip)

        return self.get_trip_review(actor=actor, trip_id=trip_id)

    def reprocess_trips(
        self,
        user_id: str | None = None,
        trip_id: str | None = None,
        model_version: str | None = None,
        feature_version: str | None = None,
    ) -> dict:
        stmt = select(Trip)

        if user_id:
            stmt = stmt.where(Trip.user_id == user_id)
        if trip_id:
            stmt = stmt.where(Trip.id == trip_id)
        if model_version:
            stmt = stmt.where(Trip.model_version == model_version)
        if feature_version:
            stmt = stmt.where(Trip.feature_version == feature_version)

        trips = self.db.execute(stmt.order_by(Trip.started_at.desc())).scalars().all()

        reprocessed = 0
        failed = 0
        completed_trip_ids: list[str] = []

        for trip in trips:
            try:
                self.finalize_trip(user_id=trip.user_id, trip_id=trip.id, delete_raw=False, force_reprocess=True)
                reprocessed += 1
                completed_trip_ids.append(trip.id)
            except Exception:
                failed += 1
                self.db.rollback()

        return {
            "matched": len(trips),
            "reprocessed": reprocessed,
            "failed": failed,
            "trip_ids": completed_trip_ids,
        }

    def finalize_trip(
        self,
        user_id: str,
        trip_id: str,
        delete_raw: bool = False,
        force_reprocess: bool = False,
    ) -> dict:
        trip = self._load_trip(user_id=user_id, trip_id=trip_id)

        if getattr(trip, "processed_at", None) and not delete_raw and not force_reprocess:
            return self._build_response(trip, self._load_breakdown(trip), already_processed=True)

        sample_rows = self._load_samples(user_id=user_id, trip_id=trip.id)
        sample_payload = self._samples_to_payload(sample_rows)
        sample_ids = [row.id for row in sample_rows]

        # Release the DB transaction so the pipeline can run without holding
        # a long-lived connection lock (important for SQLite WAL mode).
        # Only rollback if there is actually an active transaction to avoid
        # discarding state from sibling operations in the same session.
        if self.db.in_transaction():
            self.db.rollback()

        pipeline_result = run_trip_pipeline(sample_payload, self.cfg)

        trip_features = pipeline_result["trip_features"]
        rule_score = pipeline_result["score"]
        rule_breakdown = pipeline_result["breakdown"]
        feature_version = pipeline_result["feature_version"]
        confidence = pipeline_result["confidence"]
        generated_events = pipeline_result.get("event_instances", [])

        if self._is_not_enough_samples(rule_breakdown):
            # H-6 fix: previously the trip (and its raw samples) were DELETED here.
            # Short trips were permanently lost and admin GETs ran destructive
            # cleanup. Instead, preserve the trip and its samples so later
            # reprocessing (e.g. after threshold changes) can re-evaluate it.
            trip = self._load_trip(user_id=user_id, trip_id=trip_id)
            trip.score = None
            trip.score_breakdown = json.dumps(
                {
                    "error": "not_enough_samples",
                    "rule_breakdown": rule_breakdown,
                    "trip_preserved": True,
                }
            )
            trip.feature_version = feature_version
            trip.confidence = confidence
            trip.processed_at = datetime.now(timezone.utc)
            trip.raw_deleted = False
            try:
                self.db.add(trip)
                commit_with_retry(self.db)
                self.db.refresh(trip)
            except Exception:
                self.db.rollback()
                raise
            return self._build_not_enough_samples_response(
                trip_id=trip.id,
                feature_version=feature_version,
                confidence=confidence,
                rule_breakdown=rule_breakdown,
                processed_at=trip.processed_at,
            )

        ml_prediction: int | None = None
        ml_risk_probability: float | None = None
        model_version = MODEL_VERSION_RULES_V1
        ml_used = False

        if trip_features and confidence >= ML_CONFIDENCE_THRESHOLD:
            try:
                ml_result = self.model_scorer.predict(trip_features)
                ml_prediction = ml_result["prediction"]
                ml_risk_probability = ml_result.get("risk_probability")
                model_version = str(ml_result["model_version"])
                ml_used = True
            except Exception as exc:
                rule_breakdown["ml_error"] = str(exc)
        elif trip_features:
            rule_breakdown["low_confidence"] = True

        ml_blend_weight = (
            adaptive_ml_blend_weight(confidence, self.model_scorer.calibration_metrics)
            if ml_used
            else None
        )

        final_score = self._compute_final_score(
            rule_score=rule_score,
            ml_prediction=ml_prediction,
            ml_risk_probability=ml_risk_probability,
            confidence=confidence,
            ml_blend_weight=ml_blend_weight,
        )
        risk_probability = self._risk_probability_from_score(final_score, ml_risk_probability)
        risk_level = self._risk_level_from_score(final_score)

        human_reasons = build_human_reasons(
            trip_features=trip_features,
            ml_prediction=ml_prediction,
            ml_risk_probability=ml_risk_probability,
        )
        if trip_features and confidence < ML_CONFIDENCE_THRESHOLD:
            human_reasons.append(LOW_CONFIDENCE_REASON)
            human_reasons.append(LOW_CONFIDENCE_SCORE_REASON)

        persisted_breakdown = {
            "rule_score": rule_score,
            "rule_breakdown": rule_breakdown,
            "ml_prediction": ml_prediction,
            "ml_risk_probability": ml_risk_probability,
            "ml_blend_weight": ml_blend_weight,
            "confidence": confidence,
            "final_score": final_score,
            "risk_level": risk_level,
            "risk_probability": risk_probability,
            "decision_source": self._decision_source(ml_used),
            "reasons": human_reasons,
            "generated_events": generated_events,
            "trip_features": trip_features,
        }

        trip = self._load_trip(user_id=user_id, trip_id=trip_id)
        trip.score = final_score
        trip.score_breakdown = json.dumps(persisted_breakdown)
        trip.feature_version = feature_version
        trip.model_version = model_version
        trip.confidence = confidence
        trip.risk_probability = risk_probability
        trip.risk_level = risk_level
        trip.processed_at = datetime.now(timezone.utc)
        trip.raw_deleted = False

        try:
            self._replace_generated_events(
                user_id=user_id,
                trip_id=trip.id,
                generated_events=generated_events,
            )

            if delete_raw and sample_ids:
                self.db.execute(
                    delete(SensorSample).where(
                        SensorSample.user_id == user_id,
                        SensorSample.trip_id == trip.id,
                        SensorSample.id.in_(sample_ids),
                    )
                )
                trip.raw_deleted = True

            self.db.add(trip)
            commit_with_retry(self.db)
            self.db.refresh(trip)
        except Exception:
            self.db.rollback()
            raise

        if not force_reprocess:
            try:
                reviewed_trip_count = (
                    self._count_reviewed_real_trips() if settings.auto_retrain_enabled else None
                )
                maybe_schedule_auto_retrain(
                    completed_trip_count=self._count_completed_trips(),
                    reviewed_trip_count=reviewed_trip_count,
                )
            except Exception:
                logger.exception("Failed to schedule automatic retraining after trip finalization.")

        return self._build_response(trip, persisted_breakdown, already_processed=False)

    def _replace_generated_events(
        self,
        user_id: str,
        trip_id: str,
        generated_events: list[dict],
    ) -> None:
        existing = (
            self.db.query(DrivingEvent)
            .filter(
                DrivingEvent.user_id == user_id,
                DrivingEvent.trip_id == trip_id,
                DrivingEvent.event_type.in_(self.GENERATED_EVENT_TYPES),
            )
            .all()
        )

        for ev in existing:
            self.db.delete(ev)

        for item in generated_events:
            occurred_at_raw = item.get("occurred_at")
            occurred_at = None
            if occurred_at_raw:
                occurred_at = datetime.fromisoformat(str(occurred_at_raw).replace("Z", "+00:00"))
            ev = DrivingEvent(
                user_id=user_id,
                trip_id=trip_id,
                event_type=item["event_type"],
                value=float(item["value"]),
                confidence=item.get("confidence"),
                severity=item.get("severity"),
                duration_s=item.get("duration_s"),
                occurred_at=occurred_at,
                lat=item.get("lat"),
                lon=item.get("lon"),
            )
            self.db.add(ev)

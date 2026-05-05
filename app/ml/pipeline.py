# File role: ML/scoring pipeline component used during trip finalization to derive features, score, and confidence.
# Connects to: .config, .preprocessing, .features.
# Key symbols/vars: FEATURE_VERSION, MODEL_VERSION, run_trip_pipeline.
# app/ml/pipeline.py
from __future__ import annotations
from typing import Dict, Any, List
from .config import FeatureConfigV1
from .event_generation import generate_trip_events
from .preprocessing import preprocess_samples
from .features import compute_per_sample_features, aggregate_trip_features
from .scoring_rules import score_trip_rules_v1

FEATURE_VERSION = "fv1"
MODEL_VERSION = "rules_v1"

def run_trip_pipeline(samples: List[Dict[str, Any]], cfg: FeatureConfigV1) -> Dict[str, Any]:
    df = preprocess_samples(samples, cfg.max_gap_s, cfg.ema_alpha, cfg.input_speed_unit)
    if df.empty or len(df) < cfg.min_samples_for_scoring:
        # not enough data to score
        return {
            "feature_version": FEATURE_VERSION,
            "model_version": MODEL_VERSION,
            "score": None,
            "confidence": 0.0,
            "breakdown": {"error": "not_enough_samples"},
            "trip_features": {},
            "event_instances": [],
        }

    per = compute_per_sample_features(df)
    trip_features = aggregate_trip_features(
        per,
        aggressive_turn_threshold=cfg.aggressive_turn_threshold,
        harsh_brake_dv=cfg.harsh_brake_dv,
        harsh_accel_dv=cfg.harsh_accel_dv,
        emergency_brake_dv=cfg.emergency_brake_dv,
        emergency_brake_min_speed_mps=cfg.emergency_brake_min_speed_mps,
        min_event_duration_s=cfg.min_event_duration_s,
        merge_gap_s=cfg.merge_gap_s,
    )
    breakdown = score_trip_rules_v1(trip_features, cfg.w_brake, cfg.w_accel, cfg.w_turn, cfg.w_jerk, cfg.w_speed_var)
    event_instances = generate_trip_events(
        per,
        trip_features,
        harsh_brake_dv=cfg.harsh_brake_dv,
        harsh_accel_dv=cfg.harsh_accel_dv,
        emergency_brake_dv=cfg.emergency_brake_dv,
        emergency_brake_min_speed_mps=cfg.emergency_brake_min_speed_mps,
        aggressive_turn_threshold=cfg.aggressive_turn_threshold,
        min_event_duration_s=cfg.min_event_duration_s,
        merge_gap_s=cfg.merge_gap_s,
    )

    return {
        "feature_version": FEATURE_VERSION,
        "model_version": MODEL_VERSION,
        "score": breakdown["score"],
        "confidence": trip_features["confidence"],
        "breakdown": breakdown,
        "trip_features": trip_features,
        "event_instances": event_instances,
    }

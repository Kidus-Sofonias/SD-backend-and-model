# File role: ML/scoring pipeline component used during trip finalization to derive features, score, and confidence.
# Connects to: .config, .preprocessing, .features.
# Key symbols/vars: FEATURE_VERSION, MODEL_VERSION, run_trip_pipeline.
# app/ml/pipeline.py
from __future__ import annotations
from typing import Dict, Any, List
from .config import FeatureConfigV2
from .event_generation import generate_trip_events
from .preprocessing import preprocess_samples
from .features import compute_per_sample_features, aggregate_trip_features
from .scoring_rules import score_trip_rules_v2
from .schemas import FEATURE_VERSION, MODEL_VERSION_RULES_V1

MODEL_VERSION = MODEL_VERSION_RULES_V1


def run_trip_pipeline(samples: List[Dict[str, Any]], cfg: FeatureConfigV2) -> Dict[str, Any]:
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
        turn_min_duration_s=cfg.turn_min_duration_s,
        harsh_brake_dv=cfg.harsh_brake_dv,
        harsh_accel_dv=cfg.harsh_accel_dv,
        emergency_brake_dv=cfg.emergency_brake_dv,
        emergency_brake_min_speed_mps=cfg.emergency_brake_min_speed_mps,
        min_event_duration_s=cfg.min_event_duration_s,
        merge_gap_s=cfg.merge_gap_s,
        unstable_motion_jerk_threshold=cfg.unstable_motion_jerk_threshold,
        overspeed_threshold_mps=cfg.overspeed_threshold_mps,
        overspeed_min_duration_s=cfg.overspeed_min_duration_s,
        severe_overspeed_threshold_mps=cfg.severe_overspeed_threshold_mps,
        severe_overspeed_min_duration_s=cfg.severe_overspeed_min_duration_s,
    )
    breakdown = score_trip_rules_v2(
        trip_features,
        cfg.w_emergency_brake,
        cfg.w_brake,
        cfg.w_accel,
        cfg.w_turn,
        cfg.w_overspeed,
        cfg.w_severe_overspeed,
        cfg.w_unstable_motion,
        cfg.w_jerk,
        cfg.w_speed_var,
        cfg.w_density,
        cfg.jerk_normalize_low,
        cfg.jerk_normalize_high,
        cfg.speed_var_normalize_high,
        cfg.density_normalize_low,
        cfg.density_normalize_high,
    )
    event_instances = generate_trip_events(
        per,
        trip_features,
        harsh_brake_dv=cfg.harsh_brake_dv,
        harsh_accel_dv=cfg.harsh_accel_dv,
        emergency_brake_dv=cfg.emergency_brake_dv,
        emergency_brake_min_speed_mps=cfg.emergency_brake_min_speed_mps,
        aggressive_turn_threshold=cfg.aggressive_turn_threshold,
        turn_min_duration_s=cfg.turn_min_duration_s,
        min_event_duration_s=cfg.min_event_duration_s,
        merge_gap_s=cfg.merge_gap_s,
        unstable_motion_jerk_threshold=cfg.unstable_motion_jerk_threshold,
        overspeed_threshold_mps=cfg.overspeed_threshold_mps,
        overspeed_min_duration_s=cfg.overspeed_min_duration_s,
        severe_overspeed_threshold_mps=cfg.severe_overspeed_threshold_mps,
        severe_overspeed_min_duration_s=cfg.severe_overspeed_min_duration_s,
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

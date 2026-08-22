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
from .scoring_rules import score_trip_rules_v3
from .schemas import FEATURE_VERSION, MODEL_VERSION_RULES_V1
from .vehicle_profiles import vehicle_feature_row

MODEL_VERSION = MODEL_VERSION_RULES_V1


def run_trip_pipeline(
    samples: List[Dict[str, Any]],
    cfg: FeatureConfigV2,
    vehicle_profile: Any = None,
) -> Dict[str, Any]:
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

    per = compute_per_sample_features(df, nominal_dt_s=cfg.nominal_dt_s)
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
        # Phase 10 (hackathon): noise robustness & event impact
        event_cooldown_s=cfg.event_cooldown_s,
        dv_min_speed_delta_mps=cfg.dv_min_speed_delta_mps,
        dv_single_sample_peak_mps2=cfg.dv_single_sample_peak_mps2,
        unstable_cooldown_s=cfg.unstable_cooldown_s,
        turn_min_speed_mps=cfg.turn_min_speed_mps,
        nominal_dt_s=cfg.nominal_dt_s,
        brake_severity_ref_mps2=cfg.brake_severity_ref_mps2,
        accel_severity_ref_mps2=cfg.accel_severity_ref_mps2,
        turn_severity_ref_mps2=cfg.turn_severity_ref_mps2,
        unstable_severity_ref_mps3=cfg.unstable_severity_ref_mps3,
        overspeed_severity_ref_mps=cfg.overspeed_severity_ref_mps,
        severe_overspeed_severity_ref_mps=cfg.severe_overspeed_severity_ref_mps,
        density_distance_normalize_high=cfg.density_distance_normalize_high,
        density_min_duration_s=cfg.density_min_duration_s,
    )
    breakdown = score_trip_rules_v3(
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
        cfg.density_distance_normalize_high,
        cfg.density_min_duration_s,
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
        # Phase 10 (hackathon): noise robustness & event impact
        event_cooldown_s=cfg.event_cooldown_s,
        dv_min_speed_delta_mps=cfg.dv_min_speed_delta_mps,
        dv_single_sample_peak_mps2=cfg.dv_single_sample_peak_mps2,
        nominal_dt_s=cfg.nominal_dt_s,
        unstable_cooldown_s=cfg.unstable_cooldown_s,
        turn_min_speed_mps=cfg.turn_min_speed_mps,
        brake_severity_ref_mps2=cfg.brake_severity_ref_mps2,
        accel_severity_ref_mps2=cfg.accel_severity_ref_mps2,
        turn_severity_ref_mps2=cfg.turn_severity_ref_mps2,
        unstable_severity_ref_mps3=cfg.unstable_severity_ref_mps3,
        overspeed_severity_ref_mps=cfg.overspeed_severity_ref_mps,
        severe_overspeed_severity_ref_mps=cfg.severe_overspeed_severity_ref_mps,
    )

    # Phase 8b: merge the vehicle context into the feature dict so the model
    # contract (FEATURE_COLUMNS_FV1) is complete for every trip — defaulting
    # to the universal 1400 kg reference when no vehicle profile exists.
    trip_features.update(vehicle_feature_row(vehicle_profile))

    return {
        "feature_version": FEATURE_VERSION,
        "model_version": MODEL_VERSION,
        "score": breakdown["score"],
        "confidence": trip_features["confidence"],
        "breakdown": breakdown,
        "trip_features": trip_features,
        "event_instances": event_instances,
    }

"""Shared event utility functions used across the ML pipeline for event detection.

Provides a single source of truth for segment detection logic that was previously
duplicated between features.py and event_generation.py.
"""

from __future__ import annotations

import numpy as np


def event_segments(
    mask: np.ndarray,
    timestamps: np.ndarray,
    min_duration_s: float,
    merge_gap_s: float,
    cooldown_s: float = 0.0,
) -> list[tuple[int, int]]:
    """Return sustained event index ranges from a boolean mask.

    Merges adjacent segments that are closer than merge_gap_s apart,
    filters out any segment shorter than min_duration_s, then applies a
    per-category cooldown: a segment whose START is within cooldown_s of the
    previous kept segment's START is dropped (Phase 10 noise dedup).

    Args:
        mask: Boolean array where True indicates an event condition.
        timestamps: Array of timestamps corresponding to each sample.
        min_duration_s: Minimum duration in seconds for a sustained event.
        merge_gap_s: Maximum gap in seconds between segments to merge.
        cooldown_s: Minimum seconds between consecutive kept events of the
            same category (0 disables).

    Returns:
        List of (start, end) index tuples for each sustained event.
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []

    groups: list[tuple[int, int]] = []
    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            groups.append((start, prev))
            start = i
            prev = i
    groups.append((start, prev))

    merged: list[list[int]] = []
    for s, e in groups:
        if not merged:
            merged.append([s, e])
            continue

        _, prev_e = merged[-1]
        gap = timestamps[s] - timestamps[prev_e]
        if gap <= merge_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # CRIT-1: at GPS sample rates (0.5-2 Hz) real braking/acceleration events often
    # span only 1-2 samples. Filtering by an absolute wall-clock duration (e.g.
    # 0.25 s) silently discarded those events, producing "0 events despite clearly
    # detected behavior". The duration floor is now interpreted relative to the
    # actual sampling interval: a segment must cover at least
    # round(min_duration_s / median_dt) samples (min 1). At 10 Hz a 0.25 s event
    # still needs ~3 samples; at 1 Hz a single sample is a real, counted event.
    median_dt = 0.0
    if len(timestamps) >= 2:
        median_dt = float(np.median(np.diff(timestamps)))
    min_samples = max(1, int(round(min_duration_s / median_dt))) if median_dt > 0 else 1

    segments: list[tuple[int, int]] = []
    for s, e in merged:
        if (e - s + 1) >= min_samples:
            segments.append((s, e))

    if cooldown_s > 0 and segments:
        segments = apply_cooldown(segments, timestamps, cooldown_s)

    return segments


def apply_cooldown(
    segments: list[tuple[int, int]],
    timestamps: np.ndarray,
    cooldown_s: float,
) -> list[tuple[int, int]]:
    """Drop segments whose START is within ``cooldown_s`` of the previous
    KEPT segment's start (Phase 10 noise dedup, one physical maneuver = one
    event). Applied AFTER signal filtering so a filtered-out first segment
    never suppresses its genuine successors."""
    if cooldown_s <= 0 or not segments:
        return segments
    kept: list[tuple[int, int]] = [segments[0]]
    last_start_t = float(timestamps[segments[0][0]])
    for s, e in segments[1:]:
        start_t = float(timestamps[s])
        if start_t - last_start_t >= cooldown_s:
            kept.append((s, e))
            last_start_t = start_t
    return kept


def count_events(
    mask: np.ndarray,
    timestamps: np.ndarray,
    min_duration_s: float,
    merge_gap_s: float,
    cooldown_s: float = 0.0,
) -> int:
    """Count sustained events from a boolean mask.

    Example use: harsh braking mask, harsh acceleration mask, aggressive turning mask.
    """
    return len(event_segments(mask, timestamps, min_duration_s, merge_gap_s, cooldown_s))


def filter_net_speed_delta(
    segments: list[tuple[int, int]],
    speed_raw: np.ndarray,
    *,
    direction: int,
    min_delta_mps: float,
    extreme_peak_mps2: float = 0.0,
    nominal_dt_s: float = 0.5,
) -> list[tuple[int, int]]:
    """Keep brake/accel segments that moved the vehicle's speed materially.

    GPS speed noise creates isolated single-sample dv crossings that move the
    actual speed very little. A genuine maneuver changes speed by at least
    ``min_delta_mps`` over its extent, measured with one sample of context on
    each side so a 1-sample event at low GPS rates is judged against its
    neighbours rather than trivially dropped.

    A short segment whose peak single-sample speed change exceeds
    ``extreme_peak_mps2 * nominal_dt_s`` (a reference |dv| in m/s) is kept even
    when the windowed net change is ~0 -- e.g. a panic stop captured in a
    single GPS fix: the speed dips and recovers, so net change is small, but
    the instantaneous deceleration is unmistakable and far above noise.

    The escape threshold is deliberately RATE-INDEPENDENT: expressed in m/s of
    speed change rather than m/s^2. A genuine maneuver sheds a constant |dv|
    per sample regardless of sampling frequency (a panic stop over one 1 Hz
    fix sheds ~11 m/s; over one 2 Hz fix ~5.5 m/s; both pass the 4.5 m/s
    reference), while per-sample GPS noise amplitude stays roughly constant
    across rates, so its m/s^2 value grows as 1/dt and would otherwise
    increasingly pass a fixed m/s^2 threshold at high sample rates.

    Note this is safe at high sample rates too: a genuine stop at 10 Hz spans
    many samples (a single 10 Hz sample would need ~45 m/s^2), so it is caught
    by the windowed net-delta check, and single-sample segments at high rates
    are noise -- which the m/s reference rejects.
    """
    if min_delta_mps <= 0 and extreme_peak_mps2 <= 0:
        return segments
    ref_speed_change = extreme_peak_mps2 * max(float(nominal_dt_s), 1e-9)
    n = len(speed_raw)
    if n < 2:
        # No speed differences exist; nothing can be a genuine maneuver.
        return segments
    kept: list[tuple[int, int]] = []
    for s, e in segments:
        lo = max(0, s - 1)
        hi = min(n - 1, e + 1)
        net = direction * (float(speed_raw[hi]) - float(speed_raw[lo]))
        if min_delta_mps > 0 and net >= min_delta_mps:
            kept.append((s, e))
            continue
        if ref_speed_change > 0:
            # Peak single-sample |dv| (m/s) with one sample of context either
            # side, so a 1-sample event is judged against its neighbours.
            peak_abs_dv = max(
                abs(float(speed_raw[i + 1]) - float(speed_raw[i]))
                for i in range(max(0, s - 1), min(n - 1, e + 1))
            )
            if peak_abs_dv >= ref_speed_change:
                kept.append((s, e))
    return kept


def event_severity_and_confidence(
    *,
    event_type: str,
    value: float,
    duration_s: float,
    activation: float,
    reference: float,
) -> tuple[float, float]:
    """Per-event severity (0..1) and confidence (0..1) for an event.

    Severity = how far the peak is past the activation threshold, normalised
    against a reference peak (both in the same units, value already positive).
    Confidence = margin-over-threshold + how long the signal persisted, so a
    short low-margin spike (typical sensor noise) scores low.

    Phase 10 (hackathon) — keeps Event / Confidence / Severity / Score-impact
    as separate concepts instead of conflating them with the event type.
    """
    activation = float(activation)
    reference = float(reference)
    if reference <= activation:
        severity = 0.0
    else:
        severity = float(np.clip((float(value) - activation) / (reference - activation), 0.0, 1.0))
    duration_factor = min(1.0, max(float(duration_s), 0.0) / 2.0)
    confidence = float(np.clip(0.4 + 0.4 * severity + 0.2 * duration_factor, 0.3, 0.98))
    return severity, confidence

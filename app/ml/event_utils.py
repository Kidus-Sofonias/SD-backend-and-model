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
) -> list[tuple[int, int]]:
    """Return sustained event index ranges from a boolean mask.

    Merges adjacent segments that are closer than merge_gap_s apart,
    then filters out any segment shorter than min_duration_s.

    Args:
        mask: Boolean array where True indicates an event condition.
        timestamps: Array of timestamps corresponding to each sample.
        min_duration_s: Minimum duration in seconds for a sustained event.
        merge_gap_s: Maximum gap in seconds between segments to merge.

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

    segments: list[tuple[int, int]] = []
    for s, e in merged:
        duration = timestamps[e] - timestamps[s]
        if duration >= min_duration_s:
            segments.append((s, e))

    return segments


def count_events(
    mask: np.ndarray,
    timestamps: np.ndarray,
    min_duration_s: float,
    merge_gap_s: float,
) -> int:
    """Count sustained events from a boolean mask.

    Example use: harsh braking mask, harsh acceleration mask, aggressive turning mask.
    """
    return len(event_segments(mask, timestamps, min_duration_s, merge_gap_s))

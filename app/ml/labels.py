"""Shared review-label tier classification (hackathon).

Single source of truth for deciding whether a stored trip review label is real
human ground truth (admin UI), a synthetic generator label, or a score-derived
demo label (seed script). Used by:

- ``scripts.build_training_dataset`` (label priority tiers)
- ``scripts.reviewed_model_analysis`` (real-review analysis)
- ``app.services.trip_processing_service`` (review-gated retrain counter)

Keeping this in one place prevents the three consumers from drifting apart.
"""

from __future__ import annotations


def review_label_tier(reviewed_label_source: str | None) -> str:
    """Classify a review label source into ``reviewed_real``,
    ``reviewed_synthetic`` or ``reviewed_demo``.

    Case-insensitive (``source.lower()``) so behavior is identical on SQLite
    and PostgreSQL. A missing source is treated as ``reviewed_real`` (the
    admin review default).
    """
    source = (reviewed_label_source or "reviewed_real").strip().lower()
    if "synthetic" in source:
        return "reviewed_synthetic"
    if "demo" in source:
        return "reviewed_demo"
    return "reviewed_real"


def is_real_review_source(reviewed_label_source: str | None) -> bool:
    """True when the label came from the admin review screen (human ground
    truth), not from the synthetic generator or the demo seed script."""
    return review_label_tier(reviewed_label_source) == "reviewed_real"

# File role: Backward-compatible wrapper for the typo-named ML inference module.
# New code should import from app.ml.inference.

from app.ml.inference import MODELS_DIR, ModelScorer

__all__ = ["MODELS_DIR", "ModelScorer"]

# File role: Runtime ML model loader and predictor for backend inference.
# Loads the latest trained sklearn model artifact and scores a trip feature row.
# Connects to:
# - app.ml.schemas
# - artifacts/models
# Key symbols/vars:
# - MODELS_DIR
# - ModelScorer

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from app.ml.model_registry import get_production_model_version, model_path_for
from app.ml.schemas import FEATURE_COLUMNS_FV1, FEATURE_VERSION


MODELS_DIR = Path("artifacts/models")


class ModelScorer:
    def __init__(self) -> None:
        self.model = None
        self.model_version: str | None = None
        self.feature_version = FEATURE_VERSION

    def load_latest(self) -> bool:
        model_version = get_production_model_version()
        if not model_version:
            return False

        self.model_version = model_version

        model_path = model_path_for(self.model_version)
        if not model_path.exists():
            # Backward-compatible fallback for older artifact names.
            fallback = MODELS_DIR / f"best_model_{self.model_version}_{FEATURE_VERSION}.joblib"
            if fallback.exists():
                model_path = fallback
            else:
                return False

        self.model = joblib.load(model_path)
        return True

    def predict(self, trip_features: dict) -> dict:
        if self.model is None:
            loaded = self.load_latest()
            if not loaded:
                raise RuntimeError("No trained model artifact found")

        row = {col: trip_features[col] for col in FEATURE_COLUMNS_FV1}
        features_df = pd.DataFrame([row])

        pred = int(self.model.predict(features_df)[0])

        result = {
            "prediction": pred,  # 0=safe, 1=risky
            "model_version": self.model_version,
            "feature_version": self.feature_version,
        }

        if hasattr(self.model, "predict_proba"):
            result["risk_probability"] = float(self.model.predict_proba(features_df)[0][1])
        else:
            result["risk_probability"] = float(pred)

        return result

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.ml.schemas import FEATURE_VERSION


MODELS_DIR = Path("artifacts/models")
PRODUCTION_MANIFEST_PATH = MODELS_DIR / f"production_model_{FEATURE_VERSION}.json"


def metadata_path_for(version: str) -> Path:
    return MODELS_DIR / f"metadata_{FEATURE_VERSION}_{version}.json"


def model_path_for(version: str) -> Path:
    return MODELS_DIR / f"model_{FEATURE_VERSION}_{version}.joblib"


def load_metadata(version: str) -> dict[str, Any]:
    path = metadata_path_for(version)
    if not path.exists():
        raise FileNotFoundError(f"Metadata artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_production_manifest() -> dict[str, Any] | None:
    if not PRODUCTION_MANIFEST_PATH.exists():
        return None
    return json.loads(PRODUCTION_MANIFEST_PATH.read_text(encoding="utf-8"))


def save_production_manifest(payload: dict[str, Any]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PRODUCTION_MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_production_model_version() -> str | None:
    manifest = load_production_manifest()
    if manifest and manifest.get("model_version"):
        return str(manifest["model_version"])

    metadata_files = sorted(MODELS_DIR.glob(f"metadata_{FEATURE_VERSION}_*.json"))
    if not metadata_files:
        return None

    latest_metadata = json.loads(metadata_files[-1].read_text(encoding="utf-8"))
    model_version = latest_metadata.get("model_version")
    return str(model_version) if model_version else None

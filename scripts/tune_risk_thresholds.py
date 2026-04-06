"""Phase 8 threshold-sweep entrypoint for reviewed-label model tuning."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.schemas import FEATURE_VERSION
from scripts.reviewed_model_analysis import (
    DEFAULT_THRESHOLDS,
    load_reviewed_trip_rows,
    score_reviewed_trip_rows,
)
from scripts.reporting_utils import build_threshold_report


REPORTS_DIR = Path("artifacts/reports")


def run_threshold_tuning(
    *,
    model_version: str | None = None,
    include_synthetic: bool = False,
    thresholds: list[float] | None = None,
) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    threshold_values = thresholds or DEFAULT_THRESHOLDS
    rows = load_reviewed_trip_rows(include_synthetic=include_synthetic)
    scored = score_reviewed_trip_rows(rows, model_version=model_version)
    report = build_threshold_report(scored, threshold_values)
    report.update(
        {
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "model_version": None if scored.empty else scored["model_version"].iloc[0],
            "feature_version": FEATURE_VERSION,
            "reviewed_real_row_count": int((scored.get("label_tier") == "reviewed_real").sum()) if "label_tier" in scored else 0,
            "reviewed_synthetic_row_count": int((scored.get("label_tier") == "reviewed_synthetic").sum()) if "label_tier" in scored else 0,
        }
    )

    version = str(report["model_version"] or "unknown")
    out_path = REPORTS_DIR / f"threshold_tuning_{FEATURE_VERSION}_{version}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(out_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multiple risky-trip probability thresholds")
    parser.add_argument("--model-version", required=False, help="Model version to analyze")
    parser.add_argument("--include-synthetic", action="store_true", help="Include reviewed synthetic labels")
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=None,
        help="Optional threshold sweep values. Example: --thresholds 0.3 0.4 0.5 0.6 0.7",
    )
    args = parser.parse_args()

    result = run_threshold_tuning(
        model_version=args.model_version,
        include_synthetic=args.include_synthetic,
        thresholds=args.thresholds,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

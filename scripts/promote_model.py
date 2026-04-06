from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.model_registry import load_metadata, save_production_manifest
from app.ml.schemas import FEATURE_VERSION


REPORTS_DIR = Path("artifacts/reports")


def _bootstrap_threshold_check(candidate_metadata: dict) -> dict:
    metrics = candidate_metadata.get("metrics", {})
    checks = {
        "bootstrap_risky_trip_f1_min": bool(float(metrics.get("risky_trip_f1", 0.0)) >= 0.55),
        "bootstrap_false_positive_rate_max": bool(float(metrics.get("false_positive_rate", 1.0)) <= 0.35),
        "bootstrap_brier_score_max": bool(float(metrics.get("brier_score", 1.0)) <= 0.25),
    }
    checks["promote"] = all(checks.values())
    return checks


def promote_from_report(compare_report: dict, regression_tests_passed: bool = True) -> dict:
    checks = dict(compare_report.get("promotion_check", {}))
    checks["regression_tests_passed"] = bool(regression_tests_passed)
    checks["promote"] = bool(checks.get("promote")) and checks["regression_tests_passed"]

    candidate_version = str(compare_report["candidate_model_version"])
    if not checks["promote"]:
        return {
            "promoted": False,
            "candidate_model_version": candidate_version,
            "checks": checks,
        }

    candidate_metadata = load_metadata(candidate_version)
    manifest = {
        "feature_version": FEATURE_VERSION,
        "model_version": candidate_version,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "source_compare_report": compare_report,
        "metrics": candidate_metadata.get("metrics", {}),
    }
    save_production_manifest(manifest)
    return {
        "promoted": True,
        "candidate_model_version": candidate_version,
        "checks": checks,
        "manifest_path": "artifacts/models/production_model_fv1.json",
    }


def bootstrap_promote_model(candidate_version: str, regression_tests_passed: bool = True) -> dict:
    candidate_metadata = load_metadata(candidate_version)
    checks = _bootstrap_threshold_check(candidate_metadata)
    checks["regression_tests_passed"] = bool(regression_tests_passed)
    checks["promote"] = bool(checks["promote"]) and checks["regression_tests_passed"]

    if not checks["promote"]:
        return {
            "promoted": False,
            "candidate_model_version": candidate_version,
            "checks": checks,
            "bootstrap": True,
        }

    manifest = {
        "feature_version": FEATURE_VERSION,
        "model_version": candidate_version,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "promotion_type": "bootstrap",
        "metrics": candidate_metadata.get("metrics", {}),
        "bootstrap_checks": checks,
    }
    save_production_manifest(manifest)
    return {
        "promoted": True,
        "candidate_model_version": candidate_version,
        "checks": checks,
        "manifest_path": "artifacts/models/production_model_fv1.json",
        "bootstrap": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a candidate model if comparison checks pass")
    parser.add_argument("--compare-report", required=True, help="Path to compare report JSON")
    parser.add_argument(
        "--allow-failed-regression",
        action="store_true",
        help="Override regression test gate for manual promotion",
    )
    args = parser.parse_args()

    report_path = Path(args.compare_report)
    if not report_path.exists():
        report_path = REPORTS_DIR / args.compare_report
    if not report_path.exists():
        raise FileNotFoundError(f"Compare report not found: {args.compare_report}")

    compare_report = json.loads(report_path.read_text(encoding="utf-8"))
    result = promote_from_report(
        compare_report=compare_report,
        regression_tests_passed=not args.allow_failed_regression,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

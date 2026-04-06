from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.model_registry import get_production_model_version
from app.db.session import SessionLocal
from scripts.build_training_dataset import main as build_training_dataset_main
from scripts.compare_models import run_compare
from scripts.generate_synthetic_trips import count_completed_trips, generate_synthetic_trips
from scripts.promote_model import bootstrap_promote_model, promote_from_report
from scripts.train_model import run_training


REPORTS_DIR = Path("artifacts/reports")
PYTEST_TEMP_DIR = BACKEND_ROOT / ".pytest_temp"
PYTEST_CACHE_DIR = BACKEND_ROOT / ".pytest_cache_local"


def _safe_remove_tree(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path)
    except OSError:
        # Best-effort cleanup only; do not fail the model refresh if temp files are still in use.
        pass


def _run_regression_tests() -> dict[str, Any]:
    _safe_remove_tree(PYTEST_TEMP_DIR)
    _safe_remove_tree(PYTEST_CACHE_DIR)
    PYTEST_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    PYTEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TMP"] = str(PYTEST_TEMP_DIR)
    env["TEMP"] = str(PYTEST_TEMP_DIR)
    env["TMPDIR"] = str(PYTEST_TEMP_DIR)
    env.pop("PYTEST_ADDOPTS", None)

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests",
                f"--basetemp={PYTEST_TEMP_DIR}",
                "-o",
                f"cache_dir={PYTEST_CACHE_DIR}",
            ],
            cwd=BACKEND_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        return {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "basetemp": str(PYTEST_TEMP_DIR),
            "cache_dir": str(PYTEST_CACHE_DIR),
            "stdout_tail": result.stdout.strip().splitlines()[-20:],
            "stderr_tail": result.stderr.strip().splitlines()[-20:],
        }
    finally:
        _safe_remove_tree(PYTEST_TEMP_DIR)
        _safe_remove_tree(PYTEST_CACHE_DIR)


def _top_up_completed_trips(
    *,
    target_trip_count: int | None,
    user_id: str | None,
    samples_per_trip: int,
    dt: float,
    seed: int,
) -> dict[str, Any] | None:
    if not target_trip_count:
        return None

    db = SessionLocal()
    try:
        current_completed = count_completed_trips(db)
    finally:
        db.close()

    additional_needed = max(target_trip_count - current_completed, 0)
    if additional_needed <= 0:
        return {
            "target_trip_count": target_trip_count,
            "completed_before": current_completed,
            "completed_after": current_completed,
            "generated_count": 0,
            "samples_per_trip": samples_per_trip,
            "dt": dt,
            "seed": seed,
            "user_id": user_id,
        }

    generated = generate_synthetic_trips(
        count=additional_needed,
        user_id=user_id,
        samples_per_trip=samples_per_trip,
        dt=dt,
        seed=seed,
    )
    return {
        "target_trip_count": target_trip_count,
        "completed_before": current_completed,
        "completed_after": current_completed + int(generated["created_count"]),
        "generated_count": int(generated["created_count"]),
        "samples_per_trip": samples_per_trip,
        "dt": dt,
        "seed": seed,
        "user_id": generated["user_id"],
    }


def main(
    *,
    target_trip_count: int | None = None,
    user_id: str | None = None,
    samples_per_trip: int = 240,
    dt: float = 0.5,
    seed: int = 42,
    skip_tests: bool = False,
) -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    current_version = get_production_model_version()
    synthetic_top_up = _top_up_completed_trips(
        target_trip_count=target_trip_count,
        user_id=user_id,
        samples_per_trip=samples_per_trip,
        dt=dt,
        seed=seed,
    )

    build_training_dataset_main()
    training_report = run_training()
    if not training_report:
        raise RuntimeError("Training did not produce a report.")

    candidate_version = str(training_report["best_model_version"])

    comparison_report = None
    if current_version:
        comparison_report = run_compare(current_version=current_version, candidate_version=candidate_version)

    regression = {
        "passed": True,
        "skipped": True,
        "reason": "Regression tests skipped by CLI flag.",
    } if skip_tests else _run_regression_tests()

    promotion = None
    if comparison_report:
        promotion = promote_from_report(
            compare_report=comparison_report,
            regression_tests_passed=bool(regression["passed"]),
        )
    else:
        promotion = bootstrap_promote_model(
            candidate_version=candidate_version,
            regression_tests_passed=bool(regression["passed"]),
        )

    cycle_report = {
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "current_model_version": current_version,
        "candidate_model_version": candidate_version,
        "synthetic_top_up": synthetic_top_up,
        "training_report": training_report,
        "comparison_report": comparison_report,
        "regression": regression,
        "promotion": promotion,
    }

    out_path = REPORTS_DIR / f"refresh_cycle_{candidate_version}.json"
    out_path.write_text(json.dumps(cycle_report, indent=2), encoding="utf-8")
    print(json.dumps(cycle_report, indent=2))
    return cycle_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild dataset, retrain, compare, test, and promote the best model")
    parser.add_argument("--target-trip-count", type=int, default=None, help="Top up completed trips to this count using synthetic data before training")
    parser.add_argument("--user-id", type=str, default=None, help="Existing user ID to assign generated synthetic trips to")
    parser.add_argument("--samples-per-trip", type=int, default=240, help="Samples per generated synthetic trip")
    parser.add_argument("--dt", type=float, default=0.5, help="Seconds between generated samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic trip generation")
    parser.add_argument("--skip-tests", action="store_true", help="Skip regression tests during the refresh cycle")
    args = parser.parse_args()
    main(
        target_trip_count=args.target_trip_count,
        user_id=args.user_id,
        samples_per_trip=args.samples_per_trip,
        dt=args.dt,
        seed=args.seed,
        skip_tests=args.skip_tests,
    )

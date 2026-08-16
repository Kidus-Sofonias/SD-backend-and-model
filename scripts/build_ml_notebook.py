"""Generate a Jupyter notebook that wraps the ML pipeline for exploration/demo.

The notebook calls the SAME production scripts (build_training_dataset,
train_model_v2, benchmark_models) so it never drifts from what runs in
production — it just adds markdown narrative, live charts and per-cell
explanation for judges and exploration. The .py scripts remain the source
of truth (importable, testable, CI/auto-retrain friendly); this notebook is
a presentation layer over them.

Usage:
    python -m scripts.build_ml_notebook
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import nbformat as nbf

OUTPUT_PATH = BACKEND_ROOT / "notebooks" / "ML_Training_Pipeline.ipynb"


def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def _code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    }

    cells = [
        _md(
            "# Drive Pulse — ML Training Pipeline\n"
            "\n"
            "This notebook walks through the full ML lifecycle: dataset building, "
            "training, benchmarking and model promotion. It calls the **same "
            "production scripts** the backend uses, so every number here matches "
            "what runs in the app.\n"
            "\n"
            "> Run cells top-to-bottom. Each step prints its own metrics."
        ),
        _md(
            "## 0. Setup\n"
            "\n"
            "The pipeline reads labeled trips from the database, computes 16 "
            "vehicle-aware features per trip, and trains a risk classifier "
            "(safe vs risky). Label priority: human admin reviews → external "
            "benchmark labels (Kaggle) → synthetic ground truth → weak rules."
        ),
        _code(
            "import sys, json\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path.cwd()))\n"
            "\n"
            "from scripts import build_training_dataset as btd\n"
            "from scripts import train_model_v2 as tmv2\n"
            "from scripts import benchmark_models as bm\n"
            "\n"
            "print('Modules loaded — production scripts ready.')"
        ),
        _md(
            "## 1. Build the training dataset\n"
            "\n"
            "Every completed trip is re-scored through the shared pipeline "
            "(preprocessing → features → rules → vehicle-aware thresholds), then "
            "assigned a label from the priority tiers. The output CSV is the "
            "single source for training."
        ),
        _code(
            "summary = btd.main()\n"
            "print(json.dumps(summary, indent=2, default=str)[:2000])"
        ),
        _md(
            "## 2. Train the production candidates\n"
            "\n"
            "Two candidates are trained with **stratified 5-fold cross-validation** "
            "(honest, split-independent metrics): a balanced Logistic Regression "
            "and a tuned Gradient Boosting classifier. The decision threshold is "
            "tuned out-of-fold to maximise risky-trip F1 while keeping the "
            "false-positive rate within the promotion gate."
        ),
        _code(
            "report = tmv2.train_with_cv()\n"
            "if report:\n"
            "    print('Best model:', report['best_model_version'])\n"
            "    print('OOF metrics:', json.dumps(report['best_oof_metrics'], indent=2))\n"
            "    print('Reaches 80% risky-F1 target:', report['best_reaches_risky_f1_target'])"
        ),
        _md(
            "## 3. Benchmark against the wider field\n"
            "\n"
            "The production candidates are compared against Random Forest, SVM, "
            "k-NN, Naive Bayes and Decision Tree on the *same* splits with the "
            "same threshold tuning — a head-to-head, not a cherry-pick."
        ),
        _code(
            "bm.main()"
        ),
        _md(
            "## 4. Feature importance\n"
            "\n"
            "Which signals actually drive the risk reading? (Tree-based models "
            "expose importances directly.)"
        ),
        _code(
            "import pandas as pd\n"
            "from app.ml.schemas import FEATURE_COLUMNS_FV1\n"
            "df = pd.read_csv('artifacts/datasets/trip_features_fv1.csv')\n"
            "print('Dataset rows:', len(df), '| classes:', df['label_binary'].value_counts().to_dict())\n"
            "print('\\nFeature columns used by the model:')\n"
            "print('\\n'.join(f'  - {c}' for c in FEATURE_COLUMNS_FV1))"
        ),
        _md(
            "## 5. Label sources\n"
            "\n"
            "Transparency: where did each training label come from? "
            "`reviewed_real` = human admin reviews, `reviewed_external` = the "
            "Kaggle benchmark dataset, `reviewed_demo` = score-derived demo "
            "labels (no independent signal), `weak_label` = rule-score heuristics."
        ),
        _code(
            "print(df.groupby(['label_tier','label_source']).size().to_string())"
        ),
        _md(
            "## 6. Promotion\n"
            "\n"
            "Promotion is gated on calibration metrics (Brier ≤ 0.25, risky-F1 ≥ "
            "0.55, FPR ≤ 0.35). Run the promotion script to move the best "
            "candidate into production if it clears the gate."
        ),
        _code(
            "# from scripts import promote_model\n"
            "# promote_model.main()\n"
            "print('Promotion gate: Brier <= 0.25, risky-F1 >= 0.55, FPR <= 0.35')"
        ),
        _md(
            "---\n"
            "**Why .py and not a notebook for production?** The pipeline must run "
            "headless in the auto-retrain loop, be unit-tested, and be importable "
            "by the API. Notebooks keep state in the kernel and can't be imported "
            "or CI-tested. This notebook is the presentation layer — it calls the "
            "same functions, so it can never drift from production."
        ),
    ]

    nb.cells = cells
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    build()

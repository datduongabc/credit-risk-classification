"""Project-wide constants: paths, dataset source, column roles and the random seed."""

from __future__ import annotations

import os
from pathlib import Path

SEED = 42

# Public download sources, tried in order, so the notebook needs no API keys or personal drive
# (assignment requirement). Kaggle answers anonymous downloads from Colab with HTTP 404, so the
# primary source is a GitHub release mirror of the CC0 dataset; Kaggle stays as a fallback.
DATASET_URLS = (
    "https://github.com/datduongabc/credit-risk-classification/releases/download/data-v1/credit_risk_dataset.csv",
    "https://www.kaggle.com/api/v1/datasets/download/laotse/credit-risk-dataset",
)
# Pins the exact file the reported results were produced from, whichever source served it.
DATASET_SHA256 = "ce3c6d2167717bf1627d1c0c81cbccd28323cd4aa7b96d542599366d5ff6aac8"
DATASET_PAGE = "https://www.kaggle.com/datasets/laotse/credit-risk-dataset"
CSV_NAME = "credit_risk_dataset.csv"

PROJECT_ROOT = Path(os.environ.get("CREDITRISK_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "generated"

TARGET = "loan_status"

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
NOMINAL_FEATURES = ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]
# loan_grade has a natural risk order (A best ... G worst), so it can also be ordinal-encoded.
ORDINAL_FEATURES = ["loan_grade"]
GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]
CATEGORICAL_FEATURES = NOMINAL_FEATURES + ORDINAL_FEATURES
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CLASS_NAMES = ["Non-default (0)", "Default (1)"]


def ensure_dirs() -> None:
    for d in (DATA_DIR, FEATURES_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)

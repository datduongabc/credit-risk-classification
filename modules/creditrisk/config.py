"""Project-wide constants: paths, dataset source, column roles and the random seed."""

from __future__ import annotations

import os
from pathlib import Path

SEED = 42

# Public, unauthenticated Kaggle download endpoint, so the notebook can fetch the data on
# Colab without API keys or mounting a personal drive (assignment requirement).
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/laotse/credit-risk-dataset"
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

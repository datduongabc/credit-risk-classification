"""Automatic EDA overview with ydata-profiling, written to reports/eda_profile.html.

ydata-profiling pins older numpy/pandas/matplotlib releases, so installing it inside the Colab
notebook could downgrade libraries mid-session and force a runtime restart (breaking
"Run all"). It therefore runs once, in its own environment:

    python -m venv .venv-profile && . .venv-profile/bin/activate
    pip install ydata-profiling "setuptools<81"
    python scripts/make_profile.py

The notebook's targeted figures (modules/creditrisk/eda.py) complement this overview with
class-conditional views that a generic profiler does not produce.
"""

from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.kaggle.com/api/v1/datasets/download/laotse/credit-risk-dataset"


def main() -> None:
    csv = ROOT / "data" / "credit_risk_dataset.csv"
    df = pd.read_csv(csv) if csv.exists() else pd.read_csv(URL, compression="zip")
    report = ProfileReport(
        df,
        title="Credit Risk Dataset - automatic profile",
        correlations={"spearman": {"calculate": True}, "pearson": {"calculate": False}},
        interactions={"targets": ["loan_status"]},
        explorative=True,
    )
    out = ROOT / "reports" / "eda_profile.html"
    report.to_file(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

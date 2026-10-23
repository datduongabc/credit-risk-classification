"""Pipeline steps that scikit-learn does not provide out of the box.

Everything else in the pipeline is a stock scikit-learn / imbalanced-learn component; only
grade-aware imputation and IQR winsorizing need custom code. Stateless transforms are thin
``FunctionTransformer`` wrappers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

ENGINEERED_FEATURES = ["log_income", "log_loan_amnt", "income_per_hist_year", "rate_x_burden"]


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Fill a numeric column with the median of its group (learned on the training fold).

    Interest rates are set from the loan grade, so the grade-wise median is a much better
    guess for a missing ``loan_int_rate`` than the global median that ``SimpleImputer`` uses.
    """

    def __init__(self, column: str = "loan_int_rate", group: str = "loan_grade"):
        self.column = column
        self.group = group

    def fit(self, X: pd.DataFrame, y=None):
        self.medians_ = X.groupby(self.group)[self.column].median().to_dict()
        self.global_median_ = float(X[self.column].median())
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        fill = X[self.group].map(self.medians_).astype(float).fillna(self.global_median_)
        X[self.column] = X[self.column].fillna(fill)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class IQRClipper(BaseEstimator, TransformerMixin):
    """Winsorize each column to [Q1 - k*IQR, Q3 + k*IQR] with bounds learned on training data."""

    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_, self.upper_ = q1 - self.k * iqr, q3 + self.k * iqr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _add_ratio_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["log_income"] = np.log1p(X["person_income"])
    X["log_loan_amnt"] = np.log1p(X["loan_amnt"])
    X["income_per_hist_year"] = X["person_income"] / (X["cb_person_cred_hist_length"] + 1)
    # Interest rate times the share of income borrowed approximates the repayment burden.
    X["rate_x_burden"] = X["loan_int_rate"] * X["loan_percent_income"]
    return X


def _ratio_feature_names(_, input_features):
    return np.concatenate([input_features, ENGINEERED_FEATURES])


def feature_engineer() -> FunctionTransformer:
    """Domain ratio features; NaN-tolerant because imputation happens afterwards."""
    return FunctionTransformer(_add_ratio_features, feature_names_out=_ratio_feature_names)


def log_transform() -> FunctionTransformer:
    """log1p for the right-skewed, non-negative numeric columns (income, amounts)."""
    return FunctionTransformer(np.log1p, feature_names_out="one-to-one")

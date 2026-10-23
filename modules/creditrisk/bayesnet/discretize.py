"""Turn the mixed-type credit table into integer-coded discrete variables for the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from .. import config


class Discretizer:
    """Quantile-bin numeric columns (bins learned on training data) and code categoricals.

    Missing values stay ``-1`` so that inference treats them as unobserved rather than
    imputing them.
    """

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        self.variables_ = list(config.FEATURES) + [config.TARGET]
        self.binners_, self.states_ = {}, {}
        for col in config.NUMERIC_FEATURES:
            vals = X[col].dropna().to_numpy().reshape(-1, 1)
            kb = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy="quantile", quantile_method="averaged_inverted_cdf")
            kb.fit(vals)
            self.binners_[col] = kb
            edges = kb.bin_edges_[0]
            self.states_[col] = [f"[{lo:g}, {hi:g})" for lo, hi in zip(edges[:-1], edges[1:])]
        for col in config.CATEGORICAL_FEATURES:
            cats = config.GRADE_ORDER if col == "loan_grade" else sorted(X[col].dropna().unique())
            self.states_[col] = list(cats)
        self.states_[config.TARGET] = [0, 1]
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray:
        out = np.full((len(X), len(self.variables_)), -1, dtype=np.int64)
        for j, col in enumerate(self.variables_):
            if col == config.TARGET:
                if y is not None:
                    out[:, j] = np.asarray(y, dtype=np.int64)
                continue
            s = X[col]
            mask = s.notna().to_numpy()
            if col in self.binners_:
                out[mask, j] = self.binners_[col].transform(s[mask].to_numpy().reshape(-1, 1))[:, 0]
            else:
                lookup = {c: k for k, c in enumerate(self.states_[col])}
                out[mask, j] = s[mask].map(lookup).fillna(-1).astype(np.int64).to_numpy()
        return out

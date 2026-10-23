"""Exploratory data analysis: summary tables and figures saved for the report."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

from . import config

PALETTE = {0: "#4c72b0", 1: "#dd8452"}


def _save(fig, name):
    if name:
        config.ensure_dirs()
        fig.savefig(config.FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=130, bbox_inches="tight")
    return fig


def overview(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, missing values, cardinality and range or top categories."""
    rows = []
    for col in df.columns:
        s = df[col]
        numeric = pd.api.types.is_numeric_dtype(s) and col != config.TARGET
        rows.append(
            {
                "column": col,
                "type": "numeric" if numeric else "categorical",
                "missing": int(s.isna().sum()),
                "missing_pct": 100 * s.isna().mean(),
                "unique": int(s.nunique()),
                "summary": (
                    f"{s.min():g} .. {s.max():g} (median {s.median():g})"
                    if numeric
                    else ", ".join(f"{k} ({v})" for k, v in s.value_counts().head(4).items())
                ),
            }
        )
    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df[config.NUMERIC_FEATURES].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    d["skew"] = df[config.NUMERIC_FEATURES].skew()
    return d


def plot_target(df, name="eda_target"):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = df[config.TARGET].value_counts().sort_index()
    bars = ax.bar(["Non-default (0)", "Default (1)"], counts.values, color=[PALETTE[0], PALETTE[1]])
    for b, c in zip(bars, counts.values):
        ax.annotate(f"{c:,}\n({100 * c / counts.sum():.1f}%)", (b.get_x() + b.get_width() / 2, c), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of loans")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_title("Target distribution (loan_status)")
    return _save(fig, name)


def plot_numeric_distributions(df, name="eda_numeric"):
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, cols):
        data = df[[col, config.TARGET]].dropna()
        log = col in ("person_income", "loan_amnt")
        sns.histplot(
            data, x=col, hue=config.TARGET, ax=ax, bins=40, stat="density", common_norm=False,
            element="step", palette=PALETTE, log_scale=log,
        )
        ax.set_title(col + (" (log scale)" if log else ""), fontsize=10)
        ax.set_xlabel("")
    axes.flat[-1].axis("off")
    fig.suptitle("Numeric features by loan status after removing impossible values (densities normalized per class)")
    fig.tight_layout()
    return _save(fig, name)


def plot_outliers(df, name="eda_boxplots"):
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(1, len(cols), figsize=(18, 4))
    for ax, col in zip(axes, cols):
        log = col == "person_income"  # a single 6M income would flatten the linear axis
        sns.boxplot(y=df[col], ax=ax, color="#9ecae1", fliersize=1.5, log_scale=log)
        ax.set_title(col + (" (log)" if log else ""), fontsize=9)
        ax.set_ylabel("")
    fig.suptitle("Box plots of the raw numeric features (before cleaning)")
    fig.tight_layout()
    return _save(fig, name)


def plot_categorical(df, name="eda_categorical"):
    cols = config.CATEGORICAL_FEATURES
    fig, axes = plt.subplots(2, len(cols), figsize=(18, 7.5))
    for j, col in enumerate(cols):
        order = config.GRADE_ORDER if col == "loan_grade" else df[col].value_counts().index.tolist()
        sns.countplot(data=df, x=col, order=order, ax=axes[0, j], color="#9ecae1")
        axes[0, j].set_title(f"{col}: counts", fontsize=10)
        rate = df.groupby(col)[config.TARGET].mean().reindex(order)
        axes[1, j].bar(range(len(order)), rate.values, color=PALETTE[1])
        axes[1, j].axhline(df[config.TARGET].mean(), color="k", ls="--", lw=0.8)
        axes[1, j].set_xticks(range(len(order)), order)
        axes[1, j].set_title(f"{col}: default rate", fontsize=10)
        axes[1, j].set_ylim(0, 1)
        for ax in axes[:, j]:
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.set_xlabel("")
    fig.tight_layout()
    return _save(fig, name)


def plot_correlations(df, name="eda_correlation"):
    d = df[config.NUMERIC_FEATURES + [config.TARGET]].copy()
    d["loan_grade_code"] = df["loan_grade"].map({g: i for i, g in enumerate(config.GRADE_ORDER)})
    d["prev_default_Y"] = (df["cb_person_default_on_file"] == "Y").astype(int)
    corr = d.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 8})
    ax.set_title("Spearman correlation (numeric features, ordinal grade, target)")
    fig.tight_layout()
    return _save(fig, name), corr


def plot_missingness(df, name="eda_missing"):
    cols = ["person_emp_length", "loan_int_rate"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    miss = df[cols].isna()
    rates = pd.DataFrame(
        {"missing": [df.loc[miss[c], config.TARGET].mean() for c in cols],
         "observed": [df.loc[~miss[c], config.TARGET].mean() for c in cols]},
        index=cols,
    )
    rates.plot.bar(ax=axes[0], color=["#c44e52", "#55a868"], rot=0)
    axes[0].set_ylabel("Default rate")
    axes[0].set_title("Default rate when the value is missing vs observed")
    by_grade = df.groupby("loan_grade")["loan_int_rate"].apply(lambda s: s.isna().mean()).reindex(config.GRADE_ORDER)
    axes[1].bar(config.GRADE_ORDER, 100 * by_grade.values, color="#8172b2")
    axes[1].set_ylabel("% missing")
    axes[1].set_title("loan_int_rate missing rate by loan grade")
    fig.tight_layout()
    return _save(fig, name), rates


def plot_rate_by_grade(df, name="eda_rate_grade"):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="loan_grade", y="loan_int_rate", order=config.GRADE_ORDER, ax=ax, color="#9ecae1", fliersize=1.5)
    ax.set_title("Interest rate is largely determined by the loan grade")
    return _save(fig, name)


def plot_burden_scatter(df, name="eda_burden", n=6000):
    sample = df.sample(min(n, len(df)), random_state=config.SEED)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=sample, x="person_income", y="loan_percent_income", hue=config.TARGET, palette=PALETTE,
        s=8, alpha=0.5, ax=ax, edgecolor=None,
    )
    ax.set_xscale("log")
    ax.set_title("Loan-to-income ratio vs income (sample of 6,000 loans)")
    return _save(fig, name)


def mutual_information(df) -> pd.Series:
    """MI between each raw feature and the target (categoricals as codes, NaN as median)."""
    X = df[config.FEATURES].copy()
    discrete = []
    for col in X.columns:
        if col in config.CATEGORICAL_FEATURES:
            X[col] = X[col].astype("category").cat.codes
            discrete.append(True)
        else:
            X[col] = X[col].fillna(X[col].median())
            discrete.append(False)
    mi = mutual_info_classif(X, df[config.TARGET], discrete_features=np.array(discrete), random_state=config.SEED)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def plot_mutual_information(mi: pd.Series, name="eda_mutual_info"):
    fig, ax = plt.subplots(figsize=(7, 4))
    mi.sort_values().plot.barh(ax=ax, color="#4c72b0")
    ax.set_xlabel("Mutual information with loan_status (nats)")
    ax.set_title("Univariate relevance of each feature")
    fig.tight_layout()
    return _save(fig, name)

"""Metrics, cross-validation helpers and evaluation plots shared by every model family.

Metric policy: missing a defaulter (false negative) costs a lender far more than rejecting a
good applicant (false positive), so recall on the default class matters most. Recall alone is
degenerate (flagging everyone gives recall 1.0), so models are ranked by PR-AUC, which is
threshold-free and robust to the 22% class imbalance, and the operating threshold is tuned
for F2, which weights recall twice as much as precision.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from . import config

F2_SCORER = make_scorer(fbeta_score, beta=2)

CV_SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "f2": F2_SCORER,
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "accuracy": "accuracy",
}
PRIMARY_METRIC = "pr_auc"


def cv_splitter(n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)


def cross_validate_pipeline(pipe, X, y, n_splits: int = 5, n_jobs: int | None = None) -> dict:
    t0 = time.perf_counter()
    res = cross_validate(pipe, X, y, cv=cv_splitter(n_splits), scoring=CV_SCORING, n_jobs=n_jobs)
    out = {}
    for name in CV_SCORING:
        scores = res[f"test_{name}"]
        out[name] = float(np.mean(scores))
        out[f"{name}_std"] = float(np.std(scores))
    out["fit_time"] = float(np.mean(res["fit_time"]))
    out["wall_time"] = time.perf_counter() - t0
    return out


def scores_of(model, X) -> np.ndarray:
    """Continuous score for ranking metrics: probability if available, else decision value."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "f2": fbeta_score(y_true, y_pred, beta=2),
    }
    if y_score is not None:
        m["roc_auc"] = roc_auc_score(y_true, y_score)
        m["pr_auc"] = average_precision_score(y_true, y_score)
    return {k: float(v) for k, v in m.items()}


def plot_roc_pr(curves: dict[str, tuple[np.ndarray, np.ndarray]], path=None, title_suffix=""):
    """curves maps a model label to (y_true, y_score)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, (y_true, y_score) in curves.items():
        RocCurveDisplay.from_predictions(y_true, y_score, name=label, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_true, y_score, name=label, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_title(f"ROC curve{title_suffix}")
    base = float(np.mean(next(iter(curves.values()))[0]))
    axes[1].axhline(base, color="k", ls="--", lw=0.8, label=f"Chance (AP={base:.2f})")
    axes[1].set_title(f"Precision-recall curve{title_suffix}")
    for ax in axes:
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrices(preds: dict[str, tuple[np.ndarray, np.ndarray]], path=None, ncols=3):
    n = len(preds)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    for ax, (label, (y_true, y_pred)) in zip(axes.flat, preds.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=["Non-default", "Default"], ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(label, fontsize=10)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def f2_threshold(y_true, y_score) -> float:
    """F2-optimal threshold on a validation set (mirrors the traditional pipeline's policy)."""
    grid = np.linspace(0.02, 0.98, 97)
    scores = [fbeta_score(y_true, (y_score >= t).astype(int), beta=2) for t in grid]
    return float(grid[int(np.argmax(scores))])

"""Experiment suites for the traditional pipeline.

Each ablation suite varies one group of pipeline options while holding the rest at a
reference configuration, so differences can be attributed to that step. All scores are
5-fold stratified cross-validation on the training split; the test split is reserved for the
final comparison in ``evaluate_on_test``.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TunedThresholdClassifierCV, cross_validate

from . import config, data
from .evaluation import CV_SCORING, F2_SCORER, PRIMARY_METRIC, compute_metrics, cross_validate_pipeline, cv_splitter, scores_of
from .pipeline import MODEL_LABELS, PipelineConfig, build_pipeline

# Reference configurations: linear/distance models need scaling, trees do not. Balanced class
# weights are the reference because the default class is the minority (about 22%).
REF_LINEAR = PipelineConfig(model="logreg", class_weight="balanced")
REF_TREE = PipelineConfig(model="random_forest", scaler="none", class_weight="balanced")
REF_KNN = PipelineConfig(model="knn")


def run_suite(name: str, variants: Iterable[tuple[dict, PipelineConfig]], X, y, n_jobs=-1, verbose=True):
    """Cross-validate each (labels, config) pair and return one row per variant."""
    rows = []
    for labels, cfg in variants:
        res = cross_validate_pipeline(build_pipeline(cfg), X, y, n_jobs=n_jobs)
        rows.append({"suite": name, **labels, **res})
        if verbose:
            tag = ", ".join(f"{k}={v}" for k, v in labels.items())
            print(
                f"[{name}] {tag:55s} PR-AUC={res['pr_auc']:.4f}  F2={res['f2']:.4f}  "
                f"recall={res['recall']:.4f}  ({res['wall_time']:.1f}s)"
            )
    df = pd.DataFrame(rows)
    config.ensure_dirs()
    df.to_csv(config.RESULTS_DIR / f"suite_{name}.csv", index=False)
    return df


def imputation_variants():
    for ref in (REF_LINEAR, REF_TREE):
        for imp in ("mean", "median", "group_median", "knn", "iterative"):
            yield {"model": MODEL_LABELS[ref.model], "imputer": imp, "indicator": False}, ref.with_(imputer=imp)
        # Missingness itself might carry signal; test it once on the reference imputer.
        yield {"model": MODEL_LABELS[ref.model], "imputer": "median", "indicator": True}, ref.with_(add_indicator=True)


def scaling_variants():
    scalers = [
        ("none", "none", (0.0, 1.0)),
        ("standard", "standard", (0.0, 1.0)),
        ("minmax [0,1]", "minmax", (0.0, 1.0)),
        ("minmax [-1,1]", "minmax", (-1.0, 1.0)),
        ("robust", "robust", (0.0, 1.0)),
    ]
    for ref in (REF_LINEAR, REF_KNN, PipelineConfig(model="svm", class_weight="balanced")):
        for label, scaler, rng in scalers:
            if ref.model == "svm" and scaler == "none":
                # Unscaled RBF-SVM converges extremely slowly (minutes per fold) and is clearly
                # worse in development runs; skipped to keep "Run all" practical on Colab.
                continue
            yield {"model": MODEL_LABELS[ref.model], "scaler": label}, ref.with_(scaler=scaler, feature_range=rng)


def outlier_fe_variants():
    for ref in (REF_LINEAR, REF_TREE):
        for outlier in ("none", "clip_iqr", "log"):
            for fe in (False, True):
                yield (
                    {"model": MODEL_LABELS[ref.model], "outlier": outlier, "feat_eng": fe},
                    ref.with_(outlier=outlier, feature_engineering=fe),
                )


def encoding_variants():
    for ref in (REF_LINEAR, REF_KNN, REF_TREE):
        for enc in ("ordinal", "onehot"):
            yield {"model": MODEL_LABELS[ref.model], "grade_encoding": enc}, ref.with_(grade_encoding=enc)


def dimensionality_variants():
    for ref in (REF_LINEAR, REF_KNN, REF_TREE.with_(scaler="standard")):
        for var in (None, 0.80, 0.90, 0.95, 0.99):
            label = "none" if var is None else f"PCA {int(var * 100)}%"
            yield {"model": MODEL_LABELS[ref.model], "reduction": label}, ref.with_(pca_variance=var)
        for k in (5, 10, 15):
            yield {"model": MODEL_LABELS[ref.model], "reduction": f"SelectKBest k={k}"}, ref.with_(select_k=k)


def imbalance_variants():
    for ref in (REF_LINEAR, REF_TREE, PipelineConfig(model="hist_gb", scaler="none")):
        for strategy in ("none", "class_weight", "smote", "undersample"):
            cfg = ref.with_(
                class_weight="balanced" if strategy == "class_weight" else None,
                resampler=strategy if strategy in ("smote", "undersample") else "none",
            )
            yield {"model": MODEL_LABELS[ref.model], "imbalance": strategy}, cfg


SUITES = {
    "imputation": imputation_variants,
    "scaling": scaling_variants,
    "outlier_fe": outlier_fe_variants,
    "encoding": encoding_variants,
    "dimensionality": dimensionality_variants,
    "imbalance": imbalance_variants,
}


# Preprocessing for the tuned comparison follows the ablation findings. Scale-sensitive models
# get grade-aware imputation, engineered ratios, IQR clipping, one-hot grade and
# standardization; trees keep the ordinal grade and skip scaling. Neither family uses PCA.
PREP_SCALED = PipelineConfig(
    imputer="group_median", feature_engineering=True, outlier="clip_iqr", grade_encoding="onehot"
)
PREP_TREE = PipelineConfig(imputer="group_median", feature_engineering=True, scaler="none")
CW = {"model__class_weight": [None, "balanced"]}

PARAM_GRIDS: dict[str, tuple[PipelineConfig, dict]] = {
    "logreg": (PREP_SCALED.with_(model="logreg"), {"model__C": [0.01, 0.1, 1, 10], **CW}),
    "linear_svm": (PREP_SCALED.with_(model="linear_svm"), {"model__C": [0.01, 0.1, 1], **CW}),
    "svm": (PREP_SCALED.with_(model="svm", class_weight="balanced"), {"model__C": [1, 10], "model__gamma": ["scale", 0.1]}),
    "knn": (PREP_SCALED.with_(model="knn"), {"model__n_neighbors": [15, 31, 51], "model__weights": ["uniform", "distance"]}),
    "naive_bayes": (PREP_SCALED.with_(model="naive_bayes", outlier="log"), {"model__var_smoothing": [1e-9, 1e-6, 1e-3]}),
    "decision_tree": (
        PREP_TREE.with_(model="decision_tree"),
        {"model__max_depth": [6, 10, 16], "model__min_samples_leaf": [1, 20], **CW},
    ),
    "random_forest": (
        PREP_TREE.with_(model="random_forest"),
        {"model__max_features": ["sqrt", 0.5], "model__min_samples_leaf": [1, 3], **CW},
    ),
    "hist_gb": (
        PREP_TREE.with_(model="hist_gb"),
        {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [31, 63], "model__max_iter": [300], **CW},
    ),
}


def tune_models(X, y, models: Iterable[str] | None = None, n_jobs=-1, verbose=True):
    """Grid-search each model with 5-fold CV, refitting the candidate with the best PR-AUC."""
    rows, searches = [], {}
    for name in models or PARAM_GRIDS:
        cfg, grid = PARAM_GRIDS[name]
        t0 = time.perf_counter()
        gs = GridSearchCV(build_pipeline(cfg), grid, scoring=CV_SCORING, refit=PRIMARY_METRIC, cv=cv_splitter(), n_jobs=n_jobs)
        gs.fit(X, y)
        i, cvr = gs.best_index_, gs.cv_results_
        row = {"model": MODEL_LABELS[name], "key": name, "best_params": gs.best_params_}
        for metric in CV_SCORING:
            row[f"cv_{metric}"] = float(cvr[f"mean_test_{metric}"][i])
            row[f"cv_{metric}_std"] = float(cvr[f"std_test_{metric}"][i])
        row["n_candidates"] = len(cvr["params"])
        row["search_time"] = time.perf_counter() - t0
        rows.append(row)
        searches[name] = gs
        if verbose:
            print(f"[tune] {MODEL_LABELS[name]:24s} PR-AUC={row['cv_pr_auc']:.4f}  best={gs.best_params_}  ({row['search_time']:.0f}s)")
    df = pd.DataFrame(rows)
    config.ensure_dirs()
    df.assign(best_params=df["best_params"].astype(str)).to_csv(config.RESULTS_DIR / "tuning.csv", index=False)
    return df, searches


def tune_threshold(estimator, X, y, n_jobs=-1) -> TunedThresholdClassifierCV:
    """Wrap a tuned pipeline with an F2-optimal decision threshold chosen by internal CV."""
    tuned = TunedThresholdClassifierCV(estimator, scoring=F2_SCORER, cv=cv_splitter(), n_jobs=n_jobs)
    return tuned.fit(X, y)


def evaluate_on_test(models: dict, X_test, y_test, thresholded: dict | None = None) -> pd.DataFrame:
    """Test metrics at the default threshold and, when given, at the F2-tuned threshold."""
    rows = []
    for name, est in models.items():
        t0 = time.perf_counter()
        score = scores_of(est, X_test)
        pred_time = time.perf_counter() - t0
        row = {"model": MODEL_LABELS.get(name, name), "key": name, "predict_time": pred_time}
        row.update(compute_metrics(y_test, est.predict(X_test), score))
        if thresholded and name in thresholded:
            tt = thresholded[name]
            tuned = compute_metrics(y_test, tt.predict(X_test))
            row["threshold"] = float(tt.best_threshold_)
            row.update({f"{k}_tuned": v for k, v in tuned.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def importance_frame(estimator, X, y, n_repeats=5, n_jobs=-1) -> pd.DataFrame:
    """Permutation importance on the raw input columns (drop in PR-AUC when a column is shuffled)."""
    r = permutation_importance(
        estimator, X, y, scoring="average_precision", n_repeats=n_repeats, random_state=config.SEED, n_jobs=n_jobs
    )
    return (
        pd.DataFrame({"feature": X.columns, "importance": r.importances_mean, "std": r.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- impossible values
def _impossible_to_nan(X, ref=None):
    X = X.copy()
    bad_age, bad_emp = data.impossible_masks(X)
    X.loc[bad_age, "person_age"] = np.nan
    X.loc[bad_emp, "person_emp_length"] = np.nan
    return X


def _impossible_to_stat(stat):
    """Replace impossible values by a statistic of the valid training values."""

    def apply(X, ref):
        X = X.copy()
        bad_age, bad_emp = data.impossible_masks(X)
        ok_age, ok_emp = data.impossible_masks(ref)
        X.loc[bad_age, "person_age"] = getattr(ref.loc[~ok_age, "person_age"], stat)()
        X.loc[bad_emp, "person_emp_length"] = getattr(ref.loc[~ok_emp, "person_emp_length"], stat)()
        return X

    return apply


def _impossible_to_cap(X, ref=None):
    X = X.copy()
    bad_age, bad_emp = data.impossible_masks(X)
    X.loc[bad_age, "person_age"] = 100.0
    X.loc[bad_emp, "person_emp_length"] = (X.loc[bad_emp, "person_age"] - 14).clip(lower=0)
    return X


# Each strategy is (treatment of the training fold, treatment of the validation fold). A deployed
# model cannot refuse to score an applicant, so "drop" still treats impossible validation rows
# as missing values.
IMPOSSIBLE_STRATEGIES = {
    "keep as is": (lambda X, ref: X, lambda X, ref: X),
    "drop rows": ("drop", _impossible_to_nan),
    "replace by mean": (_impossible_to_stat("mean"), _impossible_to_stat("mean")),
    "replace by median": (_impossible_to_stat("median"), _impossible_to_stat("median")),
    "cap to plausible value": (_impossible_to_cap, _impossible_to_cap),
    "missing, then pipeline imputer": (_impossible_to_nan, _impossible_to_nan),
}


def run_impossible_values(estimators: dict, verbose=True) -> pd.DataFrame:
    """CV PR-AUC of each way of handling impossible ages / employment lengths.

    Uses the deduplicated but otherwise raw data with the project's train/test split, so the
    validation rows are identical across strategies and only the treatment differs.
    """
    raw = data.load_raw().drop_duplicates().reset_index(drop=True)
    raw[config.NUMERIC_FEATURES] = raw[config.NUMERIC_FEATURES].astype(float)
    X, _, y, _ = data.split(raw)
    rows = []
    for strategy, (treat_fit, treat_val) in IMPOSSIBLE_STRATEGIES.items():
        for label, est in estimators.items():
            scores = []
            for tr, va in cv_splitter().split(X, y):
                X_fit, y_fit, X_val = X.iloc[tr], y.iloc[tr], X.iloc[va]
                if treat_fit == "drop":
                    bad_age, bad_emp = data.impossible_masks(X_fit)
                    keep = ~(bad_age | bad_emp)
                    X_fit2, y_fit2 = X_fit[keep], y_fit[keep]
                else:
                    X_fit2, y_fit2 = treat_fit(X_fit, X_fit), y_fit
                proba = clone(est).fit(X_fit2, y_fit2).predict_proba(treat_val(X_val, X_fit))[:, 1]
                scores.append(average_precision_score(y.iloc[va], proba))
            rows.append({"strategy": strategy, "model": label, "pr_auc": float(np.mean(scores)), "pr_auc_std": float(np.std(scores))})
            if verbose:
                print(f"[impossible] {strategy:32s} {label:24s} PR-AUC={np.mean(scores):.5f}")
    out = pd.DataFrame(rows)
    config.ensure_dirs()
    out.to_csv(config.RESULTS_DIR / "impossible_values.csv", index=False)
    return out


# ------------------------------------------------------------------------------------ headroom
HGB_WIDE_SPACE = {
    "model__learning_rate": loguniform(0.01, 0.3),
    "model__max_leaf_nodes": randint(15, 128),
    "model__min_samples_leaf": randint(5, 100),
    "model__l2_regularization": loguniform(1e-3, 10),
    "model__max_iter": randint(200, 1000),
    "model__max_features": uniform(0.5, 0.5),
    "model__class_weight": [None, "balanced"],
}


def run_headroom(X, y, searches: dict, n_iter=40, n_jobs=-1, verbose=True):
    """Does more optimization help? A wide random search for gradient boosting and a stacked
    ensemble of the three strongest tuned model families, compared with the grid-search result."""
    rows = [{"approach": "Hist. Gradient Boosting, grid search (8 candidates)",
             "cv_pr_auc": float(searches["hist_gb"].best_score_),
             "cv_pr_auc_std": float(searches["hist_gb"].cv_results_["std_test_pr_auc"][searches["hist_gb"].best_index_])}]
    t0 = time.perf_counter()
    rs = RandomizedSearchCV(
        build_pipeline(PARAM_GRIDS["hist_gb"][0]), HGB_WIDE_SPACE, n_iter=n_iter, scoring="average_precision",
        cv=cv_splitter(), n_jobs=n_jobs, random_state=config.SEED,
    ).fit(X, y)
    rows.append({"approach": f"Hist. Gradient Boosting, random search ({n_iter} candidates, 7 hyper-parameters)",
                 "cv_pr_auc": float(rs.best_score_),
                 "cv_pr_auc_std": float(rs.cv_results_["std_test_score"][rs.best_index_]),
                 "time": time.perf_counter() - t0})
    stack = StackingClassifier(
        [(k, searches[k].best_estimator_) for k in ("hist_gb", "random_forest", "logreg")],
        final_estimator=LogisticRegression(max_iter=1000), cv=5,
    )
    t0 = time.perf_counter()
    res = cross_validate(stack, X, y, cv=cv_splitter(), scoring="average_precision", n_jobs=n_jobs)
    rows.append({"approach": "Stacking (Hist. GB + Random Forest + Logistic Regression)",
                 "cv_pr_auc": float(res["test_score"].mean()), "cv_pr_auc_std": float(res["test_score"].std()),
                 "time": time.perf_counter() - t0})
    out = pd.DataFrame(rows)
    out["gain_vs_grid"] = out["cv_pr_auc"] - out["cv_pr_auc"].iloc[0]
    config.ensure_dirs()
    out.to_csv(config.RESULTS_DIR / "headroom.csv", index=False)
    if verbose:
        print(out.round(4).to_string(index=False))
    return out, rs

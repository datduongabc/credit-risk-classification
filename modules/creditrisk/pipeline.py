"""Configurable traditional ML pipeline.

Every step (imputation, outlier handling, encoding, scaling, feature selection, PCA,
class-imbalance handling and the classifier) is chosen through a ``PipelineConfig`` so that
experiments are plain data and the same builder serves the notebook, the tests and the report.
All statistics are learned inside the pipeline, so cross-validation never leaks fold data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from functools import partial
from typing import Any

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers IterativeImputer)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from . import config
from .transformers import (
    ENGINEERED_FEATURES,
    GroupMedianImputer,
    IQRClipper,
    feature_engineer,
    log_transform,
)

IMPUTERS = ("mean", "median", "group_median", "knn", "iterative")
OUTLIERS = ("none", "clip_iqr", "log")
SCALERS = ("none", "standard", "minmax", "robust")
GRADE_ENCODINGS = ("ordinal", "onehot")
RESAMPLERS = ("none", "smote", "undersample")
MODELS = (
    "logreg",
    "linear_svm",
    "svm",
    "knn",
    "naive_bayes",
    "decision_tree",
    "random_forest",
    "hist_gb",
)
MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "svm": "SVM (RBF)",
    "knn": "k-NN",
    "naive_bayes": "Gaussian Naive Bayes",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "hist_gb": "Hist. Gradient Boosting",
}


@dataclass(frozen=True)
class PipelineConfig:
    imputer: str = "median"
    add_indicator: bool = False
    feature_engineering: bool = False
    outlier: str = "none"
    grade_encoding: str = "ordinal"
    scaler: str = "standard"
    feature_range: tuple[float, float] = (0.0, 1.0)
    select_k: int | None = None
    pca_variance: float | None = None
    resampler: str = "none"
    class_weight: str | None = None
    model: str = "logreg"
    model_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        checks = [
            ("imputer", self.imputer, IMPUTERS),
            ("outlier", self.outlier, OUTLIERS),
            ("scaler", self.scaler, SCALERS),
            ("grade_encoding", self.grade_encoding, GRADE_ENCODINGS),
            ("resampler", self.resampler, RESAMPLERS),
            ("model", self.model, MODELS),
        ]
        for name, value, allowed in checks:
            if value not in allowed:
                raise ValueError(f"{name}={value!r} is not one of {allowed}")
        if self.pca_variance is not None and not 0 < self.pca_variance < 1:
            raise ValueError("pca_variance must be in (0, 1)")

    def with_(self, **changes) -> "PipelineConfig":
        return replace(self, **changes)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["feature_range"] = list(self.feature_range)
        return d


def _numeric_imputer(cfg: PipelineConfig):
    if cfg.imputer in ("mean", "median"):
        return SimpleImputer(strategy=cfg.imputer, add_indicator=cfg.add_indicator)
    if cfg.imputer == "group_median":
        # loan_int_rate is already filled by GroupMedianImputer; the rest falls back to median.
        return SimpleImputer(strategy="median", add_indicator=cfg.add_indicator)
    if cfg.imputer == "knn":
        # KNN distances need comparable scales, so neighbours are found in standardized space.
        return SkPipeline(
            [
                ("std", StandardScaler()),
                ("knn", KNNImputer(n_neighbors=5, add_indicator=cfg.add_indicator)),
            ]
        )
    return IterativeImputer(
        max_iter=10, random_state=config.SEED, add_indicator=cfg.add_indicator
    )


def _scaler(cfg: PipelineConfig):
    return {
        "none": "passthrough",
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(feature_range=tuple(cfg.feature_range)),
        "robust": RobustScaler(),
    }[cfg.scaler]


def _outlier(cfg: PipelineConfig):
    if cfg.outlier == "clip_iqr":
        return IQRClipper(k=1.5)
    return log_transform() if cfg.outlier == "log" else "passthrough"


def build_preprocessor(cfg: PipelineConfig, numeric_features: list[str]) -> ColumnTransformer:
    numeric = SkPipeline(
        [
            # Outlier handling first: both transforms are NaN-tolerant, and the KNN imputer
            # below returns standardized values that a log transform could not accept.
            ("outlier", _outlier(cfg)),
            ("impute", _numeric_imputer(cfg)),
            ("scale", _scaler(cfg)),
        ]
    )
    nominal = SkPipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    if cfg.grade_encoding == "ordinal":
        grade = SkPipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(categories=[config.GRADE_ORDER])),
                # Put the 0..6 grade code on the same footing as the scaled numerics.
                ("scale", _scaler(cfg)),
            ]
        )
    else:
        grade = OneHotEncoder(
            categories=[config.GRADE_ORDER], handle_unknown="ignore", sparse_output=False
        )
    return ColumnTransformer(
        [
            ("num", numeric, numeric_features),
            ("nom", nominal, config.NOMINAL_FEATURES),
            ("grade", grade, config.ORDINAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def build_model(cfg: PipelineConfig):
    p = dict(cfg.model_params)
    cw = cfg.class_weight
    seed = config.SEED
    if cfg.model == "logreg":
        return LogisticRegression(max_iter=2000, class_weight=cw, **p)
    if cfg.model == "linear_svm":
        return LinearSVC(class_weight=cw, random_state=seed, max_iter=5000, **p)
    if cfg.model == "svm":
        return SVC(kernel="rbf", class_weight=cw, cache_size=1000, random_state=seed, **p)
    if cfg.model == "knn":
        return KNeighborsClassifier(**{"n_neighbors": 15, **p})
    if cfg.model == "naive_bayes":
        return GaussianNB(**p)
    if cfg.model == "decision_tree":
        return DecisionTreeClassifier(class_weight=cw, random_state=seed, **p)
    if cfg.model == "random_forest":
        return RandomForestClassifier(
            **{"n_estimators": 300, "n_jobs": -1, "random_state": seed, "class_weight": cw, **p}
        )
    return HistGradientBoostingClassifier(random_state=seed, class_weight=cw, **p)


def build_pipeline(cfg: PipelineConfig) -> Pipeline:
    numeric = list(config.NUMERIC_FEATURES)
    steps: list[tuple[str, Any]] = []
    if cfg.imputer == "group_median":
        steps.append(("group_impute", GroupMedianImputer()))
    if cfg.feature_engineering:
        steps.append(("engineer", feature_engineer()))
        numeric += ENGINEERED_FEATURES
    steps.append(("preprocess", build_preprocessor(cfg, numeric)))
    if cfg.select_k is not None:
        steps.append(
            (
                "select",
                SelectKBest(partial(mutual_info_classif, random_state=config.SEED), k=cfg.select_k),
            )
        )
    if cfg.pca_variance is not None:
        pca = PCA(n_components=cfg.pca_variance, svd_solver="full", random_state=config.SEED)
        steps.append(("pca", pca))
    if cfg.resampler == "smote":
        steps.append(("resample", SMOTE(random_state=config.SEED)))
    elif cfg.resampler == "undersample":
        steps.append(("resample", RandomUnderSampler(random_state=config.SEED)))
    steps.append(("model", build_model(cfg)))
    return Pipeline(steps)


def transform_only(pipe: Pipeline, X):
    """Run the fitted feature steps (everything except resampling and the model)."""
    Xt = X
    for name, step in pipe.steps[:-1]:
        if name == "resample":
            continue
        Xt = step.transform(Xt)
    return Xt


def feature_names(pipe: Pipeline) -> list[str]:
    """Names of the columns that reach the model (PCA components are named pc1, pc2, ...)."""
    names = None
    for name, step in pipe.steps[:-1]:
        if name == "resample":
            continue
        if name == "pca":
            names = [f"pc{i + 1}" for i in range(step.n_components_)]
        else:
            names = list(step.get_feature_names_out(names))
    return names


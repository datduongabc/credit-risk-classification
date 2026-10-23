import numpy as np
import pandas as pd
import pytest

from creditrisk import config
from creditrisk.data import clean
from creditrisk.pipeline import PipelineConfig, build_pipeline, feature_names
from creditrisk.transformers import GroupMedianImputer, IQRClipper


def synthetic_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    grade = rng.choice(config.GRADE_ORDER, n)
    df = pd.DataFrame(
        {
            "person_age": rng.integers(20, 70, n).astype(float),
            "person_income": rng.lognormal(11, 0.5, n),
            "person_emp_length": rng.integers(0, 10, n).astype(float),
            "loan_amnt": rng.integers(500, 35000, n).astype(float),
            "loan_int_rate": 5 + 2 * pd.Series(grade).map({g: i for i, g in enumerate(config.GRADE_ORDER)}).to_numpy() + rng.random(n),
            "loan_percent_income": rng.random(n) * 0.6,
            "cb_person_cred_hist_length": rng.integers(2, 20, n).astype(float),
            "person_home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n),
            "loan_intent": rng.choice(["EDUCATION", "MEDICAL", "VENTURE"], n),
            "cb_person_default_on_file": rng.choice(["Y", "N"], n),
            "loan_grade": grade,
        }
    )
    df.loc[rng.random(n) < 0.1, "loan_int_rate"] = np.nan
    df.loc[rng.random(n) < 0.05, "person_emp_length"] = np.nan
    y = pd.Series((df["loan_percent_income"] + rng.random(n) * 0.3 > 0.5).astype(int))
    return df, y


def test_clean_drops_duplicates_and_nulls_impossible_values():
    df, y = synthetic_frame(10)
    df[config.TARGET] = y
    df.loc[0, "person_age"] = 144
    df.loc[1, "person_emp_length"] = 123
    df = pd.concat([df, df.iloc[[2]]], ignore_index=True)
    cleaned, report = clean(df)
    assert report.n_duplicates == 1
    assert np.isnan(cleaned.loc[0, "person_age"]) and np.isnan(cleaned.loc[1, "person_emp_length"])


def test_group_median_imputer_uses_grade_medians():
    X = pd.DataFrame({"loan_grade": ["A", "A", "B", "B", "B"], "loan_int_rate": [5.0, np.nan, 10.0, 12.0, np.nan]})
    out = GroupMedianImputer().fit(X).transform(X)
    assert out["loan_int_rate"].tolist() == [5.0, 5.0, 10.0, 12.0, 11.0]


def test_iqr_clipper_learns_bounds_on_fit_data():
    clip = IQRClipper(k=1.5).fit(np.array([[1.0], [2.0], [3.0], [4.0]]))
    assert clip.transform(np.array([[100.0]]))[0, 0] == pytest.approx(4 + 1.5 * 1.5 - 0.75)


@pytest.mark.parametrize(
    "cfg",
    [
        PipelineConfig(),
        PipelineConfig(imputer="knn", outlier="log", scaler="minmax", feature_range=(-1, 1)),
        PipelineConfig(imputer="group_median", feature_engineering=True, grade_encoding="onehot", pca_variance=0.9),
        PipelineConfig(imputer="iterative", add_indicator=True, select_k=5, resampler="smote", model="random_forest"),
        PipelineConfig(outlier="clip_iqr", resampler="undersample", model="svm"),
        PipelineConfig(model="hist_gb", scaler="none", class_weight="balanced"),
    ],
)
def test_every_configuration_fits_and_predicts(cfg):
    X, y = synthetic_frame()
    pipe = build_pipeline(cfg).fit(X, y)
    assert pipe.predict(X).shape == (len(X),)
    assert len(feature_names(pipe)) > 0


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        PipelineConfig(scaler="zscore")
    with pytest.raises(ValueError):
        PipelineConfig(pca_variance=1.5)

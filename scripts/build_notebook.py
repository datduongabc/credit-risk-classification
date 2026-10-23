"""Generate notebooks/credit_risk_pipeline.ipynb from the cell list below.

Keeping the notebook as code makes it reviewable in diffs; regenerate after editing:
    python scripts/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf

REPO_URL = "https://github.com/datduongabc/credit-risk-classification.git"
REPO_DIR = "credit-risk-classification"

cells: list[tuple[str, str]] = []


def md(text: str) -> None:
    cells.append(("md", text.strip("\n")))


def code(text: str) -> None:
    cells.append(("code", text.strip("\n")))


md(
    f"""
# Credit Risk Prediction - Machine Learning Assignment (CO3117)

**Ho Chi Minh City University of Technology (HCMUT) - Faculty of Computer Science and Engineering**
Semester 1, academic year 2025-2026 - Advisor: Dr. Le Thanh Sach - Student: Duong Nguyen Tan Dat (2352222)

Task: predict whether a borrower will **default** on a loan (`loan_status = 1`) from the public
[Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (32,581 loans, 11 features).

| Part | Content |
|---|---|
| 1 | Setup: clone the code, install extra libraries, download the data |
| 2 | Exploratory data analysis (EDA) |
| 3 | Configurable traditional ML pipeline and ablation experiments |
| 4 | Hyper-parameter tuning, threshold tuning and test-set comparison of 8 classifiers |
| 5 | Saving extracted features (`.npy` / `.h5`) |
| 6 | Deep learning pipeline (entity-embedding MLP, TabNet) and comparison |
| 7 | Talented-engineering extension: Bayesian network with exact and approximate inference |
| 8 | Overall comparison and conclusions |

Source code: {REPO_URL.removesuffix('.git')}

**Runtime:** about 30 minutes on a standard Colab CPU runtime. *Runtime -> Run all* runs everything with no
manual steps. No personal cloud storage is mounted. The data comes from a public URL.
"""
)

md("## 1. Setup")
code(
    f"""
import os, subprocess, sys
from pathlib import Path

REPO_URL = "{REPO_URL}"
REPO_DIR = "{REPO_DIR}"

# Locate the project root: the repository itself when run locally, otherwise a fresh clone (Colab).
if Path("modules/creditrisk").is_dir():
    pass
elif Path("../modules/creditrisk").is_dir():
    os.chdir("..")
else:
    if not Path(REPO_DIR).is_dir():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
    os.chdir(REPO_DIR)

ROOT = Path.cwd()
os.environ["CREDITRISK_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT / "modules"))
print("Project root:", ROOT)
"""
)
code(
    """
# Libraries that Colab does not ship by default (no-op when already installed).
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")], check=True)
"""
)
code(
    """
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from IPython.display import display

from creditrisk import config, data, eda, evaluation, experiments, pipeline, reporting, deep
from creditrisk.bayesnet import experiments as bnx

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)
config.ensure_dirs()
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
import sklearn
# Parallel CV workers each hold distance matrices (k-NN) and forests in memory; capping the
# workers and sklearn's chunk size keeps peak RAM bounded on machines with many cores.
CORES = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
N_JOBS = min(4, CORES)
sklearn.set_config(working_memory=256)
print(f"{CORES} CPU cores available, using {N_JOBS} parallel workers")
T_START = time.perf_counter()
"""
)
code(
    """
csv_path = data.download_dataset()  # public Kaggle endpoint, see config.DATASET_URL
print("Downloaded from:", config.DATASET_URL)
raw = data.load_raw(csv_path)
print(raw.shape)
raw.head()
"""
)

# ----------------------------------------------------------------------------------- EDA
md(
    """
## 2. Exploratory data analysis

A generic, automatic overview of every column (histograms, missing values, correlations, alerts) is
available in [`reports/eda_profile.html`](../reports/eda_profile.html), generated with
[ydata-profiling](https://github.com/ydataai/ydata-profiling) by `scripts/make_profile.py`. It is generated
outside the notebook because that library pins older numpy/pandas versions. Installing it inside Colab
could force a runtime restart and break *Run all*.

The sections below focus on the questions the generic profile does not answer: how each feature relates
to **default**, and what the findings imply for the pipeline.

### 2.1 Overview of the columns
"""
)
code(
    """
ov = eda.overview(raw)
display(ov)
print("Exact duplicate rows:", raw.duplicated().sum())
"""
)
code(
    """
num_summary = eda.numeric_summary(raw)
display(num_summary.round(2))
"""
)
md(
    """
`person_age` reaches 144 and `person_emp_length` reaches 123 years, which are impossible values.
`person_income` is extremely right-skewed (skew > 30). Two columns have missing values:
`person_emp_length` (2.7%) and `loan_int_rate` (9.6%).
"""
)
md("### 2.2 Target distribution")
code("eda.plot_target(raw); plt.show()")
md(
    """
About 21.8% of the loans default. The classes are imbalanced, so accuracy is misleading: always predicting
"non-default" already scores 78%. We therefore rank models by **PR-AUC** and tune the decision
threshold for **F2**, which weights recall on the default class twice as much as precision. Missing a
defaulter costs a lender much more than turning down a good applicant.
"""
)
md("### 2.3 Numeric features")
code(
    """
# Histograms use the rule-cleaned data (section 3.1) so that the five impossible ages and two impossible
# employment lengths do not stretch the axes; the box plots below show the raw values.
eda.plot_numeric_distributions(data.clean(raw)[0]); plt.show()
"""
)
code("eda.plot_outliers(raw); plt.show()")
md("### 2.4 Categorical features and default rate per category")
code("eda.plot_categorical(raw); plt.show()")
md("### 2.5 Correlations")
code(
    """
fig, corr = eda.plot_correlations(raw); plt.show()
corr[config.TARGET].drop(config.TARGET).sort_values(key=abs, ascending=False).round(3)
"""
)
md("### 2.6 Missing values")
code(
    """
fig, miss_rates = eda.plot_missingness(raw); plt.show()
display(miss_rates.round(3))
eda.plot_rate_by_grade(raw); plt.show()
"""
)
md(
    """
The two columns behave differently:
* A missing `loan_int_rate` says nothing about the outcome. Its default rate is about the same as for observed rates,
  and its missing share is flat across grades. It looks *missing completely at random*. The rate itself is almost a
  function of `loan_grade`, which motivates the **grade-wise median imputer**.
* A missing `person_emp_length` is informative: those applicants default about 50% more often than the rest. The
  missingness carries signal, which motivates the optional **missing-value indicator** columns tested in section 3.3.
"""
)
md("### 2.7 Relevance of individual features")
code(
    """
mi = eda.mutual_information(raw)
eda.plot_mutual_information(mi); plt.show()
eda.plot_burden_scatter(raw); plt.show()
mi.round(4)
"""
)

# ------------------------------------------------------------------------- preprocessing
md(
    """
## 3. Traditional machine learning pipeline

### 3.1 Cleaning and train/test split
Rule-based cleaning (no learned statistics, so it cannot leak test information) removes exact duplicates and
turns impossible values (age > 100, employment longer than age minus 14) into missing values. The
imputer then fills them. A stratified 80/20 split holds out a test set that is used **only** for the final
comparisons.
"""
)
code(
    """
X_train, X_test, y_train, y_test, cleaning = data.load_splits()
print(cleaning.as_dict())
print("train:", X_train.shape, "default rate %.4f" % y_train.mean())
print("test: ", X_test.shape, "default rate %.4f" % y_test.mean())
"""
)
md(
    """
#### Is "impossible value -> missing" the right choice?
There are several ways to handle an age of 144: keep it, drop the row, replace it by the mean or median, cap it
at a plausible value, or mark it missing and let the pipeline's imputer fill it. The comparison below scores
every option with 5-fold CV on the same validation rows, for a tree ensemble and a linear model.
"""
)
code(
    """
impossible = experiments.run_impossible_values({
    "Hist. Gradient Boosting": pipeline.build_pipeline(experiments.PREP_TREE.with_(model="hist_gb")),
    "Logistic Regression": pipeline.build_pipeline(experiments.PREP_SCALED.with_(model="logreg")),
}, verbose=False)
display(impossible.pivot(index="strategy", columns="model", values="pr_auc").round(5))
print("typical fold standard deviation: %.4f" % impossible["pr_auc_std"].mean())
"""
)
md(
    """
Only 7 training rows (and no test row) are affected, so every strategy scores within a fraction of the fold
standard deviation of the others. Performance cannot separate them, so the choice is made on methodology.
"Missing, then imputer" keeps the row's ten valid columns. It learns the fill value inside each training fold,
which avoids leakage. It can use any configured imputer (MICE can infer age from credit-history length). And
it is the only option that also works for a new applicant at prediction time, where a row cannot be dropped.

### 3.2 A configurable pipeline
Every step is chosen through a `PipelineConfig`:

| Step | Options |
|---|---|
| Missing values | mean, median, grade-wise median, KNN, iterative (MICE); optional missing indicators |
| Feature engineering | optional log income, log amount, income per credit-history year, rate x burden |
| Outliers | none, IQR clipping (winsorizing), log transform |
| Encoding | one-hot for nominal features; loan grade ordinal or one-hot |
| Scaling | none, StandardScaler, MinMaxScaler(`feature_range`), RobustScaler |
| Feature selection / reduction | SelectKBest (mutual information), PCA with a retained-variance level |
| Class imbalance | none, class weights, SMOTE, random under-sampling |
| Model | Logistic Regression, Linear SVM, RBF SVM, k-NN, Gaussian NB, Decision Tree, Random Forest, Hist. Gradient Boosting |

All learned steps sit inside one `imblearn.Pipeline`, so cross-validation refits them on each training fold.
"""
)
code(
    """
example = pipeline.PipelineConfig(imputer="group_median", feature_engineering=True, outlier="clip_iqr",
                                  grade_encoding="onehot", scaler="minmax", feature_range=(-1, 1),
                                  pca_variance=0.95, resampler="smote", model="svm")
example_pipe = pipeline.build_pipeline(example)
display(example_pipe)
"""
)
md(
    """
### 3.3 Ablation experiments (5-fold stratified CV on the training set)
Each suite changes one step and keeps the rest at a reference configuration.
"""
)
code(
    """
suites = {}
for name, variants in experiments.SUITES.items():
    t0 = time.perf_counter()
    suites[name] = experiments.run_suite(name, variants(), X_train, y_train, n_jobs=N_JOBS, verbose=False)
    print(f"{name:15s} {len(suites[name]):2d} configurations in {time.perf_counter() - t0:.0f}s")
"""
)
code(
    """
COLS = ["pr_auc", "pr_auc_std", "roc_auc", "f2", "recall", "precision"]
def show(name, keys):
    df = suites[name][keys + COLS]
    display(df.round(4))
    return df

show("imputation", ["model", "imputer", "indicator"])
reporting.plot_suite(suites["imputation"].assign(variant=lambda d: d.imputer + np.where(d.indicator, " +ind", "")),
                     x="variant", title="Imputation strategy (5-fold CV)", name="suite_imputation"); plt.show()
"""
)
code(
    """
show("scaling", ["model", "scaler"])
reporting.plot_suite(suites["scaling"], x="scaler", title="Feature scaling (5-fold CV)", name="suite_scaling"); plt.show()
"""
)
code(
    """
show("outlier_fe", ["model", "outlier", "feat_eng"])
reporting.plot_suite(suites["outlier_fe"].assign(variant=lambda d: d.outlier + np.where(d.feat_eng, " +FE", "")),
                     x="variant", title="Outlier handling and feature engineering (5-fold CV)", name="suite_outlier_fe"); plt.show()
"""
)
code(
    """
show("encoding", ["model", "grade_encoding"])
reporting.plot_suite(suites["encoding"], x="grade_encoding", title="Loan grade encoding (5-fold CV)", name="suite_encoding"); plt.show()
"""
)
code(
    """
show("dimensionality", ["model", "reduction"])
reporting.plot_suite(suites["dimensionality"], x="reduction", title="PCA and feature selection (5-fold CV)", name="suite_dimensionality"); plt.show()
"""
)
code(
    """
show("imbalance", ["model", "imbalance"])
reporting.plot_suite(suites["imbalance"], x="imbalance", metric="f2", title="Class-imbalance handling, F2 at threshold 0.5 (5-fold CV)", name="suite_imbalance"); plt.show()
"""
)
md(
    """
**Take-aways used for the final configurations**:
- Scaling is essential for distance- and margin-based models (k-NN, SVM).
- The engineered ratio features, IQR clipping and a one-hot loan grade help the linear models.
- The imputation method barely matters at these missing-value levels.
- PCA never helps: it is neutral at 99% retained variance and harmful below, because the ~20 encoded features are not redundant.
- Class weights raise F2 at the default threshold without hurting PR-AUC.

## 4. Model comparison
### 4.1 Hyper-parameter search (5-fold CV, refit on PR-AUC)
"""
)
code(
    """
tuning, searches = experiments.tune_models(X_train, y_train, n_jobs=N_JOBS)
display(tuning[["model", "best_params", "cv_pr_auc", "cv_pr_auc_std", "cv_roc_auc", "cv_f2", "cv_recall", "search_time"]].round(4))
"""
)
md(
    """
### 4.2 Decision threshold tuned for F2
`TunedThresholdClassifierCV` picks the threshold that maximizes F2 with internal cross-validation on the
training set. The test set never influences it.
"""
)
code(
    """
best = {k: gs.best_estimator_ for k, gs in searches.items()}
thresholded = {k: experiments.tune_threshold(est, X_train, y_train, n_jobs=N_JOBS) for k, est in best.items()}
test_results = experiments.evaluate_on_test(best, X_test, y_test, thresholded)
test_results.to_csv(config.RESULTS_DIR / "test_traditional.csv", index=False)
display(test_results[["model", "pr_auc", "roc_auc", "accuracy", "f1", "threshold", "f2_tuned", "recall_tuned", "precision_tuned"]]
        .sort_values("pr_auc", ascending=False).round(4))
"""
)
code(
    """
curves = {pipeline.MODEL_LABELS[k]: (y_test, evaluation.scores_of(m, X_test)) for k, m in best.items()}
evaluation.plot_roc_pr(curves, path=config.FIGURES_DIR / "roc_pr_traditional.pdf", title_suffix=" (test set)"); plt.show()
preds = {pipeline.MODEL_LABELS[k]: (y_test, t.predict(X_test)) for k, t in thresholded.items()}
evaluation.plot_confusion_matrices(preds, path=config.FIGURES_DIR / "confusion_traditional.pdf", ncols=4); plt.show()
"""
)
md(
    """
### 4.3 Is there headroom left?
Several grid optima sit on the edge of their grids, which can mean the search space was too narrow. Two
stronger optimizers test this for the best model family: a 40-candidate random search over seven
gradient-boosting hyper-parameters, and a stacked ensemble of the three strongest tuned families.
"""
)
code(
    """
headroom, wide_search = experiments.run_headroom(X_train, y_train, searches, n_jobs=N_JOBS, verbose=False)
display(headroom.round(4))
print("best random-search parameters:", {k.replace("model__", ""): (round(v, 4) if isinstance(v, float) else v)
                                         for k, v in wide_search.best_params_.items()})
"""
)
md(
    """
Neither the wider search nor stacking improves PR-AUC by more than the fold-to-fold standard deviation. The
tuned gradient-boosting model sits on a plateau, so the remaining error comes from the overlap of the classes in
feature space (section 2), not from under-optimization. The grid-search models are kept.
"""
)
md("### 4.4 What drives the predictions? (permutation importance of the best model on the test set)")
code(
    """
best_key = test_results.sort_values("pr_auc", ascending=False).iloc[0]["key"]
imp = experiments.importance_frame(best[best_key], X_test, y_test, n_jobs=N_JOBS)
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(imp["feature"][::-1], imp["importance"][::-1], xerr=imp["std"][::-1], color="#4c72b0")
ax.set_xlabel("Drop in PR-AUC when the column is shuffled")
ax.set_title(f"Permutation importance - {pipeline.MODEL_LABELS[best_key]}")
fig.tight_layout(); reporting.save_figure(fig, "importance_best"); plt.show()
imp.round(4)
"""
)
md(
    """
### 4.5 Error analysis of the best model
Which borrowers does the best model miss (false negatives) at its F2 threshold?
"""
)
code(
    """
pred_best = thresholded[best_key].predict(X_test)
err = X_test.assign(y=y_test.values, pred=pred_best)
err["outcome"] = np.select([(err.y == 1) & (err.pred == 1), (err.y == 1) & (err.pred == 0), (err.y == 0) & (err.pred == 1)],
                           ["TP", "FN", "FP"], "TN")
error_profile = err.groupby("outcome")[["loan_percent_income", "loan_int_rate", "person_income", "loan_amnt"]].median()
error_profile["count"] = err["outcome"].value_counts()
error_profile["share_grade_A_B"] = err.groupby("outcome")["loan_grade"].apply(lambda s: s.isin(["A", "B"]).mean())
error_profile.round(3)
"""
)

# ------------------------------------------------------------------------------ features
md(
    """
## 5. Saving the extracted features
The feature matrices produced by the fitted preprocessing steps are saved under `features/` as `.npy`
files and collected in `features/credit_features.h5`. They can be reused without re-running the pipeline.
"""
)
code(
    """
from creditrisk.pipeline import feature_names, transform_only

saved = {}
for key in ("logreg", best_key):
    pipe = best[key]
    saved[key] = reporting.save_features(
        f"traditional_{key}",
        {"X_train": np.asarray(transform_only(pipe, X_train), dtype=np.float32),
         "X_test": np.asarray(transform_only(pipe, X_test), dtype=np.float32),
         "y_train": y_train.to_numpy(np.int8), "y_test": y_test.to_numpy(np.int8)},
        meta={"feature_names": feature_names(pipe), "config": experiments.PARAM_GRIDS[key][0].as_dict()},
    )
pca_pipe = pipeline.build_pipeline(experiments.PREP_SCALED.with_(pca_variance=0.95)).fit(X_train, y_train)
reporting.save_features("traditional_pca95", {
    "X_train": np.asarray(transform_only(pca_pipe, X_train), dtype=np.float32),
    "X_test": np.asarray(transform_only(pca_pipe, X_test), dtype=np.float32)},
    meta={"explained_variance_ratio": pca_pipe.named_steps["pca"].explained_variance_ratio_.round(5).tolist()})
for p in sorted(config.FEATURES_DIR.rglob("*.npy")):
    print(p.relative_to(ROOT), np.load(p).shape)
"""
)

# ---------------------------------------------------------------------------- deep learning
md(
    """
## 6. Deep learning pipeline
The deep pipeline learns its representation end to end. Numeric columns are only imputed, clipped and
standardized. Categorical columns become integer codes that feed trainable **entity embeddings**.

* **MLP**: embeddings + numeric features -> [Linear -> BatchNorm -> ReLU -> Dropout] x L -> logit,
  trained with AdamW and a class-weighted BCE loss. Early stopping uses the validation PR-AUC.
* **TabNet** (Arik & Pfister, 2021): sequential attention that selects the features to use at each decision step.

### 6.1 MLP architecture and training variants
"""
)
code(
    """
torch.set_num_threads(CORES)
mlp_results, mlp_models = deep.run_mlp_variants(X_train, y_train, X_test, y_test)
display(mlp_results[["model", "val_pr_auc", "pr_auc", "roc_auc", "f2_tuned", "recall_tuned", "precision_tuned", "epochs", "fit_time"]].round(4))
"""
)
code(
    """
ref_mlp = mlp_models["MLP 128-64 (reference)"]
hist = pd.DataFrame(ref_mlp.history_)
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(hist.epoch, hist.train_loss, label="train loss (weighted BCE)", color="#4c72b0")
ax2 = ax1.twinx(); ax2.plot(hist.epoch, hist.val_pr_auc, label="validation PR-AUC", color="#dd8452")
ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax2.set_ylabel("PR-AUC")
fig.legend(loc="center right"); ax1.set_title("Training curve of the reference MLP")
fig.tight_layout(); reporting.save_figure(fig, "mlp_training"); plt.show()
"""
)
md("### 6.2 TabNet")
code(
    """
tabnet = deep.TabNetWrapper().fit(X_train, y_train)
tabnet_row = deep.evaluate_network("TabNet", tabnet, X_test, y_test)
display(pd.DataFrame([tabnet_row]).round(4))
fig, ax = plt.subplots(figsize=(7, 4))
tabnet.feature_importances().sort_values().plot.barh(ax=ax, color="#55a868")
ax.set_title("TabNet attention-based feature importance"); fig.tight_layout()
reporting.save_figure(fig, "tabnet_importance"); plt.show()
"""
)
md(
    """
### 6.3 Learned embeddings as features
The 64-dimensional penultimate layer of the reference MLP is saved as a feature set. A plain logistic
regression trained on it shows how much of the network's power lies in the learned representation.
"""
)
code(
    """
from sklearn.linear_model import LogisticRegression
emb_train, emb_test = ref_mlp.embed(X_train), ref_mlp.embed(X_test)
reporting.save_features("deep_mlp_embeddings", {"X_train": emb_train.astype(np.float32), "X_test": emb_test.astype(np.float32),
                                                "y_train": y_train.to_numpy(np.int8), "y_test": y_test.to_numpy(np.int8)},
                        meta={"source": "penultimate layer of MLP 128-64 (reference)"})
lr_emb = LogisticRegression(max_iter=2000, class_weight="balanced").fit(emb_train, y_train)
p = lr_emb.predict_proba(emb_test)[:, 1]
emb_row = {"model": "LogReg on MLP embeddings", **evaluation.compute_metrics(y_test, (p >= 0.5).astype(int), p)}
display(pd.DataFrame([emb_row]).round(4))
"""
)
md("### 6.4 Traditional vs deep learning on the test set")
code(
    """
dl_table = pd.concat([mlp_results[mlp_results.model == "MLP 128-64 (reference)"], pd.DataFrame([tabnet_row])], ignore_index=True)
dl_table.to_csv(config.RESULTS_DIR / "test_deep.csv", index=False)
curves_dl = {
    pipeline.MODEL_LABELS[best_key]: (y_test, evaluation.scores_of(best[best_key], X_test)),
    "Logistic Regression": (y_test, evaluation.scores_of(best["logreg"], X_test)),
    "MLP (embeddings)": (y_test, ref_mlp.predict_proba(X_test)[:, 1]),
    "TabNet": (y_test, tabnet.predict_proba(X_test)[:, 1]),
}
evaluation.plot_roc_pr(curves_dl, path=config.FIGURES_DIR / "roc_pr_traditional_vs_deep.pdf", title_suffix=" (test set)"); plt.show()
"""
)

# ------------------------------------------------------------------------------- Bayesian net
md(
    """
## 7. Extension: Bayesian network (Talented Engineering program)
Implemented from scratch in `modules/creditrisk/bayesnet/`:

* `graph.py`: DAG with cycle checks, Kahn topological sort and Markov blankets.
* `network.py`: CPTs estimated by MAP with a Dirichlet prior (Laplace smoothing), computed per family on the rows
  where that family is fully observed (missing values need no imputation), plus ancestral sampling.
* `structure.py`: Naive Bayes, TAN (Chow-Liu tree on conditional mutual information), greedy hill climbing on BIC,
  and an expert-designed DAG.
* `inference.py`: **exact** inference by variable elimination (barren-node pruning, min-weight elimination order),
  checked against brute-force enumeration in the unit tests. **Approximate** inference by rejection sampling,
  likelihood weighting and Gibbs sampling.

Numeric features are discretized into quantile bins learned on the training split.

### 7.1 Discretization granularity and smoothing, chosen per structure on a validation split of the training set
"""
)
code(
    """
bn_sens = bnx.run_sensitivity(X_train, y_train)
sens_table = bn_sens.pivot_table(index=["n_bins", "alpha"], columns="structure", values="pr_auc")
display(sens_table.round(4))
# Each structure gets the (bins, alpha) that maximizes its own validation PR-AUC.
BN_SETTINGS = bnx.select_settings(bn_sens)
print("Selected (n_bins, alpha) per structure:", BN_SETTINGS)
"""
)
md("### 7.2 Structures and classification performance on the test set")
code(
    """
bn_results, bn_models = bnx.run_classification(X_train, y_train, X_test, y_test, settings=BN_SETTINGS)
display(bn_results[["structure", "n_bins", "alpha", "edges", "parameters", "test_loglik", "pr_auc", "roc_auc", "threshold", "f2_tuned", "recall_tuned", "precision_tuned", "inference_time"]].round(4))
"""
)
code(
    """
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
for ax, (name, m) in zip(axes.flat, bn_models.items()):
    bnx.draw_dag(m["bn"].dag, f"{name} ({len(m['bn'].dag.edges)} edges)", ax=ax)
fig.tight_layout(); reporting.save_figure(fig, "bn_structures"); plt.show()
trace = bn_models["Hill-climb BIC"]["trace"]
fig, ax = plt.subplots(figsize=(6, 3.5)); ax.plot(trace, marker="o", ms=3)
ax.set_xlabel("hill-climbing step"); ax.set_ylabel("BIC score"); ax.set_title("Structure search progress")
fig.tight_layout(); reporting.save_figure(fig, "bn_bic_trace"); plt.show()
"""
)
md(
    """
### 7.3 Reasoning with the networks (exact queries by variable elimination)
Predictive queries (cause to effect) and diagnostic queries (effect to cause), answered exactly. The empirical
column gives the conditional frequency in the training data. A network whose answers drift from it makes an
independence assumption that the data contradict.
"""
)
code(
    """
bn_best_name = bn_results.sort_values("pr_auc", ascending=False).iloc[0]["structure"]
print("Best BN classifier:", bn_best_name)
# Compare exact answers of the best (TAN) and the expert network with the empirical frequencies.
queries = bnx.example_queries({name: (bn_models[name]["bn"], bn_models[name]["disc"].states_) for name in ("TAN", "Expert DAG")},
                              X_train, y_train)
queries.to_csv(config.RESULTS_DIR / "bn_queries.csv", index=False)
queries.round(4)
"""
)
md(
    """
### 7.4 Exact vs approximate inference
For random test borrowers we condition on 2, 5 or 10 of their observed features and compare the sampled
estimate of P(default | e) with the exact value from variable elimination.
"""
)
code(
    """
tan = bn_models["TAN"]
inf = bnx.run_inference_comparison(tan["bn"], tan["disc"].transform(X_test), verbose=False)
display(inf.round(4))
approx = inf[inf.method != "Variable elimination"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
for ax, (k, grp) in zip(axes, approx.groupby("evidence_size")):
    for method, g in grp.groupby("method"):
        ax.plot(g.samples, g.mae, marker="o", label=method)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"{k} evidence variables")
    ax.set_xlabel("number of samples"); ax.grid(alpha=0.3)
axes[0].set_ylabel("mean |estimate - exact|"); axes[0].legend()
fig.tight_layout(); reporting.save_figure(fig, "bn_inference_error"); plt.show()
"""
)

# ------------------------------------------------------------------------------ comparison
md("## 8. Overall comparison (test set, 6,484 loans)")
code(
    """
def pick(df, name_col, name, family, label=None):
    r = df[df[name_col] == name].iloc[0]
    return {"family": family, "model": label or name, "pr_auc": r.pr_auc, "roc_auc": r.roc_auc,
            "f2": r.f2_tuned, "recall": r.recall_tuned, "precision": r.precision_tuned, "accuracy": r.accuracy_tuned}

overall = pd.DataFrame(
    [pick(test_results, "key", k, "Traditional", pipeline.MODEL_LABELS[k]) for k in best]
    + [pick(mlp_results, "model", "MLP 128-64 (reference)", "Deep learning", "MLP (entity embeddings)"),
       pick(pd.DataFrame([tabnet_row]), "model", "TabNet", "Deep learning")]
    + [pick(bn_results, "structure", s, "Bayesian network", f"BN - {s}") for s in bn_results.structure]
).sort_values("pr_auc", ascending=False).reset_index(drop=True)
overall.to_csv(config.RESULTS_DIR / "overall.csv", index=False)
display(overall.round(4))
fig, ax = plt.subplots(figsize=(9, 5.5))
colors = overall.family.map({"Traditional": "#4c72b0", "Deep learning": "#dd8452", "Bayesian network": "#55a868"})
ax.barh(overall.model[::-1], overall.pr_auc[::-1], color=colors[::-1])
for fam, c in [("Traditional", "#4c72b0"), ("Deep learning", "#dd8452"), ("Bayesian network", "#55a868")]:
    ax.barh([], [], color=c, label=fam)
ax.set_xlim(0.5, 1); ax.set_xlabel("PR-AUC on the test set"); ax.legend(loc="lower right")
ax.set_title("All models ranked by PR-AUC")
fig.tight_layout(); reporting.save_figure(fig, "overall_pr_auc"); plt.show()
"""
)
md(
    """
## 9. Conclusions
* **Gradient-boosted trees win.** Tree ensembles capture the strong interactions between grade, loan-to-income
  ratio and home ownership. Linear models cannot represent them without hand-crafted features.
* **Preprocessing matters most for the scale-sensitive models.** Scaling, clipping and the encoding choice
  change k-NN, SVM and logistic regression substantially, while trees are almost unaffected. PCA never helps
  on this low-dimensional data.
* **Deep learning is competitive but does not beat boosting** on 26k rows of tabular data, in line with the
  literature (Grinsztajn et al., 2022). The entity-embedding MLP beats every linear model.
* **Bayesian networks** give interpretable, probabilistic reasoning, including diagnostic queries and native
  handling of missing features. They lose some accuracy to discretization. TAN, which learns its structure
  for predicting the class, is the best network classifier. Likelihood weighting and Gibbs sampling scale with
  evidence, while rejection sampling collapses when many variables are observed.
"""
)
code('print(f"Total runtime: {(time.perf_counter() - T_START) / 60:.1f} minutes")')


def build(path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1] / "notebooks" / "credit_risk_pipeline.ipynb")

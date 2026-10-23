"""Turn the notebook's saved results (results/*.csv) into LaTeX tables for the report.

Run after the notebook:  python scripts/export_report_tables.py
Tables land in reports/generated/, which reports/latex/report.tex includes, so every number in
the PDF comes from the same run as the notebook.
"""

import ast
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules"))

from creditrisk import config, data, eda  # noqa: E402
from creditrisk.pipeline import MODEL_LABELS  # noqa: E402
from creditrisk.reporting import latex_table, write_values  # noqa: E402

R = config.RESULTS_DIR
SUITE_TITLES = {
    "imputation": (["model", "imputer", "indicator"], "Imputation strategies"),
    "scaling": (["model", "scaler"], "Scaling methods"),
    "outlier_fe": (["model", "outlier", "feat_eng"], "Outlier handling and feature engineering"),
    "encoding": (["model", "grade_encoding"], "Encoding of the loan grade"),
    "dimensionality": (["model", "reduction"], "PCA and univariate feature selection"),
    "imbalance": (["model", "imbalance"], "Class-imbalance strategies"),
}
TUNED = {"f2_tuned": "F2", "recall_tuned": "recall", "precision_tuned": "precision", "accuracy_tuned": "accuracy"}


def main() -> None:
    raw = data.load_raw()
    X_train, X_test, y_train, _, cleaning = data.load_splits()

    ov = eda.overview(raw)
    latex_table(
        ov[["column", "type", "missing", "missing_pct", "unique"]].rename(columns={"missing_pct": "missing (%)"}),
        "tab_overview", "Columns of the raw dataset.", "tab:overview", float_fmt="{:.2f}",
    )
    num = eda.numeric_summary(raw)[["mean", "std", "min", "1%", "50%", "99%", "max", "skew"]]
    latex_table(num.reset_index().rename(columns={"index": "feature"}), "tab_numeric",
                "Descriptive statistics of the numeric features (raw data).", "tab:numeric", float_fmt="{:.2f}",
                size="\\scriptsize")

    for name, (keys, title) in SUITE_TITLES.items():
        df = pd.read_csv(R / f"suite_{name}.csv")
        latex_table(
            df[keys + ["pr_auc", "pr_auc_std", "roc_auc", "f2", "recall", "precision"]].rename(columns={"pr_auc_std": "sd"}),
            f"tab_suite_{name}", f"{title}: 5-fold CV on the training set (F2, recall and precision at threshold 0.5).",
            f"tab:suite_{name}", size="\\scriptsize",
        )

    imp = pd.read_csv(R / "impossible_values.csv")
    imp_t = imp.pivot(index="strategy", columns="model", values="pr_auc").reindex(imp.strategy.unique()).reset_index()
    imp_t["strategy"] = [f"{i + 1}. {name}" for i, name in enumerate(imp_t["strategy"])]
    latex_table(
        imp_t, "tab_impossible",
        "Handling the 7 impossible training values: 5-fold CV PR-AUC on identical validation rows "
        f"(mean fold standard deviation {imp['pr_auc_std'].mean():.4f}).",
        "tab:impossible", float_fmt="{:.5f}",
    )

    head = pd.read_csv(R / "headroom.csv")
    latex_table(
        head[["approach", "cv_pr_auc", "cv_pr_auc_std", "gain_vs_grid"]].rename(
            columns={"cv_pr_auc": "CV PR-AUC", "cv_pr_auc_std": "sd", "gain_vs_grid": "gain vs grid"}),
        "tab_headroom", "Headroom check: stronger optimization of the best model family (5-fold CV on the training set).",
        "tab:headroom", column_format="p{8.2cm}rrr",
    )

    tuning = pd.read_csv(R / "tuning.csv")
    tuning["best_params"] = tuning.best_params.map(
        lambda s: ", ".join(f"{k.replace('model__', '')}={v}" for k, v in ast.literal_eval(s).items())
    )
    latex_table(
        tuning[["model", "best_params", "cv_pr_auc", "cv_pr_auc_std", "cv_roc_auc", "cv_f2", "search_time"]].rename(
            columns={"cv_pr_auc": "CV PR-AUC", "cv_pr_auc_std": "sd", "cv_roc_auc": "CV ROC-AUC", "cv_f2": "CV F2",
                     "search_time": "time (s)"}),
        "tab_tuning", "Grid-search results (5-fold CV, refit on PR-AUC; F2 at threshold 0.5).", "tab:tuning",
        size="\\scriptsize", column_format="lp{5.4cm}rrrrr", col_formats={"time (s)": "{:.1f}"},
    )

    test = pd.read_csv(R / "test_traditional.csv").sort_values("pr_auc", ascending=False)
    latex_table(
        test[["model", "pr_auc", "roc_auc", "threshold", *TUNED]].rename(columns=TUNED),
        "tab_test_traditional",
        "Test-set results of the tuned traditional models; F2, recall, precision and accuracy at the F2-tuned threshold.",
        "tab:test_traditional", bold_max=["pr_auc", "roc_auc", "F2", "recall"], size="\\small",
    )

    mlp = pd.read_csv(R / "dl_mlp_variants.csv")
    latex_table(
        mlp[["model", "val_pr_auc", "pr_auc", "roc_auc", "f2_tuned", "recall_tuned", "epochs", "fit_time"]].rename(
            columns={"val_pr_auc": "val PR-AUC", "f2_tuned": "F2", "recall_tuned": "recall", "fit_time": "time (s)"}),
        "tab_mlp", "MLP variants: validation PR-AUC (used for selection) and test metrics.", "tab:mlp",
        size="\\scriptsize", col_formats={"time (s)": "{:.1f}"},
    )
    deep = pd.read_csv(R / "test_deep.csv")
    latex_table(
        deep[["model", "pr_auc", "roc_auc", "threshold", *TUNED, "fit_time"]].rename(columns={**TUNED, "fit_time": "time (s)"}),
        "tab_deep", "Deep learning models on the test set (threshold tuned for F2 on a validation split).", "tab:deep",
        size="\\small", col_formats={"time (s)": "{:.1f}"},
    )

    sens = pd.read_csv(R / "bn_sensitivity.csv")
    latex_table(
        sens.pivot_table(index=["n_bins", "alpha"], columns="structure", values="pr_auc").reset_index(),
        "tab_bn_sensitivity", "Bayesian networks: validation PR-AUC by number of bins and Dirichlet pseudo-count $\\alpha$.",
        "tab:bn_sens", size="\\small", col_formats={"alpha": "{:g}"},
    )
    bn = pd.read_csv(R / "bn_classification.csv")
    latex_table(
        bn[["structure", "n_bins", "alpha", "edges", "parameters", "test_loglik", "pr_auc", "roc_auc", *TUNED]].rename(
            columns={**TUNED, "test_loglik": "test log-lik.", "n_bins": "bins"}),
        "tab_bn_results",
        "Bayesian-network structures on the test set, each with the bins and $\\alpha$ selected on validation data.",
        "tab:bn_results",
        bold_max=["pr_auc", "roc_auc"], size="\\scriptsize",
        col_formats={"alpha": "{:g}", "test log-lik.": "{:.0f}"},
    )
    queries = pd.read_csv(R / "bn_queries.csv")
    latex_table(queries, "tab_bn_queries", "Exact answers of two networks (variable elimination) against the empirical frequency in the training data; support is the number of training rows matching the evidence.", "tab:bn_queries")

    inf = pd.read_csv(R / "bn_inference.csv")
    approx = inf[inf.method != "Variable elimination"]
    mae = approx.pivot_table(index=["method", "samples"], columns="evidence_size", values="mae").reset_index()
    mae.columns = ["method", "samples"] + [f"MAE, {c} evid." for c in mae.columns[2:]]
    acc = approx[approx.method == "Rejection sampling"].pivot_table(index="samples", columns="evidence_size", values="acceptance_rate")
    mae["samples"] = mae["samples"].astype(int)
    latex_table(mae, "tab_bn_inference",
                "Mean absolute error of approximate $P(\\text{default} \\mid \\bm{e})$ against variable elimination "
                "(60 random test applicants per evidence size).", "tab:bn_inference")
    acc = acc.reset_index()
    acc.columns = ["samples"] + [f"{c} evidence vars" for c in acc.columns[1:]]
    acc["samples"] = acc["samples"].astype(int)
    latex_table(acc, "tab_bn_acceptance", "Acceptance rate of rejection sampling.", "tab:bn_acceptance", float_fmt="{:.5f}")
    ve_time = inf[inf.method == "Variable elimination"].set_index("evidence_size")["time_per_query"]

    overall = pd.read_csv(R / "overall.csv")
    latex_table(
        overall, "tab_overall",
        "All model families on the test set, ranked by PR-AUC (F2, recall, precision and accuracy at the tuned threshold).",
        "tab:overall", bold_max=["pr_auc", "roc_auc", "f2", "recall"], size="\\small",
    )

    best = test.iloc[0]
    write_values({
        "n_raw": cleaning.n_raw, "n_clean": cleaning.n_clean, "n_dup": cleaning.n_duplicates,
        "n_train": len(X_train), "n_test": len(X_test), "default_rate": float(y_train.mean()),
        "best_model": MODEL_LABELS[best.key], "best_pr_auc": float(best.pr_auc),
        "best_f_two": float(best.f2_tuned),
        # Percentages read naturally in sentences such as "detects 85.1% of the defaulters".
        "best_recall": f"{100 * best.recall_tuned:.1f}\\%", "best_precision": f"{100 * best.precision_tuned:.1f}\\%",
        "ve_ms": f"{1000 * ve_time.max():.1f}",
    })
    print(sorted(p.name for p in config.TABLES_DIR.glob("*.tex")))


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd

from creditrisk import config, deep, reporting


def test_seeds_derive_from_project_seed():
    assert deep.SEEDS[0] == config.SEED
    assert len(set(deep.SEEDS)) == len(deep.SEEDS) == 5


def test_summarize_seeds_gives_mean_and_sample_std_per_model():
    runs = pd.DataFrame(
        {"model": ["A", "A", "A", "B", "B"], "seed": [42, 43, 44, 42, 43], "pr_auc": [0.8, 0.9, 1.0, 0.5, 0.7]}
    )
    out = deep.summarize_seeds(runs).set_index("model")
    assert list(out.index) == ["A", "B"]
    assert out.loc["A", "n_seeds"] == 3
    assert np.isclose(out.loc["A", "pr_auc"], 0.9) and np.isclose(out.loc["A", "pr_auc_std"], 0.1)
    assert np.isclose(out.loc["B", "pr_auc_std"], np.std([0.5, 0.7], ddof=1))
    assert "seed" not in out and "seed_std" not in out


def test_latex_table_shows_spread_only_where_present(tmp_path, monkeypatch):
    for name in ("DATA_DIR", "FEATURES_DIR", "RESULTS_DIR", "FIGURES_DIR", "TABLES_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name)
    df = pd.DataFrame({"model": ["HGB", "MLP"], "pr_auc": [0.9, 0.86], "pr_auc_std": [np.nan, 0.0012]})
    tex = reporting.latex_table(df, "t", "c", "l", bold_max=["pr_auc"], spread={"pr_auc": "pr_auc_std"})
    assert "HGB & \\textbf{0.9000} \\\\" in tex
    assert "MLP & 0.8600 $\\pm$ 0.0012 \\\\" in tex
    assert "std" not in tex

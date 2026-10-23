"""Saving extracted features and exporting result tables / plots for the LaTeX report."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config


def save_features(name: str, arrays: dict[str, np.ndarray], meta: dict | None = None) -> Path:
    """Write each array as features/<name>/<key>.npy and add them to features/credit_features.h5."""
    out = config.FEATURES_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    for key, arr in arrays.items():
        np.save(out / f"{key}.npy", arr)
    if meta:
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    with h5py.File(config.FEATURES_DIR / "credit_features.h5", "a") as h5:
        grp = h5.require_group(name)
        for key, arr in arrays.items():
            if key in grp:
                del grp[key]
            grp.create_dataset(key, data=arr, compression="gzip")
        for k, v in (meta or {}).items():
            grp.attrs[k] = json.dumps(v, default=str)
    return out


def save_figure(fig, name: str):
    config.ensure_dirs()
    fig.savefig(config.FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=130, bbox_inches="tight")
    return fig


def plot_suite(df: pd.DataFrame, x: str, hue: str = "model", metric: str = "pr_auc", title: str = "", name: str | None = None):
    """Grouped bar chart of one CV metric with its standard deviation as error bars."""
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * df[x].nunique() * df[hue].nunique() / 2), 4))
    order = list(dict.fromkeys(df[x].astype(str)))
    sns.barplot(data=df.assign(**{x: df[x].astype(str)}), x=x, y=metric, hue=hue, order=order, ax=ax, errorbar=None)
    # Error bars from the stored fold standard deviations.
    for container, (_, grp) in zip(ax.containers, df.groupby(hue, sort=False)):
        grp = grp.set_index(grp[x].astype(str)).reindex(order)
        xs = [bar.get_x() + bar.get_width() / 2 for bar in container]
        ax.errorbar(xs, grp[metric].values, yerr=grp[f"{metric}_std"].values, fmt="none", ecolor="k", capsize=2, lw=0.8)
    lo = max(0.0, df[metric].min() - 0.1)
    ax.set_ylim(lo, min(1.0, df[metric].max() + 0.05))
    ax.set_ylabel(metric.upper().replace("_", "-"))
    ax.set_xlabel("")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if name:
        save_figure(fig, name)
    return fig


def latex_table(
    df: pd.DataFrame,
    name: str,
    caption: str,
    label: str,
    float_fmt: str = "{:.4f}",
    bold_max: list[str] | None = None,
    column_format: str | None = None,
    size: str = "\\small",
) -> str:
    """Booktabs table written to reports/generated/<name>.tex (bolding the best per column)."""
    df = df.copy()
    for col in bold_max or []:
        best = df[col].max()
        df[col] = [f"\\textbf{{{float_fmt.format(v)}}}" if v == best else float_fmt.format(v) for v in df[col]]
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda v: "--" if pd.isna(v) else float_fmt.format(v))
    df.columns = [str(c).replace("_", "\\_") for c in df.columns]
    df = df.map(lambda v: str(v).replace("_", "\\_").replace("%", "\\%") if isinstance(v, str) and "\\textbf" not in v else v)
    body = df.to_latex(index=False, escape=False, column_format=column_format or "l" * len(df.columns))
    tex = (
        "\\begin{table}[H]\n\\centering\n"
        f"{size}\n{body}"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    )
    config.ensure_dirs()
    (config.TABLES_DIR / f"{name}.tex").write_text(tex)
    return tex


def write_values(values: dict, name: str = "values") -> None:
    """Scalar results as LaTeX macros (\\newcommand) so the report text quotes live numbers."""
    lines = []
    for key, val in values.items():
        macro = "".join(ch for ch in key.title() if ch.isalpha())
        text = f"{val:.4f}" if isinstance(val, float) else str(val)
        lines.append(f"\\newcommand{{\\{macro}}}{{{text}}}")
    config.ensure_dirs()
    (config.TABLES_DIR / f"{name}.tex").write_text("\n".join(lines) + "\n")

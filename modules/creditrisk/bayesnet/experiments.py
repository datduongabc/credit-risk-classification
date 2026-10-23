"""Bayesian-network experiments on the credit data: structures, classification, inference."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .. import config
from ..evaluation import compute_metrics, f2_threshold
from .discretize import Discretizer
from .graph import DAG
from .inference import (
    gibbs_sampling,
    likelihood_weighting,
    predict_proba,
    rejection_sampling,
    variable_elimination,
)
from .network import BayesianNetwork
from .structure import expert_dag, hill_climb_dag, naive_bayes_dag, tan_dag

TARGET = config.TARGET
SHORT = {
    "person_age": "age",
    "person_income": "income",
    "person_emp_length": "emp_len",
    "loan_amnt": "amount",
    "loan_int_rate": "int_rate",
    "loan_percent_income": "pct_income",
    "cb_person_cred_hist_length": "hist_len",
    "person_home_ownership": "home",
    "loan_intent": "intent",
    "cb_person_default_on_file": "prev_default",
    "loan_grade": "grade",
    "loan_status": "DEFAULT",
}


STRUCTURES = ("Naive Bayes", "TAN", "Hill-climb BIC", "Expert DAG")


def learn_structure(name: str, data: np.ndarray, variables: list[str], cards: dict[str, int], max_parents: int = 3):
    """Return (DAG, info) for one named structure; info holds the BIC trace for hill climbing."""
    if name == "Naive Bayes":
        return naive_bayes_dag(variables, TARGET), {}
    if name == "TAN":
        return tan_dag(data, variables, TARGET, cards), {}
    if name == "Hill-climb BIC":
        t0 = time.perf_counter()
        dag, trace = hill_climb_dag(data, variables, cards, max_parents=max_parents)
        return dag, {"trace": trace, "time": time.perf_counter() - t0}
    if name == "Expert DAG":
        return expert_dag(variables), {}
    raise ValueError(f"unknown structure {name!r}")


def select_settings(sensitivity: pd.DataFrame) -> dict[str, tuple[int, float]]:
    """Best (n_bins, alpha) per structure on the validation split.

    Each structure gets its own setting: denser graphs (TAN) overfit with fine bins, while
    sparse ones (Naive Bayes, hill climbing with the BIC penalty) can afford more bins, so a
    single shared setting would favour some structures at the expense of others.
    """
    best = sensitivity.loc[sensitivity.groupby("structure")["pr_auc"].idxmax()]
    return {r.structure: (int(r.n_bins), float(r.alpha)) for r in best.itertuples()}


def run_classification(
    X_train, y_train, X_test, y_test, settings=None, n_bins=5, alpha=1.0, max_parents=3, verbose=True, save=True
):
    """Fit each structure on the training split and evaluate P(default | features) on test.

    ``settings`` maps a structure name to its (n_bins, alpha); without it every structure
    uses ``n_bins`` and ``alpha``.
    """
    settings = settings or {name: (n_bins, alpha) for name in STRUCTURES}
    # Validation split of the training data picks each network's F2 threshold.
    tr_idx, val_idx = train_test_split(
        np.arange(len(X_train)), test_size=0.15, stratify=y_train, random_state=config.SEED
    )
    rows, models, discs = [], {}, {}
    for name, (bins, a) in settings.items():
        if bins not in discs:
            disc = Discretizer(bins).fit(X_train)
            discs[bins] = (disc, disc.transform(X_train, y_train), disc.transform(X_test))
        disc, D_train, D_test = discs[bins]
        variables, states = disc.variables_, disc.states_
        cards = {v: len(states[v]) for v in variables}
        dag, info = learn_structure(name, D_train, variables, cards, max_parents)
        D_val = D_train[val_idx].copy()
        D_val[:, variables.index(TARGET)] = -1

        sub = BayesianNetwork(dag, states).fit(D_train[tr_idx], a)
        thr = f2_threshold(np.asarray(y_train)[val_idx], predict_proba(sub, D_val, TARGET)[:, 1])
        bn = BayesianNetwork(dag, states).fit(D_train, a)
        t0 = time.perf_counter()
        proba = predict_proba(bn, D_test, TARGET)[:, 1]
        infer_time = time.perf_counter() - t0
        row = {
            "structure": name,
            "n_bins": bins,
            "alpha": a,
            "edges": len(dag.edges),
            "parameters": bn.n_parameters(),
            "test_loglik": bn.log_likelihood(disc.transform(X_test, y_test)),
            "inference_time": infer_time,
            "threshold": thr,
            **compute_metrics(y_test, (proba >= 0.5).astype(int), proba),
            **{f"{k}_tuned": v for k, v in compute_metrics(y_test, (proba >= thr).astype(int)).items()},
        }
        if "time" in info:
            row["learn_time"] = info["time"]
        rows.append(row)
        models[name] = {"bn": bn, "disc": disc, "proba": proba, "threshold": thr, **info}
        if verbose:
            print(
                f"[BN] {name:15s} bins={bins:2d} alpha={a:<4g} edges={row['edges']:2d} params={row['parameters']:5d} "
                f"PR-AUC={row['pr_auc']:.4f} F2(tuned)={row['f2_tuned']:.4f} ({infer_time:.1f}s)"
            )
    df = pd.DataFrame(rows)
    if save:
        config.ensure_dirs()
        df.to_csv(config.RESULTS_DIR / "bn_classification.csv", index=False)
    return df, models


def run_sensitivity(X_train, y_train, bins=(3, 5, 8, 12), alphas=(0.1, 1.0, 10.0)):
    """How discretization granularity and the Dirichlet pseudo-count change each structure.

    Scored on a validation split of the training data so that the chosen setting is not
    tuned on the test set.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=config.SEED
    )
    frames = []
    for b in bins:
        for a in alphas:
            df, _ = run_classification(X_fit, y_fit, X_val, y_val, n_bins=b, alpha=a, verbose=False, save=False)
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(config.RESULTS_DIR / "bn_sensitivity.csv", index=False)
    return out


def run_inference_comparison(
    bn: BayesianNetwork,
    D_test: np.ndarray,
    n_queries: int = 60,
    evidence_sizes=(2, 5, 10),
    sample_sizes=(100, 1_000, 10_000),
    seed: int = config.SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Error of approximate P(default | e) against variable elimination, by evidence size."""
    rng = np.random.default_rng(seed)
    feats = [v for v in bn.variables if v != TARGET]
    rows = []
    for k in evidence_sizes:
        queries = []
        for r in rng.choice(len(D_test), n_queries, replace=False):
            observed = [v for v in feats if D_test[r, bn.index[v]] >= 0]
            chosen = rng.choice(observed, min(k, len(observed)), replace=False)
            queries.append({v: int(D_test[r, bn.index[v]]) for v in chosen})
        t0 = time.perf_counter()
        exact = [variable_elimination(bn, TARGET, e)[1] for e in queries]
        ve_time = (time.perf_counter() - t0) / n_queries
        rows.append({"evidence_size": k, "method": "Variable elimination", "samples": 0, "mae": 0.0, "max_error": 0.0, "time_per_query": ve_time})
        for n in sample_sizes:
            for method in ("Rejection sampling", "Likelihood weighting", "Gibbs sampling"):
                errs, acc = [], []
                t0 = time.perf_counter()
                for e, p in zip(queries, exact):
                    if method == "Rejection sampling":
                        est, accepted = rejection_sampling(bn, TARGET, e, n, rng)
                        acc.append(accepted / n)
                    elif method == "Likelihood weighting":
                        est = likelihood_weighting(bn, TARGET, e, n, rng)
                    else:
                        est = gibbs_sampling(bn, TARGET, e, n, rng, burn_in=100, n_chains=20)
                    errs.append(abs(est[1] - p))
                row = {
                    "evidence_size": k,
                    "method": method,
                    "samples": n,
                    "mae": float(np.mean(errs)),
                    "max_error": float(np.max(errs)),
                    "time_per_query": (time.perf_counter() - t0) / n_queries,
                }
                if acc:
                    row["acceptance_rate"] = float(np.mean(acc))
                rows.append(row)
                if verbose:
                    extra = f" accept={row['acceptance_rate']:.4f}" if acc else ""
                    print(f"[inference] |e|={k:2d} {method:21s} N={n:6d} MAE={row['mae']:.4f}{extra}")
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "bn_inference.csv", index=False)
    return df


# (description, query variable, query states, evidence on raw values); all variables involved
# are categorical, so the empirical frequency in the training data is directly comparable.
QUERIES = [
    ("P(default)", TARGET, (1,), {}),
    ("P(default | grade=A)", TARGET, (1,), {"loan_grade": "A"}),
    ("P(default | grade=D)", TARGET, (1,), {"loan_grade": "D"}),
    ("P(default | grade=G)", TARGET, (1,), {"loan_grade": "G"}),
    ("P(default | home=RENT)", TARGET, (1,), {"person_home_ownership": "RENT"}),
    ("P(default | home=OWN)", TARGET, (1,), {"person_home_ownership": "OWN"}),
    ("P(default | grade=D, home=OWN)", TARGET, (1,), {"loan_grade": "D", "person_home_ownership": "OWN"}),
    ("P(default | intent=MEDICAL)", TARGET, (1,), {"loan_intent": "MEDICAL"}),
    ("P(default | intent=VENTURE)", TARGET, (1,), {"loan_intent": "VENTURE"}),
    # Diagnostic (effect -> cause) queries: reasoning against the arrow direction.
    ("P(prev_default=Y | default=1)", "cb_person_default_on_file", ("Y",), {TARGET: 1}),
    ("P(prev_default=Y | default=0)", "cb_person_default_on_file", ("Y",), {TARGET: 0}),
    ("P(grade in D..G | default=1)", "loan_grade", ("D", "E", "F", "G"), {TARGET: 1}),
    ("P(grade in D..G | default=0)", "loan_grade", ("D", "E", "F", "G"), {TARGET: 0}),
]


def example_queries(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Answer QUERIES exactly (variable elimination) in each network and empirically.

    ``models`` maps a label to a fitted network together with its discretizer's states; the
    empirical column is the conditional frequency in the training data, so a large gap
    reveals an independence assumption of that network that the data contradict.
    """
    df = X_train.assign(**{TARGET: np.asarray(y_train)})
    rows = []
    for desc, query, query_states, evidence in QUERIES:
        mask = np.ones(len(df), dtype=bool)
        for var, val in evidence.items():
            mask &= (df[var] == val).to_numpy()
        row = {"query": desc, "empirical": float(df.loc[mask, query].isin(query_states).mean()), "support": int(mask.sum())}
        for label, (bn, states) in models.items():
            ev = {k: states[k].index(v) for k, v in evidence.items()}
            p = variable_elimination(bn, query, ev)
            row[label] = float(p[[states[query].index(q) for q in query_states]].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def draw_dag(dag: DAG, title: str, path=None, ax=None):
    g = nx.relabel_nodes(dag.to_networkx(), SHORT)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    try:
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    except Exception:
        # Layered layout by topological generation; keeps arrows mostly downward.
        for layer, nodes in enumerate(nx.topological_generations(g)):
            for n in nodes:
                g.nodes[n]["layer"] = layer
        pos = nx.multipartite_layout(g, subset_key="layer", align="horizontal", scale=2)
        pos = {n: (x, -y) for n, (x, y) in pos.items()}
    colors = ["#d9534f" if n == "DEFAULT" else "#9ecae1" for n in g.nodes]
    nx.draw_networkx(
        g, pos, ax=ax, node_color=colors, node_size=1500, font_size=8, arrowsize=14, edge_color="#555555",
        connectionstyle="arc3,rad=0.08",
    )
    ax.set_title(title)
    ax.axis("off")
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig

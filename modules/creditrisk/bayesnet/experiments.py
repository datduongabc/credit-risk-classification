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


def learn_structures(data: np.ndarray, variables: list[str], cards: dict[str, int], max_parents: int = 3) -> dict:
    t0 = time.perf_counter()
    hc, trace = hill_climb_dag(data, variables, cards, max_parents=max_parents)
    hc_time = time.perf_counter() - t0
    return {
        "Naive Bayes": (naive_bayes_dag(variables, TARGET), {}),
        "TAN": (tan_dag(data, variables, TARGET, cards), {}),
        "Hill-climb BIC": (hc, {"trace": trace, "time": hc_time}),
        "Expert DAG": (expert_dag(variables), {}),
    }


def run_classification(X_train, y_train, X_test, y_test, n_bins=5, alpha=1.0, max_parents=3, verbose=True, save=True):
    """Fit each structure on the training split and evaluate P(default | features) on test."""
    disc = Discretizer(n_bins).fit(X_train)
    variables, states = disc.variables_, disc.states_
    cards = {v: len(states[v]) for v in variables}
    D_train = disc.transform(X_train, y_train)
    D_test = disc.transform(X_test)  # target column left as -1 (unknown)
    structures = learn_structures(D_train, variables, cards, max_parents)

    # Validation split of the training data picks each network's F2 threshold.
    tr_idx, val_idx = train_test_split(
        np.arange(len(D_train)), test_size=0.15, stratify=y_train, random_state=config.SEED
    )
    D_val = D_train[val_idx].copy()
    D_val[:, variables.index(TARGET)] = -1

    rows, models = [], {}
    for name, (dag, info) in structures.items():
        sub = BayesianNetwork(dag, states).fit(D_train[tr_idx], alpha)
        thr = f2_threshold(np.asarray(y_train)[val_idx], predict_proba(sub, D_val, TARGET)[:, 1])
        bn = BayesianNetwork(dag, states).fit(D_train, alpha)
        t0 = time.perf_counter()
        proba = predict_proba(bn, D_test, TARGET)[:, 1]
        infer_time = time.perf_counter() - t0
        D_test_full = disc.transform(X_test, y_test)
        row = {
            "structure": name,
            "edges": len(dag.edges),
            "parameters": bn.n_parameters(),
            "test_loglik": bn.log_likelihood(D_test_full),
            "inference_time": infer_time,
            "threshold": thr,
            **compute_metrics(y_test, (proba >= 0.5).astype(int), proba),
            **{f"{k}_tuned": v for k, v in compute_metrics(y_test, (proba >= thr).astype(int)).items()},
        }
        if "time" in info:
            row["learn_time"] = info["time"]
        rows.append(row)
        models[name] = {"bn": bn, "proba": proba, "threshold": thr, **info}
        if verbose:
            print(
                f"[BN] {name:15s} edges={row['edges']:2d} params={row['parameters']:5d} "
                f"PR-AUC={row['pr_auc']:.4f} F2(tuned)={row['f2_tuned']:.4f} ({infer_time:.1f}s)"
            )
    df = pd.DataFrame(rows)
    if save:
        config.ensure_dirs()
        df.to_csv(config.RESULTS_DIR / "bn_classification.csv", index=False)
    return df, models, disc


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
            df, _, _ = run_classification(X_fit, y_fit, X_val, y_val, n_bins=b, alpha=a, verbose=False, save=False)
            frames.append(df.assign(n_bins=b, alpha=a))
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


def example_queries(bn: BayesianNetwork, states: dict) -> pd.DataFrame:
    """Readable predictive and diagnostic queries answered exactly by variable elimination."""

    def code(var, value):
        return states[var].index(value)

    rows = []

    def add(desc, query, evidence, query_states=(1,)):
        p = variable_elimination(bn, query, {k: code(k, v) for k, v in evidence.items()})
        idx = [states[query].index(s) for s in query_states]
        rows.append({"query": desc, "probability": float(p[idx].sum())})

    add("P(default)", TARGET, {})
    for g in ("A", "D", "G"):
        add(f"P(default | grade={g})", TARGET, {"loan_grade": g})
    add("P(default | home=RENT)", TARGET, {"person_home_ownership": "RENT"})
    add("P(default | home=OWN)", TARGET, {"person_home_ownership": "OWN"})
    add("P(default | grade=D, home=OWN)", TARGET, {"loan_grade": "D", "person_home_ownership": "OWN"})
    add("P(default | intent=MEDICAL)", TARGET, {"loan_intent": "MEDICAL"})
    add("P(default | intent=VENTURE)", TARGET, {"loan_intent": "VENTURE"})
    # Diagnostic (effect -> cause) queries: reasoning against the arrow direction.
    add("P(prev_default=Y | default=1)", "cb_person_default_on_file", {TARGET: 1}, ("Y",))
    add("P(prev_default=Y | default=0)", "cb_person_default_on_file", {TARGET: 0}, ("Y",))
    add("P(grade in D..G | default=1)", "loan_grade", {TARGET: 1}, ("D", "E", "F", "G"))
    add("P(grade in D..G | default=0)", "loan_grade", {TARGET: 0}, ("D", "E", "F", "G"))
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

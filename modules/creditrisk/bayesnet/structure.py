"""Structure construction and learning for discrete Bayesian networks.

* Naive Bayes:  class -> every feature.
* TAN:          Naive Bayes plus a maximum-weight spanning tree over the features, with
                conditional mutual information I(Xi; Xj | C) as edge weight (Chow-Liu).
* Hill climbing: greedy search over add / delete / reverse edge moves maximizing the BIC
                score, which decomposes over families so each move is re-scored locally.
* Expert:       a hand-designed DAG encoding lending domain knowledge.
"""

from __future__ import annotations

import numpy as np

from .graph import DAG
from .network import family_counts


def naive_bayes_dag(variables: list[str], target: str) -> DAG:
    return DAG(variables, [(target, v) for v in variables if v != target])


def conditional_mutual_information(data, i, j, c, cards) -> float:
    counts = family_counts(data, [i, j, c], [cards[i], cards[j], cards[c]]) + 1e-9
    p = counts / counts.sum()
    p_ic, p_jc, p_c = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True), p.sum(axis=(0, 1), keepdims=True)
    return float((p * np.log(p * p_c / (p_ic * p_jc))).sum())


def tan_dag(data: np.ndarray, variables: list[str], target: str, cards: dict[str, int]) -> DAG:
    """Tree-Augmented Naive Bayes (Friedman et al., 1997) via Prim's maximum spanning tree."""
    idx = {v: k for k, v in enumerate(variables)}
    feats = [v for v in variables if v != target]
    card_list = [cards[v] for v in variables]
    weight = {}
    for a in range(len(feats)):
        for b in range(a + 1, len(feats)):
            w = conditional_mutual_information(data, idx[feats[a]], idx[feats[b]], idx[target], card_list)
            weight[feats[a], feats[b]] = weight[feats[b], feats[a]] = w
    in_tree, edges = {feats[0]}, []
    while len(in_tree) < len(feats):
        u, v = max(
            ((u, v) for u in in_tree for v in feats if v not in in_tree), key=lambda e: weight[e]
        )
        edges.append((u, v))  # direct the tree away from its root
        in_tree.add(v)
    return DAG(variables, [(target, f) for f in feats] + edges)


def family_bic(data, child: int, parents: list[int], cards: list[int]) -> float:
    """BIC contribution of one family: log-likelihood minus (log N / 2) * free parameters."""
    cols = parents + [child]
    counts = family_counts(data, cols, [cards[c] for c in cols])
    n = counts.sum()
    parent_totals = counts.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.nansum(counts * np.log(counts / parent_totals))
    k = int(np.prod([cards[p] for p in parents])) * (cards[child] - 1)
    return float(ll - 0.5 * np.log(max(n, 1)) * k)


def hill_climb_dag(
    data: np.ndarray,
    variables: list[str],
    cards: dict[str, int],
    max_parents: int = 3,
    start: DAG | None = None,
    max_iter: int = 200,
    forbidden: set[tuple[str, str]] = frozenset(),
) -> tuple[DAG, list[float]]:
    """Greedy BIC hill climbing; returns the DAG and the score trace."""
    idx = {v: k for k, v in enumerate(variables)}
    card_list = [cards[v] for v in variables]
    dag = start.copy() if start else DAG(variables)
    cache: dict[tuple, float] = {}

    def score(child: str, parents: list[str]) -> float:
        key = (child, tuple(sorted(parents)))
        if key not in cache:
            cache[key] = family_bic(data, idx[child], [idx[p] for p in key[1]], card_list)
        return cache[key]

    total = sum(score(v, dag.parents(v)) for v in variables)
    trace = [total]
    for _ in range(max_iter):
        best_delta, best_move = 1e-6, None
        for u in variables:
            for v in variables:
                if u == v:
                    continue
                pv = dag.parents(v)
                if dag.has_edge(u, v):
                    # delete u -> v
                    delta = score(v, [p for p in pv if p != u]) - score(v, pv)
                    if delta > best_delta:
                        best_delta, best_move = delta, ("del", u, v)
                    # reverse u -> v
                    pu = dag.parents(u)
                    if len(pu) < max_parents and (v, u) not in forbidden:
                        dag.remove_edge(u, v)
                        ok = not dag.has_path(u, v)
                        dag.add_edge(u, v)
                        if ok:
                            delta = (
                                score(v, [p for p in pv if p != u]) - score(v, pv) + score(u, pu + [v]) - score(u, pu)
                            )
                            if delta > best_delta:
                                best_delta, best_move = delta, ("rev", u, v)
                elif not dag.has_edge(v, u) and len(pv) < max_parents and (u, v) not in forbidden:
                    if not dag.has_path(v, u):
                        delta = score(v, pv + [u]) - score(v, pv)
                        if delta > best_delta:
                            best_delta, best_move = delta, ("add", u, v)
        if best_move is None:
            break
        op, u, v = best_move
        if op == "add":
            dag.add_edge(u, v)
        elif op == "del":
            dag.remove_edge(u, v)
        else:
            dag.remove_edge(u, v)
            dag.add_edge(v, u)
        total += best_delta
        trace.append(total)
    return dag, trace


def expert_dag(variables: list[str]) -> DAG:
    """Causal story of a loan application, written before looking at learned structures.

    Age drives employment and credit history length; income and the requested amount
    determine the loan-to-income ratio; the lender's grade reflects past defaults and history
    and sets the interest rate; default depends on grade, burden, housing and purpose.
    """
    edges = [
        ("person_age", "person_emp_length"),
        ("person_age", "cb_person_cred_hist_length"),
        ("person_age", "person_income"),
        ("person_income", "person_home_ownership"),
        ("person_income", "loan_percent_income"),
        ("loan_amnt", "loan_percent_income"),
        ("cb_person_default_on_file", "loan_grade"),
        ("loan_grade", "loan_int_rate"),
        ("loan_grade", "loan_status"),
        ("loan_percent_income", "loan_status"),
        ("person_home_ownership", "loan_status"),
        ("loan_intent", "loan_status"),
    ]
    return DAG(variables, edges)

"""Exact and approximate inference for ``BayesianNetwork``.

Exact:       variable elimination (with barren-node pruning and a greedy min-weight
             elimination order) and brute-force enumeration (reference for testing).
Approximate: rejection sampling, likelihood weighting and Gibbs sampling.

Every function answers P(query | evidence) for a single query variable and returns a
normalized probability vector over the query's states.
"""

from __future__ import annotations

import itertools

import numpy as np

from .factor import Factor
from .network import BayesianNetwork, sample_categorical


def _relevant_variables(bn: BayesianNetwork, query: str, evidence: dict[str, int]) -> set[str]:
    """Ancestors of the query and evidence; every other node is barren and sums to one."""
    keep, stack = set(), [query, *evidence]
    while stack:
        v = stack.pop()
        if v not in keep:
            keep.add(v)
            stack.extend(bn.dag.parents(v))
    return keep


def elimination_order(factors: list[Factor], hidden: set[str], cards: dict[str, int]) -> list[str]:
    """Greedy min-weight heuristic: eliminate the variable whose resulting factor is smallest."""
    neighbors = {v: set() for v in hidden}
    for f in factors:
        for v in f.variables:
            if v in neighbors:
                neighbors[v] |= set(f.variables) - {v}
    order, remaining = [], set(hidden)
    while remaining:
        best = min(remaining, key=lambda v: np.prod([cards[u] for u in neighbors[v]] or [1]) * cards[v])
        order.append(best)
        remaining.discard(best)
        for u in neighbors[best]:
            if u in neighbors:
                neighbors[u] |= neighbors[best] - {u}
                neighbors[u].discard(best)
    return order


def variable_elimination(bn: BayesianNetwork, query: str, evidence: dict[str, int]) -> np.ndarray:
    evidence = {k: v for k, v in evidence.items() if k != query}
    relevant = _relevant_variables(bn, query, evidence)
    factors = [bn.factor(v).reduce(evidence) for v in bn.variables if v in relevant]
    hidden = relevant - set(evidence) - {query}
    for var in elimination_order(factors, hidden, bn.card):
        involved = [f for f in factors if var in f.variables]
        factors = [f for f in factors if var not in f.variables]
        prod = involved[0]
        for f in involved[1:]:
            prod = prod.product(f)
        factors.append(prod.marginalize(var))
    result = factors[0]
    for f in factors[1:]:
        result = result.product(f)
    return result.normalize().values


def enumeration(bn: BayesianNetwork, query: str, evidence: dict[str, int]) -> np.ndarray:
    """Sum the full joint over all hidden assignments. Exponential; used only in tests."""
    hidden = [v for v in bn.variables if v not in evidence and v != query]
    out = np.zeros(bn.card[query])
    for q in range(bn.card[query]):
        for combo in itertools.product(*(range(bn.card[h]) for h in hidden)):
            assign = {**evidence, query: q, **dict(zip(hidden, combo))}
            p = 1.0
            for v in bn.variables:
                idx = tuple(assign[u] for u in bn.dag.parents(v)) + (assign[v],)
                p *= bn.cpts[v][idx]
            out[q] += p
    return out / out.sum()


def rejection_sampling(bn, query, evidence, n_samples, rng) -> tuple[np.ndarray, int]:
    """Forward-sample and keep samples consistent with the evidence.

    Returns the estimate and the number of accepted samples; with unlikely evidence most
    samples are rejected, which is exactly the weakness the experiments illustrate.
    """
    samples, _ = bn.sample(n_samples, rng)
    mask = np.ones(n_samples, dtype=bool)
    for v, s in evidence.items():
        mask &= samples[:, bn.index[v]] == s
    accepted = samples[mask, bn.index[query]]
    if len(accepted) == 0:
        return np.full(bn.card[query], 1.0 / bn.card[query]), 0
    return np.bincount(accepted, minlength=bn.card[query]) / len(accepted), int(mask.sum())


def likelihood_weighting(bn, query, evidence, n_samples, rng) -> np.ndarray:
    samples, w = bn.sample(n_samples, rng, evidence=evidence)
    est = np.bincount(samples[:, bn.index[query]], weights=w, minlength=bn.card[query])
    return est / est.sum()


def gibbs_sampling(bn, query, evidence, n_samples, rng, burn_in: int = 200, n_chains: int = 20) -> np.ndarray:
    """Gibbs sampling with ``n_chains`` parallel chains initialized by likelihood weighting.

    Each non-evidence variable is resampled from P(X | Markov blanket), which is proportional
    to P(X | pa(X)) * prod over children C of P(C | pa(C)).
    """
    hidden = [v for v in bn.variables if v not in evidence]
    state, _ = bn.sample(n_chains, rng, evidence=evidence)
    steps = burn_in + int(np.ceil(n_samples / n_chains))
    counts = np.zeros(bn.card[query])
    qi = bn.index[query]
    for t in range(steps):
        for v in hidden:
            vi, k = bn.index[v], bn.card[v]
            logp = np.zeros((n_chains, k))
            for s in range(k):
                state[:, vi] = s
                for u in [v, *bn.dag.children(v)]:
                    idx = tuple(state[:, bn.index[p]] for p in bn.dag.parents(u)) + (state[:, bn.index[u]],)
                    logp[:, s] += np.log(bn.cpts[u][idx])
            p = np.exp(logp - logp.max(axis=1, keepdims=True))
            state[:, vi] = sample_categorical(p / p.sum(axis=1, keepdims=True), rng)
        if t >= burn_in:
            counts += np.bincount(state[:, qi], minlength=bn.card[query])
    return counts / counts.sum()


def predict_proba(bn: BayesianNetwork, data: np.ndarray, target: str) -> np.ndarray:
    """P(target | observed features) for every row; missing entries (-1) are not evidence.

    Rows with identical evidence share one inference call, which matters because binned
    features repeat a lot.
    """
    ti = bn.index[target]
    feature_cols = [i for i in range(data.shape[1]) if i != ti]
    cache: dict[tuple, np.ndarray] = {}
    out = np.empty((len(data), bn.card[target]))
    for r, row in enumerate(data):
        key = tuple(row[feature_cols])
        if key not in cache:
            ev = {bn.variables[i]: int(row[i]) for i in feature_cols if row[i] >= 0}
            cache[key] = variable_elimination(bn, target, ev)
        out[r] = cache[key]
    return out

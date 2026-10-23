"""Discrete Bayesian network: CPT estimation, likelihood and ancestral sampling.

Data are integer-coded arrays with one column per variable and ``-1`` marking a missing
value. Parameters are estimated per family from the rows where that family is fully observed
(available-case analysis), so no imputation is needed for training, and at prediction time a
missing feature is simply left out of the evidence and marginalized by inference.
"""

from __future__ import annotations

import numpy as np

from .factor import Factor
from .graph import DAG


def family_counts(data: np.ndarray, cols: list[int], cards: list[int]) -> np.ndarray:
    """Contingency table over ``cols`` from rows where all of them are observed."""
    sub = data[:, cols]
    sub = sub[(sub >= 0).all(axis=1)]
    flat = np.ravel_multi_index(sub.T, cards) if len(cols) else np.zeros(len(sub), dtype=int)
    return np.bincount(flat, minlength=int(np.prod(cards))).reshape(cards).astype(float)


class BayesianNetwork:
    def __init__(self, dag: DAG, states: dict[str, list]):
        self.dag = dag
        self.states = states
        self.variables = list(dag.nodes)
        self.card = {v: len(states[v]) for v in self.variables}
        self.index = {v: i for i, v in enumerate(self.variables)}
        self.cpts: dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------- learning
    def fit(self, data: np.ndarray, alpha: float = 1.0) -> "BayesianNetwork":
        """Maximum a posteriori CPTs with a symmetric Dirichlet(alpha) prior (Laplace smoothing)."""
        for v in self.variables:
            pa = self.dag.parents(v)
            cols = [self.index[p] for p in pa] + [self.index[v]]
            counts = family_counts(data, cols, [self.card[p] for p in pa] + [self.card[v]]) + alpha
            self.cpts[v] = counts / counts.sum(axis=-1, keepdims=True)
        return self

    def factor(self, v: str) -> Factor:
        return Factor(self.dag.parents(v) + [v], self.cpts[v])

    def factors(self) -> list[Factor]:
        return [self.factor(v) for v in self.variables]

    def n_parameters(self) -> int:
        """Free parameters: (r - 1) per parent configuration of each variable."""
        return int(sum(self.cpts[v][..., 0].size * (self.card[v] - 1) for v in self.variables))

    def log_likelihood(self, data: np.ndarray) -> float:
        """Sum over variables of log P(x_v | pa_v) on rows where the family is observed."""
        ll = 0.0
        for v in self.variables:
            cols = [self.index[p] for p in self.dag.parents(v)] + [self.index[v]]
            sub = data[:, cols]
            sub = sub[(sub >= 0).all(axis=1)]
            ll += float(np.log(self.cpts[v][tuple(sub.T)]).sum())
        return ll

    # ---------------------------------------------------------------- sampling
    def sample(self, n: int, rng: np.random.Generator, evidence: dict[str, int] | None = None):
        """Ancestral (forward) sampling. With ``evidence`` the evidence nodes are clamped and
        the returned weights are the likelihood-weighting weights prod P(e | pa(e))."""
        evidence = evidence or {}
        out = np.empty((n, len(self.variables)), dtype=np.int64)
        weights = np.ones(n)
        for v in self.dag.topological_order():
            pa_idx = tuple(out[:, self.index[p]] for p in self.dag.parents(v))
            probs = self.cpts[v][pa_idx] if pa_idx else np.broadcast_to(self.cpts[v], (n, self.card[v]))
            if v in evidence:
                out[:, self.index[v]] = evidence[v]
                weights *= probs[:, evidence[v]]
            else:
                out[:, self.index[v]] = sample_categorical(probs, rng)
        return out, weights


def sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One draw per row of a (n, k) probability matrix, vectorized via inverse CDF."""
    cdf = np.cumsum(probs, axis=1)
    u = rng.random(len(probs))[:, None] * cdf[:, -1:]
    return np.minimum((u > cdf).sum(axis=1), probs.shape[1] - 1)

"""Discrete factors (potential tables) with the three operations exact inference needs."""

from __future__ import annotations

import numpy as np


class Factor:
    """phi(variables) stored as an ndarray with one axis per variable, in ``variables`` order."""

    def __init__(self, variables: list[str], values: np.ndarray):
        values = np.asarray(values, dtype=float)
        if values.ndim != len(variables):
            raise ValueError("one axis per variable is required")
        self.variables = list(variables)
        self.values = values

    @property
    def cards(self) -> dict[str, int]:
        return dict(zip(self.variables, self.values.shape))

    def _aligned(self, variables: list[str]) -> np.ndarray:
        """View of the values broadcastable over ``variables`` (a superset of ours)."""
        perm = [self.variables.index(v) for v in variables if v in self.variables]
        arr = np.transpose(self.values, perm)
        shape = [self.cards[v] if v in self.variables else 1 for v in variables]
        return arr.reshape(shape)

    def product(self, other: "Factor") -> "Factor":
        variables = self.variables + [v for v in other.variables if v not in self.variables]
        return Factor(variables, self._aligned(variables) * other._aligned(variables))

    def marginalize(self, variable: str) -> "Factor":
        axis = self.variables.index(variable)
        return Factor([v for v in self.variables if v != variable], self.values.sum(axis=axis))

    def reduce(self, evidence: dict[str, int]) -> "Factor":
        index, keep = [], []
        for v in self.variables:
            if v in evidence:
                index.append(evidence[v])
            else:
                index.append(slice(None))
                keep.append(v)
        return Factor(keep, self.values[tuple(index)])

    def normalize(self) -> "Factor":
        return Factor(self.variables, self.values / self.values.sum())

    def __repr__(self) -> str:
        return f"Factor({self.variables}, shape={self.values.shape})"

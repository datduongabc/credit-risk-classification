"""Minimal directed acyclic graph with cycle checking and Kahn's topological sort."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable


class DAG:
    def __init__(self, nodes: Iterable[str], edges: Iterable[tuple[str, str]] = ()):
        self.nodes: list[str] = list(nodes)
        self._parents: dict[str, list[str]] = {n: [] for n in self.nodes}
        self._children: dict[str, list[str]] = {n: [] for n in self.nodes}
        for u, v in edges:
            self.add_edge(u, v)

    def parents(self, node: str) -> list[str]:
        return list(self._parents[node])

    def children(self, node: str) -> list[str]:
        return list(self._children[node])

    @property
    def edges(self) -> list[tuple[str, str]]:
        return [(p, c) for c in self.nodes for p in self._parents[c]]

    def has_edge(self, u: str, v: str) -> bool:
        return u in self._parents[v]

    def has_path(self, src: str, dst: str) -> bool:
        stack, seen = [src], {src}
        while stack:
            n = stack.pop()
            if n == dst:
                return True
            for c in self._children[n]:
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return False

    def add_edge(self, u: str, v: str) -> None:
        if u == v or self.has_edge(u, v):
            raise ValueError(f"invalid edge {u} -> {v}")
        if self.has_path(v, u):
            raise ValueError(f"edge {u} -> {v} would create a cycle")
        self._parents[v].append(u)
        self._children[u].append(v)

    def remove_edge(self, u: str, v: str) -> None:
        self._parents[v].remove(u)
        self._children[u].remove(v)

    def copy(self) -> "DAG":
        return DAG(self.nodes, self.edges)

    def topological_order(self) -> list[str]:
        indeg = {n: len(self._parents[n]) for n in self.nodes}
        queue = deque(n for n in self.nodes if indeg[n] == 0)
        order = []
        while queue:
            n = queue.popleft()
            order.append(n)
            for c in self._children[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        if len(order) != len(self.nodes):
            raise ValueError("graph has a cycle")
        return order

    def markov_blanket(self, node: str) -> set[str]:
        mb = set(self._parents[node]) | set(self._children[node])
        for c in self._children[node]:
            mb |= set(self._parents[c])
        mb.discard(node)
        return mb

    def to_networkx(self):
        import networkx as nx

        g = nx.DiGraph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from(self.edges)
        return g

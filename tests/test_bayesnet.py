import numpy as np
import pytest
from sklearn.naive_bayes import CategoricalNB

from creditrisk.bayesnet.graph import DAG
from creditrisk.bayesnet.inference import (
    enumeration,
    gibbs_sampling,
    likelihood_weighting,
    predict_proba,
    rejection_sampling,
    variable_elimination,
)
from creditrisk.bayesnet.network import BayesianNetwork
from creditrisk.bayesnet.structure import hill_climb_dag, naive_bayes_dag, tan_dag


def random_network(seed=0):
    """Small multi-parent network with random CPTs (a 'sprinkler'-like shape)."""
    rng = np.random.default_rng(seed)
    dag = DAG(["A", "B", "C", "D", "E"], [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E"), ("A", "E")])
    states = {"A": [0, 1], "B": [0, 1, 2], "C": [0, 1], "D": [0, 1, 2], "E": [0, 1]}
    bn = BayesianNetwork(dag, states)
    for v in dag.nodes:
        shape = [bn.card[p] for p in dag.parents(v)] + [bn.card[v]]
        raw = rng.random(shape) + 0.1
        bn.cpts[v] = raw / raw.sum(axis=-1, keepdims=True)
    return bn


def test_dag_rejects_cycles_and_sorts_topologically():
    dag = DAG(["a", "b", "c"], [("a", "b"), ("b", "c")])
    with pytest.raises(ValueError):
        dag.add_edge("c", "a")
    order = dag.topological_order()
    assert order.index("a") < order.index("b") < order.index("c")
    assert dag.markov_blanket("b") == {"a", "c"}


@pytest.mark.parametrize(
    "query,evidence",
    [("A", {}), ("A", {"E": 1}), ("D", {"A": 0, "E": 1}), ("B", {"C": 1, "E": 0}), ("E", {"B": 2})],
)
def test_variable_elimination_matches_enumeration(query, evidence):
    bn = random_network()
    np.testing.assert_allclose(variable_elimination(bn, query, evidence), enumeration(bn, query, evidence), atol=1e-12)


def test_sampling_methods_converge_to_exact():
    bn = random_network(1)
    rng = np.random.default_rng(0)
    query, evidence = "A", {"D": 2, "E": 1}
    exact = variable_elimination(bn, query, evidence)
    lw = likelihood_weighting(bn, query, evidence, 200_000, rng)
    rs, accepted = rejection_sampling(bn, query, evidence, 200_000, rng)
    gibbs = gibbs_sampling(bn, query, evidence, 100_000, rng, burn_in=100, n_chains=50)
    assert accepted > 1000
    for est in (lw, rs, gibbs):
        np.testing.assert_allclose(est, exact, atol=0.01)


def test_fit_recovers_generating_cpts():
    bn = random_network(2)
    data, _ = bn.sample(200_000, np.random.default_rng(0))
    learned = BayesianNetwork(bn.dag, {v: list(range(bn.card[v])) for v in bn.variables}).fit(data, alpha=1)
    for v in bn.variables:
        np.testing.assert_allclose(learned.cpts[v], bn.cpts[v], atol=0.02)


def test_naive_bayes_structure_equals_sklearn_categorical_nb():
    rng = np.random.default_rng(3)
    X = rng.integers(0, 4, size=(2000, 3))
    y = (X[:, 0] + rng.integers(0, 3, 2000) > 3).astype(int)
    data = np.column_stack([X, y])
    variables = ["x0", "x1", "x2", "y"]
    bn = BayesianNetwork(naive_bayes_dag(variables, "y"), {v: list(range(4)) for v in variables[:3]} | {"y": [0, 1]})
    bn.fit(data, alpha=1.0)
    ours = predict_proba(bn, np.column_stack([X, np.full(len(X), -1)]), "y")
    ref = CategoricalNB(alpha=1.0).fit(X, y)
    # CategoricalNB smooths the class prior differently (no pseudocount), so compare with its
    # likelihood but our prior: use log-probabilities difference across classes.
    ref_ll = ref.predict_joint_log_proba(X) - ref.class_log_prior_
    prior = np.log(bn.cpts["y"])
    ref_post = np.exp(ref_ll + prior)
    ref_post /= ref_post.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(ours, ref_post, atol=1e-10)


def test_missing_evidence_is_marginalized():
    bn = random_network(4)
    row = np.array([[-1, 1, -1, 2, 0]])  # A unknown is the query; C unknown is marginalized
    np.testing.assert_allclose(predict_proba(bn, row, "A")[0], variable_elimination(bn, "A", {"B": 1, "D": 2, "E": 0}))


def test_structure_learners_produce_valid_dags():
    bn = random_network(5)
    data, _ = bn.sample(20_000, np.random.default_rng(1))
    cards = bn.card
    tan = tan_dag(data, bn.variables, "A", cards)
    tan.topological_order()
    # TAN: class has no parents, each feature has the class plus at most one feature parent.
    assert tan.parents("A") == []
    assert all("A" in tan.parents(v) and len(tan.parents(v)) <= 2 for v in bn.variables if v != "A")
    hc, trace = hill_climb_dag(data, bn.variables, cards, max_parents=2)
    hc.topological_order()
    assert all(b >= a for a, b in zip(trace, trace[1:]))
    assert all(len(hc.parents(v)) <= 2 for v in bn.variables)

from typing import Dict, Set, Type

import pytest

from src.construction.random_scm import generate_random_scm
from src.symbolic import (
    Decomposability,
    Distribution,
    GaussianDistribution,
    Smoothness,
    SymbolicArithmeticCircuit,
    UniformDistribution,
)


@pytest.fixture
def input_dists_factory():
    """Factory for creating a dictionary of input distributions for a given scope."""

    def _create(scope: Set[int], dist_type: str = "gaussian") -> Dict[int, Distribution]:
        dists = {}
        for var in scope:
            if dist_type == "gaussian":
                dists[var] = GaussianDistribution(var, loc=0.0, scale=1.0)
            elif dist_type == "uniform":
                dists[var] = UniformDistribution(var, low=-1.0, high=1.0)
            else:
                raise ValueError(f"Unknown dist_type: {dist_type}")
        return dists

    return _create


@pytest.fixture
def synthetic_data():
    """Generates synthetic data from a random SCM."""

    def _generate(n_vars: int, n_samples: int, expected_degree: float = 2.0):
        scm = generate_random_scm(n_vars, expected_degree)
        df = scm.sample(n_samples)
        # Convert to numpy array for builders that expect it
        return df.to_numpy()

    return _generate


def assert_alternating_structure(graph, start_node_id: int, layer1_type: Type, layer2_type: Type):
    """
    Verifies that the graph follows an alternating structure:
    layer1_type -> layer2_type -> layer1_type ...
    """
    visited = set()
    queue = [(start_node_id, layer1_type)]

    while queue:
        node_id, expected_type = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)

        node_data = graph.get_node_data(node_id)
        assert isinstance(node_data, expected_type), (
            f"Node {node_id} is {type(node_data)}, expected {expected_type}"
        )

        next_type = layer2_type if expected_type == layer1_type else layer1_type

        # We assume leaf nodes of the graph (not necessarily the circuit)
        # are always of one of these types or mark the end of alternation.
        children = graph.get_children(node_id)
        for child_id in children:
            queue.append((child_id, next_type))

from typing import Callable, Dict

import numpy as np

from src.symbolic.scm import AdditiveNoiseMechanism, StructuralCausalModel


class LinearLogic:
    def __init__(self, coefficients: Dict[str, float]):
        self.coefficients = coefficients

    def __call__(self, **kwargs) -> np.ndarray:
        first_val = next(iter(kwargs.values()))
        res = np.zeros_like(first_val, dtype=np.float64)
        for k, v in kwargs.items():
            res += self.coefficients[k] * v
        return res

    def __str__(self):
        terms = [f"{v:.2f}*{k}" for k, v in self.coefficients.items()]
        return " + ".join(terms) if terms else "0"


class GaussianNoise:
    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def __call__(self, n_samples: int) -> np.ndarray:
        return np.random.normal(self.loc, self.scale, size=n_samples)

    def __str__(self):
        return f"N({self.loc:.2f}, {self.scale:.2f})"


class UniformNoise:
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def __call__(self, n_samples: int) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=n_samples)

    def __str__(self):
        return f"U({self.low:.2f}, {self.high:.2f})"


def generate_random_scm(n_nodes: int, expected_degree: float = 2.0) -> StructuralCausalModel:
    """
    Generate a random Structural Causal Model (DAG) of a given size and sparsity.
    Relationships are restricted to polynomial exponent <= 1 (i.e. linear combinations)
    to prevent numerical explosions in large graphs.

    Args:
        n_nodes: Number of nodes in the generated SCM.
        expected_degree: The expected degree (in-degree + out-degree) of each node.

    Returns:
        A StructuralCausalModel with randomly initialized variables and mechanisms.
    """
    scm = StructuralCausalModel()

    # Calculate probability of an edge.
    # Total expected edges = n_nodes * expected_degree / 2
    # Total possible edges in a DAG (i < j) = n_nodes * (n_nodes - 1) / 2
    p = expected_degree / (n_nodes - 1) if n_nodes > 1 else 0
    p = min(max(p, 0), 1)  # Bound probability between 0 and 1

    # Generate upper triangular adjacency matrix for the DAG
    # (True means an edge exists from row -> col)
    adj_matrix = np.triu(np.random.rand(n_nodes, n_nodes) < p, k=1)

    node_names = [f"X_{i}" for i in range(n_nodes)]

    for i in range(n_nodes):
        name = node_names[i]

        # Parent indices correspond to the True values in column i
        parent_indices = np.where(adj_matrix[:, i])[0]
        parents = [node_names[p_idx] for p_idx in parent_indices]

        # 1. Deterministic Logic (Linear combination, exponents <= 1)
        # We assign random coefficients between -1.0 and 1.0, but we also scale them
        # based on the number of parents to avoid output explosions as depth increases
        scaling_factor = 1.0 / max(1.0, np.sqrt(len(parents)))
        coeffs = {p_name: np.random.uniform(-1.0, 1.0) * scaling_factor for p_name in parents}

        # Logic closure is passed only if there are parents
        logic = LinearLogic(coeffs) if parents else None

        # 2. Additive Noise (50% Gaussian, 50% Uniform mixing)
        noise_type = np.random.choice(["gaussian", "uniform"])

        if noise_type == "gaussian":
            loc = float(np.random.uniform(-0.5, 0.5))
            scale = float(np.random.uniform(0.1, 1.0))
            noise_dist = GaussianNoise(loc, scale)
        else:
            low = float(np.random.uniform(-1.0, -0.1))
            high = float(np.random.uniform(0.1, 1.0))
            noise_dist = UniformNoise(low, high)

        # Add to the graph
        mechanism = AdditiveNoiseMechanism(logic=logic, noise_dist=noise_dist)
        scm.add_variable(name, mechanism=mechanism, parents=parents)

    return scm




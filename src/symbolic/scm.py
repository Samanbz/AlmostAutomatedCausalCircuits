from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd

from .base import DirectedAcyclicGraph


class Mechanism(ABC):
    """Abstract base class for data generation logic."""

    @abstractmethod
    def __call__(self, n_samples: int, **parents: np.ndarray) -> np.ndarray:
        pass


class AdditiveNoiseMechanism(Mechanism):
    """
    Y = f(Parents) + Noise
    """

    def __init__(
        self, logic: Optional[Callable[..., np.ndarray]], noise_dist: Callable[[int], np.ndarray]
    ):
        self.logic = logic
        self.noise_dist = noise_dist

    def __call__(self, n_samples: int, **parents: np.ndarray) -> np.ndarray:
        # Generate noise (Ensure size=n_samples is handled by the generic utils or passed callable)
        noise = self.noise_dist(n_samples)

        if self.logic is None:
            return noise

        try:
            deterministic = self.logic(**parents)
        except TypeError as e:
            raise TypeError(f"Mechanism arguments mismatch. {e}") from e

        return deterministic + noise


class StructuralCausalModel(DirectedAcyclicGraph[str, Mechanism, Any]):
    """
    SCM specific implementation of a DAG.
    Nodes hold Mechanisms. Edges are implicitly causal links (Any data).
    """

    def add_variable(self, name: str, mechanism: Mechanism, parents: List[str] = None):
        """
        High-level wrapper to add a node and its incoming edges.
        """
        self.add_node(name, mechanism)
        if parents:
            for parent in parents:
                self.add_edge(source=parent, target=name)

    def sample(self, n_samples: int) -> pd.DataFrame:
        """
        Orchestrates the feed-forward sampling process.
        """
        data = {}

        # Use the generic DAG's topological sort
        for node_id in self.topological_sort():
            # 1. Get Parent Data
            parent_ids = self.get_parents(node_id)
            parent_data = {pid: data[pid] for pid in parent_ids}

            # 2. Get Mechanism (Node Payload)
            mechanism = self.get_node_data(node_id)

            # 3. Execute
            data[node_id] = mechanism(n_samples=n_samples, **parent_data)

        return pd.DataFrame(data, index=range(n_samples))

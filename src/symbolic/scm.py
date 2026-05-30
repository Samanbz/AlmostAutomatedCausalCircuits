from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd

from .base import DirectedAcyclicGraph


class Mechanism(ABC):
    """Abstract base class for data generation logic."""

    @abstractmethod
    def __call__(self, n_samples: int, **parents: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, noise: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        """Evaluate the mechanism with a given exogenous noise."""
        pass

    @abstractmethod
    def abduct(self, value: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        """Infer the exogenous noise given the output value and parent values."""
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

    def evaluate(self, noise: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        if self.logic is None:
            return noise

        try:
            deterministic = self.logic(**parents)
        except TypeError as e:
            raise TypeError(f"Mechanism arguments mismatch. {e}") from e

        return deterministic + noise

    def abduct(self, value: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        if self.logic is None:
            return value

        try:
            deterministic = self.logic(**parents)
        except TypeError as e:
            raise TypeError(f"Mechanism arguments mismatch. {e}") from e

        return value - deterministic

    def __str__(self) -> str:
        logic_str = str(self.logic) if self.logic else "0"
        noise_str = str(self.noise_dist) if self.noise_dist else "Noise"
        return f"{logic_str} + {noise_str}"


class ConstantMechanism(Mechanism):
    """
    Mechanism that ignores parents and returns a constant array.
    """

    def __init__(self, value: float):
        self.value = value

    def __call__(self, n_samples: int, **parents: np.ndarray) -> np.ndarray:
        return np.full(n_samples, self.value, dtype=np.float64)

    def evaluate(self, noise: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        return np.full(len(noise), self.value, dtype=np.float64)

    def abduct(self, value: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        return np.zeros_like(value)

    def __str__(self) -> str:
        return str(self.value)


class StructuralCausalModel(DirectedAcyclicGraph[Union[str, int], Mechanism, Any]):
    """
    SCM specific implementation of a DAG.
    Nodes hold Mechanisms. Edges are implicitly causal links (Any data).
    """

    def __init__(self):
        super().__init__()
        self.exogenous_variables = set()
        self.observable_variables = set()

    def add_variable(
        self,
        name: Union[str, int],
        mechanism: Mechanism,
        parents: List[Union[str, int]] = None,
        is_exogenous: bool = False,
    ):
        """
        High-level wrapper to add a node and its incoming edges.
        """
        self._add_node_explicit(name, mechanism)
        if is_exogenous:
            self.exogenous_variables.add(name)
        else:
            self.observable_variables.add(name)

        if parents:
            for parent in parents:
                self.add_edge(source=parent, target=name)

    def sample(self, n_samples: int, given_noises: dict = None) -> pd.DataFrame:
        """
        Orchestrates the feed-forward sampling process.
        """
        data = {}
        given_noises = given_noises or {}

        # Use the generic DAG's topological sort
        for node_id in self.topological_sort():
            # 1. Get Parent Data
            parent_ids = self.get_parents(node_id)
            parent_data = {pid: data[pid] for pid in parent_ids}

            # 2. Get Mechanism (Node Payload)
            mechanism = self.get_node_data(node_id)

            # 3. Execute
            if node_id in given_noises:
                data[node_id] = mechanism.evaluate(noise=given_noises[node_id], **parent_data)
            else:
                data[node_id] = mechanism(n_samples=n_samples, **parent_data)

        return pd.DataFrame(data, index=range(n_samples))

    def abduct(self, factual_data: pd.DataFrame) -> dict:
        """
        Calculates the exact exogenous noises U that generated the factual_data.
        """
        abducted_noises = {}

        for node_id in self.topological_sort():
            mechanism = self.get_node_data(node_id)
            parent_ids = self.get_parents(node_id)
            parent_data = {pid: factual_data[pid].values for pid in parent_ids}

            if node_id in factual_data:
                abducted_noises[node_id] = mechanism.abduct(
                    value=factual_data[node_id].values, **parent_data
                )
            else:
                # If a node is missing in factual data, we can't abduct it directly,
                # but let's assume fully observed factual data for now.
                pass

        return abducted_noises

    def intervene(self, interventions: dict) -> "StructuralCausalModel":
        """
        Returns a new SCM under the do-operator interventions.
        """
        new_scm = StructuralCausalModel()

        for node_id in self.topological_sort():
            is_exo = node_id in self.exogenous_variables
            if node_id in interventions:
                new_scm.add_variable(
                    name=node_id,
                    mechanism=ConstantMechanism(interventions[node_id]),
                    parents=[],
                    is_exogenous=is_exo,
                )
            else:
                parents = self.get_parents(node_id)
                mechanism = self.get_node_data(node_id)
                new_scm.add_variable(
                    name=node_id, mechanism=mechanism, parents=parents, is_exogenous=is_exo
                )
        return new_scm

    def get_ancestors(self, nodes: set) -> set:
        """Returns the ancestors of the given nodes using directed edges (includes the nodes themselves)."""
        ancestors = set(nodes)
        queue = list(nodes)
        while queue:
            node = queue.pop(0)
            for parent in self.get_parents(node):
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.append(parent)
        return ancestors

    def get_c_components(self, nodes: set) -> List[set]:
        """
        Returns a list of sets of nodes, representing connected components of the given observable nodes,
        where two observable nodes are connected if they share an exogenous parent.
        """
        components = []
        unvisited = set(nodes)

        # Build adjacency list
        adj = {n: set() for n in nodes}
        for exo in self.exogenous_variables:
            children = set(self.get_children(exo))
            intersected = list(children.intersection(nodes))
            for i in range(len(intersected)):
                for j in range(i + 1, len(intersected)):
                    u, v = intersected[i], intersected[j]
                    adj[u].add(v)
                    adj[v].add(u)

        while unvisited:
            start = unvisited.pop()
            comp = {start}
            queue = [start]
            while queue:
                curr = queue.pop(0)
                for neighbor in adj[curr]:
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        comp.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)
        return components

    def remove_incoming_edges(self, target_nodes: set) -> "StructuralCausalModel":
        """Returns a new SCM with all directed edges pointing into target_nodes removed (computes G_X)."""
        new_scm = StructuralCausalModel()
        for node_id in self.topological_sort():
            mechanism = self.get_node_data(node_id)
            is_exo = node_id in self.exogenous_variables

            parents = self.get_parents(node_id)
            if node_id in target_nodes:
                parents = []

            new_scm.add_variable(
                name=node_id, mechanism=mechanism, parents=parents, is_exogenous=is_exo
            )
        return new_scm

    def subgraph(self, nodes: set) -> "StructuralCausalModel":
        """Returns a new SCM containing only the specified nodes (and their relevant exogenous parents)."""
        new_scm = StructuralCausalModel()

        exo_parents = set()
        for node in nodes:
            for parent in self.get_parents(node):
                if parent in self.exogenous_variables:
                    exo_parents.add(parent)

        nodes_to_include = nodes.union(exo_parents)

        for node_id in self.topological_sort():
            if node_id in nodes_to_include:
                mechanism = self.get_node_data(node_id)
                is_exo = node_id in self.exogenous_variables

                parents = [p for p in self.get_parents(node_id) if p in nodes_to_include]
                new_scm.add_variable(
                    name=node_id, mechanism=mechanism, parents=parents, is_exogenous=is_exo
                )
        return new_scm

    def empirical_log_density(
        self, query: dict, evidence: dict = None, n_samples: int = 20000000, eps: float = 0.05
    ) -> float:
        """
        Numerically estimate log P(query | evidence) using eps-bin counting.
        query: {var_name: value}
        evidence: {var_name: value}
        """
        df = self.sample(n_samples)

        def get_mask(conditions):
            if not conditions:
                return np.ones(n_samples, dtype=bool)
            mask = np.ones(n_samples, dtype=bool)
            for k, v in conditions.items():
                k_str = str(k) if str(k) in df.columns else k
                k_int = (
                    int(k) if isinstance(k, str) and k.isdigit() and int(k) in df.columns else k_str
                )
                col = k_int if k_int in df.columns else k_str
                mask &= (df[col] >= v - eps) & (df[col] <= v + eps)
            return mask

        evidence_mask = get_mask(evidence)
        count_evidence = np.sum(evidence_mask)

        if count_evidence == 0:
            return -np.inf

        joint_cond = {**query}
        if evidence:
            joint_cond.update(evidence)

        joint_mask = get_mask(joint_cond)
        count_joint = np.sum(joint_mask)

        if count_joint == 0:
            return -np.inf

        dim_query = len(query)
        density = (count_joint / count_evidence) / ((2 * eps) ** dim_query)

        return float(np.log(density))

    def empirical_counterfactual_log_density(
        self,
        query: dict,
        intervention: dict,
        evidence: dict,
        n_samples: int = 20000000,
        eps: float = 0.05,
    ) -> float:
        """
        Numerically estimate the counterfactual log probability log P(Y_{X=x} = y | E = e)
        using eps-bin counting (rejection sampling for abduction).

        query: {var_name: value} (e.g. {"Y": y}, the counterfactual outcome)
        intervention: {var_name: value} (e.g. {"X": x}, the do-intervention)
        evidence: {var_name: value} (e.g. {"X": x', "Y": y'}, the factual observation)
        """
        # 1. Sample from the factual SCM
        df = self.sample(n_samples)

        def get_mask(df_target, conditions):
            if not conditions:
                return np.ones(len(df_target), dtype=bool)
            mask = np.ones(len(df_target), dtype=bool)
            for k, v in conditions.items():
                col = k if k in df_target.columns else str(k)
                mask &= (df_target[col] >= v - eps) & (df_target[col] <= v + eps)
            return mask

        # Filter factual samples that match the evidence
        evidence_mask = get_mask(df, evidence)
        count_evidence = np.sum(evidence_mask)

        if count_evidence == 0:
            return -np.inf

        # 2. Abduction: Reverse-engineer the noises for the matching factual samples
        filtered_df = df[evidence_mask].reset_index(drop=True)
        abducted_noises = self.abduct(filtered_df)

        # 3. Action & Prediction: Apply intervention and predict using the abducted noises
        scm_intervened = self.intervene(intervention)
        cf_df = scm_intervened.sample(n_samples=count_evidence, given_noises=abducted_noises)

        # 4. Filter counterfactual samples that match the query
        cf_mask = get_mask(cf_df, query)
        count_cf_joint = np.sum(cf_mask)

        if count_cf_joint == 0:
            return -np.inf

        dim_query = len(query)
        density = (count_cf_joint / count_evidence) / ((2 * eps) ** dim_query)

        return float(np.log(density))


class LinearLogic:
    def __init__(self, coefficients: dict[str, float]):
        self.coefficients = coefficients

    def __call__(self, **kwargs) -> np.ndarray:
        if not kwargs:
            return 0.0
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

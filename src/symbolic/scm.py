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

        return self.evaluate(noise, **parents)

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


class BinaryMechanism(Mechanism):
    """
    Y = 1 if U < P(Y=1 | Parents) else 0
    where U ~ Uniform(0, 1)
    """

    def __init__(self, logic: Optional[Callable[..., np.ndarray]] = None, base_p: float = 0.5):
        self.logic = logic
        self.base_p = base_p

    def __call__(self, n_samples: int, **parents: np.ndarray) -> np.ndarray:
        noise = np.random.uniform(0, 1, size=n_samples)
        return self.evaluate(noise, **parents)

    def evaluate(self, noise: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        if self.logic is None:
            probs = np.full(len(noise), self.base_p, dtype=np.float64)
        else:
            probs = self.logic(**parents)
        return (noise < probs).astype(np.float64)

    def abduct(self, value: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Abduction is ill-posed for threshold-based binary SCMs.")

    def __str__(self) -> str:
        if self.logic is None:
            return f"Bernoulli({self.base_p})"
        return f"Bernoulli({self.logic})"


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
    def __init__(self, coefficients: dict[str, float], intercept: float = 0.0):
        self.coefficients = coefficients
        self.intercept = intercept

    def __call__(self, **kwargs) -> np.ndarray:
        if not kwargs:
            return self.intercept
        first_val = next(iter(kwargs.values()))
        res = np.full_like(first_val, self.intercept, dtype=np.float64)
        for k, v in kwargs.items():
            res += self.coefficients[k] * v
        return res

    def __str__(self):
        terms = [f"{v:.2f}*{k}" for k, v in self.coefficients.items()]
        eq = " + ".join(terms) if terms else "0"
        return f"{self.intercept:.2f} + {eq}"


class LogisticLogic:
    def __init__(self, coefficients: dict[str, float], intercept: float = 0.0):
        self.coefficients = coefficients
        self.intercept = intercept

    def __call__(self, **kwargs) -> np.ndarray:
        if not kwargs:
            return np.full(1, 1 / (1 + np.exp(-self.intercept)), dtype=np.float64)

        first_val = next(iter(kwargs.values()))
        logits = np.full_like(first_val, self.intercept, dtype=np.float64)
        for k, v in kwargs.items():
            logits += self.coefficients[k] * v
        return 1 / (1 + np.exp(-logits))

    def __str__(self):
        terms = [f"{v:.2f}*{k}" for k, v in self.coefficients.items()]
        eq = " + ".join(terms) if terms else "0"
        return f"σ({eq} + {self.intercept:.2f})"


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


def build_synthetic_continuous_scm(
    num_confounders: int = 4,
    confounding_strength: float = 1.0,
    seed: Optional[int] = None,
) -> StructuralCausalModel:
    """Builds a strictly linear-Gaussian SCM with random means and bounded variance.

    Coefficients are chosen so that, for typical random seeds, both X and Y (and the
    interventional distribution P(Y|do(X))) lie mostly inside [-20, 20].

    Args:
        num_confounders: Number of Z variables.
        confounding_strength: A knob to control the magnitude of confounding.
            0.0 means no confounding (P(Y|X) == P(Y|do(X))),
            > 0.0 increases the divergence.
        seed: Optional random seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)

    scm = StructuralCausalModel()

    # Z0 is exogenous with a random (non-zero) mean.
    scm.add_variable(
        name="Z0",
        mechanism=AdditiveNoiseMechanism(
            logic=None,
            noise_dist=GaussianNoise(np.random.uniform(-0.5, 0.5), np.random.uniform(0.1, 0.5)),
        ),
        parents=[],
        is_exogenous=True,
    )

    # Branching tree for Z variables with random intercepts.
    for i in range(1, num_confounders):
        parent_i = (i - 1) // 2
        parent_name = f"Z{parent_i}"
        scm.add_variable(
            name=f"Z{i}",
            mechanism=AdditiveNoiseMechanism(
                logic=LinearLogic(
                    {parent_name: np.random.uniform(-0.8, 0.8)},
                    intercept=np.random.uniform(-0.5, 0.5),
                ),
                noise_dist=GaussianNoise(0.0, np.random.uniform(0.1, 0.5)),
            ),
            parents=[parent_name],
            is_exogenous=False,
        )

    # X depends on all Z.  Coefficients are moderate so the marginal stays bounded.
    x_intercept = np.random.uniform(-0.5, 0.5)
    x_logic = {f"Z{i}": np.random.uniform(-1, 1) for i in range(num_confounders)}
    scm.add_variable(
        name="X",
        mechanism=AdditiveNoiseMechanism(
            logic=LinearLogic(x_logic, intercept=x_intercept),
            noise_dist=GaussianNoise(0.0, np.random.uniform(0.1, 0.5)),
        ),
        parents=[f"Z{i}" for i in range(num_confounders)],
        is_exogenous=False,
    )

    # Y depends on X and all Z.  The Z coefficients create the backdoor confounding.
    y_intercept = np.random.uniform(-0.5, 0.5)
    y_logic = {"X": np.random.uniform(-0.5, 0.5)}
    for i in range(num_confounders):
        y_logic[f"Z{i}"] = np.random.uniform(-0.2, 1) * confounding_strength

    scm.add_variable(
        name="Y",
        mechanism=AdditiveNoiseMechanism(
            logic=LinearLogic(y_logic, intercept=y_intercept),
            noise_dist=GaussianNoise(0.0, np.random.uniform(0.1, 0.5)),
        ),
        parents=["X"] + [f"Z{i}" for i in range(num_confounders)],
        is_exogenous=False,
    )

    return scm


def build_synthetic_binary_scm(
    num_confounders: int = 4, confounding_strength: float = 1.0
) -> StructuralCausalModel:
    """Builds a binary SCM with a branching confounder tree.

    Args:
        num_confounders: Number of Z variables.
        confounding_strength: A knob to control the magnitude of confounding.
            0.0 means no confounding (P(Y|X) == P(Y|do(X))),
            > 0.0 increases the divergence.
    """
    scm = StructuralCausalModel()

    # Z0 is strictly exogenous
    scm.add_variable(
        name="Z0",
        mechanism=BinaryMechanism(logic=None, base_p=0.5),
        parents=[],
        is_exogenous=True,
    )

    # Building a branching tree for Z variables (Logistic logic)
    z_vars = ["Z0"]
    for i in range(1, num_confounders):
        parent_i = (i - 1) // 2
        z_var = f"Z{i}"
        z_vars.append(z_var)
        scm.add_variable(
            name=z_var,
            mechanism=BinaryMechanism(logic=LogisticLogic({f"Z{parent_i}": 2.0}, intercept=-1.0)),
            parents=[f"Z{parent_i}"],
            is_exogenous=False,
        )

    # X depends on Z strongly to create high confounding
    x_logic = {f"Z{i}": 2.0 for i in range(num_confounders)}
    scm.add_variable(
        "X",
        mechanism=BinaryMechanism(
            logic=LogisticLogic(coefficients=x_logic, intercept=-float(num_confounders))
        ),
        parents=[f"Z{i}" for i in range(num_confounders)],
        is_exogenous=False,
    )

    # Y is caused by X and Z
    # We want X to positively influence Y (+2.0)
    # We want Z to negatively influence Y (-confounding_strength) to create a backdoor path
    y_coeffs = {"X": 2.0}
    for z in z_vars:
        y_coeffs[z] = -1.0 * confounding_strength

    # Intercept compensates for Z so that the interventional distribution stays relatively stable
    scm.add_variable(
        "Y",
        mechanism=BinaryMechanism(
            logic=LogisticLogic(
                coefficients=y_coeffs, intercept=confounding_strength * (num_confounders / 2.0)
            )
        ),
        parents=["X"] + z_vars,
        is_exogenous=False,
    )

    return scm

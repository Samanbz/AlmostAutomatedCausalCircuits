"""The structural causal model graph: sampling and intervention.

``StructuralCausalModel`` is a DAG whose nodes hold :class:`~.mechanisms.Mechanism`
payloads. It tracks ``hidden_variables`` (endogenous nodes dropped from released
datasets, for semi-Markovian variants) and exposes an exact :meth:`ground_truth`
engine for mechanism families with analytical ground truth.

Graph-topology algorithms (ancestors, districts, d-separation, ...) live in
:class:`src.symbolic.causal_graph.CausalGraph` — obtain one from an SCM via
:func:`src.construction.latent_projection.latent_projection`.
"""

from typing import TYPE_CHECKING, Any, List, Union

import pandas as pd

from ..base import DirectedAcyclicGraph
from .mechanisms import ConstantMechanism, Mechanism, TabularMechanism


if TYPE_CHECKING:
    from .ground_truth import GroundTruth


class StructuralCausalModel(DirectedAcyclicGraph[Union[str, int], Mechanism, Any]):
    """
    SCM specific implementation of a DAG.
    Nodes hold Mechanisms. Edges are implicitly causal links (Any data).
    """

    def __init__(self):
        super().__init__()
        self.exogenous_variables = set()
        self.observable_variables = set()
        self._hidden_variables = set()

    @property
    def hidden_variables(self) -> set:
        """Endogenous variables dropped from released datasets (semi-Markovian variants).

        Resolved via ``__dict__`` so SCMs pickled before this attribute existed still
        unpickle correctly.
        """
        return self.__dict__.setdefault("_hidden_variables", set())

    def add_variable(
        self,
        name: Union[str, int],
        mechanism: Mechanism,
        parents: List[Union[str, int]] = None,
        is_exogenous: bool = False,
        hidden: bool = False,
    ):
        """
        High-level wrapper to add a node and its incoming edges.
        """
        self._add_node_explicit(name, mechanism)
        if is_exogenous:
            self.exogenous_variables.add(name)
        else:
            self.observable_variables.add(name)
        if hidden:
            self.hidden_variables.add(name)

        if parents:
            for parent in parents:
                self.add_edge(source=parent, target=name)

    def validate_clg_structure(self):
        """Enforce the CLG restriction: discrete nodes may only have discrete parents."""
        for node_id in self.topological_sort():
            if isinstance(self.get_node_data(node_id), TabularMechanism):
                for parent in self.get_parents(node_id):
                    if not isinstance(self.get_node_data(parent), TabularMechanism):
                        raise ValueError(
                            f"CLG structure violation: discrete node '{node_id}' has "
                            f"non-discrete parent '{parent}'."
                        )

    def ground_truth(self) -> "GroundTruth":
        """Exact ground-truth engine for this SCM (see ``ground_truth.py``)."""
        from .ground_truth import GroundTruth

        return GroundTruth(self)

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

    def sample_dataset(self, n_samples: int, drop_hidden: bool = True) -> pd.DataFrame:
        """Sample the full SCM; drop hidden-variable columns for release."""
        df = self.sample(n_samples)
        if drop_hidden:
            hidden = [h for h in self.hidden_variables if h in df.columns]
            df = df.drop(columns=hidden)
        return df

    def intervene(self, interventions: dict) -> "StructuralCausalModel":
        """
        Returns a new SCM under the do-operator interventions.
        """
        new_scm = StructuralCausalModel()

        for node_id in self.topological_sort():
            is_exo = node_id in self.exogenous_variables
            hidden = node_id in self.hidden_variables
            if node_id in interventions:
                new_scm.add_variable(
                    name=node_id,
                    mechanism=ConstantMechanism(interventions[node_id]),
                    parents=[],
                    is_exogenous=is_exo,
                    hidden=hidden,
                )
            else:
                parents = self.get_parents(node_id)
                mechanism = self.get_node_data(node_id)
                new_scm.add_variable(
                    name=node_id,
                    mechanism=mechanism,
                    parents=parents,
                    is_exogenous=is_exo,
                    hidden=hidden,
                )
        return new_scm

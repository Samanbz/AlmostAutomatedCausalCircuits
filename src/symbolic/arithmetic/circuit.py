from typing import TYPE_CHECKING, Any, Dict, List, Type

import torch

from src.utils import BitSet

from ..base import DirectedAcyclicGraph
from .nodes import (
    ArithmeticNode,
    HadamardProductNode,
    KroneckerProductNode,
    LeafNode,
    ProductNode,
    SumNode,
    UniversalSumNode,
)


if TYPE_CHECKING:
    from .properties import Property


class SymbolicArithmeticCircuit(DirectedAcyclicGraph[int, ArithmeticNode, Any]):
    """
    A DAG representing a symbolic arithmetic circuit.
    Nodes are identified by integers and contain Node objects (SumNode, KroneckerProductNode, etc.).
    Edges can optionally contain data (e.g., weights for sum nodes) and represent descendence and
    not computational flow.
    """

    def get_nodes_with_scope(self, scope: BitSet, node_type: Type[ArithmeticNode]) -> List[int]:
        """Returns a list of node IDs that have the given scope and are of the specified type."""
        return [
            node_id
            for node_id, node in self._nodes.items()
            if isinstance(node, node_type) and node.scope == scope
        ]  # TODO: improve by using a top-down pruning search

    def unfold(self) -> "SymbolicArithmeticCircuit":
        """
        Unfolds the tensorized circuit into an explicit scalar circuit.
        Returns a new SymbolicArithmeticCircuit where all nodes have unit_count=1.
        Edges carry explicit scalar weights if available.
        """
        import copy

        new_ac = SymbolicArithmeticCircuit()

        # Mapping from original node ID to a list of unfolded node IDs
        unfold_map: Dict[int, List[int]] = {}

        for node_id in self.topological_sort(reverse=True):
            node = self.get_node_data(node_id)
            h = node.unit_count
            unfolded_ids = []

            if isinstance(node, LeafNode):
                for i in range(h):
                    new_node = copy.copy(node)
                    new_node.unit_count = 1
                    if hasattr(node, "unit_supports") and node.unit_supports:
                        new_node.unit_supports = [node.unit_supports[i]]
                    # Do not copy weights as leaf nodes don't have them in the same way sum nodes do
                    new_id = new_ac.add_node(new_node)
                    unfolded_ids.append(new_id)

            elif isinstance(node, KroneckerProductNode):
                children = self.get_children(node_id)
                assert len(children) == 2, "Kronecker product must have exactly two children"
                left_child, right_child = children
                left_unfolded = unfold_map[left_child]
                right_unfolded = unfold_map[right_child]

                for i in range(len(left_unfolded)):
                    for j in range(len(right_unfolded)):
                        new_node = KroneckerProductNode(support=node.support, unit_count=1)
                        new_id = new_ac.add_node(new_node)
                        new_ac.add_edge(new_id, left_unfolded[i])
                        new_ac.add_edge(new_id, right_unfolded[j])
                        unfolded_ids.append(new_id)

            elif isinstance(node, HadamardProductNode):
                children = self.get_children(node_id)
                assert len(children) == 2, "Hadamard product must have exactly two children"
                left_child, right_child = children
                left_unfolded = unfold_map[left_child]
                right_unfolded = unfold_map[right_child]

                assert len(left_unfolded) == len(right_unfolded) == h, (
                    "Hadamard child unit counts must match"
                )

                for i in range(h):
                    new_node = HadamardProductNode(support=node.support, unit_count=1)
                    new_id = new_ac.add_node(new_node)
                    new_ac.add_edge(new_id, left_unfolded[i])
                    new_ac.add_edge(new_id, right_unfolded[i])
                    unfolded_ids.append(new_id)

            elif isinstance(node, UniversalSumNode):
                children = self.get_children(node_id)
                assert len(children) == 1, (
                    "Sum node currently expects exactly one product child layer"
                )
                child_id = children[0]
                child_unfolded = unfold_map[child_id]
                h_in = len(child_unfolded)

                for i in range(h):
                    new_node = UniversalSumNode(
                        support=node.support, unit_count=1, md_set=node.md_set
                    )
                    new_id = new_ac.add_node(new_node)
                    unfolded_ids.append(new_id)

                    for j in range(h_in):
                        weight = node.weights[i, j].item() if hasattr(node, "weights") else None
                        new_ac.add_edge(new_id, child_unfolded[j], data=weight)

            elif isinstance(node, SumNode):
                children = self.get_children(node_id)
                assert len(children) == 1, (
                    "Sum node currently expects exactly one product child layer"
                )
                child_id = children[0]
                child_unfolded = unfold_map[child_id]
                h_in = len(child_unfolded)

                prods_per_sum = h_in // h

                for i in range(h):
                    new_node = SumNode(support=node.support, unit_count=1, md_set=node.md_set)
                    new_id = new_ac.add_node(new_node)
                    unfolded_ids.append(new_id)

                    start_idx = i * prods_per_sum
                    end_idx = start_idx + prods_per_sum
                    for j, child_prod_idx in enumerate(range(start_idx, end_idx)):
                        weight = (
                            node.weights[i, j].item()
                            if hasattr(node, "weights")
                            and node.weights.shape[0] > i
                            and node.weights.shape[1] > j
                            else None
                        )
                        new_ac.add_edge(new_id, child_unfolded[child_prod_idx], data=weight)

            unfold_map[node_id] = unfolded_ids

        return new_ac

    def check_property(self, property: "Property") -> bool:
        """Checks if the given property holds for the specified node."""
        for node_id in self.topological_sort():
            if not property.check(node_id, self):
                return False
        return True

    def add_edge(self, source: int, target: int, data: Any = None) -> None:
        """Adds a directed edge from source to target with optional data."""
        super().add_edge(source, target, data)

    def is_sum_node(self, node_id: int) -> bool:
        return isinstance(self.get_node_data(node_id), SumNode) or isinstance(
            self.get_node_data(node_id), LeafNode
        )

    def is_product_node(self, node_id: int) -> bool:
        return isinstance(self.get_node_data(node_id), ProductNode)

    def set_edge_data(self, source: int, target: int, data: Any) -> None:
        """Sets the data for an existing edge from source to target."""
        if source not in self._adj or target not in self._adj[source]:
            raise KeyError(f"Edge from '{source}' to '{target}' does not exist.")
        self._adj[source][target] = data
        self._rev_adj[target][source] = data

    def get_node_config(
        self, show_node_id: bool = False, show_unit_supports: bool = False
    ) -> Dict[type, Dict[str, Any]]:
        self._show_unit_supports = show_unit_supports
        config = super().get_node_config(show_node_id=show_node_id)
        return config

    def _get_base_node_config(self) -> Dict[type, Dict[str, Any]]:
        config = super()._get_base_node_config()
        show_unit_supports = getattr(self, "_show_unit_supports", False)

        def leaf_label(node: LeafNode) -> str:
            label = "Leaf"
            if hasattr(node, "scope"):
                label += f"\n{sorted(node.scope)}"
            if show_unit_supports:
                if getattr(node, "unit_supports", None) is not None and len(node.unit_supports) > 1:
                    supports_str = "\n".join(str(s) for s in node.unit_supports)
                    label += f"\n{supports_str}"
                elif hasattr(node, "var_support"):
                    # node: Distribution
                    label += f"\n{node.var_support}"
            return label

        def node_label(node: ArithmeticNode) -> str:
            label = ""
            if isinstance(node, SumNode):
                label = "+"
            elif isinstance(node, KroneckerProductNode):
                label = "x_K"
            elif isinstance(node, HadamardProductNode):
                label = "x_H"
            elif isinstance(node, LeafNode):
                label = leaf_label(node)

            if hasattr(node, "unit_count") and node.unit_count > 1:
                label += f"\n(units={node.unit_count})"

            if show_unit_supports and getattr(node, "unit_supports", None) is not None:
                supports_str = "\n".join(str(s) for s in node.unit_supports)
                label += f"\n{supports_str}"

            return label

        config.update(
            {
                SumNode: {"color": "#ff9999", "label": node_label, "shape": "diamond"},
                KroneckerProductNode: {"color": "#9999ff", "label": node_label, "shape": "box"},
                HadamardProductNode: {"color": "#ffcc99", "label": node_label, "shape": "box"},
                LeafNode: {"color": "#99ff99", "label": node_label, "shape": "ellipse"},
            }
        )
        return config

def eval_circuit(ac: SymbolicArithmeticCircuit, data: torch.Tensor) -> torch.Tensor:
    """Evaluates the circuit on the given data."""
    outputs = {}
    for node_id in ac.topological_sort(reverse=True):
        node = ac.get_node_data(node_id)
        child_ids = ac.get_children(node_id)
        child_outs = [outputs[cid] for cid in child_ids]
        outputs[node_id] = node.forward(data, child_outs)

    roots = ac.get_roots()
    if not roots:
        return torch.empty(data.shape[0], 0)
    # Return the first root's output
    return outputs[roots[0]]

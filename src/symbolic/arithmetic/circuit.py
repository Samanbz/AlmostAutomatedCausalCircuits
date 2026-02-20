from typing import TYPE_CHECKING, Any, Dict, List, Type

from src.utils import BitSet

from ..base import DirectedAcyclicGraph
from .nodes import ArithmeticNode, LeafNode, ProductNode, SumNode


if TYPE_CHECKING:
    from .properties import Property


class SymbolicArithmeticCircuit(DirectedAcyclicGraph[int, ArithmeticNode, Any]):
    """
    A DAG representing a symbolic arithmetic circuit.
    Nodes are identified by integers and contain Node objects (SumNode, ProductNode, etc.).
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

    def check_property(self, property: "Property") -> bool:
        """Checks if the given property holds for the specified node."""
        for node_id in self.topological_sort():
            if not property.check(node_id, self):
                return False
        return True

    def get_node_config(self, show_node_id=False) -> Dict[type, Dict[str, Any]]:
        def leaf_label(node: LeafNode) -> str:
            label = "Leaf"
            if hasattr(node, "scope"):
                label += f"\n{sorted(node.scope)}"
            if hasattr(node, "var_support"):
                # node: Distribution
                label += f"\n{node.var_support}"
            return label

        node_ids = {n: k for k, n in self._nodes.items()}

        def node_label(node: ArithmeticNode) -> str:
            label = ""
            if isinstance(node, SumNode):
                label = "+"
            elif isinstance(node, ProductNode):
                label = "x"
            elif isinstance(node, LeafNode):
                label = leaf_label(node)
            if show_node_id:
                label += f"\nID: {node_ids[node]}"

            return label

        return {
            SumNode: {"color": "#ff9999", "label": node_label, "shape": "diamond"},
            ProductNode: {"color": "#9999ff", "label": node_label, "shape": "box"},
            LeafNode: {"color": "#99ff99", "label": node_label, "shape": "ellipse"},
        }

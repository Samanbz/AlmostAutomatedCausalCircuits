from typing import TYPE_CHECKING, Any, Dict, List, Type

from src.graph import DirectedAcyclicGraph
from src.utils import BitSet

from .nodes import ArithmeticNode, LeafNode, ProductNode, SumNode


if TYPE_CHECKING:
    from .properties import Property


class SymbolicArithmeticCircuit(DirectedAcyclicGraph[int, ArithmeticNode, Any]):
    """
    A DAG representing a symbolic arithmetic circuit.
    Nodes are identified by integers and contain Node objects (SumNode, ProductNode, etc.).
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

    @property
    def node_config(self) -> Dict[type, Dict[str, Any]]:
        def leaf_label(node: LeafNode) -> str:
            label = "Leaf"
            if hasattr(node, "scope"):
                label += f"\n{sorted(node.scope)}"
            if hasattr(node, "var_support"):
                # node: Distribution
                label += f"\n{node.var_support}"
            return label

        return {
            SumNode: {"color": "#ff9999", "label": "+", "shape": "diamond"},
            ProductNode: {"color": "#9999ff", "label": "x", "shape": "box"},
            LeafNode: {"color": "#99ff99", "label": leaf_label, "shape": "ellipse"},
        }

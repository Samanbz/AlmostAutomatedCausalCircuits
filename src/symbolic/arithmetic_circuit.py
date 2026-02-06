from typing import Any, Dict, List, Type

from src.graph import DirectedAcyclicGraph, Node
from src.utils import BitSet, Support


class ArithmeticNode(Node):
    """Represents an arithmetic operation node."""

    def __init__(self, scope: BitSet):
        self.scope = scope

    def __repr__(self):
        return f"{self.__class__.__name__}(scope={self.scope})"


class SumNode(ArithmeticNode):
    """Represents a sum operation."""

    pass


class ProductNode(ArithmeticNode):
    """Represents a product operation."""

    pass


class LeafNode(ArithmeticNode):
    """Represents a leaf distribution (e.g., Gaussian) in the SPN."""

    pass


class SymbolicArithmeticCircuit(DirectedAcyclicGraph[int, ArithmeticNode, Any]):
    """
    A DAG representing a symbolic arithmetic circuit.
    Nodes are identified by integers and contain Node objects (SumNode, ProductNode, etc.).
    """

    # TODO: Should we maybe store support at the nodes and assign them top-down like scope?
    def get_support(self, node_id: int) -> Support:
        from .distributions import Distribution  # avoid circular import

        if self.is_leaf(node_id):
            node = self.get_node_data(node_id)
            assert isinstance(node, Distribution), "Leaf nodes must be of type LeafNode"
            return node.support
        children = self.get_children(node_id)
        support = Support()
        for child_id in children:
            child_support = self.get_support(child_id)
            support = support.union(child_support)
        return support

    def get_nodes_with_scope(self, scope: BitSet, node_type: Type[ArithmeticNode]) -> List[int]:
        """Returns a list of node IDs that have the given scope and are of the specified type."""
        return [
            node_id
            for node_id, node in self._nodes.items()
            if isinstance(node, node_type) and node.scope == scope
        ]  # TODO: improve by using a top-down pruning search

    @property
    def node_config(self) -> Dict[type, Dict[str, Any]]:
        def leaf_label(node: LeafNode) -> str:
            label = "Leaf"
            if hasattr(node, "scope"):
                label += f"\n{sorted(node.scope)}"
            if hasattr(node, "support"):
                # node: Distribution
                label += f"\n{node.support}"
            return label

        return {
            SumNode: {"color": "#ff9999", "label": "+", "shape": "diamond"},
            ProductNode: {"color": "#9999ff", "label": "x", "shape": "box"},
            LeafNode: {"color": "#99ff99", "label": leaf_label, "shape": "ellipse"},
        }

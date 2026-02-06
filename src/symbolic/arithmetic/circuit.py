from typing import TYPE_CHECKING, Any, Dict, List, Type

from src.graph import DirectedAcyclicGraph
from src.utils import BitSet, Support

from .distributions import Distribution
from .nodes import ArithmeticNode, LeafNode, ProductNode, SumNode


if TYPE_CHECKING:
    from .properties import Property


class SymbolicArithmeticCircuit(DirectedAcyclicGraph[int, ArithmeticNode, Any]):
    """
    A DAG representing a symbolic arithmetic circuit.
    Nodes are identified by integers and contain Node objects (SumNode, ProductNode, etc.).
    """

    # TODO: Should we maybe store support at the nodes and assign them top-down like scope?
    def get_support(self, node_id: int) -> Support:
        if self.is_leaf(node_id):
            node = self.get_node_data(node_id)
            assert isinstance(node, Distribution), "Leaf nodes must be of type LeafNode"
            # FIXME: assuming distribution scope is a single variable, which we can, but it's not
            # an elegant solution. Should later be refactored so that Distribution also has a
            # support property of type Support, which also includes the scope.
            return Support({node.scope.min(): node.support})
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
            if hasattr(node, "support"):
                label += f"\n{node.support}"
            return label

        return {
            SumNode: {"color": "#ff9999", "label": "+", "shape": "diamond"},
            ProductNode: {"color": "#9999ff", "label": "x", "shape": "box"},
            LeafNode: {"color": "#99ff99", "label": leaf_label, "shape": "ellipse"},
        }

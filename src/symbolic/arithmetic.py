from typing import Any

from src.graph import DirectedAcyclicGraph, Node
from src.utils import BitSet


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

    pass

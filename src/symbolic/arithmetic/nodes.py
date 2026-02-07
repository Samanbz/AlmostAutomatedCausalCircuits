from src.graph import Node
from src.utils import BitSet, Support


class ArithmeticNode(Node):
    """Represents an arithmetic operation node."""

    def __init__(self, support: Support):
        self.support = support

    @property
    def scope(self) -> BitSet:
        return self.support.scope

    def __repr__(self):
        return f"{self.__class__.__name__}(scope={self.support.scope})"


class SumNode(ArithmeticNode):
    """Represents a sum operation."""

    pass


class ProductNode(ArithmeticNode):
    """Represents a product operation."""

    pass


class LeafNode(ArithmeticNode):
    """Represents a leaf distribution (e.g., Gaussian) in the SPN."""

    pass

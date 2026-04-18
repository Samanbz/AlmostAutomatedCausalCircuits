from typing import List

import torch

from src.utils import BitSet, Support

from ..base import Node


class ArithmeticNode(Node):
    """Represents an arithmetic operation node."""

    def __init__(self, support: Support, unit_count: int = 1):
        self.support = support
        self.unit_count = unit_count

    @property
    def scope(self) -> BitSet:
        return self.support.scope

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(scope={self.support.scope}, unit_count={self.unit_count})"
        )


class SumNode(ArithmeticNode):
    """Represents a sum operation."""

    pass


class ProductNode(ArithmeticNode):
    """Represents a product operation."""

    pass


class LeafNode(ArithmeticNode):
    """Represents a leaf distribution (e.g., Gaussian) in the SPN."""

    pass

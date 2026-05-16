from typing import List, Optional

from src.utils import BitSet, Support

from ..base import Node


class ArithmeticNode(Node):
    """Represents an arithmetic operation node."""

    def __init__(
        self,
        support: Support,
        unit_count: int = 1,
        unit_supports: Optional[List[Support]] = None,
        md_set: Optional[BitSet] = None,
    ):
        self.support = support
        self.unit_count = unit_count
        # If unit_supports is not provided, all units share the same support
        self.unit_supports = unit_supports or [support] * unit_count
        # Marginal determinism set for this node's region (None if unknown)
        self.md_set = md_set

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


class UniversalSumNode(SumNode):
    """Represents a dense, unpartitioned sum operation over all children products."""

    pass


class ProductNode(ArithmeticNode):
    """Represents a product operation."""

    pass


class KroneckerProductNode(ProductNode):
    """Represents a product operation (Cartesian/Kronecker)."""

    pass


class HadamardProductNode(ProductNode):
    """Represents an element-wise (Hadamard) product operation."""

    pass


class UnaryProductNode(ArithmeticNode):
    """Represents a unary product operation."""

    pass


class LeafNode(ArithmeticNode):
    """Represents a leaf distribution (e.g., Gaussian) in the SPN."""

    pass


class ConstantLeafNode(LeafNode):
    """Leaf that always returns log(1) = 0 for all inputs.

    Used to represent a marginalized variable in a compiled estimand circuit.
    The variable is still in scope (so the circuit remains smooth/decomposable)
    but contributes nothing to the density.
    """

    def __init__(self, var: int, unit_count: int = 1, md_set: Optional[BitSet] = None):
        from src.utils import ContinuousInterval

        var_support = ContinuousInterval(float("-inf"), float("inf"), False, False)
        super().__init__(
            support=Support({var: var_support}),
            unit_count=unit_count,
            md_set=md_set,
        )
        self.var = var

    def __repr__(self):
        return f"ConstantLeafNode(var={self.var}, units={self.unit_count})"


class InverseLeafNode(LeafNode):
    """Leaf that wraps another leaf and negates its log-density.

    Used for POW(-1) operations: if the base leaf returns log p(x),
    this returns -log p(x) = log p(x)^{-1}.
    """

    def __init__(self, base_leaf: LeafNode, power: int = -1):
        super().__init__(
            support=base_leaf.support,
            unit_count=base_leaf.unit_count,
            unit_supports=base_leaf.unit_supports,
            md_set=base_leaf.md_set,
        )
        self.base_leaf = base_leaf
        self.power = power
        if hasattr(base_leaf, "var"):
            self.var = base_leaf.var

    def __repr__(self):
        return f"InverseLeafNode(base={self.base_leaf}, power={self.power})"


class ProductLeafNode(LeafNode):
    """Leaf that combines two leaves by adding their log-densities.

    Used for DetProd (support-compatible product) at the leaf level:
    log(p_A(x) * p_B(x)) = log p_A(x) + log p_B(x).
    """

    def __init__(self, leaf_a: LeafNode, leaf_b: LeafNode):
        super().__init__(
            support=leaf_a.support,
            unit_count=leaf_a.unit_count,
            unit_supports=leaf_a.unit_supports,
            md_set=leaf_a.md_set,
        )
        self.leaf_a = leaf_a
        self.leaf_b = leaf_b
        if hasattr(leaf_a, "var"):
            self.var = leaf_a.var

    def __repr__(self):
        return f"ProductLeafNode(a={self.leaf_a}, b={self.leaf_b})"

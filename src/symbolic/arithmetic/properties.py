from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Set

from src.logger import logger as g_logger
from src.utils import BitSet, Support

from .nodes import (
    HadamardProductNode,
    KroneckerProductNode,
    LeafNode,
    ProductNode,
    SumNode,
    UnaryProductNode,
    UniversalSumNode,
)


logger = g_logger.getChild(__name__)


if TYPE_CHECKING:
    from .circuit import SymbolicArithmeticCircuit


class Property(ABC):
    """
    Abstract base class for structural and support properties of arithmetic circuits.
    """

    @abstractmethod
    def check(self, node_id: int, circuit: "SymbolicArithmeticCircuit") -> bool:
        """Check if the property holds for the given node."""
        raise NotImplementedError


class Smoothness(Property):
    """
    Represents the property of smoothness for arithmetic circuits.

    A circuit is smooth if for every sum node, all children have the same scope.
    """

    def check(self, node_id: int, circuit: "SymbolicArithmeticCircuit") -> bool:
        node = circuit.get_node_data(node_id)
        if not isinstance(node, SumNode):
            return True

        children = circuit.get_children(node_id)
        return all(circuit.get_node_data(child_id).scope == node.scope for child_id in children)


class Decomposability(Property):
    """
    Represents the property of decomposability for arithmetic circuits.

    A circuit is decomposable if for every product node, the scopes of its children are disjoint.
    """

    def check(self, node_id: int, circuit: "SymbolicArithmeticCircuit") -> bool:
        node = circuit.get_node_data(node_id)
        if not isinstance(node, ProductNode):
            return True

        children = circuit.get_children(node_id)
        child_scopes = [circuit.get_node_data(child_id).scope for child_id in children]
        for i in range(len(child_scopes)):
            for j in range(i + 1, len(child_scopes)):
                if not child_scopes[i].intersection(child_scopes[j]).is_empty:
                    return False
        return True


class Determinism(Property):
    """
    Represents the property of determinism for arithmetic circuits.

    A circuit is deterministic if for every sum node, the supports of its children are disjoint.
    """

    def check(self, node_id: int, circuit: "SymbolicArithmeticCircuit") -> bool:
        node = circuit.get_node_data(node_id)
        if not isinstance(node, SumNode):
            return True

        children = circuit.get_children(node_id)
        child_supports = [circuit.get_node_data(child_id).support for child_id in children]
        for i in range(len(child_supports)):
            for j in range(i + 1, len(child_supports)):
                if not child_supports[i].intersect(child_supports[j]).is_empty:
                    return False
        return True


class StructuredDecomposability(Property):
    """
    Represents the property of structured decomposability for arithmetic circuits.

    A circuit is structured decomposable if it is decomposable and for every pair of product nodes
    with the same scope, their children have the same scopes (i.e., they decompose in the same way).
    """

    def __init__(self):
        self.cache: Dict[BitSet, bool] = {}

    def check(self, node_id: int, circuit: "SymbolicArithmeticCircuit") -> bool:
        node = circuit.get_node_data(node_id)
        if not isinstance(node, ProductNode):
            return True

        if node.scope in self.cache:
            return self.cache[node.scope]

        if not Decomposability().check(node_id, circuit):
            self.cache[node.scope] = False
            return False

        prod_node_ids = circuit.get_nodes_with_scope(node.scope, node_type=ProductNode)
        node_children = circuit.get_children(node_id)

        if len(node_children) <= 1:
            self.cache[node.scope] = True
            return True

        node_child_scopes = [circuit.get_node_data(child_id).scope for child_id in node_children]
        node_child_scopes.sort(key=lambda s: s.min)

        for prod_node_id in prod_node_ids:
            if prod_node_id == node_id:
                continue

            prod_children = circuit.get_children(prod_node_id)
            if len(prod_children) <= 1:
                continue

            prod_child_scopes = [
                circuit.get_node_data(child_id).scope for child_id in prod_children
            ]
            prod_child_scopes.sort(key=lambda s: s.min)

            if node_child_scopes != prod_child_scopes:
                self.cache[node.scope] = False
                return False

        self.cache[node.scope] = True
        return True


class Compatibility(Property):
    # TODO
    pass


class MarginalDeterminism(Property):
    """
    Checks marginal determinism w.r.t. a target scope Q. For every SumNode
    whose scope overlaps Q, all child product units must have pairwise
    disjoint supports on Q.

    Since intermediate unit_supports are not stored on internal nodes for
    efficiency, this property recursively resolves them from the leaves.
    """

    def __init__(self, target_scope: Set):
        self.target_scope = BitSet(target_scope)
        self._support_cache: Dict[int, List[Support]] = {}

    def _get_unit_supports(
        self, node_id: int, circuit: "SymbolicArithmeticCircuit"
    ) -> List[Support]:
        if node_id in self._support_cache:
            return self._support_cache[node_id]

        node = circuit.get_node_data(node_id)
        child_ids = circuit.get_children(node_id)

        if isinstance(node, LeafNode):
            res = node.unit_supports
        elif isinstance(node, UnaryProductNode):
            res = node.unit_supports
        elif isinstance(node, HadamardProductNode):
            assert len(child_ids) == 2
            l_sups = self._get_unit_supports(child_ids[0], circuit)
            r_sups = self._get_unit_supports(child_ids[1], circuit)
            res = [l_sups[i].union(r_sups[i]) for i in range(node.unit_count)]
        elif isinstance(node, KroneckerProductNode):
            assert len(child_ids) == 2
            l_sups = self._get_unit_supports(child_ids[0], circuit)
            r_sups = self._get_unit_supports(child_ids[1], circuit)
            res = [ls.union(rs) for ls in l_sups for rs in r_sups]
            assert len(res) == node.unit_count
        elif isinstance(node, UniversalSumNode):
            overall_support = Support()
            for ch_id in child_ids:
                for sup in self._get_unit_supports(ch_id, circuit):
                    overall_support = overall_support.union(sup)
            res = [overall_support] * node.unit_count
        elif isinstance(node, SumNode):
            res = []
            for unit_id in range(node.unit_count):
                unit_support = Support()
                for ch_id in child_ids:
                    ch_sups = self._get_unit_supports(ch_id, circuit)
                    if not ch_sups:
                        continue
                    pps = len(ch_sups) // node.unit_count
                    if pps == 0:
                        # If child has fewer units than parent, this mapping is ambiguous.
                        # For now, assume element-wise if same or error.
                        pps = 1

                    start, end = unit_id * pps, (unit_id + 1) * pps
                    for i in range(start, min(end, len(ch_sups))):
                        unit_support = unit_support.union(ch_sups[i])
                res.append(unit_support)
        else:
            res = node.unit_supports

        self._support_cache[node_id] = res
        return res

    def check(self, node_id: int, circuit: "SymbolicArithmeticCircuit") -> bool:
        node = circuit.get_node_data(node_id)
        if not isinstance(node, SumNode) or isinstance(node, UniversalSumNode):
            return True

        # If target scope doesn't intersect node scope, it's trivially deterministic
        if node.scope.intersection(self.target_scope).is_empty:
            return True

        child_ids = circuit.get_children(node_id)
        for unit_id in range(node.unit_count):
            unit_child_supports = []
            for ch_id in child_ids:
                ch_sups = self._get_unit_supports(ch_id, circuit)
                if not ch_sups:
                    continue

                pps = len(ch_sups) // node.unit_count
                if pps == 0:
                    pps = 1

                start = unit_id * pps
                end = min(start + pps, len(ch_sups))
                unit_child_supports.extend(ch_sups[start:end])

            for j in range(len(unit_child_supports)):
                for k in range(j + 1, len(unit_child_supports)):
                    l_marg = unit_child_supports[j].filter_by_vars(self.target_scope)
                    r_marg = unit_child_supports[k].filter_by_vars(self.target_scope)
                    overlap = l_marg.intersect(r_marg)
                    if not overlap.is_empty:
                        logger.warning(
                            f"Marginal determinism violated at node {node_id} for unit {unit_id} "
                            f"between child product units {j} and {k} on scope {self.target_scope}."
                        )
                        return False
        return True

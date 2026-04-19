from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Set

from src.utils import BitSet, Support

from .nodes import LeafNode, ProductNode, SumNode


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
        if isinstance(node, (ProductNode, LeafNode)):
            return True  # Smoothness only applies to SumNodes

        children = circuit.get_children(node_id)
        return all(circuit.get_node_data(child_id).scope == node.scope for child_id in children)


class Decomposability(Property):
    """
    Represents the property of decomposability for arithmetic circuits.

    A circuit is decomposable if for every product node, the scopes of its children are disjoint.
    """

    def check(self, node_id: int, circuit: "SymbolicArithmeticCircuit") -> bool:
        node = circuit.get_node_data(node_id)
        if isinstance(node, (SumNode, LeafNode)):
            return True  # Decomposability only applies to ProductNodes

        children = circuit.get_children(node_id)

        # Check if scopes of children are disjoint
        child_scopes = [circuit.get_node_data(child_id).scope for child_id in children]
        for i in range(len(child_scopes)):
            for j in range(i + 1, len(child_scopes)):
                if not child_scopes[i].intersection(child_scopes[j]).is_empty:
                    print(
                        f"Decomposability violated at ProductNode {node_id}. Child {children[i]} scope {child_scopes[i]} intersects with child {children[j]} scope {child_scopes[j]}"
                    )
                    return False
        return True


class Determinism(Property):
    """
    Represents the property of determinism for arithmetic circuits.

    A circuit is deterministic if for every sum node, the supports of its children are disjoint.
    """

    def check(self, node_id: int, circuit: "SymbolicArithmeticCircuit") -> bool:
        node = circuit.get_node_data(node_id)
        if isinstance(node, (ProductNode, LeafNode)):
            return True  # Determinism only applies to SumNodes

        children = circuit.get_children(node_id)

        # Check if supports of children are disjoint
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
        # Caches results for nodes with a certain scope for avoid redundant checks
        # We won't assume that product nodes at the same depth have the same scope
        self.cache: Dict[BitSet, bool] = {}
        self.cache_circuit: SymbolicArithmeticCircuit = None

    def check(self, node_id: int, circuit: "SymbolicArithmeticCircuit") -> bool:
        node = circuit.get_node_data(node_id)
        if isinstance(node, (SumNode, LeafNode)):
            return True  # Structured decomposability only applies to ProductNodes

        # Check if the result is already cached for this node's scope
        if node.scope in self.cache:
            return self.cache[node.scope]

        # First check if the node is decomposable
        if not Decomposability().check(node_id, circuit):
            self.cache[node.scope] = False
            return False

        prod_node_ids = circuit.get_nodes_with_scope(node.scope, node_type=ProductNode)

        node_children = circuit.get_children(node_id)
        node_child_scopes = [circuit.get_node_data(child_id).scope for child_id in node_children]

        # Enforce an ordering on the child scopes. Since node is decomposable, child scopes are
        # disjoint, so this works reliably.
        node_child_scopes.sort(key=lambda s: s.min)
        for prod_node_id in prod_node_ids:
            if prod_node_id == node_id:
                continue  # Skip self

            prod_children = circuit.get_children(prod_node_id)
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
    # Should replace StructuredDecomposability eventually (compatible to self)
    # How do we elegantly check two circuits at the same time?
    # TODO
    pass


class MarginalDeterminism(Property):
    """
    Represents the property of marginal determinism for arithmetic circuits.

    A circuit is marginal deterministic with respect to a subset Q ⊆ V if for every sum node T:
    - The restricted scope φ_Q(T) is empty (i.e., NO overlap between node scope and Q).
    - The sum node T is Q-deterministic (children are functionally disjoint or identical on Q).
    """

    def __init__(self, target_scope: Set):
        self.target_scope = BitSet(target_scope)
        self.sig_cache: Dict[int, frozenset] = {}
        self.supp_cache: Dict[int, Support] = {}

    def _get_marginal_signature(self, node_id: int, circuit: "SymbolicArithmeticCircuit") -> frozenset:
        if node_id in self.sig_cache:
            return self.sig_cache[node_id]

        node = circuit.get_node_data(node_id)
        intersection_scope = node.scope.intersection(self.target_scope)

        if intersection_scope.is_empty:
            res = frozenset()
            self.sig_cache[node_id] = res
            return res

        if isinstance(node, LeafNode):
            res = frozenset([node_id])
            self.sig_cache[node_id] = res
            return res

        children = circuit.get_children(node_id)
        if isinstance(node, ProductNode):
            res = set()
            for child_id in children:
                child_sig = self._get_marginal_signature(child_id, circuit)
                res.update(child_sig)
            res = frozenset(res)
            self.sig_cache[node_id] = res
            return res

        if isinstance(node, SumNode):
            res = set()
            for child_id in children:
                child_sig = self._get_marginal_signature(child_id, circuit)
                res.update(child_sig)
            res = frozenset(res)
            self.sig_cache[node_id] = res
            return res

        raise ValueError(f"Unknown node type: {type(node)}")

    def _get_marginal_support(self, node_id: int, circuit: "SymbolicArithmeticCircuit") -> Support:
        if node_id in self.supp_cache:
            return self.supp_cache[node_id]

        node = circuit.get_node_data(node_id)
        intersection_scope = node.scope.intersection(self.target_scope)

        if intersection_scope.is_empty:
            res = Support()
            self.supp_cache[node_id] = res
            return res

        if isinstance(node, LeafNode):
            res = node.support.filter_by_vars(intersection_scope)
            self.supp_cache[node_id] = res
            return res

        children = circuit.get_children(node_id)
        res = Support()
        for child_id in children:
            child_support = self._get_marginal_support(child_id, circuit)
            res = res.union(child_support)
        self.supp_cache[node_id] = res
        return res

    def check(self, node_id: int, circuit: "SymbolicArithmeticCircuit") -> bool:
        node = circuit.get_node_data(node_id)

        if not isinstance(node, SumNode):
            return True

        intersection_scope = node.scope.intersection(self.target_scope)
        if intersection_scope.is_empty:
            return True

        children = circuit.get_children(node_id)
        child_sigs = []
        child_supports = []

        for child_id in children:
            child_sigs.append(self._get_marginal_signature(child_id, circuit))
            child_supports.append(self._get_marginal_support(child_id, circuit))

        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                sig1 = child_sigs[i]
                sig2 = child_sigs[j]

                # If the children use the EXACT SAME subcircuit for the variables in Q,
                # they are functionally identical on Q, so it's a valid mixing operation over non-Q.
                if len(sig1) > 0 and sig1 == sig2:
                    continue

                # Otherwise, they represent a mixture on Q, so their supports on Q MUST be mutually exclusive.
                s1 = child_supports[i]
                s2 = child_supports[j]

                intersection = s1.intersect(s2)
                if not intersection.is_empty:
                    return False

        return True

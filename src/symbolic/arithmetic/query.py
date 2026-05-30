import copy
from typing import Any, Dict, Set, Tuple

import torch

from src.utils import BitSet

from ..id_ast import (
    CondNode,
    ConstantNode,
    DetProdNode,
    EstimandAST,
    InstNode,
    MargNode,
    PNode,
    PowNode,
    ProdNode,
)
from .circuit import SymbolicArithmeticCircuit
from .nodes import (
    ArithmeticNode,
    CartesianLeafNode,
    ConstantLeafNode,
    HadamardProductNode,
    InverseLeafNode,
    KroneckerProductNode,
    LeafNode,
    ProductLeafNode,
    ProductNode,
    SumNode,
    UniversalSumNode,
)


class CircuitCompilationError(Exception):
    pass


class InstantiatedLeafNode(LeafNode):
    """Placeholder leaf node to represent clamping a variable to a specific value."""

    def __init__(self, base_leaf: LeafNode, value: Any):
        super().__init__(
            support=base_leaf.support,
            unit_count=base_leaf.unit_count,
            unit_supports=base_leaf.unit_supports,
            md_set=base_leaf.md_set,
        )
        self.base_leaf = base_leaf
        self.value = value
        if hasattr(base_leaf, "var"):
            self.var = base_leaf.var

    def forward(self, data: torch.Tensor, children_outputs: list = None) -> torch.Tensor:
        data_clamped = data.clone()
        data_clamped[:, self.var] = self.value
        return self.base_leaf.forward(data_clamped, children_outputs)

    def __repr__(self):
        return f"InstantiatedLeafNode(base={self.base_leaf}, val={self.value})"


def _marginalize(ac: SymbolicArithmeticCircuit, marg_vars: Set[int]) -> SymbolicArithmeticCircuit:
    """
    Returns a new circuit where leaves corresponding to marg_vars are replaced with ConstantLeafNode(1).
    """
    new_ac = SymbolicArithmeticCircuit(node_allocator=ac.node_allocator)
    mapping = {}

    for node_id in ac.topological_sort(reverse=True):
        node = ac.get_node_data(node_id)
        if isinstance(node, LeafNode) and getattr(node, "var", None) in marg_vars:
            new_node = ConstantLeafNode(
                var=node.var, unit_count=node.unit_count, md_set=node.md_set
            )
            new_id = new_ac.add_node(new_node)
        else:
            new_node = copy.copy(node)
            new_id = new_ac.add_node(new_node)
            for child_id, edge_data in ac.get_outgoing_edges(node_id):
                new_ac.add_edge(new_id, mapping[child_id], edge_data)
        mapping[node_id] = new_id

    return new_ac


def _inverse(
    ac: SymbolicArithmeticCircuit,
) -> SymbolicArithmeticCircuit:
    """
    Returns a new circuit representing the reciprocal density.
    Recursively inverts weights and leaves. Only valid for deterministic circuits.
    """
    new_ac = SymbolicArithmeticCircuit(node_allocator=ac.node_allocator)
    mapping = {}

    def push_inverse(node_id: int) -> int:
        if node_id in mapping:
            return mapping[node_id]

        node = ac.get_node_data(node_id)
        child_ids = ac.get_children(node_id)

        if isinstance(node, LeafNode):
            new_node = InverseLeafNode(base_leaf=node, power=-1)
            new_id = new_ac.add_node(new_node)
            mapping[node_id] = new_id
            return new_id

        # Structural inversion: P^-1(x) = sum (w_i^-1 * P_i^-1(x))
        # (A * B)^-1 = A^-1 * B^-1
        new_node = copy.copy(node)
        if isinstance(node, SumNode) and hasattr(node, "weights") and node.weights is not None:
            new_node.weights = 1.0 / node.weights.clamp(min=1e-12)

        new_id = new_ac.add_node(new_node)
        mapping[node_id] = new_id

        # Preserve edges and their data
        for child_id, edge_data in ac.get_outgoing_edges(node_id):
            inv_child = push_inverse(child_id)
            new_ac.add_edge(new_id, inv_child, edge_data)

        return new_id

    roots = ac.get_roots()
    for rid in roots:
        push_inverse(rid)

    return new_ac


def _multiply(
    ac1: SymbolicArithmeticCircuit,
    ac2: SymbolicArithmeticCircuit,
) -> SymbolicArithmeticCircuit:
    """
    Returns a new circuit representing the product of ac1 and ac2.
    Implements structured product (matched sums/products) when MD-sets overlap.
    """
    new_ac = SymbolicArithmeticCircuit(node_allocator=ac1.node_allocator)
    memo = {}
    ac1_copy_memo = {}
    ac2_copy_memo = {}

    def copy_node(ac_src: SymbolicArithmeticCircuit, src_id: int, copy_memo: Dict[int, int]) -> int:
        if src_id in copy_memo:
            return copy_memo[src_id]
        node = ac_src.get_node_data(src_id)
        new_node = copy.copy(node)
        new_id = new_ac.add_node(new_node)
        copy_memo[src_id] = new_id
        for child_id, edge_data in ac_src.get_outgoing_edges(src_id):
            new_child_id = copy_node(ac_src, child_id, copy_memo)
            new_ac.add_edge(new_id, new_child_id, edge_data)
        return new_id

    def recurse(n1_id: int, n2_id: int, is_deterministic: bool = False) -> int:
        if (n1_id, n2_id, is_deterministic) in memo:
            return memo[(n1_id, n2_id, is_deterministic)]

        n1 = ac1.get_node_data(n1_id)
        n2 = ac2.get_node_data(n2_id)

        c_intersect = n1.scope.intersection(n2.scope)

        # Base Case 1: Disjoint scopes
        if c_intersect.is_empty:
            new_support = copy.copy(n1.support)
            for v, inter in n2.support.intervals.items():
                new_support.intervals[v] = inter
            new_node = KroneckerProductNode(
                support=new_support, unit_count=n1.unit_count * n2.unit_count
            )
            new_node.md_set = BitSet()
            new_id = new_ac.add_node(new_node)
            l_id = copy_node(ac1, n1_id, ac1_copy_memo)
            r_id = copy_node(ac2, n2_id, ac2_copy_memo)
            new_ac.add_edge(new_id, l_id)
            new_ac.add_edge(new_id, r_id)
            memo[(n1_id, n2_id, is_deterministic)] = new_id
            return new_id

        # Base Case 2: Both are leaves
        if isinstance(n1, LeafNode) and isinstance(n2, LeafNode):
            new_node = ProductLeafNode(n1, n2) if is_deterministic else CartesianLeafNode(n1, n2)
            if getattr(n1, "md_set", None) and getattr(n2, "md_set", None):
                new_node.md_set = n1.md_set.union(n2.md_set)
            else:
                new_node.md_set = BitSet.universal()
            new_id = new_ac.add_node(new_node)
            memo[(n1_id, n2_id, is_deterministic)] = new_id
            return new_id

        c1_children = ac1.get_children(n1_id)
        c2_children = ac2.get_children(n2_id)

        # Case 3: Sum Nodes - check for MD matching
        if isinstance(n1, (SumNode, UniversalSumNode)) and isinstance(
            n2, (SumNode, UniversalSumNode)
        ):
            shared_md = n1.md_set.intersection(n2.md_set) if n1.md_set and n2.md_set else BitSet()

            # If they share an MD set over the common variables, we MUST match indices i==j
            if not shared_md.is_empty and n1.unit_count == n2.unit_count:
                new_support = copy.copy(n1.support)
                for v, inter in n2.support.intervals.items():
                    new_support.intervals[v] = inter

                new_node = type(n1)(
                    support=new_support, unit_count=n1.unit_count, md_set=n1.md_set.union(n2.md_set)
                )
                if hasattr(n1, "weights") and hasattr(n2, "weights"):
                    if n1.weights is not None and n2.weights is not None:
                        # Only multiply if dimensions match
                        if n1.weights.shape == n2.weights.shape:
                            new_node.weights = n1.weights * n2.weights

                new_id = new_ac.add_node(new_node)
                memo[(n1_id, n2_id, is_deterministic)] = new_id

                # Match children element-wise
                for c1, c2 in zip(c1_children, c2_children):
                    child_prod_id = recurse(c1, c2, is_deterministic=True)
                    new_ac.add_edge(new_id, child_prod_id)
                return new_id

            # Default: Cross-multiply sum children (Universal/independent)
            new_support = copy.copy(n1.support)
            for v, inter in n2.support.intervals.items():
                new_support.intervals[v] = inter
            new_node = SumNode(
                support=new_support,
                unit_count=n1.unit_count * n2.unit_count,
                md_set=BitSet.universal(),
            )

            if hasattr(n1, "weights") and hasattr(n2, "weights"):
                if n1.weights is not None and n2.weights is not None:
                    # Kronecker product of weights: (h1, pps1) x (h2, pps2) -> (h1*h2, pps1*pps2)
                    new_node.weights = torch.kron(n1.weights, n2.weights)

            new_id = new_ac.add_node(new_node)
            memo[(n1_id, n2_id, is_deterministic)] = new_id

            # SPN block structure: SumNode has one child (ProductNode)
            # which we create by recursing on the product of the input children.
            if c1_children and c2_children:
                child_prod_id = recurse(c1_children[0], c2_children[0], is_deterministic=False)
                new_ac.add_edge(new_id, child_prod_id)

            return new_id

        # Case 4: Product Nodes
        if is_deterministic and isinstance(n1, ProductNode):
            # Matched product of layers: use the same node type as n1
            new_support = copy.copy(n1.support)
            for v, inter in n2.support.intervals.items():
                new_support.intervals[v] = inter

            # Create a node of the same type
            new_node = type(n1)(
                support=new_support,
                unit_count=n1.unit_count,
            )
            new_id = new_ac.add_node(new_node)
            memo[(n1_id, n2_id, is_deterministic)] = new_id

            if len(c1_children) == 2 and len(c2_children) == 2:
                # Recurse and align left with left, right with right
                l_id = recurse(c1_children[0], c2_children[0], is_deterministic=True)
                r_id = recurse(c1_children[1], c2_children[1], is_deterministic=True)
                new_ac.add_edge(new_id, l_id)
                new_ac.add_edge(new_id, r_id)
            else:
                # Fallback for unary products or mismatched arity
                for c1, c2 in zip(c1_children, c2_children):
                    child_prod_id = recurse(c1, c2, is_deterministic=True)
                    new_ac.add_edge(new_id, child_prod_id)
            return new_id

        # General Fallback: If no match found, cross-multiply at the root level of these sub-circuits
        new_support = copy.copy(n1.support)
        for v, inter in n2.support.intervals.items():
            new_support.intervals[v] = inter
        new_node = KroneckerProductNode(
            support=new_support, unit_count=n1.unit_count * n2.unit_count
        )
        new_id = new_ac.add_node(new_node)
        l_id = copy_node(ac1, n1_id, ac1_copy_memo)
        r_id = copy_node(ac2, n2_id, ac2_copy_memo)
        new_ac.add_edge(new_id, l_id)
        new_ac.add_edge(new_id, r_id)
        memo[(n1_id, n2_id, is_deterministic)] = new_id
        return new_id

    ac1_roots = ac1.get_roots()
    ac2_roots = ac2.get_roots()
    if not ac1_roots or not ac2_roots:
        return new_ac

    recurse(ac1_roots[0], ac2_roots[0])
    return new_ac


def _instantiate(
    ac: SymbolicArithmeticCircuit,
    instantiations: Dict[int, Any],
) -> SymbolicArithmeticCircuit:
    """
    Returns a new circuit where leaves specified in instantiations are clamped to specific values.
    """
    new_ac = SymbolicArithmeticCircuit(node_allocator=ac.node_allocator)
    mapping = {}

    for node_id in ac.topological_sort(reverse=True):
        node = ac.get_node_data(node_id)
        if isinstance(node, LeafNode) and getattr(node, "var", None) in instantiations:
            val = instantiations[node.var]
            new_node = InstantiatedLeafNode(base_leaf=node, value=val)
            new_id = new_ac.add_node(new_node)
        else:
            new_node = copy.copy(node)
            new_id = new_ac.add_node(new_node)
            for child_id, edge_data in ac.get_outgoing_edges(node_id):
                new_ac.add_edge(new_id, mapping[child_id], edge_data)
        mapping[node_id] = new_id

    return new_ac


def compile_query(
    ast: EstimandAST,
    base_ac: SymbolicArithmeticCircuit,
    base_ac_root_id: int,  # kept for signature compatibility
    var_to_id: Dict[str, int],
) -> Tuple[SymbolicArithmeticCircuit, int]:
    """
    Compiles an EstimandAST into a SymbolicArithmeticCircuit by recursively
    applying the functional transformations.
    """

    def _build_recursive(ast_node_id: int) -> SymbolicArithmeticCircuit:
        ast_node = ast.get_node_data(ast_node_id)

        if isinstance(ast_node, PNode):
            return base_ac

        elif isinstance(ast_node, MargNode):
            child_id = ast.get_children(ast_node_id)[0]
            child_ac = _build_recursive(child_id)
            # Map string var names to int IDs
            marg_ids = {var_to_id[v] for v in ast_node.marginalize_vars if v in var_to_id}
            return _marginalize(child_ac, marg_ids)

        elif isinstance(ast_node, (ProdNode, DetProdNode)):
            children_ids = ast.get_children(ast_node_id)
            if not children_ids:
                empty_ac = SymbolicArithmeticCircuit()
                return empty_ac

            acc_ac = _build_recursive(children_ids[0])
            for cid in children_ids[1:]:
                next_ac = _build_recursive(cid)
                acc_ac = _multiply(acc_ac, next_ac)
            return acc_ac

        elif isinstance(ast_node, PowNode):
            child_id = ast.get_children(ast_node_id)[0]
            child_ac = _build_recursive(child_id)
            if ast_node.power == -1:
                return _inverse(child_ac)
            return child_ac

        elif isinstance(ast_node, InstNode):
            child_id = ast.get_children(ast_node_id)[0]
            child_ac = _build_recursive(child_id)
            insts = {var_to_id[v]: val for v, val in ast_node.variables.items() if v in var_to_id}
            return _instantiate(child_ac, insts)

        elif isinstance(ast_node, ConstantNode):
            ac = SymbolicArithmeticCircuit()
            # Constant 1 leaf
            leaf = ConstantLeafNode(var=-1)  # dummy var
            ac.add_node(leaf)
            return ac

        return base_ac

    final_ac = _build_recursive(ast.get_root())
    roots = final_ac.get_roots()
    root_id = roots[0] if roots else -1
    return final_ac, root_id

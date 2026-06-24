import copy
from typing import Any, Dict, Optional, Set, Tuple

import torch

from src.logger import logger as g_logger
from src.symbolic.arithmetic.nodes.leaf import GaussianMixture
from src.utils import BitSet, Support

from ..id_ast import (
    ConstantNode,
    DetProdNode,
    EstimandAST,
    InstNode,
    MargNode,
    PNode,
    PowNode,
    ProdNode,
)
from ..vtree import VNode, VTree
from .circuit import SymbolicArithmeticCircuit
from .nodes import (
    CartesianLeafNode,
    ConstantLeafNode,
    InverseLeafNode,
    KroneckerProductNode,
    LeafNode,
    ProductLeafNode,
    SumNode,
)


logger = g_logger.getChild("query")
logger.setLevel("DEBUG")


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
    Also updates the md-sets of the new vtree and circuit nodes to universal if their md-set intersects marg_vars.
    """
    new_ac = SymbolicArithmeticCircuit(node_allocator=ac.node_allocator)
    new_ac.vtree = copy.deepcopy(ac.vtree)

    marg_vars_bitset = BitSet(marg_vars)

    # Update md-sets in the new vtree
    for v_id in new_ac.vtree._nodes:
        v_node = new_ac.vtree.get_node_data(v_id)
        if v_node.md_set is not None and not v_node.md_set.intersection(marg_vars_bitset).is_empty:
            v_node.md_set = BitSet.universal()
    mapping = {}

    for node_id in ac.topological_sort(reverse=True):
        node = ac.get_node_data(node_id)

        # Determine the new md_set for this node
        new_md_set = getattr(node, "md_set", None)
        if new_md_set is not None and not new_md_set.intersection(marg_vars_bitset).is_empty:
            new_md_set = BitSet.universal()

        if isinstance(node, LeafNode) and getattr(node, "var", None) in marg_vars:
            new_node = ConstantLeafNode(var=node.var, unit_count=node.unit_count, md_set=new_md_set)
            new_id = new_ac.add_node(new_node)
        else:
            new_node = copy.copy(node)
            if hasattr(new_node, "md_set"):
                new_node.md_set = new_md_set

            new_id = new_ac.add_node(new_node)
            for child_id, edge_data in ac.get_outgoing_edges(node_id):
                new_ac.add_edge(new_id, mapping[child_id], edge_data)
        mapping[node_id] = new_id

        if node_id in ac.sum_to_vtree:
            v_id = ac.sum_to_vtree[node_id]
            new_ac.sum_to_vtree[new_id] = v_id
            new_ac.vtree_to_sum.setdefault(v_id, []).append(new_id)

    return new_ac


def _inverse(
    ac: SymbolicArithmeticCircuit,
) -> SymbolicArithmeticCircuit:
    """
    Returns a new circuit representing the reciprocal density.
    Recursively inverts weights and leaves. Only valid for deterministic circuits.
    """
    new_ac = SymbolicArithmeticCircuit(node_allocator=ac.node_allocator)
    new_ac.vtree = ac.vtree
    mapping = {}

    def push_inverse(node_id: int) -> int:
        if node_id in mapping:
            return mapping[node_id]

        node = ac.get_node_data(node_id)

        if isinstance(node, LeafNode):
            new_node = InverseLeafNode(base_leaf=node, power=-1)
            new_id = new_ac.add_node(new_node)
            mapping[node_id] = new_id
            if node_id in ac.sum_to_vtree:
                v_id = ac.sum_to_vtree[node_id]
                new_ac.sum_to_vtree[new_id] = v_id
                new_ac.vtree_to_sum.setdefault(v_id, []).append(new_id)
            return new_id

        # Structural inversion in log-space: log(1/w) = -log(w)
        new_node = copy.copy(node)
        if (
            isinstance(node, SumNode)
            and hasattr(node, "log_weights")
            and node.log_weights is not None
        ):
            # Mask out sparsity bounds so they don't become huge positive numbers
            # An absent edge (weight approx 0) must remain absent.
            LOG_ZERO = -46.0517  # torch.log(torch.tensor(1e-20))
            mask = node.log_weights < -15.0
            new_weights = torch.where(mask, torch.tensor(LOG_ZERO, device=node.log_weights.device), -node.log_weights)
            new_node.log_weights = new_weights.detach().requires_grad_(True)

        new_id = new_ac.add_node(new_node)
        mapping[node_id] = new_id
        if node_id in ac.sum_to_vtree:
            v_id = ac.sum_to_vtree[node_id]
            new_ac.sum_to_vtree[new_id] = v_id
            new_ac.vtree_to_sum.setdefault(v_id, []).append(new_id)

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
    Implements the structural product algorithm for compatible circuits.
    """
    new_ac = SymbolicArithmeticCircuit(node_allocator=ac1.node_allocator)
    new_ac.vtree = VTree()

    def get_vtree_info(v1_id: int, v2_id: int):
        vt1, vt2 = ac1.vtree, ac2.vtree
        return (
            vt1.get_node_data(v1_id),
            vt2.get_node_data(v2_id),
            vt1.get_children_pair(v1_id),
            vt2.get_children_pair(v2_id),
        )

    def check_deferred_product(v1_id: int, v2_id: int) -> Optional[Tuple[bool, bool, int, int]]:
        v1_node, v2_node, v1_children, v2_children = get_vtree_info(v1_id, v2_id)
        common_scope = v1_node.scope.intersection(v2_node.scope)

        if v2_children:
            l2, r2 = v2_children
            if l2 is not None and common_scope.is_subset(ac2.vtree.get_node_data(l2).scope):
                return True, True, l2, r2
            if r2 is not None and common_scope.is_subset(ac2.vtree.get_node_data(r2).scope):
                return True, False, r2, l2

        if v1_children:
            l1, r1 = v1_children
            if l1 is not None and common_scope.is_subset(ac1.vtree.get_node_data(l1).scope):
                return False, True, l1, r1
            if r1 is not None and common_scope.is_subset(ac1.vtree.get_node_data(r1).scope):
                return False, False, r1, l1

        return None

    def check_matching_children(
        v1_id: int, v2_id: int
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        v1_node, v2_node, v1_children, v2_children = get_vtree_info(v1_id, v2_id)
        common_scope = v1_node.scope.intersection(v2_node.scope)

        if not v1_children or not v2_children:
            return None

        matched_children = []

        for c1 in v1_children:
            for c2 in v2_children:
                if c1 is not None and c2 is not None:
                    c1_scope = ac1.vtree.get_node_data(c1).scope
                    c2_scope = ac2.vtree.get_node_data(c2).scope
                    if c1_scope.intersection(common_scope) == c2_scope.intersection(common_scope):
                        matched_children.append((c1, c2))

        return tuple(matched_children) if matched_children else None

    def copy_subcircuit(ac: SymbolicArithmeticCircuit, v_id: int) -> Tuple[int, int]:
        s_id = ac.vtree_to_sum[v_id][0]
        s_node = ac.get_node_data(s_id)
        v_node = ac.vtree.get_node_data(v_id)

        new_v_node = VNode(scope=v_node.scope, md_set=v_node.md_set)
        new_v_id = new_ac.vtree.add_node(new_v_node)

        if ac.is_sum_node(s_id) and isinstance(s_node, LeafNode):
            new_node = copy.copy(s_node)
            new_id = new_ac.add_node(new_node)
            new_ac.sum_to_vtree[new_id] = new_v_id
            new_ac.vtree_to_sum.setdefault(new_v_id, []).append(new_id)
            return new_id, new_v_id

        p_id = ac.get_children(s_id)[0]
        p_node = ac.get_node_data(p_id)

        children = ac.vtree.get_children_pair(v_id)
        if children is None:
            new_node = copy.copy(s_node)
            new_id = new_ac.add_node(new_node)
            new_ac.sum_to_vtree[new_id] = new_v_id
            new_ac.vtree_to_sum.setdefault(new_v_id, []).append(new_id)
            return new_id, new_v_id

        l_v_id, r_v_id = children
        l_new_id, l_new_v_id = copy_subcircuit(ac, l_v_id)
        r_new_id, r_new_v_id = copy_subcircuit(ac, r_v_id)

        new_ac.vtree.add_children(new_v_id, l_new_v_id, r_new_v_id)

        new_p_node = copy.copy(p_node)
        new_p_id = new_ac.add_node(new_p_node)
        new_ac.add_edge(new_p_id, l_new_id)
        new_ac.add_edge(new_p_id, r_new_id)

        new_s_node = copy.copy(s_node)
        new_s_id = new_ac.add_node(new_s_node)
        new_ac.add_edge(new_s_id, new_p_id)

        new_ac.sum_to_vtree[new_s_id] = new_v_id
        new_ac.vtree_to_sum.setdefault(new_v_id, []).append(new_s_id)

        return new_s_id, new_v_id

    def recurse(v1_id: int, v2_id: int, parent_det: bool = False) -> Tuple[int, int]:
        v1_node, v2_node, v1_children, v2_children = get_vtree_info(v1_id, v2_id)
        common_scope = v1_node.scope.intersection(v2_node.scope)
        logger.debug(
            "_multiply recurse: v1=%d(scope=%s) v2=%d(scope=%s) common=%s",
            v1_id,
            list(v1_node.scope),
            v2_id,
            list(v2_node.scope),
            list(common_scope),
        )

        assert v1_node.md_set is not None and v2_node.md_set is not None, (
            "MD-sets must be defined for all vtree nodes in both circuits for multiplication."
        )
        is_det = common_scope.is_superset(v1_node.md_set) and common_scope.is_superset(
            v2_node.md_set
        )

        # Case 1: Disjoint scopes
        if common_scope.is_empty:
            l_new_id, l_new_v_id = copy_subcircuit(ac1, v1_id)
            r_new_id, r_new_v_id = copy_subcircuit(ac2, v2_id)

            s1_id = ac1.vtree_to_sum[v1_id][0]
            s2_id = ac2.vtree_to_sum[v2_id][0]
            s1_node = ac1.get_node_data(s1_id)
            s2_node = ac2.get_node_data(s2_id)

            new_unit_count = s1_node.unit_count * s2_node.unit_count
            l_new_node = new_ac.get_node_data(l_new_id)
            r_new_node = new_ac.get_node_data(r_new_id)

            new_p_unit_count = getattr(l_new_node, "unit_count", 1) * getattr(
                r_new_node, "unit_count", 1
            )
            new_p_node = KroneckerProductNode(support=None, unit_count=new_p_unit_count)
            new_p_id = new_ac.add_node(new_p_node)
            new_ac.add_edge(new_p_id, l_new_id)
            new_ac.add_edge(new_p_id, r_new_id)

            new_v_node = VNode(scope=v1_node.scope.union(v2_node.scope), md_set=BitSet.universal())
            new_v_id = new_ac.vtree.add_node(new_v_node)
            new_ac.vtree.add_children(new_v_id, l_new_v_id, r_new_v_id)

            logger.debug(
                "  -> Case 1 disjoint: new_p_id=%d unit_count=%d", new_p_id, new_p_unit_count
            )
            return new_p_id, new_v_id

        # Case 2: Leaf nodes
        if not v1_children and not v2_children:
            l1_id = ac1.vtree_to_sum[v1_id][0]
            l2_id = ac2.vtree_to_sum[v2_id][0]
            leaf1_node = ac1.get_node_data(l1_id)
            leaf2_node = ac2.get_node_data(l2_id)

            # if one of the operands is a constant node, simply return the non-constant operand
            if is_det or parent_det:
                new_node = ProductLeafNode(leaf_a=leaf1_node, leaf_b=leaf2_node)
            else:
                new_node = CartesianLeafNode(leaf_a=leaf1_node, leaf_b=leaf2_node)

            if isinstance(leaf1_node, GaussianMixture) and isinstance(leaf2_node, GaussianMixture):
                if is_det or parent_det:
                    unit_count = leaf1_node.unit_count
                    log_weights = (
                        (leaf1_node.log_weights + leaf2_node.log_weights)
                        .detach()
                        .requires_grad_(True)
                    )
                    new_node = GaussianMixture(
                        base_dist=ProductLeafNode(
                            leaf_a=leaf1_node.base_dist, leaf_b=leaf2_node.base_dist
                        ),
                        log_weights=log_weights,
                    )
                else:
                    unit_count = leaf1_node.unit_count * leaf2_node.unit_count
                    log_weights = (
                        (
                            leaf1_node.log_weights.unsqueeze(1).unsqueeze(3)
                            + leaf2_node.log_weights.unsqueeze(0).unsqueeze(2)
                        )
                        .detach()
                        .requires_grad_(True)
                    )

                    new_node = GaussianMixture(
                        base_dist=CartesianLeafNode(
                            leaf_a=leaf1_node.base_dist, leaf_b=leaf2_node.base_dist
                        ),
                        log_weights=log_weights.reshape(unit_count, -1),
                    )

            new_id = new_ac.add_node(new_node)

            new_v_node = VNode(
                scope=v1_node.scope.union(v2_node.scope),
                md_set=v1_node.md_set.union(v2_node.md_set),
            )
            new_v_id = new_ac.vtree.add_node(new_v_node)

            new_ac.sum_to_vtree[new_id] = new_v_id
            new_ac.vtree_to_sum.setdefault(new_v_id, []).append(new_id)

            logger.debug("  -> Case 2 leaf: new_id=%d", new_id)
            return new_id, new_v_id

        # Case 3: Deferred product
        if res := check_deferred_product(v1_id, v2_id):
            is_v2_split, match_is_left, matched_child_id, unmatched_child_id = res
            logger.debug(
                "  -> Case 3 deferred: is_v2_split=%s match_is_left=%s", is_v2_split, match_is_left
            )

            if is_v2_split:
                recurse_res, recurse_v_id = recurse(v1_id, matched_child_id, is_det)
                copied_res, copied_v_id = copy_subcircuit(ac2, unmatched_child_id)
                parent_ac = ac2
                parent_v_id = v2_id
            else:
                recurse_res, recurse_v_id = recurse(matched_child_id, v2_id, is_det)
                copied_res, copied_v_id = copy_subcircuit(ac1, unmatched_child_id)
                parent_ac = ac1
                parent_v_id = v1_id

            if match_is_left:
                l_new_id, r_new_id = recurse_res, copied_res
                l_new_v_id, r_new_v_id = recurse_v_id, copied_v_id
            else:
                l_new_id, r_new_id = copied_res, recurse_res
                l_new_v_id, r_new_v_id = copied_v_id, recurse_v_id

            # The rest of the circuit is copied up from the parent (bigger) circuit
            parent_sum_id = parent_ac.vtree_to_sum[parent_v_id][0]
            parent_sum_node = parent_ac.get_node_data(parent_sum_id)
            p_id = parent_ac.get_children(parent_sum_id)[0]
            p_node = parent_ac.get_node_data(p_id)

            new_p_unit_count = p_node.unit_count
            new_p_node = KroneckerProductNode(support=None, unit_count=new_p_unit_count)

            new_p_id = new_ac.add_node(new_p_node)
            new_ac.add_edge(new_p_id, l_new_id)
            new_ac.add_edge(new_p_id, r_new_id)

            # Copy the parent sum node (weights and all)
            new_s_node = copy.copy(parent_sum_node)
            new_s_id = new_ac.add_node(new_s_node)
            new_ac.add_edge(new_s_id, new_p_id)

            parent_v_node = parent_ac.vtree.get_node_data(parent_v_id)
            new_v_node = VNode(scope=parent_v_node.scope, md_set=parent_v_node.md_set)
            new_v_id = new_ac.vtree.add_node(new_v_node)
            new_ac.vtree.add_children(new_v_id, l_new_v_id, r_new_v_id)

            new_ac.sum_to_vtree[new_s_id] = new_v_id
            new_ac.vtree_to_sum.setdefault(new_v_id, []).append(new_s_id)

            logger.debug(
                "     deferred sum copied: unit_count=%d log_weights=%s sparse=%s",
                new_s_node.unit_count,
                list(new_s_node.log_weights.shape) if new_s_node.log_weights is not None else None,
                new_s_node.sparse,
            )

            return new_s_id, new_v_id

        # Case 4: Matching children
        if res := check_matching_children(v1_id, v2_id):
            l1_id, l2_id = res[0]
            r1_id, r2_id = res[1]

            s1_id, s2_id = ac1.vtree_to_sum[v1_id][0], ac2.vtree_to_sum[v2_id][0]
            s1_node, s2_node = ac1.get_node_data(s1_id), ac2.get_node_data(s2_id)
            p1_id = ac1.get_children(s1_id)[0]
            p2_id = ac2.get_children(s2_id)[0]

            children_1 = ac1.get_children(p1_id)
            children_2 = ac2.get_children(p2_id)

            assert children_1 is not None and children_2 is not None

            c_l1_id, c_r1_id = children_1
            c_l2_id, c_r2_id = children_2
            l1_node, r1_node = ac1.get_node_data(c_l1_id), ac1.get_node_data(c_r1_id)
            l2_node, r2_node = ac2.get_node_data(c_l2_id), ac2.get_node_data(c_r2_id)

            h_l1, h_r1 = getattr(l1_node, "unit_count", 1), getattr(r1_node, "unit_count", 1)
            h_l2, h_r2 = getattr(l2_node, "unit_count", 1), getattr(r2_node, "unit_count", 1)

            l_new_id, l_new_v_id = recurse(*res[0], is_det)
            r_new_id, r_new_v_id = recurse(*res[1], is_det)

            l_new_node = new_ac.get_node_data(l_new_id)
            r_new_node = new_ac.get_node_data(r_new_id)
            h_l_new = getattr(l_new_node, "unit_count", 1)
            h_r_new = getattr(r_new_node, "unit_count", 1)

            new_p_unit_count = h_l_new * h_r_new
            new_p_node = KroneckerProductNode(support=None, unit_count=new_p_unit_count)

            new_p_id = new_ac.add_node(new_p_node)
            new_ac.add_edge(new_p_id, l_new_id)
            new_ac.add_edge(new_p_id, r_new_id)

            lw1, lw2 = getattr(s1_node, "log_weights", None), getattr(s2_node, "log_weights", None)
            assert lw1 is not None and lw2 is not None

            l_unit_supports = getattr(l_new_node, "unit_supports", None) or [Support()] * h_l_new
            r_unit_supports = getattr(r_new_node, "unit_supports", None) or [Support()] * h_r_new
            p_unit_supports = []

            logger.debug(
                f"  -> Case 4 matching: h_l1={h_l1} h_r1={h_r1} h_l2={h_l2} h_r2={h_r2} h_l_new={h_l_new} h_r_new={h_r_new}"
                f" new_p_unit_count={new_p_unit_count} is_det={is_det}"
            )

            for i in range(h_l_new):
                for j in range(h_r_new):
                    p_unit_supports.append(
                        Support.fast_disjoint_union(l_unit_supports[i], r_unit_supports[j])
                    )

            new_p_node.unit_supports = p_unit_supports

            if is_det:
                w_out = lw1 + lw2
                new_unit_count = s1_node.unit_count
            else:
                U1, C1 = lw1.shape
                U2, C2 = lw2.shape

                w1_3d = lw1.view(U1, h_l1, h_r1)
                w2_3d = lw2.view(U2, h_l2, h_r2)

                expand_L = h_l_new == h_l1 * h_l2
                expand_R = h_r_new == h_r1 * h_r2

                if expand_L and expand_R:
                    w_out = w1_3d.unsqueeze(1).unsqueeze(3).unsqueeze(5) + w2_3d.unsqueeze(
                        0
                    ).unsqueeze(2).unsqueeze(4)
                elif not expand_L and expand_R:
                    w_out = w1_3d.unsqueeze(1).unsqueeze(4) + w2_3d.unsqueeze(0).unsqueeze(3)
                elif expand_L and not expand_R:
                    w_out = w1_3d.unsqueeze(1).unsqueeze(3) + w2_3d.unsqueeze(0).unsqueeze(2)
                else:
                    raise CircuitCompilationError(
                        "Neither child was expanded in non-deterministic kronecker product. Unexpected behavior."
                    )

                if parent_det:
                    w_out = w_out.diagonal(dim1=0, dim2=1).movedim(-1, 0)
                    new_unit_count = min(U1, U2)
                else:
                    w_out = w_out.reshape(U1, U2, h_l_new, h_r_new)
                    new_unit_count = U1 * U2

            w_out_reshaped = w_out.reshape(new_unit_count, -1)

            # Compute sum node unit supports
            s_unit_supports = []
            s_support = Support()
            for u in range(new_unit_count):
                u_supp = Support()
                for c in range(new_p_unit_count):
                    if w_out_reshaped[u, c].item() > -1e4:
                        u_supp = u_supp.union(p_unit_supports[c])
                s_unit_supports.append(u_supp)
                s_support = s_support.union(u_supp)

            new_s_node = SumNode(
                support=s_support,
                unit_supports=s_unit_supports,
                unit_count=new_unit_count,
                sparse=getattr(s1_node, "sparse", True),
            )
            new_s_node.log_weights = w_out_reshaped.detach().requires_grad_(True)

            new_s_id = new_ac.add_node(new_s_node)
            new_ac.add_edge(new_s_id, new_p_id)

            new_v_node = VNode(
                scope=v1_node.scope.union(v2_node.scope),
                md_set=v1_node.md_set.union(v2_node.md_set),
            )
            new_v_id = new_ac.vtree.add_node(new_v_node)
            new_ac.vtree.add_children(new_v_id, l_new_v_id, r_new_v_id)

            new_ac.sum_to_vtree[new_s_id] = new_v_id
            new_ac.vtree_to_sum.setdefault(new_v_id, []).append(new_s_id)
            return new_s_id, new_v_id
        raise CircuitCompilationError(
            f"Incompatible circuits for multiplication at vtree nodes {v1_id} and {v2_id}."
        )

    ac1_roots, ac2_roots = ac1.get_roots(), ac2.get_roots()
    if not ac1_roots or not ac2_roots:
        return new_ac

    v1_root, v2_root = ac1.sum_to_vtree[ac1_roots[0]], ac2.sum_to_vtree[ac2_roots[0]]
    recurse(v1_root, v2_root)
    return new_ac


def _instantiate(
    ac: SymbolicArithmeticCircuit, instantiations: Dict[int, Any]
) -> SymbolicArithmeticCircuit:
    """
    Returns a new circuit where leaves specified in instantiations are clamped to specific values.
    """
    new_ac = SymbolicArithmeticCircuit(node_allocator=ac.node_allocator)
    new_ac.vtree = ac.vtree
    mapping = {}

    for node_id in ac.topological_sort(reverse=True):
        node = ac.get_node_data(node_id)
        if isinstance(node, LeafNode) and getattr(node, "var", None) in instantiations:
            val = instantiations[node.var]
            if isinstance(node, GaussianMixture):
                new_node = GaussianMixture(
                    base_dist=InstantiatedLeafNode(base_leaf=node.base_dist, value=val),
                    log_weights=node.log_weights.detach().requires_grad_(True),
                )
            else:
                new_node = InstantiatedLeafNode(base_leaf=node, value=val)
            new_id = new_ac.add_node(new_node)
        else:
            new_node = copy.copy(node)
            new_id = new_ac.add_node(new_node)
            for child_id, edge_data in ac.get_outgoing_edges(node_id):
                new_ac.add_edge(new_id, mapping[child_id], edge_data)
        mapping[node_id] = new_id
        if node_id in ac.sum_to_vtree:
            v_id = ac.sum_to_vtree[node_id]
            new_ac.sum_to_vtree[new_id] = v_id
            new_ac.vtree_to_sum.setdefault(v_id, []).append(new_id)

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
            # Map string var names to int IDs, or keep as int if already int
            marg_ids = {var_to_id[v] if v in var_to_id else v for v in ast_node.marginalize_vars}
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
            insts = {
                var_to_id[v] if v in var_to_id else v: val for v, val in ast_node.variables.items()
            }
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

import copy
from typing import Any, Dict, Optional, Set, Tuple

import torch

from src.logger import logger as g_logger
from src.symbolic.arithmetic.nodes.leaf_layer import MixtureLeafLayer
from src.symbolic.arithmetic.weights import (
    LOG_ZERO,
    DenseWeights,
    MixingCondWeights,
    ProductWeights,
)
from src.symbolic.vtree import VNode, VTree
from src.utils import BitSet

from ..id_ast import (
    CondNode,
    ConstantNode,
    EstimandAST,
    InstNode,
    MargNode,
    PNode,
    ProdNode,
)
from .circuit import SymbolicArithmeticCircuit
from .nodes import (
    ConstantRegionNode,
    IndicatorLeafLayer,
    LeafLayer,
    ProductLeafLayer,
    SumLayer,
)


logger = g_logger.getChild("query")
logger.setLevel("ERROR")


class CircuitCompilationError(Exception):
    pass


class InstantiatedLeafLayer(LeafLayer):
    """Placeholder leaf node to represent clamping a variable to a specific value."""

    def __init__(self, base_leaf: LeafLayer, value: Any):
        super().__init__(
            support=base_leaf.support,
            num_nodes=base_leaf.num_nodes,
            num_groups=base_leaf.num_groups,
            node_supports=base_leaf.node_supports,
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
        return f"InstantiatedLeafLayer(base={self.base_leaf}, val={self.value})"


def _marginalize(ac: SymbolicArithmeticCircuit, marg_vars: Set[int]) -> SymbolicArithmeticCircuit:
    """
    Returns a new circuit representing the marginalized density.
    Nodes keep their original scope, but track marginalized variables in `marg_scope`.
    Leaves that are fully marginalized are replaced with a ConstantRegionNode.
    """
    new_ac = SymbolicArithmeticCircuit(node_allocator=ac.node_allocator)
    new_ac.vtree = copy.deepcopy(ac.vtree)
    marg_vars_bitset = BitSet(marg_vars)

    def marginalize_recursive(node_id: int) -> int:
        node = ac.get_node_data(node_id)
        v_id = ac.sum_to_vtree.get(node_id)

        new_marg_scope = node.marg_scope
        new_md_set = node.md_set
        new_node = copy.copy(node)

        if v_id is not None:  # must be sum or leaf node
            # update marg_scope and md_set
            new_marg_scope = node.marg_scope.union(node.scope.intersection(marg_vars_bitset))
            if not marg_vars_bitset.intersection(node.md_set).is_empty:
                new_md_set = BitSet.universal()

        if isinstance(node, LeafLayer) and new_marg_scope == node.scope:
            new_node = ConstantRegionNode(
                scope=node.scope,
                num_nodes=node.num_nodes,
                num_groups=node.num_groups,
                md_set=new_md_set,
            )

        new_node.marg_scope = new_marg_scope
        new_node.md_set = new_md_set

        new_node_id = new_ac.add_node(new_node)

        if v_id is not None:
            v_node = new_ac.vtree.get_node_data(v_id)
            v_node.md_set = new_md_set
            new_ac.sum_to_vtree[new_node_id] = v_id
            new_ac.vtree_to_sum[v_id] = new_node_id

        for child_id, edge_data in ac.get_outgoing_edges(node_id):
            new_child_id = marginalize_recursive(child_id)
            new_ac.add_edge(new_node_id, new_child_id, edge_data)

        return new_node_id

    for root_id in ac.get_roots():
        marginalize_recursive(root_id)

    return new_ac


def _conditional(ac: SymbolicArithmeticCircuit, cond_vars: Set[int]) -> SymbolicArithmeticCircuit:
    """
    Returns a new circuit representing the conditional distribution.
    Implements Wang's algorithm:
    - Nodes that are MD with respect to cond_vars have their weights stripped.
    - Other nodes are kept as is (since norm_consts are 1.0 for the joint base AC).
    """
    new_ac = SymbolicArithmeticCircuit(node_allocator=ac.node_allocator)
    new_ac.vtree = copy.deepcopy(ac.vtree)
    cond_vars_bitset = BitSet(cond_vars)

    def recurse(node_id: int) -> int:
        node = ac.get_node_data(node_id)

        if isinstance(node, LeafLayer):
            if not node.md_set.is_universal and node.scope.min in cond_vars_bitset:
                # uniformize the leaf, turn it into an indicator function.
                new_node = IndicatorLeafLayer(base_leaf=node)
            elif node.scope.min not in cond_vars_bitset:
                # simply copy the leaf
                new_node = copy.copy(node)
            else:  # triggers only if is universal and in cond_vars
                raise ValueError(
                    "Circuit is not marginal deterministic with respect to the conditioning set."
                )
            new_id = new_ac.add_node(new_node)
            new_ac.sum_to_vtree[new_id] = ac.sum_to_vtree.get(node_id)
            if ac.sum_to_vtree.get(node_id) is not None:
                new_ac.vtree_to_sum[ac.sum_to_vtree.get(node_id)] = new_id
            return new_id

        node: SumLayer

        # Since node is definitely an inner sum node at this point, these are all truthy.
        vnode_id = ac.sum_to_vtree.get(
            node_id
        )  # Must make sure we're only recursing over sum nodes
        vnode = ac.vtree.get_node_data(vnode_id)
        l_vnode_id, r_vnode_id = ac.vtree.get_children_pair(vnode_id)
        l_vnode = ac.vtree.get_node_data(l_vnode_id)
        r_vnode = ac.vtree.get_node_data(r_vnode_id)

        cond_vars_in_scope = cond_vars_bitset.intersection(vnode.scope)

        if cond_vars_in_scope.is_empty:
            # No need to normalize, for same reason as above. Just copy and continue
            new_node = copy.copy(node)
        elif cond_vars_in_scope.is_superset(node.md_set):
            # Distinguish between synthesizing, left/right mixing
            if vnode.md_set == l_vnode.md_set.union(r_vnode.md_set):  # Synthesizing, uniformize all
                log_weights = node.log_weights.uniformize()
            elif vnode.md_set == l_vnode.md_set:
                if cond_vars_in_scope.is_superset(r_vnode.md_set.union(l_vnode.md_set)):
                    log_weights = node.log_weights.uniformize()
                else:
                    log_weights = MixingCondWeights(node.log_weights, other_child_axis="right")
            elif (
                vnode.md_set == r_vnode.md_set
            ):  # Right mixing, unformize right using the left child values
                if cond_vars_in_scope.is_superset(r_vnode.md_set.union(l_vnode.md_set)):
                    log_weights = node.log_weights.uniformize()
                else:
                    log_weights = MixingCondWeights(node.log_weights, other_child_axis="left")
            else:
                raise ValueError(
                    "Invalid layer type. Must be either synthesizing or left/right mixing layer."
                )
            new_node = copy.copy(node)
            new_node.log_weights = log_weights
        else:
            raise ValueError(
                "Circuit is not marginal deterministic with respect to the conditioning set."
            )

        new_id = new_ac.add_node(new_node)
        new_ac.sum_to_vtree[new_id] = vnode_id
        new_ac.vtree_to_sum[vnode_id] = new_id
        for child_id, edge_data in ac.get_outgoing_edges(node_id):
            new_child_id = recurse(child_id)
            new_ac.add_edge(new_id, new_child_id, edge_data)

        return new_id

    for root_id in ac.get_roots():
        recurse(root_id)

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
            vt1.get_parent(v1_id),
            vt2.get_parent(v2_id),
            vt1.get_node_data(v1_id),
            vt2.get_node_data(v2_id),
            vt1.get_children_pair(v1_id),
            vt2.get_children_pair(v2_id),
        )

    def check_deferred_product(v1_id: int, v2_id: int) -> Optional[Tuple[bool, bool, int, int]]:
        _, _, v1_node, v2_node, v1_children, v2_children = get_vtree_info(v1_id, v2_id)
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
        _, _, v1_node, v2_node, v1_children, v2_children = get_vtree_info(v1_id, v2_id)
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
        s_id = ac.vtree_to_sum[v_id]
        s_node = ac.get_node_data(s_id)
        v_node = ac.vtree.get_node_data(v_id)

        new_v_node = VNode(scope=v_node.scope, md_set=v_node.md_set)
        new_v_id = new_ac.vtree.add_node(new_v_node)

        if ac.is_sum_node(s_id) and isinstance(s_node, LeafLayer):
            new_node = copy.copy(s_node)
            new_id = new_ac.add_node(new_node)
            new_ac.sum_to_vtree[new_id] = new_v_id
            new_ac.vtree_to_sum[new_v_id] = new_id
            return new_id, new_v_id

        children = ac.vtree.get_children_pair(v_id)
        if children is None:
            new_node = copy.copy(s_node)
            new_id = new_ac.add_node(new_node)
            new_ac.sum_to_vtree[new_id] = new_v_id
            new_ac.vtree_to_sum[new_v_id] = new_id
            return new_id, new_v_id

        l_v_id, r_v_id = children
        l_new_id, l_new_v_id = copy_subcircuit(ac, l_v_id)
        r_new_id, r_new_v_id = copy_subcircuit(ac, r_v_id)

        new_ac.vtree.add_children(new_v_id, l_new_v_id, r_new_v_id)

        new_s_node = copy.copy(s_node)
        new_id = new_ac.add_node(new_s_node)
        new_ac.add_edge(new_id, l_new_id)
        new_ac.add_edge(new_id, r_new_id)

        new_ac.sum_to_vtree[new_id] = new_v_id
        new_ac.vtree_to_sum[new_v_id] = new_id

        return new_id, new_v_id

    def recurse(v1_id: int, v2_id: int) -> Tuple[int, int]:
        v1_parent_id, v2_parent_id, v1_node, v2_node, v1_children, v2_children = get_vtree_info(
            v1_id, v2_id
        )
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

        # Case 1: Disjoint scopes
        if common_scope.is_empty:
            l_new_id, l_new_v_id = copy_subcircuit(ac1, v1_id)
            r_new_id, r_new_v_id = copy_subcircuit(ac2, v2_id)

            s1_id = ac1.vtree_to_sum[v1_id]
            s2_id = ac2.vtree_to_sum[v2_id]
            s1_node = ac1.get_node_data(s1_id)
            s2_node = ac2.get_node_data(s2_id)

            new_num_nodes = s1_node.num_nodes * s2_node.num_nodes
            new_num_groups = s1_node.num_groups * s2_node.num_groups
            l_new_node = new_ac.get_node_data(l_new_id)
            r_new_node = new_ac.get_node_data(r_new_id)

            new_node = SumLayer(
                num_nodes=new_num_nodes,
                num_groups=new_num_groups,
                md_set=BitSet(),
                support=l_new_node.support.union(r_new_node.support),
            )

            ref_w = s1_node.log_weights
            device = ref_w.log_weights.device if hasattr(ref_w, "log_weights") else ref_w.device
            lw = torch.full(
                (
                    new_num_groups,
                    new_num_nodes,
                    s1_node.num_groups,
                    s1_node.num_nodes,
                    s2_node.num_groups,
                    s2_node.num_nodes,
                ),
                LOG_ZERO,
                dtype=torch.float32,
                device=device,
            )

            # Weights are of size [G, U, G_L, L, G_R, R]. Since sum nodes are naive, for each index of G,U, there is only one non-zero entry of [G_L, L, G_R, R]
            # Since we have G_L*G_R groups of L*R nodes each, each combination of (G_L, L, G_R, R) corresponds to a unique (G, U) index. Therefore, we can use an identity matrix for the weights.
            gl = torch.arange(s1_node.num_groups, device=device)
            gr = torch.arange(s2_node.num_groups, device=device)
            ul = torch.arange(s1_node.num_nodes, device=device)
            ur = torch.arange(s2_node.num_nodes, device=device)
            GL, GR, UL, UR = torch.meshgrid(gl, gr, ul, ur, indexing="ij")
            g_idx = GL * s2_node.num_groups + GR
            u_idx = UL * s2_node.num_nodes + UR
            lw[g_idx, u_idx, GL, UL, GR, UR] = 0.0

            new_node.log_weights = DenseWeights(lw)
            new_vnode = VNode(scope=new_node.scope, md_set=new_node.md_set)
            new_id = new_ac.add_node(new_node)
            new_v_id = new_ac.vtree.add_node(new_vnode)
            new_ac.add_edge(new_id, l_new_id)
            new_ac.add_edge(new_id, r_new_id)
            new_ac.vtree.add_children(new_v_id, l_new_v_id, r_new_v_id)
            new_ac.sum_to_vtree[new_id] = new_v_id
            new_ac.vtree_to_sum[new_v_id] = new_id

            return new_id, new_v_id

        # Case 2: Leaf nodes
        if not v1_children and not v2_children:
            l1_id = ac1.vtree_to_sum[v1_id]
            l2_id = ac2.vtree_to_sum[v2_id]
            leaf1_node = ac1.get_node_data(l1_id)
            leaf2_node = ac2.get_node_data(l2_id)

            v1_parent_node = (
                ac1.vtree.get_node_data(v1_parent_id) if v1_parent_id is not None else None
            )
            v2_parent_node = (
                ac2.vtree.get_node_data(v2_parent_id) if v2_parent_id is not None else None
            )

            is_disjoint = not leaf1_node.md_set.is_universal and not leaf2_node.md_set.is_universal
            # if one of the operands is a constant node, simply return the non-constant operand
            new_node = ProductLeafLayer(
                leaf_a=leaf1_node, leaf_b=leaf2_node, expand_leaves=not is_disjoint
            )

            if isinstance(leaf1_node, MixtureLeafLayer) and isinstance(
                leaf2_node, MixtureLeafLayer
            ):
                new_log_weights = ProductWeights(
                    leaf1_node.log_weights,
                    leaf2_node.log_weights,
                    expand_L=not is_disjoint,
                    expand_R=not is_disjoint,
                    expand_U=not is_disjoint,
                )
                new_node = MixtureLeafLayer(
                    base_dist=ProductLeafLayer(
                        leaf_a=leaf1_node.base_dist,
                        leaf_b=leaf2_node.base_dist,
                        expand_leaves=not is_disjoint,
                    ),
                    log_weights=new_log_weights,
                )

            new_id = new_ac.add_node(new_node)

            new_v_node = VNode(
                scope=v1_node.scope.union(v2_node.scope),
                md_set=v1_node.md_set.union(v2_node.md_set),
            )
            new_v_id = new_ac.vtree.add_node(new_v_node)

            new_ac.sum_to_vtree[new_id] = new_v_id
            new_ac.vtree_to_sum[new_v_id] = new_id

            logger.debug("  -> Case 2 leaf: new_id=%d", new_id)
            return new_id, new_v_id

        # Case 3: Deferred product
        if res := check_deferred_product(v1_id, v2_id):
            is_v2_split, match_is_left, matched_child_id, unmatched_child_id = res
            logger.debug(
                "  -> Case 3 deferred: is_v2_split=%s match_is_left=%s", is_v2_split, match_is_left
            )

            if is_v2_split:
                recurse_res, recurse_v_id = recurse(v1_id, matched_child_id)
                copied_res, copied_v_id = copy_subcircuit(ac2, unmatched_child_id)
                parent_ac = ac2
                parent_v_id = v2_id
            else:
                recurse_res, recurse_v_id = recurse(matched_child_id, v2_id)
                copied_res, copied_v_id = copy_subcircuit(ac1, unmatched_child_id)
                parent_ac = ac1
                parent_v_id = v1_id

            if match_is_left:
                l_new_id, r_new_id = recurse_res, copied_res
                l_new_v_id, r_new_v_id = recurse_v_id, copied_v_id
            else:
                l_new_id, r_new_id = copied_res, recurse_res
                l_new_v_id, r_new_v_id = copied_v_id, recurse_v_id

            # So we have some matched child and some unmatched child. Matched child has dimensions (G_L*G_R, L*R). Assume that (G_R, R) is the root of the smaller circuit; if we don't assume this is (1,1). Unmathced child has dimensions (G_U, U)
            # Somehow G_R and R have to propagate upwards to the larger circuit's root.

            # For now we will assume G_R=1, R=1 for simplicity. Then we just have to copy the circuit upwards.
            new_node = copy.copy(parent_ac.get_node_data(parent_ac.vtree_to_sum[parent_v_id]))
            new_id = new_ac.add_node(new_node)
            new_v_node = VNode(
                scope=parent_ac.vtree.get_node_data(parent_v_id).scope,
                md_set=parent_ac.vtree.get_node_data(parent_v_id).md_set,
            )
            new_v_id = new_ac.vtree.add_node(new_v_node)
            new_ac.add_edge(new_id, l_new_id)
            new_ac.add_edge(new_id, r_new_id)
            new_ac.sum_to_vtree[new_id] = new_v_id
            new_ac.vtree_to_sum[new_v_id] = new_id
            return new_id, new_v_id

        # Case 4: Matching children
        if res := check_matching_children(v1_id, v2_id):
            s1_id, s2_id = ac1.vtree_to_sum[v1_id], ac2.vtree_to_sum[v2_id]
            v1_parent_node = (
                ac1.vtree.get_node_data(v1_parent_id) if v1_parent_id is not None else None
            )
            v2_parent_node = (
                ac2.vtree.get_node_data(v2_parent_id) if v2_parent_id is not None else None
            )
            v1_l_node, v1_r_node = (
                ac1.vtree.get_node_data(v1_children[0]),
                ac1.vtree.get_node_data(v1_children[1]),
            )
            v2_l_node, v2_r_node = (
                ac2.vtree.get_node_data(v2_children[0]),
                ac2.vtree.get_node_data(v2_children[1]),
            )
            assert (
                v1_l_node is not None
                and v1_r_node is not None
                and v2_l_node is not None
                and v2_r_node is not None
            ), "Encountered leaf in matching children case."

            s1_node, s2_node = ac1.get_node_data(s1_id), ac2.get_node_data(s2_id)
            children_1 = ac1.get_children(s1_id)
            children_2 = ac2.get_children(s2_id)

            assert children_1 is not None and children_2 is not None

            c_l1_id, c_r1_id = children_1
            c_l2_id, c_r2_id = children_2
            l1_node, r1_node = ac1.get_node_data(c_l1_id), ac1.get_node_data(c_r1_id)
            l2_node, r2_node = ac2.get_node_data(c_l2_id), ac2.get_node_data(c_r2_id)

            H_L1, H_R1 = l1_node.num_nodes, r1_node.num_nodes
            H_L2, H_R2 = l2_node.num_nodes, r2_node.num_nodes

            l_new_id, l_new_v_id = recurse(*res[0])
            r_new_id, r_new_v_id = recurse(*res[1])

            l_new_node = new_ac.get_node_data(l_new_id)
            r_new_node = new_ac.get_node_data(r_new_id)

            U_disjoint = not v1_node.md_set.is_universal and not v2_node.md_set.is_universal
            L_disjoint = not v1_l_node.md_set.is_universal and not v2_l_node.md_set.is_universal
            R_disjoint = not v1_r_node.md_set.is_universal and not v2_r_node.md_set.is_universal

            pw = ProductWeights(
                s1_node.log_weights,
                s2_node.log_weights,
                expand_U=not U_disjoint,
                expand_L=not L_disjoint,
                expand_R=not R_disjoint,
            )

            H_L_new = (H_L1 * H_L2) if pw.expand_L else H_L1
            H_R_new = (H_R1 * H_R2) if pw.expand_R else H_R1

            new_node = SumLayer(
                num_nodes=H_L_new * H_R_new,
                num_groups=s1_node.num_groups * s2_node.num_groups,
                md_set=v1_node.md_set.union(v2_node.md_set),
                support=l_new_node.support.union(r_new_node.support),
            )
            new_node.log_weights = pw
            new_id = new_ac.add_node(new_node)
            new_v_node = VNode(
                scope=v1_node.scope.union(v2_node.scope),
                md_set=v1_node.md_set.union(v2_node.md_set),  # NOTE: is this correct?
            )
            new_v_id = new_ac.vtree.add_node(new_v_node)
            new_ac.add_edge(new_id, l_new_id)
            new_ac.add_edge(new_id, r_new_id)
            new_ac.vtree.add_children(new_v_id, l_new_v_id, r_new_v_id)
            new_ac.sum_to_vtree[new_id] = new_v_id
            new_ac.vtree_to_sum[new_v_id] = new_id
            return new_id, new_v_id

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
        if isinstance(node, LeafLayer) and getattr(node, "var", None) in instantiations:
            val = instantiations[node.var]
            if isinstance(node, MixtureLeafLayer):
                new_node = MixtureLeafLayer(
                    base_dist=InstantiatedLeafLayer(base_leaf=node.base_dist, value=val),
                    log_weights=node.log_weights.detach().requires_grad_(True),
                )
            else:
                new_node = InstantiatedLeafLayer(base_leaf=node, value=val)
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
            new_ac.vtree_to_sum[v_id] = new_id

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

        elif isinstance(ast_node, ProdNode):
            children_ids = ast.get_children(ast_node_id)
            if not children_ids:
                empty_ac = SymbolicArithmeticCircuit()
                return empty_ac

            acc_ac = _build_recursive(children_ids[0])
            for cid in children_ids[1:]:
                next_ac = _build_recursive(cid)
                acc_ac = _multiply(acc_ac, next_ac)
            return acc_ac

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
            leaf = ConstantRegionNode(scope=BitSet(), num_nodes=1, num_groups=1)
            ac.add_node(leaf)
            return ac

        elif isinstance(ast_node, CondNode):
            child_id = ast.get_children(ast_node_id)[0]
            child_ac = _build_recursive(child_id)

            den_ids = {var_to_id[v] if v in var_to_id else v for v in ast_node.den_vars}
            return _conditional(child_ac, den_ids)

        return base_ac

    final_ac = _build_recursive(ast.get_root())
    roots = final_ac.get_roots()
    root_id = roots[0] if roots else -1
    return final_ac, root_id

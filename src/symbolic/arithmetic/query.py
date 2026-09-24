import contextvars
import copy
from functools import wraps
from typing import Any, Dict, Optional, Set, Tuple

import torch

from src.logger import logger as g_logger
from src.symbolic.arithmetic.weights import (
    LOG_ZERO,
    MixingCondWeights,
    ProductWeights,
    SparseWeights,
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
from ..identification import prod_determinism
from .circuit import SymbolicArithmeticCircuit
from .nodes import (
    ConstantLayer,
    IndicatorLeafLayer,
    InstantiatedLeafLayer,
    LeafLayer,
    MixtureLeafLayer,
    ProductLeafLayer,
    SumLayer,
)


logger = g_logger.getChild("query")
logger.setLevel("ERROR")

# Indentation-aware debug logging for the recursive _multiply algorithm.
# The depth is tracked in a ContextVar so nested/recursive calls indent correctly.
_rec_depth = contextvars.ContextVar("multiply_depth", default=0)


def _mlog(msg: str, *args):
    """Log a _multiply message indented by the current recursion depth."""
    logger.debug("  " * _rec_depth.get() + msg, *args)


def _depth_traced(fn):
    """Increment the multiply log depth for the duration of one call."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = _rec_depth.set(_rec_depth.get() + 1)
        try:
            return fn(*args, **kwargs)
        finally:
            _rec_depth.reset(token)

    return wrapper


class CircuitCompilationError(Exception):
    pass


CoordMap = Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]

_SHARED, _SIDE1, _SIDE2 = 0, 1, 2


def _factor_coord(res_size: int, factors: Tuple[Tuple[int, int], ...], k: int) -> torch.Tensor:
    """Sub-coordinate of factor ``k`` for each coordinate in ``range(res_size)``,
    where ``factors`` is the major->minor factorization of the axis."""
    suffix = 1
    for _, n in factors[k + 1 :]:
        suffix *= n
    p = torch.arange(res_size)
    return (p // suffix) % factors[k][1]


def _case3_axis_lock(
    axis: str,
    res_size: int,
    factors: Tuple[Tuple[int, int], ...],
    matched_side: int,
    small_side: int,
    slot_size: int,
    small_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resolve one Case-3 axis: how ``w_big``'s matched-slot axis maps into the
    recursion-result grid, and which result coordinate the small side's parent axis
    ties to.

    Returns (matched_idx, lock_idx), integer tensors of length ``res_size``.
    ``matched_idx`` selects the big weights' matched-slot axis (an identity mapping
    when the result grid IS the slot). ``lock_idx`` is what the small-side parent
    coordinate must equal: the free factor's coordinate, or — when the small layer's
    units coincide with the matched layer's units (a shared overlay) — the matched
    coordinate itself.
    """
    matched = [k for k, (s, _) in enumerate(factors) if s in (_SHARED, matched_side)]
    free = [k for k, (s, _) in enumerate(factors) if s == small_side]
    if any(s not in (_SHARED, matched_side, small_side) for s, _ in factors):
        raise CircuitCompilationError(
            f"Case 3 {axis}-axis map {factors} carries sides outside the matched/"
            f"small operand pair ({matched_side}, {small_side})."
        )
    if len(matched) > 1 or len(free) > 1:
        raise CircuitCompilationError(
            f"Case 3 {axis}-axis map {factors} has multi-factor matched/free blocks; "
            f"this requires unit-correspondence tracking beyond a single factor."
        )
    if not matched:
        raise CircuitCompilationError(
            f"Case 3 {axis}-axis map {factors} addresses no matched-slot coordinate."
        )
    if factors[matched[0]][1] != slot_size:
        raise CircuitCompilationError(
            f"Case 3 {axis}-axis: matched factor size {factors[matched[0]][1]} != "
            f"big weights' slot size {slot_size}."
        )
    matched_idx = _factor_coord(res_size, factors, matched[0])
    if free:
        if factors[free[0]][1] != small_size:
            raise CircuitCompilationError(
                f"Case 3 {axis}-axis: free factor size {factors[free[0]][1]} does not "
                f"match the small side's grid {small_size}; the whole-side lock cannot "
                f"be established (unit-correspondence tracking required)."
            )
        return matched_idx, _factor_coord(res_size, factors, free[0])
    if small_size == 1:
        return matched_idx, torch.zeros(res_size, dtype=torch.long)
    # No free coordinate, yet the small side has a non-trivial grid: only valid if the
    # small layer's units ARE the matched layer's units (shared overlay, side 0).
    if factors[matched[0]][0] == _SHARED and factors[matched[0]][1] == small_size:
        return matched_idx, matched_idx
    raise CircuitCompilationError(
        f"Case 3 {axis}-axis: small side has grid {small_size} but the recursion "
        f"result carries no free coordinate for it (map {factors}); cannot establish "
        f"the whole-side lock without guessing."
    )


def _circuit_device(ac: SymbolicArithmeticCircuit) -> torch.device:
    """Device of the circuit's weights (torch defaults to CPU, so constants
    created during query compilation must take the device from a real tensor)."""
    for nid in ac.topological_sort():
        node = ac.get_node_data(nid)
        w = getattr(node, "log_weights", None)
        t = w if isinstance(w, torch.Tensor) else getattr(w, "log_weights", None)
        if isinstance(t, torch.Tensor):
            return t.device
    return torch.device("cpu")


def _marginal_unit_values(ac: SymbolicArithmeticCircuit, node_id: int) -> torch.Tensor:
    """Exact per-(group, unit) values the subtree at ``node_id`` transmits when
    all its leaves are marginalized (forced to log 1).

    Computed by direct evaluation of the subtree, so it is exact regardless of
    weight normalization — e.g. in conditional circuits with stripped or
    renormalized weights, where a unit's transmitted value need not be log 1.
    Captured here because the subtree's weights are lost when it is replaced by
    a constant.
    """
    device = _circuit_device(ac)
    dummy = torch.zeros(1, 1, device=device)
    cache = {}

    def rec(nid: int) -> torch.Tensor:
        if nid in cache:
            return cache[nid]
        node = ac.get_node_data(nid)
        if isinstance(node, LeafLayer):
            out = torch.zeros(1, node.num_groups, node.num_nodes, device=device)
        else:
            child_outs = [rec(cid) for cid in ac.get_children(nid)]
            out = node.forward(dummy, child_outs)
        cache[nid] = out
        return out

    with torch.no_grad():
        return rec(node_id).squeeze(0).detach()  # [G, U]


def _marginalize(ac: SymbolicArithmeticCircuit, marg_vars: Set[int]) -> SymbolicArithmeticCircuit:
    new_ac = SymbolicArithmeticCircuit(node_allocator=ac.node_allocator)
    new_ac.vtree = copy.deepcopy(ac.vtree)
    marg_vars: BitSet = BitSet(marg_vars)

    def marginalize_recursive(node_id: int) -> int:
        node = ac.get_node_data(node_id)
        v_id = ac.sum_to_vtree.get(node_id)

        new_md_set = (
            node.md_set if marg_vars.intersection(node.md_set).is_empty else BitSet.universal()
        )
        new_scope = node.scope.difference(marg_vars)

        if new_scope.is_empty:  # Fully marginalized: empty-scope scalar constant
            logger.debug(
                f"Marginalizing node {node_id} with scope {node.scope} to constant with G={node.num_groups}, U={node.num_nodes}"
            )
            new_node = ConstantLayer(
                scope=new_scope,
                md_set=BitSet(),
                num_groups=node.num_groups,
                num_nodes=node.num_nodes,
                unit_values=_marginal_unit_values(ac, node_id),
            )
        else:
            new_node = copy.copy(node)
            new_node.scope = new_scope
            new_node.md_set = new_md_set

        new_node_id = new_ac.add_node(new_node)
        v_node = new_ac.vtree.get_node_data(v_id)
        v_node.scope = new_scope
        v_node.md_set = new_md_set
        new_ac.sum_to_vtree[new_node_id] = v_id
        new_ac.vtree_to_sum[v_id] = new_node_id

        if new_scope.is_empty:
            return new_node_id

        for child_id in ac.get_children(node_id):
            new_child_id = marginalize_recursive(child_id)
            new_ac.add_edge(new_node_id, new_child_id)

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
            if node.scope.is_empty:
                # Empty-scope constant (fully marginalized subtree): nothing to
                # condition, copy as is. Its mass is folded into big weights.
                new_node = copy.copy(node)
            elif not node.md_set.is_universal and node.scope.min in cond_vars_bitset:
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


def _prod_md_set(md1: BitSet, md2: BitSet, shared_vars: BitSet) -> BitSet:
    """Determinism label of a product node, per the T-ID UpdateMult rule
    (:func:`src.symbolic.identification.prod_determinism`): claim the union
    only where both operands guarantee determinism over the shared scope;
    otherwise make no claim (universal label).
    """
    if md1.is_universal or md2.is_universal:
        return BitSet.universal()
    q = prod_determinism(frozenset(md1), frozenset(md2), frozenset(shared_vars))
    return BitSet(q) if q is not None else BitSet.universal()


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
            l2_id, r2_id = v2_children
            l2 = ac2.vtree.get_node_data(l2_id)
            r2 = ac2.vtree.get_node_data(r2_id)
            if v1_node.scope.intersection(common_scope) == l2.scope.intersection(common_scope):
                return True, True, l2_id, r2_id
            elif v1_node.scope.intersection(common_scope) == r2.scope.intersection(common_scope):
                return True, False, r2_id, l2_id

        if v1_children:
            l1_id, r1_id = v1_children
            l1 = ac1.vtree.get_node_data(l1_id)
            r1 = ac1.vtree.get_node_data(r1_id)
            if v2_node.scope.intersection(common_scope) == l1.scope.intersection(common_scope):
                return False, True, l1_id, r1_id
            elif v2_node.scope.intersection(common_scope) == r1.scope.intersection(common_scope):
                return False, False, r1_id, l1_id

        return None

    def check_matching_children(
        v1_id: int, v2_id: int
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        _, _, v1_node, v2_node, v1_children, v2_children = get_vtree_info(v1_id, v2_id)
        common_scope = v1_node.scope.intersection(v2_node.scope)

        if not v1_children or not v2_children:
            return None

        l1_id, r1_id = v1_children
        l2_id, r2_id = v2_children
        l1 = ac1.vtree.get_node_data(l1_id)
        r1 = ac1.vtree.get_node_data(r1_id)
        l2 = ac2.vtree.get_node_data(l2_id)
        r2 = ac2.vtree.get_node_data(r2_id)

        if l1.scope.intersection(common_scope) == l2.scope.intersection(
            common_scope
        ) and r1.scope.intersection(common_scope) == r2.scope.intersection(common_scope):
            return ((l1_id, l2_id), (r1_id, r2_id))
        elif l1.scope.intersection(common_scope) == r2.scope.intersection(
            common_scope
        ) and r1.scope.intersection(common_scope) == l2.scope.intersection(common_scope):
            return ((l1_id, r2_id), (r1_id, l2_id))
        else:
            return None

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

    def recurse(v1_id: int, v2_id: int) -> Tuple[int, int, CoordMap]:
        """Returns (node_id, vtree_id, coord_map): ``coord_map`` describes, per parent
        grid axis (G, U), the ordered (side, size) factors of the result layer's grid
        (major -> minor; side 1 = ac1, side 2 = ac2, side 0 = shared overlay). Case 3
        consumes the child's map to re-index weights; nested deferred products flip
        the major side, so the map — not the dims — is authoritative for coordinates.
        """
        v1_big_id, v2_big_id, v1_node, v2_node, v1_children, v2_children = get_vtree_info(
            v1_id, v2_id
        )
        common_scope = v1_node.scope.intersection(v2_node.scope)
        _mlog(
            "recurse: v1=%d(scope=%s) v2=%d(scope=%s) v1_chs=%s v2_chs=%s common=%s",
            v1_id,
            list(v1_node.scope),
            v2_id,
            list(v2_node.scope),
            list(v1_children) if v1_children else [],
            list(v2_children) if v2_children else [],
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

            G_1, H_1, G_2, H_2 = (
                s1_node.num_groups,
                s1_node.num_nodes,
                s2_node.num_groups,
                s2_node.num_nodes,
            )
            G = G_1 * G_2
            H = H_1 * H_2

            l_new_node = new_ac.get_node_data(l_new_id)
            r_new_node = new_ac.get_node_data(r_new_id)

            new_node = SumLayer(
                num_nodes=G,
                num_groups=H,
                scope=v1_node.scope.union(v2_node.scope),
                md_set=_prod_md_set(s1_node.md_set, s2_node.md_set, common_scope),
            )

            lw_size = (G, H, G_1, H_1, G_2, H_2)
            device = s1_node.log_weights.device
            lw = torch.full(lw_size, LOG_ZERO, dtype=torch.float32, device=device)

            # Trivial sums as kronecker product
            for g_l in range(G_1):
                for h_l in range(H_1):
                    for g_r in range(G_2):
                        for h_r in range(H_2):
                            g = g_l * G_2 + g_r
                            h = h_l * H_2 + h_r
                            lw[g, h, g_l, h_l, g_r, h_r] = 0.0

            new_node.log_weights = SparseWeights(lw)
            _mlog(
                "-> Case 1 disjoint scopes: G=%d, H=%d (G_1=%d, H_1=%d x G_2=%d, H_2=%d)",
                G,
                H,
                G_1,
                H_1,
                G_2,
                H_2,
            )
            new_vnode = VNode(scope=new_node.scope, md_set=new_node.md_set)
            new_id = new_ac.add_node(new_node)
            new_v_id = new_ac.vtree.add_node(new_vnode)
            new_ac.add_edge(new_id, l_new_id)
            new_ac.add_edge(new_id, r_new_id)
            new_ac.vtree.add_children(new_v_id, l_new_v_id, r_new_v_id)
            new_ac.sum_to_vtree[new_id] = new_v_id
            new_ac.vtree_to_sum[new_v_id] = new_id

            # g = g_l * G_2 + g_r, h = h_l * H_2 + h_r: side-1 major on both axes.
            cmap: CoordMap = (
                ((_SIDE1, G_1), (_SIDE2, G_2)),
                ((_SIDE1, H_1), (_SIDE2, H_2)),
            )
            return new_id, new_v_id, cmap

        # Case 2: Leaf nodes
        if not v1_children and not v2_children:
            l1_id = ac1.vtree_to_sum[v1_id]
            l2_id = ac2.vtree_to_sum[v2_id]
            leaf1_node = ac1.get_node_data(l1_id)
            leaf2_node = ac2.get_node_data(l2_id)

            is_disjoint = not leaf1_node.md_set.is_universal and not leaf2_node.md_set.is_universal
            new_node = ProductLeafLayer(
                leaf_a=leaf1_node,
                leaf_b=leaf2_node,
                expand_leaves=not is_disjoint,
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
                md_set=_prod_md_set(v1_node.md_set, v2_node.md_set, common_scope),
            )
            new_v_id = new_ac.vtree.add_node(new_v_node)

            new_ac.sum_to_vtree[new_id] = new_v_id
            new_ac.vtree_to_sum[new_v_id] = new_id

            _mlog(
                "-> Case 2 leaf: new_id=%d, G=%d, H=%d (G_a=%d, G_a=%d x G_b=%d, G_b=%d, expand=%s)",
                new_id,
                new_node.num_groups,
                new_node.num_nodes,
                leaf1_node.num_groups,
                leaf1_node.num_nodes,
                leaf2_node.num_groups,
                leaf2_node.num_nodes,
                not is_disjoint,
            )
            # ProductLeafLayer/ProductWeights: G side-1 major (g = g_a * G_b + g_b)
            # when enumerated; a single shared factor when groups are tied.
            # U shared on overlay (diagonal), side-tagged when one side broadcasts.
            g_a, u_a = leaf1_node.num_groups, leaf1_node.num_nodes
            g_b, u_b = leaf2_node.num_groups, leaf2_node.num_nodes
            g_factors = ((_SIDE1, g_a), (_SIDE2, g_b))
            if not is_disjoint:
                u_factors = ((_SIDE1, u_a), (_SIDE2, u_b))
            elif u_a == 1 and u_b == 1:
                u_factors = ((_SHARED, 1),)
            elif u_a == 1:
                u_factors = ((_SIDE2, u_b),)
            elif u_b == 1:
                u_factors = ((_SIDE1, u_a),)
            else:
                u_factors = ((_SHARED, u_a),)
            cmap: CoordMap = (
                g_factors,
                u_factors,
            )
            return new_id, new_v_id, cmap
        # Case 3: Deferred product
        if res := check_deferred_product(v1_id, v2_id):
            v2_big, match_is_left, matched_child_id, unmatched_child_id = res
            _mlog(
                "-> Case 3 deferred: v2_big=%s match_is_left=%s",
                v2_big,
                match_is_left,
            )

            if v2_big:
                recurse_res, recurse_v_id, child_cmap = recurse(v1_id, matched_child_id)
                copied_res, copied_v_id = copy_subcircuit(ac2, unmatched_child_id)
            else:
                recurse_res, recurse_v_id, child_cmap = recurse(matched_child_id, v2_id)
                copied_res, copied_v_id = copy_subcircuit(ac1, unmatched_child_id)

            if match_is_left:
                l_new_id, r_new_id = recurse_res, copied_res
                l_new_v_id, r_new_v_id = recurse_v_id, copied_v_id
            else:
                l_new_id, r_new_id = copied_res, recurse_res
                l_new_v_id, r_new_v_id = copied_v_id, recurse_v_id

            l_new_node = new_ac.get_node_data(l_new_id)
            r_new_node = new_ac.get_node_data(r_new_id)

            G_L, H_L, G_R, H_R = (
                l_new_node.num_groups,
                l_new_node.num_nodes,
                r_new_node.num_groups,
                r_new_node.num_nodes,
            )

            _mlog(
                "-> Case 3 deferred children: G_L=%d, H_L=%d x G_R=%d, H_R=%d",
                G_L,
                H_L,
                G_R,
                H_R,
            )

            small_ac = ac1 if v2_big else ac2
            small_v_id = v1_id if v2_big else v2_id
            big_ac = ac2 if v2_big else ac1
            big_v_id = v2_id if v2_big else v1_id

            small_node = small_ac.get_node_data(small_ac.vtree_to_sum[small_v_id])
            G_s, U_s = small_node.num_groups, small_node.num_nodes

            big_node = big_ac.get_node_data(big_ac.vtree_to_sum[big_v_id])
            G_b, U_b = big_node.num_groups, big_node.num_nodes
            w_big = big_node.log_weights

            res_node = new_ac.get_node_data(recurse_res)
            # Weights are authoritative for grid dims (node metadata can disagree).
            res_lw = getattr(res_node, "log_weights", None)
            if res_lw is not None and hasattr(res_lw, "shape"):
                G_res, U_res = res_lw.shape[0], res_lw.shape[1]
            else:
                G_res, U_res = res_node.num_groups, res_node.num_nodes
            G_m = w_big.shape[2] if match_is_left else w_big.shape[4]
            U_m = w_big.shape[3] if match_is_left else w_big.shape[5]
            assert G_res % G_m == 0 and U_res % U_m == 0, (
                f"recursion result grid (G_res={G_res}, U_res={U_res}) is not a "
                f"multiple of the matched slot (G_m={G_m}, U_m={U_m})"
            )
            # Accumulated whole-side factor at the matched slot:
            # G_res = G_m * G_x, U_res = U_m * U_x. The split is read off the child's
            # coordinate map (matched/free factors), never inferred from dimensions:
            # coincidentally equal dims pick the wrong lock and silently drop mass.
            g_factors, u_factors = child_cmap
            g_prod = 1
            for _, n in g_factors:
                g_prod *= n
            u_prod = 1
            for _, n in u_factors:
                u_prod *= n
            assert (g_prod, u_prod) == (G_res, U_res), (
                f"child coordinate map {child_cmap} is inconsistent with the "
                f"recursion result grid ({G_res}, {U_res})"
            )
            # big's matched child is operand 2 of the child recursion iff v2_big.
            matched_side = _SIDE2 if v2_big else _SIDE1
            small_side = _SIDE1 if v2_big else _SIDE2
            s_m_g, lock_g = _case3_axis_lock(
                "G", G_res, g_factors, matched_side, small_side, G_m, G_s
            )
            s_m_u, lock_u = _case3_axis_lock(
                "U", U_res, u_factors, matched_side, small_side, U_m, U_s
            )
            # Weights carry over unchanged only when the result grid IS the matched
            # slot (single matched factor per axis => s_m == arange) and the small
            # side is trivial (zero ties). Otherwise the locked branch below must
            # materialize the re-index / shared-unit tie.
            identity = (
                (G_s, U_s) == (1, 1)
                and len(g_factors) <= 1
                and len(u_factors) <= 1
                and all(s in (_SHARED, matched_side) for s, _ in g_factors)
                and all(s in (_SHARED, matched_side) for s, _ in u_factors)
            )

            _mlog(
                "-> Case 3 deferred sizes: G_b=%d, U_b=%d, G_s=%d, U_s=%d, G_m=%d, U_m=%d, G_res=%d, U_res=%d, cmap=%s -> G=%d, U=%d",
                G_b,
                U_b,
                G_s,
                U_s,
                G_m,
                U_m,
                G_res,
                U_res,
                child_cmap,
                G_s * G_b,
                U_s * U_b,
            )

            new_node = copy.copy(big_node)

            if identity:
                # No free factor and trivial whole side: result grid IS the matched
                # slot; the locks are identity/zero ties and the weights carry over.
                new_node.log_weights = w_big
            else:
                w_lw = w_big.log_weights
                if hasattr(w_big, "_zero_pattern"):
                    zp = w_big._zero_pattern()
                elif hasattr(w_big, "mask"):
                    zp = w_big.mask == 0
                else:
                    zp = None
                dev = w_lw.device
                gb = torch.arange(G_s * G_b, device=dev) // G_s
                gs = torch.arange(G_s * G_b, device=dev) % G_s
                ub = torch.arange(U_s * U_b, device=dev) // U_s
                us = torch.arange(U_s * U_b, device=dev) % U_s
                # Matched-slot re-index and whole-side lock, from the coordinate map.
                s_m_g, s_m_u = s_m_g.to(dev), s_m_u.to(dev)
                lock_g, lock_u = lock_g.to(dev), lock_u.to(dev)

                w_exp = w_lw[gb][:, ub]  # [G', U', G_L, L, G_R, R]
                if zp is not None:
                    zp_exp = zp[gb][:, ub]
                gs_b = gs.view(-1, 1, 1, 1, 1, 1)
                us_b = us.view(1, -1, 1, 1, 1, 1)
                if match_is_left:
                    w_exp = w_exp.index_select(2, s_m_g).index_select(3, s_m_u)
                    if zp is not None:
                        zp_exp = zp_exp.index_select(2, s_m_g).index_select(3, s_m_u)
                    sx_b = lock_g.view(1, 1, -1, 1, 1, 1)
                    su_b = lock_u.view(1, 1, 1, -1, 1, 1)
                else:
                    w_exp = w_exp.index_select(4, s_m_g).index_select(5, s_m_u)
                    if zp is not None:
                        zp_exp = zp_exp.index_select(4, s_m_g).index_select(5, s_m_u)
                    sx_b = lock_g.view(1, 1, 1, 1, -1, 1)
                    su_b = lock_u.view(1, 1, 1, 1, 1, -1)
                # Zero-tie locks (trivial small grid) impose no constraint.
                pen = (gs_b != sx_b) | (us_b != su_b)
                logw = torch.where(pen, torch.full_like(w_exp, LOG_ZERO), w_exp)
                alive = ~pen
                if zp is not None:
                    alive = alive & ~zp_exp
                new_node.log_weights = SparseWeights(logw, alive.to(torch.float32))
            new_node.num_groups = G_s * G_b
            new_node.num_nodes = U_s * U_b
            new_node.scope = big_node.scope.union(small_node.scope)
            new_node.md_set = _prod_md_set(big_node.md_set, small_node.md_set, common_scope)

            new_id = new_ac.add_node(new_node)
            new_v_node = VNode(
                scope=new_node.scope,
                md_set=new_node.md_set,
            )
            new_v_id = new_ac.vtree.add_node(new_v_node)
            new_ac.add_edge(new_id, l_new_id)
            new_ac.add_edge(new_id, r_new_id)
            new_ac.vtree.add_children(new_v_id, l_new_v_id, r_new_v_id)
            new_ac.sum_to_vtree[new_id] = new_v_id
            new_ac.vtree_to_sum[new_v_id] = new_id
            # New node's parent grid is big-major (g = g_b * G_s + g_s); big is
            # operand 2 of this level's call iff v2_big.
            big_side = _SIDE2 if v2_big else _SIDE1
            cmap: CoordMap = (
                ((big_side, G_b), (small_side, G_s)),
                ((big_side, U_b), (small_side, U_s)),
            )
            return new_id, new_v_id, cmap

        # Case 4: Matching children
        if res := check_matching_children(v1_id, v2_id):
            _mlog(f"-> Case 4 matching children: res={res}")
            s1_id, s2_id = ac1.vtree_to_sum[v1_id], ac2.vtree_to_sum[v2_id]

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

            G_1, G_2 = s1_node.num_groups, s2_node.num_groups
            H_1, H_2 = s1_node.num_nodes, s2_node.num_nodes
            G_L1, G_L2 = l1_node.num_groups, l2_node.num_groups
            G_R1, G_R2 = r1_node.num_groups, r2_node.num_groups
            H_L1, H_L2 = l1_node.num_nodes, l2_node.num_nodes
            H_R1, H_R2 = r1_node.num_nodes, r2_node.num_nodes

            l_new_id, l_new_v_id, l_cmap = recurse(*res[0])
            r_new_id, r_new_v_id, r_cmap = recurse(*res[1])

            l_new_node = new_ac.get_node_data(l_new_id)
            r_new_node = new_ac.get_node_data(r_new_id)

            U_disjoint = not v1_node.md_set.is_universal and not v2_node.md_set.is_universal
            L_disjoint = not v1_l_node.md_set.is_universal and not v2_l_node.md_set.is_universal
            R_disjoint = not v1_r_node.md_set.is_universal and not v2_r_node.md_set.is_universal

            def _axis_expand(child_g, child_u, g1, g2, u1, u2, md_expand, axis):
                prod_g, max_g = g1 * g2, max(g1, g2)
                if prod_g != max_g:
                    if child_g == prod_g:
                        e_g = True
                    elif child_g == max_g:
                        e_g = False
                    else:
                        raise CircuitCompilationError(
                            f"Case 4 {axis} child grid G={child_g} matches neither "
                            f"the expanded ({prod_g}) nor the shared ({max_g}) layout."
                        )
                else:
                    e_g = md_expand
                prod_u, max_u = u1 * u2, max(u1, u2)
                if prod_u != max_u:
                    if child_u == prod_u:
                        e_u = True
                    elif child_u == max_u:
                        e_u = False
                    else:
                        raise CircuitCompilationError(
                            f"Case 4 {axis} child grid U={child_u} matches neither "
                            f"the expanded ({prod_u}) nor the shared ({max_u}) layout."
                        )
                else:
                    e_u = md_expand
                return e_g, e_u

            e_gl, e_lu = _axis_expand(
                l_new_node.num_groups,
                l_new_node.num_nodes,
                G_L1,
                G_L2,
                H_L1,
                H_L2,
                not L_disjoint,
                "left",
            )
            e_gr, e_ru = _axis_expand(
                r_new_node.num_groups,
                r_new_node.num_nodes,
                G_R1,
                G_R2,
                H_R1,
                H_R2,
                not R_disjoint,
                "right",
            )

            pw = ProductWeights(
                s1_node.log_weights,
                s2_node.log_weights,
                expand_U=not U_disjoint,
                expand_L=e_gl and e_lu,
                expand_R=e_gr and e_ru,
                expand_GL=e_gl,
                expand_GR=e_gr,
                expand_Lu=e_lu,
                expand_Ru=e_ru,
            )

            def _raw_weights(w):
                return w.log_weights if hasattr(w, "log_weights") else w

            def _side_axis_maps(factors, size, side1_dim, side2_dim):
                """(side1_coords, side2_coords) index tensors for one child axis.

                Returns None when the axis is not in a clean operand-tagged
                form or does not cover the operand's child dims exactly.
                """
                if len(factors) == 2 and {f[0] for f in factors} == {_SIDE1, _SIDE2}:
                    i1 = 0 if factors[0][0] == _SIDE1 else 1
                    c1 = _factor_coord(size, factors, i1)
                    c2 = _factor_coord(size, factors, 1 - i1)
                elif len(factors) == 1:
                    (s, _n) = factors[0]
                    if s == _SHARED:
                        c1 = c2 = _factor_coord(size, factors, 0)
                    elif s == _SIDE1:
                        c1 = _factor_coord(size, factors, 0)
                        c2 = torch.zeros(size, dtype=torch.long)
                    else:
                        c1 = torch.zeros(size, dtype=torch.long)
                        c2 = _factor_coord(size, factors, 0)
                else:
                    return None
                if size == 0 or int(c1.max()) + 1 != side1_dim or int(c2.max()) + 1 != side2_dim:
                    return None
                return c1, c2

            l_cmap_g, l_cmap_u = l_cmap
            r_cmap_g, r_cmap_u = r_cmap
            u_shared = (
                len(l_cmap_u) == 1
                and len(r_cmap_u) == 1
                and l_cmap_u[0][0] == _SHARED
                and r_cmap_u[0][0] == _SHARED
            )
            exactly_one_shared_u = (len(l_cmap_u) == 1 and l_cmap_u[0][0] == _SHARED) != (
                len(r_cmap_u) == 1 and r_cmap_u[0][0] == _SHARED
            )

            composed_w = None
            if not exactly_one_shared_u:
                lm_g = _side_axis_maps(l_cmap_g, l_new_node.num_groups, G_L1, G_L2)
                lm_u = _side_axis_maps(l_cmap_u, l_new_node.num_nodes, H_L1, H_L2)
                rm_g = _side_axis_maps(r_cmap_g, r_new_node.num_groups, G_R1, G_R2)
                rm_u = _side_axis_maps(r_cmap_u, r_new_node.num_nodes, H_R1, H_R2)
                if None not in (lm_g, lm_u, rm_g, rm_u):
                    t1 = _raw_weights(s1_node.log_weights)
                    t2 = _raw_weights(s2_node.log_weights)
                    dev = t1.device
                    W1 = (
                        t1.index_select(2, lm_g[0].to(dev))
                        .index_select(3, lm_u[0].to(dev))
                        .index_select(4, rm_g[0].to(dev))
                        .index_select(5, rm_u[0].to(dev))
                    )
                    W2 = (
                        t2.index_select(2, lm_g[1].to(dev))
                        .index_select(3, lm_u[1].to(dev))
                        .index_select(4, rm_g[1].to(dev))
                        .index_select(5, rm_u[1].to(dev))
                    )
                    if u_shared:
                        composed_w = (W1[:, None] + W2[None, :]).reshape(
                            G_1 * G_2, H_1, *W1.shape[2:]
                        )
                    else:
                        composed_w = (W1[:, None, :, None] + W2[None, :, None, :]).reshape(
                            G_1 * G_2, H_1 * H_2, *W1.shape[2:]
                        )

            expand_U = not u_shared if composed_w is not None else pw.expand_U

            G_new = G_1 * G_2
            H_new = (H_1 * H_2) if expand_U else H_1
            G_L_new = G_L1 * G_L2
            H_L_new = (H_L1 * H_L2) if pw.expand_L else H_L1
            G_R_new = G_R1 * G_R2
            H_R_new = (H_R1 * H_R2) if pw.expand_R else H_R1

            _mlog(
                "-> Case 4 weights: U_disjoint=%s, L_disjoint=%s, R_disjoint=%s, "
                "composed=%s -> G=%d, H=%d, G_L=%d, H_L=%d, G_R=%d, H_R=%d",
                U_disjoint,
                L_disjoint,
                R_disjoint,
                composed_w is not None,
                G_new,
                H_new,
                G_L_new,
                H_L_new,
                G_R_new,
                H_R_new,
            )

            new_node = SumLayer(
                scope=v1_node.scope.union(v2_node.scope),
                md_set=_prod_md_set(v1_node.md_set, v2_node.md_set, common_scope),
                num_groups=G_new,
                num_nodes=H_new,
            )
            new_node.log_weights = SparseWeights(composed_w) if composed_w is not None else pw
            new_id = new_ac.add_node(new_node)
            new_v_node = VNode(
                scope=v1_node.scope.union(v2_node.scope),
                md_set=_prod_md_set(v1_node.md_set, v2_node.md_set, common_scope),
            )
            new_v_id = new_ac.vtree.add_node(new_v_node)
            new_ac.add_edge(new_id, l_new_id)
            new_ac.add_edge(new_id, r_new_id)
            new_ac.vtree.add_children(new_v_id, l_new_v_id, r_new_v_id)
            new_ac.sum_to_vtree[new_id] = new_v_id
            new_ac.vtree_to_sum[new_v_id] = new_id
            # G side-1 major (g = g1 * G2 + g2); U shared on overlay (diagonal),
            # side-tagged when one side's unit axis broadcasts.
            if expand_U:
                u_factors = ((_SIDE1, H_1), (_SIDE2, H_2))
            elif H_1 == 1 and H_2 == 1:
                u_factors = ((_SHARED, 1),)
            elif H_1 == 1:
                u_factors = ((_SIDE2, H_2),)
            elif H_2 == 1:
                u_factors = ((_SIDE1, H_1),)
            else:
                u_factors = ((_SHARED, H_1),)
            cmap: CoordMap = (
                ((_SIDE1, G_1), (_SIDE2, G_2)),
                u_factors,
            )
            return new_id, new_v_id, cmap

        raise CircuitCompilationError(
            f"Incompatible circuits for multiplication at vtree nodes {v1_id} and {v2_id}."
        )

    recurse = _depth_traced(recurse)

    ac1_roots, ac2_roots = ac1.get_roots(), ac2.get_roots()
    if not ac1_roots or not ac2_roots:
        return new_ac

    v1_root, v2_root = ac1.sum_to_vtree[ac1_roots[0]], ac2.sum_to_vtree[ac2_roots[0]]
    recurse(v1_root, v2_root)  # coordinate map is not needed at the top level
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
    var_to_id: Dict[str, int],
) -> Tuple[SymbolicArithmeticCircuit, int]:
    """
    Compiles an EstimandAST into a SymbolicArithmeticCircuit by recursively
    applying the functional transformations.
    """

    def _build_recursive(ast_node_id: int, recursion_level: int = 0) -> SymbolicArithmeticCircuit:
        ast_node = ast.get_node_data(ast_node_id)

        if isinstance(ast_node, PNode):
            logger.debug(f"{'  ' * recursion_level}PNode")
            return base_ac

        elif isinstance(ast_node, MargNode):
            logger.debug(f"{'  ' * recursion_level}MARG({ast_node.marginalize_vars})")
            child_id = ast.get_children(ast_node_id)[0]
            child_ac = _build_recursive(child_id, recursion_level + 1)
            # Map string var names to int IDs, or keep as int if already int
            marg_ids = {var_to_id[v] if v in var_to_id else v for v in ast_node.marginalize_vars}
            return _marginalize(child_ac, marg_ids)

        elif isinstance(ast_node, ProdNode):
            logger.debug(f"{'  ' * recursion_level}PROD")
            children_ids = ast.get_children(ast_node_id)
            if not children_ids:
                empty_ac = SymbolicArithmeticCircuit()
                return empty_ac

            children_ids = sorted(
                children_ids,
                key=lambda cid: (
                    0 if isinstance(ast.get_node_data(cid), CondNode) else 1,
                    str(ast.get_node_data(cid)),
                ),
            )

            acc_ac = _build_recursive(children_ids[0], recursion_level + 1)
            for cid in children_ids[1:]:
                next_ac = _build_recursive(cid, recursion_level + 1)
                acc_ac = _multiply(acc_ac, next_ac)
            return acc_ac

        elif isinstance(ast_node, InstNode):
            logger.debug(f"{'  ' * recursion_level}INST({ast_node.variables})")
            child_id = ast.get_children(ast_node_id)[0]
            child_ac = _build_recursive(child_id, recursion_level + 1)
            insts = {
                var_to_id[v] if v in var_to_id else v: val for v, val in ast_node.variables.items()
            }
            return _instantiate(child_ac, insts)

        elif isinstance(ast_node, ConstantNode):
            logger.debug(f"{'  ' * recursion_level}CONST")
            ac = SymbolicArithmeticCircuit()
            # Constant 1 leaf
            leaf = ConstantLayer(scope=BitSet(), num_nodes=1, num_groups=1)
            ac.add_node(leaf)
            return ac

        elif isinstance(ast_node, CondNode):
            logger.debug(f"{'  ' * recursion_level}COND(..|{ast_node.den_vars})")
            child_id = ast.get_children(ast_node_id)[0]
            child_ac = _build_recursive(child_id, recursion_level + 1)

            den_ids = {var_to_id[v] if v in var_to_id else v for v in ast_node.den_vars}
            return _conditional(child_ac, den_ids)

        return base_ac

    final_ac = _build_recursive(ast.get_root())
    return final_ac

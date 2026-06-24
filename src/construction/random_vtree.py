import random
from typing import List, Optional, Set

from src.symbolic.vtree import VNode, VTree
from src.utils import BitSet


def _get_conj_len(conj_len: Optional[int], max_len: int) -> int:
    if conj_len is not None:
        return conj_len
    return random.randint(1, max(1, max_len // 2))


def _check_termination(conj_len: Optional[int], scope_size: int) -> bool:
    if conj_len is not None and scope_size <= conj_len + 1:
        return True
    if scope_size <= 1:
        return True
    return False


def construct_random_vtree(
    vars: Set[int], conj_len: Optional[int] = None, top_split_A: Optional[list[int]] = None
) -> VTree:
    """Constructs a random binary variable tree (vtree) for the given number of variables.

    Args:
        vars (set[int]): The set of variables.
        conj_len (Optional[int]): The number of variables to consider at each split. If None, is sampled randomly at each split.
        top_split_A (Optional[list[int]]): If provided, the root node will split `vars` into `top_split_A` and the rest.
    """
    vt = VTree()

    if top_split_A is not None:
        conj_vars = list(top_split_A)
        rest_vars = list(set(vars) - set(top_split_A))
        random.shuffle(conj_vars)
        random.shuffle(rest_vars)
        ordering = conj_vars + rest_vars
    else:
        ordering = list(vars)
        random.shuffle(ordering)

    root_scope = BitSet(ordering)
    root_node = VNode(scope=root_scope)
    root_id = vt.add_node(root_node)

    if top_split_A is not None:
        left_scope = BitSet(conj_vars)
        left_vnode = VNode(scope=left_scope)
        left_id = vt.add_node(left_vnode)

        right_scope = BitSet(rest_vars)
        right_vnode = VNode(scope=right_scope)
        right_id = vt.add_node(right_vnode)

        vt.add_children(root_id, left_id, right_id)

        remaining = [(left_id, left_vnode, conj_vars), (right_id, right_vnode, rest_vars)]
    else:
        remaining = [(root_id, root_node, ordering)]

    while remaining:
        curr_id, curr_vnode, curr_scope_ordered = remaining.pop(0)
        if _check_termination(conj_len, len(curr_scope_ordered)):
            continue

        curr_conj_len = _get_conj_len(conj_len, len(curr_scope_ordered))

        conj_vars = curr_scope_ordered[:curr_conj_len]
        rest_vars = curr_scope_ordered[curr_conj_len:]

        left_scope = BitSet(conj_vars)
        left_vnode = VNode(scope=left_scope)
        left_id = vt.add_node(left_vnode)

        right_scope = BitSet(rest_vars)
        right_vnode = VNode(scope=right_scope)
        right_id = vt.add_node(right_vnode)

        vt.add_children(curr_id, left_id, right_id)

        remaining.append((left_id, left_vnode, conj_vars))
        remaining.append((right_id, right_vnode, rest_vars))

    return vt


def construct_random_md_vtree(
    vars: Set[int],
    md_sets: List[Set[int]],
    conj_len: Optional[int] = None,
    top_split_A: Optional[list[int]] = None,
) -> VTree:
    """Constructs a random binary variable tree and labels it with Marginally Deterministic sets."""
    base_vtree = construct_random_vtree(vars, conj_len, top_split_A)

    base_vtree.compute_md_labeling(md_sets)
    return base_vtree

import random
from typing import List, Optional, Set

from src.symbolic.vtree import MDVTree, VNode, VTree
from src.utils import BitSet, NodeAllocator


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


def construct_random_vtree(vars: Set[int], conj_len: Optional[int] = None) -> VTree:
    """Constructs a random binary variable tree (vtree) for the given number of variables.

    Args:
        vars (set[int]): The set of variables.
        conj_len (Optional[int]): The number of variables to consider at each split. If None, is sampled randomly at each split.
    """
    vt = VTree()

    ordering = list(vars)
    random.shuffle(ordering)

    root_scope = BitSet(ordering)
    root_node = VNode(scope=root_scope)

    vt.add_node(0, root_node)

    allocator = NodeAllocator(start=1)
    remaining = [(0, root_node, ordering)]

    while remaining:
        curr_id, curr_vnode, curr_scope_ordered = remaining.pop(0)
        if _check_termination(conj_len, len(curr_scope_ordered)):
            continue

        curr_conj_len = _get_conj_len(conj_len, len(curr_scope_ordered))

        conj_vars = curr_scope_ordered[:curr_conj_len]
        rest_vars = curr_scope_ordered[curr_conj_len:]

        left_scope = BitSet(conj_vars)
        left_vnode = VNode(scope=left_scope)
        left_id = allocator.next_id()

        right_scope = BitSet(rest_vars)
        right_vnode = VNode(scope=right_scope)
        right_id = allocator.next_id()

        vt.add_node(left_id, left_vnode)
        vt.add_node(right_id, right_vnode)
        vt.add_children(curr_id, left_id, right_id)

        remaining.append((left_id, left_vnode, conj_vars))
        remaining.append((right_id, right_vnode, rest_vars))

    return vt


def construct_random_md_vtree(
    vars: Set[int], max_subset_size: int, conj_len: Optional[int] = None
) -> MDVTree:
    """Constructs a random binary variable tree and labels it with Marginally Deterministic sets."""
    base_vtree = construct_random_vtree(vars, conj_len)

    md_sets = []

    # Choose a single random path down the VTree to ensure increasing sequence of overlapping subsets
    curr_vid = base_vtree.get_root()
    while curr_vid is not None:
        n = base_vtree.get_node_data(curr_vid)
        # Note: the subsets must be small enough
        if len(n.scope) <= max_subset_size:
            md_sets.append(set(n.scope))

        children = base_vtree.get_children_pair(curr_vid)
        if not children:
            break

        # Randomly choose one branch to go down
        curr_vid = random.choice(children)

    return MDVTree.from_vtree(base_vtree, md_sets)

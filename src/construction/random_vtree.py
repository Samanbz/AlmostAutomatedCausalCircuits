import random
from typing import Optional

from src.symbolic.vtree import MDVTree, VNode, VTree
from src.utils import BitSet


def construct_random_vtree(vars: set[int], conj_len: Optional[int] = None) -> VTree:
    """Constructs a random binary variable tree (vtree) for the given number of variables.

    Args:
        vars (set[int]): The set of variables.
        conj_len (Optional[int]): The number of variables to consider at each split. If None, is sampled randomly at each split.
    """

    def get_conj_len(max_len: int) -> int:
        if conj_len is not None:
            return conj_len
        return random.randint(1, max(1, max_len // 2))

    def check_termination(scope_size: int) -> bool:
        if conj_len is not None and scope_size <= conj_len + 1:
            return True
        if scope_size == 1:
            return True
        return False

    vt = VTree()

    ordering = list(vars)
    random.shuffle(ordering)

    root_scope = BitSet(ordering)
    root_node = VNode(scope=root_scope)

    vt.add_node(0, root_node)
    node_counter = 1

    remaining = [(0, root_node, ordering)]

    while remaining:
        curr_id, curr_vnode, curr_scope_ordered = remaining.pop(0)
        if check_termination(len(curr_scope_ordered)):
            continue

        decompositions = []

        # Get or sample conj_len
        curr_conj_len = get_conj_len(len(curr_scope_ordered))

        # Select the next conj_len variables in the original ordering
        conj_vars = curr_scope_ordered[:curr_conj_len]
        decompositions.append(conj_vars)

        decompositions.append(curr_scope_ordered[curr_conj_len:])

        assert len(decompositions) == 2, "Decomposition count mismatch."
        assert sum(len(d) for d in decompositions) == len(curr_vnode.scope), (
            "Decomposed variable count does not match current scope size."
        )

        # Create children - first is left (conj_vars), second is right (rest)
        left_vars, right_vars = decompositions

        # Left child (conjunction variables)
        left_scope = BitSet(left_vars)
        left_vnode = VNode(scope=left_scope)
        left_id = node_counter
        node_counter += 1

        # Right child (remaining variables)
        right_scope = BitSet(right_vars)
        right_vnode = VNode(scope=right_scope)
        right_id = node_counter
        node_counter += 1

        # Add both nodes and children atomically
        vt.add_node(left_id, left_vnode)
        vt.add_node(right_id, right_vnode)
        vt.add_children(curr_id, left_id, right_id)

        remaining.append((left_id, left_vnode, left_vars))
        remaining.append((right_id, right_vnode, right_vars))

    return vt


def construct_random_md_vtree(
    vars: set[int], max_subset_size: int, conj_len: Optional[int] = None
) -> MDVTree:
    """Constructs a random binary variable tree and labels it with Marginally Deterministic sets."""
    base_vtree = construct_random_vtree(vars, conj_len)

    md_sets = []

    def get_maximal(vid: int):
        n = base_vtree.get_node_data(vid)
        if len(n.scope) <= max_subset_size:
            md_sets.append(set(n.scope))
            return

        children = base_vtree.get_children_pair(vid)
        if children:
            get_maximal(children[0])
            get_maximal(children[1])

    get_maximal(base_vtree.get_root())

    return MDVTree.from_vtree(base_vtree, md_sets)

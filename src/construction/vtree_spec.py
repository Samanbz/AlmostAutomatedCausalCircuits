"""Build a VTree from a parenthesized variable-name spec (Newick-style).

Config-pinned vtrees for experiments whose COAST kernels require a specific
decomposition (e.g. colliderdoor's ``((Z,X),(M,Y))`` so that the {Z,X}
determinism side and the {M,Y} density side sit under one pseudo-mixing root).
The MD labeling is computed afterwards by ``VTree.compute_md_labeling`` — the
spec fixes only the shape, not the labels.
"""

from typing import Dict, Set, Tuple

from src.symbolic.vtree import VNode, VTree
from src.utils import BitSet


def build_vtree_from_spec(spec: str, var_to_id: Dict[str, int]) -> VTree:
    """Parse ``spec`` into a binary VTree over the named variables.

    Args:
        spec: Newick-like string, e.g. ``"((Z,X),(M,Y))"``. Every variable in
            ``var_to_id`` must appear exactly once; internal nodes are binary.
        var_to_id: variable name to column id.

    Returns:
        The shaped VTree (no MD labeling — caller runs ``compute_md_labeling``).
    """
    tokens = spec.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()
    pos = 0
    seen: Set[str] = set()

    def parse() -> Tuple[int, BitSet]:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError(f"Unexpected end of vtree spec: {spec!r}")
        tok = tokens[pos]
        if tok == "(":
            pos += 1
            left, left_scope = parse()
            if pos >= len(tokens) or tokens[pos] != ",":
                raise ValueError(f"Expected ',' in vtree spec: {spec!r}")
            pos += 1
            right, right_scope = parse()
            if pos >= len(tokens) or tokens[pos] != ")":
                raise ValueError(f"Expected ')' in vtree spec: {spec!r}")
            pos += 1
            vid = vt.add_node(VNode(left_scope | right_scope))
            vt.add_children(vid, left, right)
            return vid, left_scope | right_scope
        if tok in {",", ")"}:
            raise ValueError(f"Unexpected {tok!r} in vtree spec: {spec!r}")
        pos += 1
        if tok not in var_to_id:
            raise ValueError(f"Unknown variable {tok!r} in vtree spec (have {sorted(var_to_id)})")
        if tok in seen:
            raise ValueError(f"Variable {tok!r} appears twice in vtree spec: {spec!r}")
        seen.add(tok)
        return vt.add_node(VNode(BitSet([var_to_id[tok]]))), BitSet([var_to_id[tok]])

    vt = VTree()
    _root, root_scope = parse()
    if pos != len(tokens):
        raise ValueError(f"Trailing tokens in vtree spec: {spec!r}")
    missing = set(var_to_id) - set(seen)
    if missing:
        raise ValueError(f"vtree spec {spec!r} is missing variables: {sorted(missing)}")
    return vt

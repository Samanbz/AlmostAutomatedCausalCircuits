"""Tests for Newick-style VTree specs (``src/construction/vtree_spec.py``)."""

import pytest

from src.construction.vtree_spec import build_vtree_from_spec


VAR_TO_ID = {"Z": 0, "X": 1, "M": 2, "Y": 3}


def _scopes(vt):
    return {vid: frozenset(vt.get_node_data(vid).scope) for vid in vt.topological_sort()}


def test_shape_matches_variable_partition():
    vt = build_vtree_from_spec("((Z,X),(M,Y))", VAR_TO_ID)
    scopes = set(_scopes(vt).values())
    root = vt.get_node_data(vt.get_root())
    assert len(root.scope) == 4
    assert frozenset([0, 1]) in scopes  # {Z,X}
    assert frozenset([2, 3]) in scopes  # {M,Y}
    assert all(frozenset([i]) in scopes for i in range(4))


def test_md_labeling_from_spec_matches_manual_construction():
    from src.symbolic.vtree import VNode, VTree
    from src.utils import BitSet

    vt = build_vtree_from_spec("((Z,X),(M,Y))", VAR_TO_ID)
    md_sets = [{0, 1}, {0, 1, 2}]
    vt.compute_md_labeling(md_sets)

    manual = VTree()
    v_z = manual.add_node(VNode(BitSet([0])))
    v_x = manual.add_node(VNode(BitSet([1])))
    v_m = manual.add_node(VNode(BitSet([2])))
    v_y = manual.add_node(VNode(BitSet([3])))
    v_zx = manual.add_node(VNode(BitSet([0, 1])))
    manual.add_children(v_zx, v_z, v_x)
    v_my = manual.add_node(VNode(BitSet([2, 3])))
    manual.add_children(v_my, v_m, v_y)
    v_root = manual.add_node(VNode(BitSet([0, 1, 2, 3])))
    manual.add_children(v_root, v_zx, v_my)
    manual.compute_md_labeling(md_sets)

    def _labels(v):
        return {
            frozenset(v.get_node_data(k).scope): v.get_node_data(k).md_set
            for k in v.topological_sort()
        }

    assert _labels(vt) == _labels(manual)


@pytest.mark.parametrize(
    "spec",
    [
        "((Z,X),(M,Z))",  # duplicate variable
        "((Z,X),(M))",  # non-binary internal node
        "((Z,X),(M,Q))",  # unknown variable
        "((Z,X),(M,Y)",  # unbalanced parens
        "((Z,X),(M,Y)),",  # trailing tokens
        "((Z,X),(M,_))",  # missing Y
    ],
)
def test_invalid_specs_raise(spec):
    with pytest.raises(ValueError):
        build_vtree_from_spec(spec, VAR_TO_ID)

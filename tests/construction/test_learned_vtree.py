import pytest
import torch

from src.construction.learned_vtree import (
    construct_optimal_md_vtree,
    construct_optimal_vtree,
    learn_liang_ps_md_vtree,
)
from src.utils.bitset import BitSet


# 1.1 Structural validations


def test_base_case_execution():
    """Base case: 1-variable and 2-variable vtrees."""
    # 1 Variable
    data_1v = torch.randn(10, 1)
    vt_1 = construct_optimal_vtree(data_1v)
    assert len(vt_1._nodes) == 1

    md_vt_1 = construct_optimal_md_vtree(data_1v, md_sets=[{0}])
    assert len(md_vt_1._nodes) == 1

    # 2 Variables
    data_2v = torch.randn(10, 2)
    vt_2 = construct_optimal_vtree(data_2v)
    assert len(vt_2._nodes) == 3  # Root + 2 leaves

    md_vt_2 = construct_optimal_md_vtree(data_2v, md_sets=[{0}])
    assert len(md_vt_2._nodes) == 3


def test_root_identification(complex_vtree):
    """Root: single node with no incoming edges spanning all variables."""
    root_id = complex_vtree.get_root()
    parents = complex_vtree.get_parents(root_id)
    assert len(parents) == 0
    assert complex_vtree.get_node_data(root_id).scope == BitSet({0, 1, 2, 3})


def test_md_set_universality(synthetic_data):
    """MD labels propagate to ancestors: a non-universal node's md-set must be
    either the union of its children's md-sets or equal to one child's."""
    md_vt = construct_optimal_md_vtree(synthetic_data, md_sets=[{0}, {0, 1}])

    for vid in md_vt.topological_sort():
        node = md_vt.get_node_data(vid)
        children = md_vt.get_children_pair(vid)
        if children is None:
            continue

        l_vid, r_vid = children
        l_child = md_vt.get_node_data(l_vid)
        r_child = md_vt.get_node_data(r_vid)

        if node.md_set.is_universal:
            continue

        assert (
            node.md_set == l_child.md_set.union(r_child.md_set)
            or node.md_set == l_child.md_set
            or node.md_set == r_child.md_set
        ), f"Invalid MD-Set propagation at node {vid}"


# 1.2 Edge cases


def test_disjoint_md_sets_validation(synthetic_data):
    """MD sets must be closed under intersection."""
    with pytest.raises(ValueError, match="closed under intersection"):
        construct_optimal_md_vtree(synthetic_data, md_sets=[{0}, {1}, {2}])


# 1.3 PS-root MD vtree learner (learn_liang_ps_md_vtree)


def _frontdoor_like_data(n=2000, seed=0):
    """Columns (X, M, Y): X -> M -> Y with confounding on (X, Y)."""
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, generator=g)
    x = u + torch.randn(n, generator=g)
    m = 1.5 * x + torch.randn(n, generator=g)
    y = m + 2.0 * u + torch.randn(n, generator=g)
    return torch.stack([x, m, y], dim=1)


def _layer_type(vt, vid):
    from src.construction.circuit_builder import LayerType

    children = vt.get_children_pair(vid)
    if children is None:
        return None
    node = vt.get_node_data(vid)
    l_vid, r_vid = children
    return LayerType.from_md_sets(
        node.md_set, vt.get_node_data(l_vid).md_set, vt.get_node_data(r_vid).md_set
    )


def _ps_layers(vt):
    from src.construction.circuit_builder import LayerType

    return [
        vid
        for vid in vt.topological_sort()
        if _layer_type(vt, vid) in (LayerType.PS_LEFT_MIXING, LayerType.PS_RIGHT_MIXING)
    ]


def test_ps_md_vtree_root_ps_and_y_on_density_side():
    """Frontdoor family [{M,X},{X}]: root must be a left pseudo-mixing layer with
    X on the selector side and Y on the density (right) side; a single PS layer."""
    data = _frontdoor_like_data()
    vt = learn_liang_ps_md_vtree(data, md_sets=[{0, 1}, {0}], y_vars={2})

    root_id = vt.get_root()
    l_vid, r_vid = vt.get_children_pair(root_id)
    assert set(vt.get_node_data(l_vid).scope) == {0}  # selector side = {X}
    assert 2 in set(vt.get_node_data(r_vid).scope)  # Y on the density side

    ps = _ps_layers(vt)
    assert ps == [root_id], f"expected only the root PS layer, got {ps}"


def test_ps_md_vtree_warns_on_second_ps_layer(caplog):
    """An inclusion chain of determinisms ({M,Z,X} > {M,X} > {X}) forces PS
    layers below the root — a warning must be logged."""
    import logging

    g = torch.Generator().manual_seed(0)
    x = torch.randn(2000, generator=g)
    z = x + torch.randn(2000, generator=g)
    m = z + x + torch.randn(2000, generator=g)
    y = m + z + torch.randn(2000, generator=g)
    data = torch.stack([x, m, z, y], dim=1)

    with caplog.at_level(logging.WARNING, logger="mcc.learned_vtree"):
        vt = learn_liang_ps_md_vtree(data, md_sets=[{0, 1, 2}, {0, 1}, {0}], y_vars={3})

    assert len(_ps_layers(vt)) >= 2
    assert any(
        "pseudo-mixing" in r.message and "will not be normalized" in r.message
        for r in caplog.records
    )


def test_ps_md_vtree_singleton_family_is_classical_at_root(caplog):
    """A single-determinism family yields a classical disjoint mixing layer at
    the root (no pseudo-mixing needed) — logged at info, no warning."""
    import logging

    data = _frontdoor_like_data()
    with caplog.at_level(logging.INFO, logger="mcc.learned_vtree"):
        vt = learn_liang_ps_md_vtree(data, md_sets=[{0}], y_vars={2})
    assert not _ps_layers(vt)
    assert any("classical disjoint mixing layer" in r.message for r in caplog.records)


def test_ps_md_vtree_rejects_outcomes_in_md_set():
    data = _frontdoor_like_data()
    with pytest.raises(ValueError, match="outcome"):
        learn_liang_ps_md_vtree(data, md_sets=[{0, 2}, {0}], y_vars={2})


def test_ps_md_vtree_rejects_non_closed_family():
    data = _frontdoor_like_data()
    with pytest.raises(ValueError, match="closed under intersection"):
        learn_liang_ps_md_vtree(data, md_sets=[{0}, {1}], y_vars={2})

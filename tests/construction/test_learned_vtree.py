import networkx as nx
import pytest
import torch

from src.construction.learned_vtree import (
    LearnedVTreeBuilder,
    apply_md_constraints,
    build_skeleton,
    construct_optimal_md_vtree,
    construct_optimal_vtree,
)
from src.utils.bitset import BitSet


# 1.1 Structural Validations


def test_full_connectedness():
    """1.1.1 Full Connectedness: Verify that a vtree built without a provided DAG has a complete skeleton graph."""
    num_vars = 4
    skeleton = build_skeleton(None, num_vars)
    assert skeleton.number_of_nodes() == num_vars
    assert skeleton.number_of_edges() == (num_vars * (num_vars - 1)) // 2
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            assert skeleton.has_edge(i, j)


def test_moralized_skeleton():
    """1.1.2 Moralized Skeleton: Verify that if a DAG is provided, the skeleton graph adds an undirected edge between parents."""
    dag = nx.DiGraph()
    dag.add_edges_from([(0, 2), (1, 2)])
    skeleton = build_skeleton(dag, 3)

    assert skeleton.has_edge(0, 2)
    assert skeleton.has_edge(1, 2)
    # Check for moralized edge
    assert skeleton.has_edge(0, 1)


def test_base_case_execution():
    """1.1.3 Base Case Execution: Test construct_optimal_vtree and construct_optimal_md_vtree on trivial inputs (1 variable, 2 variables)."""
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
    """1.1.4 Root Identification: Verify that md_vtree.get_root() correctly identifies the single node with no incoming edges."""
    root_id = complex_vtree.get_root()
    parents = complex_vtree.get_parents(root_id)
    assert len(parents) == 0
    assert complex_vtree.get_node_data(root_id).scope == BitSet({0, 1, 2, 3})


def test_md_set_universality(synthetic_data):
    """1.1.5 MD-Set Universality: Check that MD sets properly propagate to ancestors.
    If a parent is not universal, its MD-set must be either the union of its children's MD-sets
    or equal to the MD-set of its left/right child."""
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


# 1.2 MD Constraints mapping


def test_md_set_uncuttable_edges():
    """1.2.1 MD-Set Uncuttable Edges: Verify that edges in the skeleton receive the designated md_weight."""
    skeleton = build_skeleton(None, 3)
    apply_md_constraints(skeleton, md_sets=[{0, 1}], md_weight=1e9)

    assert skeleton[0][1]["weight"] == 1e9
    # The edge (1, 2) should not have the high weight constraint initially
    assert "weight" not in skeleton[1][2] or skeleton[1][2]["weight"] != 1e9


def test_joint_constraint_satisfaction(synthetic_data):
    """1.2.2 Joint Constraint Satisfaction: Verify that variables in a joint MD-set are never partitioned
    into separate sub-trees until they are explicitly resolved as a joint MD-set at a specific node."""
    md_vt = construct_optimal_md_vtree(synthetic_data, md_sets=[{0, 1}])

    # Trace the tree from the root. If a node contains {0, 1} in its scope, it should not split 0 and 1
    # into different children, EXCEPT when that node itself is evaluating exactly the md-set.
    # In practice, the high weight guarantees they stay together in the VTree until the very end.

    for vid in md_vt.topological_sort():
        node = md_vt.get_node_data(vid)
        children = md_vt.get_children_pair(vid)
        if children is None:
            continue

        l_vid, r_vid = children
        l_child = md_vt.get_node_data(l_vid)
        r_child = md_vt.get_node_data(r_vid)

        # If both variables are in the parent scope, check how they split
        if 0 in node.scope and 1 in node.scope:
            l_has_one = (0 in l_child.scope) ^ (1 in l_child.scope)
            r_has_one = (0 in r_child.scope) ^ (1 in r_child.scope)

            # They should only be split if this node is resolving the {0, 1} group.
            # But wait, any split of {0, 1} means they are separated.
            # Since their mutual edge weight is 1e9, they should be the VERY LAST things split.
            if l_has_one or r_has_one:
                # This must be the node where their scope is exactly {0, 1}
                assert node.scope == BitSet({0, 1})


# 1.3 Edge Cases


def test_disjoint_md_sets_validation(synthetic_data):
    """1.3.1 Disjoint MD Sets Validation: Test md_sets=[{0}, {1}, {2}].
    The builder should explicitly raise a ValueError because MD sets must be
    closed under union."""
    with pytest.raises(ValueError, match="closed under intersection"):
        construct_optimal_md_vtree(synthetic_data, md_sets=[{0}, {1}, {2}])


def test_bisection_fallback():
    """1.3.3 Bisection Fallback: Test behavior when a graph partitioning algorithm fails to find a split."""
    builder = LearnedVTreeBuilder()
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    # Fully connected with 1e9 weight
    G.add_edge(0, 1, weight=1e9)
    G.add_edge(1, 2, weight=1e9)
    G.add_edge(0, 2, weight=1e9)

    # Bisection should fallback to arbitrary split
    l_vars, r_vars = builder._bisect_graph(G, [0, 1, 2])
    assert len(l_vars) > 0 and len(r_vars) > 0
    assert set(l_vars).union(set(r_vars)) == {0, 1, 2}
    assert set(l_vars).intersection(set(r_vars)) == set()

import pytest
import torch
from src.construction.random_scm import generate_random_scm

from src.symbolic import (
    GaussianDistribution,
    KroneckerProductNode,
    LeafNode,
    PartitionNode,
    RegionNode,
    SumNode,
    VNode,
    VTree,
)
from src.utils import BitSet


@pytest.fixture
def synthetic_data():
    """A fixture returning a basic 2D tensor of random variables generated from an SCM."""
    scm = generate_random_scm(n_nodes=4, expected_degree=2.0)
    df = scm.sample(100)
    return torch.from_numpy(df.values.copy()).float()


@pytest.fixture
def basic_input_dists():
    """A dictionary mapping variable IDs to standard instantiated GaussianDistributions."""
    return {v: GaussianDistribution(var=v, mean=0.0, stddev=1.0) for v in range(4)}


@pytest.fixture
def trivial_vtree():
    """A pre-constructed 2-variable VTree without MD-sets."""
    vt = VTree()
    n0_id = vt.add_node(VNode(scope=BitSet({0, 1}), md_set=BitSet()))
    n1_id = vt.add_node(VNode(scope=BitSet({0}), md_set=BitSet()))
    n2_id = vt.add_node(VNode(scope=BitSet({1}), md_set=BitSet()))
    vt.add_children(n0_id, n1_id, n2_id)
    return vt


@pytest.fixture
def complex_vtree():
    """A 4-variable VTree with nested MD-sets."""
    vt = VTree()
    n0_id = vt.add_node(VNode(scope=BitSet({0, 1, 2, 3}), md_set=BitSet({0, 1})))

    n1_id = vt.add_node(VNode(scope=BitSet({0, 1}), md_set=BitSet({0, 1})))
    n2_id = vt.add_node(VNode(scope=BitSet({0}), md_set=BitSet({0})))
    n3_id = vt.add_node(VNode(scope=BitSet({1}), md_set=BitSet({1})))
    vt.add_children(n1_id, n2_id, n3_id)

    n4_id = vt.add_node(VNode(scope=BitSet({2, 3}), md_set=BitSet()))
    n5_id = vt.add_node(VNode(scope=BitSet({2}), md_set=BitSet()))
    n6_id = vt.add_node(VNode(scope=BitSet({3}), md_set=BitSet()))
    vt.add_children(n4_id, n5_id, n6_id)

    vt.add_children(n0_id, n1_id, n4_id)
    return vt


def assert_isomorphic(rg, ac, rg_node_id, ac_node_id, visited=None):
    """
    Recursively checks graph equivalence between a given RegionGraph and its compiled
    ArithmeticCircuit counterpart, ensuring identical DAG skeletons.
    """
    if visited is None:
        visited = set()

    if (rg_node_id, ac_node_id) in visited:
        return
    visited.add((rg_node_id, ac_node_id))

    rg_node = rg.get_node_data(rg_node_id)
    ac_node = ac.get_node_data(ac_node_id)

    # Check structural correspondence
    if isinstance(rg_node, RegionNode):
        if not rg.get_children(rg_node_id):
            assert isinstance(ac_node, LeafNode), (
                f"Expected LeafNode for leaf RegionNode {rg_node_id}"
            )
        else:
            assert isinstance(ac_node, SumNode), f"Expected SumNode for RegionNode {rg_node_id}"
    elif isinstance(rg_node, PartitionNode):
        assert isinstance(ac_node, KroneckerProductNode), (
            f"Expected KroneckerProductNode for PartitionNode {rg_node_id}"
        )
    else:
        pytest.fail(f"Unknown RegionGraph node type: {type(rg_node)}")

    # Check scope equivalence
    assert rg_node.scope == ac_node.scope, f"Scope mismatch at RG {rg_node_id} / AC {ac_node_id}"

    # Check children correspondence
    rg_children = sorted(rg.get_children(rg_node_id))
    ac_children = sorted(ac.get_children(ac_node_id))

    assert len(rg_children) == len(ac_children), (
        f"Children count mismatch at RG {rg_node_id} / AC {ac_node_id}"
    )

    for rg_child, ac_child in zip(rg_children, ac_children):
        assert_isomorphic(rg, ac, rg_child, ac_child, visited)


@pytest.fixture(scope="module")
def device():
    """A fixture that returns the appropriate torch device (GPU if available, else CPU)."""
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

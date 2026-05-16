import torch

from src.construction.circuit_builder import MDCircuitBuilder
from src.construction.region_graph_builder import MDRegionGraphBuilder
from src.symbolic import (
    GaussianDistribution,
    HadamardProductNode,
    KroneckerProductNode,
    MarginalDeterminism,
    MDVNode,
    MDVTree,
    SumNode,
    SymbolicArithmeticCircuit,
)
from src.utils import BitSet, ContinuousInterval
from src.utils.support import Support


# 4. Property Validation


def test_single_variable_md_validation(basic_input_dists):
    """4.1.1 Single-Variable MD Validation: Compile an MDNet with md_sets=[{0}].
    Assert spn.check_property(MarginalDeterminism(target_scope={0})) == True."""
    vt = MDVTree()
    vt.add_node(0, MDVNode(scope=BitSet({0, 1}), md_set=BitSet({0})))
    vt.add_node(1, MDVNode(scope=BitSet({0}), md_set=BitSet({0})))
    vt.add_node(2, MDVNode(scope=BitSet({1}), md_set=BitSet({1})))
    vt.add_children(0, 1, 2)

    rg = MDRegionGraphBuilder(basic_input_dists, vt).build()
    ac = MDCircuitBuilder(rg=rg, h=4, input_dists=basic_input_dists).build()

    is_md = ac.check_property(MarginalDeterminism(target_scope={0}))
    assert is_md is True


def test_joint_variable_md_validation(basic_input_dists):
    """4.1.2 Joint-Variable MD Validation: Compile an MDNet with md_sets=[{0, 1}].
    Assert spn.check_property(MarginalDeterminism(target_scope={0, 1})) == True."""
    vt = MDVTree()
    vt.add_node(0, MDVNode(scope=BitSet({0, 1, 2}), md_set=BitSet({0, 1})))
    vt.add_node(1, MDVNode(scope=BitSet({0, 1}), md_set=BitSet({0, 1})))
    vt.add_node(2, MDVNode(scope=BitSet({2}), md_set=BitSet({2})))
    vt.add_children(0, 1, 2)

    vt.add_node(3, MDVNode(scope=BitSet({0}), md_set=BitSet({0})))
    vt.add_node(4, MDVNode(scope=BitSet({1}), md_set=BitSet({1})))
    vt.add_children(1, 3, 4)

    rg = MDRegionGraphBuilder(basic_input_dists, vt).build()
    ac = MDCircuitBuilder(rg=rg, h=4, input_dists=basic_input_dists).build()

    is_md = ac.check_property(MarginalDeterminism(target_scope={0, 1}))
    assert is_md is True


def test_negative_md_validation(basic_input_dists):
    """4.1.3 Negative MD Validation: Compile an MDNet with md_sets=[{0}].
    Assert checking on unconstrained scope {1} returns False."""
    vt = MDVTree()
    vt.add_node(0, MDVNode(scope=BitSet({0, 1}), md_set=BitSet({0})))
    vt.add_node(1, MDVNode(scope=BitSet({0}), md_set=BitSet({0})))
    # Explicitly make Node 2 unconstrained so it duplicates support and causes overlap
    vt.add_node(2, MDVNode(scope=BitSet({1}), md_set=BitSet.universal()))
    vt.add_children(0, 1, 2)

    rg = MDRegionGraphBuilder(basic_input_dists, vt).build()
    ac = MDCircuitBuilder(rg=rg, h=4, input_dists=basic_input_dists).build()

    is_md_1 = ac.check_property(MarginalDeterminism(target_scope={1}))
    assert is_md_1 is False


def test_support_mutually_exclusive_check():
    """4.1.4 Support Mutually Exclusive Check: Manually construct a small AC
    where a SumNode mixes scopes {0, 1} but children's supports on {0} overlap.
    Assert MarginalDeterminism returns False."""

    # We build an AC manually
    ac = SymbolicArithmeticCircuit()

    # Leaves for var 0 with overlapping support
    dist0_a = GaussianDistribution(var=0, mean=0.0, stddev=1.0)
    dist0_a = dist0_a.constrain_to(ContinuousInterval(-1.0, 1.0))  # overlap

    dist0_b = GaussianDistribution(var=0, mean=0.0, stddev=1.0)
    dist0_b = dist0_b.constrain_to(ContinuousInterval(0.0, 2.0))  # overlap

    dist1 = GaussianDistribution(var=1, mean=0.0, stddev=1.0)

    # Node allocator mock
    ac.add_node(1, dist0_a)
    ac.add_node(2, dist1)

    ac.add_node(3, dist0_b)
    ac.add_node(4, dist1)

    p1 = KroneckerProductNode(support=dist0_a.support.union(dist1.support), unit_count=1)
    p2 = KroneckerProductNode(support=dist0_b.support.union(dist1.support), unit_count=1)
    ac.add_node(5, p1)
    ac.add_edge(5, 1)
    ac.add_edge(5, 2)

    ac.add_node(6, p2)
    ac.add_edge(6, 3)
    ac.add_edge(6, 4)

    s = SumNode(support=p1.support.union(p2.support), unit_count=1)
    ac.add_node(7, s)
    ac.add_edge(7, 5, data=torch.tensor([[0.5]]))
    ac.add_edge(7, 6, data=torch.tensor([[0.5]]))

    is_md = ac.check_property(MarginalDeterminism(target_scope={0}))
    assert is_md is False


def test_md_hadamard_unweighted_per_unit_supports():
    """Regression: edge_data is None branch must use per-unit supports,
    not whole-node support, for HadamardProduct children.

    Two HadamardProducts with disjoint per-unit supports but overlapping
    whole-node supports. The check should pass because per-unit pairs are
    all disjoint on the target scope."""
    ac = SymbolicArithmeticCircuit()

    # Leaf distributions for var 0 (target variable) with disjoint per-unit supports
    # HP_A will use units covering [0,1) and [3,4) on var 0
    # HP_B will use units covering [1,2) and [2,3) on var 0
    leaf_a0 = GaussianDistribution(var=0, mean=0.0, stddev=1.0)
    leaf_a0 = leaf_a0.constrain_to(ContinuousInterval(0.0, 4.0))
    leaf_a0.unit_count = 2
    leaf_a0.unit_supports = [
        Support({0: ContinuousInterval(0.0, 1.0)}),
        Support({0: ContinuousInterval(3.0, 4.0)}),
    ]

    leaf_b0 = GaussianDistribution(var=0, mean=0.0, stddev=1.0)
    leaf_b0 = leaf_b0.constrain_to(ContinuousInterval(1.0, 3.0))
    leaf_b0.unit_count = 2
    leaf_b0.unit_supports = [
        Support({0: ContinuousInterval(1.0, 2.0)}),
        Support({0: ContinuousInterval(2.0, 3.0)}),
    ]

    # Leaf distributions for var 1 (non-target, shared across both HPs)
    leaf_a1 = GaussianDistribution(var=1, mean=0.0, stddev=1.0)
    leaf_a1.unit_count = 2
    leaf_a1.unit_supports = [leaf_a1.support, leaf_a1.support]

    leaf_b1 = GaussianDistribution(var=1, mean=0.0, stddev=1.0)
    leaf_b1.unit_count = 2
    leaf_b1.unit_supports = [leaf_b1.support, leaf_b1.support]

    ac.add_node(1, leaf_a0)
    ac.add_node(2, leaf_a1)
    ac.add_node(3, leaf_b0)
    ac.add_node(4, leaf_b1)

    # HP_A: overall support on var 0 = [0, 4), per-unit = [0,1) and [3,4)
    hp_a = HadamardProductNode(support=leaf_a0.support.union(leaf_a1.support), unit_count=2)
    hp_a.unit_supports = [
        leaf_a0.unit_supports[0].union(leaf_a1.unit_supports[0]),
        leaf_a0.unit_supports[1].union(leaf_a1.unit_supports[1]),
    ]
    ac.add_node(5, hp_a)
    ac.add_edge(5, 1)
    ac.add_edge(5, 2)

    # HP_B: overall support on var 0 = [1, 3), per-unit = [1,2) and [2,3)
    hp_b = HadamardProductNode(support=leaf_b0.support.union(leaf_b1.support), unit_count=2)
    hp_b.unit_supports = [
        leaf_b0.unit_supports[0].union(leaf_b1.unit_supports[0]),
        leaf_b0.unit_supports[1].union(leaf_b1.unit_supports[1]),
    ]
    ac.add_node(6, hp_b)
    ac.add_edge(6, 3)
    ac.add_edge(6, 4)

    # SumNode connected to both HPs without edge weights
    s = SumNode(support=hp_a.support.union(hp_b.support), unit_count=1)
    ac.add_node(7, s)
    ac.add_edge(7, 5)  # no edge data → triggers edge_data is None branch
    ac.add_edge(7, 6)

    # Whole-node supports overlap ([0,4) ∩ [1,3) = [1,3) on var 0),
    # but all per-unit pairs are disjoint: [0,1)∩[1,2)=∅, [0,1)∩[2,3)=∅,
    # [3,4)∩[1,2)=∅, [3,4)∩[2,3)=∅
    is_md = ac.check_property(MarginalDeterminism(target_scope={0}))
    assert is_md is True

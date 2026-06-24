import torch

from src.construction.circuit_builder import CircuitBuilder
from src.construction.region_graph_builder import RegionGraphBuilder
from src.symbolic import (
    GaussianDistribution,
    KroneckerProductNode,
    MarginalDeterminism,
    SumNode,
    SymbolicArithmeticCircuit,
    VNode,
    VTree,
)
from src.utils import BitSet, ContinuousInterval
from src.utils.support import Support


# 4. Property Validation


def test_single_variable_md_validation(basic_input_dists):
    """4.1.1 Single-Variable MD Validation: Compile an MDNet with md_sets=[{0}].
    Assert spn.check_property(MarginalDeterminism(target_scope={0})) == True."""
    vt = VTree()
    vt._add_node_explicit(0, VNode(scope=BitSet({0, 1}), md_set=BitSet({0})))
    vt._add_node_explicit(1, VNode(scope=BitSet({0}), md_set=BitSet({0})))
    vt._add_node_explicit(2, VNode(scope=BitSet({1}), md_set=BitSet({1})))
    vt.add_children(0, 1, 2)

    rg = RegionGraphBuilder(basic_input_dists, vt).build()
    ac = CircuitBuilder(rg=rg, h=4, input_dists=basic_input_dists).build()

    is_md = ac.check_property(MarginalDeterminism(target_scope={0}))
    assert is_md is True


def test_joint_variable_md_validation(basic_input_dists):
    """4.1.2 Joint-Variable MD Validation: Compile an MDNet with md_sets=[{0, 1}].
    Assert spn.check_property(MarginalDeterminism(target_scope={0, 1})) == True."""
    vt = VTree()
    vt._add_node_explicit(0, VNode(scope=BitSet({0, 1, 2}), md_set=BitSet({0, 1})))
    vt._add_node_explicit(1, VNode(scope=BitSet({0, 1}), md_set=BitSet({0, 1})))
    vt._add_node_explicit(2, VNode(scope=BitSet({2}), md_set=BitSet({2})))
    vt.add_children(0, 1, 2)

    vt._add_node_explicit(3, VNode(scope=BitSet({0}), md_set=BitSet({0})))
    vt._add_node_explicit(4, VNode(scope=BitSet({1}), md_set=BitSet({1})))
    vt.add_children(1, 3, 4)

    rg = RegionGraphBuilder(basic_input_dists, vt).build()
    ac = CircuitBuilder(rg=rg, h=4, input_dists=basic_input_dists).build()

    is_md = ac.check_property(MarginalDeterminism(target_scope={0, 1}))
    assert is_md is True


def test_negative_md_validation(basic_input_dists):
    """4.1.3 Negative MD Validation: Compile an MDNet with md_sets=[{0}].
    Assert checking on unconstrained scope {1} returns False."""
    vt = VTree()
    vt._add_node_explicit(0, VNode(scope=BitSet({0, 1}), md_set=BitSet({0})))
    vt._add_node_explicit(1, VNode(scope=BitSet({0}), md_set=BitSet({0})))
    # Explicitly make Node 2 unconstrained so it duplicates support and causes overlap
    vt._add_node_explicit(2, VNode(scope=BitSet({1}), md_set=BitSet.universal()))
    vt.add_children(0, 1, 2)

    rg = RegionGraphBuilder(basic_input_dists, vt).build()
    ac = CircuitBuilder(rg=rg, h=4, input_dists=basic_input_dists).build()

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
    dist0_a.var_support = ContinuousInterval(-1.0, 1.0)
    dist0_a.support = Support({0: dist0_a.var_support})

    dist0_b = GaussianDistribution(var=0, mean=0.0, stddev=1.0)
    dist0_b.var_support = ContinuousInterval(0.0, 2.0)
    dist0_b.support = Support({0: dist0_b.var_support})

    dist1 = GaussianDistribution(var=1, mean=0.0, stddev=1.0)

    # Node allocator mock
    ac._add_node_explicit(1, dist0_a)
    ac._add_node_explicit(2, dist1)

    ac._add_node_explicit(3, dist0_b)
    ac._add_node_explicit(4, dist1)

    p1 = KroneckerProductNode(support=dist0_a.support.union(dist1.support), unit_count=1)
    p2 = KroneckerProductNode(support=dist0_b.support.union(dist1.support), unit_count=1)
    ac._add_node_explicit(5, p1)
    ac.add_edge(5, 1)
    ac.add_edge(5, 2)

    ac._add_node_explicit(6, p2)
    ac.add_edge(6, 3)
    ac.add_edge(6, 4)

    s = SumNode(support=p1.support.union(p2.support), unit_count=1)
    ac._add_node_explicit(7, s)
    ac.add_edge(7, 5, data=torch.tensor([[0.5]]))
    ac.add_edge(7, 6, data=torch.tensor([[0.5]]))

    is_md = ac.check_property(MarginalDeterminism(target_scope={0}))
    assert is_md is False




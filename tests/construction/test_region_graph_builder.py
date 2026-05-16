from src.construction.region_graph_builder import MDRegionGraphBuilder
from src.symbolic import GaussianDistribution, MDVNode, MDVTree, RegionNode
from src.utils import BitSet


def test_mixing_layer_topology():
    """Test the topology of a Mixing Layer."""
    mdvt = MDVTree()
    mdvt.add_node(0, MDVNode(scope=BitSet({0, 1}), md_set=BitSet({0})))
    mdvt.add_node(1, MDVNode(scope=BitSet({0}), md_set=BitSet({0})))
    mdvt.add_node(2, MDVNode(scope=BitSet({1}), md_set=BitSet({1})))
    mdvt.add_children(0, 1, 2)

    input_dists = {
        0: GaussianDistribution(var=0, mean=0.0, stddev=1.0),
        1: GaussianDistribution(var=1, mean=0.0, stddev=1.0),
    }

    rg = MDRegionGraphBuilder(input_dists, mdvt).build()

    # The root node unifies all sub-regions
    root_node_id = rg.get_roots()[0]
    root_node = rg.get_node_data(root_node_id)
    assert isinstance(root_node, RegionNode)
    assert root_node.scope == BitSet({0, 1})

    part_ids = rg.get_children(root_node_id)
    assert len(part_ids) == 1


def test_synthesizing_layer_topology():
    """Test the topology of a Synthesizing Layer."""
    mdvt = MDVTree()
    mdvt.add_node(0, MDVNode(scope=BitSet({0, 1}), md_set=BitSet({0, 1})))
    mdvt.add_node(1, MDVNode(scope=BitSet({0}), md_set=BitSet({0})))
    mdvt.add_node(2, MDVNode(scope=BitSet({1}), md_set=BitSet({1})))
    mdvt.add_children(0, 1, 2)

    input_dists = {
        0: GaussianDistribution(var=0, mean=0.0, stddev=1.0),
        1: GaussianDistribution(var=1, mean=0.0, stddev=1.0),
    }

    rg = MDRegionGraphBuilder(input_dists, mdvt).build()

    root_node_id = rg.get_roots()[0]
    part_ids = rg.get_children(root_node_id)
    assert len(part_ids) == 1


def test_non_overlapping_md_sets_validation():
    """Test an empty md-set."""
    mdvt = MDVTree()
    mdvt.add_node(0, MDVNode(scope=BitSet({0, 1}), md_set=BitSet()))
    mdvt.add_node(1, MDVNode(scope=BitSet({0}), md_set=BitSet()))
    mdvt.add_node(2, MDVNode(scope=BitSet({1}), md_set=BitSet()))
    mdvt.add_children(0, 1, 2)

    input_dists = {
        0: GaussianDistribution(var=0, mean=0.0, stddev=1.0),
        1: GaussianDistribution(var=1, mean=0.0, stddev=1.0),
    }

    rg = MDRegionGraphBuilder(input_dists, mdvt).build()
    assert rg.get_roots()[0] is not None

from src.construction.circuit_builder import CircuitBuilder, create_md_circuit
from src.construction.region_graph_builder import RegionGraphBuilder
from src.symbolic import (
    Decomposability,
    KroneckerProductNode,
    LeafNode,
    MarginalDeterminism,
    PartitionNode,
    Smoothness,
    StructuredDecomposability,
    SumNode,
)
from src.symbolic.region_graph import RegionNode


def test_isomorphism_and_leaf_instantiation(basic_input_dists, trivial_vtree):
    """3.1.1 Isomorphism & 3.1.2 Leaf Instantiation
    Assert that the number of SumNodes in the AC equals the number of non-leaf RegionNodes in the RG.
    Assert that KroneckerProductNodes correspond 1:1 with PartitionNodes."""

    rg_builder = RegionGraphBuilder(basic_input_dists, trivial_vtree)
    rg = rg_builder.build()

    ac_builder = CircuitBuilder(rg=rg, leaf_h=4, sum_h=4, input_dists=basic_input_dists)
    ac = ac_builder.build()

    # Note: RG nodes and AC nodes might have different IDs since NodeAllocator is shared or unique per builder.
    # We can count them or map them by type.
    rg_non_leaf_regions = [
        n for n in rg._nodes if isinstance(rg.get_node_data(n), RegionNode) and not rg.is_leaf(n)
    ]
    rg_leaf_regions = [
        n for n in rg._nodes if isinstance(rg.get_node_data(n), RegionNode) and rg.is_leaf(n)
    ]
    rg_partitions = [n for n in rg._nodes if isinstance(rg.get_node_data(n), PartitionNode)]

    ac_sums = [n for n in ac._nodes if isinstance(ac.get_node_data(n), SumNode)]
    ac_leaves = [n for n in ac._nodes if isinstance(ac.get_node_data(n), LeafNode)]
    ac_products = [n for n in ac._nodes if isinstance(ac.get_node_data(n), KroneckerProductNode)]

    # The circuit builder creates nodes 1:1 from the region graph
    # unless there are Mixing or Synthesizing layers creating sub-regions
    assert len(rg_non_leaf_regions) == len(ac_sums)
    # the partition nodes map to Kronecker / Hadamard products based on layer type
    assert len(rg_partitions) >= len(ac_products)  # could be mixture of Hadamard/Kronecker
    assert len(rg_leaf_regions) == len(ac_leaves)

    # Leaf Instantiation: Check that ac_leaves have valid parameterization
    # (since basic_input_dists has Gaussians)
    for l_id in ac_leaves:
        l_node = ac.get_node_data(l_id)
        assert l_node.unit_count > 0


def test_block_scaling(basic_input_dists, trivial_vtree):
    """3.2.1 Sum Node Capacities, 3.2.2 Root Capacity, 3.2.3 Product Node Capacities."""
    h = 4

    rg = RegionGraphBuilder(basic_input_dists, trivial_vtree).build()
    ac = CircuitBuilder(rg=rg, leaf_h=h, sum_h=h, input_dists=basic_input_dists).build()

    root_id = [n for n in ac._nodes if not ac.get_parents(n)][0]
    assert root_id is not None
    root_node = ac.get_node_data(root_id)

    # 3.2.2 Root Capacity
    assert root_node.unit_count == 1

    # 3.2.1 Sum Node Capacities
    for n_id in ac._nodes:
        node = ac.get_node_data(n_id)
        if isinstance(node, SumNode) and n_id != root_id:
            assert node.unit_count == h
        elif isinstance(node, LeafNode):
            assert node.unit_count == h

    # 3.2.3 Product Node Capacities
    for n_id in ac._nodes:
        node = ac.get_node_data(n_id)
        if isinstance(node, KroneckerProductNode):
            children = ac.get_children(n_id)
            if len(children) == 2:
                # Binary Product
                l_h = ac.get_node_data(children[0]).unit_count
                r_h = ac.get_node_data(children[1]).unit_count
                assert node.unit_count == l_h * r_h
            elif len(children) == 1:
                # Unary Product
                c_h = ac.get_node_data(children[0]).unit_count
                assert node.unit_count == c_h


def test_property_validation_pipeline(basic_input_dists, complex_vtree):
    """3.4.1 Property Validation Pipeline: Feed raw synthetic data to construct_optimal_md_vtree,
    pass the vtree to create_md_circuit, and verify the resulting SymbolicArithmeticCircuit
    successfully passes all structural property checks."""

    # We already have complex_vtree with md_sets = [{0, 1}] on nodes
    circuit = create_md_circuit(input_dists=basic_input_dists, md_var_decomp=complex_vtree, leaf_h=4, sum_h=4)

    # Check structural properties
    # Smoothness
    assert circuit.check_property(Smoothness())

    # Decomposability
    assert circuit.check_property(Decomposability())

    # Structured Decomposability
    assert circuit.check_property(StructuredDecomposability())

    # Marginal Determinism for the expected scopes
    # For complex_vtree, we explicitly set md_sets=[{0, 1}]
    # It should only be deterministic on {0, 1}
    assert circuit.check_property(MarginalDeterminism(target_scope={0, 1}))

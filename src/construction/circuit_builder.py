import copy
from typing import Dict

import torch

from src.symbolic import (
    Distribution,
    MDRegionGraph,
    PartitionNode,
    ProductNode,
    RegionNode,
    SumNode,
    SymbolicArithmeticCircuit,
)
from src.symbolic.vtree import MDVTree
from src.utils import BitSet, Support
from src.utils.node_allocator import NodeAllocator


def get_support(scope: BitSet, input_dists: Dict[int, Distribution]) -> Support:
    """Utility function to create a full support for a given scope."""
    intervals = {i: input_dists[i].support for i in scope}
    return Support(intervals)


class MDCircuitBuilder:
    def __init__(self, h: int):
        self.h = h
        self.node_allocator = NodeAllocator()

    def build(self, rg: MDRegionGraph) -> SymbolicArithmeticCircuit:
        """
        Constructs a blockified SymbolicArithmeticCircuit from an MDRegionGraph.
        Each symbolic SumNode represents a block of h units, except for the root(s) which have 1 unit.
        ProductNodes represent h*h units.
        """
        circuit = SymbolicArithmeticCircuit()

        srg_roots = [n for n in rg._nodes if not rg.get_parents(n)]
        srg_id_to_circuit_id: Dict[int, int] = {}

        # Pass 1: Create Region Nodes (Sums and Leaves)
        for rg_id in rg.topological_sort():
            rg_node = rg.get_node_data(rg_id)
            if not isinstance(rg_node, RegionNode):
                continue

            is_leaf_region = len(rg.get_children(rg_id)) == 0
            circuit_id = self.node_allocator.next_id()
            node_h = 1 if rg_id in srg_roots else self.h

            if is_leaf_region:
                var_id = list(rg_node.scope)[0]
                base_dist = rg_node.support[var_id]
                leaf_node = copy.deepcopy(base_dist)
                leaf_node.unit_count = node_h
                circuit.add_node(circuit_id, leaf_node)
            else:
                node_support = Support({v: d.var_support for v, d in rg_node.support.items()})
                sum_node = SumNode(support=node_support, unit_count=node_h)
                circuit.add_node(circuit_id, sum_node)

            srg_id_to_circuit_id[rg_id] = circuit_id

        # Pass 2: Create Partition Nodes (Products) and connect them
        for p_id in rg.topological_sort():
            p_node = rg.get_node_data(p_id)
            if not isinstance(p_node, PartitionNode):
                continue

            parents = rg.get_parents(p_id)
            children = rg.get_children(p_id)

            # Only support arity 1 (unary root merging) or 2 (standard product)
            arity = len(children)

            # Create a ProductNode representing h*h units (or h*1 for unary)
            prod_id = self.node_allocator.next_id()
            p_node_support = Support({v: d.var_support for v, d in p_node.support.items()})

            if arity == 2:
                l_rg_id, r_rg_id = children
                l_circuit_id = srg_id_to_circuit_id[l_rg_id]
                r_circuit_id = srg_id_to_circuit_id[r_rg_id]

                h_l = circuit.get_node_data(l_circuit_id).unit_count
                h_r = circuit.get_node_data(r_circuit_id).unit_count

                prod_node = ProductNode(support=p_node_support, unit_count=h_l * h_r)
                circuit.add_node(prod_id, prod_node)
                circuit.add_edge(prod_id, l_circuit_id)
                circuit.add_edge(prod_id, r_circuit_id)

                for parent_rg_id in parents:
                    parent_circuit_id = srg_id_to_circuit_id[parent_rg_id]
                    h_out = circuit.get_node_data(parent_circuit_id).unit_count

                    # We use dense weights over the implicit blocks
                    weight_matrix = torch.randn(h_out, h_l * h_r)
                    # We can store it as (h_out, h_l, h_r) for the Tucker layer to parse
                    weight_matrix = torch.softmax(weight_matrix.view(h_out, -1), dim=-1).view(
                        h_out, h_l, h_r
                    )
                    circuit.add_edge(parent_circuit_id, prod_id, data=weight_matrix)

            elif arity == 1:
                c_rg_id = children[0]
                c_circuit_id = srg_id_to_circuit_id[c_rg_id]
                h_c = circuit.get_node_data(c_circuit_id).unit_count

                prod_node = ProductNode(support=p_node_support, unit_count=h_c)
                circuit.add_node(prod_id, prod_node)
                circuit.add_edge(prod_id, c_circuit_id)

                for parent_rg_id in parents:
                    parent_circuit_id = srg_id_to_circuit_id[parent_rg_id]
                    h_out = circuit.get_node_data(parent_circuit_id).unit_count

                    weight_matrix = torch.randn(h_out, h_c)
                    weight_matrix = torch.softmax(weight_matrix, dim=-1)
                    circuit.add_edge(parent_circuit_id, prod_id, data=weight_matrix)

        return circuit


def create_md_circuit(
    input_dists: Dict[int, Distribution],
    md_var_decomp: MDVTree,
    h: int,
    num_sums: int = 1,
    num_inputs: int = 1,
) -> SymbolicArithmeticCircuit:
    """
    Creates an MD-Circuit end-to-end.
    """
    from src.construction.region_graph_builder import MDRegionGraphBuilder

    rg_builder = MDRegionGraphBuilder(input_dists, md_var_decomp, num_sums, num_inputs)
    rg = rg_builder.build()

    c_builder = MDCircuitBuilder(h=h)
    return c_builder.build(rg)

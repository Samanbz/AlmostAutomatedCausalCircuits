import copy
import random
from typing import Dict, List

from src.symbolic import (
    DataPartitionNode,
    DataRegionGraph,
    DataRegionNode,
    Distribution,
    PartitionNode,
    ProductNode,
    RegionGraph,
    RegionNode,
    SumNode,
    SymbolicArithmeticCircuit,
)
from src.utils import BitSet, Support


def get_support(scope: BitSet, input_dists: Dict[int, Distribution]) -> Support:
    """Utility function to create a full support for a given scope."""
    intervals = {i: input_dists[i].support for i in scope}
    return Support(intervals)


def normalize_circuit_weights(circuit: SymbolicArithmeticCircuit):
    for node_id in circuit.topological_sort():
        if isinstance(circuit.get_node_data(node_id), SumNode):
            outgoing = circuit.get_outgoing_edges(node_id)
            if not outgoing:
                continue

            total_weight = sum(edge[1] for edge in outgoing)
            if total_weight > 0:
                for child_id, weight in outgoing:
                    circuit.set_edge_data(node_id, child_id, weight / total_weight)





def build_circuit_from_region_graph(
    rg: RegionGraph,
    num_classes: int,
    num_sums: int,
    num_inputs: int,
    input_dists: Dict[int, Distribution],
) -> SymbolicArithmeticCircuit:
    spn = SymbolicArithmeticCircuit()
    node_allocator = NodeAllocator()
    region_to_spn_nodes: Dict[int, List[int]] = {}

    rg_root_ids = [n for n in rg._nodes if not rg.get_parents(n)]
    if not rg_root_ids:
        raise ValueError("Invalid RegionGraph: No root found.")
    rg_root_id = rg_root_ids[0]

    # Pass 1
    for rid in rg.topological_sort():
        r_node = rg.get_node_data(rid)
        if not isinstance(r_node, RegionNode):
            continue

        outgoing_edges = rg._adj[rid]
        is_leaf_region = len(outgoing_edges) == 0
        is_root_region = rid == rg_root_id

        spn_ids = []
        if is_leaf_region:
            for _ in range(num_inputs):
                sid = node_allocator.next_id()
                if len(r_node.scope) != 1:
                    raise ValueError(f"Leaf region scope must be size 1, got {len(r_node.scope)}")
                var_id = list(r_node.scope)[0]
                leaf_dist = copy.deepcopy(input_dists[var_id])
                spn.add_node(sid, leaf_dist)
                spn_ids.append(sid)
        else:
            count = num_classes if is_root_region else num_sums
            for _ in range(count):
                sid = node_allocator.next_id()
                s_node = SumNode(support=get_support(r_node.scope, input_dists))
                spn.add_node(sid, s_node)
                spn_ids.append(sid)

        region_to_spn_nodes[rid] = spn_ids

    # Pass 2
    for pid in rg.topological_sort():
        p_node = rg.get_node_data(pid)
        if not isinstance(p_node, PartitionNode):
            continue

        parents = rg.get_parents(pid)
        if not parents:
            raise ValueError("Partition node has no parent region.")
        parent_region_id = parents[0]

        children = list(rg._adj[pid].keys())
        if len(children) != 2:
            raise ValueError("Partition node must have exactly two child regions.")
        r1_id, r2_id = children

        parent_spn_ids = region_to_spn_nodes.get(parent_region_id, [])
        r1_spn_ids = region_to_spn_nodes.get(r1_id, [])
        r2_spn_ids = region_to_spn_nodes.get(r2_id, [])

        parent_r_node = rg.get_node_data(parent_region_id)

        for n1 in r1_spn_ids:
            for n2 in r2_spn_ids:
                prod_id = node_allocator.next_id()
                prod_node = ProductNode(support=get_support(parent_r_node.scope, input_dists))
                spn.add_node(prod_id, prod_node)
                spn.add_edge(prod_id, n1)
                spn.add_edge(prod_id, n2)

                for parent_sum_id in parent_spn_ids:
                    weight = random.random()
                    spn.add_edge(parent_sum_id, prod_id, data=weight)

    normalize_circuit_weights(spn)
    return spn


def build_circuit_from_data_region_graph(
    rg: DataRegionGraph,
    input_dists: Dict[int, Distribution],
    alpha: float = 0.01,
) -> SymbolicArithmeticCircuit:
    circuit = SymbolicArithmeticCircuit()
    node_allocator = NodeAllocator()
    region_to_ac_nodes: Dict[int, List[int]] = {}

    # Pass 1
    for rg_node_id in rg.topological_sort():
        rg_node = rg.get_node_data(rg_node_id)
        if not isinstance(rg_node, DataRegionNode):
            continue

        is_leaf_region = len(rg._adj[rg_node_id]) == 0
        ac_ids = []

        if is_leaf_region:
            assert len(rg_node.scope) == 1, f"Leaf region scope must be 1, got {rg_node.scope}"
            var_id = list(rg_node.scope)[0]
            base_dist = input_dists[var_id]

            for _ in range(rg_node.num_inputs):
                leaf_id = node_allocator.next_id()
                dist_copy = copy.deepcopy(base_dist)
                leaf_node = dist_copy.constrain_to(rg_node.constraints.get(var_id))
                circuit.add_node(leaf_id, leaf_node)
                ac_ids.append(leaf_id)
        else:
            full_support = Support({var: input_dists[var].var_support for var in rg_node.scope})
            node_support = full_support.intersect(rg_node.constraints)

            for _ in range(rg_node.num_sums):
                sum_id = node_allocator.next_id()
                sum_node = SumNode(support=node_support)
                circuit.add_node(sum_id, sum_node)
                ac_ids.append(sum_id)

        region_to_ac_nodes[rg_node_id] = ac_ids

    # Pass 2
    for p_id in rg.topological_sort():
        p_node = rg.get_node_data(p_id)
        if not isinstance(p_node, DataPartitionNode):
            continue

        parents = rg.get_parents(p_id)
        assert len(parents) == 1, "DataPartitionNode must have exactly one parent."
        parent_region_id = parents[0]
        parent_r_node = rg.get_node_data(parent_region_id)

        children = list(rg._adj[p_id].keys())
        assert len(children) == 2, "Partition node must have exactly two child regions."
        l_r_id, r_r_id = children

        parent_sums = region_to_ac_nodes.get(parent_region_id, [])
        l_children = region_to_ac_nodes.get(l_r_id, [])
        r_children = region_to_ac_nodes.get(r_r_id, [])

        full_support = Support({var: input_dists[var].var_support for var in parent_r_node.scope})
        node_support = full_support.intersect(p_node.constraints)

        num_partitions = len(rg.get_children(parent_region_id))
        proportion_base = (len(p_node.row_ids) + alpha) / (
            len(parent_r_node.row_ids) + alpha * num_partitions
        )

        for l_ac in l_children:
            for r_ac in r_children:
                prod_id = node_allocator.next_id()
                prod_node = ProductNode(support=node_support)
                circuit.add_node(prod_id, prod_node)
                circuit.add_edge(prod_id, l_ac)
                circuit.add_edge(prod_id, r_ac)

                for parent_sum_id in parent_sums:
                    weight = proportion_base
                    if parent_r_node.num_sums > 1 or num_partitions > 1:
                        weight = proportion_base * (0.5 + random.random())
                    circuit.add_edge(parent_sum_id, prod_id, data=weight)

    normalize_circuit_weights(circuit)
    return circuit

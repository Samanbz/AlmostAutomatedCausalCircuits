import random
from typing import Dict, List

from src.symbolic import (
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


def construct_random_region_graph(
    num_features: int, depth: int, num_repetitions: int
) -> RegionGraph:
    """
    Algorithm 2: Random Region Graph construction.
    Returns a RegionGraph DAG.
    """
    rg = RegionGraph()
    node_counter = 0

    # Helper to get unique ID
    def next_id():
        nonlocal node_counter
        nid = node_counter
        node_counter += 1
        return nid

    # Cache regions by scope to avoid duplicates
    scope_to_id: Dict[BitSet, int] = {}

    def get_or_create_region_node(scope: BitSet) -> int:
        if scope in scope_to_id:
            return scope_to_id[scope]

        nid = next_id()
        r_node = RegionNode(scope=scope)
        rg.add_node(nid, r_node)
        scope_to_id[scope] = nid
        return nid

    root_scope = BitSet(range(num_features))

    def split_recursive(scope: BitSet, d: int):
        # Ensure region exists
        rid = get_or_create_region_node(scope)

        if len(scope) < 2:
            return

        # 1. Create a partition for this splitting step
        partition_id = next_id()
        p_node = PartitionNode(scope=scope)
        rg.add_node(partition_id, p_node)

        # Connect Region -> Partition
        rg.add_edge(rid, partition_id)

        # 2. Define sub-scopes
        shuffled = list(scope)
        random.shuffle(shuffled)
        mid = len(shuffled) // 2
        scope1 = BitSet(shuffled[:mid])
        scope2 = BitSet(shuffled[mid:])

        # 3. Get or Create sub-regions
        r1_id = get_or_create_region_node(scope1)
        r2_id = get_or_create_region_node(scope2)

        # 4. Connect Partition -> Sub-Regions
        rg.add_edge(partition_id, r1_id)
        rg.add_edge(partition_id, r2_id)

        # 5. Recurse
        if d > 1:
            # Note: Checking length again to be safe, though split logic handles it
            if len(scope1) > 1:
                split_recursive(scope1, d - 1)
            if len(scope2) > 1:
                split_recursive(scope2, d - 1)

    # Main loop for R repetitions
    for _ in range(num_repetitions):
        split_recursive(root_scope, depth)

    return rg


def construct_spn_from_region_graph(
    rg: RegionGraph,
    num_classes: int,
    num_sums: int,
    num_inputs: int,
    input_dists: Dict[int, Distribution],
) -> SymbolicArithmeticCircuit:
    """
    Algorithm 1: Construct SPN from Region Graph.
    """
    spn = SymbolicArithmeticCircuit()
    node_counter = 0

    # Helper to get unique ID
    def next_id():
        nonlocal node_counter
        nid = node_counter
        node_counter += 1
        return nid

    # Map: region_node_id -> List[spn_node_id]
    region_to_spn_nodes: Dict[int, List[int]] = {}

    # Identify RG root (node with no parents)
    rg_root_ids = [n for n in rg._nodes if not rg.get_parents(n)]
    if not rg_root_ids:
        # Fallback if graph is empty or cyclic (unlikely by construction)
        raise ValueError("Invalid RegionGraph: No root found.")
    rg_root_id = rg_root_ids[0]

    # Pass 1: Create Region-level nodes (Sums or Leaves)
    # We iterate topological, so we might visit ParentRegions before ChildrenRegions.
    # But to create SumNodes we don't need Children yet.
    for rid in rg.topological_sort():
        r_node = rg.get_node_data(rid)

        if not isinstance(r_node, RegionNode):
            continue

        # A region is a leaf region if it has no child partitions.
        outgoing_edges = rg._adj[rid]
        is_leaf_region = len(outgoing_edges) == 0
        is_root_region = rid == rg_root_id

        spn_ids = []
        if is_leaf_region:
            for _ in range(num_inputs):
                sid = next_id()
                # Create a specialized leaf node by reusing the distribution for the scope
                # NOTE: input_dists is a dict[var_id -> Distribution].
                # A leaf region typically covers ONE variable for univariate leaves.
                # If scope has > 1 variable, we need multiple leaves or a multivariate leaf.
                # RAT-SPN usually assumes univariate leaves at bottom.

                # Check scope size
                if len(r_node.scope) != 1:
                    raise ValueError(f"Leaf region scope must be size 1, got {len(r_node.scope)}")

                var_id = list(r_node.scope)[0]
                dist_template = input_dists[var_id]

                # We should clone/copy the distribution to have unique instances
                import copy

                leaf_dist = copy.deepcopy(dist_template)

                spn.add_node(sid, leaf_dist)
                spn_ids.append(sid)
        else:
            count = num_classes if is_root_region else num_sums
            for _ in range(count):
                sid = next_id()
                s_node = SumNode(support=get_support(r_node.scope, input_dists))
                spn.add_node(sid, s_node)
                spn_ids.append(sid)

        region_to_spn_nodes[rid] = spn_ids

    # Pass 2: Create Partition-level nodes (Products) and connect
    # We iterate again to ensure all region nodes (children of partitions) are created.
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

        # Parent scope
        parent_r_node = rg.get_node_data(parent_region_id)

        # Create Product nodes: Cartesian product of children tensors
        # N1 x N2
        for n1 in r1_spn_ids:
            for n2 in r2_spn_ids:
                prod_id = node_counter
                node_counter += 1

                prod_node = ProductNode(support=get_support(parent_r_node.scope, input_dists))
                spn.add_node(prod_id, prod_node)

                # Connect Product -> Children (Sum/Leaf of subregions)
                spn.add_edge(prod_id, n1)  # Edge: Product -> Left Child
                spn.add_edge(prod_id, n2)  # Edge: Product -> Right Child

                # Connect Parent Region Sums -> This Product
                for parent_sum_id in parent_spn_ids:
                    # Edge: Sum -> Product
                    weight = random.random()
                    spn.add_edge(parent_sum_id, prod_id, data=weight)

    # Normalize weights for all Sum nodes
    for node_id in spn.topological_sort():
        if isinstance(spn.get_node_data(node_id), SumNode):
            outgoing = spn.get_outgoing_edges(node_id)
            if not outgoing:
                continue

            total_weight = sum(edge[1] for edge in outgoing)
            if total_weight > 0:
                for child_id, weight in outgoing:
                    spn.set_edge_data(node_id, child_id, weight / total_weight)

    return spn


def create_rat_spn(
    num_features: int,
    num_classes: int,
    depth: int,
    num_repetitions: int,
    num_sums: int,
    num_inputs: int,
    input_dists: Dict[int, Distribution],
) -> SymbolicArithmeticCircuit:
    """
    High-level function to create a RAT-SPN arithmetic circuit.

    Args:
        num_features: Number of input variables/features.
        num_classes: Number of classes (roots), C in the paper.
        depth: Depth of the region graph splitting, D in the paper.
        num_repetitions: Number of random split repetitions, R in the paper.
        num_sums: Number of sum nodes per region, S in the paper.
        num_inputs: Number of input distributions per leaf region, I in the paper.

    Returns:
        A SymbolicArithmeticCircuit representing the SPN.
    """
    rg = construct_random_region_graph(num_features, depth, num_repetitions)
    spn = construct_spn_from_region_graph(rg, num_classes, num_sums, num_inputs, input_dists)
    return spn

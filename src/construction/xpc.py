import random
from typing import Dict, List, Tuple

import numpy as np

from src.logging import logger as g_logger
from src.symbolic import (
    DataPartitionNode,
    DataRegionGraph,
    DataRegionNode,
    Distribution,
    MDVTree,
    ProductNode,
    SumNode,
    SymbolicArithmeticCircuit,
    VTree,
)
from src.utils import BitSet, DataSlice, Support


logger = g_logger.getChild("xpc")


def get_random_constraints(
    conj_dists: Dict[int, Distribution],
) -> Tuple[Support, Support]:
    """Generate random constraints by sampling from distributions and cutting at the sampled point."""
    constraints = Support()
    r_constraints = Support()

    for var_idx, dist in conj_dists.items():
        rand_cut_point = dist.sample()
        left_dist, right_dist = dist.split_at(rand_cut_point)
        if random.random() < 0.5:
            left_dist, right_dist = right_dist, left_dist
        constraints.add(var_idx, left_dist.var_support)
        r_constraints.add(var_idx, right_dist.var_support)

    return constraints, r_constraints


def apply_logical_constraints(
    data_slice: DataSlice,
    constraints: Support,
) -> Tuple[BitSet, BitSet]:
    """
    Applies constraints to the subset of data defined by row_ids.
    Returns (subset_row_ids, rest_row_ids).
    """
    if not data_slice.row_ids:
        return [], []

    assert all(i in data_slice.col_ids for i in list(constraints.intervals.keys())), (
        "Constraint variables must be in data slice columns."
    )

    mask = np.ones(len(data_slice.row_ids), dtype=bool)

    for var_idx, interval in constraints:
        vals = data_slice.get_column(var_idx)
        mask &= interval.contains(vals)

    # Convert active rows to indices
    current_row_mask = data_slice.row_ids.to_numpy(data_slice.data.shape[0])
    active_indices = np.flatnonzero(current_row_mask)

    # Split indices based on mask
    subset_indices = active_indices[mask]
    rest_indices = active_indices[~mask]

    # Reconstruct BitSets
    subset_ids_mask = np.zeros_like(current_row_mask)
    subset_ids_mask[subset_indices] = True
    subset_ids = BitSet.from_bool_mask(subset_ids_mask)

    rest_ids_mask = np.zeros_like(current_row_mask)
    rest_ids_mask[rest_indices] = True
    rest_ids = BitSet.from_bool_mask(rest_ids_mask)

    assert rest_ids == data_slice.row_ids.difference(subset_ids), "Row ID partitioning error."

    return subset_ids, rest_ids


def partition_randomly(
    data_slice: DataSlice,
    conj_vars: BitSet,
    split_arity: int,
    min_examples: int,
    input_dists: Dict[int, Distribution],
    max_tries: int = 20,
) -> List[DataSlice]:
    slices: List[DataSlice] = []

    assert all(var in input_dists for var in conj_vars), (
        "All conjunction variables must have input distributions."
    )

    conj_dists = {var: input_dists[var] for var in conj_vars}

    logger.debug("")
    tries = 0
    while len(slices) < split_arity - 1:
        if tries == 0:
            logger.debug(
                f"Partitioning data slice with {len(data_slice)} examples on vars {conj_vars}. Slice {len(slices) + 1}/{split_arity - 1}"
            )
        if tries >= max_tries:
            logger.debug(
                f"Failed to partition data slice with {len(data_slice)} examples on vars {conj_vars} after {max_tries} tries."
            )
            break
        if len(data_slice) < 2 * min_examples:
            logger.debug(
                f"Insufficient examples ({len(data_slice)}) to partition data slice on vars {conj_vars}."
            )
            break

        constraints, r_constraints = get_random_constraints(
            conj_dists=conj_dists,
        )
        subset_row_ids, rest_row_ids = apply_logical_constraints(
            data_slice=data_slice,
            constraints=constraints,
        )
        if len(subset_row_ids) < min_examples or len(rest_row_ids) < min_examples:
            logger.debug(
                f"\tRejected {len(subset_row_ids)}/{len(rest_row_ids)} split due to insufficient examples. Tries {tries}/{max_tries}."
            )
            tries += 1
            continue

        logger.debug(f"\tAccepted split with {len(subset_row_ids)}/{len(rest_row_ids)} examples.")
        logger.debug("")

        new_slice = DataSlice(
            data_slice.data,
            subset_row_ids,
            data_slice.col_ids,
            constraints=data_slice.constraints.intersect(constraints),
        )
        slices.append(new_slice)

        data_slice = DataSlice(
            data_slice.data,
            rest_row_ids,
            data_slice.col_ids,
            constraints=data_slice.constraints.intersect(r_constraints),
        )
        tries = 0

    if len(slices) > 0:
        slices.append(data_slice)
        return slices
    else:
        return []


def construct_random_data_region_graph(
    data: np.ndarray,
    input_dists: Dict[int, Distribution],
    min_examples: int,
    split_arity: int,
    var_decomp: VTree,
) -> DataRegionGraph:
    assert split_arity >= 2, "Split arity must be at least 2."
    assert data.ndim == 2, "Data must be a 2D array."
    assert var_decomp is not None, "Variable decomposition (VTree) is required."

    node_counter = 0

    def next_id():
        nonlocal node_counter
        nid = node_counter
        node_counter += 1
        return nid

    rg = DataRegionGraph()
    num_rows, num_cols = data.shape

    root_region = DataRegionNode(scope=BitSet.full(num_cols), row_ids=BitSet.full(num_rows))
    root_id = next_id()
    rg.add_node(root_id, root_region)

    # Map DataRegionNode ID -> VTree Node ID
    rnode_to_vnode: Dict[int, int] = {root_id: var_decomp.get_root()}

    P: List[Tuple[int, DataRegionNode]] = [(root_id, root_region)]

    while P:
        # Random region from P
        idx = random.randint(0, len(P) - 1)
        r_id, r_node = P.pop(idx)

        v_id = rnode_to_vnode[r_id]

        if var_decomp.is_leaf(v_id):
            continue

        # Get children (left is always conj_vars), v_id is not leaf, so children are not None
        conj_v_id, rest_v_id = var_decomp.get_children_pair(v_id)

        conj_vars = var_decomp.get_node_data(conj_v_id).scope

        current_data_slice = r_node.get_data_slice(data)
        current_constraints = r_node.constraints
        constrainted_input_dists = {
            var: dist.constrain_to(current_constraints[var]) if var in current_constraints else dist
            for var, dist in input_dists.items()
        }

        data_slices = partition_randomly(
            data_slice=current_data_slice,
            conj_vars=conj_vars,
            split_arity=split_arity,
            min_examples=min_examples,
            input_dists=constrainted_input_dists,
        )

        new_partitions: List[Tuple[int, DataPartitionNode]] = []
        for data_slice in data_slices:
            partition = DataPartitionNode(
                scope=r_node.scope,
                row_ids=data_slice.row_ids,
                constraints=data_slice.constraints,  # carry over constraints
            )
            partition_id = next_id()
            rg.add_node(partition_id, partition)
            rg.add_edge(r_id, partition_id)
            new_partitions.append((partition_id, partition))

        for p_id, p_node in new_partitions:
            # Filter constraints for children based on their scope.
            l_constraints = p_node.constraints.filter_by_vars(conj_vars)
            l_region = DataRegionNode(
                scope=conj_vars,
                row_ids=p_node.row_ids,
                constraints=l_constraints,
            )

            r_scope = r_node.scope.difference(conj_vars)
            r_constraints = p_node.constraints.filter_by_vars(r_scope)
            r_region = DataRegionNode(
                scope=r_scope,
                row_ids=p_node.row_ids,
                constraints=r_constraints,
            )
            l_region_id = next_id()
            r_region_id = next_id()

            rg.add_node(l_region_id, l_region)
            rg.add_node(r_region_id, r_region)
            rg.add_edge(p_id, l_region_id)
            rg.add_edge(p_id, r_region_id)

            # Map new regions to corresponding VNodes
            rnode_to_vnode[l_region_id] = conj_v_id
            if rest_v_id is not None:
                rnode_to_vnode[r_region_id] = rest_v_id

            # Add to Queue
            P.append((l_region_id, l_region))
            P.append((r_region_id, r_region))

    return rg


def construct_random_md_data_region_graph(
    data: np.ndarray,
    input_dists: Dict[int, Distribution],
    min_examples: int,
    split_arity: int,
    md_var_decomp: MDVTree,
) -> DataRegionGraph:
    assert split_arity >= 2, "Split arity must be at least 2."
    assert data.ndim == 2, "Data must be a 2D array."
    assert md_var_decomp is not None, (
        "Marginal deterministic variable decomposition (MDVTree) is required."
    )

    node_counter = 0

    def next_id():
        nonlocal node_counter
        nid = node_counter
        node_counter += 1
        return nid

    rg = DataRegionGraph()
    num_rows, num_cols = data.shape

    root_region = DataRegionNode(scope=BitSet.full(num_cols), row_ids=BitSet.full(num_rows))
    root_id = next_id()
    rg.add_node(root_id, root_region)

    # Map DataRegionNode ID -> VTree Node ID
    rg_nid_to_md_vid: Dict[int, int] = {root_id: md_var_decomp.get_root()}

    def partition_recursive(r_id: int):
        r_node = rg.get_node_data(r_id)
        md_vid = rg_nid_to_md_vid[r_id]
        md_vnode = md_var_decomp.get_node_data(md_vid)
        if md_var_decomp.is_leaf(md_vid):
            return  # No further partitioning needed

        current_data_slice = r_node.get_data_slice(data)
        current_constraints = r_node.constraints
        constrainted_input_dists = {
            var: dist.constrain_to(current_constraints[var]) if var in current_constraints else dist
            for var, dist in input_dists.items()
        }

        data_slices = partition_randomly(
            data_slice=current_data_slice,
            conj_vars=md_vnode.md_set,
            split_arity=split_arity,
            min_examples=min_examples,
            input_dists=constrainted_input_dists,
        )

        for data_slice in data_slices:
            l_md_child_id, r_md_child_id = md_var_decomp.get_children_pair(
                md_vid
            )  # Guaranteed to be truthy since md_vid is not a leaf
            l_md_child = md_var_decomp.get_node_data(l_md_child_id)
            r_md_child = md_var_decomp.get_node_data(r_md_child_id)

            p_node_id = next_id()
            p_node = DataPartitionNode(
                scope=md_vnode.scope,
                row_ids=data_slice.row_ids,
                constraints=data_slice.constraints,
            )
            rg.add_node(p_node_id, p_node)

            l_r_node_id = next_id()
            l_r_node = DataRegionNode(
                scope=l_md_child.scope,
                row_ids=data_slice.row_ids,
                constraints=data_slice.constraints,
            )
            rg.add_node(l_r_node_id, l_r_node)
            rg_nid_to_md_vid[l_r_node_id] = l_md_child_id

            r_r_node_id = next_id()
            r_r_node = DataRegionNode(
                scope=r_md_child.scope,
                row_ids=data_slice.row_ids,
                constraints=data_slice.constraints,
            )
            rg.add_node(r_r_node_id, r_r_node)
            rg_nid_to_md_vid[r_r_node_id] = r_md_child_id

            rg.add_edge(p_node_id, l_r_node_id)
            rg.add_edge(p_node_id, r_r_node_id)

            rg.add_edge(r_id, p_node_id)

            partition_recursive(l_r_node_id)
            partition_recursive(r_r_node_id)

    partition_recursive(root_id)
    return rg


def construct_spn_from_region_graph(
    rg: DataRegionGraph,
    input_dists: Dict[int, Distribution],
) -> SymbolicArithmeticCircuit:
    node_counter = 1

    def next_id():
        nonlocal node_counter
        nid = node_counter
        node_counter += 1
        return nid

    circuit = SymbolicArithmeticCircuit()

    rg_root = rg.get_node_data(rg.get_roots()[0])
    ac_root = SumNode(
        support=Support({var: input_dists[var].var_support for var in rg_root.scope}).intersect(
            rg_root.constraints
        )
    )

    circuit.add_node(0, ac_root)

    rg_node_to_ac_node: Dict[int, int] = {}
    rg_node_to_ac_node[rg.get_roots()[0]] = 0

    for r_id in rg.topological_sort():
        r_node = rg.get_node_data(r_id)
        if r_id == rg.get_roots()[0]:
            continue

        logger.debug(f"Processing RG Node ID {r_id} with scope {r_node.scope}.")
        if isinstance(r_node, DataRegionNode):
            rg_node_parents = rg.get_parents(r_id)  # TODO Use tree?
            logger.debug(f"\tParents: {rg_node_parents}")
            assert len(rg_node_parents) == 1, (
                "DataRegionNode must have exactly one parent DataPartitionNode."
            )
            p_id = rg_node_parents[0]
            p_ac_id = rg_node_to_ac_node[p_id]
            p_ac_node = circuit.get_node_data(p_ac_id)
            assert isinstance(p_ac_node, ProductNode), (
                "Parent AC node must be a ProductNode corresponding to the partition."
            )

            if rg.is_leaf(r_id):
                # Create Naive Factorization
                for i in r_node.scope:
                    leaf_node = input_dists[i].constrain_to(r_node.constraints.get(i))
                    leaf_id = next_id()
                    circuit.add_node(leaf_id, leaf_node)
                    circuit.add_edge(p_ac_id, leaf_id)
            else:
                # Create SumNode for inner RegionNode
                region_ac_node = SumNode(
                    support=Support(
                        {var: input_dists[var].var_support for var in r_node.scope}
                    ).intersect(r_node.constraints)
                )
                region_ac_id = next_id()
                circuit.add_node(region_ac_id, region_ac_node)
                circuit.add_edge(p_ac_id, region_ac_id)
                rg_node_to_ac_node[r_id] = region_ac_id

        elif isinstance(r_node, DataPartitionNode):
            partition_ac_node = ProductNode(
                support=Support(
                    {var: input_dists[var].var_support for var in r_node.scope}
                ).intersect(r_node.constraints)
            )
            partition_ac_id = next_id()
            circuit.add_node(partition_ac_id, partition_ac_node)

            rg_node_parents = rg.get_parents(r_id)
            assert len(rg_node_parents) == 1, (
                "DataPartitionNode must have exactly one parent DataRegionNode."
            )
            r_parent_id = rg_node_parents[0]
            r_parent_ac_id = rg_node_to_ac_node[r_parent_id]
            r_parent_ac_node = circuit.get_node_data(r_parent_ac_id)
            assert isinstance(r_parent_ac_node, SumNode), (
                "Parent AC node must be a SumNode corresponding to the region."
            )

            circuit.add_edge(r_parent_ac_id, partition_ac_id)
            rg_node_to_ac_node[r_id] = partition_ac_id
        else:
            raise ValueError(f"Unknown region graph node type: {type(r_node)}")

    return circuit

import random
from typing import Dict, List, Tuple

import numpy as np

from src.logging import logger as g_logger
from src.symbolic import (
    DataPartitionNode,
    DataRegionGraph,
    DataRegionNode,
    Distribution,
    GaussianDistribution,
    SymbolicArithmeticCircuit,
    UniformDistribution,
    VTree,
)
from src.utils import BitSet, DataSlice, Interval


logger = g_logger.getChild("xpc")


def get_random_cut_point(dist: Distribution) -> float:
    if isinstance(dist, GaussianDistribution):
        return np.random.normal(dist.mean, dist.stddev)
    elif isinstance(dist, UniformDistribution):
        return np.random.uniform(dist.low, dist.high)
    return 0.0


def get_random_logical_constraints(
    conj_dists: Dict[int, Distribution],
) -> Dict[int, Interval]:
    constraints = {}

    for var_idx, dist in conj_dists.items():
        val = get_random_cut_point(dist)

        # Randomly choose split direction
        if random.random() < 0.5:
            # (-inf, val]
            interval = Interval(
                float("-inf"), val, include_low=False, include_high=True
            )  # TODO: Review
        else:
            # (val, inf)
            interval = Interval(
                val, float("inf"), include_low=False, include_high=False
            )  # TODO: Review

        constraints[var_idx] = interval

    return constraints


def apply_logical_constraints(
    data_slice: DataSlice,
    constraints: Dict[int, Interval],
) -> Tuple[BitSet, BitSet]:
    """
    Applies constraints to the subset of data defined by row_ids.
    Returns (subset_row_ids, rest_row_ids).
    """
    if not data_slice.row_ids:
        return [], []

    assert all(i in data_slice.col_ids for i in list(constraints.keys())), (
        "Constraint variables must be in data slice columns."
    )

    mask = np.ones(len(data_slice.row_ids), dtype=bool)

    for var_idx, interval in constraints.items():
        vals = data_slice.get_column(var_idx)
        mask &= interval.contains(vals)

    # Convert active rows to indices
    current_row_mask = data_slice.row_ids.to_bool_mask(data_slice.data.shape[0])
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

        constraints = get_random_logical_constraints(
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
            data_slice.data, subset_row_ids, data_slice.col_ids, constraints=constraints
        )
        slices.append(new_slice)
        data_slice = DataSlice(data_slice.data, rest_row_ids, data_slice.col_ids)
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
) -> Tuple[DataRegionGraph, List[int]]:
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

    P = [(root_id, root_region)]
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
        data_slices = partition_randomly(
            data_slice=current_data_slice,
            conj_vars=conj_vars,
            split_arity=split_arity,
            min_examples=min_examples,
            input_dists=input_dists,
        )

        new_partitions: List[Tuple[int, DataPartitionNode]] = []
        for data_slice in data_slices:
            partition = DataPartitionNode(
                scope=r_node.scope,
                row_ids=data_slice.row_ids,
                constraints=data_slice.constraints,
            )
            partition_id = next_id()
            rg.add_node(partition_id, partition)
            rg.add_edge(r_id, partition_id)
            new_partitions.append((partition_id, partition))

        for p_id, p_node in new_partitions:
            assert conj_vars == BitSet(list(p_node.constraints.keys())), (
                "Constraint variable mismatch."
            )
            l_region = DataRegionNode(
                scope=conj_vars,
                row_ids=p_node.row_ids,
                constraints=p_node.constraints,
            )
            r_region = DataRegionNode(
                scope=r_node.scope.difference(conj_vars),
                row_ids=p_node.row_ids,
                # No constraints on right region
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


def construct_spn_from_region_graph(
    rg: DataRegionGraph,
    data: np.ndarray,
) -> SymbolicArithmeticCircuit:
    pass  # TODO

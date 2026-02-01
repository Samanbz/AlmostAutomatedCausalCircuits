import random
from typing import Dict, List, Tuple

import numpy as np

from src.symbolic import (
    DataPartitionNode,
    DataRegionGraph,
    DataRegionNode,
    Distribution,
    GaussianDistribution,
    UniformDistribution,
)
from src.utils import BitSet, DataSlice, Interval


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


def rand_splits(
    data_slice: DataSlice,
    conj_len: int,
    split_arity: int,
    min_examples: int,
    input_dists: Dict[int, Distribution],
) -> Tuple[List[DataSlice], BitSet]:
    slices: List[DataSlice] = []

    conj_vars = random.sample(list(data_slice.col_ids), conj_len)
    conj_dists = {var: input_dists[var] for var in conj_vars}

    while len(slices) < split_arity - 1:
        constraints = get_random_logical_constraints(
            conj_dists=conj_dists,
        )
        subset_row_ids, rest_row_ids = apply_logical_constraints(
            data_slice=data_slice,
            constraints=constraints,
        )
        if len(subset_row_ids) < min_examples or len(rest_row_ids) < min_examples:
            print(f"\tRejected split {len(subset_row_ids)}/{len(rest_row_ids)}")
            continue
        print(f"\tAccepted split {len(subset_row_ids)}/{len(rest_row_ids)}")
        new_slice = DataSlice(data_slice.data, subset_row_ids, data_slice.col_ids)
        slices.append(new_slice)
        data_slice = DataSlice(data_slice.data, rest_row_ids, data_slice.col_ids)

    if len(slices) > 0:
        slices.append(data_slice)
        return slices, BitSet(conj_vars)
    else:
        return [], BitSet.from_int(0)


def random_region_graph(
    data: np.ndarray,
    input_dists: Dict[int, Distribution],
    min_examples: int,
    split_arity: int,
    conj_len: int,
) -> DataRegionGraph:
    node_counter = 0

    def next_id():
        nonlocal node_counter
        nid = node_counter
        node_counter += 1
        return nid

    rg = DataRegionGraph()
    num_rows, num_cols = data.shape

    root_region = DataRegionNode(
        scope=BitSet.full(num_cols), row_ids=BitSet.full(num_rows), constraints={}
    )
    root_id = next_id()
    rg.add_node(root_id, root_region)

    P = [(root_id, root_region)]
    while P:
        print(f"P: {P}")
        # Random region from P
        idx = random.randint(0, len(P) - 1)
        r_id, r_node = P.pop(idx)
        if len(r_node.scope) < conj_len or len(r_node.row_ids) < 2 * min_examples:
            print(f"Skipping region node {r_id} due to insufficient scope size or examples.")
            continue

        data_slice = r_node.get_data_slice(data)
        print(f"Processing region node {r_id} with {len(r_node.row_ids)} examples.")
        data_slices, conj_vars = rand_splits(
            data_slice=data_slice,
            conj_len=conj_len,
            split_arity=split_arity,
            min_examples=min_examples,
            input_dists=input_dists,
        )

        if data_slices:
            new_partitions: List[Tuple[int, DataPartitionNode]] = []
            for data_slice in data_slices:
                partition = DataPartitionNode(scope=r_node.scope, row_ids=data_slice.row_ids)
                partition_id = next_id()
                rg.add_node(partition_id, partition)
                rg.add_edge(r_id, partition_id)
                new_partitions.append((partition_id, partition))
            for p_id, p_node in new_partitions:
                l_region = DataRegionNode(
                    scope=conj_vars,
                    row_ids=p_node.row_ids,
                )
                r_region = DataRegionNode(
                    scope=r_node.scope.difference(conj_vars),
                    row_ids=p_node.row_ids,
                )
                l_region_id = next_id()
                r_region_id = next_id()
                rg.add_node(l_region_id, l_region)
                rg.add_node(r_region_id, r_region)
                rg.add_edge(p_id, l_region_id)
                rg.add_edge(p_id, r_region_id)
                P.append((r_region_id, r_region))

    return rg

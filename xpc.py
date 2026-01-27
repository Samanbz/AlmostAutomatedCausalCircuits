import random
from typing import Dict, List, Tuple

import numpy as np

from graph import (
    DataPartitionNode,
    DataRegionGraph,
    DataRegionNode,
    Distribution,
    GaussianDistribution,
    Interval,
    RegionGraph,
    UniformDistribution,
)


def get_random_cut_point(dist: Distribution) -> float:
    if isinstance(dist, GaussianDistribution):
        return np.random.normal(dist.mean, dist.stddev)
    elif isinstance(dist, UniformDistribution):
        return np.random.uniform(dist.low, dist.high)
    return 0.0


def get_random_logical_constraints(
    input_dists: Dict[int, Distribution], conj_vars: List[int]
) -> List[Tuple[int, Interval]]:
    constraints = []

    for var_idx in conj_vars:
        assert var_idx in input_dists, "Distribution for variable not found."

        dist = input_dists[var_idx]
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

        constraints.append((var_idx, interval))

    return constraints


def apply_logical_constraints(
    data: np.ndarray, row_ids: List[int], constraints: List[Tuple[int, Interval]]
) -> Tuple[List[int], List[int]]:
    """
    Applies constraints to the subset of data defined by row_ids.
    Returns (subset_row_ids, rest_row_ids).
    """
    if not row_ids:
        return [], []

    data_slice = data[row_ids]
    mask = np.ones(len(data_slice), dtype=bool)

    for var_idx, interval in constraints:
        vals = data_slice[:, var_idx]

        cond = np.full(vals.shape, True, dtype=bool)
        if interval.include_low:
            cond &= vals >= interval.low
        else:
            cond &= vals > interval.low

        if interval.include_high:
            cond &= vals <= interval.high
        else:
            cond &= vals < interval.high

        mask &= cond

    row_ids_arr = np.array(row_ids)
    subset_ids = row_ids_arr[mask].tolist()
    rest_ids = row_ids_arr[~mask].tolist()

    return subset_ids, rest_ids


def rand_splits(
    row_ids: List[int],
    data_slice: np.ndarray,
    min_examples: int,
    conj_vars: List[int],
    split_arity: int,
    input_dists: Dict[int, Distribution],
) -> List[Tuple[List[int], np.ndarray]]:
    rest = (row_ids, data_slice)
    splits = []

    while len(splits) < split_arity:
        constraints = get_random_logical_constraints(
            distributions=input_dists,
            scope=tuple(conj_vars),
        )
        subset_ids, rest_ids = apply_logical_constraints(
            data=data_slice, row_ids=row_ids, constraints=constraints
        )

        if len(subset_ids) >= min_examples and len(rest_ids) >= min_examples:
            splits.append((subset_ids, rest[subset_ids]))
            rest = rest[rest_ids]

    if len(splits) == 0:
        return [rest]
    else:
        return []


def random_region_graph(
    data: np.ndarray,
    input_dists: Dict[int, Distribution],
    min_examples: int,
    split_arity: int,
    conj_len: int,
) -> RegionGraph:
    node_counter = 0

    def next_id():
        nonlocal node_counter
        nid = node_counter
        node_counter += 1
        return nid

    rg = DataRegionGraph()
    num_rows, num_cols = data.shape
    root_scope = tuple(range(num_cols))

    root_region = DataRegionNode(scope=root_scope, row_ids=list(range(num_rows)), constraints={})
    root_id = next_id()
    rg.add_node(root_id, root_region)

    P = [(root_id, root_region)]

    while P:
        # Random region from P
        idx = random.randint(0, len(P) - 1)
        r_id, r_node = P.pop(idx)

        conj_vars = random.sample(list(range(data.shape[1])), conj_len)
        data_slice = r_node.get_data_slice(data)
        data_splits = rand_splits(
            r_node.row_ids,
            data_slice,
            min_examples,
            conj_vars,
            split_arity,
            input_dists,
        )

        if data_splits:
            new_paritions: List[Tuple[int, DataPartitionNode]] = []
            for row_ids, _ in data_splits:
                partition = DataPartitionNode(scope=r_node.scope, row_ids=row_ids)
                partition_id = next_id()
                rg.add_node(partition_id, partition)
                rg.add_edge(r_id, partition_id)
                new_paritions.append((partition_id, partition))
            for p_id, p_node in new_paritions:
                l_region = DataRegionNode(
                    scope=conj_vars,
                    row_ids=p_node.row_ids,
                )
                r_region = DataRegionNode(
                    scope=tuple(set(r_node.scope) - set(conj_vars)),
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

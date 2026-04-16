import itertools
import random
from typing import Dict, List, Tuple

import numpy as np

from src.logger import logger as g_logger
from src.symbolic import Distribution
from src.utils import BitSet, DataSlice, DiscreteInterval, Support


logger = g_logger.getChild("partitioning")


def get_random_constraints(
    conj_dists: Dict[int, Distribution],
) -> Tuple[Support, Support]:
    """Generate random constraints by sampling from distributions and cutting at the sampled point."""
    constraints = Support()
    r_constraints = Support()

    for varg_node_idx, dist in conj_dists.items():
        rand_cut_point = dist.sample()
        left_dist, right_dist = dist.split_at(rand_cut_point)
        if random.random() < 0.5:
            left_dist, right_dist = right_dist, left_dist
        constraints.add(varg_node_idx, left_dist.var_support)
        r_constraints.add(varg_node_idx, right_dist.var_support)

    return constraints, r_constraints


def get_factorial_constraints(
    conj_dists: Dict[int, Distribution],
) -> List[Support]:
    """Generate mutually exclusive constraints forming a full factorial grid split."""
    var_splits = {}
    for var, dist in conj_dists.items():
        if isinstance(dist.var_support, DiscreteInterval) and len(dist.var_support.values) > 1:
            valid_cuts = sorted(dist.var_support.values)[:-1]
            rand_cut_point = random.choice(valid_cuts)
        else:
            rand_cut_point = dist.sample()

        logger.debug(
            f"Variable {var}: sampled cut point {rand_cut_point} from distribution {dist}."
        )
        left_support, right_support = dist.var_support.split_at(rand_cut_point)
        # Randomize left/right order to avoid bias
        opts = (
            [right_support, left_support]
            if random.random() < 0.5
            else [left_support, right_support]
        )
        var_splits[var] = opts

    vars_list = list(conj_dists.keys())
    supports = []

    # Generate all 2^k combinations
    for combo in itertools.product(*[var_splits[v] for v in vars_list]):
        s = Support()
        for i, v in enumerate(vars_list):
            s.add(v, combo[i])
        supports.append(s)

    return supports


def apply_constraint(
    data_slice: DataSlice,
    constraint: Support,
) -> BitSet:
    """
    Applies a single hyper-rectangle constraint to the data slice
    and returns the row IDs that fall into it.
    """
    if not data_slice.row_ids:
        return BitSet()

    assert all(i in data_slice.col_ids for i in list(constraint.intervals.keys())), (
        "Constraint variables must be in data slice columns."
    )

    mask = np.ones(len(data_slice.row_ids), dtype=bool)

    for varg_node_idx, interval in constraint:
        vals = data_slice.get_column(varg_node_idx)
        mask &= interval.contains(vals)

    # Convert active rows to indices
    current_row_mask = data_slice.row_ids.to_numpy(data_slice.data.shape[0])
    active_indices = np.flatnonzero(current_row_mask)

    # Filter indices based on mask
    subset_indices = active_indices[mask]

    subset_ids_mask = np.zeros_like(current_row_mask)
    subset_ids_mask[subset_indices] = True
    subset_ids = BitSet.from_bool_mask(subset_ids_mask)

    return subset_ids


def partition_randomly(
    data_slice: DataSlice,
    split_arity: int,  # Kept for signature compatibility, but overridden by 2^k geometry
    min_examples: int,
    conj_dists: Dict[int, Distribution],
    max_tries: int = 20,
    subsample_size: int = None,
) -> List[DataSlice]:
    if len(data_slice) < 2 * min_examples:
        logger.debug(
            f"Insufficient examples ({len(data_slice)}) to partition data slice on vars {list(conj_dists.keys())}."
        )
        return []

    logger.debug(
        f"Factorial partitioning data slice with {len(data_slice)} examples on {len(conj_dists)} vars."
    )

    test_slice = data_slice
    test_min_examples = min_examples
    if subsample_size is not None and len(data_slice) > subsample_size:
        active_indices = np.flatnonzero(data_slice.row_ids.to_numpy(data_slice.data.shape[0]))
        subsample_indices = np.random.choice(active_indices, size=subsample_size, replace=False)
        sub_mask = np.zeros(data_slice.data.shape[0], dtype=bool)
        sub_mask[subsample_indices] = True
        test_slice = DataSlice(
            data_slice.data,
            BitSet.from_bool_mask(sub_mask),
            data_slice.col_ids,
            data_slice.constraints,
        )
        test_min_examples = max(1, int(min_examples * (subsample_size / len(data_slice))))

    for _ in range(max_tries):
        supports = get_factorial_constraints(conj_dists)
        populated_branches = 0

        # First pass: Test efficiency on the (sub)sample
        for constraint in supports:
            subset_row_ids = apply_constraint(
                data_slice=test_slice,
                constraint=constraint,
            )
            if len(subset_row_ids) >= test_min_examples:
                populated_branches += 1

        # For a split to be useful, it should distribute data into at least 2 branches
        if populated_branches >= 2:
            logger.debug(
                f"\tAccepted factorial split with {populated_branches}/{len(supports)} heavily populated branches on test sample."
            )

            # Second pass: Apply accepted constraint to full data slice
            candidate_slices = []
            for constraint in supports:
                full_subset_row_ids = apply_constraint(
                    data_slice=data_slice,
                    constraint=constraint,
                )
                new_slice = DataSlice(
                    data_slice.data,
                    full_subset_row_ids,
                    data_slice.col_ids,
                    constraints=data_slice.constraints.intersect(constraint),
                )
                candidate_slices.append(new_slice)

            return candidate_slices

    logger.debug(f"Failed to find a viable factorial partition after {max_tries} tries.")
    return []

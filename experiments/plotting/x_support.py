"""X-leaf support utilities for slice plots."""

from typing import List, Tuple

import numpy as np

from experiments.utils.circuit_inspect import (
    choose_x_values_from_intervals,
    get_x_leaf_support_intervals,
)


def get_x_support_contexts(
    ac,
    data_info: dict,
    df_obs=None,
    pad_fraction: float = 0.05,
) -> Tuple[np.ndarray, List[str], List[tuple]]:
    """Return one representative X value per leaf support interval.

    Args:
        ac: trained arithmetic circuit.
        data_info: dict returned by ``experiments.utils.data.load_data``.
        df_obs: optional observational dataframe for clipping infinite tails.
        pad_fraction: padding used when clipping infinite support bounds.

    Returns:
        (x_values, labels, intervals): arrays of X contexts, human-readable
        labels, and the raw support intervals ``(low, high)`` the contexts
        were chosen from (used for histogram sample selection).
    """
    intervals = get_x_leaf_support_intervals(ac, data_info["x_id"])
    if not intervals:
        raise ValueError("No X-leaf support intervals found in the circuit.")
    x_values, labels = choose_x_values_from_intervals(
        intervals,
        df_obs=df_obs,
        x_id_name=data_info["var_to_id"].get(data_info["x_id"], "X"),
        pad_fraction=pad_fraction,
    )
    return x_values, labels, intervals

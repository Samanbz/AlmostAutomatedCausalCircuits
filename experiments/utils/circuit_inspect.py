"""Introspection helpers for X-leaf supports and split points."""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


def resolve_context_xs(
    ac,
    data_info: Dict[str, Any],
    eval_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, List[str]]:
    """Pick representative X values: one per X-leaf support, or a linspace fallback."""
    df_obs = data_info["df_obs"]
    x_intervals = get_x_leaf_support_intervals(ac, data_info["x_id"])
    return choose_x_values_from_intervals(x_intervals, df_obs=df_obs, x_id_name="X")


def get_x_split_points(ac, x_id: int) -> List[float]:
    """Return sorted finite X-leaf support boundaries."""
    split_points = set()
    for leaf_id in ac.get_leaves():
        leaf = ac.get_node_data(leaf_id)
        if getattr(leaf, "var", None) != x_id:
            continue
        if hasattr(leaf, "_split_points"):
            with torch.no_grad():
                for b in leaf._split_points().cpu().numpy().ravel():
                    if not math.isinf(b):
                        split_points.add(float(b))
        else:
            supports = getattr(leaf, "node_supports", None)
            if supports is None:
                continue
            for supp in supports:
                interval = supp.intervals.get(x_id)
                if interval is None:
                    continue
                if not math.isinf(interval.low):
                    split_points.add(float(interval.low))
                if not math.isinf(interval.high):
                    split_points.add(float(interval.high))
    return sorted(split_points)


def get_x_leaf_support_intervals(ac, x_id: int) -> List[Tuple[float, float]]:
    """Return support intervals for the X leaf."""
    for leaf_id in ac.get_leaves():
        leaf = ac.get_node_data(leaf_id)
        if getattr(leaf, "var", None) != x_id:
            continue
        node_supports = getattr(leaf, "node_supports", None)
        if not node_supports:
            continue
        intervals = []
        for supp in node_supports:
            iv = supp.intervals.get(x_id)
            if iv is None:
                continue
            intervals.append((float(iv.low), float(iv.high)))
        return intervals
    return []


def choose_x_values_from_intervals(
    intervals: List[Tuple[float, float]],
    df_obs: Optional[pd.DataFrame] = None,
    x_id_name: str = "X",
    pad_fraction: float = 0.05,
) -> Tuple[np.ndarray, List[str]]:
    """Pick one representative X value inside each leaf support interval."""
    if df_obs is not None and x_id_name in df_obs.columns:
        x_data = df_obs[x_id_name].values
        x_min, x_max = float(np.percentile(x_data, 0.5)), float(np.percentile(x_data, 99.5))
    else:
        x_min, x_max = float("-inf"), float("inf")

    finite_bounds = [b for low, high in intervals for b in (low, high) if math.isfinite(b)]
    if finite_bounds:
        span = max(finite_bounds) - min(finite_bounds)
        if span <= 0:
            # Degenerate case: e.g. two half-open tails sharing a single finite
            # split point. Fall back to the observational data range (if given)
            # so the tail padding stays non-zero.
            span = (x_max - x_min) if x_min > float("-inf") and x_max < float("inf") else 1.0
    else:
        span = 1.0
    pad = pad_fraction * span

    values, labels = [], []
    for i, (low, high) in enumerate(intervals):
        if math.isfinite(low) and math.isfinite(high):
            x_val = 0.5 * (low + high)
            label = f"X={x_val:.2f} (support {i})"
        elif math.isfinite(low):
            candidate = low + pad
            if x_max < float("inf"):
                candidate = min(candidate, x_max)
            x_val = max(low, candidate)
            label = f"X={x_val:.2f} (right tail {i})"
        elif math.isfinite(high):
            candidate = high - pad
            if x_min > float("-inf"):
                candidate = max(candidate, x_min)
            x_val = min(high, candidate)
            label = f"X={x_val:.2f} (left tail {i})"
        else:
            x_val = 0.5 * (x_min + x_max) if x_min > float("-inf") and x_max < float("inf") else 0.0
            label = f"X={x_val:.2f} (full range {i})"
        values.append(x_val)
        labels.append(label)
    return np.array(values), labels

"""Empirical ground-truth density estimates from observational/interventional data."""

from typing import Optional

import numpy as np
import pandas as pd


def py_given_x_kde(
    df: pd.DataFrame,
    x_val: float,
    grid_y: np.ndarray,
    x_col: str = "X",
    y_col: str = "Y",
    x_bandwidth: Optional[float] = None,
) -> np.ndarray:
    """Gaussian conditional KDE estimate of P(Y | X = x_val).

    Uses a hard 3-bandwidth window for speed and falls back to a uniform
    histogram if too few samples are available.
    """
    x_vals = df[x_col].values
    y_vals = df[y_col].values

    if x_bandwidth is None:
        x_bandwidth = max(0.1, (x_vals.max() - x_vals.min()) / 20.0)

    window = 1.0 * x_bandwidth
    mask = np.abs(x_vals - x_val) <= window
    x_sel = x_vals[mask]
    y_sel = y_vals[mask]
    n = len(y_sel)

    if n == 0:
        return np.zeros_like(grid_y)

    x_weights = np.exp(-0.5 * ((x_sel - x_val) / x_bandwidth) ** 2)
    weight_sum = x_weights.sum()
    if weight_sum <= 0:
        return np.zeros_like(grid_y)

    std = np.std(y_sel)
    iqr = np.subtract(*np.percentile(y_sel, [75, 25]))
    h = 0.9 * min(std, iqr / 1.34) * n ** (-0.2) if n > 1 else 1.0
    y_bandwidth = max(h, 1e-6)

    diffs = grid_y[:, None] - y_sel[None, :]
    kernels = np.exp(-0.5 * (diffs / y_bandwidth) ** 2)
    kernels /= y_bandwidth * np.sqrt(2.0 * np.pi)

    return (x_weights[None, :] * kernels).sum(axis=1) / weight_sum


def empirical_densities_at_x(
    df_obs: pd.DataFrame,
    df_do: pd.DataFrame,
    x_val: float,
    grid_y: np.ndarray,
    x_col: str = "X",
    y_col: str = "Y",
) -> dict:
    """Return empirical P(Y|X=x) and P(Y|do(X=x)) as dictionaries."""
    return {
        "obs": py_given_x_kde(df_obs, x_val, grid_y, x_col=x_col, y_col=y_col),
        "do": py_given_x_kde(df_do, x_val, grid_y, x_col=x_col, y_col=y_col),
    }


def py_given_x_hist(
    df: pd.DataFrame,
    x_interval: tuple,
    bin_edges: np.ndarray,
    x_col: str = "X",
    y_col: str = "Y",
) -> tuple:
    """Plain density histogram of Y over the samples whose X falls in ``x_interval``.

    This is the unbiased, assumption-free counterpart to :func:`py_given_x_kde`:
    no bandwidth choice, no kernel shape — just the empirical counts in the
    X-leaf support bin, normalized to a density on ``bin_edges``.

    Returns:
        (density, n): per-bin density (0 for empty bins) and the sample count.
    """
    lo, hi = x_interval
    x_vals = df[x_col].values
    y_vals = df[y_col].values
    mask = (x_vals >= lo) & (x_vals <= hi)
    n = int(mask.sum())
    if n == 0:
        return np.zeros(len(bin_edges) - 1, dtype=float), 0
    density, _ = np.histogram(y_vals[mask], bins=bin_edges, density=True)
    return density.astype(float), n


def empirical_histograms_at_x(
    df_obs: pd.DataFrame,
    df_do: pd.DataFrame,
    x_interval: tuple,
    bin_edges: np.ndarray,
    x_col: str = "X",
    y_col: str = "Y",
) -> dict:
    """Histogram-based empirical P(Y|X in I) and P(Y|do(X in I)).

    ``df_do`` rows are paired draws from do(X = x_i); selecting the rows whose
    observed/interventional X value lies in the support interval I yields the
    empirical interventional Y distribution for that X region.
    """
    obs, n_obs = py_given_x_hist(df_obs, x_interval, bin_edges, x_col, y_col)
    do, n_do = py_given_x_hist(df_do, x_interval, bin_edges, x_col, y_col)
    return {"obs": obs, "do": do, "n_obs": n_obs, "n_do": n_do}

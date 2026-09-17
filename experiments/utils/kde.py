"""Gaussian KDE estimates of conditional densities P(Y | X) from dataframes."""

from typing import Optional

import numpy as np
import pandas as pd


def py_given_x_kde(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_val: float,
    grid_y: np.ndarray,
    x_bandwidth: Optional[float] = None,
    y_bandwidth: Optional[float] = None,
) -> np.ndarray:
    """Fast Gaussian conditional KDE estimate of P(Y | X = x_val)."""
    x_vals = df[x_col].values
    y_vals = df[y_col].values

    if x_bandwidth is None:
        x_bandwidth = max(0.1, (x_vals.max() - x_vals.min()) / 20.0)

    window = 3.0 * x_bandwidth
    mask = np.abs(x_vals - x_val) <= window
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]
    n = len(y_vals)
    if n == 0:
        return np.zeros_like(grid_y)

    x_weights = np.exp(-0.5 * ((x_vals - x_val) / x_bandwidth) ** 2)
    weight_sum = x_weights.sum()
    if weight_sum <= 0:
        return np.zeros_like(grid_y)

    if y_bandwidth is None:
        std = np.std(y_vals)
        iqr = np.subtract(*np.percentile(y_vals, [75, 25]))
        h = 0.9 * min(std, iqr / 1.34) * n ** (-0.2) if n > 1 else 1.0
        y_bandwidth = max(h, 1e-6)

    diffs = grid_y[:, None] - y_vals[None, :]
    y_kernels = np.exp(-0.5 * (diffs / y_bandwidth) ** 2)
    y_kernels /= y_bandwidth * np.sqrt(2.0 * np.pi)
    kde = (x_weights[None, :] * y_kernels).sum(axis=1) / weight_sum
    return kde


def py_given_x_surface(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    grid_x: np.ndarray,
    y_edges: np.ndarray,
    min_x_count: int = 100,
    x_bandwidth: Optional[float] = None,
) -> np.ndarray:
    """P(Y | X) surface (rows=Y bins, cols=X bins) via conditional KDE."""
    grid_y = (y_edges[:-1] + y_edges[1:]) / 2.0
    surface = np.zeros((len(grid_y), len(grid_x)))
    x_vals = df[x_col].values
    if x_bandwidth is None:
        x_bandwidth = max(0.1, (grid_x[-1] - grid_x[0]) / 20.0)

    for i, x_c in enumerate(grid_x):
        window = 3.0 * x_bandwidth
        mask = np.abs(x_vals - x_c) <= window
        if mask.sum() >= min_x_count:
            surface[:, i] = py_given_x_kde(df, x_col, y_col, x_c, grid_y, x_bandwidth=x_bandwidth)
        else:
            surface[:, i] = np.nan
    return surface

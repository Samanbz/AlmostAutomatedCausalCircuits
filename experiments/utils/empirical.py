"""Empirical ground-truth estimates of conditional densities P(Y | X) from dataframes.

Two estimators, both deliberately simple and parameter-light:

* 2-D heatmaps use a plain histogram of the empirical joint, column-normalised
  to a conditional density. No kernel bandwidths at all.
* 1-D slices use the ``SLICE_WINDOW_COUNT`` samples nearest in X (an adaptive
  window: it narrows where data is dense), with a small Gaussian KDE in Y and
  bootstrap confidence bands so estimation noise is visible instead of looking
  like circuit error.

Caveat (inherent to any empirical conditional, not a bug of these estimators):
averaging Y over an X-window of half-width ``w`` when the conditional mean has
slope ``m`` inflates the apparent conditional std by roughly ``m * w``. The
window half-width is therefore reported per slice so the inflation is
transparent, and it is largest exactly for steep conditionals.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# Fixed estimator constants (kept in code, not config, since they are not
# meant to be tuned per experiment).
SLICE_WINDOW_COUNT = 3000
SLICE_BOOTSTRAP_REPS = 200


def py_given_x_surface_hist(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    min_x_count: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """P(Y | X) surface (rows=Y bins, cols=X bins) as a plain 2-D histogram.

    Each X column is normalised to a conditional density (integrates to 1 over
    the Y bins). Columns with fewer than ``min_x_count`` samples are NaN.

    Returns ``(surface, x_counts)`` where ``x_counts[i]`` is the number of
    samples in X-bin ``i`` (0 for masked columns).
    """
    counts, _, _ = np.histogram2d(df[x_col].values, df[y_col].values, bins=[x_edges, y_edges])
    x_counts = counts.sum(axis=1)
    dy = y_edges[1] - y_edges[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        surface = counts.T / (x_counts * dy)
    surface[:, x_counts < min_x_count] = np.nan
    return surface, x_counts


def _y_kde(y_vals: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Gaussian KDE of ``y_vals`` on ``grid_y`` with Silverman bandwidth."""
    n = len(y_vals)
    if n < 2:
        return np.zeros_like(grid_y)
    std = np.std(y_vals)
    iqr = np.subtract(*np.percentile(y_vals, [75, 25]))
    h = max(0.9 * min(std, iqr / 1.34) * n ** (-0.2), 1e-6)
    diffs = grid_y[:, None] - y_vals[None, :]
    kernels = np.exp(-0.5 * (diffs / h) ** 2)
    return kernels.sum(axis=1) / (n * h * np.sqrt(2.0 * np.pi))


def py_given_x_slice(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_val: float,
    grid_y: np.ndarray,
    window_count: int = SLICE_WINDOW_COUNT,
    n_boot: int = SLICE_BOOTSTRAP_REPS,
    seed: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Empirical P(Y | X = x_val) from the ``window_count`` nearest X samples.

    Returns ``(density, boot_lo, boot_hi, half_width, n_used)`` where
    ``boot_lo``/``boot_hi`` are pointwise percentile bootstrap bands and
    ``half_width`` is the X distance to the farthest sample in the window
    (the quantity that controls the window-inflation of the apparent
    conditional variance, see module docstring).
    """
    x_vals = df[x_col].values
    y_vals = df[y_col].values
    n_total = len(y_vals)
    n_used = min(window_count, n_total)
    if n_used < 2:
        zeros = np.zeros_like(grid_y)
        return zeros, zeros, zeros, 0.0, n_used

    order = np.argsort(np.abs(x_vals - x_val), kind="stable")[:n_used]
    y_win = y_vals[order]
    half_width = float(np.abs(x_vals[order] - x_val).max())

    density = _y_kde(y_win, grid_y)

    if n_boot > 0:
        rng = np.random.default_rng(seed)
        boots = np.empty((n_boot, len(grid_y)))
        for b in range(n_boot):
            resample = rng.integers(0, n_used, n_used)
            boots[b] = _y_kde(y_win[resample], grid_y)
        boot_lo = np.percentile(boots, 2.5, axis=0)
        boot_hi = np.percentile(boots, 97.5, axis=0)
    else:
        boot_lo = boot_hi = density

    return density, boot_lo, boot_hi, half_width, n_used

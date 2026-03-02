from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.symbolic import StructuralCausalModel


def synthesize_data(scm: StructuralCausalModel, n_samples: int) -> pd.DataFrame:
    """
    A clean wrapper for the data generation process.
    """
    print(f"Starting data generation for {n_samples} samples...")
    try:
        df = scm.sample(n_samples)
        print("Data generation successful.")
        return df
    except Exception as e:
        print(f"Data generation failed: {e}")
        raise e


def normal(mean: float, std: float) -> Callable[[int], np.ndarray]:
    """Returns a lambda that safely generates Gaussian noise for n samples."""
    return lambda n: np.random.normal(loc=mean, scale=std, size=n)


def uniform(low: float, high: float) -> Callable[[int], np.ndarray]:
    """Returns a lambda that safely generates Uniform noise for n samples."""
    return lambda n: np.random.uniform(low, high, size=n)


def plot_bool_matrix(matrix: np.ndarray, title: str = "Boolean Matrix"):
    """
    Utility function to visualize a boolean matrix as a grid.
    Black for 1 (True), White for 0 (False).
    """
    rows, cols = matrix.shape
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap="binary", vmin=0, vmax=1, aspect="equal", interpolation="nearest")

    # Gridlines configuration
    ax = plt.gca()

    if rows * cols < 400:  # Only show labels if matrix is small enough
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
    else:
        # For large matrices, turn off labels to avoid clutter
        ax.set_xticks([])
        ax.set_yticks([])

    # Minor ticks at half-integers (-0.5, 0.5...) for gridlines
    # Use simpler grid generation for large matrices to avoid performance hit
    if rows * cols < 2500:
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)
    else:
        # If too big, skip the gridlines or they will just make it gray block
        pass

    # Ensure major gridlines are off (in case they are on by default)
    ax.grid(which="major", visible=False)
    # Remove minor tick marks (the little lines sticking out)
    ax.tick_params(which="minor", size=0)
    ax.tick_params(which="major", size=0)

    plt.title(title)
    plt.tight_layout()
    plt.show()

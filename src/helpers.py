from typing import Callable

import numpy as np
import pandas as pd
import torch

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


def create_sparse_stochastic_matrix(n, m=None, sparsity=0.9):
    if m is None:
        m = n
    A = torch.rand(n, m)
    mask = (torch.rand_like(A) > sparsity).float()
    A_sparse = A * mask
    zero_rows = A_sparse.sum(dim=1) == 0
    if zero_rows.any():
        A_sparse[zero_rows, torch.randint(0, m, (zero_rows.sum(),))] = 1.0
    row_sums = A_sparse.sum(dim=1, keepdim=True)
    return A_sparse / row_sums


def create_banded_stochastic_matrix(n, m=None, bandwidth=1):
    if m is None:
        m = n
    A = torch.zeros(n, m)
    for i in range(n):
        if n > 1:
            center = i * (m - 1) / (n - 1)
        else:
            center = (m - 1) / 2

        c = int(center + 0.5)
        start = max(0, c - bandwidth)
        end = min(m, c + bandwidth + 1)

        if start < end:
            A[i, start:end] = torch.rand(end - start)

    row_sums = A.sum(dim=1, keepdim=True)
    # Handle rows that might sum to zero (possible in rectangular banded matrices) by avoiding division by zero
    row_sums[row_sums == 0] = 1.0
    return A / row_sums


def create_dense_stochastic_matrix(n, m=None):
    """
    Generates a dense n x m matrix where all elements are > 0 and each row sums to 1.
    """
    if m is None:
        m = n
    A = torch.rand(n, m) + 1e-6
    row_sums = A.sum(dim=1, keepdim=True)
    return A / row_sums

import numpy as np
import pandas as pd
from typing import Callable
from scm import StructuralCausalModel

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
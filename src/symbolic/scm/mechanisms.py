"""Mechanism base classes, noise samplers, and tractability interfaces.

Every mechanism implements the :class:`Mechanism` sampling contract. Mechanisms that
admit *analytical* ground truth additionally implement one of two interfaces:

- :class:`TabularMechanism` — discrete mechanisms exposing an exact conditional
  probability table (CPT).
- :class:`ConditionalLinearGaussian` — continuous mechanisms whose conditional law,
  given the configuration of their discrete parents, is linear in their continuous
  parents plus Gaussian-mixture noise.

The exact ground-truth engine in ``ground_truth.py`` dispatches on these interfaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class Mechanism(ABC):
    """Abstract base class for data generation logic."""

    @abstractmethod
    def __call__(self, n_samples: int, **parents: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, noise: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        """Evaluate the mechanism with a given exogenous noise."""
        pass

    @abstractmethod
    def abduct(self, value: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        """Infer the exogenous noise given the output value and parent values."""
        pass


class ConstantMechanism(Mechanism):
    """
    Mechanism that ignores parents and returns a constant array.
    """

    def __init__(self, value: float):
        self.value = value

    def __call__(self, n_samples: int, **parents: np.ndarray) -> np.ndarray:
        return np.full(n_samples, self.value, dtype=np.float64)

    def evaluate(self, noise: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        return np.full(len(noise), self.value, dtype=np.float64)

    def abduct(self, value: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        return np.zeros_like(value)

    def __str__(self) -> str:
        return str(self.value)


class GaussianNoise:
    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def __call__(self, n_samples: int) -> np.ndarray:
        return np.random.normal(self.loc, self.scale, size=n_samples)

    def __str__(self):
        return f"N({self.loc:.2f}, {self.scale:.2f})"


class GaussianMixtureNoise:
    """One-dimensional Gaussian-mixture noise: sum_k weights[k] * N(means[k], stds[k]^2)."""

    def __init__(self, weights: Sequence[float], means: Sequence[float], stds: Sequence[float]):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.means = np.asarray(means, dtype=np.float64)
        self.stds = np.asarray(stds, dtype=np.float64)
        if not (self.weights.shape == self.means.shape == self.stds.shape):
            raise ValueError("weights, means and stds must have the same length.")
        if self.weights.ndim != 1 or len(self.weights) < 1:
            raise ValueError("weights must be a non-empty 1-D array.")
        if np.any(self.stds <= 0):
            raise ValueError("stds must be positive.")
        if np.any(self.weights < 0) or not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("weights must be non-negative and sum to 1.")

    @property
    def n_components(self) -> int:
        return len(self.weights)

    @classmethod
    def single(cls, mean: float, std: float) -> "GaussianMixtureNoise":
        return cls([1.0], [mean], [std])

    def __call__(self, n_samples: int) -> np.ndarray:
        components = np.random.choice(self.n_components, size=n_samples, p=self.weights)
        return np.random.normal(self.means[components], self.stds[components])

    def __str__(self):
        terms = [
            f"{w:.2f}*N({m:.2f}, {s:.2f})" for w, m, s in zip(self.weights, self.means, self.stds)
        ]
        return " + ".join(terms)


class UniformNoise:
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def __call__(self, n_samples: int) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=n_samples)

    def __str__(self):
        return f"U({self.low:.2f}, {self.high:.2f})"


class LinearLogic:
    def __init__(self, coefficients: dict[str, float], intercept: float = 0.0):
        self.coefficients = coefficients
        self.intercept = intercept

    def __call__(self, **kwargs) -> np.ndarray:
        if not kwargs:
            return self.intercept
        first_val = next(iter(kwargs.values()))
        res = np.full_like(first_val, self.intercept, dtype=np.float64)
        for k, v in kwargs.items():
            res += self.coefficients[k] * v
        return res

    def __str__(self):
        terms = [f"{v:.2f}*{k}" for k, v in self.coefficients.items()]
        eq = " + ".join(terms) if terms else "0"
        return f"{self.intercept:.2f} + {eq}"


class LogisticLogic:
    def __init__(self, coefficients: dict[str, float], intercept: float = 0.0):
        self.coefficients = coefficients
        self.intercept = intercept

    def __call__(self, **kwargs) -> np.ndarray:
        if not kwargs:
            return np.full(1, 1 / (1 + np.exp(-self.intercept)), dtype=np.float64)

        first_val = next(iter(kwargs.values()))
        logits = np.full_like(first_val, self.intercept, dtype=np.float64)
        for k, v in kwargs.items():
            logits += self.coefficients[k] * v
        return 1 / (1 + np.exp(-logits))

    def __str__(self):
        terms = [f"{v:.2f}*{k}" for k, v in self.coefficients.items()]
        eq = " + ".join(terms) if terms else "0"
        return f"σ({eq} + {self.intercept:.2f})"


@dataclass
class GaussianParams:
    """Parameters of Y | discrete-config = intercept + sum(coefs * X) + noise."""

    intercept: float
    coefficients: Dict[str, float]
    noise: GaussianMixtureNoise


class TabularMechanism(Mechanism):
    """Interface for discrete mechanisms with an exact conditional probability table."""

    cardinality: int

    @abstractmethod
    def cpt(self, parent_names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Exact P(V | parents) of shape (*parent_cardinalities, cardinality).

        Axes follow ``parent_names`` order. Mechanisms that store their parent layout
        may be called with ``parent_names=None``.
        """
        pass

    @abstractmethod
    def parent_cardinalities(self) -> Dict[str, int]:
        pass


class ConditionalLinearGaussian(Mechanism):
    """Interface for continuous mechanisms in conditional-linear-Gaussian form.

    Given the joint configuration ``d`` of the discrete parents, the mechanism is
    ``Y = intercept_d + sum_j coef_d[j] * X_j + noise_d`` with Gaussian-mixture noise.
    The plain linear-GMM case is the special case with no discrete parents.
    """

    @property
    @abstractmethod
    def discrete_parent_cardinalities(self) -> Dict[str, int]:
        pass

    @property
    @abstractmethod
    def continuous_parent_names(self) -> List[str]:
        pass

    @abstractmethod
    def gaussian_params(self, discrete_config: Tuple[int, ...] = ()) -> GaussianParams:
        """Parameters for one discrete parent configuration (ordered as
        ``discrete_parent_cardinalities``)."""
        pass


def mixed_radix_index(values: Dict[str, np.ndarray], cardinalities: Dict[str, int]) -> np.ndarray:
    """Flatten per-row parent configurations into a single integer index.

    ``cardinalities`` iteration order defines the digit order (first entry = most
    significant digit). Values must be integer-valued arrays in ``[0, card)``.
    """
    if not cardinalities:
        return np.zeros(len(next(iter(values.values()))) if values else 1, dtype=np.int64)
    names = list(cardinalities)
    missing = [n for n in names if n not in values]
    if missing:
        raise KeyError(f"Missing parent values for {missing}.")
    index = np.zeros(len(values[names[0]]), dtype=np.int64)
    for name in names:
        vals = np.asarray(values[name], dtype=np.int64)
        card = cardinalities[name]
        if np.any(vals < 0) or np.any(vals >= card):
            raise ValueError(f"Parent '{name}' has values outside [0, {card}).")
        index = index * card + vals
    return index

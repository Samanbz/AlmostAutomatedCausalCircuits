"""Continuous mechanisms in conditional-linear-Gaussian (CLG) form.

- :class:`AdditiveNoiseMechanism` — legacy ``Y = f(parents) + noise``; exposes exact
  Gaussian parameters when the logic is linear and the noise is Gaussian / GMM.
- :class:`LinearGMMMechanism` — linear mechanism with Gaussian-mixture noise.
- :class:`CLGMechanism` — discrete parents act as regime switches selecting among
  linear-Gaussian mechanisms (Lauritzen & Wermuth 1989).
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .mechanisms import (
    ConditionalLinearGaussian,
    GaussianMixtureNoise,
    GaussianNoise,
    GaussianParams,
    LinearLogic,
    Mechanism,
    mixed_radix_index,
)


class AdditiveNoiseMechanism(Mechanism):
    """
    Y = f(Parents) + Noise
    """

    def __init__(
        self, logic: Optional[Callable[..., np.ndarray]], noise_dist: Callable[[int], np.ndarray]
    ):
        self.logic = logic
        self.noise_dist = noise_dist

    def __call__(self, n_samples: int, **parents: np.ndarray) -> np.ndarray:
        # Generate noise (Ensure size=n_samples is handled by the generic utils or passed callable)
        noise = self.noise_dist(n_samples)

        return self.evaluate(noise, **parents)

    def evaluate(self, noise: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        if self.logic is None:
            return noise

        try:
            deterministic = self.logic(**parents)
        except TypeError as e:
            raise TypeError(f"Mechanism arguments mismatch. {e}") from e

        return deterministic + noise

    def abduct(self, value: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        if self.logic is None:
            return value

        try:
            deterministic = self.logic(**parents)
        except TypeError as e:
            raise TypeError(f"Mechanism arguments mismatch. {e}") from e

        return value - deterministic

    def gaussian_params(self, discrete_config: Tuple[int, ...] = ()) -> GaussianParams:
        """Exact Gaussian parameters when the mechanism is linear with Gaussian/GMM noise."""
        if discrete_config:
            raise ValueError("AdditiveNoiseMechanism has no discrete parents.")
        if self.logic is not None and not isinstance(self.logic, LinearLogic):
            raise ValueError(f"No exact ground truth for non-linear logic {type(self.logic)}.")
        if isinstance(self.noise_dist, GaussianNoise):
            noise = GaussianMixtureNoise.single(self.noise_dist.loc, self.noise_dist.scale)
        elif isinstance(self.noise_dist, GaussianMixtureNoise):
            noise = self.noise_dist
        else:
            raise ValueError(f"No exact ground truth for noise of type {type(self.noise_dist)}.")
        if self.logic is None:
            return GaussianParams(intercept=0.0, coefficients={}, noise=noise)
        return GaussianParams(
            intercept=float(self.logic.intercept),
            coefficients=dict(self.logic.coefficients),
            noise=noise,
        )

    def __str__(self) -> str:
        logic_str = str(self.logic) if self.logic else "0"
        noise_str = str(self.noise_dist) if self.noise_dist else "Noise"
        return f"{logic_str} + {noise_str}"


class LinearGMMMechanism(AdditiveNoiseMechanism, ConditionalLinearGaussian):
    """Y = intercept + sum_j coefficients[j] * X_j + U, U ~ GaussianMixtureNoise."""

    def __init__(
        self,
        coefficients: Dict[str, float],
        intercept: float,
        noise: GaussianMixtureNoise,
    ):
        # `self.coefficients` and the logic's dict must be the SAME object: sampling
        # reads `logic`, ground truth reads `coefficients` — mutating one and not the
        # other (e.g. effect engineering) would silently desync data from ground truth.
        self.coefficients = dict(coefficients)
        super().__init__(logic=LinearLogic(self.coefficients, intercept), noise_dist=noise)
        self.intercept = float(intercept)
        self.noise = noise

    @property
    def discrete_parent_cardinalities(self) -> Dict[str, int]:
        return {}

    @property
    def continuous_parent_names(self) -> List[str]:
        return list(self.coefficients)

    def gaussian_params(self, discrete_config: Tuple[int, ...] = ()) -> GaussianParams:
        if discrete_config:
            raise ValueError("LinearGMMMechanism has no discrete parents.")
        return GaussianParams(
            intercept=self.intercept, coefficients=dict(self.coefficients), noise=self.noise
        )

    def __str__(self) -> str:
        return f"{self.logic} + {self.noise}"


class CLGMechanism(ConditionalLinearGaussian):
    """Conditional linear Gaussian mechanism.

    Y | D = d, X = x ~ N(intercepts[d] + coefs[d] @ x, stds[d]^2)

    The discrete parents D act as regime switches: each joint configuration d selects
    its own linear-Gaussian mechanism. ``evaluate``/``abduct`` use a *standard* normal
    noise (scaled by stds[d] internally), so abduction is exact.
    """

    def __init__(
        self,
        discrete_parent_cardinalities: Dict[str, int],
        continuous_parent_names: List[str],
        intercepts: np.ndarray,
        coefficients: np.ndarray,
        stds: np.ndarray,
    ):
        self._discrete_parent_cardinalities = dict(discrete_parent_cardinalities)
        self._continuous_parent_names = list(continuous_parent_names)
        self.intercepts = np.asarray(intercepts, dtype=np.float64)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.stds = np.asarray(stds, dtype=np.float64)

        n_configs = int(np.prod(list(self._discrete_parent_cardinalities.values()), dtype=np.int64))
        n_cont = len(self._continuous_parent_names)
        if self.intercepts.shape != (n_configs,):
            raise ValueError(
                f"intercepts must have shape ({n_configs},), got {self.intercepts.shape}."
            )
        if self.coefficients.shape != (n_configs, n_cont):
            raise ValueError(
                f"coefficients must have shape ({n_configs}, {n_cont}), got {self.coefficients.shape}."
            )
        if self.stds.shape != (n_configs,) or np.any(self.stds <= 0):
            raise ValueError(f"stds must have shape ({n_configs},) and be positive.")

    @property
    def discrete_parent_cardinalities(self) -> Dict[str, int]:
        return dict(self._discrete_parent_cardinalities)

    @property
    def continuous_parent_names(self) -> List[str]:
        return list(self._continuous_parent_names)

    def _config_means(self, parents: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Per-row config index and conditional mean."""
        discrete_values = {n: parents[n] for n in self._discrete_parent_cardinalities}
        config_idx = mixed_radix_index(discrete_values, self._discrete_parent_cardinalities)
        mean = self.intercepts[config_idx].copy()
        for j, name in enumerate(self._continuous_parent_names):
            mean = mean + self.coefficients[config_idx, j] * np.asarray(
                parents[name], dtype=np.float64
            )
        return config_idx, mean

    def gaussian_params(self, discrete_config: Tuple[int, ...] = ()) -> GaussianParams:
        cards = self._discrete_parent_cardinalities
        if len(discrete_config) != len(cards):
            raise ValueError(
                f"Expected a config of {len(cards)} discrete parents, got {discrete_config}."
            )
        idx = 0
        for name, value in zip(cards, discrete_config):
            if not (0 <= value < cards[name]):
                raise ValueError(f"Config value {value} out of range for parent '{name}'.")
            idx = idx * cards[name] + int(value)
        return GaussianParams(
            intercept=float(self.intercepts[idx]),
            coefficients={
                name: float(self.coefficients[idx, j])
                for j, name in enumerate(self._continuous_parent_names)
            },
            noise=GaussianMixtureNoise.single(0.0, float(self.stds[idx])),
        )

    def __call__(self, n_samples: int, **parents: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0.0, 1.0, size=n_samples)
        return self.evaluate(noise, **parents)

    def evaluate(self, noise: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        config_idx, mean = self._config_means(parents)
        return mean + self.stds[config_idx] * np.asarray(noise, dtype=np.float64)

    def abduct(self, value: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        config_idx, mean = self._config_means(parents)
        return (np.asarray(value, dtype=np.float64) - mean) / self.stds[config_idx]

    def __str__(self) -> str:
        return f"CLG(configs={len(self.intercepts)}, cont_parents={self._continuous_parent_names})"

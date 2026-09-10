"""Discrete mechanisms with exact conditional probability tables.

- :class:`BinaryMechanism` — legacy logistic-threshold binary mechanism.
- :class:`RegionalDiscreteMechanism` — CausalProfiler-style regional mechanism
  (Panayiotou et al. 2026, App. D): continuous uniform noise partitioned into R
  regions, each region carrying a distinct mapping from parent configurations to
  values. R = 1 is deterministic; larger R is more stochastic.
- :class:`DirichletCPTMechanism` — plain CPT with Dirichlet-sampled rows.
"""

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .mechanisms import TabularMechanism, mixed_radix_index


class BinaryMechanism(TabularMechanism):
    """
    Y = 1 if U < P(Y=1 | Parents) else 0
    where U ~ Uniform(0, 1)
    """

    def __init__(self, logic: Optional[Callable[..., np.ndarray]] = None, base_p: float = 0.5):
        self.logic = logic
        self.base_p = base_p

    @property
    def cardinality(self) -> int:
        return 2

    def parent_cardinalities(self) -> Dict[str, int]:
        if self.logic is None:
            return {}
        return dict.fromkeys(self.logic.coefficients, 2)

    def cpt(self, parent_names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Exact CPT of shape (2, ..., 2, 2); axes follow ``parent_names`` order."""
        cards = self.parent_cardinalities()
        names = list(cards) if parent_names is None else list(parent_names)
        if set(names) != set(cards):
            raise ValueError(f"parent_names {names} do not match mechanism parents {list(cards)}.")
        if not names:
            p1 = np.asarray(self.logic(), dtype=np.float64).item() if self.logic else self.base_p
            return np.array([1.0 - p1, p1])
        grid = np.indices([2] * len(names), dtype=np.float64).reshape(len(names), -1)
        parent_data = {name: grid[i] for i, name in enumerate(names)}
        p1 = np.asarray(self.logic(**parent_data), dtype=np.float64).reshape([2] * len(names))
        return np.stack([1.0 - p1, p1], axis=-1)

    def __call__(self, n_samples: int, **parents: np.ndarray) -> np.ndarray:
        noise = np.random.uniform(0, 1, size=n_samples)
        return self.evaluate(noise, **parents)

    def evaluate(self, noise: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        if self.logic is None:
            probs = np.full(len(noise), self.base_p, dtype=np.float64)
        else:
            probs = self.logic(**parents)
        return (noise < probs).astype(np.float64)

    def abduct(self, value: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Abduction is ill-posed for threshold-based binary SCMs.")

    def __str__(self) -> str:
        if self.logic is None:
            return f"Bernoulli({self.base_p})"
        return f"Bernoulli({self.logic})"


class RegionalDiscreteMechanism(TabularMechanism):
    """Regional discrete mechanism (CausalProfiler App. D, Definition D.2).

    U ~ Unif[0, 1] is partitioned into R consecutive regions by sorted uniform cut
    points. Each region r carries a distinct mapping m_r : Omega_PA -> Omega_V, and
    f_V(pa, u) = m_r(pa) when u falls in region r. Since U is uniform, the exact
    conditional P(V | pa) is the sum of region widths over regions whose mapping sends
    pa to V.
    """

    def __init__(
        self,
        cardinality: int,
        parent_cardinalities: Dict[str, int],
        cut_points: np.ndarray,
        mappings: np.ndarray,
    ):
        self._cardinality = int(cardinality)
        self._parent_cardinalities = dict(parent_cardinalities)
        self.cut_points = np.asarray(cut_points, dtype=np.float64)
        self.mappings = np.asarray(mappings, dtype=np.int64)

        n_configs = int(np.prod(list(self._parent_cardinalities.values()), dtype=np.int64))
        if self.mappings.shape != (len(self.cut_points) - 1, n_configs):
            raise ValueError(
                f"mappings must have shape (n_regions, {n_configs}), got {self.mappings.shape}."
            )
        if self.cut_points[0] != 0.0 or self.cut_points[-1] != 1.0:
            raise ValueError("cut_points must start at 0 and end at 1.")
        if np.any(np.diff(self.cut_points) <= 0):
            raise ValueError("cut_points must be strictly increasing.")
        if np.any(self.mappings < 0) or np.any(self.mappings >= self._cardinality):
            raise ValueError("mappings contain values outside [0, cardinality).")

    @property
    def cardinality(self) -> int:
        return self._cardinality

    @property
    def n_regions(self) -> int:
        return len(self.cut_points) - 1

    def parent_cardinalities(self) -> Dict[str, int]:
        return dict(self._parent_cardinalities)

    @property
    def max_regions(self) -> int:
        """Number of distinct mappings Omega_PA -> Omega_V: C^(prod of parent cards)."""
        n_configs = int(np.prod(list(self._parent_cardinalities.values()), dtype=np.int64))
        return self._cardinality**n_configs

    @classmethod
    def random(
        cls,
        cardinality: int,
        parent_cardinalities: Dict[str, int],
        n_regions: int,
        rng: np.random.Generator,
    ) -> "RegionalDiscreteMechanism":
        """Sample a mechanism via sample-rejection (CausalProfiler Alg. 4)."""
        n_configs = int(np.prod(list(parent_cardinalities.values()), dtype=np.int64))
        max_regions = cardinality**n_configs
        n_regions = min(int(n_regions), max_regions)
        if n_regions < 1:
            raise ValueError("n_regions must be >= 1.")

        if n_regions == 1:
            cut_points = np.array([0.0, 1.0])
        else:
            interior = np.sort(rng.uniform(0.0, 1.0, size=n_regions - 1))
            cut_points = np.concatenate([[0.0], interior, [1.0]])

        mappings = np.empty((n_regions, n_configs), dtype=np.int64)
        for r in range(n_regions):
            while True:
                candidate = rng.integers(0, cardinality, size=n_configs)
                if r == 0 or not np.any(np.all(mappings[:r] == candidate, axis=1)):
                    mappings[r] = candidate
                    break
        return cls(cardinality, parent_cardinalities, cut_points, mappings)

    def cpt(self, parent_names: Optional[Sequence[str]] = None) -> np.ndarray:
        cards = self._parent_cardinalities
        names = list(cards) if parent_names is None else list(parent_names)
        if set(names) != set(cards):
            raise ValueError(f"parent_names {names} do not match mechanism parents {list(cards)}.")

        widths = np.diff(self.cut_points)
        n_configs = self.mappings.shape[1]
        flat = np.zeros((n_configs, self._cardinality), dtype=np.float64)
        configs = np.arange(n_configs)
        for r in range(self.n_regions):
            flat[configs, self.mappings[r]] += widths[r]

        table = flat.reshape([cards[n] for n in cards] + [self._cardinality])
        if names != list(cards):
            permutation = [list(cards).index(n) for n in names]
            table = np.transpose(table, axes=permutation + [len(names)])
        return table

    def __call__(self, n_samples: int, **parents: np.ndarray) -> np.ndarray:
        noise = np.random.uniform(0.0, 1.0, size=n_samples)
        return self.evaluate(noise, **parents)

    def evaluate(self, noise: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        config_idx = mixed_radix_index(parents, self._parent_cardinalities)
        noise = np.asarray(noise, dtype=np.float64)
        region = np.clip(
            np.searchsorted(self.cut_points, noise, side="right") - 1, 0, self.n_regions - 1
        )
        return self.mappings[region, config_idx]

    def abduct(self, value: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Abduction is set-valued for regional discrete mechanisms.")

    def __str__(self) -> str:
        return f"Regional(C={self._cardinality}, R={self.n_regions})"


class DirichletCPTMechanism(TabularMechanism):
    """Discrete mechanism with an explicit CPT; rows sampled from Dirichlet(alpha * 1)."""

    def __init__(
        self,
        cardinality: int,
        parent_cardinalities: Dict[str, int],
        table: np.ndarray,
    ):
        self._cardinality = int(cardinality)
        self._parent_cardinalities = dict(parent_cardinalities)
        self.table = np.asarray(table, dtype=np.float64)
        expected = tuple(self._parent_cardinalities.values()) + (self._cardinality,)
        if self.table.shape != expected:
            raise ValueError(f"table must have shape {expected}, got {self.table.shape}.")
        if np.any(self.table < 0) or not np.allclose(self.table.sum(axis=-1), 1.0):
            raise ValueError("table rows must be non-negative and sum to 1.")

    @property
    def cardinality(self) -> int:
        return self._cardinality

    def parent_cardinalities(self) -> Dict[str, int]:
        return dict(self._parent_cardinalities)

    @classmethod
    def random(
        cls,
        cardinality: int,
        parent_cardinalities: Dict[str, int],
        alpha: float,
        rng: np.random.Generator,
    ) -> "DirichletCPTMechanism":
        n_configs = int(np.prod(list(parent_cardinalities.values()), dtype=np.int64))
        flat = rng.dirichlet(np.full(cardinality, float(alpha)), size=n_configs)
        table = flat.reshape(tuple(parent_cardinalities.values()) + (cardinality,))
        return cls(cardinality, parent_cardinalities, table)

    def cpt(self, parent_names: Optional[Sequence[str]] = None) -> np.ndarray:
        cards = self._parent_cardinalities
        names = list(cards) if parent_names is None else list(parent_names)
        if set(names) != set(cards):
            raise ValueError(f"parent_names {names} do not match mechanism parents {list(cards)}.")
        if names == list(cards):
            return self.table
        permutation = [list(cards).index(n) for n in names]
        return np.transpose(self.table, axes=permutation + [len(names)])

    def __call__(self, n_samples: int, **parents: np.ndarray) -> np.ndarray:
        noise = np.random.uniform(0.0, 1.0, size=n_samples)
        return self.evaluate(noise, **parents)

    def evaluate(self, noise: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        config_idx = mixed_radix_index(parents, self._parent_cardinalities)
        flat = self.table.reshape(-1, self._cardinality)
        cumprobs = np.cumsum(flat[config_idx], axis=1)
        noise = np.asarray(noise, dtype=np.float64)
        return (cumprobs <= noise[:, None]).sum(axis=1).astype(np.int64)

    def abduct(self, value: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Abduction is set-valued for tabular discrete mechanisms.")

    def __str__(self) -> str:
        return f"CPT(C={self._cardinality})"

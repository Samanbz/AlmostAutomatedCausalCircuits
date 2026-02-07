from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np
from scipy import stats

from src.utils import ContinuousInterval, DiscreteInterval, Interval, Support

from .nodes import LeafNode


class Distribution(LeafNode, ABC):
    """Base class for probability distributions used as leaves in SPNs."""

    def __init__(self, var: int, var_support: Interval):
        super().__init__(Support({var: var_support}))
        self.var = var
        self.var_support = var_support

    @abstractmethod
    def sample(self) -> Any:
        """Sample a value from the distribution."""
        pass

    @abstractmethod
    def split_at(self, cut_point: Any) -> tuple["Distribution", "Distribution"]:
        """Split the distribution at a cut point into two distributions."""
        pass

    @abstractmethod
    def constrain_to(self, interval: Optional[Interval]) -> "TruncatedDistribution":
        """Return a truncated distribution constrained to the given interval. If interval is None, return self."""
        pass


class GaussianDistribution(Distribution):
    """Represents a Gaussian distribution leaf node."""

    def __init__(self, var: int, mean: float, stddev: float):
        super().__init__(var, ContinuousInterval(float("-inf"), float("inf")))
        self.mean = mean
        self.stddev = stddev

    def sample(self) -> float:
        """Sample from the Gaussian distribution."""
        return np.random.normal(self.mean, self.stddev)

    def split_at(
        self, cut_point: float
    ) -> tuple["TruncatedGaussianDistribution", "TruncatedGaussianDistribution"]:
        """Split the Gaussian at a cut point, returning two truncated Gaussians."""
        left_support, right_support = self.var_support.split_at(cut_point)
        left_dist = TruncatedGaussianDistribution(self.var, self, left_support)
        right_dist = TruncatedGaussianDistribution(self.var, self, right_support)
        return left_dist, right_dist

    def constrain_to(self, interval: Optional[ContinuousInterval]) -> "TruncatedDistribution":
        """Return a truncated Gaussian distribution constrained to the given interval."""
        if interval is None:
            return self
        return TruncatedGaussianDistribution(self.var, self, interval)

    def __repr__(self):
        return f"GaussianDistribution(var={self.var}, mean={self.mean}, stddev={self.stddev})"


class CategoricalDistribution(Distribution):
    """Represents a Categorical distribution leaf node."""

    def __init__(self, var: int, categories: List[Any], probabilities: List[float]):
        super().__init__(var, DiscreteInterval(range(len(categories))))
        self.categories = categories
        self.probabilities = probabilities

    def sample(self) -> Any:
        """Sample from the categorical distribution according to probabilities."""
        return np.random.choice(self.categories, p=self.probabilities)

    def split_at(
        self, cut_point: Any
    ) -> tuple["TruncatedCategoricalDistribution", "TruncatedCategoricalDistribution"]:
        """Split the categorical distribution at a cut point."""
        left_support, right_support = self.var_support.split_at(cut_point)
        left_dist = TruncatedCategoricalDistribution(self.var, self, left_support)
        right_dist = TruncatedCategoricalDistribution(self.var, self, right_support)
        return left_dist, right_dist

    def constrain_to(self, interval: Optional[DiscreteInterval]) -> "TruncatedDistribution":
        """Return a truncated categorical distribution constrained to the given interval."""
        if interval is None:
            return self
        return TruncatedCategoricalDistribution(self.var, self, interval)

    def __repr__(self):
        return f"CategoricalDistribution(var={self.var}, categories={self.categories}, probabilities={self.probabilities})"


class UniformDistribution(Distribution):
    """Represents a Uniform distribution leaf node."""

    def __init__(self, var: int, low: float, high: float):
        super().__init__(var, ContinuousInterval(low, high))
        self.low = low
        self.high = high

    def sample(self) -> float:
        """Sample uniformly from the distribution."""
        return np.random.uniform(self.low, self.high)

    def split_at(
        self, cut_point: float
    ) -> tuple["TruncatedUniformDistribution", "TruncatedUniformDistribution"]:
        """Split the uniform distribution at a cut point."""
        left_support, right_support = self.var_support.split_at(cut_point)
        left_dist = TruncatedUniformDistribution(self.var, self, left_support)
        right_dist = TruncatedUniformDistribution(self.var, self, right_support)
        return left_dist, right_dist

    def constrain_to(self, interval: Optional[ContinuousInterval]) -> "TruncatedDistribution":
        """Return a truncated uniform distribution constrained to the given interval."""
        if interval is None:
            return self
        return TruncatedUniformDistribution(self.var, self, interval)

    def __repr__(self):
        return f"UniformDistribution(var={self.var}, low={self.low}, high={self.high})"


class TruncatedDistribution(Distribution):
    """
    Base class for distributions restricted to a specific interval.
    Uses rejection sampling by default.
    """

    def __init__(self, var: int, base_distribution: Distribution, var_support: Interval):
        super().__init__(var, var_support)
        self.base_distribution = base_distribution

    def sample(self, max_attempts: int = 1000) -> Any:
        """Sample from base distribution using rejection sampling."""
        for _ in range(max_attempts):
            value = self.base_distribution.sample()
            if self.var_support.contains(value):
                return value
        raise ValueError(
            f"Failed to sample from TruncatedDistribution after {max_attempts} attempts. "
            f"Support interval may be too restrictive."
        )

    def split_at(self, cut_point: Any) -> tuple["TruncatedDistribution", "TruncatedDistribution"]:
        """Further split the truncated distribution at a cut point."""
        left_support, right_support = self.var_support.split_at(cut_point)
        left_dist = self.__class__(self.var, self.base_distribution, left_support)
        right_dist = self.__class__(self.var, self.base_distribution, right_support)
        return left_dist, right_dist

    def constrain_to(self, interval: Optional[Interval]) -> "TruncatedDistribution":
        """Return a further truncated distribution constrained to the given interval."""
        if interval is None:
            return self
        new_support = self.var_support.intersect(interval)
        return self.__class__(self.var, self.base_distribution, new_support)

    def __repr__(self):
        return f"TruncatedDistribution(var={self.var}, base={self.base_distribution}, support={self.var_support})"


class TruncatedGaussianDistribution(TruncatedDistribution):
    """
    Gaussian distribution truncated to a specific interval.
    Uses scipy.stats.truncnorm for efficient sampling.
    """

    def __init__(
        self,
        var: int,
        base_distribution: GaussianDistribution,
        support: ContinuousInterval,
    ):
        super().__init__(var, base_distribution, support)
        if not isinstance(base_distribution, GaussianDistribution):
            raise TypeError("Base distribution must be GaussianDistribution")
        if not isinstance(support, ContinuousInterval):
            raise TypeError("Support must be ContinuousInterval")
        self.mean = base_distribution.mean
        self.stddev = base_distribution.stddev

    def sample(self) -> float:
        """Sample from truncated Gaussian using scipy.stats.truncnorm."""
        # Convert bounds to standardized form for truncnorm
        self.var_support: ContinuousInterval
        a = (self.var_support.low - self.mean) / self.stddev
        b = (self.var_support.high - self.mean) / self.stddev
        return stats.truncnorm.rvs(a, b, loc=self.mean, scale=self.stddev)

    def __repr__(self):
        return f"TruncatedGaussianDistribution(var={self.var}, mean={self.mean}, stddev={self.stddev}, support={self.var_support})"


class TruncatedUniformDistribution(TruncatedDistribution):
    """
    Uniform distribution truncated to a specific interval.
    This is just a uniform distribution over the truncated range.
    """

    def __init__(
        self,
        var: int,
        base_distribution: UniformDistribution,
        support: ContinuousInterval,
    ):
        super().__init__(var, base_distribution, support)
        if not isinstance(base_distribution, UniformDistribution):
            raise TypeError("Base distribution must be UniformDistribution")
        if not isinstance(support, ContinuousInterval):
            raise TypeError("Support must be ContinuousInterval")

    def sample(self) -> float:
        """Sample uniformly from the truncated interval."""
        self.var_support: ContinuousInterval
        return np.random.uniform(self.var_support.low, self.var_support.high)

    def __repr__(self):
        return f"TruncatedUniformDistribution(var={self.var}, support={self.var_support})"


class TruncatedCategoricalDistribution(TruncatedDistribution):
    """
    Categorical distribution truncated to a subset of categories.
    Renormalizes probabilities over the allowed categories.
    """

    def __init__(
        self,
        var: int,
        base_distribution: CategoricalDistribution,
        support: DiscreteInterval,
    ):
        super().__init__(var, base_distribution, support)
        if not isinstance(base_distribution, CategoricalDistribution):
            raise TypeError("Base distribution must be CategoricalDistribution")
        if not isinstance(support, DiscreteInterval):
            raise TypeError("Support must be DiscreteInterval")

        # Filter and renormalize probabilities for categories in support
        self.categories = []
        self.probabilities = []
        for cat, prob in zip(base_distribution.categories, base_distribution.probabilities):
            if self.var_support.contains(cat):
                self.categories.append(cat)
                self.probabilities.append(prob)

        # Renormalize probabilities
        total_prob = sum(self.probabilities)
        if total_prob > 0:
            self.probabilities = [p / total_prob for p in self.probabilities]
        else:
            raise ValueError("No valid categories in truncated support")

    def sample(self) -> Any:
        """Sample from renormalized categorical distribution."""
        return np.random.choice(self.categories, p=self.probabilities)

    def __repr__(self):
        return f"TruncatedCategoricalDistribution(var={self.var}, categories={self.categories}, probabilities={self.probabilities})"

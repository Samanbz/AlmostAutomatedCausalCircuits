from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
from scipy import stats

from src.utils import BitSet, ContinuousInterval, DiscreteInterval, Interval

from .arithmetic_circuit import LeafNode


class Distribution(LeafNode, ABC):
    """Base class for probability distributions used as leaves in SPNs."""

    support: Interval

    @abstractmethod
    def sample(self) -> Any:
        """Sample a value from the distribution."""
        pass

    @abstractmethod
    def split_at(self, cut_point: Any) -> tuple["Distribution", "Distribution"]:
        """Split the distribution at a cut point into two distributions."""
        pass

    @abstractmethod
    def constrain_to(self, interval: Interval) -> "TruncatedDistribution":
        """Return a truncated distribution constrained to the given interval."""
        pass


class GaussianDistribution(Distribution):
    """Represents a Gaussian distribution leaf node."""

    def __init__(self, scope: BitSet, mean: float, stddev: float):
        super().__init__(scope)
        self.mean = mean
        self.stddev = stddev
        self.support = ContinuousInterval.infinite()

    def sample(self) -> float:
        """Sample from the Gaussian distribution."""
        return np.random.normal(self.mean, self.stddev)

    def split_at(
        self, cut_point: float
    ) -> tuple["TruncatedGaussianDistribution", "TruncatedGaussianDistribution"]:
        """Split the Gaussian at a cut point, returning two truncated Gaussians."""
        left_support, right_support = self.support.split_at(cut_point)
        left_dist = TruncatedGaussianDistribution(self.scope, self, left_support)
        right_dist = TruncatedGaussianDistribution(self.scope, self, right_support)
        return left_dist, right_dist

    def constrain_to(self, interval: ContinuousInterval) -> "TruncatedDistribution":
        """Return a truncated Gaussian distribution constrained to the given interval."""
        return TruncatedGaussianDistribution(self.scope, self, interval)

    def __repr__(self):
        return f"GaussianDistribution(scope={self.scope}, mean={self.mean}, stddev={self.stddev})"


class CategoricalDistribution(Distribution):
    """Represents a Categorical distribution leaf node."""

    def __init__(self, scope: BitSet, categories: List[Any], probabilities: List[float]):
        super().__init__(scope)
        self.categories = categories
        self.probabilities = probabilities
        self.support = DiscreteInterval(categories)

    def sample(self) -> Any:
        """Sample from the categorical distribution according to probabilities."""
        return np.random.choice(self.categories, p=self.probabilities)

    def split_at(
        self, cut_point: Any
    ) -> tuple["TruncatedCategoricalDistribution", "TruncatedCategoricalDistribution"]:
        """Split the categorical distribution at a cut point."""
        left_support, right_support = self.support.split_at(cut_point)
        left_dist = TruncatedCategoricalDistribution(self.scope, self, left_support)
        right_dist = TruncatedCategoricalDistribution(self.scope, self, right_support)
        return left_dist, right_dist

    def constrain_to(self, interval: DiscreteInterval) -> "TruncatedDistribution":
        """Return a truncated categorical distribution constrained to the given interval."""
        return TruncatedCategoricalDistribution(self.scope, self, interval)

    def __repr__(self):
        return f"CategoricalDistribution(scope={self.scope}, categories={self.categories}, probabilities={self.probabilities})"


class UniformDistribution(Distribution):
    """Represents a Uniform distribution leaf node."""

    def __init__(self, scope: BitSet, low: float, high: float):
        super().__init__(scope)
        self.low = low
        self.high = high
        self.support = ContinuousInterval(low, high, include_low=True, include_high=False)

    def sample(self) -> float:
        """Sample uniformly from the distribution."""
        return np.random.uniform(self.low, self.high)

    def split_at(
        self, cut_point: float
    ) -> tuple["TruncatedUniformDistribution", "TruncatedUniformDistribution"]:
        """Split the uniform distribution at a cut point."""
        left_support, right_support = self.support.split_at(cut_point)
        left_dist = TruncatedUniformDistribution(self.scope, self, left_support)
        right_dist = TruncatedUniformDistribution(self.scope, self, right_support)
        return left_dist, right_dist

    def constrain_to(self, interval: ContinuousInterval) -> "TruncatedDistribution":
        """Return a truncated uniform distribution constrained to the given interval."""
        return TruncatedUniformDistribution(self.scope, self, interval)

    def __repr__(self):
        return f"UniformDistribution(scope={self.scope}, low={self.low}, high={self.high})"


class TruncatedDistribution(Distribution):
    """
    Base class for distributions restricted to a specific interval.
    Uses rejection sampling by default.
    """

    def __init__(self, scope: BitSet, base_distribution: Distribution, support: Interval):
        super().__init__(scope)
        self.base_distribution = base_distribution
        self.support = support

    def sample(self, max_attempts: int = 1000) -> Any:
        """Sample from base distribution using rejection sampling."""
        for _ in range(max_attempts):
            value = self.base_distribution.sample()
            if self.support.contains(value):
                return value
        raise ValueError(
            f"Failed to sample from TruncatedDistribution after {max_attempts} attempts. "
            f"Support interval may be too restrictive."
        )

    def split_at(self, cut_point: Any) -> tuple["TruncatedDistribution", "TruncatedDistribution"]:
        """Further split the truncated distribution at a cut point."""
        left_support, right_support = self.support.split_at(cut_point)
        left_dist = self.__class__(self.scope, self.base_distribution, left_support)
        right_dist = self.__class__(self.scope, self.base_distribution, right_support)
        return left_dist, right_dist

    def constrain_to(self, interval: Interval) -> "TruncatedDistribution":
        """Return a further truncated distribution constrained to the given interval."""
        new_support = self.support.intersect(interval)
        return self.__class__(self.scope, self.base_distribution, new_support)

    def __repr__(self):
        return f"TruncatedDistribution(scope={self.scope}, base={self.base_distribution}, support={self.support})"


class TruncatedGaussianDistribution(TruncatedDistribution):
    """
    Gaussian distribution truncated to a specific interval.
    Uses scipy.stats.truncnorm for efficient sampling.
    """

    def __init__(
        self, scope: BitSet, base_distribution: GaussianDistribution, support: ContinuousInterval
    ):
        super().__init__(scope, base_distribution, support)
        if not isinstance(base_distribution, GaussianDistribution):
            raise TypeError("Base distribution must be GaussianDistribution")
        if not isinstance(support, ContinuousInterval):
            raise TypeError("Support must be ContinuousInterval")
        self.mean = base_distribution.mean
        self.stddev = base_distribution.stddev

    def sample(self) -> float:
        """Sample from truncated Gaussian using scipy.stats.truncnorm."""
        # Convert bounds to standardized form for truncnorm
        self.support: ContinuousInterval
        a = (self.support.low - self.mean) / self.stddev
        b = (self.support.high - self.mean) / self.stddev
        return stats.truncnorm.rvs(a, b, loc=self.mean, scale=self.stddev)

    def __repr__(self):
        return f"TruncatedGaussianDistribution(scope={self.scope}, mean={self.mean}, stddev={self.stddev}, support={self.support})"


class TruncatedUniformDistribution(TruncatedDistribution):
    """
    Uniform distribution truncated to a specific interval.
    This is just a uniform distribution over the truncated range.
    """

    def __init__(
        self, scope: BitSet, base_distribution: UniformDistribution, support: ContinuousInterval
    ):
        super().__init__(scope, base_distribution, support)
        if not isinstance(base_distribution, UniformDistribution):
            raise TypeError("Base distribution must be UniformDistribution")
        if not isinstance(support, ContinuousInterval):
            raise TypeError("Support must be ContinuousInterval")

    def sample(self) -> float:
        """Sample uniformly from the truncated interval."""
        self.support: ContinuousInterval
        return np.random.uniform(self.support.low, self.support.high)

    def __repr__(self):
        return f"TruncatedUniformDistribution(scope={self.scope}, support={self.support})"


class TruncatedCategoricalDistribution(TruncatedDistribution):
    """
    Categorical distribution truncated to a subset of categories.
    Renormalizes probabilities over the allowed categories.
    """

    def __init__(
        self, scope: BitSet, base_distribution: CategoricalDistribution, support: DiscreteInterval
    ):
        super().__init__(scope, base_distribution, support)
        if not isinstance(base_distribution, CategoricalDistribution):
            raise TypeError("Base distribution must be CategoricalDistribution")
        if not isinstance(support, DiscreteInterval):
            raise TypeError("Support must be DiscreteInterval")

        # Filter and renormalize probabilities for categories in support
        self.categories = []
        self.probabilities = []
        for cat, prob in zip(base_distribution.categories, base_distribution.probabilities):
            if support.contains(cat):
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
        return f"TruncatedCategoricalDistribution(scope={self.scope}, categories={self.categories}, probabilities={self.probabilities})"

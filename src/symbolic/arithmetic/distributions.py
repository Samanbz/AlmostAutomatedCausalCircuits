from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np
from scipy import stats

from src.utils import ContinuousInterval, DiscreteInterval, Interval, Support

from .nodes import LeafNode


class Distribution(LeafNode, ABC):
    """Base class for probability distributions used as leaves in SPNs."""

    def __init__(self, var: int, var_support: Interval, unit_count: int = 1):
        super().__init__(Support({var: var_support}), unit_count=unit_count)
        self.var = var
        self.var_support = var_support

    @abstractmethod
    def sample(self) -> Any:
        """Sample a value from the distribution."""
        pass

    @property
    @abstractmethod
    def _truncated_class(self) -> type["TruncatedDistribution"]:
        """Return the class used for truncated versions of this distribution."""
        pass

    def split_at(self, cut_point: Any) -> tuple["TruncatedDistribution", "TruncatedDistribution"]:
        """Split the distribution at a cut point into two distributions."""
        left_support, right_support = self.var_support.split_at(cut_point)
        left_dist = self._truncated_class(
            self.var, self, left_support, unit_count=self.unit_count
        )
        right_dist = self._truncated_class(
            self.var, self, right_support, unit_count=self.unit_count
        )
        return left_dist, right_dist

    def split(self, n: int) -> List["TruncatedDistribution"]:
        """
        Splits the distribution into n disjoint truncated distributions.
        """
        if n <= 1:
            return [self]

        # If support is infinite, we cannot split uniformly.
        # We check if this class or its base distribution (if truncated)
        # provides a specialized split.
        if np.isinf(self.var_support.low) or np.isinf(self.var_support.high):
            # Specialized split for Gaussian and its truncated versions
            if isinstance(self, (GaussianDistribution, TruncatedGaussianDistribution)):
                # We'll override split in TruncatedGaussianDistribution too
                pass
            else:
                raise ValueError(
                    f"Cannot split infinite interval uniformly for {self.__class__.__name__}. "
                    "Use specialized distribution splitting."
                )

        sub_intervals = self.var_support.split(n)
        return [self.constrain_to(interval) for interval in sub_intervals]

    def constrain_to(self, interval: Optional[Interval]) -> "TruncatedDistribution":
        """Return a truncated distribution constrained to the given interval."""
        if interval is None:
            return self
        return self._truncated_class(self.var, self, interval, unit_count=self.unit_count)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.var == other.var and self.var_support == other.var_support

    def __hash__(self):
        return hash((self.__class__, self.var, self.var_support))


class GaussianDistribution(Distribution):
    """Represents a Gaussian distribution leaf node."""

    def __init__(self, var: int, mean: float, stddev: float, unit_count: int = 1):
        super().__init__(
            var,
            ContinuousInterval(float("-inf"), float("inf"), include_low=False, include_high=False),
            unit_count=unit_count,
        )
        self.mean = mean
        self.stddev = stddev

    def sample(self) -> float:
        """Sample from the Gaussian distribution."""
        return np.random.normal(self.mean, self.stddev)

    @property
    def _truncated_class(self) -> type["TruncatedGaussianDistribution"]:
        return TruncatedGaussianDistribution

    def split(self, n: int) -> List["TruncatedGaussianDistribution"]:
        """
        Splits the Gaussian distribution into n disjoint truncated Gaussian distributions using quantiles.
        """
        if n <= 1:
            return [self]

        # Use quantiles for better Gaussian partitioning
        probs = np.linspace(0, 1, n + 1)
        cut_points = stats.norm.ppf(probs, loc=self.mean, scale=self.stddev)

        intervals = []
        for i in range(n):
            low, high = cut_points[i], cut_points[i + 1]
            # Ensure -inf/inf remain exclusive
            include_low = i != 0
            include_high = (i == n - 1) and self.var_support.include_high
            intervals.append(ContinuousInterval(low, high, include_low, include_high))

        return [self.constrain_to(interval) for interval in intervals]

    def __repr__(self):
        return f"GaussianDistribution(var={self.var}, mean={self.mean}, stddev={self.stddev})"

    def __eq__(self, other):
        return (
            super().__eq__(other) and self.mean == other.mean and self.stddev == other.stddev
        )

    def __hash__(self):
        return hash((super().__hash__(), self.mean, self.stddev))


class CategoricalDistribution(Distribution):
    """Represents a Categorical distribution leaf node."""

    def __init__(
        self, var: int, categories: List[Any], probabilities: List[float], unit_count: int = 1
    ):
        super().__init__(var, DiscreteInterval(range(len(categories))), unit_count=unit_count)
        self.categories = categories
        self.probabilities = probabilities

    def sample(self) -> Any:
        """Sample from the categorical distribution according to probabilities."""
        return np.random.choice(self.categories, p=self.probabilities)

    @property
    def _truncated_class(self) -> type["TruncatedCategoricalDistribution"]:
        return TruncatedCategoricalDistribution

    def __repr__(self):
        return f"CategoricalDistribution(var={self.var}, categories={self.categories}, probabilities={self.probabilities})"

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.categories == other.categories
            and self.probabilities == other.probabilities
        )

    def __hash__(self):
        return hash((super().__hash__(), tuple(self.categories), tuple(self.probabilities)))


class UniformDistribution(Distribution):
    """Represents a Uniform distribution leaf node."""

    def __init__(self, var: int, low: float, high: float, unit_count: int = 1):
        super().__init__(var, ContinuousInterval(low, high), unit_count=unit_count)
        self.low = low
        self.high = high

    def sample(self) -> float:
        """Sample uniformly from the distribution."""
        return np.random.uniform(self.low, self.high)

    @property
    def _truncated_class(self) -> type["TruncatedUniformDistribution"]:
        return TruncatedUniformDistribution

    def split_at(
        self, cut_point: float
    ) -> tuple["TruncatedUniformDistribution", "TruncatedUniformDistribution"]:
        """Split the uniform distribution at a cut point."""
        left_support, right_support = self.var_support.split_at(cut_point)
        left_dist = TruncatedUniformDistribution(
            self.var, self, left_support, unit_count=self.unit_count
        )
        right_dist = TruncatedUniformDistribution(
            self.var, self, right_support, unit_count=self.unit_count
        )
        return left_dist, right_dist

    def constrain_to(self, interval: Optional[ContinuousInterval]) -> "TruncatedDistribution":
        """Return a truncated uniform distribution constrained to the given interval."""
        if interval is None:
            return self
        return TruncatedUniformDistribution(self.var, self, interval, unit_count=self.unit_count)

    def __repr__(self):
        return f"UniformDistribution(var={self.var}, low={self.low}, high={self.high})"

    def __eq__(self, other):
        return super().__eq__(other) and self.low == other.low and self.high == other.high

    def __hash__(self):
        return hash((super().__hash__(), self.low, self.high))


class TruncatedDistribution(Distribution):
    """
    Base class for distributions restricted to a specific interval.
    Uses rejection sampling by default.
    """

    def __init__(
        self,
        var: int,
        base_distribution: Distribution,
        var_support: Interval,
        unit_count: int = 1,
    ):
        super().__init__(var, var_support, unit_count=unit_count)
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

    @property
    def _truncated_class(self) -> type["TruncatedDistribution"]:
        """Return the class used for further truncation."""
        return self.__class__

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

    def __eq__(self, other):
        return super().__eq__(other) and self.base_distribution == other.base_distribution

    def __hash__(self):
        return hash((super().__hash__(), self.base_distribution))


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
        unit_count: int = 1,
    ):
        super().__init__(var, base_distribution, support, unit_count=unit_count)
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

    def split(self, n: int) -> List["TruncatedGaussianDistribution"]:
        """
        Splits the truncated Gaussian into n disjoint truncated Gaussian distributions using quantiles.
        """
        if n <= 1:
            return [self]

        # Use quantiles within the truncated range
        self.var_support: ContinuousInterval
        
        # Get CDF at bounds
        p_low = stats.norm.cdf(self.var_support.low, loc=self.mean, scale=self.stddev)
        p_high = stats.norm.cdf(self.var_support.high, loc=self.mean, scale=self.stddev)
        
        probs = np.linspace(p_low, p_high, n + 1)
        cut_points = stats.norm.ppf(probs, loc=self.mean, scale=self.stddev)

        intervals = []
        for i in range(n):
            low, high = cut_points[i], cut_points[i + 1]
            include_low = (i != 0) or self.var_support.include_low
            include_high = (i == n - 1) and self.var_support.include_high
            intervals.append(ContinuousInterval(low, high, include_low, include_high))

        return [self.constrain_to(interval) for interval in intervals]

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
        unit_count: int = 1,
    ):
        super().__init__(var, base_distribution, support, unit_count=unit_count)
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
        unit_count: int = 1,
    ):
        super().__init__(var, base_distribution, support, unit_count=unit_count)
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

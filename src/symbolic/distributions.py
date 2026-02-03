from typing import Any, List

from src.utils import BitSet, Interval

from .arithmetic_circuit import LeafNode


class Distribution(LeafNode):
    """Base class for probability distributions used as leaves in SPNs."""

    pass


class GaussianDistribution(Distribution):
    """Represents a Gaussian distribution leaf node."""

    def __init__(self, scope: BitSet, mean: float, stddev: float):
        super().__init__(scope)
        self.mean = mean
        self.stddev = stddev

    def __repr__(self):
        return f"GaussianDistribution(scope={self.scope}, mean={self.mean}, stddev={self.stddev})"


class CategoricalDistribution(Distribution):
    """Represents a Categorical distribution leaf node."""

    def __init__(self, scope: BitSet, categories: List[Any], probabilities: List[float]):
        super().__init__(scope)
        self.categories = categories
        self.probabilities = probabilities

    def __repr__(self):
        return f"CategoricalDistribution(scope={self.scope}, categories={self.categories}, probabilities={self.probabilities})"


class UniformDistribution(Distribution):
    """Represents a Uniform distribution leaf node."""

    def __init__(self, scope: BitSet, low: float, high: float):
        super().__init__(scope)
        self.low = low
        self.high = high

    def __repr__(self):
        return f"UniformDistribution(scope={self.scope}, low={self.low}, high={self.high})"


class TruncatedDistribution(Distribution):
    """
    Represents a distribution restricted to a specific interval.
    """

    def __init__(
        self,
        scope: BitSet,
        base_distribution: Distribution,
        interval: Interval,
    ):
        super().__init__(scope)
        self.base_distribution = base_distribution
        self.interval = interval

    def __repr__(self):
        return f"TruncatedDistribution(scope={self.scope}, base={self.base_distribution}, interval={self.interval})"

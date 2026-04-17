from abc import ABC, abstractmethod
from typing import Any, List, Set, Tuple, Union

import numpy as np


class Interval(ABC):
    """Abstract base class for intervals."""

    @abstractmethod
    def contains(self, value: Union[float, int, np.ndarray]) -> Union[bool, np.ndarray]:
        """Checks if the value is contained in the interval."""
        pass

    @abstractmethod
    def intersect(self, other: "Interval") -> "Interval":
        """Returns the intersection of this interval with another."""
        pass

    @abstractmethod
    def union(self, other: "Interval") -> "Interval":
        """
        Returns the union of this interval with another.
        Assumes continuity for continuous intervals and set union for discrete intervals.
        """
        pass

    @abstractmethod
    def split_at(self, cut_point: Any) -> Tuple["Interval", "Interval"]:
        """
        Splits the interval at the given cut point into two intervals.
        """
        pass

    @abstractmethod
    def split(self, n: int) -> List["Interval"]:
        """
        Splits the interval into n disjoint sub-intervals.
        """
        pass

    @property
    @abstractmethod
    def is_empty(self) -> bool:
        """Returns True if the interval is empty."""
        pass


class ContinuousInterval(Interval):
    """Represents a mathematical interval [low, high), (low, high], etc. for continuous data."""

    def __init__(
        self,
        low: float,
        high: float,
        include_low: bool = True,
        include_high: bool = False,
    ):
        self.low = low
        self.high = high
        self.include_low = include_low
        self.include_high = include_high

    @classmethod
    def infinite(cls) -> "ContinuousInterval":
        return cls(float("-inf"), float("inf"), False, False)

    def contains(self, value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        lower_check = (value >= self.low) if self.include_low else (value > self.low)
        upper_check = (value <= self.high) if self.include_high else (value < self.high)
        if hasattr(value, "__len__") and not isinstance(value, str):  # Handle array-like
            return lower_check & upper_check
        return lower_check and upper_check

    def intersect(self, other: "Interval") -> "ContinuousInterval":
        if not isinstance(other, ContinuousInterval):
            raise TypeError("Can only intersect ContinuousInterval with ContinuousInterval.")

        new_low = max(self.low, other.low)
        new_high = min(self.high, other.high)

        # Determine strictness of bounds
        if self.low == other.low:
            new_include_low = self.include_low and other.include_low
        else:
            new_include_low = self.include_low if self.low > other.low else other.include_low

        if self.high == other.high:
            new_include_high = self.include_high and other.include_high
        else:
            new_include_high = self.include_high if self.high < other.high else other.include_high

        return ContinuousInterval(new_low, new_high, new_include_low, new_include_high)

    def union(self, other: "Interval") -> "ContinuousInterval":
        if not isinstance(other, ContinuousInterval):
            raise TypeError("Can only union ContinuousInterval with ContinuousInterval.")

        new_low = min(self.low, other.low)
        new_high = max(self.high, other.high)

        # Determine strictness of bounds
        if self.low == other.low:
            new_include_low = self.include_low or other.include_low
        else:
            new_include_low = self.include_low if self.low < other.low else other.include_low

        if self.high == other.high:
            new_include_high = self.include_high or other.include_high
        else:
            new_include_high = self.include_high if self.high > other.high else other.include_high

        return ContinuousInterval(new_low, new_high, new_include_low, new_include_high)

    def split_at(self, cut_point: float) -> Tuple["ContinuousInterval", "ContinuousInterval"]:
        """
        Splits the interval at the given cut point into two intervals.
        Cutpoint is exclusive to the right interval.
        """
        is_contained = self.contains(cut_point)
        if isinstance(is_contained, np.ndarray):
            if not is_contained.all():
                raise ValueError("Cut point must be within the interval.")
        elif not is_contained:
            raise ValueError("Cut point must be within the interval.")

        left = ContinuousInterval(self.low, cut_point, self.include_low, True)
        right = ContinuousInterval(cut_point, self.high, False, self.include_high)
        return left, right

    def split(self, n: int) -> List["ContinuousInterval"]:
        """
        Splits the interval into n disjoint sub-intervals.
        """
        if n <= 1:
            return [self]

        if np.isinf(self.low) or np.isinf(self.high):
            raise ValueError(
                "Cannot split infinite interval uniformly. Use specialized distribution splitting."
            )

        step = (self.high - self.low) / n
        intervals = []
        for i in range(n):
            low = self.low + i * step
            high = self.low + (i + 1) * step
            include_low = self.include_low if i == 0 else False
            include_high = self.include_high if i == n - 1 else True
            intervals.append(ContinuousInterval(low, high, include_low, include_high))
        return intervals

    @property
    def is_empty(self) -> bool:
        if self.low > self.high:
            return True
        if self.low == self.high:
            return not (self.include_low and self.include_high)
        return False

    def __repr__(self):
        left = "[" if self.include_low else "("
        right = "]" if self.include_high else ")"
        return f"{left}{self.low:.2f}, {self.high:.2f}{right}"

    def __eq__(self, other):
        if not isinstance(other, ContinuousInterval):
            return False
        return (
            self.low == other.low
            and self.high == other.high
            and self.include_low == other.include_low
            and self.include_high == other.include_high
        )

    def __hash__(self):
        return hash((self.low, self.high, self.include_low, self.include_high))


class DiscreteInterval(Interval):
    """Represents a set of values for discrete/ordinal data."""

    def __init__(self, values: Union[List[int], Set[int], np.ndarray]):
        self.values = np.sort(np.unique(values))

    def contains(self, value: Union[int, np.ndarray]) -> Union[bool, np.ndarray]:
        if isinstance(value, (np.ndarray, list)):
            return np.isin(value, self.values)
        return np.isin(value, self.values).item() if np.isscalar(value) else value in self.values

    def intersect(self, other: "Interval") -> "DiscreteInterval":
        if not isinstance(other, DiscreteInterval):
            raise TypeError("Can only intersect DiscreteInterval with DiscreteInterval.")

        common_values = np.intersect1d(self.values, other.values)
        return DiscreteInterval(common_values)

    def union(self, other: "Interval") -> "DiscreteInterval":
        if not isinstance(other, DiscreteInterval):
            raise TypeError("Can only union DiscreteInterval with DiscreteInterval.")

        all_values = np.union1d(self.values, other.values)
        return DiscreteInterval(all_values)

    @property
    def is_empty(self) -> bool:
        return len(self.values) == 0

    def split_at(self, cut_point: int) -> Tuple["DiscreteInterval", "DiscreteInterval"]:
        left_vals = self.values[self.values <= cut_point]
        right_vals = self.values[self.values > cut_point]
        return DiscreteInterval(left_vals), DiscreteInterval(right_vals)

    def split(self, n: int) -> List["DiscreteInterval"]:
        """
        Splits the interval into n disjoint sub-intervals.
        """
        if n <= 1 or len(self.values) == 0:
            return [self]

        split_vals = np.array_split(self.values, n)
        return [DiscreteInterval(vals) for vals in split_vals if len(vals) > 0]

    def __repr__(self):
        return f"DiscreteInterval({list(self.values)})"

    def __eq__(self, other):
        if not isinstance(other, DiscreteInterval):
            return False
        return np.array_equal(self.values, other.values)

    def __hash__(self):
        return hash(tuple(self.values))

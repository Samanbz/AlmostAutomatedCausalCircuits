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
    def split_at(self, cut_point: Any) -> Tuple["Interval", "Interval"]:
        """
        Splits the interval at the given cut point into two intervals.
        """
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

    def __repr__(self):
        left = "[" if self.include_low else "("
        right = "]" if self.include_high else ")"
        return f"{left}{self.low:.2f}, {self.high:.2f}{right}"


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

    def split_at(self, cut_point: int) -> Tuple["DiscreteInterval", "DiscreteInterval"]:
        left_vals = self.values[self.values <= cut_point]
        right_vals = self.values[self.values > cut_point]
        return DiscreteInterval(left_vals), DiscreteInterval(right_vals)

    def __repr__(self):
        return f"DiscreteInterval({list(self.values)})"

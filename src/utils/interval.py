import numpy as np


class Interval:
    """Represents a mathematical interval [low, high), (low, high], etc."""

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

    def contains(self, value: float | np.ndarray) -> bool | np.ndarray:
        lower_check = (value >= self.low) if self.include_low else (value > self.low)
        upper_check = (value <= self.high) if self.include_high else (value < self.high)
        return lower_check & upper_check

    def intersect(self, other: "Interval") -> "Interval":
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

        return Interval(new_low, new_high, new_include_low, new_include_high)

    def __repr__(self):
        left = "[" if self.include_low else "("
        right = "]" if self.include_high else ")"
        return f"{left}{self.low}, {self.high}{right}"

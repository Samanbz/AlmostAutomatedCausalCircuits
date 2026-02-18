from typing import Dict

from .bitset import BitSet
from .interval import Interval


class Support:
    def __init__(self, intervals: Dict[int, Interval] = None):
        self.intervals = intervals if intervals is not None else {}

    def add(self, var: int, interval: Interval):
        self.intervals[var] = interval

    def get(self, var: int) -> Interval:
        return self.intervals.get(var, None)

    @property
    def scope(self) -> BitSet:
        return BitSet(self.intervals.keys())

    def intersect(self, other: "Support") -> "Support":
        new_intervals = {}
        for var, interval in self.intervals.items():
            if var in other.intervals:
                new_intervals[var] = interval.intersect(other.intervals[var])
            else:
                new_intervals[var] = interval
        for var, interval in other.intervals.items():
            if var not in self.intervals:
                new_intervals[var] = interval
        return Support(new_intervals)

    def union(self, other: "Support") -> "Support":
        new_intervals = {}
        for var, interval in self.intervals.items():
            if var in other.intervals:
                new_intervals[var] = interval.union(other.intervals[var])
            else:
                new_intervals[var] = interval
        for var, interval in other.intervals.items():
            if var not in self.intervals:
                new_intervals[var] = interval
        return Support(new_intervals)

    def filter_by_vars(self, vars: BitSet) -> "Support":
        """
        Returns a new Support object containing only the intervals for the variables in the given BitSet.

        :param vars: A BitSet representing the variables to keep in the support.
        Variables not in this set will be removed from the resulting Support.
        :type vars: BitSet
        :return: A new Support object with intervals only for the specified variables.
        :rtype: Support
        """
        new_intervals = {var: interval for var, interval in self.intervals.items() if var in vars}
        return Support(new_intervals)

    @property
    def is_empty(self) -> bool:
        if not self.intervals:
            return False  # Empty dict implies full support (no constraints)
        return any(interval.is_empty for interval in self.intervals.values())

    def __iter__(self):
        return iter(self.intervals.items())

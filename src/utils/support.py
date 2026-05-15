from typing import Dict

from .bitset import BitSet
from .interval import Interval


class Support:
    __slots__ = ("intervals",)

    def __init__(self, intervals: Dict[int, Interval] = None):
        self.intervals = intervals if intervals is not None else {}

    def add(self, var: int, interval: Interval):
        self.intervals[var] = interval

    def get(self, var: int) -> Interval:
        return self.intervals.get(var, None)

    @property
    def scope(self) -> BitSet:
        return BitSet(self.intervals.keys())

    def __eq__(self, other):
        if not isinstance(other, Support):
            return False
        if len(self.intervals) != len(other.intervals):
            return False
        for var, interval in self.intervals.items():
            if var not in other.intervals or interval != other.intervals[var]:
                return False
        return True

    def intersect(self, other: "Support") -> "Support":
        if not self.intervals:
            return self
        if not other.intervals:
            return other

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
        if not self.intervals:
            return other
        if not other.intervals:
            return self

        new_intervals = self.intervals.copy()
        for var, other_interval in other.intervals.items():
            if var in new_intervals:
                self_interval = new_intervals[var]
                if self_interval is not other_interval:
                    new_intervals[var] = self_interval.union(other_interval)
            else:
                new_intervals[var] = other_interval
        return Support(new_intervals)

    @staticmethod
    def fast_disjoint_union(s1: "Support", s2: "Support") -> "Support":
        """Fast union for supports with disjoint scopes."""
        return Support({**s1.intervals, **s2.intervals})

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
            return True
        return any(interval.is_empty for interval in self.intervals.values())

    def __iter__(self):
        return iter(self.intervals.items())

    def __getitem__(self, key: int) -> Interval:
        return self.intervals[key]

    def __contains__(self, key: int) -> bool:
        return key in self.intervals

    def __repr__(self):
        newline = "\n"
        tab = "\t"
        return f"Support({newline}{newline.join(f'{tab}{k}: {v}' for k, v in self.intervals.items())}{newline})"

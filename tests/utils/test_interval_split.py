import numpy as np
import pytest

from src.utils import ContinuousInterval, DiscreteInterval


def test_continuous_interval_split():
    # Test finite interval split
    ci = ContinuousInterval(0, 10, include_low=True, include_high=False)
    n = 5
    splits = ci.split(n)

    assert len(splits) == n
    assert splits[0].low == 0
    assert splits[0].high == 2
    assert splits[0].include_low is True
    assert splits[0].include_high is True

    assert splits[1].low == 2
    assert splits[1].high == 4
    assert splits[1].include_low is False
    assert splits[1].include_high is True

    assert splits[-1].low == 8
    assert splits[-1].high == 10
    assert splits[-1].include_low is False
    assert splits[-1].include_high is False


def test_continuous_interval_infinite_split():
    ci = ContinuousInterval(float("-inf"), float("inf"))
    with pytest.raises(ValueError, match="Cannot split infinite interval"):
        ci.split(2)


def test_discrete_interval_split():
    di = DiscreteInterval([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    n = 3
    splits = di.split(n)

    assert len(splits) == 3
    # np.array_split partitions as [1,2,3,4], [5,6,7], [8,9,10] or similar
    all_vals = []
    for s in splits:
        all_vals.extend(s.values)
    assert len(all_vals) == 10
    assert np.all(np.sort(all_vals) == di.values)

    # Check disjointness
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            intersection = splits[i].intersect(splits[j])
            assert intersection.is_empty

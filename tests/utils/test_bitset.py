import numpy as np
import pytest

from src.utils.bitset import BitSet


class TestBitSet:
    def test_universal(self):
        u = BitSet.universal()

        # Membership
        assert 0 in u
        assert 100 in u
        assert 100000 in u

        # Length and iteration should fail
        with pytest.raises(OverflowError, match="(?i)infinite set"):
            len(u)

        # iter(u) just returns a generator, doesn't execute code yet.
        # We need to try to consume it to trigger the error.
        with pytest.raises(OverflowError, match="(?i)infinite set"):
            next(iter(u))

        with pytest.raises(OverflowError, match="(?i)infinite set"):
            list(u)

        # Operations
        bs = BitSet([1, 2, 3])

        # Union with universal is universal
        assert u.union(bs) == u
        assert bs.union(u) == u

        # Intersection with universal is the set itself
        assert u.intersection(bs) == bs
        assert bs.intersection(u) == bs

        # Difference
        # bs - u = empty
        assert bs.difference(u).is_empty

        # u - bs = infinite set (all ints except 1, 2, 3)
        diff = u.difference(bs)
        assert 0 in diff
        assert 1 not in diff
        assert 2 not in diff
        assert 3 not in diff
        assert 4 in diff
        with pytest.raises(OverflowError):
            len(diff)

    def test_init_empty(self):
        bs = BitSet()
        assert len(bs) == 0
        assert list(bs) == []

    def test_init_iterable(self):
        bs = BitSet([1, 3, 5])
        assert len(bs) == 3
        assert 1 in bs
        assert 3 in bs
        assert 5 in bs
        assert 2 not in bs

    def test_init_bitset(self):
        bs1 = BitSet([1, 2])
        bs2 = BitSet(bs1)
        assert bs1 == bs2
        assert bs1 is not bs2  # Should be a new object (based on implementation logic)

    def test_from_int(self):
        # 1 + 4 = 5 (binary 101) -> indices 0 and 2
        bs = BitSet.from_int(5)
        assert 0 in bs
        assert 2 in bs
        assert 1 not in bs
        assert len(bs) == 2

    def test_full(self):
        bs = BitSet.full(3)  # 111 binary = 7
        assert len(bs) == 3
        assert 0 in bs
        assert 1 in bs
        assert 2 in bs
        assert 3 not in bs

    def test_add_remove_discard(self):
        bs = BitSet()
        bs.add(1)
        assert 1 in bs

        bs.discard(1)
        assert 1 not in bs

        bs.discard(1)  # Should not raise

        bs.add(2)
        bs.remove(2)
        assert 2 not in bs

        with pytest.raises(KeyError):
            bs.remove(2)

    def test_set_operations(self):
        bs1 = BitSet([1, 2])
        bs2 = BitSet([2, 3])

        # Union
        assert bs1.union(bs2) == BitSet([1, 2, 3])
        assert (bs1 | bs2) == BitSet([1, 2, 3])

        # Intersection
        assert bs1.intersection(bs2) == BitSet([2])
        assert (bs1 & bs2) == BitSet([2])

        # Difference
        assert bs1.difference(bs2) == BitSet([1])
        assert (bs1 - bs2) == BitSet([1])

        # Symmetric Difference
        assert bs1.symmetric_difference(bs2) == BitSet([1, 3])
        assert (bs1 ^ bs2) == BitSet([1, 3])

    def test_subset_superset(self):
        bs1 = BitSet([1, 2])
        bs2 = BitSet([1, 2, 3])

        assert bs1.issubset(bs2)
        assert not bs2.issubset(bs1)

        assert bs2.issuperset(bs1)
        assert not bs1.issuperset(bs2)

    def test_from_bool_mask(self):
        mask = np.array([True, False, True], dtype=bool)  # indices 0, 2
        bs = BitSet.from_bool_mask(mask)
        assert 0 in bs
        assert 2 in bs
        assert 1 not in bs
        assert len(bs) == 2

    def test_to_bool_mask(self):
        bs = BitSet([0, 2])
        mask = bs.to_bool_mask(3)
        expected = np.array([True, False, True], dtype=bool)
        np.testing.assert_array_equal(mask, expected)

        # Test size larger than max element
        mask_large = bs.to_bool_mask(5)
        expected_large = np.array([True, False, True, False, False], dtype=bool)
        np.testing.assert_array_equal(mask_large, expected_large)

    def test_iter(self):
        elements = [1, 5, 10]
        bs = BitSet(elements)
        assert sorted(bs) == sorted(elements)

    def test_eq(self):
        bs1 = BitSet([1, 2])
        bs2 = BitSet([2, 1])
        assert bs1 == bs2
        assert bs1 != BitSet([1])
        assert bs1 != "not a bitset"

    def test_copy(self):
        bs = BitSet([1, 2])
        copied = bs.copy()
        assert bs == copied
        assert bs is not copied

    def test_hash(self):
        bs1 = BitSet([1, 2])
        bs2 = BitSet([1, 2])
        bs3 = BitSet([1])

        assert hash(bs1) == hash(bs2)
        assert hash(bs1) != hash(bs3)

        # Note: BitSet is mutable, so hash changes if modified
        h_before = hash(bs3)
        bs3.add(2)
        h_after = hash(bs3)

        assert h_after != h_before
        assert h_after == hash(bs1)

    def test_is_empty(self):
        bs = BitSet()
        assert bs.is_empty

        bs.add(1)
        assert not bs.is_empty

        bs.remove(1)
        assert bs.is_empty

    def test_min_max(self):
        bs = BitSet([1, 5, 10])
        assert bs.min == 1
        assert bs.max == 10

        bs = BitSet([3])
        assert bs.min == 3
        assert bs.max == 3

        empty_bs = BitSet()
        with pytest.raises(ValueError, match="BitSet is empty"):
            empty_bs.min()
        with pytest.raises(ValueError, match="BitSet is empty"):
            empty_bs.max()

    def test_repr(self):
        bs = BitSet([1, 2])
        # The list order in repr depends on iteration order which depends on bit significance
        # BitSet iteration yields indices from LSB to MSB (0...N)
        assert repr(bs) == "BitSet([1, 2])"

        bs = BitSet()
        assert repr(bs) == "BitSet([])"

from typing import Iterable

import numpy as np


class BitSet:
    __slots__ = ("_val",)  # type: int

    def __init__(self, elements: Iterable[int] = ()):
        self._val = 0
        if isinstance(elements, BitSet):
            self._val = elements._val
        elif elements:
            for e in elements:
                self._val |= 1 << int(e)

    @classmethod
    def from_int(cls, val: int) -> "BitSet":
        new_bs = cls()
        new_bs._val = val
        return new_bs

    @classmethod
    def from_bool_mask(cls, mask: np.ndarray) -> "BitSet":
        # Ensure mask is uint8 (0/1) before packing
        if mask.dtype != bool and mask.dtype != np.uint8:
            mask = mask.astype(bool)
        packed = np.packbits(mask.view(np.uint8), bitorder="little")
        val = int.from_bytes(packed.tobytes(), byteorder="little")
        return cls.from_int(val)

    @classmethod
    def full(cls, size: int) -> "BitSet":
        val = (1 << size) - 1 if size > 0 else 0
        return cls.from_int(val)

    @classmethod
    def universal(cls) -> "BitSet":
        """Returns a BitSet representing the universal set (all integers)."""
        return cls.from_int(-1)

    def add(self, element: int):
        self._val |= 1 << int(element)

    def remove(self, element: int):
        mask = 1 << int(element)
        if not (self._val & mask):
            raise KeyError(element)
        self._val &= ~mask

    def discard(self, element: int):
        self._val &= ~(1 << int(element))

    def __contains__(self, element: int) -> bool:
        return bool((self._val >> int(element)) & 1)

    def __iter__(self):
        if self._val < 0:
            raise OverflowError("Cannot iterate over infinite set")
        v = self._val
        while v:
            lsb = v & -v
            idx = lsb.bit_length() - 1
            yield idx
            v ^= lsb

    def __len__(self) -> int:
        if self._val < 0:
            raise OverflowError("Infinite set has no length")
        return self._val.bit_count()

    def __repr__(self) -> str:
        return f"BitSet({list(self) if self._val >= 0 else 'Univ.'})"

    def __eq__(self, other) -> bool:
        if isinstance(other, BitSet):
            return self._val == other._val
        return False

    def __hash__(self) -> int:
        return hash(self._val)

    @property
    def is_empty(self) -> bool:
        return self._val == 0

    @property
    def is_universal(self) -> bool:
        return self._val == -1

    @property
    def min(self) -> int:
        if self.is_empty:
            raise ValueError("BitSet is empty")
        return (self._val & -self._val).bit_length() - 1

    def max(self) -> int:  # TODO: test
        if self.is_empty:
            raise ValueError("BitSet is empty")
        return self._val.bit_length() - 1

    def union(self, other: "BitSet") -> "BitSet":
        if not isinstance(other, BitSet):
            return NotImplemented
        return BitSet.from_int(self._val | other._val)

    def intersection(self, other: "BitSet") -> "BitSet":
        if not isinstance(other, BitSet):
            return NotImplemented
        return BitSet.from_int(self._val & other._val)

    def difference(self, other: "BitSet") -> "BitSet":
        if not isinstance(other, BitSet):
            return NotImplemented
        return BitSet.from_int(self._val & ~other._val)

    def symmetric_difference(self, other: "BitSet") -> "BitSet":
        if not isinstance(other, BitSet):
            return NotImplemented
        return BitSet.from_int(self._val ^ other._val)

    def issubset(self, other: "BitSet") -> bool:
        return (self._val & other._val) == self._val

    def issuperset(self, other: "BitSet") -> bool:
        return (self._val & other._val) == other._val

    def copy(self) -> "BitSet":
        return BitSet.from_int(self._val)

    def __and__(self, other):
        return self.intersection(other)

    def __or__(self, other):
        return self.union(other)

    def __sub__(self, other):
        return self.difference(other)

    def __xor__(self, other):
        return self.symmetric_difference(other)

    def to_bool_mask(self, size: int) -> np.ndarray:
        n_bytes = (size + 7) // 8
        val = self._val & ((1 << size) - 1)
        bytes_val = val.to_bytes(n_bytes, byteorder="little")
        bits = np.unpackbits(np.frombuffer(bytes_val, dtype=np.uint8), bitorder="little")
        return bits[:size].view(bool)

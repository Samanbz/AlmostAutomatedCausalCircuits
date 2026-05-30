from abc import ABC, abstractmethod
from typing import Generic, TypeVar


K = TypeVar("K")


class NodeAllocator(ABC, Generic[K]):
    @abstractmethod
    def next_id(self) -> K:
        """Returns the next unique node ID."""
        pass


class IncrementalNodeAllocator(NodeAllocator[int]):
    def __init__(self, start: int = 1):
        self.counter = start

    def next_id(self) -> int:
        nid = self.counter
        self.counter += 1
        return nid

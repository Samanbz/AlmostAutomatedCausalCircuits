from typing import List, Optional

import torch

from src.symbolic.base import Node
from src.utils import BitSet, Support


class ArithmeticNode(Node):
    """Represents an arithmetic operation node."""

    def __init__(
        self,
        support: Support = None,
        unit_count: int = 1,
        unit_supports: Optional[List[Support]] = None,
        md_set: Optional[BitSet] = None,
    ):
        self.support = support
        self.unit_count = unit_count
        # If unit_supports is not provided, all units share the same support
        self.unit_supports = (
            unit_supports or [support] * unit_count if support is not None else None
        )
        # Marginal determinism set for this node's region (None if unknown)
        self.md_set = md_set

    @property
    def scope(self) -> BitSet:
        if self.support is None:
            return BitSet.empty()
        return self.support.scope

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self):
        scope_str = self.support.scope if self.support is not None else None
        return f"{self.__class__.__name__}(scope={scope_str}, unit_count={self.unit_count})"

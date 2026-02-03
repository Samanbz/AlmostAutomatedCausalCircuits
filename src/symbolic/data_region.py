from typing import Any, Dict

import numpy as np

from src.graph import DirectedAcyclicGraph
from src.utils import BitSet, DataSlice, Interval

from .region import RegionGraphNode


class DataRegionGraphNode(RegionGraphNode):
    """RegionGraphNode that holds data slice information and constraints."""

    def __init__(
        self,
        scope: BitSet,
        row_ids: BitSet,
    ):
        super().__init__(scope)
        self.row_ids = row_ids

    def get_data_slice(self, data: np.ndarray) -> DataSlice:
        return DataSlice(data, self.row_ids, self.scope)


class DataRegionNode(DataRegionGraphNode):
    pass


class DataPartitionNode(DataRegionGraphNode):
    def __init__(
        self,
        scope: BitSet,
        row_ids: BitSet,
        constraints: Dict[int, Interval] = None,
    ):
        DataRegionGraphNode.__init__(self, scope, row_ids)
        self.constraints = constraints if constraints is not None else {}


class DataRegionGraph(DirectedAcyclicGraph[int, DataRegionGraphNode, Any]):
    """RegionGraph that holds data slices at each node."""

    pass

from typing import Any, Dict

import numpy as np

from src.utils import BitSet, DataSlice, Support

from .base import DirectedAcyclicGraph, Node
from .region_graph import RegionGraphNode


class DataRegionGraphNode(RegionGraphNode):
    """RegionGraphNode that holds data slice information and constraints."""

    def __init__(
        self,
        scope: BitSet,
        row_ids: BitSet,
        constraints: Support = None,
    ):
        super().__init__(scope)
        self.row_ids = row_ids
        self.constraints = constraints if constraints is not None else Support()

    def get_data_slice(self, data: np.ndarray) -> DataSlice:
        return DataSlice(data, self.row_ids, self.scope, self.constraints)


class DataRegionNode(DataRegionGraphNode):
    pass


class DataPartitionNode(DataRegionGraphNode):
    pass


class DataRegionGraph(DirectedAcyclicGraph[int, DataRegionGraphNode, Any]):
    """RegionGraph that holds data slices at each node."""

    def _get_base_node_config(self) -> Dict[type, Dict[str, Any]]:
        config = super()._get_base_node_config()

        config.update(
            {
                DataRegionNode: {
                    "color": "#ffcc99",
                    "label": lambda n: f"Region\nScope: {list(n.scope)}\nRows: {len(n.row_ids)}\nConstraints: {n.constraints}",
                    "shape": "box",
                },
                DataPartitionNode: {
                    "color": "#99ccff",
                    "label": lambda n: f"Partition\nScope: {list(n.scope)}\nRows: {len(n.row_ids)}\nConstraints: {n.constraints}",
                    "shape": "ellipse",
                },
            }
        )
        return config

import numpy as np

from .bitset import BitSet
from .support import Support


class DataSlice:
    def __init__(
        self,
        data: np.ndarray,
        row_ids: BitSet,
        col_ids: BitSet,
        constraints: Support = None,
    ):
        self.data = data
        self.row_ids = row_ids
        self.col_ids = col_ids
        self.constraints = constraints if constraints is not None else Support()

    def get_data(self) -> np.ndarray:
        row_mask = self.row_ids.to_bool_mask(self.data.shape[0])
        col_mask = self.col_ids.to_bool_mask(self.data.shape[1])
        return self.data[row_mask][:, col_mask]

    def get_column(self, global_col_idx: int) -> np.ndarray:
        row_mask = self.row_ids.to_bool_mask(self.data.shape[0])
        return self.data[row_mask, global_col_idx]

    def __len__(self):
        return len(self.row_ids)

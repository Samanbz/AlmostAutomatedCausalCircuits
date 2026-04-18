from typing import List

import torch
import torch.nn as nn

from src.compilation.tensorized_circuit import TensorizedCircuit, TensorizedLayer, TuckerLayer
from src.symbolic import SymbolicArithmeticCircuit
from src.utils.monarch import MonarchMatrix


class MonarchTuckerLayer(TuckerLayer):
    """
    Hardware-optimized Tucker layer that uses Monarch matrices
    to represent the weighted sum of implicit products.
    """

    def __init__(
        self,
        out_idx: List[int],
        left_indices: List[int],
        right_indices: List[int],
        W: torch.Tensor,
        z_mask: torch.Tensor = None,
    ):
        nn.Module.__init__(self)
        self.out_idx = out_idx
        self.register_buffer("left_indices", torch.tensor(left_indices, dtype=torch.long))
        self.register_buffer("right_indices", torch.tensor(right_indices, dtype=torch.long))

        if z_mask is None:
            z_mask = torch.zeros(len(out_idx), dtype=torch.bool)
        self.register_buffer("z_mask", z_mask)

        # W shape: (G_P, G_L, G_R, h_out, h_l, h_r)
        self.G_P, self.G_L, self.G_R, self.h_out, self.h_l, self.h_r = W.shape

        # We need to map each valid (p, l, r) connection to a Monarch Matrix
        # Find non-zero blocks
        W_sum = W.abs().sum(dim=(3, 4, 5))
        valid_mask = W_sum > 0

        self.register_buffer("valid_mask", valid_mask)

        # Extract valid blocks and factorize
        valid_blocks = W[valid_mask]  # (num_valid, h_out, h_l, h_r)
        num_valid = valid_blocks.shape[0]

        if num_valid > 0:
            flattened_weights = valid_blocks.view(num_valid, self.h_out, self.h_l * self.h_r)

            ls, rs = [], []
            for i in range(num_valid):
                monarch = MonarchMatrix.from_dense(flattened_weights[i])
                ls.append(monarch.L)
                rs.append(monarch.R)

            self.L = nn.Parameter(torch.stack(ls))  # (P, b, c, k)
            self.R = nn.Parameter(torch.stack(rs))  # (P, k, b, b1)

            self.b, self.c, self.k, self.b1 = monarch.b, monarch.c, monarch.k, monarch.b1
        else:
            self.L = None
            self.R = None

    def _compute_tucker(self, global_buffer: torch.Tensor) -> torch.Tensor:
        batch_size = global_buffer.shape[0]
        res = torch.zeros(
            batch_size, self.G_P, self.h_out, device=global_buffer.device, dtype=global_buffer.dtype
        )

        if self.L is None:
            return res.flatten(1)

        L = global_buffer[:, self.left_indices].view(batch_size, self.G_L, self.h_l)
        R = global_buffer[:, self.right_indices].view(batch_size, self.G_R, self.h_r)

        # Implicit products: (B, G_L, G_R, h_l, h_r) -> (B, G_L, G_R, hl*hr)
        products = (L.unsqueeze(2).unsqueeze(-1) * R.unsqueeze(1).unsqueeze(-2)).flatten(3)

        # Filter valid products: (B, num_valid, hl*hr)
        # valid_mask: (G_P, G_L, G_R)
        # We need to gather the (l, r) pairs for each valid p.
        # Let's extract the p, l, r indices of valid blocks
        p_idx, l_idx, r_idx = torch.where(self.valid_mask)

        # valid_products: (B, num_valid, hl*hr)
        valid_products = products[:, l_idx, r_idx, :]

        num_valid = valid_products.shape[1]

        # Apply Batched Monarch Matrix: (Batch, num_valid, hl*hr) -> (Batch, num_valid, h_out)
        # products view: (Batch, num_valid, k, b1) -> (num_valid*k, Batch, b1)
        x = valid_products.view(batch_size, num_valid, self.k, self.b1)
        x_in = x.permute(1, 2, 0, 3).reshape(num_valid * self.k, batch_size, self.b1)

        # R_mat: (num_valid*k, b1, b)
        R_mat = self.R.transpose(2, 3).reshape(num_valid * self.k, self.b1, self.b)

        y = torch.bmm(x_in, R_mat)  # (num_valid*k, Batch, b)

        # Permute and multiply by L factor
        # y: (num_valid, k, Batch, b) -> (num_valid, b, Batch, k) -> (num_valid*b, Batch, k)
        y = y.view(num_valid, self.k, batch_size, self.b).permute(0, 3, 2, 1)
        y_in = y.reshape(num_valid * self.b, batch_size, self.k)

        # L_mat: (num_valid*b, k, c)
        L_mat = self.L.transpose(2, 3).reshape(num_valid * self.b, self.k, self.c)

        z = torch.bmm(y_in, L_mat)  # (num_valid*b, Batch, c)

        # Reshape back to (Batch, num_valid, h_out)
        out = (
            z.view(num_valid, self.b, batch_size, self.c)
            .permute(2, 0, 3, 1)
            .reshape(batch_size, num_valid, -1)
        )

        # Scatter add to res
        # res: (B, G_P, h_out)
        # out: (B, num_valid, h_out)
        # p_idx: (num_valid,)

        p_idx_expanded = p_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.h_out)
        res.scatter_add_(1, p_idx_expanded, out)

        return res.flatten(1)

    def forward(
        self, buf_joint: torch.Tensor, buf_xz: torch.Tensor = None, buf_z: torch.Tensor = None
    ) -> torch.Tensor:
        res_joint = self._compute_tucker(buf_joint)
        if buf_xz is None or buf_z is None:
            buf_joint[:, self.out_idx] = res_joint
            return res_joint
        res_xz = self._compute_tucker(buf_xz)
        res_z = self._compute_tucker(buf_z)
        interventional_math = res_joint - res_xz + res_z
        interventional_math = torch.nan_to_num(interventional_math, nan=0.0)
        z_mask_expanded = self.z_mask.unsqueeze(0).expand_as(res_joint)
        res_final = torch.where(z_mask_expanded, interventional_math, res_joint)
        buf_joint[:, self.out_idx] = res_final
        buf_xz[:, self.out_idx] = res_xz
        buf_z[:, self.out_idx] = res_z
        return res_final


class MonarchCircuit(TensorizedCircuit):
    """
    Subclass of TensorizedCircuit that replaces standard dense Tucker layers
    with memory-efficient, hardware-optimized structured sparse `MonarchMatrix` layers.
    """

    def __init__(self, symbolic_circuit: SymbolicArithmeticCircuit, respect_sparsity: bool = True):
        self.respect_sparsity = respect_sparsity
        super().__init__(symbolic_circuit)

    def _build_tucker_layer(
        self, circuit: SymbolicArithmeticCircuit, sum_node_ids: List[int], out_idx: List[int]
    ):
        # Re-use parameter gathering from TensorizedCircuit but return Monarch layer
        base_layer = super()._build_tucker_layer(circuit, sum_node_ids, out_idx)
        return MonarchTuckerLayer(
            out_idx,
            base_layer.left_indices.tolist(),
            base_layer.right_indices.tolist(),
            base_layer.weights.data,
        )

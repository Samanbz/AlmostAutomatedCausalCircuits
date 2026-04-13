import math
from typing import List, Tuple

import torch
import torch.nn as nn

from src.compilation.monarch import LogMonarchMatrix
from src.compilation.tensorized_circuit import TensorizedCircuit, TensorizedLayer
from src.symbolic import SymbolicArithmeticCircuit


class MonarchSumLayer(TensorizedLayer):
    """
    Hardware-efficient layer that paramaterizes sum logic using a sequence of block-diagonal
    Monarch structured sparse matrices in LOG space. Replaces dense tensor sum layers.
    """

    def __init__(
        self,
        out_idx: List[int],
        in_idx: List[int],
        connections: torch.Tensor,
        z_mask: torch.Tensor = None,
        respect_sparsity: bool = True,
    ):
        super().__init__(out_idx)
        self.in_idx = in_idx

        if z_mask is None:
            z_mask = torch.zeros(len(out_idx), dtype=torch.bool)
        self.register_buffer("z_mask", z_mask)

        # Dimensions
        out_dim, in_dim = connections.shape
        b, c, k, b1 = self._auto_factorize(out_dim, in_dim)

        # Build logical equivalent from dense, using Log space
        self.log_monarch = LogMonarchMatrix.from_dense(
            connections.clone(), b=b, c=c, k=k, b1=b1, respect_sparsity=respect_sparsity
        )
        # Store metadata
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.register_buffer("connections_mask", connections == 0)

    def _auto_factorize(self, out_dim: int, in_dim: int) -> Tuple[int, int, int, int]:
        """Automatically compute integer block factors for arbitrary rectangular dimensions."""
        if out_dim == in_dim and math.isqrt(in_dim) ** 2 == in_dim:
            m = math.isqrt(in_dim)
            return m, m, m, m

        def get_mid_factor(n):
            for i in range(math.isqrt(n), 0, -1):
                if n % i == 0:
                    return i
            return 1

        c = get_mid_factor(out_dim)
        b = out_dim // c

        if in_dim % c == 0:
            k = c
            b1 = in_dim // k
        else:
            k = get_mid_factor(in_dim)
            b1 = in_dim // k

        return b, c, k, b1

    def _compute_logsumexp(self, global_buffer: torch.Tensor) -> torch.Tensor:
        input_values = global_buffer[:, self.in_idx]
        out = self.log_monarch(input_values)
        return out

    def forward(
        self, buf_joint: torch.Tensor, buf_xz: torch.Tensor = None, buf_z: torch.Tensor = None
    ) -> torch.Tensor:
        res_joint = self._compute_logsumexp(buf_joint)

        if buf_xz is None or buf_z is None:
            buf_joint[:, self.out_idx] = res_joint
            return res_joint

        res_xz = self._compute_logsumexp(buf_xz)
        res_z = self._compute_logsumexp(buf_z)

        # Causal adjustment calculation in LOG space
        interventional_math = res_joint - res_xz + res_z
        interventional_math = torch.nan_to_num(interventional_math, nan=float("-inf"))

        z_mask_expanded = self.z_mask.unsqueeze(0).expand_as(res_joint)
        res_final = torch.where(z_mask_expanded, interventional_math, res_joint)

        buf_joint[:, self.out_idx] = res_final
        buf_xz[:, self.out_idx] = res_xz
        buf_z[:, self.out_idx] = res_z

        return res_final

    def update_params(self):
        with torch.no_grad():
            self._update_factor(self.log_monarch.log_R)
            self._update_factor(self.log_monarch.log_L)

    def _update_factor(self, param: nn.Parameter):
        responsibilities = param.grad
        if responsibilities is None:
            return

        current_probs = torch.exp(param)
        structural_zeros = current_probs == 0

        total_resp = responsibilities.sum(dim=-1, keepdim=True)
        mask = total_resp > 0

        smoothed_resp = responsibilities + 1e-5
        smoothed_resp.masked_fill_(structural_zeros, 0.0)

        new_probs = smoothed_resp / smoothed_resp.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        new_probs = torch.where(mask, new_probs, current_probs)
        new_probs.masked_fill_(structural_zeros, 0.0)

        param.copy_(
            torch.where(
                structural_zeros,
                torch.tensor(float("-inf"), device=param.device),
                torch.log(new_probs.clamp(min=1e-10)),
            )
        )
        param.grad.zero_()


class MonarchCircuit(TensorizedCircuit):
    """
    Subclass of TensorizedCircuit that replaces standard dense sum layers
    with memory-efficient, hardware-optimized structured sparse `MonarchMatrix` layers.
    """

    def __init__(self, symbolic_circuit: SymbolicArithmeticCircuit, respect_sparsity: bool = True):
        self.respect_sparsity = respect_sparsity
        super().__init__(symbolic_circuit)

    def _build_sum_layer(
        self,
        out_idx: List[int],
        in_idx: List[int],
        connections: torch.Tensor,
        z_mask: torch.Tensor = None,
    ) -> TensorizedLayer:
        """Override the Factory pattern from TensorizedCircuit"""
        return MonarchSumLayer(
            out_idx, in_idx, connections, z_mask=z_mask, respect_sparsity=self.respect_sparsity
        )

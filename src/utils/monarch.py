import math
from typing import Tuple

import torch


class MonarchMatrix:
    @staticmethod
    def _auto_factorize(out_dim: int, in_dim: int) -> tuple[int, int, int, int]:
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

    @staticmethod
    def random(shape: Tuple[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(shape) >= 2, "Shape must have at least 2 dimensions"
        *rest, out_dim, in_dim = shape

        b, c, k, b1 = MonarchMatrix._auto_factorize(out_dim, in_dim)
        L = torch.randn(*rest, b, c, k)
        R = torch.randn(*rest, k, b, b1)
        return L, R

    @staticmethod
    def from_dense(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a dense weight matrix into Monarch factors L and R using rank-1 SVD approximation."""
        assert W.dim() >= 2, "Weight matrix must have at least 2 dimensions"
        *rest, out_dim, in_dim = W.shape
        b, c, k, b1 = MonarchMatrix._auto_factorize(out_dim, in_dim)

        W_reshaped = W.reshape(*rest, b, c, k, b1)
        W_perm = W_reshaped.permute(*range(len(rest)), -4, -2, -3, -1)

        U, S, Vh = torch.linalg.svd(W_perm, full_matrices=False)

        L = torch.abs(U[..., 0]) * torch.sqrt(S[..., 0]).unsqueeze(-1)
        R = torch.abs(Vh[..., 0, :]) * torch.sqrt(S[..., 0]).unsqueeze(-1)

        L = L.permute(*range(len(rest)), -3, -1, -2)
        R = R.permute(*range(len(rest)), -2, -3, -1)

        return L, R

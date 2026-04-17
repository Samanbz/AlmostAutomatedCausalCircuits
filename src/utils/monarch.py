import math

import torch
import torch.nn as nn

from src.compilation.base_circuit import SafeLogSumExp


def log_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batched matrix multiplication in log space using SafeLogSumExp.
    Computes log(exp(A) @ exp(B))
    A: (b, n, m)
    B: (b, m, p)
    returns: (b, n, p)
    """
    return SafeLogSumExp.apply(A.unsqueeze(-1) + B.unsqueeze(1), 2)


def _get_rectangular_slices(A: torch.Tensor, b: int, c: int, k: int, b1: int):
    """
    Reshapes the rectangular matrix A into (b * k) independent (c x b1) slices.
    """
    A_4d = A.view(c, b, k, b1)  # Axes: (l, j, k, i)
    M_batched = A_4d.permute(1, 2, 0, 3).reshape(b * k, c, b1)
    return M_batched


def _batched_greedy_mwvc(A_batched: torch.Tensor, mask_batched: torch.Tensor):
    """
    GPU-Parallel Greedy Minimum Weight Vertex Cover generalized for rectangular slices.
    """
    B, c, b1 = A_batched.shape
    device = A_batched.device

    F_curr = (1.0 - mask_batched).float()
    allowed_A = A_batched * mask_batched

    C_row = (allowed_A**2).sum(dim=2)
    C_col = (allowed_A**2).sum(dim=1)

    row_del = torch.zeros((B, c), dtype=torch.bool, device=device)
    col_del = torch.zeros((B, b1), dtype=torch.bool, device=device)

    eps = 1e-8

    for _ in range(c + b1):
        row_edges = F_curr.sum(dim=2)
        col_edges = F_curr.sum(dim=1)

        if row_edges.max() == 0:
            break

        row_eff = row_edges / (C_row + eps)
        col_eff = col_edges / (C_col + eps)

        row_eff[row_del] = -1.0
        col_eff[col_del] = -1.0

        max_r_eff, best_r = row_eff.max(dim=1)
        max_c_eff, best_c = col_eff.max(dim=1)

        active_batches = row_edges.sum(dim=1) > 0

        choose_row = active_batches & (max_r_eff >= max_c_eff)
        choose_col = active_batches & (~choose_row)

        if choose_row.any():
            r_idx = best_r[choose_row]
            row_del[choose_row, r_idx] = True
            F_curr[choose_row, r_idx, :] = 0.0

        if choose_col.any():
            c_idx = best_c[choose_col]
            col_del[choose_col, c_idx] = True
            F_curr[choose_col, :, c_idx] = 0.0

    return row_del, col_del


class MonarchMatrix(nn.Module):
    def __init__(
        self, b: int, c: int, k: int, b1: int, L: torch.Tensor = None, R: torch.Tensor = None
    ):
        """
        Creates a Generalized Rectangular Monarch Matrix.
        If L and R are not provided, initializes them randomly.
        """
        super().__init__()

        self.b = b
        self.c = c
        self.k = k
        self.b1 = b1

        self.in_dim = k * b1
        self.out_dim = c * b

        if L is not None and R is not None:
            assert L.shape == (b, c, k), f"Expected L shape {(b, c, k)}, got {L.shape}"
            assert R.shape == (k, b, b1), f"Expected R shape {(k, b, b1)}, got {R.shape}"
            self.L = nn.Parameter(L)
            self.R = nn.Parameter(R)
        else:
            # Standard PyTorch initialization (scaled by 1 / sqrt(fan_in))
            self.L = nn.Parameter(torch.randn(b, c, k) / math.sqrt(k))
            self.R = nn.Parameter(torch.randn(k, b, b1) / math.sqrt(b1))

    @property
    def shape(self):
        return (self.out_dim, self.in_dim)

    def forward(self, x):
        return self.matmul(x)

    def __matmul__(self, x):
        return self.matmul(x)

    def matmul(self, x):
        is_1d = x.dim() == 1
        if is_1d:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        # 1. Reshape x and multiply by R
        x_reshaped = x.view(batch_size, self.k, self.b1)
        x_perm = x_reshaped.permute(1, 0, 2).contiguous()  # (k, batch, b1)

        R_perm = self.R.transpose(1, 2)  # (k, b1, b)
        y = torch.bmm(x_perm, R_perm)  # (k, batch, b)

        # 2. Permute intermediate result and multiply by L
        y_perm = y.permute(2, 1, 0).contiguous()  # (b, batch, k)

        L_perm = self.L.transpose(1, 2)  # (b, k, c)
        z = torch.bmm(y_perm, L_perm)  # (b, batch, c)

        # 3. Permute back and flatten to output vector
        out = z.permute(1, 2, 0).contiguous().view(batch_size, self.out_dim)

        if is_1d:
            out = out.squeeze(0)

        return out

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

    @classmethod
    def from_dense(
        cls,
        mat: torch.Tensor,
        b: int = None,
        c: int = None,
        k: int = None,
        b1: int = None,
        respect_sparsity: bool = True,
    ):
        assert mat.dim() == 2
        out_dim, in_dim = mat.shape

        if b is None or c is None or k is None or b1 is None:
            b, c, k, b1 = cls._auto_factorize(out_dim, in_dim)

        assert out_dim == c * b, f"out_dim ({out_dim}) must equal c * b ({c * b})"
        assert in_dim == k * b1, f"in_dim ({in_dim}) must equal k * b1 ({k * b1})"

        M_batched = _get_rectangular_slices(mat, b, c, k, b1)

        if respect_sparsity:
            mask = (mat != 0).float()
            mask_batched = _get_rectangular_slices(mask, b, c, k, b1)
            row_del, col_del = _batched_greedy_mwvc(M_batched, mask_batched)

        # Use SVD to get the best rank-1 approximation for each slice
        # M_batched is (b*k, c, b1)
        U, S, Vh = torch.linalg.svd(M_batched, full_matrices=False)
        # Take first singular value and vectors
        u = U[:, :, 0] * torch.sqrt(S[:, 0:1])  # (b*k, c)
        v = Vh[:, 0, :] * torch.sqrt(S[:, 0:1])  # (b*k, b1)

        if respect_sparsity:
            u = u.masked_fill(row_del, 0.0)
            v = v.masked_fill(col_del, 0.0)

        L_tilde = u.view(b, k, c).permute(0, 2, 1)  # (b, c, k)
        R_tilde = v.view(b, k, b1).permute(1, 0, 2)  # (k, b, b1)

        return cls(b=b, c=c, k=k, b1=b1, L=L_tilde, R=R_tilde)

    @classmethod
    def random_like(
        cls,
        mat: torch.Tensor,
        b: int = None,
        c: int = None,
        k: int = None,
        b1: int = None,
        respect_sparsity: bool = True,
    ):
        assert mat.dim() == 2
        out_dim, in_dim = mat.shape

        if b is None or c is None or k is None or b1 is None:
            b, c, k, b1 = cls._auto_factorize(out_dim, in_dim)

        # 1. Initialize random blocks
        L = torch.randn(b, c, k, device=mat.device) / math.sqrt(k)
        R = torch.randn(k, b, b1, device=mat.device) / math.sqrt(b1)

        if respect_sparsity:
            mask = (mat != 0).float()
            mask_batched = _get_rectangular_slices(mask, b, c, k, b1)
            row_del, col_del = _batched_greedy_mwvc(
                torch.ones_like(mask_batched), mask_batched
            )  # weight by ones since we don't care about values

            L = L.masked_fill(row_del.view(b, k, c).permute(0, 2, 1), 0.0)
            R = R.masked_fill(col_del.view(b, k, b1).permute(1, 0, 2), 0.0)

        return cls(b=b, c=c, k=k, b1=b1, L=L, R=R)

    def to_dense(self):
        M_4d = torch.einsum("blk,kbi->lbki", self.L, self.R)
        return M_4d.reshape(self.out_dim, self.in_dim)


class LogMonarchMatrix(nn.Module):
    def __init__(
        self,
        b: int,
        c: int,
        k: int,
        b1: int,
        log_L: torch.Tensor = None,
        log_R: torch.Tensor = None,
    ):
        super().__init__()
        self.b = b
        self.c = c
        self.k = k
        self.b1 = b1
        self.out_dim = c * b
        self.in_dim = k * b1

        if log_L is not None and log_R is not None:
            assert log_L.shape == (b, c, k), f"Expected log_L shape {(b, c, k)}, got {log_L.shape}"
            assert log_R.shape == (k, b, b1), (
                f"Expected log_R shape {(k, b, b1)}, got {log_R.shape}"
            )
            self.log_L = nn.Parameter(log_L)
            self.log_R = nn.Parameter(log_R)
        else:
            self.log_L = nn.Parameter(torch.randn(b, c, k) / math.sqrt(k))
            self.log_R = nn.Parameter(torch.randn(k, b, b1) / math.sqrt(b1))

    @property
    def shape(self):
        return (self.out_dim, self.in_dim)

    @classmethod
    def from_dense(
        cls,
        mat: torch.Tensor,
        b: int = None,
        c: int = None,
        k: int = None,
        b1: int = None,
        respect_sparsity: bool = True,
    ):
        base_monarch = MonarchMatrix.from_dense(
            mat, b=b, c=c, k=k, b1=b1, respect_sparsity=respect_sparsity
        )

        L_val = torch.abs(base_monarch.L.detach())
        R_val = torch.abs(base_monarch.R.detach())

        # Normalize factors to properly initialize log-sum layer parameters
        L_sums = L_val.sum(dim=-1, keepdim=True)
        L_val = torch.where(L_sums == 0, L_val, L_val / L_sums)

        R_sums = R_val.sum(dim=-1, keepdim=True)
        R_val = torch.where(R_sums == 0, R_val, R_val / R_sums)

        # Any exactly 0 values map to -inf, the rest use log
        log_L = torch.where(
            L_val == 0,
            torch.tensor(float("-inf"), device=L_val.device),
            torch.log(L_val.clamp(min=1e-12)),
        )
        log_R = torch.where(
            R_val == 0,
            torch.tensor(float("-inf"), device=R_val.device),
            torch.log(R_val.clamp(min=1e-12)),
        )

        return cls(
            b=base_monarch.b,
            c=base_monarch.c,
            k=base_monarch.k,
            b1=base_monarch.b1,
            log_L=log_L,
            log_R=log_R,
        )

    @classmethod
    def random_like(
        cls,
        mat: torch.Tensor,
        b: int = None,
        c: int = None,
        k: int = None,
        b1: int = None,
        respect_sparsity: bool = True,
    ):
        base_monarch = MonarchMatrix.random_like(
            mat, b=b, c=c, k=k, b1=b1, respect_sparsity=respect_sparsity
        )

        L_val = torch.abs(base_monarch.L.detach())
        R_val = torch.abs(base_monarch.R.detach())

        # Normalize factors to properly initialize log-sum layer parameters
        L_sums = L_val.sum(dim=-1, keepdim=True)
        L_val = torch.where(L_sums == 0, L_val, L_val / L_sums)

        R_sums = R_val.sum(dim=-1, keepdim=True)
        R_val = torch.where(R_sums == 0, R_val, R_val / R_sums)

        # Any exactly 0 values map to -inf, the rest use log
        log_L = torch.where(
            L_val == 0,
            torch.tensor(float("-inf"), device=L_val.device),
            torch.log(L_val.clamp(min=1e-12)),
        )
        log_R = torch.where(
            R_val == 0,
            torch.tensor(float("-inf"), device=R_val.device),
            torch.log(R_val.clamp(min=1e-12)),
        )

        return cls(
            b=base_monarch.b,
            c=base_monarch.c,
            k=base_monarch.k,
            b1=base_monarch.b1,
            log_L=log_L,
            log_R=log_R,
        )

    def forward(self, log_x):
        is_1d = log_x.dim() == 1
        if is_1d:
            log_x = log_x.unsqueeze(0)

        batch_size = log_x.shape[0]

        log_x_reshaped = log_x.view(batch_size, self.k, self.b1)
        log_x_perm = log_x_reshaped.permute(1, 0, 2).contiguous()

        log_R_perm = self.log_R.transpose(1, 2)
        log_y = log_bmm(log_x_perm, log_R_perm)

        log_y_perm = log_y.permute(2, 1, 0).contiguous()

        log_L_perm = self.log_L.transpose(1, 2)
        log_z = log_bmm(log_y_perm, log_L_perm)

        out = log_z.permute(1, 2, 0).contiguous().view(batch_size, self.out_dim)

        if is_1d:
            out = out.squeeze(0)

        return out

    def to_dense(self):
        # We compute the dense log_M_4d shape (c, b, k, b1), which matches lbki
        # log_L is (b, c, k). We permute to (c, b, k) -> l, b, k, and add a dimension for i
        log_L_perm = self.log_L.permute(1, 0, 2).unsqueeze(-1)
        # log_R is (k, b, b1). We permute to (b, k, b1) -> b, k, i, and add a dimension for l
        log_R_perm = self.log_R.permute(1, 0, 2).unsqueeze(0)

        log_M_4d = log_L_perm + log_R_perm
        return log_M_4d.reshape(self.out_dim, self.in_dim)

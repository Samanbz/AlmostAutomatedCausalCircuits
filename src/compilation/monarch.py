import math

import torch
import torch.nn as nn


def _get_slices(A: torch.Tensor):
    """
    Reshapes the matrix A into m^2 independent m x m slices.
    """
    n = A.shape[0]
    m = int(math.sqrt(n))
    A_4d = A.view(m, m, m, m)
    M_batched = A_4d.permute(1, 2, 0, 3).reshape(m * m, m, m)
    return M_batched


def _batched_greedy_mwvc(A_batched: torch.Tensor, mask_batched: torch.Tensor):
    """
    GPU-Parallel Greedy Minimum Weight Vertex Cover.
    Runs entirely on device tensor math without loops over slices.
    """
    B, m, _ = A_batched.shape
    device = A_batched.device

    # F_curr tracks the remaining forbidden edges (1 = forbidden, 0 = allowed)
    F_curr = (1.0 - mask_batched).float()

    # Calculate costs (L2 energy of ALLOWED connections in each row/col)
    allowed_A = A_batched * mask_batched
    C_row = (allowed_A**2).sum(dim=2)  # shape: (B, m)
    C_col = (allowed_A**2).sum(dim=1)  # shape: (B, m)

    row_del = torch.zeros((B, m), dtype=torch.bool, device=device)
    col_del = torch.zeros((B, m), dtype=torch.bool, device=device)

    eps = 1e-8

    # The loop runs at most 2m times (max possible row/col deletions)
    for _ in range(m * 2):
        row_edges = F_curr.sum(dim=2)
        col_edges = F_curr.sum(dim=1)

        # If no forbidden edges remain in ANY batch, we are perfectly done
        if row_edges.max() == 0:
            break

        # Efficiency = Edges removed / Cost of destroying valid data
        row_eff = row_edges / (C_row + eps)
        col_eff = col_edges / (C_col + eps)

        # Ignore already deleted rows/cols
        row_eff[row_del] = -1.0
        col_eff[col_del] = -1.0

        # Find best row and col for each batch
        max_r_eff, best_r = row_eff.max(dim=1)
        max_c_eff, best_c = col_eff.max(dim=1)

        active_batches = row_edges.sum(dim=1) > 0

        # Decide whether row or col is more efficient for each active batch
        choose_row = active_batches & (max_r_eff >= max_c_eff)
        choose_col = active_batches & (~choose_row)

        if choose_row.any():
            r_idx = best_r[choose_row]
            row_del[choose_row, r_idx] = True
            # Zero out the forbidden edges that this row just covered
            F_curr[choose_row, r_idx, :] = 0.0

        if choose_col.any():
            c_idx = best_c[choose_col]
            col_del[choose_col, c_idx] = True
            # Zero out the forbidden edges that this column just covered
            F_curr[choose_col, :, c_idx] = 0.0

    return row_del, col_del


class MonarchMatrix(nn.Module):
    def __init__(self, L: torch.Tensor, R: torch.Tensor):
        super().__init__()
        assert L.shape == R.shape and L.dim() == 3
        self.m = L.shape[0]
        self.n = self.m * self.m
        self.L = nn.Parameter(L)
        self.R = nn.Parameter(R)

    @property
    def shape(self):
        return (self.n, self.n)

    def forward(self, x):
        return self.matmul(x)

    def __matmul__(self, x):
        return self.matmul(x)

    def matmul(self, x):
        """
        Steps to multiply x by a Monarch matrix M = PLP^T R:
        1. Multiply R by x: y_{kj} = \sum_i R_{kji} x_{ki}
        2. Multiply PLP^T by y: z_{lj} = \sum_k L_{jlk} y_{kj}
        3. Reshape z back into a vector of size n, and return this.
        """
        # Assume x can be 1D (n,) or 2D (batch, n)
        is_1d = x.dim() == 1
        if is_1d:
            x = x.unsqueeze(0)  # (1, n)

        batch_size = x.shape[0]

        # Reshape x to (batch, m, m). x_{b, k, i}
        x_reshaped = x.view(batch_size, self.m, self.m)

        # 1. Multiply R by x: y_{b, k, j} = \sum_i R_{k, j, i} x_{b, k, i}
        # Use BMM. Treat k as batch dim.
        # x: (b, k, i) -> permute to (k, b, i)
        x_perm = x_reshaped.permute(1, 0, 2).contiguous()

        # R: (k, j, i) -> permute to (k, i, j)
        R_perm = self.R.transpose(1, 2)

        # y: (k, b, j)
        y = torch.bmm(x_perm, R_perm)

        # 2. Multiply PLP^T by y: z_{b, l, j} = \sum_k L_{j, l, k} y_{b, k, j}
        # Use BMM. Treat j as batch dim.
        # y: (k, b, j) -> permute to (j, b, k)
        y_perm = y.permute(2, 1, 0).contiguous()

        # L: (j, l, k) -> permute to (j, k, l)
        L_perm = self.L.transpose(1, 2)

        # z: (j, b, l)
        z = torch.bmm(y_perm, L_perm)

        # 3. Reshape z back into a vector of size n
        # z: (j, b, l) -> permute to (b, l, j)
        out = z.permute(1, 2, 0).contiguous().view(batch_size, self.n)

        if is_1d:
            out = out.squeeze(0)

        return out

    @classmethod
    def from_dense(cls, mat: torch.Tensor, m: int = None, respect_sparsity: bool = True):
        """
        Projection of a dense n x n matrix into Monarch factors L and R.
        Uses greedy MWVC to handle sparsity constraints if respected_sparsity is True.
        """
        assert mat.dim() == 2 and mat.shape[0] == mat.shape[1]
        n = mat.shape[0]

        if m is None:
            m = int(math.sqrt(n))
        assert n == m * m, "n must be a perfect square"

        M_batched = _get_slices(mat)

        if respect_sparsity:
            mask = (mat != 0).float()
            mask_batched = _get_slices(mask)

            # 1. Find optimal rows/cols to sacrifice to cover all zeros
            row_del, col_del = _batched_greedy_mwvc(M_batched, mask_batched)

            # 2. Modify the target matrix BEFORE SVD
            M_safe = M_batched.clone()

            # Broadcast col_del from (B, m) -> (B, 1, m) -> (B, m, m)
            row_mask = row_del.unsqueeze(-1)
            col_mask = col_del.unsqueeze(1)

            # Zero out the sacrificed rows and columns
            M_safe = M_safe * (~row_mask).float() * (~col_mask).float()
        else:
            M_safe = M_batched

        # 3. Exact SVD on the mathematically safe matrix
        U, S, Vh = torch.linalg.svd(M_safe, full_matrices=False)

        u = U[:, :, 0] * torch.sqrt(S[:, 0]).unsqueeze(1)
        v = Vh[:, 0, :] * torch.sqrt(S[:, 0]).unsqueeze(1)

        u = u.view(m, m, m)
        v = v.view(m, m, m)

        L_tilde = u.permute(0, 2, 1)
        R_tilde = v.permute(1, 0, 2)

        return cls(L_tilde, R_tilde)

    def to_dense(self):
        """
        Reconstructs the full n x n matrix from the Monarch factors L and R.
        """
        # Simply unpack the 0-th dimension to get the m blocks of size (m, m)
        L = torch.block_diag(*self.L)
        R = torch.block_diag(*self.R)

        indices = torch.arange(self.n).view(self.m, self.m).t().contiguous().view(-1)
        P = torch.eye(self.n, device=self.L.device)[indices]

        return P @ L @ P.T @ R

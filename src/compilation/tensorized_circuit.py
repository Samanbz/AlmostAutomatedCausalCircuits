"""
Tensorized Circuit: GPU-optimized nn.Module compiled from a FusedCircuit.

All operations in log-semiring. Forward pass uses a flat unit buffer of shape
[total_units, B] (units-first, batch-last) for maximally coalesced memory
access. Each layer has precomputed gather/scatter index tensors registered as
non-trainable buffers so the forward pass contains only indexed reads/writes
and no shape arithmetic.
"""

import math
from abc import ABC
from typing import List

import torch
import torch.nn as nn

from src.compilation.folded_circuit import FoldedGaussianInputLayer
from src.compilation.fused_circuit import (
    CPTLayer,
    FusedCircuit,
    FusedInputLayer,
    SparseHadamardLayer,
    SparseKroneckerLayer,
    TuckerLayer,
)


class TensorizedGaussianInput(nn.Module):
    """Truncated Gaussian input layer in log-space.

    Returns output in [N, h, B] layout (units-first, batch-last) to match
    the flat buffer's coalesced memory layout.
    """

    def __init__(
        self,
        means: torch.Tensor,
        stds: torch.Tensor,
        lows: torch.Tensor,
        highs: torch.Tensor,
        scopes: List[int],
        leaf_modes: torch.Tensor | None = None,
    ):
        super().__init__()

        # Add a small amount of random jitter to break symmetry for un-split Gaussians.
        jitter = torch.randn_like(means) * 0.1
        means = means + jitter

        self.means = nn.Parameter(means)
        self.stds = nn.Parameter(stds)
        self.register_buffer("lows", lows)
        self.register_buffer("highs", highs)

        signs = torch.ones_like(means)
        self.register_buffer("signs", signs)
        self.scopes = scopes
        self.num_nodes = means.shape[0]
        self.h_out = means.shape[1]

        # Per-node leaf mode: 0=normal, 1=constant(log1=0), 2=inverse(-log_pdf)
        if leaf_modes is None:
            leaf_modes = torch.zeros(self.num_nodes, dtype=torch.long)
        self.register_buffer("leaf_modes", leaf_modes)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data: [B, num_features]
        x_raw = data[:, self.scopes]  # [B, N]
        x = x_raw.T.unsqueeze(1)  # [N, 1, B]
        is_nan = torch.isnan(x)
        x_safe = torch.where(is_nan, torch.zeros_like(x), x)

        self._saved_x = x_safe.detach()  # [N, 1, B]
        self._nan_mask = is_nan.detach()  # [N, 1, B]

        means = self.means.unsqueeze(-1)  # [N, h, 1]
        stds = self.stds.unsqueeze(-1)  # [N, h, 1]

        z = (x_safe - means) / stds  # [N, h, B]
        log_pdf = -0.5 * math.log(2 * math.pi) - 0.5 * z**2 - torch.log(stds)

        sqrt2 = math.sqrt(2)
        z_hi = (self.highs.unsqueeze(-1) - means) / (stds * sqrt2)  # [N, h, 1]
        z_lo = (self.lows.unsqueeze(-1) - means) / (stds * sqrt2)  # [N, h, 1]

        cdf_hi = 0.5 * (1 + torch.erf(z_hi))
        cdf_lo = 0.5 * (1 + torch.erf(z_lo))
        log_Z = torch.log((cdf_hi - cdf_lo).clamp(min=1e-10))
        log_pdf = log_pdf - log_Z

        oob = (x_safe < self.lows.unsqueeze(-1)) | (x_safe > self.highs.unsqueeze(-1))
        # Use -1e30 rather than finfo.min: two finfo.min values added together in a
        # Hadamard product overflow float32 to -inf, which then causes NaN via -inf - (-inf).
        log_zero = -1e30
        log_pdf = torch.where(oob, torch.full_like(log_pdf, log_zero), log_pdf)
        log_pdf = torch.where(is_nan, torch.zeros_like(log_pdf), log_pdf)

        # Apply leaf modes: 0=normal, 1=constant(0), 2=inverse(-log_pdf), 3=indicator
        if self.leaf_modes.any():
            modes = self.leaf_modes.view(-1, 1, 1)  # [N, 1, 1]
            is_constant = modes == 1
            is_inverse = modes == 2
            is_indicator = modes == 3

            log_pdf = torch.where(is_constant, torch.zeros_like(log_pdf), log_pdf)

            # Inverse: negate but preserve OOB entries
            inv_result = -log_pdf
            inv_result = torch.where(
                log_pdf < -1e10, torch.full_like(log_pdf, log_zero), inv_result
            )
            log_pdf = torch.where(is_inverse, inv_result, log_pdf)

            # Indicator: 0 if in-support, -inf if OOB.
            # This is the result of Gaussian × InverseGaussian: the density
            # cancels but the support structure (which bin contains x) is preserved.
            indicator_result = torch.where(
                oob, torch.full_like(log_pdf, log_zero), torch.zeros_like(log_pdf)
            )
            indicator_result = torch.where(is_nan, torch.zeros_like(log_pdf), indicator_result)
            log_pdf = torch.where(is_indicator, indicator_result, log_pdf)

        self._saved_log_pdf = log_pdf
        log_pdf.requires_grad_(True)
        log_pdf.retain_grad()
        self._leaf_output = log_pdf

        return log_pdf  # [N, h, B]

    def update_params(self, responsibilities: torch.Tensor) -> None:
        """Update means, stds and truncation bounds via EM sufficient statistics.

        Args:
            responsibilities: [N, h, B] — gradient of log-likelihood w.r.t. log_pdf.
        """
        with torch.no_grad():
            x = self._saved_x  # [N, 1, B]
            nan_mask = self._nan_mask  # [N, 1, B]
            N, h = self.means.shape

            resp = responsibilities.clamp(min=0)
            resp = resp.masked_fill(nan_mask.expand_as(resp), 0)  # [N, h, B]

            is_split = ~((self.lows[:, 0] == float("-inf")) & (self.highs[:, 0] == float("inf")))

            if is_split.any():
                node_resp = resp.sum(dim=1)  # [N, B] — aggregate over h
                node_resp = node_resp.masked_fill(nan_mask.squeeze(1), 0)
                total_md = node_resp.sum(dim=-1)  # [N]
                valid_md = total_md > 1e-5

                x_flat = x.squeeze(1)  # [N, B]
                weighted_x = (node_resp * x_flat).sum(dim=-1)  # [N]
                weighted_x2 = (node_resp * x_flat**2).sum(dim=-1)  # [N]

                new_mean_md = weighted_x / total_md.clamp(min=1e-15)
                new_var_md = weighted_x2 / total_md.clamp(min=1e-15) - new_mean_md**2
                new_std_md = torch.sqrt(new_var_md.clamp(min=1e-5))

                probs = torch.linspace(0, 1, h + 1, device=self.means.device)
                z_quantiles = torch.erfinv(2 * probs - 1) * math.sqrt(2)

                new_lows = self.lows.clone()
                new_highs = self.highs.clone()

                m_expanded = new_mean_md.unsqueeze(-1).expand(N, h)
                s_expanded = new_std_md.unsqueeze(-1).expand(N, h)
                v_expanded_md = valid_md.unsqueeze(-1).expand(N, h)

                for i in range(N):
                    if is_split[i] and valid_md[i]:
                        for j in range(h):
                            new_lows[i, j] = new_mean_md[i] + new_std_md[i] * z_quantiles[j]
                            new_highs[i, j] = new_mean_md[i] + new_std_md[i] * z_quantiles[j + 1]

                mask_md = is_split.unsqueeze(-1).expand(N, h) & v_expanded_md
                self.means.data.copy_(torch.where(mask_md, m_expanded, self.means))
                self.stds.data.copy_(torch.where(mask_md, s_expanded, self.stds))
                self.lows.data.copy_(torch.where(mask_md, new_lows, self.lows))
                self.highs.data.copy_(torch.where(mask_md, new_highs, self.highs))

            if (~is_split).any():
                total_std = resp.sum(dim=-1)  # [N, h] — sum over B
                valid_std = total_std > 1e-5

                weighted_x_std = (resp * x).sum(dim=-1)  # [N, h]
                weighted_x2_std = (resp * x**2).sum(dim=-1)  # [N, h]

                new_mean_std = weighted_x_std / total_std.clamp(min=1e-15)
                new_var_std = weighted_x2_std / total_std.clamp(min=1e-15) - new_mean_std**2
                new_std_std = torch.sqrt(new_var_std.clamp(min=1e-5))

                mask_std = (~is_split).unsqueeze(-1).expand(N, h) & valid_std
                self.means.data.copy_(torch.where(mask_std, new_mean_std, self.means))
                self.stds.data.copy_(torch.where(mask_std, new_std_std, self.stds))


class TensorizedFusedLayer(ABC, nn.Module):
    weights: torch.Tensor
    log_coeff: torch.Tensor
    left_idx: torch.Tensor
    right_idx: torch.Tensor
    num_nodes: int
    h_out: int
    h_in: int

    gather_L: torch.Tensor
    gather_R: torch.Tensor
    scatter_idx: torch.Tensor

    def __init__(
        self,
        num_nodes: int,
        h_out: int,
        h_in: int,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.h_out = h_out
        self.h_in = h_in

        weights = torch.rand(num_nodes, h_out, h_in)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        self.log_w = nn.Parameter(torch.log(weights.clamp(min=1e-12)))
        self.register_buffer("log_coeff", torch.zeros(num_nodes, h_out))


class TensorizedSparseKronecker(TensorizedFusedLayer):
    """Block-diagonal Kronecker contraction in log-semiring.

    Reads children via precomputed flat gather indices. All computation in
    [N, h, B] layout (units-first) for coalesced memory access.
    """

    def __init__(
        self,
        num_nodes: int,
        h_out: int,
        h_in: int,
        h_left: int,
        h_right: int,
    ):
        super().__init__(
            num_nodes=num_nodes,
            h_out=h_out,
            h_in=h_in,
        )

        self.h_left = h_left
        self.h_right = h_right

    def forward(self, buf: torch.Tensor) -> torch.Tensor:
        # buf: [total_units, B]
        B = buf.shape[1]
        N = self.num_nodes

        L = buf[self.gather_L].reshape(N, self.h_left, B)  # [N, h_left, B]
        R = buf[self.gather_R].reshape(N, self.h_right, B)  # [N, h_right, B]

        # Log-domain outer product → [N, h_out, h_in, B]
        S = (L.unsqueeze(2) + R.unsqueeze(1)).reshape(N, self.h_out, self.h_in, B)

        m = S.max(dim=2, keepdim=True)[0]  # [N, h_out, 1, B]
        exp_S = torch.exp(S - m)  # [N, h_out, h_in, B]

        # Block-diagonal weight dot products via bmm: [N*h_out, 1, h_in] @ [N*h_out, h_in, B]
        exp_W = torch.exp(self.log_w).reshape(N * self.h_out, 1, self.h_in)
        exp_S_flat = exp_S.reshape(N * self.h_out, self.h_in, B)
        out = torch.bmm(exp_W, exp_S_flat).reshape(N, self.h_out, B)

        result = torch.log(out.clamp(min=1e-20)) + m.squeeze(2)  # [N, h_out, B]
        return result + self.log_coeff.unsqueeze(-1)  # apply per-unit coefficient


class TensorizedSparseHadamard(TensorizedFusedLayer):
    """Diagonal mixing contraction in log-semiring.

    Reads children via precomputed flat gather indices. All computation in
    [N, h, B] layout (units-first) for coalesced memory access.
    """

    def __init__(
        self,
        num_nodes: int,
        h_out: int,
        h_in: int,
        h_child: int,
    ):
        super().__init__(
            num_nodes=num_nodes,
            h_out=h_out,
            h_in=h_in,
        )

        self.h_child = h_child

    def forward(self, buf: torch.Tensor) -> torch.Tensor:
        # buf: [total_units, B]
        B = buf.shape[1]
        N = self.num_nodes

        # Diagonal pairing: same index k from each child, reshaped into [h_out, h_in] blocks
        L = buf[self.gather_L].reshape(N, self.h_out, self.h_in, B)
        R = buf[self.gather_R].reshape(N, self.h_out, self.h_in, B)

        S = L + R  # [N, h_out, h_in, B]

        m = S.max(dim=2, keepdim=True)[0]  # [N, h_out, 1, B]
        exp_S = torch.exp(S - m)  # [N, h_out, h_in, B]

        exp_W = torch.exp(self.log_w).reshape(N * self.h_out, 1, self.h_in)
        exp_S_flat = exp_S.reshape(N * self.h_out, self.h_in, B)
        out = torch.bmm(exp_W, exp_S_flat).reshape(N, self.h_out, B)

        result = torch.log(out.clamp(min=1e-20)) + m.squeeze(2)  # [N, h_out, B]
        return result + self.log_coeff.unsqueeze(-1)  # apply per-unit coefficient


class TensorizedCPT(TensorizedFusedLayer):
    """Dense Hadamard-product contraction in log-semiring (Candecomp-Transposed).

    Like SparseHadamard but with dense (non-partitioned) weights: each output
    unit sums over ALL h_child Hadamard product units.  Structurally identical
    to Tucker but the product is Hadamard (L[k]+R[k]) rather than Kronecker
    (L[i]+R[j] for all i,j).
    """

    def __init__(
        self,
        num_nodes: int,
        h_out: int,
        h_child: int,
    ):
        super().__init__(
            num_nodes=num_nodes,
            h_out=h_out,
            h_in=h_child,  # dense: all h_child product units
        )
        self.h_child = h_child

    def forward(self, buf: torch.Tensor) -> torch.Tensor:
        B = buf.shape[1]
        N = self.num_nodes

        # Hadamard product: diagonal pairing L[k] + R[k]
        L = buf[self.gather_L].reshape(N, self.h_child, B)  # [N, h_child, B]
        R = buf[self.gather_R].reshape(N, self.h_child, B)  # [N, h_child, B]
        S = L + R  # [N, h_child, B]

        m = S.max(dim=1, keepdim=True)[0]  # [N, 1, B]
        exp_S = torch.exp(S - m)  # [N, h_child, B]

        # Dense weight contraction: [N, h_out, h_child] @ [N, h_child, B]
        exp_W = torch.exp(self.log_w)
        out = torch.bmm(exp_W, exp_S)  # [N, h_out, B]

        result = torch.log(out.clamp(min=1e-20)) + m  # [N, h_out, B]
        return result + self.log_coeff.unsqueeze(-1)


class TensorizedTucker(TensorizedFusedLayer):
    """Dense Kronecker-product contraction in log-semiring (Universal Layer).

    Reads children via precomputed flat gather indices. All computation in
    [N, h, B] layout (units-first) for coalesced memory access.
    """

    def __init__(
        self,
        num_nodes: int,
        h_out: int,
        h_left: int,
        h_right: int,
        weights: torch.Tensor = None,
    ):
        super().__init__(
            num_nodes=num_nodes,
            h_out=h_out,
            h_in=h_left * h_right,
            weights=weights,
        )

        self.h_left = h_left
        self.h_right = h_right
        self.h_child = h_left * h_right

    def forward(self, buf: torch.Tensor) -> torch.Tensor:
        # buf: [total_units, B]
        B = buf.shape[1]
        N = self.num_nodes

        L = buf[self.gather_L].reshape(N, self.h_left, B)  # [N, h_left, B]
        R = buf[self.gather_R].reshape(N, self.h_right, B)  # [N, h_right, B]

        # Log-domain outer product → [N, h_child, B]
        S = (L.unsqueeze(2) + R.unsqueeze(1)).reshape(N, self.h_child, B)

        m = S.max(dim=1, keepdim=True)[0]  # [N, 1, B]
        exp_S = torch.exp(S - m)  # [N, h_child, B]

        # Dense weight contraction: [N, h_out, h_child] @ [N, h_child, B] → [N, h_out, B]
        exp_W = torch.exp(self.log_w)
        out = torch.bmm(exp_W, exp_S)  # [N, h_out, B]

        result = torch.log(out.clamp(min=1e-20)) + m  # [N, h_out, B]
        return result + self.log_coeff.unsqueeze(-1)  # apply per-unit coefficient


class CompiledCircuit(nn.Module):
    """Base class for compiled circuits. Subclasses only override layer creation."""

    def __init__(self):
        super().__init__()
        self.input_layers = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.max_var_index = 0
        # Per-node variable-scope sets, populated by subclass __init__.
        self.node_scopes: List[set] = []
        # Per-node marginal determinism sets (BitSet or None), populated by subclass __init__.
        self.node_md_sets: list = []

        # Flat buffer tracking — populated by _register_input_flat / _register_layer_flat.
        self._node_flat_offsets: List[int] = []  # flat unit start for each logical node
        self._total_units: int = 0

    # ------------------------------------------------------------------
    # Flat buffer helpers — called once per layer during __init__
    # ------------------------------------------------------------------

    def _register_input_flat(self, layer: nn.Module, num_nodes: int, h_out: int) -> None:
        """Register scatter index on an input layer and update flat buffer tracking."""
        write_start = self._total_units
        scatter = (
            torch.arange(num_nodes * h_out, dtype=torch.long) + write_start
        )  # [N*h_out] — contiguous block
        layer.register_buffer("scatter_idx", scatter)
        for i in range(num_nodes):
            self._node_flat_offsets.append(write_start + i * h_out)
        self._total_units += num_nodes * h_out

    def _register_layer_flat(
        self,
        layer: nn.Module,
        left_idx: torch.Tensor,
        right_idx: torch.Tensor,
        num_nodes: int,
        h_left: int,
        h_right: int,
        h_out: int,
    ) -> None:
        """Register gather/scatter indices on an internal layer and update flat buffer tracking."""
        write_start = self._total_units
        offsets = torch.tensor(self._node_flat_offsets, dtype=torch.long)

        # Gather indices: for each child node, read h contiguous units starting at its offset.
        lo = offsets[left_idx]  # [N] — flat start of each left child
        ro = offsets[right_idx]  # [N] — flat start of each right child

        gather_L = (lo.unsqueeze(1) + torch.arange(h_left)).reshape(-1)  # [N*h_left]
        gather_R = (ro.unsqueeze(1) + torch.arange(h_right)).reshape(-1)  # [N*h_right]

        scatter = torch.arange(num_nodes * h_out, dtype=torch.long) + write_start  # [N*h_out]

        layer.register_buffer("gather_L", gather_L)
        layer.register_buffer("gather_R", gather_R)
        layer.register_buffer("scatter_idx", scatter)

        for i in range(num_nodes):
            self._node_flat_offsets.append(write_start + i * h_out)
        self._total_units += num_nodes * h_out

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _compile_input(self, fnode: FusedInputLayer) -> nn.Module:
        fl = fnode.folded_layer
        if isinstance(fl, FoldedGaussianInputLayer):
            scopes = fnode.scopes if fnode.scopes else list(range(fl.num_nodes))
            self.max_var_index = max(self.max_var_index, max(scopes) if scopes else 0)
            leaf_modes = getattr(fl, "leaf_modes", None)
            return TensorizedGaussianInput(
                fl.means.clone(),
                fl.stddevs.clone(),
                fl.lows.clone(),
                fl.highs.clone(),
                scopes,
                leaf_modes=leaf_modes.clone() if leaf_modes is not None else None,
            )
        raise NotImplementedError(f"Input layer type {type(fl).__name__} not yet compiled")

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = self._pad(data)
        B = data.shape[0]

        buf = torch.full((self._total_units, B), -1e20, device=data.device, dtype=data.dtype)

        for il in self.input_layers:
            out = il(data)  # [N, h, B]
            buf[il.scatter_idx] = out.reshape(-1, B)

        for layer in self.layers:
            out = layer(buf)  # [N, h_out, B]
            buf[layer.scatter_idx] = out.reshape(-1, B)

        return buf[self._node_flat_offsets[-1]]  # [B]

    def forward_buffer(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns the full flat buffer (per-unit values at every node).

        Used by circuit-algebraic operations (e.g., conditional) that need
        intermediate per-unit values for normalization corrections.
        """
        data = self._pad(data)
        B = data.shape[0]

        buf = torch.full((self._total_units, B), -1e20, device=data.device, dtype=data.dtype)

        for il in self.input_layers:
            out = il(data)  # [N, h, B]
            buf[il.scatter_idx] = out.reshape(-1, B)

        for layer in self.layers:
            out = layer(buf)  # [N, h_out, B]
            buf[layer.scatter_idx] = out.reshape(-1, B)

        return buf

    def _pad(self, data: torch.Tensor) -> torch.Tensor:
        if data.shape[1] <= self.max_var_index:
            pad = torch.full(
                (data.shape[0], self.max_var_index + 1 - data.shape[1]),
                float("nan"),
                device=data.device,
                dtype=data.dtype,
            )
            return torch.cat([data, pad], dim=1)
        return data

    def em_step(self, data: torch.Tensor, step_size: float = 1.0, smoothing: float = 1e-6) -> float:
        """Executes one online/mini-batch EM step.

        Args:
            data: batch of observations.
            step_size: blend factor (1.0 = full replacement, <1.0 = online EM).
            smoothing: Laplace smoothing to prevent dead paths.
        """
        self.train()
        log_probs = self.forward(data)
        mean_ll = log_probs.mean().item()

        loss = log_probs.sum()
        loss.backward()

        for il in self.input_layers:
            if hasattr(il, "_leaf_output") and il._leaf_output.grad is not None:
                il.update_params(il._leaf_output.grad)

        for layer in self.layers:
            if not hasattr(layer, "log_w") or layer.log_w.grad is None:
                continue

            grad_clean = layer.log_w.grad.nan_to_num(0.0).clamp(min=0.0)
            batch_counts = grad_clean + smoothing
            batch_probs = batch_counts / batch_counts.sum(dim=-1, keepdim=True)

            current_probs = torch.exp(layer.log_w.data)
            new_probs = (1.0 - step_size) * current_probs + step_size * batch_probs
            layer.log_w.data.copy_(torch.log(new_probs.clamp(min=1e-12)))
            layer.log_w.grad.zero_()

        self.zero_grad()
        return mean_ll


class TensorizedCircuit(CompiledCircuit):
    """Dense compiled circuit from a FusedCircuit."""

    def __init__(self, fused: FusedCircuit):
        super().__init__()

        for fid in fused.topological_sort():
            fnode = fused.get_node_data(fid)

            if isinstance(fnode, FusedInputLayer):
                layer = self._compile_input(fnode)
                self.input_layers.append(layer)

                scopes_list = fnode.scopes if fnode.scopes else list(range(layer.num_nodes))
                fnode_md_sets = getattr(fnode, "md_sets", None)
                for i, var_idx in enumerate(scopes_list):
                    self.node_scopes.append({var_idx})
                    self.node_md_sets.append(fnode_md_sets[i] if fnode_md_sets else None)

                self._register_input_flat(layer, layer.num_nodes, layer.h_out)

            elif isinstance(fnode, SparseKroneckerLayer):
                layer = TensorizedSparseKronecker(
                    fnode.num_nodes,
                    fnode.h_out,
                    fnode.h_in,
                    fnode.h_left,
                    fnode.h_right,
                )
                self.layers.append(layer)

                fnode_md_sets = getattr(fnode, "md_sets", None)
                for i in range(fnode.num_nodes):
                    l_idx = fnode.left_indices[i].item()
                    r_idx = fnode.right_indices[i].item()
                    self.node_scopes.append(self.node_scopes[l_idx] | self.node_scopes[r_idx])
                    self.node_md_sets.append(fnode_md_sets[i] if fnode_md_sets else None)

                self._register_layer_flat(
                    layer,
                    fnode.left_indices,
                    fnode.right_indices,
                    fnode.num_nodes,
                    fnode.h_left,
                    fnode.h_right,
                    fnode.h_out,
                )

            elif isinstance(fnode, SparseHadamardLayer):
                layer = TensorizedSparseHadamard(
                    fnode.num_nodes,
                    fnode.h_out,
                    fnode.h_in,
                    fnode.h_child,
                )
                self.layers.append(layer)

                fnode_md_sets = getattr(fnode, "md_sets", None)
                for i in range(fnode.num_nodes):
                    l_idx = fnode.left_indices[i].item()
                    r_idx = fnode.right_indices[i].item()
                    self.node_scopes.append(self.node_scopes[l_idx] | self.node_scopes[r_idx])
                    self.node_md_sets.append(fnode_md_sets[i] if fnode_md_sets else None)

                self._register_layer_flat(
                    layer,
                    fnode.left_indices,
                    fnode.right_indices,
                    fnode.num_nodes,
                    fnode.h_child,
                    fnode.h_child,
                    fnode.h_out,
                )

            elif isinstance(fnode, CPTLayer):
                layer = TensorizedCPT(
                    fnode.num_nodes,
                    fnode.h_out,
                    fnode.h_child,
                    weights=getattr(fnode, "weights", None),
                )
                self.layers.append(layer)

                fnode_md_sets = getattr(fnode, "md_sets", None)
                for i in range(fnode.num_nodes):
                    l_idx = fnode.left_indices[i].item()
                    r_idx = fnode.right_indices[i].item()
                    self.node_scopes.append(self.node_scopes[l_idx] | self.node_scopes[r_idx])
                    self.node_md_sets.append(fnode_md_sets[i] if fnode_md_sets else None)

                self._register_layer_flat(
                    layer,
                    fnode.left_indices,
                    fnode.right_indices,
                    fnode.num_nodes,
                    fnode.h_child,
                    fnode.h_child,
                    fnode.h_out,
                )

            elif isinstance(fnode, TuckerLayer):
                layer = TensorizedTucker(
                    fnode.num_nodes,
                    fnode.h_out,
                    fnode.h_left,
                    fnode.h_right,
                )
                self.layers.append(layer)

                fnode_md_sets = getattr(fnode, "md_sets", None)
                for i in range(fnode.num_nodes):
                    l_idx = fnode.left_indices[i].item()
                    r_idx = fnode.right_indices[i].item()
                    self.node_scopes.append(self.node_scopes[l_idx] | self.node_scopes[r_idx])
                    self.node_md_sets.append(fnode_md_sets[i] if fnode_md_sets else None)

                self._register_layer_flat(
                    layer,
                    fnode.left_indices,
                    fnode.right_indices,
                    fnode.num_nodes,
                    fnode.h_left,
                    fnode.h_right,
                    fnode.h_out,
                )

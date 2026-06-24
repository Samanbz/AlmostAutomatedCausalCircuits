import math
from abc import ABC
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn

from src.compilation.fused_circuit import (
    CPTLayer,
    FusedCircuit,
    FusedInputLayer,
    SparseHadamardLayer,
    SparseKroneckerLayer,
    TuckerLayer,
)


class TensorizedGaussianInput(nn.Module):
    """Stateless truncated Gaussian input layer in log-space."""

    def __init__(self, scopes: List[int], num_nodes: int, h_out: int):
        super().__init__()
        self.scopes = scopes
        self.num_nodes = num_nodes
        self.h_out = h_out

    def forward(
        self,
        data: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        lows: torch.Tensor,
        highs: torch.Tensor,
        leaf_modes: torch.Tensor,
    ) -> torch.Tensor:
        # data: [B, D]
        x_raw = data[:, self.scopes]
        x = x_raw.T.unsqueeze(1)  # [num_nodes, 1, B]
        is_nan = torch.isnan(x)
        x_safe = torch.where(is_nan, torch.zeros_like(x), x)

        self._saved_x = x_safe.detach()
        self._nan_mask = is_nan.detach()

        m_exp = means.unsqueeze(-1)  # [num_nodes, h, 1]
        s_exp = stds.unsqueeze(-1)

        z = (x_safe - m_exp) / s_exp
        log_pdf = -0.5 * math.log(2 * math.pi) - 0.5 * z**2 - torch.log(s_exp)

        sqrt2 = math.sqrt(2)
        z_hi = (highs.unsqueeze(-1) - m_exp) / (s_exp * sqrt2)
        z_lo = (lows.unsqueeze(-1) - m_exp) / (s_exp * sqrt2)

        cdf_hi = 0.5 * (1 + torch.erf(z_hi))
        cdf_lo = 0.5 * (1 + torch.erf(z_lo))
        log_Z = torch.log((cdf_hi - cdf_lo).clamp(min=1e-12))
        log_pdf = log_pdf - log_Z

        # Apply multiplier to density part.
        # multiplier: 1.0 = Normal, -1.0 = Inverse, 0.0 = Indicator/Marginalized
        modes = leaf_modes.view(-1, 1, 1)
        log_pdf = modes * log_pdf

        # Apply OOB mask LAST. This turns mode 0.0 into a Selector/Indicator:
        # returns 0.0 if in-bounds (selector), -inf if out-of-bounds.
        oob = (x_safe < lows.unsqueeze(-1)) | (x_safe > highs.unsqueeze(-1))
        log_pdf = torch.where(oob, torch.full_like(log_pdf, float("-inf")), log_pdf)

        # Handle NaNs from input data (marginalization via data-masking)
        log_pdf = torch.where(is_nan, torch.zeros_like(log_pdf), log_pdf)

        if log_pdf.requires_grad:
            log_pdf.retain_grad()
        self._leaf_output = log_pdf
        return log_pdf
    def update_params(
        self,
        responsibilities: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        lows: torch.Tensor,
        highs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x = self._saved_x
            nan_mask = self._nan_mask
            N, h = means.shape
            resp = responsibilities.clamp(min=0).masked_fill(
                nan_mask.expand_as(responsibilities), 0
            )
            is_split = ~((lows[:, 0] == float("-inf")) & (highs[:, 0] == float("inf")))
            new_m, new_s, new_l, new_h = means.clone(), stds.clone(), lows.clone(), highs.clone()

            if is_split.any():
                node_resp = resp.sum(dim=1).masked_fill(nan_mask.squeeze(1), 0)
                total_md = node_resp.sum(dim=-1)
                valid_md = total_md > 1e-5
                x_flat = x.squeeze(1)
                wx_md = (node_resp * x_flat).sum(dim=-1)
                wx2_md = (node_resp * x_flat**2).sum(dim=-1)
                m_md = wx_md / total_md.clamp(min=1e-15)
                v_md = wx2_md / total_md.clamp(min=1e-15) - m_md**2
                s_md = torch.sqrt(v_md.clamp(min=1e-5))

                # Apply MD updates only to split nodes
                md_mask = is_split & valid_md
                new_m[md_mask] = m_md[md_mask].unsqueeze(-1)
                new_s[md_mask] = s_md[md_mask].unsqueeze(-1)

                probs = torch.linspace(0, 1, h + 1, device=means.device)
                zq = torch.erfinv(2 * probs - 1) * math.sqrt(2)

                # Update boundaries for split nodes
                md_indices = torch.where(md_mask)[0]
                for i in md_indices:
                    for j in range(h):
                        new_l[i, j] = m_md[i] + s_md[i] * zq[j]
                        new_h[i, j] = m_md[i] + s_md[i] * zq[j + 1]

            if (~is_split).any():
                total_s = resp.sum(dim=-1)
                valid_s = total_s > 1e-5
                wx_s = (resp * x).sum(dim=-1)
                wx2_s = (resp * x**2).sum(dim=-1)
                m_s = wx_s / total_s.clamp(min=1e-15)
                v_s = wx2_s / total_s.clamp(min=1e-15) - m_s**2
                s_s = torch.sqrt(v_s.clamp(min=1e-5))

                # Update means/stds for non-split nodes
                s_mask = (~is_split).unsqueeze(-1).expand(N, h) & valid_s
                new_m = torch.where(s_mask, m_s, new_m)
                new_s = torch.where(s_mask, s_s, new_s)
            return new_m, new_s, new_l, new_h


class TensorizedFusedLayer(ABC, nn.Module):
    def __init__(self, num_nodes: int, h_out: int, h_in: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.h_out = h_out
        self.h_in = h_in


class TensorizedSparseKronecker(TensorizedFusedLayer):
    def forward(self, buf, log_w, gather_L, gather_R):
        B = buf.shape[1]
        N = self.num_nodes
        h_o = log_w.shape[1]
        h_i = log_w.shape[2]
        h_l = gather_L.numel() // N
        h_r = gather_R.numel() // N
        L = buf[gather_L].reshape(N, h_l, B)
        R = buf[gather_R].reshape(N, h_r, B)
        S = (L.unsqueeze(2) + R.unsqueeze(1)).reshape(N, h_o, h_i, B)
        m = S.max(dim=2, keepdim=True)[0]
        m_safe = m.masked_fill(m == float("-inf"), 0.0)
        exp_S = torch.exp(S - m_safe)
        exp_W = torch.exp(log_w).reshape(N * h_o, 1, h_i)
        out = torch.bmm(exp_W, exp_S.reshape(N * h_o, h_i, B)).reshape(N, h_o, B)
        return torch.log(out.clamp(min=1e-20)) + m.squeeze(2)


class TensorizedSparseHadamard(TensorizedFusedLayer):
    def forward(self, buf, log_w, gather_L, gather_R):
        B = buf.shape[1]
        N = self.num_nodes
        h_o = log_w.shape[1]
        h_i = log_w.shape[2]
        L = buf[gather_L].reshape(N, h_o, h_i, B)
        R = buf[gather_R].reshape(N, h_o, h_i, B)
        S = L + R
        m = S.max(dim=2, keepdim=True)[0]
        m_safe = m.masked_fill(m == float("-inf"), 0.0)
        exp_S = torch.exp(S - m_safe)
        exp_W = torch.exp(log_w).reshape(N * h_o, 1, h_i)
        out = torch.bmm(exp_W, exp_S.reshape(N * h_o, h_i, B)).reshape(N, h_o, B)
        return torch.log(out.clamp(min=1e-20)) + m.squeeze(2)


class TensorizedCPT(TensorizedFusedLayer):
    def forward(self, buf, log_w, gather_L, gather_R):
        B = buf.shape[1]
        N = self.num_nodes
        h_c = gather_L.numel() // N
        L = buf[gather_L].reshape(N, h_c, B)
        R = buf[gather_R].reshape(N, h_c, B)
        S = L + R
        m = S.max(dim=1, keepdim=True)[0]
        m_safe = m.masked_fill(m == float("-inf"), 0.0)
        exp_S = torch.exp(S - m_safe)
        out = torch.bmm(torch.exp(log_w), exp_S)
        return torch.log(out.clamp(min=1e-20)) + m


class TensorizedTucker(TensorizedFusedLayer):
    def forward(self, buf, log_w, gather_L, gather_R):
        B = buf.shape[1]
        N = self.num_nodes
        h_l = gather_L.numel() // N
        h_r = gather_R.numel() // N
        L = buf[gather_L].reshape(N, h_l, B)
        R = buf[gather_R].reshape(N, h_r, B)
        S = (L.unsqueeze(2) + R.unsqueeze(1)).reshape(N, h_l * h_r, B)
        m = S.max(dim=1, keepdim=True)[0]
        m_safe = m.masked_fill(m == float("-inf"), 0.0)
        exp_S = torch.exp(S - m_safe)
        out = torch.bmm(torch.exp(log_w), exp_S)
        return torch.log(out.clamp(min=1e-20)) + m


@dataclass
class InputLayerState:
    modes: torch.Tensor  # [num_nodes] multiplier tensor
    scopes: torch.Tensor  # [num_nodes]
    means: torch.Tensor  # [num_nodes, h]
    stds: torch.Tensor  # [num_nodes, h]
    lows: torch.Tensor  # [num_nodes, h]
    highs: torch.Tensor  # [num_nodes, h]
    scatter: torch.Tensor  # [num_nodes * h_out]


@dataclass
class SumLayerState:
    weights: torch.Tensor  # [N, h_out, h_in]
    gather_L: torch.Tensor  # [N * h_in]
    gather_R: torch.Tensor  # [N * h_in]
    scatter: torch.Tensor  # [N * h_out]


@dataclass
class CircuitState:
    input_states: List[InputLayerState]
    sum_states: List[SumLayerState]


class CompiledCircuit(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layers = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.params = nn.ParameterDict()
        self.max_var_index = 0
        self._node_flat_offsets = []
        self._layer_start_offsets = []
        self._layer_h_out = []
        self._total_units = 0

    def _register_input_flat(self, layer_idx, num_nodes, h_out):
        self._layer_start_offsets.append(self._total_units)
        self._layer_h_out.append(h_out)
        scatter = torch.arange(num_nodes * h_out, dtype=torch.long) + self._total_units
        self.register_buffer(f"input_{layer_idx}_scatter", scatter)
        for i in range(num_nodes):
            self._node_flat_offsets.append(self._total_units + i * h_out)
        self._total_units += num_nodes * h_out

    def _register_layer_flat(self, layer_idx, left_idx, right_idx, num_nodes, h_l, h_r, h_out):
        self._layer_start_offsets.append(self._total_units)
        self._layer_h_out.append(h_out)
        offsets = torch.tensor(self._node_flat_offsets, dtype=torch.long)
        lo = offsets[left_idx]
        ro = offsets[right_idx]
        gather_L = (lo.unsqueeze(1) + torch.arange(h_l)).reshape(-1)
        gather_R = (ro.unsqueeze(1) + torch.arange(h_r)).reshape(-1)
        scatter = torch.arange(num_nodes * h_out, dtype=torch.long) + self._total_units
        self.register_buffer(f"layer_{layer_idx}_gather_L", gather_L)
        self.register_buffer(f"layer_{layer_idx}_gather_R", gather_R)
        self.register_buffer(f"layer_{layer_idx}_scatter", scatter)
        for i in range(num_nodes):
            self._node_flat_offsets.append(self._total_units + i * h_out)
        self._total_units += num_nodes * h_out

    def forward(self, data, state: CircuitState = None):
        B = data.shape[0]
        device = next(self.parameters()).device
        data = data.to(device)
        buf = torch.full((self._total_units, B), -1e20, device=device, dtype=data.dtype)

        if state is None:
            for i, il in enumerate(self.input_layers):
                m = self.params[f"input_{i}_means"]
                s = self.params[f"input_{i}_stds"]
                l = getattr(self, f"input_{i}_lows")
                h = getattr(self, f"input_{i}_highs")
                sc = getattr(self, f"input_{i}_scatter")
                modes = getattr(self, f"input_{i}_modes")
                buf[sc] = il(data, m, s, l, h, modes).reshape(-1, B)

            for i, layer in enumerate(self.layers):
                w = self.params[f"layer_{i}_weights"]
                gL = getattr(self, f"layer_{i}_gather_L")
                gR = getattr(self, f"layer_{i}_gather_R")
                sc = getattr(self, f"layer_{i}_scatter")
                buf[sc] = layer(buf, w, gL, gR).reshape(-1, B)
        else:
            for i, il in enumerate(self.input_layers):
                s = state.input_states[i]
                buf[s.scatter] = il(data, s.means, s.stds, s.lows, s.highs, s.modes).reshape(-1, B)

            for i, layer in enumerate(self.layers):
                s = state.sum_states[i]
                buf[s.scatter] = layer(buf, s.weights, s.gather_L, s.gather_R).reshape(-1, B)

        return buf[self._node_flat_offsets[-1]]

    def em_step(self, data, step_size=1.0, smoothing=1e-6):
        self.train()
        device = next(self.parameters()).device
        data = data.to(device)
        log_probs = self.forward(data)
        loss = log_probs.sum()
        loss.backward()

        for i, il in enumerate(self.input_layers):
            if hasattr(il, "_leaf_output") and il._leaf_output.grad is not None:
                m, s = self.params[f"input_{i}_means"], self.params[f"input_{i}_stds"]
                l, h = getattr(self, f"input_{i}_lows"), getattr(self, f"input_{i}_highs")
                nm, ns, nl, nh = il.update_params(il._leaf_output.grad, m, s, l, h)

                self.params[f"input_{i}_means"].data.copy_(nm)
                self.params[f"input_{i}_stds"].data.copy_(ns)
                getattr(self, f"input_{i}_lows").copy_(nl)
                getattr(self, f"input_{i}_highs").copy_(nh)
        for i in range(len(self.layers)):
            w = self.params[f"layer_{i}_weights"]
            if w.grad is not None:
                gc = w.grad.nan_to_num(0.0).clamp(min=0.0)
                bc = gc + smoothing
                bp = bc / bc.sum(dim=-1, keepdim=True)
                nw = (1.0 - step_size) * torch.exp(w.data) + step_size * bp
                w.data.copy_(torch.log(nw.clamp(min=1e-12)))
                w.grad.zero_()
        self.zero_grad()
        return log_probs.mean().item()


class TensorizedCircuit(CompiledCircuit):
    def __init__(self, fused: FusedCircuit):
        super().__init__()
        for fid in fused.topological_sort():
            fn = fused.get_node_data(fid)
            if isinstance(fn, FusedInputLayer):
                l_idx = len(self.input_layers)
                fl = fn.folded_layer
                sc = fn.scopes if fn.scopes else list(range(fl.num_nodes))
                self.max_var_index = max(self.max_var_index, max(sc) if sc else 0)
                il = TensorizedGaussianInput(sc, fl.num_nodes, fl.h_out)
                self.input_layers.append(il)
                self.params[f"input_{l_idx}_means"] = nn.Parameter(fl.means.clone())
                self.params[f"input_{l_idx}_stds"] = nn.Parameter(fl.stddevs.clone())
                self.register_buffer(f"input_{l_idx}_lows", fl.lows.clone())
                self.register_buffer(f"input_{l_idx}_highs", fl.highs.clone())
                mo = getattr(fl, "leaf_modes", torch.ones(fl.num_nodes, dtype=torch.float32))
                self.register_buffer(f"input_{l_idx}_modes", mo.clone().float())
                self._register_input_flat(l_idx, fl.num_nodes, fl.h_out)
            else:
                l_idx = len(self.layers)
                if isinstance(fn, SparseKroneckerLayer):
                    l = TensorizedSparseKronecker(fn.num_nodes, fn.h_out, fn.h_in)
                    hl, hr = fn.h_left, fn.h_right
                elif isinstance(fn, SparseHadamardLayer):
                    l = TensorizedSparseHadamard(fn.num_nodes, fn.h_out, fn.h_in)
                    hl, hr = fn.h_child, fn.h_child
                elif isinstance(fn, CPTLayer):
                    l = TensorizedCPT(fn.num_nodes, fn.h_out, fn.h_child)
                    hl, hr = fn.h_child, fn.h_child
                elif isinstance(fn, TuckerLayer):
                    l = TensorizedTucker(fn.num_nodes, fn.h_out, fn.h_left * fn.h_right)
                    hl, hr = fn.h_left, fn.h_right
                else:
                    raise ValueError
                self.layers.append(l)
                if hasattr(fn, "log_weights") and fn.log_weights is not None:
                    w = torch.log(fn.log_weights.clamp(min=1e-12))
                else:
                    w = torch.rand(fn.num_nodes, fn.h_out, l.h_in)
                    w = w / w.sum(dim=-1, keepdim=True)
                    w = torch.log(w.clamp(min=1e-12))
                self.params[f"layer_{l_idx}_weights"] = nn.Parameter(w)
                self._register_layer_flat(
                    l_idx, fn.left_indices, fn.right_indices, fn.num_nodes, hl, hr, fn.h_out
                )

    def get_base_state(self) -> CircuitState:
        """Return a structured state cloned from the circuit's current parameters."""
        input_states = []
        for i, il in enumerate(self.input_layers):
            input_states.append(
                InputLayerState(
                    modes=getattr(self, f"input_{i}_modes").clone(),
                    scopes=torch.tensor(il.scopes, dtype=torch.long),
                    means=self.params[f"input_{i}_means"].clone(),
                    stds=self.params[f"input_{i}_stds"].clone(),
                    lows=getattr(self, f"input_{i}_lows").clone(),
                    highs=getattr(self, f"input_{i}_highs").clone(),
                    scatter=getattr(self, f"input_{i}_scatter").clone(),
                )
            )

        sum_states = []
        for i, l in enumerate(self.layers):
            sum_states.append(
                SumLayerState(
                    weights=self.params[f"layer_{i}_weights"].clone(),
                    gather_L=getattr(self, f"layer_{i}_gather_L").clone(),
                    gather_R=getattr(self, f"layer_{i}_gather_R").clone(),
                    scatter=getattr(self, f"layer_{i}_scatter").clone(),
                )
            )
        return CircuitState(input_states=input_states, sum_states=sum_states)

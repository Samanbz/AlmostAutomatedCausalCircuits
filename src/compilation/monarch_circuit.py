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
from src.compilation.tensorized_circuit import (
    CompiledCircuit,
    TensorizedSparseHadamard,
    TensorizedSparseKronecker,
)
from src.utils.monarch import MonarchMatrix


class MonarchTucker(nn.Module):
    """Dense Kronecker-product contraction in log-semiring (Monarch Factorized).

    Reads children via precomputed flat gather indices. All computation in
    [N, h, B] layout (units-first) for coalesced memory access.
    """

    def __init__(
        self,
        left_indices: torch.Tensor,
        right_indices: torch.Tensor,
        num_nodes: int,
        h_out: int,
        h_left: int,
        h_right: int,
    ):
        super().__init__()
        self.register_buffer("left_idx", left_indices)
        self.register_buffer("right_idx", right_indices)

        self.num_nodes = num_nodes
        self.h_out = h_out
        self.h_left = h_left
        self.h_right = h_right
        self.h_child = h_left * h_right

        self.b, self.c, self.k, self.b1 = MonarchMatrix._auto_factorize(self.h_out, self.h_child)

        L = torch.rand(num_nodes, self.b, self.c, self.k)
        L_prob = L / L.sum(dim=-1, keepdim=True)
        R = torch.rand(num_nodes, self.k, self.b, self.b1)
        R_prob = R / R.sum(dim=-1, keepdim=True)

        self.log_L = nn.Parameter(torch.log(L_prob.clamp(min=1e-12)))
        self.log_R = nn.Parameter(torch.log(R_prob.clamp(min=1e-12)))

    def _monarch_contract(self, S: torch.Tensor) -> torch.Tensor:
        """Two-stage Monarch factored Tucker contraction.

        Args:
            S: [N, k, b1, B] — log-domain outer product of left/right children.

        Returns:
            [N, h_out, B]
        """
        N, k, b1, B = S.shape
        b = self.b
        c = self.c

        # Stage 1: contract S with log_R [N, k, b, b1] over b1.
        # Result U [N, k, b, B] = sum_{b1} R[N, k, b, b1] * exp(S - m1)
        m1 = S.max(dim=2, keepdim=True)[0]  # [N, k, 1, B]
        exp_S = torch.exp(S - m1)  # [N, k, b1, B]
        exp_R = torch.exp(self.log_R)  # [N, k, b, b1]

        # bmm: [N*k, b, b1] @ [N*k, b1, B] → [N*k, b, B]
        exp_R_flat = exp_R.reshape(N * k, b, b1)
        exp_S_flat = exp_S.reshape(N * k, b1, B)
        U_flat = torch.bmm(exp_R_flat, exp_S_flat)  # [N*k, b, B]
        U = torch.log(U_flat.clamp(min=1e-20)).reshape(N, k, b, B) + m1  # [N, k, b, B]

        # Stage 2: contract U with log_L [N, b, c, k] over k.
        # Result out [N, b, c, B] = sum_k L[N, b, c, k] * exp(U - m2)
        m2 = U.max(dim=1, keepdim=True)[0]  # [N, 1, b, B]
        exp_U = torch.exp(U - m2)  # [N, k, b, B]
        exp_L = torch.exp(self.log_L)  # [N, b, c, k]

        # Rearrange exp_U to [N, b, k, B] then bmm: [N*b, c, k] @ [N*b, k, B] → [N*b, c, B]
        exp_U_perm = exp_U.permute(0, 2, 1, 3).reshape(N * b, k, B)  # [N*b, k, B]
        exp_L_flat = exp_L.reshape(N * b, c, k)  # [N*b, c, k]
        out_flat = torch.bmm(exp_L_flat, exp_U_perm)  # [N*b, c, B]

        log_out = torch.log(out_flat.clamp(min=1e-20)).reshape(N, b, c, B)
        # m2: [N, 1, b, B] → permute to [N, b, 1, B] to broadcast with [N, b, c, B]
        out = log_out + m2.permute(0, 2, 1, 3)  # [N, b, c, B]
        return out.reshape(N, self.h_out, B)

    def forward_root_combined(self, buf_c: torch.Tensor, buf_m: torch.Tensor) -> torch.Tensor:
        """2-channel backdoor combination: causal + marginal channels.

        Args:
            buf_c: [total_units, B] — causal channel flat buffer.
            buf_m: [total_units, B] — marginal channel flat buffer.

        Returns:
            [N, h_out, B]
        """
        B = buf_c.shape[1]
        N = self.num_nodes

        L_c = buf_c[self.gather_L].reshape(N, self.h_left, B)
        R_c = buf_c[self.gather_R].reshape(N, self.h_right, B)
        S_c = (L_c.unsqueeze(2) + R_c.unsqueeze(1)).reshape(N, self.k, self.b1, B)

        L_m = buf_m[self.gather_L].reshape(N, self.h_left, B)
        R_m = buf_m[self.gather_R].reshape(N, self.h_right, B)
        S_m = (L_m.unsqueeze(2) + R_m.unsqueeze(1)).reshape(N, self.k, self.b1, B)

        return self._monarch_contract(S_c + S_m)

    def forward(
        self,
        buf: torch.Tensor,
        log_w: torch.Tensor,
        gather_L: torch.Tensor,
        gather_R: torch.Tensor,
        log_coeff: torch.Tensor,
    ) -> torch.Tensor:
        B = buf.shape[1]
        N = self.num_nodes

        L = buf[gather_L].reshape(N, self.h_left, B)  # [N, h_left, B]
        R = buf[gather_R].reshape(N, self.h_right, B)  # [N, h_right, B]

        # Log-domain outer product → [N, k, b1, B]
        S = (L.unsqueeze(2) + R.unsqueeze(1)).reshape(N, self.k, self.b1, B)
        return self._monarch_contract(S) + log_coeff.unsqueeze(-1)



class MonarchCPT(nn.Module):
    """Dense Hadamard-product contraction in log-semiring (Monarch Factorized CPT).

    Like MonarchTucker but the product is Hadamard (L[k]+R[k]) instead of
    Kronecker (L[i]+R[j]).  h_child = h (not h*h).
    """

    def __init__(
        self,
        num_nodes: int,
        h_out: int,
        h_child: int,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.h_out = h_out
        self.h_child = h_child

        self.b, self.c, self.k, self.b1 = MonarchMatrix._auto_factorize(self.h_out, self.h_child)

        L = torch.rand(num_nodes, self.b, self.c, self.k)
        L_prob = L / L.sum(dim=-1, keepdim=True)
        R = torch.rand(num_nodes, self.k, self.b, self.b1)
        R_prob = R / R.sum(dim=-1, keepdim=True)

        self.log_L = nn.Parameter(torch.log(L_prob.clamp(min=1e-12)))
        self.log_R = nn.Parameter(torch.log(R_prob.clamp(min=1e-12)))

    def _monarch_contract(self, S: torch.Tensor) -> torch.Tensor:
        N, k, b1, B = S.shape
        b = self.b
        c = self.c

        m1 = S.max(dim=2, keepdim=True)[0]
        exp_S = torch.exp(S - m1)
        exp_R = torch.exp(self.log_R)

        exp_R_flat = exp_R.reshape(N * k, b, b1)
        exp_S_flat = exp_S.reshape(N * k, b1, B)
        U_flat = torch.bmm(exp_R_flat, exp_S_flat)
        U = torch.log(U_flat.clamp(min=1e-20)).reshape(N, k, b, B) + m1

        m2 = U.max(dim=1, keepdim=True)[0]
        exp_U = torch.exp(U - m2)
        exp_L = torch.exp(self.log_L)

        exp_U_perm = exp_U.permute(0, 2, 1, 3).reshape(N * b, k, B)
        exp_L_flat = exp_L.reshape(N * b, c, k)
        out_flat = torch.bmm(exp_L_flat, exp_U_perm)

        log_out = torch.log(out_flat.clamp(min=1e-20)).reshape(N, b, c, B)
        out = log_out + m2.permute(0, 2, 1, 3)
        return out.reshape(N, self.h_out, B)

    def forward(
        self,
        buf: torch.Tensor,
        log_w: torch.Tensor,
        gather_L: torch.Tensor,
        gather_R: torch.Tensor,
        log_coeff: torch.Tensor,
    ) -> torch.Tensor:
        B = buf.shape[1]
        N = self.num_nodes

        # Hadamard product: diagonal pairing
        L = buf[gather_L].reshape(N, self.h_child, B)
        R = buf[gather_R].reshape(N, self.h_child, B)
        S = (L + R).reshape(N, self.k, self.b1, B)
        return self._monarch_contract(S) + log_coeff.unsqueeze(-1)



class MonarchCircuit(CompiledCircuit):
    def __init__(self, fused: FusedCircuit):
        super().__init__()

        for fid in fused.topological_sort():
            fnode = fused.get_node_data(fid)
            layer_idx = len(self.layers) if not isinstance(fnode, FusedInputLayer) else len(self.input_layers)

            if isinstance(fnode, FusedInputLayer):
                layer = self._compile_input(fnode)
                self.input_layers.append(layer)

                scopes_list = fnode.scopes if fnode.scopes else list(range(layer.num_nodes))
                for var_idx in scopes_list:
                    self.node_scopes.append({var_idx})

                # Need to register params for the input layer in central storage so _compile_input works properly
                fl = fnode.folded_layer
                self.params[f"input_{layer_idx}_means"] = nn.Parameter(fl.means.clone())
                self.params[f"input_{layer_idx}_stds"] = nn.Parameter(fl.stddevs.clone())
                self.register_buffer(f"input_{layer_idx}_lows", fl.lows.clone())
                self.register_buffer(f"input_{layer_idx}_highs", fl.highs.clone())
                modes = getattr(fl, "leaf_modes", torch.zeros(fl.num_nodes, dtype=torch.long))
                self.register_buffer(f"input_{layer_idx}_modes", modes.clone())

                self._register_input_flat(layer_idx, layer.num_nodes, layer.h_out)

            elif isinstance(fnode, SparseKroneckerLayer):
                layer = TensorizedSparseKronecker(
                    fnode.num_nodes,
                    fnode.h_out,
                    fnode.h_in,
                    fnode.h_left,
                    fnode.h_right,
                )
                self.layers.append(layer)

                weights = torch.rand(fnode.num_nodes, fnode.h_out, fnode.h_in)
                weights = weights / weights.sum(dim=-1, keepdim=True)
                self.params[f"layer_{layer_idx}_weights"] = nn.Parameter(torch.log(weights.clamp(min=1e-12)))
                self.register_buffer(f"layer_{layer_idx}_log_coeff", torch.zeros(fnode.num_nodes, fnode.h_out))

                for i in range(fnode.num_nodes):
                    l_idx = fnode.left_indices[i].item()
                    r_idx = fnode.right_indices[i].item()
                    self.node_scopes.append(self.node_scopes[l_idx] | self.node_scopes[r_idx])

                self._register_layer_flat(
                    layer_idx,
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

                weights = torch.rand(fnode.num_nodes, fnode.h_out, fnode.h_in)
                weights = weights / weights.sum(dim=-1, keepdim=True)
                self.params[f"layer_{layer_idx}_weights"] = nn.Parameter(torch.log(weights.clamp(min=1e-12)))
                self.register_buffer(f"layer_{layer_idx}_log_coeff", torch.zeros(fnode.num_nodes, fnode.h_out))

                for i in range(fnode.num_nodes):
                    l_idx = fnode.left_indices[i].item()
                    r_idx = fnode.right_indices[i].item()
                    self.node_scopes.append(self.node_scopes[l_idx] | self.node_scopes[r_idx])

                self._register_layer_flat(
                    layer_idx,
                    fnode.left_indices,
                    fnode.right_indices,
                    fnode.num_nodes,
                    fnode.h_child,
                    fnode.h_child,
                    fnode.h_out,
                )

            elif isinstance(fnode, CPTLayer):
                layer = MonarchCPT(
                    fnode.num_nodes,
                    fnode.h_out,
                    fnode.h_child,
                )
                self.layers.append(layer)

                # Dummy weights for forward compatibility
                self.params[f"layer_{layer_idx}_weights"] = nn.Parameter(torch.zeros(1))
                self.register_buffer(f"layer_{layer_idx}_log_coeff", torch.zeros(fnode.num_nodes, fnode.h_out))

                for i in range(fnode.num_nodes):
                    l_idx = fnode.left_indices[i].item()
                    r_idx = fnode.right_indices[i].item()
                    self.node_scopes.append(self.node_scopes[l_idx] | self.node_scopes[r_idx])

                self._register_layer_flat(
                    layer_idx,
                    fnode.left_indices,
                    fnode.right_indices,
                    fnode.num_nodes,
                    fnode.h_child,
                    fnode.h_child,
                    fnode.h_out,
                )

            elif isinstance(fnode, TuckerLayer):
                layer = MonarchTucker(
                    fnode.left_indices,
                    fnode.right_indices,
                    fnode.num_nodes,
                    fnode.h_out,
                    fnode.h_left,
                    fnode.h_right,
                )
                self.layers.append(layer)

                # Dummy weights for forward compatibility
                self.params[f"layer_{layer_idx}_weights"] = nn.Parameter(torch.zeros(1))
                self.register_buffer(f"layer_{layer_idx}_log_coeff", torch.zeros(fnode.num_nodes, fnode.h_out))

                for i in range(fnode.num_nodes):
                    l_idx = fnode.left_indices[i].item()
                    r_idx = fnode.right_indices[i].item()
                    self.node_scopes.append(self.node_scopes[l_idx] | self.node_scopes[r_idx])

                self._register_layer_flat(
                    layer_idx,
                    fnode.left_indices,
                    fnode.right_indices,
                    fnode.num_nodes,
                    fnode.h_left,
                    fnode.h_right,
                    fnode.h_out,
                )

    def em_step(self, data: torch.Tensor, step_size: float = 1.0, smoothing: float = 1e-6) -> float:
        """Batched EM for MonarchCircuit."""
        self.train()
        data = self._pad(data)
        log_prob = self.forward(data)
        mean_ll = log_prob.mean().item()

        loss = log_prob.sum()
        loss.backward()

        for i, il in enumerate(self.input_layers):
            if hasattr(il, "_leaf_output") and il._leaf_output.grad is not None:
                means = self.params[f"input_{i}_means"]
                stds = self.params[f"input_{i}_stds"]
                lows = getattr(self, f"input_{i}_lows")
                highs = getattr(self, f"input_{i}_highs")

                new_means, new_stds, new_lows, new_highs = il.update_params(
                    il._leaf_output.grad, means, stds, lows, highs
                )

                self.params[f"input_{i}_means"].data.copy_(new_means)
                self.params[f"input_{i}_stds"].data.copy_(new_stds)
                getattr(self, f"input_{i}_lows").copy_(new_lows)
                getattr(self, f"input_{i}_highs").copy_(new_highs)

        with torch.no_grad():
            for layer in self.layers:
                if isinstance(layer, (MonarchTucker, MonarchCPT)):
                    if layer.log_R.grad is not None:
                        grad_R = layer.log_R.grad.nan_to_num(0.0).clamp(min=0.0)
                        batch_counts_R = grad_R + smoothing
                        batch_probs_R = batch_counts_R / batch_counts_R.sum(dim=-1, keepdim=True)
                        curr_probs_R = torch.exp(layer.log_R.data)
                        new_probs_R = (1.0 - step_size) * curr_probs_R + step_size * batch_probs_R
                        layer.log_R.data.copy_(torch.log(new_probs_R.clamp(min=1e-20)))
                        layer.log_R.grad.zero_()

                    if layer.log_L.grad is not None:
                        grad_L = layer.log_L.grad.nan_to_num(0.0).clamp(min=0.0)
                        batch_counts_L = grad_L + smoothing
                        batch_probs_L = batch_counts_L / batch_counts_L.sum(dim=-1, keepdim=True)
                        curr_probs_L = torch.exp(layer.log_L.data)
                        new_probs_L = (1.0 - step_size) * curr_probs_L + step_size * batch_probs_L
                        layer.log_L.data.copy_(torch.log(new_probs_L.clamp(min=1e-20)))
                        layer.log_L.grad.zero_()
                else:
                    if not hasattr(layer, "log_w") or layer.log_w.grad is None:
                        continue
                    grad_clean = layer.log_w.grad.nan_to_num(0.0).clamp(min=0.0)
                    batch_counts = grad_clean + smoothing
                    batch_probs = batch_counts / batch_counts.sum(dim=-1, keepdim=True)
                    current_probs = torch.exp(layer.log_w.data)
                    new_probs = (1.0 - step_size) * current_probs + step_size * batch_probs
                    layer.log_w.data.copy_(torch.log(new_probs.clamp(min=1e-20)))
                    layer.log_w.grad.zero_()

        self.zero_grad()
        return mean_ll

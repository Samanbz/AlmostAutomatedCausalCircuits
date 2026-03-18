import math
from typing import List

import torch
from torch import nn

from src.compilation.base_circuit import (
    GaussianInputLayer,
    ProductLayer,
    SafeLogSumExp,
    TensorizedLayer,
)
from src.compilation.monarch import MonarchMatrix
from src.compilation.padding import pad_to_uniform_depth
from src.compilation.tensorized_circuit import SumLayer, TensorizedCircuit
from src.symbolic import SymbolicArithmeticCircuit


def log_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batched matrix multiplication in log space using SafeLogSumExp.
    Computes log(exp(A) @ exp(B))
    A: (b, n, m)
    B: (b, m, p)
    returns: (b, n, p)
    """
    return SafeLogSumExp.apply(A.unsqueeze(-1) + B.unsqueeze(1), 2)


class MonarchSumLayer(TensorizedLayer):
    def __init__(
        self, node_ids: List[int], connections: torch.Tensor, respect_sparsity: bool = True
    ):
        super().__init__(node_ids)
        self.node_ids = node_ids

        out_dim, in_dim = connections.shape
        b, c, k, b1 = None, None, None, None
        if out_dim == in_dim and math.isqrt(in_dim) ** 2 == in_dim:
            m = math.isqrt(in_dim)
            b, c, k, b1 = m, m, m, m
        else:

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

        M = MonarchMatrix.from_dense(
            connections.clone(), b=b, c=c, k=k, b1=b1, respect_sparsity=respect_sparsity
        )
        self.b = b
        self.c = c
        self.k = k
        self.b1 = b1
        self.in_dim = in_dim
        self.out_dim = out_dim

        L_val = torch.abs(M.L.detach())
        R_val = torch.abs(M.R.detach())

        R_val = R_val / R_val.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        L_val = L_val / L_val.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        self.log_L = nn.Parameter(torch.log(L_val + 1e-10), requires_grad=True)
        self.log_R = nn.Parameter(torch.log(R_val + 1e-10), requires_grad=True)

        self.register_buffer("connections_mask", connections == 0)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        batch_size = input_values.shape[0]

        log_x_reshaped = input_values.view(batch_size, self.k, self.b1)
        log_x_perm = log_x_reshaped.permute(1, 0, 2).contiguous()

        log_R_perm = self.log_R.transpose(1, 2)
        log_y = log_bmm(log_x_perm, log_R_perm)

        log_y_perm = log_y.permute(2, 1, 0).contiguous()

        log_L_perm = self.log_L.transpose(1, 2)
        log_z = log_bmm(log_y_perm, log_L_perm)

        out = log_z.permute(1, 2, 0).contiguous().view(batch_size, self.out_dim)

        return out

    def update_params(self):
        with torch.no_grad():
            self._update_factor(self.log_R)
            self._update_factor(self.log_L)

    def _update_factor(self, param: nn.Parameter):
        responsibilities = param.grad
        if responsibilities is None:
            return

        total_resp = responsibilities.sum(dim=-1, keepdim=True)
        mask = total_resp > 0

        smoothed_resp = responsibilities + 1e-5
        new_probs = smoothed_resp / smoothed_resp.sum(dim=-1, keepdim=True)

        current_probs = torch.exp(param)
        new_probs = torch.where(mask, new_probs, current_probs)

        param.copy_(torch.log(new_probs))
        param.grad.zero_()


class MonarchCircuit(TensorizedCircuit):
    def __init__(self, symbolic_circuit: SymbolicArithmeticCircuit, respect_sparsity: bool = True):
        nn.Module.__init__(self)
        symbolic_circuit = pad_to_uniform_depth(symbolic_circuit)
        self.layers = nn.ModuleList()

        node_layers = symbolic_circuit.layered_topological_sort(reverse=True)
        input_node_ids = next(node_layers)

        input_nodes = [symbolic_circuit.get_node_data(n_id) for n_id in input_node_ids]
        means = [n.mean for n in input_nodes]
        stds = [n.stddev for n in input_nodes]
        lows = [n.var_support.low for n in input_nodes]
        highs = [n.var_support.high for n in input_nodes]
        scopes = [n.var for n in input_nodes]

        input_layer = GaussianInputLayer(
            input_node_ids,
            torch.tensor(means, dtype=torch.float32),
            torch.tensor(stds, dtype=torch.float32),
            torch.tensor(lows, dtype=torch.float32),
            torch.tensor(highs, dtype=torch.float32),
            scopes,
        )

        self.layers.append(input_layer)

        for layer_node_ids in node_layers:
            is_sum_layer = all(symbolic_circuit.is_sum_node(node_id) for node_id in layer_node_ids)
            is_product_layer = all(
                symbolic_circuit.is_product_node(node_id) for node_id in layer_node_ids
            )
            assert is_sum_layer or is_product_layer, "Mixed layer types are not supported"

            if is_product_layer:
                prev_layer_node_ids = self.layers[-1].node_ids
                left_idx, right_idx = self._compute_product_layer_indices(
                    symbolic_circuit, layer_node_ids, prev_layer_node_ids
                )
                layer = ProductLayer(layer_node_ids, left_idx, right_idx)
                self.layers.append(layer)

            elif is_sum_layer:
                prev_layer_node_ids = self.layers[-1].node_ids
                connections = self._compute_sum_layer_connections(
                    symbolic_circuit, layer_node_ids, prev_layer_node_ids
                )
                try:
                    layer = MonarchSumLayer(
                        layer_node_ids, connections, respect_sparsity=respect_sparsity
                    )
                except Exception as e:
                    layer = SumLayer(layer_node_ids, connections)
                self.layers.append(layer)

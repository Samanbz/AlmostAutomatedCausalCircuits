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
from src.compilation.monarch import LogMonarchMatrix, MonarchMatrix
from src.compilation.tensorized_circuit import SumLayer, TensorizedCircuit
from src.symbolic import SymbolicArithmeticCircuit


class MonarchSumLayer(TensorizedLayer):
    def __init__(
        self,
        out_idx: List[int],
        in_idx: List[int],
        connections: torch.Tensor,
        respect_sparsity: bool = True,
    ):
        super().__init__(out_idx)
        self.in_idx = in_idx

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

        log_L = torch.log(L_val.clamp(min=1e-12))
        log_R = torch.log(R_val.clamp(min=1e-12))

        self.log_monarch = LogMonarchMatrix(b, c, k, b1, log_L, log_R)

        self.log_L = self.log_monarch.log_L
        self.log_R = self.log_monarch.log_R

        self.register_buffer("connections_mask", connections == 0)
        self.register_buffer("row_log_sums_cache", None)
        self._cache_normalization()

    def _cache_normalization(self):
        with torch.no_grad():
            zero_input = torch.zeros(
                (1, self.log_monarch.in_dim), dtype=self.log_L.dtype, device=self.log_L.device
            )
            sums = self.log_monarch(zero_input)
            self.row_log_sums_cache = torch.where(
                torch.isneginf(sums), torch.zeros_like(sums), sums
            )

    def forward(self, global_buffer: torch.Tensor) -> torch.Tensor:
        input_values = global_buffer[:, self.in_idx]

        # Fast forward pass in log space without instantiating the dense full matrix
        out = self.log_monarch(input_values)

        # Normalize the outputs
        out = out - self.row_log_sums_cache

        global_buffer[:, self.out_idx] = out
        return out

    def update_params(self):
        with torch.no_grad():
            self._update_factor(self.log_R)
            self._update_factor(self.log_L)
            self._cache_normalization()

    def _update_factor(self, param: nn.Parameter):
        responsibilities = param.grad
        if responsibilities is None:
            return

        current_probs = torch.exp(param)
        structural_zeros = current_probs == 0

        total_resp = responsibilities.sum(dim=-1, keepdim=True)
        mask = total_resp > 0

        smoothed_resp = responsibilities + 1e-5
        # Do not give mass to structural zeros
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
    def __init__(self, symbolic_circuit: SymbolicArithmeticCircuit, respect_sparsity: bool = True):
        nn.Module.__init__(self)

        self.layers = nn.ModuleList()

        self.node_to_idx = {}
        idx = 0
        node_layers_iter = symbolic_circuit.layered_topological_sort(reverse=True)

        try:
            input_node_ids = next(node_layers_iter)
        except StopIteration:
            self.num_nodes = 0
            return

        self.input_layers = nn.ModuleList()
        gaussian_nodes, gaussian_ids = [], []
        uniform_nodes, uniform_ids = [], []
        categorical_nodes, categorical_ids = [], []

        for n_id in input_node_ids:
            node = symbolic_circuit.get_node_data(n_id)
            if hasattr(node, "mean") and hasattr(node, "stddev"):
                gaussian_nodes.append(node)
                gaussian_ids.append(n_id)
            elif hasattr(node, "var_support") and not hasattr(node, "categories") and not hasattr(node, "mean"):
                uniform_nodes.append(node)
                uniform_ids.append(n_id)
            elif hasattr(node, "categories") and hasattr(node, "probabilities"):
                categorical_nodes.append(node)
                categorical_ids.append(n_id)
            else:
                raise ValueError(f"Unsupported distribution node: {node}")

        if gaussian_nodes:
            means = [n.mean for n in gaussian_nodes]
            stds = [n.stddev for n in gaussian_nodes]
            lows = [n.var_support.low for n in gaussian_nodes]
            highs = [n.var_support.high for n in gaussian_nodes]
            scopes = [n.var for n in gaussian_nodes]
            out_idx = []
            for n_id in gaussian_ids:
                self.node_to_idx[n_id] = idx
                out_idx.append(idx)
                idx += 1
            self.input_layers.append(GaussianInputLayer(
                out_idx,
                torch.tensor(means, dtype=torch.float32),
                torch.tensor(stds, dtype=torch.float32),
                torch.tensor(lows, dtype=torch.float32),
                torch.tensor(highs, dtype=torch.float32),
                scopes,
            ))

        if uniform_nodes:
            lows = [n.var_support.low for n in uniform_nodes]
            highs = [n.var_support.high for n in uniform_nodes]
            scopes = [n.var for n in uniform_nodes]
            out_idx = []
            for n_id in uniform_ids:
                self.node_to_idx[n_id] = idx
                out_idx.append(idx)
                idx += 1
            self.input_layers.append(UniformInputLayer(
                out_idx,
                torch.tensor(lows, dtype=torch.float32),
                torch.tensor(highs, dtype=torch.float32),
                scopes,
            ))

        if categorical_nodes:
            scopes = [n.var for n in categorical_nodes]
            out_idx = []
            for n_id in categorical_ids:
                self.node_to_idx[n_id] = idx
                out_idx.append(idx)
                idx += 1
            
            import math
            max_cats = max(len(n.categories) for n in categorical_nodes)
            cats_tensor = torch.full((len(categorical_nodes), max_cats), float("nan"))
            probs_tensor = torch.full((len(categorical_nodes), max_cats), float("-inf"))
            
            for i, n in enumerate(categorical_nodes):
                for j, (cat, prob) in enumerate(zip(n.categories, n.probabilities)):
                    cats_tensor[i, j] = float(cat)
                    probs_tensor[i, j] = math.log(prob) if prob > 0 else float("-inf")
                    
            self.input_layers.append(CategoricalInputLayer(
                out_idx,
                cats_tensor,
                probs_tensor,
                scopes,
            ))

        for layer_node_ids in node_layers_iter:
            is_sum_layer = all(symbolic_circuit.is_sum_node(node_id) for node_id in layer_node_ids)
            is_product_layer = all(
                symbolic_circuit.is_product_node(node_id) for node_id in layer_node_ids
            )
            assert is_sum_layer or is_product_layer, "Mixed layer types are not supported"

            out_idx = []
            for n_id in layer_node_ids:
                self.node_to_idx[n_id] = idx
                out_idx.append(idx)
                idx += 1

            if is_product_layer:
                left_idx, right_idx = self._compute_product_layer_indices(
                    symbolic_circuit, layer_node_ids
                )
                layer = ProductLayer(out_idx, left_idx, right_idx)
                self.layers.append(layer)

            elif is_sum_layer:
                child_node_ids = sorted(
                    list(
                        {
                            child_id
                            for node_id in layer_node_ids
                            for child_id in symbolic_circuit.get_children(node_id)
                        }
                    )
                )
                in_idx = [self.node_to_idx[n_id] for n_id in child_node_ids]
                connections = self._compute_sum_layer_connections(
                    symbolic_circuit, layer_node_ids, child_node_ids
                )
                try:
                    layer = MonarchSumLayer(
                        out_idx,
                        in_idx,
                        connections,
                        respect_sparsity=respect_sparsity,
                    )
                except Exception as e:
                    print(f"MonarchSumLayer error: {e}")
                    layer = SumLayer(out_idx, in_idx, connections)
                self.layers.append(layer)

        self.num_nodes = idx

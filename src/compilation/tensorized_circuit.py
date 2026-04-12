import math
from typing import List, Tuple

import torch
from torch import nn

from src.compilation.base_circuit import (
    CategoricalInputLayer,
    GaussianInputLayer,
    ProductLayer,
    SafeLogSumExp,
    TensorizedLayer,
    UniformInputLayer,
)
from src.logger import logger as g_logger
from src.symbolic import SymbolicArithmeticCircuit


logger = g_logger.getChild(__name__)


class SumLayer(TensorizedLayer):
    def __init__(self, out_idx: List[int], in_idx: List[int], connections: torch.Tensor):
        super().__init__(out_idx)
        self.in_idx = in_idx
        self.register_buffer("connections", connections)
        # Connections represent weights. Convert to log space.
        # Add a tiny epsilon to avoid log(0) for exactly zero weights.
        epsilon = 1e-10
        self.log_connections = nn.Parameter(torch.log(connections + epsilon), requires_grad=True)

    def forward(self, global_buffer: torch.Tensor) -> torch.Tensor:
        input_values = global_buffer[:, self.in_idx]
        # input_values shape: (Batch_size, Num_children)
        # log_connections shape: (Num_nodes, Num_children)

        # We want to compute log(sum_i (w_i * exp(log_p_i))) for each sum node
        # Compute log(w_i) + log_p_i -> shape: (Batch_size, Num_nodes, Num_children)

        # input_values: (Batch_size, 1, Num_children)
        # log_connections: (1, Num_nodes, Num_children)
        input_expanded = input_values.unsqueeze(1)
        log_conn_expanded = self.log_connections.unsqueeze(0)

        # log_terms: (Batch_size, Num_nodes, Num_children)
        log_terms = input_expanded + log_conn_expanded

        # We only want to sum over children that actually have a connection.
        # To be safe, we can mask out inputs for zero weights by setting them to -inf
        zero_mask = (self.connections == 0).unsqueeze(0)
        log_terms = log_terms.masked_fill(zero_mask, float("-inf"))

        # LogSumExp over the children dimension (dim=2) using safe backward logic
        res = SafeLogSumExp.apply(log_terms, 2)
        global_buffer[:, self.out_idx] = res
        return res

    def update_params(self):
        with torch.no_grad():
            responsibilities = self.log_connections.grad
            total_resp = responsibilities.sum(dim=-1, keepdim=True)

            # Only update nodes that received some gradient.
            # Dead nodes (total_resp == 0) keep their previous parameters.
            mask = total_resp > 0

            # Using Laplace smoothing (1e-5) only for nodes we are updating
            smoothed_resp = responsibilities + 1e-5
            new_probs = smoothed_resp / smoothed_resp.sum(dim=-1, keepdim=True)

            # Apply new probabilities where mask is True, keep old where False
            current_probs = torch.exp(self.log_connections)
            new_probs = torch.where(mask, new_probs, current_probs)

            self.log_connections.copy_(torch.log(new_probs))
            self.log_connections.grad.zero_()


class TensorizedCircuit(nn.Module):
    def __init__(self, symbolic_circuit: SymbolicArithmeticCircuit):
        super().__init__()
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
                # Uniform distributions
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
            
            # Categories might have different lengths. Pad them.
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

        # build layers bottom-up
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
                layer = SumLayer(out_idx, in_idx, connections)
                self.layers.append(layer)

        self.num_nodes = idx

    def _compute_sum_layer_connections(
        self, circuit: SymbolicArithmeticCircuit, sum_node_ids: List[int], child_node_ids: List[int]
    ) -> torch.Tensor:
        connections = torch.zeros(len(sum_node_ids), len(child_node_ids), dtype=torch.float32)
        for i, sum_node_id in enumerate(sum_node_ids):
            child_id_to_weight = dict(circuit.get_outgoing_edges(sum_node_id))
            for j, child_id in enumerate(child_node_ids):
                if child_id in child_id_to_weight:
                    connections[i, j] = child_id_to_weight[child_id]
        return connections

    def _compute_product_layer_indices(
        self, circuit: SymbolicArithmeticCircuit, node_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        left_idx = []
        right_idx = []
        for node_id in node_ids:
            children = circuit.get_children(node_id)
            assert len(children) == 2, "Only binary product nodes are supported"

            left_pos = self.node_to_idx[children[0]]
            right_pos = self.node_to_idx[children[1]]

            left_idx.append(left_pos)
            right_idx.append(right_pos)
        return left_idx, right_idx

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        batch_size = data.shape[0]
        global_buffer = torch.empty(
            (batch_size, self.num_nodes), dtype=data.dtype, device=data.device
        )

        # evaluate all input layers
        for layer in self.input_layers:
            layer.forward(global_buffer, data)

        # evaluate internal layers
        for layer in self.layers:
            layer.forward(global_buffer)

        output_idx = self.layers[-1].out_idx
        return global_buffer[:, output_idx]

    def update_params(self):
        for layer in self.layers:
            if isinstance(layer, SumLayer):
                layer.update_params()
        for layer in getattr(self, "input_layers", []):
            if isinstance(layer, (GaussianInputLayer, CategoricalInputLayer)):
                layer.update_params()


def em(circuit: TensorizedCircuit, data: torch.Tensor):
    # E-step: Forward pass to compute responsibilities
    log_likelihoods = circuit.forward(data)
    logger.info(f"LLs: {log_likelihoods}")

    # Clear previous gradients
    circuit.zero_grad()

    # M-step: Backward pass to populate .grad with responsibilities
    loss = log_likelihoods.sum()
    loss.backward()

    # Update parameters using computed responsibilities
    circuit.update_params()

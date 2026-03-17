from typing import List, Tuple

import torch
from torch import nn

from src.compilation.base_circuit import (
    GaussianInputLayer,
    ProductLayer,
    SafeLogSumExp,
    TensorizedLayer,
)
from src.logger import logger as g_logger
from src.symbolic import SymbolicArithmeticCircuit


logger = g_logger.getChild(__name__)


class SumLayer(TensorizedLayer):
    def __init__(self, node_ids: List[int], connections: torch.Tensor):
        super().__init__(node_ids)
        self.register_buffer("connections", connections)
        # Connections represent weights. Convert to log space.
        # Add a tiny epsilon to avoid log(0) for exactly zero weights.
        epsilon = 1e-10
        self.log_connections = nn.Parameter(torch.log(connections + epsilon), requires_grad=True)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
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
        return SafeLogSumExp.apply(log_terms, 2)

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

        node_layers = symbolic_circuit.layered_topological_sort(reverse=True)
        input_node_ids = next(node_layers)

        # assuming all input nodes are gaussian for now
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

        # build layers bottom-up
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
                layer = SumLayer(layer_node_ids, connections)
                self.layers.append(layer)

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
        self,
        circuit: SymbolicArithmeticCircuit,
        node_ids: List[int],
        prev_layer_node_ids: List[int],
    ) -> Tuple[List[int], List[int]]:
        left_idx = []
        right_idx = []
        for node_id in node_ids:
            children = circuit.get_children(node_id)
            assert len(children) == 2, "Only binary product nodes are supported"

            # Find the indices of these children in the previous layer's output
            # rather than using their absolute node_ids
            left_pos = prev_layer_node_ids.index(children[0])
            right_pos = prev_layer_node_ids.index(children[1])

            left_idx.append(left_pos)
            right_idx.append(right_pos)
        return left_idx, right_idx

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        output = input_values
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def update_params(self):
        for layer in self.layers:
            if isinstance(layer, (SumLayer, GaussianInputLayer)):
                layer.update_params()
            if isinstance(layer, SumLayer):
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

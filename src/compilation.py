import math
from typing import List, Tuple

import torch

from src.logging import logger as g_logger
from src.symbolic import SymbolicArithmeticCircuit
from src.symbolic.arithmetic.distributions import (
    GaussianDistribution,
    TruncatedGaussianDistribution,
)


logger = g_logger.getChild(__name__)


class TensorizedLayer:
    def __init__(self, node_ids: List[int]):
        self.node_ids = node_ids


class ProductLayer(TensorizedLayer):
    def __init__(self, node_ids: List[int], left_idx: List[int], right_idx: List[int]):
        super().__init__(node_ids)
        self.left_idx = left_idx
        self.right_idx = right_idx

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        # Sum of log-probabilities
        return input_values[:, self.left_idx] + input_values[:, self.right_idx]


class SumLayer(TensorizedLayer):
    def __init__(self, node_ids: List[int], connections: torch.Tensor):
        super().__init__(node_ids)
        self.connections = connections
        # Connections represent weights. Convert to log space.
        # Add a tiny epsilon to avoid log(0) for exactly zero weights.
        epsilon = 1e-10
        self.log_connections = torch.log(connections + epsilon)

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

        # LogSumExp over the children dimension (dim=2)
        return torch.logsumexp(log_terms, dim=2)


class InputLayer(TensorizedLayer):
    def __init__(
        self,
        node_ids: List[int],
        means: torch.Tensor,
        stds: torch.Tensor,
        lows: torch.Tensor,
        highs: torch.Tensor,
        scopes: List[int],
    ):
        super().__init__(node_ids)
        self.means = means
        self.stds = stds
        self.lows = lows
        self.highs = highs
        self.scopes = scopes

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        # input_values: (Batch_size, Num_vars)
        # Gather x values for each node based on self.scopes
        x = input_values[:, self.scopes]  # Shape: (Batch_size, Num_nodes)

        # Calculate standard normal variables
        z = (x - self.means) / self.stds

        # PDF of standard normal
        phi_z = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)

        # PDF of original normal
        pdf_x = phi_z / self.stds

        # Normalization constant for truncation
        cdf_high = 0.5 * (1 + torch.erf((self.highs - self.means) / (self.stds * math.sqrt(2))))
        cdf_low = 0.5 * (1 + torch.erf((self.lows - self.means) / (self.stds * math.sqrt(2))))
        # Z = cdf_high - cdf_low
        Z = cdf_high - cdf_low

        # Truncated PDF
        trunc_log_pdf = torch.log(pdf_x) - torch.log(Z)

        # Zero out values outside the bounds (log(0) = -inf)
        out_of_bounds = (x < self.lows) | (x > self.highs)
        trunc_log_pdf = torch.where(
            out_of_bounds, torch.full_like(trunc_log_pdf, float("-inf")), trunc_log_pdf
        )

        return trunc_log_pdf


class TensorizedCircuit:
    def __init__(self, symbolic_circuit: SymbolicArithmeticCircuit):
        self.layers: List[TensorizedLayer] = []

        node_layers = symbolic_circuit.layered_topological_sort(reverse=True)
        input_node_ids = next(node_layers)

        # FIXME: should support more than just guassians
        means, stds, lows, highs, scopes = [], [], [], [], []
        for node_id in input_node_ids:
            node = symbolic_circuit.get_node_data(node_id)
            if not isinstance(node, (GaussianDistribution, TruncatedGaussianDistribution)):
                raise ValueError(f"Unsupported leaf node type: {type(node)}")

            means.append(node.mean)
            stds.append(node.stddev)
            lows.append(node.var_support.low)
            highs.append(node.var_support.high)
            scopes.append(node.var)

        input_layer = InputLayer(
            input_node_ids,
            torch.tensor(means, dtype=torch.float32),
            torch.tensor(stds, dtype=torch.float32),
            torch.tensor(lows, dtype=torch.float32),
            torch.tensor(highs, dtype=torch.float32),
            scopes,
        )  # NOTE: Can we guarantee all inputs to be in the same (first) layer?
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
            logger.debug(f"Forwarding through layer with node IDs: {layer.node_ids}")
            logger.debug(f"Input:\n{output}")
            output = layer.forward(output)
            logger.debug(f"Output:\n{output}")
        return output

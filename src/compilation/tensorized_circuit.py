import math
from typing import List, Tuple

import torch
from torch import nn

from src.logger import logger as g_logger
from src.symbolic import SymbolicArithmeticCircuit


logger = g_logger.getChild(__name__)


class TensorizedLayer(nn.Module):
    def __init__(self, node_ids: List[int]):
        super().__init__()
        self.node_ids = node_ids

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ProductLayer(TensorizedLayer):
    def __init__(self, node_ids: List[int], left_idx: List[int], right_idx: List[int]):
        super().__init__(node_ids)
        self.left_idx = left_idx
        self.right_idx = right_idx

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        # Sum of log-probabilities
        return input_values[:, self.left_idx] + input_values[:, self.right_idx]


class SafeLogSumExp(torch.autograd.Function):
    """
    A mathematically stable logsumexp that returns 0 gradients
    instead of NaN when all inputs are -inf.
    """

    @staticmethod
    def forward(ctx, input, dim):
        output = torch.logsumexp(input, dim=dim)
        ctx.save_for_backward(input, output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        dim = ctx.dim

        # When output is -inf, all inputs are -inf.
        # This causes NaN in softmax exp(input - output).
        # We mask these out to prevent NaNs.
        out_unsqueezed = output.unsqueeze(dim)
        mask = torch.isneginf(out_unsqueezed)

        safe_output = torch.where(mask, torch.zeros_like(out_unsqueezed), out_unsqueezed)
        safe_input = torch.where(mask, torch.zeros_like(input), input)

        softmax = torch.exp(safe_input - safe_output)

        # For originally -inf rows, clear the softmax to 0 so they get 0 gradient
        softmax = torch.where(mask, torch.zeros_like(softmax), softmax)

        grad_input = grad_output.unsqueeze(dim) * softmax
        return grad_input, None


class SumLayer(TensorizedLayer):
    def __init__(self, node_ids: List[int], connections: torch.Tensor):
        super().__init__(node_ids)
        self.connections = connections
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


class GaussianInputLayer(TensorizedLayer):
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
        self.means = nn.Parameter(means, requires_grad=False)
        self.stds = nn.Parameter(stds, requires_grad=False)
        self.register_buffer("lows", lows)
        self.register_buffer("highs", highs)
        self.scopes = scopes

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        # input_values: (Batch_size, Num_vars)
        # Gather x values for each node based on self.scopes
        x = input_values[:, self.scopes]  # Shape: (Batch_size, Num_nodes)

        self.saved_x = x.detach()

        # Calculate standard normal variables
        z = (x - self.means) / self.stds

        # PDF of standard normal
        # Log PDF of standard normal: ln(1/sqrt(2pi)) - 0.5 * z^2
        log_phi_z = -0.5 * math.log(2 * math.pi) - 0.5 * (z**2)

        # Log PDF of original normal
        log_pdf_x = log_phi_z - torch.log(self.stds)

        # Normalization constant for truncation
        cdf_high = 0.5 * (1 + torch.erf((self.highs - self.means) / (self.stds * math.sqrt(2))))
        cdf_low = 0.5 * (1 + torch.erf((self.lows - self.means) / (self.stds * math.sqrt(2))))
        # Z = cdf_high - cdf_low
        Z = cdf_high - cdf_low

        # Truncated PDF
        # clamp Z to eliminate log(0) causing infinite gradients
        trunc_log_pdf = log_pdf_x - torch.log(Z.clamp(min=1e-10))

        # Zero out values outside the bounds (log(0) = -inf)
        out_of_bounds = (x < self.lows) | (x > self.highs)
        trunc_log_pdf = torch.where(
            out_of_bounds, torch.full_like(trunc_log_pdf, float("-inf")), trunc_log_pdf
        )

        self.saved_log_pdf = trunc_log_pdf.requires_grad_(True)
        self.saved_log_pdf.retain_grad()

        return trunc_log_pdf

    def update_params(self):
        with torch.no_grad():
            # Avoid negative responsibility adding to stats
            valid_grads = torch.clamp(self.saved_log_pdf.grad, min=0.0)  # NOTE: can I omit this?

            responsibilities = valid_grads + 1e-15

            total_resp = responsibilities.sum(dim=0)

            valid_mask = total_resp > 1e-5

            sufficient_stats = torch.stack([self.saved_x, self.saved_x**2], dim=0)

            exp_params = (responsibilities.unsqueeze(0) * sufficient_stats).sum(
                dim=1
            ) / total_resp.clamp(min=1e-15)

            new_means = exp_params[0]
            new_vars = exp_params[1] - (new_means**2)
            # Use a safe minimal variance to prevent STD falling strictly to 0
            new_stds = torch.sqrt(new_vars.clamp(min=1e-5))

            self.means.copy_(torch.where(valid_mask, new_means, self.means))
            self.stds.copy_(torch.where(valid_mask, new_stds, self.stds))


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

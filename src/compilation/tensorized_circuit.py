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
    def __init__(
        self,
        out_idx: List[int],
        in_idx: List[int],
        connections: torch.Tensor,
        z_mask: torch.Tensor = None,
    ):
        super().__init__(out_idx)
        self.in_idx = in_idx
        self.register_buffer("connections", connections)

        if z_mask is None:
            z_mask = torch.zeros(len(out_idx), dtype=torch.bool)
        self.register_buffer("z_mask", z_mask)

        # Connections represent weights. Convert to log space.
        # Add a tiny epsilon to avoid log(0) for exactly zero weights.
        epsilon = 1e-10
        self.log_connections = nn.Parameter(torch.log(connections + epsilon), requires_grad=True)

    def _compute_logsumexp(self, global_buffer: torch.Tensor) -> torch.Tensor:
        input_values = global_buffer[:, self.in_idx]
        input_expanded = input_values.unsqueeze(1)
        log_conn_expanded = self.log_connections.unsqueeze(0)

        log_terms = input_expanded + log_conn_expanded

        zero_mask = (self.connections == 0).unsqueeze(0)
        log_terms = log_terms.masked_fill(zero_mask, float("-inf"))

        res = SafeLogSumExp.apply(log_terms, 2)
        return res

    def forward(
        self, buf_joint: torch.Tensor, buf_xz: torch.Tensor = None, buf_z: torch.Tensor = None
    ) -> torch.Tensor:
        res_joint = self._compute_logsumexp(buf_joint)

        if buf_xz is None or buf_z is None:
            buf_joint[:, self.out_idx] = res_joint
            return res_joint

        res_xz = self._compute_logsumexp(buf_xz)
        res_z = self._compute_logsumexp(buf_z)

        interventional_math = res_joint - res_xz + res_z
        interventional_math = torch.nan_to_num(interventional_math, nan=float("-inf"))

        z_mask_expanded = self.z_mask.unsqueeze(0).expand_as(res_joint)
        res_final = torch.where(z_mask_expanded, interventional_math, res_joint)

        buf_joint[:, self.out_idx] = res_final
        # Also propagate down correctly to the underlying path buffers if needed
        # (Assuming the network will read them)
        buf_xz[:, self.out_idx] = res_xz
        buf_z[:, self.out_idx] = res_z

        return res_final

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
    def __init__(self, symbolic_circuit: SymbolicArithmeticCircuit, target_vars: set = None):
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
            elif (
                hasattr(node, "var_support")
                and not hasattr(node, "categories")
                and not hasattr(node, "mean")
            ):
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
            self.input_layers.append(
                GaussianInputLayer(
                    out_idx,
                    torch.tensor(means, dtype=torch.float32),
                    torch.tensor(stds, dtype=torch.float32),
                    torch.tensor(lows, dtype=torch.float32),
                    torch.tensor(highs, dtype=torch.float32),
                    scopes,
                )
            )

        if uniform_nodes:
            lows = [n.var_support.low for n in uniform_nodes]
            highs = [n.var_support.high for n in uniform_nodes]
            scopes = [n.var for n in uniform_nodes]
            out_idx = []
            for n_id in uniform_ids:
                self.node_to_idx[n_id] = idx
                out_idx.append(idx)
                idx += 1
            self.input_layers.append(
                UniformInputLayer(
                    out_idx,
                    torch.tensor(lows, dtype=torch.float32),
                    torch.tensor(highs, dtype=torch.float32),
                    scopes,
                )
            )

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

            self.input_layers.append(
                CategoricalInputLayer(
                    out_idx,
                    cats_tensor,
                    probs_tensor,
                    scopes,
                )
            )

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
                # Prepare and store node scopes for the layer before building
                node_scopes = []
                for n_id in layer_node_ids:
                    node = symbolic_circuit.get_node_data(n_id)
                    node_scopes.append(
                        set(node.scope.features)
                        if hasattr(node.scope, "features")
                        else set(node.scope)
                    )
                layer = self._build_product_layer(out_idx, left_idx, right_idx)
                layer.node_scopes = node_scopes
                self.layers.append(layer)

            elif is_sum_layer:
                child_node_ids = sorted(
                    {
                        child_id
                        for node_id in layer_node_ids
                        for child_id in symbolic_circuit.get_children(node_id)
                    }
                )
                in_idx = [self.node_to_idx[n_id] for n_id in child_node_ids]
                connections = self._compute_sum_layer_connections(
                    symbolic_circuit, layer_node_ids, child_node_ids
                )

                node_scopes = []
                for n_id in layer_node_ids:
                    node = symbolic_circuit.get_node_data(n_id)
                    node_scopes.append(
                        set(node.scope.features)
                        if hasattr(node.scope, "features")
                        else set(node.scope)
                    )

                layer = self._build_sum_layer(out_idx, in_idx, connections)
                layer.node_scopes = node_scopes
                self.layers.append(layer)

        self.num_nodes = idx
        if target_vars is not None:
            self.set_target_vars(target_vars)

    def _build_sum_layer(
        self,
        out_idx: List[int],
        in_idx: List[int],
        connections: torch.Tensor,
        z_mask: torch.Tensor = None,
    ) -> TensorizedLayer:
        return SumLayer(out_idx, in_idx, connections, z_mask=z_mask)

    def _build_product_layer(
        self,
        out_idx: List[int],
        left_idx: List[int],
        right_idx: List[int],
        z_mask: torch.Tensor = None,
    ) -> TensorizedLayer:
        return ProductLayer(out_idx, left_idx, right_idx, z_mask=z_mask)

    def set_target_vars(self, target_vars: set):
        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        for layer in self.layers:
            if hasattr(layer, "node_scopes") and layer.node_scopes:
                mask = []
                for scope in layer.node_scopes:
                    mask.append(not scope.isdisjoint(target_vars))
                layer.z_mask = torch.tensor(mask, dtype=torch.bool, device=device)
            else:
                if hasattr(layer, "z_mask"):
                    layer.z_mask = torch.zeros(len(layer.out_idx), dtype=torch.bool, device=device)

    def backdoor(
        self, data: torch.Tensor, do_vars: List[int], z_vars: List[int], query_vars: List[int]
    ) -> torch.Tensor:
        """
        Dynamically adjusts causal intervention masks and computes the backdoor probability
        for P(query_vars | do(do_vars)) by marginalizing out Z across structural pathways.
        """
        self.set_target_vars(set(z_vars + do_vars))
        d_joint = data.clone()
        d_xz = data.clone()
        for c in query_vars:
            d_xz[:, c] = float("nan")
        d_z = data.clone()
        for c in do_vars + query_vars:
            d_z[:, c] = float("nan")

        return self(d_joint, d_xz, d_z)

    def _compute_scope_mask(
        self, circuit: SymbolicArithmeticCircuit, node_ids: List[int], target_vars: set
    ) -> torch.Tensor:
        mask = torch.zeros(len(node_ids), dtype=torch.bool)
        if not target_vars:
            return mask

        for i, n_id in enumerate(node_ids):
            node = circuit.get_node_data(n_id)
            if hasattr(node, "scope"):
                # scope is an iterable/set
                scope_set = set(node.scope)
                if not scope_set.isdisjoint(target_vars):
                    mask[i] = True

        return mask

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

    def forward(
        self, data_joint: torch.Tensor, data_xz: torch.Tensor = None, data_z: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size = data_joint.shape[0]

        buf_joint = torch.empty(
            (batch_size, self.num_nodes), dtype=data_joint.dtype, device=data_joint.device
        )
        buf_xz = None
        buf_z = None

        if data_xz is not None and data_z is not None:
            buf_xz = torch.empty(
                (batch_size, self.num_nodes), dtype=data_xz.dtype, device=data_xz.device
            )
            buf_z = torch.empty(
                (batch_size, self.num_nodes), dtype=data_z.dtype, device=data_z.device
            )

        # evaluate all input layers
        for layer in self.input_layers:
            layer.forward(buf_joint, data_joint)
            if buf_xz is not None and buf_z is not None:
                layer.forward(buf_xz, data_xz)
                layer.forward(buf_z, data_z)

        # evaluate internal layers
        for layer in self.layers:
            if buf_xz is not None and buf_z is not None:
                layer.forward(buf_joint, buf_xz, buf_z)
            else:
                layer.forward(buf_joint)

        output_idx = self.layers[-1].out_idx
        return buf_joint[:, output_idx]

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

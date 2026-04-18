import math
from typing import Dict, List, Tuple

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


class TuckerLayer(TensorizedLayer):
    """
    Computes a fused Sum-Product layer in probability space using einsum.
    """

    def __init__(
        self,
        out_idx: List[int],
        left_indices: List[int],
        right_indices: List[int],
        W: torch.Tensor,
        z_mask: torch.Tensor = None,
    ):
        super().__init__(out_idx)
        # W shape: (G_P, G_L, G_R, h_out, h_l, h_r)
        self.register_buffer("left_indices", torch.tensor(left_indices, dtype=torch.long))
        self.register_buffer("right_indices", torch.tensor(right_indices, dtype=torch.long))

        if z_mask is None:
            z_mask = torch.zeros(len(out_idx), dtype=torch.bool)
        self.register_buffer("z_mask", z_mask)

        # Keep weights in probability space directly for einsum
        self.weights = nn.Parameter(W)
        self.G_P, self.G_L, self.G_R, self.h_out, self.h_l, self.h_r = W.shape

    def _compute_tucker(self, global_buffer: torch.Tensor) -> torch.Tensor:
        # global_buffer is in PROBABILITY SPACE
        L = global_buffer[:, self.left_indices].view(-1, self.G_L, self.h_l)
        R = global_buffer[:, self.right_indices].view(-1, self.G_R, self.h_r)

        # 'bli, brj, plroij -> bpo'
        out = torch.einsum("bli, brj, plroij -> bpo", L, R, self.weights)
        return out.flatten(1)  # (B, G_P * h_out)

    def forward(
        self, buf_joint: torch.Tensor, buf_xz: torch.Tensor = None, buf_z: torch.Tensor = None
    ) -> torch.Tensor:
        res_joint = self._compute_tucker(buf_joint)

        if buf_xz is None or buf_z is None:
            buf_joint[:, self.out_idx] = res_joint
            return res_joint

        res_xz = self._compute_tucker(buf_xz)
        res_z = self._compute_tucker(buf_z)

        interventional_math = res_joint - res_xz + res_z
        interventional_math = torch.nan_to_num(interventional_math, nan=0.0)

        z_mask_expanded = self.z_mask.unsqueeze(0).expand_as(res_joint)
        res_final = torch.where(z_mask_expanded, interventional_math, res_joint)

        buf_joint[:, self.out_idx] = res_final
        buf_xz[:, self.out_idx] = res_xz
        buf_z[:, self.out_idx] = res_z

        return res_final


class TensorizedCircuit(nn.Module):
    def __init__(self, symbolic_circuit: SymbolicArithmeticCircuit, target_vars: set = None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_layers = nn.ModuleList()
        self.output_idx = []
        self.num_nodes = 0
        self.max_var_index = -1
        self.identity_idx = -1

        self.node_to_idx: Dict[int, List[int]] = {}
        idx = 0

        # Pass 1: Setup Leaf Nodes
        gaussian_nodes, uniform_nodes, categorical_nodes = [], [], []
        leaf_ids = symbolic_circuit.get_leaves()
        for n_id in leaf_ids:
            node = symbolic_circuit.get_node_data(n_id)
            node_unit_count = getattr(node, "unit_count", 1)
            self.node_to_idx[n_id] = list(range(idx, idx + node_unit_count))
            idx += node_unit_count

            if hasattr(node, "mean") and hasattr(node, "stddev"):
                gaussian_nodes.append((node, self.node_to_idx[n_id]))
            elif (
                hasattr(node, "var_support")
                and not hasattr(node, "categories")
                and not hasattr(node, "mean")
            ):
                uniform_nodes.append((node, self.node_to_idx[n_id]))
            elif hasattr(node, "categories"):
                categorical_nodes.append((node, self.node_to_idx[n_id]))

        # Initialize input layers (we'll modify them slightly to output probabilities instead of log-probs, or we just exp() after)
        self._init_input_layers(gaussian_nodes, uniform_nodes, categorical_nodes)

        self.identity_idx = idx
        idx += 1

        # We must build layers by depth to guarantee topological ordering
        # Since the SPN alternates Sum and Product, and we want to fuse Sum->Product->Sum,
        # we extract the "SumNode" layers.

        sum_nodes = [
            n
            for n in symbolic_circuit.topological_sort(reverse=True)
            if symbolic_circuit.is_sum_node(n) and n not in leaf_ids
        ]

        # We need to group sum nodes by depth.
        # Compute depth of each node
        depths = {}
        for n in symbolic_circuit.topological_sort():
            parents = symbolic_circuit.get_parents(n)
            if not parents:
                depths[n] = 0
            else:
                depths[n] = max(depths[p] for p in parents) + 1

        # Group sum nodes by depth
        depth_to_sums = {}
        for n in sum_nodes:
            d = depths[n]
            if d not in depth_to_sums:
                depth_to_sums[d] = []
            depth_to_sums[d].append(n)

        # Process layers from highest depth (bottom) to lowest (root)
        for d in sorted(depth_to_sums.keys(), reverse=True):
            layer_sum_ids = depth_to_sums[d]

            # Allocate indices for these sum nodes
            layer_out_idx = []
            for n_id in layer_sum_ids:
                node = symbolic_circuit.get_node_data(n_id)
                node_unit_count = getattr(node, "unit_count", 1)
                self.node_to_idx[n_id] = list(range(idx, idx + node_unit_count))
                layer_out_idx.extend(self.node_to_idx[n_id])
                idx += node_unit_count

            layer = self._build_tucker_layer(symbolic_circuit, layer_sum_ids, layer_out_idx)
            self.layers.append(layer)

        # Roots are depth 0
        self.output_idx = []
        for n in depth_to_sums.get(0, []):
            self.output_idx.extend(self.node_to_idx[n])

        self.num_nodes = idx

        if target_vars is not None:
            self.set_target_vars(target_vars)

    def _init_input_layers(self, gaussian_nodes, uniform_nodes, categorical_nodes):
        if gaussian_nodes:
            layer_out_idx = [i for _, indices in gaussian_nodes for i in indices]
            means, stds, lows, highs, scopes = [], [], [], [], []
            for node, indices in gaussian_nodes:
                for _ in indices:
                    means.append(node.mean)
                    stds.append(node.stddev)
                    lows.append(node.var_support.low)
                    highs.append(node.var_support.high)
                    scopes.append(node.var)
            self.input_layers.append(
                GaussianInputLayer(
                    layer_out_idx,
                    torch.tensor(means, dtype=torch.float32),
                    torch.tensor(stds, dtype=torch.float32),
                    torch.tensor(lows, dtype=torch.float32),
                    torch.tensor(highs, dtype=torch.float32),
                    scopes,
                )
            )
            self.max_var_index = max([s for s in scopes] + [-1])

        if uniform_nodes:
            layer_out_idx = [i for _, indices in uniform_nodes for i in indices]
            lows, highs, scopes = [], [], []
            for node, indices in uniform_nodes:
                for _ in indices:
                    lows.append(node.var_support.low)
                    highs.append(node.var_support.high)
                    scopes.append(node.var)
            self.input_layers.append(
                UniformInputLayer(
                    layer_out_idx,
                    torch.tensor(lows, dtype=torch.float32),
                    torch.tensor(highs, dtype=torch.float32),
                    scopes,
                )
            )

    def _build_tucker_layer(
        self, circuit: SymbolicArithmeticCircuit, sum_node_ids: List[int], out_idx: List[int]
    ):
        # We need to discover the G_P, G_L, G_R groups.
        G_P = len(sum_node_ids)
        h_out = circuit.get_node_data(sum_node_ids[0]).unit_count

        # Discover left and right children groups.
        # The sum nodes connect to ProductNodes. The ProductNodes connect to (left_child, right_child)
        left_nodes = set()
        right_nodes = set()

        for p_id in sum_node_ids:
            prod_children = circuit.get_children(p_id)
            for prod_id in prod_children:
                children = circuit.get_children(prod_id)
                if len(children) == 2:
                    left_nodes.add(children[0])
                    right_nodes.add(children[1])
                elif len(children) == 1:
                    left_nodes.add(children[0])

        left_nodes = sorted(list(left_nodes))
        right_nodes = sorted(list(right_nodes))

        G_L = max(1, len(left_nodes))
        G_R = max(1, len(right_nodes))

        h_l = circuit.get_node_data(left_nodes[0]).unit_count if left_nodes else 1
        h_r = circuit.get_node_data(right_nodes[0]).unit_count if right_nodes else 1

        W = torch.zeros(G_P, G_L, G_R, h_out, h_l, h_r)

        for p_idx, p_id in enumerate(sum_node_ids):
            prod_children = circuit.get_children(p_id)
            for prod_id in prod_children:
                weight_matrix = circuit.get_edge_data(p_id, prod_id)  # (h_out, h_l * h_r)

                children = circuit.get_children(prod_id)
                if len(children) == 2:
                    l_id, r_id = children
                    l_idx = left_nodes.index(l_id)
                    r_idx = right_nodes.index(r_id)
                    W[p_idx, l_idx, r_idx] = weight_matrix.view(h_out, h_l, h_r)
                elif len(children) == 1:
                    l_id = children[0]
                    l_idx = left_nodes.index(l_id)
                    W[p_idx, l_idx, 0] = weight_matrix.view(h_out, h_l, 1)

        # Normalize W so that each sum node (G_P, h_out) marginalizes to 1 over its children (G_L, G_R, h_l, h_r)
        W = W / W.sum(dim=(1, 2, 4, 5), keepdim=True).clamp(min=1e-10)

        left_indices = []
        for l_id in left_nodes:
            left_indices.extend(self.node_to_idx[l_id])
        if not left_indices:
            left_indices = [self.identity_idx]

        right_indices = []
        for r_id in right_nodes:
            right_indices.extend(self.node_to_idx[r_id])
        if not right_indices:
            right_indices = [self.identity_idx]

        return TuckerLayer(out_idx, left_indices, right_indices, W)

    def set_target_vars(self, target_vars: set):
        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        for layer in self.layers:
            layer.z_mask = torch.zeros(len(layer.out_idx), dtype=torch.bool, device=device)

    def forward(
        self, data_joint: torch.Tensor, data_xz: torch.Tensor = None, data_z: torch.Tensor = None
    ) -> torch.Tensor:
        def _pad_if_needed(data: torch.Tensor) -> torch.Tensor:
            if data is None:
                return None
            if data.shape[1] <= self.max_var_index:
                padding = torch.full(
                    (data.shape[0], self.max_var_index + 1 - data.shape[1]),
                    float("nan"),
                    device=data.device,
                    dtype=data.dtype,
                )
                return torch.cat([data, padding], dim=1)
            return data

        data_joint = _pad_if_needed(data_joint)
        batch_size = data_joint.shape[0]

        buf_joint = torch.zeros(
            (batch_size, self.num_nodes), dtype=data_joint.dtype, device=data_joint.device
        )

        # Input layers evaluate in LOG space. We must torch.exp them to probability space!
        for layer in self.input_layers:
            layer.forward(buf_joint, data_joint)

        # Convert log-probabilities to probabilities
        buf_joint = torch.exp(buf_joint)

        # Handle identity node
        buf_joint = buf_joint.clone()
        buf_joint[:, self.identity_idx] = 1.0

        for layer in self.layers:
            buf_joint = buf_joint.clone()
            # If layer.forward modifies buf_joint inplace, we need to return it, or pass it and the layer modifies the cloned one. Wait, if layer modifies the cloned tensor inplace, the caller's buf_joint won't see the modification unless layer.forward mutates the tensor we pass.
            # But wait, buf_joint.clone() creates a NEW tensor. If we pass buf_joint to layer.forward, it mutates the new tensor. And buf_joint will point to that new tensor.
            layer.forward(buf_joint)

        # Return in LOG space
        out_prob = buf_joint[:, self.output_idx]
        return torch.log(out_prob.clamp(min=1e-30))


def em(circuit: TensorizedCircuit, data: torch.Tensor, iterations: int = 5):
    # E-step: Forward pass to compute likelihoods
    for i in range(iterations):
        # Forward pass returns log probabilities
        log_probs = circuit.forward(data)

        # Compute log-likelihood
        log_likelihood = log_probs.sum()
        print(f"Iteration {i}: Log-Likelihood = {log_likelihood.item():.4f}")

        # Standard backprop to compute gradients (responsibilities)
        circuit.zero_grad()
        log_likelihood.backward()

        # M-step: Update parameters using the gradients
        with torch.no_grad():
            for layer in circuit.layers:
                if isinstance(layer, TuckerLayer):
                    pass
            # Standard SGD update for weights in self.parameters()
            with torch.no_grad():
                for param in circuit.parameters():
                    if param.grad is not None:
                        # EM update: expected counts = param.data * param.grad
                        expected_counts = param.data * param.grad.clamp(min=0.0)
                        param.data = expected_counts + 1e-4

                        if param.data.ndim == 6:
                            # Normalize TuckerLayer weights: (G_P, G_L, G_R, h_out, h_l, h_r)
                            param.data /= param.data.sum(dim=(1, 2, 4, 5), keepdim=True).clamp(
                                min=1e-10
                            )
                        else:
                            param.data /= param.data.sum(dim=-1, keepdim=True).clamp(min=1e-10)

                        param.grad.zero_()

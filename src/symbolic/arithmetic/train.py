from typing import Dict, List, Optional

import numpy as np
import torch

from .circuit import SymbolicArithmeticCircuit
from .nodes import (
    GaussianDistribution,
    HadamardProductNode,
    KroneckerProductNode,
    LeafNode,
    ProductNode,
    SumNode,
    UniversalSumNode,
)


class SymbolicEMTrainer:
    """
    Implements Batch Expectation-Maximization for SymbolicArithmeticCircuit.
    """

    def __init__(self, ac: SymbolicArithmeticCircuit):
        self.ac = ac
        self.topo_order = list(ac.topological_sort(reverse=True))  # Leaves to Root
        self.rev_topo_order = self.topo_order[::-1]  # Root to Leaves

    def _get_node_unit_count(self, node_id: int) -> int:
        return self.ac.get_node_data(node_id).unit_count

    def train(
        self, data: torch.Tensor, n_iter: int = 10, batch_size: int = 100, log_interval: int = 1
    ):
        """
        Runs batch EM for n_iter iterations.
        """
        N = data.shape[0]
        for i in range(n_iter):
            # Shuffle data for stochastic batching (optional, here we do full batch)
            indices = torch.randperm(N)
            for start in range(0, N, batch_size):
                batch_data = data[indices[start : start + batch_size]]
                self.em_step(batch_data)
            if (i + 1) % log_interval == 0:
                print(f"Iteration {i + 1}/{n_iter} complete.")

    def em_step(self, data: torch.Tensor):
        B = data.shape[0]
        # 1. E-step: Forward Pass (compute log-probabilities)
        node_outputs = {}  # node_id -> [B, unit_count]

        for node_id in self.topo_order:
            node = self.ac.get_node_data(node_id)
            child_ids = self.ac.get_children(node_id)
            child_outs = [node_outputs[cid] for cid in child_ids]
            node_outputs[node_id] = node.forward(data, child_outs)

        # 2. E-step: Backward Pass (compute responsibilities)
        # messages[node_id] = P(node, x) / P(root, x) * weights...
        # We work in probability space for messages (normalized by root likelihood)
        node_messages = {}  # node_id -> [B, unit_count]

        roots = self.ac.get_roots()
        root_id = roots[0]
        # Initialize root message with 1.0 (normalized)
        node_messages[root_id] = torch.ones(B, 1, device=data.device)

        # Sufficient statistics accumulators
        weight_updates = {}  # sum_node_id -> [unit_count, prods_per_sum]
        gaussian_updates = {}  # leaf_id -> {'sum_m': [unit_count], 'sum_mx': [unit_count], 'sum_mxx': [unit_count]}

        for node_id in self.rev_topo_order:
            m_parent = node_messages[node_id]
            node = self.ac.get_node_data(node_id)
            child_ids = self.ac.get_children(node_id)

            if isinstance(node, (SumNode, UniversalSumNode)):
                # S = sum_j w_j * C_j
                # Message to child C (which is always 1 child - a product block):
                if isinstance(node, SumNode):
                    # m_child = m_parent * (w_j * C_j) / S
                    child_id = child_ids[0]
                    child_out = node_outputs[child_id]
                    h_out = node.unit_count
                    h_in = child_out.shape[1]

                    weights = getattr(node, "weights", None)

                    if isinstance(node, UniversalSumNode):
                        if weights is None:
                            weights = torch.ones(h_out, h_in, device=data.device) / h_in
                            node.weights = weights

                        # rel_prob[b, i, j] = exp(child_prob[b, j] - node_prob[b, i])
                        # log_rel_prob: [B, h_out, h_in]
                        log_rel_prob = child_out.unsqueeze(1) - node_outputs[node_id].unsqueeze(2)
                        m_child_block = m_parent.unsqueeze(2) * (weights.unsqueeze(0) * torch.exp(log_rel_prob))

                        if child_id not in node_messages:
                            node_messages[child_id] = torch.zeros(B, h_in, device=data.device)
                        node_messages[child_id] += m_child_block.sum(dim=1)

                        # Accumulate weight sufficient statistics
                        weight_updates[node_id] = m_child_block.sum(dim=0)
                    else:
                        # Regular partitioned SumNode
                        prods_per_sum = h_in // h_out
                        if weights is None:
                            weights = torch.ones(h_out, prods_per_sum, device=data.device) / prods_per_sum
                            node.weights = weights

                        # numerical stability: use exp(child_out - node_out)
                        rel_prob = torch.exp(
                            child_out.reshape(B, h_out, prods_per_sum) - node_outputs[node_id].unsqueeze(2)
                        )

                        # Message to the child (block)
                        m_child_block = m_parent.unsqueeze(2) * (weights.unsqueeze(0) * rel_prob)

                        if child_id not in node_messages:
                            node_messages[child_id] = torch.zeros(B, h_in, device=data.device)
                        node_messages[child_id] += m_child_block.reshape(B, h_in)

                        # Accumulate weight sufficient statistics
                        weight_updates[node_id] = m_child_block.sum(dim=0)

            elif isinstance(node, ProductNode):
                if isinstance(node, KroneckerProductNode) and len(child_ids) == 2:
                    L_id, R_id = child_ids
                    hL = self._get_node_unit_count(L_id)
                    hR = self._get_node_unit_count(R_id)

                    m_p_reshaped = m_parent.reshape(B, hL, hR)

                    if L_id not in node_messages:
                        node_messages[L_id] = torch.zeros(B, hL, device=data.device)
                    if R_id not in node_messages:
                        node_messages[R_id] = torch.zeros(B, hR, device=data.device)

                    node_messages[L_id] += m_p_reshaped.sum(dim=2)
                    node_messages[R_id] += m_p_reshaped.sum(dim=1)
                else:
                    # HadamardProductNode or single child: pass message through
                    for cid in child_ids:
                        if cid not in node_messages:
                            node_messages[cid] = torch.zeros(
                                B, self._get_node_unit_count(cid), device=data.device
                            )
                        node_messages[cid] += m_parent

            elif isinstance(node, GaussianDistribution):
                # Accumulate stats for mean/std
                # x is data[:, node.var]
                x = data[:, node.var].unsqueeze(1)  # [B, 1]
                m = m_parent  # [B, h]

                if node_id not in gaussian_updates:
                    gaussian_updates[node_id] = {
                        "sum_m": torch.zeros(node.unit_count, device=data.device),
                        "sum_mx": torch.zeros(node.unit_count, device=data.device),
                        "sum_mxx": torch.zeros(node.unit_count, device=data.device),
                    }

                gaussian_updates[node_id]["sum_m"] += m.sum(dim=0)
                gaussian_updates[node_id]["sum_mx"] += (m * x).sum(dim=0)
                gaussian_updates[node_id]["sum_mxx"] += (m * (x**2)).sum(dim=0)

        # 3. M-step: Update Parameters
        for node_id, update in weight_updates.items():
            node = self.ac.get_node_data(node_id)
            # Normalize weights
            new_weights = update / (update.sum(dim=1, keepdim=True) + 1e-12)
            node.weights = new_weights

        for node_id, stats in gaussian_updates.items():
            node = self.ac.get_node_data(node_id)
            sum_m = stats["sum_m"] + 1e-12
            new_mean = stats["sum_mx"] / sum_m
            new_var = (stats["sum_mxx"] / sum_m) - (new_mean**2)

            node.mean = new_mean
            node.stddev = torch.sqrt(new_var.clamp(min=1e-6))

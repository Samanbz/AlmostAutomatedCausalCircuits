import logging
import time

import torch

from src.logger import logger as mcc_logger

from .circuit import SymbolicArithmeticCircuit


logger = mcc_logger.getChild("EMTrainer")
logger.setLevel(logging.INFO)


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    if seconds < 3600.0:
        minutes = int(seconds // 60)
        secs = int(round(seconds % 60))
        return f"{minutes}m {secs:02d}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(round(seconds % 60))
    return f"{hours}h {minutes:02d}m {secs:02d}s"


class SymbolicEMTrainer:
    """
    Implements Batch Expectation-Maximization for SymbolicArithmeticCircuit
    using PyTorch's autograd engine.
    """

    def __init__(self, ac: SymbolicArithmeticCircuit, leaf_lr: float = 0.05):
        self.ac = ac
        self.topo_order = list(ac.topological_sort(reverse=True))

        leaf_params = list(self.ac.leaf_parameters())
        if leaf_params:
            self.optimizer = torch.optim.Adam(leaf_params, lr=leaf_lr)
        else:
            self.optimizer = None

    def train(
        self,
        data: torch.Tensor,
        n_iter: int = 10,
        batch_size: int = 100,
        step_size: float = 1.0,
        decay_rate: float = 1.0,
        decay_every: int = 1,
        log_interval: int = 1,
    ):
        """
        Runs batch EM for n_iter iterations.

        Args:
            data: Training data tensor.
            n_iter: Number of full passes over the dataset.
            batch_size: Mini-batch size.
            step_size: Initial EMA step size (learning rate) for the M-step.
            decay_rate: Multiplicative decay factor applied to step_size every
                ``decay_every`` mini-batch steps. 1.0 means no decay.
            decay_every: Number of mini-batch steps between decay applications.
            log_interval: Print batch NLL every ``log_interval`` mini-batch steps.
        """
        N = data.shape[0]
        current_step_size = step_size
        step_count = 0
        epoch_nll_accum = 0.0
        epoch_batches = 0

        total_epochs = n_iter + 1
        train_start = time.perf_counter()
        ema_epoch_duration = None
        ema_alpha = 0.3

        for epoch in range(total_epochs):
            epoch_start = time.perf_counter()
            indices = torch.randperm(N)

            for start in range(0, N, batch_size):
                batch_data = data[indices[start : start + batch_size]]
                nll = self.em_step(batch_data, current_step_size)
                step_count += 1
                epoch_nll_accum += nll
                epoch_batches += 1

            epoch_duration = time.perf_counter() - epoch_start
            if ema_epoch_duration is None:
                ema_epoch_duration = epoch_duration
            else:
                ema_epoch_duration = (
                    ema_alpha * epoch_duration + (1.0 - ema_alpha) * ema_epoch_duration
                )

            if log_interval > 0 and (epoch % log_interval == 0):
                elapsed = time.perf_counter() - train_start
                remaining_epochs = total_epochs - (epoch + 1)
                eta = ema_epoch_duration * remaining_epochs
                logger.info(
                    "Epoch %d | Step %d | Batch NLL %.4f | step_size %.4f | "
                    "epoch %s | elapsed %s | ETA %s",
                    epoch,
                    step_count,
                    nll,
                    current_step_size,
                    _format_duration(epoch_duration),
                    _format_duration(elapsed),
                    _format_duration(eta),
                )

            if decay_rate < 1.0 and epoch % decay_every == 0:
                current_step_size *= decay_rate

            epoch_nll_accum = 0.0
            epoch_batches = 0

    def em_step(self, data: torch.Tensor, step_size: float = 1.0, smoothing: float = 1e-6):
        """
        Performs a single E-step and M-step update on a Mini-batch.

        Returns:
            float: Negative log-likelihood for this batch.
        """
        # 1. Forward Pass
        from .circuit import eval_circuit

        log_probs = eval_circuit(self.ac, data, verbose=False)

        valid_mask = ~torch.isinf(log_probs).squeeze() & ~torch.isnan(log_probs).squeeze()
        nll = -log_probs[valid_mask].mean().item()

        # 2. Backward Pass
        loss = log_probs[valid_mask].sum()
        if torch.isnan(loss):
            logger.error("Loss is NaN!")

        loss.backward()

        for node_id in self.topo_order:
            node = self.ac.get_node_data(node_id)
            if hasattr(node, "means") and hasattr(node, "stddevs"):
                if torch.isnan(node.means).any():
                    logger.error(f"Node {node_id} means is NaN before M-step!")
                if node.means.grad is not None and torch.isnan(node.means.grad).any():
                    logger.error(f"Node {node_id} means.grad is NaN!")

        # 3. M-step: Delegation
        with torch.no_grad():
            for node_id in self.topo_order:
                node = self.ac.get_node_data(node_id)
                if hasattr(node, "update_params"):
                    node.update_params(data, step_size, smoothing, valid_mask)

        # 4. Gradient step for leaf parameters
        if getattr(self, "optimizer", None) is not None:
            # We want to maximize the sum of log_probs, but standard PyTorch optimizers minimize.
            # The gradients were accumulated from loss.backward() where loss = sum(log_probs).
            # So we negate the gradients for the leaf parameters before stepping.
            for param_group in self.optimizer.param_groups:
                for p in param_group["params"]:
                    if p.grad is not None:
                        p.grad.data.neg_()
            self.optimizer.step()

        # Zero grads for the circuit (tensors on nodes)
        self.ac.zero_grad()
        if getattr(self, "optimizer", None) is not None:
            self.optimizer.zero_grad()

        return nll

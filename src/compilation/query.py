"""
Query API for probabilistic and causal inference on compiled circuits.

All functions accept a CompiledCircuit (TensorizedCircuit or MonarchCircuit)
and return log-space results.
"""

from typing import List

import torch

from src.compilation.tensorized_circuit import CompiledCircuit


def marginal(
    circuit: CompiledCircuit,
    data: torch.Tensor,
    marginalize_vars: List[int],
) -> torch.Tensor:
    """Compute log P(X_{observed}) by marginalizing out specified variables.

    Sets the given variable columns to NaN so the circuit treats them as
    marginalized (log-probability contribution = 0).

    Args:
        circuit: compiled circuit.
        data: [B, num_features] observation tensor.
        marginalize_vars: variable indices to marginalize out.

    Returns:
        [B] log marginal probabilities.
    """
    d = data.clone()
    for v in marginalize_vars:
        d[:, v] = float("nan")
    return circuit(d)

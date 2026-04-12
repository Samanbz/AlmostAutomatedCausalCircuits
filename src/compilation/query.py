from typing import List, Optional

import torch
from torch import nn


def marginal(
    circuit: nn.Module, data: torch.Tensor, marg_vars: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Computes the log marginal probability of the data.

    If marg_vars is provided, those variables are marginalized out by setting their
    values to NaN before evaluating the circuit. If marg_vars is None, the function
    assumes that the input data already contains NaNs for variables that should
    be marginalized.

    Args:
        circuit: A compiled tensorized circuit.
        data: The input tensor of shape (batch_size, num_features).
        marg_vars: An optional list of feature indices to marginalize out.

    Returns:
        torch.Tensor: The log marginal probability for each sample in the batch.
    """
    if marg_vars is not None and len(marg_vars) > 0:
        data = data.clone()
        data[:, marg_vars] = float("nan")

    return circuit(data)


def conditional(
    circuit: nn.Module,
    data: torch.Tensor,
    query_vars: List[int],
    evidence_vars: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Computes the log conditional probability log P(X_Q | X_E).

    All variables not present in query_vars or evidence_vars are implicitly integrated out.
    If evidence_vars is None, any variable in the data that is not in query_vars and
    is not fully NaN is considered evidence.

    Args:
        circuit: A compiled tensorized circuit.
        data: The input tensor of shape (batch_size, num_features).
        query_vars: List of indices for the query variables.
        evidence_vars: Optional list of indices for the evidence variables.

    Returns:
        torch.Tensor: The log conditional probability for each sample in the batch.
    """
    num_vars = data.shape[1]

    if evidence_vars is None:
        # Infer evidence variables as all non-query variables that are not completely NaN
        evidence_vars = []
        for i in range(num_vars):
            if i not in query_vars and not torch.isnan(data[:, i]).all():
                evidence_vars.append(i)

    # Variables to marginalize for the joint P(X_Q, X_E)
    marg_joint = [i for i in range(num_vars) if i not in query_vars and i not in evidence_vars]

    # Variables to marginalize for the evidence P(X_E)
    marg_evidence = [i for i in range(num_vars) if i not in evidence_vars]

    log_prob_joint = marginal(circuit, data, marg_vars=marg_joint)
    log_prob_evidence = marginal(circuit, data, marg_vars=marg_evidence)

    # log P(X_Q | X_E) = log P(X_Q, X_E) - log P(X_E)
    return log_prob_joint - log_prob_evidence

import torch


def estimate_pairwise_mi(
    data: torch.Tensor, edges: list[tuple[int, int]] | None = None
) -> dict[tuple[int, int], float]:
    """
    Computes pairwise mutual information for given edges using a Gaussian approximation
    (Pearson correlation). This is an ultra-fast GPU-friendly heuristic to guide VTree formation.

    Args:
        data: Tensor of shape (num_samples, num_variables)
        edges: Optional list of (u, v) tuples specifying which edges to evaluate.
               If None, calculates for all pairs.

    Returns:
        Dictionary mapping (u, v) to an empirical mutual information score.
    """
    n_vars = data.shape[1]

    # Standardize data
    data_mean = data.mean(dim=0, keepdim=True)
    data_std = data.std(dim=0, unbiased=False, keepdim=True)

    # Avoid division by zero
    data_std = torch.where(data_std == 0, torch.ones_like(data_std), data_std)
    z = (data - data_mean) / data_std

    # Compute correlation matrix R
    n_samples = data.shape[0]
    R = (z.T @ z) / n_samples

    # Clamp to avoid numerical issues near +/- 1.0 before log
    R_clamped = torch.clamp(R, -0.999, 0.999)

    # MI approximation for bivariate normal
    MI_matrix = -0.5 * torch.log(1.0 - R_clamped**2 + 1e-12)

    mi_dict = {}
    if edges is None:
        edges = [(i, j) for i in range(n_vars) for j in range(i + 1, n_vars)]

    # Extract specifically requested edges
    # Can also be vectorised safely since we have the full matrix
    for u, v in edges:
        mi_val = MI_matrix[u, v].item()

        # Ensure non-negative
        mi_dict[(u, v)] = max(0.0, mi_val)
        mi_dict[(v, u)] = max(0.0, mi_val)

    return mi_dict

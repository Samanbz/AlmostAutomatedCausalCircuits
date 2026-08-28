import numpy as np
import torch

from src.symbolic.arithmetic.nodes import GaussianDistribution


def test_gaussian_leaf_truncated_integration():
    """
    Test that a GaussianDistribution leaf with disjoint truncated supports
    numerically integrates to 1.0 for each unit independently.
    """
    var = 0
    h = 4

    # Initialize a multi-unit Gaussian
    dist = GaussianDistribution(
        var=var,
        base_mean=0.0,
        base_stddev=1.0,
    )

    # Create disjoint intervals using split_support
    # This splits the real line into 'h' equiprobable intervals for each unit based on its mean/std
    # Actually split_support uses the mean of the means and stds across units in the current implementation.
    intervals = dist.split_support(split_count=h)

    from src.symbolic.arithmetic.nodes.leaf_layer import GaussianLeafLayer
    node_supports = []
    from src.utils import Support

    for i in range(h):
        node_supports.append(Support({var: intervals[i]}))

    leaf = GaussianLeafLayer(dist, num_nodes=h, num_groups=1, node_supports=node_supports)

    # Numerically integrate each unit using Monte Carlo
    # We sample uniformly over a large interval covering the supports
    # Since intervals can be -inf to inf, we sample widely using a normal proposal
    N = 100000
    torch.manual_seed(42)
    np.random.seed(42)

    proposal_mean = 0.0
    proposal_std = 10.0
    samples = torch.randn(N, 1) * proposal_std + proposal_mean

    # Construct data tensor
    data = torch.zeros(N, 1)
    data[:, var] = samples.squeeze()

    # Forward pass
    log_probs = leaf.forward(data)  # shape: [N, 1, h]
    log_probs = log_probs.squeeze(1)  # shape: [N, h]

    # Log proposal density
    proposal_log_probs = (
        -0.5 * np.log(2 * np.pi * proposal_std**2)
        - 0.5 * ((samples - proposal_mean) / proposal_std) ** 2
    )

    log_integrand = log_probs - proposal_log_probs

    # Integrate each unit
    log_integrals = torch.logsumexp(log_integrand, dim=0) - np.log(N)
    integrals = torch.exp(log_integrals)

    print("\nUnit Integrals:", integrals.tolist())

    # Assert each unit integrates to 1.0 within MC variance
    for i in range(h):
        assert torch.isclose(integrals[i], torch.tensor(1.0), atol=0.05), (
            f"Unit {i} integrates to {integrals[i].item():.4f}, expected ~1.0"
        )

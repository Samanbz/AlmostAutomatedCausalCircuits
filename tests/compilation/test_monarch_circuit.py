import numpy as np
import pytest
import torch

from src.compilation.folded_circuit import CircuitFolder
from src.compilation.fused_circuit import CircuitFuser
from src.compilation.monarch_circuit import MonarchCircuit
from src.compilation.query import marginal
from src.compilation.tensorized_circuit import TensorizedCircuit
from src.construction import (
    CircuitBuilder,
    RegionGraphBuilder,
    construct_optimal_md_vtree,
)
from src.symbolic import GaussianDistribution
from src.symbolic.scm import (
    AdditiveNoiseMechanism,
    GaussianNoise,
    LinearLogic,
    StructuralCausalModel,
)


@pytest.fixture
def minimal_scm():
    """Minimal SCM X -> Z, X <- Y -> Z with linear Gaussian mechanisms, for testing queries against ground truth."""
    scm = StructuralCausalModel()

    scm.add_variable(
        "1",
        mechanism=AdditiveNoiseMechanism(logic=None, noise_dist=GaussianNoise(0.0, 1.0)),
        parents=[],
    )
    scm.add_variable(
        "0",
        mechanism=AdditiveNoiseMechanism(
            logic=LinearLogic({"1": 2.0}), noise_dist=GaussianNoise(0.0, 1.0)
        ),
        parents=["1"],
    )
    scm.add_variable(
        "2",
        mechanism=AdditiveNoiseMechanism(
            logic=LinearLogic({"0": 2.0, "1": 1.5}), noise_dist=GaussianNoise(0.0, 1.0)
        ),
        parents=["0", "1"],
    )
    return scm


@pytest.fixture
def compiled_circuits(minimal_scm):
    torch.manual_seed(42)
    np.random.seed(42)

    data = minimal_scm.sample(100)
    data_tensor = torch.from_numpy(data.values.copy()).float()

    md_vtree = construct_optimal_md_vtree(data_tensor, md_sets=[])
    input_dists = {v: GaussianDistribution(var=v, mean=0.0, stddev=1.0) for v in range(3)}
    region_graph = RegionGraphBuilder(md_vtree=md_vtree, input_dists=input_dists).build()
    spn = CircuitBuilder(rg=region_graph, h=8, input_dists=input_dists).build()

    folded = CircuitFolder(spn).build()
    fused = CircuitFuser(folded).build()

    tc = TensorizedCircuit(fused)
    mc = MonarchCircuit(fused)
    return tc, mc, minimal_scm


def test_monarch_and_tensorized_similar(compiled_circuits):
    """
    1. both monarch and tensorized circuit return negative log likelihoods which are similar
    """
    tc, mc, scm = compiled_circuits
    sample = torch.from_numpy(scm.sample(16).values.copy()).float()

    with torch.no_grad():
        ll_tc = tc(sample)
        ll_mc = mc(sample)

    assert ll_tc.shape == (16,)
    assert ll_mc.shape == (16,)

    assert (ll_tc < 0).all(), "Tensorized Circuit should return negative log-likelihoods"
    assert (ll_mc < 0).all(), "Monarch Circuit should return negative log-likelihoods"

    # Check similarity. Both should roughly be in the same log probability regime.
    # Due to random init, they aren't exactly equal, but the means should be reasonably close.
    mean_diff = torch.abs(ll_tc.mean() - ll_mc.mean())
    assert mean_diff < 5.0, (
        f"Mean log-likelihoods diverge too much: tc={ll_tc.mean():.4f}, mc={ll_mc.mean():.4f}"
    )


def test_em_increases_log_likelihood(compiled_circuits):
    """
    2. log likelihood increases after EM (full batch)
    """
    tc, mc, scm = compiled_circuits
    train_data = torch.from_numpy(scm.sample(256).values.copy()).float()

    # TC EM step
    with torch.no_grad():
        ll_tc_initial = tc(train_data).mean().item()
    tc.em_step(train_data, step_size=1.0)  # Full batch
    with torch.no_grad():
        ll_tc_after = tc(train_data).mean().item()
    assert ll_tc_after > ll_tc_initial, "TensorizedCircuit EM should increase log-likelihood"

    # MC EM step
    with torch.no_grad():
        ll_mc_initial = mc(train_data).mean().item()
    mc.em_step(train_data, step_size=1.0)  # Full batch
    with torch.no_grad():
        ll_mc_after = mc(train_data).mean().item()
    assert ll_mc_after > ll_mc_initial, "MonarchCircuit EM should increase log-likelihood"


def test_queries_verified_against_ground_truth(compiled_circuits):
    """
    3. marginal, conditional and backdoor queries verified against ground truth.
    Using minimal SCM X(0) -> Y(1) -> Z(2).
    Don't rely on the circuit for ground truth, rather on the scm and count manually!
    """
    tc, mc, scm = compiled_circuits

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    tc = tc.to(device)
    mc = mc.to(device)
    # Train both circuits for a few steps to fit the SCM reasonably well
    train_data = torch.from_numpy(scm.sample(2**20).values.copy()).float().to(device)
    print(f"\nStarting training on device: {device}")

    epochs = 15
    for i in range(epochs):
        tc_epoch_ll_sum = 0.0
        mc_epoch_ll_sum = 0.0
        for batch in torch.split(train_data, 2**14):
            tc.em_step(batch, step_size=0.05)
            mc.em_step(batch, step_size=0.05)
            tc_epoch_ll_sum += tc(batch).sum().item()
            mc_epoch_ll_sum += mc(batch).sum().item()

        tc_epoch_mean_ll = tc_epoch_ll_sum / len(train_data)
        mc_epoch_mean_ll = mc_epoch_ll_sum / len(train_data)
        print(
            f"Epoch {i + 1}/{epochs}: TC mean LL = {tc_epoch_mean_ll:.4f}, MC mean LL = {mc_epoch_mean_ll:.4f}"
        )

    # Generate a large dataset to "count manually"
    N_test = 20000000
    eps = 0.05

    # Query point
    x_q, y_q, z_q = 0.5, 0.5, 0.5
    sample = torch.tensor([[x_q, y_q, z_q]]).to(device).float()

    # Column ordering in SCM: "1" corresponds to var 0, "0" corresponds to var 1, "2" corresponds to var 2.
    # Therefore, variable "0" is X, "1" is Y, "2" is Z.
    log_density_xy_counted = scm.empirical_log_density(
        query={"0": x_q, "1": y_q}, n_samples=N_test, eps=eps
    )

    log_cond_counted = scm.empirical_log_density(
        query={"2": z_q},
        evidence={"1": y_q},  # evidence on Y (var 0, "1" in SCM)
        n_samples=N_test,
        eps=eps,
    )

    def _verify_circuit_queries(circuit, name, tolerance=0.4):
        with torch.no_grad():
            # Marginal P(X, Y)
            ll_marginal_xy = marginal(circuit, sample, marginalize_vars=[2])[0].item()
            marg_diff = abs(ll_marginal_xy - log_density_xy_counted)
            # It's an approximation, so tolerance must be slightly loose
            print(
                f"{name}: Marginal P(X,Y) API {ll_marginal_xy:.4f} vs counted {log_density_xy_counted:.4f}"
            )
            assert marg_diff < tolerance, (
                f"{name}: Marginal P(X,Y) API {ll_marginal_xy:.4f} vs counted {log_density_xy_counted:.4f}"
            )

            # Conditional P(Z | Y)
            ll_cond_zy = conditional(circuit, sample, query_vars=[2], evidence_vars=[0])[0].item()
            cond_diff = abs(ll_cond_zy - log_cond_counted)
            print(
                f"{name}: Conditional P(Z|Y) API {ll_cond_zy:.4f} vs counted {log_cond_counted:.4f}"
            )
            assert cond_diff < tolerance, (
                f"{name}: Conditional P(Z|Y) API {ll_cond_zy:.4f} vs counted {log_cond_counted:.4f}"
            )

    _verify_circuit_queries(tc, "TensorizedCircuit", tolerance=0.4)
    _verify_circuit_queries(mc, "MonarchCircuit", tolerance=0.4)

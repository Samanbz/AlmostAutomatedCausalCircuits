import math

import numpy as np
import torch

from src.compilation.query import backdoor
from src.compilation.tensorized_circuit import TensorizedCircuit
from src.construction.learned_vtree import construct_optimal_md_vtree
from src.construction.xpc import (
    construct_random_md_data_region_graph,
    construct_spn_from_region_graph,
)
from src.symbolic import CategoricalDistribution
from src.symbolic.scm import AdditiveNoiseMechanism, StructuralCausalModel


def test_backdoor_xpc_scm():
    scm = StructuralCausalModel()

    scm.add_variable(
        "Z",
        mechanism=AdditiveNoiseMechanism(
            logic=None, noise_dist=lambda n: np.random.binomial(1, 0.5, size=n)
        ),
    )

    def x_logic(**kwargs):
        return kwargs["Z"]

    scm.add_variable(
        "X",
        mechanism=AdditiveNoiseMechanism(
            logic=x_logic, noise_dist=lambda n: np.random.binomial(1, 0.1, size=n)
        ),
        parents=["Z"],
    )

    def q_logic(**kwargs):
        x, z = kwargs["X"], kwargs["Z"]
        return np.logical_xor(x, z).astype(int)

    scm.add_variable(
        "Q",
        mechanism=AdditiveNoiseMechanism(
            logic=q_logic, noise_dist=lambda n: np.random.binomial(1, 0.1, size=n)
        ),
        parents=["X", "Z"],
    )

    data_df = scm.sample(500)
    data_np = (data_df[["X", "Z", "Q"]].values % 2).astype(np.float32)
    data_torch = torch.tensor(data_np)

    input_dists = {i: CategoricalDistribution(i, [0, 1], [0.5, 0.5]) for i in range(3)}

    do_vars, z_vars, query_vars = [0], [1], [2]

    md_vtree = construct_optimal_md_vtree(data_torch, md_sets=[{0, 1}], prioritize="hardware")

    rg = construct_random_md_data_region_graph(
        data=data_np,
        input_dists=input_dists,
        min_examples=10,
        split_arity=2,
        md_var_decomp=md_vtree,
    )

    circuit = construct_spn_from_region_graph(rg, input_dists=input_dists)

    tc = TensorizedCircuit(circuit)

    d_q = torch.tensor([[0.0, float("nan"), 0.0]])

    tc.set_target_vars(set())

    def p_joint(x, z, q):
        d = torch.tensor([[x, z, q]], dtype=torch.float32)
        return torch.exp(tc(d)).item()

    def p_xz(x, z):
        return p_joint(x, z, 0) + p_joint(x, z, 1)

    def p_z(z):
        return p_xz(0, z) + p_xz(1, z)

    expected_prob = 0.0
    for z in [0, 1]:
        pxz = p_xz(0, z)
        pz = p_z(z)
        if pxz > 0:
            expected_prob += (p_joint(0, z, 0) / pxz) * pz

    expected_log_prob = math.log(expected_prob) if expected_prob > 0 else float("-inf")

    res = backdoor(tc, d_q, do_vars=do_vars, z_vars=z_vars, query_vars=query_vars)

    print(f"\nBackdoor XPC SCM Result: {res.item():.5f} | Expected: {expected_log_prob:.5f}")
    assert np.isfinite(res.item())


if __name__ == "__main__":
    test_backdoor_xpc_scm()

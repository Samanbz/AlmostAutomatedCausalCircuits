import numpy as np
import pytest
import torch

from src.construction.circuit_builder import create_md_circuit
from src.symbolic.arithmetic.circuit import eval_circuit
from src.symbolic.arithmetic.nodes import GaussianDistribution
from src.symbolic.arithmetic.query import _instantiate, compile_query
from src.symbolic.arithmetic.train import SymbolicEMTrainer
from src.symbolic.id_ast import make_p
from src.symbolic.scm import AdditiveNoiseMechanism, StructuralCausalModel
from src.symbolic.vtree import VNode, VTree
from src.utils import BitSet


def test_em_training_basic():
    # 1. Generate synthetic data from Z -> X
    # Z ~ N(0, 1), X = Z + N(0, 0.1)
    # We will use 2 components for Z (one for Z < 0, one for Z > 0)
    # to see if EM can learn the mixture.

    np.random.seed(42)
    torch.manual_seed(42)

    N = 1000
    z_data = np.random.normal(0, 1, N)
    x_data = z_data + np.random.normal(0, 0.1, N)
    data = torch.tensor(np.stack([z_data, x_data], axis=1), dtype=torch.float32)

    # 2. Build VTree
    # Z is in MD-set
    vt = VTree()
    id_z = vt.add_node(VNode(BitSet([0]), md_set=BitSet([0])))
    id_x = vt.add_node(VNode(BitSet([1]), md_set=BitSet()))
    id_root = vt.add_node(VNode(BitSet([0, 1]), md_set=BitSet([0])))
    vt.add_children(id_root, id_z, id_x)

    # 3. Initialize Circuit
    # We use 2 units for Z.
    dists = {
        0: GaussianDistribution(var=0, mean=torch.tensor([-1.0, 1.0]), stddev=0.5, unit_count=2),
        1: GaussianDistribution(var=1, mean=torch.tensor([0.0, 0.0]), stddev=1.0, unit_count=2),
    }

    ac = create_md_circuit(dists, vt, h=2, initialize_weights=True)

    # 4. Initial Likelihood
    def get_avg_ll(circuit, dataset):
        roots = circuit.get_roots()
        log_probs = eval_circuit(circuit, dataset)
        return log_probs.mean().item()

    initial_ll = get_avg_ll(ac, data)
    print(f"Initial LL: {initial_ll:.4f}")

    # 5. Train
    trainer = SymbolicEMTrainer(ac)
    # We expect Z to separate the units.
    # Unit 0 should capture Z < 0, Unit 1 should capture Z > 0.
    # Then X mean should become ~ -1 for unit 0 and ~ 1 for unit 1.

    trainer.train(data, n_iter=20, batch_size=200)

    final_ll = get_avg_ll(ac, data)
    print(f"Final LL: {final_ll:.4f}")

    assert final_ll > initial_ll

    # Check learned means
    leaf_ids = [
        nid
        for nid in ac.topological_sort()
        if isinstance(ac.get_node_data(nid), GaussianDistribution)
    ]
    z_leaf = None
    x_leaf = None
    for lid in leaf_ids:
        node = ac.get_node_data(lid)
        if node.var == 0:
            z_leaf = node
        if node.var == 1:
            x_leaf = node

    print(f"Z Means: {z_leaf.mean}")
    print(f"X Means: {x_leaf.mean}")

    # Z means should be roughly -0.8 and 0.8 (expected value of N(0,1) given Z<0 or Z>0)
    # and X means should follow Z.
    assert torch.abs(z_leaf.mean[0] + z_leaf.mean[1]) < 0.5  # Symmetry
    assert torch.abs(x_leaf.mean[0] - z_leaf.mean[0]) < 0.3
    assert torch.abs(x_leaf.mean[1] - z_leaf.mean[1]) < 0.3

    # 6. Verify Causal Query: P(Z | do(X)) vs P(Z | X)
    # Since Z -> X, P(Z | do(X)) should be P(Z) = N(0, 1)
    # P(Z | X) should be biased towards X.

    scm = StructuralCausalModel()
    scm.add_variable("Z", AdditiveNoiseMechanism(None, lambda n: np.random.normal(0, 1, n)))
    scm.add_variable(
        "X",
        AdditiveNoiseMechanism(lambda Z: Z, lambda n: np.random.normal(0, 0.1, n)),
        parents=["Z"],
    )

    var_to_id = {"Z": 0, "X": 1}
    P_ast = make_p({"Z", "X"})

    # Interventional P(Z | do(X))
    from src.symbolic.identification import identify

    ast_do, _, _ = identify({"Z"}, {"X"}, P_ast, scm)
    q_do_ac, _ = compile_query(ast_do, ac, -1, var_to_id)

    # Observational P(Z | X)
    ast_obs_joint, _, _ = identify({"Z", "X"}, set(), P_ast, scm)
    ast_obs_marg, _, _ = identify({"X"}, set(), P_ast, scm)
    q_zx_ac, _ = compile_query(ast_obs_joint, ac, -1, var_to_id)
    q_x_ac, _ = compile_query(ast_obs_marg, ac, -1, var_to_id)

    # Evaluate at X=2.0
    x_val = 2.0
    q_do_x2 = _instantiate(q_do_ac, {1: x_val})
    q_zx_x2 = _instantiate(q_zx_ac, {1: x_val})
    q_x_x2 = _instantiate(q_x_ac, {1: x_val})

    z_vals = torch.linspace(-3, 3, 100).unsqueeze(1)
    eval_data = torch.zeros(100, 2)
    eval_data[:, 0] = z_vals.squeeze()

    prob_do = torch.exp(eval_circuit(q_do_x2, eval_data))[:, 0]
    prob_obs = torch.exp(
        eval_circuit(q_zx_x2, eval_data)[:, 0] - eval_circuit(q_x_x2, eval_data)[:, 0]
    )

    # Normalize
    dz = z_vals[1] - z_vals[0]
    prob_do /= prob_do.sum() * dz
    prob_obs /= prob_obs.sum() * dz

    # In P(Z|do(X=2)), Z should be centered at 0
    mean_do = ((z_vals.squeeze() * prob_do).sum() * dz).item()
    # In P(Z|X=2), Z should be centered near 2
    mean_obs = ((z_vals.squeeze() * prob_obs).sum() * dz).item()

    print(f"Mean P(Z | do(X=2)): {mean_do:.4f}")
    print(f"Mean P(Z | X=2): {mean_obs:.4f}")

    assert abs(mean_do) < 0.5
    assert mean_obs > 0.5


if __name__ == "__main__":
    test_em_training_basic()

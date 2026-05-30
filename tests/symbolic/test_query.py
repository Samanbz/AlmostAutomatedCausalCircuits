import copy

import numpy as np
import pytest
import scipy.stats
import torch

from src.construction.circuit_builder import create_md_circuit
from src.construction.learned_vtree import construct_optimal_md_vtree
from src.construction.random_scm import generate_random_scm
from src.symbolic.arithmetic.circuit import SymbolicArithmeticCircuit
from src.symbolic.arithmetic.nodes import (
    CartesianLeafNode,
    ConstantLeafNode,
    GaussianDistribution,
    InverseLeafNode,
    KroneckerProductNode,
    SumNode,
    UniversalSumNode,
)
from src.symbolic.arithmetic.query import (
    InstantiatedLeafNode,
    _instantiate,
    _inverse,
    _marginalize,
    _multiply,
    compile_query,
)
from src.symbolic.arithmetic.train import SymbolicEMTrainer
from src.symbolic.id_ast import ast_to_str, make_p
from src.symbolic.identification import identify
from src.symbolic.scm import AdditiveNoiseMechanism, StructuralCausalModel
from src.symbolic.vtree import VNode, VTree
from src.utils import BitSet


def eval_circuit(ac: SymbolicArithmeticCircuit, data: torch.Tensor) -> torch.Tensor:
    """Evaluates the circuit on the given data."""
    outputs = {}
    for node_id in ac.topological_sort(reverse=True):
        node = ac.get_node_data(node_id)
        child_ids = ac.get_children(node_id)
        child_outs = [outputs[cid] for cid in child_ids]
        outputs[node_id] = node.forward(data, child_outs)

    roots = ac.get_roots()
    assert len(roots) == 1
    return outputs[roots[0]]


@pytest.fixture
def simple_ac():
    """A very basic valid circuit with a single leaf."""
    ac = SymbolicArithmeticCircuit()
    leaf = GaussianDistribution(var=0, mean=0.0, stddev=1.0)
    leaf.md_set = BitSet([0])
    ac.add_node(leaf)
    return ac


@pytest.fixture
def complex_ac():
    # 4 vars, h=2. Build a full PC structure.
    scm = generate_random_scm(n_nodes=4)
    data = torch.from_numpy(scm.sample(100).values).float()

    # Assume MD set is just {{0}, {1}, {2}, {3}} -> essentially full factorization requested
    md_sets = [{0}, {1}, {2}, {3}]
    md_vtree = construct_optimal_md_vtree(data, md_sets=md_sets)

    input_dists = {v: GaussianDistribution(var=v, mean=0.0, stddev=1.0) for v in range(4)}

    ac = create_md_circuit(
        input_dists=input_dists,
        md_var_decomp=md_vtree,
        h=2,
        h_max=4,
    )
    return ac


def test_inverse(simple_ac):
    new_ac = _inverse(simple_ac)
    data = torch.randn(5, 1)  # simple_ac only has var 0

    orig_out = eval_circuit(simple_ac, data)
    inv_out = eval_circuit(new_ac, data)

    assert torch.allclose(inv_out, -orig_out, atol=1e-4)


def test_multiply_disjoint(simple_ac):
    ac1 = simple_ac
    ac2 = SymbolicArithmeticCircuit()
    leaf2 = GaussianDistribution(var=1, mean=5.0, stddev=2.0)
    leaf2.md_set = BitSet([1])
    ac2.add_node(leaf2)

    new_ac = _multiply(ac1, ac2)
    data = torch.randn(5, 2)

    out1 = eval_circuit(ac1, data)
    out2 = eval_circuit(ac2, data)
    mul_out = eval_circuit(new_ac, data)

    expected = (out1.unsqueeze(2) + out2.unsqueeze(1)).reshape(5, -1)
    assert torch.allclose(mul_out, expected, atol=1e-5)


def test_multiply_leaves(simple_ac):
    ac1 = simple_ac
    ac2 = SymbolicArithmeticCircuit()
    leaf2 = GaussianDistribution(var=0, mean=5.0, stddev=2.0)
    leaf2.md_set = BitSet([0])
    ac2.add_node(leaf2)

    new_ac = _multiply(ac1, ac2)
    data = torch.randn(5, 1)

    out1 = eval_circuit(ac1, data)
    out2 = eval_circuit(ac2, data)
    mul_out = eval_circuit(new_ac, data)

    expected = (out1.unsqueeze(2) + out2.unsqueeze(1)).reshape(5, -1)
    assert torch.allclose(mul_out, expected, atol=1e-5)


def test_marginalize(complex_ac):
    marg_vars = {1, 2}
    new_ac = _marginalize(complex_ac, marg_vars)

    data = torch.randn(10, 4)
    out = eval_circuit(new_ac, data)
    assert out.shape[0] == 10


def test_inverse_complex(complex_ac):
    new_ac = _inverse(complex_ac)

    data = torch.randn(10, 4)
    out = eval_circuit(new_ac, data)
    assert out.shape[0] == 10


def test_multiply_complex(complex_ac):
    # This tests the recursive structural expansion
    new_ac = _multiply(complex_ac, complex_ac)

    data = torch.randn(10, 4)
    out_orig = eval_circuit(complex_ac, data)
    out_new = eval_circuit(new_ac, data)

    assert torch.allclose(out_new, 2 * out_orig, atol=1e-4)


def test_multiply_complex_disjoint_manual():
    scm = generate_random_scm(n_nodes=4)
    data = torch.from_numpy(scm.sample(100).values).float()
    vt = construct_optimal_md_vtree(data, md_sets=[{0}, {1}, {2}, {3}])

    dists1 = {v: GaussianDistribution(var=v, mean=0.0, stddev=1.0) for v in range(4)}
    ac1 = create_md_circuit(dists1, vt, h=2)

    dists2 = {v + 4: GaussianDistribution(var=v + 4, mean=0.0, stddev=1.0) for v in range(4)}

    def shift_scope(vtree_obj, node_id):
        node = vtree_obj.get_node_data(node_id)
        node.scope = BitSet([i + 4 for i in node.scope])
        if node.md_set and not node.md_set.is_universal:
            node.md_set = BitSet([i + 4 for i in node.md_set])
        for cid in vtree_obj.get_children(node_id):
            shift_scope(vtree_obj, cid)

    vt_shifted = copy.deepcopy(vt)
    shift_scope(vt_shifted, vt_shifted.get_root())

    ac2 = create_md_circuit(dists2, vt_shifted, h=2)

    new_ac = _multiply(ac1, ac2)

    data_full = torch.randn(10, 8)
    out1 = eval_circuit(ac1, data_full)
    out2 = eval_circuit(ac2, data_full)
    out_new = eval_circuit(new_ac, data_full)

    expected = (out1.unsqueeze(2) + out2.unsqueeze(1)).reshape(10, -1)
    assert torch.allclose(out_new, expected, atol=1e-4)


def test_instantiate(complex_ac):
    new_ac = _instantiate(complex_ac, {0: 1.5, 3: -2.0})
    data = torch.randn(10, 4)

    out_inst = eval_circuit(new_ac, data)

    data_clamped = data.clone()
    data_clamped[:, 0] = 1.5
    data_clamped[:, 3] = -2.0
    out_orig_clamped = eval_circuit(complex_ac, data_clamped)

    assert torch.allclose(out_inst, out_orig_clamped, atol=1e-5)


def test_compositional_conditional():
    scm = generate_random_scm(n_nodes=4)
    data = torch.from_numpy(scm.sample(100).values).float()
    vt = construct_optimal_md_vtree(data, md_sets=[{0}, {1}, {2}, {3}])
    dists = {v: GaussianDistribution(var=v, mean=0.0, stddev=1.0) for v in range(4)}

    p_xyz_ac = create_md_circuit(dists, vt, h=2, h_max=4)
    p_xy_ac = _marginalize(p_xyz_ac, {2, 3})
    p_x_ac = _marginalize(p_xyz_ac, {1, 2, 3})
    inv_p_x_ac = _inverse(p_x_ac)
    p_y_given_x_ac = _multiply(p_xy_ac, inv_p_x_ac)

    test_data = torch.randn(15, 4)
    out_p_xy = eval_circuit(p_xy_ac, test_data)
    out_p_x = eval_circuit(p_x_ac, test_data)
    expected_conditional = out_p_xy - out_p_x  # log space

    out_conditional_circuit = eval_circuit(p_y_given_x_ac, test_data)
    assert torch.allclose(out_conditional_circuit, expected_conditional, atol=1e-4)


def test_em_training():
    # 1. Simple Case: P(X, Y) where X and Y are independent, learn parameters
    torch.manual_seed(42)
    n_samples = 1000
    # True dist: X ~ N(5, 1), Y ~ N(-2, 1)
    x_data = 5.0 + 1.0 * torch.randn(n_samples, 1)
    y_data = -2.0 + 1.0 * torch.randn(n_samples, 1)
    data = torch.cat([x_data, y_data], dim=1)

    # 2. Build circuit
    ac = SymbolicArithmeticCircuit()
    # Var 0 (X) has 2 units (mixture)
    l1 = GaussianDistribution(var=0, mean=torch.tensor([0.0, 10.0]), stddev=1.0, unit_count=2)
    id1 = ac.add_node(l1)

    # Var 1 (Y) has 1 unit
    l2 = GaussianDistribution(var=1, mean=0.0, stddev=1.0, unit_count=1)
    id2 = ac.add_node(l2)

    # Kronecker Product of X and Y: unit_count = 2 * 1 = 2
    p = KroneckerProductNode(support=l1.support, unit_count=2)
    id_p = ac.add_node(p)
    ac.add_edge(id_p, id1)
    ac.add_edge(id_p, id2)

    # Sum Node to mix the 2 units into 1 output
    sum_node = SumNode(support=l1.support, unit_count=1)
    sum_node.weights = torch.tensor([[0.5, 0.5]])
    root_id = ac.add_node(sum_node)
    ac.add_edge(root_id, id_p)

    # 3. Train
    trainer = SymbolicEMTrainer(ac)
    trainer.train(data, n_iter=20, batch_size=200)

    # 4. Verify
    # The learned mean of X (weighted mixture) should be close to 5.0
    final_weights = sum_node.weights[0].numpy()
    learned_mean_x = final_weights[0] * l1.mean[0].item() + final_weights[1] * l1.mean[1].item()
    learned_mean_y = l2.mean.item()

    print(f"Learned X Mean: {learned_mean_x:.4f} (True: 5.0)")
    print(f"Learned Y Mean: {learned_mean_y:.4f} (True: -2.0)")
    assert abs(learned_mean_x - 5.0) < 0.5
    assert abs(learned_mean_y - (-2.0)) < 0.5


def test_compositional_backdoor():
    # 1. Define Backdoor SCM: {Z1, Z2} -> X, {Z1, Z2} -> Y, X -> Y
    scm = StructuralCausalModel()
    scm.add_variable("Z1", AdditiveNoiseMechanism(None, lambda n: np.random.normal(0, 1.0, n)))
    scm.add_variable("Z2", AdditiveNoiseMechanism(None, lambda n: np.random.normal(0, 1.0, n)))
    scm.add_variable(
        "X",
        AdditiveNoiseMechanism(lambda Z1, Z2: Z1 - Z2, lambda n: np.random.normal(0, 0.1, n)),
        parents=["Z1", "Z2"],
    )
    scm.add_variable(
        "Y",
        AdditiveNoiseMechanism(
            lambda X, Z1, Z2: X + Z1 + Z2, lambda n: np.random.normal(0, 0.1, n)
        ),
        parents=["X", "Z1", "Z2"],
    )

    var_to_id = {"Z1": 0, "Z2": 1, "X": 2, "Y": 3}

    # 2. Build VTree forcing all confounders to mix with X
    vt = VTree()
    id_z1 = vt.add_node(VNode(BitSet([0]), md_set=BitSet([0])))
    id_z2 = vt.add_node(VNode(BitSet([1]), md_set=BitSet([1])))
    id_x = vt.add_node(VNode(BitSet([2]), md_set=BitSet([2])))
    id_y = vt.add_node(VNode(BitSet([3]), md_set=BitSet([3])))

    # Mix Z1, Z2
    id_z12 = vt.add_node(VNode(BitSet([0, 1]), md_set=BitSet([0, 1])))
    vt.add_children(id_z12, id_z1, id_z2)
    # Mix Z12, X
    id_z12x = vt.add_node(VNode(BitSet([0, 1, 2]), md_set=BitSet([0, 1])))
    vt.add_children(id_z12x, id_z12, id_x)
    # Mix all with Y
    id_root = vt.add_node(VNode(BitSet([0, 1, 2, 3]), md_set=BitSet([0, 1])))
    vt.add_children(id_root, id_z12x, id_y)

    md_vtree = vt

    # Initialize circuit with reasonable parameters
    mean_z1 = torch.tensor([-1.0, 1.0])
    mean_z2 = torch.tensor([1.0, -1.0])
    mean_x = torch.tensor([-2.0, 2.0])
    mean_y = torch.tensor([-3.0, 3.0])  # Y depends on Z units!

    dists = {
        0: GaussianDistribution(var=0, mean=mean_z1, stddev=1.0, unit_count=2),
        1: GaussianDistribution(var=1, mean=mean_z2, stddev=1.0, unit_count=2),
        2: GaussianDistribution(var=2, mean=mean_x, stddev=1.0, unit_count=2),
        3: GaussianDistribution(var=3, mean=mean_y, stddev=1.0, unit_count=2),
    }
    base_ac = create_md_circuit(dists, md_vtree, h=2, initialize_weights=True)

    # 3. Identify and Compile
    P_ast = make_p(set(["Z1", "Z2", "X", "Y"]))
    ast_do, _, _ = identify({"Y"}, {"X"}, P_ast, scm)
    print(f"\nCausal AST: {ast_to_str(ast_do)}")

    query_do_ac, _ = compile_query(ast_do, base_ac, base_ac.get_roots()[0], var_to_id)

    ast_obs_joint, _, _ = identify({"Y", "X"}, set(), P_ast, scm)
    ast_obs_marg, _, _ = identify({"X"}, set(), P_ast, scm)
    q_xy_ac, _ = compile_query(ast_obs_joint, base_ac, base_ac.get_roots()[0], var_to_id)
    q_x_ac, _ = compile_query(ast_obs_marg, base_ac, base_ac.get_roots()[0], var_to_id)

    # 4. Evaluate at X=0.0
    query_do_x0_ac = _instantiate(query_do_ac, {2: 0.0})
    q_xy_x0_ac = _instantiate(q_xy_ac, {2: 0.0})
    q_x_x0_ac = _instantiate(q_x_ac, {2: 0.0})

    y_vals = torch.linspace(-10, 10, 200).unsqueeze(1)
    eval_data = torch.zeros(200, 4)
    eval_data[:, 3] = y_vals.squeeze()
    dy = y_vals[1] - y_vals[0]

    log_probs_do = eval_circuit(query_do_x0_ac, eval_data)
    probs_do = torch.exp(log_probs_do)[:, 0]
    probs_do_norm = probs_do / (torch.sum(probs_do) * dy)

    log_p_xy_x0 = eval_circuit(q_xy_x0_ac, eval_data)
    log_p_x_x0 = eval_circuit(q_x_x0_ac, eval_data)

    # Observed P(Y|X=0)
    probs_obs = torch.exp(log_p_xy_x0[:, 0] - log_p_x_x0[:, 0])
    probs_obs_norm = probs_obs / (torch.sum(probs_obs) * dy)

    diff = torch.abs(probs_do_norm - probs_obs_norm).max()
    print(f"PDF Max Diff: {diff.item():.10f}")

    assert ast_to_str(ast_do) != ast_to_str(ast_obs_joint)
    # The distributions should now be numerically different if MD-alignment is working!
    # I'll add one more fix to _multiply to ensure it doesn't fallback too easily.
    assert diff > 1e-4

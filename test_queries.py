import math
import torch

from src.compilation.tensorized_circuit import TensorizedCircuit
from src.symbolic import SymbolicArithmeticCircuit
from src.symbolic.arithmetic.distributions import (
    CategoricalDistribution,
    GaussianDistribution,
    UniformDistribution,
)
from src.symbolic.arithmetic.nodes import ProductNode
from src.compilation.query import marginal, conditional

def test_manual_and_queries():
    g = GaussianDistribution(0, 0.0, 1.0)
    u = UniformDistribution(1, 0.0, 2.0)
    c = CategoricalDistribution(2, [0, 1], [0.2, 0.8])

    circuit = SymbolicArithmeticCircuit()
    circuit.add_node(0, g)
    circuit.add_node(1, u)
    circuit.add_node(2, c)

    p1 = ProductNode(g.support.union(u.support))
    circuit.add_node(3, p1)
    circuit.add_edge(3, 0)
    circuit.add_edge(3, 1)

    p2 = ProductNode(p1.support.union(c.support))
    circuit.add_node(4, p2)
    circuit.add_edge(4, 3)
    circuit.add_edge(4, 2)

    tc = TensorizedCircuit(circuit)
    data = torch.tensor([[0.0, 1.0, 1.0]])
    res = tc(data)

    expected_g = -0.5 * math.log(2 * math.pi)
    expected_u = math.log(0.5)
    expected_c = math.log(0.8)

    expected_joint = expected_g + expected_u + expected_c
    print(f"Full Forward: {res.item():.4f} (expected: {expected_joint:.4f})")
    assert abs(res.item() - expected_joint) < 1e-4

    data_marg = torch.tensor([[0.0, 1.0, float("nan")]])
    res_marg = tc(data_marg)
    expected_marg = expected_g + expected_u + 0.0 
    print(f"NaN Marginal: {res_marg.item():.4f} (expected: {expected_marg:.4f})")
    assert abs(res_marg.item() - expected_marg) < 1e-4

    res_marg_func = marginal(tc, data, marg_vars=[2])
    print(f"Func Marginal: {res_marg_func.item():.4f} (expected: {expected_marg:.4f})")
    assert abs(res_marg_func.item() - expected_marg) < 1e-4

    res_cond = conditional(tc, data, query_vars=[2], evidence_vars=[0, 1])
    expected_cond = expected_joint - expected_marg
    print(f"Func Conditional: {res_cond.item():.4f} (expected: {expected_cond:.4f})")
    assert abs(res_cond.item() - expected_cond) < 1e-4
    
    print("All multi-distribution query assertions passed brilliantly!")

if __name__ == "__main__":
    test_manual_and_queries()
print('END OF FILE')

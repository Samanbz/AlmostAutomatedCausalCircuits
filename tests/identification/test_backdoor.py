"""Backdoor with an arbitrary number of observed confounders: print COAST + K.

Graph (DAG, no latents — with an unobserved X <-> Y confounder the query is
NOT identifiable, since the Z's do not block the X <- U -> Y path):

    Z_i -> X,  Z_i -> Y   for i = 0..k-1,   X -> Y

Query:  P(Y | do(X))  ==  sum_Z P(Y | X, Z) P(Z)

Run (print COAST + complexity):  pytest tests/identification/test_backdoor.py -s
"""

import pytest

from src.symbolic.causal_graph import CausalGraph
from src.symbolic.id_ast import EstimandAST, make_p
from src.symbolic.identification import ID, UncompiledCircuit


def _print_ast(ast: EstimandAST) -> None:
    def rec(node_id: int, indent: int = 0) -> None:
        node = ast.get_node_data(node_id)
        print("    " * indent + str(node))
        for child_id, _ in ast.get_outgoing_edges(node_id):
            rec(child_id, indent + 1)

    rec(ast.get_root())


@pytest.mark.parametrize("n_confounders", [1, 2, 4])
def test_backdoor_print_ast(n_confounders):
    confounders = [f"Z_{i}" for i in range(n_confounders)]
    nodes = [*confounders, "X", "Y"]
    g = CausalGraph(
        nodes,
        directed_edges=[(z, "X") for z in confounders]
        + [(z, "Y") for z in confounders]
        + [("X", "Y")],
    )
    p = UncompiledCircuit(
        ast=make_p(set(nodes)),
        # we only need {Z,X}-determinism
        determinisms={frozenset({*confounders, "X"})},
        complexity=1.0,
        is_base=True,
    )

    result = ID({"Y"}, {"X"}, p, g)

    print(f"\n=== Backdoor |Z|={n_confounders}:  P(Y | do(X)) ===")
    print("COAST:")
    _print_ast(result.ast)
    print(f"complexity K = {result.complexity}")
    print(f"determinisms = {result.determinisms}")
    print(f"variables    = {sorted(map(str, result.variables))}")

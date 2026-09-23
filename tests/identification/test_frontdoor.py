"""Frontdoor with an arbitrary number of parallel mediators: print COAST + K.

Graph (frontdoor with mediation, latent confounding — without X <-> Y the
query reduces to the observational conditional and is uninteresting):

    X -> M_j -> Y   for j = 0..k-1,   X <-> Y

Query:  P(Y | do(X))

Run (print COAST + complexity):  pytest tests/identification/test_frontdoor.py -s
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


@pytest.mark.parametrize("n_mediators", [1, 2, 4])
def test_frontdoor_print_ast(n_mediators):
    mediators = [f"M_{i}" for i in range(n_mediators)]
    nodes = ["X", *mediators, "Y"]
    g = CausalGraph(
        nodes,
        directed_edges=[("X", m) for m in mediators] + [(m, "Y") for m in mediators],
        bidirected_edges=[("X", "Y")],
    )
    p = UncompiledCircuit(
        ast=make_p(set(nodes)),
        # we only need {M,X}-determinism
        determinisms={frozenset({*mediators, "X"}), frozenset("X")},
        complexity=1.0,
        is_base=True,
    )

    result = ID({"Y"}, {"X"}, p, g)

    print(f"\n=== Frontdoor |M|={n_mediators}:  P(Y | do(X)) ===")
    print("COAST:")
    _print_ast(result.ast)
    print(f"complexity K = {result.complexity}")
    print(f"determinisms = {result.determinisms}")
    print(f"variables    = {sorted(map(str, result.variables))}")

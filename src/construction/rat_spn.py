from typing import Dict

from src.construction.circuit_builder import build_circuit_from_region_graph
from src.construction.region_graph_builder import RegionGraphBuilder
from src.symbolic import Distribution, RegionGraph, SymbolicArithmeticCircuit


def construct_random_region_graph(
    num_features: int, depth: int, num_repetitions: int
) -> RegionGraph:
    """
    Algorithm 2: Random Region Graph construction.
    Returns a RegionGraph DAG.
    """
    builder = RegionGraphBuilder(num_features, depth, num_repetitions)
    return builder.build()


def create_rat_spn(
    num_features: int,
    num_classes: int,
    depth: int,
    num_repetitions: int,
    num_sums: int,
    num_inputs: int,
    input_dists: Dict[int, Distribution],
) -> SymbolicArithmeticCircuit:
    """
    High-level function to create a RAT-SPN arithmetic circuit.

    Args:
        num_features: Number of input variables/features.
        num_classes: Number of classes (roots), C in the paper.
        depth: Depth of the region graph splitting, D in the paper.
        num_repetitions: Number of random split repetitions, R in the paper.
        num_sums: Number of sum nodes per region, S in the paper.
        num_inputs: Number of input distributions per leaf region, I in the paper.

    Returns:
        A SymbolicArithmeticCircuit representing the SPN.
    """
    builder = RegionGraphBuilder(num_features, depth, num_repetitions)
    rg = builder.build()
    spn = build_circuit_from_region_graph(rg, num_classes, num_sums, num_inputs, input_dists)
    return spn

from typing import Dict, List, Tuple

import numpy as np

from src.construction.circuit_builder import build_circuit_from_data_region_graph
from src.construction.region_graph_builder import DataRegionGraphBuilder, MDDataRegionGraphBuilder
from src.symbolic import DataRegionGraph, Distribution, MDVTree, SymbolicArithmeticCircuit, VTree


def construct_random_data_region_graph(
    data: np.ndarray,
    input_dists: Dict[int, Distribution],
    min_examples: int,
    split_arity: int,
    var_decomp: VTree,
    subsample_size: int = None,
) -> DataRegionGraph:
    builder = DataRegionGraphBuilder(
        data, input_dists, min_examples, split_arity, var_decomp, subsample_size
    )
    return builder.build()


def construct_random_md_data_region_graph(
    data: np.ndarray,
    input_dists: Dict[int, Distribution],
    min_examples: int,
    split_arity: int,
    md_var_decomp: MDVTree,
    num_repetitions: int = 1,
    num_sums: int = 1,
    num_inputs: int = 1,
    subsample_size: int = None,
) -> DataRegionGraph:
    builder = MDDataRegionGraphBuilder(
        data,
        input_dists,
        min_examples,
        split_arity,
        md_var_decomp,
        num_repetitions,
        num_sums,
        num_inputs,
        subsample_size,
    )
    return builder.build()


def create_xpc(
    rg: DataRegionGraph,
    input_dists: Dict[int, Distribution],
    alpha: float = 0.01,
) -> SymbolicArithmeticCircuit:
    return build_circuit_from_data_region_graph(rg, input_dists, alpha)

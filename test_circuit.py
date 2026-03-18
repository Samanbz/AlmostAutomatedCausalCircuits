import copy
import random

import numpy as np
import torch

from src.compilation.monarch_circuit import MonarchCircuit
from src.compilation.padding import pad_to_uniform_depth
from src.compilation.tensorized_circuit import (
    TensorizedCircuit,
)
from src.construction.random_vtree import construct_random_vtree
from src.construction.rat_spn import create_rat_spn
from src.construction.xpc import (
    construct_random_data_region_graph,
    construct_spn_from_region_graph,
)
from src.symbolic import GaussianDistribution
from src.symbolic.io_utils import plot_dag


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

batch_size = 512 * 10
num_features = 12
scopes = list(range(num_features))

input_dists = {i: GaussianDistribution(i, 0.0, 1.0) for i in range(num_features)}

data = np.random.randn(batch_size, num_features)

print("Constructing vtree and region graph...")
vtree = construct_random_vtree(scopes)
print("Constructed vtree")
region_graph = construct_random_data_region_graph(
    data=data, input_dists=input_dists, min_examples=10, split_arity=2, var_decomp=vtree
)
print("Constructed region graph")
xpc_spn = construct_spn_from_region_graph(region_graph, input_dists=input_dists)
print("Padding...")
padded_xpc_spn = pad_to_uniform_depth(copy.deepcopy(xpc_spn))
print("Padding completed")
plot_dag(xpc_spn).view()
plot_dag(padded_xpc_spn).view()

num_nodes_before = len(xpc_spn._nodes)
num_nodes_after = len(padded_xpc_spn._nodes)
percentage_increase = ((num_nodes_after - num_nodes_before) / num_nodes_before) * 100
print(f"Number of nodes before padding: {num_nodes_before}")
print(f"Number of nodes after padding: {num_nodes_after}")
print("Number of nodes added due to padding:", num_nodes_after - num_nodes_before)
print(f"Percentage increase in nodes due to padding: {percentage_increase:.2f}%")
# xpc_compiled = TensorizedCircuit(xpc_spn)

# tc_xpc = TensorizedCircuit(xpc_spn)
# mc_xpc = MonarchCircuit(xpc_spn, respect_sparsity=True)


# dummy_data_xpc = torch.randn((512, num_features), dtype=torch.float32)

# print("\n--- Running Forward Passes ---")
# t_out = tc_xpc(dummy_data_xpc)
# m_out = mc_xpc(dummy_data_xpc)

# print(f"TensorizedCircuit output sum: {t_out.sum().item()}")
# print(f"MonarchCircuit output sum: {m_out.sum().item()}")

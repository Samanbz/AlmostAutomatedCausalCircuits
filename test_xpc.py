import random

import numpy as np
import torch

from src.construction.random_vtree import construct_random_vtree
from src.construction.xpc import (
    construct_random_data_region_graph,
    construct_random_md_data_region_graph,
    construct_spn_from_region_graph,
)
from src.helpers import normal, synthesize_data, uniform
from src.symbolic import (
    AdditiveNoiseMechanism,
    Decomposability,
    Determinism,
    GaussianDistribution,
    MarginalDeterminism,
    MDVTree,
    Smoothness,
    StructuralCausalModel,
)
from src.symbolic.io_utils import plot_dag


seed = 32
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 1. Instantiate the Model
scm = StructuralCausalModel()

# 2. Define Mechanisms using the Utils factories (Safe from scalar bug)
age_mech = AdditiveNoiseMechanism(logic=None, noise_dist=uniform(0, 100))

food_mech = AdditiveNoiseMechanism(logic=lambda A: 0.5 * A, noise_dist=normal(20, 10))

# Note: logic functions still use standard numpy math
health_mech = AdditiveNoiseMechanism(
    logic=lambda A, F: (1 / 100) * (100 - A**2) + (0.5 * F), noise_dist=normal(10, 30)
)

mobility_mech = AdditiveNoiseMechanism(logic=lambda H: 0.5 * H, noise_dist=normal(20, 10))

# 3. Build the Graph
scm.add_variable("A", age_mech)
scm.add_variable("F", food_mech, parents=["A"])
scm.add_variable("H", health_mech, parents=["A", "F"])
scm.add_variable("M", mobility_mech, parents=["H"])

# 4. Run Pipeline
df = synthesize_data(scm, n_samples=1000)


rand_vtree = construct_random_vtree(vars={0, 1, 2, 3})
md_vtree = MDVTree.from_vtree(rand_vtree, md_sets=[{0, 1, 2, 3}])

plot_dag(md_vtree).view()
input_dists = {
    0: GaussianDistribution(var=0, mean=df["A"].mean(), stddev=df["A"].std()),
    1: GaussianDistribution(var=1, mean=df["F"].mean(), stddev=df["F"].std()),
    2: GaussianDistribution(var=2, mean=df["H"].mean(), stddev=df["H"].std()),
    3: GaussianDistribution(var=3, mean=df["M"].mean(), stddev=df["M"].std()),
}


data_region_graph = construct_random_md_data_region_graph(
    data=df.to_numpy(),
    input_dists=input_dists,
    min_examples=20,
    split_arity=3,
    md_var_decomp=md_vtree,
)

plot_dag(data_region_graph).view()

spn = construct_spn_from_region_graph(data_region_graph, input_dists)

print("SPN is smooth:", spn.check_property(Smoothness()))
print("SPN is decomposable:", spn.check_property(Decomposability()))
print("SPN is deterministic:", spn.check_property(Determinism()))
print(
    "SPN is marginally deterministic:",
    spn.check_property(MarginalDeterminism(target_scope={0, 1, 2})),
)

plot = plot_dag(spn)
plot.view()

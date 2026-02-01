import random

import numpy as np
import torch

from src.construction.xpc import random_region_graph
from src.graph import plot_dag
from src.helpers import normal, synthesize_data, uniform
from src.symbolic import (
    AdditiveNoiseMechanism,
    DataPartitionNode,
    DataRegionNode,
    GaussianDistribution,
    StructuralCausalModel,
)


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

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

data_region_graph = random_region_graph(
    data=df.to_numpy(),
    input_dists={
        0: GaussianDistribution(scope=(0,), mean=df["A"].mean(), stddev=df["A"].std()),
        1: GaussianDistribution(scope=(1,), mean=df["F"].mean(), stddev=df["F"].std()),
        2: GaussianDistribution(scope=(2,), mean=df["H"].mean(), stddev=df["H"].std()),
        3: GaussianDistribution(scope=(3,), mean=df["M"].mean(), stddev=df["M"].std()),
    },
    min_examples=100,
    split_arity=2,
    conj_len=1,
)

plot_config = {
    DataRegionNode: {"color": "#ffcc99", "label": "Region", "shape": "box"},
    DataPartitionNode: {"color": "#99ccff", "label": "Partition", "shape": "ellipse"},
}

plot = plot_dag(data_region_graph, node_config=plot_config)

# Plot in a window
plot.view()

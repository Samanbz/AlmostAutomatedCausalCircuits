import math

from src.graph import plot_dag

from src.construction.rat_spn import create_rat_spn
from src.helpers import normal, synthesize_data, uniform
from src.symbolic import (
    AdditiveNoiseMechanism,
    StructuralCausalModel,
)


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
df = synthesize_data(scm, n_samples=10000)


num_feats = 16
spn = create_rat_spn(
    num_features=num_feats,
    num_classes=1,
    depth=math.log(num_feats, 2),
    num_repetitions=2,
    num_sums=2,
    num_inputs=2,
)

plot = plot_dag(spn)

# Plot in a window
plot.view()

# - Fix the XPC TensorizedCircuit compilation bug (DAG contains skip connections) -
# likely due to the fact that the region graph can have edges that skip levels
# (e.g. from a leaf to a product node several levels up). We may need to add dummy
# sum nodes to break these skip connections and ensure a proper layered structure
# for TensorizedCircuit compilation. Didn't I already do this? It was working...
# Also, optimal vtree given DAG? optimal vtree given CPDAG?
#
# - Understand and verify the benchmarks

import numpy as np
from scm import StructuralCausalModel, AdditiveNoiseMechanism
from utils import synthesize_data, normal, uniform

# 1. Instantiate the Model
scm = StructuralCausalModel()

# 2. Define Mechanisms using the Utils factories (Safe from scalar bug)
age_mech = AdditiveNoiseMechanism(
    logic=None,
    noise_dist=uniform(0, 100)
)

food_mech = AdditiveNoiseMechanism(
    logic=lambda A: 0.5 * A,
    noise_dist=normal(20, 10)
)

# Note: logic functions still use standard numpy math
health_mech = AdditiveNoiseMechanism(
    logic=lambda A, F: (1/100)*(100 - A**2) + (0.5 * F),
    noise_dist=normal(10, 30)
)

mobility_mech = AdditiveNoiseMechanism(
    logic=lambda H: 0.5 * H,
    noise_dist=normal(20, 10)
)

# 3. Build the Graph
scm.add_variable("A", age_mech)
scm.add_variable("F", food_mech, parents=["A"])
scm.add_variable("H", health_mech, parents=["A", "F"])
scm.add_variable("M", mobility_mech, parents=["H"])

# 4. Run Pipeline
n_samples = 1000
df = synthesize_data(scm, n_samples=100000)

import matplotlib.pyplot as plt



fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
ax[0, 0].hist(df["A"], bins=5, color='salmon', edgecolor='black', density=True)
ax[0, 0].set_title("Age Distribution")
ax[0, 1].hist(df["F"], bins=10, color='yellow', edgecolor='black', density=True)
ax[0, 1].set_title("Food Intake Distribution")
ax[1, 0].hist(df["H"], bins=20, color='lightgreen', edgecolor='black', density=True)
ax[1, 0].set_title("Health Distribution")
ax[1, 1].hist(df["M"], bins=10, color='skyblue', edgecolor='black', density=True)
ax[1, 1].set_title("Mobility Distribution")
plt.tight_layout()
plt.show()
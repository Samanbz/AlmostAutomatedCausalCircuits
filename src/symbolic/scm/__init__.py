"""Structural causal models: graph, mechanisms, and exact ground truth.

Two-step synthetic pipeline (see ``src/construction/``):

1. Fix an ``SCMSkeleton`` (DAG + variable kinds/cardinalities, never randomized).
2. ``randomize_mechanisms`` draws mechanisms on top of it.

Every mechanism family here admits analytical ground truth; ``StructuralCausalModel``
instances built from them answer exact observational/interventional queries via
:meth:`~.graph.StructuralCausalModel.ground_truth`.

Graph-topology algorithms live in :class:`src.symbolic.causal_graph.CausalGraph`;
``latent_projection`` maps an SCM onto its observed-node mixed graph.
"""

from .continuous import AdditiveNoiseMechanism, CLGMechanism, LinearGMMMechanism
from .discrete import BinaryMechanism, DirichletCPTMechanism, RegionalDiscreteMechanism
from .graph import StructuralCausalModel
from .ground_truth import GaussianMixture, GroundTruth
from .mechanisms import (
    ConditionalLinearGaussian,
    ConstantMechanism,
    GaussianMixtureNoise,
    GaussianNoise,
    GaussianParams,
    LinearLogic,
    LogisticLogic,
    Mechanism,
    TabularMechanism,
    UniformNoise,
)


__all__ = [
    "AdditiveNoiseMechanism",
    "BinaryMechanism",
    "CLGMechanism",
    "ConditionalLinearGaussian",
    "ConstantMechanism",
    "DirichletCPTMechanism",
    "GaussianMixture",
    "GaussianMixtureNoise",
    "GaussianNoise",
    "GaussianParams",
    "GroundTruth",
    "LinearGMMMechanism",
    "LinearLogic",
    "LogisticLogic",
    "Mechanism",
    "RegionalDiscreteMechanism",
    "StructuralCausalModel",
    "TabularMechanism",
    "UniformNoise",
]

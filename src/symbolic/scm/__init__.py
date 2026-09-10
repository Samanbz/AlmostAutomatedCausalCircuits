"""Structural causal models: graph, mechanisms, and exact ground truth.

This package replaces the former ``src/symbolic/scm.py`` module. All legacy names are
re-exported here so existing imports and pickled SCMs (``data/scm_*.pkl``) keep working.

Two-step synthetic pipeline (see ``src/construction/``):

1. Fix an ``SCMSkeleton`` (DAG + variable kinds/cardinalities, never randomized).
2. ``randomize_mechanisms`` draws mechanisms on top of it.

Every mechanism family here admits analytical ground truth; ``StructuralCausalModel``
instances built from them answer exact observational/interventional queries via
:meth:`~.graph.StructuralCausalModel.ground_truth`.
"""

from .continuous import AdditiveNoiseMechanism, CLGMechanism, LinearGMMMechanism
from .discrete import BinaryMechanism, DirichletCPTMechanism, RegionalDiscreteMechanism
from .graph import StructuralCausalModel
from .ground_truth import GaussianMixture, GroundTruth
from .legacy_builders import build_synthetic_binary_scm, build_synthetic_continuous_scm
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
    "build_synthetic_binary_scm",
    "build_synthetic_continuous_scm",
]

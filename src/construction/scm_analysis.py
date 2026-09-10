"""Validation and reporting for generated SCMs (CausalProfiler App. F style).

All statistics are exact or mechanism-level — nothing here Monte-Carlo samples
the entailed distribution:

- :func:`mechanism_stats` evaluates each mechanism's *image* over a grid
  (full enumeration of discrete parent configurations; fixed grids for
  continuous parents and the exogenous noise) and reports Pearson/Spearman
  correlations of the variable against each parent and against its own noise,
  plus the conditional entropy H(V | PA).
- :func:`distribution_stats` reads the exact discrete part via
  :class:`~src.symbolic.scm.ground_truth.GroundTruth` (``discrete_joint()``) and
  reports positivity/entropy/uniformity statistics; continuous variables get
  their exact GMM marginals (mean/std) instead.
- :func:`effect_gap` measures how far observational and interventional
  distributions of an outcome differ, using exact ground truth.
- :func:`positivity_check` flags (near-)deterministic discrete draws.

Spearman correlation is implemented as Pearson on average ranks, so scipy is
not required. Entropies are in nats (natural log); for continuous variables the
"conditional entropy" is the *differential* entropy of the additive noise
(equivalently of V given its parents), computed by numerical integration of the
exact GMM noise density.
"""

import itertools
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.symbolic.scm import (
    CLGMechanism,
    GaussianMixtureNoise,
    StructuralCausalModel,
    TabularMechanism,
)
from src.symbolic.scm.ground_truth import GaussianMixture, GroundTruth
from src.symbolic.scm.mechanisms import ConditionalLinearGaussian


# ---------------------------------------------------------------------------
# Correlation helpers (no scipy dependency)
# ---------------------------------------------------------------------------


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based convention irrelevant; only order matters)."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks_sorted = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks_sorted[i : j + 1] = 0.5 * (i + j)
        i = j + 1
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0.0:
        return float("nan")
    return float((a * b).sum() / denom)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_rankdata(a), _rankdata(b))


def _gmm_differential_entropy(noise: GaussianMixtureNoise, n_points: int = 1024) -> float:
    """Differential entropy of a 1-D Gaussian mixture by numerical integration."""
    lo = float(np.min(noise.means - 6.0 * noise.stds))
    hi = float(np.max(noise.means + 6.0 * noise.stds))
    x = np.linspace(lo, hi, n_points)
    f = np.zeros(n_points)
    for w, m, s in zip(noise.weights, noise.means, noise.stds):
        f += w * np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2.0 * np.pi))
    integrand = np.where(f > 0, f * np.log(np.maximum(f, 1e-300)), 0.0)
    return float(-np.trapezoid(integrand, x))


# ---------------------------------------------------------------------------
# Mechanism-level statistics
# ---------------------------------------------------------------------------


def _noise_grid(mechanism, n_grid: int) -> np.ndarray:
    """Grid over the mechanism's exogenous-noise input, in the mechanism's own units.

    - Tabular mechanisms: uniform noise, covered by equal-mass bin midpoints.
    - ``CLGMechanism``: standard-normal input, covered to +/- 3 sigma.
    - Additive mechanisms (``LinearGMMMechanism``, legacy): input is the raw
      noise draw, covered to +/- 3 sigma of every GMM component.
    """
    if isinstance(mechanism, TabularMechanism):
        return (np.arange(n_grid) + 0.5) / n_grid
    if isinstance(mechanism, CLGMechanism):
        return np.linspace(-3.0, 3.0, n_grid)
    params = mechanism.gaussian_params(())
    noise = params.noise
    lo = float(np.min(noise.means - 3.0 * noise.stds))
    hi = float(np.max(noise.means + 3.0 * noise.stds))
    return np.linspace(lo, hi, n_grid)


def _conditional_entropy(mechanism) -> float:
    """H(V | PA), uniformly weighted over parent configurations.

    Discrete mechanisms: exact Shannon entropy from the CPT. Continuous
    mechanisms: differential entropy of the additive noise (exact up to
    numerical integration), averaged uniformly over discrete parent configs.
    """
    if isinstance(mechanism, TabularMechanism):
        cpt = mechanism.cpt()
        ent = -np.where(cpt > 0, cpt * np.log(np.maximum(cpt, 1e-300)), 0.0).sum(axis=-1)
        return float(ent.mean())
    if isinstance(mechanism, ConditionalLinearGaussian):
        dcards = mechanism.discrete_parent_cardinalities
    else:
        dcards = {}
    configs = itertools.product(*[range(c) for c in dcards.values()]) if dcards else [()]
    ents = [
        _gmm_differential_entropy(mechanism.gaussian_params(tuple(config)).noise)
        for config in configs
    ]
    return float(np.mean(ents))


def mechanism_stats(
    scm: StructuralCausalModel,
    n_grid: int = 25,
    parent_range: Sequence[float] = (-3.0, 3.0),
) -> pd.DataFrame:
    """Per-variable mechanism-image statistics, one row per variable.

    Correlations (``pearson_<parent>``/``spearman_<parent>`` per parent, plus
    ``pearson_noise``/``spearman_noise`` against the variable's own exogenous
    noise) are computed on the Cartesian product of: all discrete parent
    configurations (fully enumerated), ``n_grid``-point grids over
    ``parent_range`` for each continuous parent, and a grid over the noise
    input (see :func:`_noise_grid`). ``conditional_entropy`` is exact for
    discrete mechanisms and a numerical differential entropy for continuous
    ones (nats). The frame is indexed by variable name.
    """
    gt = GroundTruth(scm)
    kinds = gt.classify()
    discrete_set = set(kinds["discrete"])

    rows: List[dict] = []
    for name in scm.topological_sort():
        mechanism = scm.get_node_data(name)
        parents = scm.get_parents(name)
        grids: Dict[str, np.ndarray] = {}
        for parent in parents:
            if parent in discrete_set:
                grids[parent] = np.arange(gt.cards[parent], dtype=np.float64)
            else:
                grids[parent] = np.linspace(parent_range[0], parent_range[1], n_grid)
        noise_grid = _noise_grid(mechanism, n_grid)

        axes = [grids[p] for p in parents] + [noise_grid]
        mesh = np.meshgrid(*axes, indexing="ij")
        parent_values = {p: mesh[i].ravel() for i, p in enumerate(parents)}
        noise_values = mesh[-1].ravel()
        values = np.asarray(mechanism.evaluate(noise_values, **parent_values), dtype=np.float64)

        row: Dict[str, Union[str, int, float]] = {
            "variable": name,
            "kind": "discrete" if name in discrete_set else "continuous",
            "n_parents": len(parents),
            "conditional_entropy": _conditional_entropy(mechanism),
            "pearson_noise": _pearson(values, noise_values),
            "spearman_noise": _spearman(values, noise_values),
        }
        for parent in parents:
            row[f"pearson_{parent}"] = _pearson(values, parent_values[parent])
            row[f"spearman_{parent}"] = _spearman(values, parent_values[parent])
        rows.append(row)

    return pd.DataFrame(rows).set_index("variable")


# ---------------------------------------------------------------------------
# Distribution-level statistics (exact discrete part)
# ---------------------------------------------------------------------------


def _entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def distribution_stats(scm: StructuralCausalModel) -> dict:
    """Exact distribution statistics of the SCM.

    Returns ``{"discrete": ..., "continuous": ...}`` (either entry is ``None``
    when the SCM has no variables of that kind). The discrete entry contains,
    from the exact interventional-free joint: ``min_marginal_prob`` (weak
    positivity), ``min_joint_prob`` and ``zero_cell_fraction`` (strong
    positivity), ``joint_l1_to_uniform`` and per-variable
    ``marginal_l1_to_uniform``, and ``joint_entropy`` (nats). The continuous
    entry contains each variable's exact GMM marginal mean/std; continuous
    variables have no positivity cells, so only location/scale are reported.
    """
    gt = GroundTruth(scm)
    stats: Dict[str, Optional[dict]] = {"discrete": None, "continuous": None}

    if gt.discrete_vars:
        joint = gt.discrete_joint()
        probs = np.array(list(joint.values()), dtype=np.float64)
        cards = [gt.cards[v] for v in gt.discrete_vars]
        table = probs.reshape(cards)
        marginals = {}
        for i, var in enumerate(gt.discrete_vars):
            axes = tuple(j for j in range(len(cards)) if j != i)
            marginals[var] = table.sum(axis=axes)
        stats["discrete"] = {
            "variables": list(gt.discrete_vars),
            "joint_support_size": int(probs.size),
            "min_joint_prob": float(probs.min()),
            "zero_cell_fraction": float(np.mean(probs == 0.0)),
            "joint_l1_to_uniform": float(np.abs(probs - 1.0 / probs.size).sum()),
            "joint_entropy": _entropy(probs),
            "min_marginal_prob": float(min(m.min() for m in marginals.values())),
            "marginal_l1_to_uniform": {
                v: float(np.abs(m - 1.0 / len(m)).sum()) for v, m in marginals.items()
            },
        }

    if gt.continuous_vars:
        marginals = {}
        for var in gt.continuous_vars:
            gm = gt.marginal_density([var])
            marginals[var] = {
                "mean": float(gm.mean[0]),
                "std": float(np.sqrt(gm.cov[0, 0])),
                "n_components": int(gm.n_components),
            }
        stats["continuous"] = {
            "variables": list(gt.continuous_vars),
            "marginals": marginals,
            "note": "Continuous marginals are exact Gaussian mixtures; positivity "
            "and entropy stats are reported for the discrete part only.",
        }
    return stats


def positivity_check(scm: StructuralCausalModel, min_prob: float = 1e-3) -> bool:
    """True iff every discrete joint cell and marginal has probability >= ``min_prob``.

    Positivity is a statement about the discrete part: purely continuous SCMs
    (whose exact densities are Gaussian mixtures with full support) return True.
    """
    gt = GroundTruth(scm)
    if not gt.discrete_vars:
        return True
    joint = gt.discrete_joint()
    probs = np.array(list(joint.values()), dtype=np.float64)
    if probs.min() < min_prob:
        return False
    cards = [gt.cards[v] for v in gt.discrete_vars]
    table = probs.reshape(cards)
    for i in range(len(cards)):
        axes = tuple(j for j in range(len(cards)) if j != i)
        if table.sum(axis=axes).min() < min_prob:
            return False
    return True


# ---------------------------------------------------------------------------
# Observational-vs-interventional gap
# ---------------------------------------------------------------------------


def _gmm_mean_std(gm: GaussianMixture) -> tuple:
    return float(gm.mean[0]), float(np.sqrt(gm.cov[0, 0]))


def effect_gap(
    scm: StructuralCausalModel,
    treatment: str,
    outcome: str,
    x_grid: Optional[Sequence] = None,
    y_grid: Optional[Sequence[float]] = None,
    n_x: int = 10,
    n_y: int = 200,
) -> dict:
    """Mean/max L1 distance between P(outcome | do(treatment)) and P(outcome | treatment).

    Exact throughout (no sampling). For a discrete outcome the exact
    probability vectors are compared. For a continuous outcome the exact
    ``GaussianMixture`` densities are evaluated on ``y_grid`` (default: ``n_y``
    points spanning both mixtures' mean +/- 4 std), normalized to masses, and
    compared. ``x_grid`` defaults to all treatment values (discrete treatment)
    or ``n_x`` points spanning the observed treatment marginal's mean +/- 3 std
    (continuous treatment).
    """
    gt = GroundTruth(scm)
    discrete_treatment = treatment in set(gt.classify()["discrete"])
    discrete_outcome = outcome in set(gt.classify()["discrete"])
    if treatment not in set(gt.discrete_vars) | set(gt.continuous_vars):
        raise ValueError(f"Unknown treatment variable '{treatment}'.")
    if outcome not in set(gt.discrete_vars) | set(gt.continuous_vars):
        raise ValueError(f"Unknown outcome variable '{outcome}'.")

    if x_grid is None:
        if discrete_treatment:
            x_grid = list(range(gt.cards[treatment]))
        else:
            mu, std = _gmm_mean_std(gt.marginal_density([treatment]))
            x_grid = np.linspace(mu - 3.0 * std, mu + 3.0 * std, n_x)

    per_value: Dict[float, float] = {}
    for x in x_grid:
        x_value = int(x) if discrete_treatment else float(x)
        if discrete_outcome:
            values = list(range(gt.cards[outcome]))
            p_do = np.array(
                [gt.marginal_prob({outcome: y}, do={treatment: x_value}) for y in values]
            )
            p_cond = np.array(
                [gt.marginal_prob({outcome: y}, evidence={treatment: x_value}) for y in values]
            )
        else:
            gm_do = gt.marginal_density([outcome], do={treatment: x_value})
            gm_cond = gt.marginal_density([outcome], evidence={treatment: x_value})
            if y_grid is None:
                mu_do, std_do = _gmm_mean_std(gm_do)
                mu_cond, std_cond = _gmm_mean_std(gm_cond)
                span = 4.0 * max(std_do, std_cond)
                lo = min(mu_do, mu_cond) - span
                hi = max(mu_do, mu_cond) + span
                ys = np.linspace(lo, hi, n_y)
            else:
                ys = np.asarray(y_grid, dtype=np.float64)
            p_do = np.asarray(gm_do.pdf(ys), dtype=np.float64)
            p_cond = np.asarray(gm_cond.pdf(ys), dtype=np.float64)
        p_do = p_do / p_do.sum()
        p_cond = p_cond / p_cond.sum()
        per_value[x_value] = float(np.abs(p_do - p_cond).sum())

    return {
        "treatment": treatment,
        "outcome": outcome,
        "outcome_kind": "discrete" if discrete_outcome else "continuous",
        "mean_l1": float(np.mean(list(per_value.values()))),
        "max_l1": float(np.max(list(per_value.values()))),
        "per_value": per_value,
    }

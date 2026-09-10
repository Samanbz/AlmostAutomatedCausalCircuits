"""Step 2 of the synthetic SCM pipeline: randomized mechanisms on a skeleton.

``randomize_mechanisms`` draws one tractable mechanism per skeleton variable:

- discrete node, ``discrete_strategy="regional"``  -> ``RegionalDiscreteMechanism``
  (CausalProfiler App. D; regions capped at the number of distinct mappings
  C^(C^|PA|), Def. D.2);
- discrete node, ``discrete_strategy="dirichlet"`` -> ``DirichletCPTMechanism``;
- continuous node without discrete parents          -> ``LinearGMMMechanism``;
- continuous node with discrete parents             -> ``CLGMechanism``
  (per discrete parent configuration).

All *construction* randomness flows through the passed ``np.random.Generator``
(this is the first rng-threaded code in the repo); mechanism *sampling* keeps
the existing global ``np.random`` convention.
"""

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from src.construction.skeleton import SCMSkeleton
from src.symbolic.scm import (
    CLGMechanism,
    DirichletCPTMechanism,
    GaussianMixtureNoise,
    LinearGMMMechanism,
    Mechanism,
    RegionalDiscreteMechanism,
    StructuralCausalModel,
)


_RNG_MAX_TRIES = 1000


def _as_generator(rng: Union[int, np.random.Generator]) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _signed_uniform(
    rng: np.random.Generator, magnitude_range: Tuple[float, float], size
) -> np.ndarray:
    """Uniform magnitudes in ``magnitude_range`` with independent random signs."""
    lo, hi = magnitude_range
    magnitudes = rng.uniform(lo, hi, size=size)
    signs = rng.choice([-1.0, 1.0], size=size)
    return magnitudes * signs


def _log_uniform_stds(
    rng: np.random.Generator, sigma2_range: Tuple[float, float], size
) -> np.ndarray:
    """Standard deviations whose *variance* is log-uniform in ``sigma2_range``."""
    lo, hi = sigma2_range
    sigma2 = np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))
    return np.sqrt(sigma2)


def _sample_gmm_means(
    rng: np.random.Generator,
    n_components: int,
    mean_range: Tuple[float, float],
    min_gap: float,
) -> np.ndarray:
    """Rejection-sample means in ``mean_range`` with min pairwise gap ``min_gap``."""
    lo, hi = mean_range
    for _ in range(_RNG_MAX_TRIES):
        candidates = rng.uniform(lo, hi, size=n_components)
        if n_components == 1 or np.min(np.diff(np.sort(candidates))) >= min_gap:
            return candidates
    raise ValueError(
        f"Could not place {n_components} means in {mean_range} with min pairwise gap "
        f"{min_gap} after {_RNG_MAX_TRIES} tries; widen the range or shrink the gap."
    )


def _parse_edge_coefs(skeleton: SCMSkeleton, edge_coefs: Dict[str, float]) -> Dict[str, Dict]:
    """Validate ``{"parent->child": value}`` overrides and group them by child."""
    parsed: Dict[str, Dict] = {}
    for key, value in (edge_coefs or {}).items():
        parts = key.split("->")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"edge_coefs key {key!r} must look like 'parent->child'.")
        parent, child = parts
        if (parent, child) not in set(skeleton.edges):
            raise ValueError(f"edge_coefs: no edge {parent!r} -> {child!r} in the skeleton.")
        if skeleton.spec_of(parent).kind != "continuous":
            raise ValueError(
                f"edge_coefs: parent {parent!r} is discrete — linear coefficients are only "
                "defined for continuous parents (confounding through discrete parents lives "
                "in the CPTs / CLG regimes and cannot be pinned this way)."
            )
        parsed.setdefault(child, {})[parent] = float(value)
    return parsed


def _apply_edge_coefs(mechanism: Mechanism, overrides: Dict[str, float]) -> None:
    """Pin linear coefficients on ``mechanism`` in place (child already validated)."""
    for parent, value in overrides.items():
        if isinstance(mechanism, LinearGMMMechanism):
            mechanism.coefficients[parent] = value
        elif isinstance(mechanism, CLGMechanism):
            idx = mechanism.continuous_parent_names.index(parent)
            mechanism.coefficients[:, idx] = value
        else:  # pragma: no cover - guarded by _parse_edge_coefs
            raise ValueError(
                f"Cannot pin coefficient on {parent!r}: mechanism {type(mechanism).__name__} "
                "has no linear coefficients."
            )


def _linear_gmm(name: str, scm: StructuralCausalModel) -> LinearGMMMechanism:
    mechanism = scm.get_node_data(name)
    if not isinstance(mechanism, LinearGMMMechanism):
        raise ValueError(
            f"effect engineering: {name!r} has a {type(mechanism).__name__}, not a linear "
            "mechanism — --direct_effect/--confounding_strength need continuous variables "
            "with linear mechanisms."
        )
    return mechanism


def _engineer_effect(
    skeleton: SCMSkeleton,
    scm: StructuralCausalModel,
    direct_effect,
    confounding_strength,
) -> None:
    """Translate ``--direct_effect`` / ``--confounding_strength`` into coefficients.

    Semantics (backdoor-style skeletons with continuous linear mechanisms):

    * ``direct_effect V`` — every treatment->outcome edge coefficient is pinned to V,
      so P(Y|do(X)) has slope V in X.
    * ``confounding_strength S`` — the confounder->X coefficients stay random; the
      confounder->Y coefficients are solved so the *observational* regression slope of
      Y on X, measured over the central +/-1 std window of X, becomes
      (direct effect) + S.

    The solve is exact rather than closed-form because the noise is Gaussian-mixture:
    E[Y|X=x] is a ratio of mixtures and its slope is not Cov/Var. But E[Y|X=x] is
    affine in the confounder->Y coefficient vector (Y = w_xy X + sum_z c_z Z + noise),
    so one probe — set all confounder->Y coefficients to 1, measure the slope through
    the exact ground truth, rescale to hit S — lands exactly.

    Thus ``--direct_effect -2 --confounding_strength 2`` yields an observational
    regression with zero slope over the bulk of X while do(X) moves Y with slope -2.
    Sign of S gives the bias direction; S = 0 makes P(Y|X) and P(Y|do(X)) coincide in
    slope over that window (up to the mixture's mild nonlinearity in the tails).

    Raises for discrete confounders (bias lives in CPTs/CLG regimes there), multiple
    treatments (no single slope to target), frontdoor skeletons (no direct edge;
    effects are mediated) and non-root confounders.
    """
    x_names = [v.name for v in skeleton.variables if v.name == "X" or v.name.startswith("X_")]
    y_names = [v.name for v in skeleton.variables if v.name == "Y" or v.name.startswith("Y_")]
    edge_set = set(skeleton.edges)

    if direct_effect is not None:
        direct_edges = [(x, y) for x in x_names for y in y_names if (x, y) in edge_set]
        if not direct_edges:
            raise ValueError(
                "effect engineering: no treatment->outcome (X*->Y*) edges in this skeleton "
                "(frontdoor effects are mediated) — --direct_effect does not apply."
            )
        for x, y in direct_edges:
            _linear_gmm(y, scm).coefficients[x] = float(direct_effect)

    if confounding_strength is None:
        return
    if len(x_names) != 1:
        raise ValueError(
            "effect engineering: --confounding_strength targets the regression slope of a "
            f"single treatment, but this skeleton has {len(x_names)} treatments (X*)."
        )
    (x_name,) = x_names
    _linear_gmm(x_name, scm)  # validate X carries a linear mechanism

    confounders = [
        p for p in skeleton.parents_of(x_name) if any((p, y) in edge_set for y in y_names)
    ]
    if not confounders:
        raise ValueError(
            "effect engineering: no confounders (shared parents of X and Y) in this "
            "skeleton — cannot engineer --confounding_strength."
        )
    discrete = [z for z in confounders if skeleton.spec_of(z).kind != "continuous"]
    usable = [z for z in confounders if skeleton.spec_of(z).kind == "continuous"]
    if discrete and not usable:
        raise ValueError(
            "effect engineering: the confounders are discrete — their bias lives in the "
            "CPTs / CLG regimes and cannot be set via a linear strength. Use continuous "
            "confounders (--kind continuous, or --z_kind continuous under --kind mixed)."
        )
    for z in usable:
        if skeleton.parents_of(z):
            raise ValueError(
                f"effect engineering: confounder {z!r} is not a root node; the bias solve "
                "assumes root confounders (as built by the backdoor factory)."
            )

    # Probe: unit confounder->Y coefficients with the direct X->Y path removed,
    # measure the confounding-only slope through the exact ground truth, rescale
    # (the slope is affine in c). GroundTruth snapshots mechanism parameters at
    # construction, so it is rebuilt inside obs_slope after every mutation.
    # The direct path is only touched when it exists: assigning coefficients[x]
    # on a mediated outcome (no X->Y edge) would make the mechanism declare a
    # phantom parent and break GroundTruth's parent consistency check.
    saved_direct = {}
    has_direct = {}
    for y in y_names:
        y_mech = _linear_gmm(y, scm)
        has_direct[y] = (x_name, y) in edge_set
        saved_direct[y] = y_mech.coefficients.get(x_name, 0.0)
        if has_direct[y]:
            y_mech.coefficients[x_name] = 0.0
        for z in usable:
            if (z, y) in edge_set:
                y_mech.coefficients[z] = 1.0

    def obs_slope():
        gt = scm.ground_truth()
        gm_x = gt.marginal_density([x_name])
        mu = float(gm_x.mean[0])
        std = float(np.sqrt(gm_x.cov[0, 0]))
        g0 = gt.marginal_density(y_names, evidence={x_name: mu - std})
        g1 = gt.marginal_density(y_names, evidence={x_name: mu + std})
        return (g1.mean - g0.mean) / (2.0 * std)

    slope_per_unit = obs_slope()
    for i, y in enumerate(y_names):
        unit = float(slope_per_unit[i])
        if abs(unit) < 1e-8:
            raise ValueError(
                f"effect engineering: confounders have no measured influence on {y!r} "
                "through X — cannot engineer --confounding_strength (check the skeleton)."
            )
        scale = float(confounding_strength) / unit
        y_mech = _linear_gmm(y, scm)
        if has_direct[y]:
            y_mech.coefficients[x_name] = saved_direct[y]
        for z in usable:
            if (z, y) in edge_set:
                y_mech.coefficients[z] = scale


def _random_discrete_mechanism(
    spec,
    parent_cardinalities: Dict[str, int],
    rng: np.random.Generator,
    discrete_strategy: str,
    regions: Union[int, Dict[str, int]],
    dirichlet_alpha: float,
) -> Mechanism:
    if discrete_strategy == "regional":
        # Missing dict entries fall back to the function-level default of 2.
        n_regions = regions.get(spec.name, 2) if isinstance(regions, dict) else regions
        return RegionalDiscreteMechanism.random(
            cardinality=spec.cardinality,
            parent_cardinalities=parent_cardinalities,
            n_regions=n_regions,
            rng=rng,
        )
    if discrete_strategy == "dirichlet":
        return DirichletCPTMechanism.random(
            cardinality=spec.cardinality,
            parent_cardinalities=parent_cardinalities,
            alpha=dirichlet_alpha,
            rng=rng,
        )
    raise ValueError(
        f"Unknown discrete_strategy {discrete_strategy!r}; expected 'regional' or 'dirichlet'."
    )


def _random_linear_gmm_mechanism(
    continuous_parents: List[str],
    rng: np.random.Generator,
    gmm_components: Sequence[int],
    coef_range: Tuple[float, float],
    intercept_range: Tuple[float, float],
    sigma2_range: Tuple[float, float],
    gmm_mean_range: Tuple[float, float],
    gmm_min_mean_gap: float,
) -> LinearGMMMechanism:
    coefficients = {
        name: float(coef)
        for name, coef in zip(
            continuous_parents, _signed_uniform(rng, coef_range, len(continuous_parents))
        )
    }
    intercept = float(rng.uniform(*intercept_range))
    n_components = int(rng.choice(list(gmm_components)))
    weights = rng.dirichlet(np.ones(n_components))
    means = _sample_gmm_means(rng, n_components, gmm_mean_range, gmm_min_mean_gap)
    stds = _log_uniform_stds(rng, sigma2_range, n_components)
    return LinearGMMMechanism(
        coefficients=coefficients,
        intercept=intercept,
        noise=GaussianMixtureNoise(weights=weights, means=means, stds=stds),
    )


def _random_clg_mechanism(
    discrete_parent_cardinalities: Dict[str, int],
    continuous_parents: List[str],
    rng: np.random.Generator,
    coef_range: Tuple[float, float],
    intercept_range: Tuple[float, float],
    sigma2_range: Tuple[float, float],
) -> CLGMechanism:
    n_configs = int(np.prod(list(discrete_parent_cardinalities.values()), dtype=np.int64))
    intercepts = rng.uniform(*intercept_range, size=n_configs)
    coefficients = _signed_uniform(rng, coef_range, (n_configs, len(continuous_parents)))
    stds = _log_uniform_stds(rng, sigma2_range, n_configs)
    return CLGMechanism(
        discrete_parent_cardinalities=discrete_parent_cardinalities,
        continuous_parent_names=continuous_parents,
        intercepts=intercepts,
        coefficients=coefficients,
        stds=stds,
    )


def randomize_mechanisms(
    skeleton: SCMSkeleton,
    rng: Union[int, np.random.Generator],
    *,
    discrete_strategy: str = "regional",
    regions: Union[int, Dict[str, int]] = 2,
    dirichlet_alpha: float = 1.0,
    gmm_components: Sequence[int] = (2, 3, 4),
    coef_range: Tuple[float, float] = (0.5, 2.0),
    intercept_range: Tuple[float, float] = (-1.0, 1.0),
    sigma2_range: Tuple[float, float] = (0.25, 2.0),
    gmm_mean_range: Tuple[float, float] = (-3.0, 3.0),
    gmm_min_mean_gap: float = 1.0,
    edge_coefs: Dict[str, float] = None,
    direct_effect: float = None,
    confounding_strength: float = None,
) -> StructuralCausalModel:
    """Draw random mechanisms on top of ``skeleton`` and return the SCM.

    Nodes are added in a topological order of the skeleton; hidden variables
    get mechanisms like any other node and are flagged via ``hidden=True``.

    ``edge_coefs`` pins linear coefficients after the random draw, e.g.
    ``{"X->Y": 0.0}``: the ``X->Y`` coefficient of Y's mechanism is set to 0
    while everything else stays random. Only edges with a *continuous* parent
    can be pinned (see ``_parse_edge_coefs`` for validation).

    ``direct_effect`` / ``confounding_strength`` are the high-level dials for the
    gap between P(Y|do(X)) and P(Y|X); see ``_engineer_effect`` for the exact
    semantics. They are applied after ``edge_coefs`` and take precedence on the
    treatment->outcome and confounder->outcome edges.
    """
    rng = _as_generator(rng)
    overrides_by_child = _parse_edge_coefs(skeleton, edge_coefs)
    scm = StructuralCausalModel()

    for name in skeleton.topological_order():
        spec = skeleton.spec_of(name)
        parents = skeleton.parents_of(name)
        discrete_parents = [p for p in parents if skeleton.spec_of(p).kind == "discrete"]
        continuous_parents = [p for p in parents if skeleton.spec_of(p).kind == "continuous"]

        if spec.kind == "discrete":
            mechanism = _random_discrete_mechanism(
                spec,
                parent_cardinalities={p: skeleton.spec_of(p).cardinality for p in parents},
                rng=rng,
                discrete_strategy=discrete_strategy,
                regions=regions,
                dirichlet_alpha=dirichlet_alpha,
            )
        elif discrete_parents:
            mechanism = _random_clg_mechanism(
                discrete_parent_cardinalities={
                    p: skeleton.spec_of(p).cardinality for p in discrete_parents
                },
                continuous_parents=continuous_parents,
                rng=rng,
                coef_range=coef_range,
                intercept_range=intercept_range,
                sigma2_range=sigma2_range,
            )
        else:
            mechanism = _random_linear_gmm_mechanism(
                continuous_parents=continuous_parents,
                rng=rng,
                gmm_components=gmm_components,
                coef_range=coef_range,
                intercept_range=intercept_range,
                sigma2_range=sigma2_range,
                gmm_mean_range=gmm_mean_range,
                gmm_min_mean_gap=gmm_min_mean_gap,
            )

        scm.add_variable(name, mechanism=mechanism, parents=parents, hidden=spec.hidden)
        if name in overrides_by_child:
            _apply_edge_coefs(mechanism, overrides_by_child[name])

    if direct_effect is not None or confounding_strength is not None:
        _engineer_effect(skeleton, scm, direct_effect, confounding_strength)

    return scm


def sample_dataset(scm: StructuralCausalModel, n: int, drop_hidden: bool = True):
    """Thin wrapper over ``StructuralCausalModel.sample_dataset`` for pipeline ergonomics."""
    return scm.sample_dataset(n, drop_hidden=drop_hidden)

"""Unified exact ground-truth engine for supported SCM classes.

Every supported SCM is a mixture over an enumerable discrete part of
linear-Gaussian-mixture (GMM) continuous joints:

- **pure discrete** — empty continuous part; answers come straight from variable
  elimination over the exact CPTs.
- **pure continuous** — a single discrete configuration; answers come from
  ancestral linear-Gaussian propagation.
- **mixed (CLG)** — discrete nodes have only discrete parents
  (:meth:`~.graph.StructuralCausalModel.validate_clg_structure`), so the discrete
  part is self-contained and each discrete configuration ``d`` induces a
  continuous GMM; the joint law is ``sum_d p_d * GMM_d``.

Mechanisms opt in via :class:`~.mechanisms.TabularMechanism` (discrete) or by
exposing ``gaussian_params`` (:class:`~.mechanisms.ConditionalLinearGaussian`
and the linear-Gaussian special cases of ``AdditiveNoiseMechanism``). Anything
else raises ``ValueError`` — no analytical ground truth exists for it.

Hidden variables are ordinary nodes inside the engine: they are marginalized
analytically like any other non-target variable.
"""

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .graph import StructuralCausalModel
from .mechanisms import ConditionalLinearGaussian, GaussianParams, TabularMechanism


_LOG_2PI = np.log(2.0 * np.pi)

# A labeled discrete factor: axis i of the table corresponds to names[i].
Factor = Tuple[Tuple[str, ...], np.ndarray]


def _mvn_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    d = len(mean)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance matrix is not positive definite.")
    delta = x - mean
    sol = np.linalg.solve(cov, delta)
    return float(-0.5 * (d * _LOG_2PI + logdet + delta @ sol))


class GaussianMixture:
    """A d-dimensional Gaussian mixture with labeled dimensions.

    ``weights`` (K,), ``means`` (K, d), ``covs`` (K, d, d); ``var_names`` labels
    the d dimensions. Weights are renormalized on construction.
    """

    def __init__(
        self,
        weights: Sequence[float],
        means: np.ndarray,
        covs: np.ndarray,
        var_names: Sequence[str],
    ):
        self.var_names = tuple(var_names)
        d = len(self.var_names)
        self.weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        self.means = np.asarray(means, dtype=np.float64)
        self.covs = np.asarray(covs, dtype=np.float64)
        k = len(self.weights)
        if self.means.shape != (k, d) or self.covs.shape != (k, d, d):
            raise ValueError(
                f"means must have shape ({k}, {d}) and covs ({k}, {d}, {d}); "
                f"got {self.means.shape} and {self.covs.shape}."
            )
        if k == 0 or np.any(self.weights < 0):
            raise ValueError("weights must be non-empty and non-negative.")
        total = self.weights.sum()
        if total <= 0:
            raise ValueError("weights must have a positive sum.")
        self.weights = self.weights / total

    @property
    def n_components(self) -> int:
        return len(self.weights)

    @property
    def dim(self) -> int:
        return len(self.var_names)

    @property
    def mean(self) -> np.ndarray:
        """Mixture mean, shape (d,)."""
        return (self.weights[:, None] * self.means).sum(axis=0)

    @property
    def cov(self) -> np.ndarray:
        """Mixture covariance (law of total covariance), shape (d, d)."""
        m = self.mean
        second = (
            self.weights[:, None, None]
            * (self.covs + np.einsum("ki,kj->kij", self.means, self.means))
        ).sum(axis=0)
        return second - np.outer(m, m)

    def marginal(self, names: Sequence[str]) -> "GaussianMixture":
        """Marginal mixture over ``names`` (order preserved)."""
        idx = []
        for n in names:
            if n not in self.var_names:
                raise KeyError(f"Unknown variable '{n}' in mixture over {self.var_names}.")
            idx.append(self.var_names.index(n))
        return GaussianMixture(
            self.weights,
            self.means[:, idx],
            self.covs[:, idx][:, :, idx],
            tuple(names),
        )

    def condition(self, evidence: Dict[str, float]) -> "GaussianMixture":
        """Condition on exact point evidence (per-component Gaussian conditioning).

        Component weights are renormalized by each component's evidence density.
        """
        if not evidence:
            return self
        ev_names = list(evidence)
        x = np.array([float(evidence[n]) for n in ev_names], dtype=np.float64)
        idx_e = [self.var_names.index(n) for n in ev_names]
        idx_r = [i for i, n in enumerate(self.var_names) if n not in evidence]
        rest_names = tuple(self.var_names[i] for i in idx_r)

        mu_e = self.means[:, idx_e]
        mu_r = self.means[:, idx_r]
        s_ee = self.covs[:, idx_e][:, :, idx_e]
        s_re = self.covs[:, idx_r][:, :, idx_e]
        s_rr = self.covs[:, idx_r][:, :, idx_r]

        new_means = np.zeros((self.n_components, len(idx_r)))
        new_covs = np.zeros((self.n_components, len(idx_r), len(idx_r)))
        densities = np.zeros(self.n_components)
        for k in range(self.n_components):
            delta = x - mu_e[k]
            sol = np.linalg.solve(s_ee[k], delta)
            new_means[k] = mu_r[k] + s_re[k] @ sol
            new_covs[k] = s_rr[k] - s_re[k] @ np.linalg.solve(s_ee[k], s_re[k].T)
            densities[k] = np.exp(_mvn_logpdf(x, mu_e[k], s_ee[k]))

        new_weights = self.weights * densities
        if new_weights.sum() <= 0:
            raise ValueError("Evidence has zero density under every mixture component.")
        return GaussianMixture(new_weights, new_means, new_covs, rest_names)

    def evidence_pdf(self, evidence: Dict[str, float]) -> float:
        """Mixture density evaluated at the evidence point (1.0 for empty evidence)."""
        if not evidence:
            return 1.0
        marg = self.marginal(list(evidence))
        x = np.array([float(evidence[n]) for n in evidence], dtype=np.float64)
        total = 0.0
        for w, m, c in zip(marg.weights, marg.means, marg.covs):
            total += w * np.exp(_mvn_logpdf(x, m, c))
        return float(total)

    def pdf(self, points: np.ndarray):
        """Mixture density at points.

        Accepts a single point of shape (d,) or a batch of shape (N, d); for
        1-dimensional mixtures a 1-D array is treated as N scalar points.
        """
        points = np.asarray(points, dtype=np.float64)
        single = points.ndim == 1 and self.dim > 1
        pts = points[None, :] if single else points.reshape(-1, self.dim)
        if pts.shape[1] != self.dim:
            raise ValueError(f"Expected points with {self.dim} coordinates, got {pts.shape[1]}.")
        out = np.zeros(len(pts))
        for w, m, c in zip(self.weights, self.means, self.covs):
            out += w * np.exp([_mvn_logpdf(p, m, c) for p in pts])
        return float(out[0]) if single else out

    def __repr__(self) -> str:
        return f"GaussianMixture(K={self.n_components}, vars={self.var_names})"


# ---------------------------------------------------------------------------
# Labeled-factor engine for the discrete part
# ---------------------------------------------------------------------------


def _clamp_factor(factor: Factor, assignment: Dict[str, int]) -> Factor:
    """Condition a factor on a point assignment (drops the clamped axes)."""
    names, table = factor
    names = list(names)
    for var, value in assignment.items():
        if var not in names:
            continue
        axis = names.index(var)
        table = np.take(table, int(value), axis=axis)
        del names[axis]
    return tuple(names), table


def _factor_product(f1: Factor, f2: Factor) -> Factor:
    """Product of two labeled factors via axis-aligned broadcasting."""
    names1, t1 = f1
    names2, t2 = f2
    names = list(names1) + [n for n in names2 if n not in names1]
    a = t1.reshape(list(t1.shape) + [1] * (len(names) - len(names1)))
    order = [names2.index(n) for n in names if n in names2]
    b = np.transpose(t2, axes=order)
    b = b.reshape([t2.shape[names2.index(n)] if n in names2 else 1 for n in names])
    return tuple(names), a * b


def _variable_elimination(factors: List[Factor], keep: set) -> Factor:
    """Eliminate every variable not in ``keep``; returns one factor over ``keep``."""
    factors = list(factors)
    present = set()
    for names, _ in factors:
        present.update(names)
    for var in [v for v in present if v not in keep]:
        involved = [f for f in factors if var in f[0]]
        factors = [f for f in factors if var not in f[0]]
        acc = involved[0]
        for f in involved[1:]:
            acc = _factor_product(acc, f)
        names, table = acc
        axis = names.index(var)
        table = table.sum(axis=axis)
        names = list(names)
        del names[axis]
        factors.append((tuple(names), table))
    if not factors:
        return (), np.array(1.0)
    acc = factors[0]
    for f in factors[1:]:
        acc = _factor_product(acc, f)
    return acc


# ---------------------------------------------------------------------------
# Continuous propagation helpers
# ---------------------------------------------------------------------------


def _append_constant(gm: GaussianMixture, name: str, value: float) -> GaussianMixture:
    """Append a degenerate dimension clamped to a constant (continuous do)."""
    d = gm.dim
    new_means = np.zeros((gm.n_components, d + 1))
    new_means[:, :d] = gm.means
    new_means[:, d] = value
    new_covs = np.zeros((gm.n_components, d + 1, d + 1))
    new_covs[:, :d, :d] = gm.covs
    return GaussianMixture(gm.weights.copy(), new_means, new_covs, gm.var_names + (name,))


def _extend_with_gaussian(
    gm: GaussianMixture,
    name: str,
    params: GaussianParams,
    clamped: Dict[str, float],
) -> GaussianMixture:
    """Extend the running GMM by ``V = intercept + coefs @ parents + noise``.

    Each existing component crosses each noise component. Parents clamped by a
    continuous do enter the mean with their clamped value and contribute nothing
    to the covariance.
    """
    idx = {n: i for i, n in enumerate(gm.var_names)}
    noise = params.noise
    k_noise = noise.n_components
    g_old, d = gm.n_components, gm.dim

    unknown = [p for p in params.coefficients if p not in idx]
    if unknown:
        raise ValueError(
            f"Coefficient parents {unknown} of node '{name}' are not continuous ancestors."
        )

    new_weights = np.repeat(gm.weights, k_noise) * np.tile(noise.weights, g_old)
    new_means = np.zeros((g_old * k_noise, d + 1))
    new_covs = np.zeros((g_old * k_noise, d + 1, d + 1))

    free = [(p, c) for p, c in params.coefficients.items() if p not in clamped]
    for k in range(k_noise):
        rows = np.arange(k, g_old * k_noise, k_noise)  # row g*k_noise + k <-> (g, k)
        new_means[rows, :d] = gm.means
        new_covs[rows, :d, :d] = gm.covs

        mu = np.full(g_old, params.intercept, dtype=np.float64)
        for parent, coef in params.coefficients.items():
            if parent in clamped:
                mu += coef * clamped[parent]
            else:
                mu += coef * gm.means[:, idx[parent]]
        new_means[rows, d] = mu + noise.means[k]

        cross = np.zeros((g_old, d))  # Sigma @ w
        for parent, coef in free:
            cross += coef * gm.covs[:, :, idx[parent]]
        new_covs[rows, d, :d] = cross
        new_covs[rows, :d, d] = cross
        variance = noise.stds[k] ** 2 + sum(coef * cross[:, idx[parent]] for parent, coef in free)
        new_covs[rows, d, d] = variance

    return GaussianMixture(new_weights, new_means, new_covs, gm.var_names + (name,))


def _stack_mixtures(
    mixes: List[GaussianMixture], weights: np.ndarray, var_names: Sequence[str]
) -> GaussianMixture:
    """Concatenate same-dimension mixtures into one, weighting by ``weights``."""
    all_weights = np.concatenate([w * m.weights for w, m in zip(weights, mixes)])
    all_means = np.concatenate([m.means for m in mixes], axis=0)
    all_covs = np.concatenate([m.covs for m in mixes], axis=0)
    return GaussianMixture(all_weights, all_means, all_covs, tuple(var_names))


class GroundTruth:
    """Exact ground-truth engine for an SCM with tractable mechanisms.

    Node classification: :class:`~.mechanisms.TabularMechanism` -> discrete;
    mechanisms exposing ``gaussian_params`` (:class:`ConditionalLinearGaussian`,
    linear/Gaussian ``AdditiveNoiseMechanism``) -> continuous; everything else
    raises ``ValueError`` (no analytical ground truth).
    """

    def __init__(self, scm: StructuralCausalModel):
        self.scm = scm
        scm.validate_clg_structure()
        self._topo = list(scm.topological_sort())

        discrete, continuous = [], []
        for name in self._topo:
            mech = scm.get_node_data(name)
            if isinstance(mech, TabularMechanism):
                discrete.append(name)
            elif hasattr(mech, "gaussian_params"):
                continuous.append(name)
            else:
                raise ValueError(
                    f"No analytical ground truth for node '{name}' "
                    f"(mechanism {type(mech).__name__})."
                )
        self.discrete_vars: Tuple[str, ...] = tuple(discrete)
        self.continuous_vars: Tuple[str, ...] = tuple(continuous)
        self._discrete_set = frozenset(discrete)
        self._continuous_set = frozenset(continuous)
        self.cards: Dict[str, int] = {n: int(scm.get_node_data(n).cardinality) for n in discrete}
        self._check_parent_consistency()

    def _check_parent_consistency(self):
        for name in self.discrete_vars:
            mech = self.scm.get_node_data(name)
            if set(mech.parent_cardinalities()) != set(self.scm.get_parents(name)):
                raise ValueError(
                    f"CPT parents of discrete node '{name}' do not match the graph parents."
                )
        for name in self.continuous_vars:
            mech = self.scm.get_node_data(name)
            if isinstance(mech, ConditionalLinearGaussian):
                dp = set(mech.discrete_parent_cardinalities)
                if not dp <= self._discrete_set:
                    raise ValueError(
                        f"Discrete parents {sorted(dp - self._discrete_set)} of continuous node "
                        f"'{name}' are not discrete variables."
                    )
                expected = dp | set(mech.continuous_parent_names)
                if expected != set(self.scm.get_parents(name)):
                    raise ValueError(
                        f"Declared parents of continuous node '{name}' do not match the graph."
                    )

    def classify(self) -> Dict[str, List[str]]:
        """Which variables are treated as discrete vs continuous."""
        return {"discrete": list(self.discrete_vars), "continuous": list(self.continuous_vars)}

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _split_by_kind(self, mapping: Optional[dict], what: str):
        """Split a do/evidence dict into (discrete, continuous), validating names."""
        disc, cont = {}, {}
        for key, value in (mapping or {}).items():
            if key in self._discrete_set:
                ivalue = int(value)
                if not 0 <= ivalue < self.cards[key]:
                    raise ValueError(
                        f"{what} value {value} out of range [0, {self.cards[key]}) for '{key}'."
                    )
                disc[key] = ivalue
            elif key in self._continuous_set:
                cont[key] = float(value)
            else:
                raise ValueError(f"Unknown variable '{key}' in {what}.")
        return disc, cont

    # ------------------------------------------------------------------
    # Discrete part
    # ------------------------------------------------------------------

    def _discrete_factors(self, d_do: Dict[str, int]) -> List[Factor]:
        """Truncated factorization: delete do'd CPTs, clamp do values elsewhere."""
        factors = []
        for name in self.discrete_vars:
            if name in d_do:
                continue
            mech = self.scm.get_node_data(name)
            parents = [p for p in self.scm.get_parents(name) if p in self._discrete_set]
            table = np.asarray(mech.cpt(parent_names=parents), dtype=np.float64)
            factors.append((tuple(parents) + (name,), table))
        if d_do:
            factors = [_clamp_factor(f, d_do) for f in factors]
        return factors

    def _joint_table(self, do: Optional[dict]) -> np.ndarray:
        """Interventional joint table over ``self.discrete_vars`` (axis order)."""
        d_do, _ = self._split_by_kind(do, "do")
        factors = self._discrete_factors(d_do)
        names, table = _variable_elimination(factors, set(self.discrete_vars))
        names = list(names)
        for var in self.discrete_vars:
            if var in names:
                continue
            # do-clamped variable: point mass at the intervened value.
            expanded = np.zeros(table.shape + (self.cards[var],))
            expanded[..., d_do[var]] = table
            table = expanded
            names.append(var)
        order = [names.index(v) for v in self.discrete_vars]
        return np.transpose(table, axes=order)

    def discrete_joint(self, do: Optional[dict] = None) -> Dict[Tuple[int, ...], float]:
        """Exact (interventional) joint over all discrete variables.

        Keys are configurations in ``self.discrete_vars`` order.
        """
        if not self.discrete_vars:
            return {(): 1.0}
        table = self._joint_table(do)
        configs = itertools.product(*[range(self.cards[v]) for v in self.discrete_vars])
        return {tuple(int(x) for x in c): float(table[c]) for c in configs}

    def _discrete_posterior(
        self, do: Optional[dict], d_evidence: Dict[str, int]
    ) -> Tuple[np.ndarray, List[Dict[str, int]]]:
        """Posterior over full discrete configurations given discrete evidence.

        Returns (probabilities, config dicts); configs enumerate the non-evidence
        variables and are extended with the evidence assignment.
        """
        if not self.discrete_vars:
            return np.ones(1), [dict(d_evidence)]
        table = self._joint_table(do)
        remaining = list(self.discrete_vars)
        for var, value in d_evidence.items():
            axis = remaining.index(var)
            table = np.take(table, value, axis=axis)
            del remaining[axis]
        flat = table.reshape(-1)
        total = flat.sum()
        if total <= 0:
            raise ValueError("Evidence has zero probability under the intervention.")
        configs = itertools.product(*[range(self.cards[v]) for v in remaining])
        keys = []
        for c in configs:
            key = dict(zip(remaining, (int(x) for x in c)))
            key.update(d_evidence)
            keys.append(key)
        return flat / total, keys

    # ------------------------------------------------------------------
    # Continuous part
    # ------------------------------------------------------------------

    def _continuous_ancestors(self, needed: set) -> set:
        ancestors = set()
        stack = [n for n in needed if n in self._continuous_set]
        while stack:
            node = stack.pop()
            if node in ancestors:
                continue
            ancestors.add(node)
            for parent in self.scm.get_parents(node):
                if parent in self._continuous_set:
                    stack.append(parent)
        return ancestors

    def _continuous_gmm(
        self, config: Dict[str, int], target_names: List[str], do: Optional[dict]
    ) -> GaussianMixture:
        """Joint GMM over the ancestors of ``target_names`` for one discrete config.

        Only query-relevant continuous ancestors (plus do-clamped variables) are
        materialized; the running GMM is extended one dimension per node in
        topological order.
        """
        _, c_do = self._split_by_kind(do, "do")
        needed = set(target_names) | set(c_do)
        ancestors = self._continuous_ancestors(needed)
        order = [n for n in self._topo if n in ancestors]

        gm = GaussianMixture(np.ones(1), np.zeros((1, 0)), np.zeros((1, 0, 0)), ())
        for name in order:
            if name in c_do:
                gm = _append_constant(gm, name, c_do[name])
                continue
            mech = self.scm.get_node_data(name)
            if isinstance(mech, ConditionalLinearGaussian):
                dcards = mech.discrete_parent_cardinalities
            else:
                dcards = {}
            sub_config = tuple(int(config[p]) for p in dcards)
            params = mech.gaussian_params(sub_config)
            gm = _extend_with_gaussian(gm, name, params, c_do)
        return gm.marginal(list(target_names))

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def marginal_prob(
        self,
        targets: Dict[str, int],
        do: Optional[dict] = None,
        evidence: Optional[dict] = None,
    ) -> float:
        """Exact P(targets | do, evidence) for discrete targets."""
        targets = dict(targets)
        if not targets:
            raise ValueError("targets must be non-empty.")
        for name, value in targets.items():
            if name not in self._discrete_set:
                raise ValueError(
                    f"marginal_prob is for discrete targets; '{name}' is not discrete."
                )
            if not 0 <= int(value) < self.cards[name]:
                raise ValueError(f"Target value {value} out of range for '{name}'.")

        d_ev, c_ev = self._split_by_kind(evidence, "evidence")
        _, c_do = self._split_by_kind(do, "do")

        if c_ev:
            # Continuous evidence reweights whole discrete configurations, so VE
            # over the discrete part alone is insufficient — enumerate configs.
            probs, keys = self._discrete_posterior(do, d_ev)
            weighted = []
            for p, key in zip(probs, keys):
                gm = self._continuous_gmm(key, list(c_ev), do)
                weighted.append(p * gm.evidence_pdf(c_ev))
            total = sum(weighted)
            if total <= 0:
                raise ValueError("Evidence has zero density under the model.")
            matches = sum(
                w
                for w, key in zip(weighted, keys)
                if all(key[t] == int(v) for t, v in targets.items())
            )
            return float(matches / total)

        d_do, _ = self._split_by_kind(do, "do")
        kept = {}
        for name, value in targets.items():
            if name in d_do or name in d_ev:
                fixed = d_do.get(name, d_ev.get(name))
                if int(value) != fixed:
                    return 0.0
            else:
                kept[name] = int(value)
        factors = self._discrete_factors(d_do)
        factors = [_clamp_factor(f, d_ev) for f in factors]
        names, table = _variable_elimination(factors, set(kept))
        order = [list(names).index(t) for t in kept]
        table = np.transpose(table, axes=order)
        total = table.sum()
        if total <= 0:
            raise ValueError("Evidence has zero probability under the intervention.")
        entry = table[tuple(kept.values())] if kept else table
        return float(entry / total)

    def marginal_density(
        self,
        targets: List[str],
        do: Optional[dict] = None,
        evidence: Optional[dict] = None,
    ) -> GaussianMixture:
        """Exact marginal density model for continuous targets.

        Returns a :class:`GaussianMixture` over ``targets``: a mixture over
        discrete configurations ``d`` (posterior-weighted given discrete
        evidence) of the per-``d`` GMMs, each conditioned on the continuous
        evidence and reweighted by its evidence density.
        """
        targets = list(targets)
        if not targets:
            raise ValueError("targets must be non-empty.")
        for name in targets:
            if name not in self._continuous_set:
                raise ValueError(
                    f"marginal_density is for continuous targets; '{name}' is not continuous."
                )
        d_ev, c_ev = self._split_by_kind(evidence, "evidence")

        probs, keys = self._discrete_posterior(do, d_ev)
        all_names = list(targets) + [n for n in c_ev if n not in targets]
        mixes, weights = [], []
        for p, key in zip(probs, keys):
            if p == 0.0:
                continue
            gm = self._continuous_gmm(key, all_names, do)
            weight = p
            if c_ev:
                weight *= gm.evidence_pdf(c_ev)
                gm = gm.condition(c_ev)
            mixes.append(gm.marginal(targets))
            weights.append(weight)
        if not mixes:
            raise ValueError("No discrete configuration supports the query.")
        weights = np.asarray(weights, dtype=np.float64)
        if weights.sum() <= 0:
            raise ValueError("Evidence has zero density under the model.")
        weights /= weights.sum()
        return _stack_mixtures(mixes, weights, targets)

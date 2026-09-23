import math
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np
import torch
from scipy import stats

from src.utils import BitSet, ContinuousInterval, DiscreteInterval, Interval

from .base import ArithmeticNode


# Numerical floor to prevent NaN gradients in logsumexp when all branches are -inf
LOG_FLOOR = float("-inf")


class LeafLayer(ArithmeticNode):
    """Represents a leaf distribution (e.g., Gaussian) in the SPN."""

    def num_parameters(self) -> int:
        """Number of trainable scalar parameters in this leaf (0 by default)."""
        return 0


class ConstantLayer(LeafLayer):
    """Leaf that always returns log(1) = 0 for all inputs.

    Used to represent a marginalized variable in a compiled estimand circuit.
    The variable is still in scope (so the circuit remains smooth/decomposable)
    but contributes nothing to the density.

    ``unit_values`` stores the exact per-(group, unit) log-values the replaced
    subtree transmits when all its leaves are marginalized (set to log 1) —
    computed by direct evaluation in ``_marginalize``, so it stays exact even
    when the per-unit normalization assumption breaks (e.g. conditional
    circuits with stripped/renormalized weights). A plain marginalized leaf
    has all-zero values; a sparse SumLayer has 0 on alive units and -inf on
    units with an all-zero mask row. Standalone evaluation transmits these
    values; structural products (``_multiply``) replicate the partner layer's
    weights over them.
    """

    def __init__(
        self,
        scope: BitSet,
        md_set: Optional[BitSet] = None,
        num_groups: int = 1,
        num_nodes: int = 1,
        unit_values: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            scope=scope,
            num_nodes=num_nodes,
            num_groups=num_groups,
            md_set=md_set,
        )
        if unit_values is not None and unit_values.shape != (num_groups, num_nodes):
            raise ValueError(
                f"unit_values shape {tuple(unit_values.shape)} does not match "
                f"(num_groups, num_nodes) = ({num_groups}, {num_nodes})"
            )
        self.unit_values = unit_values.detach() if unit_values is not None else None

    def to(self, device: torch.device):
        if self.unit_values is not None:
            self.unit_values = self.unit_values.to(device)
        return super().to(device)

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        B = data.shape[0]
        if self.unit_values is not None:
            return self.unit_values.unsqueeze(0).expand(B, self.num_groups, self.num_nodes)
        return torch.zeros(B, self.num_groups, self.num_nodes, device=data.device)

    def __repr__(self):
        return f"ConstantLayer(scope={list(self.scope)}, num_nodes={self.num_nodes}), num_groups={self.num_groups})"


class ProductLeafLayer(LeafLayer):
    """Leaf that combines two leaves by adding their log-densities.

    Used for DetProd (support-compatible product) at the leaf level:
    log(p_A(x) * p_B(x)) = log p_A(x) + log p_B(x).
    """

    def __init__(self, leaf_a: LeafLayer, leaf_b: LeafLayer, expand_leaves: bool = False):
        super().__init__(
            scope=leaf_a.scope,
            num_nodes=leaf_a.num_nodes
            if not expand_leaves
            else leaf_a.num_nodes * leaf_b.num_nodes,
            num_groups=leaf_a.num_groups * leaf_b.num_groups,
            md_set=leaf_a.md_set,
        )
        assert expand_leaves or leaf_a.num_nodes == leaf_b.num_nodes, (
            "ProductLeafLayer requires both leaves to have the same number of nodes."
        )
        assert leaf_a.scope == leaf_b.scope or (leaf_a.scope.is_empty or leaf_b.scope.is_empty), (
            f"ProductLeafLayer requires both leaves to have the same scope or if not one leaf to be a constant. leaf_a.scope={leaf_a.scope}, leaf_b.scope={leaf_b.scope}"
        )
        self.leaf_a = leaf_a
        self.leaf_b = leaf_b
        self.expand_leaves = expand_leaves
        if hasattr(leaf_a, "var"):
            self.var = leaf_a.var

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        out_a = self.leaf_a.forward(data, children_outputs)
        out_b = self.leaf_b.forward(data, children_outputs)
        B, G_A, U_A = out_a.shape
        _, G_B, U_B = out_b.shape

        if self.expand_leaves:
            outer = out_a.view(B, G_A, 1, U_A, 1) + out_b.view(B, 1, G_B, 1, U_B)
        else:
            assert U_A == U_B, (
                "ProductLeafLayer requires both leaves to have the same number of units if expand_leaves is False."
            )
            outer = out_a.view(B, G_A, 1, U_A) + out_b.view(B, 1, G_B, U_B)

        new_U = U_A * U_B if self.expand_leaves else U_A

        return outer.reshape(B, G_A * G_B, new_U)

    def __repr__(self):
        return f"ProductLeafLayer(a={self.leaf_a}, b={self.leaf_b}, expand_leaves={self.expand_leaves})"


class IndicatorLeafLayer(LeafLayer):
    """Leaf node that returns 1"""

    def __init__(self, base_leaf: LeafLayer):
        super().__init__(
            scope=base_leaf.scope,
            num_nodes=base_leaf.num_nodes,
            num_groups=base_leaf.num_groups,
            md_set=base_leaf.md_set,
        )
        self.base_leaf = base_leaf

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        base_out = self.base_leaf.forward(data, children_outputs)
        indicator_out = base_out > LOG_FLOOR
        return torch.where(
            indicator_out, torch.zeros_like(base_out), torch.full_like(base_out, float("-inf"))
        )


class InstantiatedLeafLayer(LeafLayer):
    """Placeholder leaf node to represent clamping a variable to a specific value."""

    def __init__(self, base_leaf: LeafLayer, value: Any):
        super().__init__(
            scope=base_leaf.scope,
            num_nodes=base_leaf.num_nodes,
            num_groups=base_leaf.num_groups,
            md_set=base_leaf.md_set,
        )
        self.base_leaf = base_leaf
        self.node_supports = getattr(base_leaf, "node_supports", None)
        self.base_leaf = base_leaf
        self.value = value
        if hasattr(base_leaf, "var"):
            self.var = base_leaf.var

    def forward(self, data: torch.Tensor, children_outputs: list = None) -> torch.Tensor:
        data_clamped = data.clone()
        data_clamped[:, self.var] = self.value
        return self.base_leaf.forward(data_clamped, children_outputs)

    def __repr__(self):
        return f"InstantiatedLeafLayer(base={self.base_leaf}, val={self.value})"


class Distribution(ABC):
    """Base class for probability distributions used as leaves in SPNs."""

    def __init__(self, var: int, var_support: Interval):
        self.var = var
        self.var_support = var_support

    @abstractmethod
    def split_support(self, split_count: int, strategy: str = "quantile") -> List[Interval]:
        """Split the variable's support into `split_count` intervals for unit supports."""
        pass

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.var == other.var and self.var_support == other.var_support

    def __hash__(self):
        return hash((self.__class__, self.var, self.var_support))


class GaussianDistribution(Distribution):
    """Represents a Gaussian distribution specification."""

    def __init__(
        self,
        var: int,
        base_mean: float,
        base_stddev: float,
    ):
        super().__init__(
            var,
            ContinuousInterval(float("-inf"), float("inf"), include_low=False, include_high=False),
        )
        self.base_mean = float(base_mean)
        self.base_stddev = float(base_stddev)

    def split_support(self, split_count: int) -> List[Interval]:
        """For a Gaussian, we can split the real line into intervals.
        Supported strategies: 'quantile', 'perturbed_quantile'.
        """
        quantiles = np.linspace(0, 1, split_count + 1)

        boundaries = stats.norm.ppf(quantiles, loc=self.base_mean, scale=self.base_stddev)
        intervals = []
        for i in range(split_count):
            low = boundaries[i]
            high = boundaries[i + 1]
            intervals.append(
                ContinuousInterval(
                    low,
                    high,
                    include_low=True,
                    include_high=(i == split_count - 1),
                )
            )
        return intervals

    def __repr__(self):
        return f"GaussianDistribution(var={self.var}, base_mean={self.base_mean}, base_stddev={self.base_stddev})"

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.base_mean == other.base_mean
            and self.base_stddev == other.base_stddev
        )

    def __hash__(self):
        return hash((super().__hash__(), self.base_mean, self.base_stddev))


class SplineDistribution(Distribution):
    """Base distribution spec for learnable, continuous, disjoint-support spline leaves."""

    def __init__(self, var: int, base_mean: float, base_stddev: float):
        super().__init__(
            var,
            ContinuousInterval(float("-inf"), float("inf"), include_low=False, include_high=False),
        )
        self.base_mean = float(base_mean)
        self.base_stddev = float(base_stddev)

    def split_support(self, split_count: int) -> List[Interval]:
        """Split the real line into ``split_count`` Gaussian-quantile intervals."""
        quantiles = np.linspace(0, 1, split_count + 1)
        boundaries = stats.norm.ppf(quantiles, loc=self.base_mean, scale=self.base_stddev)
        intervals = []
        for i in range(split_count):
            low = boundaries[i]
            high = boundaries[i + 1]
            intervals.append(
                ContinuousInterval(
                    low,
                    high,
                    include_low=True,
                    include_high=(i == split_count - 1),
                )
            )
        return intervals

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(var={self.var}, "
            f"base_mean={self.base_mean}, base_stddev={self.base_stddev})"
        )

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.base_mean == other.base_mean
            and self.base_stddev == other.base_stddev
        )

    def __hash__(self):
        return hash((super().__hash__(), self.base_mean, self.base_stddev))


class LogLinearSplineDistribution(SplineDistribution):
    """Spec for piecewise log-linear (truncated exponential) spline leaves."""

    pass


class LinearSplineDistribution(SplineDistribution):
    """Spec for piecewise linear spline leaves."""

    pass


class QuadraticSplineDistribution(SplineDistribution):
    """Spec for piecewise quadratic spline leaves."""

    pass


class RationalQuadraticSplineDistribution(SplineDistribution):
    """Spec for rational-quadratic spline CDF leaves."""

    pass


class GaussianLeafLayer(LeafLayer):
    """Represents a Gaussian distribution leaf node."""

    def __init__(
        self,
        spec: GaussianDistribution,
        num_nodes: int = 1,
        num_groups: int = 1,
        node_supports: Optional[List[ContinuousInterval]] = None,
    ):
        super().__init__(BitSet([spec.var]), num_nodes=num_nodes, num_groups=num_groups)
        self.var = spec.var
        self.node_supports = node_supports

        base_m = spec.base_mean
        base_std = spec.base_stddev

        new_means = torch.zeros((num_groups, num_nodes), dtype=torch.float32)
        new_stddevs = torch.full((num_groups, num_nodes), base_std, dtype=torch.float32)

        if node_supports is not None:
            assert len(node_supports) == num_nodes
            for i in range(num_nodes):
                iv = node_supports[i]
                if iv is not None:
                    if iv.low > float("-inf") and iv.high < float("inf"):
                        for g in range(num_groups):
                            m_g = iv.low + (iv.high - iv.low) * (g + 0.5) / num_groups
                            base_s = (iv.high - iv.low) / (max(2.0, num_groups * 2.0))
                            s_g = base_s * float(np.random.uniform(0.8, 1.2))
                            new_means[g, i] = m_g
                            new_stddevs[g, i] = max(s_g, 1e-4)
                    elif iv.low > float("-inf"):
                        for g in range(num_groups):
                            m_g = iv.low + base_std * (g + 0.5)
                            s_g = base_std * float(np.random.uniform(0.8, 1.2))
                            new_means[g, i] = m_g
                            new_stddevs[g, i] = max(s_g, 1e-4)
                    elif iv.high < float("inf"):
                        for g in range(num_groups):
                            m_g = iv.high - base_std * (num_groups - g - 0.5)
                            s_g = base_std * float(np.random.uniform(0.8, 1.2))
                            new_means[g, i] = m_g
                            new_stddevs[g, i] = max(s_g, 1e-4)
                    else:
                        for g in range(num_groups):
                            m_g = base_m + base_std * (g - (num_groups - 1) / 2.0)
                            s_g = base_std * float(np.random.uniform(0.8, 1.2))
                            new_means[g, i] = m_g
                            new_stddevs[g, i] = max(s_g, 1e-4)
                else:
                    for g in range(num_groups):
                        m_g = base_m + base_std * (g - (num_groups - 1) / 2.0)
                        s_g = base_std * float(np.random.uniform(0.8, 1.2))
                        new_means[g, i] = m_g
                        new_stddevs[g, i] = max(s_g, 1e-4)
        else:
            for g in range(num_groups):
                for i in range(num_nodes):
                    # Spatially space out means slightly around the base mean
                    m_g = base_m + float(np.random.uniform(-base_std, base_std))
                    # Variance diversity
                    s_g = base_std * float(np.random.uniform(0.8, 1.2))
                    new_means[g, i] = m_g
                    new_stddevs[g, i] = max(s_g, 1e-4)

        self.means = new_means.requires_grad_(True)
        # Keep stddevs strictly positive by storing the unconstrained log-parameter
        # and exposing the positive value through a property.
        self._raw_stddevs = torch.log(new_stddevs.clamp(min=1e-6)).requires_grad_(True)

    @property
    def stddevs(self):
        return torch.exp(self._raw_stddevs)

    def to(self, device: torch.device):
        if isinstance(self.means, torch.Tensor):
            self.means = self.means.to(device)
            if not self.means.is_leaf:
                self.means = self.means.detach().requires_grad_(self.means.requires_grad)
        if isinstance(self._raw_stddevs, torch.Tensor):
            self._raw_stddevs = self._raw_stddevs.to(device)
            if not self._raw_stddevs.is_leaf:
                self._raw_stddevs = self._raw_stddevs.detach().requires_grad_(
                    self._raw_stddevs.requires_grad
                )
        return super().to(device)

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        x = data[:, self.var].unsqueeze(1).unsqueeze(2)  # [B, 1, 1]

        # means and stddevs are of size [num_groups, num_nodes], reshape to [B, num_groups, num_nodes] for broadcasting
        means = self.means.unsqueeze(0)  # [1, num_groups, num_nodes]
        stddevs = self.stddevs.unsqueeze(0)  # [1, num_groups, num_nodes]
        vars = stddevs**2
        log_scale = torch.log(stddevs * math.sqrt(2 * math.pi))
        log_prob = -((x - means) ** 2) / (2 * vars) - log_scale

        if hasattr(self, "node_supports") and self.node_supports and len(self.node_supports) > 1:
            H = log_prob.shape[2]
            assert H == len(self.node_supports) and H == means.shape[2], (
                "Invalid H, node_supports, or means shape"
            )
            sqrt2 = math.sqrt(2)
            for j in range(len(self.node_supports)):
                iv: ContinuousInterval = self.node_supports[j]
                if iv is not None and (iv.low > float("-inf") or iv.high < float("inf")):
                    # Strictly check the interval flags!
                    oob_low = (x < iv.low) if getattr(iv, "include_low", True) else (x <= iv.low)
                    oob_high = (
                        (x > iv.high) if getattr(iv, "include_high", True) else (x >= iv.high)
                    )
                    oob = oob_low | oob_high

                    log_prob[:, :, j] = torch.where(
                        oob.squeeze(1),
                        torch.full_like(log_prob[:, :, j], float("-inf")),
                        log_prob[:, :, j],
                    )
                    # Keep normalization so the truncated density integrates to ~1.
                    if iv.low > float("-inf"):
                        z_lo = (iv.low - means[:, :, j]) / (stddevs[:, :, j] * sqrt2)
                        cdf_lo = 0.5 * (1 + torch.erf(z_lo))
                    else:
                        cdf_lo = torch.zeros_like(means[:, :, j])

                    if iv.high < float("inf"):
                        z_hi = (iv.high - means[:, :, j]) / (stddevs[:, :, j] * sqrt2)
                        cdf_hi = 0.5 * (1 + torch.erf(z_hi))
                    else:
                        cdf_hi = torch.ones_like(means[:, :, j])

                    log_Z = torch.log((cdf_hi - cdf_lo).clamp(min=1e-12))
                    log_prob[:, :, j] = log_prob[:, :, j] - log_Z

        if log_prob.requires_grad:
            log_prob.retain_grad()

        # Marginalize over NaN values
        is_nan = torch.isnan(x)
        log_prob = torch.where(
            is_nan,
            torch.zeros_like(log_prob),
            log_prob,
        )

        return log_prob  # [B, num_groups, num_nodes]

    def __repr__(self):
        return f"GaussianLeafLayer(var={self.var}, means={self.means.shape}, stddevs={self.stddevs.shape})"

    def num_parameters(self) -> int:
        return int(self.means.numel() + self._raw_stddevs.numel())


class MixtureLeafLayer(LeafLayer):
    """Represents a mixture of Gaussian distributions as a single leaf node.
    Outputs `num_nodes` units, each being a distinct mixture over `h_in` base Gaussians.
    """

    def __init__(
        self,
        base_dist: LeafLayer,
        log_weights=None,
    ):
        super().__init__(
            scope=base_dist.scope,
            num_nodes=base_dist.num_nodes,
            num_groups=base_dist.num_groups,
            md_set=base_dist.md_set if hasattr(base_dist, "md_set") else None,
        )
        self.base_dist = base_dist
        self.log_weights = log_weights

    def to(self, device: torch.device):
        if getattr(self, "log_weights", None) is not None:
            self.log_weights = self.log_weights.to(device)
        if hasattr(self, "base_dist") and hasattr(self.base_dist, "to"):
            self.base_dist.to(device)
        return super().to(device)

    @property
    def var(self) -> Optional[int]:
        return getattr(self.base_dist, "var", None)

    def split_support(self) -> List[Interval]:
        raise NotImplementedError("Support splitting for MixtureLeafLayer is not implemented.")

    def num_parameters(self) -> int:
        from src.symbolic.arithmetic.weights import Weights

        total = self.base_dist.num_parameters() if self.base_dist is not None else 0
        if isinstance(self.log_weights, Weights):
            total += self.log_weights.num_parameters()
        elif isinstance(self.log_weights, torch.Tensor):
            total += int(self.log_weights.numel())
        return total

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        base_log_probs = self.base_dist.forward(data, children_outputs)

        return self.log_weights.forward(base_log_probs)

    def update_params(
        self, data: torch.Tensor, step_size: float = 1.0, smoothing: float = 1e-6, mask=None
    ):
        """
        Updates the mixture weights using EMA on the computed gradient counts.
        """
        if self.log_weights is not None:
            self.log_weights.update_params(step_size=step_size, smoothing=smoothing)


class CategoricalDistribution(Distribution):
    """Represents a Categorical distribution specification."""

    def __init__(
        self,
        var: int,
        categories: List[Any],
        probabilities: List[float],
    ):
        super().__init__(
            var,
            DiscreteInterval(range(len(categories))),
        )
        assert categories is not None and probabilities is not None, (
            "Categories and probabilities must be provided."
        )
        assert len(categories) == len(probabilities), (
            "Categories and probabilities must have the same length."
        )
        self.categories = categories
        self.probabilities = probabilities

    def split_support(self, split_count: int, strategy: str = "quantile") -> List[Interval]:
        intervals = self.var_support.split(split_count)
        while len(intervals) < split_count:
            intervals.append(DiscreteInterval(range(0)))
        return intervals

    def __repr__(self):
        return f"CategoricalDistribution(var={self.var}, categories={len(self.categories)})"

    def __eq__(self, other):
        return super().__eq__(other) and self.categories == other.categories

    def __hash__(self):
        return hash((super().__hash__(), tuple(self.categories)))


class CategoricalLeafLayer(LeafLayer):
    """Represents a Categorical distribution leaf node."""

    def __init__(
        self,
        spec: CategoricalDistribution,
        num_nodes: int = 1,
        num_groups: int = 1,
        node_supports: Optional[List[DiscreteInterval]] = None,
    ):
        super().__init__(BitSet([spec.var]), num_nodes=num_nodes, num_groups=num_groups)
        self.var = spec.var
        self.categories = spec.categories

        probs = torch.tensor(spec.probabilities, dtype=torch.float32).repeat(
            num_groups, num_nodes, 1
        )  # [num_groups, num_nodes, num_categories]
        logits = torch.log(probs.clamp(min=1e-20))
        # Add random perturbation to break symmetry
        logits = logits + torch.randn_like(logits) * 1.5
        self.logits = logits.requires_grad_(True)
        self.node_supports = node_supports

    def to(self, device: torch.device):
        if isinstance(self.logits, torch.Tensor):
            self.logits = self.logits.to(device)
            if not self.logits.is_leaf:
                self.logits = self.logits.detach().requires_grad_(self.logits.requires_grad)
        return super().to(device)

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        x_float = data[:, self.var]
        mask = torch.isnan(x_float)
        x = torch.where(mask, torch.zeros_like(x_float), x_float).long()
        logits = self.logits.clone()

        if hasattr(self, "node_supports") and self.node_supports and len(self.node_supports) > 1:
            for j in range(self.num_nodes):
                iv = self.node_supports[j]
                if iv is not None:
                    # Mask out unsupported categories before softmax
                    for c in range(len(self.categories)):
                        if not iv.contains(c):
                            logits[:, j, c] = float("-inf")

        # Handle rows that are all -inf
        is_all_inf = torch.isinf(logits).all(dim=-1)
        # Temporarily replace all -inf rows with 0s for softmax
        safe_logits = torch.where(is_all_inf.unsqueeze(-1), torch.zeros_like(logits), logits)
        log_probs = torch.log_softmax(safe_logits, dim=-1)
        # Restore -inf for all-inf rows
        log_probs = torch.where(is_all_inf.unsqueeze(-1), float("-inf"), log_probs)

        res = log_probs[:, :, x].permute(2, 0, 1)
        res = torch.where(mask.view(-1, 1, 1), torch.zeros_like(res), res)

        if res.requires_grad:
            res.retain_grad()
        return res

    def __repr__(self):
        return f"CategoricalLeafLayer(var={self.var}, categories={len(self.categories)})"

    def num_parameters(self) -> int:
        return int(self.logits.numel())


class SplineLeafLayer(LeafLayer):
    """Base class for learnable continuous disjoint-support spline leaves.

    A leaf group models a 1-D marginal as a uniform mixture of ``num_nodes``
    children with adjacent, disjoint supports.  Boundary heights are shared
    between neighbours, which makes the mixture density continuous at every
    split point by construction.  Subclasses differ in the shape of the child
    density (log-linear, linear, quadratic, rational-quadratic).
    """

    def __init__(
        self,
        spec: SplineDistribution,
        num_nodes: int = 1,
        num_groups: int = 1,
        node_supports: Optional[List[ContinuousInterval]] = None,
    ):
        if node_supports is None or len(node_supports) <= 1:
            raise ValueError(
                f"{self.__class__.__name__} requires disjoint node_supports with len > 1"
            )
        super().__init__(
            scope=BitSet([spec.var]),
            num_nodes=num_nodes,
            num_groups=num_groups,
        )
        self.node_supports = node_supports
        self.var = spec.var
        self.num_nodes = num_nodes
        self.num_groups = num_groups

        # Extract finite split points from node_supports (right boundaries of
        # the first N-1 intervals).  These are shared across all leaf groups so
        # that every group has the same node supports; the parent MD layers rely
        # on a node index meaning the same interval in every group.
        init_splits = []
        for i in range(num_nodes - 1):
            iv = node_supports[i]
            if iv is None or iv.high >= float("inf"):
                raise ValueError(
                    f"{self.__class__.__name__} requires a finite right boundary for interval {i}"
                )
            init_splits.append(float(iv.high))
        init_splits_t = torch.tensor(init_splits, dtype=torch.float32).unsqueeze(0)  # [1, N-1]

        # Initialize boundary heights as N * Gaussian pdf at the split points.
        h_np = num_nodes * stats.norm.pdf(
            init_splits_t.numpy(), loc=spec.base_mean, scale=spec.base_stddev
        )
        h_init = torch.tensor(h_np, dtype=torch.float32)
        h_init = h_init * torch.exp(torch.randn_like(h_init) * 0.05)
        h_init = h_init.clamp(min=1e-6)

        self._log_heights = torch.log(h_init).requires_grad_(True)  # [1, N-1]
        self._b1 = init_splits_t[:, 0:1].clone().requires_grad_(True)  # [1, 1]

    def _heights(self) -> torch.Tensor:
        # Clamp to a tiny positive floor so that exponential tails never become
        # exactly zero (which would produce -inf log-densities).
        return torch.exp(self._log_heights).clamp(min=1e-6)

    @abstractmethod
    def _interior_widths(self) -> torch.Tensor:
        """Return interior child widths of shape [G, N-2]."""
        pass

    @abstractmethod
    def _log_densities(self, x: torch.Tensor, h: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return per-child log-densities of shape [B, G, N].

        Args:
            x: input values, shape [B, 1].
            h: boundary heights, shape [G, N-1].
            b: split points, shape [G, N-1].
        """
        pass

    def _split_points(self) -> torch.Tensor:
        widths = self._interior_widths()
        if widths.numel() == 0:
            return self._b1
        return self._b1 + torch.cat([torch.zeros_like(self._b1), widths.cumsum(dim=-1)], dim=-1)

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        x = data[:, self.var].unsqueeze(1)  # [B, 1]
        h = self._heights()  # [1, N-1]
        b = self._split_points()  # [1, N-1]

        log_probs = self._log_densities(x, h, b)  # [B, 1, N]

        # Marginal determinism: only the active child may be non -inf.
        idx = (x.unsqueeze(-1) > b.unsqueeze(0)).long().sum(dim=-1)  # [B, 1]
        valid = idx.unsqueeze(-1) == torch.arange(self.num_nodes, device=x.device).view(1, 1, -1)
        log_probs = torch.where(valid, log_probs, torch.full_like(log_probs, float("-inf")))

        # Broadcast the single-group density to all leaf groups.  All groups
        # share the same split points, so they must have identical supports.
        log_probs = log_probs.expand(-1, self.num_groups, -1)  # [B, G, N]

        # NaN values are marginalized out (contribute log(1) = 0).  Use
        # ``unsqueeze(-1)`` so the condition broadcasts correctly against the
        # [B, G, N] log-prob tensor.
        is_nan = torch.isnan(x).unsqueeze(-1)
        log_probs = torch.where(is_nan, torch.zeros_like(log_probs), log_probs)

        if log_probs.requires_grad:
            log_probs.retain_grad()
        return log_probs

    def to(self, device: torch.device):
        for attr_name in ["_log_heights", "_b1"]:
            attr = getattr(self, attr_name, None)
            if isinstance(attr, torch.Tensor):
                attr = attr.to(device)
                if not attr.is_leaf:
                    attr = attr.detach().requires_grad_(attr.requires_grad)
                setattr(self, attr_name, attr)
        return super().to(device)

    def num_parameters(self) -> int:
        return int(self._log_heights.numel() + self._b1.numel())


class LogLinearSplineLeafLayer(SplineLeafLayer):
    """Piecewise log-linear child densities (truncated exponentials).

    Each interior child has the form
        P_i(x) = H_{i-1} * exp((H_i - H_{i-1}) * (x - b_{i-1}))
    on [b_{i-1}, b_i].  The width is fully determined by the boundary heights
    so that each child integrates to 1 and the density is continuous at knots.
    """

    def _interior_widths(self) -> torch.Tensor:
        h = self._heights()
        dh = h[:, 1:] - h[:, :-1]
        z = dh / h[:, :-1].clamp(min=1e-12)
        # Stable log1p(z)/z with Taylor fallback.
        small = z.abs() < 1e-4
        z_safe = torch.where(small, torch.ones_like(z), z)
        ratio_factor = torch.where(small, 1.0 - z / 2 + z**2 / 3, torch.log1p(z) / z_safe)
        return ratio_factor / h[:, :-1]

    def _log_densities(self, x: torch.Tensor, h: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        xg = x.unsqueeze(-1)  # [B, 1, 1]
        # Left tail child 0 on (-inf, b_0].
        log_p0 = torch.log(h[:, 0:1]).unsqueeze(0) + h[:, 0:1].unsqueeze(0) * (
            xg - b[:, 0:1].unsqueeze(0)
        )
        # Interior children.
        log_h_left = torch.log(h[:, :-1]).unsqueeze(0)  # [1, G, N-2]
        rate = (h[:, 1:] - h[:, :-1]).unsqueeze(0)
        log_p_int = log_h_left + rate * (xg - b[:, :-1].unsqueeze(0))
        # Right tail child N-1 on [b_{N-2}, +inf).
        log_pN = torch.log(h[:, -1:]).unsqueeze(0) - h[:, -1:].unsqueeze(0) * (
            xg - b[:, -1:].unsqueeze(0)
        )
        return torch.cat([log_p0, log_p_int, log_pN], dim=-1)


class LinearSplineLeafLayer(SplineLeafLayer):
    """Piecewise linear child densities.

    Each interior child linearly interpolates between the boundary heights
    H_{i-1} and H_i on [b_{i-1}, b_i].  The width is pinned by mass-1:
        w_i = 2 / (H_{i-1} + H_i).
    """

    def _interior_widths(self) -> torch.Tensor:
        h = self._heights()
        return 2.0 / (h[:, :-1] + h[:, 1:]).clamp(min=1e-12)

    def _log_densities(self, x: torch.Tensor, h: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        xg = x.unsqueeze(-1)  # [B, 1, 1]
        widths = self._interior_widths()  # [G, N-2]

        # Left tail.
        log_p0 = torch.log(h[:, 0:1]).unsqueeze(0) + h[:, 0:1].unsqueeze(0) * (
            xg - b[:, 0:1].unsqueeze(0)
        )
        # Interior: P_i(x) = H_{i-1}(1-u) + H_i u,  u = (x - b_{i-1}) / w_i.
        u = (xg - b[:, :-1].unsqueeze(0)) / widths.unsqueeze(0).clamp(min=1e-12)
        lin = h[:, :-1].unsqueeze(0) * (1 - u) + h[:, 1:].unsqueeze(0) * u
        log_p_int = torch.log(lin.clamp(min=1e-12))
        # Right tail.
        log_pN = torch.log(h[:, -1:]).unsqueeze(0) - h[:, -1:].unsqueeze(0) * (
            xg - b[:, -1:].unsqueeze(0)
        )
        return torch.cat([log_p0, log_p_int, log_pN], dim=-1)


class QuadraticSplineLeafLayer(SplineLeafLayer):
    """Piecewise quadratic child densities with one free width per interior child.

    Interior child i on [b_{i-1}, b_i] is
        P_i(x) = H_{i-1}(1-u) + H_i u + c_i u(1-u),
    with u = (x - b_{i-1}) / w_i.  The coefficient
        c_i = 6/w_i - 3(H_{i-1} + H_i)
    encloses unit area.  Widths are learnable but capped so that P_i stays
    positive on its interval.
    """

    def __init__(
        self,
        spec: SplineDistribution,
        num_nodes: int = 1,
        num_groups: int = 1,
        node_supports: Optional[List[ContinuousInterval]] = None,
    ):
        super().__init__(spec, num_nodes, num_groups, node_supports)

        # Initialize free interior widths from the quantile spacing (shared across groups).
        splits = []
        for i in range(num_nodes - 1):
            splits.append(self.node_supports[i].high)
        splits_t = torch.tensor(splits, dtype=torch.float32).unsqueeze(0)  # [1, N-1]
        h_init = self._heights().detach()
        cap = 0.999 * 6.0 / (h_init[:, :-1] + h_init[:, 1:]).clamp(min=1e-12)
        if num_nodes > 2:
            w_init = splits_t[:, 1:] - splits_t[:, :-1]
            w_init = w_init.clamp(min=1e-6, max=None)
            ratio = (w_init / cap).clamp(min=1e-6, max=0.999)
            logit_init = torch.log(ratio / (1 - ratio))
            self._logit_widths = logit_init.requires_grad_(True)  # [1, N-2]
        else:
            self._logit_widths = None

    def _interior_widths(self) -> torch.Tensor:
        h = self._heights()
        if self.num_nodes <= 2:
            return torch.zeros(1, 0, device=h.device)
        cap = 0.999 * 6.0 / (h[:, :-1] + h[:, 1:]).clamp(min=1e-12)
        return cap * torch.sigmoid(self._logit_widths)

    def _log_densities(self, x: torch.Tensor, h: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        xg = x.unsqueeze(-1)  # [B, 1, 1]

        # Left tail.
        log_p0 = torch.log(h[:, 0:1]).unsqueeze(0) + h[:, 0:1].unsqueeze(0) * (
            xg - b[:, 0:1].unsqueeze(0)
        )

        # Interior quadratic children.
        widths = self._interior_widths()
        if widths.numel() > 0:
            u = (xg - b[:, :-1].unsqueeze(0)) / widths.unsqueeze(0).clamp(min=1e-12)
            c = 6.0 / widths - 3.0 * (h[:, :-1] + h[:, 1:])
            quad = (
                h[:, :-1].unsqueeze(0) * (1 - u)
                + h[:, 1:].unsqueeze(0) * u
                + c.unsqueeze(0) * u * (1 - u)
            )
            log_p_int = torch.log(quad.clamp(min=1e-12))
        else:
            log_p_int = torch.empty(xg.shape[0], h.shape[0], 0, device=x.device, dtype=x.dtype)

        # Right tail.
        log_pN = torch.log(h[:, -1:]).unsqueeze(0) - h[:, -1:].unsqueeze(0) * (
            xg - b[:, -1:].unsqueeze(0)
        )
        return torch.cat([log_p0, log_p_int, log_pN], dim=-1)

    def to(self, device: torch.device):
        if self._logit_widths is not None and isinstance(self._logit_widths, torch.Tensor):
            self._logit_widths = self._logit_widths.to(device)
            if not self._logit_widths.is_leaf:
                self._logit_widths = self._logit_widths.detach().requires_grad_(
                    self._logit_widths.requires_grad
                )
        return super().to(device)


class RationalQuadraticSplineLeafLayer(SplineLeafLayer):
    """Rational-quadratic spline CDF leaf (Neural Spline Flow style).

    The CDF knot heights are fixed at ``k / N`` so each child encloses unit
    mass.  Interior bins use the Gregory--Delbourgo rational-quadratic spline
    with shared knot derivatives; exponential tails are spliced at the two
    outermost knots.  Learnable parameters: boundary heights (PDF values at
    knots) and positive interior bin widths.
    """

    def __init__(
        self,
        spec: SplineDistribution,
        num_nodes: int = 1,
        num_groups: int = 1,
        node_supports: Optional[List[ContinuousInterval]] = None,
    ):
        super().__init__(spec, num_nodes, num_groups, node_supports)

        if num_nodes <= 2:
            self._log_widths = None
            return

        # Initialise widths from the quantile spacing (shared across groups).
        splits = [self.node_supports[i].high for i in range(num_nodes - 1)]
        splits_t = torch.tensor(splits, dtype=torch.float32).unsqueeze(0)  # [1, N-1]
        w_init = splits_t[:, 1:] - splits_t[:, :-1]
        w_init = w_init.clamp(min=1e-6)
        self._log_widths = torch.log(w_init).requires_grad_(True)  # [1, N-2]

    def _interior_widths(self) -> torch.Tensor:
        if self._log_widths is None:
            return torch.zeros(1, 0, device=self._b1.device)
        return torch.exp(self._log_widths).clamp(min=1e-6)

    def _log_densities(self, x: torch.Tensor, h: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        xg = x.unsqueeze(-1)  # [B, 1, 1]

        # Left exponential tail on (-inf, b_0].
        log_p0 = torch.log(h[:, 0:1]).unsqueeze(0) + h[:, 0:1].unsqueeze(0) * (
            xg - b[:, 0:1].unsqueeze(0)
        )

        # Right exponential tail on [b_{N-2}, +inf).
        log_pN = torch.log(h[:, -1:]).unsqueeze(0) - h[:, -1:].unsqueeze(0) * (
            xg - b[:, -1:].unsqueeze(0)
        )

        if self.num_nodes <= 2:
            return torch.cat([log_p0, log_pN], dim=-1)

        # Interior RQ bins indexed 1 .. N-2.
        # Left/right boundaries and heights for those bins.
        bL = b[:, :-1]  # [G, N-2]
        hL = h[:, :-1]  # [G, N-2]
        hR = h[:, 1:]  # [G, N-2]
        w = self._interior_widths()  # [G, N-2]

        alpha = (xg - bL.unsqueeze(0)) / w.unsqueeze(0).clamp(min=1e-12)
        alpha = alpha.clamp(0.0, 1.0)

        # RQ parameters for the *full* CDF segment that rises by 1/N over width
        # w.  The shared boundary heights h are child-density values, so the
        # full-CDF knot derivatives are h / N.
        inv_n = 1.0 / self.num_nodes
        s = inv_n / w.clamp(min=1e-12)
        deltaL = hL * inv_n
        deltaR = hR * inv_n

        one_m_alpha = 1.0 - alpha
        numerator = (
            deltaL * one_m_alpha * one_m_alpha
            + 2.0 * s * alpha * one_m_alpha
            + deltaR * alpha * alpha
        )
        denominator = s + (deltaL + deltaR - 2.0 * s) * alpha * one_m_alpha
        g_prime = (s * numerator) / (denominator * denominator).clamp(min=1e-12)

        # Child density = g' / w  (mass 1 by construction).
        log_p_int = torch.log(g_prime.clamp(min=1e-12)) - torch.log(w.unsqueeze(0).clamp(min=1e-12))

        return torch.cat([log_p0, log_p_int, log_pN], dim=-1)

    def to(self, device: torch.device):
        if self._log_widths is not None and isinstance(self._log_widths, torch.Tensor):
            self._log_widths = self._log_widths.to(device)
            if not self._log_widths.is_leaf:
                self._log_widths = self._log_widths.detach().requires_grad_(
                    self._log_widths.requires_grad
                )
        return super().to(device)

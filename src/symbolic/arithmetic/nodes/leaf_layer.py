import math
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np
import torch
from scipy import stats

from src.utils import BitSet, ContinuousInterval, DiscreteInterval, Interval, Support

from .base import ArithmeticNode


# Numerical floor to prevent NaN gradients in logsumexp when all branches are -inf
LOG_FLOOR = float("-inf")


class LeafLayer(ArithmeticNode):
    """Represents a leaf distribution (e.g., Gaussian) in the SPN."""

    pass


class ConstantRegionNode(LeafLayer):
    """Leaf that always returns log(1) = 0 for all inputs.

    Used to represent a marginalized variable in a compiled estimand circuit.
    The variable is still in scope (so the circuit remains smooth/decomposable)
    but contributes nothing to the density.
    """

    def __init__(
        self,
        scope: BitSet,
        num_nodes: int = 1,
        num_groups: int = 1,
        md_set: Optional[BitSet] = None,
    ):
        from src.utils import ContinuousInterval

        super().__init__(
            support=Support(
                {
                    var: ContinuousInterval(float("-inf"), float("inf"), False, False)
                    for var in scope
                }
            ),
            num_nodes=num_nodes,
            num_groups=num_groups,
            md_set=md_set,
        )

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        B = data.shape[0]
        return torch.zeros(B, self.num_groups, self.num_nodes, device=data.device)

    def __repr__(self):
        return f"ConstantRegionNode(scope={list(self.scope)}, num_nodes={self.num_nodes}), num_groups={self.num_groups})"


class ProductLeafLayer(LeafLayer):
    """Leaf that combines two leaves by adding their log-densities.

    Used for DetProd (support-compatible product) at the leaf level:
    log(p_A(x) * p_B(x)) = log p_A(x) + log p_B(x).
    """

    def __init__(self, leaf_a: LeafLayer, leaf_b: LeafLayer, expand_leaves: bool = False):
        super().__init__(
            support=leaf_a.support,
            num_nodes=leaf_a.num_nodes
            if not expand_leaves
            else leaf_a.num_nodes * leaf_b.num_nodes,
            num_groups=leaf_a.num_groups * leaf_b.num_groups,
            md_set=leaf_a.md_set,
        )
        assert leaf_a.num_groups == leaf_b.num_groups, (
            "ProductLeafLayer requires both leaves to have the same number of groups."
        )
        assert leaf_a.num_nodes == leaf_b.num_nodes, (
            "ProductLeafLayer requires both leaves to have the same number of nodes."
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
            support=base_leaf.support,
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


class Distribution(ABC):
    """Base class for probability distributions used as leaves in SPNs."""

    def __init__(self, var: int, var_support: Interval):
        self.var = var
        self.var_support = var_support
        self.support = Support({var: var_support})

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

    def split_support(self, split_count: int, strategy: str = "quantile") -> List[Interval]:
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


class GaussianLeafLayer(LeafLayer):
    """Represents a Gaussian distribution leaf node."""

    def __init__(
        self,
        spec: GaussianDistribution,
        num_nodes: int = 1,
        num_groups: int = 1,
        node_supports: Optional[List[Support]] = None,
    ):
        super().__init__(spec.support, num_nodes=num_nodes, num_groups=num_groups)
        self.var = spec.var
        self.node_supports = node_supports

        base_m = spec.base_mean
        base_std = spec.base_stddev

        new_means = torch.zeros((num_groups, num_nodes), dtype=torch.float32)
        new_stddevs = torch.full((num_groups, num_nodes), base_std, dtype=torch.float32)

        if node_supports is not None:
            assert len(node_supports) == num_nodes
            for i in range(num_nodes):
                iv: ContinuousInterval = node_supports[i].get(self.var)
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
                iv: ContinuousInterval = self.node_supports[j].get(self.var)
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

        self._last_forward_output = log_prob
        return log_prob  # [B, num_groups, num_nodes]

    def __repr__(self):
        return f"GaussianLeafLayer(var={self.var}, means={self.means.shape}, stddevs={self.stddevs.shape})"


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
            support=base_dist.support,
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

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        base_log_probs = self.base_dist.forward(data, children_outputs)
        base_log_probs = base_log_probs.clamp(min=-1e10)  # NOTE: Take a second look

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
    ):
        super().__init__(spec.support, num_nodes=num_nodes, num_groups=num_groups)
        self.var = spec.var
        self.categories = spec.categories

        probs = torch.tensor(spec.probabilities, dtype=torch.float32).repeat(
            num_groups, num_nodes, 1
        )  # [num_groups, num_nodes, num_categories]
        logits = torch.log(probs.clamp(min=1e-20))
        # Add random perturbation to break symmetry
        logits = logits + torch.randn_like(logits) * 1.5
        self.logits = logits.requires_grad_(True)

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
                iv = self.node_supports[j].get(self.var)
                if iv is not None:
                    # Mask out unsupported categories before softmax
                    for c in range(len(self.categories)):
                        if not iv.contains(c):
                            logits[:, j, c] = float("-inf")

        # Handle rows that are all -inf
        is_all_inf = torch.isinf(logits).all(dim=-1)
        # Temporarily replace all -inf rows with 0s for softmax
        safe_logits = torch.where(is_all_inf.unsqueeze(1), torch.zeros_like(logits), logits)
        log_probs = torch.log_softmax(safe_logits, dim=-1)
        # Restore -inf for all-inf rows
        log_probs = torch.where(is_all_inf.unsqueeze(1), float("-inf"), log_probs)

        res = log_probs[:, :, x].permute(2, 0, 1)
        res = torch.where(mask.view(-1, 1, 1), torch.zeros_like(res), res)

        if res.requires_grad:
            res.retain_grad()
        self._last_forward_output = res
        return res

    def __repr__(self):
        return f"CategoricalLeafLayer(var={self.var}, categories={len(self.categories)})"

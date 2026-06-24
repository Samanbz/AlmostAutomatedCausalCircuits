import copy
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


class LeafNode(ArithmeticNode):
    """Represents a leaf distribution (e.g., Gaussian) in the SPN."""

    pass


class ConstantLeafNode(LeafNode):
    """Leaf that always returns log(1) = 0 for all inputs.

    Used to represent a marginalized variable in a compiled estimand circuit.
    The variable is still in scope (so the circuit remains smooth/decomposable)
    but contributes nothing to the density.
    """

    def __init__(self, var: int, unit_count: int = 1, md_set: Optional[BitSet] = None):
        from src.utils import ContinuousInterval

        var_support = ContinuousInterval(float("-inf"), float("inf"), False, False)
        super().__init__(
            support=Support({var: var_support}),
            unit_count=unit_count,
            md_set=md_set,
        )
        self.var = var

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        B = data.shape[0]
        return torch.zeros(B, self.unit_count, device=data.device)

    def __repr__(self):
        return f"ConstantLeafNode(var={self.var}, units={self.unit_count})"


class InverseLeafNode(LeafNode):
    """Leaf that wraps another leaf and negates its log-density.

    Used for POW(-1) operations: if the base leaf returns log p(x),
    this returns -log p(x) = log p(x)^{-1}.
    """

    def __init__(self, base_leaf: LeafNode, power: int = -1):
        super().__init__(
            support=base_leaf.support,
            unit_count=base_leaf.unit_count,
            unit_supports=base_leaf.unit_supports,
            md_set=base_leaf.md_set,
        )
        self.base_leaf = base_leaf
        self.power = power
        if hasattr(base_leaf, "var"):
            self.var = base_leaf.var

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        base_out = self.base_leaf.forward(data, children_outputs)
        # Mask out -inf BEFORE multiplying to avoid generating +inf or NaNs
        is_dead = torch.isinf(base_out)
        safe_base = torch.where(is_dead, torch.zeros_like(base_out), base_out)
        inv = safe_base * self.power
        # Restore the -inf floor for the dead branches
        return torch.where(is_dead, float("-inf"), inv)

    def __repr__(self):
        return f"InverseLeafNode(base={self.base_leaf}, power={self.power})"


class CartesianLeafNode(LeafNode):
    """Leaf that computes the Cartesian outer sum of two leaves in log-space.
    Output size is h_A * h_B.
    """

    def __init__(self, leaf_a: LeafNode, leaf_b: LeafNode):
        # Merge supports: since they are over different variables or overlapping,
        # we can just use the union of intervals for representation.
        support_union = copy.copy(leaf_a.support)
        for var, interval in leaf_b.support.intervals.items():
            support_union.intervals[var] = interval

        super().__init__(
            support=support_union,
            unit_count=leaf_a.unit_count * leaf_b.unit_count,
            md_set=leaf_a.md_set,
        )
        self.leaf_a = leaf_a
        self.leaf_b = leaf_b

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        out_a = self.leaf_a.forward(data, children_outputs)  # [B, h_A]
        out_b = self.leaf_b.forward(data, children_outputs)  # [B, h_B]
        B = out_a.shape[0]
        h_A = out_a.shape[1]
        h_B = out_b.shape[1]

        outer = out_a.unsqueeze(2) + out_b.unsqueeze(1)  # [B, h_A, h_B]
        return outer.reshape(B, h_A * h_B)

    def __repr__(self):
        return f"CartesianLeafNode(a={self.leaf_a}, b={self.leaf_b})"


class ProductLeafNode(LeafNode):
    """Leaf that combines two leaves by adding their log-densities.

    Used for DetProd (support-compatible product) at the leaf level:
    log(p_A(x) * p_B(x)) = log p_A(x) + log p_B(x).
    """

    def __init__(self, leaf_a: LeafNode, leaf_b: LeafNode):
        super().__init__(
            support=leaf_a.support,
            unit_count=leaf_a.unit_count,
            unit_supports=leaf_a.unit_supports,
            md_set=leaf_a.md_set,
        )
        self.leaf_a = leaf_a
        self.leaf_b = leaf_b
        if hasattr(leaf_a, "var"):
            self.var = leaf_a.var

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        out_a = self.leaf_a.forward(data, children_outputs)
        out_b = self.leaf_b.forward(data, children_outputs)
        return out_a + out_b

    def __repr__(self):
        return f"ProductLeafNode(a={self.leaf_a}, b={self.leaf_b})"


class Distribution(LeafNode, ABC):
    """Base class for probability distributions used as leaves in SPNs."""

    def __init__(self, var: int, var_support: Interval, unit_count: int = 1):
        super().__init__(Support({var: var_support}), unit_count=unit_count)
        self.var = var
        self.var_support = var_support

    @abstractmethod
    def sample(self) -> Any:
        """Sample a value from the distribution."""
        pass

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
    """Represents a Gaussian distribution leaf node."""

    def __init__(self, var: int, mean: float, stddev: float, unit_count: int = 1):
        super().__init__(
            var,
            ContinuousInterval(float("-inf"), float("inf"), include_low=False, include_high=False),
            unit_count=unit_count,
        )
        self.mean = mean
        self.stddev = stddev

    def sample(self) -> float:
        """Sample from the Gaussian distribution."""
        return np.random.normal(self.mean, self.stddev)

    def split_support(self, split_count: int, strategy: str = "quantile") -> List[Interval]:
        """For a Gaussian, we can split the real line into intervals.
        Supported strategies: 'quantile', 'perturbed_quantile'.
        """
        mean = float(torch.as_tensor(self.mean).detach().mean())
        std = float(torch.as_tensor(self.stddev).detach().mean())

        if strategy == "perturbed_quantile":
            # Add some random noise to the quantiles, keeping 0 and 1 fixed
            inner_quantiles = np.linspace(0, 1, split_count + 1)[1:-1]
            noise = np.random.uniform(
                -0.5 / split_count, 0.5 / split_count, size=len(inner_quantiles)
            )
            # ensure they stay sorted and within (0, 1)
            inner_quantiles = np.clip(np.sort(inner_quantiles + noise), 1e-6, 1 - 1e-6)
            quantiles = np.concatenate([[0.0], inner_quantiles, [1.0]])
        else:  # default to standard quantile
            quantiles = np.linspace(0, 1, split_count + 1)

        boundaries = stats.norm.ppf(quantiles, loc=mean, scale=std)
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

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        x = data[:, self.var].unsqueeze(1)  # [B, 1]

        # Ensure mean and stddev are tensors of shape [1, unit_count]
        mean = torch.as_tensor(self.mean, device=data.device, dtype=data.dtype).view(1, -1)
        std = torch.as_tensor(self.stddev, device=data.device, dtype=data.dtype).view(1, -1)
        std = std.abs().clamp(min=1e-4)

        # Broadcast scalar parameters to all units
        if mean.numel() == 1 and self.unit_count > 1:
            mean = mean.expand(1, self.unit_count)
        if std.numel() == 1 and self.unit_count > 1:
            std = std.expand(1, self.unit_count)

        var = std**2
        log_scale = torch.log(std * math.sqrt(2 * math.pi))
        log_prob = -((x - mean) ** 2) / (2 * var) - log_scale

        if hasattr(self, "unit_supports") and self.unit_supports and len(self.unit_supports) > 1:
            h = log_prob.shape[1]
            assert h == len(self.unit_supports) and h == mean.shape[1], (
                "Invalid h, unit_supports, or mean shape"
            )
            sqrt2 = math.sqrt(2)
            for j in range(len(self.unit_supports)):
                iv = self.unit_supports[j].get(self.var)
                if iv is not None and (iv.low > float("-inf") or iv.high < float("inf")):
                    # Strictly check the interval flags!
                    oob_low = (x < iv.low) if getattr(iv, "include_low", True) else (x <= iv.low)
                    oob_high = (
                        (x > iv.high) if getattr(iv, "include_high", True) else (x >= iv.high)
                    )
                    oob = oob_low | oob_high

                    log_prob[:, j] = torch.where(
                        oob.squeeze(1),
                        torch.full_like(log_prob[:, j], float("-inf")),
                        log_prob[:, j],
                    )
                    # Keep normalization so the truncated density integrates to ~1.
                    if iv.low > float("-inf"):
                        z_lo = (iv.low - mean[:, j]) / (std[:, j] * sqrt2)
                        cdf_lo = 0.5 * (1 + torch.erf(z_lo))
                    else:
                        cdf_lo = torch.zeros_like(mean[:, j])

                    if iv.high < float("inf"):
                        z_hi = (iv.high - mean[:, j]) / (std[:, j] * sqrt2)
                        cdf_hi = 0.5 * (1 + torch.erf(z_hi))
                    else:
                        cdf_hi = torch.ones_like(mean[:, j])

                    log_Z = torch.log((cdf_hi - cdf_lo).clamp(min=1e-12))
                    log_prob[:, j] = log_prob[:, j] - log_Z

        if log_prob.requires_grad:
            log_prob.retain_grad()
        self._last_forward_output = log_prob
        return log_prob  # [B, unit_count]

    def __repr__(self):
        return f"GaussianDistribution(var={self.var}, mean={self.mean}, stddev={self.stddev})"

    def __eq__(self, other):
        return super().__eq__(other) and self.mean == other.mean and self.stddev == other.stddev

    def __hash__(self):
        return hash((super().__hash__(), self.mean, self.stddev))


class GaussianMixture(LeafNode):
    """Represents a mixture of Gaussian distributions as a single leaf node.
    Outputs `unit_count` units, each being a distinct mixture over `h_in` base Gaussians.
    """

    def __init__(
        self,
        base_dist: LeafNode,
        log_weights: torch.Tensor,
    ):
        super().__init__(
            support=base_dist.support,
            unit_count=log_weights.shape[0],
            unit_supports=None,  # TODO: set this?
            md_set=base_dist.md_set if hasattr(base_dist, "md_set") else None,
        )
        self.base_dist = base_dist
        self.log_weights = log_weights  # shape [unit_count, h_in]

    @property
    def var(self) -> Optional[int]:
        return getattr(self.base_dist, "var", None)

    @property
    def mean(self) -> torch.Tensor:
        return self.base_dist.mean

    @property
    def stddev(self) -> torch.Tensor:
        return self.base_dist.stddev

    def sample(self) -> float:
        raise NotImplementedError("Sampling from GaussianMixture is not implemented.")

    def split_support(self) -> List[Interval]:
        raise NotImplementedError("Support splitting for GaussianMixture is not implemented.")

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        log_probs = self.base_dist.forward(data, children_outputs)  # [B, h_in]
        log_probs = log_probs.clamp(min=-1e10)
        log_probs = self.log_weights.unsqueeze(0) + log_probs.unsqueeze(1)  # [B, unit_count, h_in]
        log_probs = torch.logsumexp(log_probs, dim=2)  # [B, unit_count]
        return log_probs

    def update_params(
        self, data: torch.Tensor, step_size: float = 1.0, smoothing: float = 1e-6, mask=None
    ):
        """
        Updates the mixture weights using EMA on the computed gradient counts.
        """
        if self.log_weights is None or self.log_weights.grad is None:
            return

        # Gradients of LL w.r.t log_weights are exactly the expected counts
        counts = self.log_weights.grad.nan_to_num(0.0).clamp(min=0.0)

        # Normalize to get the batch M-step estimate
        probs = (counts + smoothing) / (
            counts.sum(dim=-1, keepdim=True) + smoothing * counts.shape[-1]
        )

        # EMA update in probability space
        old_probs = torch.exp(self.log_weights.data)
        new_probs = (1.0 - step_size) * old_probs + step_size * probs

        self.log_weights.data.copy_(torch.log(new_probs.clamp(min=1e-20)))
        self.log_weights.grad.zero_()


class CategoricalDistribution(Distribution):
    """Represents a Categorical distribution leaf node."""

    def __init__(
        self, var: int, categories: List[Any], probabilities: List[float], unit_count: int = 1
    ):
        super().__init__(var, DiscreteInterval(range(len(categories))), unit_count=unit_count)
        self.categories = categories
        probs = torch.tensor(probabilities, dtype=torch.float32)
        logits = torch.log(probs.clamp(min=1e-20)).unsqueeze(0).expand(unit_count, -1).clone()
        if unit_count > 1:
            logits = logits + torch.randn_like(logits) * 0.1
        self.logits = logits.requires_grad_(True)

    def split_support(self, split_count: int, strategy: str = "quantile") -> List[Interval]:
        intervals = self.var_support.split(split_count)
        while len(intervals) < split_count:
            intervals.append(DiscreteInterval(range(0)))
        return intervals

    def sample(self) -> Any:
        """Sample from the categorical distribution according to probabilities."""
        probs = torch.softmax(self.logits[0], dim=-1).detach().numpy()
        return np.random.choice(self.categories, p=probs)

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        x = data[:, self.var].long()
        logits = self.logits.clone()

        if hasattr(self, "unit_supports") and self.unit_supports and len(self.unit_supports) > 1:
            for j in range(self.unit_count):
                iv = self.unit_supports[j].get(self.var)
                if iv is not None:
                    # Mask out unsupported categories before softmax
                    for c in range(len(self.categories)):
                        if not iv.contains(c):
                            logits[j, c] = float("-inf")

        log_probs = torch.log_softmax(logits, dim=-1)
        res = log_probs.t()[x]

        if res.requires_grad:
            res.retain_grad()
        self._last_forward_output = res
        return res

    def __repr__(self):
        return f"CategoricalDistribution(var={self.var}, categories={len(self.categories)})"

    def __eq__(self, other):
        return super().__eq__(other) and self.categories == other.categories

    def __hash__(self):
        return hash((super().__hash__(), tuple(self.categories)))


class UniformDistribution(Distribution):
    """Represents a Uniform distribution leaf node."""

    def __init__(self, var: int, low: float, high: float, unit_count: int = 1):
        super().__init__(var, ContinuousInterval(low, high), unit_count=unit_count)
        self.low = low
        self.high = high

    def split_support(self, split_count: int, strategy: str = "quantile") -> List[Interval]:
        return self.var_support.split(split_count)

    def sample(self) -> float:
        """Sample uniformly from the distribution."""
        return np.random.uniform(self.low, self.high)

    def __repr__(self):
        return f"UniformDistribution(var={self.var}, low={self.low}, high={self.high})"

    def __eq__(self, other):
        return super().__eq__(other) and self.low == other.low and self.high == other.high

    def __hash__(self):
        return hash((super().__hash__(), self.low, self.high))

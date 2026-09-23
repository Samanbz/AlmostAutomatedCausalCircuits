import math
import random
from enum import Enum
from typing import Dict, Optional

import torch

from src.logger import logger as g_logger
from src.symbolic.arithmetic.circuit import SymbolicArithmeticCircuit
from src.symbolic.arithmetic.nodes.leaf_layer import (
    CategoricalDistribution,
    CategoricalLeafLayer,
    Distribution,
    GaussianDistribution,
    GaussianLeafLayer,
    LinearSplineDistribution,
    LinearSplineLeafLayer,
    LogLinearSplineDistribution,
    LogLinearSplineLeafLayer,
    MixtureLeafLayer,
    QuadraticSplineDistribution,
    QuadraticSplineLeafLayer,
    RationalQuadraticSplineDistribution,
    RationalQuadraticSplineLeafLayer,
)
from src.symbolic.arithmetic.nodes.sum_layer import SumLayer
from src.symbolic.arithmetic.weights import DenseWeights, SparseWeights
from src.symbolic.vtree import VNode, VTree
from src.utils import BitSet
from src.utils.node_allocator import IncrementalNodeAllocator


logger = g_logger.getChild("CircuitBuilder")
logger.setLevel("WARNING")


def find_closest_factor(n: int, max_val: int = None) -> int:
    """Find the largest factor of n that is less than or equal to sqrt(n)."""
    max_val = math.floor(max_val)
    if max_val > n:
        raise ValueError("max_val must be less than or equal to n")
    for i in range(max_val, 0, -1):
        if n % i == 0:
            return i
    return 1


class LayerType(Enum):
    UNIVERSAL = "UNIVERSAL"
    LEFT_MIXING = "LEFT_MIXING"
    PS_LEFT_MIXING = "PS_LEFT_MIXING"
    RIGHT_MIXING = "RIGHT_MIXING"
    PS_RIGHT_MIXING = "PS_RIGHT_MIXING"
    SYNTHESIZING = "SYNTHESIZING"

    @staticmethod
    def from_md_sets(parent: BitSet, left: BitSet, right: BitSet) -> "LayerType":
        if parent.is_universal:
            return LayerType.UNIVERSAL
        elif left == parent:
            return LayerType.LEFT_MIXING if right.is_universal else LayerType.PS_LEFT_MIXING
        elif right == parent:
            return LayerType.RIGHT_MIXING if left.is_universal else LayerType.PS_RIGHT_MIXING
        else:
            return LayerType.SYNTHESIZING

    @staticmethod
    def from_vtree(vtree: VTree, vid: int) -> "LayerType":
        vnode = vtree.get_node_data(vid)
        if vnode.md_set.is_universal:
            return LayerType.UNIVERSAL
        elif vtree.is_leaf(vid) and not vnode.md_set.is_universal:
            return LayerType.SYNTHESIZING
        else:
            children = vtree.get_children(vid)
            assert len(children) == 2, "VTree must be binary"
            l_vnode = vtree.get_node_data(children[0])
            r_vnode = vtree.get_node_data(children[1])
            return LayerType.from_md_sets(vnode.md_set, l_vnode.md_set, r_vnode.md_set)


class CircuitBuilder:
    """Constructs a SymbolicArithmeticCircuit directly from a VTree."""

    def __init__(
        self,
        vtree: VTree,
        num_nodes: int,
        input_dists: Dict[int, Distribution] = None,
        initialize_weights: bool = False,
        weight_softmax_temperature: float = 0.5,
        max_leaf_num_nodes: Optional[int] = None,
        max_leaf_num_groups: Optional[int] = None,
        max_sum_num_groups: Optional[int] = None,
        leaf_mixture_num_nodes: Optional[int] = None,
        leaf_mixture_num_groups: Optional[int] = None,
        fairness_temperature: float = 1.0,
    ):
        self.vtree = vtree
        self.num_nodes = num_nodes
        self.input_dists = input_dists
        self.node_allocator = IncrementalNodeAllocator()

        self.weight_softmax_temperature = weight_softmax_temperature
        self.initialize_weights = initialize_weights
        self.fairness_temperature = fairness_temperature

        self.max_leaf_num_nodes = max_leaf_num_nodes
        self.max_leaf_num_groups = max_leaf_num_groups
        self.max_sum_num_groups = max_sum_num_groups
        self.leaf_mixture_num_nodes = leaf_mixture_num_nodes
        self.leaf_mixture_num_groups = leaf_mixture_num_groups

        self.built_nodes = {}

    def _handle_leaf(
        self,
        vnode: VNode,
        G: int,
        H: int,
        is_constrained: bool,
    ):
        var_id = vnode.scope.min
        dist = self.input_dists[var_id]

        G_b = (
            G
            if (is_constrained or self.leaf_mixture_num_groups is None)
            else self.leaf_mixture_num_groups
        )
        H_b = (
            H
            if (is_constrained or self.leaf_mixture_num_nodes is None)
            else self.leaf_mixture_num_nodes
        )

        if not is_constrained and vnode.md_set.is_universal:
            leaf_supports = [dist.var_support] * H_b
        else:
            leaf_supports = dist.split_support(H_b)

        if isinstance(dist, GaussianDistribution) or len(leaf_supports) == 1:
            base_mean = float(getattr(dist, "base_mean", 0.0))
            base_stddev = float(getattr(dist, "base_stddev", 1.0))
            if base_stddev <= 0:
                base_stddev = 1.0
            leaf_layer = GaussianLeafLayer(
                GaussianDistribution(var=var_id, base_mean=base_mean, base_stddev=base_stddev),
                num_nodes=H_b,
                num_groups=G_b,
                node_supports=leaf_supports if is_constrained else None,
            )
        elif isinstance(dist, CategoricalDistribution):
            leaf_layer = CategoricalLeafLayer(dist, num_nodes=H, num_groups=G)
        elif is_constrained and isinstance(dist, LogLinearSplineDistribution):
            leaf_layer = LogLinearSplineLeafLayer(
                dist, num_nodes=H_b, num_groups=G_b, node_supports=leaf_supports
            )
        elif is_constrained and isinstance(dist, LinearSplineDistribution):
            leaf_layer = LinearSplineLeafLayer(
                dist, num_nodes=H_b, num_groups=G_b, node_supports=leaf_supports
            )
        elif is_constrained and isinstance(dist, QuadraticSplineDistribution):
            leaf_layer = QuadraticSplineLeafLayer(
                dist, num_nodes=H_b, num_groups=G_b, node_supports=leaf_supports
            )
        elif is_constrained and isinstance(dist, RationalQuadraticSplineDistribution):
            leaf_layer = RationalQuadraticSplineLeafLayer(
                dist, num_nodes=H_b, num_groups=G_b, node_supports=leaf_supports
            )
        else:
            raise NotImplementedError(f"Unsupported distribution type: {type(dist)}")

        leaf_layer.node_supports = leaf_supports
        leaf_layer.md_set = vnode.md_set

        if not is_constrained:
            raw_weights = (
                torch.randn(
                    (
                        G,
                        H,
                        G_b,
                        H_b,
                        1,
                        1,
                    )
                )
                * 0.1
            )
            for g in range(G_b):
                raw_weights[g % G, :, g, :, :, :] += 0.5
            weights = torch.exp(raw_weights)
            weights = weights / weights.sum(dim=(2, 3, 4, 5), keepdim=True)
            log_weights = torch.log(weights + 1e-20).detach().requires_grad_(True)
            logger.debug(
                f"Generated log weights for mixture leaf node with scope {vnode.scope}:\n{log_weights.view(G, H, G_b, H_b)}"
            )
            mixture_leaf_layer = MixtureLeafLayer(
                base_dist=leaf_layer, log_weights=DenseWeights(log_weights)
            )
            leaf_layer = mixture_leaf_layer

        return leaf_layer

    def _sample_balanced_assignment(
        self, num_items: int, num_bins: int, temperature: float
    ) -> list[list[int]]:
        """Sample an assignment of items to bins biased toward balanced loads.

        The score for each bin is ``-temperature * (count / target)`` plus a small
        Gaussian jitter. ``temperature = 0`` recovers uniform random assignment;
        larger values make perfectly balanced assignments the most likely outcome
        while still allowing unbalanced ones.
        """
        if num_bins == 0:
            raise ValueError("num_bins must be positive")
        if num_items == 0:
            return [[] for _ in range(num_bins)]

        counts = [0] * num_bins
        target = num_items / num_bins
        assignments: list[list[int]] = [[] for _ in range(num_bins)]

        for item in range(num_items):
            scores = torch.tensor(
                [
                    -temperature * (counts[b] / target if target > 0 else 0.0)
                    + random.gauss(0.0, 1.0)
                    for b in range(num_bins)
                ],
                dtype=torch.float32,
            )
            probs = torch.softmax(scores, dim=0)
            sampled = int(torch.multinomial(probs, 1).item())
            assignments[sampled].append(item)
            counts[sampled] += 1

        return assignments

    def _sample_balanced_assignment_filled(
        self, num_items: int, num_bins: int, temperature: float, max_tries: int = 200
    ) -> list[list[int]]:
        """Balanced assignment, retried until every bin is non-empty.

        An empty bin would be a sum unit with no child combination — a dead
        unit that training cannot revive. Requires ``num_bins <= num_items``,
        which ``_cap_dead_parents`` enforces dimensionally.
        """
        for _ in range(max_tries):
            assignments = self._sample_balanced_assignment(num_items, num_bins, temperature)
            if all(len(row) > 0 for row in assignments):
                return assignments
        # Deterministic fallback: round-robin (non-empty whenever bins <= items).
        assignments = [[] for _ in range(num_bins)]
        for item in range(num_items):
            assignments[item % num_bins].append(item)
        return assignments

    def _cap_constrained_leaf_units(self, child_vid: int, G: int, H: int, n: int):
        """A constrained leaf unit must own at least one category of its variable.

        Splitting fewer categories into more units mints empty-support units whose
        mass is identically 0 (dead units). Unconstrained leaves are normalized
        mixtures over the full support, so they are unaffected.
        """
        if not self.vtree.is_leaf(child_vid):
            return G, H
        cvnode = self.vtree.get_node_data(child_vid)
        if cvnode.md_set.is_universal:
            return G, H
        dist = self.input_dists[cvnode.scope.min]
        n_categories = len(getattr(dist, "categories", []) or [])
        if n_categories and H > n_categories:
            H = n_categories
            G = max(1, n // H)
        return G, H

    def _get_child_dimensions(
        self,
        layer_type: LayerType,
        l_vid: int,
        r_vid: int,
        n: int,
        l_is_constrained: bool,
        r_is_constrained: bool,
    ):
        """Child dims for a layer, with dead-parent caps applied (see
        ``_cap_dead_parents``)."""
        G_l, H_l, G_r, H_r = self._child_dimensions_raw(
            layer_type, l_vid, r_vid, n, l_is_constrained, r_is_constrained
        )
        G_l, H_l = self._cap_dead_parents(l_vid, G_l, H_l, n)
        G_r, H_r = self._cap_dead_parents(r_vid, G_r, H_r, n)
        return G_l, H_l, G_r, H_r

    def _cap_dead_parents(self, vid: int, G: int, H: int, n: int):
        """Cap a node's (G, H) so that no sum unit is born dead.

        A constrained parent unit is dead unless it recruits at least one
        allowed child combination: in (pseudo-)left-mixing layers each parent
        needs a distinct left child unit (so at most H_L parents), right-mixing
        symmetrically (H_R), and synthesizing layers recruit unique
        (h_l, h_r) combinations (so at most H_L * H_R parents). Training cannot
        revive a structurally dead unit, so the construction must not create
        any in the first place.
        """
        if self.vtree.is_leaf(vid):
            return G, H
        vnode = self.vtree.get_node_data(vid)
        if vnode.md_set.is_universal:
            return G, H  # dense layer: every combination is available
        layer_type = LayerType.from_vtree(self.vtree, vid)
        l_vid, r_vid = self.vtree.get_children_pair(vid)
        l_is_constrained = not self.vtree.get_node_data(l_vid).md_set.is_universal
        r_is_constrained = not self.vtree.get_node_data(r_vid).md_set.is_universal
        G_L, H_L, G_R, H_R = self._get_child_dimensions(
            layer_type, l_vid, r_vid, n, l_is_constrained, r_is_constrained
        )
        if layer_type == LayerType.LEFT_MIXING:
            slots = H_L
        elif layer_type == LayerType.PS_LEFT_MIXING:
            # PS parents share left indices by design; the construction assigns
            # (h_l, h_r) pairs to parents, so the only deadness risk is an empty
            # parent (H <= H_L * H_R), not a shared left unit.
            slots = H_L * H_R
        elif layer_type == LayerType.RIGHT_MIXING:
            slots = H_R
        elif layer_type == LayerType.PS_RIGHT_MIXING:
            slots = H_L * H_R
        elif layer_type == LayerType.SYNTHESIZING:
            slots = H_L * H_R
        else:
            return G, H
        if G * H > slots:
            H = max(1, slots // G)
            if G * H > slots:
                G = max(1, slots // H)
        return G, H

    def _child_dimensions_raw(
        self,
        layer_type: LayerType,
        l_vid: int,
        r_vid: int,
        n: int,
        l_is_constrained: bool,
        r_is_constrained: bool,
    ):
        G_l, H_l, G_r, H_r = 1, n, 1, n

        if self.max_sum_num_groups is not None:
            G_l = min(G_l, self.max_sum_num_groups)
            G_r = min(G_r, self.max_sum_num_groups)
            H_l = n // G_l
            H_r = n // G_r
            G_l, H_l = self._cap_constrained_leaf_units(l_vid, G_l, H_l, n)
            G_r, H_r = self._cap_constrained_leaf_units(r_vid, G_r, H_r, n)
            return G_l, H_l, G_r, H_r
        a = find_closest_factor(n, max_val=math.sqrt(n))
        b = n // a

        if layer_type == LayerType.UNIVERSAL:
            # doesn't really matter since the weights are dense over everything anyways
            pass
        elif layer_type == LayerType.LEFT_MIXING:
            G_l, H_l, G_r, H_r = 1, n, n, 1
        elif layer_type == LayerType.PS_LEFT_MIXING:
            G_l, H_l, G_r, H_r = a, b, b, a
        elif layer_type == LayerType.RIGHT_MIXING:
            G_l, H_l, G_r, H_r = n, 1, 1, n
        elif layer_type == LayerType.PS_RIGHT_MIXING:
            G_l, H_l, G_r, H_r = b, a, a, b
        elif layer_type == LayerType.SYNTHESIZING:
            l_child_type = LayerType.from_vtree(self.vtree, l_vid)
            r_child_type = LayerType.from_vtree(self.vtree, r_vid)

            l_is_synth_like = l_child_type == LayerType.SYNTHESIZING
            r_is_synth_like = r_child_type == LayerType.SYNTHESIZING

            if l_is_synth_like and not r_is_synth_like:
                favor_right = True
            elif r_is_synth_like and not l_is_synth_like:
                favor_right = False
            else:
                favor_right = True

            if favor_right:
                h_l, G_l = b, a
                G_r, h_r = b, a
            else:
                h_r, G_r = b, a
                h_l, G_l = a, b

            G_l, h_l = self._cap_constrained_leaf_units(l_vid, G_l, h_l, n)
            G_r, h_r = self._cap_constrained_leaf_units(r_vid, G_r, h_r, n)
            return G_l, h_l, G_r, h_r

        # Handle leaf cases
        if (
            self.vtree.is_leaf(l_vid)
            and l_is_constrained
            and (not self.vtree.is_leaf(r_vid) or not r_is_constrained)
            and self.max_leaf_num_nodes is not None
        ):
            # if only one child is a constrained leaf and max_leaf_num_nodes is set, then adapt the sibling sum layer's num_groups to match
            G_l = (
                n // self.max_leaf_num_nodes
                if self.max_leaf_num_groups is None
                else self.max_leaf_num_groups
            )
            H_l = self.max_leaf_num_nodes
            G_r = self.max_leaf_num_nodes
            H_r = n // self.max_leaf_num_nodes
        elif (
            self.vtree.is_leaf(r_vid)
            and r_is_constrained
            and (not self.vtree.is_leaf(l_vid) or not r_is_constrained)
            and self.max_leaf_num_nodes is not None
        ):
            G_r = (
                n // self.max_leaf_num_nodes
                if self.max_leaf_num_groups is None
                else self.max_leaf_num_groups
            )
            H_r = self.max_leaf_num_nodes
            G_l = self.max_leaf_num_nodes
            H_l = n // self.max_leaf_num_nodes
        elif (
            self.vtree.is_leaf(l_vid)
            and l_is_constrained
            and self.vtree.is_leaf(r_vid)
            and r_is_constrained
            and self.max_leaf_num_nodes is not None
        ):
            # if both children are constrained leaves and max_leaf_num_nodes is set, then adapt both sum layers' num_groups to match
            G_l = (
                n // self.max_leaf_num_nodes
                if self.max_leaf_num_groups is None
                else self.max_leaf_num_groups
            )
            H_l = self.max_leaf_num_nodes
            G_r = (
                n // self.max_leaf_num_nodes
                if self.max_leaf_num_groups is None
                else self.max_leaf_num_groups
            )
            H_r = self.max_leaf_num_nodes

        if (G_l == 1 or G_r == 1) and layer_type == LayerType.SYNTHESIZING:
            logger.warning(
                f"Synthesizing SumLayer and vtree children ({l_vid}, {r_vid}) has G_l={G_l}, G_r={G_r}. This may lead to degenerate behavior."
            )
        G_l, H_l = self._cap_constrained_leaf_units(l_vid, G_l, H_l, n)
        G_r, H_r = self._cap_constrained_leaf_units(r_vid, G_r, H_r, n)
        return G_l, H_l, G_r, H_r

    def _generate_weights(
        self,
        G: int,
        H: int,
        G_L: int,
        H_L: int,
        G_R: int,
        H_R: int,
        layer_type: LayerType,
        is_constrained: bool,
        fairness_temperature: float = 2.0,
    ) -> torch.Tensor:
        """Generates global weight assignments ensuring determinism constraints."""

        w = torch.zeros((G, H, G_L, H_L, G_R, H_R))

        if layer_type == LayerType.UNIVERSAL:
            w = torch.exp(torch.randn((G, H, G_L, H_L, G_R, H_R)) * self.weight_softmax_temperature)

        elif layer_type == LayerType.LEFT_MIXING:
            assert is_constrained, "LEFT_MIXING should only occur for constrained nodes"
            h_l_assignments = self._sample_balanced_assignment_filled(H_L, H, fairness_temperature)
            g_l_assignments = self._sample_balanced_assignment(H_L, G_L, fairness_temperature)

            for g in range(G):
                for h in range(H):
                    h_ls = h_l_assignments[h]
                    g_ls = [
                        [h_l in g_l_assignments[g_t] for g_t in range(G_L)].index(True)
                        for h_l in h_ls
                    ]
                    for idx, h_l in enumerate(h_ls):
                        w[g, h, g_ls[idx], h_l, :, :] = torch.exp(
                            torch.randn((G_R, H_R)) * self.weight_softmax_temperature
                        )

        elif layer_type == LayerType.PS_LEFT_MIXING:
            assert is_constrained, "PS_LEFT_MIXING should only occur for constrained nodes"
            flat_assignments = self._sample_balanced_assignment_filled(
                H_L * H_R, H, fairness_temperature
            )
            assignments = [
                [(i // H_R, i % H_R) for i in row] for row in flat_assignments
            ]  # convert to (left, right) pairs

            print(f"flat_assignments: {assignments}")
            g_l_assignments = self._sample_balanced_assignment(H_L, G_L, fairness_temperature)
            g_ls = [
                [h_l in g_l_assignments[g_t] for g_t in range(G_L)].index(True)
                for h_l in range(H_L)
            ]
            for g in range(G):
                for h in range(H):
                    for h_l, h_r in assignments[h]:
                        g_l = g_ls[h_l]
                        g_r = h_l % G_R
                        w[g, h, g_l, h_l, g_r, h_r] = torch.exp(
                            torch.randn(1) * self.weight_softmax_temperature
                        )

        elif layer_type == LayerType.RIGHT_MIXING:
            assert is_constrained, "RIGHT_MIXING should only occur for constrained nodes"
            h_r_assignments = self._sample_balanced_assignment_filled(H_R, H, fairness_temperature)
            g_r_assignments = self._sample_balanced_assignment(H_R, G_R, fairness_temperature)
            for g in range(G):
                for h in range(H):
                    h_rs = h_r_assignments[h]
                    g_rs = [
                        [h_r in g_r_assignments[g_t] for g_t in range(G_R)].index(True)
                        for h_r in h_rs
                    ]
                    for idx, h_r in enumerate(h_rs):
                        w[g, h, :, :, g_rs[idx], h_r] = torch.exp(
                            torch.randn((G_L, H_L)) * self.weight_softmax_temperature
                        )

        elif layer_type == LayerType.PS_RIGHT_MIXING:
            assert is_constrained, "PS_RIGHT_MIXING should only occur for constrained nodes"
            flat_assignments = self._sample_balanced_assignment_filled(
                H_R * H_L, H, fairness_temperature
            )
            print(f"flat_assignments: {assignments}")

            assignments = [
                [(i // H_R, i % H_R) for i in row] for row in flat_assignments
            ]  # convert to (left, right) pairs
            g_r_assignments = self._sample_balanced_assignment(H_R, G_R, fairness_temperature)
            g_rs = [
                [h_r in g_r_assignments[g_t] for g_t in range(G_R)].index(True)
                for h_r in range(H_R)
            ]
            for g in range(G):
                for h in range(H):
                    for h_l, h_r in assignments[h]:
                        g_r = g_rs[h_r]
                        g_l = h_r % G_L
                        w[g, h, g_l, h_l, g_r, h_r] = torch.exp(
                            torch.randn(1) * self.weight_softmax_temperature
                        )

        elif layer_type == LayerType.SYNTHESIZING:
            assert is_constrained, "SYNTHESIZING should only occur for constrained nodes"
            # assign each combination of left and right child nodes to a unique sum node
            flat_assignments = self._sample_balanced_assignment_filled(
                H_L * H_R, H, fairness_temperature
            )
            assignments = [
                [(i // H_R, i % H_R) for i in row] for row in flat_assignments
            ]  # convert to (left, right) pairs
            for g in range(G):
                for h in range(H):
                    for h_l, h_r in assignments[h]:
                        # If H_R < G_L or H_L < G_R, then some groups will be unused,
                        # but this only occurs when both leaves are constrained and max_leaf_num_nodes is set (i.e. synthesizing).
                        # If we know this will happen, we can additionally set max_leaf_num_groups to avoid the wasted leaf groups.
                        g_l = h_r % G_L
                        g_r = h_l % G_R
                        w[g, h, g_l, h_l, g_r, h_r] = torch.exp(
                            torch.randn(1) * self.weight_softmax_temperature
                        )
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        w_sum = w.sum(dim=(2, 3, 4, 5), keepdim=True)

        safe_w_sum = torch.where(w_sum == 0, torch.ones_like(w_sum), w_sum)
        w = w / safe_w_sum

        for g in range(G):
            for n in range(H):
                w_to_print = (w > 0).int()
                logger.debug(
                    f"LayerType={layer_type}, is_constrained={is_constrained}, g={g}, n={n},\nG_L={G_L}, H_L={H_L}, G_R={G_R}, H_R={H_R},\nw[g,n,:,:,:, :]=\n{w_to_print[g, n, :, :, :, :].view(G_L * H_L, G_R * H_R)}"
                )

        return w

    def _build_recursive(
        self,
        ac: SymbolicArithmeticCircuit,
        vid: int,
        G: int = 1,
        H: int = 1,
    ) -> int:
        if vid in self.built_nodes:
            return self.built_nodes[vid]

        vnode = self.vtree.get_node_data(vid)
        is_constrained = not vnode.md_set.is_universal

        if self.vtree.is_leaf(vid):
            leaf_layer = self._handle_leaf(vnode, G, H, is_constrained)

            leaf_id = ac._add_node(leaf_layer)
            ac.sum_to_vtree[leaf_id] = vid
            ac.vtree_to_sum[vid] = leaf_id
            self.built_nodes[vid] = leaf_id
            return leaf_id

        # Internal node logic
        layer_type = LayerType.from_vtree(self.vtree, vid)

        children = self.vtree.get_children(vid)
        assert len(children) == 2, "VTree must be binary"

        l_vnode = self.vtree.get_node_data(children[0])
        r_vnode = self.vtree.get_node_data(children[1])
        l_is_constrained = not l_vnode.md_set.is_universal
        r_is_constrained = not r_vnode.md_set.is_universal

        G_L, H_L, G_R, H_R = self._get_child_dimensions(
            layer_type,
            children[0],
            children[1],
            self.num_nodes,
            l_is_constrained,
            r_is_constrained,
        )
        logger.debug(
            f"\nBuilding node {vid} with scope {vnode.scope} ,mdset {vnode.md_set}, layer type {layer_type}, G={G}, H={H}, G_L={G_L}, H_L={H_L}, G_R={G_R}, H_R={H_R}"
        )

        l_id = self._build_recursive(ac, children[0], G=G_L, H=H_L)
        r_id = self._build_recursive(ac, children[1], G=G_R, H=H_R)

        sum_layer = SumLayer(
            scope=vnode.scope,
            md_set=vnode.md_set,
            num_nodes=H,
            num_groups=G,
        )

        logger.debug(
            f"Generating weights for node {vid} with scope {vnode.scope}")
        # print(f"Generating weights for node {vid} with scope {vnode.scope}")
        w = self._generate_weights(
            G,
            H,
            G_L,
            H_L,
            G_R,
            H_R,
            layer_type,
            is_constrained,
            self.fairness_temperature,
        )

        log_w = torch.log(w + 1e-20).detach().requires_grad_(True)
        sparse_flag = layer_type != LayerType.UNIVERSAL

        if sparse_flag:
            mask = (w > 0).float()
            sum_layer.log_weights = SparseWeights(log_w, mask=mask)
        else:
            sum_layer.log_weights = DenseWeights(log_w)

        sum_id = ac._add_node(sum_layer)
        ac._add_edge(sum_id, l_id)
        ac._add_edge(sum_id, r_id)

        ac.sum_to_vtree[sum_id] = vid
        ac.vtree_to_sum[vid] = sum_id
        self.built_nodes[vid] = sum_id
        return sum_id

    def build(self) -> SymbolicArithmeticCircuit:
        """Construct a SymbolicArithmeticCircuit directly from the VTree."""
        circuit = SymbolicArithmeticCircuit(vtree=self.vtree)
        root_vid = self.vtree.get_roots()[0]
        self._build_recursive(circuit, root_vid, G=1, H=1)
        return circuit


def create_md_circuit(
    input_dists: Dict[int, Distribution],
    md_var_decomp: VTree,
    num_nodes: int,
    initialize_weights: bool = False,
    fairness_temperature: float = 1.0,
    weight_softmax_temperature: float = 0.5,
    max_leaf_num_nodes: Optional[int] = None,
    max_leaf_num_groups: Optional[int] = None,
    max_sum_num_groups: Optional[int] = None,
    leaf_mixture_num_nodes: Optional[int] = None,
    leaf_mixture_num_groups: Optional[int] = None,
) -> SymbolicArithmeticCircuit:
    """Creates an MD-Circuit end-to-end."""

    c_builder = CircuitBuilder(
        vtree=md_var_decomp,
        num_nodes=num_nodes,
        input_dists=input_dists,
        initialize_weights=initialize_weights,
        fairness_temperature=fairness_temperature,
        weight_softmax_temperature=weight_softmax_temperature,
        max_leaf_num_nodes=max_leaf_num_nodes,
        max_leaf_num_groups=max_leaf_num_groups,
        max_sum_num_groups=max_sum_num_groups,
        leaf_mixture_num_nodes=leaf_mixture_num_nodes,
        leaf_mixture_num_groups=leaf_mixture_num_groups,
    )
    return c_builder.build()
